import os
from torch import Tensor
from contextlib import nullcontext
from typing import Literal, Optional

from .transformer_layer import TransformerLayer
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core import InferenceParams, tensor_parallel
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.config_logger import has_config_logger_enabled
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding

if os.getenv("DIST_BACKEND", "easyllm") == "easyllm":
    from llm.utils.env import dist_env
elif os.getenv("DIST_BACKEND", "easyllm") == "megatron":
    from megatron.core import mpu as dist_env

try:
    from megatron.core.transformer.custom_layers.transformer_engine import (
        TENorm,
        get_cpu_offload_context,
    )

    HAVE_TE = True
    LayerNormImpl = TENorm
except ImportError:
    HAVE_TE = False
    get_cpu_offload_context = None
    try:
        import apex  # noqa

        LayerNormImpl = FusedLayerNorm
    except ModuleNotFoundError:
        from megatron.core.transformer.torch_layer_norm import WrappedTorchLayerNorm

        LayerNormImpl = WrappedTorchLayerNorm

from .pipeline_module import PipelineParallelModule


class LlaMAModel(PipelineParallelModule):
    """GPT Transformer language model.

    Args:
        config (TransformerConfig): Transformer config
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers
        vocab_size (int): Vocabulary size
        max_sequence_length (int): maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional): Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional): Include an output layer (used with pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool, optional): Defaults to False.
        parallel_output (bool, optional): Do not gather the outputs, keep them split across tensor parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool, optional): When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional):  Position embedding type.. Defaults to 'learned_absolute'.
        rotary_percent (float, optional): Percent of rotary dimension to use for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.  # noqa
        rotary_base (int, optional): Base period for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 10000.
        seq_len_interpolation_factor (Optional[float], optional): scale of linearly interpolating RoPE for longer sequences. The value must be a float larger than 1.0. Defaults to None.  # noqa
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        seq_len_interpolation_factor: Optional[float] = None,
    ) -> None:
        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
        )

    def build_module_list(self):
        module_list = list()
        # embedding
        embedding_param = dict(
            name="word_embedding",
            type=LanguageModelEmbedding,
            kwargs=dict(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=self.position_embedding_type
            )
        )
        module_list.append(embedding_param)
        # transformer layer
        transformer_layer_params = dict(
            name="decoder_layer",
            type=TransformerLayer,
            kwargs=dict(
                config=self.config
            )
        )
        if hasattr(self.transformer_layer_spec, "submodules") and self.transformer_layer_spec.submodules is not None:
            transformer_layer_params["kwargs"]["submodules"] = self.transformer_layer_spec.submodules
            transformer_layer_params['kwargs']['pre_process'] = self.pre_process
        for layer_idx in range(self.config.num_layers):
            transformer_layer_params["layer_number"] = layer_idx + 1
            module_list.append(transformer_layer_params)
        # final layernorm after transformer layers
        layer_norm_params = dict(
            name="layernorm_before_head",
            type=LayerNormImpl,
            kwargs=dict(
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon
            )
        )
        module_list.append(layer_norm_params)
        # lm head
        if self.config.defer_embedding_wgrad_compute:
            # The embedding activation buffer preserves a reference to the input activations
            # of the final embedding projection layer GEMM. It will hold the activations for
            # all the micro-batches of a global batch for the last pipeline stage. Once we are
            # done with all the back props for all the microbatches for the last pipeline stage,
            # it will be in the pipeline flush stage. During this pipeline flush we use the
            # input activations stored in embedding activation buffer and gradient outputs stored
            # in gradient buffer to calculate the weight gradients for the embedding final linear layer.
            self.embedding_activation_buffer = []
            self.grad_output_buffer = []
        else:
            self.embedding_activation_buffer = None
            self.grad_output_buffer = None
        lm_head_params = dict(
            name="lm_head",
            type=tensor_parallel.ColumnParallelLinear,
            kwargs=dict(
                input_size=self.config.hidden_size,
                output_size=self.vocab_size,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=False, # self.pre_process and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
            )
        )
        module_list.append(lm_head_params)

        return module_list

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
    ) -> Tensor:
        rotary_pos_emb = None
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = input_ids
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        if self.module_list[self._local_start]["name"] == "decoder_layer":
            if self.position_embedding_type == 'rope':
                rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                    inference_params, self.forward_funcs[0], decoder_input, self.config
                )
                rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)
        
        for idx in range(len(self.forward_funcs)):
            if self.pre_process and idx == 0:
                # embeddings
                decoder_input = self.forward_funcs[idx](input_ids=decoder_input, position_ids=position_ids)
                if self.position_embedding_type == 'rope':
                    rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                        inference_params, self.forward_funcs[idx + 1], decoder_input, self.config
                    )
                    rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)
            elif self.module_list[self._local_start + idx]["name"] == "decoder_layer":
                if self.config.sequence_parallel:
                    rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
                else:
                    rng_context = nullcontext()
                if self.config.fp8:
                    pass
                else:
                    fp8_context = nullcontext()
                with rng_context and fp8_context:
                    # Forward pass.
                    if self.config.recompute_granularity == 'full' and self.training:
                        if dist_env.get_pipeline_model_parallel_rank() == 0:
                            tf_idx = idx - 1
                        else:
                            tf_idx = idx
                        hidden_states = self._checkpointed_forward(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=None,
                            context_mask=None,
                            rotary_pos_emb=rotary_pos_emb,
                            packed_seq_params=packed_seq_params,
                            tf_idx=tf_idx
                        )
                    else:
                        decoder_input, context = self.forward_funcs[idx](
                            hidden_states=decoder_input,
                            attention_mask=attention_mask,
                            context=None,
                            context_mask=None,
                            rotary_pos_emb=rotary_pos_emb,
                            inference_params=inference_params,
                            packed_seq_params=packed_seq_params,
                        )
            elif self.module_list[self._local_start + idx]["name"] == "layernorm_before_head":
                decoder_input = self.forward_funcs[idx](decoder_input)
            elif self.module_list[self._local_start + idx]["name"] == "lm_head":
                # logits and loss
                output_weight = None
                if self.share_embeddings_and_output_weights:
                    output_weight = self.shared_embedding_or_output_weight()
                logits, _ = self.forward_funcs[idx](decoder_input, weight=output_weight)

                if has_config_logger_enabled(self.config):
                    payload = OrderedDict(
                        {
                            'input_ids': input_ids,
                            'position_ids': position_ids,
                            'attention_mask': attention_mask,
                            'decoder_input': decoder_input,
                            'logits': logits,
                        }
                    )
                    log_config_to_disk(self.config, payload, prefix='input_and_logits')

                if labels is None:
                    # [s b h] => [b s h]
                    # return logits.transpose(0, 1).contiguous()
                    decoder_input = logits.transpose(0, 1).contiguous()
                else:
                    # loss
                    decoder_input = self.compute_language_model_loss(labels, logits)

        return decoder_input