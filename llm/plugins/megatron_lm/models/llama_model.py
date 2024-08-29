import os
import torch
import torch.distributed as dist
from torch import Tensor
from bisect import bisect_left
from typing import Dict, Literal, Optional

if os.getenv("DIST_BACKEND", "easyllm") == "easyllm":
    from llm.utils.env import dist_env
elif os.getenv("DIST_BACKEND", "easyllm") == "megatron":
    from megatron.core import mpu as dist_env
from megatron.training import get_args
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core import InferenceParams, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.transformer_layer import TransformerLayer

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

# from .transformer_block import DynamicTransformerBlock
from megatron.core.transformer.transformer_block import TransformerBlock, TransformerBlockSubmodules
from megatron.core.transformer.transformer_layer import BaseTransformerLayer
from .utils import (
    partition_uniform,
    partition_balanced
)


class LlaMAModel(LanguageModule):
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
        super().__init__(config=config)
        if dist.is_initialized():
            self.global_rank = dist.get_rank()
        else:
            self.global_rank = -1

        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        # partition
        self.module_list = self.build_module_list()
        # initialize partition
        self._partition_layers()

        # These 2 attributes are needed for TensorRT-LLM export.
        self.max_position_embeddings = max_sequence_length
        self.rotary_percent = rotary_percent

        if self.position_embedding_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                use_cpu_initialization=self.config.use_cpu_initialization,
            )
        # model build
        self.forward_funcs = []
        self.build()
        if self.pre_process:
            self.embedding = self.forward_funcs[0]
        if self.post_process:
            self.output_layer = self.forward_funcs[-1]

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()
        
        if has_config_logger_enabled(self.config):
            log_config_to_disk(
                self.config, self.state_dict(), prefix=f'{type(self).__name__}_init_ckpt'
            )

    def build(self):
        for local_idx, layer in enumerate(self.part_module_list):
            layer_idx = local_idx + self._local_start

            module_type = layer['type']
            module_kwargs = layer['kwargs']
            if layer['name'] == "lm_head":
                if self.post_process:
                    module_kwargs['skip_weight_param_allocation'] = self.pre_process and self.share_embeddings_and_output_weights
                else:
                    continue
            module = module_type(**module_kwargs)
            name = str(layer_idx)
            self.forward_funcs.append(module)
            self.add_module(name, module)

    def _partition_layers(self):
        num_stages = dist_env.get_pipeline_model_parallel_world_size()
        stage_id = dist_env.get_pipeline_model_parallel_rank()

        args = get_args()
        method = args.pp_partition_method
        method = method.lower()

        if method == "uniform":
            num_layers = len(self.module_list)
            self.parts = partition_uniform(num_items=num_layers, num_parts=num_stages)
        elif method == "parameters":
            param_counts = self._count_layer_params()
            self.parts = partition_balanced(weights=param_counts, num_parts=num_stages)
        elif "manual" in method:
            parts = method.split("manual:")[1].split(',')
            self.parts = [int(item) for item in parts]
        elif method.startswith('type:'):
            # TODO
            pass
        elif method == 'profile':
            raise NotImplementedError(f'Partitioning method {method} not implemented.')
        else:
            raise NotImplementedError(f'Partitioning method {method} not implemented.')

        # Print some information on the partitioning.
        if self.global_rank == 0:
            for stage in range(num_stages):
                start = self.parts[stage]
                stop = self.parts[stage + 1]
                print(f'stage={stage} layers={stop - start}')
                for idx, layer in enumerate(self.module_list[start:stop]):
                    name = layer['name']
                    print(f'    {idx+start:2d}: {name}')
        args.pp_partition_parts = self.parts
        # setting according module list
        self._local_start = self.parts[stage_id]
        self._local_stop = self.parts[stage_id + 1]
        self.part_module_list = self.module_list[self._local_start:self._local_stop]
 
    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        # self.decoder.set_input_tensor(input_tensor[0])
        for idx in range(len(self.part_module_list)):
            if self.part_module_list[idx]["name"] == "decoder_layer":
                self.forward_funcs[idx].set_input_tensor(input_tensor[0])
                break

    def _count_layer_params(self):
        """Count the trainable parameters in individual layers.

        This routine will only build one layer at a time.

        Returns:
            A list of the number of parameters in each layer.
        """
        param_counts = [0] * len(self.module_list)
        for idx, layer in enumerate(self.module_list):
            module_type = layer["type"]
            kwargs = layer["kwargs"]
            l = module_type(**kwargs)
            params = filter(lambda p: p.requires_grad, l.parameters())
            param_counts[idx] = sum(p.numel() for p in params)
        return param_counts

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

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[Dict] = None
    ) -> ShardedStateDict:
        """Sharded state dict implementation for GPTModel backward-compatibility (removing extra state).

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the GPTModel
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        output_layer_extra_state_key = f'{prefix}output_layer._extra_state'

        # Old GPT checkpoints only stored the output layer weight key. So we remove the _extra_state key
        # but check that it doesn't contain any data anyway
        output_extra_state = sharded_state_dict.pop(output_layer_extra_state_key, None)
        assert not (
            output_extra_state and output_extra_state.data
        ), f'Expected output layer extra state to be empty, got: {output_extra_state}'

        return sharded_state_dict
