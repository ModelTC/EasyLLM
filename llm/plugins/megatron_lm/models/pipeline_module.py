import os
from torch import Tensor
import torch.distributed as dist
from typing import Dict, Literal, Optional

from megatron.training import get_args
from megatron.core.transformer.enums import ModelType
from megatron.core import parallel_state, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.custom_layers.transformer_engine import te_checkpoint
from megatron.core.models.common.language_module.language_module import LanguageModule

if os.getenv("DIST_BACKEND", "easyllm") == "easyllm":
    from llm.utils.env import dist_env
elif os.getenv("DIST_BACKEND", "easyllm") == "megatron":
    from megatron.core import mpu as dist_env

from .utils import (
    partition_uniform,
    partition_balanced
)
from .embeddings import RotaryEmbedding


class PipelineParallelModule(LanguageModule):
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

        # checkpointing
        if (self.config.recompute_granularity == 'full' and \
            self.config.recompute_method == 'dynamic_seqlen'
        ):
            args = get_args()
            self._seq_len_to_recompute_layer = args.seq_len_to_recompute_layer
    
    def _get_recompute_layer_num(self, seq_len):
        seq_len_keys = sorted(list(self._seq_len_to_recompute_layer.keys()))
        for key in seq_len_keys:
            if seq_len <= key:
                layer_num = self._seq_len_to_recompute_layer[key]
                if isinstance(layer_num, list):
                    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
                    layer_num = layer_num[pp_rank]
                return layer_num
        return -1

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

    def build_module_list(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

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

    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        packed_seq_params: PackedSeqParams,
        tf_idx: int
    ):
        """Forward method with activation checkpointing."""

        def custom(start: int, end: int):
            def custom_forward(
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
            ):
                for index in range(start, end):
                    # layer = self._get_layer(index)
                    hidden_states, context = self.forward_funcs[index](
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        context=context,
                        context_mask=context_mask,
                        rotary_pos_emb=rotary_pos_emb,
                        inference_params=None,
                        packed_seq_params=packed_seq_params,
                    )
                return hidden_states, context

            return custom_forward

        def checkpoint_handler(forward_func):
            if self.config.fp8:
                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                )
            else:
                return tensor_parallel.checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                )

        if self.config.recompute_method == 'dynamic_seqlen':
            # Checkpoint the input activation according input sequence length
            if dist_env.get_pipeline_model_parallel_rank() == 0:
                l = tf_idx + 1
            else:
                l = tf_idx

            if hidden_states == None:
                hidden_states = self.forward_funcs[l].input_tensor
            seq_len = hidden_states.shape[0]
            recompute_layer_num = self._get_recompute_layer_num(seq_len)

            if tf_idx < recompute_layer_num:
                hidden_states, context = checkpoint_handler(custom(l, l + 1))
            else:
                hidden_states, context = custom(l, l + 1)(
                    hidden_states,
                    attention_mask,
                    context,
                    context_mask,
                    rotary_pos_emb,
                )
        else:
            raise ValueError("Invalid activation recompute method.")
        
        return hidden_states

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