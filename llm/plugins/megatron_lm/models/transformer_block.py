from typing import Union

from torch import Tensor

from megatron.training import get_args
from megatron.core.transformer.transformer_block import TransformerBlock, TransformerBlockSubmodules
from megatron.core import mpu
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import BaseTransformerLayer
from megatron.core.packed_seq_params import PackedSeqParams

try:
    from megatron.core.transformer.custom_layers.transformer_engine import (
        TEDelayedScaling,
        TENorm,
        get_cpu_offload_context,
        te_checkpoint,
    )

    HAVE_TE = True
    LayerNormImpl = TENorm
except ImportError:
    HAVE_TE = False
    get_cpu_offload_context = None
    try:
        import apex

        LayerNormImpl = FusedLayerNorm
    except ModuleNotFoundError:
        from megatron.core.transformer.torch_layer_norm import WrappedTorchLayerNorm

        LayerNormImpl = WrappedTorchLayerNorm

def partition_uniform(num_items, num_parts):
    import numpy
    parts = [0] * (num_parts + 1)
    # First check for the trivial edge case
    if num_items <= num_parts:
        for p in range(num_parts + 1):
            parts[p] = min(p, num_items)
        return parts

    chunksize = num_items // num_parts
    residual = num_items - (chunksize * num_parts)

    parts = numpy.arange(0, (num_parts + 1) * chunksize, chunksize)

    for i in range(residual):
        parts[i + 1:] += 1
    parts = parts.tolist()

    return parts


def get_num_layers_to_build(config: TransformerConfig) -> int:

    # pipeline_ranks = config.pipeline_model_parallel_size

    # num_layers_per_pipeline_rank = config.num_layers // pipeline_ranks
    args = get_args()
    method = args.pp_partition_method
    
    method = method.lower()

    # Each stage gets a simple uniform number of layers.
    if method == 'uniform':
        num_layers = config.num_layers
        parts = partition_uniform(num_items=num_layers, num_parts=mpu.get_pipeline_model_parallel_world_size())
        # recompute for megatron-lm
        for idx in range(len(parts) - 1, 0, -1):
            parts[idx] = parts[idx] - parts[idx - 1]
        parts.pop(0)
        args.pp_partition_parts = parts
        num_layers_per_pipeline_rank = parts[mpu.get_pipeline_model_parallel_rank()]
    elif method == 'parameters':
        ## TODO
        parts = args.pp_partition_parts
        assert parts is not None, "parameters type parts should not be None."
        # recompute for megatron-lm
        for idx in range(len(parts) - 1, 0, -1):
            parts[idx] = parts[idx] - parts[idx - 1]
        parts.pop(0)
        parts[0] -= 1
        parts[-1] -= 2
        args.pp_partition_parts = parts
        num_layers_per_pipeline_rank = parts[mpu.get_pipeline_model_parallel_rank()]
    elif "manual" in method:
        parts = method.split("manual:")[1].split(',')
        parts = [int(item) for item in parts]
        # recompute for megatron-lm
        for idx in range(len(parts) - 1, 0, -1):
            parts[idx] = parts[idx] - parts[idx - 1]
        parts.pop(0)
        args.pp_partition_parts = parts
        num_layers_per_pipeline_rank = parts[mpu.get_pipeline_model_parallel_rank()]
    elif method.startswith('type:'):
        ## TODO
        pass
    elif method == 'profile':
        raise NotImplementedError(f'Partitioning method {method} not implemented.')
    else:
        raise NotImplementedError(f'Partitioning method {method} not implemented.')
    

    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
        # Interleaved pipeline parallelism:
        # Number of layers in each model chunk is the number of layers in the stage,
        # divided by the number of model chunks in a stage.
        # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0]  [2]  [4]  [6]
        # Stage 1: [1]  [3]  [5]  [7]
        # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0, 1]  [4, 5]
        # Stage 1: [2, 3]  [6, 7]

        vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

        assert num_layers_per_pipeline_rank % vp_size == 0, "number of layers per pipe stage should be divided by vp_size."
        num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size

        num_layers_to_build = num_layers_per_virtual_rank

    else:
        # Non-interleaved pipeline parallelism:
        # Each stage gets a contiguous set of layers.

        num_layers_to_build = num_layers_per_pipeline_rank

    return num_layers_to_build

def _get_block_submodules(
    config: TransformerConfig,
    spec: Union[TransformerBlockSubmodules, ModuleSpec],
) -> TransformerBlockSubmodules:

    # Transformer block submodules.
    if isinstance(spec, TransformerBlockSubmodules):
        return spec

    # ModuleSpec here is generally assumed to be for a transformer layer that
    # is implemented in `transformer_layer.py` or if it subclasses
    # `BaseTransformerLayer` from the `transformer_layer.py` file.
    elif isinstance(spec, ModuleSpec):
        if issubclass(spec.module, TransformerBlock):
            return spec.submodules
        elif issubclass(spec.module, BaseTransformerLayer):
            num_layers = get_num_layers_to_build(config)
            return TransformerBlockSubmodules(
                layer_specs=[spec] * num_layers,
                layer_norm=LayerNormImpl,
            )
        else:
            raise Exception(f"specialize for {spec.module.__name__}.")
    else:
        raise Exception(f"specialize for {type(spec).__name__}.")


class DynamicTransformerBlock(TransformerBlock):
    def __init__(
        self,
        config: TransformerConfig,
        spec: Union[TransformerBlockSubmodules, ModuleSpec],
        post_layer_norm: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
    ):
        super().__init__(
            config=config,
            spec=spec,
            post_layer_norm=post_layer_norm,
            pre_process=pre_process,
            post_process=post_process
        )

        self.submodules = _get_block_submodules(config, spec)

        self._build_layers()
        self.num_layers_per_pipeline_rank = len(self.layers)

        if (self.config.recompute_granularity == 'full' and \
            self.config.recompute_method == 'dynamic_seqlen'
        ):
            args = get_args()
            self._seq_len_to_recompute_layer = args.seq_len_to_recompute_layer

    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        packed_seq_params: PackedSeqParams,
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
                    layer = self._get_layer(index)
                    hidden_states, context = layer(
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

        if self.config.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            l = 0
            while l < self.num_layers_per_pipeline_rank:
                hidden_states, context = checkpoint_handler(
                    custom(l, l + self.config.recompute_num_layers)
                )

                l += self.config.recompute_num_layers

        elif self.config.recompute_method == 'block':
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            recompute_skip_num_layers = 0
            for l in range(self.num_layers_per_pipeline_rank):
                # Skip recomputation when input grad computation is not needed.
                # Need to have at least one input tensor with gradient computation
                # for re-enterant autograd engine.
                if self.config.fp8 and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if (
                    l >= recompute_skip_num_layers
                    and l < self.config.recompute_num_layers + recompute_skip_num_layers
                ):
                    hidden_states, context = checkpoint_handler(custom(l, l + 1))
                else:
                    hidden_states, context = custom(l, l + 1)(
                        hidden_states,
                        attention_mask,
                        context,
                        context_mask,
                        rotary_pos_emb,
                    )

        elif self.config.recompute_method == 'dynamic_seqlen':
            # Checkpoint the input activation according input sequence length
            seq_len = hidden_states.shape[0]
            recompute_layer_num = self._get_recompute_layer_num(seq_len)

            for l in range(self.num_layers_per_pipeline_rank):
                if l < recompute_layer_num:
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
