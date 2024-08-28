from bisect import bisect_left
from typing import Literal, Optional

from megatron.core import mpu
from megatron.training import get_args
from megatron.core import tensor_parallel
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding

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

from .transformer_block import DynamicTransformerBlock


class LlaMAModel(GPTModel):
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
        rotary_percent (float, optional): Percent of rotary dimension to use for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional): Base period for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 10000.
        seq_len_interpolation_factor (Optional[float], optional): scale of linearly interpolating RoPE for longer sequences. The value must be a float larger than 1.0. Defaults to None.
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
            seq_len_interpolation_factor=seq_len_interpolation_factor
        )

        # update pp partition method - parameters args
        self.transformer_layer_spec = transformer_layer_spec
        self.position_embedding_type = position_embedding_type
        self.update_parameters_pp_partition()

        # Transformer.
        self.decoder = DynamicTransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

    def update_parameters_pp_partition(self):
        args = get_args()
        if args.pp_partition_method != "parameters":
            return

        # count layer params
        num_layers = 1 + self.config.num_layers + 1 + 1
        param_counts = [0] * num_layers
        # word embedding
        layer = LanguageModelEmbedding(
            config=self.config,
            vocab_size=self.vocab_size,
            max_sequence_length=self.max_sequence_length,
            position_embedding_type=self.position_embedding_type,
        )
        params = filter(lambda p: p.requires_grad, layer.parameters())
        param_counts[0] = sum(p.numel() for p in params)
        # transformer layer & final norm
        def build_layer(layer_spec, layer_number):
            return build_module(
                layer_spec,
                config=self.config,
                layer_number=layer_number,
            )

        spec = self.transformer_layer_spec
        if isinstance(self.transformer_layer_spec, ModuleSpec):
            if issubclass(self.transformer_layer_spec.module, TransformerBlock):
                spec = self.transformer_layer_spec.submodules
            elif issubclass(self.transformer_layer_spec.module, BaseTransformerLayer):
                spec = TransformerBlockSubmodules(
                    layer_specs=[self.transformer_layer_spec],
                    layer_norm=LayerNormImpl,
                )
            else:
                raise Exception(f"specialize for {self.transformer_layer_spec.module.module.__name__}.")
        layer = build_layer(spec.layer_specs, 1)
        params = filter(lambda p: p.requires_grad, layer.parameters())
        for _ in range(1, self.config.num_layers + 1):
            param_counts[_] = sum(p.numel() for p in params)
        layer = build_module(
            spec.layer_norm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        params = filter(lambda p: p.requires_grad, layer.parameters())
        param_counts[self.config.num_layers + 1] = sum(p.numel() for p in params)
        # lm_head
        layer = tensor_parallel.ColumnParallelLinear(
            self.config.hidden_size,
            self.vocab_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            gather_output=not self.parallel_output,
            skip_weight_param_allocation=self.pre_process
            and self.share_embeddings_and_output_weights,
            embedding_activation_buffer=self.embedding_activation_buffer,
            grad_output_buffer=self.grad_output_buffer,
        )
        params = filter(lambda p: p.requires_grad, layer.parameters())
        param_counts[self.config.num_layers + 2] = sum(p.numel() for p in params)
        # partition
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

        def prefix_sum_inc(weights):
            """ Compute an inclusive prefix sum.

            Example:
                >>> prefix_sum_inc([3,4,5])
                [3, 7, 12]
            """
            weights_ = [w for w in weights]
            for x in range(1, len(weights_)):
                weights_[x] += weights_[x - 1]
            return weights_

        def _lprobe(weights, num_parts, bottleneck):
            num_items = len(weights)
            total_weight = weights[-1]

            # initialize partitioning
            parts = [0] * (num_parts + 1)
            for p in range(1, num_parts + 1):
                parts[p] = num_items

            bsum = bottleneck  # running sum of target weight for pth partition
            chunksize = num_items // num_parts
            step = chunksize
            for p in range(1, num_parts):
                # Jump to the next bucket
                while (step < num_items) and (weights[step] < bsum):
                    step += chunksize

                # Find the end index of partition p
                parts[p] = bisect_left(weights, bsum, lo=step - chunksize, hi=min(step, num_items))
                # Nothing more to partition, return early
                if parts[p] == num_items:
                    # See if the current partition is overweight.
                    part_size = weights[-1] - weights[parts[p - 1]]
                    return parts, part_size < bottleneck

                # Next partition target
                bsum = weights[parts[p] - 1] + bottleneck

            return parts, bsum >= total_weight

        def _rb_partition_balanced(weights, num_parts, eps):
            total_weight = weights[-1]
            lower = total_weight / num_parts  # best case heaviest partition
            upper = total_weight  # worst case heaviest partition

            # Do a binary search for the best partitioning
            while upper > lower + eps:
                mid = lower + ((upper - lower) / 2)
                parts, success = _lprobe(weights, num_parts, mid)
                if success:
                    upper = mid
                else:
                    lower = mid + eps
            return upper

        def partition_balanced(weights, num_parts, eps=1e-3):
            num_items = len(weights)
            # First check for the trivial edge case
            if num_items <= num_parts:
                return partition_uniform(num_items, num_parts)

            weights_ = prefix_sum_inc(weights)

            # Find the smallest bottleneck (weight of heaviest partition)
            bottleneck = _rb_partition_balanced(weights_, num_parts, eps=eps)

            # Now compute that partitioning
            parts, success = _lprobe(weights_, num_parts, bottleneck)
            assert success

            return parts

        parts = partition_balanced(weights=param_counts, num_parts=mpu.get_pipeline_model_parallel_world_size())
        args.pp_partition_parts = parts