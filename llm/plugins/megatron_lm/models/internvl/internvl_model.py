# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import logging
from collections import namedtuple
from functools import partial
from typing import List, Optional
import copy

import torch
from torch import Tensor

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.models.gpt import GPTModel
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_viewless_tensor
from megatron.core.transformer.spec_utils import ModuleSpec, build_module

from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.packed_seq_params import PackedSeqParams

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

import os
from megatron.training import get_args
if os.getenv("DIST_BACKEND", "easyllm") == "easyllm":
    from llm.utils.env import dist_env
elif os.getenv("DIST_BACKEND", "easyllm") == "megatron":
    from megatron.core import mpu as dist_env
from ..utils import (
    partition_uniform,
    partition_balanced
)


from ..pipeline_module import PipelineParallelModule

from .vision_embedding import VisionEmbedding
from .vision_transformer import VisionTransformerLayer
from .vision_extract_feat import VisionExtractFeat
from .word_embedings import EmbeddingPipe
from ..embeddings import RotaryEmbedding

# class InternVLModel(PipelineParallelModule):
# class InternVLModel(MegatronModule):
class InternVLModel(LanguageModule):
    def __init__(
        self,
        num_layers: int,
        num_vit_layers: int,

        vision_embedding_config: dict,
        vision_transformer_config: dict,
        vision_extract_feat_config: dict,
        drop_path_rate: float,

        word_embedding_config: dict,
        fp32_residual_connection: bool,

        language_transformer_config: TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,

        vocab_size: int,
        parallel_output: bool = True,
        language_position_embedding_type: str = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        seq_len_interpolation_factor: Optional[float] = None,
        
        share_embeddings_and_output_weights: bool = False,
        # language_max_sequence_length: int,
        # vision_embedding_layer_spec: ModuleSpec,
        # vision_transformer_layer_spec: ModuleSpec,
        # drop_vision_class_token: bool,
        # vision_projection_config: TransformerConfig,
        # vision_projection_layer_spec: ModuleSpec,
        # vision_projection_type: str = "mlp",
        # allow_missing_vision_projection_checkpoint: bool = False,
        # parallel_output: bool = True,
        # language_rotary_percent: float = 1.0,
        # pre_process: bool = True,
        # post_process: bool = True,
        # add_encoder: bool = True,
        # add_decoder: bool = True,
        # img_h: int = 336,
        # img_w: int = 336,
        # patch_dim: int = 14,
        # language_rotary_base: int = 10000,
        # img_embedding_idx: int = 0,
    ) -> None:
        # super().__init__(config=language_transformer_config)

        # if has_config_logger_enabled(language_transformer_config):
        #     log_config_to_disk(language_transformer_config, locals(), prefix=type(self).__name__)

        # logging.getLogger(__name__).warning(
        #     "LLaVA model is under development and may be missing features."
        # )

        # self.pre_process = pre_process
        # self.post_process = post_process
        # self.add_encoder = add_encoder
        # self.add_decoder = add_decoder
        # self.img_embedding_idx = img_embedding_idx

        # self.encoder_hidden_state = None
        # self.vision_model = None
        # self.vision_projection = None
        # self.language_model = None

        language_transformer_config.variable_seq_lengths = True
        # language_transformer_config.bias_dropout_fusion = False # DEBUG
        language_transformer_config.deallocate_pipeline_outputs = True
        # language_transformer_config.apply_rope_fusion = False # DEBUG
        super().__init__(config=language_transformer_config)

        # post init in layer build
        self.pre_process = False
        self.post_process = False

        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights

        self.num_vit_layers = num_vit_layers

        self.vision_embedding_config = vision_embedding_config
        self.vision_transformer_config = vision_transformer_config
        self.vision_extract_feat_config = vision_extract_feat_config
        self.drop_path_rate = drop_path_rate

        self.word_embedding_config = word_embedding_config
        self.fp32_residual_connection = fp32_residual_connection

        self.language_transformer_config = language_transformer_config
        self.language_transformer_layer_spec = language_transformer_layer_spec

        self.vocab_size = vocab_size
        self.parallel_output = parallel_output

        self.language_position_embedding_type = language_position_embedding_type
        # These 2 attributes are needed for TensorRT-LLM export.
        # self.max_position_embeddings = language_max_sequence_length
        self.rotary_percent = rotary_percent
        if self.language_position_embedding_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                use_cpu_initialization=self.config.use_cpu_initialization,
            )

        # partition
        self.module_list = self.build_module_list()
        # initialize partition
        self._partition_layers()
        self.forward_funcs = []
        self.build()

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

        from llm.plugins.megatron_lm.models.internvl.utils import load_ckpt_pretrained
        self.load_ckpt_pretrained = load_ckpt_pretrained

        self.model_kwargs = {"num_vit_layers": num_vit_layers, 
                             "num_layers": num_layers, 
                            #  "checkpoint_activations": checkpoint_activations
                            }

    def build_module_list(self):
        module_list = list()

        # vision embedding
        vision_embedding_param = dict(
            name="vision_embedding",
            type=VisionEmbedding,
            kwargs=self.vision_embedding_config
        )
        module_list.append(vision_embedding_param)

        # visoin transformer layer
        num_vit_layers = self.num_vit_layers
        # vision_transformer_layer_param = dict(
        #     type=VisionTransformerLayer,
        #     kwargs=self.vision_transformer_config
        # )
        self.vision_transformer_config["te_config"] = self.language_transformer_config
        vision_transformer_layer_param = dict(
            type=TEVisionTransformerLayer,
            kwargs=self.vision_transformer_config
        )
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, num_vit_layers)]
        for layer_idx in range(num_vit_layers):
            layer_param = copy.deepcopy(vision_transformer_layer_param)
            layer_param['name'] = f"vision_transformer_layer_{layer_idx}"
            layer_param['kwargs'].update({'vision_layer_number': layer_idx})
            layer_param['kwargs'].update({'drop_path_rate': dpr[layer_idx]})
            module_list.append(layer_param)

        # vision extract feat
        vision_extract_feat_param = dict(
            name="vision_extract_feat",
            type=VisionExtractFeat,
            kwargs=self.vision_extract_feat_config
        )
        module_list.append(vision_extract_feat_param)

        # word embedding
        self.word_embedding_config.update({"fp32_residual_connection": self.fp32_residual_connection})
        if torch.distributed.get_rank() == 0:
            print(self.word_embedding_config)
        word_embedding_param = dict(
            name="word_embedding",
            type=EmbeddingPipe,
            kwargs=self.word_embedding_config
        )
        module_list.append(word_embedding_param)

        # language transformer layer
        transformer_layer_params = dict(
            # name="language_transformer_layer",
            type=TransformerLayer,
            kwargs=dict(
                config=self.language_transformer_config,
                submodules=self.language_transformer_layer_spec.submodules,
            )
        )
        for layer_idx in range(self.language_transformer_config.num_layers):
            layer_param = copy.deepcopy(transformer_layer_params)
            layer_param['name'] = f"language_transformer_layer_{layer_idx}"
            layer_param['kwargs']["layer_number"] = layer_idx
            module_list.append(layer_param)

        # final layernorm after transformer layers
        layer_norm_params = dict(
            name="final_layernorm",
            type=LayerNormImpl,
            kwargs=dict(
                config=self.language_transformer_config,
                hidden_size=self.language_transformer_config.hidden_size,
                eps=self.language_transformer_config.layernorm_epsilon
            )
        )
        module_list.append(layer_norm_params)

        # lm head
        if self.language_transformer_config.defer_embedding_wgrad_compute:
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
                input_size=self.language_transformer_config.hidden_size,
                output_size=self.vocab_size,
                config=self.language_transformer_config,
                init_method=self.language_transformer_config.init_method,
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

    def set_input_tensor(self, input_tensor: Tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for vlm'

        self.input_tensor = input_tensor[0]

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_flags,
        labels: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
    ) -> torch.Tensor:

        """Forward function of the LLaVA model.

        Args:
            images (torch.Tensor): input image of shape [batch, img_h, img_w].
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): attention mask for the language model [batch, 1, combined_seq_len, combined_seq_len].
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            inference_params (InferenceParams): Inference-time parameters including KV cache.
        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape [b, s, vocab_size].
        """
        if not parallel_state.is_pipeline_first_stage():
            hidden_states = self.input_tensor

            # Viewless tensor.
            # - We only need to create a viewless tensor in the case of micro batch
            #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
            #   above creates a view tensor, and '.contiguous()' is a pass-through.
            #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
            #   the need to make it viewless.
            #
            #   However, we don't explicitly check mbs == 1 here because
            #   make_viewless_tensor() has negligible overhead when its input
            #   is already viewless.
            #
            # - For the 'else' case above, calling make_viewless_tensor() here is
            #   likely redundant, since p2p_communication.py (likely originator)
            #   already creates viewless tensors. That said, make_viewless_tensor()
            #   is called here to be future-proof and corner-case-proof.
            hidden_states = make_viewless_tensor(
                inp=hidden_states,
                requires_grad=True,
                keep_graph=True,
            )

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        vit_idx = 0
        llm_idx = 0
        rank = torch.distributed.get_rank()

        def custom_forward(forward_func, *args, is_recompute=True):
            if is_recompute:
                return tensor_parallel.checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    *args,
                )
            else:
                return forward_func(*args)

        # for layer in self.layers:
        for idx in range(len(self.forward_funcs)):
            module_name = self.module_list[self._local_start + idx]["name"]
            func = self.forward_funcs[idx]
            is_recompute = True # TODO: add condition check

            # if self._local_start + idx == 0:
            #     print(f"rank:{torch.distributed.get_rank()}, idx:{idx}, "
            #       f"module_name:{module_name}, hidden_states:{pixel_values.shape}")
            # else:
            #     print(f"rank:{torch.distributed.get_rank()}, idx:{idx}, "
            #         f"module_name:{module_name}, hidden_states:{hidden_states.shape}")

            if isinstance(func, VisionEmbedding):
                hidden_states = func(pixel_values)
                # hidden_states = custom_forward(func, pixel_values)
                # if torch.distributed.get_rank() == 0:
                #     torch.save(hidden_states, 'data/rank0_vit_embed.pt')

            elif isinstance(func, VisionTransformerLayer):
                hidden_states = func(hidden_states)
                # hidden_states = custom_forward(func, hidden_states)
                # if torch.distributed.get_rank() == 0:
                #     torch.save(hidden_states, f'data/rank0_vit_transformer_{vit_idx}.pt')
                #     vit_idx += 1

            elif isinstance(func, VisionExtractFeat):
                hidden_states = func(hidden_states)
                # if torch.distributed.get_rank() == 0:
                #     torch.save(hidden_states, 'data/rank0_vit_extract_feat.pt')

            elif isinstance(func, EmbeddingPipe):
                vit_embedding = hidden_states
                hidden_states = func(vit_embedding, input_ids, position_ids, image_flags)
                # if torch.distributed.get_rank() == 0:
                #     torch.save(hidden_states, 'data/rank0_word_embedding.pt')

            elif isinstance(func, TransformerLayer):
                if self.language_position_embedding_type == 'rope':
                    rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                        inference_params, self.forward_funcs[idx], hidden_states, self.config
                    )
                    rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

                hidden_states, _ = self.forward_funcs[idx](
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    inference_params=inference_params,
                    packed_seq_params=packed_seq_params,
                )
                # if torch.distributed.get_rank() == 0:
                #     torch.save(hidden_states, f'data/llm_transformer_{llm_idx}.pt')
                #     llm_idx += 1
                # llm_idx = self.module_list[self._local_start + idx]['kwargs']["layer_number"]
                # torch.save(hidden_states, f'data/rank{rank}_llm_transformer_{llm_idx}.pt')


            elif module_name == "final_layernorm":
                hidden_states = func(hidden_states)
                # if torch.distributed.get_rank() == 7:
                #     torch.save(hidden_states, 'data/final_layernorm.pt')

            elif module_name == "lm_head":
                # logits and loss
                # output_weight = None
                # if self.share_embeddings_and_output_weights:
                #     output_weight = self.shared_embedding_or_output_weight()
                logits, _ = self.forward_funcs[idx](hidden_states, weight=None)

                # if torch.distributed.get_rank() == 7:
                #     torch.save(logits, 'data/logits.pt')
                    # import pdb;pdb.set_trace()

                if labels is None:
                    # [s b h] => [b s h]
                    return logits.transpose(0, 1).contiguous()

                loss = self.compute_language_model_loss(labels, logits)
                # if torch.distributed.get_rank() == 7:
                #     torch.save(logits, 'data/loss.pt')
                return loss

        return hidden_states

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
        
    def _partition_balanced(self, weights, num_parts):
        """
        use dynamic programming solve `The Linear Partition Problem`.
        see https://www8.cs.umu.se/kurser/TDBAfl/VT06/algorithms/BOOK/BOOK2/NODE45.HTM
        """
        import numpy as np
        n = len(weights)
        m = num_parts

        if n <= m:
            return partition_uniform(n, m)

        dp_max = np.full((n + 1, m + 1), np.inf)
        dp_min = np.full((n + 1, m + 1), np.inf)
        dp_cost = np.full((n + 1, m + 1), np.inf)
        position = np.zeros((n + 1, m + 1), dtype=int)
        prefix_sum = np.zeros((n + 1))
        prefix_sum[1:] = np.cumsum(weights)

        dp_max[0, 0] = 0
        dp_cost[0, 0] = 0
        for i in range(1, n + 1):
            for j in range(1, min(i, m) + 1):
                for k in range(i):
                    max_sum = max(dp_max[k, j - 1], prefix_sum[i] - prefix_sum[k])
                    min_sum = min(dp_min[k, j - 1], prefix_sum[i] - prefix_sum[k])
                    cost = max_sum - min_sum
                    if dp_cost[i, j] >= cost:
                        dp_cost[i, j] = cost
                        dp_max[i, j] = max_sum
                        dp_min[i, j] = min_sum
                        position[i, j] = k

        parts = [n]
        for i in reversed(range(1, m + 1)):
            parts.append(position[parts[-1], i])
        parts.reverse()

        return parts

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
            self.parts = self._partition_balanced(weights=param_counts, num_parts=num_stages)
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
        if torch.distributed.get_rank() == 0:
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
        print(f'rank:{torch.distributed.get_rank()}, self.parts:{self.parts}')
        # if torch.distributed.get_rank() == 0:
        #     import pdb;pdb.set_trace()

    def build(self):
        for local_idx, layer in enumerate(self.part_module_list):
            layer_idx = local_idx + self._local_start

            module_type = layer['type']
            module_kwargs = layer['kwargs']
            # if layer['name'] == "lm_head":
            #     if self.post_process:
            #         module_kwargs['skip_weight_param_allocation'] = self.pre_process and self.share_embeddings_and_output_weights
            #     else:
            #         continue
            module = module_type(**module_kwargs)
            # name = str(layer_idx)
            # name = layer['name'] + '_' + str(layer_idx)
            name = layer['name']
            self.forward_funcs.append(module)
            self.add_module(name, module)

            if layer['name'] == "word_embedding":
                self.pre_process = True
                self.embedding = module

            if layer['name'] == "lm_head":
                self.post_process = True
                self.output_layer = module
