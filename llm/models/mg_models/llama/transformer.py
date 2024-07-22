# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer."""
try:
    from .transformer_engine import TEDotProductAttention
except:    # noqa
    TEDotProductAttention = None

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from llm.utils.env import dist_env
from ..base_modules.modules.meg_module import MegatronModule
from ..base_modules.modules.enums import AttnMaskType, AttnType, PositionEmbeddingType
from ..base_modules.layers.fused_layer_norm import build_layer_norm
from ..base_modules.layers.fused_softmax import FusedScaleMaskSoftmax
from ..base_modules.utils import (attention_mask_func,
                                  openai_gelu,
                                  erf_gelu,
                                  get_initializer_from_cfg,
                                  get_torch_dtype,
                                  get_layer_type,
                                  get_attn_mask_type,
                                  get_position_embedding_type)
from ..base_modules.layers import ColumnParallelLinear, RowParallelLinear
import deepspeed

from ..base_modules.layers.glu_activations import GLU_ACTIVATIONS
from .positional_embeddings import (RotaryEmbedding,
                                    apply_rotary_pos_emb_torch,
                                    apply_rotary_pos_emb,
                                    InternLM2DynamicNTKScalingRotaryEmbedding,
                                    LlamaDynamicYaRNScaledRotaryEmbedding)

from llm.models.hf_models.utils.flash_utils import FlashRotaryEmbedding
# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


try:
    from einops import rearrange
except ImportError:
    rearrange = None

if os.environ.get('ACCELERATOR_BACKEND') == 'CUDA':
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func, flash_attn_varlen_qkvpacked_func
        from flash_attn.bert_padding import unpad_input, pad_input
    except ImportError:
        flash_attn_varlen_kvpacked_func, flash_attn_varlen_qkvpacked_func = None, None
        unpad_input, pad_input = None, None
elif os.environ.get('ACCELERATOR_BACKEND') == 'TORCH_NPU':
    from .npu_flash import flash_attn_varlen_kvpacked_func, flash_attn_varlen_qkvpacked_func
    from .npu_flash import unpad_input, pad_input
elif os.environ.get('ACCELERATOR_BACKEND') == 'DEEPLINK_DIPU':
    from deeplink_ext.easyllm_ops import flash_attn_varlen_kvpacked_func, flash_attn_varlen_qkvpacked_func
    from deeplink_ext.easyllm_ops.bert_padding import unpad_input, pad_input
else:
    print("no backend support")


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    if n_rep == 1:
        return hidden_states

    # batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if len(hidden_states.shape) == 3:
        slen, num_key_value_heads, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :].expand(slen, num_key_value_heads, n_rep, head_dim)
        return hidden_states.reshape(slen, num_key_value_heads * n_rep, head_dim)
    elif len(hidden_states.shape) == 4:
        slen, bs, num_key_value_heads, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, :, None, :].expand(slen, bs, num_key_value_heads, n_rep, head_dim)
        return hidden_states.reshape(slen, bs, num_key_value_heads * n_rep, head_dim)


class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        assert flash_attn_varlen_qkvpacked_func is not None, ('Please install FlashAttention first, ' 'e.g., with pip install flash-attn')  # noqa
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.causal = causal

    def forward(self, q, k, v, key_padding_mask=None, cu_seqlens=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
                if unpadded: (nnz, 3, h, d)
            key_padding_mask: a bool tensor of shape (B, S)
        """
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda
        batch_size = q.shape[0]
        seqlen = q.shape[1]
        if key_padding_mask is None:
            q = rearrange(q, 'b s ... -> (b s) ...')
            k = rearrange(k, 'b s ... -> (b s) ...')
            v = rearrange(v, 'b s ... -> (b s) ...')
            max_s = seqlen
            if cu_seqlens is None:
                if os.environ.get('ACCELERATOR_BACKEND', 'CUDA') != 'CUDA':
                    cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32)
                else:
                    cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=q.device)
            else:
                cu_seqlens = cu_seqlens.view(-1).int()
            if k.shape[-2] == q.shape[-2]:
                qkv = torch.stack([q, k, v], dim=1)
                output = flash_attn_varlen_qkvpacked_func(
                    qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=self.causal
                )
            else:
                kv = torch.stack([k, v], dim=1)
                output = flash_attn_varlen_kvpacked_func(
                    q, kv, cu_seqlens, cu_seqlens, max_s, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=self.causal
                )
            output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        else:
            nheads = q.shape[-2]
            q = rearrange(q, 'b s h d -> b s (h d)')
            k = rearrange(k, 'b s h d -> b s (h d)')
            v = rearrange(v, 'b s h d -> b s (h d)')
            q_unpad, indices_q, cu_seqlens_q, max_s_q = unpad_input(q, key_padding_mask)
            k_unpad, _, cu_seqlens_k, max_s_k = unpad_input(k, key_padding_mask)
            v_unpad, _, _, _ = unpad_input(v, key_padding_mask)
            q_unpad = rearrange(q_unpad, 'nnz (h d) -> nnz h d', h=nheads)
            k_unpad = rearrange(k_unpad, 'nnz (h d) -> nnz h d', h=nheads)
            v_unpad = rearrange(v_unpad, 'nnz (h d) -> nnz h d', h=nheads)
            if q.shape[-2] == k.shape[-2]:
                qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
                output = flash_attn_varlen_qkvpacked_func(
                    qkv_unpad, cu_seqlens_q, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=self.causal)
            else:
                kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
                output_unpad = flash_attn_varlen_kvpacked_func(
                    q_unpad, kv_unpad, cu_seqlens_q, cu_seqlens_k, max_s_q, max_s_k,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=self.causal
                )
            output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                         indices_q, batch_size, seqlen), 'b s (h d) -> b s h d', h=nheads)

        return output


class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(self,
                 init_method,
                 output_layer_init_method,
                 hidden_size=1024,
                 intermediate_size=4096,
                 use_cpu_initialization=False,
                 params_dtype=torch.half,
                 bias_gelu_fusion=False,
                 glu_activation=None,
                 use_openai_gelu=False,  # openai_gelu
                 onnx_safe=False,
                 sync_tp_duplicated_parameters=True,
                 sequence_parallel=False,
                 ):
        super(ParallelMLP, self).__init__()

        self.sequence_parallel = sequence_parallel
        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            # GLU is a special activation that divides the dimension by a factor 2.
            intermediate_size,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            bias=False,
            use_cpu_initialization=use_cpu_initialization,
            params_dtype=params_dtype,
            sequence_parallel=sequence_parallel)

        self.up_proj = ColumnParallelLinear(
            hidden_size,
            # GLU is a special activation that divides the dimension by a factor 2.
            intermediate_size,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            bias=False,
            use_cpu_initialization=use_cpu_initialization,
            params_dtype=params_dtype,
            sequence_parallel=sequence_parallel)

        self.activation_func = F.gelu
        if glu_activation:
            self.activation_func = GLU_ACTIVATIONS[glu_activation]
        elif use_openai_gelu:
            self.activation_func = openai_gelu
        elif onnx_safe:
            self.activation_func = erf_gelu

        # Project back to h.
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            bias=False,
            use_cpu_initialization=use_cpu_initialization,
            params_dtype=params_dtype,
            sync_tp_duplicated_parameters=sync_tp_duplicated_parameters,
            sequence_parallel=sequence_parallel)

    def forward(self, hidden_states):

        # [s, b, 4hp]
        h1, _ = self.gate_proj(hidden_states)
        h2, _ = self.up_proj(hidden_states)
        h1 = self.activation_func(h1) * h2
        # [s, b, h]
        output, output_bias = self.down_proj(h1)
        return output, output_bias


class ParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self,
                 init_method,
                 output_layer_init_method,
                 layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.padding,
                 fp16=False,
                 bf16=True,
                 position_embedding_type=None,
                 position_embedding_kwargs=None,
                 apply_query_key_layer_scaling=False,
                 attention_softmax_in_fp32=False,
                 kv_channels=0,
                 num_attention_heads=1,
                 num_kv_attention_heads=1,
                 hidden_size=2048,
                 masked_softmax_fusion=False,
                 attention_dropout=0,
                 params_dtype=torch.half,
                 use_cpu_initialization=False,
                 sync_tp_duplicated_parameters=True,
                 sequence_parallel=False,
                 use_flash_attn=False,
                 use_matmul=False,
                 qkv_pack=False,
                 qkv_bias=False,
                 o_bias=False,
                 transformer_engine_config=None,
                 te_layer_number=1,
                 te_attention_type="self",
                 te_attention_dropout=0.0):
        super(ParallelAttention, self).__init__()
        self.fp16 = fp16
        self.bf16 = bf16
        self.use_matmul = use_matmul
        self.position_embedding_type = position_embedding_type

        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.sequence_parallel = sequence_parallel

        self.use_flash_attn = use_flash_attn
        if self.use_flash_attn:
            assert attention_type == AttnType.self_attn, ('FlashAttention code path only supports '
                                                          'self-attention for now')
            if rearrange is None:
                raise ImportError('einops is not installed, please install with pip install einops')
        projection_size = kv_channels * num_attention_heads
        projection_size_kv = kv_channels * num_kv_attention_heads

        # Per attention head and per partition values.
        def partion_attention_head(proj_size, num_heads):
            world_size = dist_env.get_tensor_model_parallel_world_size()
            assert proj_size % world_size == 0, '{} is not divisible by {}'.format(
                proj_size, world_size)
            hidden_size_per_partition = proj_size // world_size
            assert proj_size % num_heads == 0, '{} is not divisible by {}'.format(
                proj_size, num_heads)
            hidden_size_per_attention_head = proj_size // num_heads
            assert num_heads % world_size == 0, '{} is not divisible by {}'.format(
                num_heads, world_size)
            num_attention_heads_per_partition = num_heads // world_size
            return hidden_size_per_partition, hidden_size_per_attention_head, num_attention_heads_per_partition

        self.hidden_size_per_partition, self.hidden_size_per_attention_head, self.num_attention_heads_per_partition = \
            partion_attention_head(projection_size, num_attention_heads)

        self.hidden_size_kv_per_partition, self.hidden_size_per_kv_attention_head, self.num_kv_attention_heads_per_partition = partion_attention_head(projection_size_kv, num_kv_attention_heads)  # noqa

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            self.qkv_pack = qkv_pack
            self.q_size = projection_size // dist_env.get_tensor_model_parallel_world_size()
            self.kv_size = projection_size_kv // dist_env.get_tensor_model_parallel_world_size()
            if not self.qkv_pack:
                self.q_proj = ColumnParallelLinear(
                    hidden_size,
                    projection_size,
                    gather_output=False,
                    init_method=init_method,
                    bias=qkv_bias,
                    use_cpu_initialization=use_cpu_initialization,
                    params_dtype=params_dtype,
                    sequence_parallel=sequence_parallel)

                self.k_proj = ColumnParallelLinear(
                    hidden_size,
                    projection_size_kv,
                    gather_output=False,
                    init_method=init_method,
                    bias=qkv_bias,
                    use_cpu_initialization=use_cpu_initialization,
                    params_dtype=params_dtype,
                    sequence_parallel=sequence_parallel)

                self.v_proj = ColumnParallelLinear(
                    hidden_size,
                    projection_size_kv,
                    gather_output=False,
                    init_method=init_method,
                    bias=qkv_bias,
                    use_cpu_initialization=use_cpu_initialization,
                    params_dtype=params_dtype,
                    sequence_parallel=sequence_parallel)
            else:
                self.wqkv = ColumnParallelLinear(
                    hidden_size,
                    projection_size + projection_size_kv * 2,
                    gather_output=False,
                    init_method=init_method,
                    bias=qkv_bias,
                    use_cpu_initialization=use_cpu_initialization,
                    params_dtype=params_dtype,
                    sequence_parallel=sequence_parallel)
        else:
            raise NotImplementedError("Not implementented for cross-attention")

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        # if unflash_attn, self.apply_query_key_layer_scaling = False
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16, self.bf16,
            self.attn_mask_type,
            masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout)

        if self.use_flash_attn:
            self.core_attention_flash = FlashAttention(
                causal=True, attention_dropout=attention_dropout
            )
        # Output.
        self.o_proj = RowParallelLinear(
            projection_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            bias=o_bias,
            use_cpu_initialization=use_cpu_initialization,
            params_dtype=params_dtype,
            sync_tp_duplicated_parameters=sync_tp_duplicated_parameters,
            sequence_parallel=sequence_parallel)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

        if self.position_embedding_type == PositionEmbeddingType.rotary:
            self.rotary_emb = RotaryEmbedding(
                self.hidden_size_per_attention_head,
                precision=params_dtype,
                **position_embedding_kwargs or {},
            )
        elif self.position_embedding_type == PositionEmbeddingType.flash:
            self.rotary_emb = FlashRotaryEmbedding(dim=self.hidden_size_per_attention_head)
        elif self.position_embedding_type == PositionEmbeddingType.dynamicntk:
            self.rotary_emb = InternLM2DynamicNTKScalingRotaryEmbedding(
                self.hidden_size_per_attention_head,
                precision=params_dtype,
                **position_embedding_kwargs or {}
            )
        elif self.position_embedding_type == PositionEmbeddingType.yarn:
            self.rotary_emb = LlamaDynamicYaRNScaledRotaryEmbedding(
                self.hidden_size_per_attention_head,
                precision=params_dtype,
                **position_embedding_kwargs or {},
            )

        if TEDotProductAttention is not None:
            self.te_core_attention = TEDotProductAttention(
                config=transformer_engine_config,
                layer_number=te_layer_number,
                attn_mask_type=AttnMaskType(self.attn_mask_type),
                attention_type=te_attention_type,
                attention_dropout=te_attention_dropout
            )
        else:
            assert dist_env.get_context_parallel_world_size() == 1, "context_parallel_size must be 1!"  # noqa
            self.te_attention_dropout = None

    def forward(self,
                hidden_states,
                attention_mask,
                cu_seqlens=None,
                position_ids=None,
                layer_past=None,
                get_key_value=False,
                alibi=None):
        # hidden_states: [sq, b, h]
        assert self.attention_type == AttnType.self_attn
        # sq, b, np * hn
        if not self.qkv_pack:
            query_layer, _ = self.q_proj(hidden_states)
            key_layer, _ = self.k_proj(hidden_states)
            value_layer, _ = self.v_proj(hidden_states)
        else:
            qkv_layer, _ = self.wqkv(hidden_states)
            query_layer = qkv_layer[..., :self.q_size]
            key_layer = qkv_layer[..., self.q_size:(self.q_size + self.kv_size)]
            value_layer = qkv_layer[..., (self.q_size + self.kv_size):]
        bs, sq, sk = query_layer.shape[1], query_layer.shape[0], key_layer.shape[0]
        nq_head = self.num_attention_heads_per_partition
        nk_head = self.num_kv_attention_heads_per_partition
        # Rotary embeddings
        if self.position_embedding_type == PositionEmbeddingType.rotary or \
           self.position_embedding_type == PositionEmbeddingType.dynamicntk or self.position_embedding_type == PositionEmbeddingType.yarn:
            # [sq, b, np, hn] -> [sq, b * np, hn]
            query_layer = query_layer.reshape(sq, bs * nq_head, -1)
            # [sk, b, np, hn] -> [sk, b * np, hn]
            key_layer = key_layer.reshape(sk, bs * nk_head, -1)
            value_layer = value_layer.reshape(sk, bs * nk_head, -1)
            apply_rotary_fn = apply_rotary_pos_emb_torch if self.bf16 else apply_rotary_pos_emb

            seq_len = key_layer.shape[0]
            offset = 0
            if layer_past is not None and layer_past.numel() > 0:
                offset = layer_past[0].shape[0]
                seq_len += offset
            cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)
            query_layer, key_layer = apply_rotary_fn(query_layer, key_layer, cos, sin, offset=offset, position_ids=position_ids)  # noqa
        elif self.position_embedding_type == PositionEmbeddingType.flash:
            query_layer = query_layer.reshape(sq, bs, nq_head, -1)
            key_layer = key_layer.reshape(sk, bs, nk_head, -1)
            value_layer = value_layer.reshape(sk, bs, nk_head, -1)
            # [sq, b, np, hn] -> [b, sq, np, hn]
            query_layer = rearrange(query_layer, "s b h d -> b s h d")
            # [sk, b, np, hn] -> [b, sk, np, hn]
            key_layer = rearrange(key_layer, "s b h d -> b s h d")
            value_layer = rearrange(value_layer, "s b h d -> b s h d")
            query_len = query_layer.shape[1]
            key_len = key_layer.shape[1]

            offset = 0
            if layer_past is not None and layer_past.numel() > 0:
                offset = layer_past[0].shape[0]
                key_len += offset
            max_seqlen = None if key_len == query_len else max(key_len, query_len)

            query_layer, key_layer = self.rotary_emb(query_layer,
                                                     key_layer,
                                                     max_seqlen=max_seqlen,
                                                     seqlen_offset=offset)

        self.context_parallel_size = dist_env.get_context_parallel_world_size()
        if self.context_parallel_size > 1:
            query_layer = query_layer.view(sq, bs, nq_head, -1)
            key_layer = key_layer.view(sk, bs, nk_head, -1)
            value_layer = value_layer.view(sk, bs, nk_head, -1)
            context_layer = self.te_core_attention(
                query_layer,
                key_layer,
                value_layer,
                attention_mask,
                attn_mask_type=AttnMaskType(self.attn_mask_type),
            )
        elif self.use_flash_attn:
            if self.position_embedding_type != PositionEmbeddingType.flash:
                # [sq, b * np, hn] --> [sq, b, np, hn]
                query_layer = query_layer.reshape(sq, bs, nq_head, -1)
                key_layer = key_layer.reshape(sk, bs, nk_head, -1)
                value_layer = value_layer.reshape(sk, bs, nk_head, -1)
                query_layer, key_layer, value_layer = [rearrange(x, 's b ... -> b s ...').contiguous() for x in (query_layer, key_layer, value_layer)]  # noqa

            if attention_mask is None:
                qk_mask = None
            else:
                if len(attention_mask.shape) == 2:
                    qk_mask = attention_mask
                else:
                    qk_mask = ~attention_mask[:, 0, :, 0]
                if cu_seqlens is not None:
                    qk_mask = None
            if alibi is None:
                matmul_result = None
            with dist_env.get_cuda_rng_tracker().fork():
                context_layer = self.core_attention_flash(query_layer, key_layer, value_layer, qk_mask, cu_seqlens)
            context_layer = rearrange(context_layer, 'b s h d -> s b (h d)').contiguous()
        else:
            if nq_head != nk_head:
                n_rep = nq_head // nk_head
                key_layer = repeat_kv(key_layer, n_rep)

            if self.use_matmul:
                # align peft
                matmul_result = torch.matmul(
                    query_layer.transpose(0, 1), key_layer.transpose(0, 1).transpose(1, 2)
                ) / self.norm_factor
            else:
                # preallocting result tensor: [b * np, sq, sk]
                if alibi is None:
                    matmul_result = torch.empty(
                        bs * nq_head,
                        sq,
                        sk,
                        dtype=query_layer.dtype,
                        device=torch.cuda.current_device())
                    beta = 0.0
                else:
                    matmul_result = alibi[:bs * nq_head, :, :sk]
                    if not hasattr(self, "logged_alibi"):
                        print("Using Alibi.")
                        self.logged_alibi = True

                    if self.apply_query_key_layer_scaling:
                        beta = 1.0 / self.layer_number
                    else:
                        beta = 1.0

                # Raw attention scores. [b * np, sq, sk]
                matmul_result = torch.baddbmm(
                    matmul_result,
                    query_layer.transpose(0, 1),  # [b * np, sq, hn]
                    key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                    beta=beta, alpha=(1.0 / self.norm_factor))
                value_layer = value_layer.reshape(sk, bs, nk_head, -1)
                if nq_head != nk_head:
                    value_layer = repeat_kv(value_layer, n_rep)

            # change view to [b, np, sq, sk]
            attention_scores = matmul_result.view(bs, nq_head, sq, sk)
            # ==================================================
            # Update attention mask for inference. [b, np, sq, sk]
            # ==================================================

            if get_key_value:
                with torch.no_grad():
                    # TODO @thomasw21 Handle case where `attention_mask` is None
                    if layer_past is not None:
                        attention_mask = attention_mask[
                            ...,
                            attention_scores.size(3) - 1,
                            :attention_scores.size(3)].unsqueeze(2)
                    else:
                        attention_mask = attention_mask[
                            ...,
                            :attention_scores.size(3),
                            :attention_scores.size(3)]

            # ===========================
            # Attention probs and dropout
            # ===========================

            # attention scores and attention mask [b, np, sq, sk]
            # attention_probs = self.scale_mask_softmax(attention_scores.clone(), None) # attention_mask)
            batch_size, seq_length = hidden_states.shape[1], hidden_states.shape[0]
            past_key_values_length = 0
            from .attn_mask_utils import _prepare_4d_causal_attention_mask
            attention_mask = _prepare_4d_causal_attention_mask(
                None, (batch_size, seq_length), hidden_states.transpose(0, 1), past_key_values_length
            )
            attention_scores = attention_scores + attention_mask
            attention_probs = nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query_layer.dtype)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            with dist_env.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)

            # =========================
            # Context layer. [sq, b, hp]
            # =========================

            # value_layer -> context layer.
            # [sk, b, np, hn] --> [b, np, sq, hn]

            # context layer shape: [b, np, sq, hn]
            output_size = (value_layer.size(1),
                           value_layer.size(2),
                           query_layer.size(0),
                           value_layer.size(3))

            # change view [sk, b * np, hn]
            value_layer = value_layer.reshape(value_layer.size(0),
                                              output_size[0] * output_size[1], -1)

            # change view [b * np, sq, sk]
            attention_probs = attention_probs.reshape(output_size[0] * output_size[1],
                                                      output_size[2], -1)

            # matmul: [b * np, sq, hn]
            context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

            # change view [b, np, sq, hn]
            context_layer = context_layer.reshape(*output_size)

            # [b, np, sq, hn] --> [sq, b, np, hn]
            context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

            # [sq, b, np, hn] --> [sq, b, hp]
            new_context_layer_shape = context_layer.size()[:-2] + \
                (self.hidden_size_per_partition,)
            context_layer = context_layer.reshape(*new_context_layer_shape)

        # Output. [sq, b, h]
        output, bias = self.o_proj(context_layer)

        return output, bias


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor # noqa
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)
    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(x, bias, residual, prob):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor # noqa
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x, bias, residual, prob):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor # noqa
    return bias_dropout_add(x, bias, residual, prob, False)


class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(self,
                 layer_number,
                 layer_type='encoder',
                 self_attn_mask_type='padding',
                 position_embedding_type=None,
                 position_embedding_kwargs=None,
                 params_dtype='half',
                 layer_norm=None,
                 fp16=False,
                 bf16=True,
                 fp32_residual_connection=False,
                 apply_residual_connection_post_layernorm=False,
                 apply_query_key_layer_scaling=False,
                 attention_softmax_in_fp32=False,
                 kv_channels=0,
                 num_attention_heads=0,
                 num_kv_attention_heads=0,
                 hidden_size=2048,
                 masked_softmax_fusion=False,
                 attention_dropout=0,
                 use_cpu_initialization=None,
                 bias_gelu_fusion=False,
                 glu_activation=None,
                 use_openai_gelu=False,  # openai_gelu
                 onnx_safe=None,
                 sync_tp_duplicated_parameters=False,
                 hidden_dropout=0,
                 bias_dropout_fusion=False,
                 intermediate_size=4096,
                 seq_length=1024,
                 micro_batch_size=1,
                 sequence_parallel=False,
                 use_flash_attn=False,
                 initializer=None,
                 output_initializer=None,
                 use_matmul=False,
                 qkv_pack=False,
                 attention_qkv_bias=False,
                 attention_o_bias=False,
                 transformer_engine_config=None,
                 ):

        super(ParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number
        self.layer_type = get_layer_type(layer_type)
        position_embedding_type = get_position_embedding_type(position_embedding_type)
        params_dtype = get_torch_dtype(params_dtype)

        self.apply_residual_connection_post_layernorm \
            = apply_residual_connection_post_layernorm

        self.bf16 = bf16
        self.fp32_residual_connection = fp32_residual_connection
        self.sequence_parallel = sequence_parallel

        # Layernorm on the input data.
        self.input_layernorm = build_layer_norm(layer_norm)

        default_parallel_attn_params = dict(
            init_method=get_initializer_from_cfg(initializer),
            output_layer_init_method=get_initializer_from_cfg(output_initializer),
            layer_number=layer_number,
            fp16=fp16,
            bf16=bf16,
            position_embedding_type=position_embedding_type,
            position_embedding_kwargs=position_embedding_kwargs,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            attention_softmax_in_fp32=attention_softmax_in_fp32,
            kv_channels=kv_channels,
            num_attention_heads=num_attention_heads,
            num_kv_attention_heads=num_kv_attention_heads,
            hidden_size=hidden_size,
            masked_softmax_fusion=masked_softmax_fusion,
            attention_dropout=attention_dropout,
            params_dtype=params_dtype,
            use_cpu_initialization=use_cpu_initialization,
            sync_tp_duplicated_parameters=sync_tp_duplicated_parameters,
            sequence_parallel=sequence_parallel,
            use_flash_attn=use_flash_attn,
            use_matmul=use_matmul,
            qkv_pack=qkv_pack,
            qkv_bias=attention_qkv_bias,
            o_bias=attention_o_bias
        )

        # Self attention.
        self.self_attn = ParallelAttention(
            attention_type=AttnType.self_attn,
            attn_mask_type=get_attn_mask_type(self_attn_mask_type),
            transformer_engine_config=transformer_engine_config,
            **default_parallel_attn_params)
        self.hidden_dropout = hidden_dropout
        self.bias_dropout_fusion = bias_dropout_fusion

        # Layernorm on the attention output
        self.post_attention_layernorm = build_layer_norm(layer_norm)

        mlp_params = dict(
            init_method=get_initializer_from_cfg(initializer),
            output_layer_init_method=get_initializer_from_cfg(output_initializer),
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cpu_initialization=use_cpu_initialization,
            params_dtype=params_dtype,
            bias_gelu_fusion=bias_gelu_fusion,
            glu_activation=glu_activation,
            use_openai_gelu=use_openai_gelu,  # openai_gelu
            onnx_safe=onnx_safe,
            sync_tp_duplicated_parameters=sync_tp_duplicated_parameters,
            sequence_parallel=sequence_parallel,
        )
        # MLP
        self.mlp = ParallelMLP(**mlp_params)

        # Alibi
        if position_embedding_type == PositionEmbeddingType.alibi:
            self.alibi = self._build_alibi_tensor(
                seq_length, num_attention_heads, micro_batch_size).to(
                torch.cuda.current_device())
            if params_dtype == torch.float16:
                self.alibi = self.alibi.to(torch.float16)
            elif params_dtype == torch.bfloat16:
                self.alibi = self.alibi.to(torch.bfloat16)
        else:
            self.alibi = None

    def forward(self,
                hidden_states,
                attention_mask,
                cu_seqlens=None,
                position_ids=None,
                layer_past=None,
                get_key_value=False):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, _ = \
            self.self_attn(layernorm_output,
                           attention_mask,
                           cu_seqlens,
                           position_ids,
                           layer_past=layer_past,
                           get_key_value=get_key_value,
                           alibi=self.alibi)

        if get_key_value:
            attention_output, presents = attention_output

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = attention_output + residual
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # MLP.
        mlp_output, _ = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = mlp_output + residual

        if get_key_value:
            output = [output, presents]

        return output

    @staticmethod
    def _build_alibi_tensor(max_seq_len, num_attention_heads, batch_size):
        # Based on
        # https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
        """Returns tensor shaped (batch_size * num_attention_heads, 1, max_seq_len)"""

        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = (2 ** (-2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio ** i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                    :n - closest_power_of_2]

        slopes = torch.Tensor(get_slopes(num_attention_heads))
        alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0).expand(
            num_attention_heads, -1, -1)

        # Select the part of the tensor that corresponds to our tensor parallel index.
        tp_world_size = dist_env.get_tensor_model_parallel_world_size()
        tp_index = dist_env.get_tensor_model_parallel_rank()
        alibi = alibi.reshape((tp_world_size, -1, *alibi.shape[1:]))[tp_index]

        alibi = alibi.repeat(batch_size, 1, 1)
        return alibi


class ParallelTransformerLayerPipe(ParallelTransformerLayer):
    """Extends ParallelTransformerLayer to forward attention_mask through the pipeline.

    Forward has two usages that affect attention mask communication:

    1) forward((input, attn_mask) , **kwargs) -> (output, mask)
       When the attention mask is provided as the second positional
       argument, typical pipeline behavior is used and both the output
       *and* mask are returned in a tuple. This tuple is then forwarded
       to the next stage in the pipeline.

       This version is useful if masks are dynamic.

    2) forward(input, **kwargs) -> output
       When the mask is static over all samples, it is advantageous to
       cache the mask and avoid communicating it.
    """

    def forward(self, inputs, **kwargs):
        assert torch.is_tensor(inputs) or isinstance(inputs, tuple)
        if torch.is_tensor(inputs) or len(inputs) == 1:
            hidden_states, attention_mask = inputs, None
            return super().forward(hidden_states, attention_mask, **kwargs)
        elif len(inputs) == 2:
            # Attention mask is an activation.
            hidden_states, attention_mask = inputs[0], inputs[1]
            return super().forward(*inputs, **kwargs), attention_mask
        elif len(inputs) == 4:
            hidden_states, attention_mask, cu_seqlens, position_ids = inputs[0], inputs[1], inputs[2], inputs[3]
            return super().forward(*inputs, **kwargs), attention_mask, cu_seqlens, position_ids
        else:
            raise RuntimeError('Received more inputs than understood.')
