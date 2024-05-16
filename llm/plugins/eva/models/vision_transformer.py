import torch
import torch.nn as nn
from typing import Any, Optional, Tuple, Union

from transformers.activations import ACT2FN
from llm.models.mg_models.base_modules.modules.meg_module import MegatronModule
from llm.models.mg_models.base_modules.utils import get_torch_dtype

from llm.utils.env import dist_env
from llm.models.mg_models.base_modules.layers import ColumnParallelLinear, RowParallelLinear
from llm.models.mg_models.base_modules.layers.fused_layer_norm import build_layer_norm
from llm.models.mg_models.llama.transformer import FlashAttention

try:
    from einops import rearrange
except ImportError:
    rearrange = None


class EVAAttention(MegatronModule):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_dropout,
        use_flash_attn=False,
        params_dtype=torch.half,
        sequence_parallel=False,
        qv_bias=False
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(attention_dropout)
        self.use_flash_attn = use_flash_attn
        self.sequence_parallel = sequence_parallel

        self.qkv = ColumnParallelLinear(
            self.embed_dim,
            3 * self.embed_dim,
            gather_output=False,
            bias=False,
            params_dtype=params_dtype,
            sequence_parallel=sequence_parallel
        )

        if qv_bias:
            q_bias = nn.Parameter(torch.zeros(self.embed_dim))
            v_bias = nn.Parameter(torch.zeros(self.embed_dim))
            self.register_parameter("q_bias", q_bias)
            self.register_parameter("v_bias", v_bias)
        else:
            q_bias = None
            v_bias = None

        if q_bias is not None:
            qkv_bias = torch.cat((q_bias, torch.zeros_like(v_bias, requires_grad=False), v_bias))
            self.qkv.bias = nn.Parameter(qkv_bias)

        self.projection = RowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            input_is_parallel=True,
            bias=True,
            params_dtype=params_dtype,
            sequence_parallel=sequence_parallel
        )

        if self.use_flash_attn:
            self.core_attention_flash = FlashAttention(
                causal=True, attention_dropout=attention_dropout
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None
    ):
        bsz, tgt_len, embed_dim = hidden_states.size()

        if self.sequence_parallel:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
            tgt_len *= dist_env.get_tensor_model_parallel_world_size()
        mixed_qkv, _ = self.qkv(hidden_states)
        if self.sequence_parallel:
            mixed_qkv = mixed_qkv.transpose(0, 1).contiguous()
        mixed_qkv = mixed_qkv.reshape(bsz, tgt_len, 3, self.num_heads // dist_env.get_tensor_model_parallel_world_size(), embed_dim // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        query_states, key_states, value_states = (
            mixed_qkv[0],
            mixed_qkv[1],
            mixed_qkv[2],
        )

        # attention
        if self.use_flash_attn:
            query_states = query_states.permute(0, 2, 1, 3)
            key_states = key_states.permute(0, 2, 1, 3)
            value_states = value_states.permute(0, 2, 1, 3)
            if attention_mask is None:
                qk_mask = None
            else:
                if len(attention_mask.shape) == 2:
                    qk_mask = attention_mask
                else:
                    qk_mask = ~attention_mask[:, 0, :, 0]
                if cu_seqlens is not None:
                    qk_mask = None
            with dist_env.get_cuda_rng_tracker().fork():
                context_layer = self.core_attention_flash(query_states, key_states, value_states, qk_mask, cu_seqlens)
            context_layer = rearrange(context_layer, 'b s h d -> b s (h d)').contiguous()
        else:
            attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))

            attention_scores = attention_scores * self.scale

            # Normalize the attention scores to probabilities.
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)

            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            context_layer = torch.matmul(attention_probs, value_states).permute(0, 2, 1, 3)

            new_context_layer_shape = context_layer.size()[:-2] + (self.embed_dim // dist_env.get_tensor_model_parallel_world_size(),)
            context_layer = context_layer.reshape(new_context_layer_shape)

        if self.sequence_parallel:
            context_layer = context_layer.transpose(0, 1).contiguous()
        output, _ = self.projection(context_layer)
        if self.sequence_parallel:
            output = output.transpose(0, 1).contiguous()
        return output


class EVAMLP(MegatronModule):
    def __init__(
        self,
        hidden_act,
        hidden_size,
        intermediate_size,
        params_dtype=torch.half,
        sequence_parallel=False
    ):
        super().__init__()
        self.sequence_parallel = sequence_parallel
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            gather_output=False,
            params_dtype=params_dtype,
            sequence_parallel=sequence_parallel
        )

        self.fc2 = RowParallelLinear(
            intermediate_size,
            hidden_size,
            input_is_parallel=True,
            params_dtype=params_dtype,
            sequence_parallel=sequence_parallel
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.sequence_parallel:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        hidden_states, _ = self.fc1(hidden_states)
        if self.sequence_parallel:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        hidden_states = self.activation_fn(hidden_states)
        if self.sequence_parallel:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        hidden_states, _ = self.fc2(hidden_states)
        if self.sequence_parallel:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        return hidden_states


class ParallelVisionTransformerLayerPipe(MegatronModule):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        attention_dropout,
        hidden_act,
        layer_norm_eps,
        vision_layer_number,
        use_flash_attn,
        params_dtype=torch.half,
        sequence_parallel=False,
        num_eva_layers=None,
        vit_select_layer=None,
        layer_norm=None,
        postnorm=False,
        qv_bias=False
    ):
        super().__init__()
        self.embed_dim = hidden_size
        params_dtype = get_torch_dtype(params_dtype)
        if layer_norm["type"] == "torch":
            self.layer_norm1 = nn.LayerNorm(layer_norm["kwargs"]["normalized_shape"], eps=1e-06)
        else:
            self.layer_norm1 = build_layer_norm(layer_norm)
        self.self_attn = EVAAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout,
            use_flash_attn,
            params_dtype=params_dtype,
            sequence_parallel=sequence_parallel,
            qv_bias=qv_bias
        )
        self.mlp = EVAMLP(
            hidden_act,
            hidden_size,
            intermediate_size,
            params_dtype=params_dtype,
            sequence_parallel=sequence_parallel
        )
        if layer_norm["type"] == "torch":
            self.layer_norm2 = nn.LayerNorm(layer_norm["kwargs"]["normalized_shape"], eps=1e-06)
        else:
            self.layer_norm2 = build_layer_norm(layer_norm)
        self.vision_layer_number = vision_layer_number
        self.num_eva_layers = num_eva_layers
        self.vit_select_layer = vit_select_layer
        assert isinstance(self.num_eva_layers, int), "self.num_eva_layers must be int"
        self.postnorm = postnorm

    def forward(self, inputs, **kwargs):
        if len(inputs) == 6:
            ori_hidden_states, input_ids, position_ids, attention_mask, image_flags, labels = inputs
        elif len(inputs) == 7:
            ori_hidden_states, input_ids, position_ids, attention_mask, image_flags, labels, cu_seqlens = inputs
        else:
            raise NotImplementedError

        if (image_flags == 0).all():
            fake_loss = ori_hidden_states
            for param in self.layer_norm2.parameters():
                fake_loss += (param * 0).sum()

            return (fake_loss, *inputs[1:])

        if len(ori_hidden_states.shape) == 4:
            hidden_states = ori_hidden_states[-1]
        else:
            hidden_states = ori_hidden_states
        residual = hidden_states

        if self.postnorm:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                head_mask=None,
            )
            hidden_states = self.layer_norm1(hidden_states)
            hidden_states = hidden_states + residual
            residual = hidden_states
            hidden_states = self.mlp(hidden_states)
            hidden_states = self.layer_norm2(hidden_states)
        else:
            hidden_states = self.layer_norm1(hidden_states)
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                head_mask=None,
            )
            hidden_states = self.layer_norm2(hidden_states)
            hidden_states = self.mlp(hidden_states)

        hidden_states = hidden_states + residual
        if self.vision_layer_number >= self.num_eva_layers + self.vit_select_layer:
            if len(ori_hidden_states.shape) == 3:
                ori_hidden_states = ori_hidden_states.unsqueeze(0)
            ori_hidden_states = torch.cat([ori_hidden_states, hidden_states.unsqueeze(0)])
        else:
            ori_hidden_states = hidden_states

        if len(inputs) == 6:
            return ori_hidden_states, input_ids, position_ids, attention_mask, image_flags, labels
        elif len(inputs) == 7:
            return ori_hidden_states, input_ids, position_ids, attention_mask, image_flags, labels, cu_seqlens
        else:
            raise NotImplementedError