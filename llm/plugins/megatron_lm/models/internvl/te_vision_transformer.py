import copy
import itertools
import torch
import torch.nn as nn
from typing import Optional

from transformers.activations import ACT2FN
# from llm.models.mg_models.base_modules.modules.meg_module import MegatronModule
# from llm.models.mg_models.base_modules.utils import get_torch_dtype

# from llm.utils.env import dist_env
# from llm.models.mg_models.base_modules.layers import ColumnParallelLinear, RowParallelLinear
# from llm.models.mg_models.base_modules.layers.fused_layer_norm import build_layer_norm
from llm.models.mg_models.llama.transformer import FlashAttention

from .meg_module import MegatronModule
from .utils import get_torch_dtype, DropPath
from .column_parallel_linear import ColumnParallelLinear
from .row_parallel_linear import RowParallelLinear
from .fused_layer_norm import build_layer_norm

from megatron.core import parallel_state, tensor_parallel
# from megatron.core.transformer import MegatronModule
# from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

try:
    from einops import rearrange
except ImportError:
    rearrange = None

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


try:
    from megatron.core.transformer.custom_layers.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TEColumnParallelLinear,
        TEDotProductAttention,
        TENorm,
        TERowParallelGroupedLinear,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

from megatron.core.transformer.enums import AttnMaskType


class InternAttention(MegatronModule):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_dropout,
        proj_dropout,
        use_flash_attn=False,
        params_dtype=torch.half,
        sequence_parallel=False,
        qk_normalization=False,
        layer_norm=None,
        te_config=None,
        layer_number=None,
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(attention_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.use_flash_attn = use_flash_attn
        self.sequence_parallel = sequence_parallel

        # self.qkv = ColumnParallelLinear(
        #     self.embed_dim,
        #     3 * self.embed_dim,
        #     gather_output=False,
        #     bias=False,
        #     params_dtype=params_dtype,
        #     sequence_parallel=sequence_parallel,
        #     num_attention_heads=num_attention_heads
        # )
        te_config.sequence_parallel = sequence_parallel
        self.qkv = TEColumnParallelLinear(
            self.embed_dim,
            3 * self.embed_dim,
            config=te_config,
            init_method=te_config.init_method,
            gather_output=False,
            bias=te_config.add_bias_linear or te_config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv'
        )

        # if config.qkv_bias:
        # if True:
        #     q_bias = nn.Parameter(torch.zeros(self.embed_dim))
        #     v_bias = nn.Parameter(torch.zeros(self.embed_dim))
        # else:
        #     q_bias = None
        #     v_bias = None

        # if q_bias is not None:
        #     qkv_bias = torch.cat((q_bias, torch.zeros_like(v_bias, requires_grad=False), v_bias))
        #     self.qkv.bias = nn.Parameter(qkv_bias)

        # self.projection = RowParallelLinear(
        #     self.embed_dim,
        #     self.embed_dim,
        #     input_is_parallel=True,
        #     bias=True,
        #     params_dtype=params_dtype,
        #     sequence_parallel=sequence_parallel,
        #     num_attention_heads=num_attention_heads
        # )
        self.projection = TERowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            config=te_config,
            init_method=te_config.output_layer_init_method,
            bias=te_config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj'
        )
        self.rank = parallel_state.get_tensor_model_parallel_rank()
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.num_heads_partions = [self.num_heads // world_size] * world_size
        if self.num_heads % world_size > 0:
            v_mode = self.num_heads % world_size
            for idx in range(v_mode):
                self.num_heads_partions[world_size - 1 - idx] += 1

        if self.use_flash_attn:
            # self.core_attention_flash = FlashAttention(
            #     causal=False, attention_dropout=attention_dropout
            # )
            self.core_attention_flash = TEDotProductAttention(
                config=te_config,
                layer_number=layer_number,
                attn_mask_type=AttnMaskType.causal,
                attention_type="self"
            )

        self.qk_normalization = qk_normalization
        # self.qk_normalization = False
        self.qk_normalized_shapes = [0] + [_ * (self.embed_dim // self.num_heads) for _ in self.num_heads_partions]
        self.qk_normalized_shapes = list(itertools.accumulate(self.qk_normalized_shapes))
        qk_layer_norm = copy.deepcopy(layer_norm)
        qk_layer_norm['type'] = 'qk_rms_norm'
        qk_layer_norm['kwargs']['part_shapes'] = self.qk_normalized_shapes
        qk_layer_norm['kwargs']['sync_tp_duplicated_parameters'] = False
        if self.qk_normalization:
            self.q_norm = build_layer_norm(qk_layer_norm)
            self.k_norm = build_layer_norm(qk_layer_norm)

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
            tgt_len *= parallel_state.get_tensor_model_parallel_world_size()
        mixed_qkv, _ = self.qkv(hidden_states)
        if self.sequence_parallel:
            mixed_qkv = mixed_qkv.transpose(0, 1).contiguous()

        # attention
        if self.use_flash_attn:
            mixed_qkv, _ = self.qkv(hidden_states)
            # import torch
            # if torch.distributed.get_rank() == 0:
            #     import pdb;pdb.set_trace()
            # torch.distributed.barrier()
            mixed_qkv = mixed_qkv.reshape(bsz, tgt_len, 3, self.num_heads_partions[self.rank], embed_dim // self.num_heads)
            # mixed_qkv = mixed_qkv.reshape(bsz, tgt_len, 3, self.num_heads // dist_env.get_tensor_model_parallel_world_size(), embed_dim // self.num_heads)
            q, k, v = mixed_qkv.unbind(2)
            if self.qk_normalization:
                q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
                k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            query_states, key_states, value_states = q, k, v

            if attention_mask is None:
                qk_mask = None
            else:
                if len(attention_mask.shape) == 2:
                    qk_mask = attention_mask
                else:
                    qk_mask = ~attention_mask[:, 0, :, 0]
                if cu_seqlens is not None:
                    qk_mask = None
            with tensor_parallel.get_cuda_rng_tracker().fork():
                # context_layer = self.core_attention_flash(query_states, key_states, value_states, qk_mask)
                context_layer = self.core_attention_flash(query_states, key_states, value_states, qk_mask, attn_mask_type=AttnMaskType.causal)
            # context_layer = rearrange(context_layer, 'b s h d -> b s (h d)').contiguous()
        else:
            mixed_qkv = mixed_qkv.reshape(bsz, tgt_len, 3, self.num_heads // parallel_state.get_tensor_model_parallel_world_size(), embed_dim // self.num_heads)
            query_states, key_states, value_states = mixed_qkv.unbind(2)
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

            new_context_layer_shape = context_layer.size()[:-2] + (self.embed_dim // parallel_state.get_tensor_model_parallel_world_size(),)
            context_layer = context_layer.reshape(new_context_layer_shape)

        if self.sequence_parallel:
            context_layer = context_layer.transpose(0, 1).contiguous()
        output, _ = self.projection(context_layer)
        output = self.proj_drop(output)
        if self.sequence_parallel:
            output = output.transpose(0, 1).contiguous()
        return output


class InternMLP(MegatronModule):
    def __init__(
        self,
        hidden_act,
        hidden_size,
        intermediate_size,
        params_dtype=torch.half,
        sequence_parallel=False,
        te_config=None
    ):
        super().__init__()
        self.sequence_parallel = sequence_parallel
        self.activation_fn = ACT2FN[hidden_act]
        # self.fc1 = ColumnParallelLinear(
        #     hidden_size,
        #     intermediate_size,
        #     gather_output=False,
        #     # params_dtype=params_dtype,
        #     # sequence_parallel=sequence_parallel
        # )
        self.fc1 = TEColumnParallelLinear(
            hidden_size,
            intermediate_size,
            config=te_config,
            init_method=te_config.init_method,
            gather_output=False,
            bias=te_config.add_bias_linear,
            skip_bias_add=True,
            is_expert=False, # is_expert,
            tp_comm_buffer_name='fc1'
        )

        # self.fc2 = RowParallelLinear(
        #     intermediate_size,
        #     hidden_size,
        #     input_is_parallel=True,
        #     # params_dtype=params_dtype,
        #     # sequence_parallel=sequence_parallel
        # )
        self.fc2 = TERowParallelLinear(
            intermediate_size,
            hidden_size,
            config=te_config,
            init_method=te_config.output_layer_init_method,
            bias=te_config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False, # is_expert,
            tp_comm_buffer_name='fc2'
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


class TEVisionTransformerLayer(MegatronModule):
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
        num_vit_layers=None,
        vit_select_layer=None,
        layer_norm=None,
        initializer_factor=0.1,
        drop_path_rate=0.0,
        qk_normalization=False,
        atten_layer_norm=None,
        proj_dropout=0.0,
        te_config=None
    ):
        super().__init__()
        self.embed_dim = hidden_size
        params_dtype = get_torch_dtype(params_dtype)
        # self.norm1 = build_layer_norm(layer_norm)
        self.norm1 = TENorm(
            config=te_config,
            hidden_size=layer_norm["kwargs"]["normalized_shape"],
            eps=layer_norm["kwargs"]["eps"]
        )
        self.attn = InternAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout,
            proj_dropout,
            use_flash_attn,
            params_dtype=params_dtype,
            sequence_parallel=sequence_parallel,
            qk_normalization=qk_normalization,
            layer_norm=atten_layer_norm,
            te_config=te_config,
            layer_number=vision_layer_number
        )
        self.mlp = InternMLP(
            hidden_act,
            hidden_size,
            intermediate_size,
            params_dtype=params_dtype,
            sequence_parallel=sequence_parallel,
            te_config=te_config
        )
        # self.norm2 = build_layer_norm(layer_norm)
        self.norm2 = TENorm(
            config=te_config,
            hidden_size=layer_norm["kwargs"]["normalized_shape"],
            eps=layer_norm["kwargs"]["eps"]
        )
        self.vision_layer_number = vision_layer_number
        self.num_vit_layers = num_vit_layers
        self.vit_select_layer = vit_select_layer
        self.ls1 = nn.Parameter(initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(initializer_factor * torch.ones(self.embed_dim))
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # self.drop_path1 = nn.Identity()
        # self.drop_path2 = nn.Identity()
        assert isinstance(self.num_vit_layers, int), "self.num_vit_layers must be int"

    def forward(self, ori_hidden_states):
        # if len(inputs) == 4:
        #     ori_hidden_states, input_ids, position_ids, image_flags = inputs
        # elif len(inputs) == 5:
        #     ori_hidden_states, input_ids, position_ids, image_flags, cu_seqlens = inputs
        # if len(inputs) == 6:
        #     ori_hidden_states, input_ids, position_ids, attention_mask, image_flags, labels = inputs
        # elif len(inputs) == 7:
        #     ori_hidden_states, input_ids, position_ids, attention_mask, image_flags, labels, cu_seqlens = inputs
        # if (image_flags == 0).all():
        #     fake_loss = ori_hidden_states
        #     fake_loss += (self.ls1.data * 0).sum()
        #     return (fake_loss, *inputs[1:])
        # ori_hidden_states, input_ids, position_ids, image_flags = inputs
        if len(ori_hidden_states.shape) == 4:
            hidden_states = ori_hidden_states[-1]
        else:
            hidden_states = ori_hidden_states

        # if torch.distributed.get_rank() == 0 and self.vision_layer_number == 2:
        #     import pdb;pdb.set_trace()

        # print(f'hidden_states: {hidden_states.shape}')
        # hidden_states = hidden_states + self.drop_path1(self.attn(self.norm1(hidden_states=hidden_states,head_mask=attention_mask)) * self.ls1)
        hidden_states = hidden_states + self.drop_path1(self.attn(self.norm1(hidden_states)) * self.ls1)

        # hidden_states = hidden_states + self.drop_path2(self.mlp(self.norm2(hidden_states=hidden_states,head_mask=attention_mask)) * self.ls2)
        hidden_states = hidden_states + self.drop_path2(self.mlp(self.norm2(hidden_states)) * self.ls2)

        if self.vision_layer_number >= self.num_vit_layers + self.vit_select_layer:
            if len(ori_hidden_states.shape) == 3:
                ori_hidden_states = ori_hidden_states.unsqueeze(0)
            ori_hidden_states = torch.cat([ori_hidden_states, hidden_states.unsqueeze(0)])
        else:
            ori_hidden_states = hidden_states
        if self.vision_layer_number == self.num_vit_layers - 1:
            ori_hidden_states = ori_hidden_states[self.vit_select_layer]

        # print(f"rank:{torch.distributed.get_rank()},"
        #       f"ori_hidden_states: {ori_hidden_states.shape}",
        #       f"vision_layer_number:{self.vision_layer_number},"
        #       f"num_vit_layers:{self.num_vit_layers},"
        #       f"vit_select_layer:{self.vit_select_layer}")

        # if len(inputs) == 4:
        #     return ori_hidden_states, input_ids, position_ids, image_flags
        # elif len(inputs) == 5:
        #     return ori_hidden_states, input_ids, position_ids, image_flags, cu_seqlens
        # if len(inputs) == 6:
        #     return ori_hidden_states, input_ids, position_ids, attention_mask, image_flags, labels
        # elif len(inputs) == 7:
        #     return ori_hidden_states, input_ids, position_ids, attention_mask, image_flags, labels, cu_seqlens
        return ori_hidden_states
