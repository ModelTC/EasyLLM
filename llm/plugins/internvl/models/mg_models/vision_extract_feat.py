import torch
import torch.nn as nn
import warnings

from llm.models.mg_models.base_modules.modules.meg_module import MegatronModule
from llm.models.mg_models.base_modules.layers import ColumnParallelLinear, RowParallelLinear
from llm.models.mg_models.base_modules.utils import get_torch_dtype

from llm.utils.env import dist_env


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H * W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * W, -1)
    return x


class VisionExtractFeat(MegatronModule):
    def __init__(
        self,
        vit_hidden_size,
        llm_hidden_size,
        vit_select_layer=-2,
        params_dtype=torch.half,
        sequence_parallel=False,
        image_size=224,
        patch_size=14,
        image_fold=False,
        ps_version='v2'
    ):
        super().__init__()
        self.vit_select_layer = vit_select_layer
        params_dtype = get_torch_dtype(params_dtype)
        self.sequence_parallel = sequence_parallel

        if image_size == 224:
            hidden_size_scale = 1
            self.scale_factor = 1
        else:
            hidden_size_scale = 4
            self.scale_factor = 0.5

        self.mlp1_norm = nn.LayerNorm(vit_hidden_size * hidden_size_scale, dtype=params_dtype)
        self.mlp1_fc1 = ColumnParallelLinear(vit_hidden_size * hidden_size_scale, llm_hidden_size, gather_output=False, params_dtype=params_dtype)
        self.mlp1_act = nn.GELU()
        self.mlp1_fc2 = RowParallelLinear(llm_hidden_size, llm_hidden_size, params_dtype=params_dtype, input_is_parallel=True)

        self.image_size = image_size
        self.patch_size = patch_size

        self.image_fold = image_fold
        self.ps_version = ps_version

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, inputs, **kwargs):
        # if len(inputs) == 4:
        #     hidden_states, input_ids, position_ids, image_flags = inputs
        # elif len(inputs) == 5:
        #     hidden_states, input_ids, position_ids, image_flags, cu_seqlens = inputs
        if len(inputs) == 6:
            hidden_states, input_ids, position_ids, attention_mask, image_flags, labels = inputs
        elif len(inputs) == 7:
            hidden_states, input_ids, position_ids, attention_mask, image_flags, labels, cu_seqlens = inputs
        if (image_flags == 0).all():
            fake_loss = hidden_states
            for param in self.mlp1_norm.parameters():
                fake_loss += (param * 0).sum()
            return (fake_loss, *inputs[1:])
        if self.sequence_parallel:
            seq_len_base = (self.image_size // self.patch_size) ** 2
            hidden_states = hidden_states.transpose(0, 2).contiguous()
            hidden_states = dist_env.gather_from_sequence_parallel_region(hidden_states,
                                                                          tensor_parallel_output_grad=True)
            hidden_states = hidden_states.transpose(0, 2).contiguous()
            vit_embeds = hidden_states[self.vit_select_layer][:, 1:(1 + seq_len_base), :]
        else:
            vit_embeds = hidden_states[self.vit_select_layer][:, 1:, :]

        if self.image_fold:
            vit_embeds = window_reverse(vit_embeds, window_size=self.image_size // (self.image_fold * self.patch_size),
                                        H=self.image_size // self.patch_size, W=self.image_size // self.patch_size)

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.scale_factor)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1_norm(vit_embeds)
        vit_embeds, _ = self.mlp1_fc1(vit_embeds)
        vit_embeds = self.mlp1_act(vit_embeds)
        vit_embeds, _ = self.mlp1_fc2(vit_embeds)
        # if len(inputs) == 4:
        #     return vit_embeds, input_ids, position_ids, image_flags
        # elif len(inputs) == 5:
        #     return vit_embeds, input_ids, position_ids, image_flags, cu_seqlens
        if len(inputs) == 6:
            return vit_embeds, input_ids, position_ids, attention_mask, image_flags, labels
        elif len(inputs) == 7:
            return vit_embeds, input_ids, position_ids, attention_mask, image_flags, labels, cu_seqlens
