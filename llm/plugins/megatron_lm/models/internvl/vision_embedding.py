import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core import parallel_state, tensor_parallel
# from megatron.core.transformer import MegatronModule

# from llm.models.mg_models.base_modules.layers import ColumnParallelConv2d
# from llm.models.mg_models.base_modules.utils import get_torch_dtype

from .meg_module import MegatronModule
from .column_parallel_conv import ColumnParallelConv2d
from .utils import get_torch_dtype

def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size, assuming square window

    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    assert H % window_size == 0 and W % window_size == 0, 'H and W must be divisible by window_size'

    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows


class VisionEmbedding(MegatronModule):
    def __init__(
        self,
        hidden_size,
        image_size,
        patch_size,
        params_dtype=torch.half,
        sequence_parallel=False,
        image_fold=False
    ):
        super(VisionEmbedding, self).__init__()
        self.embed_dim = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        params_dtype = get_torch_dtype(params_dtype)
        self.sequence_parallel = sequence_parallel

        class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim),
        )

        self.register_parameter("class_embedding", class_embedding)

        self.patch_embedding = ColumnParallelConv2d(
            input_channel=3,
            output_channel=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            gather_output=True,
            bias=True,
            padding=0,
            params_dtype=params_dtype
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))
        self.register_parameter("position_embedding", position_embedding)
        self.image_fold = image_fold

    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = pos_embed.float().reshape(
            1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1).to(target_dtype)
        return pos_embed

    def forward(self, pixel_values):
        # if len(inputs) == 5:
        #     input_ids, position_ids, attention_mask, image_flags, pixel_values = inputs
        # elif len(inputs) == 6:
        #     input_ids, position_ids, attention_mask, image_flags, pixel_values, cu_seqlens = inputs
        # if len(inputs) == 6:
        #     input_ids, position_ids, attention_mask, image_flags, labels, pixel_values = inputs
        # elif len(inputs) == 7:
        #     input_ids, position_ids, attention_mask, image_flags, labels, pixel_values, cu_seqlens = inputs
        # if (image_flags == 0).all():
        #     fake_loss = 0
        #     for param in self.patch_embedding.parameters():
        #         fake_loss += (param * 0).sum()
        #     fake_loss += (self.position_embedding.data * 0).sum()
        #     fake_loss += (self.class_embedding.data * 0).sum()
        #     # if len(inputs) == 5:
        #     #     return fake_loss, input_ids, position_ids, image_flags
        #     # elif len(inputs) == 6:
        #     #     return fake_loss, input_ids, position_ids, image_flags, cu_seqlens
        #     if len(inputs) == 6:
        #         return fake_loss, input_ids, position_ids, attention_mask, image_flags, labels
        #     elif len(inputs) == 7:
        #         return fake_loss, input_ids, position_ids, attention_mask, image_flags, labels, cu_seqlens
        if self.image_fold:
            image_size = pixel_values.size(-1)  # B, C, H, W
            pixel_values = window_partition(pixel_values, window_size=image_size // self.image_fold)
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = torch.cat([
            self.position_embedding[:, :1, :],
            self._get_pos_embed(self.position_embedding[:, 1:, :], height, width)
        ], dim=1)
        embeddings = embeddings + position_embedding.to(target_dtype)
        if self.sequence_parallel:
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            pad_len = (embeddings.shape[1] // tp_size + 1) * tp_size
            if pad_len != embeddings.shape[1]:
                pad_shape = (embeddings.shape[0], pad_len, embeddings.shape[2])
                temp = torch.zeros(pad_shape).to(device=embeddings.device, dtype=embeddings.dtype)
                temp[:, :embeddings.shape[1]] = embeddings
                embeddings = temp
            # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
            embeddings = embeddings.transpose(0, 1).contiguous()
            embeddings = tensor_parallel.scatter_to_sequence_parallel_region(embeddings)
            # set to [s b h] --> [b s h].
            embeddings = embeddings.transpose(0, 1).contiguous()


        # if torch.distributed.get_rank() == 0:
        #     torch.save(pixel_values, 'data/rank0_pixel_values.pt')
        #     torch.save(embeddings, 'data/rank0_embeddings.pt')
        #     torch.save(self.patch_embedding.weight.data, 'data/rank0_weight.pt')
        #     import pdb;pdb.set_trace()

        # if len(inputs) == 5:
        #     return embeddings, input_ids, position_ids, image_flags
        # elif len(inputs) == 6:
        #     return embeddings, input_ids, positvscode-remote://ssh-remote%2Bcore202/mnt/afs_2/liangkaihuan/Codes/lm-toolchain/easyllm_vlm/llm/plugins/internvl/models/mg_models/vision_extract_feat.pyion_ids, image_flags, cu_seqlens
        # if len(inputs) == 6:
        #     return embeddings, input_ids, position_ids, attention_mask, image_flags, labels
        # elif len(inputs) == 7:
        #     return embeddings, input_ids, position_ids, attention_mask, image_flags, labels, cu_seqlens
        return embeddings
