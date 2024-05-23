import torch
import torch.nn as nn

from llm.models.mg_models.base_modules.modules.meg_module import MegatronModule
from llm.models.mg_models.base_modules.layers import ColumnParallelConv2d
from llm.models.mg_models.base_modules.utils import get_torch_dtype

from llm.utils.env import dist_env
from llm.utils.env.dist_env.dist_helper import get_tensor_model_parallel_world_size


class VisionEmbeddings(MegatronModule):
    def __init__(
        self,
        hidden_size,
        image_size,
        patch_size,
        params_dtype=torch.half,
        sequence_parallel=False,
        drop_rate=0
    ):
        super(VisionEmbeddings, self).__init__()
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
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))
        self.register_parameter("position_embedding", position_embedding)

    def forward(self, inputs, **kwargs):
        if len(inputs) == 6:
            input_ids, position_ids, attention_mask, image_flags, labels, pixel_values = inputs
        elif len(inputs) == 7:
            input_ids, position_ids, attention_mask, image_flags, labels, pixel_values, cu_seqlens = inputs
        else:
            raise NotImplementedError

        if (image_flags == 0).all():
            fake_loss = 0
            for param in self.patch_embedding.parameters():
                fake_loss += (param * 0).sum()
            fake_loss += (self.position_embedding.data * 0).sum()
            fake_loss += (self.class_embedding.data * 0).sum()

            if len(inputs) == 6:
                return fake_loss, input_ids, position_ids, attention_mask, image_flags, labels
            else:
                return fake_loss, input_ids, position_ids, attention_mask, image_flags, labels, cu_seqlens

        batch_size, num_channels, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        # [b, s, d]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = self.position_embedding
        # embeddings.size(1) == self.num_positions
        embeddings = embeddings + position_embedding[:, : embeddings.size(1), :].to(target_dtype)
        if self.sequence_parallel:
            pad_len = (embeddings.shape[1] // get_tensor_model_parallel_world_size() + 1) * get_tensor_model_parallel_world_size()
            if pad_len != embeddings.shape[1]:
                pad_shape = (embeddings.shape[0], pad_len, embeddings.shape[2])
                temp = torch.zeros(pad_shape).to(device=embeddings.device, dtype=embeddings.dtype)
                temp[:, :embeddings.shape[1]] = embeddings
                embeddings = temp
            # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
            embeddings = embeddings.transpose(0, 1).contiguous()
            embeddings = dist_env.scatter_to_sequence_parallel_region(embeddings)
            # set to [s b h] --> [b s h].
            embeddings = embeddings.transpose(0, 1).contiguous()
        if len(inputs) == 6:
            return embeddings, input_ids, position_ids, attention_mask, image_flags, labels
        elif len(inputs) == 7:
            return embeddings, input_ids, position_ids, attention_mask, image_flags, labels, cu_seqlens
        else:
            raise NotImplementedError
