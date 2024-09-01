import torch

from llm.utils.env import dist_env
from llm.utils.general.registry_factory import BATCH_FN_REGISTRY


@BATCH_FN_REGISTRY.register('flash_batch_pipe_internvl')
class InternFlashBatchFunction(object):
    def __init__(self,
                 tokenizer,
                 eod_mask_loss=True,
                 pretrain=False):
        self.tokenizer = tokenizer
        self.eod_mask_loss = eod_mask_loss
        if hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
            self.pad_token_id = tokenizer.pad_token_id
        else:
            self.pad_token_id = len(self.tokenizer) - 1
        self.pretrain = pretrain

    def __call__(self, data):
        # if self.pretrain:
        #     keys = ['labels', 'input_ids', "cu_seqlens", 'position_ids']
        # else:
        #     keys = ['labels', 'input_ids']
        # datatype = torch.int64
        # Broadcast data.
        # data_b = dist_env.broadcast_data(keys, data, datatype)

        # labels = data_b['labels'].long()
        # tokens = data_b['input_ids'].long()

        # ['input_ids', 'labels', 'cu_seqlens_llm', 'position_ids', 'pixel_values', 'cu_seqlens_vit', 'image_flags', 'attention_mask']
        if self.pretrain:
            keys = ['labels', 'input_ids', 'image_flags', 'cu_seqlens', 'position_ids']
        else:
            keys = ['labels', 'input_ids', 'image_flags']
        # data_b = dist_env.broadcast_data(['labels', 'input_ids', 'image_flags'], data, torch.int64)
        data_b = dist_env.broadcast_data(keys, data, torch.int64)
        labels = data_b['labels'].long()
        tokens = data_b['input_ids'].long()
        image_flags = data_b["image_flags"].long()
        if self.pretrain:
            cu_seqlens = data_b["cu_seqlens"].long()
            position_ids = data_b["position_ids"].long()

        data_b = dist_env.broadcast_data(['pixel_values'], data, torch.float32)
        pixel_values = data_b["pixel_values"].float()

        attention_mask = tokens.ne(self.pad_token_id)
        loss_mask = attention_mask.clone()
        if not self.pretrain:
            _, seq_length = tokens.size()
            position_ids = torch.arange(seq_length, dtype=torch.long,
                                        device=tokens.device)
            position_ids = position_ids.unsqueeze(0).expand_as(tokens)
            # return (tokens, position_ids, attention_mask, image_flags, pixel_values), (labels, loss_mask)
            return (tokens, position_ids, attention_mask, image_flags, labels, pixel_values), (labels, loss_mask)
        else:
            # cu_seqlens = data_b['cu_seqlens']
            # position_ids = data_b['position_ids']
            # return (tokens, position_ids, attention_mask, cu_seqlens), (labels, loss_mask)
            # return (tokens, position_ids, attention_mask, image_flags, pixel_values, cu_seqlens), (labels, loss_mask, cu_seqlens)
            return (tokens, position_ids, attention_mask, image_flags, labels, pixel_values, cu_seqlens), (labels, loss_mask, cu_seqlens)
