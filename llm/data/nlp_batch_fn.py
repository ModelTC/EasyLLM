import torch
import os

from llm.utils.env import dist_env
from llm.utils.general.registry_factory import BATCH_FN_REGISTRY


def get_batch_on_this_cp_rank(batch):
    # cp_size = dist_env.get_tensor_model_parallel_world_size()
    cp_size = dist_env.get_context_parallel_world_size()
    if cp_size > 1:
        cp_rank = dist_env.get_context_parallel_rank()
        for key, val in batch.items():
            # seq_dim = 1 if key != 'attention_mask' else 2
            seq_dim = 1
            val = val.view(
                *val.shape[0:seq_dim],
                2 * cp_size,
                val.shape[seq_dim] // (2 * cp_size),
                *val.shape[(seq_dim + 1) :],
            )
            index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)],
                                 device="cpu", pin_memory=True).cuda(non_blocking=True)
            # index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)],
            #                      device="cpu") # .cuda(non_blocking=True)
            val = val.index_select(seq_dim, index)
            val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2) :])
            batch[key] = val

    return batch


@BATCH_FN_REGISTRY.register('flash_batch_pipe')
class FlashBatchFunction(object):
    def __init__(self,
                 tokenizer,
                 eod_mask_loss=True,
                 pretrain=False):
        self.tokenizer = tokenizer
        self.eod_mask_loss = eod_mask_loss
        self.pad_token_id = len(self.tokenizer) - 1
        self.pretrain = pretrain
        self.keys = ['labels', 'input_ids', "cu_seqlens", 'position_ids']

    def __call__(self, data):
        datatype = torch.int64
        # Broadcast data.
        data_b = dist_env.broadcast_data(self.keys, data, datatype)

        labels = data_b['labels'].long()
        tokens = data_b['input_ids'].long()

        attention_mask = tokens.ne(self.pad_token_id)
        loss_mask = attention_mask.clone()
        if len(data_b['cu_seqlens']) == 1:
            _, seq_length = tokens.size()
            position_ids = torch.arange(seq_length, dtype=torch.long,
                                        device=tokens.device)
            position_ids = position_ids.unsqueeze(0).expand_as(tokens)
            batch = {
                "tokens": tokens,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "loss_mask": loss_mask
            }
            batch = get_batch_on_this_cp_rank(batch)
            tokens, position_ids, attention_mask, labels, loss_mask = \
                batch["tokens"], batch["position_ids"], batch["attention_mask"], batch["labels"], batch["loss_mask"]
            return (tokens, position_ids, attention_mask), (labels, loss_mask)
        else:
            cu_seqlens = data_b['cu_seqlens']
            position_ids = data_b['position_ids']
            if os.environ.get('ACCELERATOR_BACKEND', 'CUDA') != 'CUDA':
                cu_seqlens = cu_seqlens.cpu()
            return (tokens, position_ids, attention_mask, cu_seqlens), (labels, loss_mask)


@BATCH_FN_REGISTRY.register('mini_rlhf_json_batch_pipe')
class MiniRLHFJsonBatchFunction(object):
    def __init__(self, tokenizer, reset_position_ids, reset_attention_mask,
                 eod_mask_loss=True, prefix_indices=None, loss_on_targets_only=True):
        self.tokenizer = tokenizer
        self.reset_position_ids = reset_position_ids
        self.reset_attention_mask = reset_attention_mask
        self.eod_mask_loss = eod_mask_loss
        self.prefix_indices = prefix_indices
        self.loss_on_targets_only = loss_on_targets_only
        self.pad_token_id = len(self.tokenizer) - 1

    def __call__(self, data):
        # Items and their type.
        keys = ['labels', 'input_ids']
        datatype = torch.int64
        # Broadcast data.
        data_b = dist_env.broadcast_data(keys, data, datatype)
        scores = dist_env.broadcast_data(['scores'], data, torch.float32)['scores'].float()

        labels = data_b['labels'].long()
        tokens = data_b['input_ids'].long()

        attention_mask = tokens.ne(self.pad_token_id)
        loss_mask = attention_mask.clone()
        _, seq_length = tokens.size()
        position_ids = torch.arange(seq_length, dtype=torch.long,
                                    device=tokens.device)
        position_ids = position_ids.unsqueeze(0).expand_as(tokens)

        return (tokens, position_ids, attention_mask), (labels, loss_mask, scores)


@BATCH_FN_REGISTRY.register('token_batch_pipe')
class TokenBatchFunction(object):
    def __init__(self, tokenizer, reset_position_ids, reset_attention_mask,
                 eod_mask_loss=True, prefix_indices=None, loss_on_targets_only=True,
                 micro_batch_size=1):
        self.tokenizer = tokenizer
        self.reset_position_ids = reset_position_ids
        self.reset_attention_mask = reset_attention_mask
        self.eod_mask_loss = eod_mask_loss
        self.prefix_indices = prefix_indices
        self.loss_on_targets_only = loss_on_targets_only
        self.micro_batch_size = micro_batch_size
        self.pad_token_id = len(self.tokenizer) - 1

    def __call__(self, data):
        """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""

        # Move to GPU.
        tokens = data.view(self.micro_batch_size, -1).contiguous().cuda()

        labels = torch.cat([tokens[:, 1:], tokens.new_ones(tokens.shape[0], 1) * self.tokenizer.eos_token_id], dim=-1)

        attention_mask = tokens.ne(self.pad_token_id)
        loss_mask = attention_mask.clone()
        _, seq_length = tokens.size()
        position_ids = torch.arange(seq_length, dtype=torch.long,
                                    device=tokens.device)
        position_ids = position_ids.unsqueeze(0).expand_as(tokens)

        return (tokens, position_ids, attention_mask), (labels, loss_mask)


def build_batch_pipe_fn(cfg_batch_pipe, tokenizer):
    if 'kwargs' not in cfg_batch_pipe:
        cfg_batch_pipe['kwargs'] = {}
    cfg_batch_pipe['kwargs']['tokenizer'] = tokenizer
    return BATCH_FN_REGISTRY.build(cfg_batch_pipe)
