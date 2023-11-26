import torch

from llm.utils.env import dist_env
from llm.utils.general.registry_factory import BATCH_FN_REGISTRY


def get_ltor_masks_and_position_ids(
        data,
        eod_token,
        reset_position_ids,
        reset_attention_mask,
        eod_mask_loss,
        prefix_indices,
        loss_on_targets_only):
    """
    Build masks and position id for left to right model.
    :param prefix_indices: argument can have multiple types:
        - None signifies that the model is fully autoregressive.
        - List[int] the argument holds all prefix indices that split a row into an input and a target
        - List[List[int]] the argument holds all prefix indices that split documents between input and target.
    :param loss_on_targets_only: bool to determine if we should mask loss on prefix.
    """

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask or prefix_indices is not None:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask or prefix_indices is not None:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]

            # If the last eod token is not the last token of the sequence, we suppose that there is a partial document
            # We treat this case as if we add an eod token at the end of the sequence.
            if data[b][-1] != eod_token:
                eod_index = torch.cat(
                    (eod_index, torch.tensor([len(data[b])], dtype=eod_index.dtype, device=eod_index.device))
                )

            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]

                if reset_attention_mask:
                    # Prevent cross document interactions.
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0

                    # Prefix lm per document.
                    if prefix_indices:
                        assert isinstance(prefix_indices[b], list), f"prefix for a row has to be document specific, and consequently return a list, got {prefix_indices[b]}"        # noqa
                        attention_mask[b, 0, prev_index: prefix_indices[b][j], prev_index: prefix_indices[b][j]] = 1
                        if loss_on_targets_only:
                            # Last token of the prefix should predict the prefix_index id
                            loss_mask[b, prev_index: prefix_indices[b][j] - 1] = 0.0

                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)

                prev_index = i + 1

            # Prefix lm per row.
            if prefix_indices is not None and (reset_attention_mask is False):
                assert isinstance(prefix_indices[b], int), \
                    f"prefix for a row has to be row specific, and consequently return an int, got {prefix_indices[b]}"
                attention_mask[b, 0, :prefix_indices[b], :prefix_indices[b]] = 1
                if loss_on_targets_only:
                    # Last token of the prefix should predict the prefix_index id
                    loss_mask[b, :prefix_indices[b] - 1] = 0.0

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids


def get_peft_attention_mask(tokens, eod_token):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    bsz, tgt_len = tokens.shape
    atten_mask = (tokens != eod_token)
    combined_attention_mask = None
    if tgt_len > 1:
        mask = torch.full((tgt_len, tgt_len), torch.tensor(1.0)).to(tokens.device)
        mask_cond = torch.arange(mask.size(-1)).to(tokens.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(torch.bool)
        # remove past_key_values_length
        combined_attention_mask = mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)

    if atten_mask is not None:
        expanded_mask = atten_mask[:, None, None, :].expand(bsz, 1, tgt_len, tgt_len)

        if combined_attention_mask is None:
            combined_attention_mask = ~expanded_mask
        else:
            inverted_mask = ~expanded_mask
            combined_attention_mask = torch.logical_or(inverted_mask, combined_attention_mask)

    return combined_attention_mask


@BATCH_FN_REGISTRY.register('json_batch_pipe')
class JsonBatchFunction(object):
    def __init__(self, tokenizer, reset_position_ids, reset_attention_mask,
                 eod_mask_loss=True, prefix_indices=None, loss_on_targets_only=True,
                 hf_atten_mask=False):
        self.tokenizer = tokenizer
        self.reset_position_ids = reset_position_ids
        self.reset_attention_mask = reset_attention_mask
        self.eod_mask_loss = eod_mask_loss
        self.prefix_indices = prefix_indices
        self.loss_on_targets_only = loss_on_targets_only
        self.hf_atten_mask = hf_atten_mask
        self.pad_token_id = len(self.tokenizer) - 1

    def __call__(self, data):
        # Items and their type.
        keys = ['labels', 'input_ids']
        datatype = torch.int64
        # Broadcast data.
        data_b = dist_env.broadcast_data(keys, data, datatype)

        labels = data_b['labels'].long()
        tokens = data_b['input_ids'].long()

        # Get the masks and position ids.
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            self.pad_token_id,
            self.reset_position_ids,
            self.reset_attention_mask,
            self.eod_mask_loss,
            prefix_indices=self.prefix_indices,
            loss_on_targets_only=self.loss_on_targets_only
        )

        if self.hf_atten_mask:
            attention_mask = get_peft_attention_mask(tokens, self.pad_token_id)

        return (tokens, position_ids, attention_mask), (labels, loss_mask)


@BATCH_FN_REGISTRY.register('flash_batch_pipe')
class FlashBatchFunction(object):
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

        labels = data_b['labels'].long()
        tokens = data_b['input_ids'].long()
        attention_mask = tokens.ne(self.pad_token_id)
        _, seq_length = tokens.size()
        loss_mask = torch.ones(tokens.size(), dtype=torch.float, device=tokens.device)
        if self.eod_mask_loss:
            loss_mask[tokens == self.pad_token_id] = 0.0
        # Position ids.
        position_ids = torch.arange(seq_length, dtype=torch.long,
                                    device=tokens.device)
        position_ids = position_ids.unsqueeze(0).expand_as(tokens)

        return (tokens, position_ids, attention_mask), (labels, loss_mask)


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

        # Get the masks and position ids.
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            self.pad_token_id,
            self.reset_position_ids,
            self.reset_attention_mask,
            self.eod_mask_loss,
            prefix_indices=self.prefix_indices,
            loss_on_targets_only=self.loss_on_targets_only
        )

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
        # Get the attention mask and position ids.
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            self.tokenizer.eos_token_id,
            self.reset_position_ids,
            self.reset_attention_mask,
            self.eod_mask_loss,
            prefix_indices=self.prefix_indices,
            loss_on_targets_only=self.loss_on_targets_only
        )

        return (tokens, position_ids, attention_mask), (labels, loss_mask)


def build_batch_pipe_fn(cfg_batch_pipe, tokenizer):
    if 'kwargs' not in cfg_batch_pipe:
        cfg_batch_pipe['kwargs'] = {}
    cfg_batch_pipe['kwargs']['tokenizer'] = tokenizer
    return BATCH_FN_REGISTRY.build(cfg_batch_pipe)
