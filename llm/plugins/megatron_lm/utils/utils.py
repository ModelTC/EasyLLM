# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""General utilities."""
import torch

try:
    from transformer_engine.pytorch.optimizers import multi_tensor_applier
except ImportError:
    try:
        from apex.multi_tensor_apply import multi_tensor_applier
    except ImportError:
        multi_tensor_applier = None

from megatron.training import (
    get_args,
)
from megatron.core import mpu


def get_batch_on_this_cp_rank(batch):
    """ Slice batch input along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.
    """

    # With causal masking, each token only attends to its prior tokens. Simply split
    # sequence into CP chunks can result in severe load imbalance. That's to say, chunks
    # at the end of sequence have bigger workload than others. To address this issue,
    # we split sequence into 2*CP ranks. Assuming CP=2, we then get 4 chunks, chunk_0
    # and chunk_3 are assigned to GPU0, chunk_1 and chunk_2 are assigned to GPU1, so
    # that we can get balanced workload among GPUs in a context parallel group.
    args = get_args()
    cp_size = args.context_parallel_size
    if cp_size > 1:
        cp_rank = mpu.get_context_parallel_rank()
        for key, val in batch.items():
            if val is not None:
                seq_dim = 1 # if key != 'attention_mask' else 2
                val = val.view(
                    *val.shape[0:seq_dim],
                    2 * cp_size,
                    val.shape[seq_dim] // (2 * cp_size),
                    *val.shape[(seq_dim + 1) :],
                )
                index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)],
                                     device="cpu", pin_memory=True).cuda(non_blocking=True)
                val = val.index_select(seq_dim, index)
                val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2) :])
                batch[key] = val

    return batch


def get_batch_on_this_tp_rank(data_iterator):

    args = get_args()

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:

        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None
        seq_shape = torch.tensor([data["input_ids"].shape[1]], dtype=torch.int64, device=torch.cuda.current_device())

        batch = {
            'tokens': data["input_ids"].cuda(non_blocking=True),
            'labels': data["labels"].cuda(non_blocking=True),
            'loss_mask': data["loss_mask"].cuda(non_blocking=True),
            'attention_mask': None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking=True),
            'position_ids': data["position_ids"].cuda(non_blocking=True)
        }

        if args.pipeline_model_parallel_size == 1:
            # if args.variable_seq_lengths:
            #     _broadcast(seq_shape)
            _broadcast(seq_shape)
            _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])

        elif mpu.is_pipeline_first_stage():
            # if args.variable_seq_lengths:
            #     _broadcast(seq_shape)
            _broadcast(seq_shape)
            _broadcast(batch['tokens'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])

        elif mpu.is_pipeline_last_stage():
            # if args.variable_seq_lengths:
            #     _broadcast(seq_shape)
            _broadcast(seq_shape)
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])

    else:
        # if args.variable_seq_lengths:
        if True:
            seq_shape = torch.empty((1,), dtype=torch.int64, device=torch.cuda.current_device())
            if args.pipeline_model_parallel_size == 1:
                _broadcast(seq_shape)
            elif mpu.is_pipeline_first_stage():
                _broadcast(seq_shape)
            elif mpu.is_pipeline_last_stage():
                _broadcast(seq_shape)
            seq_len = seq_shape.item()
        else:
            seq_len = args.seq_length

        tokens = torch.empty((args.micro_batch_size, seq_len), dtype=torch.int64, device=torch.cuda.current_device())
        labels = torch.empty((args.micro_batch_size, seq_len), dtype=torch.int64, device=torch.cuda.current_device())
        loss_mask = torch.empty((args.micro_batch_size, seq_len), dtype=torch.bool, device=torch.cuda.current_device())
        if args.create_attention_mask_in_dataloader:
            # attention_mask=torch.empty(
            #      (args.micro_batch_size,1,args.seq_length,args.seq_length), dtype = torch.bool , device = torch.cuda.current_device()
            #  )
            attention_mask = torch.empty(
                (args.micro_batch_size, seq_len), dtype=torch.bool, device=torch.cuda.current_device()
            )
        else:
            attention_mask = None
        position_ids = torch.empty((args.micro_batch_size, seq_len), dtype=torch.int64, device=torch.cuda.current_device())

        if args.pipeline_model_parallel_size == 1:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif mpu.is_pipeline_first_stage():
            labels = None
            loss_mask = None

            _broadcast(tokens)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif mpu.is_pipeline_last_stage():
            tokens = None
            position_ids = None

            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)

        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }

    return batch
