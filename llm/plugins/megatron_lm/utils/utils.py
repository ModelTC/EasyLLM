# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""General utilities."""
import os
import sys
from datetime import datetime

import torch

try:
    from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_l2norm
except ImportError:
    try:
        from apex.multi_tensor_apply import multi_tensor_applier
    except ImportError:
        multi_tensor_applier = None

    try:
        from amp_C import multi_tensor_l2norm
    except ImportError:
        import warnings
        warnings.warn(
            f'Transformer Engine and Apex are not installed. '
            'Falling back to local implementations of '
            'multi_tensor_applier and multi_tensor_l2norm'
        )

        from megatron.core.utils import (
            local_multi_tensor_l2_norm as multi_tensor_l2norm,
            local_multi_tensor_applier as multi_tensor_applier,
        )

from megatron.training import (
    get_args,
    get_adlr_autoresume,
)
from megatron.core import DistributedDataParallel as DDP
from megatron.core import mpu
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from megatron.legacy.model import Float16Module
from megatron.legacy.model.module import param_is_not_shared


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
       seq_shape = torch.tensor([data["input_ids"].shape[1]], dtype = torch.int64, device = torch.cuda.current_device())

       batch = {
           'tokens': data["input_ids"].cuda(non_blocking = True),
           'labels': data["labels"].cuda(non_blocking = True),
           'loss_mask': data["loss_mask"].cuda(non_blocking = True),
           'attention_mask': None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking = True),
           'position_ids': data["position_ids"].cuda(non_blocking = True)
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
           seq_shape = torch.empty((1,), dtype = torch.int64 , device = torch.cuda.current_device())
           if args.pipeline_model_parallel_size == 1:
               _broadcast(seq_shape)
           elif mpu.is_pipeline_first_stage():
               _broadcast(seq_shape)
           elif mpu.is_pipeline_last_stage():
               _broadcast(seq_shape)
           seq_len = seq_shape.item()
       else:
           seq_len = args.seq_length

       tokens=torch.empty((args.micro_batch_size,seq_len), dtype = torch.int64 , device = torch.cuda.current_device())
       labels=torch.empty((args.micro_batch_size,seq_len), dtype = torch.int64 , device = torch.cuda.current_device())
       loss_mask=torch.empty((args.micro_batch_size,seq_len), dtype = torch.bool , device = torch.cuda.current_device())
       if args.create_attention_mask_in_dataloader:
           # attention_mask=torch.empty(
           #      (args.micro_batch_size,1,args.seq_length,args.seq_length), dtype = torch.bool , device = torch.cuda.current_device()
           #  )
           attention_mask=torch.empty(
               (args.micro_batch_size,seq_len), dtype = torch.bool , device = torch.cuda.current_device()
           )
       else:
           attention_mask=None
       position_ids=torch.empty((args.micro_batch_size,seq_len), dtype = torch.int64 , device = torch.cuda.current_device())

       if args.pipeline_model_parallel_size == 1:
           _broadcast(tokens)
           _broadcast(labels)
           _broadcast(loss_mask)
           _broadcast(attention_mask)
           _broadcast(position_ids)
 
       elif mpu.is_pipeline_first_stage():
           labels=None
           loss_mask=None
   
           _broadcast(tokens)
           _broadcast(attention_mask)
           _broadcast(position_ids)

       elif mpu.is_pipeline_last_stage():
           tokens=None
           position_ids=None
    
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
