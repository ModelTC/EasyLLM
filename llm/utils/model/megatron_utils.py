import torch
from functools import partial

import os
if os.getenv("DIST_BACKEND", "easyllm") == "easyllm":
    from llm.utils.env import dist_env
elif os.getenv("DIST_BACKEND", "easyllm") == "megatron":
    from megatron.core import mpu as dist_env
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)
from megatron.training import get_args
from megatron.training import get_timers
from llm.data.nlp_dataset import IGNORE_INDEX
from megatron.core.models.gpt import GPTModel
from megatron.core.utils import StragglerDetector

stimer = StragglerDetector()


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    # if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
    if (not dist_env.is_pipeline_first_stage()) and (not dist_env.is_pipeline_last_stage()):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


def loss_func(loss_mask: torch.Tensor, labels: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()
    ignore_mask = (labels == IGNORE_INDEX)
    loss_mask = loss_mask.view(-1)
    loss_mask = loss_mask * (~ignore_mask.view(-1))

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

    if args.context_parallel_size > 1:
        # torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        torch.distributed.all_reduce(loss, group=dist_env.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss[0].isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    reporting_loss = reporting_loss[0] / reporting_loss[1] / dist_env.get_data_parallel_world_size() # mpu.get_data_parallel_world_size()
    # torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())
    torch.distributed.all_reduce(reporting_loss, group=dist_env.get_data_parallel_group())

    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0] * args.context_parallel_size,
        local_num_tokens,
        # {'lm loss': (reporting_loss[0], reporting_loss[1])},
        {'lm loss': reporting_loss}
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator)
    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels)

    return output_tensor, partial(loss_func, loss_mask, labels)