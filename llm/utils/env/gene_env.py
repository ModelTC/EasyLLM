import random

import torch
import deepspeed
import numpy as np

from llm.utils.env import dist_env
from llm.utils.general import log_helper as logging


def setup_deepspeed_random_and_activation_checkpointing(base_num_layers,
                                                        checkpoint_num_layers,
                                                        split_transformers=False,
                                                        partition_activations=False,
                                                        contigious_checkpointing=False,
                                                        checkpoint_in_cpu=False,
                                                        synchronize_each_layer=False,
                                                        profile_backward=False):
    '''Optional DeepSpeed Activation Checkpointing features.
    Gives access to partition activations, contiguous memory optimizations
    and cpu checkpointing.
    Activation checkpoint requires keep track of the random states
    and setting the random seed for each MP process. Megatron uses
    dist.get_cuda_rng_tracker and dist.model_parallel_cuda_manual_seed
    for keeping track of the random states and setting the random seeds.
    Since they are used in places outside of activation checkpointing,
    we overwrite them to maintain consistency.
    This must be called before all the calls to
    dist.model_parallel_cuda_manual_seed
    '''
    num_layers = base_num_layers // checkpoint_num_layers
    num_layers = num_layers if base_num_layers % checkpoint_num_layers == 0 else num_layers + 1    # noqa
    if split_transformers:
        num_layers *= 2

    deepspeed.checkpointing.configure(
        dist_env,
        partition_activations=partition_activations,
        contiguous_checkpointing=contigious_checkpointing,
        num_checkpoints=num_layers,
        checkpoint_in_cpu=checkpoint_in_cpu,
        synchronize=synchronize_each_layer,
        profile=profile_backward)

    dist_env.checkpoint = deepspeed.checkpointing.checkpoint
    dist_env.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
    dist_env.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed    # noqa


def set_logging_verbosity(rank, log_level='info', log_level_replica='error', deepspeed=True):
    def set_verbosity_deepspeed(logging_level: str):
        if not deepspeed:
            return
        from deepspeed.utils import logger as ds_logger
        log_level = logging.log_levels[logging_level]
        ds_logger.setLevel(log_level)

    def set_verbosity_transformers(logging_level: str):
        try:
            # XXX: perhaps we need a better way of knowing when to override transformers logging
            # currently it's only when using `--tokenizer-type PretrainedFromHF`
            from transformers.utils import logging as transformers_logging
            log_level = transformers_logging.log_levels[logging_level]
            transformers_logging.set_verbosity(log_level)
        except BaseException:
            pass

    if rank == 0:
        if log_level is not None:
            set_verbosity_deepspeed(log_level)
            set_verbosity_transformers(log_level)
    else:
        if log_level_replica is not None:
            set_verbosity_deepspeed(log_level_replica)
            set_verbosity_transformers(log_level_replica)


def set_random_seed(seed_):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * dist_env.get_pipeline_model_parallel_rank())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            dist_env.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError('Seed ({}) should be a positive integer.'.format(seed))       # noqa
