# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Input/output checkpointing."""

import os
import random
import sys

import numpy as np

import torch

from megatron.core import mpu, tensor_parallel, dist_checkpointing
from megatron.core.dist_checkpointing.serialization import get_default_load_sharded_strategy
from megatron.core.dist_checkpointing.strategies.fully_parallel import \
    FullyParallelLoadStrategyWrapper
from megatron.core.num_microbatches_calculator import update_num_microbatches
from megatron.training.global_vars import get_args
from megatron.training.utils import unwrap_model, print_rank_0

# [ModelOpt]: Import
try:
    from modelopt.torch.opt.plugins import (
        restore_modelopt_state,
        restore_sharded_modelopt_state,
    )
    has_nvidia_modelopt = True
except Exception:
    has_nvidia_modelopt = False

from llm.utils.general.log_helper import default_logger as logger
from megatron.training.checkpointing import set_checkpoint_version, get_checkpoint_version


def _load_base_checkpoint(load_dir, rank0=False, sharded_state_dict=None,
                          exit_on_missing_checkpoint=False, checkpoint_step=None):
    """ Load the base state_dict from the given directory

    If rank0 is true, just loads rank 0 checkpoint, ignoring arguments.
    """
    import glob

    # Checkpoint.
    if rank0:
        checkpoint_name = find_checkpoint_rank_0(load_dir, iteration, release)  # noqa
        is_dist_ckpt = checkpoint_name is not None and dist_checkpointing.check_is_distributed_checkpoint(checkpoint_name)
    else:
        is_dist_ckpt = False
        # load_dir = cfg_loader["load_path"]
        filenames = glob.glob(os.path.join(load_dir, '*.bin'))
        # for ceph & safetensors support
        if len(filenames) == 0:
            filenames = glob.glob(os.path.join(load_dir, '*.safetensors'))
            if len(filenames) == 0:
                ceph_filenames = glob.glob(os.path.join(load_dir, '*.ceph'))
                if len(ceph_filenames) > 0:
                    ceph_paths = []
                    for item in ceph_filenames:
                        with open(item, "r") as f:
                            ceph_paths.append(f.readlines()[0].strip())
                    filenames = ceph_paths
                else:
                    return False

    # Load the checkpoint.
    if is_dist_ckpt:
        if rank0:
            state_dict = dist_checkpointing.load_common_state_dict(checkpoint_name)
            return state_dict, checkpoint_name, release  # noqa

        # at this point args are available
        args = get_args()
        if sharded_state_dict is None:
            assert not args.auto_detect_ckpt_format and not args.use_dist_ckpt, (args.auto_detect_ckpt_format, args.use_dist_ckpt)
            raise RuntimeError('Detected load from a distributed checkpoint, but neither --use-dist-ckpt nor --auto-detect-ckpt-format is set.')

        load_strategy = get_default_load_sharded_strategy(checkpoint_name)
        if args.ckpt_fully_parallel_load:
            load_strategy = FullyParallelLoadStrategyWrapper(load_strategy,
                                                             mpu.get_data_parallel_group(with_context_parallel=True))
        state_dict = dist_checkpointing.load(sharded_state_dict, checkpoint_name, load_strategy, strict=args.dist_ckpt_strictness)
        return state_dict, checkpoint_name, release  # noqa

    try:
        # state_dict = torch.load(checkpoint_name, map_location='cpu')
        def reader(filename):
            logger.info(f"loadding {filename}")
            if "s3://" in filename:
                dt = PetrelHelper.load(filename, map_location='cpu')  # noqa
            elif filename.endswith(".safetensors"):
                from safetensors.torch import load_file as safe_load_file
                dt = safe_load_file(filename)
            else:
                dt = torch.load(filename, map_location='cpu')
            return dt

        # input_models_map = {f: reader(f) for f in filenames}
        state_dict = {f: reader(f) for f in filenames}
    except ModuleNotFoundError:
        from megatron.legacy.fp16_deprecated import loss_scaler  # noqa
        # For backward compatibility.
        if not rank0:
            print_rank_0(' > deserializing using the old code structure ...')
        sys.modules['fp16.loss_scaler'] = sys.modules[
            'megatron.legacy.fp16_deprecated.loss_scaler']
        sys.modules['megatron.fp16.loss_scaler'] = sys.modules[
            'megatron.legacy.fp16_deprecated.loss_scaler']
        sys.modules['megatron.model'] = sys.modules['megatron.legacy.model']
        state_dict = torch.load(checkpoint_name, map_location='cpu')
        sys.modules.pop('fp16.loss_scaler', None)
        sys.modules.pop('megatron.fp16.loss_scaler', None)
        sys.modules.pop('megatron.model', None)
    except BaseException as e:
        print('could not load the checkpoint')
        print(e)
        sys.exit()

    return state_dict


class NotDivisibleError(Exception):
    def __init__(self, denominator, molecule, error_info):
        super().__init__()
        self._error_info = error_info
        self._molecule = molecule
        self._denominator = denominator

    def __str__(self):
        if self._error_info is None:
            return f"{self._denominator} is not divisible by {self._molecule}"
        else:
            return self._error_info.format(self._denominator, self._molecule)


class NotEqualError(Exception):
    def __init__(self, tensor_a, tensor_b, error_info):
        super().__init__()
        self._error_info = error_info
        self._tensor_a = tensor_a
        self._tensor_b = tensor_b

    def __str__(self):
        if self._error_info is None:
            return f"{self._tensor_a} is not equal to {self._tensor_b}"
        else:
            return self._error_info.format(self._tensor_a, self._tensor_b)


def check_equal(tensor_a, tensor_b, error_info=None):
    if tensor_a == tensor_b:
        return
    raise NotEqualError(tensor_a, tensor_b, error_info)


def check_divisible(denominator, molecule, error_info=None):
    if denominator % molecule == 0:
        return
    raise NotDivisibleError(denominator, molecule, error_info)


def row_split(w, tp, r):
    if w is None:
        return None
    h = w.shape[0]
    check_divisible(h, tp)
    part_len = h // tp
    return w[r * part_len: (r + 1) * part_len, ...].clone()


def column_split(w, tp, r):
    if w is None:
        return None
    dim1 = w.shape[1]
    check_divisible(dim1, tp)
    part_len = dim1 // tp
    return w[:, r * part_len: (r + 1) * part_len].clone()


def permute_qkv_weight(w, model_config, split=False):
    """
    adapt for ascendspeed llama qkv layer
    Notation:
        n_head: Number of attention heads,
        kv_heads: Number of key and value heads,
        tp: Tensor model parallel size,
        np: Number of attention heads in per tensor partition,
        gp: Number of key and value heads in per tensor partition,
    """
    n_head, hidden_size, tp, kv_heads = model_config
    if kv_heads is None:
        kv_heads = n_head

    check_divisible(n_head, tp)
    check_divisible(hidden_size, n_head)
    check_divisible(kv_heads, tp)
    check_divisible(n_head, kv_heads)
    np = n_head // tp
    gp = kv_heads // tp
    repeats = np // gp
    hn = hidden_size // n_head
    w_s0, w_s1 = w.shape
    check_equal(w_s0, (repeats + 2) * gp * hn)
    if not split:
        q, k, v = w.split([gp * repeats * hn, gp * hn, gp * hn], 0)
        return torch.cat([q.reshape(gp, repeats * hn, -1),
                          k.reshape(gp, hn, -1),
                          v.reshape(gp, hn, -1)], 1).reshape(w_s0, w_s1).contiguous().clone()
    q, k, v = w.reshape(gp, -1, w_s1).split([repeats * hn, hn, hn], 1)
    return torch.cat([q.reshape(-1, w_s1),
                      k.reshape(-1, w_s1),
                      v.reshape(-1, w_s1)], 0).reshape(w_s0, w_s1).contiguous().clone()


def load_checkpoint(model, optimizer, opt_param_scheduler, cfg_loader, load_arg='load', strict=True):
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    """
    args = get_args()
    # load_dir = getattr(args, load_arg)
    load_dir = cfg_loader["load_path"]

    model = unwrap_model(model)

    load_kwargs = {}
    is_dist_ckpt = False

    load_dir = cfg_loader["load_path"]
    weight_map = _load_base_checkpoint(load_dir, rank0=False, **load_kwargs)
    state_dict, total_count = {}, 0
    for key in weight_map:
        total_count += len(weight_map[key])
        state_dict = {**state_dict, **weight_map[key]}
    assert len(state_dict) == total_count
    iteration, release = 0, True

    # Checkpoint not loaded.
    if state_dict is None:
        # Iteration and num_floating_point_operations_so_far default to 0.
        return 0, 0

    # Set checkpoint version.
    set_checkpoint_version(state_dict.get('checkpoint_version', 0))

    num_floating_point_operations_so_far = state_dict.get('num_floating_point_operations_so_far', 0)

    # Check arguments.
    assert args.consumed_train_samples == 0
    assert args.consumed_valid_samples == 0
    if 'args' in state_dict and not args.finetune:
        checkpoint_args = state_dict['args']
        check_checkpoint_args(checkpoint_args)  # noqa
        args.consumed_train_samples = getattr(checkpoint_args,
                                              'consumed_train_samples', 0)
        update_num_microbatches(consumed_samples=args.consumed_train_samples)
        args.consumed_valid_samples = getattr(checkpoint_args,
                                              'consumed_valid_samples', 0)
    else:
        print_rank_0('could not find arguments in the checkpoint ...')

    # [ModelOpt]: loading modelopt_state (sharded or not)
    if has_nvidia_modelopt:
        if args.use_dist_ckpt:
            restore_sharded_modelopt_state(model, checkpoint_name)  # noqa
        else:
            restore_modelopt_state(model, state_dict)

    # Model.
    strict = False if args.retro_add_retriever else strict

    def get_weight_from(weight_map, layer_name):
        if layer_name in weight_map:
            return weight_map[layer_name]
        return None
    from functools import partial
    get_weight_from_name = partial(get_weight_from, state_dict)

    n_layer, n_heads, hidden_size = args.num_layers, args.num_attention_heads, args.hidden_size
    num_kv_heads = args.num_query_groups  # n_heads
    pp_rank, pp_size = mpu.get_pipeline_model_parallel_rank(), mpu.get_pipeline_model_parallel_world_size()
    tp_size, tp_rank = mpu.get_tensor_model_parallel_world_size(), mpu.get_tensor_model_parallel_rank()
    pp_n_layer = n_layer // pp_size
    # llama
    emb_w = get_weight_from_name("model.embed_tokens.weight")
    model_state_dict = model[0].state_dict()
    if pp_rank == 0:
        # model_state_dict["language_model.embedding.word_embeddings.weight"].copy_(row_split(emb_w, tp_size, tp_rank))
        model_state_dict["embedding.word_embeddings.weight"].copy_(row_split(emb_w, tp_size, tp_rank))
    if pp_rank == pp_size - 1:
        # model_state_dict["language_model.encoder.final_layernorm.weight"].copy_(get_weight_from_name("model.norm.weight").clone())
        model_state_dict["decoder.final_layernorm.weight"].copy_(get_weight_from_name("model.norm.weight").clone())
        # model_state_dict["language_model.output_layer.weight"].copy_(row_split(get_weight_from_name("lm_head.weight"), tp_size, tp_rank))
        model_state_dict["output_layer.weight"].copy_(row_split(get_weight_from_name("lm_head.weight"), tp_size, tp_rank))

    def layer_update(pp_i, parts):
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pre_idx = 0
        for _ in range(pp_rank):
            pre_idx += parts[_]
        ori_i = pre_idx + pp_i
        # ori_i = pp_n_layer * pp_rank + pp_i
        qw = row_split(get_weight_from_name(f"model.layers.{ori_i}.self_attn.q_proj.weight"), tp_size, tp_rank)
        kw = row_split(get_weight_from_name(f"model.layers.{ori_i}.self_attn.k_proj.weight"), tp_size, tp_rank)
        vw = row_split(get_weight_from_name(f"model.layers.{ori_i}.self_attn.v_proj.weight"), tp_size, tp_rank)
        permute_w = permute_qkv_weight(torch.cat([qw, kw, vw], dim=0), (n_heads, hidden_size, tp_size, num_kv_heads))
        # model_state_dict[f"language_model.encoder.layers.{pp_i}.self_attention.query_key_value.weight"].copy_(permute_w)
        model_state_dict[f"decoder.layers.{pp_i}.self_attention.linear_qkv.weight"].copy_(permute_w)
        # model_state_dict[f"language_model.encoder.layers.{pp_i}.self_attention.dense.weight"].copy_(column_split(
        model_state_dict[f"decoder.layers.{pp_i}.self_attention.linear_proj.weight"].copy_(column_split(
            get_weight_from_name(f"model.layers.{ori_i}.self_attn.o_proj.weight"), tp_size, tp_rank))

        gate_proj = row_split(
            get_weight_from_name(f"model.layers.{ori_i}.mlp.gate_proj.weight"), tp_size, tp_rank)
        up_proj = row_split(
            get_weight_from_name(f"model.layers.{ori_i}.mlp.up_proj.weight"), tp_size, tp_rank)
        gate_up_proj = row_split(
            get_weight_from_name(f"model.layers.{ori_i}.mlp.gate_up_proj.weight"), tp_size, tp_rank)
        if gate_up_proj is not None:
            model_state_dict[f"decoder.layers.{pp_i}.mlp.linear_fc1.weight"].copy_(gate_up_proj)
        else:
            # model_state_dict[f"language_model.encoder.layers.{pp_i}.mlp.proj.weight"].copy_(torch.cat(
            model_state_dict[f"decoder.layers.{pp_i}.mlp.linear_fc1.weight"].copy_(torch.cat(
                [gate_proj, up_proj], 0).contiguous().clone())
        # model_state_dict[f"language_model.encoder.layers.{pp_i}.mlp.dense_4h_to_h.weight"].copy_(column_split(
        model_state_dict[f"decoder.layers.{pp_i}.mlp.linear_fc2.weight"].copy_(column_split(
            get_weight_from_name(f"model.layers.{ori_i}.mlp.down_proj.weight"), tp_size, tp_rank))
        # model_state_dict[f"language_model.encoder.layers.{pp_i}.input_layernorm.weight"].copy_(get_weight_from_name(
        model_state_dict[f"decoder.layers.{pp_i}.input_layernorm.weight"].copy_(get_weight_from_name(
            f"model.layers.{ori_i}.input_layernorm.weight").clone())
        # model_state_dict[f"language_model.encoder.layers.{pp_i}.post_attention_layernorm.weight"].copy_(get_weight_from_name(
        model_state_dict[f"decoder.layers.{pp_i}.pre_mlp_layernorm.weight"].copy_(get_weight_from_name(
            f"model.layers.{ori_i}.post_attention_layernorm.weight").clone())

    parts = args.pp_partition_parts
    pp_n_layer = parts[mpu.get_pipeline_model_parallel_rank()]
    for pp_i in range(pp_n_layer):
        layer_update(pp_i, parts)

    # Fix up query/key/value matrix ordering if needed.
    checkpoint_version = get_checkpoint_version()
    print_rank_0(f' checkpoint version {checkpoint_version}')
    # fix_query_key_value_ordering(model, checkpoint_version)

    # Optimizer.
    if not release and not args.finetune and not args.no_load_optim:
        try:
            # Load state dict.
            if optimizer is not None and state_dict.get("optimizer", None):
                optimizer.load_state_dict(state_dict['optimizer'])

            # Load distributed optimizer's custom parameter state.
            # For distributed checkpoint it's already loaded in load_state_dict above
            if args.use_distributed_optimizer and not is_dist_ckpt:
                tracker_filename = get_checkpoint_tracker_filename(load_dir)  # noqa
                iteration, release = read_metadata(tracker_filename)  # noqa
                model_checkpoint_name = \
                    get_checkpoint_name(load_dir, iteration, release)  # noqa
                optim_checkpoint_name = \
                    get_distributed_optimizer_checkpoint_name(  # noqa
                        model_checkpoint_name)
                optimizer.load_parameter_state(optim_checkpoint_name)

            # Load scheduler.
            if opt_param_scheduler is not None:
                if 'lr_scheduler' in state_dict and state_dict.get("lr_scheduler", None):  # backward compatbility
                    opt_param_scheduler.load_state_dict(state_dict['lr_scheduler'])
                elif state_dict.get("opt_param_scheduler", None):
                    opt_param_scheduler.load_state_dict(state_dict['opt_param_scheduler'])
        except KeyError:
            print_rank_0('Unable to load optimizer from checkpoint {}. '
                         'Specify --no-load-optim or --finetune to prevent '
                         'attempting to load the optimizer state, '
                         'exiting ...'.format(checkpoint_name))  # noqa
            sys.exit()
    else:
        if (args.fp16 or args.bf16) and optimizer is not None:
            optimizer.reload_model_params()

    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            if 'rng_state' in state_dict:
                # access rng_state for data parallel rank
                if args.data_parallel_random_init:
                    rng_state = state_dict['rng_state'][mpu.get_data_parallel_rank()]
                else:
                    rng_state = state_dict['rng_state'][0]
                random.setstate(rng_state['random_rng_state'])
                np.random.set_state(rng_state['np_rng_state'])
                torch.set_rng_state(rng_state['torch_rng_state'])
                torch.cuda.set_rng_state(rng_state['cuda_rng_state'])
                # Check for empty states array
                if not rng_state['rng_tracker_states']:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(
                    rng_state['rng_tracker_states'])
            else:  # backward compatability
                if state_dict.get("random_rng_state", None):
                    random.setstate(state_dict['random_rng_state'])
                    np.random.set_state(state_dict['np_rng_state'])
                    torch.set_rng_state(state_dict['torch_rng_state'])
                    torch.cuda.set_rng_state(state_dict['cuda_rng_state'])
                    # Check for empty states array
                    if not state_dict['rng_tracker_states']:
                        raise KeyError
                    tensor_parallel.get_cuda_rng_tracker().set_states(
                        state_dict['rng_tracker_states'])
                else:
                    print_rank_0("Random_rng_state not in state_dict.")
        except KeyError:
            # print_rank_0('Unable to load rng state from checkpoint {}. '
            #              'Specify --no-load-rng or --finetune to prevent '
            #              'attempting to load the rng state, '
            #              'exiting ...'.format(checkpoint_name))
            sys.exit()

    # Some utilities want to load a checkpoint without distributed being initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(f'  successfully loaded checkpoint from {load_dir} '
                 f'[ t {mpu.get_tensor_model_parallel_rank()}, '
                 f'p {mpu.get_pipeline_model_parallel_rank()} ] '
                 f'at iteration {iteration}')

    return iteration, num_floating_point_operations_so_far
