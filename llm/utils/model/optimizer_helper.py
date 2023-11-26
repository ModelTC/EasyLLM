# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import apex
from .sophia import SophiaG


def _get_params_for_weight_decay_optimization(model):
    """Divide params into with-weight-decay and without-weight-decay groups.
    Layernorms and baises will have no weight decay but the rest will.
    """

    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}

    for module_ in model.modules():
        # if isinstance(module_, MixedFusedLayerNorm):
        if module_.__class__.__name__ == "MixedFusedLayerNorm":
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                    if p is not None])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                    if p is not None and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                    if p is not None and n == 'bias'])

    # XXX: temp hack to workaround the crash in apex FusedAdam's multi_tensor_applier
    #
    # it crashes when the param count is larger than a certain size which we hit at 200B over 80
    # A100 gpus - I think around 2.7B per gpu, so halving it works around the issue
    param_count = len(weight_decay_params['params'])
    first_half = weight_decay_params['params'][:param_count // 2]
    second_half = weight_decay_params['params'][param_count // 2:]

    first_half = {'params': first_half}
    second_half = {'params': second_half}

    return first_half, second_half, no_weight_decay_params


def build_cls_instance(module, cfg):
    """Build instance for given cls"""
    cls = getattr(module, cfg['type'])
    return cls(**cfg['kwargs'])


def filter_freeze_param_groups(param_groups):
    filter_param_groups = []
    for _, param_group in enumerate(param_groups):
        trainable_parameters = [param for param in param_group['params'] if param.requires_grad]
        if len(trainable_parameters) > 0:
            filter_param_groups.append(param_group)
    return filter_param_groups


def build_optimizer(cfg_optim, model, deepspeed=True):
    if cfg_optim.get('cpu_optimizer', False):
        raise NotImplementedError('need to add cpu adam')

    # Base optimizer.
    param_groups = _get_params_for_weight_decay_optimization(model)
    param_groups = filter_freeze_param_groups(param_groups)

    optim_type = cfg_optim['type']
    cfg_optim['kwargs']['params'] = param_groups
    if optim_type == 'Adam8bit':
        try:
            import bitsandbytes as bnb
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install bitsandbytes from https://github.com/facebookresearch/bitsandbytes.")
        optimizer = build_cls_instance(bnb.optim, cfg_optim)
    elif cfg_optim['type'] in ['FusedAdam', 'FusedSGD', 'FusedNovoGrad']:
        optimizer = build_cls_instance(apex.optimizers, cfg_optim)
    elif cfg_optim['type'] in ['SophiaG']:
        optimizer = SophiaG(**cfg_optim['kwargs'])
    else:
        optimizer = build_cls_instance(torch.optim, cfg_optim)

    if deepspeed:
        return optimizer
    else:
        raise NotImplementedError
