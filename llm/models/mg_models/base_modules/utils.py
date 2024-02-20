import math

import torch
from torch.nn.parallel import DistributedDataParallel as torchDDP

from llm.utils.env import dist_env

from llm.models.mg_models.base_modules.modules.enums import LayerType, AttnType, AttnMaskType, PositionEmbeddingType


def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def get_initializer_from_cfg(cfg):
    if cfg is None:
        return init_method_normal(sigma=0.02)
    elif cfg['type'] == 'normal':
        return init_method_normal(**cfg['kwargs'])
    elif cfg['type'] == 'scaled_normal':
        return scaled_init_method_normal(**cfg['kwargs'])
    else:
        raise NotImplementedError


def check_torch_dtype(params_dtype, fp16=False, bf16=False):
    if params_dtype is None:
        params_dtype = 'float'
        if fp16:
            assert not bf16
            params_dtype = 'half'
        if bf16:
            assert not fp16
            params_dtype = 'bfloat16'
    else:
        assert params_dtype in ['half', 'float', 'float16', 'float32', 'bfloat16']
        if params_dtype in ['half', 'float16']:
            assert fp16 and not bf16
        elif params_dtype in ['bfloat16']:
            assert bf16 and not fp16
        elif params_dtype in ['float', 'float32']:
            assert not bf16 and not fp16
        else:
            raise NotImplementedError
    return params_dtype


def get_torch_dtype(v):
    new_value = {'half': torch.half,
                 'float': torch.float,
                 'float16': torch.float16,
                 'float32': torch.float32,
                 'bfloat16': torch.bfloat16}[v]
    return new_value


def get_layer_type(v):
    new_value = {'encoder': LayerType.encoder,
                 'decoder': LayerType.decoder}[v]
    return new_value


def get_attn_type(v):
    new_value = {'self_attn': AttnType.self_attn,
                 'cross_attn': AttnType.cross_attn}[v]
    return new_value


def get_attn_mask_type(v):
    new_value = {'causal': AttnMaskType.causal,
                 'padding': AttnMaskType.padding,
                 'prefix': AttnMaskType.prefix,
                 'custom': AttnMaskType.custom}[v]
    return new_value


def get_position_embedding_type(v):
    new_value = {'rotary': PositionEmbeddingType.rotary,
                 'absolute': PositionEmbeddingType.absolute,
                 'alibi': PositionEmbeddingType.alibi,
                 'flash': PositionEmbeddingType.flash,
                 'dynamicntk': PositionEmbeddingType.dynamicntk}[v]
    return new_value


def check_keys_mapping(cfg_a, cfg_b):
    for key in cfg_a:
        if key in cfg_b:
            assert cfg_a[key] == cfg_b[key], 'the value of key {} of two configs do not match'.format(key)


@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x
                                       * (1.0 + 0.044715 * x * x)))


def openai_gelu(x):
    return gelu_impl(x)


# This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
@torch.jit.script
def erf_gelu(x):
    return x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype) + torch.ones_like(x).to(dtype=x.dtype))


def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, torch.finfo(attention_scores.dtype).min)
    return attention_scores


def unwrap_model(model, module_instances=(torchDDP)):
    unwrapped_model = model
    while isinstance(model, module_instances):
        unwrapped_model = model.module
    return unwrapped_model


def log_trainable_params(model):
    if dist_env.get_tensor_model_parallel_rank() == 0:
        trainable_params = 0
        all_param = 0
        for n, p in model.named_parameters():
            print(f"params: {n} || requires_grad: {p.requires_grad}")
            num_params = p.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(p, "ds_numel"):
                num_params = p.ds_numel
            all_param += num_params
            if p.requires_grad:
                trainable_params += num_params
        print(f"pp: {dist_env.get_pipeline_model_parallel_rank()} || trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")      # noqa


def recursive_getattr(model, module_name):
    """
    Recursively get the attribute of a module.
    Args:
        model (`torch.nn.Module`)
            The model to get the attribute from.
        module_name (`str`)
            The name of the module to get the attribute from.
    """
    split_list = module_name.split('.')
    output = model
    for name in split_list:
        output = getattr(output, name)
    return output


def recursive_setattr(model, module_name, module):
    """
    Recursively set the attribute of a module.
    Args:
        model (`torch.nn.Module`)
            The model to set the attribute in.
        module_name (`str`)
            The name of the module to set the attribute in.
        module (`torch.nn.Module`)
            The module to set the attribute to.
    """
    split_list = module_name.split('.')
    output = model
    for name in split_list[:-1]:
        output = getattr(output, name)
    output.__setattr__(split_list[-1], module)
