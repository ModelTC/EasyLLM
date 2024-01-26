import os
import re
import glob
import copy
import json

import torch

from llm.utils.env import dist_env
from llm.utils.general.log_helper import default_logger as logger
from llm.utils.tools.petrel_helper import PetrelHelper
from llm.models.mg_models.base_modules.layers import VocabParallelEmbedding, ColumnParallelLinear, RowParallelLinear
from llm.models.mg_models.base_modules.lora import (
    LoraVocabParallelEmbedding,
    LoraColumnParallelLinear,
    LoraRowParallelLinear)
from llm.models.mg_models.llama.positional_embeddings import RotaryEmbedding


def get_fp32_params_from_key(key, model, num_layers, lora_mode=False):
    if lora_mode:
        key = key.replace('base_model.model.', '')
    key_list = key.split('.')
    try:
        module = None
        if 'layers' in key:
            layer_num = int(key_list[2])
            param_key = key_list[3] + '.' + key_list[4] if key_list[3] in ('self_attn', 'mlp') else key_list[3]
            module = model.get_submodule(f'module.{layer_num + 3}.{param_key}')
            if lora_mode:
                key = key.replace('lora_A.weight', 'lora_A_weight')
                key = key.replace('lora_B.weight', 'lora_B_weight')
                key_list = key.split('.')
        else:    # ['model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight'])
            if 'model.embed_tokens' in key:
                module = model.get_submodule('module.1.word_embeddings')
                if lora_mode:
                    key = key.replace('lora_embedding_A', 'lora_embedding_A_weight')
                    key = key.replace('lora_embedding_B', 'lora_embedding_B_weight')
                    key_list = key.split('.')
            elif 'model.norm' in key:
                module = model.get_submodule(f'module.{num_layers + 4}')
            elif 'lm_head' in key:
                module = model.get_submodule((f'module.{num_layers + 5}.word_embeddings'))  # last layer
                if lora_mode:
                    key = key.replace('lora_A.weight', 'lora_embedding_A_weight')
                    key = key.replace('lora_B.weight', 'lora_embedding_B_weight')
                    key_list = key.split('.')
        if module:
            return getattr(module, key_list[-1]), module
        else:
            return None, None
    except AttributeError:
        # no module found, possible in different pipeline
        return None, None


def qwen_to_llama(dt, model):
    output_dt = {}
    hid_size = model.transformer_layer_params['hidden_size']
    num_layers = model.model_kwargs['num_layers']
    for key in dt.keys():
        if 'wte.weight' in key:
            output_dt['module.1.word_embeddings.weight'] = dt[key]
        elif 'ln_f.weight' in key:
            output_dt[f'module.{num_layers + 4}.weight'] = dt[key]
        elif 'lm_head.weight' in key:
            output_dt[f'module.{num_layers + 5}.word_embeddings.weight'] = dt[key]
        elif "transformer.h" in key:
            layer_id = int(key.split('.')[2])
            if 'attn.c_attn.weight' in key:
                qkv_weight = dt[key]
                s_size = qkv_weight.shape[0] // 3
                qkv_weight = torch.split(qkv_weight,
                                         dim=0,
                                         split_size_or_sections=[s_size, s_size, s_size])
                q_weight = qkv_weight[0].reshape(-1, hid_size)
                k_weight = qkv_weight[1].reshape(-1, hid_size)
                v_weight = qkv_weight[2].reshape(-1, hid_size)
                output_dt[f"module.{layer_id + 3}.self_attn.q_proj.weight"] = q_weight
                output_dt[f"module.{layer_id + 3}.self_attn.k_proj.weight"] = k_weight
                output_dt[f"module.{layer_id + 3}.self_attn.v_proj.weight"] = v_weight
            if 'attn.c_attn.bias' in key:
                qkv_bias = dt[key]
                s_size = qkv_bias.shape[0] // 3
                qkv_bias = torch.split(qkv_bias,
                                       dim=0,
                                       split_size_or_sections=[s_size, s_size, s_size])
                q_bias = qkv_bias[0].reshape(s_size)
                k_bias = qkv_bias[1].reshape(s_size)
                v_bias = qkv_bias[2].reshape(s_size)
                output_dt[f"module.{layer_id + 3}.self_attn.q_proj.bias"] = q_bias
                output_dt[f"module.{layer_id + 3}.self_attn.k_proj.bias"] = k_bias
                output_dt[f"module.{layer_id + 3}.self_attn.v_proj.bias"] = v_bias
            if 'attn.c_proj' in key:
                output_dt[f'module.{layer_id + 3}.self_attn.o_proj.weight'] = dt[key]
            if 'mlp.w2.weight' in key:
                output_dt[f'module.{layer_id + 3}.mlp.gate_proj.weight'] = dt[key]
            if 'mlp.c_proj.weight' in key:
                output_dt[f'module.{layer_id + 3}.mlp.down_proj.weight'] = dt[key]
            if 'mlp.w1.weight' in key:
                output_dt[f'module.{layer_id + 3}.mlp.up_proj.weight'] = dt[key]
            if 'ln_2.weight' in key:
                output_dt[f'module.{layer_id + 3}.post_attention_layernorm.weight'] = dt[key]
            if 'ln_1.weight' in key:
                output_dt[f'module.{layer_id + 3}.input_layernorm.weight'] = dt[key]
        else:
            logger.info(f"unuse keys {key}")
    return output_dt


def internlm2_to_llama2(dt, model, pack=False):
    output_dt = {}
    n_heads = model.transformer_layer_params['num_attention_heads']
    n_kv_heads = model.transformer_layer_params['num_kv_attention_heads']
    hid_size = model.transformer_layer_params['hidden_size']
    num_layers = model.model_kwargs['num_layers']
    gs = n_heads // n_kv_heads
    head_dim = hid_size // n_heads
    for key in dt.keys():
        if 'tok_embeddings' in key:
            output_dt['module.1.word_embeddings.weight'] = dt['model.tok_embeddings.weight']
        elif 'model.norm.weight' in key:
            output_dt[f'module.{num_layers + 4}.weight'] = dt[key]
        elif 'output' in key:
            output_dt[f'module.{num_layers + 5}.word_embeddings.weight'] = dt[key]
        elif "layers" in key:
            layer_id = int(key.split('.')[2])
            if 'attention.wqkv' in key:
                qkv = dt[key]
                qkv = qkv.reshape(-1, gs + 2, head_dim, hid_size)
                q = qkv[:, :gs, ...].reshape(-1, hid_size)
                k = qkv[:, -2, ...].reshape(-1, hid_size)
                v = qkv[:, -1, ...].reshape(-1, hid_size)
                if pack:
                    output_dt[f"module.{layer_id + 3}.self_attn.wqkv.weight"] = [q, k, v]
                else:
                    output_dt[f"module.{layer_id + 3}.self_attn.q_proj.weight"] = q
                    output_dt[f"module.{layer_id + 3}.self_attn.k_proj.weight"] = k
                    output_dt[f"module.{layer_id + 3}.self_attn.v_proj.weight"] = v
            if 'attention.wo' in key:
                output_dt[f'module.{layer_id + 3}.self_attn.o_proj.weight'] = dt[key]
            if 'feed_forward.w1' in key:
                output_dt[f'module.{layer_id + 3}.mlp.gate_proj.weight'] = dt[key]
            if 'feed_forward.w2' in key:
                output_dt[f'module.{layer_id + 3}.mlp.down_proj.weight'] = dt[key]
            if 'feed_forward.w3' in key:
                output_dt[f'module.{layer_id + 3}.mlp.up_proj.weight'] = dt[key]
            if 'ffn_norm.weight' in key:
                output_dt[f'module.{layer_id + 3}.post_attention_layernorm.weight'] = dt[key]
            if 'attention_norm.weight' in key:
                output_dt[f'module.{layer_id + 3}.input_layernorm.weight'] = dt[key]
        else:
            logger.info(f"unuse keys {key}")
    return output_dt


def hf_to_megatron_llama(dt, model, pack=False):
    output_dt = {}
    num_layers = model.model_kwargs['num_layers']
    for key in dt.keys():
        if 'model.embed_tokens' in key:
            output_dt['module.1.word_embeddings.weight'] = dt['model.embed_tokens.weight']
        elif 'model.norm' in key:
            output_dt[f'module.{num_layers + 4}.weight'] = dt[key]
        elif 'lm_head' in key:
            output_dt[f'module.{num_layers + 5}.word_embeddings.weight'] = dt[key]
        elif "layers" in key:
            layer_id = int(key.split('.')[2])
            if 'self_attn.q_proj' in key:
                if pack:
                    assert (key.replace("q_proj", "k_proj") in dt) and (key.replace("q_proj", "v_proj") in dt), "Pack mode only suport q_proj, k_proj, v_proj in one checkpoint, please re-formula your checkpoint file."      # noqa
                    wqkv_val = [dt[key], dt[key.replace("q_proj", "k_proj")], dt[key.replace("q_proj", "v_proj")]]
                    if 'weight' in key:
                        output_dt[f"module.{layer_id + 3}.self_attn.wqkv.weight"] = wqkv_val
                    else:
                        output_dt[f"module.{layer_id + 3}.self_attn.wqkv.bias"] = wqkv_val
                else:
                    if 'weight' in key:
                        output_dt[f"module.{layer_id + 3}.self_attn.q_proj.weight"] = dt[key]
                    else:
                        output_dt[f"module.{layer_id + 3}.self_attn.q_proj.bias"] = dt[key]
            if 'self_attn.k_proj' in key:
                if 'weight' in key:
                    output_dt[f'module.{layer_id + 3}.self_attn.k_proj.weight'] = dt[key]
                else:
                    output_dt[f'module.{layer_id + 3}.self_attn.k_proj.bias'] = dt[key]
            if 'self_attn.v_proj' in key:
                if 'weight' in key:
                    output_dt[f'module.{layer_id + 3}.self_attn.v_proj.weight'] = dt[key]
                else:
                    output_dt[f'module.{layer_id + 3}.self_attn.v_proj.bias'] = dt[key]
            if 'self_attn.o_proj' in key:
                if 'weight' in key:
                    output_dt[f'module.{layer_id + 3}.self_attn.o_proj.weight'] = dt[key]
                else:
                    output_dt[f'module.{layer_id + 3}.self_attn.o_proj.bias'] = dt[key]
            if 'gate_proj' in key:
                output_dt[f'module.{layer_id + 3}.mlp.gate_proj.weight'] = dt[key]
            if 'down_proj' in key:
                output_dt[f'module.{layer_id + 3}.mlp.down_proj.weight'] = dt[key]
            if 'up_proj' in key:
                output_dt[f'module.{layer_id + 3}.mlp.up_proj.weight'] = dt[key]
            if 'post_attention_layernorm' in key:
                output_dt[f'module.{layer_id + 3}.post_attention_layernorm.weight'] = dt[key]
            if 'input_layernorm' in key:
                output_dt[f'module.{layer_id + 3}.input_layernorm.weight'] = dt[key]
        else:
            logger.info(f"unuse keys {key}")
    return output_dt


def get_module_param(name, model):
    module = None
    try:
        keys = name.split('.')
        module = model.get_submodule(name[:-(len(keys[-1]) + 1)])
        param = getattr(module, keys[-1])
    except: # noqa
        param = None
    return param, module


def get_start_end(size, tp_world_size, tp_rank):
    base_block_size = size
    res = {}
    end = base_block_size
    begin = 0
    for i in range(tp_world_size):
        res[i] = [begin, end]
        begin = end
        end += base_block_size
    return res[tp_rank]


def load_func(filename, tp_rank, tp_world_size, model, num_layers, lora_mode, pretrain_type, init_set):
    logger.info(f"loadding {filename}")
    if "s3://" in filename:
        dt = PetrelHelper.load(filename, map_location='cpu')
    elif filename.endswith(".safetensors"):
        from safetensors.torch import load_file as safe_load_file
        dt = safe_load_file(filename)
    else:
        dt = torch.load(filename, map_location='cpu')

    is_pack = False
    if 'internlm2' in pretrain_type:
        assert pretrain_type in ["internlm2", "internlm2_pack"]
        if pretrain_type == "internlm2_pack":
            is_pack = True
        dt = internlm2_to_llama2(dt, model, pack=is_pack)
    elif pretrain_type == "qwen":
        dt = qwen_to_llama(dt, model)
    elif 'llama' in pretrain_type:
        assert pretrain_type in ["llama", "llama_pack"]
        if pretrain_type == "llama_pack":
            is_pack = True
        dt = hf_to_megatron_llama(dt, model, pack=is_pack)
    else:
        logger.info(f"{pretrain_type} is not be supported")

    for name in dt.keys():
        param, module = get_module_param(name, model)
        # param, module = get_fp32_params_from_key(name, model, num_layers, lora_mode)
        if module is None:
            continue
        slice_ = dt[name]
        if isinstance(module, VocabParallelEmbedding) and ((not lora_mode) or ('lora' not in name) or ('lora_embedding_A' in name)):  # noqa
            start, stop = get_start_end(param.shape[0], tp_world_size, tp_rank)
            tensor = slice_[start:stop]
        elif isinstance(module, ColumnParallelLinear) and ((not lora_mode) or ('lora_B' in name)):
            if is_pack and isinstance(slice_, list):
                q_start, q_stop = get_start_end(slice_[0].shape[0] // tp_world_size, tp_world_size, tp_rank)
                k_start, k_stop = get_start_end(slice_[1].shape[0] // tp_world_size, tp_world_size, tp_rank)
                v_start, v_stop = get_start_end(slice_[2].shape[0] // tp_world_size, tp_world_size, tp_rank)
                q_slice_ = slice_[0][q_start:q_stop]
                k_slice_ = slice_[1][k_start:k_stop]
                v_slice_ = slice_[2][v_start:v_stop]
                tensor = torch.cat([q_slice_, k_slice_, v_slice_], dim=0)
            else:
                start, stop = get_start_end(param.shape[0], tp_world_size, tp_rank)
                tensor = slice_[start:stop]
        elif isinstance(module, RowParallelLinear) and ((not lora_mode) or ('lora_A' in name)):
            if len(param.shape) == 1:  # for bias
                tensor = slice_
            else:
                start, stop = get_start_end(param.shape[1], tp_world_size, tp_rank)
                tensor = slice_[:, start:stop]
        else:
            tensor = slice_[:]

        if param is None and isinstance(module, RotaryEmbedding):
            continue
        elif param.shape != tensor.shape:
            print(param.shape, tensor.shape, module)
            n_tensor = param.data.clone()
            n_tensor[:tensor.size(0), :] = tensor
            temp = slice_.sum(1).argmin().item()
            n_tensor[tensor.size(0):] = slice_[temp, ] * 0.1
            param.data.copy_(n_tensor)
        else:
            # raise ValueError(f"Name {name}, module: {module} -- Current {param.shape} and got {tensor.shape}")
            param.data.copy_(tensor)
        init_set.add(name)


def load_llama_from_hf_format(load_dir,
                              model, optimizer,
                              num_layers, lora_mode=False,
                              worker=8, pretrain_type='llama'):
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
    tp_rank = dist_env.get_tensor_model_parallel_rank()
    tp_world_size = dist_env.get_tensor_model_parallel_world_size()
    init_set = set()
    # for filename in filenames[3:]:
    #     load_func(filename, tp_rank, tp_world_size, model, num_layers, lora_mode, pretrain_type, init_set)
    # import ipdb; ipdb.set_trace()
    from functools import partial
    from multiprocessing.pool import ThreadPool as Pool
    partial_func = partial(load_func, tp_rank=tp_rank, tp_world_size=tp_world_size, model=model, num_layers=num_layers, lora_mode=lora_mode, pretrain_type=pretrain_type, init_set=init_set)  # noqa
    with Pool(worker) as p:
        _ = p.map(partial_func, filenames)
    mega_model_keys = set(model.state_dict().keys())
    not_init_set = mega_model_keys - init_set
    filter_not_init_set = set()
    for k in not_init_set:
        if 'inv_freq' in k:
            pass
        else:
            filter_not_init_set.add(k)
    logger.info(f"not init set {filter_not_init_set}")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    # fix optimizer
    if optimizer is not None and hasattr(optimizer, 'refresh_fp32_params'):
        optimizer.refresh_fp32_params()
    return True


def load_lora_from_ds_format(load_dir, model, optimizer):
    # only load pth file for lora finetune models
    filenames = glob.glob(os.path.join(load_dir, 'layer_*.pt'))
    if len(filenames) == 0:
        ceph_filenames = glob.glob(os.path.join(load_dir, 'layer_*.ceph'))
        if len(ceph_filenames) > 0:
            ceph_paths = []
            for item in ceph_filenames:
                with open(item, "r") as f:
                    ceph_paths.append(f.readlines()[0].strip())
            filenames = ceph_paths
        else:
            return False
    layers_tps = {}
    for filename in filenames:
        layer_id_tp_id = re.findall(r"\d+", filename.split('/')[-1])
        assert len(layer_id_tp_id) == 2, 'model names must be (layer_[layer_id]-model_[tp_id]-model_states) format'
        lid, tip = int(layer_id_tp_id[0]), int(layer_id_tp_id[1])
        if lid not in layers_tps:
            layers_tps[lid] = {}
        layers_tps[lid][tip] = filename

    tp_rank = dist_env.get_tensor_model_parallel_rank()
    tp_world_size = dist_env.get_tensor_model_parallel_world_size()
    pt_tp_world_size = -1
    for lid in layers_tps:
        if pt_tp_world_size == -1:
            pt_tp_world_size = len(layers_tps[lid])
            assert pt_tp_world_size == tp_world_size, 'tp_size must match with the lora pt tp_size'
        tp_dt = []
        for pti in range(pt_tp_world_size):
            if pti not in layers_tps[lid]:
                raise ValueError(f"Pt file of layer {lid}, tp_rank {pti} is not found")
            filename = layers_tps[lid][pti]
            logger.info(f'loading layer_{lid}/tp_{pti}: {filename}')
            if "s3://" in filename:
                dt = PetrelHelper.load(filename, map_location='cpu')
            else:
                dt = torch.load(filename, map_location='cpu')
            tp_dt.append(dt)
        for name in tp_dt[0].keys():
            try:
                key_lists = name.split('.')
                module = model.get_submodule(f'module.{lid}.{".".join(key_lists[:-1])}')
                param = getattr(module, key_lists[-1])
            except BaseException:
                param, module = None, None
            if module is None:
                continue
            slice_ = [tp_dt[j][name] for j in range(pt_tp_world_size)]
            if isinstance(module, VocabParallelEmbedding) and \
                    'lora_embedding_A_weight' in name:
                slice_ = torch.cat(slice_, dim=0)
                print(f'embeding shape: {param.shape} {slice_.shape}')
                size = slice_.shape[0]
                block_size = size // tp_world_size
                start = tp_rank * block_size
                stop = (tp_rank + 1) * block_size
                tensor = slice_[start:stop]
            elif isinstance(module, ColumnParallelLinear) and \
                    'lora_B_weight' in name:
                slice_ = torch.cat(slice_, dim=0)
                size = slice_.shape[0]
                block_size = size // tp_world_size
                start = tp_rank * block_size
                stop = (tp_rank + 1) * block_size
                tensor = slice_[start:stop]
            elif isinstance(module, RowParallelLinear) and \
                    'lora_A_weight' in name:
                slice_ = torch.cat(slice_, dim=1)
                size = slice_.shape[1]
                block_size = size // tp_world_size
                start = tp_rank * block_size
                stop = (tp_rank + 1) * block_size
                tensor = slice_[:, start:stop]
            else:
                tensor = slice_[tp_rank]

            if param.shape != tensor.shape:
                raise ValueError(f"Name {name}, module: {module} -- Current {param.shape} and got {tensor.shape}")
            param.data.copy_(tensor)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    # fix optimizer
    if optimizer is not None and hasattr(optimizer, 'refresh_fp32_params'):
        optimizer.refresh_fp32_params()
    return True


def load_lora_ckpt_pretrained(cfg_lora, model, optimizer):
    if cfg_lora['loader'].get("debug", False):
        lora_success = True
    else:
        lora_load_mode = cfg_lora['loader'].get("load_mode", "deepspeed")
        if lora_load_mode == 'deepspeed':
            lora_success = load_lora_from_ds_format(cfg_lora['loader']['load_path'],
                                                    model, optimizer)
        elif lora_load_mode == 'huggingface':
            worker = cfg_lora['loader'].get('worker', 8)
            lora_success = load_llama_from_hf_format(cfg_lora['loader']['load_path'], model,
                                                     optimizer, model.model_kwargs['num_layers'],
                                                     lora_mode=True, worker=worker)
        else:
            logger.error('lora load mode {} is not support'.format(lora_load_mode))
            raise NotImplementedError
    if lora_success:
        if cfg_lora['loader'].get("debug", False):
            logger.info("skip loaded checkpoint for debug")
        else:
            logger.info(f"Successfully loaded lora checkpoint from {cfg_lora['loader']['load_path']}.")
    else:
        logger.info("Fail to load any lora checkpoints and will start from random.")


def load_ckpt_pretrained(cfg_loader, model, optimizer):
    if cfg_loader.get("debug", False):
        success = True
    else:
        if cfg_loader['load_mode'] == 'huggingface':
            worker = cfg_loader.get('worker', 8)
            success = load_llama_from_hf_format(cfg_loader['load_path'], model,
                                                optimizer, model.model_kwargs['num_layers'],
                                                worker=worker, pretrain_type=cfg_loader.get('pretrain_type', 'llama'))
        else:
            logger.error("Load Llama by the {} load mode is not support now".format(cfg_loader['load_mode']))
            raise NotImplementedError
    if success:
        logger.info(f"Successfully loaded checkpoint from {cfg_loader['load_path']}.")
    else:
        logger.info("Fail to load any checkpoints and will start from random.")


def save_lora_ckpt_pretrained(cfg_lora, model, iteration=None):
    from deepspeed.runtime import utils as ds_utils
    torch.distributed.barrier()
    dp_rank = dist_env.get_data_parallel_rank()
    dp_size = dist_env.get_data_parallel_world_size()
    num_layers = len(model.forward_funcs)
    config_name = cfg_lora['saver'].get('save_config_name', 'adapter_config.json')
    peft_template = {"alpha_pattern": {}, "auto_mapping": None, "bias": "none", "modules_to_save": [],
                     "init_lora_weights": True, "layers_pattern": None, "target_modules": [],
                     "fan_in_fan_out": False, "inference_mode": True,
                     "layers_to_transform": None, "peft_type": "LORA", "task_type": "CAUSAL_LM"}

    assert cfg_lora['saver'].get('save_mode', 'deepspeed') == 'deepspeed', 'only support deepspeed lora save mode now!'

    if model.checkpoint_parallel_write_pipeline:
        offsets = ds_utils.partition_uniform(num_layers, dp_size)
        start, end = offsets[dp_rank], offsets[dp_rank + 1]
    else:
        # data parallel rank 0 writes all layers
        if dp_rank != 0:
            pass
        else:
            start, end = 0, num_layers
    if model.checkpoint_parallel_write_pipeline or dp_rank == 0:
        layer_list = model.forward_funcs[start:end]
        sv_path = os.path.join(cfg_lora['saver']['save_path'],
                               cfg_lora['saver'].get('save_tag', f"global_step{iteration}"))
        os.makedirs(sv_path, exist_ok=True)
        for idx, layer in enumerate(layer_list):
            model_ckpt_path = model.ckpt_layer_path(sv_path, start + idx)
            if not hasattr(layer, 'state_dict'):
                continue
            orig_state_dict = layer.state_dict()
            lora_state_dict = {}
            for k, v in orig_state_dict.items():
                requires_save = ("lora_" in k)
                if (cfg_lora['saver']['modules_to_save'] is not None) and ('word_embeddings' in cfg_lora['saver']['modules_to_save']):      # noqa
                    requires_save = (requires_save or ('1.word_embeddings' in k))
                if (cfg_lora['saver']['modules_to_save'] is not None) and ('lm_head' in cfg_lora['saver']['modules_to_save']):      # noqa
                    requires_save = (requires_save or (('1.word_embeddings' not in k) and ('word_embeddings' in k)))
                if requires_save:
                    lora_state_dict[k] = v.clone()
            if len(lora_state_dict) == 0:
                continue
            final_state_dict = type(orig_state_dict)(lora_state_dict)
            model.checkpoint_engine.save(final_state_dict, model_ckpt_path)
        if dist_env.is_pipeline_first_stage() and dist_env.get_tensor_model_parallel_rank() == 0:
            config_path = os.path.join(sv_path, config_name)
            config = copy.deepcopy(peft_template)
            config['lora_alpha'] = cfg_lora['lora_alpha']
            config['lora_dropout'] = cfg_lora['lora_dropout']
            config['r'] = cfg_lora['lora_rank']
            if 'word_embeddings' in cfg_lora['saver']['modules_to_save']:
                config['modules_to_save'].append('embed_tokens')
            if 'lm_head' in cfg_lora['saver']['modules_to_save']:
                config['modules_to_save'].append('lm_head')
            if ('word_embeddings' in cfg_lora['target_modules']) or ('lm_head' in cfg_lora['target_modules']):
                raise NotImplementedError
            config['target_modules'].extend(cfg_lora['target_modules'])
            config['base_model_name_or_path'] = cfg_lora['base_model_name_or_path']
            # save it
            with open(config_path, "w") as writer:
                writer.write(json.dumps(config, indent=2, sort_keys=True))
    torch.distributed.barrier()


def set_train_status(model, lora_mode=False):
    model.train()
    if lora_mode:
        model.eval()
        # set lora modules to train
        lora_module = [LoraVocabParallelEmbedding,
                       LoraColumnParallelLinear,
                       LoraRowParallelLinear]
        for n, m in model.named_modules():
            for lm in lora_module:
                if isinstance(m, lm):
                    m.train()


def set_train_params(model, lora_mode=False, cfg_lora=None, is_train=True):
    for n, p in model.named_parameters():
        if not is_train:
            p.requires_grad = False
        elif lora_mode:
            assert cfg_lora is not None
            requires_grad = ("lora_" in n)
            modules_to_save = cfg_lora['saver'].get('modules_to_save', None)
            if (modules_to_save is not None) and ('word_embeddings' in modules_to_save):
                requires_grad = (requires_grad or ('1.word_embeddings' in n))
            if (modules_to_save is not None) and ('lm_head' in modules_to_save):
                requires_grad = (requires_grad or (('1.word_embeddings' not in n) and ('word_embeddings' in n)))
            if not requires_grad:
                p.requires_grad = False
            else:
                p.requires_grad = True
        else:
            p.requires_grad = True
