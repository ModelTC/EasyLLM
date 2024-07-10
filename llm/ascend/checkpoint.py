import os
import torch
import glob
from llm.utils.general.error_utils import check_divisible, check_equal
from llm.utils.general.log_helper import default_logger as logger
from llm.utils.tools.petrel_helper import PetrelHelper
from llm.utils.env import dist_env

make_vocab_size_divisible_by = 128
added_token_num = 0

def row_split(w, tp, r):
    if w is None:
        return None
    h = w.shape[0]
    check_divisible(h, tp)
    part_len = h // tp
    return w[r * part_len: (r + 1) * part_len, ...].clone()

def vocab_size_with_padding(orig_vocab_size, make_vocab_size_divisible_by, tp):
    after = orig_vocab_size
    multiple = make_vocab_size_divisible_by * tp
    while (after % multiple) != 0:
        after += 1
    return after

def column_split(w, tp, r):
    if w is None:
        return None
    dim1 = w.shape[1]
    check_divisible(dim1, tp)
    part_len = dim1 // tp
    return w[:, r * part_len: (r + 1) * part_len].clone()

def pad_embed(w, make_vocab_size_divisible_by, tp, added_token_num):
    padded_size = vocab_size_with_padding(w.shape[0] + added_token_num, make_vocab_size_divisible_by, tp)
    if padded_size == w.shape[0]:
        return w.clone()
    return torch.cat([w, w[-(padded_size - w.shape[0]):, ...]], dim=0)

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
                          v.reshape(gp, hn, -1)], 0).reshape(w_s0, w_s1).contiguous().clone()
    q, k, v = w.reshape(gp, -1, w_s1).split([repeats * hn, hn, hn], 1)
    return torch.cat([q.reshape(-1, w_s1),
                      k.reshape(-1, w_s1),
                      v.reshape(-1, w_s1)], 0).reshape(w_s0, w_s1).contiguous().clone()


def get_weight_from(weight_map, layer_name):
    if layer_name in weight_map:
        return weight_map[layer_name]
    return None

def copy_param(param, tensor):
    if tensor is not None:
        param.copy_(tensor)
    return param


class Woker:
    def __init__(self, worker_num) -> None:
        self.worker_num = worker_num
    
    def __call__(self, dst, keys):
        from multiprocessing.pool import ThreadPool as Pool
        with Pool(self.worker_num) as p:
            res = p.map(dst, keys)


def load_ckpt_from_pretrain(cfg_loader, model, optimizer, tp_rank, tp_size, pp_rank, pp_size, n_layer, hidden_size, n_heads, num_kv_heads, worker, pretrain_type):
    load_dir = cfg_loader["load_path"]
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

    def reader(filename):
        logger.info(f"loadding {filename}")
        if "s3://" in filename:
            dt = PetrelHelper.load(filename, map_location='cpu')
        elif filename.endswith(".safetensors"):
            from safetensors.torch import load_file as safe_load_file
            dt = safe_load_file(filename)
        else:
            dt = torch.load(filename, map_location='cpu')
        return dt

    input_models_map = {f: reader(f) for f in filenames}
    weight_map, total_count = {}, 0
    for key in input_models_map:
        total_count += len(input_models_map[key])
        weight_map = {**weight_map, **input_models_map[key]}
    assert len(weight_map) == total_count
    from functools import partial

    get_weight_from_name = partial(get_weight_from, weight_map)

    pp_n_layer = n_layer // pp_size
    if pretrain_type == 'llama':
        emb_w = get_weight_from_name("model.embed_tokens.weight")
    if pretrain_type == 'internlm2':
        emb_w = get_weight_from_name("model.tok_embeddings.weight")
    # emb_w = pad_embed(emb_w, make_vocab_size_divisible_by, tp_size, added_token_num)
    state_dict = model.state_dict()
    if pp_rank == 0:
        state_dict["language_model.embedding.word_embeddings.weight"].copy_(row_split(emb_w, tp_size, tp_rank))

    if pp_rank == pp_size - 1:
        if pretrain_type == 'llama':
            state_dict["language_model.encoder.final_layernorm.weight"].copy_(get_weight_from_name("model.norm.weight").clone())
            state_dict["language_model.output_layer.weight"].copy_(row_split(get_weight_from_name("lm_head.weight"), tp_size, tp_rank))
        if pretrain_type == 'internlm2':
            state_dict["language_model.encoder.final_layernorm.weight"].copy_(get_weight_from_name("model.norm.weight").clone())
            state_dict["language_model.output_layer.weight"].copy_(row_split(get_weight_from_name("output.weight"), tp_size, tp_rank))    
        # state_dict["language_model.output_layer.weight"].copy_(row_split(
        #     pad_embed(get_weight_from_name("lm_head.weight"), make_vocab_size_divisible_by,
        #                 tp_size, added_token_num), tp_size, tp_rank))
    def layer_update(pp_i):
        ori_i = pp_n_layer * pp_rank + pp_i
        qw = row_split(get_weight_from_name(f"model.layers.{ori_i}.self_attn.q_proj.weight"), tp_size, tp_rank)
        kw = row_split(get_weight_from_name(f"model.layers.{ori_i}.self_attn.k_proj.weight"), tp_size, tp_rank)
        vw = row_split(get_weight_from_name(f"model.layers.{ori_i}.self_attn.v_proj.weight"), tp_size, tp_rank)
        permute_w = permute_qkv_weight(torch.cat([qw, kw, vw], dim=0), (n_heads, hidden_size, tp_size, num_kv_heads))
        state_dict[f"language_model.encoder.layers.{pp_i}.self_attention.query_key_value.weight"].copy_(permute_w)
        state_dict[f"language_model.encoder.layers.{pp_i}.self_attention.dense.weight"].copy_(column_split(
            get_weight_from_name(f"model.layers.{ori_i}.self_attn.o_proj.weight"), tp_size, tp_rank))

        gate_proj = row_split(
            get_weight_from_name(f"model.layers.{ori_i}.mlp.gate_proj.weight"), tp_size, tp_rank)
        up_proj = row_split(
            get_weight_from_name(f"model.layers.{ori_i}.mlp.up_proj.weight"), tp_size, tp_rank)
        state_dict[f"language_model.encoder.layers.{pp_i}.mlp.proj.weight"].copy_(torch.cat(
            [gate_proj, up_proj], 0).contiguous().clone())
        state_dict[f"language_model.encoder.layers.{pp_i}.mlp.dense_4h_to_h.weight"].copy_(column_split(
            get_weight_from_name(f"model.layers.{ori_i}.mlp.down_proj.weight"), tp_size, tp_rank))
        state_dict[f"language_model.encoder.layers.{pp_i}.input_layernorm.weight"].copy_(get_weight_from_name(
            f"model.layers.{ori_i}.input_layernorm.weight").clone())
        state_dict[f"language_model.encoder.layers.{pp_i}.post_attention_layernorm.weight"].copy_(get_weight_from_name(
            f"model.layers.{ori_i}.post_attention_layernorm.weight").clone())

    def internlm2_layer_update(pp_i):
        ori_i = pp_n_layer * pp_rank + pp_i
        qkv = get_weight_from_name(f"model.layers.{ori_i}.attention.wqkv.weight")
        gs = n_heads // num_kv_heads
        head_dim = hidden_size // n_heads
        qkv = qkv.reshape(-1, gs + 2, head_dim, hidden_size)
        qw = row_split(qkv[:, :gs, ...], tp_size, tp_rank).reshape(-1, hidden_size)
        kw = row_split(qkv[:, -2, ...], tp_size, tp_rank).reshape(-1, hidden_size)
        vw = row_split(qkv[:, -1, ...], tp_size, tp_rank).reshape(-1, hidden_size)
        qkv_w = torch.cat([qw, kw, vw], dim=0)
        state_dict[f"language_model.encoder.layers.{pp_i}.self_attention.query_key_value.weight"].copy_(qkv_w)
        state_dict[f"language_model.encoder.layers.{pp_i}.self_attention.dense.weight"].copy_(column_split(
            get_weight_from_name(f"model.layers.{ori_i}.attention.wo.weight"), tp_size, tp_rank))

        gate_proj = row_split(
            get_weight_from_name(f"model.layers.{ori_i}.feed_forward.w1.weight"), tp_size, tp_rank)
        up_proj = row_split(
            get_weight_from_name(f"model.layers.{ori_i}.feed_forward.w3.weight"), tp_size, tp_rank)
        state_dict[f"language_model.encoder.layers.{pp_i}.mlp.proj.weight"].copy_(torch.cat(
            [gate_proj, up_proj], 0).contiguous().clone())
        state_dict[f"language_model.encoder.layers.{pp_i}.mlp.dense_4h_to_h.weight"].copy_(column_split(
            get_weight_from_name(f"model.layers.{ori_i}.feed_forward.w2.weight"), tp_size, tp_rank))
        state_dict[f"language_model.encoder.layers.{pp_i}.input_layernorm.weight"].copy_(get_weight_from_name(
            f"model.layers.{ori_i}.attention_norm.weight").clone())
        state_dict[f"language_model.encoder.layers.{pp_i}.post_attention_layernorm.weight"].copy_(get_weight_from_name(
            f"model.layers.{ori_i}.ffn_norm.weight").clone())

    for pp_i in range(pp_n_layer):
        if pretrain_type == 'llama':
            layer_update(pp_i)
        if pretrain_type == 'internlm2':
            internlm2_layer_update(pp_i)

    if cfg_loader["deepspeed"]:
        torch.npu.empty_cache()
    else:
        optimizer.reload_model_params()
    return True


def load_ckpt_pretrained(cfg_loader, model, optimizer):
    if cfg_loader.get("debug", False):
        success = True
    else:
        if cfg_loader['load_mode'] == 'huggingface':
            worker = cfg_loader.get('worker', 8)
            tp_rank = dist_env.get_tensor_model_parallel_rank()
            tp_size = dist_env.get_tensor_model_parallel_world_size()
            pp_rank = dist_env.get_pipeline_model_parallel_rank()
            pp_size = dist_env.get_pipeline_model_parallel_world_size()
            success = load_ckpt_from_pretrain(cfg_loader, model, optimizer, tp_rank, tp_size, pp_rank, pp_size, 
                                               model.model_kwargs.get('num_layers'),
                                               model.model_kwargs.get('hidden_size'),
                                               model.model_kwargs.get('num_attention_heads'),
                                               model.model_kwargs.get('num_kv_heads'),
                                                worker=worker, pretrain_type=cfg_loader.get('pretrain_type', 'llama'))
        else:
            logger.error("Load Llama by the {} load mode is not support now".format(cfg_loader['load_mode']))
            raise NotImplementedError
    if success:
        logger.info(f"Successfully loaded checkpoint from {cfg_loader['load_path']}.")
    else:
        logger.info("Fail to load any checkpoints and will start from random.")
