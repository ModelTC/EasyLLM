# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
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
import argparse
import json
import math
import os
import re
import glob
import shutil

import torch

from transformers import LlamaConfig
from llm.utils.tools.petrel_helper import PetrelHelper
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

"""
Sample usage:

```
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

Thereafter, models can be loaded via:

```py
from transformers import LlamaForCausalLM, LlamaTokenizer

model = LlamaForCausalLM.from_pretrained("/output/path")
tokenizer = LlamaTokenizer.from_pretrained("/output/path")
```

Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).
"""

WEIGHTS_WITH_ROW_PARALLELISM_CONTAIN = [
    "mlp.down_proj.weight",
    "self_attn.o_proj.weight",
]

LORA_WEIGHTS_WITH_ROW_PARALLELISM_CONTAIN = [
    "mlp.down_proj.lora_A_weight",
    "self_attn.o_proj.lora_A_weight",
]

WEIGHTS_WITH_COLUMN_PARALLELISM_CONTAIN = [
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "word_embeddings.weight",
]

LORA_WEIGHTS_WITH_COLUMN_PARALLELISM_CONTAIN = [
    "mlp.gate_proj.lora_B_weight",
    "mlp.up_proj.lora_B_weight",
    "self_attn.q_proj.lora_B_weight",
    "self_attn.k_proj.lora_B_weight",
    "self_attn.v_proj.lora_B_weight",
    "word_embeddings.lora_embedding_A_weight",
]

WEIGHTS_TO_AVERAGE_ENDSWITH = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "weight",
    "self_attn.rotary_emb.inv_freq",
]

LORA_WEIGHTS_TO_AVERAGE_ENDSWITH = [
    "mlp.gate_proj.lora_A_weight",
    "mlp.up_proj.lora_A_weight",
    "self_attn.q_proj.lora_A_weight",
    "self_attn.k_proj.lora_A_weight",
    "self_attn.v_proj.lora_A_weight",
    "word_embeddings.lora_embedding_B_weight",
    "mlp.down_proj.lora_B_weight",
    "self_attn.o_proj.lora_B_weight",
]


def get_ceph_path(path):
    ceph_bucket = os.environ.get('CEPHBUCKET')
    if ceph_bucket != '':
        if ceph_bucket[-1] != '/':
            ceph_bucket += '/'
        # remove /
        if path[0] == '/':
            path = path[1:]
        # remove ./
        if path[:2] == './':
            path = path[2:]
        ceph_path = ceph_bucket + path
    else:
        ceph_path = path
    return ceph_path


def ceph_save(state_dict, path):
    ceph_path = get_ceph_path(path)
    if "s3://" in path:
        path = path[5:]
    with open(path + '.ceph', 'w') as f:
        print(ceph_path, file=f)
    PetrelHelper.save(state_dict, ceph_path)


def compute_intermediate_size(n):
    return int(math.ceil(n * 8 / 3) + 255) // 256 * 256


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--tp_size",
        default=4, type=int,
        help="tp size",
    )
    parser.add_argument(
        "--dim",
        default=4096, type=int,
        help="hidden size",
    )
    parser.add_argument(
        "--n_heads",
        default=32, type=int,
        help="num attention heads",
    )
    parser.add_argument(
        "--n_layers",
        default=32, type=int,
        help="num hidden layers",
    )
    parser.add_argument(
        "--intermediate_size",
        default=-1, type=int,
        help="intermediate size",
    )
    parser.add_argument(
        "--num_key_value_heads",
        default=-1, type=int,
        help="num key value heads",
    )
    parser.add_argument(
        "--norm_eps",
        default=1e-06, type=float,
        help="tp size",
    )
    parser.add_argument(
        "--lora_mode",
        action='store_true',
        help="convert lora weight",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    args = parser.parse_args()

    if args.lora_mode:
        write_lora_model_fast(args)
    else:
        write_model_fast(args)


def read_tp_dt(tp_size, ltl, lid):
    assert tp_size == len(ltl), 'tp_size must match with the pt tp_size'
    tp_dt = []
    for pti in range(tp_size):
        if pti not in ltl:
            raise ValueError(f"Pt file of layer {lid}, tp_rank {pti} is not found")
        filename = ltl[pti]
        print(f'loading layer_{lid}/tp_{pti}: {filename}')
        if "s3://" in filename:
            dt = PetrelHelper.load(filename, map_location='cpu')
        else:
            dt = torch.load(filename, map_location='cpu')
        tp_dt.append(dt)
    return tp_dt


def load_pt_files(load_dir):
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
            raise ValueError

    layers_tps = {}
    for filename in filenames:
        layer_id_tp_id = re.findall(r"\d+", filename.split('/')[-1])
        assert len(layer_id_tp_id) == 2, 'model names must be (layer_[layer_id]-model_[tp_id]-model_states) format'
        lid, tip = int(layer_id_tp_id[0]), int(layer_id_tp_id[1])
        if lid not in layers_tps:
            layers_tps[lid] = {}
        layers_tps[lid][tip] = filename
    layers_tps_list = []
    n_layers = max(layers_tps.keys()) - 5
    for k, v in layers_tps.items():
        layers_tps_list.append((k, v))
    return layers_tps_list, n_layers


def load_save_func(layers_tp,
                   tp_size=1,
                   output_dir='./',
                   n_layer=100,
                   param_counts=[],
                   index_dict={"weight_map": {}}):
    special_layers = [1, n_layer + 4, n_layer + 5]
    layer_id = layers_tp[0]
    tps = layers_tp[1]
    state_dict = {}
    if layer_id in special_layers:
        return
    start = str(layer_id - 2).zfill(5)
    end = str(n_layer + 1).zfill(5)
    filename = f"pytorch_model-{start}-of-{end}.bin"
    tp_dt = read_tp_dt(tp_size, tps, layer_id)
    for name in tp_dt[0].keys():
        key = f"model.layers.{layer_id - 3}." + name
        if name in WEIGHTS_TO_AVERAGE_ENDSWITH:
            val = sum([tp_dt[s][name] for s in range(tp_size)])
            val /= tp_size
        elif name in WEIGHTS_WITH_COLUMN_PARALLELISM_CONTAIN:
            val = torch.cat([tp_dt[s][name] for s in range(tp_size)], dim=0)
        elif name in WEIGHTS_WITH_ROW_PARALLELISM_CONTAIN:
            val = torch.cat([tp_dt[s][name] for s in range(tp_size)], dim=1)
        else:
            assert False, 'Not recongnized key {}'.format(key)
        state_dict[key] = val
    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_counts.append(v.numel())
    file_path = os.path.join(output_dir, filename)
    if os.environ.get('CEPHBUCKET', None):
        ceph_save(state_dict, file_path)
    else:
        torch.save(state_dict, file_path)

    print(f"Save pytorch_model-{start}-of-{end}.bin successfully!")


def load_save_lora_func(layers_tp,
                        tp_size=1,
                        n_layer=100,
                        state_dict={}):
    special_layers = [1, n_layer + 5]
    layer_id = layers_tp[0]
    tps = layers_tp[1]

    if layer_id in special_layers:
        return

    tp_dt = read_tp_dt(tp_size, tps, layer_id)
    for name in tp_dt[0].keys():
        key = f"base_model.model.model.layers.{layer_id - 3}." + name
        key = key.replace('lora_A_weight', 'lora_A.weight')
        key = key.replace('lora_B_weight', 'lora_B.weight')

        if name in LORA_WEIGHTS_TO_AVERAGE_ENDSWITH:
            val = sum([tp_dt[s][name] for s in range(tp_size)])
            val /= tp_size
        elif name in LORA_WEIGHTS_WITH_COLUMN_PARALLELISM_CONTAIN:
            val = torch.cat([tp_dt[s][name] for s in range(tp_size)], dim=0)
        elif name in LORA_WEIGHTS_WITH_ROW_PARALLELISM_CONTAIN:
            val = torch.cat([tp_dt[s][name] for s in range(tp_size)], dim=1)
        else:
            assert False, 'Not recongnized key {}'.format(key)
        state_dict[key] = val


def write_lora_model_fast(args):
    model_path = args.output_dir
    load_dir = args.input_dir
    tp_size = args.tp_size
    # lora_mode = args.lora_mode
    if model_path == load_dir:
        if model_path[-1] != "/":
            model_path = model_path + "/"
        model_path = model_path + "convert_hf"
    model_path_dir = model_path
    if "s3://" in model_path_dir:
        model_path_dir = model_path_dir[5:]
    os.makedirs(model_path_dir, exist_ok=True)
    load_dir_tmp = load_dir
    if "s3://" in load_dir_tmp:
        load_dir_tmp = load_dir_tmp[5:]
    layers_tps, n_layer = load_pt_files(load_dir_tmp)

    filename = f"adapter_model.bin"
    config_name = "adapter_config.json"
    # for embendding and norm lm head
    state_dict = {}
    for layers_tp in layers_tps:
        layer_id = layers_tp[0]
        tps = layers_tp[1]
        special_layers = [1, n_layer + 5]
        if layer_id not in special_layers:
            continue
        tp_dt = read_tp_dt(tp_size, tps, layer_id)
        if layer_id == special_layers[0]:
            for name in tp_dt[0].keys():
                key = name.replace('word_embeddings', 'base_model.model.model.embed_tokens')
                key = key.replace('lora_embedding_A_weight', 'lora_embedding_A')
                key = key.replace('lora_embedding_B_weight', 'lora_embedding_B')
                if (name in WEIGHTS_TO_AVERAGE_ENDSWITH) or (name in LORA_WEIGHTS_TO_AVERAGE_ENDSWITH):
                    val = sum([tp_dt[s][name] for s in range(tp_size)])
                    val /= tp_size
                elif (name in WEIGHTS_WITH_COLUMN_PARALLELISM_CONTAIN) or (name in LORA_WEIGHTS_WITH_COLUMN_PARALLELISM_CONTAIN): # noqa
                    val = torch.cat([tp_dt[s][name] for s in range(tp_size)], dim=0)
                else:
                    assert False, 'Not recongnized key {}'.format(key)
            state_dict[key] = val
        elif layer_id == special_layers[1]:
            for name in tp_dt[0].keys():
                key = name.replace('word_embeddings', 'base_model.model.lm_head')
                key = key.replace('lora_embedding_A_weight', 'lora_A.weight')
                key = key.replace('lora_embedding_B_weight', 'lora_B.weight')
                if (name in WEIGHTS_TO_AVERAGE_ENDSWITH) or (name in LORA_WEIGHTS_TO_AVERAGE_ENDSWITH):
                    val = sum([tp_dt[s][name] for s in range(tp_size)])
                    val /= tp_size
                elif (name in WEIGHTS_WITH_COLUMN_PARALLELISM_CONTAIN) or (name in LORA_WEIGHTS_WITH_COLUMN_PARALLELISM_CONTAIN):  # noqa
                    val = torch.cat([tp_dt[s][name] for s in range(tp_size)], dim=0)
                else:
                    assert False, 'Not recongnized key {}'.format(key)
                state_dict[key] = val

    print("start saving and loading")
    from functools import partial
    from multiprocessing.pool import ThreadPool as Pool
    partial_func = partial(load_save_lora_func,
                           tp_size=tp_size,
                           state_dict=state_dict,
                           n_layer=n_layer)
    worker = int(os.environ.get('LOADWORKER', 8))
    with Pool(worker) as p:
        _ = p.map(partial_func, layers_tps)

    file_path = os.path.join(model_path_dir, filename)
    if os.environ.get('CEPHBUCKET', None):
        ceph_save(state_dict, file_path)
    else:
        torch.save(state_dict, file_path)

    shutil.copy(os.path.join(load_dir_tmp, config_name),
                os.path.join(model_path_dir, config_name))

    if os.environ.get('CEPHBUCKET', None):
        with open(os.path.join(model_path_dir, config_name), "r") as f:
            PetrelHelper.write(f, get_ceph_path(os.path.join(model_path_dir, config_name)))


def write_model_fast(args):
    model_path = args.output_dir
    load_dir = args.input_dir
    tp_size = args.tp_size
    # lora_mode = args.lora_mode
    if model_path == load_dir:
        if model_path[-1] != "/":
            model_path = model_path + "/"
        model_path = model_path + "convert_hf"
    model_path_dir = model_path
    if "s3://" in model_path_dir:
        model_path_dir = model_path_dir[5:]
    os.makedirs(model_path_dir, exist_ok=True)
    load_dir_tmp = load_dir
    if "s3://" in load_dir_tmp:
        load_dir_tmp = load_dir_tmp[5:]
    layers_tps, n_layer = load_pt_files(load_dir_tmp)
    param_counts = []
    vocab_size = []
    index_dict = {"weight_map": {}}
    start = str(n_layer + 1).zfill(5)
    end = str(n_layer + 1).zfill(5)
    filename = f"pytorch_model-{start}-of-{end}.bin"
    # for embendding and norm lm head
    state_dict = {}
    for layers_tp in layers_tps:
        layer_id = layers_tp[0]
        tps = layers_tp[1]
        special_layers = [1, n_layer + 4, n_layer + 5]
        if layer_id not in special_layers:
            continue
        tp_dt = read_tp_dt(tp_size, tps, layer_id)
        if layer_id == special_layers[0]:
            assert len(tp_dt[0]) == 1, 'model.embed_tokens.weight dict have multi keys!'
            name = list(tp_dt[0].keys())[0]
            assert name == 'word_embeddings.weight', 'model.embed_tokens.weight only map with word_embeddings.weight!'
            val = torch.cat([tp_dt[s][name] for s in range(tp_size)], dim=0)
            state_dict['model.embed_tokens.weight'] = val
        elif layer_id == special_layers[1]:
            assert len(tp_dt[0]) == 1, 'model.norm.weight dict have multi keys!'
            name = list(tp_dt[0].keys())[0]
            assert name == 'weight', 'model.norm.weight only map with weight!'
            val = sum([tp_dt[s][name] for s in range(tp_size)])
            val /= tp_size
            state_dict['model.norm.weight'] = val
        elif layer_id == special_layers[2]:
            assert len(tp_dt[0]) == 1, 'lm_head.weight dict have multi keys!'
            name = list(tp_dt[0].keys())[0]
            assert name == 'word_embeddings.weight', 'lm_head.weight only map with word_embeddings.weight!'
            val = torch.cat([tp_dt[s][name] for s in range(tp_size)], dim=0)
            state_dict['lm_head.weight'] = val
            vocab_size = val.shape[0]

    file_path = os.path.join(model_path, filename)
    if os.environ.get('CEPHBUCKET', None):
        ceph_save(state_dict, file_path)
    else:
        torch.save(state_dict, file_path)

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_counts.append(v.numel())

    print("start saving and loading")
    from functools import partial
    from multiprocessing.pool import ThreadPool as Pool
    partial_func = partial(load_save_func,
                           tp_size=tp_size,
                           output_dir=model_path,
                           n_layer=n_layer,
                           param_counts=param_counts,
                           index_dict=index_dict)
    worker = int(os.environ.get('LOADWORKER', 8))
    with Pool(worker) as p:
        _ = p.map(partial_func, layers_tps)
    param_count = sum(param_counts)
    index_dict["metadata"] = {"total_size": param_count * 2}

    write_json(index_dict, os.path.join(model_path_dir, "pytorch_model.bin.index.json"))

    if args.intermediate_size < 0:
        intermediate_size = compute_intermediate_size(args.dim)
    else:
        intermediate_size = args.intermediate_size
    if args.num_key_value_heads < 0:
        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=args.dim,
            intermediate_size=intermediate_size,
            num_attention_heads=args.n_heads,
            num_hidden_layers=n_layer,
            rms_norm_eps=args.norm_eps,
        )
    else:
        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=args.dim,
            intermediate_size=intermediate_size,
            num_attention_heads=args.n_heads,
            num_key_value_heads=args.num_key_value_heads,
            num_hidden_layers=n_layer,
            rms_norm_eps=args.norm_eps,
        )

    config.save_pretrained(model_path_dir)

    if os.environ.get('CEPHBUCKET', None):
        local_files = os.listdir(model_path_dir)
        for lf in local_files:
            if lf.endswith('.json'):
                with open(os.path.join(model_path_dir, lf), "r") as f:
                    PetrelHelper.write(f, get_ceph_path(os.path.join(model_path_dir, lf)))
            elif lf.endswith('.model'):
                with open(os.path.join(model_path_dir, lf), "rb") as f:
                    PetrelHelper.write(f, get_ceph_path(os.path.join(model_path_dir, lf)))


if __name__ == "__main__":
    main()
