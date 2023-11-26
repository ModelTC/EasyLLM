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
import gc
import json
import math
import os
import shutil

import torch

from transformers import LlamaConfig, LlamaForCausalLM
from petrel_helper import PetrelHelper
from multiprocessing.pool import ThreadPool as Pool


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

INTERMEDIATE_SIZE_MAP = {
    "7B": 11008,
    "13B": 13824,
    "30B": 17920,
    "65B": 22016,
}
NUM_SHARDS = {
    "7B": 1,
    "13B": 2,
    "30B": 4,
    "65B": 8,
    "106B": 8
}

unsed_keys = ['model.feat_mask', 'model.ffn_mask', 'model.layer_mask', 'model.mask_scale']


def get_ceph_path(path):
    ceph_bucket = os.environ.get('CEPHBUCKET')
    if ceph_bucket[-1] != '/':
        ceph_bucket += '/'
    # remove /
    if path[0] == '/':
        path = path[1:]
    # remove ./
    if path[:2] == './':
        path = path[2:]
    ceph_path = ceph_bucket + path
    return ceph_path


def _save(state_dict, path):
    if os.environ.get('CEPHBUCKET', None) is None:
        torch.save(state_dict, path)
        return
    ceph_path = get_ceph_path(path)
    with open(path + '.ceph', 'w') as f:
        print(ceph_path, file=f)
    PetrelHelper.save(state_dict, ceph_path)


def compute_intermediate_size(n):
    return int(math.ceil(n * 8 / 3) + 255) // 256 * 256


def read_json(path):
    return PetrelHelper.load_json(path)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def find_max_layer_num(st):
    num = 0
    for i, v in st.items():
        if i in unsed_keys:
            continue
        if "layer" in i:
            num = max(int(i.split('.')[2]), num)
    return num + 1


def merge_pp(pps):
    accu_num = 0
    new_res = {}
    old_key_num = 0
    for pp in pps:
        max_num = find_max_layer_num(pp)
        for k, v in pp.items():
            if k in unsed_keys:
                continue
            old_key_num += 1
            if "layer" in k:
                new_k = ''
                split = k.split('.')
                new_layer = str(int(split[2]) + accu_num)
                split[2] = new_layer
                for j in range(len(split) - 1):
                    new_k += split[j] + '.'
                new_k += split[-1]
            else:
                new_k = k
            if new_k in new_res:
                print("repeat keys", k, new_k)
            new_res[new_k.replace('model.', '')] = v
        accu_num += max_num
    print("old key num: ", old_key_num, "new key num: ", len(list(new_res.keys())))
    return new_res


def process(path):
    idx = int(path.split('_')[2][2])
    st = torch.load(os.path.basename(path), map_location="cpu")
    pp[idx] = st


def write_model(model_path, input_base_path, model_size, pp_size):
    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    params = read_json(os.path.join(input_base_path, "params.json"))
    num_shards = NUM_SHARDS[model_size]
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    n_heads_per_shard = n_heads // num_shards
    dim = params["dim"]
    dims_per_head = dim // n_heads
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))

    # permute for sliced rotary
    def permute(w):
        return w.view(n_heads, dim // n_heads // 2, 2, dim).transpose(1, 2).reshape(dim, dim)

    def permute_gqa(w):
        return w.view(num_shards, dim // n_heads // 2, 2, dim).transpose(1, 2).reshape(num_shards * (dim // n_heads), dim)  # noqa

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    # Load weights
    if model_size == "7B":
        # Not sharded
        # (The sharded implementation would also work, but this is simpler.)
        loaded = PetrelHelper.load(os.path.join(input_base_path, "consolidated.00.pth"), map_location="cpu")
    else:
        # Sharded
        loaded = []
        if pp_size == 1:
            loaded = [
                PetrelHelper.load(os.path.join(input_base_path, f"tp_{i:01d}.pt"), map_location="cpu")
                for i in range(num_shards)
            ]
        else:
            for i in range(num_shards):
                global pp
                pp = [0] * pp_size
                paths = []
                for j in range(pp_size):
                    paths.append(os.path.join(input_base_path, f"model_tp{i:01d}_pp{j:01d}.pt"))
                with Pool(8) as p:
                    _ = p.map(process, paths)
                new_st = merge_pp(pp)
                pp.clear()
                loaded.append(new_st)
    param_count = 0
    index_dict = {"weight_map": {}}
    for layer_i in range(n_layers):
        filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
        if model_size == "7B":
            # Unsharded
            state_dict = {
                f"model.layers.{layer_i}.self_attn.q_proj.weight": permute(
                    loaded[f"layers.{layer_i}.attention.wq.weight"]
                ),
                f"model.layers.{layer_i}.self_attn.k_proj.weight": permute(
                    loaded[f"layers.{layer_i}.attention.wk.weight"]
                ),
                f"model.layers.{layer_i}.self_attn.v_proj.weight": loaded[f"layers.{layer_i}.attention.wv.weight"],
                f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded[f"layers.{layer_i}.attention.wo.weight"],
                f"model.layers.{layer_i}.mlp.gate_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w1.weight"],
                f"model.layers.{layer_i}.mlp.down_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w2.weight"],
                f"model.layers.{layer_i}.mlp.up_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w3.weight"],
                f"model.layers.{layer_i}.input_layernorm.weight": loaded[f"layers.{layer_i}.attention_norm.weight"],
                f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[f"layers.{layer_i}.ffn_norm.weight"],
            }
        else:
            # Sharded
            # Note that in the 13B checkpoint, not cloning the two following weights will result in the checkpoint
            # becoming 37GB instead of 26GB for some reason.
            state_dict = {
                f"model.layers.{layer_i}.input_layernorm.weight": loaded[0][
                    f"layers.{layer_i}.attention_norm.weight"
                ].clone(),
                f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[0][
                    f"layers.{layer_i}.ffn_norm.weight"
                ].clone(),
            }
            state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = permute(
                torch.cat(
                    [
                        loaded[i][f"layers.{layer_i}.attention.wq.weight"].view(n_heads_per_shard, dims_per_head, dim)
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(dim, dim)
            )
            if pp_size == 1:
                state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = permute(
                    torch.cat(
                        [loaded[i][f"layers.{layer_i}.attention.wq.weight"].view(n_heads_per_shard, dims_per_head, dim)
                            for i in range(num_shards)],
                        dim=0,
                    ).reshape(dim, dim)
                )
            else:
                state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = permute_gqa(
                    torch.cat(
                        [
                            loaded[i][f"layers.{layer_i}.attention.wk.weight"].view(1, dims_per_head, dim)
                            for i in range(num_shards)
                        ],
                        dim=0,
                    ).reshape(-1, dim)
                )
            if pp_size == 1:
                state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = permute(
                    torch.cat(
                        [loaded[i][f"layers.{layer_i}.attention.wk.weight"].view(n_heads_per_shard, dims_per_head, dim)
                            for i in range(num_shards)],
                        dim=0,
                    ).reshape(dim, dim)
                )
            else:
                state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = torch.cat(
                    [
                        loaded[i][f"layers.{layer_i}.attention.wv.weight"].view(-1, dims_per_head, dim)
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(-1, dim)

            state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.attention.wo.weight"] for i in range(num_shards)], dim=1
            )
            state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w1.weight"] for i in range(num_shards)], dim=0
            )
            state_dict[f"model.layers.{layer_i}.mlp.down_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w2.weight"] for i in range(num_shards)], dim=1
            )
            state_dict[f"model.layers.{layer_i}.mlp.up_proj.weight"] = torch.cat(
                [loaded[i][f"layers.{layer_i}.feed_forward.w3.weight"] for i in range(num_shards)], dim=0
            )

        state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq
        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        _save(state_dict, os.path.join(tmp_model_path, filename))

    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
    if model_size == "7B":
        # Unsharded
        state_dict = {
            "model.embed_tokens.weight": loaded["tok_embeddings.weight"],
            "model.norm.weight": loaded["norm.weight"],
            "lm_head.weight": loaded["output.weight"],
        }
    else:
        state_dict = {
            "model.norm.weight": loaded[0]["norm.weight"],
            "model.embed_tokens.weight": torch.cat(
                [loaded[i]["tok_embeddings.weight"] for i in range(num_shards)], dim=1
            ),
            "lm_head.weight": torch.cat([loaded[i]["output.weight"] for i in range(num_shards)], dim=0),
        }

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    if os.environ.get('CEPHBUCKET', None) is not None:
        _save(state_dict, os.path.join(tmp_model_path, filename))
    else:
        torch.save(state_dict, os.path.join(tmp_model_path, filename))

    # Write configs
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))

    config = LlamaConfig(
        hidden_size=dim,
        intermediate_size=compute_intermediate_size(dim),
        num_attention_heads=params["n_heads"],
        num_hidden_layers=params["n_layers"],
        rms_norm_eps=params["norm_eps"],
    )
    config.save_pretrained(tmp_model_path)

    # Make space so we can load the model properly now.
    del state_dict
    del loaded
    gc.collect()

    print("Loading the checkpoint in a Llama model.")
    model = LlamaForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    # Avoid saving this as part of the config.
    del model.config._name_or_path

    print("Saving in the Transformers format.")
    model.save_pretrained(model_path, save_function=_save)
    if os.environ.get('CEPHBUCKET', None) is not None:
        for item in os.listdir(model_path):
            local_path = os.path.join(model_path, item)
            ceph_file_path = get_ceph_path(local_path)
            with open(local_path, 'rb') as f:
                PetrelHelper.write(f, ceph_file_path)
    else:
        shutil.rmtree(tmp_model_path)


# def write_tokenizer(tokenizer_path, input_tokenizer_path):
#     # Initialize the tokenizer based on the `spm` modelparams.jsonpa
#     tokenizer_class = LlamaTokenizer if LlamaTokenizerFast is None else LlamaTokenizerFast
#     print(f"Saving a {tokenizer_class.__name__} to {tokenizer_path}.")
#     tokenizer = tokenizer_class(input_tokenizer_path)
#     tokenizer.save_pretrained(tokenizer_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--model_size",
        choices=["7B", "13B", "30B", "65B", "106B", "tokenizer_only"],
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--pp_size",
        default=1,
        type=int,
        help="pp size",
    )
    args = parser.parse_args()
    if args.model_size != "tokenizer_only":
        write_model(
            model_path=args.output_dir,
            input_base_path=os.path.join(args.input_dir),
            model_size=args.model_size,
            pp_size=int(args.pp_size)
        )
    # spm_path = os.path.join(args.input_dir, "tokenizer.model")
    # write_tokenizer(args.output_dir, spm_path)


if __name__ == "__main__":
    main()
