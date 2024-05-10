import copy
import json
from llm.utils.general.yaml_loader import load_yaml
from llm.plugins.internvl.datas.data_utils import (
    IMG_CONTEXT_TOKEN,
    IMG_START_TOKEN,
    IMG_END_TOKEN,
    BOX_START_TOKEN,
    BOX_END_TOKEN,
    REF_START_TOKEN,
    REF_END_TOKEN,
    QUAD_START_TOKEN,
    QUAD_END_TOKEN
)
from llm.data import build_tokenizer
from llm.data.nlp_dataset import build_dataset
import multiprocessing
import argparse
from tqdm import tqdm
import os
import numpy as np
PROCESSES = 64


def decode_text(args):
    cfg_dataset, tokenizer, inds = args
    dataset = build_dataset(cfg_dataset, tokenizer)
    dataset.ds_name = "dummy"
    token_lengths = []
    for idx in inds:
        item = dataset.__getitem__(idx)
        if item['num_patches'] is not None:
            num_vit_patch = item['num_patches']
            num_token = len(item["input_ids"])
            image_flags = item['image_flags'].sum().item()
        else:
            num_vit_patch = -1
            num_token = -1
            image_flags = -1
        token_lengths.append(
            {
                "vit_num": num_vit_patch,
                "token_num": num_token,
                "image_flags": image_flags
            }
        )
    return token_lengths


def worker(cfg_dataset, tokenizer, ds_name, token_lengths_path):
    dataset = build_dataset(cfg_dataset, tokenizer)
    # if len(dataset) < 20*1000:
    #     PROCESSES = 16
    # elif len(dataset) < 100*1000:
    #     PROCESSES = 32
    # else:
    #     PROCESSES = 64
    with multiprocessing.Pool(PROCESSES) as pool:
        token_lengths_all = pool.map(decode_text, [(cfg_dataset, tokenizer, inds) for inds in np.array_split(range(len(dataset)), PROCESSES)])
    l_token_lengths = []
    for tmp in token_lengths_all:
        l_token_lengths.extend(tmp)
    ds_info = dataset.meta

    length_save_path = os.path.join(token_lengths_path, f"{ds_name}" + "_token_lengths.json")

    with open(length_save_path, "w") as f:
        json.dump(l_token_lengths, f, indent=4)
    if "max_dynamic_patch" in ds_info:
        info = {
            "root": ds_info["root"],
            "annotation": ds_info["annotation"],
            "data_augment": ds_info["data_augment"],
            "repeat_time": ds_info["repeat_time"],
            "length": len(dataset),
            "token_lengths": length_save_path,
            "max_dynamic_patch": ds_info["max_dynamic_patch"]
        }
    else:
        info = {
            "root": ds_info["root"],
            "annotation": ds_info["annotation"],
            "data_augment": ds_info["data_augment"],
            "repeat_time": ds_info["repeat_time"],
            "length": len(dataset),
            "token_lengths": length_save_path
        }
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="data root path",
    )
    parser.add_argument(
        "--worker",
        default=64, type=int,
        help="worker num",
    )
    parser.add_argument(
        "--token_lengths_path",
        default=None,
        help="token_lengths_path",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="token_lengths_path",
    )
    args = parser.parse_args()

    cfg_path = args.config
    token_lengths_path = args.token_lengths_path

    cfg = load_yaml(cfg_path)
    tokenizer = build_tokenizer(cfg['tokenizer'])
    tokenizer.tokenizer_path = cfg["tokenizer"]["kwargs"]["tokenizer_name_or_path"]
    tokenizer.model_max_length = cfg["tokenization"]["kwargs"].get("max_seq_length", 4096)
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)

    cfg_dataset_base = cfg["data"]["train"]["dataset"]
    num_image_token = int(
        (cfg["runtime"]["force_image_size"] // cfg["runtime"]["patch_size"]) ** 2 * (cfg["runtime"]["down_sample_ratio"] ** 2)
    )
    cfg_dataset_base["kwargs"]["num_image_token"] = num_image_token
    ds_collections = json.loads(open(cfg_dataset_base["meta_path"]).read())

    meta = {}
    for ds_name in tqdm(ds_collections.keys()):
        print(ds_name)
        repeat_time = ds_collections[ds_name]['repeat_time']
        cfg_dataset = copy.deepcopy(cfg_dataset_base)
        cfg_dataset["kwargs"]["meta"] = ds_collections[ds_name]
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            cfg_dataset['kwargs']['max_dynamic_patch'] = ds_collections[ds_name]['max_dynamic_patch']
        if 'repeat_time' in ds_collections[ds_name]:
            cfg_dataset['kwargs']['repeat_time'] = ds_collections[ds_name]['repeat_time']
        if 'data_augment' in ds_collections[ds_name]:
            cfg_dataset['kwargs']['is_train'] = ds_collections[ds_name]['data_augment']

        meta[ds_name] = worker(cfg_dataset, tokenizer, ds_name, token_lengths_path)

    with open(args.output_path, "w") as f:
        json.dump(meta.copy(), f, indent=4)
