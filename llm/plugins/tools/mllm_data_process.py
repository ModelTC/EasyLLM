import copy
import json
from llm.utils.general.yaml_loader import load_yaml
from llm.plugins.internvl.data.data_utils import (
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
from llm.data import build_tokenizer, build_dataset
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
        flag = item['image_flags'].sum().item()
        if flag == 0:
            num_vit_patch = item["pixel_values"].size(0)
            num_token = len(item["input_ids"])
            image_flags = 0
        elif flag == -1:
            num_vit_patch = -1
            num_token = -1
            image_flags = -1
        else:
            num_vit_patch = flag
            num_token = len(item["input_ids"])
            image_flags = flag

        token_lengths.append(
            {
                "vit_num": num_vit_patch,
                "token_num": num_token,
                "image_flags": image_flags
            }
        )

    return token_lengths


def worker(cfg_dataset, tokenizer, ds_name, token_lengths_path, ds_info):
    dataset = build_dataset(cfg_dataset, tokenizer)
    with multiprocessing.Pool(PROCESSES) as pool:
        token_lengths_all = pool.map(decode_text, [(cfg_dataset, tokenizer, inds) for inds in np.array_split(range(len(dataset)), PROCESSES)])
    l_token_lengths = []
    for tmp in token_lengths_all:
        l_token_lengths.extend(tmp)

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
        "--json_file",
        default=None,
        help="json file to statistics"
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

    cfg["tokenization"]["kwargs"]["parser_kwargs"]["read_img"] = False

    if cfg["data"]["train"]["dataset"]["type"] == "intern_packed":
        cfg_dataset_base = cfg["data"]["train"]["dataset"]["kwargs"]["dataset"]
    else:
        cfg_dataset_base = cfg["data"]["train"]["dataset"]

    # dataset = build_dataset(cfg_dataset_base, tokenizer)
    if args.json_file is None:
        ds_collections = json.loads(open(cfg_dataset_base["kwargs"]["json_file"]).read())
    else:
        ds_collections = json.loads(open(args.json_file).read())

    cfg_dataset_base["kwargs"]["data_format"] = "normal"
    cfg_dataset_base["kwargs"]["count_length"] = True
    import time
    t_1 = time.time()
    meta = {}
    for ds_name in tqdm(ds_collections.keys()):
        print(ds_name)
        cfg_dataset = copy.deepcopy(cfg_dataset_base)
        ds_info = {}
        cfg_dataset["kwargs"]["json_file"] = ds_collections[ds_name]["annotation"]
        ds_info["root"] = ds_collections[ds_name]["root"]
        ds_info["annotation"] = ds_collections[ds_name]["annotation"]
        ds_info["data_augment"] = ds_collections[ds_name].get("data_augment", False)
        ds_info["repeat_time"] = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            ds_info['max_dynamic_patch'] = ds_collections[ds_name]['max_dynamic_patch']

        meta[ds_name] = worker(cfg_dataset, tokenizer, ds_name, token_lengths_path, ds_info)

    with open(args.output_path, "w") as f:
        json.dump(meta.copy(), f, indent=4)

    t_2 = time.time()
    print(f"time: {t_2-t_1}")
