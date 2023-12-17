import json
from transformers import LlamaTokenizerFast
import numpy as np
import os
from multiprocessing import Pool
import argparse


def find_newline(text, end):
    while end >= 1:
        if text[end - 1] == '\n':
            end -= 1
        else:
            break
    return end


split_set = [' ', '。', '.', '，', '；', ';']


def find_space(text, end):
    while end < len(text):
        if text[end] not in split_set:
            end += 1
        else:
            break
    return end + 1


def find_split_location(text, split_len=250000):
    text_len = len(text)
    num = (text_len // split_len + 1)
    split_text = []
    start = 0
    if num == 1:
        end = text_len
        split_text.append(text)
        return split_text
    else:
        end = split_len
    while start < text_len:
        cur_index = text[start:end].rfind('\n')
        if cur_index == -1 or cur_index == 0:
            end = find_space(text, end)
            split_text.append(text[start:end])
            start = end
            end = start + split_len
        else:
            cur_index = find_newline(text, cur_index + start)
            if cur_index == start:
                cur_index = find_space(text, cur_index)
            split_text.append(text[start:cur_index])
            start = cur_index
            end = start + split_len
        if text_len - start <= split_len:
            split_text.append(text[start:])
            break

    return split_text


def get_split_tokens(merge_tokens, bin_size=256 * 1024, name='', out_folder=''):
    data_root = out_folder
    folder = get_save_folder(name)
    full_folder = os.path.join(data_root, folder)
    os.makedirs(full_folder, exist_ok=True)
    fw = open(f"{full_folder}.txt", "w")
    num = len(merge_tokens) // bin_size + 1
    for i in range(num):
        temp = merge_tokens[i * bin_size: (i + 1) * bin_size]
        temp = np.array(temp).astype(np.int32)
        size_temp = len(temp)
        if size_temp > 0:
            np.save(os.path.join(full_folder, f"{size_temp}_{i}"), temp)
            print(os.path.join(folder, f"{size_temp}_{i}.npy"), file=fw)
    fw.close()


def get_save_folder(name):
    items = name.replace('.jsonl', '').split('/')
    full_name = ''
    for idx, item in enumerate(items):
        if idx < len(items) - 1:
            full_name += f"{item}_"
        else:
            full_name += item
    return full_name


def process(item):
    root, name, tokenizer_path, split_len, bin_size, out_folder, space_id = item[0], item[1], item[2], item[3], item[4], item[5], item[6] # noqa
    tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_path)
    path = os.path.join(root, name)
    with open(path) as f:
        merge_tokens = []
        for _, line in enumerate(f.readlines()[:]):
            res = json.loads(line)
            if 'content' in res:
                content = res['content']
            if 'question' in res:
                content = res['question']
            split_texts = find_split_location(content, split_len)
            for idx, text in enumerate(split_texts):
                if idx == 0:
                    tokens = tokenizer(text, return_attention_mask=False, add_special_tokens=True)['input_ids']
                else:
                    tokens = tokenizer(text, return_attention_mask=False, add_special_tokens=False)['input_ids']
                if len(tokens) > 0:
                    if tokens[0] == space_id:
                        tokens = tokens[1:]
                    if text[-1] != ' ' and tokens[-1] == space_id:
                        tokens = tokens[:-1]
                merge_tokens.extend(tokens)
            merge_tokens.append(tokenizer.eos_token_id)
        get_split_tokens(merge_tokens, bin_size=bin_size, name=name, out_folder=out_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        help="data root path",
    )
    parser.add_argument(
        "--out_folder",
        help="data root path",
    )
    parser.add_argument(
        "--path_list",
        default='', type=str,
        help="path list",
    )
    parser.add_argument(
        "--tokenizer_path",
        default='', type=str,
        help="tokenizer_path",
    )
    parser.add_argument(
        "--worker",
        default=32, type=int,
        help="worker num",
    )
    parser.add_argument(
        "--split_len",
        default=10000, type=int,
        help="split length for faster encode",
    )
    parser.add_argument(
        "--group",
        default=4, type=int,
        help="group num",
    )
    parser.add_argument(
        "--group_id",
        default=0, type=int,
        help="group id",
    )
    parser.add_argument(
        "--bin_size",
        default=262144, type=int,
        help="bin size default 256K",
    )
    parser.add_argument(
        "--space_id",
        default=65616, type=int,
        help="space id on your tokenizer",
    )
    args = parser.parse_args()
    worker = int(args.worker)
    root = args.data_root
    path_list = args.path_list
    tokenizer_path = args.tokenizer_path
    split_len = int(args.split_len)
    out_folder = args.out_folder
    group = int(args.group)
    group_id = int(args.group_id)
    bin_size = int(args.bin_size)
    space_id = int(args.space_id)
    paths = open(path_list).readlines()[:]
    paths = [(root, item.strip(), tokenizer_path, split_len, bin_size, out_folder, space_id) for item in paths]
    if group > 1:
        assert group >= 1
        group_size, mod = divmod(len(paths), group)
        if mod > 0:
            group_size += 1
        assert group_id <= group - 1
        cur_paths = paths[group_id * group_size:(group_id + 1) * group_size]
        print(f"group: {group} current group_size: {len(cur_paths)} current group_id: {group_id}")
    else:
        cur_paths = paths

    with Pool(worker) as p:
        p.map(process, cur_paths)
