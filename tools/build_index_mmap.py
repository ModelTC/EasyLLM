import os
import argparse
import json
from functools import partial
from multiprocessing.pool import ThreadPool as Pool

import torch

from llm.utils.general.yaml_loader import load_yaml
from llm.data import build_tokenizer
from llm.data.nlp_transforms import build_transformer
from llm.data.data_utils import MMapIndexedDatasetBuilder, best_fitting_dtype


def cache_log(log_str, cache_log_freq):
    if (cache_log_freq != -1):
        print(log_str)


def data_encode_raw(json_line, transformer):
    meta = json.loads(json_line)
    meta = transformer(meta)
    return meta


def build_data_cache(data_encode, tokenizer, json_file, cache_prefix, cache_log_freq=1000, cache_worker=4):
    total_build_items = 0
    builders = MMapIndexedDatasetBuilder(cache_prefix + '.bin',
                                         dtype=best_fitting_dtype(tokenizer.vocab_size))
    for jsf in json_file:
        cache_log('File {}: start processing'.format(jsf), cache_log_freq)
        fin = open(jsf, 'r', encoding='utf-8')
        with Pool(cache_worker) as p:
            encoded_docs = p.imap(data_encode, fin, 25)
            ith_build_items = 0
            for meta in encoded_docs:
                input_ids = meta['input_ids'].int().cpu().numpy().tolist()
                if len(input_ids) == 0:
                    continue
                builders.add_item(torch.IntTensor(input_ids))
                builders.end_document()
                total_build_items += 1
                ith_build_items += 1
                if (ith_build_items % cache_log_freq == 0):
                    cache_log('File {}: {} items have been processed...'.format(jsf, ith_build_items),
                              cache_log_freq)
            cache_log('File {}: process done, {} items have been processed.'.format(jsf, ith_build_items),
                      cache_log_freq)
    builders.finalize(cache_prefix + '.idx')
    cache_log('All Files Done! Total {} items have been processed.'.format(total_build_items),
              cache_log_freq)


def main(args):
    cfg = load_yaml(args.cfg)
    cfg_tokenizer = cfg['tokenizer']
    tokenizer = build_tokenizer(cfg_tokenizer)
    cfg_train_dataset = cfg['data']['train']['dataset']['kwargs']
    cfg_transform = cfg_train_dataset['transformer']
    for trans in cfg_transform:
        if 'kwargs' in trans and trans['kwargs'].get('with_tokenizer', False):
            trans['kwargs']['tokenizer'] = tokenizer
            trans['kwargs'].pop('with_tokenizer')
    transform = build_transformer(cfg_transform)

    data_root = cfg_train_dataset['data_root']
    cache_dir = cfg_train_dataset['cache_dir']
    cache_prefix = cfg_train_dataset['cache_prefix']
    cache_prefix = os.path.join(data_root, cache_dir, cache_prefix)
    cache_worker = cfg_train_dataset['cache_worker']
    json_file = cfg_train_dataset['json_file']

    os.makedirs(os.path.join(data_root, cache_dir), exist_ok=True)
    if not isinstance(json_file, list):
        json_file = [json_file]
    data_encode_func = partial(data_encode_raw, transformer=transform)
    build_data_cache(data_encode_func, tokenizer, json_file, cache_prefix, cache_worker=cache_worker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        help="your easyllm yaml configs",
    )
    args = parser.parse_args()
    main(args)
