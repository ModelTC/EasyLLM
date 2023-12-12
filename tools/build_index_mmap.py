import os
import argparse
import json
from functools import partial

from llm.utils.general.yaml_loader import load_yaml
from llm.data import build_tokenizer
from llm.data.nlp_transforms import build_transformer
from llm.data.data_utils import build_data_cache


def data_encode_raw(json_line, transformer):
    meta = json.loads(json_line)
    meta = transformer(meta)
    return meta


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
    cache_splits = cfg_train_dataset.get('cache_splits', 1)
    cache_indices = args.cache_indices
    if cache_indices is None:
        cache_indices = cfg_train_dataset.get('cache_indices', None)
    json_file = cfg_train_dataset['json_file']

    os.makedirs(os.path.join(data_root, cache_dir), exist_ok=True)
    if not isinstance(json_file, list):
        json_file = [json_file]
    data_encode_func = partial(data_encode_raw, transformer=transform)
    print('Starting to process files', flush=True)
    build_data_cache(data_encode_func, tokenizer, json_file, cache_prefix, cache_worker=cache_worker,
                     cache_splits=cache_splits, cache_indices=cache_indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        help="your easyllm yaml configs",
    )
    parser.add_argument('--cache-indices', type=int, default=None, nargs='+', help='indices of the cache splits for manually building them.')
    args = parser.parse_args()
    main(args)
