from torch.utils.data import Dataset
import copy
import json
import os
import numpy as np
import torch
import random
from multiprocessing.pool import ThreadPool as Pool

from llm.utils.general.registry_factory import DATASET_REGISTRY
from llm.utils.general.log_helper import default_logger as logger
from llm.utils.tools.petrel_helper import PetrelHelper
from llm.data.nlp_dataset import build_dataset
from llm.data.nlp_transforms import build_transformer


IGNORE_INDEX = -100
DEFAULT_SEED = 1024


def get_token_sum(g):
    sum = 0
    for i in g:
        sum += i[2]
    return sum


def get_vit_num(g):
    vit_num = 0
    for _ in g:
        vit_num += _[1]
    return vit_num


@DATASET_REGISTRY.register('internvl_tools')
class InternvlToolsDataset(Dataset):
    def __init__(self,
                 json_file,
                 tokenizer=None,
                 transformer=None,
                 json_type='all',
                 data_format='wrap',
                 count_length=False):
        super(InternvlToolsDataset, self).__init__()
        self.json_file = json_file
        if not isinstance(json_file, list):
            json_file = [json_file]
        if data_format == 'wrap':
            self.metas = self.load_warp_meta(json_file)
        else:
            if not isinstance(json_type, list):
                if isinstance(json_file, list):
                    json_type = [json_type] * len(json_file)
                else:
                    json_type = [json_type]
            self.metas = self.load_metas(json_file, json_type=json_type)
        if transformer is not None:
            # add datset handler in transform kwargs in need of mosaic/mixup etc.
            for trans in transformer:
                if 'kwargs' in trans and trans['kwargs'].get('with_tokenizer', False):
                    trans['kwargs']['tokenizer'] = tokenizer
                    trans['kwargs'].pop('with_tokenizer')
            self.transformer = build_transformer(transformer)
        else:
            self.transformer = None
        self.count_length = count_length

    def set_seed(self, seed):
        from llm.utils.env import dist_env  # noqa
        # reset seed for different PP stage
        seed = seed + 3 * dist_env.get_data_parallel_rank()
        random.seed(seed)

    def load_warp_meta(self, json_files):
        metas = []
        for wrap_file in json_files:
            data_info = PetrelHelper.load_json(wrap_file)
            for data_name in data_info.keys():
                repeat_time = data_info[data_name]["repeat_time"]
                img_dir = data_info[data_name]["root"]
                data_file = data_info[data_name]["annotation"]
                data_augment = data_info[data_name]["data_augment"]
                for _ in range(repeat_time):
                    with PetrelHelper.open(data_file) as f:
                        for i, line in enumerate(f):
                            meta = json.loads(line)
                            if isinstance(meta, dict):
                                meta["img_dir"] = img_dir
                                meta["data_augment"] = data_augment
                            metas.append(meta)
                            if ((i + 1) % 1000 == 0):
                                logger.info('{} items of data have been loaded'.format(i + 1))
        return metas

    def load_metas(self, json_files, json_type=['all']):
        metas = []
        for idx, json_file in enumerate(json_files):
            if json_type[idx] == 'all':
                temp = PetrelHelper.load_json(json_file)
                metas.extend(temp)
            elif json_type[idx] == 'line':
                with PetrelHelper.open(json_file) as f:
                    for i, line in enumerate(f):
                        meta = json.loads(line)
                        metas.append(meta)
                        if ((i + 1) % 1000 == 0):
                            logger.info('{} items of data have been loaded'.format(i + 1))
        return metas

    def get_meta(self, idx):
        meta = self.metas[idx]
        if self.transformer is not None:
            meta = self.transformer(meta)
        return meta

    def __getitem__(self, idx):
        if self.count_length:
            try:
                meta = self.get_meta(idx)
            except Exception as e:
                print(e, "vit_num and token_num were set to -1.")
                meta = dict(input_ids=-1,
                            num_patches=-1,
                            image_flags=torch.tensor([-1], dtype=torch.long))
        else:
            meta = self.get_meta(idx)
            while meta is None:
                # self.set_seed(idx)
                # new_idx = random.randint(0, len(self.metas) - 1)
                idx = (idx + 100) % len(self.metas)
                meta = self.get_meta(idx)
        return meta

    def __len__(self):
        """
        Returns dataset length
        """
        return len(self.metas)


@DATASET_REGISTRY.register("intern_packed")
class InternPackedDataset(Dataset):
    def __init__(self,
                 dataset={},
                 tokenizer=None,
                 vit_packed_length=15,
                 llm_packed_length=4096,
                 worker=8,
                 cache_dir='./cache',
                 epoch=1,
                 ignore_idx=-100,
                 force_image_size=448,
                 patch_size=14,
                 down_sample_ratio=0.5,
                 iter_time=100,
                 llm_thresh={}):
        self.force_image_size = force_image_size
        self.patch_size = patch_size
        self.down_sample_ratio = down_sample_ratio
        self.dataset = build_dataset(dataset, copy.deepcopy(tokenizer))
        self.dataset_cfg = dataset
        self.tokenizer = tokenizer
        self.vit_packed_length = vit_packed_length
        self.llm_packed_length = llm_packed_length
        self.llm_thresh = llm_thresh

        self.vit_lengths, self.llm_lengths = [], []
        self.worker = worker
        self.pad_token_id = len(self.tokenizer) - 1
        self.ignore_idx = ignore_idx
        self.epoch = epoch
        self.iter_time = iter_time

        os.makedirs(cache_dir, exist_ok=True)
        logger.info("Begin preprocess dataset")
        # self.preprocess()
        self.preprocess_single()
        logger.info("Preprocess dataset successed")
        self.seed = DEFAULT_SEED
        self.pack_groups = self.get_packed_groups()

    def preprocess_single(self):
        dict_num_tokens = {}
        meta_info = json.loads(open(self.dataset.json_file).read())
        for idx, data_name in enumerate(meta_info.keys()):
            with open(meta_info[data_name]["token_lengths"], "r") as f:
                token_lengths = json.load(f)
            dict_num_tokens[idx] = {
                "lengths": len(self.dataset),
                "token_lengths": token_lengths  # sub_dataset.meta["token_lengths"]
            }
        self.dict_num_tokens = dict_num_tokens

    def preprocess(self):
        assert self.dataset_cfg["type"] == "internvl", "sub dataset type is not internvl."

        dict_num_tokens = {}
        num_datasets = len(self.dataset.datasets)
        for idx in range(num_datasets):
            sub_dataset = self.dataset.datasets[idx]
            if "token_lengths" in sub_dataset.meta:
                logger.info(f"Load from cache for dataset {idx}")
                assert os.path.exists(sub_dataset.meta["token_lengths"]), f"Dataset {idx} token_lengths file does not exist."
                with open(sub_dataset.meta["token_lengths"], "r") as f:
                    token_lengths = json.load(f)
                dict_num_tokens[idx] = {
                    "lengths": len(sub_dataset),
                    "token_lengths": token_lengths  # sub_dataset.meta["token_lengths"]
                }
            else:
                logger.info(f"Generate length json for dataset {idx}")
                token_lengths = []
                origin_indexs = list(range(len(sub_dataset)))
                token_lengths_dict = dict()

                def decode_text(idx):
                    meta = sub_dataset.__getitem__(idx)
                    token_lengths_dict[idx] = {
                        "vit_num": meta['pixel_values'].shape[0],
                        "token_num": len(meta['input_ids']),
                        "image_flags": meta["image_flags"].sum().item()
                    }

                with Pool(self.worker) as p:
                    _ = p.map(decode_text, origin_indexs[:])
                for idx in range(len(sub_dataset)):
                    token_lengths.append(
                        token_lengths_dict[idx]
                    )
                dict_num_tokens[idx] = {
                    "lengths": len(sub_dataset),
                    "token_lengths": token_lengths
                }
        self.dict_num_tokens = dict_num_tokens

    def _random_groups(self, token_lengths, seed=None):
        """
        tokens_length: [(idx, vit_img_num, llm_token_len)]
        """
        rng = np.random.RandomState(seed)
        index = list(range(len(token_lengths)))
        rng.shuffle(index)

        pack_groups = []
        vit_token_length_sum, llm_token_length_sum = 0, 0
        each_group = []
        for idx, sample_id in enumerate(index):
            vit_sample_length, llm_sample_length = token_lengths[sample_id][1], token_lengths[sample_id][2]
            if vit_sample_length > self.vit_packed_length or llm_sample_length > self.llm_packed_length:
                continue
            vit_token_length_sum += vit_sample_length
            llm_token_length_sum += llm_sample_length
            if vit_token_length_sum > self.vit_packed_length or llm_token_length_sum > self.llm_packed_length:
                pack_groups.append(each_group)
                vit_token_length_sum = vit_sample_length
                llm_token_length_sum = llm_sample_length
                each_group = [token_lengths[sample_id]]
            else:
                each_group.append(token_lengths[sample_id])
            if idx == len(token_lengths) - 1:
                if len(each_group) > 0:
                    pack_groups.append(each_group)
        return pack_groups

    def process_random_groups_input(self, groups, accu_length=0):
        new_groups = []
        for idx, item in enumerate(groups):
            if item["vit_num"] == -1:
                logger.info(f"item {idx} was filted.")
                continue
            new_groups.append((idx + accu_length, item['vit_num'], item['token_num']))
        return new_groups

    def iter_random_groups(self, groups, llm_thresh=None, seed=None, iter_time=300):
        if llm_thresh is None:
            llm_thresh = self.llm_packed_length
        if seed is None:
            seed = self.seed
        groups = self._random_groups(groups, seed=seed)
        if iter_time == 1:
            return groups
        output = []
        for i in range(iter_time - 1):
            logger.info(f"iter_random_groups {i} / {iter_time - 1}")
            need_process_groups = []
            for g in groups:
                vit_num = get_vit_num(g)
                llm_num = get_token_sum(g)
                if vit_num == self.vit_packed_length or llm_num >= llm_thresh:
                    output.append(g)
                else:
                    need_process_groups.extend(g)
            if len(need_process_groups) >= 0:
                groups = self._random_groups(need_process_groups, seed + i)
            else:
                break
        if len(need_process_groups) > 0:
            output.extend(self._random_groups(need_process_groups, seed + i))
        return output

    def collect_packed_info(self, packed_groups):
        info_dict = {}
        info_dict['vit_num_info'] = {}
        vit_num_min = 10000000
        vit_num_max = 0
        llm_num_min = 10000000
        llm_num_max = 0
        vit_ave_num = 0
        llm_ave_num = 0
        sample_num = 0
        for group in packed_groups:
            vit_num = get_vit_num(group)
            llm_num = get_token_sum(group)
            if vit_num not in info_dict['vit_num_info']:
                info_dict['vit_num_info'][vit_num] = 0
            info_dict['vit_num_info'][vit_num] += 1
            vit_num_min = min(vit_num_min, vit_num)
            vit_num_max = max(vit_num_max, vit_num)
            llm_num_min = min(llm_num_min, llm_num)
            llm_num_max = max(llm_num_max, llm_num)
            vit_ave_num += vit_num
            llm_ave_num += llm_num
            sample_num += len(group)
        info_dict['vit_num_min'] = vit_num_min
        info_dict['vit_num_max'] = vit_num_max
        info_dict['vit_ave_num'] = vit_ave_num / float(len(packed_groups))
        info_dict['llm_ave_num'] = llm_ave_num / float(len(packed_groups))
        info_dict['sample_num'] = sample_num
        info_dict['packed_group_num'] = len(packed_groups)
        return info_dict

    def find_best_groups(self, input_groups, step=4, step_num=20):
        best_group_num = 10000000000000
        best_groups = []
        best_info_dict = {}
        best_llm_thresh = 0
        llm_thresh = self.llm_packed_length
        for step_id in range(step_num):
            logger.info(f"find_best_groups {step_id} / {step_num}")
            groups = self.iter_random_groups(input_groups, llm_thresh, seed=self.seed, iter_time=self.iter_time)
            cur_info_dict = self.collect_packed_info(groups)
            if cur_info_dict['packed_group_num'] < best_group_num:
                best_group_num = cur_info_dict['packed_group_num']
                best_groups = groups
                best_info_dict = cur_info_dict
                best_llm_thresh = llm_thresh
            llm_thresh -= step
        print(f"llm thresh {best_llm_thresh} best info dict", best_info_dict)
        return best_groups

    def get_packed_groups(self):
        # num_datasets = len(self.dataset.datasets)
        num_datasets = len(list(self.dict_num_tokens.keys()))
        accu_length = 0
        input_groups = []
        for d_idx in range(num_datasets):
            dict_item = self.dict_num_tokens[d_idx]
            token_lengths = dict_item["token_lengths"]
            groups = self.process_random_groups_input(token_lengths, accu_length)
            logger.info(f"get_packed_groups {d_idx}.")
            input_groups.extend(groups)
            accu_length += len(token_lengths)
        if self.llm_thresh.get('thresh', None) is not None:
            groups = self.iter_random_groups(input_groups, llm_thresh=self.llm_thresh['thresh'], seed=self.seed, iter_time=self.iter_time)
        else:
            groups = self.find_best_groups(input_groups, self.llm_thresh.get('step', 4), self.llm_thresh.get('step_num', 10))
        print(self.collect_packed_info(groups), flush=True)
        logger.info("get_packed_groups done!")
        return groups

    def __getitem__(self, item: int):
        item = item % len(self.pack_groups)
        # item = random.randint(0, len(self.pack_groups) - 1)
        while True:
            try:
                groups = self.pack_groups[item]

                input_ids, pixel_values = [], []
                labels, position_ids, image_flags = [], [], []
                cu_seqlens = [0]
                for g in groups:
                    idx, num_patches, llm_length = g
                    meta = self.dataset.__getitem__(idx)
                    assert len(meta["input_ids"]) == llm_length
                    assert meta["pixel_values"].size(0) == num_patches
                    input_ids.append(meta['input_ids'])
                    pixel_values.append(meta['pixel_values'])
                    labels.append(meta['labels'])
                    cu_seqlens.append(len(meta['input_ids']))
                    position_ids.extend(list(range(len(meta['input_ids']))))
                    image_flags.append(meta.get('image_flags', torch.tensor([0], dtype=torch.long)))

                cu_seqlens = np.cumsum(np.array(cu_seqlens)).tolist()
                input_ids = torch.cat(input_ids)[:self.llm_packed_length]
                pixel_values = torch.cat(pixel_values)[:self.vit_packed_length]
                labels = torch.cat(labels)[:self.llm_packed_length]
                cu_seqlens = torch.clamp(torch.LongTensor(cu_seqlens), max=self.llm_packed_length)
                position_ids = torch.LongTensor(position_ids)[:self.llm_packed_length]
                image_flags = torch.cat(image_flags)
                if len(image_flags) == 0:  # pure llm text
                    image_flags = torch.tensor([0], dtype=torch.long)

                ret = {
                    "input_ids": input_ids,
                    "labels": labels,
                    "cu_seqlens": cu_seqlens,
                    "position_ids": position_ids,
                    "pixel_values": pixel_values,
                    "image_flags": image_flags
                }
                break
            except Exception as e:
                logger.info(f"{e}")
                # i = random.randint(0, len(self.raw_data) - 1)
                item = (item + 100) % len(self.pack_groups)
        return ret

    def __len__(self):
        n_packs = len(self.pack_groups)
        return n_packs
