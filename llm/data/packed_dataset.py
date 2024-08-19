from torch.utils.data import Dataset
import copy
import os
import numpy as np
import torch
import json
from llm.utils.general.registry_factory import DATASET_REGISTRY
from multiprocessing.pool import ThreadPool as Pool
from llm.data.nlp_dataset import build_dataset
from llm.utils.general.log_helper import default_logger as logger
from llm.utils.env import dist_env


IGNORE_INDEX = -100
DEFAULT_SEED = 1024


@DATASET_REGISTRY.register("packed")
class PackedDataset(Dataset):
    def __init__(self,
                 dataset={},
                 tokenizer=None,
                 length_path=None,
                 packed_length=4096,
                 worker=8,
                 cache_dir='./cache',
                 ignore_idx=-100,
                 packed_length_thresh=4050,
                 iter_time=1,
                 display_bin_size=128):
        self.dataset = build_dataset(dataset, copy.deepcopy(tokenizer))
        self.dataset_cfg = dataset
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        if length_path is None:
            self.length_path = os.path.join(cache_dir, "lengths.npy")
        else:
            self.length_path = length_path
        self.packed_length = packed_length
        self.lengths = []
        self.worker = worker
        self.pad_token_id = len(self.tokenizer) - 1
        self.ignore_idx = ignore_idx
        os.makedirs(cache_dir, exist_ok=True)
        logger.info("Begin preprocess dataset")
        self.preprocess()
        logger.info("Preprocess dataset successed")
        self.seed = DEFAULT_SEED
        self.pack_group = self.process_random_groups(self.tokens_lengths, packed_length, packed_length_thresh, iter_time)  # noqa
        self.num_tokens = sum(np.array(self.lengths)[:, 1])
        if dist_env.get_data_parallel_rank() == 0 and dist_env.get_pipeline_model_parallel_rank() == 0:
            self.display_groups_info(display_bin_size)

    def display_groups_info(self, display_bin_size):
        info = {}
        info['length_info'] = {}
        ave_length = 0
        min_length = 100000000.
        max_length = 0
        for g in self.pack_group:
            llm_num = float(self.get_token_sum(g))
            llm_num_bin = (llm_num // display_bin_size) * display_bin_size
            if llm_num_bin not in info['length_info']:
                info['length_info'][llm_num_bin] = 0
            info['length_info'][llm_num_bin] += 1
            min_length = min(llm_num, min_length)
            max_length = max(max_length, llm_num)
            ave_length += llm_num
        info['min_length'] = min_length
        info['max_length'] = max_length
        info['ave_length'] = float(ave_length / (len(self.pack_group)))
        info['group_num'] = len(self.pack_group)
        info['sample_num'] = len(self.lengths)
        print(json.dumps(info, indent=4, sort_keys=True))

    def random_group(self, token_lengths, seed=None, llm_packed_length=4096):
        rng = np.random.RandomState(seed)
        index = list(range(len(token_lengths)))
        rng.shuffle(index)

        pack_group = []
        llm_token_length_sum = 0
        each_group = []
        for idx, sample_id in enumerate(index):
            llm_sample_length = token_lengths[sample_id][1]
            if llm_sample_length > llm_packed_length:
                continue
            llm_token_length_sum += llm_sample_length
            if llm_token_length_sum > llm_packed_length:
                pack_group.append(each_group)
                llm_token_length_sum = llm_sample_length
                each_group = [token_lengths[sample_id]]
            else:
                each_group.append(token_lengths[sample_id])
            if idx == len(token_lengths) - 1:
                if len(each_group) > 0:
                    pack_group.append(each_group)
        return pack_group

    def get_token_sum(self, g):
        sum = 0
        for i in g:
            sum += i[1]
        return sum

    def process_random_groups(self, token_lengths, llm_max, llm_thresh=4050, iter_time=10):
        groups = self.random_group(token_lengths, self.seed, llm_max)
        if iter_time == 1:
            return groups
        output = []
        need_process_groups = []
        for i in range(iter_time - 1):
            need_process_groups = []
            for g in groups:
                llm_num = self.get_token_sum(g)
                if llm_num >= llm_thresh:
                    output.append(g)
                else:
                    need_process_groups.extend(g)
            if len(need_process_groups) >= 0:
                groups = self.random_group(need_process_groups, self.seed + i, llm_max)
            else:
                break
            if dist_env.get_data_parallel_rank() == 0 and dist_env.get_pipeline_model_parallel_rank() == 0:
                print(i + 1, len(output), len(need_process_groups))
        if len(need_process_groups) > 0:
            output.extend(self.random_group(need_process_groups, self.seed + i, llm_max))
        return output

    def preprocess(self):

        if os.path.exists(self.length_path):
            logger.info("Load from cache")
            self.lengths = np.load(self.length_path)
        else:
            logger.info("Generate length.npy")
            origin_indexs = list(range(len(self.dataset)))
            lengths_dict = dict()

            def decode_text(idx):
                meta, out_idx = self.dataset.get_meta(idx)
                lengths_dict[idx] = (out_idx, len(meta['input_ids']))
            with Pool(self.worker) as p:
                _ = p.map(decode_text, origin_indexs[:])
            for idx in range(len(self.dataset)):
                self.lengths.append([idx, lengths_dict[idx]])
            from llm.utils.env import dist_env
            if dist_env.get_data_parallel_rank() == 0 and dist_env.get_tensor_model_parallel_rank() == 0 and dist_env.get_pipeline_model_parallel_rank() == 0:  # noqa
                np.save(self.length_path, self.lengths)
        self.tokens_lengths = self.lengths

    def __getitem__(self, item: int):
        group = self.pack_group[item]

        input_ids = []
        labels = []
        cu_seqlens = [0]
        position_ids = []
        for g in group:
            index, length = g
            meta = self.dataset.__getitem__(index)
            new_length = len(meta["input_ids"])
            if new_length != length:
                print(index, f"current length {new_length} != cache {length}")
                continue

            input_ids.append(meta['input_ids'])
            labels.append(meta['labels'])
            cu_seqlens.append(len(meta['input_ids']))
            position_ids.extend(list(range(len(meta['input_ids']))))

        cu_seqlens = np.cumsum(np.array(cu_seqlens)).tolist()
        input_ids = torch.cat(input_ids)[:self.packed_length]
        labels = torch.cat(labels)[:self.packed_length]
        cu_seqlens = torch.clamp(torch.LongTensor(cu_seqlens), max=self.packed_length)
        position_ids = torch.LongTensor(position_ids)[:self.packed_length]

        return {"input_ids": input_ids, "labels": labels, "cu_seqlens": cu_seqlens, "position_ids": position_ids}

    def __len__(self):
        # Line 405 of document_to_sequence.py in metaseq is directly spliced,
        # without additional consideration of sos or eos
        n_packs = len(self.pack_group)
        return n_packs


@DATASET_REGISTRY.register("packed_dpo")
class PackedDPODataset(PackedDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, item: int):
        data = super().__getitem__(item)
        group = self.pack_group[item]
        cu_seqlens = [0]
        position_ids = []
        scores = []
        for g in group:
            index = g[0]
            meta = self.dataset.__getitem__(index)
            cu_seqlens.extend(torch.diff(meta['cu_seqlens']).tolist())  # reverse cu_seqlens to seqlens
            position_ids.append(meta['position_ids'])
            scores.append(meta['scores'])
        cu_seqlens = np.cumsum(np.array(cu_seqlens)).tolist()
        cu_seqlens = torch.clamp(torch.LongTensor(cu_seqlens), max=self.packed_length)
        position_ids = torch.LongTensor(torch.cat(position_ids, dim=0))[:self.packed_length]
        scores = torch.cat(scores)
        data.update({"cu_seqlens": cu_seqlens, "position_ids": position_ids, "scores": scores})
        return data
