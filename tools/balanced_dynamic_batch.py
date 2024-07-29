from torch.utils.data import DataLoader
import torch
import math
import json
from torch.utils.data import Dataset
import numpy as np

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


class MegatronPretrainingRandomSampler:
    def __init__(self,
                 total_samples,
                 consumed_samples,
                 micro_batch_size,
                 data_parallel_rank,
                 data_parallel_size):
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.last_batch_size = \
            self.total_samples % self.micro_batch_times_data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        # data sharding and random sampling
        bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) * self.micro_batch_size
        bucket_offset = current_epoch_samples // self.data_parallel_size
        start_idx = self.data_parallel_rank * bucket_size

        g = torch.Generator()
        g.manual_seed(self.epoch)
        random_idx = torch.randperm(bucket_size, generator=g).tolist()
        idx_range = [start_idx + x for x in random_idx[bucket_offset:]]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []


class MegatronLengthGroupSampler:
    def __init__(self,
                 total_samples,
                 consumed_samples,
                 micro_batch_size,
                 data_parallel_rank,
                 data_parallel_size,
                 lengths=None,
                 seed=233):

        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.last_batch_size = \
            self.total_samples % self.micro_batch_times_data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

        self.lengths = lengths
        self.num_samples = math.ceil(len(self.lengths) / (data_parallel_size * self.micro_batch_size))
        self.total_size = self.num_samples * data_parallel_size * self.micro_batch_size
        self.seed = seed
        self.buffer = None

    # copy from https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L38
    def split_to_even_chunks(self, indices, lengths, num_chunks):
        """
        Split a list of indices into `chunks` chunks of roughly equal lengths.
        """
        if len(indices) % num_chunks != 0:
            return [indices[i::num_chunks] for i in range(num_chunks)]

        num_indices_per_chunk = len(indices) // num_chunks

        chunks = [[] for _ in range(num_chunks)]
        chunks_lengths = [0 for _ in range(num_chunks)]
        for index in indices:
            shortest_chunk = chunks_lengths.index(min(chunks_lengths))
            chunks[shortest_chunk].append(index)
            chunks_lengths[shortest_chunk] += lengths[index]
            if len(chunks[shortest_chunk]) == num_indices_per_chunk:
                chunks_lengths[shortest_chunk] = float('inf')

        return chunks

    # copy from https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py#L88
    def get_length_grouped_indices(self, lengths, batch_size, world_size, generator=None, merge=True):
        # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
        indices = torch.randperm(len(lengths), generator=generator)
        megabatch_size = world_size * batch_size
        megabatches = [indices[i: i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
        megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
        megabatches = [self.split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

        return [i for megabatch in megabatches for batch in megabatch for i in batch]
    

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        if self.buffer is None:
            g = torch.Generator()
            g.manual_seed(self.seed)
            indices = self.get_length_grouped_indices(self.lengths, self.micro_batch_size, self.data_parallel_size, generator=g)
            self.buffer = indices
        else:
            indices = self.buffer
        active_indices = indices + indices[: (self.total_size - len(indices))]
        active_indices_split = []
        start = self.data_parallel_rank * self.micro_batch_size
        stop = (self.data_parallel_rank + 1) * self.micro_batch_size
        while start < len(active_indices):
            if stop <= len(active_indices):
                active_indices_split += active_indices[start:stop]
            else:
                active_indices_split += active_indices[start:]
            start += self.data_parallel_size * self.micro_batch_size
            stop += self.data_parallel_size * self.micro_batch_size

        current_epoch_samples = self.consumed_samples % len(active_indices)
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        # data sharding and random sampling
        bucket_offset = current_epoch_samples // self.data_parallel_size

        idx_range = active_indices_split[bucket_offset:]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []


class BalancedDataset(Dataset):
    def __init__(self,
                 json_files,
                 max_seq_length=4096,
                 llm_thresh=4050,
                 iter_time=10,
                 vit_packed_length=9,
                 init=True,
                 fast_group=False,
                 vit_packed_thresh=None):
        super(BalancedDataset, self).__init__()
        self.load_wrap_meta(json_files)
        self.seed = 1024
        self.vit_packed_length = vit_packed_length
        self.llm_packed_length = max_seq_length
        self.llm_thresh = llm_thresh
        self.collect_origin_info(self.token_lengths)
        if vit_packed_thresh is None:
            vit_packed_thresh = vit_packed_length
        if init:
            if fast_group:
                from fast_packed import fast_random_group
                self.pack_group = fast_random_group.fast_process_random_groups(self.token_lengths,
                                                                               self.seed,
                                                                               vit_packed_thresh,
                                                                               vit_packed_length,
                                                                               llm_thresh,
                                                                               max_seq_length,
                                                                               iter_time=iter_time)
            else:
                self.pack_group = self.iter_random_groups(self.token_lengths, iter_time=iter_time)
            print(json.dumps(self.collect_packed_info(self.pack_group), indent=4, sort_keys=True))
            lengths = []
            for g in self.pack_group:
                temp = 0
                for item in g:
                    temp += item[2]
                lengths.append(temp)
            self.lengths = lengths

    def get_display_bin_num(self, llm_num, display_bin_size):
        mod, div = llm_num // display_bin_size, llm_num % display_bin_size
        if div > 0:
            mod += 1
        return mod * display_bin_size

    def collect_origin_info(self, patch_set):
        info = {}
        llm_info = {}
        all_vit_length = []
        all_llm_length = []
        act_length = []
        for item in patch_set:
            vit_num, llm_num = item[1], item[2]
            if vit_num not in info:
                info[vit_num] = 0
            info[vit_num] += 1

            llm_num_bin = self.get_display_bin_num(llm_num, 256)
            if llm_num_bin not in llm_info:
                llm_info[llm_num_bin] = 0
            llm_info[llm_num_bin] += 1
            all_vit_length.append(vit_num)
            all_llm_length.append(llm_num_bin)
            act_length.append(llm_num)
        info['vit_length_mean'] = np.mean(all_vit_length)
        info['vit_length_var'] = np.var(all_vit_length)
        info['vit_length_std'] = np.std(all_vit_length)
        
        info['llm_length_mean'] = np.mean(all_llm_length)
        info['llm_length_var'] = np.var(all_llm_length)
        info['llm_length_std'] = np.std(all_llm_length)

        print("origin vit info")
        print(json.dumps(info, indent=4))
        print("origin llm info")
        print(json.dumps(llm_info, indent=4, sort_keys=True))
        self.origin_info = info

    def load_wrap_meta(self, json_files):
        token_lengths = []
        idx = 0
        for wrap_file in json_files:
            with open(wrap_file) as f:
                data_info = json.load(f)
            for data_name in data_info.keys():
                if "token_lengths" in data_info[data_name]:
                    token_length_path = data_info[data_name]['token_lengths']
                    with open(token_length_path, "r") as f:
                        token_length = json.load(f)
                    for item in token_length:
                        token_lengths.append([idx, item['vit_num'], item['token_num']])
                        idx += 1
        self.token_lengths = token_lengths

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

    def iter_random_groups(self, groups, llm_thresh=None, seed=None, iter_time=300):
        if llm_thresh is None:
            llm_thresh = self.llm_thresh
        if seed is None:
            seed = self.seed
        groups = self._random_groups(groups, seed=seed)
        if iter_time == 1:
            return groups
        output = []
        for i in range(iter_time - 1):
            print(f"iter_random_groups {i} / {iter_time - 1}")
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
            print(i, len(groups), len(need_process_groups))
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
        info_dict['llm_num_max'] = llm_num_max
        info_dict['llm_num_min'] = llm_num_min
        info_dict['vit_ave_num'] = vit_ave_num / float(len(packed_groups))
        info_dict['llm_ave_num'] = llm_ave_num / float(len(packed_groups))
        info_dict['sample_num'] = sample_num
        info_dict['packed_group_num'] = len(packed_groups)
        info_dict['ave_bs'] = sample_num / float(len(packed_groups))
        self.info_dict = info_dict
        return info_dict
    
    def __getitem__(self, idx):
        groups = self.pack_group[idx]
        vit_num = 0
        llm_num = 0
        for item in groups:
            vit_num += item[1]
            llm_num += item[2]
        return (llm_num, vit_num)

    def __len__(self):
        """
        Returns dataset length
        """
        return len(self.pack_group)

class BaseDataset(Dataset):
    def __init__(self,
                 json_files):
        super(BaseDataset, self).__init__()
        self.load_wrap_meta(json_files)

    def load_wrap_meta(self, json_files):
        lengths = []
        vit_nums = []
        for wrap_file in json_files:
            with open(wrap_file) as f:
                data_info = json.load(f)
            for data_name in data_info.keys():
                if "token_lengths" in data_info[data_name]:
                    token_length_path = data_info[data_name]['token_lengths']
                    with open(token_length_path, "r") as f:
                        token_length = json.load(f)
                    for item in token_length:
                        lengths.append(item['token_num'])
                        vit_nums.append(item['vit_num'])
        self.lengths = lengths
        self.vit_nums = vit_nums

    def __getitem__(self, idx):
        return (self.lengths[idx], self.vit_nums[idx])

    def __len__(self):
        """
        Returns dataset length
        """
        return len(self.lengths)


class BatchCollector(object):
    def __init__(self,
                 ignore_idx=-100,
                 max_seq_length=2048):
        self.ignore_idx = ignore_idx
        self.max_seq_length = max_seq_length

    def __call__(self, instances):
        vit = []
        llm = []
        for instance in instances:
            vit.append(instance[1])
            llm.append(instance[0])
        max_length = max(llm)
        pad_list = []
        seq_len = []
        vit_num = []
        for idx, instance in enumerate(llm):
            pad_list.append(max_length - instance)
            seq_len.append(instance)
            vit_num.append(vit[idx])

        return dict(
            max_length=max_length,
            pad_list=pad_list,
            seq_len=seq_len,
            vit_num=vit_num
        )

def get_pad_dist_ratio(dataset, sampler_type='random', dp_size=16, micro_bs=4):
    random_pad_token = 0
    random_all_token = 0
    random_act_token = 0
    dp_output = []
    vit_output = []
    iteration = 0
    for rank in range(dp_size):
        print("random rank", rank)
        if sampler_type == 'random':
            batch_sampler = MegatronPretrainingRandomSampler(len(dataset), 0, micro_bs, rank, dp_size)
        if sampler_type == 'length_group':
            batch_sampler = MegatronLengthGroupSampler(len(dataset), 0, micro_bs, rank, dp_size, lengths=dataset.lengths)
        batch_collator = BatchCollector()
        dataloader = DataLoader(dataset,
                                batch_sampler=batch_sampler,
                                num_workers=0,
                                collate_fn=batch_collator)
        temp_llm = []
        temp_vit = []
        for data in dataloader:
            iteration += 1
            random_pad_token += sum(data['pad_list'])
            random_all_token += data['max_length'] * micro_bs
            random_act_token += sum(data['seq_len'])
            temp_llm.append(data['max_length'])
            temp_vit.append(sum(data['vit_num']))
        dp_output.append(temp_llm)
        vit_output.append(temp_vit)
    device_group = []
    device_vit_group = []

    for idx in range(len(temp_llm)):
        temp_data = []
        temp_vit_data = []
        for i in range(dp_size):
            temp_data.append(dp_output[i][idx])
            temp_vit_data.append(vit_output[i][idx])
        llm_max_v = max(temp_data)
        llm_waste_v = 0
        for item in temp_data:
            llm_waste_v += (llm_max_v - item)
        llm_waste_ratio = llm_waste_v / float(llm_max_v * dp_size)
        vit_max_v = max(temp_vit_data)
        vit_waste_v = 0
        for item in temp_vit_data:
            vit_waste_v += (vit_max_v - item)
        vit_waste_ratio = vit_waste_v / float(vit_max_v * dp_size)
        device_group.append(llm_waste_ratio)
        device_vit_group.append(vit_waste_ratio)
    pad_ratio = random_pad_token / float(random_all_token)
    vit_dist_ratio = sum(device_vit_group) / len(device_vit_group)
    llm_dist_ratio = sum(device_group) / len(device_group)
    print("pad ratio", pad_ratio)
    print('dist ratio llm', llm_dist_ratio)
    print('dist ratio vit', vit_dist_ratio)
    return pad_ratio, vit_dist_ratio, llm_dist_ratio

def get_init_llm_vit_len(json_files):
    o_dataset = BalancedDataset(json_files=json_files, init=False)
    o_vit_bs_mean = o_dataset.origin_info['vit_length_mean']
    o_llm_len_mean = o_dataset.origin_info['llm_length_mean']
    init_ave_bs_llm_len = ((o_llm_len_mean // o_vit_bs_mean + 1) // 16) *  16

    init_llm_len = 4096
    init_vit_bs = init_llm_len // init_ave_bs_llm_len
    return init_vit_bs, init_llm_len, init_ave_bs_llm_len

if __name__ == "__main__":

    json_files = ['./internvl_sft_1.2M.json']
    best_r = 100
    result_list = []
    vit_bs, llm_len, init_ave_bs_llm_len = get_init_llm_vit_len(json_files)
    step = 1
    itertime = 50
    max_ave_bs = 0
    micro_bs = 4
    dp_size = 8

    while True:
        print(vit_bs, llm_len)
        threshs = [llm_len - 128]
        for thresh in threshs:
            balanced_dataset = BalancedDataset(json_files, llm_len, thresh, itertime, vit_bs, fast_group=False)
            ave_bs = balanced_dataset.info_dict['ave_bs']
            max_ave_bs = max(ave_bs, max_ave_bs)
            pad_r, v_dist_r, l_dist_r = get_pad_dist_ratio(balanced_dataset, sampler_type='random', dp_size=dp_size, micro_bs=1)
            if sum([pad_r, v_dist_r, l_dist_r]) < best_r:
                best_r = sum([pad_r, v_dist_r, l_dist_r])
            if ave_bs >= micro_bs and ave_bs <= micro_bs + 1:
                result_list.append((sum([pad_r, v_dist_r, l_dist_r]), v_dist_r, l_dist_r, vit_bs, llm_len, thresh, ave_bs))
        if max_ave_bs > micro_bs + 1:
            if len(result_list) > 0:
                break
            else:
                vit_bs -= step
                llm_len -= step * init_ave_bs_llm_len
        else:
            vit_bs += step
            llm_len += step * init_ave_bs_llm_len
    print("ratio, v_dist_ratio, l_dist_ratio, vit_bs, llm_len, llm_thresh, ave_bs")
    print(sorted(result_list))

    
