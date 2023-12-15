import math
import time
import itertools
from typing import Dict, Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torch.utils.data import _utils, IterDataPipe
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter, _share_dist_seed

from llm.utils.env import dist_env
from llm.utils.general.registry_factory import BATCH_COLLECTOR_REGISTRY, DATALOADER_REGISTRY
from .nlp_dataset import build_dataset
from .nlp_sampler import build_batch_sampler, InfiniteBatchSampler


@dataclass
@BATCH_COLLECTOR_REGISTRY.register('batch')
class BatchCollector(object):
    def __init__(self,
                 tokenizer,
                 ignore_idx=-100,
                 max_seq_length=2048,
                 offset_label=False):
        self.tokenizer = tokenizer
        self.ignore_idx = ignore_idx
        self.max_seq_length = max_seq_length
        self.pad_token_id = len(self.tokenizer) - 1
        self.offset_label = offset_label

    def _pad_func(self, input_ids, labels):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.ignore_idx)
        if self.offset_label:
            input_ids = input_ids[:, :-1]
            labels = input_ids[:, 1:]

        input_ids = input_ids[:, :self.max_seq_length]
        labels = labels[:, :self.max_seq_length]

        return input_ids, labels

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids, labels = self._pad_func(input_ids, labels)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.pad_token_id),
        )


@dataclass
@BATCH_COLLECTOR_REGISTRY.register('reward_batch')
class RewardBatchCollector(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer, ignore_idx=-100, max_seq_length=2048):
        self.tokenizer = tokenizer
        self.ignore_idx = ignore_idx
        self.max_seq_length = max_seq_length
        self.pad_token_id = len(self.tokenizer) - 1

    def _pad_with_alignment(self, chosed_input, reject_input):
        chosed_input_size = [len(item) for item in chosed_input]
        reject_input_size = [len(item) for item in reject_input]
        c_max_size = max(chosed_input_size)
        r_max_size = max(reject_input_size)
        max_size = max(c_max_size, r_max_size)
        chosed_pad_inputs = []
        chosed_pad_labels = []
        reject_pad_inputs = []
        reject_pad_labels = []
        for i in range(len(chosed_input)):
            chosed_pad_input = torch.LongTensor([self.pad_token_id] * min(self.max_seq_length, max_size))
            chosed_pad_label = torch.LongTensor([self.ignore_idx] * min(self.max_seq_length, max_size))
            reject_pad_input = chosed_pad_input.clone()
            reject_pad_label = chosed_pad_label.clone()
            c_size = chosed_input[i].size(0)
            r_size = reject_input[i].size(0)
            chosed_pad_input[:c_size] = chosed_input[i][:c_size]
            chosed_pad_label[:c_size] = chosed_input[i][:c_size]
            reject_pad_input[:r_size] = reject_input[i][:r_size]
            reject_pad_label[:r_size] = reject_input[i][:r_size]
            chosed_pad_inputs.append(chosed_pad_input)
            chosed_pad_labels.append(chosed_pad_label)
            reject_pad_inputs.append(reject_pad_input)
            reject_pad_labels.append(reject_pad_label)
        chosed_pad_inputs = torch.stack(chosed_pad_inputs)
        reject_pad_inputs = torch.stack(reject_pad_inputs)
        chosed_pad_labels = torch.stack(chosed_pad_labels)
        reject_pad_labels = torch.stack(reject_pad_labels)
        input_ids = torch.cat((chosed_pad_inputs, reject_pad_inputs))
        labels = torch.cat((chosed_pad_labels, reject_pad_labels))
        return input_ids, labels

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        chosed_input_ids = [instance["input_ids"][0] for instance in instances]
        reject_input_ids = [instance["input_ids"][1] for instance in instances]
        input_ids, labels = self._pad_with_alignment(chosed_input_ids, reject_input_ids)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.pad_token_id),
        )


@dataclass
@BATCH_COLLECTOR_REGISTRY.register('batch_align')
class BatchAlignCollector(BatchCollector):
    def __init__(self,
                 tokenizer,
                 alignment=1,
                 ignore_idx=-100,
                 max_seq_length=2048,
                 offset_label=True,
                 test_speed=False,
                 pretrain=False):
        super().__init__(tokenizer, ignore_idx)
        self.alignment = alignment
        self.max_seq_length = max_seq_length
        self.pad_token_id = len(self.tokenizer) - 1
        self.offset_label = offset_label
        self.test_speed = test_speed
        self.pretrain = pretrain
        if pretrain:
            self.data_keys = ["input_ids", "labels", "cu_seqlens", "position_ids"]
        else:
            self.data_keys = ["input_ids", "labels"]

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        item = tuple([instance[key] for instance in instances] for key in self.data_keys)  # noqa
        input_ids, labels = item[:2]
        if self.pretrain:
            cu_seqlens, position_ids = item[2:]
            input_ids, labels, cu_seqlens, position_ids = self._pad_func(input_ids, labels, cu_seqlens, position_ids)
        else:
            input_ids, labels = self._pad_func(input_ids, labels)
        data = dict(input_ids=input_ids,
                    labels=labels,
                    attention_mask=input_ids.ne(self.pad_token_id))
        if self.pretrain:
            data.update({"cu_seqlens": cu_seqlens, "position_ids": position_ids})
        return data

    def _pad_with_alignment(self, input, padding_value=0, is_label=False):
        if self.test_speed:  # no padding for speed test
            padding_value = 100
        sizes = [len(item) for item in input]
        max_size = max(sizes)
        new_size = math.ceil(max_size / float(self.alignment)) * self.alignment
        pad_inputs = []
        for item in input:
            pad_input = torch.LongTensor([padding_value] * min(self.max_seq_length, new_size))
            o_size = item.size(0) - 1
            if is_label:
                if self.offset_label:
                    pad_input[:o_size] = item[1:]
                else:
                    pad_input[:o_size + 1] = item[:self.max_seq_length]
            else:
                if self.offset_label:
                    pad_input[:o_size] = item[:-1]
                else:
                    pad_input[:o_size + 1] = item[:self.max_seq_length]
            pad_inputs.append(pad_input)
        return torch.stack(pad_inputs)

    def _pad_func(self, input_ids, labels, cu_seqlens=None, position_ids=None):
        input_ids = self._pad_with_alignment(input_ids, padding_value=self.pad_token_id)
        labels = self._pad_with_alignment(labels, padding_value=self.ignore_idx, is_label=True)
        if self.pretrain:
            position_ids = self._pad_with_alignment(position_ids, padding_value=0)
            flat_cu_seqlens = []
            for bidx in range(len(cu_seqlens)):
                ith_cu_seqlen = torch.clamp(cu_seqlens[bidx], max=self.max_seq_length)
                if bidx == 0:
                    flat_cu_seqlens.append(ith_cu_seqlen + bidx * self.max_seq_length)
                else:
                    flat_cu_seqlens.append((ith_cu_seqlen + bidx * self.max_seq_length)[1:])
            cu_seqlens = torch.cat(flat_cu_seqlens)
            return input_ids, labels, cu_seqlens, position_ids
        else:
            return input_ids, labels


@dataclass
@BATCH_COLLECTOR_REGISTRY.register('mini_rlhf_batch')
class MiniRLHFBatchCollector(BatchAlignCollector):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer, alignment, ignore_idx=-100, max_seq_length=2048, pad_type='right'):
        super().__init__(tokenizer, alignment, ignore_idx, max_seq_length, pad_type)

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, scores = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "scores"))       # noqa
        re_input_ids = []
        re_labels = []
        for i in range(len(input_ids)):
            re_input_ids.extend(input_ids[i])
            re_labels.extend(labels[i])

        input_ids, labels = self._pad_func(re_input_ids, re_labels)

        return dict(
            input_ids=input_ids,
            labels=labels,
            scores=torch.cat(scores, dim=0),
            attention_mask=input_ids.ne(self.pad_token_id),
        )


# TODO: remove once torch no longer has segment errors in large worker numbers.
class MultiProcessingDataLoaderIter(_MultiProcessingDataLoaderIter):
    def __init__(self, loader):
        super().__init__(loader)

    def _reset(self, loader, first_iter=False):
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        if isinstance(self._dataset, IterDataPipe):
            self._shared_seed = _share_dist_seed(loader.generator, self._pg)
            shared_rng = torch.Generator()
            shared_rng.manual_seed(self._shared_seed)
            self._dataset = torch.utils.data.graph_settings.apply_random_seed(self._dataset, shared_rng)
        self._send_idx = 0  # idx of the next task to be sent to workers
        self._rcvd_idx = 0  # idx of the next task to be returned in __next__
        # information about data not yet yielded, i.e., tasks w/ indices in range [rcvd_idx, send_idx).
        # map: task idx => - (worker_id,)        if data isn't fetched (outstanding)
        #                  \ (worker_id, data)   if data is already fetched (out-of-order)
        self._task_info = {}
        self._tasks_outstanding = 0  # always equal to count(v for v in task_info.values() if len(v) == 1)
        # A list of booleans representing whether each worker still has work to
        # do, i.e., not having exhausted its iterable dataset object. It always
        # contains all `True`s if not using an iterable-style dataset
        # (i.e., if kind != Iterable).
        # Not that this indicates that a worker still has work to do *for this epoch*.
        # It does not mean that a worker is dead. In case of `_persistent_workers`,
        # the worker will be reset to available in the next epoch.
        self._workers_status = [True for i in range(self._num_workers)]
        # Reset the worker queue cycle so it resumes next epoch at worker 0
        self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))
        # We resume the prefetching in case it was enabled
        if not first_iter:
            for idx in range(self._num_workers):
                self._index_queues[idx].put(_utils.worker._ResumeIteration(self._shared_seed))
            resume_iteration_cnt = self._num_workers
            while resume_iteration_cnt > 0:
                return_idx, return_data = self._get_data()
                if isinstance(return_idx, _utils.worker._ResumeIteration):
                    assert return_data is None
                    resume_iteration_cnt -= 1
        # prime the prefetch loop
        for _ in range(self._prefetch_factor * self._num_workers):
            # waiting to avoid the segment falut.
            time.sleep(0.01)
            self._try_put_index()


@DATALOADER_REGISTRY.register('base')
class BaseDataLoader(DataLoader):
    def __init__(self,
                 dataset,
                 batch_sampler=None,
                 batch_collator=None,
                 num_workers=0,
                 pin_memory=True,
                 seed=None,
                 **kwargs):
        if seed is None:
            generator = None
        else:
            generator = torch.Generator().manual_seed(seed)
        super(BaseDataLoader, self).__init__(
            dataset=dataset, batch_sampler=batch_sampler,
            num_workers=num_workers, generator=generator,
            collate_fn=batch_collator, pin_memory=pin_memory, **kwargs)

    # TODO: remove once torch no longer has segment errors in large worker numbers.
    def _get_iterator(self):
        if self.num_workers < 4:
            return super()._get_iterator()
        else:
            # hack code for dataloader segment fault.
            if self.num_workers == 0:
                return _SingleProcessDataLoaderIter(self)
            else:
                self.check_worker_number_rationality()
                return MultiProcessingDataLoaderIter(self)

    def get_epoch_size(self):
        if isinstance(self.batch_sampler, InfiniteBatchSampler):
            return len(self.batch_sampler.batch_sampler)   # training
        return len(self.batch_sampler)


def build_batch_collator(cfg_batch_collator, tokenizer):
    if 'kwargs' not in cfg_batch_collator:
        cfg_batch_collator['kwargs'] = {}
    cfg_batch_collator['kwargs']['tokenizer'] = tokenizer
    return BATCH_COLLECTOR_REGISTRY.build(cfg_batch_collator)


def build_data_loader(cfg_data, tokenizer):
    dataset = build_dataset(cfg_data['dataset'], tokenizer)
    cfg_data['batch_sampler']['kwargs'].update({'total_samples': len(dataset),
                                                'data_parallel_rank': dist_env.get_data_parallel_rank(),
                                                'data_parallel_size': dist_env.get_data_parallel_world_size()})
    batch_sampler = build_batch_sampler(cfg_data['batch_sampler'])
    batch_collator = build_batch_collator(cfg_data['batch_collector'], tokenizer)
    if 'kwargs' not in cfg_data['data_loader']:
        cfg_data['data_loader']['kwargs'] = {}
    cfg_data['data_loader']['kwargs'].update({'dataset': dataset,
                                              'batch_sampler': batch_sampler,
                                              'batch_collator': batch_collator})
    return DATALOADER_REGISTRY.build(cfg_data['data_loader'])
