import torch
from torch.utils.data.sampler import Sampler, BatchSampler
import math

from llm.utils.general.registry_factory import BATCH_SAMPLER_REGISTRY, SAMPLER_REGISTRY


@BATCH_SAMPLER_REGISTRY.register('megatron_pretrain')
class MegatronPretrainingSampler:
    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]


@BATCH_SAMPLER_REGISTRY.register('megatron_pretrain_random')
class MegatronPretrainingRandomSampler:
    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size):
        # Keep a copy of input params for later use.
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


@SAMPLER_REGISTRY.register('dist_test')
class TestDistributedSampler(Sampler):
    """
    Sampler that restricts data loading to a subset of the dataset, but won't align the total data
    size to be divisible by world_size bacause this will lead to duplicate detecton results
    """

    def __init__(self, dataset_size, data_parallel_rank, data_parallel_size):
        """
        Arguments:
             - dataset (:obj:`dataset`): instance of dataset object
        """
        num_replicas = data_parallel_size
        rank = data_parallel_rank

        self.dataset_size = dataset_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = len(range(rank, dataset_size, num_replicas))
        self.epoch = 0

    def __iter__(self):
        indices = torch.arange(self.dataset_size)
        indices = indices[self.rank::self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_samples


@SAMPLER_REGISTRY.register('dist')
class DistributedSampler(Sampler):
    def __init__(self, dataset_size, data_parallel_rank, data_parallel_size):
        num_replicas = data_parallel_size
        rank = data_parallel_rank
        self.dataset_size = dataset_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(self.dataset_size * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.consumed_samples = 0

    def __iter__(self):
        # deterministically shuffle based on epoch
        self.epoch = self.consumed_samples // self.total_size
        start_samples = self.consumed_samples % self.total_size
        if start_samples % self.num_replicas != 0:
            self.consumed_samples -= start_samples % self.num_replicas
            start_samples = (start_samples // self.num_replicas) * self.num_replicas

        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(self.dataset_size, generator=g).tolist()

        padding_size = self.total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        assert len(indices) == self.total_size

        # subsample
        offset_start = self.num_samples * self.rank + start_samples // self.num_replicas
        offset_end = self.num_samples * (self.rank + 1)
        indices = indices[offset_start:offset_end]
        assert len(indices) + start_samples // self.num_replicas == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_consumed_samples(self, consumed_samples):
        self.consumed_samples = consumed_samples


@BATCH_SAMPLER_REGISTRY.register('base')
class BaseBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last=False):
        super(BaseBatchSampler, self).__init__(sampler, batch_size, drop_last)

    def set_consumed_samples(self, consumed_samples):
        self.sampler.set_consumed_samples(consumed_samples)


class InfiniteBatchSampler(object):
    """Wraps a BatchSampler, resampling a specified number of iterations"""

    def __init__(self, batch_sampler, start_iter=0):
        """
        Arguments:
             - batch_sampler (:obj:`sampler`): instance of sampler object
        """
        self.batch_sampler = batch_sampler
        self.batch_size = self.batch_sampler.batch_size
        self.data_parallel_size = self.batch_sampler.sampler.num_replicas
        self.consumed_samples = 0

    def __iter__(self):
        while True:
            if hasattr(self.batch_sampler.sampler, "set_consumed_samples"):
                self.batch_sampler.sampler.set_consumed_samples(self.consumed_samples)

            for batch in self.batch_sampler:
                self.consumed_samples += len(batch) * self.data_parallel_size
                yield batch

    def __len__(self):
        return len(self.batch_sampler)

    def set_consumed_samples(self, consumed_samples):
        self.batch_sampler.sampler.set_consumed_samples(consumed_samples)
        self.consumed_samples = consumed_samples


def build_batch_sampler(cfg_batch_sample):
    if cfg_batch_sample['type'] == 'base':
        if 'sampler' in cfg_batch_sample['kwargs']:
            sampler_cfg = cfg_batch_sample['kwargs']['sampler']
            if 'kwargs' not in sampler_cfg:
                sampler_cfg['kwargs'] = {}
            sampler_cfg['kwargs'].update({'total_samples': cfg_batch_sample['kwargs']['total_samples'],
                                          'data_parallel_rank': cfg_batch_sample['kwargs']['data_parallel_rank'],
                                          'data_parallel_size': cfg_batch_sample['kwargs']['data_parallel_size']})

            sampler = SAMPLER_REGISTRY.build(sampler_cfg)
            batch_sampler = BaseBatchSampler(sampler, cfg_batch_sample['kwargs']['micro_batch_size'])
            batch_sampler = InfiniteBatchSampler(batch_sampler)
            return batch_sampler
    return BATCH_SAMPLER_REGISTRY.build(cfg_batch_sample)
