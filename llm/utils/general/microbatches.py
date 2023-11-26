# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Megatron number of micro-batches calculators."""

from abc import ABC
from abc import abstractmethod

from llm.utils.env import dist_env
from llm.utils.general.log_helper import default_logger as logger
from llm.utils.general.registry_factory import BATCH_CALCULATOR_REGISTRY


class NumMicroBatchesCalculator(ABC):

    def __init__(self):
        self.num_micro_batches = None
        self.current_global_batch_size = None

    def get(self):
        return self.num_micro_batches

    def get_current_global_batch_size(self):
        return self.current_global_batch_size

    @abstractmethod
    def update(self, consumed_samples, consistency_check):
        pass


@BATCH_CALCULATOR_REGISTRY.register('constant_num')
class ConstantNumMicroBatches(NumMicroBatchesCalculator):
    def __init__(self, global_batch_size, micro_batch_size):
        data_parallel_size = dist_env.get_data_parallel_world_size()
        micro_batch_times_data_parallel = micro_batch_size * \
            data_parallel_size
        assert global_batch_size % micro_batch_times_data_parallel == 0, \
            'global batch size ({}) is not divisible by micro batch size ({})' \
            ' times data parallel size ({})'.format(global_batch_size,
                                                    micro_batch_size,
                                                    data_parallel_size)
        self.num_micro_batches = global_batch_size // \
            micro_batch_times_data_parallel
        assert self.num_micro_batches >= 1
        self.global_batch_size = global_batch_size
        self.current_global_batch_size = global_batch_size

    def update(self, consumed_samples, consistency_check):
        pass


@BATCH_CALCULATOR_REGISTRY.register('rampup_batch_size_num')
class RampupBatchsizeNumMicroBatches(NumMicroBatchesCalculator):

    def __init__(self, start_batch_size, batch_size_increment, ramup_samples,
                 global_batch_size, micro_batch_size):
        """Batch size ramp up.
        Over
          steps = (global-batch-size - start-batch-size) / batch_size_increment
        increment batch size from start-batch-size to global-batch-size using
          rampup-samples / steps
        samples.
        Arguments:
            start_batch_size: global batch size to start with
            batch_size_increment: global batch size increments
            ramup_samples: number of samples to use ramp up global
               batch size from `start_batch_size` to `global_batch_size`
            global_batch_size: global batch size post rampup
            micro_batch_size: micro batch size
            data_parallel_size: data parallel size.
        """

        self.micro_batch_size = micro_batch_size
        self.data_parallel_size = dist_env.get_data_parallel_world_size()
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * \
            self.data_parallel_size
        assert self.micro_batch_times_data_parallel_size > 0

        assert start_batch_size > 0
        self.start_batch_size = start_batch_size

        assert global_batch_size > 0
        self.global_batch_size = global_batch_size
        diff_batch_size = self.global_batch_size - self.start_batch_size
        assert diff_batch_size >= 0
        assert batch_size_increment > 0
        self.batch_size_increment = batch_size_increment
        assert diff_batch_size % batch_size_increment == 0, 'expected ' \
            'global batch size interval ({}) to be divisible by global batch ' \
            'size increment ({})'.format(diff_batch_size, batch_size_increment)

        num_increments = diff_batch_size // self.batch_size_increment
        self.ramup_samples = ramup_samples
        assert self.ramup_samples >= 0
        self.rampup_samples_per_increment = self.ramup_samples / num_increments

        # Initialize number of microbatches.
        self.update(0, False)

    def update(self, consumed_samples, consistency_check):
        if consumed_samples > self.ramup_samples:
            self.current_global_batch_size = self.global_batch_size
        else:
            steps = int(consumed_samples / self.rampup_samples_per_increment)
            self.current_global_batch_size = self.start_batch_size + \
                steps * self.batch_size_increment
            assert self.current_global_batch_size <= self.global_batch_size

        if consistency_check:
            assert self.current_global_batch_size % \
                self.micro_batch_times_data_parallel_size == 0, 'current global ' \
                'batch size ({}) is not divisible by micro-batch-size ({}) ' \
                'times data parallel size ({})'.format(self.current_global_batch_size,
                                                       self.micro_batch_size,
                                                       self.data_parallel_size)
        self.num_micro_batches = self.current_global_batch_size // \
            self.micro_batch_times_data_parallel_size


def build_num_microbatches_calculator(cfg_batch_calculator):
    num_microbatches_calculator = BATCH_CALCULATOR_REGISTRY.build(cfg_batch_calculator)
    if isinstance(num_microbatches_calculator, ConstantNumMicroBatches):
        logger.info('setting number of micro-batches to constant {}'.format(num_microbatches_calculator.get()))
    elif isinstance(num_microbatches_calculator, RampupBatchsizeNumMicroBatches):
        logger.info('will use batch size rampup starting from global batch '
                    'size {} to global batch size {} with batch size increments '
                    '{} over {} samples.'.format(cfg_batch_calculator['kwargs']['start_batch_size'],
                                                 cfg_batch_calculator['kwargs']['global_batch_size'],
                                                 cfg_batch_calculator['kwargs']['batch_size_increment'],
                                                 cfg_batch_calculator['kwargs']['ramup_samples']))
    else:
        raise NotImplementedError
    return num_microbatches_calculator
