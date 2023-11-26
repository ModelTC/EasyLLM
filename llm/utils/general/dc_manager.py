import torch
import torch.utils.checkpoint as checkpoint
from llm.utils.general.log_helper import default_logger as logger


def cast_checkpoint(func, *args, **kwargs):
    def create_custom_forward(module, **kwargs):
        if 'use_cache' in kwargs:
            kwargs['use_cache'] = False

        def custom_forward(*inputs):
            return module(*inputs, **kwargs)
        return custom_forward

    return checkpoint.checkpoint(create_custom_forward(func, **kwargs), *args)


def common_cast_forward(m, *args, **kwargs):
    old_forward = m.forward

    def forward(*args, **kwargs):
        return cast_checkpoint(old_forward, *args, **kwargs)
    m.forward = forward
    m.old_forward = old_forward


def dc_cast_forward(module_m, name, manager):
    old_forward = module_m.forward

    def forward(*args, **kwargs):
        if name in manager.cur_checkpoint_modules:
            return cast_checkpoint(old_forward, *args, **kwargs)
        else:
            return old_forward(*args, **kwargs)
    module_m.forward = forward
    module_m.old_forward = old_forward


class SizeMapStrategy(object):
    def __init__(self, batch_size=None, size_map={}) -> None:
        self.batch_size = batch_size
        if self.batch_size is not None:
            self.size_map = {}
            for k, v in size_map.items():
                self.size_map[k * batch_size] = v
        else:
            self.size_map = size_map
        self.size_list = list(self.size_map.keys())

    def get_size(self, input_size, batch_size):
        for item in self.size_list:
            if input_size * batch_size <= item:
                return self.size_map[item]
        return self.size_map[self.size_list[-1]]

    def get_cur_checkpoint_modules(self, cur_iter, input_size, all_checkpoint_modules, batch_size):
        if self.batch_size is None:
            logger.info("set batch size auto...")
            self.batch_size = batch_size
            size_map = {}
            for k, v in self.size_map.items():
                size_map[k * batch_size] = v
            self.size_map = size_map
            self.size_list = list(self.size_map.keys())
        size = self.get_size(input_size, batch_size)
        cur_checkpoint_modules = all_checkpoint_modules[:size]
        return cur_checkpoint_modules


class DynamicCheckpointManager(object):
    def __init__(self,
                 checkpoint_modules=[],
                 debug_freq=-1,
                 strategy={}):
        self.all_checkpoint_modules = checkpoint_modules
        self.cached_strategy = {}
        self.input_size = 0
        self.input_batch_size = 1
        self.debug_freq = debug_freq
        self.strategy_name = strategy
        self.strategy = self.init_dc_strategy(strategy)

    def init_dc_strategy(self, strategy):
        if strategy['type'] == 'predefine':
            strategy = SizeMapStrategy(**strategy['kwargs'])
        return strategy

    def before_forward(self, cur_iter):
        if self.input_size in self.cached_strategy:
            self.cur_checkpoint_modules = self.cached_strategy[self.input_size]
        else:
            self.cur_checkpoint_modules = self.strategy.get_cur_checkpoint_modules(cur_iter,
                                                                                   self.input_size,
                                                                                   self.all_checkpoint_modules,
                                                                                   self.input_batch_size)
            self.cached_strategy[self.input_size] = self.cur_checkpoint_modules
        if self.debug_freq > 0 and cur_iter % self.debug_freq == 0:
            self.debug_info()

    def debug_info(self):
        logger.info("================= schedule begin =================")
        logger.info(f"input size: {self.input_size} batch_size: {self.input_batch_size}")
        logger.info(f"memory_allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        logger.info(f"max_memory_allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")
        logger.info(f"memory_reserved: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
        logger.info(f"max_memory_reserved: {torch.cuda.max_memory_reserved() / (1024 ** 2):.2f} MB")
        logger.info("checkpoint modules:")
        for module_id in self.cur_checkpoint_modules:
            logger.info(f"\t{module_id}")
        logger.info("================== schedule end ==================")
