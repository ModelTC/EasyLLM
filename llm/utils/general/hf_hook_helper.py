from llm.utils.general.hook_helper import Hook
from llm.utils.general.registry_factory import HOOK_REGISTRY
from llm.utils.general.log_helper import default_logger as logger
from llm.utils.env.hf_dist_helper import get_rank, all_reduce, get_world_size
from llm.utils.general.utils import Timers
from collections import defaultdict
import torch
import datetime


@HOOK_REGISTRY.register('dynamic_checkpoint')
class DynamicCheckpoint(Hook):
    def __init__(self,
                 runner,
                 enable=True,
                 checkpoint_patterns={},
                 strategy={},
                 debug_freq=-1,
                 level_keys="layers",
                 use_dc=True):
        super(DynamicCheckpoint, self).__init__(runner)
        self.enable = enable
        self.checkpoint_patterns = checkpoint_patterns
        self.checkpoint_set = []
        self.exclude_node = []
        self.level_keys = level_keys
        self.use_dc = use_dc
        self.debug_freq = debug_freq
        self.strategy = strategy

    def cast_forward(self):
        from .dc_manager import DynamicCheckpointManager, dc_cast_forward, common_cast_forward
        self.dc_manager = DynamicCheckpointManager(self.checkpoint_set,
                                                   debug_freq=self.debug_freq,
                                                   strategy=self.strategy)
        for name, m in self.runner_ref().model.named_modules():
            if name in self.checkpoint_set:
                if self.use_dc:
                    dc_cast_forward(m, name, self.dc_manager)
                else:
                    common_cast_forward(m)

    def recover_forward(self):
        for name, m in self.runner_ref().model.named_modules():
            if name in self.checkpoint_set:
                if hasattr(m, "old_forward"):
                    m.forward = m.old_forward

    def before_train_iter(self, cur_iter, input):
        if self.enable and self.use_dc:
            input_ids = input['input_ids']
            self.dc_manager.input_size = input_ids.shape[1]
            self.dc_manager.input_batch_size = input_ids.shape[0]
            self.dc_manager.before_forward(cur_iter)

    def after_train_iter(self, cur_iter, output):
        pass

    def is_leaf_node(self, cur_node, next_nodes):
        is_leaf = True
        for next_node in next_nodes:
            if next_node.startswith(cur_node):
                if next_node.count('.') > cur_node.count('.'):
                    is_leaf = False
        return is_leaf

    def get_bfs_checkpoint_node(self):
        main_key = ''
        for name, m in self.runner_ref().model.named_modules():
            if name != '' and len(name.split('.')) == 1:
                main_key = name
                break
        self.checkpoint_patterns[main_key] = {}
        self.checkpoint_patterns[f"{main_key}_level"] = 0
        for name, m in self.runner_ref().model.named_modules():
            if name == '':
                continue
            if hasattr(m, "inplace") and m.inplace:
                self.exclude_node.append(name)
            ck = main_key
            if name == ck or name.startswith(ck + '.'):
                level = name.count('.')
                if len(name.split('.')) > 0:
                    temp = name.split('.')[-1]
                    self.checkpoint_patterns[f"{temp}_level"] = level + 1
                max_level = max(self.checkpoint_patterns[ck].get('max_level', 0), level)
                if level not in self.checkpoint_patterns[ck]:
                    self.checkpoint_patterns[ck][level] = []
                self.checkpoint_patterns[ck]['max_level'] = max_level
                if isinstance(m, torch.nn.Embedding):
                    self.exclude_node.append(name)
                    continue
                self.checkpoint_patterns[ck][level].append(name)

    def add_leaf_node(self):
        for _, v in self.checkpoint_patterns.items():
            if isinstance(v, dict):
                for level in range(1, v['max_level'] - 1):
                    for cur_node in v[level]:
                        if self.is_leaf_node(cur_node, v[level + 1]):
                            v[level + 1].append(cur_node)

    def resort_by_level_keys(self):
        in_level_set = []
        out_level_set = []
        for key in self.checkpoint_set:
            if self.level_keys in key:
                in_level_set.append(key)
            else:
                out_level_set.append(key)
        self.checkpoint_set = in_level_set + out_level_set

    def _parse_node(self):
        for _, v in self.checkpoint_patterns.items():
            if isinstance(v, dict):
                level = self.checkpoint_patterns[f'{self.level_keys}_level']
                for item in v[level]:
                    self.checkpoint_set.append(item)
        for item in self.exclude_node:
            if item in self.checkpoint_set:
                self.checkpoint_set.remove(item)
        self.resort_by_level_keys()
        logger.info(f"All checkpoint module length: {len(self.checkpoint_set)}; names: ")
        logger.info(self.checkpoint_set)

    def parse_checkpoint_node(self):
        self.get_bfs_checkpoint_node()
        # self.add_leaf_node()
        self._parse_node()

    def before_train(self):
        if self.enable:
            if hasattr(self.runner_ref().model, "gradient_checkpointing_disable"):
                self.runner_ref().model.gradient_checkpointing_disable()
            if hasattr(self.runner_ref().model, "base_model"):
                self.runner_ref().model.base_model.gradient_checkpointing_disable()
            self.parse_checkpoint_node()
            self.cast_forward()

    def after_train(self):
        if self.enable:
            self.recover_forward()


@HOOK_REGISTRY.register('hf_train_val_logger')
class HFTrainValLLoggerHook(Hook):
    def __init__(self,
                 runner,
                 log_interval=1,
                 tensorboard=True,
                 log_dir='./log',
                 skip_iters=10):
        super(HFTrainValLLoggerHook, self).__init__(runner)
        self.log_interval = log_interval
        if tensorboard:
            self.tensorboard_writer = None
            if get_rank() == 0:
                from tensorboardX import SummaryWriter
                self.tensorboard_writer = SummaryWriter(log_dir=log_dir)
        else:
            self.tensorboard_writer = None
        self.timers = Timers()
        self.skip_iters = skip_iters

    def before_train(self):
        self.timers('train-iter-time').start()

    def after_train(self):
        logger.info("training Done!")

    def allreduce(self, output):
        """Sync loss and accuracy across gpus.
           loss and accuracy must be torch.Tensor.
           return loss and accuracy only
        """

        def filter_fn(x):
            return x.find('loss') >= 0 or x.find('accuracy') >= 0

        output = {name: val.clone() for name, val in output.items() if filter_fn(name)}

        if get_world_size() > 1:
            for name, val in output.items():
                if torch.is_tensor(val):
                    all_reduce(val)
                    output[name] = val / get_world_size()

        return {name: val.item() for name, val in output.items()}

    def get_loss(self, output):
        losses = [val for name, val in output.items() if name.find('loss') >= 0]
        output['All.loss'] = sum(losses)
        # only loss and accuracy are kept
        output = self.allreduce(output)

        losses = defaultdict(list)
        count = 1
        for name, value in output.items():
            if len(name.split('.')) == 1:
                prefix = f'output{count}'
                local_name = name
            else:
                prefix, local_name = name.split('.', 1)
            losses[prefix].append('{}:{:.4f}'.format(local_name, value))
            count += 1
        # merge prefix
        losses = sorted([
            "{}({})".format(prefix, " ".join(loss_list))
            for prefix, loss_list in losses.items()
        ])
        return output, " ".join(losses)

    def get_time(self, cur_iter, total_iters):
        elapsed_time = self.timers('train-iter-time').elapsed()
        elapsed_time_per_iteration = elapsed_time / self.log_interval
        eta_seconds = elapsed_time_per_iteration * (total_iters - cur_iter)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        return elapsed_time_per_iteration, eta_string

    def writer_scalar(self, cur_iter, output, lr):
        if get_rank() == 0 and self.tensorboard_writer is not None:
            output['lr'] = lr
            for k, v in output.items():
                if 'loss' in k or 'lr' in k:
                    self.tensorboard_writer.add_scalar('train/' + k, v, cur_iter)

    def after_train_iter(self, cur_iter, output={}):
        runner = self.runner_ref()
        total_iters = runner.train_iters
        start_iter = runner.start_iter
        lr = runner.optimizer.param_groups[0]['lr']
        output, loss_str = self.get_loss(output)
        self.writer_scalar(cur_iter, output, lr)
        cur_iter = cur_iter + 1
        if cur_iter % self.log_interval == 0:
            elapsed_time_per_iteration, eta_string = self.get_time(cur_iter, total_iters)
            progress = cur_iter / float(total_iters) * 100.
            epoch_size = runner.train_epoch_size
            epoch = runner.get_cur_train_epoch()
            global_bs = runner.global_train_batch_size
            max_epoch = runner.get_max_train_epoch()
            o_string = ' '.join([
                "Progress:[{progress:.3f}%][{cur_iter}/{total_iters}]",
                "Epoch:[{epoch}/{max_epoch}][{local_iter}/{epoch_size}]",
                "LR:{lr:.7f}",
                "BS:{global_bs}",
                "Loss:{loss:.5f}",
                "elapsed_time_per_iteration:{elapsed_time_per_iteration:.4f}",
                "ETA:{eta}"]).format(
                    progress=progress,
                    cur_iter=cur_iter,
                    total_iters=total_iters,
                    lr=lr,
                    global_bs=global_bs,
                    local_iter=cur_iter % epoch_size,
                    epoch_size=epoch_size,
                    epoch=epoch,
                    max_epoch=max_epoch,
                    loss=output['All.loss'],
                    elapsed_time_per_iteration=elapsed_time_per_iteration,
                    eta=eta_string
            )
            logger.info(o_string)
            logger.info(loss_str)
        if (cur_iter - start_iter) == self.skip_iters:
            self.timers('train-iter-time').reset()
            self.timers('train-iter-time').start()
