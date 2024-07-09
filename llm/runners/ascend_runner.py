import sys
import time
import os

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from tqdm import tqdm
from enum import Enum
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm import ascend 
from ascend.model import DistributedDataParallel as LocalDDP
from ascend.model import Float16Module
from llm.utils.general.parser_helper import parse_args
from ascend.global_vars import set_global_variables, get_args
from ascend.arguments import validate_args
from ascend.global_vars import _set_tokenizer
from ascend.global_vars import _set_num_microbatches_calculator
from ascend.arguments import core_transformer_config_from_args
import deepspeed
from deepspeed.accelerator import get_accelerator
from ascend.optimizer import build_megatron_optimizer
# from llm.utils.model.optimizer_helper import build_optimizer
from ascend.model import get_model
from ascend.checkpoint import load_ckpt_pretrained
from ascend.model.gpt_model import GPTModelPipe
from llm.utils.env import dist_env
from llm.utils.env import (initialize_distributed, get_distributed_info,
                           setup_deepspeed_random_and_activation_checkpointing,
                           set_logging_verbosity,
                           set_random_seed)
from llm.utils.general.log_helper import default_logger as logger
from llm.utils.general.cfg_helper import merge_opts_into_cfg
from llm.utils.general.hook_helper import build_hooks
from llm.utils.general.parser_helper import parse_args
from llm.utils.general.yaml_loader import load_yaml
from llm.utils.general.microbatches import build_num_microbatches_calculator
from llm.utils.model.lr_helper import build_learning_rate_scheduler
from llm.utils.model.initializer import set_defaults_if_not_set_tensor_model_parallel_attributes
from llm.utils.model.ckpt_helper import load_checkpoint, save_checkpoint
from llm.utils.general.utils import get_train_iters

from llm.data import build_tokenizer, build_batch_pipe_fn, build_data_iterator
from llm.models.mg_models import log_trainable_params, get_layer_info
from llm.models.mg_models import unwrap_model
from llm.models.mg_models.llama.utils import load_lora_ckpt_pretrained, save_lora_ckpt_pretrained


_TRAIN_START_TIME = time.time()

def set_hook_model_args():
    args = get_args()
    model_args = {}
    model_args['micro_batch_size'] = args.micro_batch_size
    model_args['seq_len'] = args.seq_length
    model_args['hidden_size'] = args.hidden_size
    model_args['num_layers'] = args.num_layers
    model_args['vocab_size'] = args.padded_vocab_size
    model_args['checkpoint_activations'] = args.checkpoint_activations
    model_args['glu_activation'] = False
    return model_args

def set_model_args(model_cfg):
    args = get_args()
    model_args = {}
    model_args['seq_len'] = args.seq_length
    model_args['hidden_size'] = args.hidden_size
    model_args['num_layers'] = args.num_layers
    model_args['vocab_size'] = args.padded_vocab_size
    model_args['num_attention_heads'] = args.num_attention_heads
    model_args['num_kv_heads'] = model_cfg.get('num_query_groups', args.num_attention_heads)
    return model_args

class BaseRunner(object):
    def __init__(self, args, cfg=None, training=True, base_type='train'):
        self.args = args
        self.config = cfg
        self.training = training
        self.base_type = base_type
        self.build()

    def build(self):
        self.set_param_components()
        self.build_env()
        if self.args.opts is not None:
            self.config = merge_opts_into_cfg(self.args.opts, self.config)
        self.build_num_microbatches_calculator()
        self.build_tokenizer()
        self.build_model()
        self.build_hooks()
        self.build_trainer()
        if self.args.deepspeed:
            self.deepspeed_init()
        self.load_checkpoint()
        self.build_data_engine()


    def set_param_components(self):
        self.start_iteration = 0
        self.consumed_train_samples = 0
        self.consumed_train_tokens = 0
        # set deepspeed configs
        self.deepspeed = self.config['runtime'].get('deepspeed', True)
        # assert self.deepspeed is True, 'only support deepspeed mode now!'
        if self.deepspeed:
            cfg_deepspeed = self.config.get('deepspeed', None)
            assert cfg_deepspeed is not None and (cfg_deepspeed.get('config', None) is not None), 'deepspeed mode must provide the configs of deepspeed!'     # noqa
            cfg_deepspeed['config'].update({'train_micro_batch_size_per_gpu': self.config['data'][self.base_type]['micro_batch_size'],     # noqa
                                            'train_batch_size': self.config['data'][self.base_type]['global_batch_size']})
            self.cfg_deepspeed = cfg_deepspeed
            assert self.args.deepspeed_config is None, 'please pass the deepspeed config in the config.json file, and the deepspeed_config is disabled now.'     # noqa
            # fp16 and bf16
            assert self.config['runtime'].get('fp16', False) == cfg_deepspeed['config'].get('fp16', {}).get('enabled', False)       # noqa
            assert self.config['runtime'].get('bf16', False) == cfg_deepspeed['config'].get('bf16', {}).get('enabled', False)       # noqa
        # set lora configs
        self.lora_mode = self.config['runtime'].get('lora_mode', False)
        cfg_lora = self.config.get('lora', None)
        if self.lora_mode:
            assert cfg_lora is not None, 'lora mode must provide the configs of lora!'
        self.cfg_lora = cfg_lora

    def build_env(self, rank=None, local_rank=None):
        cfg_runtime = self.config['runtime']
        # get env info
        rank, local_rank, world_size, tensor_model_parallel_size, \
            pipeline_model_parallel_size = get_distributed_info(cfg_runtime, self.args.launcher, self.args.port)
        # initialize env
        initialize_distributed(rank, local_rank, world_size, tensor_model_parallel_size,
                               pipeline_model_parallel_size, None if self.deepspeed else 'nccl',
                               self.args.launcher, deepspeed=self.deepspeed)
        # Initialize deepspeed random and activation checkpointing.
        seed = cfg_runtime.get('seed', 42)
        if self.deepspeed:
            num_layers, checkpoint_num_layers = get_layer_info(self.config['model'])
            cfg_activation_checkpoint = self.cfg_deepspeed.get('activation_checkpoint', {})
            cfg_activation_checkpoint.update({'base_num_layers': num_layers,
                                              'checkpoint_num_layers': checkpoint_num_layers})
            setup_deepspeed_random_and_activation_checkpointing(**cfg_activation_checkpoint)
        set_random_seed(seed, cfg_runtime.get('dp_random_init', False))
        # Set logging verbosity
        set_logging_verbosity(rank, cfg_runtime.get('log_level', 'info'),
                              cfg_runtime.get('log_level_replica', 'error'),
                              deepspeed=self.deepspeed)
        # get global start time
        global _TRAIN_START_TIME
        start_time_tensor = get_accelerator().FloatTensor([_TRAIN_START_TIME])
        torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
        self.start_time = start_time_tensor.item()
        logger.info('Initialize env done! Times (seconds): {:.3f}'.format(time.time() - self.start_time))


    def build_num_microbatches_calculator(self):
        if self.training:
            self.num_microbatches_calculator = build_num_microbatches_calculator(self.config['data']['train']['batch_calculator'])      # noqa
        else:
            self.num_microbatches_calculator = None
        _set_num_microbatches_calculator(self.num_microbatches_calculator)

    def build_tokenizer(self):
        self.tokenizer = build_tokenizer(self.config['tokenizer'])
        self.args.padded_vocab_size = self.tokenizer.padded_vocab_size
        _set_tokenizer(self.tokenizer)

    def build_hooks(self):
        self.set_hook_model_args = set_hook_model_args
        self.unwraped_model().model_kwargs = set_model_args(self.config['models'])
        cfg_hooks = self.config.get('hooks', [])
        self._hooks = build_hooks(self, cfg_hooks, is_train=self.training, add_log_if_not_exists=True)
        logger.info('build hooks done')

    def build_data_engine(self):
        cfg_data = self.config['data']
        data_types = cfg_data.get('data_types', ['train', 'test'])
        self.data_iterators = {}
        self.batch_pipe_func = {}
        for data_type in data_types:
            assert data_type in ['train', 'valid', 'test', 'infer'], 'data type only support train, valid, test, and infer'       # noqa
            self.batch_pipe_func[data_type] = build_batch_pipe_fn(cfg_data[data_type]['batch_pipe'], self.tokenizer)
            if data_type == 'infer':
                infer_type = cfg_data[data_type].get('infer_type', 'interactive')
                if infer_type == 'interactive':
                    continue        # skip build data_iterators for inference mode
            data_iterator, dataset_size = build_data_iterator(self.tokenizer, cfg_data, self.consumed_train_samples, data_type) # noqa
            self.data_iterators[data_type] = data_iterator
        if self.training:
            epoch = self.config['trainer'].get('epoch', -1)
            if epoch > 0:
                global_batch_size = self.num_microbatches_calculator.global_batch_size
                train_iters = int((dataset_size.item() // global_batch_size + 1) * epoch)
            else:
                train_iters = self.config['trainer'].get('train_iters', 100)
            self.set_train_iters(train_iters)

    def build_model(self):
        model_config = core_transformer_config_from_args(get_args())
        if self.deepspeed:
            remote_device = self.cfg_deepspeed.get('remote_device', 'none')
            zero_stage = self.cfg_deepspeed['config'].get('zero_optimization', {}).get('stage', 1.0)
            with deepspeed.zero.Init(data_parallel_group=dist_env.get_data_parallel_group(),
                                     remote_device=None if (remote_device == 'none') else remote_device,
                                     config_dict_or_path=self.cfg_deepspeed["config"],
                                     enabled=(zero_stage == 3), mpu=dist_env):
                model = GPTModelPipe(config=model_config, parallel_output=True)
                model.load_lora_ckpt_pretrained = load_lora_ckpt_pretrained
                model.load_ckpt_pretrained = load_ckpt_pretrained
                model.save_lora_ckpt_pretrained = save_lora_ckpt_pretrained
                # from .utils import set_train_params, set_train_status
                # set trainable params
                # if hasattr(model, 'set_train_params'):
                #     model.set_train_params(model, lora_mode=self.lora_mode,
                #                            cfg_lora=self.cfg_lora, is_train=self.training)
                # log trainable params
                log_trainable_params(model)
            for param in model.parameters():
                set_defaults_if_not_set_tensor_model_parallel_attributes(param)     # noqa
            self.model = model
        else:
            self.model = get_model(model_config)[0]
            self.unwraped_model().load_ckpt_pretrained = load_ckpt_pretrained
        self.config['loader'].update({"deepspeed": self.deepspeed})

    def set_train_iters(self, train_iters):
        self.total_train_iters = get_train_iters(self.num_microbatches_calculator,
                                                 train_iters,
                                                 self.config['trainer'].get('train_samples', None))

    def build_trainer(self):
        unwrapped_model = [self.model] if not self.deepspeed else [unwrap_model(self.model, (torchDDP, LocalDDP, Float16Module))]
        if self.training:
            cfg_optim = self.config['trainer']['optimizer']
            optimizer = build_megatron_optimizer(unwrapped_model)
            # optimizer = build_optimizer(cfg_optim, unwrapped_model[0], deepspeed=self.deepspeed)
            cfg_lr_scheduler = self.config['trainer']['lr_scheduler']
            cfg_lr_scheduler['kwargs']['max_lr'] = cfg_optim['lr']        # noqa
            if cfg_lr_scheduler['type'] == 'iter_base_annealing':
                cfg_lr_scheduler['kwargs']['global_batch_size'] = self.num_microbatches_calculator.global_batch_size
            lr_scheduler = build_learning_rate_scheduler(cfg_lr_scheduler, optimizer)
        else:
            optimizer = None
            lr_scheduler = None
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def deepspeed_init(self):
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            config=self.cfg_deepspeed['config'],
            args=None,
        )

        assert model.fp16_enabled() == self.config['runtime'].get('fp16', False), "fp16 config does not match deepspeed"
        assert model.bfloat16_enabled() == self.config['runtime'].get('bf16', False), "bf16 config does not match deepspeed"        # noqa

        if isinstance(model, deepspeed.PipelineEngine):
            assert model.grid.get_pipe_parallel_rank() == dist_env.get_pipeline_model_parallel_rank()
            assert model.grid.get_slice_parallel_rank() == dist_env.get_tensor_model_parallel_rank()
            assert model.grid.get_data_parallel_rank() == dist_env.get_data_parallel_rank()
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
    
    def unwraped_model(self,):
        unwrap_model_all = unwrap_model(self.model, (torchDDP, LocalDDP, Float16Module))
        return unwrap_model_all

    def load_checkpoint(self):
        torch.distributed.barrier()

        self.start_iteration, self.consumed_train_samples, \
            self.consumed_train_tokens = load_checkpoint(self.unwraped_model(), self.optimizer, self.num_microbatches_calculator,
                                                         self.config['loader'], self.lora_mode, self.cfg_lora)     # noqa
        if hasattr(self.lr_scheduler, 'consumed_train_tokens'):
            self.lr_scheduler.consumed_train_tokens = self.consumed_train_tokens
        if hasattr(self.lr_scheduler, 'num_steps'):
            self.lr_scheduler.num_steps = self.consumed_train_samples     # deepspeed based
        torch.distributed.barrier()

    def save_checkpoint(self, cur_iter):
        cfg_saver = self.config['saver']
        save_interval = cfg_saver.get('save_interval', 0)
        if (save_interval and (cur_iter + 1) % save_interval == 0) or (cur_iter + 1 == self.total_train_iters):
            if self.deepspeed:
                save_checkpoint((cur_iter + 1), self.consumed_train_samples, self.consumed_train_tokens,
                                self.model, cfg_saver, self.lora_mode, self.cfg_lora)
            else:
                unwrap_model_all = unwrap_model(self.model, (torchDDP, LocalDDP, Float16Module))
                save_checkpoint((cur_iter + 1), self.consumed_train_samples, self.consumed_train_tokens,
                                unwrap_model_all, cfg_saver, self.lora_mode, self.cfg_lora, self.optimizer, self.lr_scheduler)
            logger.info('Saving checkpoint at the {}_th iter.'.format(cur_iter + 1))

    def forward_step(self, data_iterator):
        if self.deepspeed:
            assert isinstance(self.model, deepspeed.PipelineEngine), self.model
            loss = self.model.train_batch(data_iter=data_iterator)
            return {'lm_loss': loss}
        else:
            from ascend.train import train_step
            loss_dict, _, grad_norm, _ = train_step(self.model, self.optimizer, self.lr_scheduler, data_iterator, self.model_config)
            loss_dict.update({'grad_norm': grad_norm})
            return loss_dict

    def train(self):
        data_type = 'train'
        assert data_type == self.base_type, 'Training type! But the base type is {}'.format_map(self.base_type)
        # Iterations.
        micro_batch_size = self.config['data'][data_type]['micro_batch_size']
        seq_length = self.config['data'][data_type]['seq_length']
        # hooks
        self._hooks('before_train')
        # model status
        if self.deepspeed:
            self.model.set_batch_fn(self.batch_pipe_func[data_type])
        else:
            self.model.batch_func = self.batch_pipe_func[data_type]
        if hasattr(self.model, 'set_train_status'):
            self.model.set_train_status(self.model, self.lora_mode)
        else:
            self.model.train()
        from ascend.core.core_utils import get_model_config
        self.model_config = get_model_config(self.model)
        if not self.deepspeed:
            self.model_config.grad_scale_func = self.optimizer.scale_loss
        for iteration in range(self.start_iteration, self.total_train_iters):
            self._hooks('before_train_iter', iteration)
            self.num_microbatches_calculator.update(self.consumed_train_samples, True)
            # inform deepspeed of any batch size changes
            global_batch_size = dist_env.get_data_parallel_world_size() * micro_batch_size * \
                self.num_microbatches_calculator.get()
            if self.deepspeed:
                self.model.set_train_batch_size(global_batch_size)
            # forward step
            output = self.forward_step(self.data_iterators[data_type])
            self.consumed_train_samples += global_batch_size
            self.consumed_train_tokens += global_batch_size * seq_length
            if hasattr(self.lr_scheduler, 'consumed_train_tokens'):
                self.lr_scheduler.consumed_train_tokens = self.consumed_train_tokens
            self._hooks('after_train_iter', iteration, output)
            
            self.save_checkpoint(iteration)
        self._hooks('after_train')


def main():
    args = parse_args()
    assert args.config is not None, 'please provide a config file'
    cfg = load_yaml(args.config)
    def parser_func(value_from, value_to):
        if isinstance(value_from, Enum):
            return type(value_from)[value_to]
        return value_to
    for cfg_obj in [cfg["runtime"], cfg["models"], cfg["trainer"]["optimizer"]]:
        for key in cfg_obj:
            if hasattr(args, key):
                setattr(args, key, parser_func(getattr(args, key), cfg_obj[key]))
    validate_args(args)
    set_global_variables(args, build_tokenizer=False)
    runtime_none_keys = ['seed', 'local_rank', 'tensor_model_parallel_size',
                         'pipeline_model_parallel_size', 'distributed_backend']
    runtime_store_true_keys = ['fp16', 'bf16', 'deepspeed', 'lora_mode']
    cfg['runtime'] = cfg.setdefault('runtime', {})
    for key in (runtime_none_keys + runtime_store_true_keys):
        val = getattr(args, key)
        if key in runtime_none_keys and val is not None:
            cfg['runtime'].update({key: val})
        elif key in runtime_store_true_keys and val is True:
            cfg['runtime'].update({key: val})
    runner = BaseRunner(args, cfg, training=True, base_type='train')
    runner.train()


if __name__ == "__main__":
    main()
