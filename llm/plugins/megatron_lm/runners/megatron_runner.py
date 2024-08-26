# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import dataclasses
import os
import copy
import json
import time
import torch

from megatron.training import get_args
from megatron.training import get_timers
from megatron.core.enums import ModelType
from megatron.core.utils import get_model_config
from megatron.core.distributed import finalize_model_grads
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.training.initialize import set_jit_fusion_options
from megatron.training.utils import unwrap_model
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
from megatron.training.checkpointing import save_checkpoint
from megatron.training.training import (
    get_model,
    get_optimizer_param_scheduler,
    num_floating_point_operations,
    training_log,
    save_checkpoint_and_time
)
from megatron.core.num_microbatches_calculator import (
    get_current_global_batch_size,
    get_current_running_global_batch_size
)

from llm.plugins.megatron_lm.utils.initialize import initialize_megatron
from llm.plugins.megatron_lm.utils.parser_helper import parse_args
from llm.utils.general.yaml_loader import load_yaml
from llm.utils.env import set_random_seed
from llm.utils.general.microbatches import build_num_microbatches_calculator
from llm.data import build_tokenizer, build_data_iterator
from llm.plugins.megatron_lm.utils.megatron_checkpointing import load_checkpoint
from llm.plugins.megatron_lm.utils.megatron_model_provider import model_provider
from llm.plugins.megatron_lm.utils.megatron_utils import forward_step, yaml2args
from llm.utils.general.hook_helper import build_hooks
from llm.utils.general.log_helper import default_logger as logger

if os.getenv("DIST_BACKEND", "easyllm") == "easyllm":
    from llm.utils.env import dist_env
elif os.getenv("DIST_BACKEND", "easyllm") == "megatron":
    from megatron.core import mpu as dist_env


_TRAIN_START_TIME = time.time()


class MegatronRunner(object):
    def __init__(self, args, cfg=None, training=True, base_type='train'):
        self.args = args
        self.config = copy.deepcopy(cfg)
        # mapping yaml args to megatron args
        yaml2args(self.config)
        self.training = training
        self.base_type = base_type
        self.build()
        self.display_train_info(cfg)

    def build(self):
        self.set_param_components()
        self.build_env()
        self.build_num_microbatches_calculator()
        self.build_tokenizer()
        # self.build_hooks()
        self.build_model()
        self.build_data_engine()
        self.build_trainer()
        self.load_checkpoint()
        # tensorboard_writer
        cfg_hooks = self.config['hooks']
        log_dir = "tf_logs/base"
        for cfg_hook in cfg_hooks:
            if cfg_hook['type'] == 'train_val_logger' and cfg_hook['kwargs'].get('tensorboard', True):
                log_dir = cfg_hook['kwargs'].get("log_dir", "tf_logs/base")
        self.tensorboard_writer = None
        # if torch.distributed.get_rank() == 0:
        if torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1):
            from tensorboardX import SummaryWriter
            self.tensorboard_writer = SummaryWriter(log_dir=log_dir)

    def build_hooks(self):
        cfg_hooks = self.config.get('hooks', [])
        self._hooks = build_hooks(self, cfg_hooks, is_train=self.training, add_log_if_not_exists=True)
        logger.info('build hooks done')

    def build_env(
        self,
        extra_args_provider=None,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        get_embedding_ranks=None,
        get_position_embedding_ranks=None
    ):
        cfg_runtime = self.config['runtime']
        # Initalize and get arguments, timers, and Tensorboard writer.
        initialize_megatron(
            extra_args_provider=extra_args_provider,
            args_defaults=args_defaults,
            get_embedding_ranks=get_embedding_ranks,
            get_position_embedding_ranks=get_position_embedding_ranks,
            ignore_unknown_args=True
        )
        # Set random seed.
        set_random_seed(cfg_runtime.get('seed', 42), cfg_runtime.get('dp_random_init', False))
        # get global start time
        global _TRAIN_START_TIME
        start_time_tensor = torch.cuda.FloatTensor([_TRAIN_START_TIME])
        torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
        self.start_time = start_time_tensor.item()
        logger.info('Initialize env done! Times (seconds): {:.3f}'.format(time.time() - self.start_time))

    def load_checkpoint(self):
        args = get_args()
        timers = get_timers()
        unwrapped_model = unwrap_model(self.model)

        cfg_loader = self.config["loader"]
        if cfg_loader.get("debug", False):
            args.iteration = 0
            args.num_floating_point_operations_so_far = 0
        else:
            timers('load-checkpoint', log_level=0).start(barrier=True)
            if cfg_loader.get("load_mode") == "huggingface":
                args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
                    self.model, self.optimizer, self.lr_scheduler, cfg_loader=cfg_loader)
            else:
                raise NotImplementedError('only support huggingface load mode.')
            timers('load-checkpoint').stop(barrier=True)
            timers.log(['load-checkpoint'])
        self.start_iteration = args.iteration + 1 if args.iteration == 0 else args.iteration

        # get model without FP16 and/or DDP wrappers
        if args.iteration == 0 and len(unwrapped_model) == 1 \
                and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
            logger.info("Initializing ICT from pretrained BERT model")
            unwrapped_model[0].init_state_dict_from_bert()
            if args.fp16:
                self.optimizer.reload_model_params()

    def build_model(self, model_type=ModelType.encoder_or_decoder):
        self.model = get_model(model_provider, model_type)

    def build_tokenizer(self):
        self.tokenizer = build_tokenizer(self.config['tokenizer'])

    def build_data_engine(self):
        cfg_data = self.config['data']
        data_types = cfg_data.get('data_types', ['train', 'test'])
        self.data_iterators = {}
        self.batch_pipe_func = {}
        for data_type in data_types:
            assert data_type in ['train', 'valid', 'test', 'infer'], 'data type only support train, valid, test, and infer'       # noqa
            if data_type == 'infer':
                infer_type = cfg_data[data_type].get('infer_type', 'interactive')
                if infer_type == 'interactive':
                    continue        # skip build data_iterators for inference mode
            data_iterator, dataset_size = build_data_iterator(self.tokenizer, cfg_data, self.consumed_train_samples, data_type)  # noqa
            self.data_iterators[data_type] = data_iterator
        if self.training:
            epoch = self.config['trainer'].get('epoch', -1)
            if epoch > 0:
                global_batch_size = self.num_microbatches_calculator.global_batch_size
                train_iters = int((dataset_size.item() // global_batch_size + 1) * epoch)
            else:
                train_iters = self.config['trainer'].get('train_iters', 100)

        args = get_args()
        args.do_train = True
        args.do_valid = False
        args.do_test = False
        args.train_iters = train_iters

    def build_trainer(self, no_wd_decay_cond=None, scale_lr_cond=None, lr_mult=1.0):
        args = get_args()
        timers = get_timers()
        if self.training:
            kwargs = {}
            for f in dataclasses.fields(OptimizerConfig):
                if hasattr(args, f.name):
                    kwargs[f.name] = getattr(args, f.name)
            config = OptimizerConfig(**kwargs)
            config.timers = timers
            optimizer = get_megatron_optimizer(config, self.model, no_wd_decay_cond,
                                               scale_lr_cond, lr_mult)
            lr_scheduler = get_optimizer_param_scheduler(optimizer)
        else:
            optimizer = None
            lr_scheduler = None
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def set_param_components(self):
        self.consumed_train_samples = 0
        # TODO: LoRA

    def build_num_microbatches_calculator(self):
        if self.training:
            self.num_microbatches_calculator = build_num_microbatches_calculator(self.config['data']['train']['batch_calculator'])      # noqa
            self.num_microbatches_calculator.update(self.consumed_train_samples, True)
        else:
            self.num_microbatches_calculator = None

    def display_train_info(self, cfg):
        logger.info(json.dumps(cfg, indent=4))
        dp_size = dist_env.get_data_parallel_world_size()
        tp_size = dist_env.get_tensor_model_parallel_world_size()
        pp_size = dist_env.get_pipeline_model_parallel_world_size()
        dist_info = f"dp size: {dp_size}; tp_size: {tp_size}; pp_size: {pp_size}"
        logger.info(dist_info)

    def forward_step(self):
        args = get_args()
        timers = get_timers()

        # Set grad to zero.
        for model_chunk in self.model:
            model_chunk.zero_grad_buffer()
        self.optimizer.zero_grad()

        # Forward pass.
        forward_backward_func = get_forward_backward_func()
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=self.data_iterators['train'],
            model=self.model,
            num_microbatches=self.num_microbatches_calculator.get(),  # get_num_microbatches(),
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False)

        # Empty unused memory.
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        # Vision gradients.
        if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":
            unwrapped_model = unwrap_model(self.model[0])
            unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

        # Update parameters.
        timers('optimizer', log_level=1).start(barrier=args.barrier_with_L1_time)
        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()
        timers('optimizer').stop()

        # Vision momentum.
        if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":
            unwrapped_model = unwrap_model(self.model[0])
            unwrapped_model.update_momentum(args.curr_iteration)

        # Update learning rate.
        if update_successful:
            # increment = get_num_microbatches() * \
            increment = self.num_microbatches_calculator.get() * \
                args.micro_batch_size * \
                args.data_parallel_size
            self.lr_scheduler.step(increment=increment)
            skipped_iter = 0
        else:
            skipped_iter = 1

        if dist_env.is_pipeline_last_stage(ignore_virtual=True):
            # Average loss across microbatches.
            loss_reduced = {}
            for key in losses_reduced[0].keys():
                if key not in loss_reduced:
                    loss_reduced[key] = 0
                for x in losses_reduced:
                    val = x[key]
                    # there is one dict per microbatch. in new reporting, we average
                    # over the total number of tokens across the global batch.
                    if isinstance(val, tuple) or isinstance(val, list):
                        loss_reduced[key] += val[0] / val[1]
                    else:
                        # legacy behavior. we average over the number of microbatches,
                        # and so the denominator is 1.
                        loss_reduced[key] += val
                # divide by accumulation_step
                gradient_accumulation_step = args.global_batch_size // dist_env.get_data_parallel_world_size() // args.micro_batch_size
                loss_reduced[key] /= gradient_accumulation_step
            return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad
        return {}, skipped_iter, grad_norm, num_zeros_in_grad

    def train(
        self,
        model_type=ModelType.encoder_or_decoder,
        process_non_loss_data_func=None
    ):
        args = get_args()
        timers = get_timers()

        # Temporary for transition to core datasets
        # train_valid_test_datasets_provider.is_distributed = True
        # Set pytorch JIT layer fusion options and warmup JIT functions.
        set_jit_fusion_options()
        # Context used for persisting some state between checkpoint saves.
        config = get_model_config(self.model[0])
        checkpointing_context = {}
        # set model to train mode
        for model_module in self.model:
            model_module.train()

        logger.info('training ...')
        total_loss_dict = {}
        num_floating_point_operations_so_far = args.num_floating_point_operations_so_far
        timers('interval-time', log_level=0).start(barrier=True)
        report_memory_flag = True
        total_flops = 0.0
        # Setup some training config params
        config.grad_scale_func = self.optimizer.scale_loss
        config.timers = timers
        if isinstance(self.model[0], DDP) and args.overlap_grad_reduce:
            assert config.no_sync_func is None, \
                ('When overlap_grad_reduce is True, config.no_sync_func must be None; '
                 'a custom no_sync_func is not supported when overlapping grad-reduce')
            config.no_sync_func = [model_chunk.no_sync for model_chunk in self.model]
            if len(self.model) == 1:
                config.no_sync_func = config.no_sync_func[0]
            if args.delay_grad_reduce:
                config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in self.model]
            if len(self.model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
        if args.overlap_param_gather and args.delay_param_gather:
            config.param_sync_func = [lambda x: self.optimizer.finish_param_sync(model_index, x)
                                      for model_index in range(len(self.model))]
            if len(self.model) == 1:
                config.param_sync_func = config.param_sync_func[0]
        config.finalize_model_grads_func = finalize_model_grads
        # TODO: resume training
        for iteration in range(self.start_iteration, args.train_iters + 1):
            self.num_microbatches_calculator.update(self.consumed_train_samples, True)
            args.curr_iteration = iteration
            loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = self.forward_step()
            # TODO: hook - tensorboard
            for key in loss_dict:
                if key == "lm loss":
                    avg = loss_dict[key].item()  # noqa
                    if self.tensorboard_writer is not None:
                        self.tensorboard_writer.add_scalar(f'train/lm_loss', avg, iteration)

            batch_size = dist_env.get_data_parallel_world_size() * \
                args.micro_batch_size * \
                self.num_microbatches_calculator.get()
            args.consumed_train_samples += batch_size
            num_skipped_samples_in_batch = (get_current_global_batch_size()
                                            - get_current_running_global_batch_size())
            if args.decrease_batch_size_if_needed:
                assert num_skipped_samples_in_batch >= 0
            else:
                assert num_skipped_samples_in_batch == 0
            args.skipped_train_samples += num_skipped_samples_in_batch
            num_fp_ops = num_floating_point_operations(args, batch_size)
            num_floating_point_operations_so_far += num_fp_ops
            total_flops += num_fp_ops

            # Logging.
            loss_scale = self.optimizer.get_loss_scale().item()
            params_norm = None
            # if args.log_params_norm:
            #     params_norm = calc_params_l2_norm(self.model)

            learning_rate = None
            decoupled_learning_rate = None
            for param_group in self.optimizer.param_groups:
                if param_group['is_decoupled_lr']:
                    decoupled_learning_rate = param_group['lr']
                else:
                    learning_rate = param_group['lr']
            report_memory_flag = training_log(loss_dict, total_loss_dict,
                                              learning_rate,
                                              decoupled_learning_rate,
                                              iteration, loss_scale,
                                              report_memory_flag, skipped_iter,
                                              grad_norm, params_norm, num_zeros_in_grad)

            # Checkpointing
            # saved_checkpoint = False
            # if args.exit_signal_handler:
            #     # signal_handler = get_signal_handler()
            #     if any(signal_handler.signals_received()):
            #         save_checkpoint_and_time(iteration, self.model, self.optimizer,
            #                                  self.lr_scheduler,
            #                                  num_floating_point_operations_so_far,
            #                                  checkpointing_context, train_data_iterator=self.data_iterators['train'])
            #         # print_datetime('exiting program after receiving SIGTERM.')
            #         break

            if args.save and args.save_interval and \
                    iteration % args.save_interval == 0:
                save_checkpoint_and_time(iteration, self.model, self.optimizer,
                                         self.lr_scheduler,
                                         num_floating_point_operations_so_far,
                                         checkpointing_context, train_data_iterator=self.data_iterators['train'])

            elif args.save and args.non_persistent_save_interval and \
                    iteration % args.non_persistent_save_interval == 0:
                timers('interval-time').stop()
                save_checkpoint_and_time(iteration, self.model, self.optimizer,
                                         self.lr_scheduler,
                                         num_floating_point_operations_so_far,
                                         non_persistent_ckpt=True, train_data_iterator=self.data_iterators['train'])
                timers('interval-time', log_level=0).start(barrier=True)

        if args.save and iteration != 0 and iteration % args.save_interval != 0:
            save_checkpoint(iteration, self.model, self.optimizer, self.lr_scheduler,
                            num_floating_point_operations_so_far, checkpointing_context,
                            train_data_iterator=self.data_iterators['train'])


def main():
    args = parse_args(ignore_unknown_args=True)
    if os.environ.get('ACCELERATOR_BACKEND') == 'TORCH_NPU':
        args.distributed_backend = 'hccl'
    assert args.config is not None, 'please provide a config file'
    cfg = load_yaml(args.config)
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
    if args.inference:
        # sequence_parallel is not supported in inference
        if 'kwargs' in cfg['model']:
            if 'sequence_parallel' in cfg['model']['kwargs']:
                cfg['model']['kwargs']['sequence_parallel'] = False
        runner = MegatronRunner(args, cfg, training=False, base_type='infer')
        runner.generate()
    else:
        # runner = BaseRunner(args, cfg, training=True, base_type='train')
        runner = MegatronRunner(args, cfg, training=True, base_type='train')
        runner.train()


if __name__ == "__main__":
    main()
