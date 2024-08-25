# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import dataclasses
import os
import copy
import json
import time
import torch
from functools import partial

from typing import Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
# from megatron.core import mpu

from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    # get_llama_layer_with_transformer_engine_spec,
    # get_llama_layer_local_spec
)
from llm.data.nlp_dataset import IGNORE_INDEX

from llm.utils.general.log_helper import default_logger as logger
if os.getenv("DIST_BACKEND", "easyllm") == "easyllm":
    from llm.utils.env import dist_env
elif os.getenv("DIST_BACKEND", "easyllm") == "megatron":
    from megatron.core import mpu as dist_env

from llm.utils.general.parser_helper import parse_args
from megatron.training.arguments import parse_args as parse_args_mg
from llm.utils.general.yaml_loader import load_yaml
from llm.utils.env import set_random_seed
from megatron.training.training import setup_model_and_optimizer, build_train_valid_test_data_iterators, train, train_step, num_floating_point_operations, training_log, save_checkpoint_and_time, get_model, get_optimizer_param_scheduler
from megatron.training.initialize import initialize_megatron, set_jit_fusion_options
from megatron.core.utils import get_model_config

from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.num_microbatches_calculator import get_num_microbatches, update_num_microbatches, get_current_global_batch_size, get_current_running_global_batch_size

from llm.utils.general.microbatches import build_num_microbatches_calculator
from llm.data import build_tokenizer, build_data_iterator
from llm.utils.general.utils import get_train_iters
from megatron.training.utils import unwrap_model
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
# from megatron.training.checkpointing import load_checkpoint, load_checkpoint_hf
from llm.utils.model.megatron_checkpointing import load_checkpoint

from llm.utils.model.megatron_model_provider import model_provider
from llm.utils.model.megatron_utils import forward_step
from megatron.core.pipeline_parallel import get_forward_backward_func


_TRAIN_START_TIME = time.time()


class MegatronRunner(object):
    def __init__(self, args, cfg=None, training=True, base_type='train'):
        self.args = args
        self.config = copy.deepcopy(cfg)
        # mapping yaml args to megatron args
        self.yaml2args()
        self.training = training
        self.base_type = base_type
        self.build()
        # self.display_train_info(cfg)

    def build_model_cfg(self):
        from llm.models.mg_models.llama.llama import _LLAMA_MODELS
        from llm.models.mg_models.base_modules.utils import check_keys_mapping
        from llm.models.mg_models.llama.default_cfg import update_model_cfg

        cfg_model = self.config['model']
        model_type = cfg_model['type']
        if 'kwargs' not in cfg_model:
            cfg_model['kwargs'] = {}
        cfg_model['kwargs'].update({"fp16": self.config['runtime'].get('fp16', False),
                                    "bf16": self.config['runtime'].get('bf16', False)})       # noqa
        is_qkv_pack = cfg_model['kwargs'].get("transformer_layer_params", {}).get("qkv_pack", False)
        pretrain_type = self.config['loader'].get("pretrain_type", "llama")
        if is_qkv_pack:
            assert "pack" in pretrain_type, "qkv_pack must load the model in pack type. currently support llama_pack and internlm2_pack!"
        else:
            assert "pack" not in pretrain_type, "You load the model in pack type, but do not set qkv_pack as True!"
        if model_type != "llama_custom":
            cfg_item = _LLAMA_MODELS[model_type]
            check_keys_mapping(cfg_item, cfg_model)
            cfg_model.update(cfg_item)
        cfg_model = update_model_cfg(cfg_model['kwargs'])
        return cfg_model

    def yaml2args(self, extra_args_provider=None, ignore_unknown_args=True, args_defaults=dict()):
        from megatron.training.arguments import parse_args, validate_args
        from megatron.training.yaml_arguments import validate_yaml
        from megatron.training.global_vars import set_global_variables
        cfg_model = self.build_model_cfg()
        args = parse_args_mg(extra_args_provider, ignore_unknown_args)

        args.num_layers = cfg_model['num_layers']
        args.hidden_size = cfg_model['hidden_size']
        args.num_attention_heads = cfg_model['num_attention_heads']
        args.max_position_embeddings = cfg_model["word_embedings_params"].get("max_position_embeddings")
        args.seq_length = self.config["tokenization"]["kwargs"].get("max_seq_length", 4096)
        if args.max_position_embeddings == None or args.seq_length > args.max_position_embeddings:
            args.max_position_embeddings = args.seq_length
        args.micro_batch_size = self.config['data']['train']['micro_batch_size']        
        # if args.yaml_cfg is not None:
        #     args = validate_yaml(args, args_defaults)
        # else:
        #     validate_args(args, args_defaults)
        # set_global_variables(args, build_tokenizer=False)

        logger.info("Mapping yaml args to megatron args.")
        # args = get_args()
        # model args
        # cfg_model = self.build_model_cfg()
        # args.num_layers = cfg_model['kwargs']['num_layers']
        # args.hidden_size = cfg_model['hidden_size']
        # args.num_attention_heads = cfg_model['num_attention_heads']
        args.ffn_hidden_size = cfg_model['intermediate_size']
        if cfg_model['transformer_layer_params'].get("glu_activation", "silu") == "silu":
            args.swiglu = True
        args.use_rotary_position_embeddings = True
        args.hidden_dropout = cfg_model['transformer_layer_params']['hidden_dropout']
        args.attention_dropout = cfg_model['transformer_layer_params']['attention_dropout']
        args.add_bias_linear = False
        args.norm_epsilon = cfg_model['layer_norm_params']["kwargs"]['eps']
        if cfg_model['layer_norm_params']['type'] == "rms_norm":
            args.normalization = "RMSNorm"
        if cfg_model['num_kv_attention_heads'] != cfg_model['num_attention_heads']:
            args.group_query_attention = True
            args.num_query_groups = cfg_model['num_kv_attention_heads']
        position_embedding_kwargs = cfg_model['transformer_layer_params']['position_embedding_kwargs']
        args.init_method_std = cfg_model['transformer_layer_params']['initializer']['kwargs']['sigma']
        args.rotary_base = position_embedding_kwargs.get('base', 10000)
        args.use_flash_attn = cfg_model.get('use_flash_attn', True)
        args.sequence_parallel = cfg_model.get('sequence_parallel', False)
        # training args
        args.global_batch_size = self.config['data']['train']['global_batch_size']
        args.train_iters = self.config['trainer'].get('train_iters', 0)
        args.train_epoch = self.config['trainer'].get('train_iters', -1)
        args.weight_decay = self.config['trainer']['optimizer']['kwargs'].get('weight_decay', 0.1)
        args.adam_beta1, args.adam_beta2 = self.config['trainer']['optimizer']['kwargs'].get('betas', [0.9, 0.95])
        args.adam_eps = self.config['trainer']['optimizer']['kwargs'].get('eps', 1.0e-8)
        args.clip_grad = self.config['deepspeed']['config'].get('gradient_clipping', 1.0)
        args.bf16 = self.config['runtime']['bf16']
        args.lr = self.config['trainer']['optimizer']['kwargs']['lr']
        args.lr_decay_style = self.config['trainer']['lr_scheduler']['kwargs']['decay_style']
        args.min_lr = self.config['trainer']['lr_scheduler']['kwargs'].get('min_lr', 1.0e-6)
        args.lr_warmup_iters = self.config['trainer']['lr_scheduler']['kwargs']['lr_warmup_iters']
        args.lr_decay_iters = self.config['trainer']['lr_scheduler']['kwargs']['lr_decay_iters']
        args.overlap_grad_reduce = True
        args.seed = self.config['runtime']['seed']
        args.ckpt_format = 'torch'
        # model parallel args
        args.tensor_model_parallel_size = self.config['runtime']['tensor_model_parallel_size']
        args.pipeline_model_parallel_size = self.config['runtime']['pipeline_model_parallel_size']
        args.context_parallel_size = self.config['runtime'].get('context_parallel_size', 1)
        # data args
        # args.data_path
        for cfg_hook in self.config['hooks']:
            if cfg_hook['type'] == "train_val_logger":
                args.log_interval = cfg_hook['kwargs'].get('log_interval')
        args.save_interval = self.config['saver']['save_interval']
        args.save = self.config['saver']['save_path']
        args.variable_seq_lengths = self.config['runtime'].get('dynamic', False)
        # args.model_parallel.variable_seq_lengths = self.config['runtime'].get('dynamic', False)

        if args.yaml_cfg is not None:
            args = validate_yaml(args, args_defaults)
        else:
            validate_args(args, args_defaults)
        set_global_variables(args, build_tokenizer=False)

    def build(self):
        self.set_param_components()
        self.build_env()
        self.build_num_microbatches_calculator()
        self.build_tokenizer()
        self.build_model()
        self.build_data_engine()
        self.build_trainer()
        self.load_checkpoint()

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
        from llm.utils.general.yaml_loader import load_yaml
        args = get_args()
        timers = get_timers()
        unwrapped_model = unwrap_model(self.model)

        cfg_loader = self.config["loader"]
        # if args.load is not None or args.pretrained_checkpoint is not None:
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

        # get model without FP16 and/or DDP wrappers
        if args.iteration == 0 and len(unwrapped_model) == 1 \
            and hasattr(unwrapped_model[0], 'init_state_dict_from_bert'):
            print_rank_0("Initializing ICT from pretrained BERT model")
            unwrapped_model[0].init_state_dict_from_bert()
            if args.fp16:
                self.optimizer.reload_model_params()

    def build_model(self, model_type=ModelType.encoder_or_decoder):
        self.model = get_model(model_provider, model_type)

    def build_tokenizer(self):
        self.tokenizer = build_tokenizer(self.config['tokenizer'])

    def set_train_iters(self, train_iters):
        self.total_train_iters = get_train_iters(self.num_microbatches_calculator,
                                                 train_iters,
                                                 self.config['trainer'].get('train_samples', None))

    def build_data_engine(self):
        cfg_data = self.config['data']
        data_types = cfg_data.get('data_types', ['train', 'test'])
        self.data_iterators = {}
        self.batch_pipe_func = {}
        for data_type in data_types:
            assert data_type in ['train', 'valid', 'test', 'infer'], 'data type only support train, valid, test, and infer'       # noqa
            # self.batch_pipe_func[data_type] = build_batch_pipe_fn(cfg_data[data_type]['batch_pipe'], self.tokenizer)
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
            self.set_train_iters(train_iters)

        args = get_args()
        args.do_train = True
        args.do_valid = False
        args.do_test = False

    def build_trainer(self, no_wd_decay_cond=None, scale_lr_cond=None, lr_mult=1.0):
        args = get_args()
        timers = get_timers()
        unwrapped_model = unwrap_model(self.model)
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

    def forward_step(
        self,
        config
    ):
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
            num_microbatches=self.num_microbatches_calculator.get(), # get_num_microbatches(),
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False)
        
        # Empty unused memory.
        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache()

        # Vision gradients.
        if getattr(args, 'vision_pretraining', False) and args.vision_pretraining_type == "dino":
            unwrapped_model = unwrap_model(model[0])
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
                numerator = 0
                denominator = 0
                for x in losses_reduced:
                    val = x[key]
                    # there is one dict per microbatch. in new reporting, we average
                    # over the total number of tokens across the global batch.
                    if isinstance(val, tuple) or isinstance(val, list):
                        # numerator += val[0]
                        # denominator += val[1]
                        loss_reduced[key] += val[0] / val[1]
                    else:
                        # legacy behavior. we average over the number of microbatches,
                        # and so the denominator is 1.
                        # numerator += val
                        # denominator += 1
                        loss_reduced[key] += val
                # loss_reduced[key] = numerator / denominator
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

        config = get_model_config(self.model[0])

        # Context used for persisting some state between checkpoint saves.
        checkpointing_context = {}

        if not args.skip_train:
            print_rank_0('training ...')

            if args.dataloader_type == 'cyclic' and args.retro_project_dir:
                assert args.retro_cyclic_train_iters is not None
                args.train_iters = args.retro_cyclic_train_iters
                print_rank_0("retro cyclic train iters : %d" % args.train_iters)

            iteration = 0
            if args.do_train and args.train_iters > 0:
                # Turn on training mode which enables dropout.
                for model_module in self.model:
                    model_module.train()

                # Tracking loss.
                total_loss_dict = {}
                # Iterations.
                iteration = args.iteration
                num_floating_point_operations_so_far = args.num_floating_point_operations_so_far
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
                timers('interval-time', log_level=0).start(barrier=True)

                report_memory_flag = True
                total_flops = 0.0
                while iteration < args.train_iters:
                    # Update number of microbatches first without consistency check to decide if a
                    # checkpoint should be saved. If the number of microbatches is different
                    # from the previous iteration, save a checkpoint. Then run consistency check
                    # to make sure training configuration is still valid.
                    self.num_microbatches_calculator.update(self.consumed_train_samples, True)

                    args.curr_iteration = iteration
                    loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = \
                        self.forward_step(config)
                        # self.forward_step(forward_step,
                        #                   self.data_iterators['train'],
                        #                   self.model,
                        #                   self.optimizer,
                        #                   self.lr_scheduler,
                        #                   config)
                        
                    iteration += 1
                    batch_size = dist_env.get_data_parallel_world_size() * \
                                 args.micro_batch_size * \
                                 self.num_microbatches_calculator.get()
                                 # get_num_microbatches()
                    args.consumed_train_samples += batch_size
                    num_skipped_samples_in_batch = (get_current_global_batch_size() -
                                                    get_current_running_global_batch_size())
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
                    if args.log_params_norm:
                        params_norm = calc_params_l2_norm(self.model)

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
                    saved_checkpoint = False
                    if args.exit_signal_handler:
                        signal_handler = get_signal_handler()
                        if any(signal_handler.signals_received()):
                            save_checkpoint_and_time(iteration, self.model, self.optimizer,
                                                     self.lr_scheduler,
                                                     num_floating_point_operations_so_far,
                                                     checkpointing_context, train_data_iterator=self.data_iterators['train'])
                            print_datetime('exiting program after receiving SIGTERM.')
                            exit = True
                            break

                    if args.save and args.save_interval and \
                       iteration % args.save_interval == 0:
                        save_checkpoint_and_time(iteration, self.model, self.optimizer,
                                                 self.lr_scheduler,
                                                 num_floating_point_operations_so_far,
                                                 checkpointing_context, train_data_iterator=self.data_iterators['train'])
                        saved_checkpoint = True

                    elif args.save and args.non_persistent_save_interval and \
                       iteration % args.non_persistent_save_interval == 0:
                        timers('interval-time').stop()
                        save_checkpoint_and_time(iteration, self.model, self.optimizer,
                                                 self.lr_scheduler,
                                                 num_floating_point_operations_so_far,
                                                 non_persistent_ckpt=True, train_data_iterator=self.data_iterators['train'])
                        saved_checkpoint = True
                        timers('interval-time', log_level=0).start(barrier=True)

            # print_datetime('after training is done')
            if args.save and iteration != 0 and iteration % args.save_interval != 0:
                save_checkpoint(iteration, self.model, self.optimizer, self.lr_scheduler,
                                num_floating_point_operations_so_far, checkpointing_context,
                                train_data_iterator=self.data_iterators['train'])
        else:
            print_rank_0('skipping training (--skip-train is on) ...')

            iteration = args.iteration


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
