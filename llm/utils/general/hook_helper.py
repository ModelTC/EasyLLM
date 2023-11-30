import sys
import copy
import time
import weakref
import torch
import datetime
from llm.utils.general.log_helper import default_logger as logger
from llm.utils.general.registry_factory import HOOK_REGISTRY
from llm.utils.env import dist_env
from llm.utils.model.ckpt_helper import save_checkpoint
from .utils import Timers, report_memory


class Hook(object):
    """ A mechanism that decouples functions like log, time and visualization
    from training code.
    """

    def __init__(self, runner):
        """
        Arguments:
            - runner (:obj:`Runner`): used as to accecss other variables
        """
        self.runner_ref = weakref.ref(runner)

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_train_iter(self, cur_iter, input={}):
        pass

    def after_train_iter(self, cur_iter, output={}):
        pass

    def __call__(self, hook_type, *args, **kwargs):
        fn = getattr(self, hook_type)
        return fn(*args, **kwargs)


@HOOK_REGISTRY.register('train_val_logger')
class TrainValLoggerHook(Hook):
    def __init__(self,
                 runner,
                 log_interval=1,
                 report_memory_interval=10,
                 log_params_norm=False,
                 tensorboard=True,
                 log_dir='./log'):
        super(TrainValLoggerHook, self).__init__(runner)
        self.timers = Timers()
        self.log_interval = log_interval
        self.log_params_norm = log_params_norm
        self.report_memory_interval = report_memory_interval

        # train_total_loss
        self.total_loss_dict = {}
        # Advanced, and Nan iterations.
        self.advanced_iters_key = 'advanced iterations'
        self.nan_iters_key = 'nan iterations'
        # model settings
        self.model_args = {}
        # env settings
        self.fp16 = runner.config['runtime'].get('fp16', False)
        self.bf16 = runner.config['runtime'].get('bf16', False)
        self.deepspeed = runner.deepspeed
        if tensorboard:
            self.tensorboard_writer = None
            if dist_env.get_global_rank() == 0:
                from tensorboardX import SummaryWriter
                self.tensorboard_writer = SummaryWriter(log_dir=log_dir)
        else:
            self.tensorboard_writer = None

    def before_train(self):
        """ Make sense after fetching data and before forwarding
        Arguments:
          - cur_iter (int): current iteration in training
          - input (dict): input to model
        """
        runner = self.runner_ref()
        self.model_args['micro_batch_size'] = runner.model.transformer_layer_params['micro_batch_size']
        self.model_args['seq_len'] = runner.model.transformer_layer_params['seq_length']
        self.model_args['hidden_size'] = runner.model.transformer_layer_params['hidden_size']
        self.model_args['num_layers'] = runner.model.model_kwargs['num_layers']
        self.model_args['vocab_size'] = runner.model.word_embedings_params['vocab_size']
        self.model_args['checkpoint_activations'] = runner.model.model_kwargs['checkpoint_activations']
        self.model_args['glu_activation'] = runner.model.transformer_layer_params['glu_activation']
        self.timers('train-iter-time').start()

    def after_train_iter(self, cur_iter, output={}):
        """ Make sense after forwarding
        Arguments:
          - cur_iter (int): current iteration in training
          - output (dict): output of the model
        """
        # Advanced iterations.
        self.total_loss_dict[self.advanced_iters_key] = self.total_loss_dict.get(self.advanced_iters_key, 0) + 1
        # Update losses and set nan iterations
        got_nan = False
        for key in output:
            if '_loss' in key:
                self.total_loss_dict[key] = self.total_loss_dict.get(key, torch.cuda.FloatTensor([0.0])) + output[key]
                value = output[key].float().sum().item()
                is_nan = value == float('inf') or \
                    value == -float('inf') or \
                    value != value
                got_nan = got_nan or is_nan
        self.total_loss_dict[self.nan_iters_key] = self.total_loss_dict.get(self.nan_iters_key, 0) + int(got_nan)

        runner = self.runner_ref()
        train_iters = runner.total_train_iters       # total iters
        consumed_train_samples = runner.consumed_train_samples
        consumed_train_tokens = runner.consumed_train_tokens
        learning_rate = runner.optimizer.param_groups[0]['lr']

        loss_scale = None
        if self.fp16:
            if self.deepspeed:
                loss_scale = runner.model.optimizer.cur_scale
            else:
                loss_scale = runner.optimizer.get_loss_scale().item()

        params_norm = None

        data_parallel_size = dist_env.get_data_parallel_world_size()

        batch_size = self.model_args['micro_batch_size'] * data_parallel_size \
            * runner.num_microbatches_calculator.get()

        total_iterations = self.total_loss_dict[self.advanced_iters_key]

        if (cur_iter + 1) % self.log_interval == 0:
            elapsed_time = self.timers('train-iter-time').elapsed()
            elapsed_time_per_iteration = elapsed_time / total_iterations
            eta_seconds = elapsed_time_per_iteration * (train_iters - cur_iter - 1)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            # Compute throughput.
            samples_per_sec = batch_size / elapsed_time_per_iteration
            # samples_per_sec_per_replica = samples_per_sec / data_parallel_size
            # tokens_per_sec = samples_per_sec * seq_len
            # tokens_per_sec_per_replica = tokens_per_sec / data_parallel_size

            # General TFLOPs formula (borrowed from Equation 3 in Section 5.1 of
            # https://arxiv.org/pdf/2104.04473.pdf).
            # The factor of 4 is when used with activation check-pointing,
            # otherwise it will be 3, but for 200B model, activation check-pointing will always be on.
            checkpoint_activations_factor = 4 if self.model_args['checkpoint_activations'] else 3
            # GLU activations double the hidden states in the upscaling feed-forward in each transformer layer
            # This leads to 16bsh^2 instead of 8bsh^2 per first feed-forward layer in MLP,
            # thus we increase the coefficient by 8.
            # Refer to https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/283#issue-1260805063.
            coefficient = 32 if self.model_args['glu_activation'] else 24
            flops_per_iteration = (coefficient * checkpoint_activations_factor * batch_size * self.model_args['seq_len'] * self.model_args['num_layers'] * (self.model_args['hidden_size']**2)) * (1. + (self.model_args['seq_len'] / (6. * self.model_args['hidden_size'])) + (self.model_args['vocab_size'] / (16. * self.model_args['num_layers'] * self.model_args['hidden_size'])))        # noqa
            tflops = flops_per_iteration / (elapsed_time_per_iteration * torch.distributed.get_world_size() * (10**12))

            log_string = ' iteration {:8d}/{:8d} |'.format(cur_iter + 1, train_iters)
            log_string += ' consumed samples: {:12d} |'.format(consumed_train_samples)
            log_string += ' consumed tokens: {:12d} |'.format(consumed_train_tokens)
            log_string += ' elapsed time per iteration (s): {:.2f} |'.format(elapsed_time_per_iteration)
            log_string += ' learning rate: {:.3E} |'.format(learning_rate)
            log_string += ' global batch size: {:5d} |'.format(batch_size)
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar("train/lr", learning_rate, cur_iter)
            for key in self.total_loss_dict:
                if key not in [self.advanced_iters_key, self.nan_iters_key]:
                    avg = self.total_loss_dict[key].item() / float(max(1, self.total_loss_dict[self.advanced_iters_key]))       # noqa
                    if avg > 0.0:
                        log_string += ' {}: {:.6E} |'.format(key, avg)
                        if self.tensorboard_writer is not None:
                            self.tensorboard_writer.add_scalar(f'train/{key}', avg, cur_iter)
                    self.total_loss_dict[key] = torch.cuda.FloatTensor([0.0])
            if self.fp16:
                log_string += ' loss scale: {:.1f} |'.format(loss_scale)
            grad_norm = runner.model.get_global_grad_norm()
            if grad_norm is not None:
                log_string += ' grad norm: {:.3f} |'.format(grad_norm)
            if params_norm is not None:
                log_string += ' params norm: {:.3f} |'.format(params_norm)
            log_string += ' number of nan iterations: {:3d} |'.format(
                self.total_loss_dict[self.nan_iters_key])
            log_string += ' samples per second: {:.3f} |'.format(samples_per_sec)
            log_string += ' TFLOPs: {:.2f} |'.format(tflops)
            log_string += f' ETA Time: {eta_string}'
            logger.info(log_string)

            self.total_loss_dict[self.advanced_iters_key] = 0
            self.total_loss_dict[self.nan_iters_key] = 0
            if (cur_iter % self.report_memory_interval == 0) and learning_rate > 0.:
                # Report memory after optimizer state has been initialized.
                report_memory('(after {} iterations)'.format(cur_iter + 1))


@HOOK_REGISTRY.register('early_exit')
class EarlyExitHook(Hook):
    def __init__(self, runner, exit_duration_in_mins=None, exit_interval=None):
        super(EarlyExitHook, self).__init__(runner)
        self.exit_duration_in_mins = exit_duration_in_mins
        self.exit_interval = exit_interval

    def after_train_iter(self, cur_iter, output={}):
        runner = self.runner_ref()
        # Exiting based on duration
        if self.exit_duration_in_mins is not None:
            train_time = (time.time() - runner.start_time) / 60.0
            done_cuda = torch.cuda.IntTensor([train_time > self.exit_duration_in_mins])
            torch.distributed.all_reduce(done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                save_checkpoint((cur_iter + 1), runner.consumed_train_samples, runner.consumed_train_tokens,
                                runner.model, runner.config['saver'], runner.lora_mode, runner.cfg_lora)
                logger.info('Exiting program after {} minutes'.format(train_time))
                sys.exit()

        # Exiting based on iterations
        if self.exit_interval and (cur_iter + 1) % self.exit_interval == 0:
            save_checkpoint((cur_iter + 1), runner.consumed_train_samples, runner.consumed_train_tokens,
                            runner.model, runner.config['saver'], runner.lora_mode, runner.cfg_lora)
            logger.info('Exiting program at iteration {}'.format(cur_iter + 1))
            sys.exit()


@HOOK_REGISTRY.register('empty_cache')
class EmptyCacheHook(Hook):
    def __init__(self, runner, freq=1):
        super(EmptyCacheHook, self).__init__(runner)
        self.freq = freq

    def after_train_iter(self, cur_iter, output={}):
        if cur_iter % self.freq == 0 and cur_iter > 0:
            torch.cuda.empty_cache()
            torch.cuda.memory.reset_peak_memory_stats()


class ComposeHook(object):
    """A interface that compose several hooks into one
    """

    def __init__(self, hooks):
        self.hooks = hooks

    def __call__(self, *args, **kwargs):
        results = [hook(*args, **kwargs) for hook in self.hooks]
        return results


def build_hooks(runner, cfg_list, is_train=True, add_log_if_not_exists=True):

    def add_log_hook(cfg_hooks):
        exists = any(['train_val_logger' in cfg['type'] for cfg in cfg_hooks])
        if not exists:
            cfg_hooks.insert(0, {
                'type': 'train_val_logger',
                'kwargs': {}
            })
        return cfg_hooks

    def build_single_hook(cfg):
        cfg = copy.deepcopy(cfg)
        kwargs = cfg.setdefault('kwargs', {})
        kwargs['runner'] = runner
        return HOOK_REGISTRY.build(cfg)

    if add_log_if_not_exists:
        cfg_list = add_log_hook(cfg_list)
    if not is_train:
        # TODO: add remove hooks
        pass

    hooks = [build_single_hook(cfg) for cfg in cfg_list]
    return ComposeHook(hooks)
