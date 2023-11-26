import math
from functools import partial

from torch.optim.lr_scheduler import LambdaLR

from llm.utils.general.log_helper import default_logger as logger
from llm.utils.general.registry_factory import LR_REGISTRY


class BaseAnnealingLR(object):
    """Anneals the learning rate."""

    def __init__(self,
                 optimizer,
                 max_lr,
                 min_lr,
                 warmup_steps,
                 decay_steps,
                 decay_style,
                 decay_tokens=None,
                 use_checkpoint_lr_scheduler=True,
                 override_lr_scheduler=False):
        # Class values.
        self.optimizer = optimizer

        self.max_lr = float(max_lr)
        self.min_lr = min_lr
        assert self.min_lr >= 0.0
        assert self.max_lr >= self.min_lr

        self.warmup_steps = warmup_steps
        self.num_steps = 0
        self.decay_steps = decay_steps
        assert self.decay_steps > 0
        assert self.warmup_steps < self.decay_steps

        self.decay_tokens = decay_tokens
        self.num_tokens = 0
        self.warmup_tokens = 0

        self.decay_style = decay_style

        self.override_lr_scheduler = override_lr_scheduler
        self.use_checkpoint_lr_scheduler = use_checkpoint_lr_scheduler
        if self.override_lr_scheduler:
            assert not self.use_checkpoint_lr_scheduler, 'both override and '\
                'use-checkpoint are set.'

    def state_dict(self):
        state_dict = {
            'max_lr': self.max_lr,
            'warmup_steps': self.warmup_steps,
            'num_steps': self.num_steps,
            'warmup_tokens': self.warmup_tokens,
            'num_tokens': self.num_tokens,
            'decay_style': self.decay_style,
            'decay_steps': self.decay_steps,
            'min_lr': self.min_lr
        }
        return state_dict

    def _check_and_set(self, cls_value, sd_value, name):
        """Auxiliary function for checking the values in the checkpoint and
        setting them."""
        if self.override_lr_scheduler:
            logger.info(' > overriding {} value to {}'.format(name, cls_value))
            return cls_value

        if not self.use_checkpoint_lr_scheduler:
            assert cls_value == sd_value, \
                f'AnnealingLR: class input value {cls_value} and checkpoint' \
                f'value {sd_value} for {name} do not match'
        logger.info(' > using checkpoint value {} for {}'.format(sd_value, name))
        return sd_value

    def load_state_dict(self, sd):
        if 'start_lr' in sd:
            max_lr_ = sd['start_lr']
        else:
            max_lr_ = sd['max_lr']
        self.max_lr = self._check_and_set(self.max_lr, max_lr_,
                                          'learning rate')

        self.min_lr = self._check_and_set(self.min_lr, sd['min_lr'],
                                          'minimum learning rate')

        if 'warmup_iter' in sd:
            warmup_steps_ = sd['warmup_iter']
        else:
            warmup_steps_ = sd['warmup_steps']
        self.warmup_steps = self._check_and_set(self.warmup_steps,
                                                warmup_steps_,
                                                'warmup iterations')

        if 'end_iter' in sd:
            decay_steps_ = sd['end_iter']
        else:
            decay_steps_ = sd['decay_steps']
        self.decay_steps = self._check_and_set(self.decay_steps, decay_steps_,
                                               'total number of iterations')
        self.decay_style = self._check_and_set(self.decay_style,
                                               sd['decay_style'],
                                               'decay style')

        if 'num_iters' in sd:
            num_steps = sd['num_iters']
        else:
            num_steps = sd['num_steps']
        if 'warmup_tokens' in sd:
            self.warmup_tokens = sd['warmup_tokens']
        if 'num_tokens' in sd:
            self.num_tokens = sd['num_tokens']
        self.step(num_steps)


@LR_REGISTRY.register('iter_base_annealing')
class IterBaseAnnealingLR(BaseAnnealingLR):
    def __init__(self,
                 optimizer,
                 max_lr,
                 min_lr,
                 decay_style,
                 warmup_steps=None,
                 decay_steps=None,
                 lr_decay_iters=None,
                 train_iters=None,
                 global_batch_size=None,
                 lr_warmup_fraction=None,
                 lr_warmup_iters=None,
                 decay_tokens=None,
                 use_checkpoint_lr_scheduler=True,
                 override_lr_scheduler=False):
        if decay_steps is not None:
            logger.warning('Setting decay_steps manually is not a recommended action, please be clear about what you are doing')        # noqa
        else:
            if lr_decay_iters is None:
                assert train_iters is not None
                lr_decay_iters = train_iters
            decay_steps = lr_decay_iters * global_batch_size
        if warmup_steps is not None:
            logger.warning('Setting warmup_steps manually is not a recommended action, please be clear about what you are doing')       # noqa
        else:
            if lr_warmup_fraction is not None:
                warmup_steps = lr_warmup_fraction * decay_steps
            else:
                warmup_steps = lr_warmup_iters * global_batch_size

        super(IterBaseAnnealingLR, self).__init__(optimizer,
                                                  max_lr,
                                                  min_lr,
                                                  warmup_steps,
                                                  decay_steps,
                                                  decay_style,
                                                  decay_tokens,
                                                  use_checkpoint_lr_scheduler,
                                                  override_lr_scheduler)
        # Set the learning rate
        self.step(0)
        logger.info('> learning rate decay style: {}'.format(self.decay_style))

    def get_lr(self):
        """Learning rate decay functions from:
              https://openreview.net/pdf?id=BJYwwY9ll pg. 4"""

        # Use linear warmup for the initial part.
        if self.warmup_steps > 0 and self.num_steps <= self.warmup_steps:
            return self.max_lr * float(self.num_steps) / float(self.warmup_steps)

        # If the learning rate is constant, just return the initial value.
        if self.decay_style == 'constant':
            return self.max_lr

        if self.num_steps > self.decay_steps:
            return self.min_lr

        num_steps_ = self.num_steps - self.warmup_steps
        decay_steps_ = self.decay_steps - self.warmup_steps
        decay_ratio = float(num_steps_) / float(decay_steps_)
        assert decay_ratio >= 0.0
        assert decay_ratio <= 1.0
        delta_lr = self.max_lr - self.min_lr

        if self.decay_style == 'linear':
            coeff = (1.0 - decay_ratio)
        elif self.decay_style == 'cosine':
            coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        else:
            raise Exception('{} decay style is not supported.'.format(
                self.decay_style))
        return self.min_lr + coeff * delta_lr

    def step(self, increment):
        self.num_steps += increment
        new_lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr


@LR_REGISTRY.register('sample_base_annealing')
class SampleBaseAnnealingLR(BaseAnnealingLR):
    def __init__(self,
                 optimizer,
                 max_lr,
                 min_lr,
                 decay_style,
                 init_consumed_tokens=0,
                 warmup_steps=None,
                 decay_steps=None,
                 lr_decay_samples=None,
                 train_samples=None,
                 lr_warmup_fraction=None,
                 lr_warmup_samples=None,
                 decay_tokens=None,
                 use_checkpoint_lr_scheduler=True,
                 override_lr_scheduler=False):
        if decay_steps is not None:
            logger.warning('Setting decay_steps manually is not a recommended action, please be clear about what you are doing')        # noqa
        else:
            if lr_decay_samples is None:
                assert train_samples is not None
                lr_decay_samples = train_samples
            decay_steps = lr_decay_samples
        if warmup_steps is not None:
            logger.warning('Setting warmup_steps manually is not a recommended action, please be clear about what you are doing')       # noqa
        else:
            if lr_warmup_fraction is not None:
                warmup_steps = lr_warmup_fraction * decay_steps
            else:
                warmup_steps = lr_warmup_samples
        self.consumed_train_tokens = init_consumed_tokens
        super(SampleBaseAnnealingLR, self).__init__(optimizer,
                                                    max_lr,
                                                    min_lr,
                                                    warmup_steps,
                                                    decay_steps,
                                                    decay_style,
                                                    decay_tokens,
                                                    use_checkpoint_lr_scheduler,
                                                    override_lr_scheduler)
        # Set the learning rate
        self.step(0)
        logger.info('> learning rate decay style: {}'.format(self.decay_style))

    def get_lr(self):
        """Learning rate decay functions from:
              https://openreview.net/pdf?id=BJYwwY9ll pg. 4"""

        # Use linear warmup for the initial part.
        if self.warmup_steps > 0 and self.num_steps <= self.warmup_steps:
            if self.num_steps == self.warmup_steps and \
                    self.decay_tokens is not None:
                self.warmup_tokens = self.num_tokens
            return self.max_lr * float(self.num_steps) / \
                float(self.warmup_steps)

        # If the learning rate is constant, just return the initial value.
        if self.decay_style == 'constant':
            return self.max_lr
        # token-based decay

        if self.num_tokens > self.decay_tokens:
            return self.min_lr
        num_tokens_ = self.num_tokens - self.warmup_tokens
        decay_tokens_ = self.decay_tokens - self.warmup_tokens
        decay_ratio = float(num_tokens_) / float(decay_tokens_)
        assert decay_ratio >= 0.0
        assert decay_ratio <= 1.0
        delta_lr = self.max_lr - self.min_lr

        if self.decay_style == 'linear':
            coeff = (1.0 - decay_ratio)
        elif self.decay_style == 'cosine':
            coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        else:
            raise Exception('{} decay style is not supported.'.format(
                self.decay_style))
        return self.min_lr + coeff * delta_lr

    def step(self, increment, token_num=None):
        """Set lr for all parameters groups."""
        if token_num is None:
            token_num = self.consumed_train_tokens
        self.num_tokens = token_num
        self.num_steps += increment
        new_lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr


@LR_REGISTRY.register('hf_cosine')
class HFCosinLR(LambdaLR):
    def __init__(self, **kwargs):
        last_epoch = kwargs.get("last_epoch", -1)
        warmup_steps = kwargs.pop("warmup_steps")
        training_steps = kwargs.pop("training_steps")

        lr_lambda = partial(self._get_lr_lambda,
                            warmup_steps=warmup_steps,
                            training_steps=training_steps)
        super(HFCosinLR, self).__init__(optimizer=kwargs["optimizer"],
                                        lr_lambda=lr_lambda,
                                        last_epoch=last_epoch)

    def _get_lr_lambda(self, curr_steps, *, warmup_steps, training_steps):
        if curr_steps < warmup_steps:
            return float(curr_steps) / float(max(1, warmup_steps))

        progress = float(curr_steps - warmup_steps) / float(max(1, training_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 0.5 * 2.0 * progress)))


def build_learning_rate_scheduler(cfg_lr, optimizer):
    cfg_lr['kwargs'].update({'optimizer': optimizer})
    return LR_REGISTRY.build(cfg_lr)
