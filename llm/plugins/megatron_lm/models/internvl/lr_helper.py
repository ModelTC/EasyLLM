import math
from functools import partial

from torch.optim.lr_scheduler import LambdaLR

from llm.utils.general.registry_factory import LR_REGISTRY


@LR_REGISTRY.register('cosine_internvl')
class InternVLHFCosinLR(LambdaLR):
    def __init__(self, **kwargs):
        last_epoch = kwargs.get("last_epoch", -1)
        warmup_steps = kwargs.pop("warmup_steps", 1)
        warmup_ratio = kwargs.pop("warmup_ratio", 0.01)
        training_steps = kwargs.pop("training_steps", 100)
        warmup_steps = math.ceil(training_steps * warmup_ratio) if training_steps * warmup_ratio > warmup_steps else warmup_steps

        lr_lambda = partial(self._get_lr_lambda,
                            warmup_steps=warmup_steps,
                            training_steps=training_steps)
        super(InternVLHFCosinLR, self).__init__(optimizer=kwargs["optimizer"],
                                                lr_lambda=lr_lambda,
                                                last_epoch=last_epoch)
        print(f'>>>> InternVLHFCosinLR.warmup_steps={warmup_steps}')

    def _get_lr_lambda(self, curr_steps, *, warmup_steps, training_steps):
        if curr_steps < warmup_steps:
            alpha = float(curr_steps) / float(max(1, warmup_steps))
        else:
            progress = float(curr_steps - warmup_steps) / float(max(1, training_steps - warmup_steps))
            alpha = max(0.0, 0.5 * (1.0 + math.cos(math.pi * 0.5 * 2.0 * progress)))
        return alpha
