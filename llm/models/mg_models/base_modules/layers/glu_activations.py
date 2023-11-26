import torch
from torch import nn
from torch.nn import functional as F
from functools import wraps

from llm.utils.general.log_helper import default_logger as logger


def log_debug_usage(logger, msg: str):
    def log_debug_usage_(func):
        """Helper function in order to log a message when using a function for the first time"""
        func.__logged_message__ = False

        @wraps(func)
        def wrapped(*args, **kwargs):
            if func.__logged_message__ is False:
                logger.debug(msg)
                func.__logged_message__ = True
            return func(*args, **kwargs)

        return wrapped
    return log_debug_usage_


class _GLUBaseModule(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn

    def forward(self, x):
        # dim=-1 breaks in jit for pt<1.10
        x1, x2 = x.chunk(2, dim=(x.ndim - 1))
        return x1 * self.activation_fn(x2)


class LiGLU(_GLUBaseModule):
    def __init__(self):
        super().__init__(nn.Identity())


class GEGLU(_GLUBaseModule):
    def __init__(self):
        super().__init__(F.gelu)


class ReGLU(_GLUBaseModule):
    def __init__(self):
        super().__init__(F.relu)


class SwiGLU(_GLUBaseModule):
    def __init__(self):
        super().__init__(F.silu)


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.silu(x)


# TODO: Try to remove it
liglu = log_debug_usage(logger, "Using GLU activation: LiGLU.")(torch.jit.script(LiGLU()))
geglu = log_debug_usage(logger, "Using GLU activation: GELU.")(torch.jit.script(GEGLU()))
reglu = log_debug_usage(logger, "Using GLU activation: ReGLU.")(torch.jit.script(ReGLU()))
swiglu = log_debug_usage(logger, "Using GLU activation: SwiGLU.")(torch.jit.script(SwiGLU()))
silu = log_debug_usage(logger, "Using GLU activation: SiLU.")(torch.jit.script(SiLU()))


GLU_ACTIVATIONS = {
    "geglu": geglu,
    "liglu": liglu,
    "reglu": reglu,
    "silu": silu,
    "swiglu": swiglu,
}
