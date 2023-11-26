import contextlib
from typing import Callable, Dict, Type

import torch

from transformers import Conv1D
from transformers.modeling_utils import no_init_weights


def _conv1d_init(self, nf, nx, device=None):
    super(Conv1D, self).__init__()
    self.nf = nf
    w = torch.empty(nx, nf, device=device)
    torch.nn.init.normal_(w, std=0.02)
    self.weight = torch.nn.Parameter(w)
    b = torch.empty(nf, device=device)
    torch.nn.init.zeros_(b)
    self.bias = torch.nn.Parameter(b)


_ORIGINAL_INITS: Dict[Type[torch.nn.Module], Callable] = {
    Conv1D: _conv1d_init,
    torch.nn.Linear: torch.nn.Linear.__init__,
    torch.nn.Embedding: torch.nn.Embedding.__init__,
    torch.nn.LayerNorm: torch.nn.LayerNorm.__init__,
}


def _get_fast_init(cls: Type[torch.nn.Module], device: torch.device):
    assert cls in _ORIGINAL_INITS

    def _fast_init(self, *args, **kwargs):
        # Same as torch.nn.utils.skip_init, excluding checks
        _ORIGINAL_INITS[cls](self, *args, **kwargs, device="meta")
        self.to_empty(device=device)

    return _fast_init


@contextlib.contextmanager
def fast_init(device: torch.device, init_weights: bool = False):
    """
    Avoid multiple slow initializations on cpu.
    """
    for cls in _ORIGINAL_INITS:
        cls.__init__ = _get_fast_init(cls, device)

    with contextlib.nullcontext() if init_weights else no_init_weights():
        yield

    for cls in _ORIGINAL_INITS:
        cls.__init__ = _ORIGINAL_INITS[cls]
