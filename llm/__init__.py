import os


def accelerator_init():
    if os.environ.get('ACCELERATOR_BACKEND', "CUDA") == 'CUDA':
        pass
    elif os.environ.get('ACCELERATOR_BACKEND') == 'TORCH_NPU':
        try:
            import torch_npu  # noqa
            from torch_npu.contrib import transfer_to_npu  # noqa
        except Exception as e:  # noqa
            print(e, "Warning: You did not install torch_npu")
    elif os.environ.get('ACCELERATOR_BACKEND') == 'DEEPLINK_DIPU':
        try:
            import torch_dipu  # noqa
        except Exception as e:  # noqa
            print(e, "Warning: You did not install dipu")  # noqa
    else:
        print("no backend support")


accelerator_init()

from .data import *  # noqa
from .models import *  # noqa
from .runners import *  # noqa
from .utils import *  # noqa


__version__ = '1.0.0'
