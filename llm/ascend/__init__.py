import logging

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    import deepspeed_npu
except Exception as e:
    logging.warning("Warning: You did not install torch_npu or deepspeed_npu")