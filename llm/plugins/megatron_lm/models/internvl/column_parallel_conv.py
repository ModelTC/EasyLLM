import torch
from torch import nn

from megatron.core import parallel_state, tensor_parallel

class ColumnParallelConv2d(nn.Conv2d):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size=3,
                 stride=1,
                 bias=False,
                 dilation=1,
                 groups=1,
                 padding=1,
                 gather_output=False,
                 init_method='xavier',
                 params_dtype=torch.half):
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        tp_output_channel = output_channel // tp_size
        super().__init__(input_channel,
                         tp_output_channel,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         bias=bias,
                         dilation=dilation,
                         groups=groups,
                         dtype=params_dtype)
        self.gather_output = gather_output
        self.init_method = init_method
        self._init_weights()

    def _init_weights(self):
        if self.init_method == 'xavier':
            nn.init.xavier_normal_(self.weight.data)
            if self.bias is not None:
                self.bias.data.zero_()

    def forward(self, x):
        x = super().forward(x)
        if self.gather_output:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = tensor_parallel.gather_from_tensor_model_parallel_region(x)
            x = x.permute(0, 3, 1, 2).contiguous()
        return x
