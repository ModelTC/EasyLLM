import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from llm.utils.env import dist_env
from llm.utils.model.initializer import set_tensor_model_parallel_attributes
from llm.utils.model.initializer import _initialize_affine_weight_gpu
from llm.utils.model.initializer import _initialize_affine_weight_cpu


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """

    def __init__(self, input_size, output_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False,
                 use_cpu_initialization=False,
                 params_dtype=torch.half,
                 sequence_parallel=False):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = bias
        self.gather_output = gather_output
        self.init_method = init_method
        self.stride = stride
        self.keep_master_weight_for_test = keep_master_weight_for_test
        self.use_cpu_initialization = use_cpu_initialization
        self.params_dtype = params_dtype
        # Divide the weight matrix along the last dimension.
        world_size = dist_env.get_tensor_model_parallel_world_size()
        assert output_size % world_size == 0, '{} is not divisible by {}'.format(
            output_size, world_size)
        self.output_size_per_partition = output_size // world_size
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size_per_partition,
                                                self.input_size,
                                                dtype=params_dtype))
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight, self.output_size, self.input_size,
                self.output_size_per_partition, 0, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size_per_partition, self.input_size,
                device=torch.cuda.current_device(), dtype=params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=0, stride=stride)

        if bias:
            if use_cpu_initialization:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition, dtype=params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=params_dtype))
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

        self.sequence_parallel = sequence_parallel

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.sequence_parallel:
            input_parallel = input_
        else:
            input_parallel = dist_env.copy_to_tensor_model_parallel_region(input_)
        # Matrix multiply.

        bias = self.bias if not self.skip_bias_add else None
        output_parallel = F.linear(input_parallel, self.weight, bias)
        if self.gather_output:
            # All-gather across the partitions.
            assert not self.sequence_parallel
            output = dist_env.gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias
