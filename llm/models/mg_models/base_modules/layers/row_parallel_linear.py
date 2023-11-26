import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from llm.utils.env import dist_env
from llm.utils.model.initializer import _initialize_affine_weight_gpu
from llm.utils.model.initializer import _initialize_affine_weight_cpu


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
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

    def __init__(self, input_size, output_size, bias=True,
                 input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False,
                 use_cpu_initialization=False,
                 params_dtype=torch.half,
                 sync_tp_duplicated_parameters=False,
                 sequence_parallel=False
                 ):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = bias
        self.input_is_parallel = input_is_parallel
        self.init_method = init_method
        self.stride = stride
        self.keep_master_weight_for_test = keep_master_weight_for_test
        self.use_cpu_initialization = use_cpu_initialization
        self.params_dtype = params_dtype
        # Divide the weight matrix along the last dimension.
        world_size = dist_env.get_tensor_model_parallel_world_size()
        assert input_size % world_size == 0, '{} is not divisible by {}'.format(
            input_size, world_size)
        self.input_size_per_partition = input_size // world_size
        self.skip_bias_add = skip_bias_add
        self.sequence_parallel = sequence_parallel

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size,
                                                self.input_size_per_partition,
                                                dtype=params_dtype))
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight, self.output_size, self.input_size,
                self.input_size_per_partition, 1, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size, self.input_size_per_partition,
                device=torch.cuda.current_device(), dtype=params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=1, stride=stride)
        if bias:
            if use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size,
                                                  dtype=params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size, device=torch.cuda.current_device(),
                    dtype=params_dtype))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

        self.bias_tp_auto_sync = sync_tp_duplicated_parameters

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            assert not self.sequence_parallel
            input_parallel = dist_env.scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight)
        # All-reduce across all the partitions.

        if self.sequence_parallel:
            output_ = dist_env.reduce_scatter_to_sequence_parallel_region(output_parallel, buffer_name="dist_env")
        else:
            output_ = dist_env.reduce_from_tensor_model_parallel_region(output_parallel)

        if self.bias_tp_auto_sync and self.bias is not None:
            torch.distributed.all_reduce(self.bias,
                                         op=torch.distributed.ReduceOp.AVG,
                                         group=dist_env.get_tensor_model_parallel_group())

        if not self.skip_bias_add:
            output = (output_ + self.bias) if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias
