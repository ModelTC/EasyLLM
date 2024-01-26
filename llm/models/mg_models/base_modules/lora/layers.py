
import math

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from llm.utils.env import dist_env
from llm.utils.model.initializer import _initialize_affine_weight_gpu

from ..layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding


class LoraColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, input_size, output_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False, skip_bias_add=False,
                 use_cpu_initialization=False,
                 params_dtype=torch.half,
                 sequence_parallel=False,
                 lora_rank=0, lora_alpha=0, lora_dropout=0,
                 lora_sync_tp_duplicated_parameters=True):
        super().__init__(input_size, output_size, bias, gather_output,
                         init_method, stride,
                         keep_master_weight_for_test, skip_bias_add,
                         use_cpu_initialization=use_cpu_initialization,
                         params_dtype=params_dtype,
                         sequence_parallel=sequence_parallel)
        assert not use_cpu_initialization
        assert not sequence_parallel, "Lora does not support sequence_parallel currently!"
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.lora_rank
        self.lora_dropout = lora_dropout
        self.lora_sync_tp_duplicated_parameters = lora_sync_tp_duplicated_parameters

        self.lora_A_weight = Parameter(torch.empty(
            self.lora_rank, self.input_size,
            device=torch.cuda.current_device(), dtype=params_dtype))
        self.lora_B_weight = Parameter(torch.empty(
            self.output_size_per_partition, self.lora_rank,
            device=torch.cuda.current_device(), dtype=params_dtype))
        _initialize_affine_weight_gpu(self.lora_A_weight, init.kaiming_uniform_,
                                      partition_dim=0, stride=stride, a=math.sqrt(5))
        _initialize_affine_weight_gpu(self.lora_B_weight, init.zeros_,
                                      partition_dim=0, stride=stride)

    def forward(self, input_):
        # Set up backprop all-reduce.
        input_parallel = dist_env.copy_to_tensor_model_parallel_region(input_)
        # Matrix multiply.

        bias = self.bias if not self.skip_bias_add else None
        output_parallel = F.linear(input_parallel, self.weight, bias)
        if self.training:
            output_lora = F.dropout(input_parallel, p=self.lora_dropout)
        else:
            output_lora = input_parallel
        if self.lora_sync_tp_duplicated_parameters:
            torch.distributed.all_reduce(self.lora_A_weight,
                                         op=torch.distributed.ReduceOp.AVG,
                                         group=dist_env.get_tensor_model_parallel_group())
        output_lora = F.linear(output_lora, self.lora_A_weight)
        output_lora = F.linear(output_lora, self.lora_B_weight)
        output_lora *= self.scaling
        output_parallel += output_lora
        if self.gather_output:
            # All-gather across the partitions.
            assert not self.sequence_parallel
            output = dist_env.gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class LoraRowParallelLinear(RowParallelLinear):
    def __init__(self, input_size, output_size, bias=True,
                 input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False,
                 use_cpu_initialization=False,
                 params_dtype=torch.half,
                 sync_tp_duplicated_parameters=False,
                 sequence_parallel=False,
                 lora_rank=0,
                 lora_alpha=0,
                 lora_dropout=0,
                 lora_sync_tp_duplicated_parameters=True):
        super().__init__(input_size, output_size, bias,
                         input_is_parallel, init_method, stride,
                         keep_master_weight_for_test, skip_bias_add,
                         use_cpu_initialization, params_dtype,
                         sync_tp_duplicated_parameters, sequence_parallel)
        assert not use_cpu_initialization
        assert not sequence_parallel, "Lora does not support sequence_parallel currently!"
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.lora_rank
        self.lora_dropout = lora_dropout
        self.lora_sync_tp_duplicated_parameters = lora_sync_tp_duplicated_parameters

        self.lora_A_weight = Parameter(torch.empty(
            self.lora_rank, self.input_size_per_partition,
            device=torch.cuda.current_device(), dtype=params_dtype))
        self.lora_B_weight = Parameter(torch.empty(
            self.output_size, self.lora_rank,
            device=torch.cuda.current_device(), dtype=params_dtype))
        _initialize_affine_weight_gpu(self.lora_A_weight, init.kaiming_uniform_,
                                      partition_dim=1, stride=stride, a=math.sqrt(5))
        _initialize_affine_weight_gpu(self.lora_B_weight, init.zeros_,
                                      partition_dim=1, stride=stride)

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            assert not self.sequence_parallel
            input_parallel = dist_env.scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight)
        if self.training:
            output_lora = F.dropout(input_parallel, p=self.lora_dropout)
        else:
            output_lora = input_parallel
        if self.lora_sync_tp_duplicated_parameters:
            torch.distributed.all_reduce(self.lora_B_weight,
                                         op=torch.distributed.ReduceOp.AVG,
                                         group=dist_env.get_tensor_model_parallel_group())
        output_lora = F.linear(output_lora, self.lora_A_weight)
        output_lora = F.linear(output_lora, self.lora_B_weight)
        output_lora *= self.scaling
        output_parallel += output_lora
        # All-reduce across all the partitions.
        output_ = dist_env.reduce_from_tensor_model_parallel_region(output_parallel)

        if self.bias_tp_auto_sync and self.bias:
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


class LoraVocabParallelEmbedding(VocabParallelEmbedding):
    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_,
                 use_bnb_optimizer=False,
                 use_cpu_initialization=False,
                 params_dtype=None,
                 lora_rank=0,
                 lora_alpha=0,
                 lora_dropout=0,
                 lora_sync_tp_duplicated_parameters=True):
        super().__init__(num_embeddings, embedding_dim, init_method,
                         use_bnb_optimizer, use_cpu_initialization, params_dtype)
        assert not self.use_cpu_initialization
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.lora_rank
        self.lora_dropout = lora_dropout
        self.lora_sync_tp_duplicated_parameters = lora_sync_tp_duplicated_parameters

        self.lora_embedding_A_weight = Parameter(torch.empty(
            self.num_embeddings_per_partition, self.lora_rank,
            device=torch.cuda.current_device(), dtype=self.params_dtype))
        self.lora_embedding_B_weight = Parameter(torch.empty(
            self.lora_rank, self.embedding_dim,
            device=torch.cuda.current_device(), dtype=self.params_dtype))

        _initialize_affine_weight_gpu(self.lora_embedding_A_weight, init.zeros_,
                                      partition_dim=0, stride=1)
        _initialize_affine_weight_gpu(self.lora_embedding_B_weight, init.normal_,
                                      partition_dim=0, stride=1)

    def forward(self, input_):
        if torch.any(input_ >= self.num_embeddings):
            raise ValueError(
                f"There is an input id in the input that is greater than the highest possible input id.\nInput: {input_}\nnum_embeddings: {self.num_embeddings}")       # noqa

        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | \
                         (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            # input_ is garanted to be in the range [0:self.vocab_end_index -
            # self.vocab_start_index] thanks to the first check
            masked_input = input_

        if self.lora_sync_tp_duplicated_parameters:
            torch.distributed.all_reduce(self.lora_embedding_B_weight,
                                         op=torch.distributed.ReduceOp.AVG,
                                         group=dist_env.get_tensor_model_parallel_group())
        # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)

        output_lora = F.embedding(masked_input, self.lora_embedding_A_weight,
                                  self.padding_idx, self.max_norm,
                                  self.norm_type, self.scale_grad_by_freq,
                                  self.sparse)
        output_lora = output_lora @ self.lora_embedding_B_weight * self.scaling
        output_parallel += output_lora

        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = dist_env.reduce_from_tensor_model_parallel_region(output_parallel)

        if hasattr(self, 'norm'):
            output = self.norm(output)

        return output
