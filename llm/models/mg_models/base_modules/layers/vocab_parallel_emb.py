import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from functools import partial

from llm.utils.env import dist_env
from llm.utils.model.initializer import _initialize_affine_weight_gpu
from llm.utils.model.initializer import _initialize_affine_weight_cpu
from llm.utils.model.initializer import xavier_uniform_tensor_parallel_


class VocabUtility:
    """Split the vocabulary into `world_size` chunks amd return the
        first and last index of the vocabulary belonging to the `rank`
        partition: Note that indecies in [fist, last)"""

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size,
                                                  rank, world_size):
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size, rank, world_size):
        assert global_vocab_size % world_size == 0, '{} is not divisible by {}'.format(
            global_vocab_size, world_size)
        per_partition_vocab_size = global_vocab_size // world_size
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank, world_size)


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_,
                 use_bnb_optimizer=False,
                 use_cpu_initialization=False,
                 params_dtype=None):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.init_method = init_method
        # Set the defaults for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = dist_env.get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocabulary dimension.
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, dist_env.get_tensor_model_parallel_rank(),
                self.tensor_model_parallel_size)
        self.num_embeddings_per_partition = self.vocab_end_index - \
            self.vocab_start_index

        self.use_bnb_optimizer = use_bnb_optimizer
        self.use_cpu_initialization = use_cpu_initialization
        self.params_dtype = params_dtype

        # Allocate weights and initialize.

        # only the first stage embedding runs this class' forward. The head's embedding does its own
        # thing, so don't waste memory allocating LN weights.
        # if dist.is_pipeline_first_stage() and (args.use_bnb_optimizer or args.embed_layernorm):
        #     self.norm = LayerNorm(embedding_dim)

        if self.use_bnb_optimizer:
            # for BNB we ignore the passed init_method and use torch.nn.init.xavier_uniform_
            # modified to calculate std on the unpartitioned embedding
            init_method = partial(xavier_uniform_tensor_parallel_, tp_degree=self.tensor_model_parallel_size)

        if self.use_cpu_initialization:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                dtype=self.params_dtype))
            _initialize_affine_weight_cpu(
                self.weight, self.num_embeddings, self.embedding_dim,
                self.num_embeddings_per_partition, 0, init_method)
        else:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                device=torch.cuda.current_device(), dtype=self.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=0, stride=1)

        if self.use_bnb_optimizer:
            from bitsandbytes.optim import GlobalOptimManager
            GlobalOptimManager.get_instance().override_config(self.weight, 'optim_bits', 32)
            GlobalOptimManager.get_instance().register_parameters(self.weight)

    def forward(self, input_):
        if torch.any(input_ >= self.num_embeddings):
            raise ValueError(
                f"There is an input id in the input that is greater than the highest possible input id.\nInput: {input_}\nnum_embeddings: {self.num_embeddings}") # noqa

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

        # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = dist_env.reduce_from_tensor_model_parallel_region(output_parallel)

        if hasattr(self, 'norm'):
            output = self.norm(output)

        return output
