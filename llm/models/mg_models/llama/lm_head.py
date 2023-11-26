import torch
import torch.nn.functional as F

from llm.utils.env import dist_env

from .word_embedings import EmbeddingPipe
from ..base_modules.lora.layers import LoraVocabParallelEmbedding


def lora_parallel_lm_logits(input_, word_embeddings_weight,
                            lora_embeddings_A_weight,
                            lora_embeddings_B_weight, scaling,
                            lora_dropout, is_training,
                            parallel_output, bias=None, sequence_parallel=False):
    """LM logits using word embedding weights."""
    # Parallel logits.
    if sequence_parallel:
        input_ = input_.transpose(0, 1).contiguous()
        input_parallel = dist_env.gather_from_sequence_parallel_region(input_,
                                                                       tensor_parallel_output_grad=True)
        input_parallel = input_parallel.transpose(0, 1).contiguous()
    else:
        input_parallel = dist_env.copy_to_tensor_model_parallel_region(input_)
    # Matrix multiply.
    if bias is None:
        logits_parallel = F.linear(input_parallel,
                                   word_embeddings_weight)
    else:
        logits_parallel = F.linear(input_parallel,
                                   word_embeddings_weight, bias)

    if is_training:
        lora_parallel = F.dropout(input_parallel, p=lora_dropout)
    else:
        lora_parallel = input_parallel
    lora_parallel = F.linear(lora_parallel, lora_embeddings_B_weight)
    lora_parallel = F.linear(lora_parallel, lora_embeddings_A_weight)

    lora_parallel *= scaling
    logits_parallel += lora_parallel

    # Gather if needed.
    if parallel_output:
        return logits_parallel

    return dist_env.gather_from_tensor_model_parallel_region(logits_parallel)


def parallel_lm_logits(input_, word_embeddings_weight, parallel_output,
                       bias=None, sequence_parallel=False):
    """LM logits using word embedding weights."""
    # Parallel logits.
    if sequence_parallel:
        input_ = input_.transpose(0, 1).contiguous()
        input_parallel = dist_env.gather_from_sequence_parallel_region(input_,
                                                                       tensor_parallel_output_grad=True)
        input_parallel = input_parallel.transpose(0, 1).contiguous()
    else:
        input_parallel = dist_env.copy_to_tensor_model_parallel_region(input_)
    # Matrix multiply.
    if bias is None:
        logits_parallel = F.linear(input_parallel,
                                   word_embeddings_weight)
    else:
        logits_parallel = F.linear(input_parallel,
                                   word_embeddings_weight, bias)
    # Gather if needed.
    if parallel_output:
        return logits_parallel

    return dist_env.gather_from_tensor_model_parallel_region(logits_parallel)


class EmbedddingPipeNoTied(EmbeddingPipe):
    def __init__(self,
                 *args,
                 parallel_output=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.parallel_output = parallel_output

    def forward(self, lm_output, **kwargs):
        if isinstance(self.word_embeddings, LoraVocabParallelEmbedding):
            # sycn lora_embeddings_B_weight.
            if self.word_embeddings.lora_sync_tp_duplicated_parameters:
                torch.distributed.all_reduce(self.word_embeddings.lora_embedding_B_weight,
                                             op=torch.distributed.ReduceOp.AVG,
                                             group=dist_env.get_tensor_model_parallel_group())
            return lora_parallel_lm_logits(
                lm_output,
                self.word_embeddings_weight,
                self.word_embeddings.lora_embedding_A_weight,
                self.word_embeddings.lora_embedding_B_weight,
                self.word_embeddings.scaling,
                self.word_embeddings.lora_dropout,
                self.training,
                self.parallel_output,
                sequence_parallel=self.sequence_parallel)
        else:
            return parallel_lm_logits(
                lm_output,
                self.word_embeddings_weight,
                self.parallel_output,
                sequence_parallel=self.sequence_parallel)
