# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Megatron arguments."""

import argparse
import deepspeed
import os


def parse_args():
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Megatron-LM Arguments', add_help=False,
                                     allow_abbrev=False)

    # Standard arguments.
    parser = _add_training_args(parser)
    parser = _add_inference_args(parser)
    parser = _add_medusa_args(parser)
    parser = _add_distributed_args(parser)
    if os.environ.get("ASCEND_RUNTIME", None):
        parser = _parse_extra_ascend_args(parser)
        
    parser = deepspeed.add_config_arguments(parser)

    args, _ = parser.parse_known_args()
    

    return args


def _add_training_args(parser):
    group = parser.add_argument_group(title='training')

    group.add_argument('--config', dest='config', default=None, help='settings in yaml format')
    group.add_argument('--distribute-checkpointed-activations',
                       action='store_true',
                       help='If set, distribute checkpointed activations '
                       'across model parallel group.')
    group.add_argument("--lora-mode", action="store_true",
                       help="Whether to use Lora for parameter efficient tuning")
    group.add_argument('--seed', type=int, default=None,
                       help='Random seed used for python, numpy, '
                       'pytorch, and cuda.')
    group.add_argument('--fp16', action='store_true',
                       help='Run model in fp16 mode.')
    group.add_argument('--bf16', action='store_true',
                       help='Run model in bfloat16 mode.')
    group.add_argument('--opts', type=str, default=None, nargs=argparse.REMAINDER)
    group.add_argument('--port', type=int, default=13333,
                       help='Slurm port to init.')
    return parser


def _add_inference_args(parser):
    group = parser.add_argument_group(title='inference')

    group.add_argument('--inference', action='store_true',
                       help='Very basic inference mode: not allocating optim/lr - requires ZERO_STAGE=0')
    group.add_argument('--generate-log-frequency', type=int, default=-1,
                       help='The log frequency in the generate interavtive mode.')
    group.add_argument('--generate-mode', type=str, default=None,
                       help='The sample generation mode.')
    group.add_argument('--logp-file', type=str, default="logp.jsonl",
                       help='logp file path saving logp results.')
    group.add_argument('--question-file', type=str, default=None,
                       help="Question file path.")
    group.add_argument('--sample-file', type=str, default="samples.jsonl",
                       help='Sample file path saving inference results.')
    group.add_argument('--eval_task', type=str, default="base",
                       help='Evaluation task type.')
    group.add_argument('--force_eos_id', type=int, default=None,
                       help='Forcing eos_id by user.')
    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--greedy", action='store_true', default=False,
                       help='Use greedy sampling.')
    group.add_argument("--top-p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top-k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument("--out-seq-length", type=int, default=1024,
                       help='Size of the output generated text.')
    group.add_argument("--recompute", action='store_true',
                       help='During generation recompute all attention '
                       'instead of using previously computed keys/values.')
    group.add_argument("--infer-file", type=str, default=None,
                       help="infer file")
    group.add_argument("--predictions-results", type=str, default="predictions_results.json",
                       help="predictions results file")
    group.add_argument("--with-infer-tokenization", action='store_true',
                       help="Whether using infer tokenization")
    return parser


def _add_medusa_args(parser):
    group = parser.add_argument_group(title='medusa')

    group.add_argument("--medusa-generate", action='store_true')
    group.add_argument("--medusa-temperature", type=float, default=0.7)
    group.add_argument("--medusa-max-steps", type=int, default=512)
    group.add_argument("--medusa-choices", type=str, default="mc_sim_7b_3_head_top5")
    return parser


def _add_distributed_args(parser):
    group = parser.add_argument_group(title='distributed')

    group.add_argument('--launcher', type=str, default="slurm",
                       help='huggingface runner launching mode.')
    group.add_argument('--tensor-model-parallel-size', type=int, default=None,
                       help='Degree of tensor model parallelism.')
    group.add_argument('--pipeline-model-parallel-size', type=int, default=None,
                       help='Degree of pipeline model parallelism.')
    group.add_argument('--distributed-backend', default='nccl',
                       choices=['nccl', 'gloo', 'ccl'],
                       help='Which backend to use for distributed training.')
    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher.')
    group.add_argument('--lazy-mpu-init', type=bool, required=False,
                       help='If set to True, initialize_megatron() '
                       'skips DDP initialization and returns function to '
                       'complete it instead. This is for '
                       'external DDP manager.')
    return parser


# ascend
def _parse_extra_ascend_args(first_parser, extra_args_provider=None,
               ignore_unknown_args=True):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='AscendSpeed Arguments', parents=[first_parser],
                                     allow_abbrev=False)
    
    # Standard arguments.
    parser = _add_ascend_network_size_args(parser)
    parser = _add_ascend_regularization_args(parser)
    parser = _add_ascend_training_args(parser)
    parser = _add_ascend_initialization_args(parser)
    parser = _add_ascend_learning_rate_args(parser)
    parser = _add_ascend_checkpointing_args(parser)
    parser = _add_ascend_mixed_precision_args(parser)
    parser = _add_ascend_distributed_args(parser)
    parser = _add_ascend_autoresume_args(parser)
    parser = _add_ascend_biencoder_args(parser)
    parser = _add_ascend_vit_args(parser)
    parser = _add_ascend_logging_args(parser)
    parser = _add_ascend_zero_args(parser)
    parser = _add_ascend_memoryopt_args(parser)
    parser = _add_ascend_activation_checkpoint_args(parser)
    parser = _add_ascend_distillation_args(parser)
    parser = _add_ascend_optimized_pipeline_args(parser)
    parser = _add_ascend_manual_layer_allocation(parser)
    parser = _add_ascend_lora_args(parser)
    parser = _add_ascend_transformer_engine_args(parser)
    parser = _add_ascend_retro_args(parser)


    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    return parser




def _add_ascend_network_size_args(parser):
    from ascend.enums import PositionEmbeddingType
    group = parser.add_argument_group(title='network size')

    group.add_argument('--num-layers', type=int, default=None,
                       help='Number of transformer layers.')
    group.add_argument('--encoder-num-layers', type=int, default=None,
                       help='Number of encoder transformer layers.')
    group.add_argument('--decoder-num-layers', type=int, default=None,
                       help='Number of decoder transformer layers.')
    group.add_argument('--num-experts', type=int, nargs='+' , default=[1,],
                           help='number of experts list, MoE related.')
    group.add_argument('--mlp-type', type=str, default='standard',
                           help='Only applicable when num-experts > 1, accepts [standard, residual]')
    group.add_argument('--topk', type=int, default=1,
                           help='Sets the k in TopK gating for MoE layers')
    group.add_argument('--expert-interval', type=int, default=2,
                           help='Use experts in every "expert-interval" layers')
    group.add_argument('--hidden-size', type=int, default=None,
                       help='Tansformer hidden size.')
    group.add_argument('--ffn-hidden-size', type=int, default=None,
                       help='Transformer Feed-Forward Network hidden size. '
                       'This is set to 4*hidden-size if not provided')
    group.add_argument('--num-attention-heads', type=int, default=None,
                       help='Number of transformer attention heads.')
    group.add_argument('--kv-channels', type=int, default=None,
                       help='Projection weights dimension in multi-head '
                       'attention. This is set to '
                       '   args.hidden_size // args.num_attention_heads '
                       'if not provided.')
    group.add_argument('--group-query-attention', action='store_true',
                       help='Use group-query attention.')
    group.add_argument('--num-query-groups', type=int, default=1)
    group.add_argument('--embed-layernorm', action='store_true',
                       help='Use layernorm for embedding.')
    group.add_argument('--max-position-embeddings', type=int, default=None,
                       help='Maximum number of position embeddings to use. '
                       'This is the size of position embedding.', dest="max_position_embeddings")
    group.add_argument('--position-embedding-type', type=lambda x: PositionEmbeddingType[x],
                       choices=list(PositionEmbeddingType), default=PositionEmbeddingType.absolute,
                       help='Define position embedding type ("absolute" | "rope" | "alibi"). "absolute" by default.')
    group.add_argument('--use-rotary-position-embeddings', action='store_true',
                       help='Use rotary positional embeddings or not. '
                       'Deprecated: use --position-embedding-type')
    group.add_argument('--rotary-percent', type=float, default=1.0,
                       help='Percent of rotary dimension to use, default 100%')
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                       'This is added for computational efficieny reasons.')
    group.add_argument('--pad-vocab-size-to', type=int, default=None,
                       help='Pad the vocab size to this value.'
                       'This value must be greater than the initial size of the tokenizer,'
                       'needs to be divisible by TP size and `make-vocab-size-divisible-by`.')
    group.add_argument('--normalization', default='LayerNorm',
                       choices=['LayerNorm', 'RMSNorm'],
                       help='Which normalization technique to use.')
    group.add_argument('--apply-layernorm-1p', action='store_true',
                       help='Adjust LayerNorm weights such that they are centered '
                       'around zero. This improves numerical stability.')
    group.add_argument('--layernorm-epsilon', type=float, default=1e-5,
                       help='Layer norm epsilon.')
    group.add_argument('--apply-residual-connection-post-layernorm',
                       action='store_true',
                       help='If set, use original BERT residula connection '
                       'ordering.')
    group.add_argument('--openai-gelu', action='store_true',
                       help='Use OpenAIs GeLU implementation. This option'
                       'should not be used unless for backward compatibility'
                       'reasons.')
    group.add_argument('--squared-relu', action='store_true',
                       help='Use squared relu activation instead of default gelu')
    group.add_argument('--swiglu', action='store_true',
                       help='Use gated linear units and SiLU activation instead of default gelu')
    group.add_argument('--onnx-safe', type=bool, required=False,
                       help='Use workarounds for known problems with '
                       'Torch ONNX exporter')
    group.add_argument('--bert-no-binary-head', action='store_false',
                       help='Disable BERT binary head.',
                       dest='bert_binary_head')
    group.add_argument('--no-untie-embeddings-and-output-weights', action='store_false',
                       help='Not untie embeddings and output weights.',
                       dest="untie_embeddings_and_output_weights",
                       default=True),
    group.add_argument('--mlp-layer-fusion', action='store_true',
                       help='Fuse gate and upprojection in MLP for llama families, '
                       'e.g. llama or internlm')
    group.add_argument('--use-flash-attn', action='store_true',
                       default=False,
                       help='Use flash attention')
    group.add_argument('--use-fused-rmsnorm', action='store_true', help='use fused norm')
    group.add_argument('--embedding-weights-in-fp32', action='store_true',
                       help='Cast word embedding weights to fp32 before embedding fwd.'),
    group.add_argument('--no-add-gate', action='store_false', default=True, dest="add_gate",
                       help='Do not add gate layer in model.'),
    return parser


def _add_ascend_logging_args(parser):
    group = parser.add_argument_group(title='logging')

    group.add_argument('--log-params-norm', action='store_true',
                       help='If set, calculate and log parameters norm.')
    group.add_argument('--log-num-zeros-in-grad', action='store_true',
                       help='If set, calculate and log the number of zeros in gradient.')
    group.add_argument('--timing-log-level', type=int,
                       default=0, choices=range(0,3),
                       help='Granularity level to measure and report timing. '
                       '   0: report only iteration time and make sure timing '
                       '      does not introduce extra overhead.'
                       '   1: report timing for operations that are executed '
                       '      very limited times (basically once) during '
                       '      each iteration (such as gradient all-reduce) '
                       '   2: report timing for operations that migh be '
                       '      executed numerous times during each iteration. '
                       'Note that setting the level to 1 or 2 might '
                       'cause increase in iteration time.')
    group.add_argument('--no-barrier-with-level-1-timing', action='store_false',
                       help='If not set, use barrier with level 1 time '
                       'measurements. Note that this is up to the user '
                       'to make sure calling barrier with their timers '
                       'will not result in hangs. This can happen if for '
                       'example the user adds a level 1 timer that is not '
                       'called by all ranks.',
                       dest='barrier_with_L1_time')
    group.add_argument('--timing-log-option', type=str, default='flatten',
                       choices=['flatten', 'max', 'minmax', 'all'],
                       help='Options for logging timing:'
                       '  flatten: report elapsed time in one line'
                       '  max: report the max timing across all ranks'
                       '  minmax: report min and max timings across all ranks'
                       '  all: report timings of all ranks.')
    group.add_argument('--tensorboard-log-interval', type=int, default=1,
                       help='Report to tensorboard interval.')
    group.add_argument('--tensorboard-queue-size', type=int, default=1000,
                       help='Size of the tensorboard queue for pending events '
                       'and summaries before one of the ‘add’ calls forces a '
                       'flush to disk.')
    group.add_argument('--log-timers-to-tensorboard', action='store_true',
                       help='If set, write timers to tensorboard.')
    group.add_argument('--log-batch-size-to-tensorboard', action='store_true',
                       help='If set, write batch-size to tensorboard.')
    group.add_argument('--no-log-learnig-rate-to-tensorboard',
                       action='store_false',
                       help='Disable learning rate logging to tensorboard.',
                       dest='log_learning_rate_to_tensorboard')
    group.add_argument('--no-log-loss-scale-to-tensorboard',
                       action='store_false',
                       help='Disable loss-scale logging to tensorboard.',
                       dest='log_loss_scale_to_tensorboard')
    group.add_argument('--log-validation-ppl-to-tensorboard',
                       action='store_true',
                       help='If set, write validation perplexity to '
                       'tensorboard.')
    group.add_argument('--log-optimizer-states-to-tensorboard',
                       action='store_true',
                       help='If set, write various optimizer states to '
                       'tensorboard. This feature may consume extra GPU memory.')

    return parser


def _add_ascend_regularization_args(parser):
    group = parser.add_argument_group(title='regularization')

    group.add_argument('--attention-dropout', type=float, default=0.0,
                       help='Post attention dropout probability.')
    group.add_argument('--hidden-dropout', type=float, default=0.0,
                       help='Dropout probability for hidden state transformer.')
    group.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay coefficient for L2 regularization.')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='Gradient clipping based on global L2 norm.')
    group.add_argument('--adam-beta1', type=float, default=0.9,
                       help='First coefficient for computing running averages '
                       'of gradient and its square')
    group.add_argument('--adam-beta2', type=float, default=0.999,
                       help='Second coefficient for computing running averages '
                       'of gradient and its square')
    group.add_argument('--adam-eps', type=float, default=1e-08,
                       help='Term added to the denominator to improve'
                       'numerical stability')
    group.add_argument('--sgd-momentum', type=float, default=0.9,
                       help='Momentum factor for sgd')

    return parser


def _add_ascend_training_args(parser):
    group = parser.add_argument_group(title='training')
    group.add_argument('--micro-batch-size', type=int, default=None,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size times number of micro batches.')
    group.add_argument('--batch-size', type=int, default=None,
                       help='Old batch size parameter, do not use. '
                       'Use --micro-batch-size instead')
    group.add_argument('--global-batch-size', type=int, default=None,
                       help='Training batch size. If set, it should be a '
                       'multiple of micro-batch-size times data-parallel-size. '
                       'If this value is None, then '
                       'use micro-batch-size * data-parallel-size as the '
                       'global batch size. This choice will result in 1 for '
                       'number of micro-batches.')
    group.add_argument('--rampup-batch-size', nargs='*', default=None,
                       help='Batch size ramp up with the following values:'
                       '  --rampup-batch-size <start batch size> '
                       '                      <batch size incerement> '
                       '                      <ramp-up samples> '
                       'For example:'
                       '   --rampup-batch-size 16 8 300000 \ '
                       '   --global-batch-size 1024'
                       'will start with global batch size 16 and over '
                       ' (1024 - 16) / 8 = 126 intervals will increase'
                       'the batch size linearly to 1024. In each interval'
                       'we will use approximately 300000 / 126 = 2380 samples.')
    group.add_argument('--recompute-activations', action='store_true',
                       help='recompute activation to allow for training '
                       'with larger models, sequences, and batch sizes.')
    group.add_argument('--recompute-granularity', type=str, default=None,
                       choices=['full'],
                       help='Checkpoint activations to allow for training '
                       'with larger models, sequences, and batch sizes.'
                       'It is supported at two granularities 1) full: '
                       'whole transformer layer is recomputed')
    group.add_argument('--distribute-saved-activations',
                       action='store_true',
                       help='If set, distribute recomputed activations '
                       'across model parallel group.')
    group.add_argument('--recompute-method', type=str, default='uniform',
                       choices=['uniform', 'block', "custom"],
                       help='1) uniform: uniformly divide the total number of '
                       'Transformer layers and recompute the input activation of '
                       'each divided chunk at specified granularity, '
                       '2) recompute the input activations of only a set number of '
                       'individual Transformer layers per pipeline stage and do the '
                       'rest without any recomputing at specified granularity'
                       'default) do not apply activations recompute to any layers')
    group.add_argument('--recompute-num-layers', type=int, default=1,
                       help='1) uniform: the number of Transformer layers in each '
                       'uniformly divided recompute unit, '
                       '2) block: the number of individual Transformer layers '
                       'to recompute within each pipeline stage.')


    group.add_argument('--checkpoint-activations', action='store_true',
                       help='Checkpoint activation to allow for training '
                       'with larger models, sequences, and batch sizes.')
    group.add_argument('--recomputation-layer-num', nargs='+',
                       type=int, help='Represents the number of layers to be recomputed at each stage of the pp. '
                       'The default is None. If pp=4, each stage has 8 layers, '
                       'if this parameter is equal to 4 4 4 4, '
                       'it means that each stage only needs to recompute 4 layers.')
    # group.add_argument('--distribute-checkpointed-activations',
    #                    action='store_true',
    #                    help='If set, distribute checkpointed activations '
    #                    'across model parallel group.')
    group.add_argument('--checkpoint-num-layers', type=int, default=1,
                       help='chunk size (number of layers) for checkpointing.')
    group.add_argument('--train-iters', type=int, default=None,
                       help='Total number of iterations to train over all '
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
    group.add_argument('--train-samples', type=int, default=None,
                       help='Total number of samples to train over all '
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
    group.add_argument('--train-tokens', type=int, default=None,
                       help='Total number of tokens to train over all '
                       'training runs.')
    group.add_argument('--random-ltd',
                       action='store_true',
                       help='enable random layer token drop')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Report loss and timing interval.')
    group.add_argument('--exit-interval', type=int, default=None,
                       help='Exit the program after the iteration is divisible '
                       'by this value.')
    group.add_argument('--exit-duration-in-mins', type=int, default=None,
                       help='Exit the program after this many minutes.')
    group.add_argument('--tensorboard-dir', type=str, default=None,
                       help='Write TensorBoard logs to this directory.')
    group.add_argument('--no-masked-softmax-fusion',
                       action='store_false',
                       help='Disable fusion of query_key_value scaling, '
                       'masking, and softmax.',
                       dest='masked_softmax_fusion')
    group.add_argument('--no-bias-gelu-fusion', action='store_false',
                       help='Disable bias and gelu fusion.',
                       dest='bias_gelu_fusion')
    group.add_argument('--no-bias-dropout-fusion', action='store_false',
                       help='Disable bias and dropout fusion.',
                       dest='bias_dropout_fusion')
    group.add_argument('--disable-moe-token-dropping', action='store_false',
                       help='Disable MoE expert token dropping.',
                       dest='moe_token_dropping')
    group.add_argument('--moe-train-capacity-factor', type=float, default=1.0,
                       help='The capacity of the MoE expert at training time')
    group.add_argument('--moe-eval-capacity-factor', type=float, default=1.0,
                       help='The capacity of the MoE expert at eval time.')
    group.add_argument('--moe-min-capacity', type=int, default=4,
                       help='The minimum capacity per MoE expert regardless of the capacity_factor.')
    group.add_argument('--moe-loss-coeff', type=float, default=0.1,
                       help='Scaling coefficient for adding MoE loss to model loss')
    group.add_argument('--create-moe-param-group', action='store_true',
                       help='Create separate groups for MoE params.'
                       'This is necessary for techniques like ZeRO.')
    group.add_argument('--add-bias-linear', action='store_true',
                       help='Use bias in the linear layers',
                       dest='add_bias_linear', default=False)
    group.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd', 'fused_adam', 'cadam'],
                       help='Optimizer function')
    group.add_argument('--release-fp32-grad', action='store_true',
                       help='The distributed training optimizer frees up '
                       'gradient copies of FP32 to save memory.')
    group.add_argument('--dataloader-type', type=str, default=None,
                       choices=['single', 'cyclic'],
                       help='Single pass vs multiple pass data loader')
    group.add_argument('--ds-inference', action='store_true',
                       help='DeepSpeed inference engine being used')
    group.add_argument('--cpu-optimizer', action='store_true',
                       help='Run optimizer on CPU')
    group.add_argument('--cpu_torch_adam', action='store_true',
                       help='Use Torch Adam as optimizer on CPU.')
    group.add_argument('--no-pipeline-parallel', action='store_true',
                       help='Disable pipeline parallelism')
    group.add_argument('--use-tutel', action='store_true',
                       help='Use Tutel optimization for MoE')
    # group.add_argument('--inference', action='store_true',
    #                    help='Very basic inference mode: not allocating optim/lr - requires ZERO_STAGE=0')
    group.add_argument('--use-fused-rotary-pos-emb', action='store_true',
                       help='use fused rotary pos emb')
    group.add_argument('--no-async-tensor-model-parallel-allreduce',
                       action='store_false',
                       help='Disable asynchronous execution of '
                       'tensor-model-parallel all-reduce with weight '
                       'gradient compuation of a column-linear layer.',
                       dest='async_tensor_model_parallel_allreduce')
    group.add_argument('--no-gradient-accumulation-fusion',
                       action='store_true',
                       help='Disable fusing gradient accumulation to weight '
                       'gradient computation of linear layers',
                       dest='gradient_accumulation_fusion')
    group.add_argument('--auto-recompute-device-size',
                       type=int, default=-1,
                       help='The memory size for auto selective recompute strategy. '
                            'The default is -1. If this parameter > 0, '
                            'will activate auto selective recompute. ')
    group.add_argument('--auto-recompute-profiling-step',
                       type=int, default=10,
                       help='The profiling step for auto selective recompute strategy. '
                            'The default is 10. If activate auto selective recompute, '
                            'will solve graph after step 10. ')
    group.add_argument('--z-loss-weight',
                       type=float, default=None,
                       help='add penalty item for loss function')
    group.add_argument('--lm-norm-weight',
                       action='store_true',
                       default=False,
                       help='normalize the weight of lm head before matmul')
    group.add_argument('--padding-attention-mask',
                       action='store_true',
                       default=False,
                       help='padding attention_mask')
    group.add_argument('--square-alibi-mask',
                       action='store_true',
                       default=False,
                       help='attention mask of alibi is squared')
    group.add_argument('--fill-neg-inf',
                       action='store_true',
                       default=False,
                       help='fill alibi with negative inf')
    group.add_argument('--seq-length', type=int, default=None,
                       help='Maximum sequence length to process.')
    group.add_argument('--variable-seq-lengths', action='store_true',
                       help='Use variable seq lengths or not.')
    group.add_argument('--encoder-seq-length', type=int, default=None,
                       help='Maximum encoder sequence length to process.'
                       'This should be exclusive of --seq-length')
    group.add_argument('--decoder-seq-length', type=int, default=None,
                       help="Maximum decoder sequence length to process.")
    group.add_argument('--retriever-seq-length', type=int, default=256,
                       help='Maximum sequence length for the biencoder model '
                        ' for retriever')
    group.add_argument('--keep-last-token', action='store_true', default=False,
                        help="keep last token of input_ids, attention_mask and labels during data processing")
    return parser


def _add_ascend_initialization_args(parser):
    group = parser.add_argument_group(title='initialization')

    group.add_argument('--init-method-std', type=float, default=0.02,
                       help='Standard deviation of the zero mean normal '
                       'distribution used for weight initialization.')
    group.add_argument('--init-method-xavier-uniform', action='store_true',
                       help='Enable Xavier uniform parameter initialization')

    return parser


def _add_ascend_learning_rate_args(parser):
    group = parser.add_argument_group(title='learning rate')

    group.add_argument('--lr', type=float, default=None,
                       help='Initial learning rate. Depending on decay style '
                       'and initial warmup, the learing rate at each '
                       'iteration would be different.')
    group.add_argument('--lr-decay-style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine'],
                       help='Learning rate decay function.')
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay learning rate over,'
                       ' If None defaults to `--train-iters`')
    group.add_argument('--lr-decay-samples', type=int, default=None,
                       help='number of samples to decay learning rate over,'
                       ' If None defaults to `--train-samples`')
    group.add_argument('--lr-decay-tokens', type=int, default=None,
                       help='number of tokens to decay learning rate over,'
                       ' If not None will override iter/sample-based decay')
    group.add_argument('--lr-warmup-fraction', type=float, default=None,
                       help='fraction of lr-warmup-(iters/samples) to use '
                       'for warmup (as a float)')
    group.add_argument('--lr-warmup-iters', type=int, default=0,
                       help='number of iterations to linearly warmup '
                       'learning rate over.')
    group.add_argument('--lr-warmup-samples', type=int, default=0,
                       help='number of samples to linearly warmup '
                       'learning rate over.')
    group.add_argument('--lr-warmup-tokens', type=int, default=None,
                       help='number of tokens to linearly warmup '
                       'learning rate over.')
    group.add_argument('--warmup', type=int, default=None,
                       help='Old lr warmup argument, do not use. Use one of the'
                       '--lr-warmup-* arguments above')
    group.add_argument('--min-lr', type=float, default=0.0,
                       help='Minumum value for learning rate. The scheduler'
                       'clip values below this threshold.')
    group.add_argument('--override-lr-scheduler', action='store_true',
                       help='Reset the values of the scheduler (learning rate,'
                       'warmup iterations, minimum learning rate, maximum '
                       'number of iterations, and decay style from input '
                       'arguments and ignore values from checkpoints. Note'
                       'that all the above values will be reset.')
    group.add_argument('--use-checkpoint-lr-scheduler', action='store_true',
                       help='Use checkpoint to set the values of the scheduler '
                       '(learning rate, warmup iterations, minimum learning '
                       'rate, maximum number of iterations, and decay style '
                       'from checkpoint and ignore input arguments.')

    return parser


def _add_ascend_checkpointing_args(parser):
    group = parser.add_argument_group(title='checkpointing')

    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save-interval', type=int, default=None,
                       help='Number of iterations between checkpoint saves.')
    group.add_argument('--save-to-cpu', action='store_true', default=False,
                       help='Save checkpoint to cpu tensor.')
    group.add_argument('--no-save-optim', action='store_true', default=None,
                       help='Do not save current optimizer.')
    group.add_argument('--no-save-rng', action='store_true', default=None,
                       help='Do not save current rng state.')
    group.add_argument('--load', type=str, default=None,
                       help='Directory containing a model checkpoint.')
    group.add_argument('--no-load-optim', action='store_true', default=None,
                       help='Do not load optimizer when loading checkpoint.')
    group.add_argument('--no-load-rng', action='store_true', default=None,
                       help='Do not load rng state when loading checkpoint.')
    group.add_argument('--no-load-lr-state', action='store_true',
                       help='Do not load lr state when loading checkpoint.')
    group.add_argument('--finetune', action='store_true',
                       help='Load model for finetuning. Do not load optimizer '
                       'or rng state from checkpoint and set iteration to 0. '
                       'Assumed when loading a release checkpoint.')
    group.add_argument('--no-initialization', action='store_false',
                       help='Do not perform initialization when building model, '
                       'can reduce startup time when definitely loading from a '
                       'checkpoint',
                       dest='perform_initialization')

    return parser


def _add_ascend_mixed_precision_args(parser):
    group = parser.add_argument_group(title='mixed precision')


    group.add_argument('--loss-scale', type=float, default=None,
                       help='Static loss scaling, positive power of 2 '
                       'values can improve fp16 convergence. If None, dynamic'
                       'loss scaling is used.')
    group.add_argument('--initial-loss-scale', type=float, default=2**32,
                       help='Initial loss-scale for dynamic loss scaling.')
    group.add_argument('--min-loss-scale', type=float, default=1.0,
                       help='Minimum loss scale for dynamic loss scale.')
    group.add_argument('--loss-scale-window', type=float, default=1000,
                       help='Window over which to raise/lower dynamic scale.')
    group.add_argument('--hysteresis', type=int, default=2,
                       help='hysteresis for dynamic loss scaling')
    group.add_argument('--fp32-residual-connection', action='store_true',
                       help='Move residual connections to fp32.')
    group.add_argument('--query-key-layer-scaling', action='store_true',
                       help='Scale Q * K^T by 1 / layer-number.',
                       dest='apply_query_key_layer_scaling', default=False)
    group.add_argument('--no-attention-softmax-in-fp32', action='store_false',
                       help='Not run attention masking and softmax in fp32. '
                            'This flag is ignored unless '
                            '--no-query-key-layer-scaling is specified.', default=True,
                       dest="attention_softmax_in_fp32")
    group.add_argument('--accumulate-allreduce-grads-in-fp32',
                       action='store_true',
                       help='Gradient accumulation and all-reduce in fp32.')
    group.add_argument('--fp16-lm-cross-entropy', action='store_true',
                       help='Move the cross entropy unreduced loss calculation'
                       'for lm head to fp16.')

    return parser


def _add_ascend_distributed_args(parser):
    group = parser.add_argument_group(title='distributed')

    group.add_argument('--foldx-mode', default=None,
                       choices=['aiao', 'fifo'],
                       help='Choose fold-x pipeline parallelism.')
    group.add_argument('--enable-expert-tensor-parallelism', action='store_true',
                        default=False,
                        help="use tensor parallelism for expert layers in MoE")
    group.add_argument('--sequence-parallel', action='store_true',
                       default=False,
                       help="use sequence parallelism")
    group.add_argument('--no-overlap-p2p-communication', action='store_false',
                       help='overlap pipeline parallel communication with forward and backward chunks',
                       dest='overlap_p2p_comm')
    group.add_argument('--pipeline-model-parallel-split-rank',
                       type=int, default=None,
                       help='Rank where encoder and decoder should be split.')
    group.add_argument('--moe-expert-parallel-size', type=int, default=1,
                       help='Degree of the MoE expert parallelism.')
    group.add_argument('--model-parallel-size', type=int, default=None,
                       help='Old model parallel argument, do not use. Use '
                       '--tensor-model-parallel-size instead.')
    group.add_argument('--num-layers-per-virtual-pipeline-stage', type=int, default=None,
                       help='Number of layers per virtual pipeline stage')
    group.add_argument('--DDP-impl', default='local',
                       choices=['local', 'torch'],
                       help='which DistributedDataParallel implementation '
                       'to use.')
    group.add_argument('--no-contiguous-buffers-in-local-ddp',
                       action='store_false', help='If set, dont use '
                       'contiguous buffer in local DDP.',
                       dest='use_contiguous_buffers_in_local_ddp')
    group.add_argument('--no-scatter-gather-tensors-in-pipeline', action='store_false',
                       help='Use scatter/gather to optimize communication of tensors in pipeline',
                       dest='scatter_gather_tensors_in_pipeline')
    group.add_argument('--local-rank', type=int, default=None,
                       help='Local rank passed from distributed launcher for torch2.x.')
    group.add_argument('--use-cpu-initialization', action='store_true',
                       default=None, help='If set, affine parallel weights '
                       'initialization uses CPU')
    group.add_argument('--triangle-attn', action='store_true',
                       help="use triangle attention instead self attention")
    group.add_argument('--triangle-block-size', type=int, default=512,
                       help="set the size of triangle attention blocks")
    group.add_argument('--standalone-embedding-stage', action='store_true',
                       default=False, help='If set, *input* embedding layer '
                       'is placed on its own pipeline stage, without any '
                       'transformer layers. (For T5, this flag currently only '
                       'affects the encoder embedding.)')
    group.add_argument('--use-distributed-optimizer', action='store_true',
                       help='Use distributed optimizer.')
    return parser


def _add_ascend_autoresume_args(parser):
    group = parser.add_argument_group(title='autoresume')

    group.add_argument('--adlr-autoresume', action='store_true',
                       help='Enable autoresume on adlr cluster.')
    group.add_argument('--adlr-autoresume-interval', type=int, default=1000,
                       help='Intervals over which check for autoresume'
                       'termination signal')

    return parser


def _add_ascend_biencoder_args(parser):
    group = parser.add_argument_group(title='biencoder')

    # network size
    group.add_argument('--ict-head-size', type=int, default=None,
                       help='Size of block embeddings to be used in ICT and '
                        'REALM (paper default: 128)')
    group.add_argument('--biencoder-projection-dim', type=int, default=0,
                       help='Size of projection head used in biencoder (paper'
                        ' default: 128)')
    group.add_argument('--biencoder-shared-query-context-model', action='store_true',
                        help='Whether to share the parameters of the query '
                        'and context models or not')

    # checkpointing
    group.add_argument('--ict-load', type=str, default=None,
                       help='Directory containing an ICTBertModel checkpoint')
    group.add_argument('--bert-load', type=str, default=None,
                       help='Directory containing an BertModel checkpoint '
                       '(needed to start ICT and REALM)')

    # data
    group.add_argument('--titles-data-path', type=str, default=None,
                       help='Path to titles dataset used for ICT')
    group.add_argument('--query-in-block-prob', type=float, default=0.1,
                       help='Probability of keeping query in block for '
                       'ICT dataset')
    group.add_argument('--use-one-sent-docs', action='store_true',
                       help='Whether to use one sentence documents in ICT')
    group.add_argument('--evidence-data-path', type=str, default=None,
                       help='Path to Wikipedia Evidence frm DPR paper')

    # training
    group.add_argument('--retriever-report-topk-accuracies', nargs='+', type=int,
                        default=[], help="Which top-k accuracies to report "
                        "(e.g. '1 5 20')")
    group.add_argument('--retriever-score-scaling', action='store_true',
                       help='Whether to scale retriever scores by inverse '
                        'square root of hidden size')

    # faiss index
    group.add_argument('--block-data-path', type=str, default=None,
                       help='Where to save/load BlockData to/from')
    group.add_argument('--embedding-path', type=str, default=None,
                       help='Where to save/load Open-Retrieval Embedding'
                        ' data to/from')

    # indexer
    group.add_argument('--indexer-batch-size', type=int, default=128,
                       help='How large of batches to use when doing indexing '
                       'jobs')
    group.add_argument('--indexer-log-interval', type=int, default=1000,
                       help='After how many batches should the indexer '
                       'report progress')
    return parser


def _add_ascend_vit_args(parser):
    group = parser.add_argument_group(title="vit")

    group.add_argument('--num-classes', type=int, default=1000,
                       help='num of classes in vision classificaiton task')
    group.add_argument('--img-dim', type=int, default=224,
                       help='Image size for vision classification task')
    group.add_argument('--num-channels', type=int, default=3,
                       help='Number of channels in input image data')
    group.add_argument('--patch-dim', type=int, default=16,
                       help='patch dimension used in vit')

    return parser


def _add_ascend_zero_args(parser):
    """Text generate arguments."""

    group = parser.add_argument_group('ZeRO configurations', 'configurations')
    group.add_argument("--zero-stage", type=int, default=1.0)
    group.add_argument('--zero-reduce-scatter', action='store_true',
                       help='Use reduce scatter if specified')
    group.add_argument('--zero-contigious-gradients', action='store_true',
                       help='Use contigious memory optimizaiton if specified')
    group.add_argument("--zero-reduce-bucket-size", type=int, default=0.0)
    group.add_argument("--zero-allgather-bucket-size", type=int, default=0.0)
    group.add_argument('--remote-device', type=str, default='none', choices=['none', 'cpu', 'nvme'],
                      help='Remote device for ZeRO-3 initialized parameters.')
    group.add_argument('--use-pin-memory', action='store_true',
                     help='Use pinned CPU memory for ZeRO-3 initialized model parameters.')
    return parser


def _add_ascend_memoryopt_args(parser):
    """Memory optimization arguments."""

    group = parser.add_argument_group('Memory optimizations', 'configurations')
    group.add_argument("--scattered-embeddings", action='store_true',
                       help='Save memory by scattering embedding activations. '
                            'Introduces dropout differences across MP configurations.')
    group.add_argument("--split-transformers", action='store_true',
                       help='Save memory by splitting transformer layers into two parts, '
                       'allowing for more frequent activation checkpoint savings.')
    group.add_argument("--memory-centric-tiled-linear", action="store_true",
                       help='Save memory by tiling with deepspeed.zero.TiledLinear.')
    group.add_argument("--tile-factor", type=int, default=1,
                       help='Make all linear layers the same size of [hidden/tile_factor, hidden/tile_factor]. '
                            'Must be enabled with --memory-centric-tiled-linear. '
                            'Example A: if tile_factor=1, the qkv layer [hidden, 3* hidden] '
                            'would be converted into [1,3] tiles of size [hidden,hidden]. '
                            'Example B: if tile_factor=2, the intermediate layer [4*hidden, hidden] '
                            'will be converted into [8, 2] tiles of size [hidden/2, hidden/2]. '
                            'Default is 1.')

    return parser


def _add_ascend_activation_checkpoint_args(parser):
    group = parser.add_argument_group('Activation Checkpointing',
                                      'Checkpointing Configurations')
    group.add_argument('--deepspeed-activation-checkpointing', action='store_true',
                       help='uses activation checkpointing from deepspeed')
    group.add_argument('--partition-activations', action='store_true',
                       help='partition Activations across GPUs before checkpointing.')
    group.add_argument('--contigious-checkpointing', action='store_true',
                       help='Contigious memory checkpointing for activatoins.')
    group.add_argument('--checkpoint-in-cpu', action='store_true',
                       help='Move the activation checkpoints to CPU.')
    group.add_argument('--synchronize-each-layer', action='store_true',
                       help='does a synchronize at the beginning and end of each checkpointed layer.')
    group.add_argument('--profile-backward', action='store_true',
                       help='Enables backward pass profiling for checkpointed layers.')
    group.add_argument('--checkpoint-policy', type=str, default='full', choices=['full', 'block', 'custom'],
                       help="activation checkpoint policy")
    group.add_argument('--checkpoint_block_layer', type=int, default=25,
                       help="activation checkpoint block layer number")
    return parser


def _add_ascend_distillation_args(parser):
    group = parser.add_argument_group('Knowledge distillation',
                                      'Distillation Configurations')

    group.add_argument('--num-layers-teacher', type=int, default=None,
                       help='Number of the teacher transformer layers.')
    group.add_argument('--num-experts-teacher', type=int, nargs='+', default=[1,],
                        help='number of teacher experts list, MoE related.')
    group.add_argument('--hidden-size-teacher', type=int, default=None,
                       help='Tansformer teacher hidden size.')
    group.add_argument('--num-attention-heads-teacher', type=int, default=None,
                       help='Number of teacher transformer attention heads.')

    group.add_argument('--mos', action='store_true',
                       help='Enable Mixture-of-Students via knolwedge distillation.')
    group.add_argument('--kd', action='store_true',
                       help='Enable knolwedge distillation.')
    group.add_argument('--kd-alpha-ce', default=1, type=float)
    group.add_argument('--kd-beta-ce', default=1, type=float)
    group.add_argument('--kd-temp', default=1.0, type=float)
    group.add_argument('--reset-iteration', action='store_true',
                    help='Reset the iteration count.')

    group.add_argument('--load-teacher', type=str, default=None,
                       help='Directory containing a teacher model checkpoint.')

    return parser


def _add_ascend_optimized_pipeline_args(parser):
    group = parser.add_argument_group(title='optimized_pipeline')

    group.add_argument('--optimized-pipeline', action='store_true',
                       help='Enable optimized pipeline for bubble time reduction.')
    group.add_argument('--manual-mbs', type=str, default='',
                       help='Dynamic micro batches for optimized pipeline. '
                            'The format shoud be a sequence of numbers seperated by '
                            'comma; e.g., 4,4,4,4. Two examples are provided by '
                            '--manual-mbs example-config-1, and '
                            '--manual-mbs example-config-2')

    return parser


def _add_ascend_manual_layer_allocation(parser):
    group = parser.add_argument_group(title='manual_layer_allocation')
    group.add_argument('--use-manual-layer-allocation', action='store_true',
                       help='Enable manually allocated layers for pipeline model parallel.')
    group.add_argument('--manual-layers', type=str, help='a list of number of layers, '
                                                         'seperated by comma; e.g., 4,4,4,4')

    return parser



def _add_ascend_transformer_engine_args(parser):
    group = parser.add_argument_group(title='Transformer-Engine')

    group.add_argument('--transformer-impl', default='local',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.',
                       dest='transformer_impl')

    return parser


def _add_ascend_lora_args(parser):
    group = parser.add_argument_group(title='lora')

    group.add_argument('--lora-target-modules', nargs='+', type=str, default=[],
                       help='Lora target modules.')
    group.add_argument('--lora-load', type=str, default=None,
                       help='Directory containing a lora model checkpoint.')
    group.add_argument('--lora-r', type=int, default=16,
                       help='Lora r.')
    group.add_argument('--lora-alpha', type=int, default=32,
                       help='Lora alpha.')
    group.add_argument('--lora-modules-to-save', nargs='+', type=str, default=None,
                       help='Lora modules to save.')
    group.add_argument('--lora-register-forward-hook', nargs='+', type=str,
                       default=['word_embeddings', 'input_layernorm'],
                       help='Lora register forward hook.')
    group.add_argument('--lora-adapter-name', type=str, default='default',
                       help='Lora adapter name.')

    return parser


def _add_ascend_retro_args(parser):
    from llm.utils.general.error_utils import (
        ensure_valid
    )
    group = parser.add_argument_group(title='retro')

    group.add_argument('--retro-workdir', default=None,
                       help='Retro working directory, which contains the '
                            'preprocessed data for for pretraining. This directory '
                            'is built during preprocessing (see '
                            'tools/retro/README.md), and contains subdirectories '
                            'for the chunk database and pretraining neighbors.')
    group.add_argument('--retro-add-retriever',
                       action='store_true', default=False,
                       help='Add a retriever to the transformer, for use in '
                       'pretraining a Retro model.')
    group.add_argument('--retro-encoder-layers', type=int, default=2,
                       help='Number of layers to use for the retrieval '
                       'encoder.')
    group.add_argument('--retro-encoder-hidden-dropout',
                       type=float, default=0.1, help='Hidden dropout for '
                       'retrieval encoder.')
    group.add_argument('--retro-encoder-attention-dropout',
                       type=float, default=0.1, help='Attention dropout for '
                       'retrieval encoder.')
    group.add_argument("--retro-num-neighbors", type=int, default=2,
                       help='Number of neighbors to retrieve during '
                       'pretraining.')
    group.add_argument("--retro-num-retrieved-chunks", type=int, default=2,
                       help='Number of chunks to retrieve from the retrieval '
                       'database.')
    group.add_argument("--retro-return-doc-ids", action="store_true",
                       help="Turn this on when preprocessing retro data.")

    # Enforce argument naming convention.
    for action in group._group_actions:
        prefix = action.dest.split("_")[0]
        ensure_valid(prefix == "retro", "Retro args must be prefixed with '--retro-*', for consistent " \
                                        "styling. Please fix '%s'." % ", ".join(action.option_strings))

    return parser

