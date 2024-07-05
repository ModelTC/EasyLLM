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


def parse_args():
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Megatron-LM Arguments',
                                     allow_abbrev=False)

    # Standard arguments.
    parser = _add_training_args(parser)
    parser = _add_inference_args(parser)
    parser = _add_medusa_args(parser)
    parser = _add_distributed_args(parser)
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

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
                       choices=['nccl', 'gloo', 'hccl'],
                       help='Which backend to use for distributed training.')
    group.add_argument('--local-rank', type=int, default=None,
                       help='local rank passed from distributed launcher.')
    group.add_argument('--lazy-mpu-init', type=bool, required=False,
                       help='If set to True, initialize_megatron() '
                       'skips DDP initialization and returns function to '
                       'complete it instead. This is for '
                       'external DDP manager.')
    return parser
