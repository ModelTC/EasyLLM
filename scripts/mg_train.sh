#!/bin/bash
set -x -e -o pipefail

export NCCL_DEBUG=WARN
T=`date +%m%d%H%M`

ROOT=../
export PYTHONPATH=$ROOT:$ROOT/llm/utils/tools:$PYTHONPATH
# Please export them, if you use ceph to load or save model
export PYTHONPATH=path/to/DeepSpeed:$PYTHONPATH
# export PYTHONPATH=path/to/transformers/src:path/to/peft/src:$PYTHONPATH

echo "START TIME: $(date)"

# for slurm
export CMD=" python \
    $ROOT/llm/runners/base_llm_runner.py \
    --config $2 \
    --launcher slurm" 

echo $CMD

srun --partition=$1 --mpi=pmi2 -N 1 -n8 --gres=gpu:8 --quotatype=spot --cpus-per-task=16  bash -c "$CMD"  2>&1 | tee -a mg_train_log.txt

echo "END TIME: $(date)"

# for torch
# so processes know who to talk to
# MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)  # for multi node
# MASTER_ADDR=127.0.0.1
# MASTER_PORT=6000

# GPUS_PER_NODE=8
# NNODES=1

# export LAUNCHER="python -u -m torch.distributed.run \
#     --nproc_per_node $GPUS_PER_NODE \
#     --nnodes $NNODES \
#     --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
#     --rdzv_backend c10d \
#     --max_restarts 0 \
#     --tee 3 \
#     "

# export CMD=" \
#     $ROOT/llm/runners/base_llm_runner.py \
#     --config $1 \
#     --launcher torch"

# echo $CMD

# bash -c "$LAUNCHER $CMD"  2>&1 | tee -a mg_train_log.txt

# echo "END TIME: $(date)"



