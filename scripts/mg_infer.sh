#!/bin/bash
set -x -e

# export NCCL_COLLNET_ENABLE=1
export NCCL_DEBUG=WARN
T=`date +%m%d%H%M`

ROOT=../
export PYTHONPATH=$ROOT:$ROOT/llm/utils/tools:$PYTHONPATH
# Please export them, if you use ceph to load or save model
export PYTHONPATH=path/to/DeepSpeed:$PYTHONPATH
# export PYTHONPATH=path/to/transformers/src:path/to/peft/src:$PYTHONPATH

echo "START TIME: $(date)"

# so processes know who to talk to
# MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_ADDR=127.0.0.1
MASTER_PORT=6000

GPUS_PER_NODE=$2
NNODES=1

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "

export CMD=" \
    $ROOT/llm/runners/base_llm_runner.py \
    --launcher torch \
    --config $3 \
    --deepspeed \
    --inference \
    --generate-mode $4 \
    --recompute"
# --generate-mode eval  for evaluate dataset
# --generate-mode interactive  for interactive
echo $CMD

# do not remove or the training will hang and nodes will be lost w/o this workaround
# export CUDA_LAUNCH_BLOCKING=1

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
# export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1


srun --partition=$1 --mpi=pmi2 -N 1 --gres=gpu:$GPUS_PER_NODE --quotatype=spot --cpus-per-task=16  bash -c "$LAUNCHER $CMD"  2>&1 | tee mg_infer_log.txt

echo "END TIME: $(date)"
