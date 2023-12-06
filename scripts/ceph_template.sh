#!/bin/bash
set -x -e -o pipefail

export NCCL_DEBUG=WARN
T=`date +%m%d%H%M`

ROOT=../
export PYTHONPATH=$ROOT:$ROOT/llm/utils/tools:$PYTHONPATH

# Please export them, if you use ceph to load or save model
export PYTHONPATH=path/to/DeepSpeed:$PYTHONPATH
# export PYTHONPATH=path/to/transformers/src:path/to/peft/src:$PYTHONPATH
export PETRELPATH=your_petreloss.conf
export CEPHBUCKET=s3://your_ceph_bucket

echo "START TIME: $(date)"

# slurm launch
export CMD="python \
    $ROOT/llm/runners/base_llm_runner.py \
    --config $2" 

echo $CMD

srun --partition=$1 --mpi=pmi2 -N 1 -n8 --gres=gpu:8 --quotatype=spot --cpus-per-task=16  bash -c "$LAUNCHER $CMD"  2>&1 | tee -a mg_train_log.txt

echo "END TIME: $(date)"
