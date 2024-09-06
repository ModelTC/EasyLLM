#!/bin/bash
set -x -e 


export NCCL_DEBUG=INFO
export CUDA_DEVICE_MAX_CONNECTIONS=1
export DIST_BACKEND=megatron
export ACCELERATOR_BACKEND=CUDA

T=`date +%m%d%H%M`
ROOT=/mnt/afs_2/liangkaihuan/Codes/lm-toolchain/easyllm
MG_CORE=/mnt/afs_2/liangkaihuan/Codes/lm-toolchain/megatron-lm
# DS=/mnt/afs_2/yaoyongqiang/exps/code/DeepSpeed_v0_12_3
DS=/mnt/afs_2/liangkaihuan/Codes/lm-toolchain/DeepSpeed-mtc
export PYTHONPATH=$ROOT:$MG_CORE:$ROOT/llm/utils/tools:$DS:$PYTHONPATH

echo "START TIME: $(date)"
# MASTER_ADDR=127.0.0.1
MASTER_ADDR=$(awk '/master-0/{print $1}' /etc/hosts | uniq)
# MASTER_ADDR=10.119.17.82
MASTER_PORT=6000
mkdir -p logs

GPUS_PER_NODE=8
NNODES=$2
config=$1

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    "
    #--tee 3 \

export CMD=" \
    $ROOT/llm/plugins/megatron_lm/runners/megatron_runner_vlm.py \
    --config $config \
    --launcher torch" 

echo $CMD

bash -c "$LAUNCHER $CMD" 2>&1 | tee logs/mg_train_$(basename $config).$3.log
#> logs/mg_train_$(basename $config).$3.log 2>&1 
#bash -c "$LAUNCHER $CMD"  2>&1 | tee logs/mg_train_$(basename $config).$2.log

echo "END TIME: $(date)"

