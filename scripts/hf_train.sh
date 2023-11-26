set -x -e -o pipefail

ROOT=../
export PYTHONPATH=$ROOT:$ROOT/llm/utils/tools:$PYTHONPATH
export PYTHONPATH=../../transformers/src:../../DeepSpeed:../../peft/src:$PYTHONPATH

MASTER_ADDR=127.0.0.1
MASTER_PORT=6004

GPUS_PER_NODE=8
NNODES=1


# for slurm
export CMD=" python \
    $ROOT/llm/runners/hf_runner.py \
    --config $2 \
    --launcher slurm"

echo $CMD

srun --partition=$1 --mpi=pmi2 -N 1 -n8 --gres=gpu:8 --quotatype=spot --cpus-per-task=16  bash -c "$CMD"  2>&1 | tee -a hf_train_log.txt

echo "END TIME: $(date)"


# for torch
# export LAUNCHER="python -u -m torch.distributed.run \
#     --nproc_per_node $GPUS_PER_NODE \
#     --nnodes $NNODES \
#     --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
#     --rdzv_backend c10d \
#     --max_restarts 0 \
#     --tee 3 \

# export CMD="\
#     $ROOT/llm/runners/hf_runner.py \
#     --config $1 \
#     --launcher torch"

# echo $CMD

# bash -c "$LAUNCHER $CMD"  2>&1 | tee -a main_log.txt

# echo "END TIME: $(date)"


