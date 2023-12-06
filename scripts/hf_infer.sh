set -x -e

ROOT=../
export PYTHONPATH=$ROOT:$ROOT/llm/utils/tools:$PYTHONPATH
# Please export them, if you use ceph to load or save model
export PYTHONPATH=path/to/DeepSpeed:$PYTHONPATH
# export PYTHONPATH=path/to/transformers/src:path/to/peft/src:$PYTHONPATH

export CMD=" python \
    $ROOT/llm/runners/hf_runner.py \
    --config $3 \
    --inference \
    --generate-mode $4 \
    --port 13334"
# --generate-mode eval  for evaluate dataset
# --generate-mode interactive  for interactive
echo $CMD

srun --partition=$1 --mpi=pmi2 -N 1 -n$2 --gres=gpu:$2 --quotatype=spot --cpus-per-task=16  bash -c "$CMD"  2>&1 | tee -a hf_infer_log.txt

echo "END TIME: $(date)"
