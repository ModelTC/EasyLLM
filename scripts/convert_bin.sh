set -x -e -o pipefail

DATA_ROOT=your/data/root
PATH_LIST=your/path/list
TOKENIZER_PATH=your/tokenizer/path (fast_mode)
OUT_FOLDER=your/output/root
GROUP=1
GROUP_ID=0
WORKER=32
BIN_SIZE=262144
ROOT=../
export PYTHONPATH=$ROOT:$ROOT/llm/utils/tools:$PYTHONPATH


export CMD="python -u \
    $ROOT/tools/build_pretrain_bin.py \
    --data_root $DATA_ROOT \
    --path_list $PATH_LIST \
    --tokenizer_path $TOKENIZER_PATH \
    --out_folder $OUT_FOLDER \
    --group $GROUP \
    --group_id $GROUP_ID \
    --worker $WORKER \
    --bin_size $BIN_SIZE"

bash -c "$CMD"
