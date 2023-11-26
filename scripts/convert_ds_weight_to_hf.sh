set -x -e

ROOT=../
export PYTHONPATH=$ROOT:$ROOT/llm/utils/tools:$PYTHONPATH
export PYTHONPATH=../../transformers/src:../../DeepSpeed:../../peft/src:$PYTHONPATH

input_dir=$1
output_dir=$2

# 13b example
export CMD = "python \
        $ROOT/tools/convert_ds_weight_to_hf.py \
        --tp_size 1 \
        --dim 5120 \
        --n_heads 40 \
        --n_layers 60 \
        --norm_eps 1e-5 \ 
        --num_key_value_heads 40 \
        --intermediate_size -1 \ # comupte by itself
        --output_dir $output_dir \
        --input_dir $input_dir "

# 7b example
# export CMD = "python $ROOT/tools/convert_ds_weight_to_hf.py \
#        --tp_size 1 \
#        --dim 4096 \
#        --n_heads 32 \
#        --n_layers 32 \
#        --norm_eps 1e-5 \
#        --num_key_value_heads 32 \
#        --intermediate_size 11008 \
#        --output_dir $output_dir \
#        --input_dir $input_dir "

# 70b example
# export CMD = "python $ROOT/tools/convert_ds_weight_to_hf.py \
#        --tp_size 4 \
#        --dim 8192 \
#        --n_heads 64 \
#        --n_layers 80 \
#        --norm_eps 1e-5 \
#        --num_key_value_heads 8 \
#        --intermediate_size 28672 \
#        --output_dir $output_dir \
#        --input_dir $input_dir "