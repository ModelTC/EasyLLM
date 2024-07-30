set -x -e

ROOT=/path/to/easyllm

export PYTHONPATH=$ROOT:$PYTHONPATH

# internlm2 20b example
python $ROOT/tools/convert_ds_weight_to_hf_fast.py \
	--save_intern \
	--no_save_config \
	--tp_size 4 \
	--dim 6144 \
	--n_heads 48 \
	--n_layers 48 \
	--norm_eps 1e-5 \
	--num_key_value_heads 8 \
	--intermediate_size 16384 \
	--rope_theta 1000000 \
	--max_position_embeddings 32768 \
	--input_dir $1 \
	--output_dir $2
