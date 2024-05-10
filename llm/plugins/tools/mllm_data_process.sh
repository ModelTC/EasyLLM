export PYTHONPATH=/mnt/afs_2/zhangfeizhao/mllm/internvl/easyllm:$PYTHONPATH

python mllm_data_process.py --config $1 --token_lengths_path /mnt/afs_2/zhangfeizhao/mllm/internvl/workdir/data_process
