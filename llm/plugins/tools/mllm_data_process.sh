EASYLLM=/mnt/afs_2/zhangfeizhao/temp/easyllm

export PYTHONPATH=$EASYLLM:$PYTHONPATH
export OMP_NUM_THREADS=1


python mllm_data_process.py --config $1 --json_file $2 --token_lengths_path $3 --output_path $4 2>&1 | tee -a log_statistics.txt
