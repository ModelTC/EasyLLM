# EasyLLM

Built upon Megatron-Deepspeed and HuggingFace Trainer, EasyLLM has reorganized the code logic with a focus on usability. While enhancing usability, it also ensures training efficiency.

## Install

  - Install python requirements

    ```shell
    pip install -r requirements.txt
    ```
    other dependency
    * flash-attn (dropout_layer_norm) (maybe you need to compile it by yourself)

  - Pull deepspeed & add them to pythonpath

    ```shell
    export PYTHONPATH=/path/to/DeepSpeed:$PYTHONPATH
    ```

## Train

[Train Example](./docs/train.md)

## Infer and Eval

[Infer Example](./docs/infer.md)

## Support Models
* qwen14b,
* internlm7-20b,
* baichuan1/2 (7b-13b)
* llama1-2 (7b/13b/70b)

[Model Example](./docs/model.md)

## Data

[Data Example](./docs/data.md)

## 3D Parallel config setting

[Parallel Example](docs/parallel.md)

## Speed Benchmark

[Speed Benchmark](docs/benchmark.md)

## Dynamic Checkpoint

To optimize the model training performance in terms of time and space, EasyLLM supports Dynamic Checkpoint. Based on the input token size, it enables checkpointing for some layers. The configuration file settings are as follows:

[Dynamic Checkpoint Example](docs/dc.md)

## License

This repository is released under the [Apache-2.0](LICENSE) license.

## Acknowledgement

We learned a lot from the following projects when developing EasyLLM.
- [DeepSpeed](https://github.com/microsoft/DeepSpeed.git)
- [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed.git)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM.git)
- [Flash Attention 1&2](https://github.com/Dao-AILab/flash-attention)
- [LightLLM](https://github.com/ModelTC/lightllm)
- [Huggingface Transformers](https://github.com/huggingface/transformers.git)