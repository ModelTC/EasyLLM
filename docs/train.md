### Huggingface

Suitable for LLMs with parameter sizes of 7b, 13b, 20b, 33b, supporting SFT and LoRA fine-tuning.

```shell
cd scripts
# slurm
bash hf_train.sh [partition] config.yaml
# torch
bash hf_train.sh config.yaml
```

To initiate LoRA Fine-tuning, the relevant LoRA parameters need to be set in the configuration file under the 'model' configuration. Taking Llama as an example

```yaml
...

model:
  type: LlamaForCausalLM
  kwargs:
    model_name_or_path: *file_path
    torch_dtype: bfloat16
    trust_remote_code: True
  # if using lora
  peft_model_cfg:
    lora_rank: 8
    lora_alpha: 32
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
    modules_to_save: ["embed_tokens", "lm_head"]
    lora_dropout: 0.05
```

### Megatron

Suitable for LLMs with a parameter size of 65b or above, supporting SFT and LoRA Fine-tuning.

```shell
cd scripts
# slurm
bash mg_train.sh [partition] config.yaml
# torch
bash mg_train.sh config.yaml
```

To initiate LoRA Fine-tuning, it is necessary to set LoRA-related parameters in the configuration file, using Llama as an example

```yaml
runtime:
  seed: &seed 42
  tensor_model_parallel_size: 4
  pipeline_model_parallel_size: 2
  deepspeed: True
  lora_mode: True  # True open lora training
  bf16: True
  dynamic: True  # input size is dynamic or not

...

lora:
  lora_rank: 8
  lora_alpha: 32
  lora_dropout: 0.05
  base_model_name_or_path: base/model/path
  target_modules: ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']
  saver:
    modules_to_save: ['word_embeddings', 'lm_head']
    only_save_trainable: True
    save_path: checkpoints/lora  # lora save path
    save_mode: deepspeed  # huggingface/deepspeed， save format
```