## Support Models

### Megatron

**Currently, the supported models are Llama and Llama2, with the following parameter counts:**

- Llama: llama_7b/llama_13b/llama_20b/llama_65b
- Llama2: llama2_7b/llama2_13b/llama2_70b

**An example of the configuration file is as follows:**

```yaml
model:
  type: llama_7b  # model type

# or
model:
  type: llama_custom
  kwargs:
    num_layers: 32
    hidden_size: 4096
    num_attention_heads: 32
```

### Huggingface

Currently, the supported models include Baichuan, Baichuan2, Internlm, Llama, and Qwen

An example of the configuration file is provided below:

- Baichuan

```yaml
model:
  type: BaiChuanForCausalLM
  kwargs:
    model_name_or_path: "./file_path"
    torch_dtype: auto/bfloat16/float16/float32
    trust_remote_code: True  
    _flash_attn_2_enabled: True 
    _flash_norm_2_enabled: True 
```
- Baichuan2: 支持7b与13b

  - 7b

  ```yaml
  model:
    type: BaiChuan2ForCausalLM
    kwargs:
      model_name_or_path: "./file_path"
      torch_dtype: auto/bfloat16/float16/float32
      trust_remote_code: True  
      _flash_attn_2_enabled: True   
      _flash_norm_2_enabled: True  
  ```

  - 13b(FlashAttention2 is not be supported)

  ```yaml
  model:
    type: BaiChuan2ForCausalLM13B
    kwargs:
      model_name_or_path: "./file_path"
      torch_dtype: auto/bfloat16/float16/float32
      trust_remote_code: True  
  ```

- internlm

```yaml
model:
  type: InternLMForCausalLM
  kwargs:
    model_name_or_path: "./file_path"
    torch_dtype: auto/bfloat16/float16/float32
    trust_remote_code: True  
    _flash_attn_2_enabled: True  
    _flash_norm_2_enabled: True  
```

- Llama

```yaml
model:
  type: LlamaForCausalLM
  kwargs:
    model_name_or_path: "./file_path"
    torch_dtype: auto/bfloat16/float16/float32
    trust_remote_code: True
    _flash_attn_2_enabled: True  
    _flash_norm_2_enabled: True
```

- Qwen
```yaml
model:
  type: QWenForCausalLM
  kwargs:
    model_name_or_path: "./file_path"
    torch_dtype: auto/bfloat16/float16/float32
    trust_remote_code: True  
    _flash_attn_2_enabled: True
    _flash_norm_2_enabled: True
```
