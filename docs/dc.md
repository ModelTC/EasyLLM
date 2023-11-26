## Dynamic checkpoint 

* Huggingface

```yaml
hooks:
  - type: dynamic_checkpoint
    kwargs:
      enable: True  
      debug_freq: 10
      strategy:
        type: predefine
        kwargs:
          size_map:
            512: 0  # checkpointing layer number base input size
            1024: 16
            2048: 16
```

* Megatron

```yaml
model:
  type: llama2_70b
  kwargs:
     use_flash_attn: True
     sequence_parallel: True
     pp_partition_method: parameters
     dynamic_checkpoint:
        enabled: True
        size_map:
            512: 0
            1024: 0
            2048: 0
            4096: 0
            8192: 0
```