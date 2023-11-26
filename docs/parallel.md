3D parallelism parameters.

### megatron

  - Tensor Parallel
  - Pipeline Parallel
  - GPU Num = DP * TP * PP
  - sequence_parallel Input alignment needs to be a multiple of the tensor processing size.

```yaml
runtime:
  seed: &seed 42
  tensor_model_parallel_size: 4  # TP
  pipeline_model_parallel_size: 2  # PP
  ...
```
**sequence parallel config example**

```yaml
model:
  type: llama2_70b
  kwargs:
     use_flash_attn: True
     sequence_parallel: True
```