## Evaluation

**EasyLLM supports interactive dialogue and dataset evaluation, including custom datasets and the following publicly available datasets:**

  - Examination: **C-Eval**/**CMMLU**
  - Reasoning: **HumanEval**

### Huggingface

- Interactive conversation

```shell
cd scripts
bash hf_infer.sh [partition] [n_gpus] config.yaml interactive
```

**To structure the input prompt format, set the 'infer_tokenization' setting in the configuration file, as illustrated below. (Supported prompt formats are described in the following section):**

```yaml
...

infer_tokenization: 
  type: sense_tokenization
  kwargs:
    max_seq_length: *train_seq_length 
    parser_type: base
    parser_kwargs:
      prompt_type: empty

...
```

- Dataset Evaluation

```shell
cd scripts
bash hf_infer.sh [partition] [n_gpus] config.yaml eval  
```

**To configure the 'infer_cfg' and 'infer_tokenization' settings in the configuration file, an example of the 'infer_cfg' configuration is provided below:**

  - eval_task: Supports four types of dataset evaluation

    - base: Custom dataset 
    - ceval|cmmlu|human_eval: Open Dataset

  - question_file: evaluation dataset paths, categorized into four cases based on the eval_task.

    - base: Specify up to the .jsonl level, and refer to the dataset format section for the format details.
    - ceval: OpenCompassData/data/ceval/formal_ceval [OpenCompassData donwload link](https://github.com/open-compass/opencompass/releases/download/0.1.8.rc1/OpenCompassData-core-20231110.zip)
    - cmmlu: OpenCompassData/data/cmmlu
    - human_eval: no need for specification.

```yaml
infer_cfg:
  eval_task: [base|ceval|cmmlu|human_eval]
  question_file: questions.jsonl
  result_file: results.jsonl  # evaluation results file path
  generation_cfg:  # inference config
    temperature: 0.2
    top_k: 40
    top_p: 0.9
    do_sample: True
    num_beams: 1
    repetition_penalty: 1.3
    max_new_tokens: 512
```

  - The format of the evaluation results.

  ```yaml
  {
     "0"[integer, task_id]: {
        "input": "xxx",                   # input text
        "raw_output": "xxx",              # raw model output
        "output": "xxx",                  # output after post process
        "answer": "xxx",                  # answer labels
        "infos": {                        # infos
            "count": 33,                  # decode count
            "accept_length": 33,          # output token num
            "ave_accept_length": 1        # token per decode
        }
     },
     "1": {
        "input": "xxx",
        "raw_output": "xxx",
        "output": "xxx",
        "answer": "xxx",
        "infos": {
           ...
        }
     },
     ...
  }
  ```

### Megatron

**Refer to the above HuggingFace section for the configuration file settings.**

- Interactive conversation

```shell
cd scripts
bash mg_infer.sh [partition] [n_gpus] config.yaml interactive
```

- Dataset Evaluation

```shell
cd scripts
bash mg_infer.sh [partition] [n_gpus] config.yaml eval