## Dataset Format

**EasyLLM supports two types of dataset formats.**

- Single dialogue
- Multi dialogue

### Single dialogue

```yaml
{
  "system": "xxx",
  "instruction": "xxx",
  "input": "xxx",
  "output": "xxx"
}
```

### Multi dialogue

single or mulit dialogue

```yaml
[
  {
    "role": "system", "content": "xxx"
  },
  {
    "role": "knowledge", "content": "xxx"
  },
  {
    "role": "user", "content": "xxx"
  },
  {
    "role": "assistant", "content": "xxx"
  }
  {
    "role": "user", "content": "xxx"
  },
  {
    "role": "assistant", "content": "xxx"
  }
  ......
]
```

### Data format

EasyLLM supports two types of dataset loading, corresponding to the organizational structure of two dataset formats (both in JSON format).

- all: （json.dump）
- line: (json.dumps)

#### all

- Dataset organizational structure.

```yaml
[
  item 1(dict/list),
  item 2,
  item 3,
  ...
]
```

- config setting

```yaml
dataset:
  type: base_nlp_json
  kwargs:
    json_file: /path/to/data.jsonl
    transformer: [*tokenization]
    json_type: all  
```

#### line

- Dataset organizational structure.

```yaml
item 1(dict/list)\n
item 2\n
item 3\n
...
```

- config setting

```yaml
dataset:
  type: base_nlp_json
  kwargs:
    json_file: /path/to/data.jsonl
    transformer: [*tokenization]
    json_type: line  # 
```

## Prompt format

**EasyLLM currently supports several prompt formats, which can be configured through the configuration file. The following provides a detailed introduction and usage examples:**
### base

The 'base' mode supports prompt formats without prompts and several prompts from open-source large models, including Internlm, Qwen, and Baichuan. The prompt formats are as follows:

  - empty: {raw_input_text}
  - intern: <|User|>:{raw_input_text}\n<|Bot|>:
  - qwen: <|im_start|>user\n{raw_input_text}<|im_end|>\n<|im_start|>assistant\n
  - baichuan2: {tokenizer.decode(195)}{raw_input_text}{tokenizer.decode(196)}

config setting

```yaml
tokenization: 
  type: sense_tokenization
  kwargs:
    max_seq_length: 2048  
    parser_type: base
    parser_kwargs:
      prompt_type: empty  # support types: empty/intern/qwen/baichuan2
      inference_mode: False  
```

### simple_chat

prompt format: "<|System>|>:{system_prompt}{history_text}<|Human|>:{user_new_query}\n<|Assistant|>:"

```yaml
tokenization:
  type: sense_tokenization
  kwargs:
    max_seq_length: 2048
    parser_type: simple_chat
    parser_kwargs:
      inference_mode: False
      ignore_index: -100
      keep_all_keys: False
      only_last_answer: False
      prompt_template:
        system_prompt: "<|System>|>:"
        question_prompt: "<|Human|>:"
        answer_prompt: "<|Assistant|>:"
        ...
```

### preprocess

prompt format: "{input}{output}", input is our question include system prompt and history dialogue and other..., out is our final assistant answer

```yaml
tokenization: &tokenization
  type: sense_tokenization
  kwargs:
    with_tokenizer: True
    max_seq_length: &train_seq_length 4096
    parser_type: preprocess
    parser_kwargs:
       keep_all_keys: False
       drop_meta: True # drop meta out of seq length
```


**Parser-Code**

```python
@PARSER_REGISTRY.register('preprocess')
class PreProcessParser(object):
    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 ignore_index=-100,
                 keep_all_keys=False,
                 inference_mode=False,
                 drop_meta=False):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.keep_all_keys = keep_all_keys
        self.max_seq_length = max_seq_length
        self.inference_mode = inference_mode
        self.drop_meta = drop_meta

    def __call__(self, meta):
        question = meta['inputs']
        answer = meta.get('outputs', "")
        tokenized_question = self.tokenizer(question, return_attention_mask=False)['input_ids']
        tokenized_answer = self.tokenizer(answer, return_attention_mask=False, add_special_tokens=False)['input_ids']
        if self.keep_all_keys:
            labels = tokenized_question + tokenized_answer
        else:
            labels = [self.ignore_index] * len(tokenized_question) + tokenized_answer
        if self.inference_mode:
            return tokenized_question + tokenized_answer, []
        else:
            tokenized_text = tokenized_question + tokenized_answer + [self.tokenizer.eos_token_id]
            labels = labels + [self.tokenizer.eos_token_id]
            if self.drop_meta and len(tokenized_text) > self.max_seq_length:
                return None
            # drop question to avoid no loss
            tokenized_text = tokenized_text[-self.max_seq_length:]
            labels = labels[-self.max_seq_length:]
            input_ids = torch.LongTensor(tokenized_text)
            labels = torch.LongTensor(labels)
            results = {'input_ids': input_ids, 'labels': labels}
        return results
```