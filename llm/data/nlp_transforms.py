import torch

from llm.utils.general.registry_factory import PARSER_REGISTRY, AUGMENTATION_REGISTRY


class NLPCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


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


@PARSER_REGISTRY.register('base')
class BaseParser(object):
    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 ignore_index=-100,
                 keep_all_keys=False,
                 inference_mode=False,
                 prompt_type="empty",
                 drop_meta=False):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.keep_all_keys = keep_all_keys
        self.max_seq_length = max_seq_length
        self.inference_mode = inference_mode
        assert prompt_type in ["empty", "llama", "qwen", "intern", "baichuan2"], f"{prompt_type} has not supported."
        self.prompt_type = prompt_type
        self.drop_meta = drop_meta

    def build_input(self, raw_input_text):
        if self.prompt_type == "empty":
            prompt = raw_input_text
        elif self.prompt_type == "llama":
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n\n{raw_input_text}\n\n### Response:\n\n" # noqa
        elif self.prompt_type == "qwen":
            # pre_system = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
            # prompt = f"\n<|im_start|>user\n{raw_input_text}<|im_end|>\n<|im_start|>assistant\n"
            prompt = f"user\n{raw_input_text}<|im_end|>\n<|im_start|>assistant\n"
        elif self.prompt_type == "intern":
            prompt = f"<|User|>:{raw_input_text}<eoh>\n<|Bot|>:"
        elif self.prompt_type == "baichuan2":
            prompt = f"{self.tokenizer.decode(195)}{raw_input_text}{self.tokenizer.decode(196)}"
        return prompt

    def __call__(self, meta):
        text = self.build_input(meta['text'])
        tokenized_text = self.tokenizer(text, return_attention_mask=False)['input_ids']
        if self.inference_mode:
            return tokenized_text, tokenized_text
        else:
            tokenized_text += [self.tokenizer.eos_token_id]
            if self.drop_meta and len(tokenized_text) > self.max_seq_length:
                return None
            # drop question to avoid no loss
            input_ids = torch.LongTensor(tokenized_text)[-self.max_seq_length:]
            labels = input_ids.clone()
            results = {'input_ids': input_ids, 'labels': labels}
        return results


@PARSER_REGISTRY.register('simple_chat')
class SimpleChatParser(object):
    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 ignore_index=-100,
                 keep_all_keys=False,
                 only_last_answer=False,
                 prompt_template={},
                 inference_mode=False,
                 drop_meta=False):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.keep_all_keys = keep_all_keys
        self.max_seq_length = max_seq_length
        self.only_last_answer = only_last_answer
        self.system_prompt = prompt_template.get('system_prompt', "<|System>|>:")
        self.question_prompt = prompt_template.get('qustion_prompt', "<|Human|>:")
        self.answer_prompt = prompt_template.get('answer_prompt', "<|Assistant|>:")
        self.predefine_prompt = prompt_template.get('predefine_prompt', '')
        self.system_prompt = self.system_prompt + self.predefine_prompt
        self.inference_mode = inference_mode
        self.eoh = prompt_template.get('eoh', "\n")
        self.eosys = prompt_template.get('eosys', "\n")
        self.eoa = prompt_template.get('eoa', '')
        self.drop_meta = drop_meta

    def convert2chat(self, meta):
        new_meta = []
        user, assistant = {}, {}
        if 'system' in meta:
            system = {}
            system['role'] = 'system'
            system['content'] = meta['system']
            new_meta.append(system)
        user['role'] = 'user'
        user['content'] = meta.get('instruction', '') + meta.get('input', "")
        assistant['role'] = 'assistant'
        assistant['content'] = meta.get('output', "")
        new_meta += [user, assistant]
        return new_meta

    def _process_meta(self, meta):
        '''
            process meta info:
            return
                system prompt
                history
                question
                answer
        '''
        system_prompt_context = self.system_prompt
        if isinstance(meta, dict):
            if 'messages' in meta:
                if 'system' in meta:
                    system_prompt_context += meta['system']
                meta = meta['messages']
            if 'input' in meta:
                meta = self.convert2chat(meta)
        answer, question = "", ""
        dialog_history = []
        if len(meta) <= 1:
            for item in meta:
                if item['role'] == 'user':
                    question += item.get('content', '')
                if item['role'] == 'assistant':
                    answer += item.get('content', '')
        else:
            if self.inference_mode:
                question = meta[-1]['content']
                dialog_history = meta[:-1]
            else:
                answer = meta[-1]['content']
                question = meta[-2]['content']
                dialog_history = meta[:-2]
        for item in dialog_history:
            if item['role'] == "system":
                system_prompt_context += item['content']
        rt_history = []
        for hist in dialog_history:
            if hist['role'] == "user" or hist['role'] == "assistant":
                rt_history.append(hist)
        system_prompt_context = "{}{}".format(system_prompt_context, self.eosys)
        return system_prompt_context, rt_history, question, answer

    def _get_system_tokens_labels(self, system):
        tokens_system = self.tokenizer(system, return_attention_mask=False)['input_ids']
        labels_system = [self.ignore_index] * len(tokens_system)
        return tokens_system, labels_system

    def _get_history_tokens_labels(self, history):
        tokens_history = []
        labels_history = [self.ignore_index] * len(tokens_history)
        for idx, item in enumerate(history):
            if item['role'] == "user":
                user_context = "{}{}{}".format(self.question_prompt, item['content'], self.eoh)
                token_user_context = self.tokenizer(user_context, return_attention_mask=False, add_special_tokens=False)['input_ids']  # noqa
                tokens_history += token_user_context
                labels_history += [self.ignore_index] * len(token_user_context)
            if item['role'] == "assistant":
                token_answer_prompt = self.tokenizer(self.answer_prompt, return_attention_mask=False, add_special_tokens=False)['input_ids']  # noqa
                labels_answer_prompt = [self.ignore_index] * len(token_answer_prompt)
                assis_context = "{}{}".format(item['content'], self.eoa)
                token_assis_context = self.tokenizer(assis_context, return_attention_mask=False, add_special_tokens=False)['input_ids'] + [self.tokenizer.eos_token_id]  # noqa
                # final_label
                labels_answer_prompt = labels_answer_prompt + token_assis_context
                # final_tokens
                token_assis_context = token_answer_prompt + token_assis_context
                if idx == 0:
                    labels_answer_prompt = [self.ignore_index] * len(labels_answer_prompt)
                tokens_history += token_assis_context
                labels_history += labels_answer_prompt
        return tokens_history, labels_history

    def _get_question_tokens_labels(self, question):
        new_question = "{}{}{}".format(self.question_prompt, question, self.eoh)
        tokens_question = self.tokenizer(new_question, return_attention_mask=False,
                                         add_special_tokens=False)['input_ids']
        labels_question = [self.ignore_index] * len(tokens_question)
        return tokens_question, labels_question

    def _get_answer_tokens_labels(self, answer):
        answer_prompt_tokens = self.tokenizer(self.answer_prompt, return_attention_mask=False,
                                              add_special_tokens=False)['input_ids']
        labels_answer_prompt = [self.ignore_index] * len(answer_prompt_tokens)
        if not self.inference_mode:
            answer += self.eoa
        tokens_answer = self.tokenizer(answer, return_attention_mask=False,
                                       add_special_tokens=False)['input_ids']
        if not self.inference_mode:
            tokens_answer += [self.tokenizer.eos_token_id]
        labels_answer = labels_answer_prompt + tokens_answer
        tokens_answer = answer_prompt_tokens + tokens_answer
        return tokens_answer, labels_answer

    def build_inference_meta(self, text, history=[]):
        meta_dict = {}
        meta_dict['role'] = 'user'
        meta_dict['content'] = text
        meta = history + [meta_dict]
        return meta

    def get_tokens_labels(self, meta):
        system, history, question, answer = self._process_meta(meta)
        tokens_system, labels_system = self._get_system_tokens_labels(system)
        tokens_history, labels_history = self._get_history_tokens_labels(history)
        tokens_question, labels_question = self._get_question_tokens_labels(question)
        tokens_answer, labels_answer = self._get_answer_tokens_labels(answer)

        tokens = tokens_system + tokens_history + tokens_question + tokens_answer
        if self.only_last_answer:
            labels = [self.ignore_index] * len(labels_system + labels_history + labels_question) + labels_answer  # noqa
        else:
            labels = labels_system + labels_history + labels_question + labels_answer

        if len(tokens) > self.max_seq_length:
            if self.drop_meta:
                return None, None
            outside_length = len(tokens) - self.max_seq_length
            # step1, clip history tokens from old to new
            tokens_history = tokens_history[outside_length:]
            labels_history = labels_history[outside_length:]
            tokens = tokens_system + tokens_history + tokens_question + tokens_answer
            labels = labels_system + labels_history + labels_question + labels_answer
            # step2, clip answer (When the history tokens is not enough to clip)
            if len(tokens) > self.max_seq_length:
                tokens = tokens[:self.max_seq_length]
                labels = labels[:self.max_seq_length]
        return tokens, labels

    def __call__(self, meta):
        if self.inference_mode:
            return self.get_tokens_labels(meta)
        tokens, labels = self.get_tokens_labels(meta)
        if tokens is None:
            return None
        input_ids = torch.LongTensor(tokens)
        if self.keep_all_keys:
            labels = input_ids.clone()
        else:
            labels = torch.LongTensor(labels)
        results = {'input_ids': input_ids, 'labels': labels}
        return results


@PARSER_REGISTRY.register('reward')
class RewardParser(BaseParser):
    def __init__(self, tokenizer, max_seq_length, ignore_index=-100,
                 keep_all_keys=False, return_type='token_ids', inference_mode=False):
        super().__init__(tokenizer, max_seq_length, ignore_index, keep_all_keys, return_type)
        self.inference_mode = inference_mode

    def __call__(self, meta):
        input_text = meta['input_text']
        choice = meta['choice']
        bad_answer = meta.get('bad_answer', '')
        input_text_token = self.tokenizer(input_text, return_attention_mask=False)['input_ids']
        choice_token = self.tokenizer(choice, return_attention_mask=False, add_special_tokens=False)['input_ids']
        if not self.inference_mode:
            bad_answer_token = self.tokenizer(bad_answer, return_attention_mask=False, add_special_tokens=False)['input_ids']  # noqa
            choice_input = input_text_token + choice_token + [self.tokenizer.eos]
            bad_input = input_text_token + bad_answer_token + [self.tokenizer.eos]
            choice_input_ids = torch.LongTensor(choice_input)
            bad_input_ids = torch.LongTensor(bad_input)
            if len(choice_input_ids) > self.max_seq_length:
                choice_input_ids = choice_input_ids[:self.max_seq_length]
            if len(bad_input_ids) > self.max_seq_length:
                bad_input_ids = bad_input_ids[:self.max_seq_length]
            results = {'input_ids': (choice_input_ids, bad_input_ids)}
        else:
            length = len(input_text_token) + len(choice_token)
            if length > self.max_seq_length:
                input_text_token = input_text_token[length - self.max_seq_length:]
            labels = torch.LongTensor([self.ignore_index] * len(input_text_token) + choice_token)[:self.max_seq_length]
            input_ids = input_text_token + choice_token
            input_ids = torch.LongTensor(input_ids)
            results = {'input_ids': input_ids, "labels": labels}
        return results


@PARSER_REGISTRY.register('mini_rlhf')
class MiniRLHFParser(BaseParser):
    def __init__(self, tokenizer, max_seq_length, ignore_index=-100,
                 keep_all_keys=False, return_type='token_ids', inference_mode=False):
        super().__init__(tokenizer, max_seq_length, ignore_index, keep_all_keys, return_type)
        self.inference_mode = inference_mode

    def __call__(self, meta):
        question = meta['question']
        answers = meta['answers']
        scores = meta['scores']
        input_question_token = self.tokenizer(question, return_attention_mask=False)['input_ids']
        if not self.inference_mode:
            assert len(self.tokenizer.eos_token), 'the tokenizer must have an eos_token, please check your special_tokens_map.json or set it manually'      # noqa
            all_sentense_token = []
            all_labels = []
            for ans in answers:
                ans = f"{ans}{self.tokenizer.eos_token}"
                ans_token = self.tokenizer(ans, return_attention_mask=False,
                                           add_special_tokens=False)['input_ids']
                sentense_token = torch.LongTensor(input_question_token + ans_token)[-self.max_seq_length:]
                all_sentense_token.append(sentense_token)     # noqa
                labels = torch.LongTensor([self.ignore_index] * len(input_question_token) + ans_token)[-self.max_seq_length:]        # noqa
                all_labels.append(labels)
            results = {'input_ids': all_sentense_token, 'labels': all_labels, 'scores': torch.FloatTensor(scores)}
        else:
            results = question
        return results


@AUGMENTATION_REGISTRY.register('sense_tokenization')
class SenseTokenization(object):
    def __init__(self, tokenizer, max_seq_length, parser_type=None, parser_kwargs={}, ignore_index=-100):
        parser_kwargs.update({'tokenizer': tokenizer, 'max_seq_length': max_seq_length,
                              'ignore_index': ignore_index})
        assert ('default' not in PARSER_REGISTRY) and ('chat' not in PARSER_REGISTRY), 'defualt and chat are keeped for sense tokenization, you can not register them.'     # noqa
        if parser_type is None or parser_type == 'default':
            parser_type = 'sense'
        elif parser_type == 'chat':
            parser_type = 'sense_chat'
        self.parser = build_parser(parser_type, parser_kwargs)

    def __call__(self, *args, **kwargs):
        return self.parser(*args, **kwargs)


def build_parser(parser_type, parser_kwargs):
    return PARSER_REGISTRY.build({"type": parser_type, "kwargs": parser_kwargs})


def build_augmentation(cfg):
    if 'template' in cfg['kwargs']:
        cfg['kwargs'].pop('template')
    return AUGMENTATION_REGISTRY.build(cfg)


def build_transformer(cfgs):
    transform_list = [build_augmentation(cfg) for cfg in cfgs]
    return NLPCompose(transform_list)
