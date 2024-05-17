import torch
import os
import json
import copy
from llm.utils.general.registry_factory import PARSER_REGISTRY
try:
    from PIL import Image
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
except ImportError:
    Image, T, InterpolationMode = None, None, None


@PARSER_REGISTRY.register('internvl_tools')
class InternvlToolParser(object):
    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 ignore_index=-100,
                 keep_all_keys=False,
                 inference_mode=False,
                 drop_meta=False,
                 prompt_template={},
                 use_system=True,
                 use_knowledge=True,
                 use_interpreter=True,
                 ensure_ascii=False,
                 tool_mode='merge',
                 min_dynamic_patch=1,
                 max_dynamic_patch=6,
                 image_size=224,
                 use_thumbnail=False,
                 dynamic_image_size=True,
                 num_image_token=256,
                 pad2square=False,
                 read_img=True):
        # system prompt won't be deleted,
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.keep_all_keys = keep_all_keys
        self.max_seq_length = max_seq_length
        self.inference_mode = inference_mode
        self.drop_meta = drop_meta
        self.conversation_start = prompt_template.get("conversation_start", "<|im_start|>")
        self.conversation_end = prompt_template.get("conversation_end", "<|im_end|>")
        self.action_start = prompt_template.get("action_start", "<|action_start|>")
        self.action_end = prompt_template.get("action_end", "<|action_end|>")
        # plugin tools
        self.plugin_prompt = prompt_template.get("plugin", "<|plugin|>")
        # interpreter
        self.interpreter_prompt = prompt_template.get("interpreter", "<|interpreter|>")
        # mllm token
        self.img_content_token = prompt_template.get("img_content_token", "<IMG_CONTEXT>")
        self.img_start_token = prompt_template.get("img_start_token", "<img>")
        self.img_end_token = prompt_template.get("img_end_token", "</img>")

        self.use_system = use_system
        self.use_knowledge = use_knowledge
        self.use_interpreter = use_interpreter
        self.ensure_ascii = ensure_ascii
        self.tool_mode = tool_mode

        # image setting
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.image_size = image_size
        self.use_thumbnail = use_thumbnail
        self.dynamic_image_size = dynamic_image_size
        self.num_image_token = num_image_token
        self.pad2square = pad2square
        self.read_img = read_img

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def build_img_transform(self, is_train, input_size, pad2square=False):
        if is_train:
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.),
                                    interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            if not pad2square:
                transform = T.Compose([
                    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                    T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ])
            else:
                transform = T.Compose([
                    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                    T.Lambda(lambda img: self.expand2square(img, tuple(int(x * 255) for x in (0.485, 0.456, 0.406)))),
                    T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ])

        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def get_num_patchs(self,
                       orig_width,
                       orig_height,
                       min_num=1,
                       max_num=6,
                       image_size=448,
                       use_thumbnail=False):
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # calculate the target width and height
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        if use_thumbnail and blocks != 1:
            blocks += 1
        return blocks

    def __call__(self, meta):
        if "image" in meta:
            if self.read_img:
                # TODO: read image in ceph
                img_dir = meta["img_dir"]
                is_train = meta["data_augment"]
                img_path = os.path.join(img_dir, meta['image'])
                image = Image.open(img_path).convert('RGB')

                img_transform = self.build_img_transform(is_train=is_train, input_size=self.image_size, pad2square=self.pad2square)
                if self.dynamic_image_size:
                    images = self.dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                                     image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                else:
                    images = [image]

                pixel_values = [img_transform(image) for image in images]
                pixel_values = torch.stack(pixel_values)
                num_patches = pixel_values.size(0)
            else:
                # for count length
                orig_width, orig_height = meta["width"], meta["height"]
                num_patches = self.get_num_patchs(orig_width,
                                                  orig_height,
                                                  min_num=self.min_dynamic_patch,
                                                  max_num=self.max_dynamic_patch,
                                                  image_size=self.image_size,
                                                  use_thumbnail=self.use_thumbnail)
                pixel_values = torch.empty((num_patches, 3, 448, 448))
            if not self.dynamic_image_size:
                assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

            conversations = meta['conversations']
            if '<image>' not in conversations[0]['value']:
                conversations[0]['value'] = '<image>\n' + conversations[0]['value']

            # fix bug when there are multiple <image> in conversations
            image_cnt = 0
            for idx, conv in enumerate(conversations):
                conv['value'] = conv['value'].replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')
                if idx == 0:
                    conv['value'] = '<image>\n' + conv['value']
                image_cnt += conv['value'].count('<image>')
            assert image_cnt == 1, f'There should be exactly one <image> in the conversation, but got {image_cnt}'

            roles = {'human': 'user', 'gpt': 'assistant'}
            tokens = []
            labels = []

            tokens.extend([self.tokenizer.bos_token_id])
            labels.extend([self.ignore_index])
            image_tokens = f'{self.img_start_token}{self.img_content_token * self.num_image_token * num_patches}{self.img_end_token}'
            if roles[conversations[0]['from']] != 'user':
                conversations = conversations[1:]
            for conv in conversations:
                content = conv['value'].strip()
                if content[0] == '\n':
                    content = content[1:]
                if '<image>' in content:
                    content = content.replace('<image>', image_tokens)
                if roles[conv['from']] == 'user':
                    user_info = f"{self.conversation_start}user\n{content}{self.conversation_end}\n"
                    tokenized_user = self.tokenizer(user_info, return_attention_mask=False, add_special_tokens=False)['input_ids']
                    tokens.extend(tokenized_user)
                    labels.extend([self.ignore_index] * len(tokenized_user))
                elif roles[conv['from']] == 'assistant':
                    assis_start = f"{self.conversation_start}assistant\n"
                    tokens_assistant_start = self.tokenizer(assis_start, return_attention_mask=False, add_special_tokens=False)['input_ids']
                    tokens.extend(tokens_assistant_start)
                    labels.extend([self.ignore_index] * len(tokens_assistant_start))

                    assis_info = f"{content}{self.conversation_end}\n"
                    tokenized_assistant = self.tokenizer(assis_info, return_attention_mask=False, add_special_tokens=False)['input_ids']
                    tokens.extend(tokenized_assistant)
                    labels.extend(copy.deepcopy(tokenized_assistant))
                else:
                    raise NotImplementedError(f"Not Support the role {roles[conv['from']]}!")

            input_ids = torch.LongTensor(tokens)
            labels = torch.LongTensor(labels)
            return {'input_ids': input_ids,
                    'labels': labels,
                    'pixel_values': pixel_values,
                    'image_flags': torch.tensor([1] * num_patches, dtype=torch.long)}
        else:
            if 'input' in meta:
                messages = meta['input'].get('messages', [])
                tools = meta['input'].get('tools', [])
                interpreter = meta['input'].get('interpreter', [])
            elif "messages" in meta:
                messages = meta['messages']
                tools = meta.get('tools', [])
                interpreter = meta.get('interpreter', [])
            elif "conversations" in meta:
                roles = {'human': 'user', 'gpt': 'assistant'}
                conversations = meta['conversations']
                messages = []
                for conv in conversations:
                    messages.append({'role': roles[conv['from']], 'content': conv['value']})

                tools = meta.get('tools', [])
                interpreter = meta.get('interpreter', [])
            else:
                messages = meta
                tools = []
                interpreter = []
            tokens = []
            labels = []

            tokens.extend([self.tokenizer.bos_token_id])
            labels.extend([self.ignore_index])
            conversation_messages = messages
            if self.use_system:
                if messages[0]['role'] == 'system' and messages[0]['content'] != '':
                    system = f"{self.conversation_start}system\n{messages[0]['content']}{self.conversation_end}\n"
                    tokenized_system = self.tokenizer(system, return_attention_mask=False,
                                                      add_special_tokens=False)['input_ids']
                    tokens.extend(tokenized_system)
                    labels.extend([self.ignore_index] * len(tokenized_system))
                    # remove the system item
                    conversation_messages = messages[1:]

            if self.use_interpreter and (len(interpreter) > 0):
                assert (len(interpreter) == 1) and (
                            interpreter[0]["name"] == "python_interpreter"), "Only support python interpreter now!"  # noqa
                interpreter_str = f"{self.conversation_start}system name={self.interpreter_prompt}\n{interpreter[0]['description']}\n{self.conversation_end}\n"
                tokenized_interpreter = self.tokenizer(interpreter_str, return_attention_mask=False, add_special_tokens=False)['input_ids']
                tokens.extend(tokenized_interpreter)
                labels.extend([self.ignore_index] * len(tokenized_interpreter))
            if len(tools) > 0:
                plugin_tools = []
                for tool in tools:
                    # use_interpreter=True and tool['function']['name']="python_interpreter" skip
                    if not (self.use_interpreter and (tool['function']['name'] == "python_interpreter")):
                        plugin_tools.append(copy.deepcopy(tool['function']))
                if len(plugin_tools) > 0:
                    if self.tool_mode == 'merge':
                        plugin_tools_str = json.dumps(plugin_tools, ensure_ascii=self.ensure_ascii)
                    else:
                        plugin_tools_str = ''
                        for p_idx, tool in enumerate(plugin_tools):
                            plugin_tools_str += json.dumps(tool, ensure_ascii=self.ensure_ascii)
                            if p_idx != len(plugin_tools) - 1:
                                plugin_tools_str += '\n'
                    plugin_tools_str = f"{self.conversation_start}system name={self.plugin_prompt}\n{plugin_tools_str}\n{self.conversation_end}\n"
                    tokenized_plugin_tools = self.tokenizer(plugin_tools_str, return_attention_mask=False, add_special_tokens=False)['input_ids']
                    tokens.extend(tokenized_plugin_tools)
                    labels.extend([self.ignore_index] * len(tokenized_plugin_tools))

            essential_prompt_len = len(tokens)
            last_dialog_index = len(tokens)
            dialog_len_list = []
            for item in conversation_messages:
                # assert item['role'] != 'system', "only allow system at the start of conversation"
                if self.use_knowledge:
                    # do not support knowledge yet
                    raise NotImplementedError
                if item['role'] == 'user':
                    user_info = f"{self.conversation_start}user\n{item['content']}{self.conversation_end}\n"
                    tokenized_user = self.tokenizer(user_info, return_attention_mask=False,
                                                    add_special_tokens=False)['input_ids']
                    tokens.extend(tokenized_user)
                    labels.extend([self.ignore_index] * len(tokenized_user))

                if item['role'] == 'assistant':
                    assis_start = f"{self.conversation_start}assistant\n"
                    tokens_assistant_start = self.tokenizer(assis_start, return_attention_mask=False,
                                                            add_special_tokens=False)['input_ids']
                    tokens.extend(tokens_assistant_start)
                    labels.extend([self.ignore_index] * len(tokens_assistant_start))
                    assis_info = ""
                    if item['content']:
                        assis_info = item['content']

                    if 'tool_calls' in item and len(item['tool_calls']) > 0:
                        assis_info += self.action_start
                        for tool_call in item['tool_calls']:
                            if self.use_interpreter and 'name' in tool_call['function'] and tool_call['function']['name'] == "python_interpreter":
                                assis_info += f"{self.interpreter_prompt}\n{tool_call['function']['arguments']['code']}\n"  # noqa
                            else:
                                assis_info += f"{self.plugin_prompt}\n{json.dumps(tool_call['function'], ensure_ascii=self.ensure_ascii)}\n"
                        assis_info += self.action_end
                    if not self.inference_mode:
                        assis_info += f"{self.conversation_end}\n"
                    tokenized_assistant = self.tokenizer(assis_info, return_attention_mask=False,
                                                         add_special_tokens=False)['input_ids']
                    tokens.extend(tokenized_assistant)
                    labels.extend(copy.deepcopy(tokenized_assistant))

                if item['role'] == 'tool':
                    if self.use_interpreter and 'name' in item and item['name'] == "python_interpreter":
                        response_info = f"{self.conversation_start}environment name={self.interpreter_prompt}\n{item['content']}{self.conversation_end}\n"
                    else:
                        response_info = f"{self.conversation_start}environment name={self.plugin_prompt}\n{item['content']}{self.conversation_end}\n"
                    tokenized_response = self.tokenizer(response_info, return_attention_mask=False, add_special_tokens=False)['input_ids']
                    tokens.extend(tokenized_response)
                    labels.extend([self.ignore_index] * len(tokenized_response))

                if item["role"] == "assistant" and ("tool_calls" not in item):
                    dialog_len_list.append(len(tokens) - last_dialog_index)
                    last_dialog_index = len(tokens)

            if self.inference_mode:
                infer_tokens_assistant_prompt = self.tokenizer(f"{self.conversation_start}assistant\n", return_attention_mask=False,
                                                               add_special_tokens=False)['input_ids']
                tokens.extend(infer_tokens_assistant_prompt)
                labels.extend([self.ignore_index] * len(infer_tokens_assistant_prompt))
                return tokens, []
            if self.keep_all_keys:
                labels = copy.deepcopy(tokens)
            else:
                if self.drop_meta and len(tokens) > self.max_seq_length:
                    return None
                # drop question to avoid no loss
                seq_length = 0
                for dialog_len in dialog_len_list[::-1]:
                    if (seq_length + dialog_len + essential_prompt_len) >= self.max_seq_length:
                        break
                    seq_length += dialog_len

                if seq_length == 0:
                    tokens = tokens[:essential_prompt_len]
                    labels = tokens[:essential_prompt_len]
                    # return None
                else:
                    tokens = tokens[:essential_prompt_len] + tokens[-seq_length:]
                    labels = labels[:essential_prompt_len] + labels[-seq_length:]
            image = Image.new('RGB', (224, 224), (255, 255, 255))
            images = self.dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                             image_size=self.image_size, use_thumbnail=self.use_thumbnail)
            transform = self.build_img_transform(is_train=False, input_size=self.image_size, pad2square=self.pad2square)
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)

            input_ids = torch.LongTensor(tokens)
            labels = torch.LongTensor(labels)
            results = {'input_ids': input_ids,
                       'labels': labels,
                       'pixel_values': pixel_values,
                       'image_flags': torch.tensor([0], dtype=torch.long)}
            return results
