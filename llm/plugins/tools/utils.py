import json
import torch
from copy import deepcopy
from llm.plugins.internvl.datas.conversation import get_conv_template
from llm.plugins.internvl.utils.constants import IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
    if best_ratio == (2, 3) or best_ratio == (3, 2):
        new_area = image_size * image_size * 4
        if area < new_area:
            best_ratio = (2, 2)
    if best_ratio == (1, 1) or best_ratio == (2, 2):
        if area < image_size * image_size:
            best_ratio = (1, 1)
        else:
            best_ratio = (2, 2)
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def get_num_patchs(orig_width, orig_height, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    # target_width = image_size * target_aspect_ratio[0]
    # target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    if use_thumbnail and blocks != 1:
        blocks += 1
    return blocks


def preprocess_internlm(
    template_name,
    sources,
    tokenizer,
    num_image_token,
    text_only=False,
    group_by_length=False
):
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            if text_only:
                sentence['value'] = sentence['value'].replace('<image>', '').replace('<query>', '')
            sentence['value'] = sentence['value'].strip()
            if sentence['value'][0] == '\n':
                sentence['value'] = sentence['value'][1:]
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token}{IMG_END_TOKEN}'
    new_conversations = []
    for conversation in conversations:
        conversation = conversation.replace('<image>', image_tokens)
        new_conversations.append(conversation)
    conversations = new_conversations

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=False if group_by_length else 'max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    return len(input_ids[0])


class DataProcess(object):
    def __init__(self, template_name, meta, tokenizer, num_image_token, image_size=224, dynamic_image_size=False,
                 use_thumbnail=False, min_dynamic_patch=1, max_dynamic_patch=6, repeat_time=1, is_train=False,
                 pad2square=False, group_by_length=False, read_img=False):
        super(DataProcess, self).__init__()
        self.template_name = template_name
        self.meta = meta
        self.tokenizer = tokenizer
        self.num_image_token = num_image_token
        self.group_by_length = group_by_length
        self.image_size = image_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        with open(meta['annotation'], 'r') as f:
            self.raw_data = f.readlines()
            if repeat_time < 1:
                # choice top len(self.raw_data) * repeat_time samples
                self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]

    def __len__(self):
        return len(self.raw_data)

    def multi_modal_get_item(self, data_item):
        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

        orig_width, orig_height = data_item["width"], data_item["height"]
        num_patches = get_num_patchs(orig_width, orig_height, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                     image_size=self.image_size, use_thumbnail=self.use_thumbnail)

        # if not self.dynamic_image_size:
        #     assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'
        # if self.template_name == 'Hermes-2':
        #     preprocess_function = preprocess_mpt
        # elif self.template_name == 'internlm2-chat':
        preprocess_function = preprocess_internlm
        # else:
        #     preprocess_function = preprocess

        num_tokens = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                         self.tokenizer, self.num_image_token * num_patches,
                                         group_by_length=self.group_by_length)

        ret = dict(
            num_patches=num_patches,
            num_tokens=num_tokens,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def pure_text_get_item(self, data_item):
        num_patches = 1
        preprocess_function = preprocess_internlm
        num_tokens = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                         self.tokenizer, self.num_image_token * num_patches,
                                         group_by_length=self.group_by_length)

        ret = dict(
            num_patches=num_patches,
            num_tokens=num_tokens,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long)
        )
        return ret

    def get_item(self, idx):
        idx = idx % len(self.raw_data)
        data_item = json.loads(self.raw_data[idx])
        if 'image' in data_item and data_item['image'] is not None and len(data_item['image']) != 0:
            ret = self.multi_modal_get_item(data_item)
        else:
            ret = self.pure_text_get_item(data_item)
        return ret
