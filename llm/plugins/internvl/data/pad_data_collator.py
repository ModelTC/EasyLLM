import numpy as np
import math
import torch

from dataclasses import dataclass
from llm.utils.general.registry_factory import BATCH_COLLECTOR_REGISTRY
from llm.data.nlp_dataloader import BatchAlignCollector
from llm.models.hf_models.sequence import (get_sequence_parallel_world_size,
                                           get_sequence_parallel_rank)


IGNORE_INDEX = -100


@dataclass
@BATCH_COLLECTOR_REGISTRY.register('internvl')
class InternvlCollector(BatchAlignCollector):
    def __init__(self, tokenizer, alignment=1, offset_label=True):
        alignment = math.lcm(get_sequence_parallel_world_size(), alignment)
        super().__init__(tokenizer, alignment=alignment, offset_label=offset_label)

    def __call__(self, instances, pad_id=0):
        first = instances[0]
        batch = {}

        batch_lens = [feat['input_ids'].shape for feat in instances]
        max_item_length = max(batch_lens)[0]
        max_item_length = math.ceil(max_item_length / float(self.alignment)) * self.alignment
        for idx in range(len(instances)):
            feat = instances[idx]
            temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
            if self.offset_label:
                temp_input_ids[:feat['input_ids'].shape[0] - 1] = feat['input_ids'][:-1]
            else:
                temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
            feat['input_ids'] = temp_input_ids
            temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
            if self.offset_label:
                temp_labels[:feat['labels'].shape[0] - 1] = feat['labels'][1:]
            else:
                temp_labels[:feat['labels'].shape[0]] = feat['labels']
            feat['labels'] = temp_labels
            if "position_ids" in feat:
                # position_ids
                temp_position_ids = torch.LongTensor([0] * max_item_length)
                if self.offset_label:
                    temp_position_ids[:feat['position_ids'].shape[0] - 1] = feat['position_ids'][:-1]
                else:
                    temp_position_ids[:feat['position_ids'].shape[0]] = feat['position_ids']
                feat['position_ids'] = temp_position_ids
            if "cu_seqlens" in feat:
                feat['cu_seqlens'][-1] = feat['position_ids'].size(0)
            feat['attention_mask'] = feat['input_ids'].ne(pad_id)

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if 'label' in first and first['label'] is not None:
            label = first['label'].item() if isinstance(first['label'], torch.Tensor) else first['label']
            dtype = torch.long if isinstance(label, int) else torch.float
            batch['labels'] = torch.tensor([f['label'] for f in instances], dtype=dtype)
        elif 'label_ids' in first and first['label_ids'] is not None:
            if isinstance(first['label_ids'], torch.Tensor):
                batch['labels'] = torch.stack([f['label_ids'] for f in instances])
            else:
                dtype = torch.long if isinstance(first['label_ids'][0], int) else torch.float
                batch['labels'] = torch.tensor([f['label_ids'] for f in instances], dtype=dtype)

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if k not in ('label', 'label_ids', 'pixel_values', 'image_flags') and \
                    v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in instances])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in instances]))
                else:
                    batch[k] = torch.tensor([f[k] for f in instances])
            if k in ('pixel_values', 'image_flags'):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.concat([f[k] for f in instances])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.concat(np.stack([f[k] for f in instances]))
                else:
                    batch[k] = torch.concat([f[k] for f in instances])

        return batch


@dataclass
@BATCH_COLLECTOR_REGISTRY.register('internvl_hf')
class InternvlCollector(BatchAlignCollector):
    def __init__(self, tokenizer, alignment=1, offset_label=True):
        self.sp_num = get_sequence_parallel_world_size()
        if self.sp_num > 1:
            alignment = math.lcm(get_sequence_parallel_world_size(), alignment)
        super().__init__(tokenizer, alignment=alignment, offset_label=offset_label)

    def __call__(self, instances, pad_id=0):
        if self.sp_num > 1:
            assert len(instances) == 1, "Only support batch size is 1 under sp condition"
            assert len(instances[0]) == self.sp_num
            length_list = []
            for instance in instances[0]:
                length_list.append(len(instance['input_ids']))
            max_item_length = max(length_list)

            # accum_length = 0
            # cu_seqlens_list = [torch.tensor([0], dtype=torch.int32, device=instances[0][0]['cu_seqlens'].device)]
            # for instance in instances[0]:
            #     instance['cu_seqlens'] += accum_length
            #     instance['cu_seqlens'][0] = accum_length
            #     temp_position_ids = torch.LongTensor([0] * max_item_length)
            #     temp_position_ids[:instance['position_ids'].shape[0]] = instance['position_ids']
            #     instance['position_ids'] = temp_position_ids
            #     accum_length += max_item_length
            #     instance['cu_seqlens'][-1] = accum_length
            #     cu_seqlens_list.append(instance['cu_seqlens'][1:])

            rank = get_sequence_parallel_rank()
            instances = [instances[0][rank]]
            # instances[0]['cu_seqlens'] = torch.cat(cu_seqlens_list)
        else:
            batch_lens = [feat['input_ids'].shape for feat in instances]
            max_item_length = max(batch_lens)[0]
            max_item_length = math.ceil(max_item_length / float(self.alignment)) * self.alignment
        
        first = instances[0]
        batch = {}
        for idx in range(len(instances)):
            feat = instances[idx]
            temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
            if self.offset_label:
                temp_input_ids[:feat['input_ids'].shape[0] - 1] = feat['input_ids'][:-1]
            else:
                temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
            feat['input_ids'] = temp_input_ids
            temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
            if self.offset_label:
                temp_labels[:feat['labels'].shape[0] - 1] = feat['labels'][1:]
            else:
                temp_labels[:feat['labels'].shape[0]] = feat['labels']
            feat['labels'] = temp_labels
            if "position_ids" in feat: #  and self.sp_num == 1
                # position_ids
                temp_position_ids = torch.LongTensor([0] * max_item_length)
                if self.offset_label:
                    temp_position_ids[:feat['position_ids'].shape[0] - 1] = feat['position_ids'][:-1]
                else:
                    temp_position_ids[:feat['position_ids'].shape[0]] = feat['position_ids']
                feat['position_ids'] = temp_position_ids
            if "cu_seqlens" in feat: # and self.sp_num == 1
                feat['cu_seqlens'][-1] = feat['position_ids'].size(0)
            feat['attention_mask'] = feat['input_ids'].ne(pad_id)

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if 'label' in first and first['label'] is not None:
            label = first['label'].item() if isinstance(first['label'], torch.Tensor) else first['label']
            dtype = torch.long if isinstance(label, int) else torch.float
            batch['labels'] = torch.tensor([f['label'] for f in instances], dtype=dtype)
        elif 'label_ids' in first and first['label_ids'] is not None:
            if isinstance(first['label_ids'], torch.Tensor):
                batch['labels'] = torch.stack([f['label_ids'] for f in instances])
            else:
                dtype = torch.long if isinstance(first['label_ids'][0], int) else torch.float
                batch['labels'] = torch.tensor([f['label_ids'] for f in instances], dtype=dtype)

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if k not in ('label', 'label_ids', 'pixel_values', 'image_flags') and \
                    v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in instances])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in instances]))
                else:
                    batch[k] = torch.tensor([f[k] for f in instances])
            if k in ('pixel_values', 'image_flags'):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.concat([f[k] for f in instances])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.concat(np.stack([f[k] for f in instances]))
                else:
                    batch[k] = torch.concat([f[k] for f in instances])

        return batch
