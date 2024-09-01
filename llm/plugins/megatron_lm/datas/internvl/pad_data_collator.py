import numpy as np
import math
import torch

from dataclasses import dataclass
from llm.utils.general.registry_factory import BATCH_COLLECTOR_REGISTRY
from llm.data.nlp_dataloader import BatchAlignCollector


IGNORE_INDEX = -100


@dataclass
@BATCH_COLLECTOR_REGISTRY.register('internvl')
class InternvlCollector(BatchAlignCollector):
    def __init__(self, tokenizer, alignment=1, offset_label=True):
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
            temp_input_ids[:feat['input_ids'].shape[0] - 1] = feat['input_ids'][:-1]
            # temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
            feat['input_ids'] = temp_input_ids
            temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
            temp_labels[:feat['labels'].shape[0] - 1] = feat['labels'][1:]
            # temp_labels[:feat['labels'].shape[0]] = feat['labels']
            feat['labels'] = temp_labels
            if "position_ids" in feat:
                # position_ids
                temp_position_ids = torch.LongTensor([0] * max_item_length)
                temp_position_ids[:feat['position_ids'].shape[0] - 1] = feat['position_ids'][:-1]
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
