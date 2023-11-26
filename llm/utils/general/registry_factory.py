from .registry import Registry

# tokenizer
TOKENIZER_REGISTRY = Registry()

# data
BATCH_CALCULATOR_REGISTRY = Registry()
DATASET_REGISTRY = Registry()
PARSER_REGISTRY = Registry()
AUGMENTATION_REGISTRY = Registry()
BATCH_COLLECTOR_REGISTRY = Registry()
SAMPLER_REGISTRY = Registry()
BATCH_SAMPLER_REGISTRY = Registry()
DATALOADER_REGISTRY = Registry()
BATCH_FN_REGISTRY = Registry()

# trainer
LR_REGISTRY = Registry()

# model
MODULE_ZOO_REGISTRY = Registry()

# utils
HOOK_REGISTRY = Registry()

# loss func

LOSS_REGISTRY = Registry()
