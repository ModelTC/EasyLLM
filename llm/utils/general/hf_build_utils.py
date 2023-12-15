import torch
import copy
from llm.utils.general.fast_init import fast_init
from llm.utils.general.log_helper import default_logger as logger
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from llm.data.nlp_sampler import InfiniteBatchSampler
from llm.utils.env.hf_dist_helper import (
    get_rank,
    get_world_size)
from llm.utils.general.registry_factory import (
    MODULE_ZOO_REGISTRY,
    DATASET_REGISTRY,
    DATALOADER_REGISTRY,
    BATCH_COLLECTOR_REGISTRY,
    SAMPLER_REGISTRY,
    BATCH_SAMPLER_REGISTRY,
    AUGMENTATION_REGISTRY)


def build_model(model_cfg):
    fast_device = torch.device('cuda')
    with fast_init(fast_device):
        peft_model_cfg = model_cfg.get('peft_model_cfg', None)
        model = MODULE_ZOO_REGISTRY.build(model_cfg)
        if peft_model_cfg is not None:
            model = build_peft_model(peft_model_cfg, model)
    return model


def hack_model(model):
    def hack_custom_forward(module, *args, **kwargs):
        output = module(*args, **kwargs)
        output.requires_grad = True
        return output

    def common_cast_forward(m, *args, **kwargs):
        old_forward = m.forward

        def forward(*args, **kwargs):
            return hack_custom_forward(old_forward, *args, **kwargs)
        m.forward = forward

    for _, m in model.named_modules():
        if isinstance(m, torch.nn.Embedding):
            common_cast_forward(m)
            logger.info("set nn.Embedding output requires_grad=True for gradient checkpointing")


def build_peft_model(peft_model_cfg, model):
    if peft_model_cfg.get('peft_path', None) is not None:
        model = PeftModel.from_pretrained(model, peft_model_cfg['peft_path'])
        logger.info(f"load peft model from : {peft_model_cfg['peft_path']}")
        return model
    peft_type = peft_model_cfg.get('peft_type', 'Lora')
    assert peft_type == 'Lora'
    logger.warning("Lora is only be supported!!!")
    target_modules = peft_model_cfg.get('target_modules', [])
    lora_rank = peft_model_cfg['lora_rank']
    lora_alpha = peft_model_cfg['lora_alpha']
    lora_dropout = peft_model_cfg['lora_dropout']
    modules_to_save = peft_model_cfg.get('modules_to_save', None)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        inference_mode=False,
        r=lora_rank, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        modules_to_save=modules_to_save)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    logger.info(f"model.modules_to_save: {model.modules_to_save}")
    return model


def build_dataset(cfg_dataset, tokenizer):
    if 'kwargs' not in cfg_dataset:
        cfg_dataset['kwargs'] = {}
    cfg_dataset['kwargs']['tokenizer'] = tokenizer
    return DATASET_REGISTRY.build(cfg_dataset)


def build_batch_collator(cfg_batch_collator, tokenizer):
    if 'kwargs' not in cfg_batch_collator:
        cfg_batch_collator['kwargs'] = {}
    cfg_batch_collator['kwargs']['tokenizer'] = tokenizer
    return BATCH_COLLECTOR_REGISTRY.build(cfg_batch_collator)


def build_sampler(cfg_sampler, dataset):
    cfg_sampler = copy.deepcopy(cfg_sampler)
    if 'kwargs' not in cfg_sampler:
        cfg_sampler['kwargs'] = {}
    cfg_sampler['kwargs']['dataset_size'] = len(dataset)
    cfg_sampler['kwargs']['data_parallel_rank'] = get_rank()
    cfg_sampler['kwargs']['data_parallel_size'] = get_world_size()
    return SAMPLER_REGISTRY.build(cfg_sampler)


def build_batch_sampler(cfg_batch_sampler, dataset):
    cfg_batch_sampler = copy.deepcopy(cfg_batch_sampler)
    cfg_sampler = cfg_batch_sampler['kwargs']['sampler']
    sampler = build_sampler(cfg_sampler, dataset)
    cfg_batch_sampler['kwargs']['sampler'] = sampler
    batch_sampler = BATCH_SAMPLER_REGISTRY.build(cfg_batch_sampler)
    infinite = cfg_batch_sampler.pop('infinite', True)
    if infinite:
        batch_sampler = InfiniteBatchSampler(batch_sampler)
    return batch_sampler


def build_dataloader(cfg_data, dataset, batch_collator):
    batch_sampler = build_batch_sampler(cfg_data['batch_sampler'], dataset)
    cfg_data['data_loader']['kwargs'].update({'dataset': dataset,
                                              'batch_sampler': batch_sampler,
                                              'batch_collator': batch_collator})
    return DATALOADER_REGISTRY.build(cfg_data['data_loader'])


def build_augmentation(cfg):
    if 'template' in cfg['kwargs']:
        cfg['kwargs'].pop('template')
    return AUGMENTATION_REGISTRY.build(cfg)
