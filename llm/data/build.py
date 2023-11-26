import torch

from llm.utils.env import dist_env

from .nlp_dataloader import build_data_loader


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def build_data_iterator(tokenizer, cfg_data, consumed_train_samples, data_type):
    # batch sampler setting
    batch_sample_type = cfg_data[data_type]['batch_sampler']['type']
    if 'kwargs' not in cfg_data[data_type]['batch_sampler']:
        cfg_data[data_type]['batch_sampler']['kwargs'] = {}
    assert 'consumed_samples' not in cfg_data[data_type]['batch_sampler']['kwargs'], 'Setting consumed_samples manually is not a recommended action.'       # noqa
    if data_type == 'train':
        cfg_data[data_type]['batch_sampler']['kwargs']['consumed_samples'] = consumed_train_samples
    else:
        cfg_data[data_type]['batch_sampler']['kwargs']['consumed_samples'] = 0
    if cfg_data[data_type]['batch_sampler']['kwargs'].get('micro_batch_size', None) is not None:
        assert cfg_data[data_type]['batch_sampler']['kwargs']['micro_batch_size'] == cfg_data[data_type]['micro_batch_size']       # noqa
    else:
        cfg_data[data_type]['batch_sampler']['kwargs']['micro_batch_size'] == cfg_data[data_type]['micro_batch_size']       # noqa
    # build data loader
    if dist_env.get_tensor_model_parallel_rank() == 0:
        dataloader = build_data_loader(cfg_data[data_type], tokenizer)
        flags = torch.cuda.LongTensor([len(dataloader)])
    else:
        dataloader = None
        flags = torch.cuda.LongTensor([0])
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    torch.distributed.broadcast(flags, dist_env.get_tensor_model_parallel_src_rank(),
                                group=dist_env.get_tensor_model_parallel_group())

    if dataloader is not None:
        if batch_sample_type == 'megatron_pretrain' or batch_sample_type == 'base':
            iterator = iter(dataloader)
        if batch_sample_type == 'megatron_pretrain_random':
            iterator = iter(cyclic_iter(dataloader))
    else:
        iterator = None

    return iterator, flags[0]
