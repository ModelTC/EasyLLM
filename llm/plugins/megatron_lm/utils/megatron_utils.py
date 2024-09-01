import torch
from functools import partial

import os
if os.getenv("DIST_BACKEND", "easyllm") == "easyllm":
    from llm.utils.env import dist_env
elif os.getenv("DIST_BACKEND", "easyllm") == "megatron":
    from megatron.core import mpu as dist_env
from llm.plugins.megatron_lm.utils.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank
)
from megatron.training import get_args
from megatron.training import get_timers
from llm.data.nlp_dataset import IGNORE_INDEX
from megatron.core.models.gpt import GPTModel
from megatron.core.utils import StragglerDetector
from llm.utils.general.log_helper import default_logger as logger
from megatron.core import tensor_parallel, parallel_state
from megatron.core.packed_seq_params import PackedSeqParams

stimer = StragglerDetector()
import time

def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    # if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
    if (not dist_env.is_pipeline_first_stage()) and (not dist_env.is_pipeline_last_stage()):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


def loss_func(loss_mask: torch.Tensor, labels: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()
    ignore_mask = (labels == IGNORE_INDEX)
    loss_mask = loss_mask.view(-1)
    loss_mask = loss_mask * (~ignore_mask.view(-1))

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

    if args.context_parallel_size > 1:
        # torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        torch.distributed.all_reduce(loss, group=dist_env.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss[0].isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    reporting_loss = reporting_loss[0] / reporting_loss[1] / dist_env.get_data_parallel_world_size()  # mpu.get_data_parallel_world_size()
    # torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())
    torch.distributed.all_reduce(reporting_loss, group=dist_env.get_data_parallel_group())

    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0] * args.context_parallel_size,
        local_num_tokens,
        # {'lm loss': (reporting_loss[0], reporting_loss[1])},
        {'lm loss': reporting_loss}
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator)
    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels)

    return output_tensor, partial(loss_func, loss_mask, labels)

def get_batch_vlm(data_iterator):
    """Generate a batch.

    Args:
        data_iterator: Iterable dataset.

    Returns:
        sample: A data sample with images, tokens, etc.
    """
    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    data_i = tensor_parallel.broadcast_data(["input_ids", "labels", "image_flags"], data, torch.int64)
    data_f = tensor_parallel.broadcast_data(["pixel_values"], data, torch.float32)
    data_b = tensor_parallel.broadcast_data(["attention_mask", "loss_mask"], data, torch.bool)

    input_ids = data_i["input_ids"].long()
    # position_ids = data_i["position_ids"].long()
    labels = data_i["labels"].long()
    image_flags = data_i['image_flags'].long()
    # cu_seqlens = data_i['cu_seqlens'].int()

    # data_i = tensor_parallel.broadcast_data(["cu_seqlens"], data, torch.int64)
    position_ids = None
    cu_seqlens = None

    pixel_values = data_f["pixel_values"].float()
    
    attention_mask = data_b["attention_mask"].bool()
    loss_mask = data_b["loss_mask"].bool()

    return input_ids, position_ids, labels, image_flags, cu_seqlens, pixel_values, attention_mask, loss_mask


def forward_step_vlm(data_iterator, model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    logger.info("start forward...")

    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        input_ids, position_ids, labels, image_flags, cu_seqlens, pixel_values, attention_mask, loss_mask = get_batch_vlm(
            data_iterator)
    timers('batch-generator').stop()

    with stimer:
        if cu_seqlens:
            packed_seq_params = PackedSeqParams(
                cu_seqlens_q=cu_seqlens[0],
                cu_seqlens_kv=cu_seqlens[0],
                qkv_format='thd',
                # max_seqlen_q=cu_seqlens[0][-1],
                # max_seqlen_kv=cu_seqlens[0][-1],
            )
        else:
            packed_seq_params=None


        # torch.cuda.synchronize()
        # start_time = time.time()
        # if torch.distributed.get_rank() == 0:
        #     import pdb;pdb.set_trace()
        output_tensor = model(pixel_values, input_ids, position_ids, 
                              attention_mask, image_flags, labels,
                              packed_seq_params=packed_seq_params)
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print(f'rank:{torch.distributed.get_rank()}, pp_rank:{parallel_state.get_pipeline_model_parallel_rank()}, duration:{end_time - start_time} s')

    return output_tensor, partial(loss_func, loss_mask, labels)

def build_model_cfg(config):
    from llm.models.mg_models.llama.llama import _LLAMA_MODELS
    from llm.models.mg_models.base_modules.utils import check_keys_mapping
    from llm.plugins.megatron_lm.utils.default_cfg import update_model_cfg

    cfg_model = config['model']
    model_type = cfg_model['type']
    if 'kwargs' not in cfg_model:
        cfg_model['kwargs'] = {}
    cfg_model['kwargs'].update({"fp16": config['runtime'].get('fp16', False),
                                "bf16": config['runtime'].get('bf16', False)})       # noqa
    is_qkv_pack = cfg_model['kwargs'].get("transformer_layer_params", {}).get("qkv_pack", False)
    pretrain_type = config['loader'].get("pretrain_type", "llama")
    if is_qkv_pack:
        assert "pack" in pretrain_type, "qkv_pack must load the model in pack type. currently support llama_pack and internlm2_pack!"
    else:
        assert "pack" not in pretrain_type, "You load the model in pack type, but do not set qkv_pack as True!"
    if model_type != "llama_custom":
        cfg_item = _LLAMA_MODELS[model_type]
        check_keys_mapping(cfg_item, cfg_model)
        cfg_model["kwargs"].update(cfg_item)
    cfg_model = update_model_cfg(cfg_model['kwargs'])
    return cfg_model

def build_internvl_cfg(cfg):
    cfg_model = cfg['model']
    if 'kwargs' not in cfg_model:
        cfg_model['kwargs'] = {}
    # if getattr(tokenizer, 'padded_vocab_size', None) is not None \
    #    and tokenizer.padded_vocab_size != len(tokenizer):
    #     vocab_size = tokenizer.padded_vocab_size
    # else:
    #     vocab_size = len(tokenizer)
    cfg_model['kwargs'].update({"fp16": cfg['runtime'].get('fp16', False),
                                "bf16": cfg['runtime'].get('bf16', False),
                                # "unpad_vocab_size": getattr(tokenizer, 'vocab_size'),
                                # "vocab_size": vocab_size,
                                "micro_batch_size": cfg['data']['train']['micro_batch_size'],
                                "seq_length": cfg['data']['train']['seq_length']})       # noqa
    
    from llm.plugins.megatron_lm.models.internvl.default_cfg import update_model_cfg
    cfg_model = update_model_cfg(cfg_model['kwargs'])
    return cfg_model

def yaml2args_vlm(config, extra_args_provider=None, ignore_unknown_args=True, args_defaults=dict()):
    from megatron.training.arguments import parse_args as parse_args_mg
    from megatron.training.arguments import validate_args
    from megatron.training.yaml_arguments import validate_yaml
    from megatron.training.global_vars import set_global_variables

    args = parse_args_mg(extra_args_provider, ignore_unknown_args)
    if config['model']['type'] == "intern_custom":
        cfg_model = build_internvl_cfg(config)
        args.cfg_model = cfg_model

        args.num_layers = cfg_model['num_layers']
        args.hidden_size = cfg_model['transformer_layer_params']['hidden_size']
        args.num_attention_heads = cfg_model['transformer_layer_params']['num_attention_heads']
        args.max_position_embeddings = cfg_model["word_embedings_params"].get("max_position_embeddings")
        args.seq_length = config["tokenization"]["kwargs"].get("max_seq_length", 4096)
        if args.max_position_embeddings is None or args.seq_length > args.max_position_embeddings:
            args.max_position_embeddings = args.seq_length
        args.micro_batch_size = config['data']['train']['micro_batch_size']

        logger.info("Mapping yaml args to megatron args.")
        args.ffn_hidden_size = cfg_model['transformer_layer_params']['intermediate_size']
        if cfg_model['transformer_layer_params'].get("glu_activation", "silu") == "silu":
            args.swiglu = True
        args.use_rotary_position_embeddings = True
        args.hidden_dropout = cfg_model['transformer_layer_params']['hidden_dropout']
        args.attention_dropout = cfg_model['transformer_layer_params']['attention_dropout']
        args.add_bias_linear = False
        args.norm_epsilon = cfg_model['layer_norm_params']["kwargs"]['eps']
        if cfg_model['layer_norm_params']['type'] == "rms_norm":
            args.normalization = "RMSNorm"
        if cfg_model['transformer_layer_params']['num_kv_attention_heads'] != cfg_model['transformer_layer_params']['num_attention_heads']:
            args.group_query_attention = True
            args.num_query_groups = cfg_model['transformer_layer_params']['num_kv_attention_heads']
        position_embedding_kwargs = cfg_model['transformer_layer_params']['position_embedding_kwargs']
        args.init_method_std = cfg_model['transformer_layer_params']['initializer']['kwargs']['sigma']
        args.rotary_base = position_embedding_kwargs.get('base', 10000)
        args.use_flash_attn = cfg_model.get('use_flash_attn', True)
        args.sequence_parallel = cfg_model.get('sequence_parallel', False)

    else:
        cfg_model = build_model_cfg(config)

    args.pp_partition_method = config['model']['kwargs'].get('pp_partition_method', "uniform")
    args.pp_partition_parts = None

    # args.num_layers = cfg_model['num_layers']
    # args.hidden_size = cfg_model['hidden_size']
    # args.num_attention_heads = cfg_model['num_attention_heads']
    # args.max_position_embeddings = cfg_model["word_embedings_params"].get("max_position_embeddings")
    # args.seq_length = config["tokenization"]["kwargs"].get("max_seq_length", 4096)
    # if args.max_position_embeddings is None or args.seq_length > args.max_position_embeddings:
    #     args.max_position_embeddings = args.seq_length
    # args.micro_batch_size = config['data']['train']['micro_batch_size']

    # logger.info("Mapping yaml args to megatron args.")
    # args.ffn_hidden_size = cfg_model['intermediate_size']
    # if cfg_model['transformer_layer_params'].get("glu_activation", "silu") == "silu":
    #     args.swiglu = True
    # args.use_rotary_position_embeddings = True
    # args.hidden_dropout = cfg_model['transformer_layer_params']['hidden_dropout']
    # args.attention_dropout = cfg_model['transformer_layer_params']['attention_dropout']
    # args.add_bias_linear = False
    # args.norm_epsilon = cfg_model['layer_norm_params']["kwargs"]['eps']
    # if cfg_model['layer_norm_params']['type'] == "rms_norm":
    #     args.normalization = "RMSNorm"
    # if cfg_model['num_kv_attention_heads'] != cfg_model['num_attention_heads']:
    #     args.group_query_attention = True
    #     args.num_query_groups = cfg_model['num_kv_attention_heads']
    # position_embedding_kwargs = cfg_model['transformer_layer_params']['position_embedding_kwargs']
    # position_embedding_kwargs = {} if position_embedding_kwargs is None else position_embedding_kwargs
    # args.init_method_std = cfg_model['transformer_layer_params']['initializer']['kwargs']['sigma']
    # args.rotary_base = position_embedding_kwargs.get('base', 10000)
    # args.use_flash_attn = cfg_model.get('use_flash_attn', True)
    # args.sequence_parallel = cfg_model.get('sequence_parallel', False)

    # training args
    args.global_batch_size = config['data']['train']['global_batch_size']
    args.train_iters = config['trainer'].get('train_iters', 0)
    args.train_epoch = config['trainer'].get('epoch', -1)
    args.weight_decay = config['trainer']['optimizer']['kwargs'].get('weight_decay', 0.1)
    args.adam_beta1, args.adam_beta2 = config['trainer']['optimizer']['kwargs'].get('betas', [0.9, 0.95])
    args.adam_eps = config['trainer']['optimizer']['kwargs'].get('eps', 1.0e-8)
    args.clip_grad = config['deepspeed']['config'].get('gradient_clipping', 1.0)
    args.bf16 = config['runtime']['bf16']
    args.lr = config['trainer']['optimizer']['kwargs']['lr']
    # args.lr_decay_style = config['trainer']['lr_scheduler']['kwargs']['decay_style']
    # args.min_lr = config['trainer']['lr_scheduler']['kwargs'].get('min_lr', 1.0e-6)
    # args.lr_warmup_iters = config['trainer']['lr_scheduler']['kwargs']['lr_warmup_iters']
    # args.lr_decay_iters = config['trainer']['lr_scheduler']['kwargs']['lr_decay_iters']
    args.overlap_grad_reduce = True
    args.seed = config['runtime']['seed']
    args.ckpt_format = 'torch'
    # model parallel args
    args.tensor_model_parallel_size = config['runtime']['tensor_model_parallel_size']
    args.pipeline_model_parallel_size = config['runtime']['pipeline_model_parallel_size']
    args.context_parallel_size = config['runtime'].get('context_parallel_size', 1)
    # data args
    # args.data_path
    for cfg_hook in config['hooks']:
        if cfg_hook['type'] == "train_val_logger":
            args.log_interval = cfg_hook['kwargs'].get('log_interval')
    args.save_interval = config['saver']['save_interval']
    args.save = config['saver']['save_path']
    args.no_save_optim = not config['saver'].get('save_optim', False)
    args.no_save_rng = not config['saver'].get('save_rng_state', False)
    # args.variable_seq_lengths = config['runtime'].get('dynamic', False)

    if args.yaml_cfg is not None:
        args = validate_yaml(args, args_defaults)
    else:
        validate_args(args, args_defaults)
    set_global_variables(args, build_tokenizer=False)

    args.variable_seq_lengths = config['runtime'].get('dynamic', False)

    # dynamic checkpoint
    if cfg_model['dynamic_checkpoint']['enabled']:
        args.recompute_granularity = 'full'
        if cfg_model['dynamic_checkpoint'].get('size_map', None):
            args.recompute_method = 'dynamic_seqlen'
            args.seq_len_to_recompute_layer = cfg_model['dynamic_checkpoint']['size_map']


def yaml2args(config, extra_args_provider=None, ignore_unknown_args=True, args_defaults=dict()):
    from megatron.training.arguments import parse_args as parse_args_mg
    from megatron.training.arguments import validate_args
    from megatron.training.yaml_arguments import validate_yaml
    from megatron.training.global_vars import set_global_variables

    args = parse_args_mg(extra_args_provider, ignore_unknown_args)
    if config['model']['type'] == "intern_custom":
        cfg_model = build_internvl_cfg(config)
        args.cfg_model = cfg_model

        args.num_layers = cfg_model['num_layers']
        args.hidden_size = cfg_model['transformer_layer_params']['hidden_size']
        args.num_attention_heads = cfg_model['transformer_layer_params']['num_attention_heads']
        args.max_position_embeddings = cfg_model["word_embedings_params"].get("max_position_embeddings")
        args.seq_length = config["tokenization"]["kwargs"].get("max_seq_length", 4096)
        if args.max_position_embeddings is None or args.seq_length > args.max_position_embeddings:
            args.max_position_embeddings = args.seq_length
        args.micro_batch_size = config['data']['train']['micro_batch_size']

        logger.info("Mapping yaml args to megatron args.")
        args.ffn_hidden_size = cfg_model['transformer_layer_params']['intermediate_size']
        if cfg_model['transformer_layer_params'].get("glu_activation", "silu") == "silu":
            args.swiglu = True
        args.use_rotary_position_embeddings = True
        args.hidden_dropout = cfg_model['transformer_layer_params']['hidden_dropout']
        args.attention_dropout = cfg_model['transformer_layer_params']['attention_dropout']
        args.add_bias_linear = False
        args.norm_epsilon = cfg_model['layer_norm_params']["kwargs"]['eps']
        if cfg_model['layer_norm_params']['type'] == "rms_norm":
            args.normalization = "RMSNorm"
        if cfg_model['transformer_layer_params']['num_kv_attention_heads'] != cfg_model['transformer_layer_params']['num_attention_heads']:
            args.group_query_attention = True
            args.num_query_groups = cfg_model['transformer_layer_params']['num_kv_attention_heads']
        position_embedding_kwargs = cfg_model['transformer_layer_params']['position_embedding_kwargs']
        args.init_method_std = cfg_model['transformer_layer_params']['initializer']['kwargs']['sigma']
        args.rotary_base = position_embedding_kwargs.get('base', 10000)
        args.use_flash_attn = cfg_model.get('use_flash_attn', True)
        args.sequence_parallel = cfg_model.get('sequence_parallel', False)

    else:
        cfg_model = build_model_cfg(config)

    # import os
    # if os.environ['RANK'] == '0':
    #     import pdb;pdb.set_trace()
    # else:
    #     import time
    #     time.sleep(100000)

    args.pp_partition_method = config['model']['kwargs'].get('pp_partition_method', "uniform")
    args.pp_partition_parts = None

    # args.num_layers = cfg_model['num_layers']
    # args.hidden_size = cfg_model['hidden_size']
    # args.num_attention_heads = cfg_model['num_attention_heads']
    # args.max_position_embeddings = cfg_model["word_embedings_params"].get("max_position_embeddings")
    # args.seq_length = config["tokenization"]["kwargs"].get("max_seq_length", 4096)
    # if args.max_position_embeddings is None or args.seq_length > args.max_position_embeddings:
    #     args.max_position_embeddings = args.seq_length
    # args.micro_batch_size = config['data']['train']['micro_batch_size']

    # logger.info("Mapping yaml args to megatron args.")
    # args.ffn_hidden_size = cfg_model['intermediate_size']
    # if cfg_model['transformer_layer_params'].get("glu_activation", "silu") == "silu":
    #     args.swiglu = True
    # args.use_rotary_position_embeddings = True
    # args.hidden_dropout = cfg_model['transformer_layer_params']['hidden_dropout']
    # args.attention_dropout = cfg_model['transformer_layer_params']['attention_dropout']
    # args.add_bias_linear = False
    # args.norm_epsilon = cfg_model['layer_norm_params']["kwargs"]['eps']
    # if cfg_model['layer_norm_params']['type'] == "rms_norm":
    #     args.normalization = "RMSNorm"
    # if cfg_model['num_kv_attention_heads'] != cfg_model['num_attention_heads']:
    #     args.group_query_attention = True
    #     args.num_query_groups = cfg_model['num_kv_attention_heads']
    # position_embedding_kwargs = cfg_model['transformer_layer_params']['position_embedding_kwargs']
    # args.init_method_std = cfg_model['transformer_layer_params']['initializer']['kwargs']['sigma']
    # args.rotary_base = position_embedding_kwargs.get('base', 10000)
    # args.use_flash_attn = cfg_model.get('use_flash_attn', True)
    # args.sequence_parallel = cfg_model.get('sequence_parallel', False)

    # training args
    args.global_batch_size = config['data']['train']['global_batch_size']
    args.train_iters = config['trainer'].get('train_iters', 0)
    args.train_epoch = config['trainer'].get('epoch', -1)
    args.weight_decay = config['trainer']['optimizer']['kwargs'].get('weight_decay', 0.1)
    args.adam_beta1, args.adam_beta2 = config['trainer']['optimizer']['kwargs'].get('betas', [0.9, 0.95])
    args.adam_eps = config['trainer']['optimizer']['kwargs'].get('eps', 1.0e-8)
    args.clip_grad = config['deepspeed']['config'].get('gradient_clipping', 1.0)
    args.bf16 = config['runtime']['bf16']
    args.lr = config['trainer']['optimizer']['kwargs']['lr']
    # args.lr_decay_style = config['trainer']['lr_scheduler']['kwargs']['decay_style']
    # args.min_lr = config['trainer']['lr_scheduler']['kwargs'].get('min_lr', 1.0e-6)
    # args.lr_warmup_iters = config['trainer']['lr_scheduler']['kwargs']['lr_warmup_iters']
    # args.lr_decay_iters = config['trainer']['lr_scheduler']['kwargs']['lr_decay_iters']
    args.overlap_grad_reduce = True
    args.seed = config['runtime']['seed']
    args.ckpt_format = 'torch'
    # model parallel args
    args.tensor_model_parallel_size = config['runtime']['tensor_model_parallel_size']
    args.pipeline_model_parallel_size = config['runtime']['pipeline_model_parallel_size']
    args.context_parallel_size = config['runtime'].get('context_parallel_size', 1)
    # data args
    # args.data_path
    for cfg_hook in config['hooks']:
        if cfg_hook['type'] == "train_val_logger":
            args.log_interval = cfg_hook['kwargs'].get('log_interval')
    args.save_interval = config['saver']['save_interval']
    args.save = config['saver']['save_path']
    args.no_save_optim = not config['saver'].get('save_optim', False)
    args.no_save_rng = not config['saver'].get('save_rng_state', False)
    # args.variable_seq_lengths = config['runtime'].get('dynamic', False)

    if args.yaml_cfg is not None:
        args = validate_yaml(args, args_defaults)
    else:
        validate_args(args, args_defaults)
    set_global_variables(args, build_tokenizer=False)

    args.variable_seq_lengths = config['runtime'].get('dynamic', False)

    # dynamic checkpoint
    if cfg_model['dynamic_checkpoint']['enabled']:
        args.recompute_granularity = 'full'
        if cfg_model['dynamic_checkpoint'].get('size_map', None):
            args.recompute_method = 'dynamic_seqlen'
            args.seq_len_to_recompute_layer = cfg_model['dynamic_checkpoint']['size_map']
    args.transformer_impl = config['runtime'].get('transformer_impl', "transformer_engine")
