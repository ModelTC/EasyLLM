import copy

from llm.models.mg_models.base_modules.utils import check_torch_dtype
from llm.models.mg_models.llama.default_cfg import (
    update_embeding_config,
    update_ln_config,
    update_loss_config,
    _SHARED_DEFAULT_CONFIG,
    _MODEL_DEFAULT_CONFIG,
    _TRANSFROMER_LAYER_DEFAULT_CONFIG,
    _TRANSFROMER_ENGINE_DEFAULT_CONFIG
)


def update_shared_config(cfg):
    shared_default = copy.deepcopy(_SHARED_DEFAULT_CONFIG)
    model_defuault = copy.deepcopy(_MODEL_DEFAULT_CONFIG)

    keep_list = ["num_layers", "hidden_size", "num_attention_heads", "num_kv_attention_heads",
                 "intermediate_size", "parallel_output", "fp16", "bf16", "fp32_residual_connection",
                 "pretrain_causal_attention", "checkpoint_activations", "checkpoint_num_layers",
                 "dynamic_checkpoint", "pp_partition_method", "sequence_parallel"]
    model_defuault.update(cfg)
    cfg = model_defuault
    for ck in cfg:
        assert (ck in shared_default) or (ck in model_defuault), 'invalid key: {}'.format(ck)
        if ck in shared_default:
            shared_default.update({ck: cfg[ck]})
    # Dtype check
    shared_default["params_dtype"] = check_torch_dtype(shared_default["params_dtype"],
                                                       shared_default['fp16'], shared_default['bf16'])
    # Mixed precision checks.
    fp32_residual_connection = shared_default['fp32_residual_connection']
    fp16 = shared_default['fp16']
    bf16 = shared_default['bf16']
    if fp32_residual_connection:
        assert fp16 or bf16, 'residual connection in fp32 only supported when using fp16 or bf16.'
    # Model params check
    assert (shared_default["num_layers"] is not None) and \
        (shared_default["hidden_size"] is not None) and (shared_default["num_attention_heads"] is not None)
    assert shared_default["hidden_size"] % shared_default["num_attention_heads"] == 0
    if shared_default["kv_channels"] is None:
        shared_default["kv_channels"] = shared_default["hidden_size"] // shared_default["num_attention_heads"]
    else:
        assert shared_default["kv_channels"] == shared_default["hidden_size"] // shared_default["num_attention_heads"]
    if shared_default["intermediate_size"] is None:
        shared_default["intermediate_size"] = ((int(2 * (4 * shared_default["hidden_size"]) / 3) + 255) // 256) * 256

    # Backward compatibility for num_attention_heads
    if shared_default["num_kv_attention_heads"] is None:
        shared_default["num_kv_attention_heads"] = shared_default['num_attention_heads']

    cfg.update(shared_default)
    cfg_keys = list(cfg.keys())
    for ck in cfg_keys:
        if ck not in keep_list:
            cfg.pop(ck)

    return cfg, shared_default


def update_transformer_layer_config(cfg, shared_default, ln_cfg, num_layers, cfg_engine):
    transformer_layer_defualt = copy.deepcopy(_TRANSFROMER_LAYER_DEFAULT_CONFIG)
    transformer_engine_defualt = copy.deepcopy(_TRANSFROMER_ENGINE_DEFAULT_CONFIG)
    shared_keys_mapping = {"self_attn_mask_type": "attn_mask_type",
                           "position_embedding_type": "position_embedding_type",
                           "position_embedding_kwargs": "position_embedding_kwargs",
                           "params_dtype": "params_dtype",
                           "fp16": "fp16",
                           "bf16": "bf16",
                           "fp32_residual_connection": "fp32_residual_connection",
                           "kv_channels": "kv_channels",
                           "num_attention_heads": "num_attention_heads",
                           "num_kv_attention_heads": "num_kv_attention_heads",
                           "hidden_size": "hidden_size",
                           "use_cpu_initialization": "use_cpu_initialization",
                           "sync_tp_duplicated_parameters": "sync_tp_duplicated_parameters",
                           "sequence_parallel": "sequence_parallel",
                           "use_flash_attn": "use_flash_attn",
                           "hidden_dropout": "hidden_dropout",
                           "intermediate_size": "intermediate_size",
                           "seq_length": "seq_length",
                           "micro_batch_size": "micro_batch_size"}
    for tlk in shared_keys_mapping:
        sk = shared_keys_mapping[tlk]
        # if tlk in cfg:
        # assert cfg[tlk] == shared_default[sk], "the key value of {} does not match with the shared configs and transformer layer configs".format(tlk)        # noqa
        transformer_layer_defualt.update({tlk: shared_default[sk]})
    transformer_layer_defualt.update(cfg)
    if transformer_layer_defualt["layer_norm"] is None:
        transformer_layer_defualt["layer_norm"] = ln_cfg
    if transformer_layer_defualt["output_initializer"]["kwargs"].get("num_layers", None):
        transformer_layer_defualt["output_initializer"]["kwargs"]["num_layers"] = num_layers
    transformer_engine_defualt.update(cfg_engine)
    # Assert Activation Function
    glu_activation = transformer_layer_defualt['glu_activation']
    bias_gelu_fusion = transformer_layer_defualt['bias_gelu_fusion']
    if glu_activation is not None and bias_gelu_fusion:
        raise ValueError("if glu-activation is used, please set bias-gelu-fusion to false")

    return transformer_layer_defualt


def update_model_cfg(cfg):
    word_embedings_cfg = cfg.pop('word_embedings_params', {})
    layer_norm_cfg = cfg.pop('layer_norm_params', {})
    transformer_layer_cfg = cfg.pop('transformer_layer_params', {})
    transformer_engine_cfg = cfg.pop('transformer_engine_params', {})
    lm_head_cfg = cfg.pop('lm_head_params', {})
    loss_cfg = cfg.pop('loss_params', {})

    cfg, shared_cfg = update_shared_config(cfg)

    word_embedings_cfg = update_embeding_config(word_embedings_cfg, shared_cfg, as_head=False)
    layer_norm_cfg['type'] = layer_norm_cfg.get('type', 'rms_norm')
    layer_norm_cfg['kwargs'] = update_ln_config(layer_norm_cfg.get('kwargs', {}), shared_cfg)
    transformer_layer_cfg = update_transformer_layer_config(transformer_layer_cfg, shared_cfg,
                                                            layer_norm_cfg, num_layers=cfg['num_layers'], cfg_engine=transformer_engine_cfg)
    lm_head_cfg = update_embeding_config(lm_head_cfg, shared_cfg, as_head=True)

    loss_cfg['type'] = loss_cfg.get('type', 'softmax_cross_entropy')
    loss_cfg['kwargs'] = update_loss_config(loss_cfg.get('kwargs', {}), shared_cfg)
    cfg['word_embedings_params'] = word_embedings_cfg
    cfg['layer_norm_params'] = layer_norm_cfg
    cfg['transformer_layer_params'] = transformer_layer_cfg
    cfg['lm_head_params'] = lm_head_cfg
    cfg['loss_params'] = loss_cfg

    return cfg
