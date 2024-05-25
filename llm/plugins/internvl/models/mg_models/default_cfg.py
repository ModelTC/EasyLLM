import copy
# import torch
# from ..modules.enums import AttnMaskType

# from llm.utils.general.log_helper import default_logger as logger
# from .llama import LlamaModelPipe, set_load_save_func
from llm.models.mg_models.base_modules.modules.enums import AttnMaskType
from llm.models.mg_models.base_modules.utils import check_torch_dtype, get_attn_mask_type


_SHARED_DEFAULT_CONFIG = {
    "num_layers": None,
    "vocab_size": None,
    "unpad_vocab_size": None,
    "kv_channels": None,
    "num_attention_heads": None,
    "num_kv_attention_heads": None,
    "intermediate_size": None,
    "seq_length": None,
    "hidden_size": None,
    "hidden_dropout": 0,
    "position_embedding_type": "rotary",
    "position_embedding_kwargs": None,
    "params_dtype": None,
    "pretrain_causal_attention": True,
    "use_cpu_initialization": False,
    "parallel_output": True,
    "sync_tp_duplicated_parameters": True,
    "sequence_parallel": False,
    "fp16": False,
    "bf16": False,
    "attn_mask_type": "causal",
    "fp32_residual_connection": False,
    "micro_batch_size": 1,
    "use_flash_attn": True,
    "num_husky_layers": None,
    "eps": 1e-6,
    # vision
    "vision_hidden_size": None,
    "vision_image_size": 448,
    "vision_patch_size": None,
    "vision_intermediate_size": None,
    "vision_num_attention_heads": None,
    "vision_attention_dropout": None,
    "vision_layer_norm_eps": None,
    "vit_select_layer": None,
    "img_context_token_id": None,
    "vision_sequence_parallel": False,
    "drop_path_rate": 0.2,
    "initializer_factor": 0.1,
    "qk_normalization": True,
    "proj_dropout": 0.0,
    "image_fold": False
}


_MODEL_DEFAULT_CONFIG = {
    "checkpoint_activations": True,
    "checkpoint_num_layers": 1,
    "dynamic_checkpoint": None,
    "pp_partition_method": "type:transformer|embedding",
    "profile_path": None,
    "layer_profile": False,
    "global_batch_size": 1
}


def update_shared_config(cfg):
    shared_default = copy.deepcopy(_SHARED_DEFAULT_CONFIG)
    model_defuault = copy.deepcopy(_MODEL_DEFAULT_CONFIG)

    keep_list = ["num_layers", "parallel_output", "fp16", "bf16", "fp32_residual_connection",
                 "pretrain_causal_attention", "checkpoint_activations", "checkpoint_num_layers",
                 "dynamic_checkpoint", "pp_partition_method", "sequence_parallel", "num_vit_layers",
                 "profile_path", "layer_profile", "global_batch_size"]
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


_EMBEDINGS_DEFAULT_CONFIG = {
    "num_tokentypes": 0,
    "max_position_embeddings": None,
    "use_bnb_optimizer": False,
    "initializer": {
        "type": "normal",
        "kwargs": {"sigma": 0.02}
    }
}


def update_embeding_config(cfg, shared_default, as_head=False):
    embeding_defualt = copy.deepcopy(_EMBEDINGS_DEFAULT_CONFIG)
    shared_keys_mapping = {"hidden_size": "hidden_size",
                           "vocab_size": "vocab_size",
                           "embedding_dropout_prob": "hidden_dropout",
                           "position_embedding_type": "position_embedding_type",
                           "params_dtype": "params_dtype",
                           "pretrain_causal_attention": "pretrain_causal_attention",
                           "use_cpu_initialization": "use_cpu_initialization",
                           "sequence_parallel": "sequence_parallel"}
    if as_head:
        shared_keys_mapping.update({"parallel_output": "parallel_output"})
    if not as_head:
        shared_keys_mapping.update({"img_context_token_id": "img_context_token_id"})
    for emk in shared_keys_mapping:
        sk = shared_keys_mapping[emk]
        # if emk in cfg:
        # assert cfg[emk] == shared_default[sk], "the key value of {} does not match with the shared configs and embeding configs".format(emk)        # noqa
        embeding_defualt.update({emk: shared_default[sk]})
    embeding_defualt.update(cfg)
    # Position embeding check
    position_embedding_type = shared_default['position_embedding_type']
    seq_length = shared_default['seq_length']
    max_position_embeddings = embeding_defualt['max_position_embeddings']
    if position_embedding_type == "absolute" or position_embedding_type == "alibi":
        assert max_position_embeddings is not None
        if seq_length is not None:
            assert max_position_embeddings >= seq_length
    else:
        assert max_position_embeddings is None
    return embeding_defualt


_LAYER_NORM_DEFAULT_CONFIG = {
    "eps": 1e-6
}


def update_ln_config(cfg, shared_default):
    ln_defualt = copy.deepcopy(_LAYER_NORM_DEFAULT_CONFIG)
    shared_keys_mapping = {"sync_tp_duplicated_parameters": "sync_tp_duplicated_parameters",
                           "normalized_shape": "hidden_size", "use_flash_attn": "use_flash_attn",
                           "bf16": "bf16", "sequence_parallel": "sequence_parallel", "eps": "eps"}
    for lnk in shared_keys_mapping:
        sk = shared_keys_mapping[lnk]
        # if lnk in cfg:
            # assert cfg[lnk] == shared_default[sk], "the key value of {} does not match with the shared configs and layer norm configs".format(lnk)        # noqa
        ln_defualt.update({lnk: shared_default[sk]})
    ln_defualt.update(cfg)
    return ln_defualt


_VISION_LAYER_NORM_DEFAULT_CONFIG = {
    "eps": 1e-6
}


def update_vision_ln_config(cfg, shared_default):
    ln_defualt = copy.deepcopy(_LAYER_NORM_DEFAULT_CONFIG)
    shared_keys_mapping = {"sync_tp_duplicated_parameters": "sync_tp_duplicated_parameters",
                           "normalized_shape": "vision_hidden_size", "use_flash_attn": "use_flash_attn",
                           "bf16": "bf16", "sequence_parallel": "sequence_parallel", "eps": "vision_layer_norm_eps"}
    for lnk in shared_keys_mapping:
        sk = shared_keys_mapping[lnk]
        # if lnk in cfg:
            # assert cfg[lnk] == shared_default[sk], "the key value of {} does not match with the shared configs and layer norm configs".format(lnk)        # noqa
        ln_defualt.update({lnk: shared_default[sk]})
    ln_defualt.update(cfg)
    return ln_defualt


_TRANSFROMER_LAYER_DEFAULT_CONFIG = {
    "layer_type": "encoder",
    "layer_norm": None,
    "apply_residual_connection_post_layernorm": False,
    "apply_query_key_layer_scaling": True,
    "attention_softmax_in_fp32": False,
    "masked_softmax_fusion": True,
    "attention_dropout": 0,
    "bias_gelu_fusion": False,
    "glu_activation": "silu",
    "use_openai_gelu": False,
    "onnx_safe": None,
    "bias_dropout_fusion": True,
    "initializer": {
        "type": "normal",
        "kwargs": {"sigma": 0.02}
    },
    "output_initializer": {
        "type": "scaled_normal",
        "kwargs": {"sigma": 0.02, "num_layers": 32}
    },
    'qkv_pack': False,
    'attention_qkv_bias': False,
    'attention_o_bias': False
}


def update_transformer_layer_config(cfg, shared_default, ln_cfg, num_layers):
    transformer_layer_defualt = copy.deepcopy(_TRANSFROMER_LAYER_DEFAULT_CONFIG)
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
    # Assert Activation Function
    glu_activation = transformer_layer_defualt['glu_activation']
    bias_gelu_fusion = transformer_layer_defualt['bias_gelu_fusion']
    if glu_activation is not None and bias_gelu_fusion:
        raise ValueError("if glu-activation is used, please set bias-gelu-fusion to false")
    return transformer_layer_defualt


def update_loss_config(cfg, shared_default):
    is_prefix = get_attn_mask_type(shared_default["attn_mask_type"]) is AttnMaskType.prefix
    cfg.update({"is_prefix": is_prefix})
    align_vocab_size_strict = cfg.pop('align_vocab_size_strict', False)
    if align_vocab_size_strict:
        vocab_size = shared_default['vocab_size']
        unpad_vocab_size = shared_default['unpad_vocab_size']
        if (vocab_size != unpad_vocab_size):
            cfg.update({'cut_size': vocab_size - unpad_vocab_size})
    return cfg


_VISION_EMBEDINGS_DEFAULT_CONFIG = {
    "hidden_size": None,
    "image_size": None,
    "patch_size": None,
    "params_dtype": None,
    "sequence_parallel": False
}


def update_vision_embeding_config(cfg, shared_default):
    vision_embeding_defualt = copy.deepcopy(_VISION_EMBEDINGS_DEFAULT_CONFIG)
    shared_keys_mapping = {"hidden_size": "vision_hidden_size",
                           "image_size": "vision_image_size",
                           "patch_size": "vision_patch_size",
                           "params_dtype": "params_dtype",
                           "sequence_parallel": "vision_sequence_parallel",
                           "image_fold": "image_fold"}

    for emk in shared_keys_mapping:
        sk = shared_keys_mapping[emk]
        vision_embeding_defualt.update({emk: shared_default[sk]})
    vision_embeding_defualt.update(cfg)
    return vision_embeding_defualt


_VISION_TRANSFROMER_LAYER_DEFAULT_CONFIG = {
    "hidden_size": None,
    "intermediate_size": None,
    "num_attention_heads": None,
    "attention_dropout": 0.0,
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-05,
    "use_flash_attn": False,
    "params_dtype": None,
    "sequence_parallel": False,
    "vit_select_layer": None
}


def update_vision_transformer_layer_config(cfg, shared_default, layer_norm_cfg):  # , ln_cfg, num_layers):
    vision_transformer_layer_defualt = copy.deepcopy(_VISION_TRANSFROMER_LAYER_DEFAULT_CONFIG)
    shared_keys_mapping = {"hidden_size": "vision_hidden_size",
                           "intermediate_size": "vision_intermediate_size",
                           "num_attention_heads": "vision_num_attention_heads",
                           "attention_dropout": "vision_attention_dropout",
                           "layer_norm_eps": "vision_layer_norm_eps",
                           "use_flash_attn": "use_flash_attn",
                           "params_dtype": "params_dtype",
                           "sequence_parallel": "vision_sequence_parallel",
                           "vit_select_layer": "vit_select_layer",
                           "drop_path_rate": "drop_path_rate",
                           "initializer_factor": "initializer_factor",
                           "proj_dropout": "proj_dropout",
                           "qk_normalization": "qk_normalization"}
    for emk in shared_keys_mapping:
        sk = shared_keys_mapping[emk]
        vision_transformer_layer_defualt.update({emk: shared_default[sk]})
    vision_transformer_layer_defualt.update(cfg)
    vision_transformer_layer_defualt['layer_norm'] = layer_norm_cfg
    vision_transformer_layer_defualt['atten_layer_norm'] = layer_norm_cfg
    return vision_transformer_layer_defualt


_VISION_EXTRACT_FEAT_DEFAULT_CONFIG = {
    "vit_hidden_size": None,
    "llm_hidden_size": None,
    "vit_select_layer": None,
    "params_dtype": None,
    "sequence_parallel": False,
    "image_size": None,
    "patch_size": None,
}


def update_vision_extract_feat_config(cfg, shared_default):
    vision_extract_feat_default = copy.deepcopy(_VISION_EXTRACT_FEAT_DEFAULT_CONFIG)
    shared_keys_mapping = {"vit_hidden_size": "vision_hidden_size",
                           "llm_hidden_size": "hidden_size",
                           "vit_select_layer": "vit_select_layer",
                           "params_dtype": "params_dtype",
                           "sequence_parallel": "vision_sequence_parallel",
                           "image_size": "vision_image_size",
                           "patch_size": "vision_patch_size",
                           "image_fold": "image_fold"}
    for emk in shared_keys_mapping:
        sk = shared_keys_mapping[emk]
        vision_extract_feat_default.update({emk: shared_default[sk]})
    vision_extract_feat_default.update(cfg)
    return vision_extract_feat_default


def update_model_cfg(cfg):
    word_embedings_cfg = cfg.pop('word_embedings_params', {})
    layer_norm_cfg = cfg.pop('layer_norm_params', {})
    transformer_layer_cfg = cfg.pop('transformer_layer_params', {})
    lm_head_cfg = cfg.pop('lm_head_params', {})
    loss_cfg = cfg.pop('loss_params', {})
    vision_embedings_cfg = cfg.pop('vision_embedings_params', {})
    vision_layer_norm_cfg = cfg.pop('vision_layer_norm_cfg', {})
    vision_atten_norm_cfg = cfg.pop('vision_atten_norm_cfg', {}) # noqa
    vision_transformer_layer_cfg = cfg.pop('vision_transformer_layer_params', {})
    vision_extract_feat_cfg = cfg.pop('vision_extract_feat_params', {})

    cfg, shared_cfg = update_shared_config(cfg)

    vision_embedings_cfg = update_vision_embeding_config(vision_embedings_cfg, shared_cfg)
    vision_layer_norm_cfg['type'] = vision_layer_norm_cfg.get('type', "rms_norm")
    vision_layer_norm_cfg['kwargs'] = update_vision_ln_config(vision_layer_norm_cfg.get('kwargs', {}), shared_cfg)
    vision_transformer_layer_cfg = update_vision_transformer_layer_config(vision_transformer_layer_cfg, shared_cfg, vision_layer_norm_cfg)
    vision_transformer_layer_cfg["num_vit_layers"] = cfg["num_vit_layers"]
    vision_extract_feat_cfg = update_vision_extract_feat_config(vision_extract_feat_cfg, shared_cfg)

    word_embedings_cfg = update_embeding_config(word_embedings_cfg, shared_cfg, as_head=False)
    layer_norm_cfg['type'] = layer_norm_cfg.get('type', 'rms_norm')
    layer_norm_cfg['kwargs'] = update_ln_config(layer_norm_cfg.get('kwargs', {}), shared_cfg)
    transformer_layer_cfg = update_transformer_layer_config(transformer_layer_cfg, shared_cfg,
                                                            layer_norm_cfg, num_layers=cfg['num_layers'])
    lm_head_cfg = update_embeding_config(lm_head_cfg, shared_cfg, as_head=True)

    loss_cfg['type'] = loss_cfg.get('type', 'softmax_cross_entropy')
    loss_cfg['kwargs'] = update_loss_config(loss_cfg.get('kwargs', {}), shared_cfg)
    cfg['word_embedings_params'] = word_embedings_cfg
    cfg['layer_norm_params'] = layer_norm_cfg
    cfg['transformer_layer_params'] = transformer_layer_cfg
    cfg['lm_head_params'] = lm_head_cfg
    cfg['loss_params'] = loss_cfg
    cfg['vision_embedings_params'] = vision_embedings_cfg
    cfg['vision_transformer_layer_params'] = vision_transformer_layer_cfg
    cfg['vision_extract_feat_params'] = vision_extract_feat_cfg
    cfg['drop_path_rate'] = vision_transformer_layer_cfg['drop_path_rate']

    return cfg
