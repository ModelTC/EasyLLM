from llm.utils.general.registry_factory import MODULE_ZOO_REGISTRY

from .base_modules.modules.fp16_module import Float16Module
from .base_modules.post_process import *     # noqa
from .base_modules.modules.enums import *    # noqa
from .base_modules.utils import unwrap_model, log_trainable_params
from .base_modules.lora import convert_layer_to_lora
from .llama import *        # noqa
from .llama.llama import _LLAMA_MODELS


_ALL_BASE_MODELS = {}
for key in _LLAMA_MODELS:
    _ALL_BASE_MODELS[key] = _LLAMA_MODELS[key]


def get_layer_info(cfg_model):
    model_type = cfg_model['type']
    model_kwargs = cfg_model.get('kwargs', {})
    if model_type in _ALL_BASE_MODELS:
        num_layers = _ALL_BASE_MODELS[model_type]['num_layers']
    else:
        assert model_kwargs.get('num_layers', None)
        num_layers = model_kwargs['num_layers']
    checkpoint_num_layers = model_kwargs.get('checkpoint_num_layers', 1)
    return num_layers, checkpoint_num_layers


def build_model(tokenizer, cfg, lora_mode, cfg_lora, base_type):
    cfg_model = cfg['model']
    if 'kwargs' not in cfg_model:
        cfg_model['kwargs'] = {}
    if getattr(tokenizer, 'padded_vocab_size', None) is not None \
       and tokenizer.padded_vocab_size != len(tokenizer):
        vocab_size = tokenizer.padded_vocab_size
    else:
        vocab_size = len(tokenizer)
    cfg_model['kwargs'].update({"fp16": cfg['runtime'].get('fp16', False),
                                "bf16": cfg['runtime'].get('bf16', False),
                                "unpad_vocab_size": getattr(tokenizer, 'vocab_size'),
                                "vocab_size": vocab_size,
                                "micro_batch_size": cfg['data'][base_type]['micro_batch_size'],
                                "seq_length": cfg['data'][base_type]['seq_length']})       # noqa
    is_qkv_pack = cfg_model['kwargs'].get("transformer_layer_params", {}).get("qkv_pack", False)
    pretrain_type = cfg['loader'].get("pretrain_type", "llama")
    if is_qkv_pack:
        assert "pack" in pretrain_type, "qkv_pack must load the model in pack type. currently support llama_pack and internlm2_pack!"
    else:
        assert "pack" not in pretrain_type, "You load the model in pack type, but do not set qkv_pack as True!"
    model = MODULE_ZOO_REGISTRY.build(cfg_model)
    if lora_mode and (cfg_lora is not None):
        model = convert_layer_to_lora(model, cfg_lora)
    return model


__all__ = ['build_model', 'Float16Module', 'unwrap_model', 'log_trainable_params',
           'PositionEmbeddingType', 'AttnMaskType', 'AttnMaskType', 'AttnMaskType', "get_layer_info"]
