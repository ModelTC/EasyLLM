from .modeling_llama import LlamaForCausalLM
from ..utils.model_utils import get_model_cfg
from llm.utils.general.registry_factory import MODULE_ZOO_REGISTRY


def build_model(**cfg):
    model_cfg = get_model_cfg(cfg)
    model_path = model_cfg.pop("model_path")
    model_cfg['config']._attn_implementation = "flash_attention_2"
    model = LlamaForCausalLM.from_pretrained(model_path, **model_cfg)

    if cfg.get("model_parallel", False):
        model.is_parallelizable = True
        model.model_parallel = True
    return model


# register model & tokenizer
MODULE_ZOO_REGISTRY.register("LlamaForCausalLM", build_model)
