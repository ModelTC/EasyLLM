from llm.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from ..utils.model_utils import get_model_cfg
from .medusa_model import MedusaModel
from .modeling_llama_kv import LlamaForCausalLM


@MODULE_ZOO_REGISTRY.register('medusa')
def build_model(**cfg):
    base_model_cfg = cfg.get('base_model')
    base_model_name_or_path = base_model_cfg['kwargs'].get('model_path')
    base_model = MODULE_ZOO_REGISTRY.build(base_model_cfg)
    medusa_num_heads = cfg.get('medusa_num_heads', 3)
    medusa_num_layers = cfg.get('medusa_num_layers', 1)
    medusa_head_name_or_path = cfg.get('medusa_head_name_or_path', None)
    medusa_model = MedusaModel.from_pretrained(medusa_head_name_or_path,
                                               base_model,
                                               medusa_num_heads,
                                               medusa_num_layers,
                                               base_model_name_or_path)
    return medusa_model


def build_kv_model(**cfg):
    model_cfg = get_model_cfg(cfg)
    model_path = model_cfg.pop("model_path")
    model = LlamaForCausalLM.from_pretrained(model_path, **model_cfg)

    if cfg.get("model_parallel", False):
        model.is_parallelizable = True
        model.model_parallel = True
    return model


# register model & tokenizer
MODULE_ZOO_REGISTRY.register("KVLlamaForCausalLM", build_kv_model)
