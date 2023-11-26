from ..utils.model_utils import get_model_cfg
from .modeling_internlm import InternLMForCausalLM
from .tokenization_internlm import InternLMTokenizer
from llm.utils.general.registry_factory import TOKENIZER_REGISTRY
from llm.utils.general.registry_factory import MODULE_ZOO_REGISTRY


def build_model(**cfg):
    model_cfg = get_model_cfg(cfg)
    model_path = model_cfg.pop("model_path")
    model = InternLMForCausalLM.from_pretrained(model_path, **model_cfg)

    if cfg.get("model_parallel", False):
        model.is_parallelizable = True
        model.model_parallel = True

    return model


MODULE_ZOO_REGISTRY.register("InternLMForCausalLM", build_model)
TOKENIZER_REGISTRY.register("InternLMTokenizer", InternLMTokenizer)
