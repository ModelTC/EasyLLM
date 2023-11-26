
from .tokenization_baichuan import BaiChuanTokenizer
from .modeling_baichuan import BaiChuanForCausalLM
from ..utils.model_utils import get_model_cfg
from llm.utils.general.registry_factory import TOKENIZER_REGISTRY
from llm.utils.general.registry_factory import MODULE_ZOO_REGISTRY


def build_model(**cfg):
    cfg["model_type"] = "baichuan"
    model_cfg = get_model_cfg(cfg)
    model_path = model_cfg.pop("model_path")
    model = BaiChuanForCausalLM.from_pretrained(model_path, **model_cfg)

    if cfg.get("model_parallel", False):
        model.is_parallelizable = True
        model.model_parallel = True

    return model


MODULE_ZOO_REGISTRY.register("BaiChuanForCausalLM", build_model)
TOKENIZER_REGISTRY.register("BaiChuanTokenizer", BaiChuanTokenizer)
