from .tokenization_baichuan import BaichuanTokenizer as BaiChuan2Tokenizer
from .modeling_baichuan import BaichuanForCausalLM as BaiChuan2ForCausalLM
from ...utils.model_utils import get_model_cfg
from llm.utils.general.registry_factory import TOKENIZER_REGISTRY
from llm.utils.general.registry_factory import MODULE_ZOO_REGISTRY


def build_model(**cfg):
    cfg["model_type"] = "baichuan2"
    model_cfg = get_model_cfg(cfg)
    model_path = model_cfg.pop("model_path")
    model = BaiChuan2ForCausalLM.from_pretrained(model_path, **model_cfg)

    if cfg.get("model_parallel", False):
        model.is_parallelizable = True
        model.model_parallel = True

    return model


MODULE_ZOO_REGISTRY.register("BaiChuan2ForCausalLM", build_model)
TOKENIZER_REGISTRY.register("BaiChuan2Tokenizer", BaiChuan2Tokenizer)
