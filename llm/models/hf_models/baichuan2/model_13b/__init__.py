from .tokenization_baichuan import BaichuanTokenizer
from .modeling_baichuan import BaichuanForCausalLM
from ...utils.model_utils import get_model_cfg
from llm.utils.general.registry_factory import TOKENIZER_REGISTRY
from llm.utils.general.registry_factory import MODULE_ZOO_REGISTRY


def build_model(**cfg):
    cfg["model_type"] = "baichuan2-13b"
    model_cfg = get_model_cfg(cfg)
    model_path = model_cfg.pop("model_path")
    model = BaichuanForCausalLM.from_pretrained(model_path, **model_cfg)

    if cfg.get("model_parallel", False):
        model.is_parallelizable = True
        model.model_parallel = True

    return model


MODULE_ZOO_REGISTRY.register("BaiChuan2ForCausalLM13B", build_model)
TOKENIZER_REGISTRY.register("BaiChuan2Tokenizer13B", BaichuanTokenizer)
