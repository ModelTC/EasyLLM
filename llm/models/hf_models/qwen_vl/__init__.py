from ..utils.model_utils import get_model_cfg
from .modeling_qwen import QWenLMHeadModel
from .tokenization_qwen import QWenTokenizer
from llm.utils.general.registry_factory import TOKENIZER_REGISTRY
from llm.utils.general.registry_factory import MODULE_ZOO_REGISTRY


def build_model(**cfg):
    cfg["model_type"] = "qwen_vl"
    model_cfg = get_model_cfg(cfg)
    model_path = model_cfg.pop("model_path")
    model = QWenLMHeadModel.from_pretrained(model_path, **model_cfg)

    if cfg.get("model_parallel", False):
        model.is_parallelizable = True
        model.model_parallel = True

    return model


MODULE_ZOO_REGISTRY.register("QWenForVL", build_model)
TOKENIZER_REGISTRY.register("QWenTokenizer_VL", QWenTokenizer)
