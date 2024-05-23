from llm.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from .modeling_internvl import InternVLChatModel
from .configuration_internvl_chat import InternVLChatConfig
import torch


def build_model(**cfg):
    model_name_or_path = cfg.get("model_name_or_path")
    config = InternVLChatConfig.from_pretrained(model_name_or_path)
    config.vision_config.drop_path_rate = cfg.get('drop_path_rate', 0)
    config.llm_config.attn_implementation = 'flash_attention_2'
    config.select_layer = cfg.get('select_layer', -1)
    config.image_fold = cfg.get('image_fold', False)
    config.dynamic_image_size = cfg.get('dynamic_image_size', True)
    config.use_thumbnail = cfg.get('use_thumbnail', True)
    config.ps_version = cfg.get('ps_version', "v2")
    config.min_dynamic_patch = cfg.get('min_dynamic_patch', 7)
    config.max_dynamic_patch = cfg.get('max_dynamic_patch', 7)
    config.force_image_size = cfg.get('force_image_size', 448)
    config.downsample_ratio = cfg.get('down_sample_ratio', 0.5)
    if config.force_image_size != config.vision_config.image_size:
        config.vision_config.image_size = config.force_image_size
    model = InternVLChatModel.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        config=config,
        ignore_mismatched_sizes=True)
    # model = InternVLChatModel._from_config(config, torch_dtype=torch.bfloat16)
    # model.vision_model.init_weights()

    return model


MODULE_ZOO_REGISTRY.register("InternVLChatModel", build_model)
