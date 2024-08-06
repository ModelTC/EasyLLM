from llm.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from .modeling_internvl import InternVLChatModel
from .configuration_internvl_chat import InternVLChatConfig
import torch
from .configuration_intern_vit import InternVisionConfig
from .modeling_intern_vit import InternVisionModel
from .modeling_internlm2 import InternLM2ForCausalLM
from .modeling_internvl import InternVLChatModel
from transformers import AutoConfig
from llm.utils.general.log_helper import default_logger as logger


def build_model(**cfg):
    model_name_or_path = cfg.get("model_name_or_path", None)
    if model_name_or_path is None:
        vision_config = InternVisionConfig.from_pretrained(cfg["vision_path"])
        vision_config.drop_path_rate = cfg["drop_path_rate"]
        vision_model = InternVisionModel.from_pretrained(cfg["vision_path"], torch_dtype=torch.bfloat16, config=vision_config)
        # vision_model = InternVisionModel._from_config(vision_config, torch_dtype=torch.bfloat16)

        # llm_config = AutoConfig.from_pretrained(cfg["llm_path"], trust_remote_code=True)
        from .configuration_internlm2 import InternLM2Config
        llm_config = InternLM2Config.from_pretrained(cfg["llm_path"], trust_remote_code=True)

        if llm_config.model_type == 'llama':
            from .modeling_llama import LlamaForCausalLM
            llm_config._attn_implementation = "flash_attention_2"
            llm = LlamaForCausalLM.from_pretrained(cfg["llm_path"], torch_dtype=torch.bfloat16, config=llm_config, trust_remote_code=True)
            # llm = LlamaForCausalLM._from_config(llm_config, torch_dtype=torch.bfloat16)
        elif llm_config.model_type == 'internlm2':
            llm_config.attn_implementation = "flash_attention_2"
            # llm = InternLM2ForCausalLM.from_pretrained(cfg["llm_path"], torch_dtype=torch.bfloat16, config=llm_config, trust_remote_code=True)
            llm = InternLM2ForCausalLM._from_config(llm_config, torch_dtype=torch.bfloat16)
        else:
            from transformers import AutoModelForCausalLM
            llm = AutoModelForCausalLM.from_pretrained(cfg["llm_path"], torch_dtype=torch.bfloat16, config=llm_config, trust_remote_code=True)
        

        internvl_chat_config = InternVLChatConfig(vision_config.to_dict(),
                                                  llm_config.to_dict(),
                                                  downsample_ratio=cfg["down_sample_ratio"],
                                                  pad2square=cfg["pad2square"],
                                                  template="internvl-chat",
                                                  select_layer=cfg["select_layer"],
                                                  dynamic_image_size=cfg["dynamic_image_size"],
                                                  use_thumbnail=cfg["use_thumbnail"],
                                                  ps_version='v2',
                                                  min_dynamic_patch=cfg["min_dynamic_patch"],
                                                  max_dynamic_patch=cfg["max_dynamic_patch"])
        internvl_chat_config.force_image_size = cfg["force_image_size"]
        model = InternVLChatModel(internvl_chat_config, vision_model, llm)

        if cfg.get("mlp_path", None) is not None:
            state_dict = torch.load(cfg["mlp_path"], map_location='cpu')
            message = model.mlp1.load_state_dict(state_dict)
            logger.info(message)
    else:
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
            config=config,)
            #ignore_mismatched_sizes=True)

        # model = InternVLChatModel._from_config(config, torch_dtype=torch.bfloat16)
        # model.vision_model.init_weights()

    return model


MODULE_ZOO_REGISTRY.register("InternVLChatModel", build_model)
