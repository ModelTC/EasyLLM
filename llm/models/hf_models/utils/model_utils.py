import os
import torch
from transformers import AutoConfig


def get_model_cfg(cfg):
    assert "model_name_or_path" in cfg, "model path not in model cfg!"
    model_path = cfg["model_name_or_path"]
    torch_dtype = cfg.get("torch_dtype", "torch.float32")
    assert torch_dtype in ["auto", "bfloat16", "float16", "float32"], "torch_dtype must be in [auto, bfloat16, float16, float32]!"  # noqa

    if torch_dtype != "auto":
        torch_dtype = getattr(torch, torch_dtype)

    config_kwargs = {
        "cache_dir": cfg.get("cache_dir", None),
        "revision": cfg.get("revision", None),
        "use_auth_token": cfg.get("use_auth_token", False),  # bool
        "trust_remote_code": cfg.get("trust_remote_code", False)
    }

    if cfg.get("model_type", None) == "baichuan":
        from ..baichuan.configuration_baichuan import BaiChuanConfig
        hf_cfg = BaiChuanConfig.from_pretrained(model_path, **config_kwargs)
    elif cfg.get("model_type", None) == "baichuan2":
        from ..baichuan2.model_7b.configuration_baichuan import BaichuanConfig
        hf_cfg = BaichuanConfig.from_pretrained(model_path, **config_kwargs)
    elif cfg.get("model_type", None) == "baichuan2-13b":
        from ..baichuan2.model_13b.configuration_baichuan import BaichuanConfig
        hf_cfg = BaichuanConfig.from_pretrained(model_path, **config_kwargs)
    elif cfg.get("model_type", None) == "qwen":
        from ..qwen.configuration_qwen import QWenConfig
        hf_cfg = QWenConfig.from_pretrained(model_path, **config_kwargs)
    elif cfg.get("model_type", None) == "internlm2":
        from ..internlm2.configuration_internlm2 import InternLM2Config
        hf_cfg = InternLM2Config.from_pretrained(model_path, **config_kwargs)
    else:
        hf_cfg = AutoConfig.from_pretrained(model_path, **config_kwargs)

    if cfg.get("_flash_attn_2_enabled", True):
        hf_cfg._flash_attn_2_enabled = True
    if cfg.get("_flash_norm_2_enabled", True):
        hf_cfg._flash_norm_2_enabled = True
    if cfg.get("_flash_rotary_2_enabled", False):
        hf_cfg._flash_rotary_2_enabled = True
    model_cfg = {
        "model_path": model_path,
        "from_tf": bool(".ckpt" in model_path),
        "config": hf_cfg,
        "cache_dir": config_kwargs["cache_dir"],
        "revision": config_kwargs["revision"],
        "use_auth_token": True if config_kwargs["use_auth_token"] else False,
        "torch_dtype": torch_dtype,
    }

    if cfg.get("model_parallel", False):
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        cuda_visiable = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        assert cuda_visiable is not None
        assert world_size == 1, 'When using model_parallel, world size must be 1 now!'
        cuda_devices = cuda_visiable.split(',')
        max_memory = {}
        for cd in cuda_devices:
            max_memory[int(cd)] = str(cfg.get("max_memory_use", 30)) + "GiB"
        device_map = "auto"

        model_cfg.update({"max_memory": max_memory, "device_map": device_map})

    return model_cfg
