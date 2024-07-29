import torch
import os
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import (strtobool,
                                is_safetensors_available,
                                is_bitsandbytes_available,
                                is_peft_available,
                                is_offline_mode,
                                SAFE_WEIGHTS_NAME,
                                SAFE_WEIGHTS_INDEX_NAME,
                                logging,
                                ContextManagers)
from transformers.utils.hub import convert_file_size_to_int, get_checkpoint_shard_files
from transformers.modeling_utils import no_init_weights
from transformers.integrations import deepspeed_config, is_deepspeed_zero3_enabled
from llm.utils.env import dist_env, initialize_model_parallel
try:
    initialize_model_parallel()
except:
    pass

def is_fsdp_enabled():
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and strtobool(os.environ.get("ACCELERATE_USE_FSDP", "False")) == 1
        and strtobool(os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING", "False")) == 1
    )

from transformers.utils.quantization_config import QuantizationMethod
if is_peft_available():
    from transformers.utils import find_adapter_config_file


def _add_variant(weights_name: str, variant: Optional[str] = None) -> str:
    if variant is not None:
        splits = weights_name.split(".")
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = ".".join(splits)

    return weights_name


logger = logging.get_logger(__name__)


@classmethod
def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.

        Args:
            torch_dtype (`torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model under this dtype.
        """
        torch_dtype = kwargs.pop("torch_dtype", None)
        use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)

        # override default dtype if needed
        dtype_orig = None
        if torch_dtype is not None:
            dtype_orig = cls._set_default_torch_dtype(torch_dtype)

        config = copy.deepcopy(config)  # We do not want to modify the config inplace in _from_config.
        config._attn_implementation = kwargs.pop("attn_implementation", None)
        config = cls._autoset_attn_implementation(
            config, use_flash_attention_2=use_flash_attention_2, check_device_map=False
        )

        if is_deepspeed_zero3_enabled():
            import deepspeed

            logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
            # this immediately partitions the model across all gpus, to avoid the overhead in time
            # and memory copying it on CPU or each GPU first
         #   with deepspeed.zero.Init(config_dict_or_path=deepspeed_config()):
            with deepspeed.zero.MiCS_Init(data_parallel_group=dist_env.get_data_parallel_group(),config_dict_or_path=deepspeed_config(), mpu=dist_env):
                model = cls(config, **kwargs)
        else:
            model = cls(config, **kwargs)

        # restore default dtype if it was modified
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)

        return model

@classmethod
def from_pretrained(
    cls,
    pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
    *model_args,
    config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    ignore_mismatched_sizes: bool = False,
    force_download: bool = False,
    local_files_only: bool = False,
    token: Optional[Union[str, bool]] = None,
    revision: str = "main",
    use_safetensors: bool = None,
    **kwargs,
):
    r"""
    Instantiate a pretrained pytorch model from a pre-trained model configuration.

    The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
    the model, you should first set it back in training mode with `model.train()`.

    The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
    pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
    task.

    The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
    weights are discarded.

    Parameters:
        pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
            Can be either:

                - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                    user or organization name, like `dbmdz/bert-base-german-cased`.
                - A path to a *directory* containing model weights saved using
                    [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                    this case, `from_tf` should be set to `True` and a configuration object should be provided as
                    `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                    PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
                - A path or url to a model folder containing a *flax checkpoint file* in *.msgpack* format (e.g,
                    `./flax_model/` containing `flax_model.msgpack`). In this case, `from_flax` should be set to
                    `True`.
                - `None` if you are both providing the configuration and state dictionary (resp. with keyword
                    arguments `config` and `state_dict`).
        model_args (sequence of positional arguments, *optional*):
            All remaining positional arguments will be passed to the underlying model's `__init__` method.
        config (`Union[PretrainedConfig, str, os.PathLike]`, *optional*):
            Can be either:

                - an instance of a class derived from [`PretrainedConfig`],
                - a string or path valid as input to [`~PretrainedConfig.from_pretrained`].

            Configuration for the model to use instead of an automatically loaded configuration. Configuration can
            be automatically loaded when:

                - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                    model).
                - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the
                    save directory.
                - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
                    configuration JSON file named *config.json* is found in the directory.
        state_dict (`Dict[str, torch.Tensor]`, *optional*):
            A state dictionary to use instead of a state dictionary loaded from saved weights file.

            This option can be used if you want to create a model from a pretrained configuration but load your own
            weights. In this case though, you should check if using [`~PreTrainedModel.save_pretrained`] and
            [`~PreTrainedModel.from_pretrained`] is not a simpler option.
        cache_dir (`Union[str, os.PathLike]`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the
            standard cache should not be used.
        from_tf (`bool`, *optional*, defaults to `False`):
            Load the model weights from a TensorFlow checkpoint save file (see docstring of
            `pretrained_model_name_or_path` argument).
        from_flax (`bool`, *optional*, defaults to `False`):
            Load the model weights from a Flax checkpoint save file (see docstring of
            `pretrained_model_name_or_path` argument).
        ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
            Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
            as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
            checkpoint with 3 labels).
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force the (re-)download of the model weights and configuration files, overriding the
            cached versions if they exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to delete incompletely received files. Will attempt to resume the download if such a
            file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
        output_loading_info(`bool`, *optional*, defaults to `False`):
            Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
        local_files_only(`bool`, *optional*, defaults to `False`):
            Whether or not to only look at local files (i.e., do not try to download the model).
        token (`str` or `bool`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
            the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.

            <Tip>

            To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>".

            </Tip>

        mirror (`str`, *optional*):
            Mirror source to accelerate downloads in China. If you are from China and have an accessibility
            problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
            Please refer to the mirror site for more information.
        _fast_init(`bool`, *optional*, defaults to `True`):
            Whether or not to disable fast initialization.

            <Tip warning={true}>

            One should only disable *_fast_init* to ensure backwards compatibility with `transformers.__version__ <
            4.6.0` for seeded model initialization. This argument will be removed at the next major version. See
            [pull request 11471](https://github.com/huggingface/transformers/pull/11471) for more information.

            </Tip>

        > Parameters for big model inference

        low_cpu_mem_usage(`bool`, *optional*):
            Tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
            This is an experimental feature and a subject to change at any moment.
        torch_dtype (`str` or `torch.dtype`, *optional*):
            Override the default `torch.dtype` and load the model under a specific `dtype`. The different options
            are:

            1. `torch.float16` or `torch.bfloat16` or `torch.float`: load in a specified
                `dtype`, ignoring the model's `config.torch_dtype` if one exists. If not specified
                - the model will get loaded in `torch.float` (fp32).

            2. `"auto"` - A `torch_dtype` entry in the `config.json` file of the model will be
                attempted to be used. If this entry isn't found then next check the `dtype` of the first weight in
                the checkpoint that's of a floating point type and use that as `dtype`. This will load the model
                using the `dtype` it was saved in at the end of the training. It can't be used as an indicator of how
                the model was trained. Since it could be trained in one of half precision dtypes, but saved in fp32.

            <Tip>

            For some models the `dtype` they were trained in is unknown - you may try to check the model's paper or
            reach out to the authors and ask them to add this information to the model's card and to insert the
            `torch_dtype` entry in `config.json` on the hub.

            </Tip>

        device_map (`str` or `Dict[str, Union[int, str, torch.device]]` or `int` or `torch.device`, *optional*):
            A map that specifies where each submodule should go. It doesn't need to be refined to each
            parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
            same device. If we only pass the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank
            like `1`) on which the model will be allocated, the device map will map the entire model to this
            device. Passing `device_map = 0` means put the whole model on GPU 0.

            To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
            more information about each option see [designing a device
            map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
        max_memory (`Dict`, *optional*):
            A dictionary device identifier to maximum memory. Will default to the maximum memory available for each
            GPU and the available CPU RAM if unset.
        offload_folder (`str` or `os.PathLike`, *optional*):
            If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
        offload_state_dict (`bool`, *optional*):
            If `True`, will temporarily offload the CPU state dict to the hard drive to avoid getting out of CPU
            RAM if the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to
            `True` when there is some disk offload.
        load_in_8bit (`bool`, *optional*, defaults to `False`):
            If `True`, will convert the loaded model into mixed-8bit quantized model. To use this feature please
            install `bitsandbytes` (`pip install -U bitsandbytes`).
        load_in_4bit (`bool`, *optional*, defaults to `False`):
            If `True`, will convert the loaded model into 4bit precision quantized model. To use this feature
            install the latest version of `bitsandbytes` (`pip install -U bitsandbytes`).
        quantization_config (`Union[QuantizationConfigMixin,Dict]`, *optional*):
            A dictionary of configuration parameters or a QuantizationConfigMixin object for quantization (e.g
            bitsandbytes, gptq)
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.
        variant (`str`, *optional*):
            If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin. `variant` is
            ignored when using `from_tf` or `from_flax`.
        use_safetensors (`bool`, *optional*, defaults to `None`):
            Whether or not to use `safetensors` checkpoints. Defaults to `None`. If not specified and `safetensors`
            is not installed, it will be set to `False`.

        kwargs (remaining dictionary of keyword arguments, *optional*):
            Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
            `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
            automatically loaded:

                - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
                    underlying model's `__init__` method (we assume all relevant updates to the configuration have
                    already been done)
                - If a configuration is not provided, `kwargs` will be first passed to the configuration class
                    initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that
                    corresponds to a configuration attribute will be used to override said attribute with the
                    supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
                    will be passed to the underlying model's `__init__` function.

    <Tip>

    Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to
    use this method in a firewalled environment.

    </Tip>

    Examples:

    ```python
    >>> from transformers import BertConfig, BertModel

    >>> # Download model and configuration from huggingface.co and cache.
    >>> model = BertModel.from_pretrained("bert-base-uncased")
    >>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
    >>> model = BertModel.from_pretrained("./test/saved_model/")
    >>> # Update configuration during loading.
    >>> model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
    >>> assert model.config.output_attentions == True
    >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
    >>> config = BertConfig.from_json_file("./tf_model/my_tf_model_config.json")
    >>> model = BertModel.from_pretrained("./tf_model/my_tf_checkpoint.ckpt.index", from_tf=True, config=config)
    >>> # Loading from a Flax checkpoint file instead of a PyTorch model (slower)
    >>> model = BertModel.from_pretrained("bert-base-uncased", from_flax=True)
    ```

    * `low_cpu_mem_usage` algorithm:

    This is an experimental function that loads the model using ~1x model size CPU memory

    Here is how it works:

    1. save which state_dict keys we have
    2. drop state_dict before the model is created, since the latter takes 1x model size CPU memory
    3. after the model has been instantiated switch to the meta device all params/buffers that
    are going to be replaced from the loaded state_dict
    4. load state_dict 2nd time
    5. replace the params/buffers from the state_dict

    Currently, it can't handle deepspeed ZeRO stage 3 and ignores loading errors

    """
    state_dict = kwargs.pop("state_dict", None)
    from_tf = kwargs.pop("from_tf", False)
    from_flax = kwargs.pop("from_flax", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    output_loading_info = kwargs.pop("output_loading_info", False)
    use_auth_token = kwargs.pop("use_auth_token", None)
    trust_remote_code = kwargs.pop("trust_remote_code", None)
    _ = kwargs.pop("mirror", None)
    from_pipeline = kwargs.pop("_from_pipeline", None)
    from_auto_class = kwargs.pop("_from_auto", False)
    _fast_init = kwargs.pop("_fast_init", True)
    torch_dtype = kwargs.pop("torch_dtype", None)
    low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", None)
    device_map = kwargs.pop("device_map", None)
    max_memory = kwargs.pop("max_memory", None)
    offload_folder = kwargs.pop("offload_folder", None)
    offload_state_dict = kwargs.pop("offload_state_dict", False)
    load_in_8bit = kwargs.pop("load_in_8bit", False)
    load_in_4bit = kwargs.pop("load_in_4bit", False)
    quantization_config = kwargs.pop("quantization_config", None)
    subfolder = kwargs.pop("subfolder", "")
    commit_hash = kwargs.pop("_commit_hash", None)
    variant = kwargs.pop("variant", None)
    adapter_kwargs = kwargs.pop("adapter_kwargs", {})
    adapter_name = kwargs.pop("adapter_name", "default")
    use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)

    if is_fsdp_enabled():
        low_cpu_mem_usage = True

    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError(
                "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
            )
        token = use_auth_token

    if token is not None and adapter_kwargs is not None and "token" not in adapter_kwargs:
        adapter_kwargs["token"] = token

    if use_safetensors is None and not is_safetensors_available():
        use_safetensors = False

    if is_bitsandbytes_available():
        is_8bit_serializable = version.parse(importlib.metadata.version("bitsandbytes")) > version.parse("0.37.2")
    else:
        is_8bit_serializable = False

    if trust_remote_code is True:
        logger.warning(
            "The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is"
            " ignored."
        )

    if commit_hash is None:
        if not isinstance(config, PretrainedConfig):
            # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
            resolved_config_file = cached_file(
                pretrained_model_name_or_path,
                CONFIG_NAME,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
            commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
        else:
            commit_hash = getattr(config, "_commit_hash", None)

    if is_peft_available():
        _adapter_model_path = adapter_kwargs.pop("_adapter_model_path", None)

        if _adapter_model_path is None:
            _adapter_model_path = find_adapter_config_file(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                _commit_hash=commit_hash,
                **adapter_kwargs,
            )
        if _adapter_model_path is not None and os.path.isfile(_adapter_model_path):
            with open(_adapter_model_path, "r", encoding="utf-8") as f:
                _adapter_model_path = pretrained_model_name_or_path
                pretrained_model_name_or_path = json.load(f)["base_model_name_or_path"]
    else:
        _adapter_model_path = None

    # change device_map into a map if we passed an int, a str or a torch.device
    if isinstance(device_map, torch.device):
        device_map = {"": device_map}
    elif isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
        try:
            device_map = {"": torch.device(device_map)}
        except RuntimeError:
            raise ValueError(
                "When passing device_map as a string, the value needs to be a device name (e.g. cpu, cuda:0) or "
                f"'auto', 'balanced', 'balanced_low_0', 'sequential' but found {device_map}."
            )
    elif isinstance(device_map, int):
        if device_map < 0:
            raise ValueError(
                "You can't pass device_map as a negative int. If you want to put the model on the cpu, pass device_map = 'cpu' "
            )
        else:
            device_map = {"": device_map}

    if device_map is not None:
        if low_cpu_mem_usage is None:
            low_cpu_mem_usage = True
        elif not low_cpu_mem_usage:
            raise ValueError("Passing along a `device_map` requires `low_cpu_mem_usage=True`")

    if low_cpu_mem_usage:
        if device_map is not None:
            # The max memory utils require PyTorch >= 1.10 to have torch.cuda.mem_get_info.
            require_version_core("torch>=1.10")

        if is_deepspeed_zero3_enabled():
            raise ValueError(
                "DeepSpeed Zero-3 is not compatible with `low_cpu_mem_usage=True` or with passing a `device_map`."
            )
        elif not is_accelerate_available():
            raise ImportError(
                "Using `low_cpu_mem_usage=True` or a `device_map` requires Accelerate: `pip install accelerate`"
            )

    quantization_method_from_args = None

    if quantization_config is not None:
        quantization_method_from_args = getattr(
            quantization_config, "quant_method", QuantizationMethod.BITS_AND_BYTES
        )

    if quantization_config is None and (load_in_8bit or load_in_4bit):
        quantization_method_from_args = QuantizationMethod.BITS_AND_BYTES
        quantization_config, kwargs = BitsAndBytesConfig.from_dict(
            config_dict={"load_in_8bit": load_in_8bit, "load_in_4bit": load_in_4bit},
            return_unused_kwargs=True,
            **kwargs,
        )
    elif quantization_method_from_args == QuantizationMethod.BITS_AND_BYTES:
        load_in_8bit = quantization_config.load_in_8bit
        load_in_4bit = quantization_config.load_in_4bit

        quantization_config_kwargs = {
            k: v for k, v in kwargs.items() if k in inspect.signature(BitsAndBytesConfig).parameters
        }

        if len(quantization_config_kwargs) > 0:
            raise ValueError(
                "You can't pass `load_in_8bit` or any other `BitsAndBytesConfig` argument as a kwarg when passing "
                "`quantization_config` argument at the same time."
            )

    if load_in_8bit or load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. A GPU is needed for quantization.")
        if not (is_accelerate_available() and is_bitsandbytes_available()):
            raise ImportError(
                "Using `load_in_8bit=True` requires Accelerate: `pip install accelerate` and the latest version of"
                " bitsandbytes `pip install -i https://test.pypi.org/simple/ bitsandbytes` or"
                " `pip install bitsandbytes`."
            )

        if torch_dtype is None:
            # We force the `dtype` to be float16, this is a requirement from `bitsandbytes`
            logger.info(
                f"Overriding torch_dtype={torch_dtype} with `torch_dtype=torch.float16` due to "
                "requirements of `bitsandbytes` to enable model loading in 8-bit or 4-bit. "
                "Pass your own torch_dtype to specify the dtype of the remaining non-linear layers or pass"
                " torch_dtype=torch.float16 to remove this warning."
            )
            torch_dtype = torch.float16

        if device_map is None:
            device_map = {"": torch.cuda.current_device()}
            logger.info(
                "The device_map was not initialized. "
                "Setting device_map to {'':torch.cuda.current_device()}. "
                "If you want to use the model for inference, please set device_map ='auto' "
            )
            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True

        if from_tf or from_flax:
            raise ValueError(
                "Converting into 4-bit or 8-bit weights from tf/flax weights is currently not supported, please make"
                " sure the weights are in PyTorch format."
            )

    user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
    if from_pipeline is not None:
        user_agent["using_pipeline"] = from_pipeline

    if is_offline_mode() and not local_files_only:
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True

    # Load config if we don't provide a configuration
    if not isinstance(config, PretrainedConfig):
        config_path = config if config is not None else pretrained_model_name_or_path
        config, model_kwargs = cls.config_class.from_pretrained(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            _from_auto=from_auto_class,
            _from_pipeline=from_pipeline,
            **kwargs,
        )
    else:
        model_kwargs = kwargs

    quantizer = None
    quantization_method_from_config = None
    if hasattr(config, "quantization_config"):
        quantization_method_from_config = config.quantization_config.get(
            "quant_method", QuantizationMethod.BITS_AND_BYTES
        )

    if (
        quantization_method_from_args is not None
        and quantization_method_from_args == QuantizationMethod.AWQ
        and quantization_method_from_config is None
    ):
        raise ValueError(
            "You cannot quantize with AWQ a non-quantized model using transformers, please refer to the quantization documentation"
            " to read more about how to quantize models with AWQ algorithm https://huggingface.co/docs/transformers/main_classes/quantization"
        )

    if quantization_method_from_config is not None and quantization_method_from_args is not None:
        if quantization_method_from_config != quantization_method_from_args:
            raise ValueError(
                f"The model is already quantized with {quantization_method_from_config}. "
                f"You can't quantize it again with {quantization_method_from_args}"
            )

    if (
        quantization_method_from_config in (QuantizationMethod.GPTQ, QuantizationMethod.AWQ)
        and quantization_method_from_args is not None
    ):
        loading_attr_dict = quantization_config.get_loading_attributes()
        for attr, val in loading_attr_dict.items():
            config.quantization_config[attr] = val
        quantization_method_from_args = None
        logger.warning(
            f"You passed `quantization_config` to `from_pretrained` but the model you're loading already has a "
            f"`quantization_config` attribute and has already quantized weights. However, loading attributes"
            f" (e.g. {list(loading_attr_dict.keys())}) will be overwritten with the one you passed to `from_pretrained`. The rest will be ignored."
        )
    if (
        quantization_method_from_args == QuantizationMethod.GPTQ
        or quantization_method_from_config == QuantizationMethod.GPTQ
    ):
        gptq_supports_cpu = version.parse(importlib.metadata.version("auto-gptq")) > version.parse("0.4.2")
        if not gptq_supports_cpu and not torch.cuda.is_available():
            raise RuntimeError("GPU is required to quantize or run quantize model.")
        elif not (is_optimum_available() and is_auto_gptq_available()):
            raise ImportError(
                "Loading a GPTQ quantized model requires optimum (`pip install optimum`) and auto-gptq library (`pip install auto-gptq`)"
            )
        elif version.parse(importlib.metadata.version("auto_gptq")) < version.parse("0.4.2"):
            raise ImportError(
                "You need a version of auto_gptq >= 0.4.2 to use GPTQ: `pip install --upgrade auto-gptq`"
            )
        else:
            # Need to protect the import
            from optimum.gptq import GPTQQuantizer
        if quantization_method_from_config == QuantizationMethod.GPTQ:
            quantization_config = GPTQConfig.from_dict(config.quantization_config)
            config.quantization_config = quantization_config
        if torch_dtype is None:
            torch_dtype = torch.float16
        else:
            logger.info("We suggest you to set `torch_dtype=torch.float16` for better efficiency with GPTQ.")
        quantizer = GPTQQuantizer.from_dict(quantization_config.to_dict_optimum())
    elif quantization_method_from_config == QuantizationMethod.AWQ:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required to run AWQ quantized model.")

        if not is_auto_awq_available():
            raise ImportError("Loading an AWQ quantized model requires auto-awq library (`pip install autoawq`)")

        if not is_accelerate_available():
            raise ImportError("Loading an AWQ quantized model requires accelerate (`pip install accelerate`)")

        if device_map is None:
            logger.warning(
                "You have loaded an AWQ model on CPU and have a CUDA device available, make sure to set "
                "your model on a GPU device in order to run your model."
            )
        elif device_map is not None:
            if isinstance(device_map, dict) and ("cpu" in device_map.values() or "disk" in device_map.values()):
                raise ValueError(
                    "You are attempting to load an AWQ model with a device_map that contains a CPU or disk device."
                    " This is not supported. Please remove the CPU or disk device from the device_map."
                )

        if torch_dtype is None:
            torch_dtype = torch.float16
        else:
            logger.info("We suggest you to set `torch_dtype=torch.float16` for better efficiency with AWQ.")

        # Force-set to `True` for more mem efficiency
        if low_cpu_mem_usage is None:
            low_cpu_mem_usage = True

    if (
        is_8bit_serializable
        and quantization_method_from_args == QuantizationMethod.BITS_AND_BYTES
        and load_in_8bit
    ):
        if quantization_method_from_config == QuantizationMethod.BITS_AND_BYTES:
            logger.warning(
                "You passed `quantization_config` to `from_pretrained` but the model you're loading already has a"
                " `quantization_config` attribute. The `quantization_config` attribute will be overwritten with the"
                " one you passed to `from_pretrained`."
            )
        config.quantization_config = quantization_config
    elif (
        is_8bit_serializable
        and not load_in_8bit
        and quantization_method_from_config == QuantizationMethod.BITS_AND_BYTES
    ):
        quantization_config = config.quantization_config
        if isinstance(quantization_config, dict):
            quantization_config = BitsAndBytesConfig.from_dict(quantization_config, return_unused_kwargs=False)
        elif isinstance(quantization_config, BitsAndBytesConfig):
            pass
        else:
            raise ValueError(
                f"Invalid type for `quantization_config`: {type(quantization_config)}. Should be a `dict` or a"
                " `BitsAndBytesConfig` instance."
            )

        load_in_8bit = quantization_config.load_in_8bit

        if load_in_8bit:
            if torch_dtype is None:
                torch_dtype = torch.float16
            if device_map is None:
                if torch.cuda.is_available():
                    device_map = {"": torch.cuda.current_device()}
                else:
                    raise RuntimeError("No GPU found. A GPU is needed for quantization.")
                logger.info(
                    "The device_map was not initialized. "
                    "Setting device_map to {'':torch.cuda.current_device()}. "
                    "If you want to use the model for inference, please set device_map ='auto' "
                )
                if low_cpu_mem_usage is None:
                    low_cpu_mem_usage = True

    elif (
        not is_8bit_serializable
        and not load_in_8bit
        and quantization_method_from_config == QuantizationMethod.BITS_AND_BYTES
    ):
        logger.warning(
            "Detected the presence of a `quantization_config` attribute in the model's configuration but you don't have the correct"
            " `bitsandbytes` version to support int8 serialization. Please install the latest version of `bitsandbytes` with "
            " `pip install --upgrade bitsandbytes`."
        )

    # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
    # index of the files.
    is_sharded = False
    sharded_metadata = None
    # Load model
    loading_info = None

    # Keep in fp32 modules
    keep_in_fp32_modules = None
    use_keep_in_fp32_modules = False

    if pretrained_model_name_or_path is not None:
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if is_local:
            if from_tf and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index")
            ):
                # Load from a TF 1.0 checkpoint in priority if from_tf
                archive_file = os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index")
            elif from_tf and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)
            ):
                # Load from a TF 2.0 checkpoint in priority if from_tf
                archive_file = os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)
            elif from_flax and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
            ):
                # Load from a Flax checkpoint in priority if from_flax
                archive_file = os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
            elif use_safetensors is not False and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant))
            ):
                # Load from a safetensors checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant)
                )
            elif use_safetensors is not False and os.path.isfile(
                os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
                )
            ):
                # Load from a sharded safetensors checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
                )
                is_sharded = True
            elif os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant))
            ):
                # Load from a PyTorch checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant)
                )
            elif os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant))
            ):
                # Load from a sharded PyTorch checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant)
                )
                is_sharded = True
            # At this stage we don't have a weight file so we will raise an error.
            elif os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index")
            ) or os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)):
                raise EnvironmentError(
                    f"Error no file named {_add_variant(WEIGHTS_NAME, variant)} found in directory"
                    f" {pretrained_model_name_or_path} but there is a file for TensorFlow weights. Use"
                    " `from_tf=True` to load this model from those weights."
                )
            elif os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)):
                raise EnvironmentError(
                    f"Error no file named {_add_variant(WEIGHTS_NAME, variant)} found in directory"
                    f" {pretrained_model_name_or_path} but there is a file for Flax weights. Use `from_flax=True`"
                    " to load this model from those weights."
                )
            elif use_safetensors:
                raise EnvironmentError(
                    f"Error no file named {_add_variant(SAFE_WEIGHTS_NAME, variant)} found in directory"
                    f" {pretrained_model_name_or_path}."
                )
            else:
                raise EnvironmentError(
                    f"Error no file named {_add_variant(WEIGHTS_NAME, variant)}, {TF2_WEIGHTS_NAME},"
                    f" {TF_WEIGHTS_NAME + '.index'} or {FLAX_WEIGHTS_NAME} found in directory"
                    f" {pretrained_model_name_or_path}."
                )
        elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
            archive_file = pretrained_model_name_or_path
            is_local = True
        elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path + ".index")):
            if not from_tf:
                raise ValueError(
                    f"We found a TensorFlow checkpoint at {pretrained_model_name_or_path + '.index'}, please set "
                    "from_tf to True to load from this checkpoint."
                )
            archive_file = os.path.join(subfolder, pretrained_model_name_or_path + ".index")
            is_local = True
        elif is_remote_url(pretrained_model_name_or_path):
            filename = pretrained_model_name_or_path
            resolved_archive_file = download_url(pretrained_model_name_or_path)
        else:
            # set correct filename
            if from_tf:
                filename = TF2_WEIGHTS_NAME
            elif from_flax:
                filename = FLAX_WEIGHTS_NAME
            elif use_safetensors is not False:
                filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
            else:
                filename = _add_variant(WEIGHTS_NAME, variant)

            try:
                # Load from URL or cache if already cached
                cached_file_kwargs = {
                    "cache_dir": cache_dir,
                    "force_download": force_download,
                    "proxies": proxies,
                    "resume_download": resume_download,
                    "local_files_only": local_files_only,
                    "token": token,
                    "user_agent": user_agent,
                    "revision": revision,
                    "subfolder": subfolder,
                    "_raise_exceptions_for_missing_entries": False,
                    "_commit_hash": commit_hash,
                }
                resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)

                # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
                # result when internet is up, the repo and revision exist, but the file does not.
                if resolved_archive_file is None and filename == _add_variant(SAFE_WEIGHTS_NAME, variant):
                    # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                    resolved_archive_file = cached_file(
                        pretrained_model_name_or_path,
                        _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                        **cached_file_kwargs,
                    )
                    if resolved_archive_file is not None:
                        is_sharded = True
                    elif use_safetensors:
                        if revision == "main":
                            resolved_archive_file, revision, is_sharded = auto_conversion(
                                pretrained_model_name_or_path, **cached_file_kwargs
                            )
                        cached_file_kwargs["revision"] = revision
                        if resolved_archive_file is None:
                            raise EnvironmentError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named"
                                f" {_add_variant(SAFE_WEIGHTS_NAME, variant)} or {_add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)} "
                                "and thus cannot be loaded with `safetensors`. Please make sure that the model has "
                                "been saved with `safe_serialization=True` or do not set `use_safetensors=True`."
                            )
                    else:
                        # This repo has no safetensors file of any kind, we switch to PyTorch.
                        filename = _add_variant(WEIGHTS_NAME, variant)
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path, filename, **cached_file_kwargs
                        )
                if resolved_archive_file is None and filename == _add_variant(WEIGHTS_NAME, variant):
                    # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                    resolved_archive_file = cached_file(
                        pretrained_model_name_or_path,
                        _add_variant(WEIGHTS_INDEX_NAME, variant),
                        **cached_file_kwargs,
                    )
                    if resolved_archive_file is not None:
                        is_sharded = True
                if resolved_archive_file is None:
                    # Otherwise, maybe there is a TF or Flax model file.  We try those to give a helpful error
                    # message.
                    has_file_kwargs = {
                        "revision": revision,
                        "proxies": proxies,
                        "token": token,
                    }
                    if has_file(pretrained_model_name_or_path, TF2_WEIGHTS_NAME, **has_file_kwargs):
                        raise EnvironmentError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file for TensorFlow weights."
                            " Use `from_tf=True` to load this model from those weights."
                        )
                    elif has_file(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME, **has_file_kwargs):
                        raise EnvironmentError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file for Flax weights. Use"
                            " `from_flax=True` to load this model from those weights."
                        )
                    elif variant is not None and has_file(
                        pretrained_model_name_or_path, WEIGHTS_NAME, **has_file_kwargs
                    ):
                        raise EnvironmentError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file without the variant"
                            f" {variant}. Use `variant=None` to load this model from those weights."
                        )
                    else:
                        raise EnvironmentError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(WEIGHTS_NAME, variant)}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or"
                            f" {FLAX_WEIGHTS_NAME}."
                        )
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                # to the original exception.
                raise
            except Exception as e:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a file named {_add_variant(WEIGHTS_NAME, variant)},"
                    f" {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}."
                ) from e

        if is_local:
            logger.info(f"loading weights file {archive_file}")
            resolved_archive_file = archive_file
        else:
            logger.info(f"loading weights file {filename} from cache at {resolved_archive_file}")
    else:
        resolved_archive_file = None

    # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
    if is_sharded:
        # rsolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
        resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
            pretrained_model_name_or_path,
            resolved_archive_file,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            user_agent=user_agent,
            revision=revision,
            subfolder=subfolder,
            _commit_hash=commit_hash,
        )

    if (
        is_safetensors_available()
        and isinstance(resolved_archive_file, str)
        and resolved_archive_file.endswith(".safetensors")
    ):
        with safe_open(resolved_archive_file, framework="pt") as f:
            metadata = f.metadata()

        if metadata.get("format") == "pt":
            pass
        elif metadata.get("format") == "tf":
            from_tf = True
            logger.info("A TensorFlow safetensors file is being loaded in a PyTorch model.")
        elif metadata.get("format") == "flax":
            from_flax = True
            logger.info("A Flax safetensors file is being loaded in a PyTorch model.")
        else:
            raise ValueError(
                f"Incompatible safetensors file. File metadata is not ['pt', 'tf', 'flax'] but {metadata.get('format')}"
            )

    from_pt = not (from_tf | from_flax)

    # load pt weights early so that we know which dtype to init the model under
    if from_pt:
        if not is_sharded and state_dict is None:
            # Time to load the checkpoint
            state_dict = load_state_dict(resolved_archive_file)

        # set dtype to instantiate the model under:
        # 1. If torch_dtype is not None, we use that dtype
        # 2. If torch_dtype is "auto", we auto-detect dtype from the loaded state_dict, by checking its first
        #    weights entry that is of a floating type - we assume all floating dtype weights are of the same dtype
        # we also may have config.torch_dtype available, but we won't rely on it till v5
        dtype_orig = None

        if torch_dtype is not None:
            if isinstance(torch_dtype, str):
                if torch_dtype == "auto":
                    if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
                        torch_dtype = config.torch_dtype
                        logger.info(f"Will use torch_dtype={torch_dtype} as defined in model's config object")
                    else:
                        if is_sharded and "dtype" in sharded_metadata:
                            torch_dtype = sharded_metadata["dtype"]
                        elif not is_sharded:
                            torch_dtype = get_state_dict_dtype(state_dict)
                        else:
                            one_state_dict = load_state_dict(resolved_archive_file[0])
                            torch_dtype = get_state_dict_dtype(one_state_dict)
                            del one_state_dict  # free CPU memory
                        logger.info(
                            "Since the `torch_dtype` attribute can't be found in model's config object, "
                            "will use torch_dtype={torch_dtype} as derived from model's weights"
                        )
                else:
                    raise ValueError(
                        f'`torch_dtype` can be either `torch.dtype` or `"auto"`, but received {torch_dtype}'
                    )
            dtype_orig = cls._set_default_torch_dtype(torch_dtype)

        # Check if `_keep_in_fp32_modules` is not None
        use_keep_in_fp32_modules = (cls._keep_in_fp32_modules is not None) and (
            torch_dtype == torch.float16 or load_in_4bit or load_in_8bit
        )

        if is_sharded:
            loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
        else:
            loaded_state_dict_keys = list(state_dict.keys())
        if low_cpu_mem_usage or (use_keep_in_fp32_modules and is_accelerate_available()):
            # In case some weights need to be kept in float32 and accelerate is not installed,
            # we later on want to take the path where state_dict is not None, that is the one
            # that do not require accelerate.
            state_dict = None

    config.name_or_path = pretrained_model_name_or_path

    # Instantiate model.
    init_contexts = [no_init_weights(_enable=_fast_init)]

    if is_deepspeed_zero3_enabled():
        import deepspeed

        logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
        init_contexts = [deepspeed.zero.MiCS_Init(data_parallel_group=dist_env.get_data_parallel_group(),config_dict_or_path=deepspeed_config(), mpu=dist_env)] + init_contexts
        #   init_contexts = [deepspeed.zero.Init(config_dict_or_path=deepspeed_config())] + init_contexts
    elif load_in_8bit or load_in_4bit or low_cpu_mem_usage:
        init_contexts.append(init_empty_weights())

    config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
    config = cls._autoset_attn_implementation(
        config, use_flash_attention_2=use_flash_attention_2, torch_dtype=torch_dtype, device_map=device_map
    )

    with ContextManagers(init_contexts):
        # Let's make sure we don't run the init function of buffer modules
        model = cls(config, *model_args, **model_kwargs)

    # make sure we use the model's config since the __init__ call might have copied it
    config = model.config

    # Check first if we are `from_pt`
    if use_keep_in_fp32_modules:
        if is_accelerate_available():
            low_cpu_mem_usage = True
        keep_in_fp32_modules = model._keep_in_fp32_modules
    else:
        keep_in_fp32_modules = []

    if load_in_8bit or load_in_4bit:
        from .integrations import get_keys_to_not_convert, replace_with_bnb_linear

        llm_int8_skip_modules = quantization_config.llm_int8_skip_modules
        load_in_8bit_fp32_cpu_offload = quantization_config.llm_int8_enable_fp32_cpu_offload
        if load_in_8bit:
            logger.info("Detected 8-bit loading: activating 8-bit loading for this model")
        else:
            logger.info("Detected 4-bit loading: activating 4-bit loading for this model")

        # We keep some modules such as the lm_head in their original dtype for numerical stability reasons
        if llm_int8_skip_modules is None:
            modules_to_not_convert = get_keys_to_not_convert(model)
        else:
            modules_to_not_convert = llm_int8_skip_modules

        if not isinstance(modules_to_not_convert, list):
            modules_to_not_convert = [modules_to_not_convert]

        modules_to_not_convert.extend(keep_in_fp32_modules)

        # Extend the modules to not convert to keys that are supposed to be offloaded to `cpu` or `disk`
        if isinstance(device_map, dict) and len(device_map.keys()) > 1:
            keys_on_cpu = [key for key, value in device_map.items() if value in ["disk", "cpu"]]

            if len(keys_on_cpu) > 0 and not load_in_8bit_fp32_cpu_offload:
                raise ValueError(
                    "If you want to offload some keys to `cpu` or `disk`, you need to set "
                    "`llm_int8_enable_fp32_cpu_offload=True`. Note that these modules will not be "
                    " converted to 8-bit but kept in 32-bit."
                )

            modules_to_not_convert.extend(keys_on_cpu)

        supports_4bit = version.parse(importlib.metadata.version("bitsandbytes")) >= version.parse("0.39.0")

        if load_in_4bit and not supports_4bit:
            raise ValueError(
                "You have a version of `bitsandbytes` that is not compatible with 4bit inference and training"
                " make sure you have the latest version of `bitsandbytes` installed"
            )

        model = replace_with_bnb_linear(
            model, modules_to_not_convert=modules_to_not_convert, quantization_config=quantization_config
        )
        # training in 8-bit is only available in 0.37.0+
        model._is_quantized_training_enabled = version.parse(
            importlib.metadata.version("bitsandbytes")
        ) >= version.parse("0.37.0")

        config.quantization_config = quantization_config
        model.is_8bit_serializable = is_8bit_serializable

    if load_in_8bit and torch_dtype is None:
        logger.warning(
            "You are loading your model in 8bit but you did not specify a `torch_dtype` attribute. "
            "All non-linear modules will be loaded in full precision."
            " If you want to load the other modules in other precision, please specify a `torch_dtype` attribute."
        )
    if quantization_method_from_config == QuantizationMethod.GPTQ:
        model = quantizer.convert_model(model)
        model._is_quantized_training_enabled = True
    elif quantization_method_from_config == QuantizationMethod.AWQ:
        from .integrations import fuse_awq_modules, get_keys_to_not_convert, replace_with_awq_linear

        modules_to_not_convert = get_keys_to_not_convert(model)

        if quantization_config is None:
            quantization_config = AwqConfig.from_dict(config.quantization_config)

        model, has_been_replaced = replace_with_awq_linear(
            model, quantization_config=quantization_config, modules_to_not_convert=modules_to_not_convert
        )
        model._is_quantized_training_enabled = False

        if not has_been_replaced:
            logger.warning(
                "You are loading an AWQ model but no linear modules were found in your model."
                " Please double check your model architecture, or submit an issue on github if you think this is"
                " a bug."
            )

    if quantization_method_from_config is not None:
        model.quantization_method = quantization_method_from_config
    elif quantization_method_from_args is not None:
        model.quantization_method = quantization_method_from_args
    if hasattr(model, "quantization_method"):
        model.is_quantized = True

        # We store the original dtype for quantized models as we cannot easily retrieve it
        # once the weights have been quantized
        # Note that once you have loaded a quantized model, you can't change its dtype so this will
        # remain a single source of truth
        config._pre_quantization_dtype = torch_dtype

    if isinstance(device_map, str):
        special_dtypes = {}
        if load_in_8bit or load_in_4bit:
            special_dtypes.update(
                {
                    name: torch_dtype
                    for name, _ in model.named_parameters()
                    if any(m in name for m in modules_to_not_convert)
                }
            )

        special_dtypes.update(
            {
                name: torch.float32
                for name, _ in model.named_parameters()
                if any(m in name for m in keep_in_fp32_modules)
            }
        )

        target_dtype = torch_dtype

        if load_in_4bit:
            if version.parse(importlib.metadata.version("accelerate")) > version.parse("0.19.0"):
                from accelerate.utils import CustomDtype

                target_dtype = CustomDtype.INT4
            else:
                raise ValueError(
                    "You are using `device_map='auto'` on a 4bit loaded version of the model. To automatically compute"
                    " the appropriate device map, you should upgrade your `accelerate` library, "
                    "`pip install --upgrade accelerate` or install it from source to support fp4 auto device map "
                    "calculation. You may encounter unexpected behavior, or pass your own device map"
                )
        elif load_in_8bit:
            target_dtype = torch.int8

        no_split_modules = model._get_no_split_modules(device_map)
        if device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
            raise ValueError(
                "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
                "'sequential'."
            )

        device_map_kwargs = {"no_split_module_classes": no_split_modules}
        if "special_dtypes" in inspect.signature(infer_auto_device_map).parameters:
            device_map_kwargs["special_dtypes"] = special_dtypes
        elif len(special_dtypes) > 0:
            logger.warning(
                "This model has some weights that should be kept in higher precision, you need to upgrade "
                "`accelerate` to properly deal with them (`pip install --upgrade accelerate`)."
            )
        if device_map != "sequential":
            max_memory = get_balanced_memory(
                model,
                dtype=target_dtype,
                low_zero=(device_map == "balanced_low_0"),
                max_memory=max_memory,
                **device_map_kwargs,
            )
        else:
            max_memory = get_max_memory(max_memory)
        if getattr(model, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
            # need more space for buffers that are created during quantization
            max_memory = {key: val * 0.90 for key, val in max_memory.items()}
        device_map_kwargs["max_memory"] = max_memory

        # Make sure tied weights are tied before creating the device map.
        model.tie_weights()
        device_map = infer_auto_device_map(model, dtype=target_dtype, **device_map_kwargs)

        if load_in_8bit or load_in_4bit:
            # The LM head / tied weights or any last module can stay on disk / CPU
            device_map_without_lm_head = {
                key: device_map[key] for key in device_map.keys() if key not in modules_to_not_convert
            }
            if "cpu" in device_map_without_lm_head.values() or "disk" in device_map_without_lm_head.values():
                raise ValueError(
                    """
                    Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit
                    the quantized model. If you want to dispatch the model on the CPU or the disk while keeping
                    these modules in 32-bit, you need to set `load_in_8bit_fp32_cpu_offload=True` and pass a custom
                    `device_map` to `from_pretrained`. Check
                    https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu
                    for more details.
                    """
                )
            del device_map_without_lm_head

    elif device_map is not None:
        model.tie_weights()
        tied_params = find_tied_parameters(model)
        # check if we don't have tied param in different devices
        check_tied_parameters_on_same_device(tied_params, device_map)

    if from_tf:
        if resolved_archive_file.endswith(".index"):
            # Load from a TensorFlow 1.X checkpoint - provided by original authors
            model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
        else:
            # Load from our TensorFlow 2.0 checkpoints
            try:
                from .modeling_tf_pytorch_utils import load_tf2_checkpoint_in_pytorch_model

                model, loading_info = load_tf2_checkpoint_in_pytorch_model(
                    model, resolved_archive_file, allow_missing_keys=True, output_loading_info=True
                )
            except ImportError:
                logger.error(
                    "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed."
                    " Please see https://pytorch.org/ and https://www.tensorflow.org/install/ for installation"
                    " instructions."
                )
                raise
    elif from_flax:
        try:
            from .modeling_flax_pytorch_utils import load_flax_checkpoint_in_pytorch_model

            model = load_flax_checkpoint_in_pytorch_model(model, resolved_archive_file)
        except ImportError:
            logger.error(
                "Loading a Flax model in PyTorch, requires both PyTorch and Flax to be installed. Please see"
                " https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for"
                " installation instructions."
            )
            raise
    elif from_pt:
        # restore default dtype
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)
        (
            model,
            missing_keys,
            unexpected_keys,
            mismatched_keys,
            offload_index,
            error_msgs,
        ) = cls._load_pretrained_model(
            model,
            state_dict,
            loaded_state_dict_keys,  # XXX: rename?
            resolved_archive_file,
            pretrained_model_name_or_path,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            sharded_metadata=sharded_metadata,
            _fast_init=_fast_init,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=device_map,
            offload_folder=offload_folder,
            offload_state_dict=offload_state_dict,
            dtype=torch_dtype,
            is_quantized=(getattr(model, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES),
            keep_in_fp32_modules=keep_in_fp32_modules,
        )

    model.is_loaded_in_4bit = load_in_4bit
    model.is_loaded_in_8bit = load_in_8bit

    # make sure token embedding weights are still tied if needed
    model.tie_weights()

    # Set model in evaluation mode to deactivate DropOut modules by default
    model.eval()

    # If it is a model with generation capabilities, attempt to load the generation config
    if model.can_generate() and pretrained_model_name_or_path is not None:
        try:
            model.generation_config = GenerationConfig.from_pretrained(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )
        except OSError:
            logger.info(
                "Generation config file not found, using a generation config created from the model config."
            )
            pass

    if (
        quantization_config is not None
        and quantization_config.quant_method == QuantizationMethod.AWQ
        and quantization_config.do_fuse
    ):
        model = fuse_awq_modules(model, config.quantization_config)
        model._awq_is_fused = True

    # Dispatch model with hooks on all devices if necessary
    if device_map is not None:
        device_map_kwargs = {
            "device_map": device_map,
            "offload_dir": offload_folder,
            "offload_index": offload_index,
        }
        if "skip_keys" in inspect.signature(dispatch_model).parameters:
            device_map_kwargs["skip_keys"] = model._skip_keys_device_placement
        dispatch_model(model, **device_map_kwargs)

    if quantization_method_from_args == QuantizationMethod.GPTQ:
        if quantization_config.tokenizer is None:
            quantization_config.tokenizer = pretrained_model_name_or_path
        if cls.main_input_name != "input_ids":
            raise RuntimeError("We can only quantize pure text model.")
        quantizer.quantize_model(model, quantization_config.tokenizer)
        config.quantization_config = GPTQConfig.from_dict_optimum(quantizer.to_dict())
        model._is_quantized_training_enabled = True
    if quantization_method_from_config == QuantizationMethod.GPTQ:
        model = quantizer.post_init_model(model)

    if _adapter_model_path is not None:
        model.load_adapter(
            _adapter_model_path,
            adapter_name=adapter_name,
            token=token,
            adapter_kwargs=adapter_kwargs,
        )

    if output_loading_info:
        if loading_info is None:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
                "error_msgs": error_msgs,
            }
        return model, loading_info

    return model