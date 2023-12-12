import math

from deepspeed.pipe import PipelineModule, LayerSpec
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology

from llm.utils.env import dist_env
from ..base_modules.modules.meg_module import MegatronModule
from .word_embedings import EmbeddingPipe
from .lm_head import EmbedddingPipeNoTied
from .transformer import ParallelTransformerLayerPipe
from ..base_modules.layers.fused_layer_norm import build_layer_norm

from ..base_modules.modules.fp16_module import float16_to_fp32, fp32_to_float16
# from .losses import get_cross_entropy
from .default_cfg import update_model_cfg
from .utils import load_lora_ckpt_pretrained, load_ckpt_pretrained, save_lora_ckpt_pretrained
from .utils import set_train_params, set_train_status
from ..base_modules.utils import check_keys_mapping
from llm.utils.general.registry_factory import LOSS_REGISTRY


class LlamaModelPipe(PipelineModule, MegatronModule):
    """
        LLaMA model.
        NOTE: The name of this class has to be kept as GPTModelPipe.
        I don't know why, but it is used in the code.
    """

    def __init__(
        self,
        num_layers,
        parallel_output=True,
        fp16: bool = False,
        bf16: bool = True,
        fp32_residual_connection: bool = False,
        pretrain_causal_attention: bool = False,
        checkpoint_activations: bool = True,
        checkpoint_num_layers: int = 1,
        pp_partition_method: str = 'type:transformer|embedding',
        sequence_parallel=False,
        dynamic_checkpoint=None,
        word_embedings_params=None,
        transformer_layer_params=None,
        layer_norm_params=None,
        lm_head_params=None,
        loss_params=None,
    ):

        self.parallel_output = parallel_output
        self.sequence_parallel = sequence_parallel
        # set model args
        self._set_model_kwargs(num_layers, checkpoint_activations)
        self.word_embedings_params = word_embedings_params
        self.transformer_layer_params = transformer_layer_params
        self.layer_norm_params = layer_norm_params
        self.lm_head_params = lm_head_params
        self.loss_params = loss_params

        self.specs = self.build_specs(num_layers, fp16, bf16, fp32_residual_connection, pretrain_causal_attention)
        self.loss_fn = LOSS_REGISTRY.build(self.loss_params)

        if checkpoint_activations:
            interval = checkpoint_num_layers
        else:
            interval = 0
        self.size_map = None
        self.skip_checkpoint_layer_range = -1
        if dynamic_checkpoint is not None:
            if dynamic_checkpoint['enabled']:
                self.size_map = dynamic_checkpoint['size_map']
        self.num_layers_per_pp = math.ceil(self.model_kwargs['num_layers'] / dist_env.get_pipeline_model_parallel_world_size())      # noqa
        self.start_checkpoint_layer_idx = self.num_layers_per_pp * dist_env.get_pipeline_model_parallel_rank()
        self.skip_checkpoint_layer_range += self.start_checkpoint_layer_idx

        topo = PipeModelDataParallelTopology(num_pp=dist_env.get_pipeline_model_parallel_world_size(),
                                             num_mp=dist_env.get_tensor_model_parallel_world_size(),
                                             num_dp=dist_env.get_data_parallel_world_size())

        # here one can extend the regex to include more layers to be counted towards partitioning,
        # e.g. 'type:transformer|embedding' will add up all the transformer blocks and also the first
        # and last embedding layers and then partition that transformers+2 layers - so to get a good
        # balance you may want to use less transformer layers
        #
        # caveat emptor: the current implementation of PP fails unless each stage has at least one
        # transformer layer
        if pp_partition_method is not None:
            partition_method = pp_partition_method
        else:
            partition_method = 'type:transformer'

        super().__init__(layers=self.specs,
                         loss_fn=self.loss_fn,
                         topology=topo,
                         activation_checkpoint_interval=interval,
                         partition_method=partition_method)

    def _is_checkpointable(self, funcs):
        # This is an unfortunate hack related to torch and deepspeed activation checkpoint implementations.
        # Some layers like torch.nn.Embedding will not receive grads if checkpointed, which breaks things.
        # I presume it's related to the discrete inputs that cannot require_grad? Need to revisit.
        if self.skip_checkpoint_layer_range < self.start_checkpoint_layer_idx:
            return all('ParallelTransformerLayerPipe' in f.__class__.__name__ for f in funcs)
        else:
            flag = True
            for f in funcs:
                flag = flag & ('ParallelTransformerLayerPipe' in f.__class__.__name__)
                if hasattr(f, "layer_number") and (self.skip_checkpoint_layer_range >= self.start_checkpoint_layer_idx):
                    flag = flag & (f.layer_number < self.skip_checkpoint_layer_range)
                    # flag = False
                    # print(f.layer_number, f"checkpoint {flag}")
            return flag

    def get_checkpoint_range(self, seq_len):
        size_list = sorted(list(self.size_map.keys()))
        for item in size_list:
            if seq_len <= item:
                return self.size_map[item]
        return -1

    def get_seq_len(self, forward_input):
        # first stage
        if dist_env.get_pipeline_model_parallel_rank() == 0:
            seq_len = forward_input[0].shape[1]
        else:
            if isinstance(forward_input, tuple):
                forward_input = forward_input[0]
            if self.sequence_parallel:
                seq_len = forward_input.shape[0] * dist_env.get_tensor_model_parallel_world_size()
            else:
                seq_len = forward_input.shape[0]
        return seq_len

    def forward(self, forward_input):
        seq_len = self.get_seq_len(forward_input)
        if self.size_map is not None:
            self.skip_checkpoint_layer_range = self.start_checkpoint_layer_idx + self.get_checkpoint_range(seq_len)
        return super().forward(forward_input)

    def _set_model_kwargs(self, num_layers, checkpoint_activations):
        self.model_kwargs = {"num_layers": num_layers, "checkpoint_activations": checkpoint_activations}

    def build_specs(self, num_layers, fp16, bf16, fp32_residual_connection, pretrain_causal_attention):
        specs = []

        def _to_float16(inputs):
            if fp16:
                return fp32_to_float16(inputs, lambda v: v.half())
            elif bf16:
                return fp32_to_float16(inputs, lambda v: v.bfloat16())
            else:
                return inputs

        specs.append(_to_float16)

        specs.append(LayerSpec(EmbeddingPipe, **self.word_embedings_params))

        def transpose(x):
            if isinstance(x, tuple) or isinstance(x, list):
                if fp32_residual_connection:
                    return (x[0].transpose(0, 1).contiguous().float(), *x[1:])
                else:
                    return (x[0].transpose(0, 1).contiguous(), *x[1:])
            else:
                if fp32_residual_connection:
                    return x.transpose(0, 1).contiguous().float()
                else:
                    return x.transpose(0, 1).contiguous()
        specs.append(transpose)

        for layer_idx in range(num_layers):
            self.transformer_layer_params.update({'layer_number': layer_idx})
            specs.append(LayerSpec(ParallelTransformerLayerPipe, **self.transformer_layer_params))

        # Undo data format change
        def undo(x):
            if isinstance(x, tuple) or isinstance(x, list):
                x = x[0]
            return x.transpose(0, 1).contiguous()
        specs.append(undo)

        # Final layernorm after transformer layers
        specs.append(build_layer_norm(self.layer_norm_params, layer_spec=True))

        specs.append(LayerSpec(EmbedddingPipeNoTied, **self.lm_head_params))

        # Convert to fp32 if needed
        if fp16 or bf16:
            specs.append(float16_to_fp32)

        return specs


_LLAMA_MODELS = {
    "llama_7b": {
        "num_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32
    },
    "llama_13b": {
        "num_layers": 40,
        "hidden_size": 5120,
        "num_attention_heads": 40
    },
    "llama_20b": {
        "num_layers": 60,
        "hidden_size": 5120,
        "num_attention_heads": 40
    },
    "llama_65b": {
        "num_layers": 80,
        "hidden_size": 8192,
        "num_attention_heads": 64
    },
    "llama2_7b": {
        "num_layers": 32,
        "hidden_size": 4096,
        # No MQA
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_kv_attention_heads": 32
    },
    "llama2_13b": {
        "num_layers": 40,
        "hidden_size": 5120,
        "intermediate_size": 13824,
        "num_attention_heads": 40,
        "num_kv_attention_heads": 40
    },
    "llama2_70b": {
        "num_layers": 80,
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_attention_heads": 64,
        "num_kv_attention_heads": 8  # multi-group query attention
    },
    "codellama_7b": {
        "num_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_kv_attention_heads": 32,
        "position_embedding_kwargs": {
            "base": 1.e6,   # rope_theta
        },
    },
    "codellama_13b": {
        "num_layers": 40,
        "hidden_size": 5120,
        "intermediate_size": 13824,
        "num_attention_heads": 40,
        "num_kv_attention_heads": 40,
        "position_embedding_kwargs": {
            "base": 1.e6,   # rope_theta
        },
    },
    "codellama_34b": {
        "num_layers": 48,
        "hidden_size": 8192,
        "ffn_hidden_size": 22016,
        "num_attention_heads": 64,
        "num_kv_attention_heads": 8,  # multi-group query attention
        "position_embedding_kwargs": {
            "base": 1.e6,   # rope_theta
        },
    },
}


def llama_custom(**cfg_model):
    cfg_model = update_model_cfg(cfg_model)
    model = LlamaModelPipe(**cfg_model)
    # set save and load
    model.load_lora_ckpt_pretrained = load_lora_ckpt_pretrained
    model.load_ckpt_pretrained = load_ckpt_pretrained
    model.save_lora_ckpt_pretrained = save_lora_ckpt_pretrained
    # set trainable
    model.set_train_status = set_train_status
    model.set_train_params = set_train_params
    return model


def llama_7b(**cfg_model):
    cfg_7b = _LLAMA_MODELS['llama_7b']
    check_keys_mapping(cfg_7b, cfg_model)
    cfg_model.update(cfg_7b)
    return llama_custom(**cfg_model)


def llama_13b(**cfg_model):
    cfg_13b = _LLAMA_MODELS['llama_13b']
    check_keys_mapping(cfg_13b, cfg_model)
    cfg_model.update(cfg_13b)
    return llama_custom(**cfg_model)


def llama_20b(**cfg_model):
    cfg_20b = _LLAMA_MODELS['llama_20b']
    check_keys_mapping(cfg_20b, cfg_model)
    cfg_model.update(cfg_20b)
    return llama_custom(**cfg_model)


def llama_65b(**cfg_model):
    cfg_65b = _LLAMA_MODELS['llama_65b']
    check_keys_mapping(cfg_65b, cfg_model)
    cfg_model.update(cfg_65b)
    return llama_custom(**cfg_model)


def llama2_7b(**cfg_model):
    cfg_7b = _LLAMA_MODELS['llama2_7b']
    check_keys_mapping(cfg_7b, cfg_model)
    cfg_model.update(cfg_7b)
    return llama_custom(**cfg_model)


def llama2_13b(**cfg_model):
    cfg_13b = _LLAMA_MODELS['llama2_13b']
    check_keys_mapping(cfg_13b, cfg_model)
    cfg_model.update(cfg_13b)
    return llama_custom(**cfg_model)


def llama2_70b(**cfg_model):
    cfg_70b = _LLAMA_MODELS['llama2_70b']
    check_keys_mapping(cfg_70b, cfg_model)
    cfg_model.update(cfg_70b)
    return llama_custom(**cfg_model)


def codellama_7b(**cfg_model):
    _cfg = _LLAMA_MODELS['codellama_7b']
    check_keys_mapping(_cfg, cfg_model)
    cfg_model.update(_cfg)
    return llama_custom(**cfg_model)


def codellama_13b(**cfg_model):
    _cfg = _LLAMA_MODELS['codellama_13b']
    check_keys_mapping(_cfg, cfg_model)
    cfg_model.update(_cfg)
    return llama_custom(**cfg_model)


def codellama_34b(**cfg_model):
    _cfg = _LLAMA_MODELS['codellama_34b']
    check_keys_mapping(_cfg, cfg_model)
    cfg_model.update(_cfg)
    return llama_custom(**cfg_model)
