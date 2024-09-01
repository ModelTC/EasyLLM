from megatron.core.transformer.spec_utils import import_module
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training import print_rank_0
from megatron.training import get_args
from megatron.core.models.gpt import GPTModel
import megatron.legacy.model
from typing import Union
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

try:
    from megatron.core.transformer.custom_layers.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TEColumnParallelLinear,
        TEDotProductAttention,
        TENorm,
        TERowParallelGroupedLinear,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_layer_norm import WrappedTorchLayerNorm

    warnings.warn(f'Apex is not installed. Falling back to Torch LayerNorm')
    LNImpl = WrappedTorchLayerNorm

from megatron.legacy.model.rms_norm import RMSNorm
from llm.plugins.megatron_lm.models.llama_model import LlaMAModel
from llm.plugins.megatron_lm.models.internvl.internvl_model import InternVLModel


# Helper function to get module spec for MLP/MoE
def _get_mlp_module_spec(
    use_te: bool = True, num_experts: int = None, moe_grouped_gemm: bool = False
) -> ModuleSpec:
    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                # linear_fc1=TELayerNormColumnParallelLinear if use_te else ColumnParallelLinear,
                linear_fc1=TEColumnParallelLinear if use_te else ColumnParallelLinear,
                linear_fc2=TERowParallelLinear if use_te else RowParallelLinear,
            ),
        )
    else:
        # Mixture of experts with modules in megatron core.
        if use_te and moe_grouped_gemm:
            linear_fc1 = TEColumnParallelGroupedLinear
            linear_fc2 = TERowParallelGroupedLinear
        else:
            linear_fc1 = ColumnParallelLinear
            linear_fc2 = RowParallelLinear

        use_te_grouped_gemm = use_te and TEColumnParallelGroupedLinear is not None

        return ModuleSpec(
            module=MoELayer,
            submodules=(
                MLPSubmodules(linear_fc1=linear_fc1, linear_fc2=linear_fc2)
                if not moe_grouped_gemm or use_te_grouped_gemm
                else None
            ),
        )

# Use this spec to use lower level Transformer Engine modules (required for fp8 training)


def get_llama_layer_with_transformer_engine_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=True, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=TENorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    # TENorm significantly harms convergence when used
                    # for QKLayerNorm; we instead use the Apex implementation.
                    q_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                    k_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=TENorm,  # if num_experts else IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


# Use this spec for an implementation using only modules in megatron core
def get_llama_layer_local_spec(
    num_experts: int = None, moe_grouped_gemm: bool = False, qk_layernorm: bool = False
) -> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=RMSNorm,  # LNImpl,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=RMSNorm if qk_layernorm else IdentityOp,
                    k_layernorm=RMSNorm if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=RMSNorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
            },
        ),
    )


def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else:  # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_llama_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
            else:
                transformer_layer_spec = get_llama_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)

        # model = GPTModel(
        model = LlaMAModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base
        )

    return model


def model_provider_vlm(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    # args = get_args()
    # use_te = args.transformer_impl == "transformer_engine"

    # print_rank_0('building GPT model ...')
    # # Experimental loading arguments from yaml
    # if args.yaml_cfg is not None:
    #     config = core_transformer_config_from_yaml(args, "language_model")
    # else:
    #     config = core_transformer_config_from_args(args)

    # if args.use_legacy_models:
    #     model = megatron.legacy.model.GPTModel(
    #         config,
    #         num_tokentypes=0,
    #         parallel_output=True,
    #         pre_process=pre_process,
    #         post_process=post_process,
    #     )
    # else:  # using core models
    #     if args.spec is not None:
    #         transformer_layer_spec = import_module(args.spec)
    #     else:
    #         if use_te:
    #             transformer_layer_spec = get_llama_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
    #         else:
    #             transformer_layer_spec = get_llama_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)

    #     # model = GPTModel(
    #     model = InternVLModel(
    #         config=config,
    #         transformer_layer_spec=transformer_layer_spec,
    #         vocab_size=args.padded_vocab_size,
    #         max_sequence_length=args.max_position_embeddings,
    #         pre_process=pre_process,
    #         post_process=post_process,
    #         fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
    #         parallel_output=True,
    #         share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
    #         position_embedding_type=args.position_embedding_type,
    #         rotary_percent=args.rotary_percent,
    #         rotary_base=args.rotary_base
    #     )

    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"
    
    language_transformer_config = core_transformer_config_from_args(args)
    if args.spec is not None:
        language_transformer_layer_spec = import_module(args.spec)
    else:
        if use_te:
            language_transformer_layer_spec = get_llama_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
        else:
            language_transformer_layer_spec = get_llama_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)

    cfg_model = args.cfg_model
    model = InternVLModel(
        num_vit_layers=cfg_model['num_vit_layers'],

        vision_embedding_config=cfg_model['vision_embedings_params'],
        vision_transformer_config=cfg_model['vision_transformer_layer_params'],
        vision_extract_feat_config=cfg_model['vision_extract_feat_params'],
        drop_path_rate=cfg_model['drop_path_rate'],

        word_embedding_config=cfg_model['word_embedings_params'],
        fp32_residual_connection=cfg_model['fp32_residual_connection'],

        language_transformer_config=language_transformer_config,
        language_transformer_layer_spec=language_transformer_layer_spec,

        vocab_size=args.padded_vocab_size,
        parallel_output=True
    )


    return model