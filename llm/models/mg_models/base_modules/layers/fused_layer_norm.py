import numbers
import importlib
from packaging import version
from deepspeed.pipe import LayerSpec

import torch
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F

from llm.utils.env import dist_env

try:
    from flash_attn.ops.rms_norm import rms_norm as flash_attn_rms_norm
except ImportError:
    flash_attn_rms_norm = None


# no bias version
class RMSNorm(torch.nn.Module):
    def __init__(self, sync_tp_duplicated_parameters, normalized_shape,
                 eps=1e-6, use_flash_attn=False, bf16=True, hf_mode=False,
                 sequence_parallel=False):
        super(RMSNorm, self).__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.use_flash_attn = use_flash_attn
        self.bf16 = bf16
        self.hf_mode = hf_mode

        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()

        self.layernorm_tp_auto_sync = sync_tp_duplicated_parameters
        self.sequence_parallel = sequence_parallel

    def reset_parameters(self):
        init.ones_(self.weight)

    def _norm(self, x):
        if self.bf16:
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        else:
            return x * torch.rsqrt(x.to(torch.float32).pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, input):

        if self.layernorm_tp_auto_sync or self.sequence_parallel:
            torch.distributed.all_reduce(self.weight / dist_env.get_tensor_model_parallel_world_size(),
                                         op=torch.distributed.ReduceOp.SUM,
                                         group=dist_env.get_tensor_model_parallel_group())

        if self.use_flash_attn and flash_attn_rms_norm and input.shape[-1] <= 8192:
            return flash_attn_rms_norm(input, self.weight, self.eps)

        if self.hf_mode:
            variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)
            hidden_states = input * torch.rsqrt(variance + self.eps)

            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)

            return hidden_states * self.weight

        if self.bf16:
            return self._norm(input) * self.weight
        else:
            return (self._norm(input) * self.weight).to(input.dtype)


class FusedLayerNormAffineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, normalized_shape, eps):
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        bias_ = bias.contiguous()
        output, mean, invvar = fused_mix_prec_layer_norm_cuda.forward_affine(
            input_, ctx.normalized_shape, weight_, bias_, ctx.eps)
        ctx.save_for_backward(input_, weight_, bias_, mean, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight_, bias_, mean, invvar = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        grad_input, grad_weight, grad_bias = fused_mix_prec_layer_norm_cuda.backward_affine(
            grad_output.contiguous(), mean, invvar,
            input_, ctx.normalized_shape,
            weight_, bias_, ctx.eps)
        return grad_input, grad_weight, grad_bias, None, None


class MixedFusedLayerNorm(torch.nn.Module):
    def __init__(self, sync_tp_duplicated_parameters, normalized_shape,
                 eps=1e-6, use_flash_attn=False, bf16=True, hf_mode=False,
                 sequence_parallel=False):
        super(MixedFusedLayerNorm, self).__init__()

        global fused_mix_prec_layer_norm_cuda
        fused_mix_prec_layer_norm_cuda = importlib.import_module("fused_mix_prec_layer_norm_cuda")

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.use_flash_attn = use_flash_attn

        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.bias = Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()

        self.layernorm_tp_auto_sync = sync_tp_duplicated_parameters
        self.sequence_parallel = sequence_parallel

        # Current Meg-DS cuda kernel has better throughput than torch.nn.LayerNorm
        # https://github.com/pytorch/pytorch/pull/66920
        self.use_meg_ds_fused_layer_norm = (bf16 or version.parse(torch.__version__) >= version.parse("1.11.0"))

    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input):

        if self.layernorm_tp_auto_sync or self.sequence_parallel:
            torch.distributed.all_reduce(self.weight, op=torch.distributed.ReduceOp.AVG,
                                         group=dist_env.get_tensor_model_parallel_group())
            torch.distributed.all_reduce(self.bias, op=torch.distributed.ReduceOp.AVG,
                                         group=dist_env.get_tensor_model_parallel_group())

        if self.use_meg_ds_fused_layer_norm:
            return FusedLayerNormAffineFunction.apply(
                input, self.weight, self.bias, self.normalized_shape, self.eps)
        else:
            return F.layer_norm(input, self.normalized_shape, self.weight, self.bias)


def build_layer_norm(cfg_ln, layer_spec=False):
    if cfg_ln['type'] == 'rms_norm':
        if layer_spec:
            return LayerSpec(RMSNorm, **cfg_ln['kwargs'])
        else:
            return RMSNorm(**cfg_ln['kwargs'])
    elif cfg_ln['type'] == 'mixed_fused_norm':
        if layer_spec:
            return LayerSpec(MixedFusedLayerNorm, **cfg_ln['kwargs'])
        else:
            return MixedFusedLayerNorm(**cfg_ln['kwargs'])
    else:
        raise NotImplementedError
