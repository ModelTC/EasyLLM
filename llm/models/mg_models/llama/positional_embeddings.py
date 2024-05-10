# Extracted from: https://github.com/EleutherAI/gpt-neox
import torch
import math
from typing import Optional
import torch.nn.functional as F
import torch.utils.checkpoint


class RotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, base=10000, precision=torch.half, scale_factor=1.0):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        # self.register_buffer('inv_freq', inv_freq)
        self.inv_freq = inv_freq
        self.max_seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.precision = precision
        self.scale_factor = scale_factor

    def forward(self, x, seq_dim=1, seq_len=None):
        self.inv_freq = self.inv_freq.cuda()
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = seq_len
            # follow transformers LlamaLinearScalingRotaryEmbedding implement
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=torch.float32)
            t = t / self.scale_factor
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()
            # [sx, 1 (b * np), hn]
            self.cos_cached = emb.cos()[:, None, :]
            self.sin_cached = emb.sin()[:, None, :]
            if self.precision == torch.bfloat16:
                self.cos_cached = self.cos_cached.bfloat16()
                self.sin_cached = self.sin_cached.bfloat16()
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


class InternLM2DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """InternLM2RotaryEmbedding extended with Dynamic NTK scaling.
    Credits to the Reddit users /u/bloc97 and /u/emozilla.
    """
    def __init__(self,
                 dim,
                 base=10000,
                 precision=torch.half,
                 scale_factor=1.0,
                 max_position_embeddings=None):
        super().__init__(dim, base, precision, scale_factor)
        self.base = base
        self.dim = dim
        assert max_position_embeddings is not None
        self.max_position_embeddings = max_position_embeddings
        self._set_cos_sin_cache(max_position_embeddings,
                                torch.cuda.current_device())

    def _set_cos_sin_cache(self, seq_len, device):
        self.max_seq_len_cached = seq_len
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scale_factor * seq_len / self.max_position_embeddings) - (self.scale_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float32)
        t = t / self.scale_factor
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1).to(device)

        if self.precision == torch.bfloat16:
            emb = emb.float()

        self.cos_cached = emb.cos()[:, None, :]
        self.sin_cached = emb.sin()[:, None, :]
        if self.precision == torch.bfloat16:
            self.cos_cached = self.cos_cached.bfloat16()
            self.sin_cached = self.sin_cached.bfloat16()

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]

        device = x.device
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self._set_cos_sin_cache(seq_len, device)

        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


def _get_unpad_data(padding_mask):
    seqlens_in_batch = padding_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# Inverse dim formula to find dim based on number of rotations
def _yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


# Find dim range bounds based on rotations
def _yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(_yarn_find_correction_dim(
        low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(
        high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def _yarn_get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


class LlamaDynamicYaRNScaledRotaryEmbedding(torch.nn.Module):
    def __init__(self,
                 dim,
                 base=10000,
                 precision=torch.half,
                 max_position_embeddings=8192,
                 original_max_position_embeddings=2048,
                 extrapolation_factor=1,
                 attn_factor=1,
                 beta_fast=32,
                 beta_slow=1,
                 finetuned=True):
        super().__init__()

        self.original_max_position_embeddings = original_max_position_embeddings
        self.dim = dim
        self.base = base
        self.precision = precision
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.max_seq_len_cached = None
        # self.cos_cached = None
        # self.sin_cached = None
        self.precision = precision

        if finetuned:
            self.yarn(self.max_position_embeddings / self.original_max_position_embeddings)
        else:
            inv_freq = 1.0 / \
                (base ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.mscale = 1

    def forward(self, x, seq_dim=1, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None:
            self.max_seq_len_cached = self.max_position_embeddings
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", (emb.cos()[:, None, :] * self.mscale).to(self.precision), persistent=False)
            self.register_buffer("sin_cached", (emb.sin()[:, None, :] * self.mscale).to(self.precision), persistent=False)

        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len

            self.yarn(seq_len / self.max_position_embeddings, x.device)

            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", (emb.cos()[:, None, :] * self.mscale).to(self.precision), persistent=False)
            self.register_buffer("sin_cached", (emb.sin()[:, None, :] * self.mscale).to(self.precision), persistent=False)
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

    def yarn(self, scale, device=None):
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scale * pos_freqs)

        low, high = _yarn_find_correction_range(self.beta_fast, self.beta_slow, self.dim, self.base, self.original_max_position_embeddings)
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(low, high, self.dim // 2).float().to(device)) * self.extrapolation_factor
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mscale = float(_yarn_get_mscale(scale) * self.attn_factor)


# rotary pos emb helpers:

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos, sin = cos[offset:q.shape[0] + offset, ...], sin[offset:q.shape[0] + offset, ...]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def apply_rotary_pos_emb_torch(q, k, cos, sin, offset: int = 0, position_ids=None):  # jitting fails with bf16
    if position_ids is None:
        cos, sin = cos[offset:q.shape[0] + offset, ...], sin[offset:q.shape[0] + offset, ...]
    else:
        bs_len, sen_len = position_ids.shape
        position_ids = position_ids.transpose(1, 0)
        cos = cos[position_ids.reshape(-1)]
        sin = sin[position_ids.reshape(-1)]
        if bs_len > 1:
            cos = cos.reshape(sen_len, bs_len, cos.shape[-2], cos.shape[-1])
            sin = sin.reshape(sen_len, bs_len, sin.shape[-2], sin.shape[-1])
            q = q.reshape(sen_len, bs_len, -1, q.shape[-1])
            k = k.reshape(sen_len, bs_len, -1, k.shape[-1])
    pos_emb_q, pos_emb_k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    if (position_ids is not None) and (bs_len > 1):
        pos_emb_q = pos_emb_q.reshape(q.shape)
        pos_emb_k = pos_emb_k.reshape(k.shape)
    return pos_emb_q, pos_emb_k
