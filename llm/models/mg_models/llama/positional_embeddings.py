# Extracted from: https://github.com/EleutherAI/gpt-neox
import torch


class RotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, base=10000, precision=torch.half, scale_factor=1.0):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.precision = precision
        self.scale_factor = scale_factor

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = seq_len
            # follow transformers LlamaLinearScalingRotaryEmbedding implement
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
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
