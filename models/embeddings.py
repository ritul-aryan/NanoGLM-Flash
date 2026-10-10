import torch
import torch.nn as nn
from typing import Tuple

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps)
        return x_normed * self.weight

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).
    Precomputes cos and sin frequencies up to max_seq_len.
    """
    def __init__(self, dim: int, max_seq_len: int = 1024, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # dim must be even
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq) # (seq_len, dim // 2)
        emb = torch.cat((freqs, freqs), dim=-1) # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q, k shape: (B, H, S, D)
        """
        seq_len = q.shape[2]
        total_len = offset + seq_len
        if total_len > self.cos_cached.shape[0]:
            self._build_cache(max(total_len, self.cos_cached.shape[0] * 2))
            
        cos = self.cos_cached[offset:total_len].to(dtype=q.dtype, device=q.device) # (S, D)
        sin = self.sin_cached[offset:total_len].to(dtype=q.dtype, device=q.device) # (S, D)
        
        # Reshape for broadcasting: (1, 1, S, D)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        # [-x2, x1]
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

class TiedEmbedding(nn.Module):
    """
    Token Embedding with weight tying to output LM head.
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return nn.functional.embedding(input_ids, self.weight)

    def project_to_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # (B, S, d_model) @ (d_model, vocab_size) -> (B, S, vocab_size)
        return torch.matmul(hidden_states, self.weight.t())
