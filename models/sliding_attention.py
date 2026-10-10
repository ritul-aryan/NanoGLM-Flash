import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .embeddings import RotaryEmbedding

class SlidingWindowRoPEAttention(nn.Module):
    """
    Sliding-Window Causal Multi-Head Self-Attention with Rotary Position Embeddings (RoPE).
    - Window size W = 256
    - Accelerated via torch.nn.functional.scaled_dot_product_attention (SDPA)
    - Full KV cache support for inference
    """
    def __init__(self, d_model: int, n_heads: int = 8, head_dim: int = 64, window_size: int = 256, max_seq_len: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.window_size = window_size
        
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
        
        self.rope = RotaryEmbedding(dim=head_dim, max_seq_len=max_seq_len)

    def _create_sliding_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Creates boolean mask where True means MASKED OUT (disallowed).
        Allowed: 0 <= (row - col) < window_size
        """
        rows = torch.arange(seq_len, device=device).unsqueeze(1)
        cols = torch.arange(seq_len, device=device).unsqueeze(0)
        diff = rows - cols
        # Allowed if diff >= 0 and diff < window_size
        allowed = (diff >= 0) & (diff < self.window_size)
        # In PyTorch SDPA, attn_mask can be boolean where False is masked out OR float with -inf
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.float32)
        mask = mask.masked_fill(~allowed, float("-inf"))
        return mask

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        offset: int = 0
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        x: (B, S, d_model)
        kv_cache: Optional tuple of (K, V), each (B, H, cache_len, D)
        """
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2) # (B, H, S, D)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q, k = self.rope(q, k, offset=offset)
        
        # Manage KV cache for generation
        if kv_cache is not None:
            k_past, v_past = kv_cache
            k = torch.cat([k_past, k], dim=2) # (B, H, total_len, D)
            v = torch.cat([v_past, v], dim=2)
            # Prune cache to maximum window_size + current step to cap memory
            if k.shape[2] > self.window_size + 64:
                k = k[:, :, -self.window_size:]
                v = v[:, :, -self.window_size:]
            new_kv_cache = (k, v)
        else:
            new_kv_cache = (k, v) if not self.training else None
            
        total_len = k.shape[2]
        
        if S == 1:
            # Single-token autoregressive decoding step: query attends to last min(total_len, window_size) tokens
            if total_len > self.window_size:
                k_active = k[:, :, -self.window_size:, :]
                v_active = v[:, :, -self.window_size:, :]
            else:
                k_active = k
                v_active = v
            # No mask needed for single token attending to past tokens
            y = nn.functional.scaled_dot_product_attention(q, k_active, v_active, is_causal=False)
        else:
            # Multi-token parallel sequence (training or prompt encoding)
            if S <= self.window_size and offset == 0:
                # Standard causal FlashAttention is equivalent when S <= window_size!
                y = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                # Construct sliding window mask
                mask = self._create_sliding_mask(S, device=x.device).unsqueeze(0).unsqueeze(0) # (1, 1, S, S)
                mask = mask.to(dtype=q.dtype)
                y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        y = y.transpose(1, 2).contiguous().view(B, S, self.d_model)
        out = self.out_proj(y)
        return out, new_kv_cache
