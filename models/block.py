import torch
import torch.nn as nn
from typing import Optional, Tuple, Any
from .embeddings import RMSNorm
from .linear_attention import GatedLinearAttention
from .sliding_attention import SlidingWindowRoPEAttention
from .moe import SparseMoE

class HybridBlock(nn.Module):
    """
    Hybrid Transformer Block:
    - Pre-norm RMSNorm
    - Hybrid Attention: Gated Linear Attention OR Sliding-Window RoPE Attention
    - Pre-norm RMSNorm
    - Sparse Mixture-of-Experts (1 Shared + Top-2/4 Routed SwiGLU Experts)
    """
    def __init__(
        self,
        d_model: int,
        layer_idx: int,
        n_heads: int = 8,
        head_dim: int = 64,
        sliding_window: int = 256,
        full_attn_interval: int = 3,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        top_k: int = 2,
        moe_intermediate_dim: int = 768,
        aux_loss_coef: float = 0.01,
        max_seq_len: int = 1024,
        rms_norm_eps: float = 1e-6
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = d_model
        self.is_full_attn = (layer_idx % full_attn_interval == (full_attn_interval - 1))
        
        # 1. Attention Pre-Norm & Attention Layer
        self.attn_norm = RMSNorm(d_model, eps=rms_norm_eps)
        if self.is_full_attn:
            self.attn = SlidingWindowRoPEAttention(
                d_model=d_model,
                n_heads=n_heads,
                head_dim=head_dim,
                window_size=sliding_window,
                max_seq_len=max_seq_len
            )
        else:
            self.attn = GatedLinearAttention(
                d_model=d_model,
                n_heads=n_heads,
                head_dim=head_dim
            )
            
        # 2. MoE Pre-Norm & Sparse MoE Block
        self.moe_norm = RMSNorm(d_model, eps=rms_norm_eps)
        self.moe = SparseMoE(
            d_model=d_model,
            n_routed_experts=n_routed_experts,
            n_shared_experts=n_shared_experts,
            top_k=top_k,
            intermediate_dim=moe_intermediate_dim,
            aux_loss_coef=aux_loss_coef
        )

    def forward(
        self,
        x: torch.Tensor,
        state_or_cache: Optional[Any] = None,
        offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Any]]:
        """
        x: (B, S, d_model)
        Returns: (output, aux_loss, next_state_or_cache)
        """
        # Attention sub-layer with residual
        norm_x = self.attn_norm(x)
        if self.is_full_attn:
            attn_out, next_cache = self.attn(norm_x, kv_cache=state_or_cache, offset=offset)
        else:
            attn_out, next_cache = self.attn(norm_x, state=state_or_cache)
            
        h = x + attn_out
        
        # MoE sub-layer with residual
        norm_h = self.moe_norm(h)
        moe_out, aux_loss = self.moe(norm_h)
        out = h + moe_out
        
        return out, aux_loss, next_cache
