import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

class GatedLinearAttention(nn.Module):
    """
    Gated Linear Attention (GLA) / Recurrent State-Space Layer.
    - O(T) FLOPs during parallel training
    - O(1) Memory & Time per step during autoregressive rollout generation
    """
    def __init__(self, d_model: int, n_heads: int = 8, head_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        
        # Projections
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.g_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
        
        # Learnable head-wise decay parameter gamma in (0, 1)
        # Parameterized as log(-log(decay)) or sigmoid
        self.decay_param = nn.Parameter(torch.linspace(-2.0, -0.5, n_heads)) # sigmoid decay ~ 0.88 to 0.62

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: (B, S, d_model)
        state: Optional recurrent state (B, n_heads, head_dim, head_dim)
        Returns: (output, next_state)
        """
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2) # (B, H, S, D)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        g = self.g_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Positive feature activations
        q = nn.functional.elu(q) + 1.0
        k = nn.functional.elu(k) + 1.0
        gate = torch.sigmoid(g)
        
        decay = torch.sigmoid(self.decay_param).view(1, self.n_heads, 1, 1) # (1, H, 1, 1)
        
        if state is not None or (not self.training and S == 1):
            # Recurrent O(1) step mode for inference rollouts
            if state is None:
                state = torch.zeros(B, self.n_heads, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)
                
            q_t = q[:, :, 0, :].unsqueeze(-2) # (B, H, 1, D)
            k_t = k[:, :, 0, :].unsqueeze(-1) # (B, H, D, 1)
            v_t = v[:, :, 0, :].unsqueeze(-2) # (B, H, 1, D)
            gate_t = gate[:, :, 0, :]         # (B, H, D)
            
            # S_t = decay * S_{t-1} + (k_t^T @ v_t)
            kv = torch.matmul(k_t, v_t) # (B, H, D, D)
            new_state = (state * decay.squeeze(-1)) + kv
            
            # y_t = q_t @ S_t
            y_t = torch.matmul(q_t, new_state).squeeze(-2) # (B, H, D)
            y_t = y_t * gate_t * self.scale
            
            out = y_t.transpose(1, 2).contiguous().view(B, 1, self.d_model) # (B, 1, d_model)
            out = self.out_proj(out)
            return out, new_state

        else:
            # Parallel training mode: compute causal associative scan
            # We construct a causal decay matrix: M_{i,j} = decay^{i-j} for i >= j, else 0
            idx = torch.arange(S, device=x.device)
            dist = idx.unsqueeze(1) - idx.unsqueeze(0) # (S, S): row i - col j >= 0 (strictly lower triangular causal)
            causal_mask = dist >= 0
            
            # Decay weights per head: (1, H, S, S)
            # decay: (1, H, 1, 1)
            # pow(decay, dist) where causal_mask is True
            decay_powers = torch.where(
                causal_mask,
                decay ** dist.clamp(min=0).float(),
                torch.zeros(1, device=x.device, dtype=torch.float32)
            ).to(dtype=q.dtype) # (1, H, S, S)
            
            # Attention weights: A = (Q @ K^T) * M
            # Q: (B, H, S, D), K: (B, H, S, D) -> Q @ K^T: (B, H, S, S)
            qk = torch.matmul(q, k.transpose(-2, -1)) # (B, H, S, S)
            qk_decayed = qk * decay_powers * self.scale
            
            # Row normalization across causal keys for numerical stability across sequence lengths
            row_sum = qk_decayed.sum(dim=-1, keepdim=True).clamp(min=1e-5)
            attn_weights = qk_decayed / row_sum
            
            # Out: A @ V
            y = torch.matmul(attn_weights, v) # (B, H, S, D)
            y = y * gate
            
            y = y.transpose(1, 2).contiguous().view(B, S, self.d_model)
            out = self.out_proj(y)
            return out, None
