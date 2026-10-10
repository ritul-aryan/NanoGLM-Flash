import torch
import torch.nn as nn
from typing import Tuple, Optional

class SwiGLUExpert(nn.Module):
    """
    SwiGLU Feed-Forward Network:
    SwiGLU(x) = (SiLU(W_gate @ x) * (W_up @ x)) @ W_down
    """
    def __init__(self, d_model: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

class SparseMoE(nn.Module):
    """
    Sparse Mixture of Experts with 1 Shared Expert and N Routed Experts.
    - Top-k routing (k=2)
    - Always-active Shared Expert capturing general syntax & logic
    - Auxiliary load-balancing loss calculation
    """
    def __init__(
        self,
        d_model: int,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        top_k: int = 2,
        intermediate_dim: int = 768,
        aux_loss_coef: float = 0.01
    ):
        super().__init__()
        self.d_model = d_model
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.top_k = min(top_k, n_routed_experts)
        self.aux_loss_coef = aux_loss_coef
        
        # Router
        self.router = nn.Linear(d_model, n_routed_experts, bias=False)
        
        # Shared expert (always active)
        self.shared_expert = SwiGLUExpert(d_model, intermediate_dim)
        
        # Routed experts
        self.routed_experts = nn.ModuleList([
            SwiGLUExpert(d_model, intermediate_dim)
            for _ in range(n_routed_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, S, d_model)
        Returns: (output, aux_loss)
        """
        B, S, D = x.shape
        flat_x = x.view(-1, D) # (N, D), where N = B * S
        N = flat_x.shape[0]
        
        # 1. Always evaluate shared expert
        shared_out = self.shared_expert(x) # (B, S, D)
        
        # 2. Router logits and gating probabilities
        router_logits = self.router(flat_x) # (N, n_routed)
        gating_probs = nn.functional.softmax(router_logits, dim=-1) # (N, n_routed)
        
        # 3. Top-k expert selection
        top_weights, top_indices = torch.topk(gating_probs, self.top_k, dim=-1) # (N, k)
        # Renormalize top-k weights to sum to 1
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 4. Compute load-balancing auxiliary loss during training
        if self.training:
            # Fraction of tokens routed to expert i
            # top_indices: (N, k)
            one_hot_mask = nn.functional.one_hot(top_indices, num_classes=self.n_routed_experts).float() # (N, k, n_routed)
            token_count_per_expert = one_hot_mask.sum(dim=1).mean(dim=0) # (n_routed,) fraction f_i
            avg_prob_per_expert = gating_probs.mean(dim=0) # (n_routed,) P_i
            # aux_loss = coef * n_routed * sum(f_i * P_i)
            aux_loss = self.aux_loss_coef * self.n_routed_experts * (token_count_per_expert * avg_prob_per_expert).sum()
        else:
            aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            
        # 5. Evaluate routed experts efficiently
        routed_out = torch.zeros_like(flat_x)
        
        for i, expert in enumerate(self.routed_experts):
            # Find tokens assigned to expert i
            mask = (top_indices == i) # (N, k)
            if not mask.any():
                continue
                
            token_idx, k_pos = torch.where(mask)
            expert_tokens = flat_x[token_idx] # (num_tokens, D)
            weights = top_weights[token_idx, k_pos].unsqueeze(-1) # (num_tokens, 1)
            
            # Apply expert and weight
            expert_res = expert(expert_tokens) * weights
            routed_out.index_add_(0, token_idx, expert_res)
            
        routed_out = routed_out.view(B, S, D)
        total_out = shared_out + routed_out
        return total_out, aux_loss
