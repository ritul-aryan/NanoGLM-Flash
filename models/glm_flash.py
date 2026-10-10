import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List
from configs.model_config import ModelConfig
from .embeddings import RMSNorm, TiedEmbedding
from .vision_prefix import VisionPrefixProjector
from .block import HybridBlock

class NanoGLMFlash(nn.Module):
    """
    NanoGLM-Flash Multi-Modal Causal Language Model:
    - Byte-level tokenization with tied embeddings (0.20M params)
    - 2D Patch-Projection Vision Prefix Projector (0.66M params)
    - 10 Hybrid Layers: 7 Gated Linear Attention + 3 Sliding-Window RoPE Attention
    - Sparse Mixture-of-Experts: 1 Shared Expert + 4 Routed SwiGLU Experts (top-2)
    - End-to-end BF16 / FP16 mixed precision acceleration
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 1. Tied Byte Embedding
        self.embeddings = TiedEmbedding(config.vocab_size, config.d_model)
        
        # 2. Vision Prefix Projector
        self.vision_projector = VisionPrefixProjector(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_channels=config.img_channels,
            d_model=config.d_model
        )
        
        # 3. Stack of 10 Hybrid Blocks
        self.blocks = nn.ModuleList([
            HybridBlock(
                d_model=config.d_model,
                layer_idx=i,
                n_heads=config.n_heads,
                head_dim=config.head_dim,
                sliding_window=config.sliding_window,
                full_attn_interval=config.full_attn_interval,
                n_routed_experts=config.n_routed_experts,
                n_shared_experts=config.n_shared_experts,
                top_k=config.top_k,
                moe_intermediate_dim=config.moe_intermediate_dim,
                aux_loss_coef=config.aux_loss_coef,
                max_seq_len=config.max_seq_len,
                rms_norm_eps=config.rms_norm_eps
            )
            for i in range(config.n_layers)
        ])
        
        # 4. Final Normalization
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def count_parameters(self) -> Dict[str, int]:
        total_params = sum(p.numel() for p in self.parameters())
        # Active params count (1 shared + top-2 routed experts active per block)
        emb_p = self.embeddings.weight.numel()
        vis_p = sum(p.numel() for p in self.vision_projector.parameters())
        norm_p = sum(p.numel() for p in self.final_norm.parameters())
        
        block_active_p = 0
        for b in self.blocks:
            attn_p = sum(p.numel() for p in b.attn.parameters())
            norms = sum(p.numel() for p in b.attn_norm.parameters()) + sum(p.numel() for p in b.moe_norm.parameters())
            router_p = sum(p.numel() for p in b.moe.router.parameters())
            shared_p = sum(p.numel() for p in b.moe.shared_expert.parameters())
            # top-2 routed experts active per token
            routed_p = sum(p.numel() for p in b.moe.routed_experts[0].parameters()) * b.moe.top_k
            block_active_p += (attn_p + norms + router_p + shared_p + routed_p)
            
        active_params = emb_p + vis_p + norm_p + block_active_p
        return {
            "total_params": total_params,
            "active_params": active_params,
            "sparsity_pct": (1.0 - (active_params / total_params)) * 100.0
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        input_ids: (B, S)
        images: Optional (B, 3, H, W)
        labels: Optional (B, S) or (B, num_img_patches + S)
        Returns: dict with 'logits', 'loss', 'aux_loss', 'ce_loss'
        """
        B, S = input_ids.shape
        x = self.embeddings(input_ids) # (B, S, d_model)
        
        num_prefix = 0
        if images is not None:
            img_prefix = self.vision_projector(images) # (B, 16, d_model)
            x = torch.cat([img_prefix, x], dim=1) # (B, 16 + S, d_model)
            num_prefix = img_prefix.shape[1]
            
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        # Pass through hybrid blocks
        for block in self.blocks:
            x, aux_loss, _ = block(x)
            total_aux_loss = total_aux_loss + aux_loss
            
        h = self.final_norm(x)
        logits = self.embeddings.project_to_logits(h) # (B, seq_len, vocab_size)
        
        out = {
            "logits": logits,
            "aux_loss": total_aux_loss
        }
        
        if labels is not None:
            # Shift for causal next-token cross entropy
            # If labels corresponds to input_ids only, align with text logits portion
            if labels.shape[1] == S:
                target_logits = logits[:, num_prefix:, :]
            else:
                target_logits = logits
                
            shift_logits = target_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            ce_loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            total_loss = ce_loss + total_aux_loss
            out["loss"] = total_loss
            out["ce_loss"] = ce_loss
            
        return out

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        eos_token_id: int = 258
    ) -> torch.Tensor:
        """
        Autoregressive text generation with temperature, top-k, top-p, and repetition penalty.
        """
        self.eval()
        B = prompt_tokens.shape[0]
        curr_ids = prompt_tokens.clone()
        
        # Initial forward pass to get prompt logits
        out = self.forward(curr_ids, images=images)
        next_logit = out["logits"][:, -1, :] # (B, vocab_size)
        
        for _ in range(max_new_tokens):
            # Apply repetition penalty to already generated tokens
            if repetition_penalty != 1.0:
                for b in range(B):
                    unique_tokens = torch.unique(curr_ids[b])
                    for tok in unique_tokens:
                        if next_logit[b, tok] > 0:
                            next_logit[b, tok] /= repetition_penalty
                        else:
                            next_logit[b, tok] *= repetition_penalty

            if temperature > 0:
                scaled_logits = next_logit / max(temperature, 1e-4)
                
                # Top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[:, [-1]]] = -float('Inf')
                    
                # Top-p filtering (nucleus)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    cumulative_probs = torch.cumsum(nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    scaled_logits[indices_to_remove] = -float('Inf')
                    
                probs = nn.functional.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
            else:
                next_token = torch.argmax(next_logit, dim=-1, keepdim=True) # (B, 1)
                
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
            # Check for EOS across all batch items
            if (next_token == eos_token_id).all():
                break
                
            # Forward next token with full sequence (or cached states)
            out = self.forward(curr_ids, images=images)
            next_logit = out["logits"][:, -1, :]
            
        return curr_ids
