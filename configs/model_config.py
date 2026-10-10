from dataclasses import dataclass

@dataclass
class ModelConfig:
    """
    Configuration for NanoGLM-Flash-72M multi-modal architecture.
    """
    # Vocabulary & Embeddings
    vocab_size: int = 384                 # 256 byte tokens + 128 control/special tokens
    d_model: int = 512                    # Model hidden dimension (multiple of 64 & 128)
    max_seq_len: int = 1024               # Maximum sequence length supported by RoPE
    
    # Layer Depth & Hybrid Attention Schedule
    n_layers: int = 10                    # Total transformer blocks
    n_heads: int = 8                      # Number of attention heads
    head_dim: int = 64                    # Head dimension (512 // 8 = 64, Tensor Core aligned)
    d_state: int = 64                     # Recurrent feature state dimension for Gated Linear Attention
    sliding_window: int = 256             # Sliding window size for local RoPE self-attention
    full_attn_interval: int = 3           # Full/Window Attention every 3rd layer (indices 2, 5, 8)
    
    # Sparse Mixture-of-Experts (MoE)
    n_routed_experts: int = 4             # Number of routed experts
    n_shared_experts: int = 1             # Number of shared experts (always active)
    top_k: int = 2                        # Number of routed experts selected per token
    moe_intermediate_dim: int = 768       # Intermediate dimension for SwiGLU MLP (1.5x d_model)
    aux_loss_coef: float = 0.01           # Load balancing auxiliary loss coefficient
    
    # Multi-Modal Vision Prefix
    img_size: int = 64                    # Input image spatial resolution (64x64)
    img_channels: int = 3                 # RGB image channels
    patch_size: int = 16                  # Patch size (16x16 -> 4x4 = 16 tokens)
    
    # Normalization & Numerics
    rms_norm_eps: float = 1e-6            # RMSNorm numerical stability epsilon
    rope_theta: float = 10000.0           # RoPE base frequency
    tie_embeddings: bool = True           # Tied weights between input embedding and LM head
    
    @property
    def num_patches(self) -> int:
        return (self.img_size // self.patch_size) ** 2
    
    @property
    def patch_dim(self) -> int:
        return self.patch_size * self.patch_size * self.img_channels
