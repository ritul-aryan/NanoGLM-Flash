import torch
import torch.nn as nn
from typing import Optional

class VisionPrefixProjector(nn.Module):
    """
    2D Image Patch-Projection Prefix Module.
    - Extracts 16 non-overlapping patches from 64x64 RGB images
    - Projects patches to LLM embedding dimension (d_model=512)
    - Adds learned 2D positional embeddings
    """
    def __init__(self, img_size: int = 64, patch_size: int = 16, in_channels: int = 3, d_model: int = 512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2 # 16 patches
        self.d_model = d_model
        
        # Patch extraction via strided convolution
        self.patch_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )
        
        # 2-layer MLP projection with LayerNorm
        self.ln = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=True)
        )
        
        # Learned visual positional embedding
        self.pos_emb = nn.Parameter(torch.empty(1, self.num_patches, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, 3, H, W)
        Returns: visual_prefix (B, 16, d_model)
        """
        B, C, H, W = images.shape
        # (B, C, H, W) -> (B, d_model, H/P, W/P)
        feat = self.patch_conv(images)
        # Flatten spatial dims: (B, d_model, num_patches) -> (B, num_patches, d_model)
        feat = feat.flatten(2).transpose(1, 2)
        
        # Add visual position embeddings
        feat = feat + self.pos_emb
        
        # Non-linear projection
        out = self.mlp(self.ln(feat))
        return out
