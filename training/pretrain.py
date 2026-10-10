import os
import sys
import time
import math
import torch
import torch.nn as nn
from typing import Optional, Dict
from configs.model_config import ModelConfig
from configs.pretrain_config import PretrainConfig
from models.glm_flash import NanoGLMFlash
from data.tokenizer import ByteTokenizer
from data.synthetic_text import SyntheticTextGenerator
from data.synthetic_vision import SyntheticVisionGenerator

class Pretrainer:
    """
    Pre-training and Supervised Fine-Tuning Engine for NanoGLM-Flash.
    - Mixed precision BF16 with Blackwell Tensor Core acceleration
    - Multi-modal interleaved batches (synthetic reasoning + vision)
    - Cosine learning rate schedule with linear warmup
    """
    def __init__(self, model_config: ModelConfig, pretrain_config: PretrainConfig):
        self.m_cfg = model_config
        self.p_cfg = pretrain_config
        self.device = torch.device(pretrain_config.device if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = NanoGLMFlash(model_config).to(self.device)
        self.tokenizer = ByteTokenizer(vocab_size=model_config.vocab_size)
        
        # Data generators
        self.text_gen = SyntheticTextGenerator(seed=42)
        self.vision_gen = SyntheticVisionGenerator(img_size=model_config.img_size, seed=42)
        
        # Optimizer with weight decay separation
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name or "pos_emb" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
                
        optim_groups = [
            {"params": decay_params, "weight_decay": pretrain_config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        self.optimizer = torch.optim.AdamW(
            optim_groups,
            lr=pretrain_config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        os.makedirs(pretrain_config.checkpoint_dir, exist_ok=True)

    def get_lr(self, step: int) -> float:
        """Cosine annealing learning rate with linear warmup."""
        if step < self.p_cfg.warmup_steps:
            return self.p_cfg.learning_rate * (step + 1) / self.p_cfg.warmup_steps
        if step > self.p_cfg.max_steps:
            return self.p_cfg.min_learning_rate
        decay_ratio = (step - self.p_cfg.warmup_steps) / (self.p_cfg.max_steps - self.p_cfg.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.p_cfg.min_learning_rate + coeff * (self.p_cfg.learning_rate - self.p_cfg.min_learning_rate)

    def prepare_batch(self, batch_size: int, seq_len: int, step: int):
        """Prepares pure text or multi-modal batches alternatingly."""
        # Alternate between pure text reasoning and multi-modal vision
        is_vision_step = (step % 3 == 0) # 1 in 3 steps is multi-modal
        
        if is_vision_step:
            images, texts = self.vision_gen.generate_batch(batch_size)
            images = images.to(self.device)
        else:
            texts = self.text_gen.generate_batch(batch_size)
            images = None

        # Tokenize and pad
        token_lists = [self.tokenizer.encode(t, add_bos=True, add_eos=True) for t in texts]
        
        # Max length for this batch
        batch_max_len = min(seq_len, max(len(toks) for toks in token_lists) + 4)
        
        input_ids = torch.full((batch_size, batch_max_len), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
        labels = torch.full((batch_size, batch_max_len), -100, dtype=torch.long, device=self.device)
        
        for i, toks in enumerate(token_lists):
            clipped = toks[:batch_max_len]
            input_ids[i, :len(clipped)] = torch.tensor(clipped, dtype=torch.long, device=self.device)
            labels[i, :len(clipped)] = torch.tensor(clipped, dtype=torch.long, device=self.device)
            labels[i, labels[i] == self.tokenizer.pad_token_id] = -100

        return input_ids, images, labels

    def train_step(self, step: int) -> Dict[str, float]:
        self.model.train()
        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
            
        input_ids, images, labels = self.prepare_batch(
            self.p_cfg.batch_size,
            self.p_cfg.seq_len,
            step=step
        )
        
        self.optimizer.zero_grad()
        
        # BF16 Autocast forward pass
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            outputs = self.model(input_ids, images=images, labels=labels)
            loss = outputs["loss"]
            ce_loss = outputs["ce_loss"]
            aux_loss = outputs["aux_loss"]
            
        loss.backward()
        
        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.p_cfg.grad_clip)
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "ce_loss": ce_loss.item(),
            "aux_loss": aux_loss.item(),
            "lr": lr,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        }

    def train(self, num_steps: Optional[int] = None) -> str:
        total_steps = num_steps or self.p_cfg.max_steps
        print("=" * 65)
        print(f"STARTING PRE-TRAINING ({total_steps} STEPS, BATCH_SIZE={self.p_cfg.batch_size})")
        print("=" * 65)
        
        t0 = time.time()
        running_loss = 0.0
        
        for step in range(1, total_steps + 1):
            step_start = time.perf_counter()
            metrics = self.train_step(step)
            step_time = (time.perf_counter() - step_start) * 1000.0
            
            running_loss += metrics["loss"]
            
            if step % 10 == 0 or step == 1:
                tokens_per_sec = (self.p_cfg.batch_size * self.p_cfg.seq_len) / (step_time / 1000.0)
                vram_used_gb = torch.cuda.memory_allocated(self.device) / (1024**3)
                print(
                    f"Step {step:4d}/{total_steps} | "
                    f"Loss: {metrics['loss']:.4f} (CE: {metrics['ce_loss']:.4f}, Aux: {metrics['aux_loss']:.4f}) | "
                    f"LR: {metrics['lr']:.2e} | "
                    f"Latency: {step_time:6.1f}ms | "
                    f"Throughput: {tokens_per_sec:7.0f} tok/s | "
                    f"VRAM: {vram_used_gb:4.2f} GB"
                )
                
            if step % self.p_cfg.save_interval == 0 or step == total_steps:
                save_path = os.path.join(self.p_cfg.checkpoint_dir, f"pretrain_step_{step}.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "config": self.m_cfg
                }, save_path)
                print(f"--> Saved checkpoint to {save_path}")
                
        elapsed = time.time() - t0
        print(f"Pre-training completed in {elapsed:.1f}s. Final average loss: {running_loss / total_steps:.4f}")
        final_ckpt = os.path.join(self.p_cfg.checkpoint_dir, "pretrain_final.pt")
        torch.save({
            "step": total_steps,
            "model_state_dict": self.model.state_dict(),
            "config": self.m_cfg
        }, final_ckpt)
        return final_ckpt
