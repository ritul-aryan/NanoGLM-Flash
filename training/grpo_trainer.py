import os
import sys
import time
import copy
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from configs.model_config import ModelConfig
from configs.rl_config import RLConfig
from models.glm_flash import NanoGLMFlash
from data.tokenizer import ByteTokenizer
from data.rl_coding_tasks import RLCodingBenchmark, CodingTask
from sandbox.code_sandbox import SandboxedCodeExecutor, ExecutionResult

class GRPOTrainer:
    """
    Group Relative Policy Optimization (GRPO) Post-Training Engine.
    - Zero-critic architecture (eliminates value network overhead)
    - Generates K candidate rollouts per task prompt
    - Evaluates code correctness in the isolated Python execution sandbox
    - Computes group-relative normalized advantage
    - Clipped policy surrogate loss with KL divergence constraint to reference policy
    """
    def __init__(
        self,
        model_config: ModelConfig,
        rl_config: RLConfig,
        pretrained_weights_path: Optional[str] = None
    ):
        self.m_cfg = model_config
        self.rl_cfg = rl_config
        self.device = torch.device(rl_config.device if torch.cuda.is_available() else "cpu")
        
        # 1. Initialize Actor Policy Model
        self.actor = NanoGLMFlash(model_config).to(self.device)
        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            ckpt = torch.load(pretrained_weights_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.actor.load_state_dict(state_dict)
            print(f"Loaded pretrained weights from {pretrained_weights_path}")
            
        # 2. Initialize Reference Policy (Frozen copy in BF16)
        self.ref_policy = copy.deepcopy(self.actor)
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad = False
            
        # 3. Tokenizer & Execution Sandbox
        self.tokenizer = ByteTokenizer(vocab_size=model_config.vocab_size)
        self.sandbox = SandboxedCodeExecutor(timeout_sec=rl_config.execution_timeout_sec)
        self.tasks = RLCodingBenchmark.get_all_tasks()
        
        # 4. AdamW Optimizer for Policy Actor
        self.optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=rl_config.learning_rate,
            weight_decay=rl_config.weight_decay,
            eps=1e-8
        )
        
        os.makedirs(rl_config.checkpoint_dir, exist_ok=True)

    def generate_rollouts(
        self,
        task: CodingTask,
        group_size_k: int
    ) -> Tuple[List[str], List[torch.Tensor], List[ExecutionResult]]:
        """
        Samples K rollouts for a coding prompt and evaluates each in the sandbox.
        """
        prompt_str = task.prompt
        prompt_tokens = torch.tensor(
            [self.tokenizer.encode(prompt_str, add_bos=True)],
            device=self.device
        ) # (1, prompt_len)
        
        # Tile prompt K times for parallel batch generation
        prompt_batch = prompt_tokens.repeat(group_size_k, 1) # (K, prompt_len)
        prompt_len = prompt_tokens.shape[1]
        
        self.actor.eval()
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            gen_sequences = self.actor.generate(
                prompt_batch,
                max_new_tokens=self.rl_cfg.max_gen_len,
                temperature=self.rl_cfg.temperature,
                top_p=self.rl_cfg.top_p,
                eos_token_id=self.tokenizer.eos_token_id
            ) # (K, total_len)

        completions_text: List[str] = []
        completion_token_tensors: List[torch.Tensor] = []
        eval_results: List[ExecutionResult] = []
        
        for k in range(group_size_k):
            seq = gen_sequences[k]
            comp_tokens = seq[prompt_len:] # tokens generated after prompt
            comp_text = self.tokenizer.decode(comp_tokens, skip_special_tokens=False)
            
            # Sandboxed evaluation against task test cases
            res = self.sandbox.execute_and_evaluate(
                raw_output=comp_text,
                test_cases=task.test_cases,
                correct_reward=self.rl_cfg.correct_reward,
                partial_weight=self.rl_cfg.partial_reward_weight,
                format_bonus=self.rl_cfg.format_bonus,
                timeout_penalty=self.rl_cfg.timeout_penalty
            )
            
            completions_text.append(comp_text)
            completion_token_tensors.append(seq) # Full sequence (prompt + completion)
            eval_results.append(res)
            
        return completions_text, completion_token_tensors, eval_results

    def compute_completion_log_probs(
        self,
        model: NanoGLMFlash,
        full_sequences: torch.Tensor,
        prompt_len: int
    ) -> torch.Tensor:
        """
        Computes the sum of log probabilities of the completion tokens.
        full_sequences: (K, total_len)
        Returns: (K,) token log prob sum
        """
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            out = model(full_sequences)
            logits = out["logits"] # (K, total_len, vocab_size)
            
        # Shift logits and tokens for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_tokens = full_sequences[:, 1:].contiguous()
        
        # Cross entropy loss per token (negative log likelihood)
        log_probs = nn.functional.log_softmax(shift_logits, dim=-1)
        # Gather log prob of actual target tokens
        token_log_probs = log_probs.gather(dim=-1, index=shift_tokens.unsqueeze(-1)).squeeze(-1) # (K, total_len - 1)
        
        # Mask only the completion tokens (excluding prompt tokens)
        comp_mask = torch.zeros_like(token_log_probs, dtype=torch.bool)
        comp_mask[:, prompt_len - 1:] = True
        
        completion_log_prob_sum = (token_log_probs * comp_mask.float()).sum(dim=-1) # (K,)
        return completion_log_prob_sum

    def grpo_step(self, step: int) -> Dict[str, float]:
        self.actor.train()
        self.optimizer.zero_grad()
        
        # Sample a batch of prompts
        batch_tasks = [
            self.tasks[(step * self.rl_cfg.prompt_batch_size + i) % len(self.tasks)]
            for i in range(self.rl_cfg.prompt_batch_size)
        ]
        
        total_loss = torch.tensor(0.0, device=self.device)
        all_rewards = []
        all_passes = []
        all_kls = []
        
        for task in batch_tasks:
            # 1. Generate K rollouts and evaluate
            texts, seq_tensors, eval_results = self.generate_rollouts(task, self.rl_cfg.group_size_k)
            prompt_len = len(self.tokenizer.encode(task.prompt, add_bos=True))
            
            # Stack sequences: (K, total_len)
            max_len = max(s.shape[0] for s in seq_tensors)
            padded_seqs = torch.full(
                (self.rl_cfg.group_size_k, max_len),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=self.device
            )
            for k_idx, s in enumerate(seq_tensors):
                padded_seqs[k_idx, :s.shape[0]] = s
                
            rewards = torch.tensor([r.reward for r in eval_results], device=self.device, dtype=torch.float32)
            passes = [1.0 if r.success else 0.0 for r in eval_results]
            all_rewards.extend(rewards.tolist())
            all_passes.extend(passes)
            
            # 2. Compute group-relative advantage: A_i = (R_i - mean(R)) / (std(R) + 1e-4)
            mean_r = rewards.mean()
            std_r = rewards.std() + 1e-4
            advantages = (rewards - mean_r) / std_r # (K,)
            
            # 3. Compute Actor log probs with gradients
            actor_log_probs = self.compute_completion_log_probs(self.actor, padded_seqs, prompt_len)
            
            # 4. Compute Reference Policy log probs (frozen)
            with torch.no_grad():
                ref_log_probs = self.compute_completion_log_probs(self.ref_policy, padded_seqs, prompt_len)
                old_log_probs = actor_log_probs.detach().clone()
                
            # 5. Policy ratio & clipped surrogate loss
            # r = exp(log_pi - log_pi_old)
            ratio = torch.exp(actor_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.rl_cfg.clip_eps, 1.0 + self.rl_cfg.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 6. KL divergence penalty: D_KL(pi || pi_ref) approx log_pi - log_pi_ref
            kl_div = (actor_log_probs - ref_log_probs).mean()
            all_kls.append(kl_div.item())
            
            task_loss = policy_loss + (self.rl_cfg.beta_kl * kl_div)
            total_loss = total_loss + (task_loss / self.rl_cfg.prompt_batch_size)
            
        # Backward and step
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.rl_cfg.grad_clip)
        self.optimizer.step()
        
        mean_reward = sum(all_rewards) / len(all_rewards)
        pass_rate = sum(all_passes) / len(all_passes)
        mean_kl = sum(all_kls) / len(all_kls)
        
        return {
            "loss": total_loss.item(),
            "mean_reward": mean_reward,
            "pass_rate": pass_rate,
            "mean_kl": mean_kl
        }

    def train(self, num_steps: Optional[int] = None) -> str:
        total_steps = num_steps or self.rl_cfg.max_steps
        print("=" * 65)
        print(f"STARTING GRPO POST-TRAINING ({total_steps} STEPS, K={self.rl_cfg.group_size_k})")
        print("=" * 65)
        
        t0 = time.time()
        for step in range(1, total_steps + 1):
            step_start = time.perf_counter()
            metrics = self.grpo_step(step)
            step_time = (time.perf_counter() - step_start)
            
            vram_gb = torch.cuda.memory_allocated(self.device) / (1024**3)
            print(
                f"RL Step {step:3d}/{total_steps} | "
                f"Reward: {metrics['mean_reward']:+.3f} | "
                f"Pass@1: {metrics['pass_rate']*100:5.1f}% | "
                f"KL: {metrics['mean_kl']:+.4f} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Step: {step_time:4.1f}s | "
                f"VRAM: {vram_gb:4.2f} GB"
            )
            
            if step % 20 == 0 or step == total_steps:
                save_path = os.path.join(self.rl_cfg.checkpoint_dir, f"grpo_step_{step}.pt")
                torch.save({
                    "step": step,
                    "actor_state_dict": self.actor.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "metrics": metrics
                }, save_path)
                print(f"--> Saved GRPO checkpoint to {save_path}")
                
        elapsed = time.time() - t0
        print(f"GRPO Post-Training completed in {elapsed:.1f}s.")
        final_ckpt = os.path.join(self.rl_cfg.checkpoint_dir, "grpo_final.pt")
        torch.save({
            "step": total_steps,
            "actor_state_dict": self.actor.state_dict(),
            "config": self.m_cfg
        }, final_ckpt)
        return final_ckpt
