from dataclasses import dataclass

@dataclass
class RLConfig:
    """
    Configuration for Group Relative Policy Optimization (GRPO) post-training.
    """
    group_size_k: int = 4                 # K candidate rollouts sampled per prompt
    prompt_batch_size: int = 4            # Prompts per step (total rollouts = 4 * 4 = 16 sequences)
    max_gen_len: int = 192                # Max completion tokens per rollout
    temperature: float = 0.8              # Sampling temperature for diverse rollouts
    top_p: float = 0.95                   # Nucleus sampling threshold
    
    # GRPO Loss Hyperparameters
    clip_eps: float = 0.2                 # PPO/GRPO policy ratio clipping threshold
    beta_kl: float = 0.04                 # KL penalty against reference policy
    learning_rate: float = 1e-5           # Policy actor learning rate
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    max_steps: int = 80                   # RL post-training optimization steps
    
    # Sandboxed Execution & Rewards
    execution_timeout_sec: float = 1.5    # Hard subprocess timeout for candidate code
    correct_reward: float = 1.0           # Reward when all unit test assertions pass
    partial_reward_weight: float = 0.5    # Weight for fraction of passed assertions
    format_bonus: float = 0.1             # Bonus for proper <thought> and <code> structure
    syntax_error_penalty: float = 0.0     # Penalty for unparseable syntax
    timeout_penalty: float = -0.5         # Penalty for infinite loops or execution timeouts
    
    checkpoint_dir: str = "checkpoints/rl"
    device: str = "cuda"
