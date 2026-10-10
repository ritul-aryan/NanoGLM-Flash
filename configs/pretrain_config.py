from dataclasses import dataclass

@dataclass
class PretrainConfig:
    """
    Configuration for pre-training / SFT optimization and scheduling.
    """
    batch_size: int = 8
    seq_len: int = 512
    learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    warmup_steps: int = 20
    max_steps: int = 150
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    mixed_precision: str = "bf16"         # 'bf16' or 'fp16'
    eval_interval: int = 25
    save_interval: int = 50
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda"
