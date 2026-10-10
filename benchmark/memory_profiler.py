import torch
import tabulate
from configs.model_config import ModelConfig
from models.glm_flash import NanoGLMFlash

def profile_vram_breakdown():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("        PRECISION VRAM CONSUMPTION PROFILING (BF16)         ")
    print("=" * 65)
    
    if device.type != "cuda":
        print("CUDA device not available for hardware VRAM profiling.")
        return
        
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    base_free, base_total = torch.cuda.mem_get_info(0)
    
    # 1. Instantiate Model in BF16
    config = ModelConfig()
    model = NanoGLMFlash(config).to(device=device, dtype=torch.bfloat16)
    model_bytes = torch.cuda.memory_allocated()
    
    # 2. Add Reference Policy (Frozen copy in BF16)
    ref_model = NanoGLMFlash(config).to(device=device, dtype=torch.bfloat16)
    ref_model.eval()
    ref_bytes = torch.cuda.memory_allocated() - model_bytes
    
    # 3. Initialize AdamW Optimizer States for Policy Actor
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for p in model.parameters():
        optimizer.state[p] = {
            'step': torch.tensor(0.0),
            'exp_avg': torch.zeros_like(p, dtype=torch.float32),
            'exp_avg_sq': torch.zeros_like(p, dtype=torch.float32)
        }
    opt_bytes = torch.cuda.memory_allocated() - (model_bytes + ref_bytes)
    
    # 4. Measure Dynamic Forward Activations across sequence lengths
    seq_lengths = [128, 256, 512]
    batch_size = 8
    act_results = []
    
    for s in seq_lengths:
        torch.cuda.reset_peak_memory_stats()
        x = torch.randint(0, config.vocab_size, (batch_size, s), device=device)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x, labels=x)
            loss = out["loss"]
        loss.backward()
        
        peak_bytes = torch.cuda.max_memory_allocated()
        act_overhead = peak_bytes - (model_bytes + ref_bytes + opt_bytes)
        act_results.append([f"B={batch_size}, S={s}", f"{act_overhead / (1024**2):.1f} MB", f"{peak_bytes / (1024**2):.1f} MB", f"{peak_bytes / (1024**3):.2f} GB"])
        del x, out, loss
        torch.cuda.empty_cache()
        
    # 5. Measure GRPO Rollout Generation (16 candidate rollouts)
    torch.cuda.reset_peak_memory_stats()
    prompt_rollouts = torch.randint(0, config.vocab_size, (16, 64), device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        gen_tokens = model.generate(prompt_rollouts, max_new_tokens=128)
    rollout_peak = torch.cuda.max_memory_allocated()
    del prompt_rollouts, gen_tokens
    torch.cuda.empty_cache()

    # Format table
    summary_data = [
        ["Policy Actor Model (BF16)", f"{model_bytes / (1024**2):.1f} MB", f"{model_bytes / (1024**3):.2f} GB"],
        ["Reference Policy pi_ref (BF16)", f"{ref_bytes / (1024**2):.1f} MB", f"{ref_bytes / (1024**3):.2f} GB"],
        ["AdamW Optimizer States (FP32)", f"{opt_bytes / (1024**2):.1f} MB", f"{opt_bytes / (1024**3):.2f} GB"],
        ["GRPO Rollout Inference Peak (K=16)", f"{rollout_peak / (1024**2):.1f} MB", f"{rollout_peak / (1024**3):.2f} GB"],
        ["Pre-Training SFT Peak (B=8, S=512)", act_results[-1][1], act_results[-1][3]],
    ]
    
    print("\n--- Static Weights & State Allocations ---")
    print(tabulate.tabulate(summary_data, headers=["Phase / Component", "Memory (MB)", "Memory (GB)"], tablefmt="grid"))
    
    print("\n--- Dynamic SFT Forward/Backward Footprint ---")
    print(tabulate.tabulate(act_results, headers=["Config", "Activation Buffer", "Peak VRAM", "Total GB"], tablefmt="grid"))
    
    final_free, final_total = torch.cuda.mem_get_info(0)
    print(f"\nHardware Global VRAM:  {final_total / (1024**3):.2f} GB")
    print(f"Safety Ceiling Target: 6.50 GB")
    print(f"Peak Observed VRAM:    {rollout_peak / (1024**3):.2f} GB (Leaves {(final_total - rollout_peak) / (1024**3):.2f} GB headroom)")
    print("=" * 65)

if __name__ == "__main__":
    profile_vram_breakdown()
