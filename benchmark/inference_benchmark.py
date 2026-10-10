import time
import torch
import tabulate
from configs.model_config import ModelConfig
from models.glm_flash import NanoGLMFlash

def run_inference_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("       NANOGLM-FLASH INFERENCE THROUGHPUT BENCHMARK         ")
    print("=" * 65)
    
    config = ModelConfig()
    model = NanoGLMFlash(config).to(device=device, dtype=torch.bfloat16)
    model.eval()
    
    batch_sizes = [1, 4, 8]
    gen_lengths = [64, 128]
    prompt_len = 32
    
    results = []
    
    # Warmup
    dummy = torch.randint(0, config.vocab_size, (1, 16), device=device)
    with torch.no_grad():
        _ = model.generate(dummy, max_new_tokens=8)
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    for b in batch_sizes:
        for g_len in gen_lengths:
            prompt = torch.randint(0, config.vocab_size, (b, prompt_len), device=device)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out = model.generate(prompt, max_new_tokens=g_len, temperature=0.7)
                
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            total_time = t1 - t0
            total_tokens = b * g_len
            tok_per_sec = total_tokens / total_time
            latency_per_tok = (total_time / g_len) * 1000.0 # ms per step
            
            results.append([
                b,
                prompt_len,
                g_len,
                f"{total_time:.3f} s",
                f"{latency_per_tok:.2f} ms",
                f"{tok_per_sec:.1f} tok/s"
            ])
            
    headers = ["Batch Size", "Prompt Len", "Gen Len", "Total Time", "Step Latency", "Throughput"]
    print(tabulate.tabulate(results, headers=headers, tablefmt="grid"))
    print("=" * 65)

if __name__ == "__main__":
    run_inference_benchmark()
