import os
import sys
import argparse

# Configure UTF-8 stdout encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from configs.model_config import ModelConfig
from configs.pretrain_config import PretrainConfig
from configs.rl_config import RLConfig
from training.pretrain import Pretrainer
from training.grpo_trainer import GRPOTrainer
from benchmark.memory_profiler import profile_vram_breakdown
from benchmark.inference_benchmark import run_inference_benchmark
from benchmark.eval_accuracy import evaluate_downstream
from data.tokenizer import ByteTokenizer
from data.synthetic_vision import SyntheticVisionGenerator
from models.glm_flash import NanoGLMFlash

def run_diagnostic():
    print("\n>>> RUNNING HARDWARE & VRAM DIAGNOSTIC PROFILER <<<")
    profile_vram_breakdown()

def run_pretrain(steps: int):
    print(f"\n>>> RUNNING PRE-TRAINING ({steps} STEPS) <<<")
    m_cfg = ModelConfig()
    p_cfg = PretrainConfig(max_steps=steps)
    pretrainer = Pretrainer(m_cfg, p_cfg)
    ckpt = pretrainer.train(num_steps=steps)
    print(f"\nPre-training finished successfully! Checkpoint saved to: {ckpt}")
    return ckpt

def run_rl(steps: int, pretrained_ckpt: str = None):
    print(f"\n>>> RUNNING GRPO REINFORCEMENT LEARNING ({steps} STEPS) <<<")
    m_cfg = ModelConfig()
    rl_cfg = RLConfig(max_steps=steps)
    trainer = GRPOTrainer(m_cfg, rl_cfg, pretrained_weights_path=pretrained_ckpt)
    final_ckpt = trainer.train(num_steps=steps)
    print(f"\nGRPO training finished successfully! Checkpoint saved to: {final_ckpt}")
    return final_ckpt

def run_benchmark():
    print("\n>>> RUNNING BENCHMARKS <<<")
    profile_vram_breakdown()
    print("\n")
    run_inference_benchmark()

def run_demo(model_path: str = None):
    print("\n>>> RUNNING MULTI-MODAL INFERENCE DEMO <<<")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig()
    model = NanoGLMFlash(config).to(device=device, dtype=torch.bfloat16)
    if model_path and os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("actor_state_dict", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {model_path}")
        
    tokenizer = ByteTokenizer(vocab_size=config.vocab_size)
    vis_gen = SyntheticVisionGenerator(img_size=config.img_size, seed=99)
    
    # 1. Multi-modal demo
    img_t, _ = vis_gen.generate_sample()
    img_batch = img_t.unsqueeze(0).to(device)
    prompt = "<img_start><img_end><user_start>What shape is in the image?<user_end><assistant_start>"
    tokens = torch.tensor([tokenizer.encode(prompt, add_bos=True)], device=device)
    
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        out = model.generate(tokens, images=img_batch, max_new_tokens=32, temperature=0.7)
    ans = tokenizer.decode(out[0])
    print(f"\n[Multi-Modal Query]:\n{ans}\n")
    
    # 2. Algorithmic code demo
    code_prompt = "<user_start>Write a Python function `solution(nums)` to sum even numbers.<user_end><assistant_start>"
    c_tokens = torch.tensor([tokenizer.encode(code_prompt, add_bos=True)], device=device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        c_out = model.generate(c_tokens, max_new_tokens=48, temperature=0.7)
    c_ans = tokenizer.decode(c_out[0])
    print(f"[Code Synthesis Query]:\n{c_ans}\n")

def run_terminal_chat(checkpoint_path: str = "checkpoints/pretrain_final.pt"):
    print("\n" + "=" * 65)
    print("      ⚡ NANOGLM-FLASH INTERACTIVE TERMINAL CHAT ⚡         ")
    print("      Type your message and press Enter. Type 'exit' to quit. ")
    print("=" * 65 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig()
    model = NanoGLMFlash(config).to(device=device, dtype=torch.bfloat16)
    
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("actor_state_dict", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {checkpoint_path}\n")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found, using base weights.\n")
    model.eval()
    tokenizer = ByteTokenizer(vocab_size=config.vocab_size)
    
    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye!")
                break
                
            formatted_prompt = f"<user_start>{user_input}<user_end><assistant_start>"
            tokens = torch.tensor([tokenizer.encode(formatted_prompt, add_bos=True)], device=device)
            
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                gen = model.generate(
                    tokens,
                    max_new_tokens=96,
                    temperature=0.3,
                    repetition_penalty=1.12,
                    eos_token_id=tokenizer.eos_token_id
                )
                
            raw_gen = tokenizer.decode(gen[0][tokens.shape[1]:])
            
            # Format thought vs answer
            if "<thought_start>" in raw_gen:
                parts = raw_gen.split("<thought_start>", 1)[1]
                if "<thought_end>" in parts:
                    thought, answer = parts.split("<thought_end>", 1)
                    print(f"\n🧠 NanoGLM [Thinking]: {thought.strip()}")
                    print(f"⚡ NanoGLM: {answer.replace('<assistant_end>', '').replace('<eos>', '').strip()}")
                else:
                    print(f"\n🧠 NanoGLM [Thinking]: {parts.strip()}")
            else:
                clean_ans = raw_gen.replace("<assistant_end>", "").replace("<eos>", "").strip()
                print(f"\n⚡ NanoGLM: {clean_ans}")
        except KeyboardInterrupt:
            print("\nExiting chat.")
            break

def main():
    parser = argparse.ArgumentParser(description="NanoGLM-Flash Multi-Modal & RL Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="chat",
        choices=["chat", "web", "diagnostic", "pretrain", "rl", "benchmark", "eval", "demo", "all"],
        help="Pipeline mode to execute"
    )
    parser.add_argument("--steps", type=int, default=50, help="Number of steps for pretrain or RL")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pretrain_final.pt", help="Path to checkpoint weights")
    parser.add_argument("--port", type=int, default=7860, help="Port for web server")
    args = parser.parse_args()

    if args.mode == "chat":
        run_terminal_chat(args.checkpoint)
    elif args.mode == "web":
        from web_chat import run_web_server
        run_web_server(port=args.port, checkpoint_path=args.checkpoint)
    elif args.mode == "diagnostic":
        run_diagnostic()
    elif args.mode == "pretrain":
        run_pretrain(args.steps)
    elif args.mode == "rl":
        run_rl(args.steps, args.checkpoint)
    elif args.mode == "benchmark":
        run_benchmark()
    elif args.mode == "eval":
        evaluate_downstream(args.checkpoint)
    elif args.mode == "demo":
        run_demo(args.checkpoint)
    elif args.mode == "all":
        run_diagnostic()
        ckpt = run_pretrain(args.steps)
        rl_ckpt = run_rl(args.steps, ckpt)
        run_benchmark()
        evaluate_downstream(rl_ckpt)
        run_demo(rl_ckpt)

if __name__ == "__main__":
    main()
