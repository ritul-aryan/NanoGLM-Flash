import os
import torch
import tabulate
from configs.model_config import ModelConfig
from models.glm_flash import NanoGLMFlash
from data.tokenizer import ByteTokenizer
from data.synthetic_vision import SyntheticVisionGenerator
from data.synthetic_text import SyntheticTextGenerator
from data.rl_coding_tasks import RLCodingBenchmark
from sandbox.code_sandbox import SandboxedCodeExecutor

def evaluate_downstream(model_weights_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("        DOWNSTREAM REASONING & ACCURACY EVALUATION          ")
    print("=" * 65)
    
    config = ModelConfig()
    model = NanoGLMFlash(config).to(device=device, dtype=torch.bfloat16)
    if model_weights_path and os.path.exists(model_weights_path):
        ckpt = torch.load(model_weights_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("actor_state_dict", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {model_weights_path}")
    else:
        print("Evaluating baseline initialized model...")
        
    model.eval()
    tokenizer = ByteTokenizer(vocab_size=config.vocab_size)
    sandbox = SandboxedCodeExecutor(timeout_sec=1.5)
    
    # 1. Evaluate Coding Tasks (Pass@1)
    tasks = RLCodingBenchmark.get_all_tasks()
    coding_passed = 0
    total_tasks = len(tasks)
    
    print("\n--- Evaluating Python Code Synthesis Suite ---")
    coding_table = []
    for task in tasks:
        p_tokens = torch.tensor([tokenizer.encode(task.prompt, add_bos=True)], device=device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            gen = model.generate(p_tokens, max_new_tokens=128, temperature=0.2)
        gen_text = tokenizer.decode(gen[0][p_tokens.shape[1]:])
        res = sandbox.execute_and_evaluate(gen_text, task.test_cases)
        
        status = "PASSED" if res.success else "FAILED"
        if res.success:
            coding_passed += 1
        coding_table.append([task.task_id, f"{res.tests_passed}/{res.total_tests}", f"{res.reward:.2f}", status])
        
    print(tabulate.tabulate(coding_table, headers=["Task", "Tests Passed", "Reward", "Status"], tablefmt="grid"))
    pass_pct = (coding_passed / total_tasks) * 100.0
    print(f"Code Synthesis Pass@1: {coding_passed}/{total_tasks} ({pass_pct:.1f}%)")
    
    # 2. Evaluate Visual Reasoning
    print("\n--- Evaluating Multi-Modal Visual Reasoning ---")
    vis_gen = SyntheticVisionGenerator(img_size=config.img_size, seed=123)
    vis_correct = 0
    vis_total = 8
    
    vis_table = []
    for i in range(vis_total):
        img_t, prompt_text = vis_gen.generate_sample()
        img_t = img_t.unsqueeze(0).to(device)
        # Separate prompt from expected answer
        parts = prompt_text.split("<assistant_start>")
        user_prompt = parts[0] + "<assistant_start>"
        expected_ans = parts[1].replace("<assistant_end>", "").strip()
        
        p_tokens = torch.tensor([tokenizer.encode(user_prompt, add_bos=True)], device=device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            gen = model.generate(p_tokens, images=img_t, max_new_tokens=48, temperature=0.2)
        gen_ans = tokenizer.decode(gen[0][p_tokens.shape[1]:]).replace("<assistant_end>", "").strip()
        
        # Check substring containment or match
        matched = (expected_ans in gen_ans) or (expected_ans.lower() in gen_ans.lower())
        if matched:
            vis_correct += 1
        vis_table.append([f"Sample {i+1}", expected_ans[:20], gen_ans[:25], "CORRECT" if matched else "INCORRECT"])
        
    print(tabulate.tabulate(vis_table, headers=["Sample", "Expected", "Generated", "Result"], tablefmt="grid"))
    vis_pct = (vis_correct / vis_total) * 100.0
    print(f"Visual Reasoning Score: {vis_correct}/{vis_total} ({vis_pct:.1f}%)")
    print("=" * 65)

if __name__ == "__main__":
    evaluate_downstream()
