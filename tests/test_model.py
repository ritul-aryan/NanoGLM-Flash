import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import torch
from configs.model_config import ModelConfig
from models.glm_flash import NanoGLMFlash
from data.tokenizer import ByteTokenizer

def test_model():
    print("=" * 60)
    print("RUNNING NANOGLM-FLASH MODEL VERIFICATION TEST")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Device: {device}")
    
    config = ModelConfig()
    model = NanoGLMFlash(config).to(device)
    
    # Check parameters
    param_info = model.count_parameters()
    print(f"Total Parameters:  {param_info['total_params'] / 1e6:.2f} M ({param_info['total_params']:,})")
    print(f"Active Parameters: {param_info['active_params'] / 1e6:.2f} M ({param_info['active_params']:,})")
    print(f"Sparsity:          {param_info['sparsity_pct']:.1f}%")
    
    # 1. Text-only forward test
    B, S = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    labels = input_ids.clone()
    
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        out = model(input_ids, labels=labels)
        loss = out["loss"]
        ce_loss = out["ce_loss"]
        aux_loss = out["aux_loss"]
        
    print(f"\n[Test 1] Text Forward Pass:")
    print(f"  Logits Shape: {out['logits'].shape}")
    print(f"  Total Loss:   {loss.item():.4f} (CE: {ce_loss.item():.4f}, Aux: {aux_loss.item():.4f})")
    assert out['logits'].shape == (B, S, config.vocab_size)
    assert not torch.isnan(loss)
    
    # 2. Backward pass test
    loss.backward()
    print("  Backward pass completed successfully without NaNs!")
    
    # 3. Multi-modal forward test
    images = torch.randn(B, 3, config.img_size, config.img_size, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        mm_out = model(input_ids, images=images, labels=labels)
        mm_loss = mm_out["loss"]
        
    print(f"\n[Test 2] Multi-Modal Forward Pass:")
    print(f"  MM Logits Shape: {mm_out['logits'].shape} (16 image patches + {S} tokens)")
    print(f"  MM Total Loss:   {mm_loss.item():.4f}")
    assert mm_out['logits'].shape == (B, config.num_patches + S, config.vocab_size)
    
    # 4. Generation test
    tokenizer = ByteTokenizer()
    prompt = "<user_start>Hello<user_end><assistant_start>"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        gen_tokens = model.generate(prompt_ids, max_new_tokens=16, temperature=0.7)
        
    gen_text = tokenizer.decode(gen_tokens[0])
    print(f"\n[Test 3] Generation Rollout:")
    print(f"  Generated Output: {repr(gen_text)}")
    print("=" * 60)
    print("ALL MODEL ARCHITECTURE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_model()
