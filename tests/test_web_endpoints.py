import urllib.request
import json
import sys
import base64
import io
from PIL import Image, ImageDraw

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def post_json(endpoint, payload):
    req = urllib.request.Request(
        f'http://127.0.0.1:7860{endpoint}',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))

def test_endpoints():
    print("=" * 65)
    print("RUNNING NANOGLM-FLASH SHOWCASE WEB TEST SUITE")
    print("=" * 65)

    # 1. Identity
    r_hi = post_json('/chat', {'prompt': 'Hi'})
    print("[+] Test 1 (Identity - 'Hi'):")
    print("    Thought:", r_hi['thought'])
    print("    Text:", r_hi['text'][:70], "...")
    assert "NanoGLM" in r_hi['text']

    # 2. Architecture
    r_arch = post_json('/chat', {'prompt': 'What is your architecture?'})
    print("\n[+] Test 2 (Architecture Specs):")
    print("    Thought:", r_arch['thought'])
    print("    Text snippet:", r_arch['text'][:80], "...")
    assert "72.20M" in r_arch['text'] and "Gated Linear Attention" in r_arch['text']

    # 3. MoE Explanation
    r_moe = post_json('/chat', {'prompt': 'Explain MoE routing'})
    print("\n[+] Test 3 (MoE Routing):")
    print("    Thought:", r_moe['thought'])
    print("    Text snippet:", r_moe['text'][:80], "...")
    assert "Shared Expert" in r_moe['text'] or "Routed" in r_moe['text']

    # 4. Code Generation + AST Sandbox Execution
    r_qs = post_json('/chat', {'prompt': 'Write quicksort in python'})
    print("\n[+] Test 4 (Code Generation - Quicksort):")
    print("    Thought:", r_qs['thought'])
    print("    Code snippet:", r_qs['code'][:60], "...")
    assert "quicksort" in r_qs['code']

    r_sandbox = post_json('/run_code', {'code': r_qs['code']})
    print("    Sandbox Execution:")
    print("    Success:", r_sandbox['success'])
    print("    Tests Passed:", r_sandbox['tests_passed'], "/", r_sandbox['total_tests'])
    print("    Time:", f"{r_sandbox['execution_time']*1000:.1f} ms")
    assert r_sandbox['success']

    # 5. Math evaluation
    r_math = post_json('/chat', {'prompt': 'Calculate 48 * 12'})
    print("\n[+] Test 5 (Math calculation):")
    print("    Thought:", r_math['thought'])
    print("    Text:", r_math['text'])
    assert "576" in r_math['text']

    # 6. Multi-Modal Vision
    img = Image.new("RGB", (64, 64), color=(240, 240, 245))
    draw = ImageDraw.Draw(img)
    draw.ellipse((8, 8, 24, 24), fill=(239, 68, 68))
    draw.ellipse((36, 12, 52, 28), fill=(239, 68, 68))
    draw.ellipse((20, 36, 38, 54), fill=(239, 68, 68))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    r_vis = post_json('/chat', {'prompt': 'How many circles are in this image?', 'image': img_b64})
    print("\n[+] Test 6 (Multi-Modal Vision):")
    print("    Thought:", r_vis['thought'])
    print("    Text:", r_vis['text'])
    assert "3 red circles" in r_vis['text']

    print("\n" + "=" * 65)
    print("ALL 6 ENDPOINT AND SANDBOX TESTS PASSED FLAWLESSLY!")
    print("=" * 65)

if __name__ == "__main__":
    test_endpoints()
