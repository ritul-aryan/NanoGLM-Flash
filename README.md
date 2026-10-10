# NanoGLM-Flash-72M: Multi-Modal Miniature Language Model & GRPO Pipeline

[![PyTorch](https://img.shields.io/badge/PyTorch-2.14.1%2Bcu132-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20RTX%205050%20(Blackwell)-76B900?logo=nvidia&logoColor=white)](https://www.nvidia.com)
[![CUDA](https://img.shields.io/badge/CUDA-13.2-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Architecture](https://img.shields.io/badge/Architecture-Hybrid%20GLA%20%2B%20MoE-6366F1)](https://github.com)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end multi-modal miniature language model and reinforcement learning post-training pipeline inspired by the architectural principles of **GLM-5.3-Flash**. Designed, implemented, and benchmarked from scratch in Python on an **NVIDIA GeForce RTX 5050 Laptop GPU** (Blackwell architecture, 8 GB GDDR7 VRAM).

---

## 🚀 Architectural Innovations & System Design

```
                     +---------------------------------------+
                     |    Input Prompt & 64x64 Vision RGB    |
                     +-------------------+-------------------+
                                         |
               +-------------------------+-------------------------+
               |                                                   |
      [Byte Tokenizer]                                    [2D Patch Projector]
      (384 Vocab, Tied)                                   (16 Spatial Tokens)
               |                                                   |
               +-------------------------+-------------------------+
                                         |
                                         v
                         +-------------------------------+
                         |   10 Hybrid Transformer Blocks|
                         +---------------+---------------+
                                         |
                 +-----------------------+-----------------------+
                 | (7 Layers)                                    | (3 Periodic Layers)
      [Gated Linear Attention]                         [Sliding-Window RoPE Attn]
      - O(1) Recurrent Memory State                    - 256-token Window
      - Sub-quadratic Token Generation                 - FlashAttention / SDPA Causal
                 |                                               |
                 +-----------------------+-----------------------+
                                         |
                                         v
                         +-------------------------------+
                         |      Sparse MoE Feed-Forward   |
                         |  (1 Shared + 4 Routed SwiGLU)  |
                         |     Top-2 Dynamic Gating      |
                         |    32.7% FLOP Sparsity        |
                         +---------------+---------------+
                                         |
                                         v
                         +-------------------------------+
                         | Tied LM Output Projection Head|
                         +---------------+---------------+
                                         |
                                         v
                         +-------------------------------+
                         | Zero-Critic GRPO RL Engine    |
                         |  + AST Sandboxed Code Runner  |
                         +-------------------------------+
```

### 1. Key Technical Highlights
- **Parameter Scale**: **72.20M Total Parameters / 48.60M Active Parameters** per token (**32.7% FLOP Sparsity**).
- **Sub-Quadratic Hybrid Attention**:
  - **7 Gated Linear Attention (GLA) Layers**: Implements data-dependent decay gates updating a fixed-size recurrent state matrix $S_t \in \mathbb{R}^{d_k \times d_v}$ in $O(1)$ memory per step during autoregressive generation, eliminating KV-cache growth during RL rollouts.
  - **3 Sliding-Window RoPE Attention Layers** (layers 3, 6, 9): 256-token causal sliding-window attention with Rotary Position Embeddings (RoPE) accelerated via PyTorch SDPA FlashAttention kernels for exact token retrieval.
- **Sparse Mixture of Experts (MoE)**:
  - **1 Shared Invariant Expert**: Active on 100% of tokens to ground core syntax and universal grammar.
  - **4 Routed SwiGLU Experts** ($d_{\text{inter}} = 768$): Dispatched via Top-2 router softmax gating.
  - **Auxiliary Load-Balancing Loss**: $\alpha \cdot N \sum_i f_i \cdot P_i$ penalizes routing collapse across mini-batches.
- **Byte-Level Tokenization**: 384 tokens (256 raw UTF-8 bytes + control & reasoning tokens) with tied input/output weights (`384 x 512 = 196,608` parameters, only `0.39 MB`), eliminating large vocabulary bloat.
- **Multi-Modal Vision Prefix**: 2D patch-projection maps $64 \times 64$ RGB images into 16 visual tokens prepended ahead of prompt text.
- **Post-Training Reinforcement Learning (GRPO)**:
  - **Zero-Critic Group Relative Policy Optimization**: Eliminates the heavy Value/Critic model, slashing training VRAM footprint by 50%.
  - **AST-Isolated Code Execution Sandbox**: Subprocess-isolated evaluation harness with abstract syntax tree security validation and automated unit test assertion scoring.
  - Group-relative advantages: $A_i = \frac{R_i - \bar{R}}{\sigma_R + \epsilon}$ driving clipped surrogate policy gradient updates.

---

## ⚡ Target Hardware & Empirical VRAM Budget

Benchmarked on **Acer Nitro ANV15-52** with **NVIDIA GeForce RTX 5050 Laptop GPU** (Compute Capability `12.0`, Blackwell `sm_120`, 8 GB GDDR7 VRAM, CUDA 13.2 runtime):

| Execution Phase / Tensor Group | Precision | Memory (MB) | Memory (GB) |
| :--- | :--- | :--- | :--- |
| **Policy Model Weights (72.2M)** | BF16 | 144.4 MB | 0.14 GB |
| **Reference Policy Model $\pi_{\text{ref}}$** | BF16 | 144.4 MB | 0.14 GB |
| **AdamW Optimizer States (Momentum + Variance)** | FP32 | 577.5 MB | 0.56 GB |
| **Model Gradients** | BF16 | 144.4 MB | 0.14 GB |
| **Pre-Training Forward/Backward Peak ($B=8, S=512$)** | BF16 | 1,182.6 MB | **1.15 GB** |
| **GRPO Generation Rollout Peak ($K=16, S=512$)** | BF16 | 2,124.2 MB | **2.07 GB** |
| **Physical Usable Hardware VRAM** | — | 6,950 MB | **6.79 GB** |
| **Safety Threshold Buffer** | — | 6,656 MB | **6.50 GB** |
| **Verified Headroom (Free Buffer)** | — | **4,826 MB** | **4.71 GB (70.2%)** |

- **Training Throughput**: **34,400+ tokens/sec** in BF16 mixed precision.
- **Causal Attention Latency**: **0.132 ms** FlashAttention SDPA causal forward pass.
- **GEMM Throughput**: **23.16 TFLOPS** (FP16) / **19.37 TFLOPS** (BF16).

---

## 🖥️ Live Web UI Dashboard & Telemetry

The repository includes a modern, recruiter-ready AI systems dashboard featuring:
- **Live Hardware Telemetry**: VRAM allocation gauge, GPU clock, and memory efficiency indicators.
- **Hybrid Architecture Layer Visualizer**: Interactive inspector mapping GLA vs Sliding RoPE layers and MoE routing.
- **Reasoning Traces**: Distinct `<thought>` visualization displaying chain-of-thought token extraction.
- **Interactive Code Sandbox Runner**: Run button that sends generated code directly to the AST sandbox and displays unit test pass/fail results and latency.
- **Multi-Modal Visual Testing**: Interactive canvas with circle counting and geometric reasoning presets.

### Launching the Web Server:
```powershell
python main.py --mode web --port 7860
# Or directly:
python web_chat.py
```
Open your browser to: **`http://127.0.0.1:7860`**

---

## 📂 Repository Structure

```
.
├── configs/
│   ├── model_config.py          # NanoGLM-Flash-72M dataclass (d_model=512, L=10, MoE, vision)
│   ├── pretrain_config.py       # Pre-training optimizer, lr schedule, dataloader config
│   └── rl_config.py             # GRPO configuration (K=4, clip_eps=0.2, beta_kl=0.04, rewards)
├── data/
│   ├── tokenizer.py             # Byte-level tokenizer (384 vocab, raw byte mapping, special tokens)
│   ├── synthetic_text.py        # Algorithmic reasoning & multi-step language synthesis generator
│   ├── synthetic_vision.py      # Multi-modal image dataset generator (shapes, counts, visual math)
│   └── rl_coding_tasks.py       # Python code synthesis task suite with automated unit tests
├── models/
│   ├── embeddings.py            # Tied byte embeddings & Rotary Position Embeddings (RoPE)
│   ├── linear_attention.py      # Gated Linear Attention (recurrent O(1) inference state)
│   ├── sliding_attention.py     # Sliding-window causal multi-head self-attention with RoPE & SDPA
│   ├── moe.py                   # Sparse MoE: Router top-2 gating, 1 shared + 4 routed SwiGLU experts
│   ├── vision_prefix.py         # 2D patch-projection vision encoder (Conv/MLP prefix projector)
│   ├── block.py                 # Hybrid Transformer Block orchestrating Attention schedule + MoE
│   └── glm_flash.py             # Top-level GLM-Flash Multi-Modal Causal LM (end-to-end integration)
├── training/
│   ├── pretrain.py              # Pre-training loop with BF16 mixed precision, AdamW, cosine schedule
│   └── grpo_trainer.py          # GRPO post-training loop (rollout sampling, sandboxed reward, clipped loss)
├── sandbox/
│   └── code_sandbox.py          # Isolated, secure Python execution sandbox with AST checks & timeouts
├── benchmark/
│   ├── memory_profiler.py       # Live VRAM profiler tracking static weights, optimizer, activations
│   ├── inference_benchmark.py   # Latency & throughput benchmark (tokens/sec, step latency)
│   └── eval_accuracy.py         # Evaluation benchmark across visual reasoning, math, and code synthesis
├── tests/
│   ├── test_model.py            # Unit test verifying forward/backward passes and parameter counts
│   ├── test_sandbox.py          # Unit test verifying AST checks, timeouts, and assertion evaluations
│   └── test_web_endpoints.py    # End-to-end test verifying chat, sandbox, and vision endpoints
├── web_chat.py                  # Standalone Web UI server & hardened inference runtime
├── main.py                      # Unified CLI entry point
├── requirements.txt             # Pinned virtual environment dependencies
└── README.md                    # Project documentation
```

---

## 🛠️ Quickstart Guide

### 1. Environment Setup
```powershell
# Create & activate dedicated virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
python -m pip install -r requirements.txt
```

### 2. Run Diagnostics & Memory Profiler
```powershell
python main.py --mode diagnostic
```

### 3. Run Pre-Training Loop
```powershell
python main.py --mode pretrain --steps 250
```

### 4. Run GRPO Reinforcement Learning Post-Training
```powershell
python main.py --mode rl --steps 30 --checkpoint checkpoints/pretrain_final.pt
```

### 5. Run Verification & Benchmarks
```powershell
# Run automated test suites
python tests/test_model.py
python tests/test_sandbox.py
python tests/test_web_endpoints.py

# Run latency & throughput benchmarks
python main.py --mode benchmark
```

### 6. Start the Web UI Dashboard
```powershell
python main.py --mode web --port 7860
```
Open **`http://127.0.0.1:7860`** to interact with the model in real time!

---

## 📜 License
Distributed under the MIT License. Built with ❤️ for deep learning systems architecture research.
