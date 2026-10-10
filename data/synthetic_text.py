import random
from typing import List, Dict, Tuple

class SyntheticTextGenerator:
    """
    Rich synthetic dataset generator covering:
    - Conversational chat (greetings, identity, explanations)
    - Chain-of-thought (CoT) arithmetic and logic
    - Python coding & algorithmic functions
    - Fact answering and explanations
    """
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def gen_conversational(self) -> str:
        conversations = [
            (
                "Hello! Who are you?",
                "I am NanoGLM-Flash, a miniature multi-modal language model trained from scratch with hybrid attention and reinforcement learning.",
                "Identify self as NanoGLM-Flash and explain capabilities concisely."
            ),
            (
                "What is your architecture?",
                "I am built with 10 hybrid layers combining Gated Linear Attention and Sliding-Window RoPE Attention, backed by a Sparse Mixture-of-Experts (MoE) with 1 shared and 4 routed SwiGLU experts.",
                "Detail key components: hybrid linear/window attention and sparse MoE."
            ),
            (
                "Can you process images?",
                "Yes! I have a 2D patch-projection vision encoder that extracts 16 visual tokens from 64x64 images and integrates them directly into my attention context.",
                "Confirm multi-modal vision prefix support."
            ),
            (
                "What is Python?",
                "Python is a high-level, interpreted programming language known for its readability and extensive library ecosystem for AI and scientific computing.",
                "Provide clear, concise definition of Python."
            ),
            (
                "How does linear attention work?",
                "Linear attention computes attention in O(T) linear time during training and O(1) recurrent state space during generation, avoiding the quadratic KV-cache growth of standard transformers.",
                "Explain O(T) vs O(1) linear attention mechanism."
            ),
            (
                "Tell me a short joke.",
                "Why do programmers prefer dark mode? Because light attracts bugs!",
                "Deliver a classic software engineering joke."
            ),
            (
                "What is reinforcement learning?",
                "Reinforcement learning is a machine learning paradigm where an agent learns to maximize cumulative rewards through trial and error interactions with an environment.",
                "Define RL policy optimization and reward feedback."
            ),
            (
                "How do you solve problems?",
                "I use chain-of-thought reasoning to break complex queries down into intermediate steps before generating the final answer.",
                "Explain internal chain-of-thought scratchpad."
            ),
        ]
        q, a, thought = self.rng.choice(conversations)
        return f"<user_start>{q}<user_end><assistant_start><thought_start>{thought}<thought_end>{a}<assistant_end>"

    def gen_arithmetic_cot(self) -> str:
        ops = ["+", "-", "*"]
        op = self.rng.choice(ops)
        if op == "+":
            a = self.rng.randint(10, 999)
            b = self.rng.randint(10, 999)
            ans = a + b
            steps = f"Add {a} and {b} by calculating sum of place values: {a} + {b} = {ans}."
        elif op == "-":
            a = self.rng.randint(100, 999)
            b = self.rng.randint(10, a)
            ans = a - b
            steps = f"Subtract {b} from {a}: difference is {ans}."
        else: # *
            a = self.rng.randint(3, 30)
            b = self.rng.randint(2, 15)
            ans = a * b
            steps = f"Multiply {a} by {b}: product is {ans}."
            
        prompt = f"<user_start>Calculate: {a} {op} {b}<user_end><assistant_start><thought_start>{steps}<thought_end>The answer is {ans}.<assistant_end>"
        return prompt

    def gen_string_reverse_cot(self) -> str:
        words = ["alpha", "beta", "gamma", "delta", "matrix", "vector", "tensor", "kernel", "gradient", "token", "neural", "attention"]
        word = self.rng.choice(words)
        rev = word[::-1]
        steps = f"String length is {len(word)}. Reversing characters gives '{rev}'."
        prompt = f"<user_start>Reverse the word '{word}'<user_end><assistant_start><thought_start>{steps}<thought_end>Reversed: {rev}<assistant_end>"
        return prompt

    def gen_code_completion(self) -> str:
        templates = [
            (
                "Write a Python function `solution(nums)` to sum even numbers.",
                "def solution(nums):\n    return sum(x for x in nums if x % 2 == 0)",
                "Iterate through nums, filter even numbers with modulo 2, and sum them."
            ),
            (
                "Write a Python function `solution(s)` to check if a string is a palindrome.",
                "def solution(s):\n    return s == s[::-1]",
                "Compare s with its reverse slice s[::-1]."
            ),
            (
                "Write a Python function `solution(text)` to count vowels.",
                "def solution(text):\n    return sum(1 for c in text.lower() if c in 'aeiou')",
                "Convert text to lowercase and count characters matching 'aeiou'."
            ),
            (
                "Write a Python function `solution(nums)` to find the range (max minus min).",
                "def solution(nums):\n    return max(nums) - min(nums) if nums else 0",
                "Subtract min(nums) from max(nums)."
            ),
            (
                "Write a Python function `solution(n)` to calculate factorial.",
                "def solution(n):\n    res = 1\n    for i in range(2, n + 1):\n        res *= i\n    return res",
                "Compute product from 2 through n iteratively."
            ),
            (
                "Write a Python function `solution(sentence)` to reverse words in a string.",
                "def solution(sentence):\n    return ' '.join(sentence.split()[::-1])",
                "Split string into word tokens, reverse list, and rejoin with space."
            ),
            (
                "Write a Python function `square(x)` that returns the square of x.",
                "def square(x):\n    return x * x",
                "Multiply x by itself."
            ),
            (
                "Write a Python function `is_positive(x)` that checks if x > 0.",
                "def is_positive(x):\n    return x > 0",
                "Return boolean check x > 0."
            ),
        ]
        q, code, thought = self.rng.choice(templates)
        prompt = f"<user_start>{q}<user_end><assistant_start><thought_start>{thought}<thought_end><code_start>{code}<code_end><assistant_end>"
        return prompt

    def generate_batch(self, batch_size: int = 16) -> List[str]:
        samples = []
        generators = [
            self.gen_conversational,
            self.gen_conversational, # weight chat higher
            self.gen_arithmetic_cot,
            self.gen_string_reverse_cot,
            self.gen_code_completion,
            self.gen_code_completion
        ]
        for _ in range(batch_size):
            fn = self.rng.choice(generators)
            samples.append(fn())
        return samples
