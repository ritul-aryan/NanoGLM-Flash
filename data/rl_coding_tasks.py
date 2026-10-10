from dataclasses import dataclass
from typing import List, Dict

@dataclass
class CodingTask:
    task_id: str
    prompt: str
    test_cases: List[str]
    reference_code: str
    description: str

class RLCodingBenchmark:
    """
    Algorithmic coding tasks for policy gradient RL (GRPO) evaluation.
    Each task requires implementing a function named `solution(...)`.
    """
    TASKS: List[CodingTask] = [
        CodingTask(
            task_id="sum_evens",
            prompt="<user_start>Write a Python function `solution(nums)` that returns the sum of even numbers in a list.<user_end><assistant_start>",
            test_cases=[
                "assert solution([1, 2, 3, 4]) == 6",
                "assert solution([1, 3, 5]) == 0",
                "assert solution([2, 4, 6, 8]) == 20",
                "assert solution([]) == 0"
            ],
            reference_code="def solution(nums):\n    return sum(x for x in nums if x % 2 == 0)",
            description="Sum all even integers in list"
        ),
        CodingTask(
            task_id="is_palindrome",
            prompt="<user_start>Write a Python function `solution(s)` that returns True if the string is a palindrome, else False.<user_end><assistant_start>",
            test_cases=[
                "assert solution('racecar') == True",
                "assert solution('hello') == False",
                "assert solution('a') == True",
                "assert solution('') == True"
            ],
            reference_code="def solution(s):\n    return s == s[::-1]",
            description="Check if string is a palindrome"
        ),
        CodingTask(
            task_id="count_vowels",
            prompt="<user_start>Write a Python function `solution(text)` that returns the number of vowels (a, e, i, o, u case-insensitive).<user_end><assistant_start>",
            test_cases=[
                "assert solution('hello') == 2",
                "assert solution('AI') == 2",
                "assert solution('rhythm') == 0",
                "assert solution('aeiouAEIOU') == 10"
            ],
            reference_code="def solution(text):\n    return sum(1 for c in text.lower() if c in 'aeiou')",
            description="Count vowels in string"
        ),
        CodingTask(
            task_id="max_difference",
            prompt="<user_start>Write a Python function `solution(nums)` that returns the difference between max and min in a non-empty list.<user_end><assistant_start>",
            test_cases=[
                "assert solution([1, 5, 3]) == 4",
                "assert solution([10, 10, 10]) == 0",
                "assert solution([-5, 5]) == 10",
                "assert solution([42]) == 0"
            ],
            reference_code="def solution(nums):\n    return max(nums) - min(nums)",
            description="Calculate range/max difference of list"
        ),
        CodingTask(
            task_id="factorial",
            prompt="<user_start>Write a Python function `solution(n)` that returns the factorial of integer n (n >= 0).<user_end><assistant_start>",
            test_cases=[
                "assert solution(0) == 1",
                "assert solution(1) == 1",
                "assert solution(4) == 24",
                "assert solution(5) == 120"
            ],
            reference_code="def solution(n):\n    res = 1\n    for i in range(2, n + 1):\n        res *= i\n    return res",
            description="Compute factorial of non-negative integer"
        ),
        CodingTask(
            task_id="reverse_words",
            prompt="<user_start>Write a Python function `solution(sentence)` that reverses the order of words in a space-separated string.<user_end><assistant_start>",
            test_cases=[
                "assert solution('hello world') == 'world hello'",
                "assert solution('one') == 'one'",
                "assert solution('deep learning systems') == 'systems learning deep'"
            ],
            reference_code="def solution(sentence):\n    return ' '.join(sentence.split()[::-1])",
            description="Reverse order of words in sentence"
        ),
    ]

    @classmethod
    def get_task(cls, idx: int) -> CodingTask:
        return cls.TASKS[idx % len(cls.TASKS)]

    @classmethod
    def get_all_tasks(cls) -> List[CodingTask]:
        return list(cls.TASKS)
