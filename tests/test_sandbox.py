import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandbox.code_sandbox import SandboxedCodeExecutor

def test_sandbox():
    executor = SandboxedCodeExecutor(timeout_sec=3.5)

    # 1. Valid code
    out1 = '<thought_start>sum evens<thought_end><code_start>def solution(nums):\n    return sum(x for x in nums if x % 2 == 0)<code_end>'
    r1 = executor.execute_and_evaluate(out1, ['assert solution([1,2,3,4]) == 6', 'assert solution([]) == 0'])
    print(f"Test 1 (Valid): success={r1.success}, reward={r1.reward:.2f}, passed={r1.tests_passed}/{r1.total_tests}")
    assert r1.success and r1.reward >= 1.0

    # 2. Partial pass
    out2 = 'def solution(nums):\n    return 6'
    r2 = executor.execute_and_evaluate(out2, ['assert solution([1,2,3,4]) == 6', 'assert solution([]) == 0'])
    print(f"Test 2 (Partial): success={r2.success}, reward={r2.reward:.2f}, passed={r2.tests_passed}/{r2.total_tests}")
    assert not r2.success and r2.reward > 0.0

    # 3. Forbidden code
    out3 = 'import os\nos.system("dir")'
    r3 = executor.execute_and_evaluate(out3, ['assert True'])
    print(f"Test 3 (Security): success={r3.success}, error={r3.error_message}")
    assert not r3.success and 'Forbidden' in r3.error_message

    # 4. Infinite loop timeout
    out4 = 'def solution(nums):\n    while True:\n        pass'
    r4 = executor.execute_and_evaluate(out4, ['assert solution([]) == 0'])
    print(f"Test 4 (Timeout): success={r4.success}, error={r4.error_message}")
    assert not r4.success and 'timed out' in r4.error_message

    print("ALL SANDBOX TESTS PASSED!")

if __name__ == "__main__":
    test_sandbox()
