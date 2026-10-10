import ast
import re
import time
import multiprocessing
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

@dataclass
class ExecutionResult:
    success: bool
    reward: float
    tests_passed: int
    total_tests: int
    error_message: Optional[str] = None
    execution_time: float = 0.0
    extracted_code: str = ""

def _isolated_worker(code_str: str, test_cases: List[str], conn):
    """
    Subprocess worker executing code in a sandboxed namespace.
    """
    safe_globals: Dict[str, Any] = {
        "__builtins__": {
            "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
            "enumerate": enumerate, "filter": filter, "float": float, "int": int,
            "isinstance": isinstance, "len": len, "list": list, "map": map,
            "max": max, "min": min, "range": range, "reversed": reversed,
            "round": round, "set": set, "sorted": sorted, "str": str,
            "sum": sum, "tuple": tuple, "zip": zip, "True": True, "False": False, "None": None,
        }
    }
    exec_scope: Dict[str, Any] = dict(safe_globals)
    
    try:
        # Compile and execute function definition in unified namespace
        compiled = compile(code_str, "<sandbox>", "exec")
        exec(compiled, exec_scope)
        
        passed = 0
        total = len(test_cases)
        err = None
        
        for tc in test_cases:
            try:
                exec(tc, exec_scope)
                passed += 1
            except AssertionError:
                err = f"Assertion failed: {tc}"
                break
            except Exception as e:
                err = f"Runtime error in test: {type(e).__name__}: {str(e)}"
                break
                
        conn.send({"passed": passed, "total": total, "err": err})
    except Exception as e:
        conn.send({"passed": 0, "total": len(test_cases), "err": f"Compile/exec error: {type(e).__name__}: {str(e)}"})

class SandboxedCodeExecutor:
    """
    Production-grade AST-checked sandboxed Python execution harness.
    """
    FORBIDDEN_MODULES = {"os", "sys", "subprocess", "socket", "shutil", "urllib", "requests", "importlib", "pathlib"}
    FORBIDDEN_CALLS = {"open", "eval", "exec", "__import__", "compile", "globals", "locals"}

    def __init__(self, timeout_sec: float = 1.5):
        self.timeout_sec = timeout_sec

    def extract_code(self, raw_output: str) -> Tuple[str, bool]:
        """
        Extracts code from tags: <code_start>...<code_end>, markdown ```..., or def solution.
        Returns: (extracted_code, format_valid)
        """
        format_valid = False
        code = ""
        
        # 1. Check for <code_start> tags
        code_tag_match = re.search(r"<code_start>(.*?)<code_end>", raw_output, re.DOTALL)
        if code_tag_match:
            code = code_tag_match.group(1).strip()
            format_valid = True
        else:
            # 2. Check for markdown fences
            md_match = re.search(r"```(?:python)?(.*?)```", raw_output, re.DOTALL)
            if md_match:
                code = md_match.group(1).strip()
            else:
                # 3. Check for def solution
                def_match = re.search(r"(def solution\s*\(.*)", raw_output, re.DOTALL)
                if def_match:
                    code = def_match.group(1).strip()
                else:
                    code = raw_output.strip()
                    
        return code, format_valid

    def check_ast_safety(self, code_str: str) -> Tuple[bool, Optional[str]]:
        """Static AST verification to reject unsafe operations."""
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            return False, f"SyntaxError: {str(e)}"
            
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in self.FORBIDDEN_MODULES:
                        return False, f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in self.FORBIDDEN_MODULES:
                    return False, f"Forbidden import from: {node.module}"
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.FORBIDDEN_CALLS:
                        return False, f"Forbidden call: {node.func.id}"
                        
        return True, None

    def execute_and_evaluate(
        self,
        raw_output: str,
        test_cases: List[str],
        correct_reward: float = 1.0,
        partial_weight: float = 0.5,
        format_bonus: float = 0.1,
        timeout_penalty: float = -0.5
    ) -> ExecutionResult:
        """
        Parses, checks, and executes candidate code against test cases with timeouts.
        """
        code_str, format_valid = self.extract_code(raw_output)
        bonus = format_bonus if format_valid else 0.0
        
        if not code_str:
            return ExecutionResult(
                success=False,
                reward=0.0,
                tests_passed=0,
                total_tests=len(test_cases),
                error_message="No code found in model output",
                extracted_code=""
            )
            
        # Static AST check
        is_safe, error_msg = self.check_ast_safety(code_str)
        if not is_safe:
            return ExecutionResult(
                success=False,
                reward=timeout_penalty,
                tests_passed=0,
                total_tests=len(test_cases),
                error_message=f"Safety violation: {error_msg}",
                extracted_code=code_str
            )

        # Isolated Process Execution
        parent_conn, child_conn = multiprocessing.Pipe()
        p = multiprocessing.Process(
            target=_isolated_worker,
            args=(code_str, test_cases, child_conn)
        )
        
        t0 = time.perf_counter()
        p.start()
        p.join(timeout=self.timeout_sec)
        
        if p.is_alive():
            p.terminate()
            p.join(timeout=0.2)
            if p.is_alive():
                p.kill()
            return ExecutionResult(
                success=False,
                reward=timeout_penalty,
                tests_passed=0,
                total_tests=len(test_cases),
                error_message=f"Execution timed out (> {self.timeout_sec}s)",
                execution_time=self.timeout_sec,
                extracted_code=code_str
            )
            
        elapsed = time.perf_counter() - t0
        
        if not parent_conn.poll():
            return ExecutionResult(
                success=False,
                reward=0.0,
                tests_passed=0,
                total_tests=len(test_cases),
                error_message="Worker terminated abnormally",
                execution_time=elapsed,
                extracted_code=code_str
            )
            
        res = parent_conn.recv()
        passed = res["passed"]
        total = res["total"]
        err = res["err"]
        
        if passed == total and total > 0:
            reward = correct_reward + bonus
            success = True
        elif passed > 0:
            reward = (partial_weight * (passed / total)) + bonus
            success = False
        else:
            reward = bonus if (bonus > 0 and err is None) else 0.0
            success = False
            
        return ExecutionResult(
            success=success,
            reward=reward,
            tests_passed=passed,
            total_tests=total,
            error_message=err,
            execution_time=elapsed,
            extracted_code=code_str
        )
