"""
C++ executor service
Compiles and runs C++ code in a temporary directory and returns stdout/stderr.
"""
import subprocess
import tempfile
import os
import time
import shutil


class CPPExecutor:
    def __init__(self, timeout=10):
        self.timeout = timeout

    def execute(self, code, stdin_data=None):
        result = {
            'success': False,
            'stdout': '',
            'stderr': '',
            'execution_time': 0.0,
        }

        temp_dir = tempfile.mkdtemp(prefix='cpp_exec_')
        src_path = os.path.join(temp_dir, 'main.cpp')
        bin_path = os.path.join(temp_dir, 'a.out')

        try:
            with open(src_path, 'w', encoding='utf-8') as f:
                f.write(code)

            start = time.time()

            # Compile
            compile_proc = subprocess.Popen([
                'g++', '-std=c++17', src_path, '-O2', '-o', bin_path
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            comp_out, comp_err = compile_proc.communicate(timeout=30)

            if compile_proc.returncode != 0:
                result['stderr'] = comp_err or comp_out
                return result

            # Run
            run_proc = subprocess.Popen([bin_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=temp_dir)
            try:
                stdout, stderr = run_proc.communicate(input=stdin_data, timeout=self.timeout)
                result['stdout'] = stdout
                result['stderr'] = stderr
                result['success'] = run_proc.returncode == 0
                result['execution_time'] = time.time() - start
            except subprocess.TimeoutExpired:
                run_proc.kill()
                stdout, stderr = run_proc.communicate()
                result['stderr'] = f"Execution timed out after {self.timeout} seconds\n{stderr}"
                result['success'] = False

        except Exception as e:
            result['stderr'] = str(e)
            result['success'] = False
        finally:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

        return result
