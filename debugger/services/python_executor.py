"""
Python code executor service
Handles execution of Python code in a sandboxed environment
"""

import subprocess
import tempfile
import os
import sys
import time
import base64
import re


class PythonExecutor:
    """Execute Python code and capture output"""
    
    def __init__(self, timeout=30):
        """
        Initialize the Python executor
        
        Args:
            timeout: Maximum execution time in seconds
        """
        self.timeout = timeout
        # Use the same Python interpreter that's running Django
        self.python_executable = sys.executable
    
    def execute(self, code):
        """
        Execute Python code and return results
        
        Args:
            code: Python source code to execute
            
        Returns:
            dict: Execution results with keys:
                - success: bool
                - stdout: str
                - stderr: str
                - execution_time: float
                - plots: list of base64 encoded images
        """
        result = {
            'success': False,
            'stdout': '',
            'stderr': '',
            'execution_time': 0.0,
            'plots': []
        }
        
        # Validate code first
        validation = self.validate_syntax(code)
        if not validation['valid']:
            result['stderr'] = validation['error']
            return result
        
        # Create temporary directory for plots
        temp_dir = tempfile.mkdtemp()
        plot_dir = os.path.join(temp_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Inject plot saving code
        plot_setup = f"""
import os
import sys

# Setup for matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    # Store original show function
    _original_show = plt.show
    _plot_counter = [0]
    
    def _custom_show():
        # Save the current figure
        _plot_counter[0] += 1
        plot_path = os.path.join(r'{plot_dir}', f'plot_{{_plot_counter[0]}}.png')
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    # Replace plt.show with our custom function
    plt.show = _custom_show
except ImportError:
    pass

"""
        
        modified_code = plot_setup + "\n" + code
        
        # Create a temporary file for the Python code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir=temp_dir) as tmp_file:
            tmp_file.write(modified_code)
            tmp_file_path = tmp_file.name
        
        try:
            start_time = time.time()
            
            # Execute the Python code
            process = subprocess.Popen(
                [self.python_executable, tmp_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=temp_dir
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                execution_time = time.time() - start_time
                
                result['stdout'] = stdout
                result['stderr'] = stderr
                result['execution_time'] = execution_time
                result['success'] = process.returncode == 0
                
                # Collect plot images
                if os.path.exists(plot_dir):
                    plot_files = sorted([f for f in os.listdir(plot_dir) if f.endswith('.png')])
                    for plot_file in plot_files:
                        plot_path = os.path.join(plot_dir, plot_file)
                        try:
                            with open(plot_path, 'rb') as img_file:
                                img_data = base64.b64encode(img_file.read()).decode('utf-8')
                                result['plots'].append({
                                    'name': plot_file,
                                    'data': img_data
                                })
                        except Exception as e:
                            pass
                
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                result['stderr'] = f"Execution timed out after {self.timeout} seconds\n{stderr}"
                result['success'] = False
                
        except Exception as e:
            result['stderr'] = f"Execution error: {str(e)}"
            result['success'] = False
            
        finally:
            # Clean up temporary files
            try:
                os.unlink(tmp_file_path)
                # Clean up plot directory
                if os.path.exists(plot_dir):
                    for plot_file in os.listdir(plot_dir):
                        os.unlink(os.path.join(plot_dir, plot_file))
                    os.rmdir(plot_dir)
                os.rmdir(temp_dir)
            except:
                pass
        
        return result
    
    def validate_syntax(self, code):
        """
        Validate Python code syntax without executing
        
        Args:
            code: Python source code to validate
            
        Returns:
            dict: Validation results with keys:
                - valid: bool
                - error: str (if not valid)
        """
        try:
            compile(code, '<string>', 'exec')
            return {'valid': True, 'error': None}
        except SyntaxError as e:
            return {
                'valid': False,
                'error': f"Syntax Error on line {e.lineno}: {e.msg}"
            }
        except Exception as e:
            return {
                'valid': False,
                'error': f"Validation Error: {str(e)}"
            }
