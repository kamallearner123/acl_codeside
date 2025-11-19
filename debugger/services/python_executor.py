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
from django.conf import settings


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
        
        # Set temp directory - use project's media folder for PythonAnywhere compatibility
        if hasattr(settings, 'MEDIA_ROOT') and settings.MEDIA_ROOT:
            self.temp_base_dir = os.path.join(settings.MEDIA_ROOT, 'temp_exec')
            os.makedirs(self.temp_base_dir, exist_ok=True)
            # Clean up old temp directories (older than 1 hour)
            self._cleanup_old_temp_dirs()
        else:
            self.temp_base_dir = None  # Use system temp
    
    def _cleanup_old_temp_dirs(self):
        """Clean up temporary execution directories older than 1 hour"""
        try:
            if not self.temp_base_dir or not os.path.exists(self.temp_base_dir):
                return
            
            current_time = time.time()
            max_age = 3600  # 1 hour in seconds
            
            for item in os.listdir(self.temp_base_dir):
                item_path = os.path.join(self.temp_base_dir, item)
                if os.path.isdir(item_path) and item.startswith('exec_'):
                    try:
                        # Check directory age
                        dir_age = current_time - os.path.getmtime(item_path)
                        if dir_age > max_age:
                            # Remove old directory and its contents
                            import shutil
                            shutil.rmtree(item_path, ignore_errors=True)
                    except Exception:
                        pass
        except Exception:
            pass
    
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
        # Use custom temp directory if available (for PythonAnywhere compatibility)
        if self.temp_base_dir:
            temp_dir = os.path.join(self.temp_base_dir, f'exec_{int(time.time() * 1000)}')
            os.makedirs(temp_dir, exist_ok=True)
        else:
            temp_dir = tempfile.mkdtemp()
            
        plot_dir = os.path.join(temp_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Create matplotlib config directory to avoid configuration issues
        mpl_config_dir = os.path.join(temp_dir, 'mpl_config')
        os.makedirs(mpl_config_dir, exist_ok=True)
        
        # Inject plot saving code with proper matplotlib configuration
        plot_setup = f"""
import os
import sys

# Setup for matplotlib - must be done before importing matplotlib
try:
    # Set environment variables before importing matplotlib
    os.environ['MPLCONFIGDIR'] = r'{mpl_config_dir}'
    
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
        # Use a safe filename with proper extension
        temp_script_path = os.path.join(temp_dir, f'script_{int(time.time() * 1000)}.py')
        
        try:
            with open(temp_script_path, 'w', encoding='utf-8') as tmp_file:
                tmp_file.write(modified_code)
        except Exception as e:
            result['stderr'] = f"Error writing temporary file: {str(e)}"
            return result
        
        try:
            start_time = time.time()
            
            # Execute the Python code with proper environment
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['MPLCONFIGDIR'] = mpl_config_dir  # Set matplotlib config directory
            env['PYTHONDONTWRITEBYTECODE'] = '1'  # Prevent .pyc files
            # Don't override HOME as it may cause other issues
            
            process = subprocess.Popen(
                [self.python_executable, temp_script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=temp_dir,
                env=env
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
                if os.path.exists(temp_script_path):
                    os.unlink(temp_script_path)
                # Clean up plot directory
                if os.path.exists(plot_dir):
                    for plot_file in os.listdir(plot_dir):
                        try:
                            os.unlink(os.path.join(plot_dir, plot_file))
                        except:
                            pass
                    try:
                        os.rmdir(plot_dir)
                    except:
                        pass
                # Clean up matplotlib config directory
                if os.path.exists(mpl_config_dir):
                    try:
                        import shutil
                        shutil.rmtree(mpl_config_dir, ignore_errors=True)
                    except:
                        pass
                # Clean up temp directory
                if os.path.exists(temp_dir):
                    try:
                        os.rmdir(temp_dir)
                    except:
                        pass
            except Exception as cleanup_error:
                # Log cleanup errors but don't fail the execution
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
