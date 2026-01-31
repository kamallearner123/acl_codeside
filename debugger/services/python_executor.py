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
        # Debug mode: set environment variable PY_EXEC_DEBUG=1 or Django setting PY_EXEC_DEBUG=True
        debug_mode = False
        try:
            # FORCE DEBUG MODE True for troubleshooting
            debug_mode = True 
            # debug_mode = os.environ.get('PY_EXEC_DEBUG', '0') == '1' or bool(getattr(settings, 'PY_EXEC_DEBUG', False))
        except Exception:
            debug_mode = True
        
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
        
        # Create a minimal matplotlibrc file to prevent loading defaults
        matplotlibrc_path = os.path.join(mpl_config_dir, 'matplotlibrc')
        with open(matplotlibrc_path, 'w') as f:
            f.write('backend: Agg\n')
            f.write('interactive: False\n')
            
        # Fix permissions for directories and config files
        try:
            os.chmod(temp_dir, 0o755)
            os.chmod(mpl_config_dir, 0o755)
            os.chmod(matplotlibrc_path, 0o644)
        except Exception as e:
            print(f"[DEBUG] Warning: Failed to set permissions for config dirs: {e}")
        
        # Inject plot saving code with proper matplotlib configuration
        plot_setup = f"""
import os
import sys

# Setup for matplotlib - must be done before importing matplotlib
try:
    # Completely disable matplotlib configuration loading
    os.environ['MPLCONFIGDIR'] = r'{mpl_config_dir}'
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['MATPLOTLIBRC'] = r'{mpl_config_dir}'
    
    # Disable all matplotlib configuration loading
    # Disable all matplotlib configuration loading
    os.environ['HOME'] = r'{temp_dir}'  # Prevent loading from home directory
    
    # Create a minimal matplotlibrc file to prevent loading defaults
    matplotlibrc_path = os.path.join(r'{mpl_config_dir}', 'matplotlibrc')
    with open(matplotlibrc_path, 'w') as f:
        f.write('backend: Agg\\n')
        f.write('interactive: False\\n')
        f.write('figure.figsize: 8, 6\\n')
        f.write('font.size: 10\\n')
    
    # Import matplotlib with forced backend
    import matplotlib
    matplotlib.use('Agg', force=True)
    
    # Set matplotlib to use our custom config directory
    import matplotlib.pyplot as plt
    
    # Disable interactive mode and any configuration loading
    plt.ioff()
    plt.rcParams['backend'] = 'Agg'
    
    # Disable any further configuration loading
    plt.rcParams['savefig.directory'] = r'{plot_dir}'
    
    # Store original show function
    _original_show = plt.show
    _plot_counter = [0]
    
    def _custom_show():
        # Save the current figure
        _plot_counter[0] += 1
        plot_path = os.path.join(r'{plot_dir}', f'plot_{{_plot_counter[0]}}.png')
        try:
            plt.savefig(plot_path, dpi=100, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Warning: Could not save plot: {{e}}", file=sys.stderr)
    
    # Replace plt.show with our custom function
    plt.show = _custom_show
except ImportError:
    pass
except Exception as e:
    print(f"Warning: Matplotlib setup failed: {{e}}", file=sys.stderr)

"""
        
        modified_code = plot_setup + "\n" + code
        
        # Create a temporary file for the Python code
        # Use a safe filename with proper extension
        temp_script_path = os.path.join(temp_dir, f'script_{int(time.time() * 1000)}.py')

        import traceback
        # Ensure the parent directory exists before writing the script file
        script_dir = os.path.dirname(temp_script_path)
        print(f"[DEBUG] Attempting to create script directory: {script_dir}")
        try:
            os.makedirs(script_dir, exist_ok=True)
            print(f"[DEBUG] Script directory exists or created: {script_dir}")
        except Exception as e:
            tb = traceback.format_exc()
            result['stderr'] = f"Error creating script directory {script_dir}: {str(e)}\n{tb}"
            print(f"[DEBUG] Error creating script directory {script_dir}: {e}\n{tb}")
            return result

        print(f"[DEBUG] Attempting to write script file: {temp_script_path}")
        try:
            with open(temp_script_path, 'w', encoding='utf-8') as tmp_file:
                tmp_file.write(modified_code)
            print(f"[DEBUG] Successfully wrote script file: {temp_script_path}")
            
            # Fix permissions for execution as requested
            try:
                os.chmod(temp_script_path, 0o755)
            except Exception as perm_error:
                print(f"[DEBUG] Warning: Could not change permissions: {perm_error}")
                
        except Exception as e:
            tb = traceback.format_exc()
            result['stderr'] = f"Error writing temporary file {temp_script_path}: {str(e)}\n{tb}"
            print(f"[DEBUG] Error writing script file {temp_script_path}: {e}\n{tb}")
            return result
        
        try:
            start_time = time.time()
            
            # Execute the Python code with proper environment
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['MPLCONFIGDIR'] = mpl_config_dir  # Set matplotlib config directory
            env['MPLBACKEND'] = 'Agg'  # Force Agg backend
            env['MATPLOTLIBRC'] = mpl_config_dir  # Point to our custom matplotlibrc
            env['MATPLOTLIBRC'] = mpl_config_dir  # Point to our custom matplotlibrc
            env['PYTHONDONTWRITEBYTECODE'] = '1'  # Prevent .pyc files
            env['HOME'] = temp_dir  # Set HOME to temp directory to avoid config issues
            # Set XDG directories to temp to avoid permission issues
            env['XDG_CACHE_HOME'] = os.path.join(temp_dir, '.cache')
            env['XDG_CONFIG_HOME'] = os.path.join(temp_dir, '.config')
            
            # Create XDG directories
            os.makedirs(env['XDG_CACHE_HOME'], exist_ok=True)
            os.makedirs(env['XDG_CONFIG_HOME'], exist_ok=True)
            
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
                import shutil
                # If debug_mode is enabled, preserve the temp_dir and write debug info
                if debug_mode:
                    try:
                        debug_info_path = os.path.join(temp_dir, 'debug_info.txt')
                        with open(debug_info_path, 'w', encoding='utf-8') as dbg:
                            dbg.write(f"temp_dir: {temp_dir}\n")
                            dbg.write(f"temp_script: {temp_script_path}\n")
                            dbg.write(f"plot_dir: {plot_dir}\n")
                            dbg.write(f"mpl_config_dir: {mpl_config_dir}\n")
                            dbg.write('\n--- ENVIRONMENT ---\n')
                            for k in sorted(env.keys()):
                                dbg.write(f"{k}={env[k]}\n")
                        # Also write a small listing of the temp directory
                        try:
                            with open(os.path.join(temp_dir, 'ls_temp_dir.txt'), 'w', encoding='utf-8') as lsout:
                                for root, dirs, files in os.walk(temp_dir):
                                    lsout.write(f"ROOT: {root}\n")
                                    for d in dirs:
                                        lsout.write(f"DIR: {os.path.join(root,d)}\n")
                                    for f in files:
                                        lsout.write(f"FILE: {os.path.join(root,f)}\n")
                        except Exception:
                            pass
                        # Expose debug path in result for quick inspection
                        result['debug_dir'] = temp_dir
                    except Exception:
                        pass
                    # Do not remove anything when debugging
                else:
                    if os.path.exists(temp_script_path):
                        os.unlink(temp_script_path)

                    # Clean up plot directory
                    if os.path.exists(plot_dir):
                        shutil.rmtree(plot_dir, ignore_errors=True)

                    # Clean up matplotlib config directory
                    if os.path.exists(mpl_config_dir):
                        shutil.rmtree(mpl_config_dir, ignore_errors=True)

                    # Clean up XDG directories
                    cache_dir = os.path.join(temp_dir, '.cache')
                    config_dir = os.path.join(temp_dir, '.config')
                    if os.path.exists(cache_dir):
                        shutil.rmtree(cache_dir, ignore_errors=True)
                    if os.path.exists(config_dir):
                        shutil.rmtree(config_dir, ignore_errors=True)

                    # Clean up temp directory
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    
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
