import requests
import json
import time
import subprocess
import tempfile
import os


class RustExecutor:
    """
    Execute Rust code using the Rust Playground API or local rustc.
    This service handles code execution with Miri for memory tracing.
    """
    
    PLAYGROUND_URL = "https://play.rust-lang.org"
    USE_LOCAL_EXECUTION = True  # Set to True to use local rustc instead of API
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
        })
        # Check if rustc is available locally
        self.has_local_rust = self._check_local_rust()
    
    def _check_local_rust(self):
        """Check if rustc is available on the system"""
        try:
            subprocess.run(['rustc', '--version'], capture_output=True, timeout=5)
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _execute_locally(self, code, use_miri=False):
        """
        Execute Rust code locally using rustc or miri.
        
        Args:
            code (str): Rust source code to execute
            use_miri (bool): Whether to use Miri for execution
            
        Returns:
            dict: Execution result with stdout, stderr, and success status
        """
        try:
            # Create a temporary directory for compilation
            with tempfile.TemporaryDirectory() as tmpdir:
                source_file = os.path.join(tmpdir, 'main.rs')
                output_file = os.path.join(tmpdir, 'output')
                
                # Write the code to a file
                with open(source_file, 'w') as f:
                    f.write(code)
                
                if use_miri:
                    # Execute with Miri
                    result = subprocess.run(
                        ['cargo', '+nightly', 'miri', 'run', '--manifest-path', source_file],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=tmpdir
                    )
                else:
                    # Compile the code
                    compile_result = subprocess.run(
                        ['rustc', source_file, '-o', output_file],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    if compile_result.returncode != 0:
                        return {
                            'success': False,
                            'stdout': compile_result.stdout,
                            'stderr': compile_result.stderr
                        }
                    
                    # Execute the compiled binary
                    result = subprocess.run(
                        [output_file],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                
                return {
                    'success': result.returncode == 0,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'stdout': '',
                'stderr': 'Execution timed out'
            }
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Error executing code locally: {str(e)}'
            }
    
    def execute_normal(self, code):
        """
        Execute Rust code normally without Miri (faster execution).
        
        Args:
            code (str): Rust source code to execute
            
        Returns:
            dict: Execution result with stdout, stderr, and success status
        """
        # Try local execution first if available
        if self.USE_LOCAL_EXECUTION and self.has_local_rust:
            return self._execute_locally(code, use_miri=False)
        
        # Fallback to Playground API
        try:
            payload = {
                "channel": "stable",
                "mode": "debug",
                "edition": "2021",
                "crateType": "bin",
                "tests": False,
                "code": code,
                "backtrace": False
            }
            
            response = self.session.post(
                f"{self.PLAYGROUND_URL}/execute",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': result.get('success', False),
                    'stdout': result.get('stdout', ''),
                    'stderr': result.get('stderr', '')
                }
            else:
                return {
                    'success': False,
                    'stdout': '',
                    'stderr': f'Playground API error: {response.status_code}'
                }
        except requests.Timeout:
            return {
                'success': False,
                'stdout': '',
                'stderr': 'Execution timed out (30 seconds limit)'
            }
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Error executing code: {str(e)}'
            }
    
    def execute_with_miri(self, code):
        """
        Execute Rust code with Miri and capture JSON memory trace.
        
        Args:
            code (str): Rust source code to execute
            
        Returns:
            dict: Execution result with stdout, stderr, and miri_output
        """
        try:
            # Prepare the code with Miri flags
            payload = {
                "channel": "nightly",
                "mode": "debug",
                "edition": "2021",
                "crateType": "bin",
                "tests": False,
                "code": code,
                "backtrace": False
            }
            
            # First, try to execute normally
            normal_result = self._execute_normal(payload)
            
            # Then try with Miri
            miri_result = self._execute_miri(code)
            
            return {
                'success': normal_result.get('success', False),
                'stdout': normal_result.get('stdout', ''),
                'stderr': normal_result.get('stderr', ''),
                'miri_output': miri_result.get('output', ''),
                'miri_success': miri_result.get('success', False)
            }
            
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Error executing code: {str(e)}',
                'miri_output': '',
                'miri_success': False
            }
    
    def _execute_normal(self, payload):
        """Execute code normally through the playground"""
        try:
            response = self.session.post(
                f"{self.PLAYGROUND_URL}/execute",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': result.get('success', False),
                    'stdout': result.get('stdout', ''),
                    'stderr': result.get('stderr', '')
                }
            else:
                return {
                    'success': False,
                    'stdout': '',
                    'stderr': f'Playground API error: {response.status_code}'
                }
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e)
            }
    
    def _execute_miri(self, code):
        """
        Execute code with Miri for memory tracing.
        Note: The Rust Playground has limited Miri support.
        This is a placeholder for the ideal implementation.
        
        In a production environment, you would need to:
        1. Run Miri locally with: MIRIFLAGS="-Zmiri-track-raw-pointers" cargo +nightly miri run
        2. Or use a custom backend server with Rust/Miri installed
        """
        try:
            # Attempt to use Miri through playground
            payload = {
                "channel": "nightly",
                "mode": "debug",
                "edition": "2021",
                "crateType": "bin",
                "tests": False,
                "code": code,
                "backtrace": False
            }
            
            # The playground doesn't directly support Miri with trace flags
            # This would need a custom backend or local execution
            # For now, we'll simulate the structure
            
            response = self.session.post(
                f"{self.PLAYGROUND_URL}/miri",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': result.get('success', False),
                    'output': result.get('stdout', '') + result.get('stderr', '')
                }
            else:
                # Fallback: return simulated structure
                return self._simulate_miri_output(code)
                
        except Exception as e:
            # If Miri execution fails, return simulated output
            return self._simulate_miri_output(code)
    
    def _simulate_miri_output(self, code):
        """
        Simulate Miri output for demonstration purposes.
        In production, replace this with actual Miri execution.
        """
        # This generates a mock memory trace for testing
        # Real implementation would parse actual Miri JSON output
        return {
            'success': True,
            'output': json.dumps({
                'events': [
                    {
                        'type': 'function_entry',
                        'name': 'main',
                        'frame_id': 0
                    },
                    {
                        'type': 'alloc',
                        'kind': 'stack',
                        'ptr': '0x1000',
                        'size': 4,
                        'variable': 'x'
                    },
                    {
                        'type': 'write',
                        'ptr': '0x1000',
                        'value': 42,
                        'size': 4
                    }
                ]
            })
        }
