# PythonAnywhere Python Executor Fix

## Problem

When running Python code execution on PythonAnywhere, you may encounter this error:
```
unable to load configuration from /tmp/tmpr_soaxyf/tmpufnyv5e0.py
```

## Root Cause

PythonAnywhere has restrictions on the `/tmp` directory for security reasons. The standard Python `tempfile.mkdtemp()` and `tempfile.NamedTemporaryFile()` functions create files in `/tmp`, which causes permission and configuration loading issues on PythonAnywhere's environment.

## Solution

The fix modifies the `PythonExecutor` class in `debugger/services/python_executor.py` to:

1. **Use Media Directory Instead of /tmp**: Create temporary execution directories in `media/temp_exec/` instead of the system `/tmp` directory
2. **Explicit File Management**: Use explicit file paths with proper naming instead of Python's NamedTemporaryFile
3. **Automatic Cleanup**: Implement automatic cleanup of old temporary directories (older than 1 hour)
4. **Better Error Handling**: Add proper exception handling for file operations

## Changes Made

### 1. Modified Python Executor (`debugger/services/python_executor.py`)

**Key Changes:**
- Added `self.temp_base_dir` to use `media/temp_exec/` for temporary files
- Replaced `tempfile.mkdtemp()` with custom directory creation in media folder
- Replaced `tempfile.NamedTemporaryFile()` with explicit file writing using `open()`
- Added `_cleanup_old_temp_dirs()` method to prevent disk space issues
- Added UTF-8 encoding and environment variables for better compatibility

### 2. Updated .gitignore

Added exclusion for temporary execution directory:
```
media/temp_exec/
```

### 3. Updated Deployment Documentation

Added setup instructions for creating the required directory on PythonAnywhere.

## Deployment Steps on PythonAnywhere

After pulling the latest code, run these commands in PythonAnywhere bash console:

```bash
# Navigate to project
cd ~/acl_codeside

# Create temporary execution directory
mkdir -p media/temp_exec

# Set proper permissions
chmod 755 media
chmod 755 media/temp_exec

# Reload web app
# (Use the reload button in PythonAnywhere Web tab)
```

## Testing

To verify the fix works, run this in PythonAnywhere console:

```bash
workon acl_codeside
cd ~/acl_codeside
python manage.py shell
```

Then in the Python shell:
```python
from debugger.services.python_executor import PythonExecutor

# Test basic execution
executor = PythonExecutor()
result = executor.execute("print('Hello from PythonAnywhere!')")
print(f"Success: {result['success']}")
print(f"Output: {result['stdout']}")

# Test with matplotlib
code = """
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title('Sine Wave')
plt.show()

print('Plot created successfully!')
"""
result = executor.execute(code)
print(f"Success: {result['success']}")
print(f"Output: {result['stdout']}")
print(f"Plots: {len(result['plots'])} plot(s) generated")
```

## Benefits

1. **PythonAnywhere Compatible**: Works within PythonAnywhere's security restrictions
2. **Automatic Cleanup**: Prevents disk space issues by cleaning old temp files
3. **Better Error Handling**: More informative error messages
4. **Maintains Functionality**: All features (stdout, stderr, plots) work as before
5. **No Breaking Changes**: Compatible with existing code and local development

## Local Development

The fix automatically detects the environment:
- **On PythonAnywhere**: Uses `media/temp_exec/`
- **Locally (if MEDIA_ROOT not set)**: Falls back to system `/tmp`

No configuration changes needed for local development!

## Maintenance

The automatic cleanup runs every time a PythonExecutor instance is created, removing temporary directories older than 1 hour. This prevents disk space accumulation without requiring manual maintenance.

## Security Notes

- Temporary files are created in the Django project's media directory
- Each execution gets a unique timestamped directory
- Old files are automatically cleaned up
- No changes to code execution sandboxing or timeout mechanisms

## Support

If you encounter any issues:

1. Check PythonAnywhere error logs in Web tab
2. Verify `media/temp_exec/` directory exists and has proper permissions
3. Test with the commands provided in the Testing section above
4. Check that your virtual environment is activated: `workon acl_codeside`

## Version

- **Fixed in**: November 2025
- **Tested on**: PythonAnywhere Free and Paid tiers
- **Django Version**: 5.2.5
- **Python Version**: 3.10+
