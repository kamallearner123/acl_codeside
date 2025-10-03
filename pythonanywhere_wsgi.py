"""
WSGI config for leetcode_clone project on PythonAnywhere.

This WSGI file should be uploaded to your PythonAnywhere account and configured
in the Web tab of your PythonAnywhere dashboard.

Steps to configure on PythonAnywhere:
1. Upload your project files to PythonAnywhere (using git or file upload)
2. Create a virtual environment: mkvirtualenv --python=/usr/bin/python3.10 acl_codeside
3. Install requirements: pip install -r requirements.txt
4. Update the paths below to match your PythonAnywhere directory structure
5. Copy this file content to your WSGI configuration file in PythonAnywhere Web tab
6. Set environment variables in PythonAnywhere Web tab
"""

import os
import sys

# Add your project directory to Python path
# IMPORTANT: Update this path to match your PythonAnywhere directory structure
# Replace 'yourusername' with your actual PythonAnywhere username
path = '/home/kamal123/acl_codeside'
if path not in sys.path:
    sys.path.insert(0, path)

# Set environment variable for Django settings
# Use production settings for PythonAnywhere
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'leetcode_clone.settings_prod')

# Set production environment variables if not already set
if 'DJANGO_SECRET_KEY' not in os.environ:
    # IMPORTANT: Replace this with your actual secret key in PythonAnywhere environment variables
    os.environ['DJANGO_SECRET_KEY'] = 'your-production-secret-key-here'

if 'ALLOWED_HOSTS' not in os.environ:
    # IMPORTANT: Replace with your actual PythonAnywhere domain
    os.environ['ALLOWED_HOSTS'] = 'yourusername.pythonanywhere.com,www.aptcomputinglabs.com'

# Import Django WSGI application
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()