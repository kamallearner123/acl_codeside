# PythonAnywhere Deployment Guide for ACL CodeSide

This guide will help you deploy your Django ACL CodeSide application to PythonAnywhere.

## Prerequisites

1. A PythonAnywhere account (free or paid)
2. Your Django project files
3. Basic familiarity with the PythonAnywhere dashboard

## Step-by-Step Deployment

### 1. Upload Your Code to PythonAnywhere

**Option A: Using Git (Recommended)**
```bash
# In PythonAnywhere bash console
cd ~
git clone https://github.com/kamallearner123/acl_codeside.git
cd acl_codeside
```

**Option B: Using File Upload**
- Zip your project files
- Upload via PythonAnywhere Files tab
- Extract in your home directory

### 2. Create Required Directories

```bash
# Create media directory for file uploads and temp execution
cd ~/acl_codeside
mkdir -p media/temp_exec
chmod 755 media
chmod 755 media/temp_exec
```

### 3. Create a Virtual Environment

In a PythonAnywhere Bash console:
```bash
# Create virtual environment with Python 3.10
mkvirtualenv --python=/usr/bin/python3.10 acl_codeside

# Activate the environment (should auto-activate)
workon acl_codeside

# Navigate to your project
cd ~/acl_codeside

# Install requirements
pip install -r requirements.txt
```

### 3. Configure Database

```bash
# Run migrations
python manage.py migrate

# Create superuser (optional)
python manage.py createsuperuser

# Load sample data (if available)
python manage.py load_sample_questions
```

### 4. Collect Static Files

```bash
# Collect static files for production
python manage.py collectstatic --noinput
```

### 5. Configure Web App in PythonAnywhere

1. Go to PythonAnywhere Dashboard → Web tab
2. Click "Add a new web app"
3. Choose "Manual configuration" 
4. Select Python 3.10
5. Click "Next"

### 6. Configure WSGI File

1. In Web tab, click on WSGI configuration file link
2. Replace the content with this configuration:

```python
import os
import sys

# Add your project directory to Python path
# Replace 'yourusername' with your actual PythonAnywhere username
path = '/home/yourusername/acl_codeside'
if path not in sys.path:
    sys.path.insert(0, path)

# Set environment variable for Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'leetcode_clone.settings_prod')

# Set production environment variables
os.environ['DJANGO_SECRET_KEY'] = 'your-super-secret-key-here'
os.environ['ALLOWED_HOSTS'] = 'yourusername.pythonanywhere.com,www.aptcomputinglabs.com'

# Import Django WSGI application
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
```

### 7. Configure Virtual Environment

1. In Web tab, find "Virtualenv" section
2. Enter: `/home/yourusername/.virtualenvs/acl_codeside`

### 8. Configure Static Files

1. In Web tab, find "Static files" section
2. Add these mappings:

| URL | Directory |
|-----|-----------|
| `/static/` | `/home/yourusername/acl_codeside/staticfiles/` |
| `/media/` | `/home/yourusername/acl_codeside/media/` |

### 9. Set Environment Variables

In Web tab, find "Environment variables" section and add:

| Name | Value |
|------|-------|
| `DJANGO_SECRET_KEY` | Your secret key (generate a new one for production) |
| `ALLOWED_HOSTS` | `yourusername.pythonanywhere.com,www.aptcomputinglabs.com` |
| `DATABASE_URL` | (optional, defaults to SQLite) |

### 10. Important Configuration Updates

Before deployment, update these files:

**Replace placeholders in WSGI file:**
- Replace `yourusername` with your actual PythonAnywhere username
- Replace `your-super-secret-key-here` with a new Django secret key

**Generate a new secret key:**
```python
# In PythonAnywhere console
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

### 11. Test and Deploy

1. Click "Reload" button in Web tab
2. Visit your app at `https://yourusername.pythonanywhere.com`
3. Check error logs if issues occur (in Web tab → Error log)

## Troubleshooting

### Common Issues:

1. **Import Errors**: Make sure virtual environment path is correct
2. **Static Files Not Loading**: Check static files mapping and run `collectstatic`
3. **Database Errors**: Ensure migrations are run and database file permissions are correct
4. **Secret Key Errors**: Make sure `DJANGO_SECRET_KEY` environment variable is set
5. **Python Executor Error** (`unable to load configuration from /tmp/...`): 
   - This is fixed in the latest version by using media/temp_exec instead of /tmp
   - Make sure the `media/temp_exec` directory exists and has proper permissions:
     ```bash
     cd ~/acl_codeside
     mkdir -p media/temp_exec
     chmod 755 media/temp_exec
     ```

### Debug Steps:

1. Check error logs in Web tab
2. Test in PythonAnywhere console:
   ```bash
   workon acl_codeside
   cd ~/acl_codeside
   python manage.py check --deploy
   ```
3. Test Python code execution:
   ```bash
   workon acl_codeside
   cd ~/acl_codeside
   python manage.py shell
   ```
   Then run:
   ```python
   from debugger.services.python_executor import PythonExecutor
   executor = PythonExecutor()
   result = executor.execute("print('Hello World')")
   print(result)
   ```

### File Permissions on PythonAnywhere:

Make sure these directories have proper permissions:
```bash
chmod 755 ~/acl_codeside/media
chmod 755 ~/acl_codeside/media/temp_exec
chmod 644 ~/acl_codeside/db.sqlite3  # If using SQLite
```

### Performance Tips:

1. Use MySQL database for better performance (upgrade to paid plan)
2. Enable compression for static files
3. Set up proper caching if needed

## Security Checklist

- [ ] New secret key generated for production
- [ ] DEBUG = False in production settings
- [ ] ALLOWED_HOSTS properly configured
- [ ] SSL/HTTPS enabled (automatic on PythonAnywhere)
- [ ] Environment variables used for sensitive data

## Features of Your ACL CodeSide App

Your deployed app includes:
- User authentication and profiles
- Course management
- Coding questions and challenges
- Submission tracking
- Developer tools
- Contact functionality
- Clean Bootstrap 4 UI

## Support

- PythonAnywhere Help: https://help.pythonanywhere.com/
- Django Documentation: https://docs.djangoproject.com/
- Your app repository: https://github.com/kamallearner123/acl_codeside