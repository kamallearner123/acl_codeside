#!/bin/bash

# PythonAnywhere Setup Script for ACL CodeSide
# Run this script in a PythonAnywhere Bash console after uploading your code

echo "🚀 Setting up ACL CodeSide on PythonAnywhere..."

# Check if we're in the right directory
if [ ! -f "manage.py" ]; then
    echo "❌ Error: manage.py not found. Please run this script from your project root directory."
    echo "   Expected: ~/acl_codeside/"
    exit 1
fi

# Install requirements
echo "📦 Installing Python requirements..."
pip install -r requirements.txt

# Run migrations
echo "🗄️ Setting up database..."
python manage.py migrate

# Collect static files
echo "📁 Collecting static files..."
python manage.py collectstatic --noinput

# Create superuser (optional)
echo "👤 Do you want to create a superuser? (y/n)"
read -r create_superuser
if [ "$create_superuser" = "y" ] || [ "$create_superuser" = "Y" ]; then
    python manage.py createsuperuser
fi

# Load sample questions (if available)
echo "📚 Loading sample questions..."
python manage.py load_sample_questions || echo "⚠️  Sample questions not loaded (command might not exist)"

# Check deployment readiness
echo "🔍 Checking deployment configuration..."
python manage.py check --deploy || echo "⚠️  Some deployment checks failed (review warnings)"

echo ""
echo "✅ Setup complete! Next steps:"
echo ""
echo "1. Configure your WSGI file in PythonAnywhere Web tab"
echo "2. Set your virtual environment path: $(pwd)/../.virtualenvs/acl_codeside"
echo "3. Configure static files mapping:"
echo "   URL: /static/ → Directory: $(pwd)/staticfiles/"
echo "   URL: /media/ → Directory: $(pwd)/media/"
echo "4. Set environment variables in Web tab:"
echo "   DJANGO_SECRET_KEY=your-secret-key"
echo "   ALLOWED_HOSTS=yourusername.pythonanywhere.com,www.aptcomputinglabs.com"
echo "5. Click 'Reload' in Web tab"
echo ""
echo "📖 See PYTHONANYWHERE_DEPLOYMENT.md for detailed instructions"
echo "🌐 Your app will be available at: https://yourusername.pythonanywhere.com"