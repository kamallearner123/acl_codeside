from .settings import *
import os

# Production overrides
DEBUG = False

# Require SECRET_KEY from env
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'django-insecure-##*^*1v*swr1k^#!0z#h#d^snnm(&pat%nbk)%o^@effwwb5+=')

# PythonAnywhere friendly ALLOWED_HOSTS
ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', 'localhost').split(',')

# Database: prefer DATABASE_URL or default to sqlite for quick deploy
import dj_database_url
DATABASES = {
    'default': dj_database_url.config(
        default=os.environ.get('DATABASE_URL', 'sqlite:///db.sqlite3')
    )
}

# Static files configuration for PythonAnywhere
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Media files configuration
MEDIA_ROOT = BASE_DIR / 'media'

# Use WhiteNoise to serve static files (helpful for development, PythonAnywhere handles this)
MIDDLEWARE = [
    'whitenoise.middleware.WhiteNoiseMiddleware',
] + MIDDLEWARE

# Update STORAGES for production with WhiteNoise
# Use CompressedStaticFilesStorage (no manifest) instead of CompressedManifestStaticFilesStorage
STORAGES = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
    },
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedStaticFilesStorage",
    },
}

# Security settings (adjust based on your needs)
# For PythonAnywhere free accounts, you might need to adjust these
SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'True') == 'True'
CSRF_COOKIE_SECURE = os.environ.get('CSRF_COOKIE_SECURE', 'True') == 'True'
SECURE_HSTS_SECONDS = int(os.environ.get('SECURE_HSTS_SECONDS', '0'))
SECURE_SSL_REDIRECT = os.environ.get('SECURE_SSL_REDIRECT', 'False') == 'True'

# Email configuration for production
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = os.environ.get('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.environ.get('EMAIL_PORT', '587'))
EMAIL_USE_TLS = os.environ.get('EMAIL_USE_TLS', 'True') == 'True'
EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER', 'your-email@gmail.com')
EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD', 'your-password')
DEFAULT_FROM_EMAIL = os.environ.get('DEFAULT_FROM_EMAIL', 'info@aptcomputinglabs.com')
