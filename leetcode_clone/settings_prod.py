from .settings import *
import os

# Production overrides
DEBUG = False

# Require SECRET_KEY from env
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY') or 'replace-me'

ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', 'localhost').split(',')

# Database: prefer DATABASE_URL or default to sqlite for quick deploy
import dj_database_url
DATABASES = {
    'default': dj_database_url.config(default=os.environ.get('DATABASE_URL', 'sqlite:///db.sqlite3'))
}

# Static files (served by WhiteNoise in the container)
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Security settings
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_HSTS_SECONDS = int(os.environ.get('SECURE_HSTS_SECONDS', '3600'))
SECURE_SSL_REDIRECT = os.environ.get('SECURE_SSL_REDIRECT', 'True') == 'True'

# Use WhiteNoise to serve static files
MIDDLEWARE = [
    'whitenoise.middleware.WhiteNoiseMiddleware',
] + MIDDLEWARE

STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'
