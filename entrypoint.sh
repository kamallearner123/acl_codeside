#!/bin/sh
set -e

echo "Waiting for DB to be ready..."
# simple wait-for; in production prefer a robust wait-for script
sleep 1

echo "Apply database migrations"
python manage.py migrate --noinput

echo "Collect static files"
python manage.py collectstatic --noinput

echo "Starting Gunicorn"
exec gunicorn leetcode_clone.wsgi:application \
    --bind 0.0.0.0:${PORT:-8000} \
    --workers 3
