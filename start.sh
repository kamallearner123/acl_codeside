#!/usr/bin/env bash
set -euo pipefail

# Resolve project root relative to this script
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$PROJECT_ROOT"

if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "Virtual environment '.venv' not found." >&2
    echo "Create it with: python -m venv .venv" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$PROJECT_ROOT/.venv/bin/activate"

python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt

python manage.py migrate
python manage.py runserver 0.0.0.0:8000
