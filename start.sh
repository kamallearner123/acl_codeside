#!/usr/bin/env bash
set -euo pipefail

# Resolve project root relative to this script
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$PROJECT_ROOT"

# Check if Rust is installed, if not, provide instructions
if ! command -v rustc &> /dev/null; then
    echo "⚠️  Rust is not installed."
    echo "To enable local Rust execution, install Rust with:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo ""
    echo "Continuing without local Rust support (will use Playground API)..."
else
    echo "✓ Rust is installed: $(rustc --version)"
    
    # Ensure nightly Rust is installed (required for Miri)
    if ! rustup toolchain list 2>/dev/null | grep -q "nightly"; then
        echo "Installing Rust nightly toolchain..."
        rustup toolchain install nightly
    fi
    
    # Check if Miri is installed
    if ! rustup component list --toolchain nightly 2>/dev/null | grep -q "miri.*installed"; then
        echo "Miri not found. To enable Miri support, run:"
        echo "  rustup +nightly component add miri"
    fi
fi

echo "Rust part is done"

VENV_IS_NEW=false
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
    VENV_IS_NEW=true
else
    echo "Virtual environment '.venv' already exists, reusing it."
fi

# shellcheck source=/dev/null
source "$PROJECT_ROOT/.venv/bin/activate"

# Check if requirements have changed or venv is fresh
REQ_STAMP="$PROJECT_ROOT/.venv/.requirements.stamp"
if [ "$VENV_IS_NEW" = true ] || [ ! -f "$REQ_STAMP" ] || [ "$PROJECT_ROOT/requirements.txt" -nt "$REQ_STAMP" ]; then
    echo "Syncing dependencies from requirements.txt..."
    python3 -m pip install -r requirements.txt
    touch "$REQ_STAMP"
else
    echo "✓ Python dependencies up-to-date (cached)."
fi

echo "Applying database migrations..."
python3 manage.py migrate

echo "Starting Django server at http://127.0.0.1:8008 ..."
python3 manage.py runserver 0.0.0.0:8008
