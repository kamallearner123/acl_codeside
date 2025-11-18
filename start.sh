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

echo ""
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
