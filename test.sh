#!/bin/bash
# Build the Rust extension and run Python tests
set -e

echo "=== Building psychopomp ==="

# Check if maturin is installed, install if not
if ! python3 -m maturin --version &> /dev/null; then
    echo "maturin not found, installing..."
    pip3 install maturin
fi

# Use maturin develop if in a virtualenv, otherwise build and install
if [ -n "$VIRTUAL_ENV" ] || [ -n "$CONDA_PREFIX" ] || [ -d ".venv" ]; then
    python3 -m maturin develop
else
    python3 -m maturin build --release
    pip3 install --force-reinstall rust_psychopomp/target/wheels/*.whl
fi

echo ""
echo "=== Running tests ==="
python3 tests/test_backend.py
