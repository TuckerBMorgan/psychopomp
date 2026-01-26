#!/bin/bash
# Build the Rust extension and run Python tests
set -e

echo "=== Building psychopomp ==="
maturin develop

echo ""
echo "=== Running tests ==="
python tests/test_backend.py
