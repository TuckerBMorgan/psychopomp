#!/bin/bash
# Build the Rust extension and install it for Python
set -e

echo "Building psychopomp..."
maturin develop

echo ""
echo "Build complete! You can now run tests with:"
echo "  python -m pytest tests/ -v"
echo "  # or"
echo "  python tests/test_backend.py"
