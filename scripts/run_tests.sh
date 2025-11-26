#!/bin/bash
# Run comprehensive tests

echo "Running VITON-MOD Tests..."
echo ""

echo "1. Running realistic editing tests..."
python tests/realistic_test.py

echo ""
echo "2. Running logo placement tests..."
python tests/logo_test.py

echo ""
echo "Tests complete! Check ./results/ for outputs."
