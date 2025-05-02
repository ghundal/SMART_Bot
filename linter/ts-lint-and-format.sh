#!/bin/bash

# Script to run TypeScript linters and formatters on the project
# Configuration files are at the root level

set -e

echo "Running TypeScript linters and formatters..."

# Check for frontend directories
if [ ! -d "src/frontend" ] && [ ! -d "frontend" ]; then
    echo "No frontend directories found. Skipping TypeScript linting."
    exit 0
fi

# Find TypeScript files
FRONTEND_DIR="src/frontend"
if [ ! -d "$FRONTEND_DIR" ]; then
    FRONTEND_DIR="frontend"
fi

# Format TypeScript files with Prettier
echo "Formatting TypeScript files with Prettier..."
npx prettier --write "$FRONTEND_DIR/**/*.{ts,tsx,js,jsx}"

# Lint TypeScript files with ESLint
echo "Linting TypeScript files with ESLint..."
ESLINT_USE_FLAT_CONFIG=true npx eslint "$FRONTEND_DIR/**/*.{ts,tsx}" --ext .ts,.tsx

echo "All TypeScript linting and formatting complete!"
