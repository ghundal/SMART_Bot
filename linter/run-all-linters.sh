#!/bin/bash

# Script to run all linters and formatters
set -e

# Run Python linters
./linter/lint-and-format.sh

# Run TypeScript linters if npm is available
if command -v npm &> /dev/null; then
    ./linter/ts-lint-and-format.sh
else
    echo "Skipping TypeScript linting (npm not installed)"
fi

# Run Docker linters
./linter/docker-lint.sh

echo "All linting and formatting complete!"
