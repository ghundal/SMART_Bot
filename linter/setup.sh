#!/bin/bash
# Script to set up linters and formatters for Python and TypeScript code
# All configuration files will be at the root level of the project

set -e

echo "Setting up Python and TypeScript linters and formatters for your project..."

# Create linter directory for scripts
mkdir -p linter

# Install Python linters and formatters using pipenv
# Make sure pipenv is installed
if ! command -v pipenv &> /dev/null; then
    echo "pipenv not found. Installing pipenv..."
    pip install pipenv
fi

# Install Python linting dependencies with pipenv
echo "Installing Python linting tools with pipenv..."
pipenv install --dev black flake8 isort pylint mypy

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "npm not found. Please install Node.js and npm to enable TypeScript linting."
else
    # Install TypeScript linting tools with npm
    echo "Installing TypeScript linting tools with npm..."
    npm install --save-dev eslint typescript @typescript-eslint/parser @typescript-eslint/eslint-plugin eslint-config-prettier prettier
fi

# Create configuration files
echo "Creating configuration files at root level..."

# Python - Black configuration
cat > pyproject.toml << 'EOF'
[tool.black]
line-length = 100
target-version = ['py38']
include = '\.pyi?$'
exclude = '''
/(
    \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
line_length = 100
multi_line_output = 3
include_trailing_comma = true
EOF

# Python - Flake8 configuration
cat > .flake8 << 'EOF'
[flake8]
max-line-length = 100
extend-ignore = E203, E501
exclude = .git,__pycache__,docs/source/conf.py,old,build,dist,.venv,tests/,tests/*,
    tests/**/*,src/frontend/
EOF

# Python - MyPy configuration
cat > mypy.ini << 'EOF'
[mypy]
mypy_path = src
python_version = 3.12
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = False
disallow_incomplete_defs = False
ignore_missing_imports = True
namespace_packages = True
explicit_package_bases = True
exclude = ('tests|src/frontend')

[mypy.plugins.numpy.*]
follow_imports = skip

[mypy-pandas.*]
ignore_missing_imports = True

[mypy-google.cloud.*]
ignore_missing_imports = True
EOF

# TypeScript - ESLint configuration using flat config format
cat > eslint.config.js << 'EOF'
// eslint.config.js - Flat config format
module.exports = [
  {
    // Expanded ignores section to cover your specific case
    ignores: [
      // Next.js build directories - with multiple variations to catch all cases
      '**/.next/**',
      '**/src/frontend/.next/**',
      '**/frontend/.next/**',
      '**/E115_SMART/src/frontend/.next/**',
      '/next/home/gurpreet/Desktop/Spring2025/CSCI115/Project/E115_SMART/src/frontend/.next/**',
      // Other common directories to ignore
      '**/node_modules/**',
      '**/dist/**',
      '**/build/**'
    ],
  },
  {
    files: ['**/*.ts', '**/*.tsx', '**/*.js', '**/*.jsx'],
    languageOptions: {
      parser: require('@typescript-eslint/parser'),
      parserOptions: {
        ecmaVersion: 'latest',
        sourceType: 'module',
      },
      globals: {
        browser: true,
        node: true,
        es6: true,
      },
    },
    plugins: {
      '@typescript-eslint': require('@typescript-eslint/eslint-plugin'),
    },
    rules: {
      // TypeScript specific rules
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/explicit-function-return-type': 'off',
      '@typescript-eslint/explicit-module-boundary-types': 'off',
      '@typescript-eslint/no-unused-vars': ['error', {
        'argsIgnorePattern': '^_',
        'varsIgnorePattern': '^_'
      }],

      // General best practices
      'no-console': ['warn', { allow: ['warn', 'error'] }],
      'prefer-const': 'error',
      'no-var': 'error',
      'eqeqeq': ['error', 'always'],
      'curly': ['error', 'all'],
      'no-duplicate-imports': 'error',
      'no-multiple-empty-lines': ['error', { 'max': 1, 'maxEOF': 1 }],
      'sort-imports': ['error', {
        'ignoreCase': true,
        'ignoreDeclarationSort': true,
        'ignoreMemberSort': false
      }]
    },
  },
  // Prettier rules instead of using extends
  {
    files: ['**/*.ts', '**/*.tsx', '**/*.js', '**/*.jsx'],
    rules: {
      'arrow-body-style': 'off',
      'prefer-arrow-callback': 'off',
    },
  },
];
EOF

# TypeScript - Prettier configuration
cat > .prettierrc << 'EOF'
{
  "semi": true,
  "trailingComma": "all",
  "singleQuote": true,
  "printWidth": 100,
  "tabWidth": 2
}
EOF

# YAML linting config
cat > .yamllint << 'EOF'
extends: default

rules:
  line-length:
    max: 120
    level: warning
  document-start: disable
EOF

# Create a helper script to run Python linters and formatters
cat > linter/lint-and-format.sh << 'EOF'
#!/bin/bash

# Script to run Python linters and formatters on the project
# Configuration files are at the root level

set -e

echo "Running Python linters and formatters..."

# Format Python files
echo "Formatting Python files with black..."
pipenv run black src/datapipeline
if [ -d "src/api" ]; then
    pipenv run black src/api
fi
if [ -d "tests" ]; then
    pipenv run black tests
fi

# Sort imports in Python files
echo "Sorting imports with isort..."
pipenv run isort src/datapipeline
if [ -d "src/api" ]; then
    pipenv run isort src/api
fi
if [ -d "tests" ]; then
    pipenv run isort tests
fi

# Lint Python files
echo "Linting Python files with flake8..."
pipenv run flake8 src/datapipeline
if [ -d "src/api" ]; then
    pipenv run flake8 src/api
fi
# Skip linting test files with flake8
echo "Skipping flake8 linting for test files..."

# Type checking Python files
echo "Type checking Python files with mypy..."
pipenv run mypy src/datapipeline
if [ -d "src/api" ]; then
    pipenv run mypy src/api
fi
# Skip type checking test files
echo "Skipping mypy type checking for test files..."

echo "All Python linting and formatting complete!"
EOF

# Create a helper script to run TypeScript linters and formatters
cat > linter/ts-lint-and-format.sh << 'EOF'
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
npx prettier --write "$FRONTEND_DIR"

# Lint TypeScript files with ESLint
echo "Linting TypeScript files with ESLint..."
ESLINT_USE_FLAT_CONFIG=true npx eslint "$FRONTEND_DIR" --ext .ts,.tsx

echo "All TypeScript linting and formatting complete!"
EOF

# Create Docker linting script
cat > linter/docker-lint.sh << 'EOF'
#!/bin/bash
# Script to set up and run Docker linting

set -e

echo "Setting up Docker linting tools..."

# Install hadolint for Dockerfile linting if not already installed
if ! command -v hadolint &> /dev/null; then
    if command -v brew &> /dev/null; then
        brew install hadolint
    elif command -v apt-get &> /dev/null; then
        wget -O /tmp/hadolint https://github.com/hadolint/hadolint/releases/latest/download/hadolint-Linux-x86_64
        chmod +x /tmp/hadolint
        sudo mv /tmp/hadolint /usr/local/bin/
    else
        echo "Please install hadolint manually: https://github.com/hadolint/hadolint"
    fi
fi

# Install yamllint for YAML linting
pipenv install --dev yamllint

# Function to lint Docker files
lint_docker_files() {
    echo "Linting Dockerfiles..."

    # Find all Dockerfiles
    dockerfiles=$(find . -name "Dockerfile*")

    if [ -z "$dockerfiles" ]; then
        echo "No Dockerfiles found."
    else
        for file in $dockerfiles; do
            echo "Linting $file"
            hadolint "$file"
        done
    fi

    echo "Linting docker-compose.yml files..."

    # Find all docker-compose.yml files
    compose_files=$(find . -name "docker-compose*.yml")

    if [ -z "$compose_files" ]; then
        echo "No docker-compose files found."
    else
        for file in $compose_files; do
            echo "Linting $file"
            pipenv run yamllint "$file"
        done
    fi
}

# Run the linting
lint_docker_files

echo "Docker linting complete!"
EOF

# Make scripts executable
chmod +x linter/lint-and-format.sh
chmod +x linter/ts-lint-and-format.sh
chmod +x linter/docker-lint.sh

# Create a main script to run all linters
cat > linter/run-all-linters.sh << 'EOF'
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
EOF

chmod +x linter/run-all-linters.sh

# Create README
cat > linter/README.md << 'EOF'
# Python and TypeScript Linting and Formatting Guide

This guide covers how to set up and use linters and formatters to maintain code quality in this project.

## Setup

Run the setup script to install all linters and formatters:

```bash
chmod +x linter/setup.sh
./linter/setup.sh
```

## Tools Included

### Python
- **Black**: Code formatter (line length: 100)
- **isort**: Import statement formatter
- **Flake8**: Style guide enforcer
- **MyPy**: Type checker

### TypeScript/JavaScript
- **ESLint**: Static code analyzer (using flat config format)
- **Prettier**: Code formatter

### Docker and YAML
- **hadolint**: Dockerfile linter
- **yamllint**: YAML linter

## Running All Linters and Formatters

You can run all linters and formatters with a single command:

```bash
./linter/run-all-linters.sh
```

## Running Individual Tools

### Python (with Pipenv)
```bash
./linter/lint-and-format.sh
```

### TypeScript/JavaScript (with npm)
```bash
./linter/ts-lint-and-format.sh
```

### Docker Files
```bash
./linter/docker-lint.sh
```

## Configuration Files

All configuration files are now at the root level of the project:

- Python:
  - `pyproject.toml` (Black, isort)
  - `.flake8`
  - `mypy.ini`

- TypeScript:
  - `eslint.config.js` (ESLint flat config)
  - `.prettierrc`

- Docker/YAML:
  - `.yamllint`

## Flat Config Format for ESLint

This project uses ESLint's new flat config format, which is the default in ESLint v9+. The key differences from the traditional format are:

- Configuration is an array of config objects
- Parser settings are specified under `languageOptions`
- Plugins are specified as objects instead of strings
- No `extends` property - configs are composed by adding objects to the array
- No `root` property (flat configs are always treated as root)

For more details, see the [ESLint Flat Config Migration Guide](https://eslint.org/docs/latest/use/configure/migration-guide).

## Customizing Configuration

You can customize the linter settings by editing the configuration files at the root level of the project.
EOF

# Create a copy of this script as the setup script
cp "$0" linter/setup.sh
chmod +x linter/setup.sh

echo "Setup complete! You can now run ./linter/run-all-linters.sh to lint and format all your code."
echo "All configuration files are now placed at the root level of the project."
