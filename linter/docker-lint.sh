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
