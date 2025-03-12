#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Set vairables
export BASE_DIR=$(pwd)
export PERSISTENT_DIR=$(pwd)/../persistent-folder/
export SECRETS_DIR=$(pwd)/../secrets/
export GCP_PROJECT="SMART"
export GOOGLE_APPLICATION_CREDENTIALS="/secrets/smart-452816-101a65261db2.json"
export IMAGE_NAME="smart_input"

# Create the network if we don't have it yet
docker network inspect smart-network >/dev/null 2>&1 || docker network create smart-network

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .

# Run All Containers
docker-compose run --rm --service-ports ${1:-$IMAGE_NAME}
