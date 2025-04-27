#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Set vairables
export BASE_DIR=$(pwd)
export PERSISTENT_DIR=$(pwd)/../../persistent-folder/
export SECRETS_DIR=$(pwd)/../../secrets/
export GCP_PROJECT="SMART"
export GOOGLE_APPLICATION_CREDENTIALS="/secrets/smart_input_key.json"
export DP_IMAGE_NAME="datapipeline"
export API_IMAGE_NAME="api"
export FRONTEND_IMAGE_NAME='frontend'
export GOOGLE_CREDENTIALS_FILE="/secrets/client_secrets.json"

# Create the network if we don't have it yet
docker network inspect smart-network >/dev/null 2>&1 || docker network create smart-network

# Build the image based on the Dockerfile
docker build -t $DP_IMAGE_NAME ./datapipeline
docker build --no-cache -t $API_IMAGE_NAME ./api
docker build --no-cache -t $FRONTEND_IMAGE_NAME ./frontend

# Run All Containers
# docker-compose up --build ${1:+"$1"}
docker-compose run --rm --service-ports ${1:-$DP_IMAGE_NAME}
