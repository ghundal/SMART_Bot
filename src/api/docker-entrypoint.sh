#!/bin/bash

echo "Container is running!!!"

# Start Ollama and capture PID
ollama serve &
OLLAMA_PID=$!

# Trap SIGINT and SIGTERM to stop Ollama gracefully
term_handler() {
  echo "Caught termination signal. Stopping Ollama and Uvicorn..."

  if kill -0 "$OLLAMA_PID" 2>/dev/null; then
    kill -TERM "$OLLAMA_PID"
    wait "$OLLAMA_PID"
  fi

  if [[ -n "$UVICORN_PID" ]] && kill -0 "$UVICORN_PID" 2>/dev/null; then
    kill -TERM "$UVICORN_PID"
    wait "$UVICORN_PID"
  fi

  exit 0
}

trap term_handler SIGINT SIGTERM

# Function to download models
download_models() {
  echo "Downloading required models..."

  # Array of models to download
  MODELS=("llama-guard3:8b" "llama3:8b" "gemma3:12b" "qllama/bge-reranker-large")

  for model in "${MODELS[@]}"; do
    echo "Pulling model: $model"
    ollama pull "$model" || echo "Warning: Failed to pull $model"
  done

  echo "Model downloads complete."
}

# this will run the api/service.py file with the instantiated app FastAPI
uvicorn_server() {
    uvicorn main_api:app --host 0.0.0.0 --port 9000 --log-level debug --reload --reload-dir ./ "$@"
}

uvicorn_server_production() {
    pipenv run uvicorn main_api:app --host 0.0.0.0 --port 9000 --lifespan on
    UVICORN_PID=$!
    wait "$UVICORN_PID"
}

export -f uvicorn_server
export -f uvicorn_server_production

echo -en "\033[92m
The following commands are available:
    uvicorn_server
        Run the Uvicorn Server
\033[0m
"
# Download NLTK stopwords
python -c "import nltk; nltk.download('stopwords')"

echo "Waiting for Ollama server to start..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
  sleep 1
done

# Download models once the server is up
download_models

if [ "${DEV}" = 1 ]; then
  pipenv shell
else
  uvicorn_server_production
fi
