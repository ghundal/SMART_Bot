#!/bin/bash

echo "Container is running!!!"

# Start Ollama and capture PID
ollama serve &
OLLAMA_PID=$!

# Trap SIGINT and SIGTERM to stop Ollama gracefully
cleanup() {
  echo "Caught termination signal. Stopping Ollama..."
  kill -TERM "$OLLAMA_PID" 2>/dev/null
  wait "$OLLAMA_PID"
  exit 0
}
trap cleanup SIGINT SIGTERM

# this will run the api/service.py file with the instantiated app FastAPI
uvicorn_server() {
    uvicorn main_api:app --host 0.0.0.0 --port 9000 --log-level debug --reload --reload-dir ./ "$@"
}

uvicorn_server_production() {
    pipenv run uvicorn main_api:app --host 0.0.0.0 --port 9000 --lifespan on
}

export -f uvicorn_server
export -f uvicorn_server_production

echo -en "\033[92m
The following commands are available:
    uvicorn_server
        Run the Uvicorn Server
\033[0m
"

python -c "import nltk; nltk.download('stopwords')"
echo "Waiting for Ollama server to start..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
  sleep 1
done

if [ "${DEV}" = 1 ]; then
  pipenv shell
else
  uvicorn_server_production
fi
