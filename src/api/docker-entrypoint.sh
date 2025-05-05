#!/bin/bash
echo "Container is running!!!"

# Store the PID of the uvicorn process
UVICORN_PID=""

# Signal handler function
handle_signal() {
  echo "Received signal, shutting down gracefully..."
  if [ -n "$UVICORN_PID" ]; then
    echo "Stopping Uvicorn process (PID: $UVICORN_PID)..."
    kill -TERM "$UVICORN_PID"
    wait "$UVICORN_PID"
  fi
  echo "Shutdown complete"
  exit 0
}

# Register signal handlers
trap handle_signal SIGINT SIGTERM

# this will run the api/service.py file with the instantiated app FastAPI
uvicorn_server() {
  uvicorn main_api:app --host 0.0.0.0 --port 9000 --log-level debug --reload --reload-dir ./ "$@" &
  UVICORN_PID=$!
  wait "$UVICORN_PID"
  UVICORN_PID=""
}

uvicorn_server_production() {
  pipenv run uvicorn main_api:app --host 0.0.0.0 --port 9000 --lifespan on &
  UVICORN_PID=$!
  wait "$UVICORN_PID"
  UVICORN_PID=""
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

if [ "${DEV}" = 1 ]; then
  # Signals will be handled by the shell itself
  pipenv shell
else
  uvicorn_server_production
fi
