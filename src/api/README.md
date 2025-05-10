# API

This module is desined to create ollama container with the downloaded models and start the api service.

## **Prerequisites**

Ensure that datapipeline is set up and running.

## **Organization**

```
── Readme.md
├── reports
|      |── MS3_SMART.pdf
|      |── Design_Document.pdf
|      └── Milestone1_SMART.pdf
├── sql
│   └── init.sql
|
└── src
    ├── api
    │   ├── rag_pipeline
    |   |       |── __init__.py
    |   |       |── config.py
    |   |       |── embedding.py
    |   |       |── language.py
    |   |       |── ollama_api.py
    |   |       |── safety.py
    |   |       └── search.py
    │   ├── routers
    |   |       |── __init__.py
    |   |       |── auth_google.py
    |   |       |── auth_middleware.py
    |   |       |── chat_api.py
    |   |       |── health.py
    |   |       └── reports.py
    |   ├── utils
    |   |       |── __init__.py
    |   |       |── chat_history.py
    |   |       |── database.py
    |   |       └── llm_rag_utils.py
    |   ├── __init__.py
    |   ├── docker_entrypoint.sh
    │   ├── Dockerfile
    │   ├── main_api.py
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   └── README.md
    ├── datapipeline
    ├── docker-compose.yml
    ├── docker-shell.sh
    └── Dockerfile.postgres
```

## **Running the api**

Execute the below command in /src

```bash
sh docker-shell.sh api
```

Run in the container to start the API service:

```
uvicorn_server
```
