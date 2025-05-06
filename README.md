# E115_SMART

This repository contains a **Retrieval-Augmented Generation (RAG)** system that integrates a **vector database** with a **Large Language Model (LLM)**. The system:

- **Chunks** text documents using semantic chunking
- **Embeds** the text into a vector space
- **Stores** embeddings in a **PostgreSQL + pgvector database**
- **Detects Language** Detects the language of the query and translates to english if needed
- **Safety** Uses LLM to ensure that query is appropriate, otherwise returns Content Violation
- **BM25 + Vector Search** if safe, retrives relevant information
- **LLM** Responds to the query in the orignal language using the relevants chunks with choice of two models
- **Reranker** Special reranker to confirm the top documents
- **Security + Audit Trail** Logs at every access and retrieval point

---

## **Overview**

Please see the design document under reports for full stack.

The following documents contain an overview of each container, its components and how to run each container.

1. [Data Pipeline](./src/datapipeline/README.md)
2.

## **Deployment**

### Preqrequsites

- Add the secrets in th github in settings -> action

### Steps

The workflows are set up in Github actions to run on commit with following steps:

- run linters
- run tests
- build images
- push the images to GCR
  <need ansible step here>
- deploy to kubernetes

- **Commit**

```
git add .
git commit -m "<message>"
git push
```

## **Manual Deployment**
