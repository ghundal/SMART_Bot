"""
Configuration settings for the Ollama RAG system.
"""

import logging

# Ollama API endpoints
OLLAMA_URL = "http://localhost:11434/api/generate"
# OLLAMA_MODEL = "gemma3:12b"
OLLAMA_MODEL = "llama3:8b"
RERANKER_MODEL = "BAAI/bge-reranker-base"
SAFETY_MODEL = "llama-guard3:8b"
EMBEDDING_MODEL = "all-mpnet-base-v2"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ollama_rag")

# Define generation config with improved parameters
GENERATION_CONFIG = {
    "max_length": 256,
    "temperature": 0.1,
    "top_p": 0.9,
    "repeat_penalty": 1.3,
    "stream": False,
}

# Token limits
MAX_INPUT_TOKENS = 4000

# Vector search threshold
VECTOR_SIMILARITY_THRESHOLD = 0.3

# Default chunk limits for searches
DEFAULT_VECTOR_K = 10
DEFAULT_BM25_K = 10
