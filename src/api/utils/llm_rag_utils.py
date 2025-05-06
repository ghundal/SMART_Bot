"""
Chat Session Utilities for SMART RAG System

This module provides helper functions for managing in-memory chat sessions within
the SMART system's Retrieval-Augmented Generation (RAG) pipeline. It supports:

1. Creating isolated chat sessions to track multi-turn user interactions.
2. Rebuilding chat context from historical messages for follow-up queries.
3. Maintaining consistent in-memory state using the `chat_sessions` dictionary.

Functions:
- `create_chat_session`: Initializes an empty session with message history and metadata.
- `rebuild_chat_session`: Reconstructs session context from past chat message records.

Global State:
- `chat_sessions`: Dictionary to track active sessions keyed by chat ID.
"""

import logging
from typing import Any, Dict, List


# Setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("llm_rag_utils")

# Initialize chat sessions
chat_sessions: Dict[str, Any] = {}


def create_chat_session():
    """Create a new chat session for tracking conversation history"""
    return {"messages": [], "metadata": {"created_at": None, "last_updated": None}}


def rebuild_chat_session(chat_history: List[Dict]) -> Dict[str, Any]:
    """
    Rebuild a chat session with complete context from chat history
    Args:
        chat_history: List of message dictionaries from chat history
    Returns:
        Dict[str, Any]: Reconstructed chat session
    """
    new_session: Dict[str, Any] = create_chat_session()
    # Extract just the role and content for each message
    for message in chat_history:
        if message.get("role") and message.get("content"):
            new_session["messages"].append({"role": message["role"], "content": message["content"]})
    return new_session
