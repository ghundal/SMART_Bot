"""
Provides utility functions for creating and managing chat sessions,
including functions for generating responses through the Ollama RAG system
and rebuilding chat context from history.
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
