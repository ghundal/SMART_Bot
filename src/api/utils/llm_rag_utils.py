'''
Provides utility functions for creating and managing chat sessions,
including functions for generating responses through the Ollama RAG system
and rebuilding chat context from history.
'''

import os
from typing import Dict, Any, List, Optional
from fastapi import HTTPException
import traceback
import logging
from api.utils.database import connect_to_postgres, SessionLocal
from sqlalchemy import text

# Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('llm_rag_utils')

# Initialize chat sessions
chat_sessions: Dict[str, Any] = {}

def create_chat_session():
    """Create a new chat session for tracking conversation history"""
    return {
        "messages": [],
        "metadata": {
            "created_at": None,
            "last_updated": None
        }
    }

def generate_chat_response(chat_session, message: Dict, user_email: str) -> str:
    """
    Generate a response using Ollama models via the hybrid search approach.
    
    Args:
        chat_session: The chat session tracking conversation history
        message: Dict containing 'content' (text)
    
    Returns:
        str: The model's response
    """
    try:
        # Import here to avoid circular imports
        from ollama import query_ollama_with_hybrid_search_multilingual, get_ch_embedding_model
        
        # Get content from message
        content = message.get("content", "")
        if not content:
            raise ValueError("Message must contain text content")

        # Get or initialize embedding model
        embedding_model = get_ch_embedding_model()
        
        # Add message to chat history
        chat_session["messages"].append({
            "role": "user",
            "content": content
        })
        
        # Get response from Ollama using hybrid search
        result = query_ollama_with_hybrid_search_multilingual(
            session=SessionLocal(),
            question=content,
            embedding_model=embedding_model,
            vector_k=10,
            bm25_k=10,
            user_email=user_email
        )
        
        # Extract response from result
        response = result.get("response", "I'm sorry, I couldn't generate a response.")
        
        # Add assistant response to chat history
        chat_session["messages"].append({
            "role": "assistant",
            "content": response
        })
        
        # Log success
        logger.info(f"Generated response with {result.get('context_count', 0)} context chunks")
        
        return response
        
    except Exception as e:
        logger.exception(f"Error generating response: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate response: {str(e)}"
        )

def rebuild_chat_session(chat_history: List[Dict]) -> Dict:
    """
    Rebuild a chat session with complete context from chat history
    
    Args:
        chat_history: List of message dictionaries from chat history
    
    Returns:
        Dict: Reconstructed chat session
    """
    new_session = create_chat_session()
    
    # Extract just the role and content for each message
    for message in chat_history:
        if message.get("role") and message.get("content"):
            new_session["messages"].append({
                "role": message["role"],
                "content": message["content"]
            })
    
    return new_session