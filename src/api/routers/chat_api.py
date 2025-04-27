"""
Implements a FastAPI router for an Ollama-based RAG chatbot with
endpoints for creating, updating, retrieving, and deleting chat sessions,
including hybrid vector search functionality.
"""

import time
import uuid
from typing import Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from ..rag_pipeline.config import DEFAULT_BM25_K, DEFAULT_VECTOR_K, OLLAMA_MODEL
from ..rag_pipeline.embedding import get_ch_embedding_model
from ..rag_pipeline.ollama import query_ollama_with_hybrid_search_multilingual
from ..utils.chat_history import ChatHistoryManager
from ..utils.database import SessionLocal
from ..utils.llm_rag_utils import chat_sessions, create_chat_session, rebuild_chat_session
from .auth_middleware import verify_token


class QueryRequest(BaseModel):
    chat_id: str
    question: str
    model_name: str
    session_id: str


# Define Router
router = APIRouter()

# Initialize chat history manager and sessions
chat_manager = ChatHistoryManager(model="ollama-rag")

# Load embedding model
embedding_model = get_ch_embedding_model()


@router.get("/chats")
async def get_chats(
    x_session_id: str = Header(None, alias="X-Session-ID"),
    limit: Optional[int] = None,
    user_email: str = Depends(verify_token),
):
    """Get all chats, optionally limited to a specific number"""
    print(f"User {user_email} retrieving chats with session: {x_session_id}")
    return chat_manager.get_recent_chats(x_session_id, limit)


@router.get("/chats/{chat_id}")
async def get_chat(
    chat_id: str,
    x_session_id: str = Header(None, alias="X-Session-ID"),
    user_email: str = Depends(verify_token),
):
    """Get a specific chat by ID"""
    print(f"User {user_email} retrieving chat {chat_id} with session: {x_session_id}")
    chat = chat_manager.get_chat(chat_id, x_session_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.post("/chats")
async def start_chat_with_llm(
    message: Dict,
    x_session_id: str = Header(None, alias="X-Session-ID"),
    user_email: str = Depends(verify_token),
):
    """Start a new chat with an initial message"""
    print(
        f"User {user_email} starting chat with content: {message.get('content')} in session: {x_session_id}"
    )

    chat_id = str(uuid.uuid4())
    current_time = int(time.time())

    # Get message content
    question = message.get("content", "")
    if not question:
        raise HTTPException(status_code=400, detail="Message content is required")

    # Generate response using Ollama
    result = query_ollama_with_hybrid_search_multilingual(
        session=SessionLocal(),
        question=question,
        embedding_model=embedding_model,
        vector_k=DEFAULT_VECTOR_K,
        bm25_k=DEFAULT_BM25_K,
        model_name=message.get("model", OLLAMA_MODEL),
        user_email=user_email,
    )

    # Create a new chat session
    chat_session = create_chat_session()
    chat_sessions[chat_id] = chat_session

    # Create chat response
    title = question[:50] + "..." if len(question) > 50 else question

    chat_response = {
        "chat_id": chat_id,
        "title": title,
        "dts": current_time,
        "messages": [
            {"message_id": str(uuid.uuid4()), "role": "user", "content": question},
            {
                "message_id": str(uuid.uuid4()),
                "role": "assistant",
                "content": result.get("response", "Sorry, I couldn't generate a response."),
            },
        ],
    }

    # Add document information if available
    if "top_documents" in result:
        chat_response["top_documents"] = result["top_documents"]

    # Save chat
    chat_manager.save_chat(chat_response, x_session_id)
    return chat_response


@router.post("/chats/{chat_id}")
async def continue_chat_with_llm(
    chat_id: str,
    message: Dict,
    x_session_id: str = Header(None, alias="X-Session-ID"),
    user_email: str = Depends(verify_token),
):
    """Add a message to an existing chat"""
    print(
        f"User {user_email} continuing chat {chat_id} with content: {message.get('content')} in session: {x_session_id}"
    )

    # Get message content
    question = message.get("content", "")
    if not question:
        raise HTTPException(status_code=400, detail="Message content is required")

    # Get existing chat
    chat = chat_manager.get_chat(chat_id, x_session_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Get or rebuild chat session
    chat_session = chat_sessions.get(chat_id)
    if not chat_session:
        chat_session = rebuild_chat_session(chat["messages"])
        chat_sessions[chat_id] = chat_session

    # Update timestamp
    current_time = int(time.time())
    chat["dts"] = current_time

    # Add message ID and role
    user_message = {
        "message_id": str(uuid.uuid4()),
        "role": "user",
        "content": question,
    }

    # Add the user message to chat history
    chat["messages"].append(user_message)

    # Generate response using Ollama
    result = query_ollama_with_hybrid_search_multilingual(
        session=SessionLocal(),
        question=question,
        embedding_model=embedding_model,
        vector_k=DEFAULT_VECTOR_K,
        bm25_k=DEFAULT_BM25_K,
        model_name=message.get("model", OLLAMA_MODEL),
        user_email=user_email,
        chat_history=chat["messages"][:-1],
    )

    # Create assistant message with response
    assistant_message = {
        "message_id": str(uuid.uuid4()),
        "role": "assistant",
        "content": result.get("response", "Sorry, I couldn't generate a response."),
    }

    # Add the assistant response to chat history
    chat["messages"].append(assistant_message)

    # Add document information if available
    if "top_documents" in result:
        chat["top_documents"] = result["top_documents"]

    # Save updated chat
    chat_manager.save_chat(chat, x_session_id)
    return chat


@router.delete("/chats/{chat_id}")
async def delete_chat(
    chat_id: str,
    x_session_id: str = Header(None, alias="X-Session-ID"),
    user_email: str = Depends(verify_token),
):
    """Delete a chat by ID"""
    print(f"User {user_email} deleting chat {chat_id} in session: {x_session_id}")

    # Remove from chat sessions if present
    if chat_id in chat_sessions:
        del chat_sessions[chat_id]

    # Delete from database
    success = chat_manager.delete_chat(chat_id, x_session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Chat not found or could not be deleted")

    return {"status": "success", "message": f"Chat {chat_id} deleted successfully"}


@router.post("/query")
async def process_query(request: QueryRequest, user_email: str = Depends(verify_token)):
    """
    Process a query from the frontend with user authentication
    Provides compatibility with the /query endpoint
    """
    # Parse the JSON body from request
    chat_id = request.chat_id
    question = request.question
    model_name = request.model_name
    session_id = request.session_id

    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")

    # If chat_id is provided, add to existing chat, otherwise create new chat
    if chat_id:
        # Get existing chat
        chat = chat_manager.get_chat(chat_id, session_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")

        # Get or rebuild chat session
        chat_session = chat_sessions.get(chat_id)
        if not chat_session:
            chat_session = create_chat_session()
            # Populate with existing messages
            for message in chat["messages"]:
                if message["role"] == "user":
                    chat_session["messages"].append({"role": "user", "content": message["content"]})
                elif message["role"] == "assistant":
                    chat_session["messages"].append(
                        {"role": "assistant", "content": message["content"]}
                    )
            chat_sessions[chat_id] = chat_session

        # Update timestamp
        current_time = int(time.time())
        chat["dts"] = current_time

        # Process the new message with RAG
        message = {"content": question, "message_id": str(uuid.uuid4()), "role": "user"}

        # Add the new message to chat history
        chat["messages"].append(message)

        # Generate response from Ollama
        result = query_ollama_with_hybrid_search_multilingual(
            session=SessionLocal(),
            question=question,
            embedding_model=embedding_model,
            vector_k=10,
            bm25_k=10,
            model_name=model_name,
            user_email=user_email,
            chat_history=chat["messages"][:-1],
        )

        # Create assistant message with response
        assistant_message = {
            "message_id": str(uuid.uuid4()),
            "role": "assistant",
            "content": result.get("response", "Sorry, I couldn't generate a response."),
        }

        # Add the assistant response to chat history
        chat["messages"].append(assistant_message)

        # Save updated chat
        chat_manager.save_chat(chat, session_id)

        # Add document information if available
        if "top_documents" in result:
            chat["top_documents"] = result["top_documents"]

        return chat

    else:
        # Create a new chat
        new_chat_id = str(uuid.uuid4())
        current_time = int(time.time())

        # Create a new chat session
        chat_session = create_chat_session()
        chat_sessions[new_chat_id] = chat_session

        # Generate response from Ollama
        result = query_ollama_with_hybrid_search_multilingual(
            session=SessionLocal(),
            question=question,
            embedding_model=embedding_model,
            vector_k=DEFAULT_VECTOR_K,
            bm25_k=DEFAULT_BM25_K,
            model_name=model_name,
            user_email=user_email,
        )

        # Create chat response structure
        title = question[:50] + "..." if len(question) > 50 else question
        chat_response = {
            "chat_id": new_chat_id,
            "title": title,
            "dts": current_time,
            "messages": [
                {"message_id": str(uuid.uuid4()), "role": "user", "content": question},
                {
                    "message_id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": result.get("response", "Sorry, I couldn't generate a response."),
                },
            ],
        }

        # Add document information if available
        if "top_documents" in result:
            chat_response["top_documents"] = result["top_documents"]

        # Save chat
        chat_manager.save_chat(chat_response, session_id)

        return chat_response
