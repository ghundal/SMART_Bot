"""
CLI interface for the Ollama RAG system with database-backed conversation memory.
"""

import time
import uuid

from rag_pipeline.config import OLLAMA_MODEL, logger
from rag_pipeline.embedding import get_ch_embedding_model
from rag_pipeline.ollama import query_ollama_with_hybrid_search_multilingual
from utils.chat_history import ChatHistoryManager
from utils.database import SessionLocal


def main():
    """Main function to run an interactive Ollama RAG system with persistent memory."""
    try:
        print("\n===== Secure Multilingual Ollama RAG System with Persistent Memory =====")
        print("Loading models... This may take a moment.")

        # Load embedding model
        embedding_model = get_ch_embedding_model()

        # Initialize chat history manager
        chat_manager = ChatHistoryManager(model=OLLAMA_MODEL)

        print("Models loaded successfully!")
        print("Content safety monitoring enabled with Llama Guard 3")
        print("Persistent conversation memory enabled - history is saved between sessions")
        print("Type 'quit', 'exit', or 'q' to end the session.")
        print("Type 'clear' or 'new' to start a new conversation.")
        print("Type 'list' to see your recent conversations.")
        print("Type 'load <chat_id>' to load a specific conversation.")
        print("You can ask questions in any language. The system willrespond accordingly.")
        print("-" * 60)

        # Ask for user email for document access control and session ID
        user_email = input("\nPlease enter your email address for document access: ").strip()
        while not user_email or "@" not in user_email:
            print("Please enter a valid email address.")
            user_email = input("Email address: ").strip()

        # Use email as the session ID for simplicity
        session_id = user_email

        print(f"\nDocuments will be filtered for: {user_email}")
        print("-" * 60)

        # Default model name for Ollama
        model_name = OLLAMA_MODEL

        # Initialize with a new chat
        current_chat_id = str(uuid.uuid4())
        current_chat = {
            "chat_id": current_chat_id,
            "title": "New conversation",
            "dts": int(time.time()),
            "messages": [],
        }

        while True:
            # Get user question
            question = input("\nPlease enter your question (or command): ").strip()

            # Check if user wants to quit
            if question.lower() in ["quit", "exit", "q"]:
                print("\nThank you for using the Multilingual Ollama System. Goodbye!")
                break

            # Check if user wants to clear conversation history
            if question.lower() in ["clear", "new"]:
                current_chat_id = str(uuid.uuid4())
                current_chat = {
                    "chat_id": current_chat_id,
                    "title": "New conversation",
                    "dts": int(time.time()),
                    "messages": [],
                }
                print("\nStarted a new conversation.")
                continue

            # Check if user wants to list conversations
            if question.lower() == "list":
                recent_chats = chat_manager.get_recent_chats(session_id, limit=10)
                print("\n=== Recent Conversations ===")
                if not recent_chats:
                    print("No saved conversations found.")
                else:
                    for chat in recent_chats:
                        print(f"ID: {chat['chat_id']} - Title: {chat['title']}")
                print("-" * 60)
                continue

            # Check if user wants to load a conversation
            if question.lower().startswith("load "):
                chat_id_to_load = question[5:].strip()
                loaded_chat = chat_manager.get_chat(chat_id_to_load, session_id)
                if loaded_chat:
                    current_chat_id = loaded_chat["chat_id"]
                    current_chat = loaded_chat
                    print(f"\nLoaded conversation: {current_chat['title']}")
                    # Print the last few messages for context
                    print("\n=== Recent Messages ===")
                    for msg in current_chat["messages"][-6:]:  # Show last 3 exchanges
                        role = "You" if msg["role"] == "user" else "Assistant"
                        print(f"{role}: {msg['content'][:100]}...")
                    print("-" * 60)
                else:
                    print(f"Conversation with ID {chat_id_to_load} not found.")
                continue

            # Skip empty questions
            if not question:
                print("Please enter a valid question.")
                continue

            print(f"\nProcessing question: {question}")

            # Create user message
            user_message = {
                "message_id": str(uuid.uuid4()),
                "role": "user",
                "content": question,
            }

            # Add user message to current chat
            current_chat["messages"].append(user_message)

            # Set chat title from first question if this is the first message
            if len(current_chat["messages"]) == 1:
                current_chat["title"] = question[:50] + "..." if len(question) > 50 else question

            # Update timestamp
            current_chat["dts"] = int(time.time())

            # Process the query with conversation history
            result = query_ollama_with_hybrid_search_multilingual(
                session=SessionLocal(),
                question=question,
                embedding_model=embedding_model,
                vector_k=10,
                bm25_k=10,
                user_email=user_email,
                model_name=model_name,
                chat_history=current_chat["messages"],  # Pass full chat history
            )

            # Create assistant message
            assistant_message = {
                "message_id": str(uuid.uuid4()),
                "role": "assistant",
                "content": result.get("response", "Sorry, I couldn't generate a response."),
            }

            # Add assistant message to current chat
            current_chat["messages"].append(assistant_message)

            # Save updated chat to database
            chat_manager.save_chat(current_chat, session_id)

            # Check if query was rejected for safety reasons
            if result.get("safety_issue", False):
                print("\n--- Safety Alert ---")
                print(result["response"])
                print("-" * 60)
                continue

            # Display language information if non-English was detected
            if result.get("detected_language") != "en":
                print(f"\nDetected language: {result.get('detected_language')}")
                print(f"Translated question: {result.get('english_question')}")

            print(
                f"\n--- Response (from {result.get('context_count', 'unknown')} context chunks) ---"
            )
            print(result["response"])  # This now includes the appended sources

            # Display document information if available
            if "top_documents" in result:
                print("\n=== Sources ===")
                for i, doc in enumerate(result["top_documents"], 1):
                    print(
                        f"{i}. {doc.get('class_name', 'N/A')} by "
                        f"{doc.get('authors', 'N/A')} ({doc.get('term', 'N/A')})"
                    )

            print("-" * 60)

    except Exception as e:
        logger.exception(f"Error in main function: {str(e)}")
        return None


if __name__ == "__main__":
    main()
