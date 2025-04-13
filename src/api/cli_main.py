"""
CLI interface for the Ollama RAG system.
"""
from rag_pipeline.config import logger, OLLAMA_MODEL
from rag_pipeline.embedding import get_ch_embedding_model
from ollama import query_ollama_with_hybrid_search_multilingual
from api.utils.database import SessionLocal

def main():
    """Main function to run an interactive Ollama RAG system with Llama Guard 3 safety."""
    try:
        print("\n===== Secure Multilingual Ollama RAG Question Answering System =====")
        print("Loading models... This may take a moment.")
        
        # Load embedding model
        embedding_model = get_ch_embedding_model()
        
        print("Models loaded successfully!")
        print("Content safety monitoring enabled with Llama Guard 3")
        print("Type 'quit', 'exit', or 'q' to end the session.")
        print("You can ask questions in any language - the system will detect and respond accordingly.")
        print("-" * 60)
        
        # Ask for user email for document access control
        user_email = input("\nPlease enter your email address for document access: ").strip()
        while not user_email or '@' not in user_email:
            print("Please enter a valid email address.")
            user_email = input("Email address: ").strip()
        
        print(f"\nDocuments will be filtered for: {user_email}")
        print("-" * 60)
        
        # Default model name for Ollama
        model_name = OLLAMA_MODEL
        
        while True:
            # Get user question
            question = input("\nPlease enter your question: ").strip()
            
            # Check if user wants to quit
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using the Multilingual Ollama System. Goodbye!")
                break
            
            # Skip empty questions
            if not question:
                print("Please enter a valid question.")
                continue
            
            print(f"\nProcessing question: {question}")
            
            # Process the query - now passing user_email
            result = query_ollama_with_hybrid_search_multilingual(
                session=SessionLocal(),
                question=question,
                embedding_model=embedding_model,
                vector_k=10,
                bm25_k=10,
                user_email=user_email,  # Pass the user email
                model_name=model_name
            )
            
            # Check if query was rejected for safety reasons
            if result.get('safety_issue', False):
                print("\n--- Safety Alert ---")
                print(result['response'])
                print("-" * 60)
                continue
            
            # Display language information if non-English was detected
            if result.get('detected_language') != 'en':
                print(f"\nDetected language: {result.get('detected_language')}")
                print(f"Translated question: {result.get('english_question')}")
            
            print(f"\n--- Response (from {result.get('context_count', 'unknown')} context chunks) ---")
            print(result['response'])  # This now includes the appended sources
            print("-" * 60)
        
    except Exception as e:
        logger.exception(f"Error in main function: {str(e)}")

    return None

if __name__ == "__main__":
    main()