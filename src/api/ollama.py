import torch
import psycopg2
import os
import logging
import numpy as np
import requests
import re
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import nltk
from database import connect_to_postgres

# Import search functions from search.py
from rag_pipeline.search import hybrid_search, retrieve_document_metadata

# Import language and safety functions from language.py
from rag_pipeline.language import detect_language, translate_text, check_query_safety_with_llama_guard

# Ollama API endpoints
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"
# OLLAMA_MODEL = 'llama3-chatqa' # only to build memory
RERANKER_MODEL = "llama3.1"
# OLLAMA_MODEL = "gemma3:12b"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ollama_model')

# Define generation config with improved parameters
GENERATION_CONFIG = {
    "max_length": 256,
    "temperature": 0.1, 
    "top_p": 0.9,
    "repeat_penalty": 1.3,
    "stream": False
}

# Token limits
MAX_INPUT_TOKENS = 4000

# DB session
engine = connect_to_postgres()
SessionLocal = sessionmaker(bind=engine)

def get_ch_embedding_model():
    """Load and return the embedding model."""
    try:
        model_name = 'all-mpnet-base-v2'
        # model_name = 'multi-qa-mpnet-base-dot-v1'
        model = SentenceTransformer(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Successfully loaded Embedding model: {model_name}")
        return model
    except Exception as e:
        logger.exception(f"Error loading Embedding model: {e}")
        raise

def embed_query(query, model):
    """Generate an embedding for the given query."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Encode query and move to the same device as the model
    embedding = model.encode(
        query, 
        show_progress_bar=False,
        device=device 
    )
    return embedding.tolist()

def format_prompt(system_instruction, context, question):
    """Format the prompt for Ollama."""
    prompt = f"""
{system_instruction}

Here is some context to help answer my question:

{context}

My question is: {question}
"""
    return prompt

def log_audit(session, user_email, query, query_embedding, chunks, response):
    try:
        document_ids = [chunk['document_id'] for chunk in chunks]
        chunk_texts = [chunk['chunk_text'] for chunk in chunks]

        # Format the embedding into a pgvector-compatible string
        embedding_str = str(query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding)

        sql = f"""
        INSERT INTO audit (
            user_email, query, query_embedding, document_ids, chunk_texts, response
        ) VALUES (
            :user_email, :query, '{embedding_str}'::vector, :document_ids, :chunk_texts, :response
        )
        """

        params = {
            "user_email": user_email,
            "query": query,
            "document_ids": document_ids,
            "chunk_texts": chunk_texts,
            "response": response
        }

        session.execute(text(sql), params)
        session.commit()
        logger.info("Successfully logged audit entry.")

    except Exception as e:
        logger.exception(f"Error logging audit: {e}")

def query_ollama_with_hybrid_search_multilingual(session, question, embedding_model, 
                                   vector_k, bm25_k, user_email='Anonymous',
                                   model_name=OLLAMA_MODEL):
    """
    Query the Ollama model using hybrid search with multilingual support.
    """
    try:
        # First safety check on original query (any language)
        is_safe_original, reason_original = check_query_safety_with_llama_guard(question)
        if not is_safe_original:
            logger.warning(f"Original query failed safety check: {reason_original}")
            return {
                "original_question": question,
                "safety_issue": True,
                "response": f"I cannot process this request: {reason_original}",
                "context_count": 0
            }
            
        # Detect original language
        original_language = detect_language(question)
        logger.info(f"Detected language: {original_language}")
        
        # Translate question to English if not already English
        if original_language != 'en':
            english_question = translate_text(question, target_lang='en', source_lang=original_language)
            logger.info(f"Translated question to English: {english_question}")
            
            # Second safety check on translated English question
            is_safe_translated, reason_translated = check_query_safety_with_llama_guard(english_question)
            if not is_safe_translated:
                logger.warning(f"Translated query failed safety check: {reason_translated}")
                # Translate the rejection reason back to the original language
                rejection_message = f"I cannot process this request: {reason_translated}"
                localized_rejection = translate_text(
                    rejection_message, 
                    target_lang=original_language, 
                    source_lang='en'
                )
                return {
                    "original_question": question,
                    "english_question": english_question,
                    "safety_issue": True,
                    "response": localized_rejection,
                    "context_count": 0
                }
        else:
            english_question = question
        
        
        # Embed the English query
        query_embedding = embed_query(english_question, embedding_model)
        logger.info(f"Generated query embedding with {len(query_embedding)} dimensions")
        
        # Perform hybrid search with reduced chunk count
        context_chunks, reranked_chunks = hybrid_search(session, english_question, query_embedding, vector_k, bm25_k)

        # Get top 3 unique document IDs from reranked chunks
        top_document_ids = []
        seen = set()
        for chunk in reranked_chunks:
            doc_id = chunk['document_id']
            if doc_id not in seen:
                seen.add(doc_id)
                top_document_ids.append(doc_id)
            if len(top_document_ids) == 3:
                break
        
        # Get document metadata
        document_metadata = retrieve_document_metadata(session, top_document_ids)
        
        # Format context chunks for the prompt
        contexts = [f"DOCUMENT {i+1}:\n{chunk['chunk_text']}" for i, chunk in enumerate(context_chunks)]
        context = "\n\n".join(contexts)
        
        # System prompt - add multilingual instruction if needed
        system_prompt = """
        You are an AI assistant specialized in machine learning, deep learning, and data science. You provide helpful, accurate, and educational responses to questions about these topics.

        When answering a query:
        1. Provide clear explanations with appropriate technical detail for the complexity of the question.
        2. When explaining concepts, include practical examples to illustrate how they work.
        3. If relevant, mention advantages, limitations, and common use cases.
        4. Break down your explanation into understandable components.
        5. Maintain a professional and educational tone throughout your responses.
        6. Prioritize information from the chunks and enhance/format with your knowledge.
        7. Keep your answers concise and to the point.
        8. If you don't know the answer, say so.
        9. Do not include unnecessary information or repetitive explanations.
        10. Format your response clearly and directly address the question.
        """
        
        # For non-English queries, specify that response should be in English first (we'll translate after)
        if original_language != 'en':
            system_prompt += "\n\nPlease respond in English. The response will be translated later."
        
        # Format the prompt with English question
        prompt = format_prompt(system_prompt, context, english_question)
        
        # Prepare the request payload for Ollama
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": GENERATION_CONFIG["stream"],
            "temperature": GENERATION_CONFIG["temperature"],
            "top_p": GENERATION_CONFIG["top_p"],
            "repeat_penalty": GENERATION_CONFIG["repeat_penalty"],
        }
        
        # Call Ollama API
        response = requests.post(OLLAMA_URL, json=payload)
        
        # Check for successful response
        if response.status_code == 200:
            result = response.json()
            english_response = result.get("response", "")
            logger.info("Successfully generated English response")
            
            # Translate response back to original language if not English
            if original_language != 'en':
                final_response = translate_text(english_response, target_lang=original_language, source_lang='en')
                logger.info(f"Translated response to {original_language}")
            else:
                final_response = english_response
        else:
            logger.error(f"Error from Ollama API: {response.status_code} - {response.text}")
            english_response = "Sorry, I encountered an error while processing your question."
            final_response = translate_text(english_response, target_lang=original_language, source_lang='en') if original_language != 'en' else english_response
        
        # Log the original question, English translation, and English response
        log_audit(
            session=session,
            user_email=user_email,
            query=question,  # Log original question
            query_embedding=query_embedding,
            chunks=context_chunks,
            response=english_response  # Log English response for consistency
        )

        session.close()

        return {
            "original_question": question,
            "detected_language": original_language,
            "english_question": english_question if original_language != 'en' else None,
            "context_count": len(context_chunks),
            "response": final_response,  # Return response in original language
            "top_documents": [
                {
                    "document_id": doc_id,
                    "page_number": next((chunk['page_number'] for chunk in reranked_chunks if chunk['document_id'] == doc_id), "N/A"),
                    "class_name": document_metadata.get(doc_id, {}).get("class_name", "N/A"),
                    "authors": document_metadata.get(doc_id, {}).get("authors", "N/A"),
                    "term": document_metadata.get(doc_id, {}).get("term", "N/A"),
                }
                for doc_id in top_document_ids
            ]
        }
    
    except Exception as e:
        logger.exception(f"Error in query_ollama_with_hybrid_search_multilingual: {str(e)}")
        # Try to translate error message
        error_response = "Sorry, I encountered an error while processing your question."
        if 'original_language' in locals() and original_language != 'en':
            try:
                error_response = translate_text(error_response, target_lang=original_language, source_lang='en')
            except:
                pass  # If translation fails, use English error
                
        return {
            "question": question,
            "error": str(e),
            "response": error_response
        }

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
            
            # Process the query
            result = query_ollama_with_hybrid_search_multilingual(
                session = SessionLocal(),
                question=question,
                embedding_model=embedding_model,
                vector_k=10,
                bm25_k=10,
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
            print(result['response'])

            # Display top 3 re-ranked document metadata
            top_docs = result.get('top_documents', [])
            if top_docs:
                print("\n--- Top 3 Relevant Documents ---")
                for doc in top_docs:
                    print(f"Document ID: {doc['document_id']}, Page: {doc['page_number']}, Class: {doc['class_name']}, "
                        f"Author(s): {doc['authors']}, Term: {doc['term']}")
            print("-" * 60)
        
    except Exception as e:
        logger.exception(f"Error in main function: {str(e)}")

    return None

if __name__ == "__main__":
    main()