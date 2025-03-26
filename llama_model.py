import torch
import psycopg2
import os
import logging
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Database connection settings
DB_NAME = "smart"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = "5432"

# Ollama API endpoints
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"
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

def connect_to_postgres():
    """Connect to the postgres and return the connection."""
    try:
        return create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    except Exception as e:
        logger.exception(f"Failed to connect to postgres: {e}")
        raise RuntimeError(f"Critical failure: Unable to connect to postgres!")

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

def format_for_pgroonga(query: str) -> str:
    stop_words = set(stopwords.words('english'))
    
    # Lowercase and tokenize
    terms = query.strip().lower().split()
    
    # Remove stopwords
    keywords = [term for term in terms if term not in stop_words]
    
    if not keywords:
        return query  # fallback if everything is filtered out
    
    return " AND ".join(keywords)

def bm25_search(session, query: str, limit):
    """
    Perform a BM25 full-text search using PGroonga with SQLAlchemy.
    """
    try:
        try:
            # Construct the SQL query
            sql = """
            SELECT 
                ch.document_id, 
                ch.page_number,
                ch.chunk_text,
                pgroonga_score(ch.*) AS score
            FROM chunk ch
            JOIN document d ON ch.document_id = d.document_id
            WHERE ch.chunk_text &@~ :query
            ORDER BY score DESC
            LIMIT :limit
            """
            query = format_for_pgroonga(query)
            params = {"query": query, "limit": limit}
            
            # Execute the query
            result = session.execute(text(sql), params)
            rows = result.fetchall()
            
            # Process the results
            search_results = []
            for row in rows:
                search_results.append({
                    'document_id': row[0],
                    'page_number': row[1],
                    'chunk_text': row[2],
                    'score': row[3] if len(row) > 3 else 0
                })
                
            return search_results
            
        finally:
            session.close()
    
    except Exception as e:
        logger.exception(f"Error in BM25 search: {str(e)}")
        raise

def vector_search(session, embedding, limit, threshold: float = 0.3):
    """
    Perform vector similarity search on the chunk embeddings using SQLAlchemy.
    """
    try:
        try:
            # Construct the SQL query
            sql = f"""
            SELECT 
                ch.document_id, 
                ch.page_number,
                ch.chunk_text,
                1 - (embedding <=> '{str(embedding)}'::vector) AS similarity
            FROM chunk ch
            JOIN document d ON ch.document_id = d.document_id
            WHERE 1 - (embedding <=> '{str(embedding)}'::vector) >= :threshold
            ORDER BY similarity DESC
            LIMIT :limit
            """
            
            params = {"limit": limit, "threshold": threshold}
            
            # Execute the query
            result = session.execute(text(sql), params)
            rows = result.fetchall()
            
            # Process the results
            search_results = []
            for row in rows:
                search_results.append({
                    'document_id': row[0],
                    'page_number': row[1],
                    'chunk_text': row[2],
                    'score': row[3] if len(row) > 3 else 0
                })
                
            return search_results
            
        finally:
            session.close()
    
    except Exception as e:
        logger.exception(f"Error in Vector search: {str(e)}")
        raise

def rerank_with_llm(chunks, query, model_name=RERANKER_MODEL):
    """
    Use an LLM to rerank chunks based on relevance to the query.
    """
    try:
        reranking_results = []
        
        # Create a scoring prompt for each chunk
        for chunk in chunks:
            prompt = f"""
            Task: Evaluate the relevance of the following text to the query.
            Query: {query}
            
            Text: {chunk['chunk_text']}
            
            On a scale of 0 to 10, how relevant is the text to the query?
            Respond with only a number from 0 to 10.
            """
            
            # Prepare the request payload for the reranker
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1,  # Low temperature for consistent scoring
            }
            
            # Call Ollama API for reranking
            response = requests.post(OLLAMA_URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                # Extract the numerical score
                score_text = result.get("response", "0").strip()
                # Try to get a numerical score, default to 0 if parsing fails
                try:
                    relevance_score = float(score_text.split()[0])
                    # Ensure score is in valid range
                    relevance_score = max(0, min(10, relevance_score))
                except (ValueError, IndexError):
                    relevance_score = 0
                
                # Add to results with the LLM-assigned score
                chunk_with_score = chunk.copy()
                chunk_with_score['llm_score'] = relevance_score
                reranking_results.append(chunk_with_score)
            else:
                logger.warning(f"Error in reranking chunk: {response.status_code}")
                chunk['llm_score'] = 0
                reranking_results.append(chunk)
        
        # Sort by LLM score in descending order
        reranked_chunks = sorted(reranking_results, key=lambda x: x['llm_score'], reverse=True)
        logger.info(f"Reranked {len(reranked_chunks)} chunks using LLM")
        
        return reranked_chunks
    
    except Exception as e:
        logger.exception(f"Error in LLM reranking: {str(e)}")
        # Return original chunks if reranking fails
        return chunks

def hybrid_search(session, query, embedding, vector_k, bm25_k):
    """
    Perform a hybrid search using both vector similarity and BM25.
    """
    try:
        # Get results from vector search
        vector_results = vector_search(session, embedding, vector_k)
        logger.info(f"Retrieved {len(vector_results)} chunks using vector search")
        
        # Get results from BM25 search
        bm25_results = bm25_search(session, query, bm25_k)
        logger.info(f"Retrieved {len(bm25_results)} chunks using BM25 search")
        
        # Combine results with more robust deduplication and selection
        # Start with a dictionary to remove duplicates
        combined_chunks = {}
        
        # Add vector search results with score
        for chunk in vector_results:
            combined_chunks[chunk['chunk_text']] = {
                'chunk': chunk,
                'vector_score': chunk.get('score', 0),
                'bm25_score': 0
            }
            
        # Add or update BM25 search results
        for chunk in bm25_results:
            if chunk['chunk_text'] in combined_chunks:
                combined_chunks[chunk['chunk_text']]['bm25_score'] = chunk.get('score', 0)
            else:
                combined_chunks[chunk['chunk_text']] = {
                    'chunk': chunk,
                    'vector_score': 0,
                    'bm25_score': chunk.get('score', 0)
                }
        
        # Calculate combined score
        for text, data in combined_chunks.items():
            data['combined_score'] = data['vector_score'] + data['bm25_score']
        
        # Sort by combined score and take top chunks
        sorted_results = sorted(
            combined_chunks.values(), 
            key=lambda x: x['combined_score'], 
            reverse=True
        )
        
        # Apply LLM reranking to combined results
        reranked_chunks = rerank_with_llm([item['chunk'] for item in sorted_results[:15]], query)
        
        # Take only the top chunks overall to keep context size reasonable
        top_results = [item['chunk'] for item in sorted_results[:7]]
        
        logger.info(f"Selected {len(top_results)} top chunks for context")
        
        return top_results, reranked_chunks
    
    except Exception as e:
        logger.exception(f"Error in hybrid search: {str(e)}")
        raise

def retrieve_document_metadata(session, document_ids):
    """
    Retrieve metadata for the documents.
    """
    try:
        if not document_ids:
            return {}
        
        # Construct the SQL query to get document metadata
        sql = """
        SELECT 
            d.document_id, 
            c.class_name,
            c.authors,
            c.term
        FROM document d
        JOIN class c ON d.class_id = c.class_id
        WHERE d.document_id IN :document_ids
        """
        
        # Execute the query
        result = session.execute(text(sql), {"document_ids": tuple(document_ids)})
        
        # Process the results
        metadata = {}
        for row in result:
            metadata[row[0]] = {
                'class_name': row[1],
                'authors': row[2],
                'term': row[3]
            }
        
        return metadata
    
    except Exception as e:
        logger.exception(f"Error retrieving document metadata: {str(e)}")
        return {}
    
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

def query_ollama_with_hybrid_search(question, embedding_model, 
                                   vector_k, bm25_k, user_email = 'Anonymous',
                                   model_name=OLLAMA_MODEL):
    """
    Query the Ollama model using hybrid search to retrieve relevant context.
    """
    try:
        # Connect to database
        connection = connect_to_postgres()
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=connection)
        session = SessionLocal()
        
        # Embed the query
        query_embedding = embed_query(question, embedding_model)
        logger.info(f"Generated query embedding with {len(query_embedding)} dimensions")
        
        # Perform hybrid search with reduced chunk count
        context_chunks, reranked_chunks = hybrid_search(session, question, query_embedding, vector_k, bm25_k)

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
        
        # Format context chunks for the prompt - more concisely
        contexts = [f"DOCUMENT {i+1}:\n{chunk['chunk_text']}" for i, chunk in enumerate(context_chunks)]
        context = "\n\n".join(contexts)
        
        # Use simplified system prompt if none provided
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
        
        # Format the prompt
        prompt = format_prompt(system_prompt, context, question)
        
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
            assistant_response = result.get("response", "")
            logger.info("Successfully generated response")
        else:
            logger.error(f"Error from Ollama API: {response.status_code} - {response.text}")
            assistant_response = "Sorry, I encountered an error while processing your question."
        
        log_audit(
            session=session,
            user_email=user_email,
            query=question,
            query_embedding=query_embedding,
            chunks=context_chunks,
            response=assistant_response
        )

        session.close()

        return {
            "question": question,
            "context_count": len(context_chunks),
            "response": assistant_response,
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
        logger.exception(f"Error in query_ollama_with_hybrid_search: {str(e)}")
        return {
            "question": question,
            "error": str(e),
            "response": "Sorry, I encountered an error while processing your question."
        }

def main():
    """Main function to run an interactive Ollama query system with hybrid search."""
    try:
        print("\n===== Improved Ollama RAG Question Answering System =====")
        print("Loading models... This may take a moment.")
        
        # Load embedding model
        embedding_model = get_ch_embedding_model()
        
        print("Models loaded successfully!")
        print("Type 'quit', 'exit', or 'q' to end the session.")
        print("-" * 60)
        
        # Default model name for Ollama - using Llama 3.1 as specified
        model_name = OLLAMA_MODEL
        
        while True:
            # Get user question
            question = input("\nPlease enter your question: ").strip()
            
            # Check if user wants to quit
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using the Ollama Question Answering System. Goodbye!")
                break
            
            # Skip empty questions
            if not question:
                print("Please enter a valid question.")
                continue
            
            print(f"\nProcessing question: {question}")
            print("Searching for relevant context...")
            
            # Use the improved query function with reduced context size
            result = query_ollama_with_hybrid_search(
                question=question,
                embedding_model=embedding_model,
                vector_k = 10,
                bm25_k = 10,
                model_name=model_name
            )
            
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