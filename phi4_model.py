import torch
import psycopg2
import os
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Database connection settings
DB_NAME = "smart"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = "5432"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('phi4_model')

# Define generation config with improved parameters
GENERATION_CONFIG = {
    "max_length": 512,
    "temperature": 0.25,  # Lower temperature for more focused output
    "top_p": 0.9,
    "repetition_penalty": 1.3,
    "do_sample": True
}

# Token limits
MAX_INPUT_TOKENS = 4000  # Reduced to prevent context overload

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
        device=device  # Ensure embeddings are computed on GPU
    )
    return embedding.tolist()

def bm25_search(session, query: str, limit: int = 5):  # Reduced from 25 to 5
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

def vector_search(session, embedding, limit: int = 5):  # Reduced from 25 to 5
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
            ORDER BY similarity DESC
            LIMIT :limit
            """
            
            params = {"limit": limit}
            
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

def hybrid_search(session, query, embedding, vector_k=5, bm25_k=5):  # Reduced from 25 to 5
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
        
        # Take only the top chunks overall to keep context size reasonable
        top_results = [item['chunk'] for item in sorted_results[:7]]  # Limited to 7 chunks total
        
        logger.info(f"Selected {len(top_results)} top chunks for context")
        
        return top_results
    
    except Exception as e:
        logger.exception(f"Error in hybrid search: {str(e)}")
        raise

def count_tokens(tokenizer, text):
    """Count the number of tokens in the text using the provided tokenizer."""
    return len(tokenizer.encode(text))

def load_phi_model():
    """Load and return the Phi-4 model and tokenizer."""
    try:
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load model and tokenizer
        model_name = "microsoft/Phi-4-mini-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with optimizations for the device
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        logger.info(f"Successfully loaded Phi-4 model: {model_name}")
        return model, tokenizer
    except Exception as e:
        logger.exception(f"Error loading Phi-4 model: {e}")
        raise

def format_prompt_with_chat_markers(system_instruction, context, question):
    """Format the prompt using Phi-4's chat markers."""
    prompt = f"<|system|>\n{system_instruction}\n<|user|>\nHere is some context to help answer my question:\n\n{context}\n\nMy question is: {question}\n<|assistant|>"
    return prompt

def query_phi_with_hybrid_search(question, model, tokenizer, embedding_model,
                                vector_k, bm25_k):
    """
    Query the Phi-4 model using hybrid search to retrieve relevant context.
    Uses proper chat format and more selective context retrieval.
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
        context_chunks = hybrid_search(session, question, query_embedding, vector_k, bm25_k)
        
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
        4. For complex topics, break down your explanation into understandable components.
        5. Maintain a professional and educational tone throughout your responses.
        6. Only use information from the provided context.
        7. Keep your answers concise and to the point.
        8. If you don't know the answer, say so.
        9. Do not include unnecessary information or repetitive explanations.
        10. Format your response clearly and directly address the question.
        """
        
        # Format the prompt with chat markers
        prompt = format_prompt_with_chat_markers(system_prompt, context, question)
        
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to("cuda" if torch.cuda.is_available() else "cpu")
        attention_mask = inputs.attention_mask.to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Log token count
        logger.info(f"Input has {len(inputs.input_ids[0])} tokens")
        
        # Generate response directly with model.generate()
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=len(input_ids[0]) + GENERATION_CONFIG["max_length"],
                temperature=GENERATION_CONFIG["temperature"],
                top_p=GENERATION_CONFIG["top_p"],
                repetition_penalty=GENERATION_CONFIG["repetition_penalty"],
                do_sample=GENERATION_CONFIG["do_sample"],
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode and extract only the assistant's reply
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = full_response.split("<|assistant|>")[-1].strip()
        
        logger.info("Successfully generated response")
        
        return {
            "question": question,
            "context_count": len(context_chunks),
            "response": assistant_response
        }
    
    except Exception as e:
        logger.exception(f"Error in query_phi_with_hybrid_search: {str(e)}")
        return {
            "question": question,
            "error": str(e),
            "response": "Sorry, I encountered an error while processing your question."
        }

def main():
    """Main function to run an interactive Phi-4 query system with hybrid search."""
    try:
        print("\n===== Improved Phi-4 RAG Question Answering System =====")
        print("Loading models... This may take a moment.")

        # Load Phi model
        phi_model, tokenizer = load_phi_model()
        
        # Load embedding model
        embedding_model = get_ch_embedding_model()

        # Load tokenizer once to avoid reloading it for each question
        model_name = "microsoft/Phi-4-mini-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("Models loaded successfully!")
        print("Type 'quit', 'exit', or 'q' to end the session.")
        print("-" * 60)
        
        while True:
            # Get user question
            question = input("\nPlease enter your question: ").strip()
            
            # Check if user wants to quit
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using the Phi-4 Question Answering System. Goodbye!")
                break
            
            # Skip empty questions
            if not question:
                print("Please enter a valid question.")
                continue
            
            print(f"\nProcessing question: {question}")
            print("Searching for relevant context...")
            
            # Use the improved query function with reduced context size
            result = query_phi_with_hybrid_search(
                question=question,
                model=phi_model,
                tokenizer=tokenizer,
                embedding_model=embedding_model,
                vector_k=5,
                bm25_k=5
            )
            
            print(f"\n--- Response (from {result.get('context_count', 'unknown')} context chunks) ---")
            print(result['response'])
            print("-" * 60)
        
    except Exception as e:
        logger.exception(f"Error in main function: {str(e)}")

    return None

if __name__ == "__main__":
    main()