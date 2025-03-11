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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LLM')

# Define generation config
GENERATION_CONFIG = {
    "max_new_tokens": 8192,
    "temperature": 0.1,
    "top_p": 0.8,
    "repetition_penalty": 2.0,
    "do_sample": True
}

# Token limits
MAX_INPUT_TOKENS = 20000 
SYSTEM_PROMPT_TOKENS = 500

# Define system instruction for ML/Data Science expertise
SYSTEM_INSTRUCTION = """
You are an AI assistant specialized in machine learning, deep learning, and data science. Your responses are based solely on the information provided in the text chunks given to you. Do not use any external knowledge or make assumptions beyond what is explicitly stated in these chunks.
When answering a query:
1. Carefully read all the text chunks provided.
2. Identify the most relevant information from these chunks to address the user's question.
3. Formulate your response prioritizing the chunk information but enhnaced with outside knowledge.
4. If the provided chunks do not contain sufficient information to answer the query, state that you don't have enough information to provide a complete answer.
5. Always maintain a professional and knowledgeable tone, befitting a data science expert.
6. If there are contradictions in the provided chunks, mention this in your response and explain the different viewpoints presented.
Remember:
- You are an expert in machine learning, deep learning, and data science, but your knowledge is limited to the information in the provided chunks.
- Do not invent information or draw from knowledge outside the chunks.
- If asked about topics unrelated to data science or machine learning, politely redirect the conversation back to related subjects.
- Be concise in your responses while ensuring you cover all relevant information from the chunks.
Your goal is to provide accurate, helpful information about machine learning, deep learning, and data science based solely on the content of the text chunks you receive with each query.
"""

def get_ch_embedding_model():
    """Load and return the embedding model."""
    try:
        model_name = 'all-mpnet-base-v2'
        # model_name = 'multi-qa-mpnet-base-dot-v1'
        model = SentenceTransformer(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Successfully loaded Embedding model: {model_name}")
        return model
    except Exception as e:
        logger.exception(f"Error loading Embedding model: {model_name}")
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

def bm25_search(session, query: str, limit: int = 25):
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
                })
                
            return search_results
            
        finally:
            session.close()
    
    except Exception as e:
        logger.exception(f"Error in BM25 search: {str(e)}")
        raise

def vector_search(session, embedding, limit: int = 25):
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
                })
                
            return search_results
            
        finally:
            session.close()
    
    except Exception as e:
        logger.exception(f"Error in Vector search: {str(e)}")
        raise

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
            device_map=device,
            trust_remote_code=True
        )
        
        logger.info(f"Successfully loaded Phi-4 model: {model_name}")
        return model, tokenizer, device
    except Exception as e:
        logger.exception(f"Error loading Phi-4 model: {e}")
        raise

def format_prompt_with_system_instruction(question, context=None):
    """Format prompt with system instruction for Phi-4"""
    structured_question = question
    
    # For certain question types, add more structure
    if any(keyword in question.lower() for keyword in ["what is", "explain", "describe", "define"]):
        structured_question = f"""
            Please provide a structured response with the following sections:
            1. DEFINITION
            2. COMPONENTS 
            3. ARCHITECTURE
            4. APPLICATIONS (if applicable)
            5. CONCLUSION

            Original question: {question}
        """
    
    if context:
        return f"<|system|>\n{SYSTEM_INSTRUCTION}\n<|user|>\nContext: {context}\n\nQuestion: {structured_question}\n<|assistant|>"
    else:
        return f"<|system|>\n{SYSTEM_INSTRUCTION}\n<|user|>\n{structured_question}\n<|assistant|>"

def count_tokens(text, tokenizer):
    """Count the number of tokens in the given text."""
    return len(tokenizer.encode(text))

def get_model_response(model, tokenizer, device, question, context):
    """Generate a response from the model based on the question and context."""
    try:
        # Format the prompt with system instruction and context
        prompt = format_prompt_with_system_instruction(question, context)
        
        # Tokenize the prompt with proper attention mask
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        # Log token count for transparency
        token_count = input_ids.size(1)
        logger.info(f"Input token count: {token_count}")
        
        # Generate answer
        logger.info("Generating model response...")
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                **GENERATION_CONFIG,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode and extract only the assistant's reply
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = full_response.split("<|assistant|>")[-1].strip()
        
        return assistant_response
    except Exception as e:
        logger.exception(f"Error generating model response: {e}")
        return "An error occurred while generating the response."

def limit_context_by_tokens(chunks, tokenizer, max_tokens):
    """
    Limit the context by using only as many chunks as will fit within the token limit.
    Assumes chunks are already sorted by relevance (highest score first).
    """
    context_parts = []
    current_tokens = 0
    
    # Sort chunks by score in descending order to prioritize most relevant chunks
    sorted_chunks = sorted(chunks, key=lambda x: x.get('score', 0), reverse=True)
    
    used_chunk_count = 0
    for chunk in sorted_chunks:
        chunk_text = chunk['chunk_text']
        chunk_tokens = count_tokens(chunk_text, tokenizer)
        
        # Check if adding this chunk would exceed the limit
        if current_tokens + chunk_tokens + 2 <= max_tokens:  # +2 for separator tokens
            context_parts.append(chunk_text)
            current_tokens += chunk_tokens + 2  # Count separator tokens
            used_chunk_count += 1
        else:
            # Stop adding chunks once we reach the token limit
            break
    
    logger.info(f"Using {used_chunk_count} chunks out of {len(chunks)} (token limit: {max_tokens})")
    return "\n---\n".join(context_parts)

def hybrid_search(session, query, embedding, vector_k=25, bm25_k=25):
    """
    Perform a hybrid search using both vector similarity and BM25.
    
    Args:
        session: Database session
        query (str): The user's question
        embedding: Query embedding vector
        vector_k (int): Number of results to retrieve from vector search
        bm25_k (int): Number of results to retrieve from BM25 search
        
    Returns:
        list: Combined unique chunks from both search methods
    """
    try:
        # Get results from vector search
        vector_results = vector_search(session, embedding, vector_k)
        logger.info(f"Retrieved {len(vector_results)} chunks using vector search")
        
        # Get results from BM25 search
        bm25_results = bm25_search(session, query, bm25_k)
        logger.info(f"Retrieved {len(bm25_results)} chunks using BM25 search")
        
        # Combine results, removing duplicates
        # Create a dictionary with chunk_text as key to remove duplicates
        combined_chunks = {}
        
        # Add vector search results
        for chunk in vector_results:
            combined_chunks[chunk['chunk_text']] = chunk
            
        # Add BM25 search results
        for chunk in bm25_results:
            combined_chunks[chunk['chunk_text']] = chunk
            
        # Convert back to list
        unique_chunks = list(combined_chunks.values())
        logger.info(f"Combined into {len(unique_chunks)} unique chunks")
        
        return unique_chunks
    
    except Exception as e:
        logger.exception(f"Error in hybrid search: {str(e)}")
        raise

def retrieve_and_generate(query, embedding_model, phi_model, tokenizer, device, top_k=25):
    """
    Main function to retrieve relevant chunks and generate a response.
    """
    try:        
        # Connect to database
        engine = connect_to_postgres()
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()
        
        # Generate embedding for query
        query_embedding = embed_query(query, embedding_model)
        
        # Get relevant chunks
        chunks = hybrid_search(session, query, query_embedding, top_k, top_k)
        
        # Calculate available token budget for context
        # Subtract tokens needed for system prompt and user question from max input tokens
        question_tokens = count_tokens(query, tokenizer)
        available_context_tokens = MAX_INPUT_TOKENS - SYSTEM_PROMPT_TOKENS - question_tokens
        
        # Limit context to fit within token budget
        context = limit_context_by_tokens(chunks, tokenizer, available_context_tokens)
        
        # Generate response
        response = get_model_response(phi_model, tokenizer, device, query, context)
        
        return response
    
    except Exception as e:
        logger.exception(f"Error in retrieve_and_generate: {e}")
        return "An error occurred during the retrieve and generate process."

def main():

    # Load models
    embedding_model = get_ch_embedding_model()
    phi_model, tokenizer, device = load_phi_model()

    print("\n===== Retrieval Augmented Generation (RAG) System =====")
    print("Type your machine learning and data science questions below.")
    print("Type 'exit', 'quit', or 'q' to end the program.")
    print("Type 'vector', 'bm25', or 'hybrid' to switch search methods (default: hybrid).\n")
    
    search_method = "hybrid"
    
    while True:
        # Get user question
        user_input = input("Question: ")
        
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        
        # Processing the query and get response
        print("Processing query using hybrid search method...")
        response = retrieve_and_generate(user_input, embedding_model, phi_model, tokenizer, device,)
        
        # Print the response
        print("\nAnswer:")
        print(response)
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()