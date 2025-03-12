import psycopg2
import os
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, relationship, Session

# Database connection settings
DB_NAME = "smart"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = "5432"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_ch_embedding_model():
    """Load and return the embedding model."""
    try:
        model_name = 'all-mpnet-base-v2'  # Alternative: 'multi-qa-mpnet-base-dot-v1'
        model = SentenceTransformer(model_name)
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
        logger.exception(f"Failed to connect to postgres!")
        raise RuntimeError(f"Critical failure: Unable to connect to postgres!")

def embed_query(query, model):
    """Generate an embedding for the given query."""
    return model.encode(query).tolist()

def bm25_search(session, query: str, limit: int = 10):
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

def vector_search(session, embedding, limit: int = 10):
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


def main():

    # Set connection to postgres
    connection = connect_to_postgres()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=connection)
    session = SessionLocal()
    query = "What is a CNN?"

    model = get_ch_embedding_model()
    embedding = model.encode(
            list(query)
        )
    results = bm25_search(session, query)
    vector_result = vector_search(session, embedding.tolist()[0])
    print(vector_result)

    return None

if __name__ == "__main__":
    main()