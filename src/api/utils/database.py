"""
Establishes a PostgreSQL database connection and provides a logging function
that records detailed query information in an audit table for tracking
and compliance purposes.
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from rag_pipeline.config import logger

def connect_to_postgres():
    """
    Creates and returns a connection to the PostgreSQL database
    """
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "smart")
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "postgres")
    
    # Create connection string
    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    # Create and return engine
    return create_engine(connection_string)

def log_audit(session, user_email, query, query_embedding, chunks, response):
    """
    Log audit information for a query.
    """
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

# Get the engine and SessionLocal from the database connection
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=connect_to_postgres())
