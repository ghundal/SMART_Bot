import logging
import re
import requests
import nltk
from nltk.corpus import stopwords
from sqlalchemy import text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('search')

# Ollama API endpoints - needed for reranking
OLLAMA_URL = "http://localhost:11434/api/generate"
RERANKER_MODEL = "llama3.1"

def format_for_pgroonga(query: str) -> str:
    """Format query for PGroonga search by removing punctuation and stopwords."""
    stop_words = set(stopwords.words('english'))

    # Remove punctuation and lowercase
    query = re.sub(r'[^\w\s]', '', query.lower())
    
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