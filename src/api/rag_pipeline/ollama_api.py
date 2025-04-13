"""
Client for Ollama API interactions.
"""
import requests
from rag_pipeline.config import logger, OLLAMA_URL, GENERATION_CONFIG

def rerank_with_llm(chunks, query, model_name):
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

def format_prompt(system_prompt, context, question, conversation_history=""):
    """
    Format a prompt for the Ollama model, now including conversation history.
    """
    return f"""
{system_prompt}

{conversation_history}

CONTEXT:
{context}

USER QUERY:
{question}

RESPONSE:
"""

def query_llm(prompt, model_name):
    """
    Query the Ollama LLM model with a formatted prompt.
    """
    try:
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
            return result.get("response", "")
        else:
            logger.error(f"Error from Ollama API: {response.status_code} - {response.text}")
            return "Sorry, I encountered an error while processing your question."
    
    except Exception as e:
        logger.exception(f"Error querying LLM: {str(e)}")
        return f"Error: {str(e)}"