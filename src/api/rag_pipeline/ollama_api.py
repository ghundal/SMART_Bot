"""
Client for Ollama model interactions via API server.
"""

import re
import requests
import time
from typing import List, Dict, Any, Optional

from .config import GENERATION_CONFIG, OLLAMA_URL, logger

# Override the RERANKER_MODEL to use llama3:8b
RERANKER_MODEL = "llama3:8b"


class OllamaAPIClient:
    def __init__(self, model_name: str):
        """
        Initialize a client for Ollama model interactions via API.

        Args:
            model_name: Name of the Ollama model
        """
        self.model_name = model_name
        self.api_base = OLLAMA_URL
        logger.info(f"Initialized OllamaAPIClient for model: {model_name}")

    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        max_tokens: int = 2048,
        max_retries: int = 2,  # Add max_retries parameter with a default of 2
    ) -> str:
        """
        Generate text using Ollama model via API with retry mechanism.

        Args:
            prompt: Input text prompt
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repeat_penalty: Penalty for repeating tokens
            max_tokens: Maximum number of tokens to generate
            max_retries: Maximum number of retry attempts

        Returns:
            Generated text response
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                # Prepare the payload for the API request
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": temperature,
                    "top_p": top_p,
                    "repeat_penalty": repeat_penalty,
                    "max_tokens": max_tokens,
                    "stream": False  # We want the full response at once
                }

                # Make the API request
                logger.info(f"Sending generate request to API for model: {self.model_name}")
                response = requests.post(
                    self.api_base,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30  # Add timeout to prevent hanging requests
                )

                if response.status_code != 200:
                    error_msg = f"Error generating text: {response.status_code} - {response.text[:200]}"
                    logger.error(error_msg)
                    last_error = error_msg

                    # If we got a 404, don't retry - the model doesn't exist
                    if response.status_code == 404:
                        break

                    # Wait before retrying
                    time.sleep(1)
                    continue

                # Extract the generated text from the response
                response_data = response.json()
                api_response = response_data.get("response", "")

                # Clean the output to match the CLI behavior
                if prompt in api_response:
                    api_response = api_response[api_response.find(prompt) + len(prompt):].strip()

                return api_response

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt+1} failed: {str(e)}. {'Retrying...' if attempt < max_retries-1 else 'Giving up.'}")
                time.sleep(1)  # Brief delay before retry

        # If we get here, all attempts failed
        error_message = f"Error after {max_retries} attempts: {last_error}"
        logger.exception(error_message)
        return f"Error: {last_error}"


def rerank_with_llm(
    chunks: List[Dict[str, Any]], query: str, model_name: str = None
) -> List[Dict[str, Any]]:
    """
    Use an Ollama model to rerank chunks based on relevance to the query.
    Falls back to original ordering if API calls fail.

    Args:
        chunks: List of document chunks to rerank
        query: The user's query
        model_name: Optional model name to override the RERANKER_MODEL from config
    """
    try:
        # Use RERANKER_MODEL if no specific model provided
        if model_name is None:
            model_name = RERANKER_MODEL

        logger.info(f"Reranking using model: {model_name}")
        logger.info(f"Using prompt-based reranking with model: {model_name}")

        # Initialize API client
        model_client = OllamaAPIClient(model_name)

        # Try a test call to see if the model is working
        test_prompt = "This is a test."
        try:
            test_response = model_client.generate_text(
                prompt=test_prompt,
                temperature=0.1,
                max_tokens=10,
                max_retries=1  # Only try once for the test
            )
            # Check if we got an error response
            if test_response.startswith("Error:"):
                logger.warning(f"Reranking model test failed: {test_response}. Using original chunk order.")
                return chunks
        except Exception as e:
            logger.warning(f"Reranking model not available: {str(e)}. Using original chunk order.")
            return chunks

        # If we got here, the test was successful - proceed with reranking
        reranking_results = []
        max_failures = 3  # Maximum number of consecutive failures before giving up
        failures = 0

        # Create a scoring prompt for each chunk
        for chunk in chunks:
            if failures >= max_failures:
                logger.warning(f"Too many consecutive failures ({failures}). Stopping reranking.")
                break

            prompt = f"""
Task: Evaluate the relevance of the following text to the query.
Query: {query}
Text: {chunk['chunk_text']}
On a scale of 0 to 10, how relevant is the text to the query?
Respond with only a number from 0 to 10.
"""
            # Generate score using API model
            score_text = model_client.generate_text(
                prompt=prompt,
                temperature=0.1,  # Low temperature for consistent scoring
                max_tokens=50,  # We only need a short response
            )

            # Check if we got an error
            if score_text.startswith("Error:"):
                failures += 1
                # Assign a default score to keep the process going
                relevance_score = 5.0  # Neutral score
            else:
                failures = 0  # Reset failure counter on success
                # Try to get a numerical score, default to 5 if parsing fails
                try:
                    # Extract the first number from the response
                    numbers = re.findall(r"\d+(?:\.\d+)?", score_text)
                    relevance_score = float(numbers[0]) if numbers else 5.0
                    # Ensure score is in valid range
                    relevance_score = max(0, min(10, relevance_score))
                except (ValueError, IndexError):
                    relevance_score = 5.0

            # Add to results with the LLM-assigned score
            chunk_with_score = chunk.copy()
            chunk_with_score["llm_score"] = relevance_score
            reranking_results.append(chunk_with_score)

        # If we have results, sort them
        if reranking_results:
            # Sort by LLM score in descending order
            reranked_chunks = sorted(reranking_results, key=lambda x: x["llm_score"], reverse=True)
            logger.info(f"Reranked {len(reranked_chunks)} chunks using Ollama model: {model_name}")
            return reranked_chunks
        else:
            # Return original chunks if no results
            return chunks

    except Exception as e:
        logger.exception(f"Error in LLM reranking: {str(e)}")
        # Return original chunks if reranking fails
        return chunks


def format_prompt(
    system_prompt: str, context: str, question: str, conversation_history: str = ""
) -> str:
    """
    Format a prompt for the Ollama model, including conversation history.
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


def query_llm(prompt: str, model_name: str) -> str:
    """
    Query the Ollama model with a formatted prompt.
    """
    try:
        # Initialize Ollama API client
        model_client = OllamaAPIClient(model_name)

        # Generate response using API model
        response = model_client.generate_text(
            prompt=prompt,
            temperature=GENERATION_CONFIG["temperature"],
            top_p=GENERATION_CONFIG["top_p"],
            repeat_penalty=GENERATION_CONFIG["repeat_penalty"],
            max_retries=2  # Add retries to make it more robust
        )

        return response
    except Exception as e:
        logger.exception(f"Error querying LLM: {str(e)}")
        return f"Error: {str(e)}"
