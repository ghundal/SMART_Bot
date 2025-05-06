"""
Async Ollama API Client for Text Generation and Reranking (RAG System)

This module defines an asynchronous client (`AsyncOllamaAPIClient`) for interacting with
the Ollama API to perform:

- Text generation using LLMs via async HTTP requests.
- Prompt-based reranking of document chunks for query relevance scoring.
- Prompt formatting for conversational query responses.

Key Features:
- Extended timeout handling (default 10 minutes) for long LLM operations.
- Graceful error recovery and fallback behavior.
- Designed for integration into RAG (retrieval-augmented generation) pipelines.
"""

import re
import aiohttp
import asyncio
from typing import List, Dict, Any

from .config import GENERATION_CONFIG, OLLAMA_URL, RERANKER_MODEL, logger

# Configure longer timeouts (10 minutes = 600 seconds)
DEFAULT_TIMEOUT = 600  # 10 minutes in seconds


class AsyncOllamaAPIClient:
    def __init__(self, model_name: str, timeout: int = DEFAULT_TIMEOUT):
        """
        Initialize an async client for Ollama model interactions via API.

        Args:
            model_name: Name of the Ollama model
            timeout: Timeout in seconds for API requests (default: 10 minutes)
        """
        self.model_name = model_name
        self.api_base = OLLAMA_URL
        self.timeout = timeout
        logger.info(
            f"Initialized AsyncOllamaAPIClient for model: {model_name} with {timeout}s timeout"
        )

    async def generate_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        max_tokens: int = 2048,
        timeout: int = None,
    ) -> str:
        """
        Generate text using Ollama model via API.

        Args:
            prompt: Input text prompt
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repeat_penalty: Penalty for repeating tokens
            max_tokens: Maximum number of tokens to generate
            timeout: Optional request-specific timeout in seconds (overrides default)

        Returns:
            Generated text response
        """
        try:
            # Use provided timeout or fall back to instance default
            request_timeout = timeout if timeout is not None else self.timeout

            # Prepare the payload for the API request
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "repeat_penalty": repeat_penalty,
                "max_tokens": max_tokens,
                "stream": False,  # We want the full response at once
            }

            # Make the API request
            logger.info(
                f"Sending generate request to API for model: {self.model_name} with {request_timeout}s timeout"
            )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_base,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=request_timeout),  # Extended timeout
                ) as response:
                    if response.status != 200:
                        error_msg = (
                            f"Error generating text: {response.status} - {await response.text()}"
                        )
                        logger.error(error_msg)
                        return f"Error: {error_msg}"

                    # Extract the generated text from the response
                    response_data = await response.json()
                    api_response = response_data.get("response", "")

                    # Clean the output to match the CLI behavior
                    if prompt in api_response:
                        api_response = api_response[
                            api_response.find(prompt) + len(prompt) :
                        ].strip()

                    return api_response

        except aiohttp.ClientError as ce:
            error_message = f"Network error in generate_text: {str(ce)}"
            logger.exception(error_message)
            return f"Error: {error_message}"
        except asyncio.TimeoutError:
            error_message = f"Request timed out after {request_timeout} seconds"
            logger.exception(error_message)
            return f"Error: {error_message}"
        except Exception as e:
            error_message = f"Error in generate_text: {str(e)}"
            logger.exception(error_message)
            return f"Error: {str(e)}"


async def rerank_with_llm(
    chunks: List[Dict[str, Any]], query: str, model_name: str = None, timeout: int = DEFAULT_TIMEOUT
) -> List[Dict[str, Any]]:
    """
    Use an Ollama model to rerank chunks based on relevance to the query.
    Falls back to original ordering if API calls fail.

    Args:
        chunks: List of document chunks to rerank
        query: The user's query
        model_name: Optional model name to override the RERANKER_MODEL from config
        timeout: Timeout in seconds for API requests (default: 10 minutes)
    """
    try:
        # Use RERANKER_MODEL if no specific model provided
        if model_name is None:
            model_name = RERANKER_MODEL

        logger.info(f"Reranking using model: {model_name} with {timeout}s timeout")
        logger.info(f"Using prompt-based reranking with model: {model_name}")

        # Initialize API client with extended timeout
        model_client = AsyncOllamaAPIClient(model_name, timeout=timeout)

        # Try a test call to see if the model is working
        test_prompt = "This is a test."
        try:
            test_response = await model_client.generate_text(
                prompt=test_prompt,
                temperature=0.1,
                max_tokens=10,
            )
            # Check if we got an error response
            if test_response.startswith("Error:"):
                logger.warning(
                    f"Reranking model test failed: {test_response}. Using original chunk order."
                )
                return chunks
        except Exception as e:
            logger.warning(f"Reranking model not available: {str(e)}. Using original chunk order.")
            return chunks

        # If we got here, the test was successful - proceed with reranking
        reranking_results = []

        # Process chunks one at a time sequentially
        for i, chunk in enumerate(chunks):
            try:
                prompt = f"""
Task: Evaluate the relevance of the following text to the query.
Query: {query}
Text: {chunk['chunk_text']}
On a scale of 0 to 10, how relevant is the text to the query?
Respond with only a number from 0 to 10.
"""
                # Process each chunk individually
                score_text = await model_client.generate_text(
                    prompt=prompt,
                    temperature=0.1,  # Low temperature for consistent scoring
                    max_tokens=50,  # We only need a short response
                )

                # Create a copy of the chunk to add the score
                chunk_with_score = chunk.copy()

                # Check if we got an error
                if score_text.startswith("Error:"):
                    logger.error(f"Error response for chunk {i}: {score_text}")
                    relevance_score = 5.0  # Default score on error
                else:
                    # Try to get a numerical score, default to 5 if parsing fails
                    try:
                        # Extract the first number from the response
                        numbers = re.findall(r"\d+(?:\.\d+)?", score_text)
                        relevance_score = float(numbers[0]) if numbers else 5.0
                        # Ensure score is in valid range
                        relevance_score = max(0, min(10, relevance_score))
                    except (ValueError, IndexError) as e:
                        logger.error(f"Failed to parse score for chunk {i}: {str(e)}")
                        relevance_score = 5.0  # Default on parsing error

                # Add score to the chunk
                chunk_with_score["llm_score"] = relevance_score
                reranking_results.append(chunk_with_score)

            except Exception as e:
                logger.error(f"Error processing chunk {i}: {str(e)}")
                # Add the original chunk with a neutral score on error
                chunk_with_score = chunk.copy()
                chunk_with_score["llm_score"] = 5.0
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


async def query_llm(prompt: str, model_name: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """
    Query the Ollama model with a formatted prompt.

    Args:
        prompt: The formatted prompt to send to the model
        model_name: Name of the Ollama model to use
        timeout: Timeout in seconds for API request (default: 10 minutes)
    """
    try:
        # Initialize Ollama API client with extended timeout
        model_client = AsyncOllamaAPIClient(model_name, timeout=timeout)

        # Generate response using API model
        response = await model_client.generate_text(
            prompt=prompt,
            temperature=GENERATION_CONFIG["temperature"],
            top_p=GENERATION_CONFIG["top_p"],
            repeat_penalty=GENERATION_CONFIG["repeat_penalty"],
        )

        return response
    except Exception as e:
        logger.exception(f"Error querying LLM: {str(e)}")
        return f"Error: {str(e)}"
