"""
Client for local Ollama model interactions without API calls.
"""

import os
import subprocess
import tempfile
from typing import List, Dict, Any

from .config import GENERATION_CONFIG, logger

# Import the reranker from the new module
from .transformer_reranker import rerank_chunks


class OllamaLocalClient:
    def __init__(self, model_name: str):
        """
        Initialize a client for local Ollama model interactions.

        Args:
            model_name: Name of the downloaded Ollama model
        """
        self.model_name = model_name
        self._ensure_model_exists()

    def _ensure_model_exists(self):
        """Check if the model exists locally."""
        try:
            # Run ollama list to check available models
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)

            # Check if our model is in the list
            if self.model_name not in result.stdout:
                logger.warning(
                    f"Model {self.model_name} not found locally. Make sure it's downloaded."
                )
                logger.info("Available models: " + result.stdout)
        except subprocess.CalledProcessError as e:
            logger.exception(f"Error checking Ollama models: {str(e)}")
        except FileNotFoundError:
            logger.error(
                "Ollama command not found. Make sure Ollama is installed and in your PATH."
            )

    def _create_temp_model(
        self, temperature: float = 0.7, top_p: float = 0.9, repeat_penalty: float = 1.1
    ) -> str:
        """
        Create a temporary model with the specified parameters.

        Returns:
            The name of the temporary model
        """
        try:
            # Create a unique model name
            temp_model_name = f"{self.model_name}-temp-{os.getpid()}"

            # Create a temporary modelfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".modelfile", delete=False) as f:
                modelfile_path = f.name
                f.write(
                    f"""
FROM {self.model_name}
PARAMETER temperature {temperature}
PARAMETER top_p {top_p}
PARAMETER repeat_penalty {repeat_penalty}
"""
                )

            # Create the temporary model
            result = subprocess.run(
                ["ollama", "create", temp_model_name, "-f", modelfile_path],
                capture_output=True,
                text=True,
            )

            # Remove the temporary modelfile
            os.unlink(modelfile_path)

            if result.returncode != 0:
                logger.error(f"Error creating temporary model: {result.stderr}")
                return self.model_name  # Fallback to original model

            return temp_model_name
        except Exception as e:
            logger.exception(f"Error creating temporary model: {str(e)}")
            return self.model_name  # Fallback to original model

    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate text using local Ollama model via CLI.

        Args:
            prompt: Input text prompt
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repeat_penalty: Penalty for repeating tokens
            max_tokens: Maximum number of tokens to generate

        Returns:
            Generated text response
        """
        try:
            # Create a temporary model with the specified parameters
            temp_model_name = self._create_temp_model(temperature, top_p, repeat_penalty)

            # Execute ollama run command with the temporary model
            cmd = ["ollama", "run", temp_model_name, prompt]

            # Execute ollama run command
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Clean up the temporary model
            cleanup_result = subprocess.run(
                ["ollama", "rm", temp_model_name], capture_output=True, text=True
            )

            if cleanup_result.returncode != 0:
                logger.warning(
                    f"Failed to remove temporary model {temp_model_name}: {cleanup_result.stderr}"
                )

            if result.returncode != 0:
                logger.error(f"Error generating text: {result.stderr}")
                return f"Error: {result.stderr}"

            # Clean the output - Ollama CLI may include additional info
            response = result.stdout.strip()

            # Try to extract just the model's response
            if prompt in response:
                response = response[response.find(prompt) + len(prompt) :].strip()

            return response
        except Exception as e:
            logger.exception(f"Error generating text: {str(e)}")
            return f"Error: {str(e)}"


def rerank_with_llm(
    chunks: List[Dict[str, Any]], query: str, model_name: str
) -> List[Dict[str, Any]]:
    """
    Use a local Ollama model to rerank chunks based on relevance to the query.

    This function detects if the model_name is a reranker model and uses the appropriate method:
    - For reranker models (like qllama/bge-reranker-large): Uses direct comparison
    - For LLM models: Uses prompt-based scoring
    """
    try:
        # Check if the model is a reranker model
        reranker_model_names = ["bge-reranker", "rerank", "colbert", "msmarco"]

        # If the model name contains any reranker keywords, use direct reranking
        if any(reranker_term in model_name.lower() for reranker_term in reranker_model_names):
            logger.info(f"Using direct transformer reranking with model: {model_name}")
            return rerank_chunks(chunks, query, model_name)

        # Otherwise use the original prompt-based approach
        logger.info(f"Using prompt-based reranking with model: {model_name}")
        reranking_results = []
        # Initialize local Ollama client
        model_client = OllamaLocalClient(model_name)

        # Create a scoring prompt for each chunk
        for chunk in chunks:
            prompt = f"""
Task: Evaluate the relevance of the following text to the query.
Query: {query}
Text: {chunk['chunk_text']}
On a scale of 0 to 10, how relevant is the text to the query?
Respond with only a number from 0 to 10.
"""
            # Generate score using local model
            score_text = model_client.generate_text(
                prompt=prompt,
                temperature=0.1,  # Low temperature for consistent scoring
                max_tokens=50,  # We only need a short response
            )

            # Try to get a numerical score, default to 0 if parsing fails
            try:
                # Extract the first number from the response
                import re

                numbers = re.findall(r"\d+(?:\.\d+)?", score_text)
                relevance_score = float(numbers[0]) if numbers else 0
                # Ensure score is in valid range
                relevance_score = max(0, min(10, relevance_score))
            except (ValueError, IndexError):
                relevance_score = 0

            # Add to results with the LLM-assigned score
            chunk_with_score = chunk.copy()
            chunk_with_score["llm_score"] = relevance_score
            reranking_results.append(chunk_with_score)

        # Sort by LLM score in descending order
        reranked_chunks = sorted(reranking_results, key=lambda x: x["llm_score"], reverse=True)
        logger.info(f"Reranked {len(reranked_chunks)} chunks using local Ollama model")
        return reranked_chunks
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
    Query the local Ollama model with a formatted prompt.
    """
    try:
        # Initialize local Ollama client
        model_client = OllamaLocalClient(model_name)

        # Generate response using local model
        response = model_client.generate_text(
            prompt=prompt,
            temperature=GENERATION_CONFIG["temperature"],
            top_p=GENERATION_CONFIG["top_p"],
            repeat_penalty=GENERATION_CONFIG["repeat_penalty"],
        )

        return response
    except Exception as e:
        logger.exception(f"Error querying local LLM: {str(e)}")
        return f"Error: {str(e)}"
