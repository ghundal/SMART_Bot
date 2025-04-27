import os
import sys
import unittest
from unittest.mock import ANY, MagicMock, call, patch

# Add the src directory to the path so we can import our module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the module to test
from src.api.rag_pipeline.ollama_api import (
    OllamaLocalClient,
    format_prompt,
    query_llm,
    rerank_with_llm,
)


class TestOllamaLocalClient(unittest.TestCase):
    """Test cases for the OllamaLocalClient class."""

    def setUp(self):
        """Set up test fixtures."""
        self.model_name = "llama3:8b"

        # Mock subprocess responses
        self.mock_list_result = MagicMock()
        self.mock_list_result.stdout = (
            f"NAME             ID        SIZE  \n{self.model_name}  abc123  4.4 GB"
        )
        self.mock_list_result.returncode = 0

    @patch("src.api.rag_pipeline.ollama_api.subprocess.run")
    def test_init_and_model_exists(self, mock_run):
        """Test initialization and model existence check."""
        # Configure mock
        mock_run.return_value = self.mock_list_result

        # Create client
        client = OllamaLocalClient(self.model_name)

        # Verify subprocess was called correctly
        mock_run.assert_called_once_with(
            ["ollama", "list"], capture_output=True, text=True, check=True
        )

        # Verify model was stored
        self.assertEqual(client.model_name, self.model_name)

    @patch("src.api.rag_pipeline.ollama_api.subprocess.run")
    def test_model_not_found(self, mock_run):
        """Test warning when model is not found."""
        # Configure mock to return a list without our model
        mock_result = MagicMock()
        mock_result.stdout = "NAME             ID        SIZE  \ndifferent-model  abc123  4.4 GB"
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        # Create client - should log a warning but not fail
        with self.assertLogs(level="WARNING") as cm:
            client = OllamaLocalClient(self.model_name)

            # Verify warning was logged
            self.assertTrue(
                any(f"Model {self.model_name} not found locally" in msg for msg in cm.output)
            )

    @patch("src.api.rag_pipeline.ollama_api.subprocess.run")
    def test_ollama_command_not_found(self, mock_run):
        """Test handling of missing Ollama command."""
        # Configure mock to raise FileNotFoundError
        mock_run.side_effect = FileNotFoundError("No such file or directory: 'ollama'")

        # Create client - should log an error but not fail
        with self.assertLogs(level="ERROR") as cm:
            client = OllamaLocalClient(self.model_name)

            # Verify error was logged
            self.assertTrue(any("Ollama command not found" in msg for msg in cm.output))

    @patch("src.api.rag_pipeline.ollama_api.tempfile.NamedTemporaryFile")
    @patch("src.api.rag_pipeline.ollama_api.subprocess.run")
    @patch("src.api.rag_pipeline.ollama_api.os.unlink")
    @patch("src.api.rag_pipeline.ollama_api.os.getpid")
    def test_create_temp_model(self, mock_getpid, mock_unlink, mock_run, mock_tempfile):
        """Test creation of temporary model with parameters."""
        # Configure mocks
        mock_getpid.return_value = 12345

        # Mock temporary file
        mock_file = MagicMock()
        mock_file.name = "/tmp/model12345.modelfile"
        mock_tempfile.return_value.__enter__.return_value = mock_file

        # Mock subprocess call to create model
        mock_create_result = MagicMock()
        mock_create_result.returncode = 0
        mock_run.return_value = mock_create_result

        # Create client
        with patch("src.api.rag_pipeline.ollama_api.subprocess.run"):
            client = OllamaLocalClient(self.model_name)

        # Call method
        temp_model_name = client._create_temp_model(temperature=0.5, top_p=0.8, repeat_penalty=1.2)

        # Verify temporary file was written correctly
        mock_file.write.assert_called_once()
        write_arg = mock_file.write.call_args[0][0]
        self.assertIn(f"FROM {self.model_name}", write_arg)
        self.assertIn("PARAMETER temperature 0.5", write_arg)
        self.assertIn("PARAMETER top_p 0.8", write_arg)
        self.assertIn("PARAMETER repeat_penalty 1.2", write_arg)

        # Verify subprocess was called correctly
        expected_temp_name = f"{self.model_name}-temp-12345"
        mock_run.assert_called_with(
            ["ollama", "create", expected_temp_name, "-f", "/tmp/model12345.modelfile"],
            capture_output=True,
            text=True,
        )

        # Verify file was cleaned up
        mock_unlink.assert_called_once_with("/tmp/model12345.modelfile")

        # Verify correct name was returned
        self.assertEqual(temp_model_name, expected_temp_name)

    @patch("src.api.rag_pipeline.ollama_api.tempfile.NamedTemporaryFile")
    @patch("src.api.rag_pipeline.ollama_api.subprocess.run")
    @patch("src.api.rag_pipeline.ollama_api.os.unlink")
    def test_create_temp_model_error(self, mock_unlink, mock_run, mock_tempfile):
        """Test error handling in temporary model creation."""
        # Configure mocks
        mock_file = MagicMock()
        mock_file.name = "/tmp/model12345.modelfile"
        mock_tempfile.return_value.__enter__.return_value = mock_file

        # Mock subprocess call to fail
        mock_create_result = MagicMock()
        mock_create_result.returncode = 1
        mock_create_result.stderr = "Error creating model"
        mock_run.return_value = mock_create_result

        # Create client
        with patch("src.api.rag_pipeline.ollama_api.subprocess.run"):
            client = OllamaLocalClient(self.model_name)

        # Call method - should log error and return original model name
        with self.assertLogs(level="ERROR") as cm:
            temp_model_name = client._create_temp_model()

            # Verify error was logged
            self.assertTrue(any("Error creating temporary model" in msg for msg in cm.output))

        # Verify original model name was returned as fallback
        self.assertEqual(temp_model_name, self.model_name)

    @patch.object(OllamaLocalClient, "_create_temp_model")
    @patch("src.api.rag_pipeline.ollama_api.subprocess.run")
    def test_generate_text(self, mock_run, mock_create_temp_model):
        """Test text generation using local Ollama model."""
        # Configure mocks
        mock_create_temp_model.return_value = "llama3:8b-temp-12345"

        mock_generate_result = MagicMock()
        mock_generate_result.returncode = 0
        mock_generate_result.stdout = "Model response text"

        mock_cleanup_result = MagicMock()
        mock_cleanup_result.returncode = 0

        # Set up mock_run to return different values for different calls
        mock_run.side_effect = [mock_generate_result, mock_cleanup_result]

        # Create client
        with patch("src.api.rag_pipeline.ollama_api.subprocess.run"):
            client = OllamaLocalClient(self.model_name)

        # Call method
        response = client.generate_text(
            prompt="What is the capital of France?", temperature=0.7, top_p=0.9
        )

        # Verify temp model was created
        mock_create_temp_model.assert_called_once_with(0.7, 0.9, 1.1)

        # Verify subprocess calls
        self.assertEqual(mock_run.call_count, 2)
        # First call - generate text
        mock_run.assert_any_call(
            ["ollama", "run", "llama3:8b-temp-12345", "What is the capital of France?"],
            capture_output=True,
            text=True,
        )
        # Second call - cleanup
        mock_run.assert_any_call(
            ["ollama", "rm", "llama3:8b-temp-12345"], capture_output=True, text=True
        )

        # Verify response
        self.assertEqual(response, "Model response text")

    @patch.object(OllamaLocalClient, "_create_temp_model")
    @patch("src.api.rag_pipeline.ollama_api.subprocess.run")
    def test_generate_text_error(self, mock_run, mock_create_temp_model):
        """Test error handling in text generation."""
        # Configure mocks
        mock_create_temp_model.return_value = "llama3:8b-temp-12345"

        mock_generate_result = MagicMock()
        mock_generate_result.returncode = 1
        mock_generate_result.stderr = "Generation error"

        mock_cleanup_result = MagicMock()
        mock_cleanup_result.returncode = 0

        # Set up mock_run to return different values for different calls
        mock_run.side_effect = [mock_generate_result, mock_cleanup_result]

        # Create client
        with patch("src.api.rag_pipeline.ollama_api.subprocess.run"):
            client = OllamaLocalClient(self.model_name)

        # Call method - should log error and return error message
        with self.assertLogs(level="ERROR") as cm:
            response = client.generate_text("Failed prompt")

            # Verify error was logged
            self.assertTrue(any("Error generating text" in msg for msg in cm.output))

        # Verify response contains error
        self.assertEqual(response, "Error: Generation error")

    @patch("src.api.rag_pipeline.ollama_api.OllamaLocalClient")
    def test_rerank_with_llm(self, mock_ollama_client_class):
        """Test LLM-based reranking of chunks."""
        # Configure mock
        mock_client = MagicMock()
        mock_client.generate_text.side_effect = ["7", "5", "9"]
        mock_ollama_client_class.return_value = mock_client

        # Test data
        chunks = [
            {"chunk_id": 1, "chunk_text": "Information about Paris"},
            {"chunk_id": 2, "chunk_text": "Information about London"},
            {"chunk_id": 3, "chunk_text": "Information about the Eiffel Tower"},
        ]

        # Call function
        reranked = rerank_with_llm(chunks, "Tell me about Paris", "llama3:8b")

        # Verify client was created with correct model
        mock_ollama_client_class.assert_called_once_with("llama3:8b")

        # Verify generate_text was called for each chunk
        self.assertEqual(mock_client.generate_text.call_count, 3)

        # Verify reranking - should be in order of scores (9, 7, 5)
        self.assertEqual(reranked[0]["chunk_id"], 3)  # Score 9
        self.assertEqual(reranked[1]["chunk_id"], 1)  # Score 7
        self.assertEqual(reranked[2]["chunk_id"], 2)  # Score 5

        # Verify scores were added
        self.assertEqual(reranked[0]["llm_score"], 9)
        self.assertEqual(reranked[1]["llm_score"], 7)
        self.assertEqual(reranked[2]["llm_score"], 5)

    @patch("src.api.rag_pipeline.ollama_api.OllamaLocalClient")
    def test_rerank_with_llm_parsing_error(self, mock_ollama_client_class):
        """Test error handling in score parsing during reranking."""
        # Configure mock to return responses that can't be parsed as numbers
        mock_client = MagicMock()
        mock_client.generate_text.side_effect = ["Not a number", "5", "nine"]
        mock_ollama_client_class.return_value = mock_client

        # Test data
        chunks = [
            {"chunk_id": 1, "chunk_text": "Information about Paris"},
            {"chunk_id": 2, "chunk_text": "Information about London"},
            {"chunk_id": 3, "chunk_text": "Information about the Eiffel Tower"},
        ]

        # Call function
        reranked = rerank_with_llm(chunks, "Tell me about Paris", "llama3:8b")

        # Verify default scores were used when parsing failed
        self.assertEqual(reranked[0]["chunk_id"], 2)  # Score 5
        self.assertEqual(reranked[1]["llm_score"], 0)  # Failed to parse "Not a number"
        self.assertEqual(reranked[2]["llm_score"], 0)  # Failed to parse "nine"

    @patch("src.api.rag_pipeline.ollama_api.OllamaLocalClient")
    def test_rerank_with_llm_error(self, mock_ollama_client_class):
        """Test error handling in LLM reranking."""
        # Configure mock to raise exception
        mock_ollama_client_class.side_effect = Exception("Reranking error")

        # Test data
        chunks = [
            {"chunk_id": 1, "chunk_text": "Information about Paris"},
            {"chunk_id": 2, "chunk_text": "Information about London"},
        ]

        # Call function - should log error and return original chunks
        with self.assertLogs(level="ERROR") as cm:
            result = rerank_with_llm(chunks, "Tell me about Paris", "llama3:8b")

            # Verify error was logged
            self.assertTrue(any("Error in LLM reranking" in msg for msg in cm.output))

        # Verify original chunks were returned
        self.assertEqual(result, chunks)

    def test_format_prompt(self):
        """Test prompt formatting."""
        system_prompt = "You are a helpful assistant."
        context = "Paris is the capital of France."
        question = "What is the capital of France?"
        history = "User: Hi\nAI: Hello"

        # Call function
        formatted = format_prompt(system_prompt, context, question, history)

        # Verify format
        self.assertIn(system_prompt, formatted)
        self.assertIn(context, formatted)
        self.assertIn(question, formatted)
        self.assertIn(history, formatted)
        self.assertIn("CONTEXT:", formatted)
        self.assertIn("USER QUERY:", formatted)
        self.assertIn("RESPONSE:", formatted)

    @patch("src.api.rag_pipeline.ollama_api.OllamaLocalClient")
    def test_query_llm(self, mock_ollama_client_class):
        """Test querying LLM with a formatted prompt."""
        # Configure mock
        mock_client = MagicMock()
        mock_client.generate_text.return_value = "The capital of France is Paris."
        mock_ollama_client_class.return_value = mock_client

        # Call function
        response = query_llm("What is the capital of France?", "llama3:8b")

        # Verify client was created with correct model
        mock_ollama_client_class.assert_called_once_with("llama3:8b")

        # Verify generate_text was called with correct parameters
        mock_client.generate_text.assert_called_once()
        args, kwargs = mock_client.generate_text.call_args
        self.assertEqual(kwargs["prompt"], "What is the capital of France?")

        # Verify correct response was returned
        self.assertEqual(response, "The capital of France is Paris.")

    @patch("src.api.rag_pipeline.ollama_api.OllamaLocalClient")
    def test_query_llm_error(self, mock_ollama_client_class):
        """Test error handling in LLM querying."""
        # Configure mock to raise exception
        mock_ollama_client_class.side_effect = Exception("Query error")

        # Call function - should log error and return error message
        with self.assertLogs(level="ERROR") as cm:
            response = query_llm("What is the capital of France?", "llama3:8b")

            # Verify error was logged
            self.assertTrue(any("Error querying local LLM" in msg for msg in cm.output))

        # Verify response contains error
        self.assertEqual(response, "Error: Query error")


if __name__ == "__main__":
    unittest.main()
