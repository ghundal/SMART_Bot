"""
Unit tests for the ollama_api.py module.

Tests the local Ollama client functionality including:
- Model existence checking
- Temporary model creation
- Text generation
- Prompt formatting
- Reranking integration
"""

import unittest
from unittest.mock import MagicMock, patch, call, ANY
import os
import subprocess
import tempfile
import sys


class TestOllamaLocalClient(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        # Set up patches for subprocess and logger
        self.mock_subprocess = MagicMock()
        self.mock_logger = MagicMock()

        # Create mock objects for subprocess run results
        self.mock_list_result = MagicMock()
        self.mock_run_result = MagicMock()
        self.mock_cleanup_result = MagicMock()

        # Configure default mock returns
        self.mock_list_result.stdout = "model1\nllama2\nmodel3"
        self.mock_list_result.returncode = 0
        self.mock_run_result.stdout = "Generated text response"
        self.mock_run_result.returncode = 0
        self.mock_cleanup_result.returncode = 0

        # Create patches for os and tempfile
        self.mock_os = MagicMock()
        self.mock_tempfile = MagicMock()
        self.mock_temp_file = MagicMock()
        self.mock_tempfile.NamedTemporaryFile.return_value.__enter__.return_value = self.mock_temp_file
        self.mock_temp_file.name = "/tmp/tempfile.modelfile"

    @patch('api.rag_pipeline.ollama_api.subprocess')
    @patch('api.rag_pipeline.ollama_api.logger')
    def test_init_model_exists(self, mock_logger, mock_subprocess):
        """Test initialization when model exists locally"""
        # Configure mock
        mock_subprocess.run.return_value.stdout = "llama2\nmodel2\nmodel3"
        mock_subprocess.run.return_value.returncode = 0

        # Import after patching
        from api.rag_pipeline.ollama_api import OllamaLocalClient

        # Create client
        client = OllamaLocalClient("llama2")

        # Verify model name was set correctly
        self.assertEqual(client.model_name, "llama2")

        # Verify subprocess was called correctly
        mock_subprocess.run.assert_called_once_with(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )

        # Verify logger was not called for warnings
        mock_logger.warning.assert_not_called()

    @patch('api.rag_pipeline.ollama_api.subprocess')
    @patch('api.rag_pipeline.ollama_api.logger')
    def test_init_model_not_exists(self, mock_logger, mock_subprocess):
        """Test initialization when model doesn't exist locally"""
        # Configure mock
        mock_subprocess.run.return_value.stdout = "model1\nmodel2\nmodel3"
        mock_subprocess.run.return_value.returncode = 0

        # Import after patching
        from api.rag_pipeline.ollama_api import OllamaLocalClient

        # Create client
        client = OllamaLocalClient("llama2")

        # Verify warning was logged
        mock_logger.warning.assert_called_once()

        # Verify info about available models was logged
        mock_logger.info.assert_called_once()

    @patch('subprocess.run')
    @patch('api.rag_pipeline.ollama_api.logger')
    def test_init_ollama_not_found(self, mock_logger, mock_subprocess_run):
        """Test initialization when Ollama is not installed"""
        # Configure mock to raise FileNotFoundError
        mock_subprocess_run.side_effect = FileNotFoundError()

        # Import after patching
        from api.rag_pipeline.ollama_api import OllamaLocalClient

        # Create client
        client = OllamaLocalClient("llama2")

        # Verify error was logged
        mock_logger.error.assert_called_once()

    @patch('api.rag_pipeline.ollama_api.subprocess')
    @patch('api.rag_pipeline.ollama_api.os')
    @patch('api.rag_pipeline.ollama_api.tempfile')
    @patch('api.rag_pipeline.ollama_api.logger')
    def test_create_temp_model(self, mock_logger, mock_tempfile, mock_os, mock_subprocess):
        """Test creating a temporary model"""
        # Configure mocks
        mock_tempfile.NamedTemporaryFile.return_value.__enter__.return_value.name = "/tmp/model.modelfile"
        # Set up subprocess.run to return success for the create command only
        create_result = MagicMock()
        create_result.returncode = 0

        # Add model check to the list of calls as well
        list_result = MagicMock()
        list_result.stdout = "model1\nllama2\nmodel3"
        list_result.returncode = 0

        mock_subprocess.run.side_effect = [list_result, create_result]

        mock_os.getpid.return_value = 12345

        # Import after patching
        from api.rag_pipeline.ollama_api import OllamaLocalClient

        # Create client
        client = OllamaLocalClient("llama2")

        # Reset mock_subprocess.run to avoid counting the call in __init__
        mock_subprocess.run.reset_mock()
        mock_subprocess.run.return_value = create_result

        # Call the method
        temp_model_name = client._create_temp_model(temperature=0.5, top_p=0.8, repeat_penalty=1.2)

        # Verify expected model name
        self.assertEqual(temp_model_name, "llama2-temp-12345")

        # Verify temporary file was created
        mock_tempfile.NamedTemporaryFile.assert_called_once_with(
            mode="w",
            suffix=".modelfile",
            delete=False
        )

        # Verify subprocess was called correctly - only once for create
        mock_subprocess.run.assert_called_once_with(
            ["ollama", "create", "llama2-temp-12345", "-f", "/tmp/model.modelfile"],
            capture_output=True,
            text=True,
        )

        # Verify temporary file was deleted
        mock_os.unlink.assert_called_once_with("/tmp/model.modelfile")

    @patch('api.rag_pipeline.ollama_api.subprocess')
    @patch('api.rag_pipeline.ollama_api.os')
    @patch('api.rag_pipeline.ollama_api.tempfile')
    @patch('api.rag_pipeline.ollama_api.logger')
    def test_create_temp_model_error(self, mock_logger, mock_tempfile, mock_os, mock_subprocess):
        """Test error handling when creating a temporary model"""
        # Configure mocks
        mock_tempfile.NamedTemporaryFile.return_value.__enter__.return_value.name = "/tmp/model.modelfile"

        # Add model check to the list of calls
        list_result = MagicMock()
        list_result.stdout = "model1\nllama2\nmodel3"
        list_result.returncode = 0

        # Set error for create command
        create_result = MagicMock()
        create_result.returncode = 1
        create_result.stderr = "Command failed"

        mock_subprocess.run.side_effect = [list_result, create_result]

        mock_os.getpid.return_value = 12345

        # Import after patching
        from api.rag_pipeline.ollama_api import OllamaLocalClient

        # Create client
        client = OllamaLocalClient("llama2")

        # Reset mock_subprocess.run to avoid counting the call in __init__
        mock_subprocess.run.reset_mock()
        mock_subprocess.run.return_value = create_result

        # Call the method
        temp_model_name = client._create_temp_model()

        # Verify fallback to original model
        self.assertEqual(temp_model_name, "llama2")

        # Verify error was logged
        mock_logger.error.assert_called_once()

    @patch('api.rag_pipeline.ollama_api.logger')
    def test_generate_text(self, mock_logger):
        """Test generating text with local Ollama model"""
        # Import after patching
        from api.rag_pipeline.ollama_api import OllamaLocalClient

        # Create a mock instance with _create_temp_model already mocked
        client = OllamaLocalClient("llama2")
        client._ensure_model_exists = MagicMock()
        client._create_temp_model = MagicMock(return_value="llama2-temp-12345")

        # Mock subprocess.run for the actual test
        with patch('api.rag_pipeline.ollama_api.subprocess.run') as mock_run:
            # Configure the run mock to return different values for run and cleanup
            run_result = MagicMock()
            run_result.returncode = 0
            run_result.stdout = "User prompt\nGenerated response text"

            cleanup_result = MagicMock()
            cleanup_result.returncode = 0

            mock_run.side_effect = [run_result, cleanup_result]

            # Call the method
            response = client.generate_text("User prompt")

            # Verify response
            self.assertEqual(response, "Generated response text")

            # Verify temp model was created
            client._create_temp_model.assert_called_once_with(0.7, 0.9, 1.1)

            # Verify ollama run was called correctly
            calls = mock_run.call_args_list
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][0][0], ["ollama", "run", "llama2-temp-12345", "User prompt"])
            self.assertEqual(calls[1][0][0], ["ollama", "rm", "llama2-temp-12345"])

    @patch('api.rag_pipeline.ollama_api.logger')
    def test_generate_text_error(self, mock_logger):
        """Test error handling in text generation"""
        # Import after patching
        from api.rag_pipeline.ollama_api import OllamaLocalClient

        # Create a mock instance with _create_temp_model already mocked
        client = OllamaLocalClient("llama2")
        client._ensure_model_exists = MagicMock()
        client._create_temp_model = MagicMock(return_value="llama2-temp-12345")

        # Mock subprocess.run for the actual test
        with patch('api.rag_pipeline.ollama_api.subprocess.run') as mock_run:
            # Configure the run mock to return error for run but success for cleanup
            run_result = MagicMock()
            run_result.returncode = 1
            run_result.stderr = "Command failed"

            cleanup_result = MagicMock()
            cleanup_result.returncode = 0

            mock_run.side_effect = [run_result, cleanup_result]

            # Call the method
            response = client.generate_text("User prompt")

            # Verify error response
            self.assertEqual(response, "Error: Command failed")

            # Verify error was logged
            mock_logger.error.assert_called_once()

    @patch('api.rag_pipeline.ollama_api.logger')
    def test_cleanup_failure(self, mock_logger):
        """Test handling cleanup failure after generation"""
        # Import after patching
        from api.rag_pipeline.ollama_api import OllamaLocalClient

        # Create a mock instance with _create_temp_model already mocked
        client = OllamaLocalClient("llama2")
        client._ensure_model_exists = MagicMock()
        client._create_temp_model = MagicMock(return_value="llama2-temp-12345")

        # Mock subprocess.run for the actual test
        with patch('api.rag_pipeline.ollama_api.subprocess.run') as mock_run:
            # Configure the run mock to return success for run but failure for cleanup
            run_result = MagicMock()
            run_result.returncode = 0
            run_result.stdout = "Generated text"

            cleanup_result = MagicMock()
            cleanup_result.returncode = 1
            cleanup_result.stderr = "Cleanup failed"

            mock_run.side_effect = [run_result, cleanup_result]

            # Reset the logger mock to avoid counting warnings from other calls
            mock_logger.reset_mock()

            # Call the method
            response = client.generate_text("User prompt")

            # Verify response still works
            self.assertEqual(response, "Generated text")

            # Verify warning was called exactly once
            mock_logger.warning.assert_called_once()

    def test_rerank_with_llm(self):
        """Test rerank_with_llm function"""
        # Create a mock for transformer_reranker module
        mock_transformer_reranker = MagicMock()
        mock_rerank_chunks = MagicMock(return_value=["chunk1", "chunk2"])
        mock_transformer_reranker.rerank_chunks = mock_rerank_chunks

        # Save original and patch the module
        original_module = sys.modules.get('api.rag_pipeline.transformer_reranker', None)
        sys.modules['api.rag_pipeline.transformer_reranker'] = mock_transformer_reranker

        try:
            # Force reload of the ollama_api module to pick up our mock
            if 'api.rag_pipeline.ollama_api' in sys.modules:
                del sys.modules['api.rag_pipeline.ollama_api']

            # Now import the module - this should use our mock transformer_reranker
            import api.rag_pipeline.ollama_api as ollama_api

            # Call the function
            chunks = ["raw_chunk1", "raw_chunk2"]
            query = "test query"
            result = ollama_api.rerank_with_llm(chunks, query)

            # Verify the mock was called correctly
            mock_rerank_chunks.assert_called_once_with(chunks, query)

            # Verify the result
            self.assertEqual(result, ["chunk1", "chunk2"])

        finally:
            # Restore original module if it existed
            if original_module:
                sys.modules['api.rag_pipeline.transformer_reranker'] = original_module
            else:
                del sys.modules['api.rag_pipeline.transformer_reranker']

    def test_format_prompt(self):
        """Test format_prompt function"""
        # Import the function
        from api.rag_pipeline.ollama_api import format_prompt

        # Call the function
        prompt = format_prompt(
            system_prompt="You are a helpful assistant.",
            context="This is the context information.",
            question="What is the answer?",
            conversation_history="User: Previous question\nAssistant: Previous answer"
        )

        # Verify prompt structure
        self.assertIn("You are a helpful assistant.", prompt)
        self.assertIn("This is the context information.", prompt)
        self.assertIn("What is the answer?", prompt)
        self.assertIn("User: Previous question\nAssistant: Previous answer", prompt)

    @patch('api.rag_pipeline.ollama_api.OllamaLocalClient')
    @patch('api.rag_pipeline.ollama_api.GENERATION_CONFIG', {
        "temperature": 0.8,
        "top_p": 0.95,
        "repeat_penalty": 1.2
    })
    def test_query_llm(self, mock_ollama_client_class):
        """Test query_llm function"""
        # Configure mock
        mock_client = MagicMock()
        mock_client.generate_text.return_value = "Generated response"
        mock_ollama_client_class.return_value = mock_client

        # Import after patching
        from api.rag_pipeline.ollama_api import query_llm

        # Call the function
        response = query_llm("Test prompt", "llama2")

        # Verify client was created with correct model
        mock_ollama_client_class.assert_called_once_with("llama2")

        # Verify generate_text was called with correct parameters
        mock_client.generate_text.assert_called_once_with(
            prompt="Test prompt",
            temperature=0.8,
            top_p=0.95,
            repeat_penalty=1.2
        )

        # Verify response
        self.assertEqual(response, "Generated response")

    @patch('api.rag_pipeline.ollama_api.OllamaLocalClient')
    @patch('api.rag_pipeline.ollama_api.logger')
    def test_query_llm_exception(self, mock_logger, mock_ollama_client_class):
        """Test query_llm with exception"""
        # Configure mock to raise exception
        mock_ollama_client_class.side_effect = Exception("Client error")

        # Import after patching
        from api.rag_pipeline.ollama_api import query_llm

        # Call the function
        response = query_llm("Test prompt", "llama2")

        # Verify error response
        self.assertEqual(response, "Error: Client error")

        # Verify exception was logged
        mock_logger.exception.assert_called_once()


if __name__ == "__main__":
    unittest.main()
