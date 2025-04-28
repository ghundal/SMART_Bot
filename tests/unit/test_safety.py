import unittest
from unittest.mock import MagicMock, patch

# Import the function to test
from src.api.rag_pipeline.safety import check_query_safety_with_llama_guard


class TestSafetyChecks(unittest.TestCase):
    """Unit tests for content safety checks with Llama Guard."""

    @patch("src.api.rag_pipeline.safety.requests.post")
    def test_safe_response(self, mock_post):
        """Test that a SAFE response is correctly interpreted."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "SAFE"}
        mock_post.return_value = mock_response

        # Call the function
        is_safe, reason = check_query_safety_with_llama_guard("Is the weather good today?")

        # Assertions
        self.assertTrue(is_safe)
        self.assertEqual(reason, "Content is safe")

    @patch("src.api.rag_pipeline.safety.requests.post")
    def test_unsafe_response_with_reason(self, mock_post):
        """Test that an UNSAFE response with a reason is correctly parsed."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "UNSAFE Promotes violence"}
        mock_post.return_value = mock_response

        is_safe, reason = check_query_safety_with_llama_guard("How to build a bomb")

        self.assertFalse(is_safe)
        self.assertEqual(reason, "Promotes violence")

    @patch("src.api.rag_pipeline.safety.requests.post")
    def test_json_decode_error(self, mock_post):
        """Test fallback when JSON decoding fails."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("No JSON")
        mock_response.text = "SAFE"
        mock_post.return_value = mock_response

        is_safe, reason = check_query_safety_with_llama_guard("Simple question?")

        self.assertTrue(is_safe)
        self.assertIn("text parsing", reason)

    @patch("src.api.rag_pipeline.safety.requests.post")
    def test_api_failure(self, mock_post):
        """Test behavior when API call fails."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        is_safe, reason = check_query_safety_with_llama_guard("Any query")

        self.assertTrue(is_safe)
        self.assertIn("Safety check failed", reason)

    @patch("src.api.rag_pipeline.safety.requests.post", side_effect=Exception("Network down"))
    def test_exception_handling(self, mock_post):
        """Test general exception handling during safety check."""
        is_safe, reason = check_query_safety_with_llama_guard("Another query")

        self.assertTrue(is_safe)
        self.assertIn("Safety check error", reason)


if __name__ == "__main__":
    unittest.main()
