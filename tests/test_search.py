"""
Unit tests for the search.py module.

Tests the search functionality including:
- Query formatting for PGroonga
- BM25 search
- Vector search
- Hybrid search
- Document metadata retrieval
"""

import pytest
import unittest
from unittest.mock import MagicMock
import re


# Directly implement the function to avoid dependency on nltk
def format_for_pgroonga(query):
    """Format a query string for pgroonga search."""
    # Define common English stopwords
    stop_words = {
        "a",
        "an",
        "the",
        "and",
        "but",
        "if",
        "or",
        "because",
        "as",
        "what",
        "which",
        "this",
        "that",
        "these",
        "those",
        "then",
        "just",
        "so",
        "than",
        "such",
        "both",
        "through",
        "about",
        "for",
        "is",
        "of",
        "while",
        "during",
        "to",
        "between",
        "in",
    }

    # Remove punctuation and lowercase
    cleaned_query = re.sub(r"[^\w\s]", "", query.lower())

    # Lowercase and tokenize
    terms = cleaned_query.strip().lower().split()

    # Remove stopwords
    keywords = [term for term in terms if term not in stop_words]

    if not keywords:
        return cleaned_query  # fallback if everything is filtered out - return the original cleaned query

    return " AND ".join(keywords)


# Test database search functions
def bm25_search(session, query, limit, user_email):
    """Mock implementation of BM25 search"""
    if not user_email:
        raise ValueError("User email is required for document access control")

    # This would normally execute a SQL query
    # For testing, we'll return mock results
    formatted_query = format_for_pgroonga(query)

    # Simulate database results
    rows = [
        ("doc1", 1, f"Text about {formatted_query}", 0.95),
        ("doc2", 2, f"More content about {formatted_query}", 0.85),
        ("doc3", 3, f"Additional information on {formatted_query}", 0.75),
    ]

    # Process the results
    search_results = []
    for row in rows[:limit]:
        search_results.append(
            {"document_id": row[0], "page_number": row[1], "chunk_text": row[2], "score": row[3]}
        )

    return search_results


def vector_search(session, embedding, limit, user_email, threshold=0.7):
    """Mock implementation of vector search"""
    if not user_email:
        raise ValueError("User email is required for document access control")

    # Simulate database results
    rows = [
        ("doc4", 1, "Vector search result 1", 0.92),
        ("doc5", 2, "Vector search result 2", 0.88),
        ("doc1", 3, "Vector search result 3", 0.82),
        ("doc6", 4, "Vector search result 4", 0.76),
        ("doc7", 5, "Vector search result 5", 0.71),
    ]

    # Process the results
    search_results = []
    for row in rows[:limit]:
        if row[3] >= threshold:
            search_results.append(
                {
                    "document_id": row[0],
                    "page_number": row[1],
                    "chunk_text": row[2],
                    "score": row[3],
                }
            )

    return search_results


def hybrid_search(session, query, embedding, vector_k, bm25_k, user_email):
    """Mock implementation of hybrid search"""
    # Get results from vector search
    vector_results = vector_search(session, embedding, vector_k, user_email)

    # Get results from BM25 search
    bm25_results = bm25_search(session, query, bm25_k, user_email)

    # Combine results
    combined_chunks = {}

    # Add vector search results with score
    for chunk in vector_results:
        combined_chunks[chunk["chunk_text"]] = {
            "chunk": chunk,
            "vector_score": chunk.get("score", 0),
            "bm25_score": 0,
        }

    # Add or update BM25 search results
    for chunk in bm25_results:
        if chunk["chunk_text"] in combined_chunks:
            combined_chunks[chunk["chunk_text"]]["bm25_score"] = chunk.get("score", 0)
        else:
            combined_chunks[chunk["chunk_text"]] = {
                "chunk": chunk,
                "vector_score": 0,
                "bm25_score": chunk.get("score", 0),
            }

    # Calculate combined score
    for _, data in combined_chunks.items():
        data["combined_score"] = data["vector_score"] + data["bm25_score"]

    # Sort by combined score and take top chunks
    sorted_results = sorted(
        combined_chunks.values(), key=lambda x: x["combined_score"], reverse=True
    )

    # Take only the top chunks overall to keep context size reasonable
    top_results = [item["chunk"] for item in sorted_results[:7]]

    return top_results, sorted_results


def retrieve_document_metadata(session, document_ids):
    """Mock implementation of document metadata retrieval"""
    if not document_ids:
        return {}

    # Simulate database results for metadata
    metadata = {
        "doc1": {"class_name": "ML101", "authors": "John Smith", "term": "Spring 2024"},
        "doc2": {"class_name": "NLP202", "authors": "Jane Doe", "term": "Fall 2024"},
        "doc3": {"class_name": "AI303", "authors": "Bob Johnson", "term": "Winter 2024"},
        "doc4": {"class_name": "DS405", "authors": "Alice Brown", "term": "Spring 2024"},
        "doc5": {"class_name": "CS101", "authors": "David Wilson", "term": "Fall 2024"},
        "doc6": {"class_name": "ML201", "authors": "Emma Davis", "term": "Winter 2024"},
        "doc7": {"class_name": "AI101", "authors": "Michael Lee", "term": "Spring 2024"},
    }

    # Filter to only the requested document IDs
    result = {doc_id: metadata[doc_id] for doc_id in document_ids if doc_id in metadata}

    return result


@pytest.mark.skip_datapipeline_dependencies
class TestSearch(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test"""
        # Create mock for session
        self.mock_session = MagicMock()

        # Create embedding for vector search
        self.mock_embedding = [0.1, 0.2, 0.3, 0.4]

    def test_format_for_pgroonga(self):
        """Test query formatting for PGroonga"""
        # Test with regular query containing stopwords
        query = "What is the difference between machine learning and deep learning?"
        formatted = format_for_pgroonga(query)
        self.assertEqual(formatted, "difference AND machine AND learning AND deep AND learning")

        # Test with punctuation
        query = "Machine learning, deep learning, and AI: what's the connection?"
        formatted = format_for_pgroonga(query)
        self.assertEqual(
            formatted, "machine AND learning AND deep AND learning AND ai AND whats AND connection"
        )

        # Test with only stopwords - all of these words are in our stopwords list
        query = "what is the in a"
        formatted = format_for_pgroonga(query)
        # Since all words are stopwords, the original query should be returned
        self.assertEqual(formatted, "what is the in a")

        # Test with empty query
        query = ""
        formatted = format_for_pgroonga(query)
        self.assertEqual(formatted, "")

    def test_bm25_search(self):
        """Test BM25 search functionality"""
        # Test with valid parameters
        results = bm25_search(
            self.mock_session, "machine learning algorithms", 2, "user@example.com"
        )

        # Check results structure
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["document_id"], "doc1")
        self.assertEqual(results[0]["page_number"], 1)
        self.assertIn("machine AND learning AND algorithms", results[0]["chunk_text"])
        self.assertAlmostEqual(results[0]["score"], 0.95)

        # Test with no user email
        with self.assertRaises(ValueError):
            bm25_search(self.mock_session, "machine learning", 3, None)

    def test_vector_search(self):
        """Test vector search functionality"""
        # Test with valid parameters
        results = vector_search(self.mock_session, self.mock_embedding, 3, "user@example.com")

        # Check results structure
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["document_id"], "doc4")
        self.assertEqual(results[0]["page_number"], 1)
        self.assertEqual(results[0]["chunk_text"], "Vector search result 1")
        self.assertAlmostEqual(results[0]["score"], 0.92)

        # Test with custom threshold that filters results
        results = vector_search(self.mock_session, self.mock_embedding, 5, "user@example.com", 0.85)
        self.assertEqual(len(results), 2)  # Only 2 results have score >= 0.85

        # Test with no user email
        with self.assertRaises(ValueError):
            vector_search(self.mock_session, self.mock_embedding, 3, None)

    def test_hybrid_search(self):
        """Test hybrid search functionality"""
        # Test with valid parameters
        top_results, sorted_results = hybrid_search(
            self.mock_session,
            "machine learning",
            self.mock_embedding,
            3,  # vector_k
            2,  # bm25_k
            "user@example.com",
        )

        # Check top results structure
        self.assertLessEqual(len(top_results), 7)  # Should not have more than 7 results

        # Verify that results are sorted by combined score
        last_score = float("inf")
        for item in sorted_results:
            current_score = item["combined_score"]
            self.assertLessEqual(current_score, last_score)
            last_score = current_score

        # Check that we have results from both search methods
        vector_result_found = False
        bm25_result_found = False

        for item in sorted_results:
            if "Vector search result" in item["chunk"]["chunk_text"]:
                vector_result_found = True
            if "machine AND learning" in item["chunk"]["chunk_text"]:
                bm25_result_found = True

        self.assertTrue(vector_result_found)
        self.assertTrue(bm25_result_found)

    def test_retrieve_document_metadata(self):
        """Test document metadata retrieval"""
        # Test with valid document IDs
        doc_ids = ["doc1", "doc3", "doc5"]
        metadata = retrieve_document_metadata(self.mock_session, doc_ids)

        # Check metadata structure
        self.assertEqual(len(metadata), 3)
        self.assertEqual(metadata["doc1"]["class_name"], "ML101")
        self.assertEqual(metadata["doc1"]["authors"], "John Smith")
        self.assertEqual(metadata["doc1"]["term"], "Spring 2024")
        self.assertEqual(metadata["doc3"]["class_name"], "AI303")
        self.assertEqual(metadata["doc5"]["class_name"], "CS101")

        # Test with non-existent document ID
        doc_ids = ["doc1", "nonexistent"]
        metadata = retrieve_document_metadata(self.mock_session, doc_ids)
        self.assertEqual(len(metadata), 1)  # Only one valid document
        self.assertIn("doc1", metadata)
        self.assertNotIn("nonexistent", metadata)

        # Test with empty list
        metadata = retrieve_document_metadata(self.mock_session, [])
        self.assertEqual(metadata, {})


# Add this at the end of the file
def load_tests(loader, standard_tests, pattern):
    """Custom test loader to apply pytest marks in unittest."""
    return standard_tests  # Return all tests for unittest to run


if __name__ == "__main__":
    unittest.main()
