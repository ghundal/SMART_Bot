import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from langchain_core.documents import Document

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the module to test - using correct path to your actual file
from src.datapipeline.Advanced_semantic_chunker import (
    AdvancedSemanticChunker,
    calculate_cosine_distances,
    combine_sentences,
    generate_local_embeddings,
    get_embedding_model,
)


class TestAdvancedSemanticChunker(unittest.TestCase):
    def setUp(self):
        # Create a mock embedding function that returns deterministic embeddings
        self.mock_embedding_function = lambda texts: [
            [0.1, 0.2, 0.3] if i % 2 == 0 else [0.4, 0.5, 0.6] for i in range(len(texts))
        ]

        # Create an instance of the chunker with the mock embedding function
        self.chunker = AdvancedSemanticChunker(
            buffer_size=1,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
            embedding_function=self.mock_embedding_function,
        )

        # Sample sentences for testing
        self.sample_text = (
            "This is the first sentence. This is the second sentence. "
            "This is the third sentence. This is the fourth sentence. "
            "This is a completely different topic. Another different topic sentence."
        )

        # Sample documents for testing
        self.sample_documents = [
            Document(page_content=self.sample_text, metadata={"source": "test_doc_1"}),
            Document(
                page_content="Another document with some text.", metadata={"source": "test_doc_2"}
            ),
        ]

    def test_get_embedding_model(self):
        # Create a fresh patch for get_embedding_model that differs from the tested function
        with patch(
            "src.datapipeline.Advanced_semantic_chunker.SentenceTransformer"
        ) as mock_transformer:
            # Setup mock
            mock_model = MagicMock()
            mock_transformer.return_value = mock_model

            # Test the function
            model = get_embedding_model("test_model")

            # Verify the model was loaded with the right name
            mock_transformer.assert_called_once_with("test_model")

            # Verify we got back the mock model
            self.assertEqual(model, mock_transformer.return_value)

    def test_generate_local_embeddings(self):
        # Create a fresh patch for get_embedding_model
        with patch(
            "src.datapipeline.Advanced_semantic_chunker.get_embedding_model"
        ) as mock_get_model:
            # Setup
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
            mock_get_model.return_value = mock_model

            texts = ["Text 1", "Text 2"]

            # Test with no preloaded model
            embeddings = generate_local_embeddings(texts, model_name="test_model")

            # Verify the model was loaded
            mock_get_model.assert_called_once_with("test_model")

            # Verify encode was called with the right texts
            mock_model.encode.assert_called_once()

            # Test with preloaded model
            preloaded_model = MagicMock()
            preloaded_model.encode.return_value = np.array([[0.5, 0.6], [0.7, 0.8]])

            embeddings = generate_local_embeddings(texts, preloaded_model=preloaded_model)

            # Verify the preloaded model was used
            preloaded_model.encode.assert_called_once()

    def test_combine_sentences(self):
        # Test input
        sentences = [
            {"sentence": "First sentence.", "index": 0},
            {"sentence": "Second sentence.", "index": 1},
            {"sentence": "Third sentence.", "index": 2},
        ]

        # Test with buffer_size=1
        result = combine_sentences(sentences, buffer_size=1)

        # Verify results
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["combined_sentence"], "First sentence. Second sentence.")
        self.assertEqual(
            result[1]["combined_sentence"], "First sentence. Second sentence. Third sentence."
        )
        self.assertEqual(result[2]["combined_sentence"], "Second sentence. Third sentence.")

        # Test with buffer_size=0
        result = combine_sentences(sentences, buffer_size=0)
        self.assertEqual(result[0]["combined_sentence"], "First sentence.")
        self.assertEqual(result[1]["combined_sentence"], "Second sentence.")
        self.assertEqual(result[2]["combined_sentence"], "Third sentence.")

    def test_calculate_cosine_distances(self):
        # Test input with embeddings
        sentences = [
            {"combined_sentence_embedding": [1.0, 0.0, 0.0], "sentence": "First."},
            {"combined_sentence_embedding": [0.0, 1.0, 0.0], "sentence": "Second."},
            {"combined_sentence_embedding": [0.0, 0.0, 1.0], "sentence": "Third."},
        ]

        # Calculate distances
        distances, updated_sentences = calculate_cosine_distances(sentences)

        # Verify results
        self.assertEqual(len(distances), 2)  # Should have distances between consecutive sentences
        self.assertAlmostEqual(distances[0], 1.0)  # Distance between orthogonal vectors is 1.0
        self.assertAlmostEqual(distances[1], 1.0)  # Distance between orthogonal vectors is 1.0

        # Check that distances were added to sentences
        self.assertAlmostEqual(updated_sentences[0]["distance_to_next"], 1.0)
        self.assertAlmostEqual(updated_sentences[1]["distance_to_next"], 1.0)

    def test_split_text_simple(self):
        # Test with a simple text
        result = self.chunker.split_text("Single sentence text.")

        # Should return the text as a single chunk for very short texts
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "Single sentence text.")

    def test_split_text_with_breakpoints(self):
        # Test with a text that has a semantic breakpoint
        # Here we're leveraging our mock embedding function which alternates
        # between two very different embeddings, creating semantic breaks

        # Mock the _calculate_breakpoint_threshold to return a predictable threshold
        with patch.object(
            self.chunker,
            "_calculate_breakpoint_threshold",
            return_value=(0.5, [0.9, 0.1, 0.9, 0.1, 0.9]),
        ):
            result = self.chunker.split_text(self.sample_text)

            # We expect the text to be broken into chunks due to our mock embeddings
            self.assertGreater(len(result), 1)

    def test_create_documents(self):
        # Test creating documents from a list of texts
        texts = ["Text 1.", "Text 2."]
        metadata = [{"source": "test1"}, {"source": "test2"}]

        # Mock split_text to control the output
        with patch.object(self.chunker, "split_text", side_effect=lambda text: [text]):
            documents = self.chunker.create_documents(texts, metadatas=metadata)

            # Verify results
            self.assertEqual(len(documents), 2)
            self.assertEqual(documents[0].page_content, "Text 1.")
            self.assertEqual(documents[0].metadata, {"source": "test1"})
            self.assertEqual(documents[1].page_content, "Text 2.")
            self.assertEqual(documents[1].metadata, {"source": "test2"})

    def test_create_documents_with_start_index(self):
        # Test with add_start_index=True
        chunker_with_start_index = AdvancedSemanticChunker(
            buffer_size=1, add_start_index=True, embedding_function=self.mock_embedding_function
        )

        # Mock split_text to control the output
        with patch.object(
            chunker_with_start_index, "split_text", side_effect=lambda text: ["Chunk 1", "Chunk 2"]
        ):
            documents = chunker_with_start_index.create_documents(
                ["Text with multiple chunks."], metadatas=[{"source": "test"}]
            )

            # Verify results
            self.assertEqual(len(documents), 2)
            self.assertEqual(documents[0].page_content, "Chunk 1")
            self.assertEqual(documents[0].metadata, {"source": "test", "start_index": 0})
            self.assertEqual(documents[1].page_content, "Chunk 2")
            self.assertEqual(
                documents[1].metadata, {"source": "test", "start_index": 7}
            )  # Length of "Chunk 1"

    def test_split_documents(self):
        # Mock create_documents to verify it's called with the right parameters
        with patch.object(self.chunker, "create_documents") as mock_create_docs:
            mock_create_docs.return_value = [Document(page_content="Chunk", metadata={})]

            self.chunker.split_documents(self.sample_documents)

            # Verify create_documents was called with the right parameters
            texts = [doc.page_content for doc in self.sample_documents]
            metadatas = [doc.metadata for doc in self.sample_documents]
            mock_create_docs.assert_called_once_with(texts, metadatas=metadatas)

    def test_transform_documents(self):
        # Test that transform_documents calls split_documents
        with patch.object(self.chunker, "split_documents") as mock_split_docs:
            mock_split_docs.return_value = [Document(page_content="Chunk", metadata={})]

            self.chunker.transform_documents(self.sample_documents)

            # Verify split_documents was called with the right parameters
            mock_split_docs.assert_called_once_with(self.sample_documents)

    def test_breakpoint_threshold_percentile(self):
        # Test the percentile threshold calculation
        chunker = AdvancedSemanticChunker(
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90,
            embedding_function=self.mock_embedding_function,
        )

        distances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        threshold, _ = chunker._calculate_breakpoint_threshold(distances)

        # 90th percentile of distances should be 0.9 - allow some tolerance
        self.assertAlmostEqual(threshold, 0.9, places=1)

    def test_breakpoint_threshold_standard_deviation(self):
        # Test the standard deviation threshold calculation
        chunker = AdvancedSemanticChunker(
            breakpoint_threshold_type="standard_deviation",
            breakpoint_threshold_amount=2,
            embedding_function=self.mock_embedding_function,
        )

        distances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        threshold, _ = chunker._calculate_breakpoint_threshold(distances)

        # Mean is 0.55, std is ~0.3, so threshold should be around 0.55 + 2*0.3 = 1.15
        expected = np.mean(distances) + 2 * np.std(distances)
        self.assertAlmostEqual(threshold, expected)

    def test_breakpoint_threshold_interquartile(self):
        # Test the interquartile threshold calculation
        chunker = AdvancedSemanticChunker(
            breakpoint_threshold_type="interquartile",
            breakpoint_threshold_amount=1.5,
            embedding_function=self.mock_embedding_function,
        )

        distances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        threshold, _ = chunker._calculate_breakpoint_threshold(distances)

        # Q1 is 0.3, Q3 is 0.8, IQR is 0.5, so threshold should be around 0.55 + 1.5*0.5 = 1.3
        q1 = np.percentile(distances, 25)
        q3 = np.percentile(distances, 75)
        iqr = q3 - q1
        expected = np.mean(distances) + 1.5 * iqr
        self.assertAlmostEqual(threshold, expected)

    def test_breakpoint_threshold_gradient(self):
        # Test the gradient threshold calculation
        chunker = AdvancedSemanticChunker(
            breakpoint_threshold_type="gradient",
            breakpoint_threshold_amount=90,
            embedding_function=self.mock_embedding_function,
        )

        distances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        threshold, gradient_list = chunker._calculate_breakpoint_threshold(distances)

        # The gradient should be constant for this linear sequence, except at endpoints
        # The 90th percentile will be the second-highest gradient value
        gradient_values = np.gradient(distances, range(0, len(distances)))
        expected = np.percentile(gradient_values, 90)
        self.assertAlmostEqual(threshold, expected)

    def test_threshold_from_clusters(self):
        # Test the threshold from clusters calculation
        chunker = AdvancedSemanticChunker(
            number_of_chunks=3, embedding_function=self.mock_embedding_function
        )

        distances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        threshold = chunker._threshold_from_clusters(distances)

        # This is a complex calculation but we can verify it gives some sensible value
        self.assertTrue(0 <= threshold <= 1.0)

    def test_invalid_breakpoint_threshold_type(self):
        # Test with an invalid threshold type - check for KeyError during initialization
        with self.assertRaises(ValueError):
            chunker = AdvancedSemanticChunker(
                breakpoint_threshold_type="invalid_type",
                embedding_function=self.mock_embedding_function,
            )

    def test_calculate_sentence_distances(self):
        # Set up mock embedding function that returns consistent embeddings
        chunker = AdvancedSemanticChunker(
            embedding_function=lambda texts: [[0.1, 0.2, 0.3] for _ in texts]
        )

        # Test the sentence distance calculation
        single_sentences_list = ["First sentence.", "Second sentence.", "Third sentence."]

        # Patch the calculate_cosine_distances function to return predictable values
        with patch(
            "src.datapipeline.Advanced_semantic_chunker.calculate_cosine_distances"
        ) as mock_calc_dist:
            mock_calc_dist.return_value = (
                [0.1, 0.2],
                [
                    {"sentence": "1", "distance_to_next": 0.1},
                    {"sentence": "2", "distance_to_next": 0.2},
                    {"sentence": "3"},
                ],
            )

            distances, sentences = chunker._calculate_sentence_distances(single_sentences_list)

            # Verify distances
            self.assertEqual(distances, [0.1, 0.2])

            # Verify sentences have distances
            self.assertEqual(sentences[0]["distance_to_next"], 0.1)
            self.assertEqual(sentences[1]["distance_to_next"], 0.2)

    def test_edge_cases_single_sentence(self):
        # Test with a single sentence
        result = self.chunker.split_text("Just one sentence.")

        # Should return the text as a single chunk
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "Just one sentence.")

    def test_edge_cases_two_sentences_gradient(self):
        # Test with two sentences and gradient threshold type
        chunker = AdvancedSemanticChunker(
            breakpoint_threshold_type="gradient", embedding_function=self.mock_embedding_function
        )

        # Should handle this without error
        result = chunker.split_text("First sentence. Second sentence.")

        # Should still split the text into sentences
        self.assertGreaterEqual(len(result), 1)

    def test_number_of_chunks_option(self):
        # Test with specified number of chunks
        chunker = AdvancedSemanticChunker(
            number_of_chunks=2, embedding_function=self.mock_embedding_function
        )

        # Create a longer text to ensure multiple chunks
        long_text = " ".join([f"This is sentence {i}." for i in range(20)])

        # Mock the threshold calculation for predictable results
        with patch.object(chunker, "_threshold_from_clusters", return_value=0.5), patch.object(
            chunker,
            "_calculate_sentence_distances",
            return_value=([0.1, 0.9, 0.2, 0.8] + [0.3] * 15, []),
        ):
            # Use a simple version of split_text for testing
            with patch.object(chunker, "split_text", return_value=["Chunk 1", "Chunk 2"]):
                # Test creating documents with number_of_chunks
                docs = chunker.create_documents([long_text], metadatas=[{"source": "test"}])

                # Should create 2 documents as specified
                self.assertEqual(len(docs), 2)


if __name__ == "__main__":
    unittest.main()
