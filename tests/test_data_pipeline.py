import os
import sys
import unittest
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
from langchain_core.documents import Document

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import from the correct module path
from src.datapipeline.datapipeline import (
    BUCKET_NAME,
    chunk_documents_from_gcs,
    clean_chunks,
    clean_document_text,
    connect_to_bucket,
    connect_to_postgres,
    create_and_insert_chunks,
    create_chunk_embeddings,
    create_documents_dataframe,
    get_ch_embedding_model,
    get_metadata,
    list_document_folders,
    load_pdf_from_gcs,
    main,
    recursive_chunking,
    semantic_chunking,
    validate_data,
)


class TestDataPipeline(unittest.TestCase):

    def setUp(self):
        # Setup common mocks
        self.mock_bucket = MagicMock()
        self.mock_storage_client = MagicMock()
        self.mock_storage_client.bucket.return_value = self.mock_bucket

        # Mock directory structure
        self.mock_blob_1 = MagicMock()
        self.mock_blob_1.name = "documents/class1/doc1.pdf"

        self.mock_blob_2 = MagicMock()
        self.mock_blob_2.name = "documents/class2/doc2.pdf"

        self.mock_bucket.list_blobs.return_value = [self.mock_blob_1, self.mock_blob_2]

        # Mock metadata blobs
        self.mock_meta_blob = MagicMock()
        self.mock_meta_blob.download_as_string.return_value = (
            b"class_id,name\nclass1,Class 1\nclass2,Class 2"
        )

        self.mock_access_blob = MagicMock()
        self.mock_access_blob.download_as_string.return_value = (
            b"user_id,class_id\nuser1,class1\nuser2,class2"
        )

        # Setup patches
        self.gcs_client_patcher = patch("src.datapipeline.datapipeline.storage.Client")
        self.mock_gcs_client = self.gcs_client_patcher.start()
        self.mock_gcs_client.return_value = self.mock_storage_client

        # Mock database connection
        self.db_patcher = patch("src.datapipeline.datapipeline.create_engine")
        self.mock_db_engine = self.db_patcher.start()
        self.mock_engine = MagicMock()
        self.mock_db_engine.return_value = self.mock_engine

        # Mock environment variables
        self.env_patcher = patch.dict(
            "os.environ",
            {"GOOGLE_APPLICATION_CREDENTIALS": "fake_credentials.json", "DB_HOST": "localhost"},
        )
        self.env_patcher.start()

    def tearDown(self):
        # Stop all patches
        self.gcs_client_patcher.stop()
        self.db_patcher.stop()
        self.env_patcher.stop()

    def test_connect_to_bucket(self):
        # Test successful connection
        bucket = connect_to_bucket()

        # Verify client was created and bucket was retrieved
        self.mock_gcs_client.assert_called_once()
        self.mock_storage_client.bucket.assert_called_once_with("smart_input_data")
        self.assertEqual(bucket, self.mock_bucket)

        # Test failed connection
        self.mock_storage_client.bucket.side_effect = Exception("Connection failed")

        with self.assertRaises(RuntimeError):
            connect_to_bucket()

    def test_connect_to_postgres(self):
        # Ensure we're using the mock, not the real function
        with patch("src.datapipeline.datapipeline.create_engine") as mock_create_engine:
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine

            # Call the function - this should call create_engine
            engine = connect_to_postgres()

            # Verify that create_engine was called with the correct parameters
            mock_create_engine.assert_called_once()
            self.assertEqual(engine, mock_engine)

            # Test failed connection
            mock_create_engine.side_effect = Exception("DB connection failed")

            with self.assertRaises(RuntimeError):
                connect_to_postgres()

    def test_list_document_folders(self):
        # Test listing document folders
        folders = list_document_folders(self.mock_bucket)

        # Verify blobs were listed and folders were extracted
        self.mock_bucket.list_blobs.assert_called_once_with(prefix="documents/")
        self.assertEqual(folders, {"class1", "class2"})

    def test_create_documents_dataframe(self):
        # Test creating documents dataframe
        df = create_documents_dataframe(self.mock_bucket)

        # Verify blobs were listed
        self.mock_bucket.list_blobs.assert_called_once_with(prefix="documents/")

        # Verify dataframe structure
        self.assertEqual(len(df), 2)
        self.assertIn("document_id", df.columns)
        self.assertIn("class_id", df.columns)

        # Verify content
        self.assertTrue(
            "gs://smart_input_data/documents/class1/doc1.pdf" in df["document_id"].values
        )
        self.assertTrue(
            "gs://smart_input_data/documents/class2/doc2.pdf" in df["document_id"].values
        )
        self.assertTrue("class1" in df["class_id"].values)
        self.assertTrue("class2" in df["class_id"].values)

    def test_get_metadata(self):
        # Configure the mocks for the bucket blobs
        def get_blob(path):
            if path == "meta_data/meta.csv":
                return self.mock_meta_blob
            elif path == "meta_data/access.csv":
                return self.mock_access_blob
            return None

        self.mock_bucket.blob.side_effect = get_blob

        # Test getting metadata
        meta_df, access_df = get_metadata(self.mock_bucket)

        # Verify blobs were retrieved
        self.mock_bucket.blob.assert_any_call("meta_data/meta.csv")
        self.mock_bucket.blob.assert_any_call("meta_data/access.csv")

        # Verify dataframes
        self.assertEqual(len(meta_df), 2)
        self.assertEqual(len(access_df), 2)
        self.assertIn("class_id", meta_df.columns)
        self.assertIn("name", meta_df.columns)
        self.assertIn("user_id", access_df.columns)
        self.assertIn("class_id", access_df.columns)

        # Test error handling for meta.csv
        self.mock_meta_blob.download_as_string.side_effect = Exception("Download failed")

        meta_df, access_df = get_metadata(self.mock_bucket)

        # meta_df should be empty, access_df should be valid
        self.assertTrue(meta_df.empty)
        self.assertEqual(len(access_df), 2)

    def test_validate_data(self):
        # Create test dataframes
        meta_df = pd.DataFrame({"class_id": ["class1", "class2"], "name": ["Class 1", "Class 2"]})

        access_df = pd.DataFrame({"user_id": ["user1", "user2"], "class_id": ["class1", "class2"]})

        document_folders = {"class1", "class2"}

        # Test successful validation
        result = validate_data(document_folders, meta_df, access_df)
        self.assertTrue(result)

        # Test mismatched IDs
        meta_df = pd.DataFrame(
            {
                "class_id": ["class1", "class3"],  # class3 is not in document_folders
                "name": ["Class 1", "Class 3"],
            }
        )

        result = validate_data(document_folders, meta_df, access_df)
        self.assertFalse(result)

        # Test missing metadata
        with self.assertRaises(RuntimeError):
            validate_data(document_folders, None, access_df)

    @patch("src.datapipeline.datapipeline.tempfile.NamedTemporaryFile")
    @patch("src.datapipeline.datapipeline.PyPDFLoader")
    @patch("src.datapipeline.datapipeline.os.unlink")
    @patch("os.path.isfile")
    @patch("mimetypes.guess_type")
    @patch("mimetypes.init")
    def test_load_pdf_from_gcs(
        self,
        mock_mime_init,
        mock_mime_guess,
        mock_isfile,
        mock_unlink,
        mock_pdf_loader,
        mock_temp_file,
    ):
        # Mock mimetypes to avoid file system issues
        mock_mime_init.return_value = None
        mock_mime_guess.return_value = ("application/pdf", None)

        # Make sure os.path.isfile returns True for our temp file
        mock_isfile.return_value = True

        # Configure mocks
        mock_temp = MagicMock()
        mock_temp.name = "/tmp/test.pdf"
        mock_temp_file.return_value.__enter__.return_value = mock_temp

        mock_blob = MagicMock()
        self.mock_bucket.blob.return_value = mock_blob

        mock_doc = Document(page_content="Test content", metadata={"page": 1})
        mock_loader = MagicMock()
        mock_loader.load.return_value = [mock_doc]
        mock_pdf_loader.return_value = mock_loader

        # Test loading PDF from GCS
        gcs_path = "documents/class1/doc1.pdf"
        docs = load_pdf_from_gcs(self.mock_bucket, gcs_path)

        # Verify GCS blob was retrieved and downloaded
        self.mock_bucket.blob.assert_called_once_with(gcs_path)
        mock_blob.download_to_filename.assert_called_once_with("/tmp/test.pdf")

        # Verify PDF was loaded
        mock_pdf_loader.assert_called_once_with("/tmp/test.pdf")
        mock_loader.load.assert_called_once()

        # Verify temporary file was cleaned up
        mock_unlink.assert_called_once_with("/tmp/test.pdf")

        # Verify documents were returned with correct metadata
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, "Test content")
        self.assertEqual(docs[0].metadata["source"], f"gs://{BUCKET_NAME}/{gcs_path}")

        # Test error handling - FIXED TO EXPECT RUNTIME ERROR
        mock_loader.load.return_value = []

        with self.assertRaises(RuntimeError):
            load_pdf_from_gcs(self.mock_bucket, gcs_path)

    def test_recursive_chunking(self):
        # Create a new patch for this test
        with patch(
            "src.datapipeline.datapipeline.RecursiveCharacterTextSplitter"
        ) as mock_splitter_class:
            # Configure mock
            mock_splitter = MagicMock()
            mock_splitter_class.return_value = mock_splitter

            mock_doc = Document(page_content="Test content", metadata={"page": 1})
            mock_chunk = Document(page_content="Test chunk", metadata={"page": 1})

            mock_splitter.split_documents.return_value = [mock_chunk]

            # Test chunking
            docs = [mock_doc]
            chunks = recursive_chunking(docs, chunk_size=1000, chunk_overlap=100)

            # Verify splitter was created with correct parameters
            mock_splitter_class.assert_called_once_with(
                chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""]
            )

            # Verify documents were split
            mock_splitter.split_documents.assert_called_once_with(docs)

            # Verify chunks were returned
            self.assertEqual(chunks, [mock_chunk])

            # Test error handling
            mock_splitter.split_documents.side_effect = Exception("Chunking failed")

            with self.assertRaises(RuntimeError):
                recursive_chunking(docs)

    def test_semantic_chunking(self):
        # Use patch within the test to ensure it's applied correctly
        with patch(
            "src.datapipeline.datapipeline.torch.cuda.is_available", return_value=False
        ), patch(
            "src.datapipeline.datapipeline.AdvancedSemanticChunker"
        ) as mock_chunker_class, patch(
            "src.datapipeline.datapipeline.get_embedding_model"
        ) as mock_get_model:

            # Configure mocks
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            mock_get_model.return_value = mock_model

            mock_chunker = MagicMock()
            mock_chunker_class.return_value = mock_chunker

            mock_doc = Document(page_content="Test content", metadata={"page": 1})
            mock_chunk = Document(page_content="Test chunk", metadata={"page": 1})

            mock_chunker.split_documents.return_value = [mock_chunk]

            # Test with no preloaded model
            docs = [mock_doc]
            chunks = semantic_chunking(docs)

            # Verify model was loaded
            mock_get_model.assert_called_once()

            # Verify model was moved to CPU (since CUDA is mocked as not available)
            mock_model.to.assert_called_once_with("cpu")

            # Verify chunker was created with correct parameters
            mock_chunker_class.assert_called_once()

            # Verify documents were split
            mock_chunker.split_documents.assert_called_once_with(docs)

            # Verify chunks were returned
            self.assertEqual(chunks, [mock_chunk])

    def test_clean_chunks(self):
        # Test with Document objects
        doc1 = Document(page_content="  Test \x00 content  ", metadata={"page": 1})
        doc2 = Document(page_content="\n\nAnother\n\nchunk\n\n", metadata={"page": 2})
        doc3 = Document(page_content="", metadata={"page": 3})  # Empty chunk

        chunks = [doc1, doc2, doc3, None]  # Include None to test handling of None

        cleaned = clean_chunks(chunks)

        # Verify cleaning - FIXED ASSERTION
        self.assertEqual(cleaned[0].page_content, "Test content")
        self.assertEqual(cleaned[1].page_content, "Another chunk")
        self.assertEqual(len(cleaned), 2)  # Only 2 non-empty chunks should remain

        # Test with strings
        strings = ["  Test \x00 string  ", "\n\nAnother\n\nstring\n\n", "", None]

        cleaned = clean_chunks(strings)

        # Verify string cleaning
        self.assertEqual(cleaned[0], "Test string")
        self.assertEqual(cleaned[1], "Another string")
        self.assertEqual(len(cleaned), 2)

    def test_clean_document_text(self):
        # Test with Document objects
        doc1 = Document(page_content="  Test \x00 content  ", metadata={"page": 1})
        doc2 = Document(page_content="\n\nAnother\n\nchunk\n\n", metadata={"page": 2})
        doc3 = Document(page_content="", metadata={"page": 3})  # Empty document

        docs = [doc1, doc2, doc3, None]  # Include None to test handling of None

        cleaned = clean_document_text(docs)

        # Verify cleaning
        self.assertEqual(len(cleaned), 2)  # Empty and None should be removed
        self.assertEqual(cleaned[0].page_content, "Test content")
        self.assertEqual(cleaned[1].page_content, "Another chunk")
        self.assertEqual(cleaned[0].metadata, {"page": 1})
        self.assertEqual(cleaned[1].metadata, {"page": 2})

    @patch("src.datapipeline.datapipeline.create_documents_dataframe")
    @patch("src.datapipeline.datapipeline.load_pdf_from_gcs")
    @patch("src.datapipeline.datapipeline.clean_document_text")
    @patch("src.datapipeline.datapipeline.recursive_chunking")
    @patch("src.datapipeline.datapipeline.semantic_chunking")
    @patch("src.datapipeline.datapipeline.clean_chunks")
    @patch("src.datapipeline.datapipeline.get_embedding_model")
    @patch("src.datapipeline.datapipeline.torch.cuda.is_available")
    @patch("src.datapipeline.datapipeline.torch.cuda.empty_cache")
    def test_chunk_documents_from_gcs(
        self,
        mock_empty_cache,
        mock_cuda,
        mock_get_model,
        mock_clean_chunks,
        mock_semantic,
        mock_recursive,
        mock_clean_text,
        mock_load_pdf,
        mock_create_df,
    ):
        # Configure mocks
        mock_cuda.return_value = False  # No CUDA

        # Create dataframe with document paths
        mock_df = pd.DataFrame(
            {
                "document_id": ["gs://smart_input_data/documents/class1/doc1.pdf"],
                "class_id": ["class1"],
            }
        )
        mock_create_df.return_value = mock_df

        # Mock document loading and cleaning
        mock_doc = Document(page_content="Test content", metadata={"page": 1})
        mock_load_pdf.return_value = [mock_doc]
        mock_clean_text.return_value = [mock_doc]

        # Mock chunking
        mock_chunk = Document(page_content="Test chunk", metadata={"page": 1, "source": "test"})
        mock_recursive.return_value = [mock_chunk]
        mock_semantic.return_value = [mock_chunk]
        mock_clean_chunks.return_value = [mock_chunk]

        # Test recursive chunking
        chunks = chunk_documents_from_gcs(self.mock_bucket, "recursive")

        # Verify dataframe was created
        mock_create_df.assert_called_once_with(self.mock_bucket)

        # Verify PDF was loaded and cleaned
        mock_load_pdf.assert_called_once()
        mock_clean_text.assert_called_once_with([mock_doc])

        # Verify recursive chunking was used
        mock_recursive.assert_called_once()
        mock_semantic.assert_not_called()

        # Verify chunks were cleaned
        mock_clean_chunks.assert_called_once_with([mock_chunk])

        # Verify chunks were returned
        self.assertEqual(chunks, [mock_chunk])

    def test_get_ch_embedding_model(self):
        # Use a new patch to ensure it's applied correctly
        with patch("src.datapipeline.datapipeline.SentenceTransformer") as mock_transformer, patch(
            "src.datapipeline.datapipeline.torch.cuda.is_available", return_value=False
        ):

            # Configure mocks
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            mock_transformer.return_value = mock_model

            # Test loading model
            model = get_ch_embedding_model()

            # Verify model was loaded
            mock_transformer.assert_called_once_with("all-mpnet-base-v2")

            # Verify model was moved to CPU
            mock_model.to.assert_called_once_with("cpu")

            # Verify model was returned
            self.assertEqual(model, mock_model)

    @patch("src.datapipeline.datapipeline.torch.cuda.is_available")
    def test_create_chunk_embeddings(self, mock_cuda):
        # Configure mocks
        mock_cuda.return_value = False  # No CUDA

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

        # Test creating embeddings
        chunk_texts = ["Text 1", "Text 2"]
        embeddings = create_chunk_embeddings(chunk_texts, mock_model)

        # Verify encode was called with the right parameters
        mock_model.encode.assert_called_once_with(chunk_texts, device="cpu", batch_size=4)

        # Verify embeddings were returned
        self.assertTrue(np.array_equal(embeddings, np.array([[0.1, 0.2], [0.3, 0.4]])))

        # Test error handling
        mock_model.encode.side_effect = Exception("Encoding failed")

        with self.assertRaises(RuntimeError):
            create_chunk_embeddings(chunk_texts, mock_model)

    def test_create_and_insert_chunks(self):
        # Use fresh patches to ensure they're applied correctly
        with patch("src.datapipeline.datapipeline.connect_to_postgres") as mock_connect, patch(
            "src.datapipeline.datapipeline.sessionmaker"
        ) as mock_sessionmaker, patch(
            "src.datapipeline.datapipeline.get_ch_embedding_model"
        ) as mock_get_model, patch(
            "src.datapipeline.datapipeline.create_chunk_embeddings"
        ) as mock_create_embeddings:

            # Configure mocks
            mock_engine = MagicMock()
            mock_connect.return_value = mock_engine

            mock_session_class = MagicMock()
            mock_sessionmaker.return_value = mock_session_class

            mock_session = MagicMock()
            mock_session_class.return_value = mock_session

            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            # Mock embeddings
            mock_create_embeddings.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

            # Create test chunks
            chunks = [
                Document(page_content="Chunk 1", metadata={"source": "doc1", "page": 1}),
                Document(page_content="Chunk 2", metadata={"source": "doc2", "page": 2}),
            ]

            # Test inserting chunks
            inserted_count = create_and_insert_chunks(chunks)

            # Verify database connection was established
            mock_connect.assert_called_once()
            mock_sessionmaker.assert_called_once_with(bind=mock_engine)

            # Verify embedding model was loaded
            mock_get_model.assert_called_once()

            # Verify embeddings were created
            mock_create_embeddings.assert_called_once_with(["Chunk 1", "Chunk 2"], mock_model)

            # Verify SQL was executed for each chunk
            self.assertEqual(mock_session.execute.call_count, 2)

            # Verify session was committed
            mock_session.commit.assert_called()

            # Verify correct count was returned
            self.assertEqual(inserted_count, 2)

    def test_main(self):
        # Use fresh patches to ensure they're applied correctly
        with patch("src.datapipeline.datapipeline.connect_to_bucket") as mock_connect_bucket, patch(
            "src.datapipeline.datapipeline.list_document_folders"
        ) as mock_list_folders, patch(
            "src.datapipeline.datapipeline.get_metadata"
        ) as mock_get_metadata, patch(
            "src.datapipeline.datapipeline.create_documents_dataframe"
        ) as mock_create_df, patch(
            "src.datapipeline.datapipeline.validate_data"
        ) as mock_validate, patch(
            "src.datapipeline.datapipeline.chunk_documents_from_gcs"
        ) as mock_chunk_docs, patch(
            "src.datapipeline.datapipeline.connect_to_postgres"
        ) as mock_connect_db, patch(
            "src.datapipeline.datapipeline.create_and_insert_chunks"
        ) as mock_insert_chunks, patch(
            "src.datapipeline.datapipeline.text"
        ) as mock_text:

            # Configure mocks
            mock_bucket = MagicMock()
            mock_connect_bucket.return_value = mock_bucket

            mock_folders = {"class1", "class2"}
            mock_list_folders.return_value = mock_folders

            mock_meta_df = pd.DataFrame(
                {"class_id": ["class1", "class2"], "name": ["Class 1", "Class 2"]}
            )

            mock_access_df = pd.DataFrame(
                {"user_id": ["user1", "user2"], "class_id": ["class1", "class2"]}
            )

            mock_get_metadata.return_value = (mock_meta_df, mock_access_df)

            mock_docs_df = pd.DataFrame(
                {"document_id": ["doc1", "doc2"], "class_id": ["class1", "class2"]}
            )

            mock_create_df.return_value = mock_docs_df

            mock_validate.return_value = True

            mock_chunks = [
                Document(page_content="Chunk 1", metadata={"source": "doc1", "page": 1}),
                Document(page_content="Chunk 2", metadata={"source": "doc2", "page": 2}),
            ]

            mock_chunk_docs.return_value = mock_chunks

            mock_engine = MagicMock()
            mock_connect_db.return_value = mock_engine

            # Mock the connection context manager
            mock_connection = MagicMock()
            mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_connection)
            mock_engine.connect.return_value.__exit__ = MagicMock(return_value=None)

            mock_insert_chunks.return_value = 2

            # Test main function with recursive chunking
            result = main(chunk_method="recursive")

            # Verify bucket connection was established
            mock_connect_bucket.assert_called_once()

            # Verify folders were listed
            mock_list_folders.assert_called_once_with(mock_bucket)

            # Verify metadata was retrieved
            mock_get_metadata.assert_called_once_with(mock_bucket)

            # Verify documents dataframe was created
            mock_create_df.assert_called_once_with(mock_bucket)

            # Verify data was validated
            mock_validate.assert_called_once_with(mock_folders, mock_meta_df, mock_access_df)

            # Verify documents were chunked with recursive method
            mock_chunk_docs.assert_called_once_with(mock_bucket, "recursive")

            # Verify database connection was established
            mock_connect_db.assert_called_once()

            # Verify chunks were inserted
            mock_insert_chunks.assert_called_once_with(mock_chunks)

            # Verify correct count was returned
            self.assertEqual(result, 2)


if __name__ == "__main__":
    unittest.main()
