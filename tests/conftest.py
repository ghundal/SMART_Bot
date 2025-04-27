import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add the src/datapipeline directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/datapipeline")))


# Define mock classes for dependencies to avoid importing actual modules
class MockSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, texts, show_progress_bar=False, batch_size=None, device=None):
        # Return a dummy embedding of consistent size
        return np.array([[0.1] * 384 for _ in range(len(texts))])

    def to(self, device):
        # Mock moving to device
        return self


# Mock mimetypes module to avoid file system dependencies
@pytest.fixture(autouse=True, scope="session")
def mock_mimetypes():
    """Mock the mimetypes module to avoid looking for mime.types file."""
    with patch("mimetypes.init") as mock_init, patch("mimetypes.guess_type") as mock_guess_type:

        # Just return a reasonable mimetype for PDFs
        mock_guess_type.return_value = ("application/pdf", None)

        # Make init a no-op
        mock_init.return_value = None

        yield


# Mock SentenceTransformer
@pytest.fixture(autouse=True, scope="session")
def mock_sentence_transformer():
    """Mock SentenceTransformer to avoid loading models."""
    with patch("sentence_transformers.SentenceTransformer", MockSentenceTransformer):
        yield


# Mock PyPDFLoader and Blob
@pytest.fixture(autouse=True, scope="session")
def mock_pdf_loader():
    """Mock PyPDFLoader to avoid loading actual PDFs."""
    from langchain_core.documents import Document

    mock_doc = Document(page_content="Test PDF content", metadata={"page": 1})

    class MockBlob:
        @staticmethod
        def from_path(file_path):
            return "mock_blob"

    class MockLoader:
        def load(self):
            return [mock_doc]

        def lazy_load(self):
            yield mock_doc

    with patch(
        "langchain_community.document_loaders.pdf.PyPDFLoader.__init__", return_value=None
    ), patch(
        "langchain_community.document_loaders.pdf.PyPDFLoader.load", return_value=[mock_doc]
    ), patch(
        "langchain_community.document_loaders.pdf.PyPDFLoader.lazy_load", return_value=[mock_doc]
    ), patch(
        "langchain_core.documents.base.Blob", MockBlob
    ), patch(
        "langchain_core.documents.base.Blob.from_path", MockBlob.from_path
    ):
        yield


# Mock os.path.isfile to return True for PDF paths
@pytest.fixture(autouse=True, scope="session")
def mock_isfile():
    """Mock os.path.isfile to return True for PDF paths."""

    def mock_is_file(path):
        if path.endswith(".pdf"):
            return True
        # For other paths, use the actual implementation
        return os.path.isfile(path)

    with patch("os.path.isfile", side_effect=mock_is_file):
        yield


# Create mocks for modules used directly in the datapipeline
@pytest.fixture(scope="session", autouse=True)
def mock_datapipeline_dependencies():
    """Mock dependencies used directly in datapipeline.py"""

    # Mock specific modules imported in datapipeline.py
    with patch("src.datapipeline.datapipeline.storage", MagicMock()), patch(
        "src.datapipeline.datapipeline.torch", MagicMock()
    ), patch("src.datapipeline.datapipeline.SentenceTransformer", MockSentenceTransformer), patch(
        "src.datapipeline.datapipeline.PyPDFLoader", MagicMock()
    ), patch(
        "src.datapipeline.datapipeline.RecursiveCharacterTextSplitter", MagicMock()
    ), patch(
        "src.datapipeline.datapipeline.sessionmaker", MagicMock()
    ), patch(
        "src.datapipeline.datapipeline.text", MagicMock()
    ):

        # Configure torch.cuda
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = False
        sys.modules["src.datapipeline"].torch = torch_mock

        yield


# Create mocks for modules used directly in Advanced_semantic_chunker
@pytest.fixture(scope="session", autouse=True)
def mock_semantic_chunker_dependencies():
    """Mock dependencies used directly in Advanced_semantic_chunker.py"""

    # Mock specific modules imported in Advanced_semantic_chunker.py
    with patch("Advanced_semantic_chunker.SentenceTransformer", MockSentenceTransformer), patch(
        "Advanced_semantic_chunker.np", np
    ):

        yield


# Create mocks for other modules
@pytest.fixture(scope="session", autouse=True)
def mock_modules():
    # Dictionary to store the mocked modules
    mocks = {}

    # Mock these modules if they're imported
    modules_to_mock = [
        "google.cloud",
        "google.cloud.storage",
        "sqlalchemy",
        "torch",
        "langchain.text_splitter",
        "transformers",
        "pypdf",
        "requests",
    ]

    # Create the mocks
    for module_name in modules_to_mock:
        if module_name not in sys.modules:
            mocks[module_name] = MagicMock()
            sys.modules[module_name] = mocks[module_name]

    # Set environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./mock_credentials.json"

    # Give access to the mocks
    yield mocks

    # Clean up the modules after tests
    for module_name in modules_to_mock:
        if module_name in sys.modules and module_name in mocks:
            del sys.modules[module_name]
