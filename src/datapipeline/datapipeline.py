import logging
import os
import tempfile

import pandas as pd
import torch
from google.cloud import storage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

try:
    # Try package-relative import first (for tests and package usage)
    from src.datapipeline.Advanced_semantic_chunker import (
        AdvancedSemanticChunker,
        get_embedding_model,
    )
except ImportError:
    try:
        # Try direct import (for when run directly)
        from Advanced_semantic_chunker import AdvancedSemanticChunker, get_embedding_model
    except ImportError:
        # Try relative import (for when in package)
        from .Advanced_semantic_chunker import AdvancedSemanticChunker, get_embedding_model

# Set up environment
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../secrets/smart_input_key.json"
GCP_PROJECT = "SMART"
BUCKET_NAME = "smart_input_data"
DOCUMENTS_FOLDER = "documents/"
METADATA_FOLDER = "meta_data/"
OUTPUT_DIR = "./data/chunks"

# Database connection settings
DB_NAME = "smart"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = "5432"

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 200

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("data_pipeline")

# Suppress PyPDF warnings about wrong pointing objects
logging.getLogger("pypdf._reader").setLevel(logging.ERROR)


def connect_to_bucket():
    """Connect to the GCS bucket and return the bucket object."""
    try:
        client = storage.Client(project=os.environ.get("GCP_PROJECT"))
        bucket = client.bucket(BUCKET_NAME)
        logger.info(f"Successfully connected to bucket: {BUCKET_NAME}")
        return bucket
    except Exception as e:
        logger.exception(f"Failed to connect to bucket {BUCKET_NAME}: {str(e)}")
        raise RuntimeError(f"Critical failure: Unable to connect to GCS bucket: {str(e)}")


def connect_to_postgres():
    """Connect to the postgres and return the connection."""
    try:
        return create_engine(
            f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )
    except Exception as e:
        logger.exception(f"Failed to connect to postgres! {str(e)}")
        raise RuntimeError("Critical failure: Unable to connect to postgres!")


def list_document_folders(bucket):
    """Retrieve unique class folder names from GCS under 'documents/'."""
    logger.info("Fetching document folder names from GCS...")

    blobs = bucket.list_blobs(prefix=DOCUMENTS_FOLDER)
    folder_names = set()

    for blob in blobs:
        parts = blob.name.split("/")
        if len(parts) > 2:
            folder_names.add(parts[1])

    logger.info(f"Found {len(folder_names)} document folders: {folder_names}")
    return folder_names


def create_documents_dataframe(bucket):
    """Create a dataframe with class ID and document GCP path."""
    logger.info("Creating documents dataframe...")

    blobs = bucket.list_blobs(prefix=DOCUMENTS_FOLDER)
    documents = []

    for blob in blobs:
        if blob.name.endswith(".pdf"):
            parts = blob.name.split("/")
            if len(parts) > 2:
                class_id = parts[1]
                document_path = f"gs://{BUCKET_NAME}/{blob.name}"  # Full GCP path
                documents.append(
                    {
                        "document_id": document_path,  # Store GCP path
                        "class_id": class_id,
                    }
                )

    docs_df = pd.DataFrame(documents)
    logger.info(f"Created documents dataframe with {len(docs_df)} entries")
    return docs_df


def get_metadata(bucket):
    """Download meta.csv and access.csv from GCS as dataframes."""
    metadata_files = ["meta.csv", "access.csv"]
    dataframes = {}

    for file_name in metadata_files:
        try:
            blob = bucket.blob(f"{METADATA_FOLDER}{file_name}")
            content = blob.download_as_string()
            logger.info(f"Downloaded {file_name} content")

            # Load into Pandas DataFrame directly from content
            dataframes[file_name] = pd.read_csv(pd.io.common.BytesIO(content))
        except Exception as e:
            logger.exception(f"Failed to download or load {file_name}: {str(e)}")
            dataframes[file_name] = pd.DataFrame()  # Return empty

    return dataframes.get("meta.csv"), dataframes.get("access.csv")


def validate_data(document_folders, meta_df, access_df):
    """Validate that all unique IDs in meta and access match document folder names."""
    logger.info("Validating consistency between metadata and document folders...")

    if meta_df is None or access_df is None:
        error_msg = "Critical failure: One or more metadata files failed to load."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Extract unique IDs from meta.csv and access.csv
    meta_ids = (
        set(meta_df["class_id"].astype(str).unique()) if "class_id" in meta_df.columns else set()
    )
    access_ids = (
        set(access_df["class_id"].astype(str).unique())
        if "class_id" in access_df.columns
        else set()
    )

    # Convert document folder names to strings for consistency
    document_folders = set(str(folder) for folder in document_folders)

    # Check if IDs match
    if meta_ids == access_ids == document_folders:
        logger.info("✅ Validation PASSED: IDs match across metadata and document folders.")
        return True
    else:
        logger.warning("❌ Validation FAILED: Mismatches detected.")

        if document_folders - meta_ids:
            logger.warning(
                f" - In documents but missing in meta.csv: {document_folders - meta_ids}"
            )
        if meta_ids - document_folders:
            logger.warning(
                f" - In meta.csv but missing in documents: {meta_ids - document_folders}"
            )

        if document_folders - access_ids:
            logger.warning(
                f" - In documents but missing in access.csv: {document_folders - access_ids}"
            )
        if access_ids - document_folders:
            logger.warning(
                f" - In access.csv but missing in documents: {access_ids - document_folders}"
            )

        if meta_ids - access_ids:
            logger.warning(f" - In meta.csv but missing in access.csv: {meta_ids - access_ids}")
        if access_ids - meta_ids:
            logger.warning(f" - In access.csv but missing in meta.csv: {access_ids - meta_ids}")

        return False


def load_pdf_from_gcs(bucket, gcs_path):
    """Load a PDF document directly from GCS using a temporary file."""
    try:
        # Remove the gs://bucket-name/ prefix if present
        if gcs_path.startswith(f"gs://{BUCKET_NAME}/"):
            gcs_path = gcs_path[len(f"gs://{BUCKET_NAME}/") :]

        blob = bucket.blob(gcs_path)

        # Create a temporary file to store the PDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_path = temp_file.name
            blob.download_to_filename(temp_path)

        # Use PyPDFLoader to load the PDF
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        # Clean up the temporary file
        os.unlink(temp_path)

        # Set the source in metadata to the GCS path including bucket name
        gcs_full_path = f"gs://{BUCKET_NAME}/{gcs_path}"
        for doc in documents:
            doc.metadata["source"] = gcs_full_path

        if not documents:
            raise ValueError(f"No content extracted from PDF: {gcs_path}")

        return documents

    except Exception as e:
        logger.exception(f"Error loading PDF from GCS {gcs_path}: {str(e)}")
        raise RuntimeError(f"Critical failure: Unable to load PDF from {gcs_path}: {str(e)}")


def recursive_chunking(
    documents, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP
):
    """Split documents using recursive chunking method."""
    logger.info(
        f"Chunking documents using recursive method with size={chunk_size}, overlap={chunk_overlap}"
    )

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Split documents into chunks
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks using recursive method")

        return chunks

    except Exception as e:
        logger.exception(f"Error with recursive chunking: {str(e)}")
        raise RuntimeError(f"Critical failure: Unable to perform recursive chunking: {str(e)}")


def semantic_chunking(
    documents,
    embedding_model="all-MiniLM-L6-v2",
    buffer_size=1,
    breakpoint_type="percentile",
    breakpoint_amount=None,
    preloaded_model=None,
):
    """Split documents using semantic chunking method."""
    logger.info(f"Chunking documents using semantic method with model: {embedding_model}")

    try:
        # Use the preloaded model
        model = preloaded_model
        if model is None:
            logger.info(f"Loading embedding model: {embedding_model}")
            model = get_embedding_model(embedding_model)

        # Move model to CUDA
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        logger.info(f"Using device: {device} for semantic chunking")

        text_splitter = AdvancedSemanticChunker(
            embedding_model=embedding_model,
            buffer_size=buffer_size,
            breakpoint_threshold_type=breakpoint_type,
            breakpoint_threshold_amount=breakpoint_amount,
            embedding_function=lambda texts: model.encode(
                texts, device=device, batch_size=4, show_progress_bar=False
            ),
        )

        # Split documents into chunks
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks using semantic method")

        return chunks

    except Exception as e:
        logger.exception(f"Error with semantic chunking: {str(e)}")
        raise RuntimeError(f"Critical failure: Unable to perform semantic chunking: {str(e)}")


def clean_chunks(chunks):
    """
    Cleans chunks by aggressively removing whitespace and NUL (0x00) characters.
    """
    cleaned_chunks = []

    for chunk in chunks:
        # Skip None chunks
        if chunk is None:
            continue

        # Convert chunk to string and remove NUL characters
        chunk_text = str(chunk).replace("\x00", "")

        # For LangChain Document objects
        if hasattr(chunk, "page_content"):
            chunk.page_content = " ".join(chunk.page_content.replace("\x00", "").split()).strip()
            if chunk.page_content:
                cleaned_chunks.append(chunk)
        # For strings
        elif isinstance(chunk, str):
            clean_text = " ".join(chunk_text.split()).strip()
            if clean_text:  # Only keep non-empty chunks
                cleaned_chunks.append(clean_text)
        # For other objects
        else:
            clean_text = " ".join(chunk_text.split()).strip()
            if clean_text:  # Only keep non-empty chunks
                cleaned_chunks.append(clean_text)

    return cleaned_chunks


def clean_document_text(documents):
    """
    Cleans document text before chunking by removing various types of problematic
    characters and normalizing whitespace.

    Args:
        documents: List of Document objects from PyPDFLoader

    Returns:
        List of Document objects with cleaned page_content
    """
    import re

    cleaned_documents = []

    for doc in documents:
        if doc is None or not hasattr(doc, "page_content"):
            continue

        # Get the current text
        text = doc.page_content

        # Remove NUL characters and other control characters
        text = text.replace("\x00", "")

        # Remove non-printable and control characters except normal whitespace
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)

        # Normalize whitespace (combine multiple spaces, tabs, newlines to single space)
        text = re.sub(r"\s+", " ", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        # Only include document if it still has content after cleaning
        if text:
            # Create a new document with the same metadata but cleaned text
            doc.page_content = text
            cleaned_documents.append(doc)

    return cleaned_documents


def chunk_documents_from_gcs(bucket, chunk_method):
    """
    Load documents from GCS, chunk them, and return/save the chunks dataframe.

    Args:
        bucket: GCS bucket object
        chunk_method: 'recursive' or 'semantic'

    Returns:
        pandas DataFrame with document_id, page, chunk_text, and embedding columns
    """
    # IMPORTANT: Convert to lowercase and add explicit logging
    chunk_method = chunk_method.lower()
    logger.info(f"Starting document chunking process using '{chunk_method}' method")

    # Get document dataframe with GCP paths
    docs_df = create_documents_dataframe(bucket)
    if docs_df.empty:
        raise RuntimeError("Critical failure: No documents found in GCS bucket")

    all_chunks = []

    # Load the embedding model once if using semantic chunking
    embedding_model = None
    if chunk_method == "semantic":
        logger.info("Loading embedding model for chunking")
        embedding_model = get_embedding_model("all-MiniLM-L6-v2").to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # Process each document
    for _, row in docs_df.iterrows():
        gcs_path = row["document_id"]

        # Extract just the filename part for logging
        filename = gcs_path.split("/")[-1] if "/" in gcs_path else gcs_path
        logger.info(f"Processing PDF: {filename}")

        # Load the PDF directly from GCS
        documents = load_pdf_from_gcs(bucket, gcs_path)

        if not documents:
            logger.warning(f"No documents loaded from {gcs_path}. Skipping.")
            continue

        # Clean the documents before chunking
        logger.info(f"Cleaning text for {filename} before chunking")
        cleaned_documents = clean_document_text(documents)

        if not cleaned_documents:
            logger.warning(f"No documents remained after cleaning for {filename}. Skipping.")
            continue

        # Apply the appropriate chunking method
        if chunk_method == "semantic":
            logger.info(f"Applying SEMANTIC chunking to {filename}")
            chunks = semantic_chunking(
                documents,
                embedding_model="all-MiniLM-L6-v2",
                buffer_size=2,
                breakpoint_type="percentile",
                breakpoint_amount=90,
                preloaded_model=embedding_model,
            )
        else:
            logger.info(f"Applying RECURSIVE chunking to {filename}")
            chunks = recursive_chunking(
                documents,
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            )
        cleaned_chunks = clean_chunks(chunks)
        all_chunks.extend(cleaned_chunks)

    # Free the semantic chunking model after processing
    if embedding_model:
        del embedding_model
        torch.cuda.empty_cache()
        logger.info("Unloaded semantic chunking model and cleared GPU cache")

    # Create DataFrame from all chunks
    if not all_chunks:
        raise RuntimeError("Critical failure: No chunks were created from any document")

    logger.info(f"Total chunks created before embedding: {len(all_chunks)}")

    return all_chunks


def get_ch_embedding_model():
    """Load and return the embedding model."""
    try:
        model_name = "all-mpnet-base-v2"
        # model_name = 'multi-qa-mpnet-base-dot-v1'
        model = SentenceTransformer(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Successfully loaded Embedding model: {model_name}")
        return model
    except Exception:
        logger.exception(f"Error loading Embedding model: {model_name}")
        raise


def create_chunk_embeddings(chunk_texts, embedding_model, batch_size=4):
    """Create embeddings for a list of chunk texts."""

    try:
        embeddings = embedding_model.encode(
            chunk_texts,
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=batch_size,
        )
        logger.info(f"Created {len(embeddings)} embeddings")
        return embeddings
    except Exception as e:
        logger.exception(f"Error creating embeddings: {str(e)}")
        raise RuntimeError(f"Critical failure: Unable to create embeddings: {str(e)}")


def create_and_insert_chunks(all_chunks):
    """Process chunks and insert directly into the database with document_id, page,
    chunk_text, and vector embeddings."""

    # Connect to the database
    engine = connect_to_postgres()
    Session = sessionmaker(bind=engine)
    session = Session()

    inserted_count = 0

    # Load the chunk embedding model
    logger.info("Loading model for chunk embeddings")
    embedding_model = get_ch_embedding_model()

    # Extract all chunk texts for batch processing
    chunk_texts = [chunk.page_content for chunk in all_chunks]

    # Create embeddings for all chunks at once (batch processing)
    logger.info(f"Creating embeddings for {len(chunk_texts)} chunks")
    embeddings = create_chunk_embeddings(chunk_texts, embedding_model)

    for i, chunk in enumerate(all_chunks):
        # Get the document_id (GCS path) from the metadata
        document_id = chunk.metadata.get("source", "unknown")

        # Extract page number
        page_number = chunk.metadata.get("page", 0)

        # Get the embedding
        embedding = embeddings[i] if i < len(embeddings) else None

        if embedding is None:
            raise RuntimeError(f"Critical failure: Missing embedding for chunk {i}")

        # Format the embedding for pgvector
        # Convert numpy array to string format pgvector expects: '[0.1,0.2,...]'
        embedding_str = str(embedding.tolist())

        try:
            # Insert directly into the database
            sql = f"""
            INSERT INTO chunk (document_id, page_number, chunk_text, embedding)
            VALUES (:document_id, :page_number, :chunk_text, '{embedding_str}'::vector)
            """

            session.execute(
                text(sql),
                {
                    "document_id": document_id,
                    "page_number": page_number,
                    "chunk_text": chunk.page_content,
                },
            )

            inserted_count += 1

            # Commit every 100 inserts to avoid large transactions
            if inserted_count % 100 == 0:
                session.commit()
                # logger.info(f"Inserted {inserted_count} chunks so far")

        except Exception as e:
            logger.error(f"Error inserting chunk {i}: {e}")
            session.rollback()

    # Final commit for any remaining inserts
    if inserted_count % 100 != 0:
        session.commit()

    logger.info(f"Successfully inserted {inserted_count} unique chunks into the database")
    return inserted_count


def main(chunk_method="recursive"):
    """
    Main function to execute the chunking pipeline.
    """
    # IMPORTANT: Convert to lowercase and add explicit logging
    chunk_method = chunk_method.lower()
    logger.info(f"🚀 Starting data pipeline with '{chunk_method}' chunking method...")

    try:
        bucket = connect_to_bucket()
        document_folders = list_document_folders(bucket)

        # Get metadata as dataframes
        meta_df, access_df = get_metadata(bucket)

        # Create documents dataframe
        docs_df = create_documents_dataframe(bucket)

        # Display the dataframes
        logger.info(f"Documents dataframe shape: {docs_df.shape}")
        logger.info(f"Meta dataframe shape: {meta_df.shape}")
        logger.info(f"Access dataframe shape: {access_df.shape}")

        # Run validation - will raise exception if validation fails
        validate_data(document_folders, meta_df, access_df)

        # Process documents and get chunks
        logger.info(f"Will use '{chunk_method}' method for chunking documents")
        all_chunks = chunk_documents_from_gcs(bucket, chunk_method)

        # Set connection to postgres
        connection = connect_to_postgres()

        # Delete all rows in tables before inserting new data
        with connection.connect() as conn:
            conn.execute(text("DELETE FROM access;"))
            conn.execute(text("DELETE FROM chunk;"))
            conn.execute(text("DELETE FROM document;"))
            conn.execute(text("DELETE FROM class;"))
            conn.commit()

        # Insert data
        meta_df.to_sql("class", con=connection, if_exists="append", index=False)

        access_df.to_sql("access", con=connection, if_exists="append", index=False)

        docs_df.to_sql("document", con=connection, if_exists="append", index=False)

        inserted_count = create_and_insert_chunks(all_chunks)

        logger.info("🎯 Data pipeline completed successfully!")

        return inserted_count

    except Exception as e:
        logger.exception(f"Pipeline execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    import argparse
    import sys

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run document processing pipeline with specified chunking method"
    )
    parser.add_argument(
        "--chunk-method",
        type=str,
        default="semantic",
        choices=["recursive", "semantic"],
        help="Method to use for document chunking (recursive or semantic)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Log the received arguments for debugging
    logger.info(f"Command line arguments: {sys.argv}")
    logger.info(f"Parsed chunk-method argument: '{args.chunk_method}'")

    try:
        # Run main with specified method
        inserted_count = main(chunk_method=args.chunk_method)
        logger.info(f"Successfully processed {inserted_count} unique chunks with embeddings")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Pipeline failed with error : {str(e)}")
        sys.exit(1)
