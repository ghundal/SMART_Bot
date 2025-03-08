import os
import logging
from google.cloud import storage
import pandas as pd

# Set up environment
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "../secrets/smart-452816-101a65261db2.json"
GCP_PROJECT = "SMART"
BUCKET_NAME = 'smart_input_data'
DOCUMENTS_FOLDER = 'documents/'
METADATA_FOLDER = 'meta_data/'
LOCAL_PDF_DIR = "./data/pdfs"
LOCAL_META_DIR = "./data/meta"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_pipeline')

def connect_to_bucket():
    """Connect to the GCS bucket and return the bucket object."""
    try:
        client = storage.Client(project=os.environ.get('GCP_PROJECT'))
        bucket = client.bucket(BUCKET_NAME)
        logger.info(f"Successfully connected to bucket: {BUCKET_NAME}")
        return bucket
    except Exception as e:
        logger.error(f"Failed to connect to bucket {BUCKET_NAME}: {str(e)}")
        raise

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


def download_pdfs_by_folder(bucket, folder_name):
    """Download PDFs from a specific folder in GCS."""
    folder_path = os.path.join(LOCAL_PDF_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    prefix = f"{DOCUMENTS_FOLDER}{folder_name}/"
    blobs = bucket.list_blobs(prefix=prefix)
    pdf_files = [blob for blob in blobs if blob.name.endswith('.pdf')]
    
    logger.info(f"Downloading {len(pdf_files)} PDFs from folder: {folder_name}")
    
    for blob in pdf_files:
        file_name = os.path.basename(blob.name)
        local_path = os.path.join(folder_path, file_name)
        try:
            blob.download_to_filename(local_path)
            logger.debug(f"Downloaded: {blob.name} -> {local_path}")
        except Exception as e:
            logger.error(f"Failed to download {blob.name}: {str(e)}")
    
    return folder_path

def download_metadata(bucket):
    """Download meta.csv and access.csv from GCS, handling errors."""
    os.makedirs(LOCAL_META_DIR, exist_ok=True)

    metadata_files = ["meta.csv", "access.csv"]
    dataframes = {}

    for file_name in metadata_files:
        local_path = os.path.join(LOCAL_META_DIR, file_name)
        try:
            blob = bucket.blob(f"{METADATA_FOLDER}{file_name}")
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded {file_name} -> {local_path}")

            # Load into Pandas DataFrame
            dataframes[file_name] = pd.read_csv(local_path)
        except Exception as e:
            logger.error(f"Failed to download or load {file_name}: {str(e)}")
            dataframes[file_name] = None  # Set None to avoid crashes later

    return dataframes.get("meta.csv"), dataframes.get("access.csv")

def validate_data(document_folders, meta_df, access_df):
    """Validate that all unique IDs in meta and access match document folder names."""
    logger.info("Validating consistency between metadata and document folders...")

    if meta_df is None or access_df is None:
        logger.warning("Skipping validation: One or more metadata files failed to load.")
        return False

    # Extract unique IDs from meta.csv and access.csv
    meta_ids = set(meta_df['id'].astype(str).unique()) if 'id' in meta_df.columns else set()
    access_ids = set(access_df['id'].astype(str).unique()) if 'id' in access_df.columns else set()

    # Convert document folder names to strings for consistency
    document_folders = set(str(folder) for folder in document_folders)

    # Check if IDs match
    if meta_ids == access_ids == document_folders:
        logger.info("✅ Validation PASSED: IDs match across metadata and document folders.")
        return True
    else:
        logger.warning("❌ Validation FAILED: Mismatches detected.")

        if document_folders - meta_ids:
            logger.warning(f" - In documents but missing in meta.csv: {document_folders - meta_ids}")
        if meta_ids - document_folders:
            logger.warning(f" - In meta.csv but missing in documents: {meta_ids - document_folders}")

        if document_folders - access_ids:
            logger.warning(f" - In documents but missing in access.csv: {document_folders - access_ids}")
        if access_ids - document_folders:
            logger.warning(f" - In access.csv but missing in documents: {access_ids - document_folders}")

        if meta_ids - access_ids:
            logger.warning(f" - In meta.csv but missing in access.csv: {meta_ids - access_ids}")
        if access_ids - meta_ids:
            logger.warning(f" - In access.csv but missing in meta.csv: {access_ids - meta_ids}")

        return False

def main():
    """Main function to execute the pipeline."""
    logger.info("🚀 Starting data pipeline...")

    try:
        bucket = connect_to_bucket()
        document_folders = list_document_folders(bucket)

        # Download PDFs
        for folder_name in document_folders:
            # Download PDFs for this folder
            download_pdfs_by_folder(bucket, folder_name)

        # Download metadata
        meta_df, access_df = download_metadata(bucket)

        # Run validation
        validate_data(document_folders, meta_df, access_df)

        logger.info("🎯 Data pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")

if __name__ == "__main__":
    main()

