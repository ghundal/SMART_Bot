import os
import logging
import pandas as pd
import uuid
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Import the AdvancedSemanticChunker from the improved module
from Advanced_semantic_chunker import AdvancedSemanticChunker, get_embedding_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('chunking_comparison')

# Constants
LOCAL_PDF_DIR = "./data/pdfs"
OUTPUT_DIR = "./data/chunks"
PLOTS_DIR = "./data/plots"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Pre-load the embedding model once, so it's ready for semantic chunking
_ = get_embedding_model('all-MiniLM-L6-v2')
logger.info("Pre-loaded embedding model for semantic chunking")

def load_documents_from_directory(directory_path):
    """Load all PDF documents from a directory using LangChain."""
    logger.info(f"Loading PDF documents from: {directory_path}")
    
    try:
        # Check if directory exists
        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return []
            
        # Check if directory contains PDF files
        pdf_files = list(Path(directory_path).glob("**/*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in directory: {directory_path}")
            return []
        
        # Use DirectoryLoader to load all PDFs in the directory
        loader = DirectoryLoader(
            directory_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        # logger.info(f"Successfully loaded {len(documents)} document pages from {directory_path}")
        return documents
    
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        return []

def recursive_chunking(documents, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """Split documents using recursive chunking method."""
    logger.info(f"Chunking documents using recursive method with size={chunk_size}, overlap={chunk_overlap}")
    
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Split documents into chunks
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks using recursive method")
        
        return chunks
    
    except Exception as e:
        logger.error(f"Error with recursive chunking: {str(e)}")
        return []

def semantic_chunking(documents, embedding_model='all-MiniLM-L6-v2', buffer_size=1, 
                     breakpoint_type='percentile', breakpoint_amount=None):
    """Split documents using semantic chunking method."""
    logger.info(f"Chunking documents using semantic method with model: {embedding_model}")
    
    try:
        text_splitter = AdvancedSemanticChunker(
            embedding_model=embedding_model,
            buffer_size=buffer_size,
            breakpoint_threshold_type=breakpoint_type,
            breakpoint_threshold_amount=breakpoint_amount
        )
        
        # Split documents into chunks
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks using semantic method")
        
        return chunks
    
    except Exception as e:
        logger.error(f"Error with semantic chunking: {str(e)}")
        return []

def create_chunks_dataframe(chunks, folder_name, method_name):
    """Convert chunks to a pandas DataFrame with metadata."""
    data = []
    
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get('source', 'unknown')
        
        # Generate a unique ID for each chunk
        chunk_id = str(uuid.uuid4())
        
        # Extract relevant metadata
        data.append({
            "chunk_id": chunk_id,
            "document": Path(source).name,
            "class": folder_name,
            "page": chunk.metadata.get('page', 0),
            "chunk_text": chunk.page_content,
            "chunk_length": len(chunk.page_content),
            "chunk_method": method_name
        })
    
    # Create DataFrame
    return pd.DataFrame(data)

def save_chunks(df, folder_name, method_name, output_dir=OUTPUT_DIR, format="csv"):
    """Save chunks DataFrame to disk with specified format (csv by default)."""
    os.makedirs(output_dir, exist_ok=True)
    
    if format.lower() == "parquet":
        try:
            output_file = os.path.join(output_dir, f"chunks-{method_name}-{folder_name}.parquet")
            df.to_parquet(output_file, index=False)
            logger.info(f"Saved {len(df)} chunks to {output_file}")
            return output_file
        except Exception as e:
            logger.warning(f"Could not save as Parquet: {str(e)}")
            logger.info("Falling back to CSV format...")
            format = "csv"
    
    if format.lower() == "csv":
        try:
            output_file = os.path.join(output_dir, f"chunks-{method_name}-{folder_name}.csv")
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(df)} chunks to {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Error saving chunks as CSV: {str(e)}")
            return None

def analyze_chunk_quality(df_recursive, df_semantic, folder_name):
    """Analyze and compare chunk quality between methods."""
    logger.info(f"Analyzing chunk quality for {folder_name}")
    
    # Compute basic statistics
    stats_recursive = df_recursive['chunk_length'].describe()
    stats_semantic = df_semantic['chunk_length'].describe()
    
    logger.info(f"\nRecursive Chunking Stats for {folder_name}:\n{stats_recursive}")
    logger.info(f"\nSemantic Chunking Stats for {folder_name}:\n{stats_semantic}")
    
    # Count extreme chunks
    short_recursive = (df_recursive['chunk_length'] < 100).sum()
    short_semantic = (df_semantic['chunk_length'] < 100).sum()
    
    long_recursive = (df_recursive['chunk_length'] > 2000).sum()
    long_semantic = (df_semantic['chunk_length'] > 2000).sum()
    
    logger.info(f"\nRecursive: {short_recursive} short chunks, {long_recursive} long chunks")
    logger.info(f"Semantic: {short_semantic} short chunks, {long_semantic} long chunks")
    
    # Create comparison plots
    create_comparison_plots(df_recursive, df_semantic, folder_name)
    
    # Return comparison metrics
    return {
        "recursive": {
            "count": len(df_recursive),
            "mean_length": stats_recursive['mean'],
            "std_length": stats_recursive['std'],
            "min_length": stats_recursive['min'],
            "max_length": stats_recursive['max'],
            "short_chunks": short_recursive,
            "long_chunks": long_recursive
        },
        "semantic": {
            "count": len(df_semantic),
            "mean_length": stats_semantic['mean'],
            "std_length": stats_semantic['std'],
            "min_length": stats_semantic['min'],
            "max_length": stats_semantic['max'],
            "short_chunks": short_semantic,
            "long_chunks": long_semantic
        }
    }

def create_comparison_plots(df_recursive, df_semantic, folder_name):
    """Create plots comparing the two chunking methods."""
    try:
        plt.figure(figsize=(12, 8))
        
        # Create a combined DataFrame for easier plotting
        df_recursive_plot = df_recursive[['chunk_length']].copy()
        df_recursive_plot['method'] = 'Recursive'
        
        df_semantic_plot = df_semantic[['chunk_length']].copy()
        df_semantic_plot['method'] = 'Semantic'
        
        df_combined = pd.concat([df_recursive_plot, df_semantic_plot])
        
        # Plot histograms
        plt.subplot(2, 1, 1)
        sns.histplot(data=df_combined, x='chunk_length', hue='method', 
                     kde=True, element='step', bins=30, alpha=0.6)
        plt.title(f'Chunk Length Distribution: {folder_name}')
        plt.xlabel('Chunk Length (characters)')
        plt.ylabel('Count')
        plt.legend(title='Method')
        
        # Plot box plots
        plt.subplot(2, 1, 2)
        sns.boxplot(data=df_combined, x='method', y='chunk_length')
        plt.title(f'Chunk Length Comparison: {folder_name}')
        plt.xlabel('Chunking Method')
        plt.ylabel('Chunk Length (characters)')
        
        # Save the plot
        plot_path = os.path.join(PLOTS_DIR, f"chunk_comparison_{folder_name}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Saved comparison plot to {plot_path}")
        
    except Exception as e:
        logger.error(f"Error creating plots: {str(e)}")

def process_folder(folder_name, folder_path):
    """Process a single folder with both chunking methods and compare results."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing folder: {folder_name}")
    logger.info(f"{'='*60}")
    
    # Load documents
    documents = load_documents_from_directory(folder_path)
    
    if not documents:
        logger.warning(f"No documents found in {folder_path}. Skipping.")
        return None
    
    logger.info(f"Loaded {len(documents)} document pages from {folder_path}")
    
    # Apply recursive chunking
    recursive_chunks = recursive_chunking(
        documents,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP
    )
    
    # Apply semantic chunking
    semantic_chunks = semantic_chunking(
        documents,
        embedding_model='all-MiniLM-L6-v2',
        buffer_size=2,
        breakpoint_type='percentile',
        breakpoint_amount=90
    )
    
    # Check if both methods produced chunks
    if not recursive_chunks or not semantic_chunks:
        logger.warning(f"Chunking failed for {folder_name}. Skipping comparison.")
        return None
    
    # Create DataFrames
    df_recursive = create_chunks_dataframe(recursive_chunks, folder_name, "recursive")
    df_semantic = create_chunks_dataframe(semantic_chunks, folder_name, "semantic")
    
    # Save DataFrames
    save_chunks(df_recursive, folder_name, "recursive")
    save_chunks(df_semantic, folder_name, "semantic")
    
    # Analyze and compare
    comparison = analyze_chunk_quality(df_recursive, df_semantic, folder_name)
    
    return comparison

def generate_summary_report(all_comparisons):
    """Generate a summary report of all folder comparisons."""
    if not all_comparisons:
        logger.warning("No comparisons to summarize.")
        return
    
    # Create summary DataFrame
    summary_data = []
    
    for folder, comparison in all_comparisons.items():
        if comparison:
            # Extract metrics for this folder
            rec = comparison["recursive"]
            sem = comparison["semantic"]
            
            # Add to summary data
            summary_data.append({
                "folder": folder,
                "recursive_chunks": rec["count"],
                "semantic_chunks": sem["count"],
                "diff_count": sem["count"] - rec["count"],
                "recursive_mean_length": rec["mean_length"],
                "semantic_mean_length": sem["mean_length"],
                "recursive_std_length": rec["std_length"],
                "semantic_std_length": sem["std_length"],
                "recursive_short_chunks": rec["short_chunks"],
                "semantic_short_chunks": sem["short_chunks"],
                "recursive_long_chunks": rec["long_chunks"],
                "semantic_long_chunks": sem["long_chunks"]
            })
    
    # Create DataFrame and save to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(OUTPUT_DIR, "chunking_comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    logger.info(f"\n{'='*60}")
    logger.info("Chunking Comparison Summary")
    logger.info(f"{'='*60}")
    logger.info(f"\n{summary_df.to_string()}")
    logger.info(f"\nSummary saved to {summary_path}")
    
    # Create summary plots
    try:
        # Chunk count comparison
        plt.figure(figsize=(12, 6))
        summary_df_plot = summary_df.melt(
            id_vars=['folder'], 
            value_vars=['recursive_chunks', 'semantic_chunks'],
            var_name='method', value_name='chunks'
        )
        sns.barplot(data=summary_df_plot, x='folder', y='chunks', hue='method')
        plt.title('Number of Chunks by Method and Folder')
        plt.xlabel('Folder')
        plt.ylabel('Number of Chunks')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "chunk_count_comparison.png"))
        plt.close()
        
        # Mean length comparison
        plt.figure(figsize=(12, 6))
        summary_df_plot = summary_df.melt(
            id_vars=['folder'], 
            value_vars=['recursive_mean_length', 'semantic_mean_length'],
            var_name='method', value_name='mean_length'
        )
        sns.barplot(data=summary_df_plot, x='folder', y='mean_length', hue='method')
        plt.title('Mean Chunk Length by Method and Folder')
        plt.xlabel('Folder')
        plt.ylabel('Mean Chunk Length (characters)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "mean_length_comparison.png"))
        plt.close()
        
        logger.info(f"Summary plots saved to {PLOTS_DIR}")
        
    except Exception as e:
        logger.error(f"Error creating summary plots: {str(e)}")

def main():
    """Main function to run the chunking comparison."""
    logger.info("Starting chunking comparison...")
    
    # List all subdirectories in the PDFs directory
    try:
        folders = [d for d in os.listdir(LOCAL_PDF_DIR) 
                  if os.path.isdir(os.path.join(LOCAL_PDF_DIR, d))]
    except FileNotFoundError:
        logger.error(f"PDF directory not found: {LOCAL_PDF_DIR}")
        return
    
    if not folders:
        logger.warning(f"No folders found in {LOCAL_PDF_DIR}")
        return
    
    logger.info(f"Found {len(folders)} folders to process: {folders}")
    
    # Process each folder and compare chunking methods
    all_comparisons = {}
    
    for folder in folders:
        folder_path = os.path.join(LOCAL_PDF_DIR, folder)
        comparison = process_folder(folder, folder_path)
        all_comparisons[folder] = comparison
    
    # Generate summary report
    generate_summary_report(all_comparisons)
    
    logger.info("Chunking comparison completed!")

if __name__ == "__main__":
    main()