#!/usr/bin/env python3
"""
Main script for processing datasets using the RAG pipeline
Supports flexible configuration for different datasets like OTT-QA
"""

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from loguru import logger
from src.core.models import ProcessingConfig, DatasetConfig, ChunkType
from src.pipeline.rag_pipeline import RAGPipeline
from src.visualization.dash_app import create_optimized_dashboard


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Process datasets using RAG pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset configuration
    parser.add_argument(
        '--dataset-name', 
        type=str, 
        default='ott-qa',
        help='Name of the dataset'
    )
    
    parser.add_argument(
        '--dataset-path', 
        type=str, 
        default='datasets',
        help='Path to the dataset directory'
    )
    
    parser.add_argument(
        '--metadata-file', 
        type=str, 
        default='datasets/ott-qa.json',
        help='Path to the metadata JSON file'
    )
    
    # Processing range configuration
    parser.add_argument(
        '--start-index', 
        type=int, 
        default=0,
        help='Starting index for processing'
    )
    
    parser.add_argument(
        '--end-index', 
        type=int, 
        default=None,
        help='Ending index for processing (None means all)'
    )
    
    parser.add_argument(
        '--chunk-size', 
        type=int, 
        default=None,
        help='Maximum number of chunks to process'
    )
    
    parser.add_argument(
        '--max-samples', 
        type=int, 
        default=None,
        help='Maximum number of samples/files to process (alias for chunk-size)'
    )
    
    parser.add_argument(
        '--filter-type', 
        type=str, 
        choices=['document', 'table'],
        default=None,
        help='Filter by source type (document/table)'
    )
    
    # Processing configuration
    parser.add_argument(
        '--similarity-threshold', 
        type=float, 
        default=0.8,
        help='Similarity threshold for merging sentences/rows'
    )
    
    parser.add_argument(
        '--table-similarity-threshold', 
        type=float, 
        default=0.7,
        help='Similarity threshold for merging table rows'
    )
    
    parser.add_argument(
        '--remove-stopwords', 
        action='store_true',
        help='Remove stopwords from text'
    )
    
    # Output configuration
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='output',
        help='Directory to save outputs'
    )
    
    parser.add_argument(
        '--save-graph', 
        action='store_true',
        help='Save the knowledge graph to JSON'
    )
    
    # Analysis configuration
    parser.add_argument(
        '--analyze-similarities', 
        action='store_true',
        help='Run similarity analysis after processing'
    )
    
    parser.add_argument(
        '--create-similarity-plots', 
        action='store_true',
        help='Create similarity visualization plots'
    )
    
    # Visualization configuration
    parser.add_argument(
        '--launch-dashboard', 
        action='store_true',
        help='Launch interactive dashboard'
    )
    
    parser.add_argument(
        '--dashboard-port', 
        type=int, 
        default=8050,
        help='Port for dashboard'
    )
    
    parser.add_argument(
        '--dashboard-host', 
        type=str, 
        default='localhost',
        help='Host for dashboard'
    )
    
    # Future LLM configuration (placeholder)
    parser.add_argument(
        '--llm-model', 
        type=str, 
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='LLM model to use for embeddings (future feature)'
    )
    
    parser.add_argument(
        '--embedding-model', 
        type=str, 
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Embedding model to use'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=32,
        help='Batch size for embedding generation (GPU optimization)'
    )
    
    parser.add_argument(
        '--use-faiss', 
        action='store_true',
        default=True,
        help='Use FAISS-accelerated graph building (default: True)'
    )
    
    parser.add_argument(
        '--no-faiss', 
        action='store_true',
        help='Disable FAISS acceleration and use standard O(N²) graph building'
    )
    
    parser.add_argument(
        '--max-neighbors', 
        type=int, 
        default=150,
        help='Maximum neighbors to consider per node in FAISS search (default: 150)'
    )
    
    parser.add_argument(
        '--faiss-gpu', 
        action='store_true',
        help='Use GPU for FAISS index (may have compatibility issues, CPU is default)'
    )
    
    # Logging
    parser.add_argument(
        '--log-level', 
        type=str, 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    return parser


def validate_arguments(args) -> bool:
    """Validate command line arguments"""
    # Check if metadata file exists
    if not os.path.exists(args.metadata_file):
        logger.error(f"Metadata file not found: {args.metadata_file}")
        return False
    
    # Check if dataset path exists
    if not os.path.exists(args.dataset_path):
        logger.error(f"Dataset path not found: {args.dataset_path}")
        return False
    
    # Validate index ranges
    if args.end_index is not None and args.end_index <= args.start_index:
        logger.error("End index must be greater than start index")
        return False
    
    # Validate thresholds
    if not (0.0 <= args.similarity_threshold <= 1.0):
        logger.error("Similarity threshold must be between 0.0 and 1.0")
        return False
    
    if not (0.0 <= args.table_similarity_threshold <= 1.0):
        logger.error("Table similarity threshold must be between 0.0 and 1.0")
        return False
    
    return True


def create_configs(args) -> tuple:
    """Create configuration objects from arguments"""
    
    # Create processing config
    processing_config = ProcessingConfig(
        sentence_similarity_threshold=args.similarity_threshold,
        table_similarity_threshold=args.table_similarity_threshold,
        remove_stopwords=args.remove_stopwords,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        use_faiss=not args.no_faiss if hasattr(args, 'no_faiss') else True,
        faiss_use_gpu=args.faiss_gpu if hasattr(args, 'faiss_gpu') else False,
        max_neighbors=args.max_neighbors
    )
    
    # Handle filter type
    filter_source_type = None
    if args.filter_type:
        if args.filter_type == 'document':
            filter_source_type = ChunkType.DOCUMENT
        elif args.filter_type == 'table':
            filter_source_type = ChunkType.TABLE
    
    # Use max_samples if provided, otherwise use chunk_size
    effective_chunk_size = args.max_samples if args.max_samples is not None else args.chunk_size
    
    # Create dataset config
    dataset_config = DatasetConfig(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        metadata_file=args.metadata_file,
        start_index=args.start_index,
        end_index=args.end_index,
        chunk_size=effective_chunk_size,
        filter_source_type=filter_source_type
    )
    
    return processing_config, dataset_config


def main():
    """Main function"""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level=args.log_level)
    
    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)
    
    # Create configurations
    processing_config, dataset_config = create_configs(args)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=== RAG Pipeline Dataset Processing ===")
    logger.info(f"Dataset: {dataset_config.dataset_name}")
    logger.info(f"Processing range: {dataset_config.start_index} to {dataset_config.end_index or 'end'}")
    if dataset_config.filter_source_type:
        logger.info(f"Filter: {dataset_config.filter_source_type.value}")
    if dataset_config.chunk_size:
        logger.info(f"Chunk limit: {dataset_config.chunk_size}")
    
    try:
        # Initialize pipeline
        pipeline = RAGPipeline(processing_config)
        
        # Process dataset
        logger.info("Starting dataset processing...")
        knowledge_graph = pipeline.process_from_json_metadata(dataset_config)
        
        if len(knowledge_graph.nodes) == 0:
            logger.warning("No nodes were created in the knowledge graph")
            return
        
        # Print statistics
        stats = knowledge_graph.get_graph_statistics()
        logger.info("=== Knowledge Graph Statistics ===")
        for key, value in stats.items():
            if isinstance(value, dict):
                logger.info(f"{key}:")
                for subkey, subvalue in value.items():
                    logger.info(f" {subkey}: {subvalue}")
            else:
                logger.info(f"{key}: {value}")
        
        # Save graph if requested
        if args.save_graph:
            output_file = os.path.join(args.output_dir, f"{args.dataset_name}_graph.json")
            knowledge_graph.export_to_json(output_file)
            logger.info(f"Graph saved to: {output_file}")
        
        # Run similarity analysis if requested
        if args.analyze_similarities:
            logger.info("Running similarity analysis...")
            
            cmd = [sys.executable, "similarity_analysis.py", "--output-dir", args.output_dir]
            if args.create_similarity_plots:
                cmd.append("--create-plots")
            
            try:
                subprocess.run(cmd, check=True)
                logger.info("Similarity analysis completed")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error running similarity analysis: {e}")
            except FileNotFoundError:
                logger.warning("similarity_analysis.py not found. Run it manually for detailed analysis.")
        
        # Launch dashboard if requested
        if args.launch_dashboard:
            logger.info(f"Launching dashboard at http://{args.dashboard_host}:{args.dashboard_port}")
            app = create_optimized_dashboard(knowledge_graph)
            app.run(host=args.dashboard_host, port=args.dashboard_port, debug=False)
        
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()