#!/usr/bin/env python3
"""
STARK Dataset Demo Script
Demonstrates how to use the STARK Graph Analysis Pipeline
Supports both limited demo mode and full dataset processing
"""

import sys
import json
import argparse
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stark_graph_analysis_pipeline import STARKGraphAnalysisPipeline

def run_stark_pipeline(max_products=None, k_neighbors=200, use_gpu=True, num_threads=64, cache_suffix=""):
    """
    Run the STARK pipeline with configurable parameters
    
    Args:
        max_products: Maximum number of products to process (None = all)
        k_neighbors: Number of neighbors for similarity analysis
        use_gpu: Whether to use GPU acceleration
        num_threads: Number of threads for parallel processing
        cache_suffix: Suffix to add to cache directory name
    """
    
    mode = "FULL DATASET" if max_products is None else f"LIMITED ({max_products} products)"
    logger.info(f"Starting STARK Dataset Processing - {mode}")
    logger.info("=" * 60)
    
    # Set cache directory based on mode
    if max_products is None:
        cache_dir = f"/shared/khoja/CogComp/output/stark_full_cache{cache_suffix}"
    else:
        cache_dir = f"/shared/khoja/CogComp/output/stark_demo_cache{cache_suffix}"
    
    # Initialize pipeline
    logger.info("Initializing STARK pipeline...")
    pipeline = STARKGraphAnalysisPipeline(
        stark_dataset_file="/shared/khoja/CogComp/datasets/STARK/node_info.json",
        cache_dir=cache_dir,
        use_gpu=use_gpu,
        num_threads=num_threads
    )
    
    try:
        # Step 1: Load STARK chunks
        if max_products is None:
            logger.info("Step 1: Loading ALL STARK chunks (full dataset)...")
            pipeline.load_stark_chunks() # No limit = full dataset
        else:
            logger.info(f"Step 1: Loading STARK chunks (limited to {max_products} products)...")
            pipeline.load_stark_chunks(max_products=max_products)
        
        # Show statistics
        product_chunks = [c for c in pipeline.chunks if c.chunk_type == "table"]
        review_chunks = [c for c in pipeline.chunks if c.chunk_type == "document"]
        
        logger.info(f"Loaded {len(product_chunks)} product chunks (tables)")
        logger.info(f"Loaded {len(review_chunks)} review chunks (documents)")
        logger.info(f"Total chunks: {len(pipeline.chunks)}")
        
        # Step 2: Build HNSW index
        logger.info("Step 2: Building HNSW index...")
        pipeline.build_hnsw_index()
        logger.info(f"HNSW index built with {pipeline.faiss_index.ntotal} embeddings")
        
        # Step 3: Analyze similarities
        logger.info(f"Step 3: Analyzing similarities (k_neighbors={k_neighbors})...")
        pipeline.analyze_stark_similarities(k_neighbors=k_neighbors)
        logger.info(f"Calculated {len(pipeline.similarity_data)} similarity pairs")
        
        # Show edge type distribution
        edge_types = {}
        for sim in pipeline.similarity_data:
            edge_types[sim.edge_type] = edge_types.get(sim.edge_type, 0) + 1
        
        logger.info("Edge type distribution:")
        for edge_type, count in edge_types.items():
            logger.info(f" {edge_type}: {count}")
        
        # Step 4: Generate analysis reports
        logger.info("Step 4: Generating analysis reports...")
        pipeline.generate_stark_analysis_reports()
        logger.info(f"Analysis reports saved to: {pipeline.cache_dir}/stark_analysis_reports")
        
        # Step 5: Extract and save unique edges (FINAL OUTPUT)
        logger.info("Step 5: Extracting and saving unique edges...")
        edges_file = pipeline.extract_and_save_unique_edges()
        
        if edges_file is None:
            logger.warning("No edges were generated. This can happen with very small datasets.")
            logger.warning("Try using more products or checking if similarities are being calculated.")
            logger.info("=" * 60)
            logger.info(f"STARK {mode} processing completed with no edges generated.")
            return None
        
        # Load and show final edge statistics
        with open(edges_file, 'r') as f:
            final_edges = json.load(f)
        
        logger.info(f"Final output saved to: {edges_file}")
        logger.info(f"Total unique edges: {len(final_edges)}")
        
        # Show edge breakdown by type and reason
        edge_breakdown = {}
        for edge in final_edges:
            key = f"{edge['edge_type']}_{edge['reason']}"
            edge_breakdown[key] = edge_breakdown.get(key, 0) + 1
        
        logger.info("Final edge breakdown:")
        for key, count in edge_breakdown.items():
            logger.info(f" {key}: {count}")
        
        logger.info("=" * 60)
        logger.info(f"STARK {mode} processing completed successfully!")
        logger.info(f"Results available at: {edges_file}")
        
        return edges_file
        
    except Exception as e:
        logger.error(f"STARK pipeline failed: {e}")
        raise

def show_sample_edges(edges_file: Path, num_samples: int = 5):
    """Show sample edges from the final output"""
    
    logger.info(f"Showing {num_samples} sample edges from results...")
    
    with open(edges_file, 'r') as f:
        edges = json.load(f)
    
    for i, edge in enumerate(edges[:num_samples]):
        logger.info(f"\nSample Edge {i+1}:")
        logger.info(f" Type: {edge['edge_type']}")
        logger.info(f" Reason: {edge['reason']}")
        logger.info(f" Source: {edge['source_chunk_id']}")
        logger.info(f" Target: {edge['target_chunk_id']}")
        logger.info(f" Similarity: {edge['semantic_similarity']:.3f}")
        logger.info(f" Entity Count: {edge['entity_count']}")

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="STARK Dataset Graph Analysis Pipeline")
    
    # Dataset processing options
    parser.add_argument("--mode", choices=["demo", "full"], default="demo",
                       help="Processing mode: 'demo' for limited dataset, 'full' for complete dataset")
    parser.add_argument("--max-products", type=int, default=100,
                       help="Maximum number of products to process in demo mode (default: 100)")
    
    # Performance options
    parser.add_argument("--k-neighbors", type=int, default=200,
                       help="Number of neighbors for similarity analysis (default: 200)")
    parser.add_argument("--num-threads", type=int, default=64,
                       help="Number of threads for parallel processing (default: 64)")
    parser.add_argument("--no-gpu", action="store_true",
                       help="Disable GPU acceleration (use CPU only)")
    
    # Output options
    parser.add_argument("--cache-suffix", type=str, default="",
                       help="Suffix to add to cache directory name")
    parser.add_argument("--show-samples", type=int, default=3,
                       help="Number of sample edges to display (default: 3)")
    
    args = parser.parse_args()
    
    # Determine processing parameters
    if args.mode == "full":
        max_products = None
        logger.info(" Running STARK pipeline on FULL dataset (957K+ products)")
        logger.warning(" This will take several hours and requires significant GPU memory!")
    else:
        max_products = args.max_products
        logger.info(f" Running STARK pipeline in DEMO mode ({max_products} products)")
    
    use_gpu = not args.no_gpu
    
    # Display configuration
    logger.info("Configuration:")
    logger.info(f" Mode: {args.mode}")
    logger.info(f" Max products: {max_products or 'ALL'}")
    logger.info(f" K-neighbors: {args.k_neighbors}")
    logger.info(f" Threads: {args.num_threads}")
    logger.info(f" GPU: {'Enabled' if use_gpu else 'Disabled'}")
    logger.info(f" Cache suffix: '{args.cache_suffix}'")
    
    try:
        # Run the pipeline
        edges_file = run_stark_pipeline(
            max_products=max_products,
            k_neighbors=args.k_neighbors,
            use_gpu=use_gpu,
            num_threads=args.num_threads,
            cache_suffix=args.cache_suffix
        )
        
        # Show sample edges if any were generated
        if edges_file is not None:
            show_sample_edges(edges_file, num_samples=args.show_samples)
        
        logger.info(" Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f" Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
