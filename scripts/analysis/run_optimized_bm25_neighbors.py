#!/usr/bin/env python3
"""
Runner script for OPTIMIZED BM25 neighbors pre-computation
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from precompute_bm25_neighbors_optimized import OptimizedBM25NeighborsPrecomputer
from loguru import logger

def main():
    """Run the OPTIMIZED BM25 neighbors pre-computation"""
    
    logger.info(" Starting OPTIMIZED BM25 Neighbors Pre-computation")
    logger.info(" Major optimizations: pre-computed neighbors + fast mappings")
    logger.info("=" * 60)
    
    # Initialize optimized pre-computer
    precomputer = OptimizedBM25NeighborsPrecomputer(
        precomputed_neighbors_dir="/shared/khoja/CogComp/output/precomputed_neighbors",
        content_mapping_file="/shared/khoja/CogComp/output/asin_to_content_mapping.json",
        output_dir="/shared/khoja/CogComp/output/precomputed_bm25_neighbors",
        batch_size=10000, # Larger batches for efficiency
        k_bm25_neighbors=100, # Keep top 100 BM25-similar neighbors
        num_threads=16 # Fewer threads, larger batches
    )
    
    try:
        # Run optimized pre-computation
        precomputer.run_precomputation()
        
        logger.info(" OPTIMIZED BM25 neighbors pre-computation completed!")
        
        # Show file sizes
        output_dir = Path("/shared/khoja/CogComp/output/precomputed_bm25_neighbors")
        logger.info(" Generated files:")
        for file in sorted(output_dir.glob("*.pkl")):
            size_mb = file.stat().st_size / (1024 * 1024)
            logger.info(f" {file.name}: {size_mb:.1f} MB")
        
    except Exception as e:
        logger.error(f" OPTIMIZED BM25 pre-computation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

