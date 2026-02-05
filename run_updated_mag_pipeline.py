#!/usr/bin/env python3
"""
Run Updated MAG Pipeline - Complete Dataset Processing
Processes ALL authors and papers with doubled chunk size
"""

import os
import sys
import time
from pathlib import Path
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from mag_chunked_pipeline import MAGChunkedPipeline

def main():
    """Run the updated MAG pipeline for complete dataset"""
    # Set PyTorch memory configuration
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    logger.info("🚀 UPDATED MAG PIPELINE - COMPLETE DATASET PROCESSING")
    logger.info("=" * 70)
    logger.info("📊 Expected Results:")
    logger.info("   • Authors: ~1,172,724 nodes (indices 0-1,172,723)")
    logger.info("   • Papers:  ~700,244 nodes (indices 1,172,724-1,872,967)")
    logger.info("   • Total:   ~1,872,968 nodes")
    logger.info("   • Chunks:  ~19 chunks (100K nodes each)")
    logger.info("   • Memory:  4GB per chunk (doubled from 2GB)")
    logger.info("=" * 70)
    
    start_time = time.time()
    
    try:
        # Initialize pipeline with updated settings
        pipeline = MAGChunkedPipeline(
            chunk_size=100000,  # 100K nodes per chunk (doubled)
            use_gpu=True,
            num_threads=32
        )
        
        # Run complete pipeline
        logger.info("🔥 Starting complete MAG dataset processing...")
        analysis_file = pipeline.run_chunked_pipeline()
        
        total_time = time.time() - start_time
        
        logger.info("🎉 UPDATED MAG PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"⏱️  Total processing time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        logger.info(f"📁 Analysis file: {analysis_file}")
        logger.info("✅ All authors and papers processed with embeddings!")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()

