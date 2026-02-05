#!/usr/bin/env python3
"""
Run MAG Pipeline Final on FULL DATASET
After successful small-scale test, now process the complete MAG dataset
"""

import os
import sys
import time
from pathlib import Path
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import our pipeline
from mag_pipeline_final import MAGPipelineFinal

def run_full_mag_pipeline():
    """Run MAG pipeline on the complete dataset"""
    logger.info("🚀 MAG PIPELINE FINAL - FULL DATASET PROCESSING")
    logger.info("=" * 60)
    logger.info("📊 Target: ALL papers (~700K) + ALL authors (~1.1M)")
    logger.info("📦 Expected: ~20 chunks of 100K nodes each")
    logger.info("⏱️  Estimated time: 8-12 hours")
    logger.info("")
    
    # Create pipeline for full dataset
    full_cache_dir = "/shared/khoja/CogComp/output/mag_final_cache"
    
    # Clean any existing cache (fresh start)
    import shutil
    if Path(full_cache_dir).exists():
        logger.info("🧹 Cleaning existing cache for fresh start...")
        shutil.rmtree(full_cache_dir)
    
    # Create pipeline with optimal settings for full dataset
    pipeline = MAGPipelineFinal(
        cache_dir=full_cache_dir,
        target_chunks=20,  # 20 chunks of ~100K nodes each
        use_gpu=True,
        num_threads=32  # Full thread utilization
    )
    
    start_time = time.time()
    
    try:
        logger.info("🚀 Starting FULL DATASET processing...")
        logger.info("💡 Processing ALL papers + ALL authors (no limits)")
        
        # Run pipeline without any limits - process everything
        analysis_file = pipeline.run_pipeline()  # max_papers=None by default
        
        total_time = time.time() - start_time
        hours = total_time / 3600
        
        logger.info("")
        logger.info("🎉 FULL DATASET PROCESSING COMPLETED!")
        logger.info("=" * 50)
        logger.info(f"⏱️  Total time: {total_time:.2f}s ({hours:.2f} hours)")
        logger.info(f"📁 Analysis file: {analysis_file}")
        logger.info("")
        logger.info("🔍 Next steps:")
        logger.info("   1. Verify embedding completeness")
        logger.info("   2. Build HNSW index for retrieval")
        logger.info("   3. Test retrieval performance")
        
        return True
        
    except Exception as e:
        total_time = time.time() - start_time
        hours = total_time / 3600
        logger.error(f"❌ FULL DATASET PROCESSING FAILED: {e}")
        logger.error(f"⏱️  Failed after: {total_time:.2f}s ({hours:.2f} hours)")
        return False

def main():
    """Main function"""
    logger.info("🧬 MAG PIPELINE FINAL - COMPLETE DATASET")
    logger.info("💡 Processing entire MAG dataset for embeddings")
    logger.info("")
    
    # Set environment variables for optimal performance
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    
    success = run_full_mag_pipeline()
    
    logger.info("")
    logger.info("=" * 60)
    if success:
        logger.info("✅ FULL DATASET PROCESSING SUCCESSFUL!")
        logger.info("🎯 Ready for HNSW index building and retrieval testing")
    else:
        logger.error("❌ FULL DATASET PROCESSING FAILED!")
        logger.error("🔧 Check logs and fix issues before retrying")
    
    return success

if __name__ == "__main__":
    main()
