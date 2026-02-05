#!/usr/bin/env python3
"""
Runner for Optimized STARK Similarity Calculation
Memory-efficient, GPU-accelerated similarity calculation with deduplication
"""

import os
import sys
import time
import torch
from pathlib import Path
from loguru import logger
import psutil
import subprocess

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from optimized_similarity_calculator import OptimizedSimilarityCalculator

def setup_cuda_environment():
    """Setup CUDA environment for optimal GPU usage"""
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,roundup_power2_divisions:16'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '0'
    os.environ['OMP_NUM_THREADS'] = '32'
    os.environ['MKL_NUM_THREADS'] = '32'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    logger.info("🔧 Set CUDA environment for optimized similarity calculation:")
    for key in ['PYTORCH_CUDA_ALLOC_CONF', 'CUDA_LAUNCH_BLOCKING', 'OMP_NUM_THREADS']:
        logger.info(f"    {key}={os.environ.get(key)}")

def check_system_resources():
    """Check and display system resources"""
    logger.info("🔍 System Resource Check:")
    
    # GPU info
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"    GPU {i}: {gpu_memory:.1f} GB total ({gpu_name})")
    
    # RAM info
    ram = psutil.virtual_memory()
    logger.info(f"    RAM: {ram.available/1024**3:.1f} GB available / {ram.total/1024**3:.1f} GB total")
    
    # CPU info
    logger.info(f"    CPU: {psutil.cpu_count()} cores")

def check_pipeline_status():
    """Check if the chunked pipeline has completed successfully"""
    cache_dir = Path("/shared/khoja/CogComp/output/stark_chunked_cache")
    
    # Check for required files
    hnsw_index = cache_dir / "final" / "hnsw_index.faiss"
    hnsw_mapping = cache_dir / "final" / "hnsw_mapping.pkl"
    embeddings_dir = cache_dir / "embeddings"
    
    if not hnsw_index.exists():
        logger.error("❌ HNSW index not found! Run the chunked pipeline first.")
        return False
    
    if not hnsw_mapping.exists():
        logger.error("❌ HNSW mapping not found! Run the chunked pipeline first.")
        return False
    
    if not embeddings_dir.exists() or not list(embeddings_dir.glob("*.json")):
        logger.error("❌ Embedding files not found! Run the chunked pipeline first.")
        return False
    
    # Check cache status
    embedding_files = list(embeddings_dir.glob("*.json"))
    total_size = sum(f.stat().st_size for f in embedding_files) / (1024**3)
    
    logger.info("✅ Chunked pipeline prerequisites found:")
    logger.info(f"    📁 HNSW index: {hnsw_index.stat().st_size / (1024**2):.1f} MB")
    logger.info(f"    📁 HNSW mapping: {hnsw_mapping.stat().st_size / (1024**2):.1f} MB")
    logger.info(f"    📁 Embedding files: {len(embedding_files)} files ({total_size:.1f} GB)")
    
    return True

def run_optimized_similarity_calculation(batch_size: int = 1000, k_neighbors: int = 200):
    """
    Run the optimized similarity calculation
    
    Args:
        batch_size: Number of nodes to process in one batch
        k_neighbors: Number of neighbors to find for each node
    """
    
    logger.info("🚀 Starting Optimized STARK Similarity Calculation")
    logger.info(f"📦 Batch size: {batch_size:,} nodes")
    logger.info(f"🔍 K-neighbors: {k_neighbors}")
    logger.info("💡 GPU-accelerated with intelligent deduplication")
    logger.info("=" * 70)
    
    # Setup environment
    setup_cuda_environment()
    check_system_resources()
    
    # Check prerequisites
    if not check_pipeline_status():
        logger.error("❌ Prerequisites not met. Please run the chunked pipeline first.")
        return None
    
    # Initialize calculator
    logger.info("\n🔧 Initializing Optimized Similarity Calculator...")
    start_time = time.time()
    
    calculator = OptimizedSimilarityCalculator(
        cache_dir="/shared/khoja/CogComp/output/stark_chunked_cache",
        batch_size=batch_size,
        similarity_cache_size=1000000,  # Cache for 1M pairs
        use_gpu=True
    )
    
    init_time = time.time() - start_time
    logger.info(f"✅ Calculator initialized in {init_time:.2f} seconds")
    
    # Run similarity calculation
    logger.info("\n🎯 Starting Optimized Similarity Calculation...")
    logger.info("💫 This will process ~2M nodes with intelligent batching!")
    
    try:
        output_file = calculator.calculate_all_similarities(k_neighbors=k_neighbors)
        
        total_time = time.time() - start_time
        
        logger.info("\n🎉 Optimized Similarity Calculation completed successfully!")
        logger.info(f"⏱️  Total execution time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        logger.info(f"📁 Final edges: {output_file}")
        
        # Show final statistics
        stats_file = output_file.parent / f"stark_edge_statistics_{output_file.stem.split('_')[-1]}.json"
        if stats_file.exists():
            logger.info(f"📊 Edge statistics: {stats_file}")
        
        return output_file
        
    except Exception as e:
        logger.error(f"❌ Similarity calculation failed: {e}")
        raise

def main():
    """Main function with argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized STARK Similarity Calculator")
    parser.add_argument("--batch-size", type=int, default=1000,
                       help="Number of nodes per batch (default: 1000)")
    parser.add_argument("--k-neighbors", type=int, default=200,
                       help="Number of neighbors per node (default: 200)")
    parser.add_argument("--mode", choices=["test", "full"], default="full",
                       help="Run mode: test (smaller batches) or full (optimized)")
    
    args = parser.parse_args()
    
    # Adjust parameters based on mode
    if args.mode == "test":
        batch_size = min(args.batch_size, 100)  # Smaller batches for testing
        k_neighbors = min(args.k_neighbors, 50)  # Fewer neighbors for testing
        logger.info("🧪 Running in TEST mode with reduced parameters")
    else:
        batch_size = args.batch_size
        k_neighbors = args.k_neighbors
        logger.info("🚀 Running in FULL mode with optimized parameters")
    
    try:
        result = run_optimized_similarity_calculation(
            batch_size=batch_size,
            k_neighbors=k_neighbors
        )
        
        if result:
            logger.info("✅ Similarity calculation completed successfully!")
            logger.info(f"📊 Results saved to: {result}")
        
    except KeyboardInterrupt:
        logger.warning("⚠️  Calculation interrupted by user")
        logger.info("💡 Progress is cached - you can resume by running again")
        
    except Exception as e:
        logger.error(f"❌ Calculation failed with error: {e}")
        logger.info("💡 Check the logs above for details. Some progress may be cached.")
        sys.exit(1)

if __name__ == "__main__":
    main()

