#!/usr/bin/env python3
"""
Enhanced Runner for 2-Stage STARK Similarity Calculation
Optimized for 8x NVIDIA TITAN RTX GPUs (192GB total memory)
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

from optimized_similarity_calculator_v2 import EnhancedSimilarityCalculator

def setup_titan_rtx_environment():
    """Setup optimal environment for 8x TITAN RTX GPUs"""
    # TITAN RTX optimized settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,roundup_power2_divisions:16,max_split_size_mb:2048'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async for better performance
    os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '0'
    os.environ['OMP_NUM_THREADS'] = '64'  # More threads for your system
    os.environ['MKL_NUM_THREADS'] = '64'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'  # All 8 GPUs
    
    # TITAN RTX memory optimization
    os.environ['PYTORCH_CUDA_MEMORY_STRATEGY'] = 'native'
    
    logger.info("🔧 TITAN RTX Environment Configuration:")
    for key in ['PYTORCH_CUDA_ALLOC_CONF', 'OMP_NUM_THREADS', 'CUDA_VISIBLE_DEVICES']:
        logger.info(f"    {key}={os.environ.get(key)}")

def check_titan_rtx_resources():
    """Check and display TITAN RTX system resources"""
    logger.info("🔍 TITAN RTX System Resource Check:")
    
    # GPU info
    if torch.cuda.is_available():
        total_gpu_memory = 0
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            total_gpu_memory += gpu_memory
            logger.info(f"    GPU {i}: {gpu_memory:.1f} GB ({gpu_name})")
        
        logger.info(f"    🚀 Total GPU Memory: {total_gpu_memory:.1f} GB")
        
        # Check GPU utilization
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                utilizations = [int(x.strip()) for x in result.stdout.strip().split('\n')]
                avg_util = sum(utilizations) / len(utilizations)
                logger.info(f"    📊 Average GPU Utilization: {avg_util:.1f}%")
        except:
            pass
    
    # RAM info
    ram = psutil.virtual_memory()
    logger.info(f"    RAM: {ram.available/1024**3:.1f} GB available / {ram.total/1024**3:.1f} GB total")
    
    # CPU info
    logger.info(f"    CPU: {psutil.cpu_count()} cores ({psutil.cpu_count(logical=False)} physical)")

def check_data_prerequisites():
    """Check if required data files exist"""
    cache_dir = Path("/shared/khoja/CogComp/output/stark_chunked_cache")
    
    # Check for required files
    hnsw_index = cache_dir / "final" / "hnsw_index.faiss"
    hnsw_mapping = cache_dir / "final" / "hnsw_mapping.pkl"
    embeddings_dir = cache_dir / "embeddings"
    
    if not hnsw_index.exists():
        logger.error("❌ HNSW index not found! Run the chunked pipeline first.")
        logger.info("💡 Required file: /shared/khoja/CogComp/output/stark_chunked_cache/final/hnsw_index.faiss")
        return False
    
    if not hnsw_mapping.exists():
        logger.error("❌ HNSW mapping not found! Run the chunked pipeline first.")
        logger.info("💡 Required file: /shared/khoja/CogComp/output/stark_chunked_cache/final/hnsw_mapping.pkl")
        return False
    
    if not embeddings_dir.exists() or not list(embeddings_dir.glob("*.json")):
        logger.error("❌ Embedding files not found! Run the chunked pipeline first.")
        logger.info("💡 Required directory: /shared/khoja/CogComp/output/stark_chunked_cache/embeddings/")
        return False
    
    # Check cache status
    embedding_files = list(embeddings_dir.glob("*.json"))
    total_size = sum(f.stat().st_size for f in embedding_files) / (1024**3)
    
    logger.info("✅ Data prerequisites found:")
    logger.info(f"    📁 HNSW index: {hnsw_index.stat().st_size / (1024**2):.1f} MB")
    logger.info(f"    📁 HNSW mapping: {hnsw_mapping.stat().st_size / (1024**2):.1f} MB")
    logger.info(f"    📁 Embedding files: {len(embedding_files)} files ({total_size:.1f} GB)")
    
    return True

def run_enhanced_similarity_calculation(
    neighbor_batch_size: int = 100000,  # TITAN RTX optimized
    similarity_batch_size: int = 50000,  # TITAN RTX optimized
    k_neighbors: int = 200
):
    """
    Run the enhanced 2-stage similarity calculation
    
    Args:
        neighbor_batch_size: Nodes per batch for neighbor search (Stage 1)
        similarity_batch_size: Pairs per batch for similarity computation (Stage 2)
        k_neighbors: Number of neighbors per node
    """
    
    logger.info("🚀 Starting Enhanced 2-Stage STARK Similarity Calculation")
    logger.info(f"📋 Stage 1 batch size: {neighbor_batch_size:,} nodes")
    logger.info(f"🔥 Stage 2 batch size: {similarity_batch_size:,} pairs")
    logger.info(f"🔍 K-neighbors: {k_neighbors}")
    logger.info("💡 Optimized for 8x TITAN RTX GPUs (192GB total)")
    logger.info("=" * 80)
    
    # Setup environment
    setup_titan_rtx_environment()
    check_titan_rtx_resources()
    
    # Check prerequisites
    if not check_data_prerequisites():
        logger.error("❌ Prerequisites not met. Please run the chunked pipeline first.")
        return None
    
    # Initialize enhanced calculator
    logger.info("\n🔧 Initializing Enhanced 2-Stage Calculator...")
    start_time = time.time()
    
    calculator = EnhancedSimilarityCalculator(
        cache_dir="/shared/khoja/CogComp/output/stark_chunked_cache",
        neighbor_batch_size=neighbor_batch_size,
        similarity_batch_size=similarity_batch_size,
        use_gpu=True
    )
    
    init_time = time.time() - start_time
    logger.info(f"✅ Calculator initialized in {init_time:.2f} seconds")
    
    # Run 2-stage similarity calculation
    logger.info("\n🎯 Starting Enhanced 2-Stage Similarity Calculation...")
    logger.info("💫 Processing ~2M nodes with TITAN RTX optimization!")
    
    try:
        output_file = calculator.calculate_all_similarities_2stage(k_neighbors=k_neighbors)
        
        total_time = time.time() - start_time
        
        logger.info("\n🎉 Enhanced 2-Stage Calculation completed successfully!")
        logger.info(f"⏱️  Total execution time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        logger.info(f"📁 Final edges: {output_file}")
        
        return output_file
        
    except Exception as e:
        logger.error(f"❌ Enhanced calculation failed: {e}")
        raise

def estimate_computation_time(num_nodes: int = 2000000, k_neighbors: int = 200):
    """
    Estimate computation time for 8x TITAN RTX setup
    
    Based on TITAN RTX performance benchmarks:
    - HNSW search: ~50,000 queries/sec per GPU
    - Similarity computation: ~100,000 pairs/sec total
    """
    
    logger.info("⏱️  Performance Estimation for 8x TITAN RTX:")
    
    # Stage 1: Neighbor search
    total_queries = num_nodes
    queries_per_sec_total = 50000 * 8  # 8 GPUs
    stage1_time = total_queries / queries_per_sec_total
    
    # Stage 2: Similarity computation
    total_pairs = num_nodes * k_neighbors / 2  # Approximate unique pairs
    pairs_per_sec_total = 100000  # Conservative estimate
    stage2_time = total_pairs / pairs_per_sec_total
    
    total_estimated_time = stage1_time + stage2_time
    
    logger.info(f"    📊 Nodes: {num_nodes:,}")
    logger.info(f"    🔍 K-neighbors: {k_neighbors}")
    logger.info(f"    📋 Stage 1 (neighbor search): ~{stage1_time:.0f} seconds ({stage1_time/60:.1f} minutes)")
    logger.info(f"    🔥 Stage 2 (similarities): ~{stage2_time:.0f} seconds ({stage2_time/60:.1f} minutes)")
    logger.info(f"    ⏱️  Total estimated time: ~{total_estimated_time:.0f} seconds ({total_estimated_time/60:.1f} minutes)")
    
    return total_estimated_time

def main():
    """Main function with argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced 2-Stage STARK Similarity Calculator")
    parser.add_argument("--neighbor-batch-size", type=int, default=100000,
                       help="Stage 1: Nodes per batch for neighbor search (default: 100000)")
    parser.add_argument("--similarity-batch-size", type=int, default=50000,
                       help="Stage 2: Pairs per batch for similarity computation (default: 50000)")
    parser.add_argument("--k-neighbors", type=int, default=200,
                       help="Number of neighbors per node (default: 200)")
    parser.add_argument("--mode", choices=["estimate", "test", "full"], default="full",
                       help="Run mode: estimate (time only), test (small batches), full (optimized)")
    
    args = parser.parse_args()
    
    # Adjust parameters based on mode
    if args.mode == "estimate":
        estimate_computation_time(k_neighbors=args.k_neighbors)
        return
    elif args.mode == "test":
        neighbor_batch_size = min(args.neighbor_batch_size, 10000)
        similarity_batch_size = min(args.similarity_batch_size, 5000)
        k_neighbors = min(args.k_neighbors, 50)
        logger.info("🧪 Running in TEST mode with reduced parameters")
    else:
        neighbor_batch_size = args.neighbor_batch_size
        similarity_batch_size = args.similarity_batch_size
        k_neighbors = args.k_neighbors
        logger.info("🚀 Running in FULL mode with TITAN RTX optimization")
    
    try:
        result = run_enhanced_similarity_calculation(
            neighbor_batch_size=neighbor_batch_size,
            similarity_batch_size=similarity_batch_size,
            k_neighbors=k_neighbors
        )
        
        if result:
            logger.info("✅ Enhanced 2-stage calculation completed successfully!")
            logger.info(f"📊 Results saved to: {result}")
        
    except KeyboardInterrupt:
        logger.warning("⚠️  Calculation interrupted by user")
        logger.info("💡 Progress is cached - you can resume by running again")
        
    except Exception as e:
        logger.error(f"❌ Calculation failed with error: {e}")
        logger.info("💡 Check the logs above for details. Progress may be cached.")
        sys.exit(1)

if __name__ == "__main__":
    main()
