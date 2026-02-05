#!/usr/bin/env python3
"""
STARK Chunked Pipeline Runner
Memory-efficient, fault-tolerant runner for processing STARK dataset in chunks
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

from stark_chunked_pipeline import STARKChunkedPipeline

def setup_cuda_environment():
    """Setup CUDA environment for optimal memory usage"""
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,roundup_power2_divisions:16'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '0'
    os.environ['OMP_NUM_THREADS'] = '32'
    os.environ['MKL_NUM_THREADS'] = '32'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    logger.info("🔧 Set CUDA environment for chunked processing:")
    for key in ['PYTORCH_CUDA_ALLOC_CONF', 'CUDA_LAUNCH_BLOCKING', 'OMP_NUM_THREADS']:
        logger.info(f"    {key}={os.environ.get(key)}")

def check_system_resources():
    """Check and display system resources"""
    logger.info("🔍 System Resource Check for Chunked Processing:")
    
    # GPU info
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            try:
                # Try to get current GPU memory usage
                result = subprocess.run(['nvidia-ml-py3', '-c', f'import pynvml; pynvml.nvmlInit(); h=pynvml.nvmlDeviceGetHandleByIndex({i}); m=pynvml.nvmlDeviceGetMemoryInfo(h); print(f"{{(m.total-m.used)/1024**3:.1f}}")'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    free_memory = float(result.stdout.strip())
                    logger.info(f"    GPU {i}: {free_memory:.1f} GB free ({gpu_name})")
                else:
                    logger.info(f"    GPU {i}: {gpu_memory:.1f} GB total ({gpu_name})")
            except:
                logger.info(f"    GPU {i}: {gpu_memory:.1f} GB total ({gpu_name})")
    
    # RAM info
    ram = psutil.virtual_memory()
    logger.info(f"    RAM: {ram.available/1024**3:.1f} GB available / {ram.total/1024**3:.1f} GB total")
    
    # CPU info
    logger.info(f"    CPU: {psutil.cpu_count()} cores")

def monitor_chunk_progress(pipeline):
    """Monitor and display chunk processing progress"""
    if not pipeline.chunks_info:
        return
    
    total_chunks = len(pipeline.chunks_info)
    completed = len([c for c in pipeline.chunks_info if c.status == 'completed'])
    processing = len([c for c in pipeline.chunks_info if c.status == 'processing'])
    failed = len([c for c in pipeline.chunks_info if c.status == 'failed'])
    pending = total_chunks - completed - processing - failed
    
    logger.info(f"📊 Chunk Progress: {completed}/{total_chunks} completed, {processing} processing, {failed} failed, {pending} pending")
    
    if completed > 0:
        total_time = sum(c.processing_time or 0 for c in pipeline.chunks_info if c.processing_time)
        avg_time = total_time / completed
        est_remaining = avg_time * pending
        logger.info(f"⏱️  Average time per chunk: {avg_time:.1f}s, Estimated remaining: {est_remaining/60:.1f} minutes")

def run_chunked_pipeline_demo(chunk_size: int = 50000, max_products: int = None):
    """
    Run the STARK chunked pipeline demo
    
    Args:
        chunk_size: Number of products per chunk (default: 50,000)
        max_products: Maximum products to process (None for all)
    """
    
    mode = "FULL DATASET" if max_products is None else f"LIMITED ({max_products:,} products)"
    logger.info(f"🚀 Starting STARK Chunked Pipeline Demo - {mode}")
    logger.info(f"📦 Chunk size: {chunk_size:,} products per chunk")
    logger.info("💡 Fault-tolerant design prevents memory issues and enables resumability")
    logger.info("=" * 70)
    
    # Setup environment
    setup_cuda_environment()
    check_system_resources()
    
    # Initialize pipeline
    logger.info("\n🔧 Initializing Chunked Pipeline...")
    start_time = time.time()
    
    pipeline = STARKChunkedPipeline(
        cache_dir="/shared/khoja/CogComp/output/stark_chunked_cache",
        chunk_size=chunk_size,
        use_gpu=True,
        num_threads=32
    )
    
    init_time = time.time() - start_time
    logger.info(f"✅ Pipeline initialized in {init_time:.2f} seconds")
    
    # Display cache status
    logger.info("\n📁 Cache Status:")
    cache_files = list(pipeline.cache_dir.rglob("*"))
    if cache_files:
        logger.info(f"    Found {len(cache_files)} existing cache files")
        if pipeline.chunk_manifest_file.exists():
            pipeline._load_chunk_manifest()
            monitor_chunk_progress(pipeline)
    else:
        logger.info("    No existing cache found - fresh start")
    
    # Run pipeline
    logger.info("\n🎯 Starting Chunked Pipeline Execution...")
    logger.info("💫 Processing data in manageable chunks for maximum reliability!")
    
    try:
        analysis_file = pipeline.run_chunked_pipeline(
            max_products=max_products,
            k_neighbors=200
        )
        
        total_time = time.time() - start_time
        
        logger.info("\n🎉 Chunked Pipeline Demo completed successfully!")
        logger.info(f"⏱️  Total execution time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        logger.info(f"📁 Final analysis: {analysis_file}")
        
        # Final progress summary
        monitor_chunk_progress(pipeline)
        
        return analysis_file
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        
        # Show progress even on failure
        if pipeline.chunks_info:
            monitor_chunk_progress(pipeline)
            logger.info("💡 Pipeline is resumable - you can restart and it will continue from where it left off!")
        
        raise

def main():
    """Main function with argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="STARK Chunked Pipeline Runner")
    parser.add_argument("--chunk-size", type=int, default=50000, 
                       help="Number of products per chunk (default: 50000)")
    parser.add_argument("--max-products", type=int, default=None,
                       help="Maximum products to process (default: all)")
    parser.add_argument("--mode", choices=["test", "full"], default="full",
                       help="Run mode: test (small sample) or full (all data)")
    
    args = parser.parse_args()
    
    # Set parameters based on mode
    if args.mode == "test":
        chunk_size = min(args.chunk_size, 10000)  # Smaller chunks for testing
        max_products = 20000  # Process only 20k products for testing
        logger.info("🧪 Running in TEST mode with limited data")
    else:
        chunk_size = args.chunk_size
        max_products = args.max_products
        logger.info("🚀 Running in FULL mode")
    
    try:
        result = run_chunked_pipeline_demo(
            chunk_size=chunk_size,
            max_products=max_products
        )
        
        logger.info("✅ Pipeline execution completed successfully!")
        logger.info(f"📊 Results saved to: {result}")
        
    except KeyboardInterrupt:
        logger.warning("⚠️  Pipeline interrupted by user")
        logger.info("💡 You can resume the pipeline by running the same command again")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {e}")
        logger.info("💡 Check the logs above for details. Pipeline is resumable.")
        sys.exit(1)

if __name__ == "__main__":
    main()

