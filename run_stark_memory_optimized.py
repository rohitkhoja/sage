#!/usr/bin/env python3
"""
Memory-Optimized STARK Pipeline Runner
Sets optimal CUDA memory configuration and runs with conservative settings
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from loguru import logger

def set_cuda_memory_config():
    """Set optimal CUDA memory configuration"""
    # Set CUDA memory configuration for better memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For debugging memory issues
    
    logger.info("🔧 Set CUDA memory configuration:")
    logger.info(f"   PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}")
    logger.info(f"   CUDA_LAUNCH_BLOCKING={os.environ['CUDA_LAUNCH_BLOCKING']}")

def check_gpu_memory():
    """Check available GPU memory"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            memory_free = [int(x.strip()) for x in result.stdout.strip().split('\n')]
            logger.info("🔍 GPU Memory Status:")
            for i, mem in enumerate(memory_free):
                logger.info(f"   GPU {i}: {mem} MB free")
            return min(memory_free)
        else:
            logger.warning("Could not check GPU memory")
            return None
    except Exception as e:
        logger.warning(f"Error checking GPU memory: {e}")
        return None

def main():
    """Run memory-optimized STARK pipeline"""
    logger.info("🚀 Memory-Optimized STARK Pipeline Runner")
    logger.info("=" * 60)
    
    # Set CUDA memory configuration
    set_cuda_memory_config()
    
    # Check GPU memory
    min_free_memory = check_gpu_memory()
    if min_free_memory and min_free_memory < 8000:  # Less than 8GB free
        logger.warning(f"⚠️  Low GPU memory detected: {min_free_memory} MB")
        logger.warning("   Consider clearing GPU memory or reducing batch sizes")
    
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    logger.info("🏃‍♂️ Starting ultra-fast pipeline with memory optimization...")
    
    # Run the optimized pipeline
    try:
        result = subprocess.run([
            sys.executable, 
            'run_stark_ultrafast_demo.py', 
            '--mode', 'full',
            '--k-neighbors', '200'
        ], env=os.environ, check=True)
        
        logger.info("🎉 Pipeline completed successfully!")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Pipeline failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        logger.info("⏹️  Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
