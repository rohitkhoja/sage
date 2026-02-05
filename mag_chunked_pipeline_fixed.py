#!/usr/bin/env python3
"""
Fixed MAG Chunked Pipeline with Ultra-Aggressive Memory Management
Restarts process every 10 chunks to prevent memory fragmentation
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from loguru import logger
import torch
import gc

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from mag_chunked_pipeline import MAGChunkedPipeline

@dataclass
class ChunkProgress:
    """Track progress of chunk processing"""
    completed_chunks: List[int]
    failed_chunks: List[int]
    current_batch: int
    total_chunks: int

class MAGChunkedPipelineFixed:
    """
    Fixed MAG pipeline that restarts process every 10 chunks to prevent memory issues
    """
    
    def __init__(self, cache_dir: str = "/shared/khoja/CogComp/output/mag_chunked_cache_fixed"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.cache_dir / "progress.json"
        self.chunks_per_batch = 10  # Process 10 chunks per batch to prevent memory issues
        
        # Set ultra-aggressive PyTorch memory configuration
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
        
        logger.info("🔧 Fixed MAG Chunked Pipeline initialized")
        logger.info(f"   📦 Chunks per batch: {self.chunks_per_batch}")
        logger.info(f"   💾 Cache directory: {self.cache_dir}")
    
    def load_progress(self) -> ChunkProgress:
        """Load progress from file"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                return ChunkProgress(**data)
            except Exception as e:
                logger.warning(f"Failed to load progress: {e}")
        
        return ChunkProgress(
            completed_chunks=[],
            failed_chunks=[],
            current_batch=0,
            total_chunks=0
        )
    
    def save_progress(self, progress: ChunkProgress):
        """Save progress to file"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(asdict(progress), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def run_single_batch(self, start_chunk: int, end_chunk: int) -> bool:
        """Run a single batch of chunks (10 chunks max)"""
        logger.info(f"🚀 Running batch: chunks {start_chunk}-{end_chunk-1}")
        
        # Create pipeline for this batch
        pipeline = MAGChunkedPipeline(
            cache_dir=str(self.cache_dir),
            chunk_size=50000,  # Smaller chunks for better memory management
            use_gpu=True,
            num_threads=8
        )
        
        try:
            # Process only the specified chunk range
            success = pipeline.run_chunked_pipeline(
                start_chunk=start_chunk,
                end_chunk=end_chunk
            )
            
            if success:
                logger.info(f"✅ Batch {start_chunk}-{end_chunk-1} completed successfully")
                return True
            else:
                logger.error(f"❌ Batch {start_chunk}-{end_chunk-1} failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Batch {start_chunk}-{end_chunk-1} failed with error: {e}")
            return False
        finally:
            # Ultra-aggressive cleanup before returning
            self._ultra_aggressive_cleanup()
    
    def _ultra_aggressive_cleanup(self):
        """Ultra-aggressive cleanup to free all possible memory"""
        logger.info("🧹 Performing ultra-aggressive cleanup...")
        
        # Clear PyTorch cache on all GPUs
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # Force memory compaction
                    torch.cuda.reset_peak_memory_stats()
        
        # Multiple rounds of garbage collection
        for _ in range(5):
            gc.collect()
            time.sleep(0.1)
        
        # Clear Python cache
        if hasattr(sys, '_clear_type_cache'):
            sys._clear_type_cache()
        
        logger.info("✅ Ultra-aggressive cleanup completed")
    
    def run_fixed_pipeline(self, max_chunks: Optional[int] = None):
        """Run the fixed pipeline with batch processing"""
        logger.info("🚀 Starting Fixed MAG Chunked Pipeline")
        logger.info("=" * 60)
        
        # Load existing progress
        progress = self.load_progress()
        
        # Determine total chunks to process
        if max_chunks:
            total_chunks = max_chunks
        else:
            # Get total chunks from the dataset
            pipeline = MAGChunkedPipeline(
                cache_dir=str(self.cache_dir),
                chunk_size=50000,
                use_gpu=True,
                num_threads=8
            )
            total_chunks = pipeline.get_total_chunks()
        
        progress.total_chunks = total_chunks
        
        logger.info(f"📊 Total chunks to process: {total_chunks}")
        logger.info(f"📊 Already completed: {len(progress.completed_chunks)}")
        logger.info(f"📊 Failed chunks: {len(progress.failed_chunks)}")
        
        # Process chunks in batches
        current_chunk = 0
        batch_number = 0
        
        while current_chunk < total_chunks:
            end_chunk = min(current_chunk + self.chunks_per_batch, total_chunks)
            
            logger.info(f"🔄 Processing Batch {batch_number + 1}: chunks {current_chunk}-{end_chunk-1}")
            
            # Run this batch
            success = self.run_single_batch(current_chunk, end_chunk)
            
            if success:
                # Mark chunks as completed
                for chunk_id in range(current_chunk, end_chunk):
                    if chunk_id not in progress.completed_chunks:
                        progress.completed_chunks.append(chunk_id)
                
                logger.info(f"✅ Batch {batch_number + 1} completed successfully")
            else:
                # Mark chunks as failed
                for chunk_id in range(current_chunk, end_chunk):
                    if chunk_id not in progress.failed_chunks:
                        progress.failed_chunks.append(chunk_id)
                
                logger.error(f"❌ Batch {batch_number + 1} failed")
            
            # Save progress
            progress.current_batch = batch_number + 1
            self.save_progress(progress)
            
            # Move to next batch
            current_chunk = end_chunk
            batch_number += 1
            
            # Small delay between batches
            time.sleep(2)
        
        # Final summary
        logger.info("🎉 Fixed MAG Pipeline completed!")
        logger.info(f"✅ Completed chunks: {len(progress.completed_chunks)}")
        logger.info(f"❌ Failed chunks: {len(progress.failed_chunks)}")
        
        if progress.failed_chunks:
            logger.warning(f"⚠️  Failed chunks: {progress.failed_chunks}")
            logger.info("💡 You can retry failed chunks by running the pipeline again")

def main():
    """Main execution function"""
    logger.info("🔧 Fixed MAG Chunked Pipeline")
    logger.info("=" * 50)
    
    # Create fixed pipeline
    pipeline = MAGChunkedPipelineFixed()
    
    # Run the fixed pipeline
    pipeline.run_fixed_pipeline()

if __name__ == "__main__":
    main()

