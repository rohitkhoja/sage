#!/usr/bin/env python3
"""
STARK Resumable Pipeline Demo Script
Test the resumable pipeline with stage-by-stage execution
"""

import sys
import json
import argparse
import time
from pathlib import Path
from loguru import logger
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from stark_graph_analysis_pipeline_resumable import STARKResumablePipeline

def run_resumable_pipeline_demo(max_products=None, k_neighbors=200, gpu_batch_size=50000):
    """
    Run the resumable STARK pipeline with detailed stage information
    """
    
    mode = "FULL DATASET" if max_products is None else f"LIMITED ({max_products} products)"
    logger.info(f"🚀 Starting STARK Resumable Pipeline Demo - {mode}")
    logger.info("=" * 70)
    
    # Display system information
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"🔥 Multi-GPU System: {num_gpus} GPUs available")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        logger.info(f"   GPU Batch Size: {gpu_batch_size:,} chunks per GPU")
    else:
        logger.info("🖥️  CPU Mode: No GPUs available")
    
    # Initialize pipeline
    logger.info("\n🔧 Initializing Resumable Pipeline...")
    start_time = time.time()
    
    pipeline = STARKResumablePipeline(
        cache_dir="/shared/khoja/CogComp/output/stark_resumable_demo_cache",
        gpu_batch_size=gpu_batch_size
    )
    
    init_time = time.time() - start_time
    logger.info(f"✅ Pipeline initialized in {init_time:.2f} seconds")
    
    # Show cache status
    logger.info("\n📁 Stage Cache Status:")
    for stage_name, cache_file in pipeline.stage_files.items():
        status = "✅ EXISTS" if cache_file.exists() else "❌ MISSING"
        logger.info(f"   {stage_name}: {status}")
    
    try:
        total_start_time = time.time()
        
        # Run full pipeline
        logger.info("\n🎯 Starting Pipeline Execution...")
        edges_file = pipeline.run_full_pipeline(max_products=max_products, k_neighbors=k_neighbors)
        
        total_time = time.time() - total_start_time
        
        # Show final results
        if edges_file and edges_file.exists():
            with open(edges_file, 'r') as f:
                final_edges = json.load(f)
            
            logger.info("\n📊 PIPELINE RESULTS:")
            logger.info(f"   ⏱️  Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
            logger.info(f"   📦 Total chunks processed: {len(pipeline.chunks):,}")
            logger.info(f"   🔗 Total similarities calculated: {len(pipeline.similarity_data):,}")
            logger.info(f"   🎯 Final unique edges: {len(final_edges):,}")
            logger.info(f"   💾 Results saved to: {edges_file}")
            
            # Show edge breakdown
            edge_breakdown = {}
            for edge in final_edges:
                key = f"{edge['edge_type']}_{edge['reason']}"
                edge_breakdown[key] = edge_breakdown.get(key, 0) + 1
            
            logger.info("\n📈 Edge Breakdown by Type and Reason:")
            for key, count in edge_breakdown.items():
                logger.info(f"   - {key}: {count:,}")
            
            # Show sample edges
            logger.info(f"\n🔍 Sample Edges (showing first 3):")
            for i, edge in enumerate(final_edges[:3]):
                logger.info(f"   Edge {i+1}:")
                logger.info(f"      Type: {edge['edge_type']}")
                logger.info(f"      Reason: {edge['reason']}")
                logger.info(f"      Source: {edge['source_chunk_id']}")
                logger.info(f"      Target: {edge['target_chunk_id']}")
                logger.info(f"      Similarity: {edge['semantic_similarity']:.3f}")
                logger.info(f"      Entity Count: {edge['entity_count']}")
            
            return edges_file
        else:
            logger.warning("❌ No edges were generated!")
            return None
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise

def test_resume_functionality():
    """Test the resume functionality by running stages individually"""
    logger.info("🔄 Testing Resume Functionality")
    logger.info("=" * 50)
    
    pipeline = STARKResumablePipeline(
        cache_dir="/shared/khoja/CogComp/output/stark_resume_test_cache",
        gpu_batch_size=10000  # Smaller batch for testing
    )
    
    try:
        # Test each stage individually
        logger.info("Stage 1: Loading chunks...")
        pipeline.stage1_load_chunks(max_products=5)  # Small test
        
        logger.info("Stage 2: Calculating field embeddings...")
        pipeline.stage2_calculate_field_embeddings()
        
        logger.info("Stage 3: Calculating content embeddings...")
        pipeline.stage3_calculate_content_embeddings()
        
        logger.info("Stage 4: Building HNSW index...")
        pipeline.stage4_build_hnsw_index()
        
        logger.info("Stage 5: Calculating similarities...")
        pipeline.stage5_calculate_similarities(k_neighbors=50)
        
        logger.info("Stage 6: Generating analysis...")
        edges_file = pipeline.stage6_generate_analysis()
        
        logger.info("✅ Resume functionality test completed!")
        logger.info(f"📁 Results: {edges_file}")
        
        # Test resume by running again (should use cache)
        logger.info("\n🔄 Testing resume by running again...")
        start_time = time.time()
        
        pipeline2 = STARKResumablePipeline(
            cache_dir="/shared/khoja/CogComp/output/stark_resume_test_cache"
        )
        edges_file2 = pipeline2.run_full_pipeline(max_products=5)
        
        resume_time = time.time() - start_time
        logger.info(f"✅ Resume test completed in {resume_time:.2f} seconds (should be very fast)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Resume test failed: {e}")
        return False

def show_cache_info(cache_dir: str):
    """Show information about cached files"""
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        logger.info(f"📁 Cache directory does not exist: {cache_dir}")
        return
    
    logger.info(f"📁 Cache Directory: {cache_dir}")
    logger.info("   Cached Files:")
    
    total_size = 0
    cache_files = list(cache_path.glob("*"))
    cache_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    for file_path in cache_files:
        if file_path.is_file():
            size_mb = file_path.stat().st_size / 1024 / 1024
            total_size += size_mb
            modified_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_path.stat().st_mtime))
            logger.info(f"   - {file_path.name}: {size_mb:.1f} MB (modified: {modified_time})")
    
    logger.info(f"   Total cache size: {total_size:.1f} MB")

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(
        description="STARK Resumable Pipeline Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick demo with 10 products
  python run_stark_resumable_demo.py --mode demo --max-products 10
  
  # Larger demo with 1000 products
  python run_stark_resumable_demo.py --mode demo --max-products 1000
  
  # Full dataset processing
  python run_stark_resumable_demo.py --mode full
  
  # Test resume functionality
  python run_stark_resumable_demo.py --test-resume
  
  # Show cache information
  python run_stark_resumable_demo.py --show-cache
        """
    )
    
    # Main options
    parser.add_argument("--mode", choices=["demo", "full"], default="demo",
                       help="Processing mode: 'demo' for limited dataset, 'full' for complete dataset")
    parser.add_argument("--max-products", type=int, default=50,
                       help="Maximum number of products to process in demo mode (default: 50)")
    
    # Performance options
    parser.add_argument("--k-neighbors", type=int, default=100,
                       help="Number of neighbors for similarity analysis (default: 100)")
    parser.add_argument("--gpu-batch-size", type=int, default=50000,
                       help="Number of chunks to process per GPU batch (default: 50000)")
    
    # Testing options
    parser.add_argument("--test-resume", action="store_true",
                       help="Test the resume functionality with a small dataset")
    parser.add_argument("--show-cache", action="store_true",
                       help="Show information about cached files")
    parser.add_argument("--cache-dir", type=str, default="/shared/khoja/CogComp/output/stark_resumable_demo_cache",
                       help="Cache directory to use")
    
    args = parser.parse_args()
    
    # Handle special operations
    if args.show_cache:
        show_cache_info(args.cache_dir)
        return
    
    if args.test_resume:
        success = test_resume_functionality()
        if success:
            logger.info("🎉 Resume functionality test passed!")
        else:
            logger.error("❌ Resume functionality test failed!")
            sys.exit(1)
        return
    
    # Determine processing parameters
    if args.mode == "full":
        max_products = None
        logger.info("🚀 Running STARK Resumable Pipeline on FULL dataset (957K+ products)")
        logger.warning("⚠️  This will take several hours but can be resumed at any point!")
    else:
        max_products = args.max_products
        logger.info(f"🎯 Running STARK Resumable Pipeline in DEMO mode ({max_products} products)")
    
    # Display configuration
    logger.info("\n🔧 CONFIGURATION:")
    logger.info(f"   Mode: {args.mode}")
    logger.info(f"   Max products: {max_products or 'ALL'}")
    logger.info(f"   K-neighbors: {args.k_neighbors}")
    logger.info(f"   GPU batch size: {args.gpu_batch_size:,}")
    logger.info(f"   Cache directory: {args.cache_dir}")
    
    # Show existing cache info
    show_cache_info(args.cache_dir)
    
    try:
        # Run the pipeline
        edges_file = run_resumable_pipeline_demo(
            max_products=max_products,
            k_neighbors=args.k_neighbors,
            gpu_batch_size=args.gpu_batch_size
        )
        
        if edges_file:
            logger.info("\n🎉 Resumable Pipeline Demo completed successfully!")
            logger.info("🚀 Pipeline is fully resumable - you can stop and restart at any time!")
        else:
            logger.warning("⚠️  Pipeline completed but no edges were generated")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Pipeline interrupted by user")
        logger.info("🔄 You can resume from where you left off by running the same command again")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
