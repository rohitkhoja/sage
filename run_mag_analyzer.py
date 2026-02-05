#!/usr/bin/env python3
"""
Simple runner script for MAG Multi-Feature Analyzer
"""

from mag_multi_feature_analyzer import MAGMultiFeatureAnalyzer
from loguru import logger

def main():
    logger.info("🚀 MAG Multi-Feature Graph-Enhanced Retrieval Analysis")
    logger.info("=" * 70)
    
    # Initialize analyzer
    # Pre-loads all content, embeddings, and neighbor graphs for ultra-fast processing
    analyzer = MAGMultiFeatureAnalyzer(
        gpu_id=0,
        load_all_embeddings=True  # Always pre-load everything
    )
    
    # Load data
    analyzer.load_data()
    
    # Run analysis on first 10 queries for testing
    logger.info("\n🧪 Running test on first 10 queries...")
    results, summary = analyzer.run_analysis(max_queries=10)
    
    logger.info("\n✅ Test completed successfully!")
    logger.info("💡 To run on all queries, set max_queries=None")

if __name__ == "__main__":
    main()

