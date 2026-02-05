#!/usr/bin/env python3
"""
Multi-Feature HNSW Index Builder
Builds separate HNSW indices for each embedding feature from STARK chunked cache
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import torch
import faiss
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
import gc
from loguru import logger

class MultiFeatureHNSWBuilder:
    """
    Builds separate HNSW indices for each embedding feature
    """
    
    def __init__(self, 
                 cache_dir: str = "/shared/khoja/CogComp/output/stark_chunked_cache",
                 output_dir: str = "/shared/khoja/CogComp/output/multi_feature_hnsw"):
        
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.embeddings_dir = self.cache_dir / "embeddings"
        
        # Feature names to process
        self.embedding_features = [
            'content_embedding',
            'title_embedding', 
            'feature_embedding',
            'detail_embedding',
            'description_embedding',
            'reviews_summary_embedding',
            'reviews_text_embedding'
        ]
        
        # Storage for embeddings and mappings
        self.feature_embeddings = {}  # feature_name -> list of embeddings
        self.feature_mappings = {}    # feature_name -> list of (chunk_id, asin)
        
        logger.info(f"🚀 Initialized Multi-Feature HNSW Builder")
        logger.info(f"   📁 Cache directory: {self.cache_dir}")
        logger.info(f"   📁 Output directory: {self.output_dir}")
    
    def load_all_embeddings(self):
        """Load all embeddings from chunk files"""
        logger.info("📥 Loading all chunk embeddings...")
        
        # Initialize storage
        for feature in self.embedding_features:
            self.feature_embeddings[feature] = []
            self.feature_mappings[feature] = []
        
        # Load all chunk files
        chunk_files = sorted(self.embeddings_dir.glob("chunk_*_embeddings.json"))
        logger.info(f"   Found {len(chunk_files)} chunk files to process")
        
        total_chunks_processed = 0
        
        for chunk_file in tqdm(chunk_files, desc="Loading chunk files"):
            logger.info(f"   Processing {chunk_file.name}...")
            
            with open(chunk_file, 'r') as f:
                chunks_data = json.load(f)
            
            chunks_in_file = 0
            
            for chunk_data in chunks_data:
                chunk_id = chunk_data.get('chunk_id', '')
                asin = chunk_data.get('asin', '')
                
                if not chunk_id or not asin:
                    continue
                
                # Process each embedding feature
                for feature in self.embedding_features:
                    embedding = chunk_data.get(feature)
                    
                    # Only include if embedding exists and is non-empty
                    if embedding and isinstance(embedding, list) and len(embedding) > 0:
                        # Check if embedding has non-zero values
                        if any(x != 0.0 for x in embedding):
                            self.feature_embeddings[feature].append(embedding)
                            self.feature_mappings[feature].append({
                                'chunk_id': chunk_id,
                                'asin': asin,
                                'chunk_type': chunk_data.get('chunk_type', ''),
                                'index': len(self.feature_embeddings[feature]) - 1
                            })
                
                chunks_in_file += 1
            
            total_chunks_processed += chunks_in_file
            logger.info(f"   ✅ Processed {chunks_in_file:,} chunks from {chunk_file.name}")
        
        # Log summary
        logger.info(f"📊 Embedding loading summary:")
        logger.info(f"   Total chunks processed: {total_chunks_processed:,}")
        
        for feature in self.embedding_features:
            count = len(self.feature_embeddings[feature])
            logger.info(f"   {feature}: {count:,} embeddings")
    
    def build_hnsw_index(self, feature_name: str) -> Tuple[Optional[faiss.Index], Optional[str]]:
        """Build HNSW index for a specific feature"""
        
        embeddings = self.feature_embeddings[feature_name]
        mappings = self.feature_mappings[feature_name]
        
        if not embeddings:
            logger.warning(f"   ⚠️  No embeddings found for {feature_name}, skipping...")
            return None, None
        
        logger.info(f"🔧 Building HNSW index for {feature_name}")
        logger.info(f"   📊 {len(embeddings):,} embeddings to index")
        
        # Convert to numpy array
        embeddings_matrix = np.array(embeddings, dtype=np.float32)
        dimension = embeddings_matrix.shape[1]
        
        logger.info(f"   📏 Embedding dimension: {dimension}")
        
        # Create HNSW index
        hnsw_index = faiss.IndexHNSWFlat(dimension, 64)  # 64 connections per node
        hnsw_index.hnsw.efConstruction = 2000  # Higher quality construction
        hnsw_index.hnsw.efSearch = 1000        # Higher quality search
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_matrix)
        
        # Build index
        logger.info(f"   🔥 Adding {len(embeddings):,} embeddings to index...")
        hnsw_index.add(embeddings_matrix)
        
        # Save index
        index_file = self.output_dir / f"{feature_name}_hnsw_index.faiss"
        faiss.write_index(hnsw_index, str(index_file))
        
        # Save mapping
        mapping_file = self.output_dir / f"{feature_name}_mapping.pkl"
        with open(mapping_file, 'wb') as f:
            pickle.dump(mappings, f)
        
        logger.info(f"   ✅ Index saved to {index_file.name}")
        logger.info(f"   ✅ Mapping saved to {mapping_file.name}")
        
        # Clear memory
        del embeddings_matrix
        gc.collect()
        
        return hnsw_index, str(index_file)
    
    def build_all_indices(self):
        """Build HNSW indices for all features"""
        logger.info("🚀 Building all HNSW indices...")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # First load all embeddings
        self.load_all_embeddings()
        
        # Build indices for each feature
        built_indices = {}
        
        for feature in self.embedding_features:
            feature_start = time.time()
            
            index, index_file = self.build_hnsw_index(feature)
            
            if index is not None:
                built_indices[feature] = {
                    'index_file': index_file,
                    'embedding_count': len(self.feature_embeddings[feature]),
                    'build_time': time.time() - feature_start
                }
            
            # Clear feature data to save memory
            self.feature_embeddings[feature] = []
            self.feature_mappings[feature] = []
            gc.collect()
        
        total_time = time.time() - start_time
        
        # Save build summary
        build_summary = {
            'build_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_build_time': total_time,
            'indices_built': built_indices,
            'embedding_features': self.embedding_features
        }
        
        summary_file = self.output_dir / "build_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(build_summary, f, indent=2)
        
        # Final summary
        logger.info("🎉 All HNSW indices built successfully!")
        logger.info(f"⏱️  Total build time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        logger.info(f"📁 Results saved to: {self.output_dir}")
        
        logger.info("\n📊 Build Summary:")
        for feature, info in built_indices.items():
            logger.info(f"   {feature}: {info['embedding_count']:,} embeddings in {info['build_time']:.1f}s")
        
        return built_indices

def main():
    """Main execution function"""
    logger.info("🚀 Multi-Feature HNSW Index Builder")
    logger.info("💡 Building separate HNSW indices for each embedding feature")
    logger.info("=" * 60)
    
    builder = MultiFeatureHNSWBuilder()
    
    try:
        results = builder.build_all_indices()
        
        logger.info("🎉 Multi-Feature HNSW building completed successfully!")
        logger.info(f"📁 All indices saved to: {builder.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Multi-Feature HNSW building failed: {e}")
        raise

if __name__ == "__main__":
    main()
