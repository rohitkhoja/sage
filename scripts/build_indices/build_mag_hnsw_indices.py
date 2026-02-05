#!/usr/bin/env python3
"""
Phase 1: Build HNSW Indices for MAG Dataset Features

This script builds separate HNSW indices for each embedding feature:
- Paper features: content, original_title, abstract, authors, fields_of_study, cites 
- Author features: content, display_name, institution

Only includes nodes where the feature value is non-empty.
Uses object_id as the key for each HNSW index.
"""

import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import numpy as np
import faiss
import gc
from loguru import logger

class MAGHNSWBuilder:
    """Build HNSW indices for all MAG features"""
    
    def __init__(self, embeddings_dir: str, output_dir: str):
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature definitions
        self.paper_features = [
            'content_embedding',
            'original_title_embedding', 
            'abstract_embedding',
            'authors_embedding',
            'fields_of_study_embedding',
            'cites_embedding'
        ]
        
        self.author_features = [
            'content_embedding',
            'display_name_embedding',
            'institution_embedding'
        ]
        
        # HNSW parameters (using FAISS)
        self.embedding_dim = 384 # MiniLM-L6-v2 dimension
        self.hnsw_params = {
            'M': 64, # Number of connections per element 
            'ef_construction': 2000, # Higher quality construction
            'ef_search': 1000, # Higher quality search
        }
        
        # Statistics
        self.stats = {
            'total_chunks': 0,
            'total_nodes': 0,
            'paper_nodes': 0,
            'author_nodes': 0,
            'feature_counts': defaultdict(int),
            'empty_feature_counts': defaultdict(int)
        }
        
    def scan_all_chunks(self) -> Dict[str, Dict[int, np.ndarray]]:
        """
        Scan all chunks and collect embeddings for each feature
        Returns: {feature_name: {object_id: embedding_vector}}
        """
        logger.info(" Scanning all chunks to collect feature embeddings...")
        
        feature_embeddings = {
            feature: {} for feature in (self.paper_features + self.author_features)
        }
        
        chunk_files = sorted(self.embeddings_dir.glob("chunk_*.json"))
        self.stats['total_chunks'] = len(chunk_files)
        
        logger.info(f" Found {len(chunk_files)} chunk files")
        
        for i, chunk_file in enumerate(chunk_files):
            logger.info(f" Processing {chunk_file.name} ({i+1}/{len(chunk_files)})")
            
            with open(chunk_file, 'r') as f:
                chunk_data = json.load(f)
            
            self.stats['total_nodes'] += len(chunk_data)
            
            for node in chunk_data:
                object_id = node['object_id']
                node_type = node['node_type']
                
                # Count node types
                if node_type == 'paper':
                    self.stats['paper_nodes'] += 1
                    relevant_features = self.paper_features
                elif node_type == 'author':
                    self.stats['author_nodes'] += 1
                    relevant_features = self.author_features
                else:
                    logger.warning(f"Unknown node type: {node_type}")
                    continue
                
                # Extract embeddings for relevant features
                for feature in relevant_features:
                    if feature in node:
                        embedding = node[feature]
                        
                        # Check if embedding is valid (non-empty)
                        if embedding and len(embedding) == self.embedding_dim:
                            feature_embeddings[feature][object_id] = np.array(embedding, dtype=np.float32)
                            self.stats['feature_counts'][feature] += 1
                        else:
                            self.stats['empty_feature_counts'][feature] += 1
                    else:
                        self.stats['empty_feature_counts'][feature] += 1
        
        logger.info(" Chunk scanning complete!")
        self._log_statistics()
        
        return feature_embeddings
    
    def _log_statistics(self):
        """Log collection statistics"""
        logger.info("\n COLLECTION STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total chunks: {self.stats['total_chunks']}")
        logger.info(f"Total nodes: {self.stats['total_nodes']:,}")
        logger.info(f"Paper nodes: {self.stats['paper_nodes']:,}")
        logger.info(f"Author nodes: {self.stats['author_nodes']:,}")
        logger.info("\n Feature Embedding Counts:")
        
        for feature in sorted(self.stats['feature_counts'].keys()):
            valid_count = self.stats['feature_counts'][feature]
            empty_count = self.stats['empty_feature_counts'][feature]
            total = valid_count + empty_count
            pct = (valid_count / total * 100) if total > 0 else 0
            logger.info(f" {feature:25}: {valid_count:8,} valid / {total:8,} total ({pct:5.1f}%)")
    
    def build_hnsw_index(self, feature_name: str, embeddings_dict: Dict[int, np.ndarray]) -> str:
        """
        Build HNSW index for a specific feature using FAISS
        Returns: path to saved index
        """
        if not embeddings_dict:
            logger.warning(f" No embeddings found for {feature_name}, skipping")
            return None
            
        logger.info(f" Building HNSW index for {feature_name} ({len(embeddings_dict):,} embeddings)")
        
        # Prepare data
        object_ids = list(embeddings_dict.keys())
        embeddings_matrix = np.array([embeddings_dict[oid] for oid in object_ids], dtype=np.float32)
        
        logger.info(f" Embedding dimension: {embeddings_matrix.shape[1]}")
        logger.info(f" Adding {len(object_ids):,} embeddings to index...")
        
        # Create FAISS HNSW index
        hnsw_index = faiss.IndexHNSWFlat(self.embedding_dim, self.hnsw_params['M'])
        hnsw_index.hnsw.efConstruction = self.hnsw_params['ef_construction']
        hnsw_index.hnsw.efSearch = self.hnsw_params['ef_search']
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_matrix)
        
        # Build index
        start_time = time.time()
        hnsw_index.add(embeddings_matrix)
        build_time = time.time() - start_time
        
        # Save index
        index_path = self.output_dir / f"{feature_name}_hnsw.faiss"
        faiss.write_index(hnsw_index, str(index_path))
        
        # Save object_id mapping 
        mapping_path = self.output_dir / f"{feature_name}_mapping.pkl"
        with open(mapping_path, 'wb') as f:
            pickle.dump(object_ids, f)
        
        logger.info(f" Index built and saved in {build_time:.2f}s")
        logger.info(f" Index: {index_path}")
        logger.info(f" Mapping: {mapping_path}")
        
        # Clear memory
        del embeddings_matrix
        gc.collect()
        
        return str(index_path)
    
    def build_all_indices(self):
        """Build HNSW indices for all features"""
        logger.info(" STARTING MAG HNSW INDEX BUILDING")
        logger.info("=" * 60)
        
        total_start_time = time.time()
        
        # Step 1: Scan all chunks and collect embeddings
        feature_embeddings = self.scan_all_chunks()
        
        # Step 2: Build HNSW indices for each feature
        logger.info("\n Building HNSW indices...")
        
        built_indices = {}
        
        all_features = self.paper_features + self.author_features
        for i, feature in enumerate(all_features):
            logger.info(f"\n--- Building index {i+1}/{len(all_features)}: {feature} ---")
            
            embeddings_dict = feature_embeddings[feature]
            index_path = self.build_hnsw_index(feature, embeddings_dict)
            
            if index_path:
                built_indices[feature] = {
                    'index_path': index_path,
                    'mapping_path': str(self.output_dir / f"{feature}_mapping.pkl"),
                    'count': len(embeddings_dict)
                }
        
        # Step 3: Save master manifest
        manifest = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_time_seconds': time.time() - total_start_time,
            'embedding_dimension': self.embedding_dim,
            'hnsw_parameters': self.hnsw_params,
            'statistics': dict(self.stats),
            'indices': built_indices
        }
        
        manifest_path = self.output_dir / 'hnsw_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        total_time = time.time() - total_start_time
        logger.info("\n ALL HNSW INDICES BUILT SUCCESSFULLY!")
        logger.info("=" * 50)
        logger.info(f" Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        logger.info(f" Output directory: {self.output_dir}")
        logger.info(f" Manifest: {manifest_path}")
        logger.info(f" Built {len(built_indices)} HNSW indices")
        
        return manifest_path

def main():
    """Main function to build all HNSW indices"""
    
    # Configuration
    embeddings_dir = "/shared/khoja/CogComp/output/mag_final_cache/embeddings"
    output_dir = "/shared/khoja/CogComp/output/mag_hnsw_indices"
    
    logger.info(" MAG HNSW INDEX BUILDER")
    logger.info(" Building separate HNSW indices for each embedding feature")
    logger.info("")
    logger.info(f" Input: {embeddings_dir}")
    logger.info(f" Output: {output_dir}")
    logger.info("")
    
    # Verify input directory exists
    if not Path(embeddings_dir).exists():
        logger.error(f" Embeddings directory not found: {embeddings_dir}")
        return False
    
    # Build indices
    try:
        builder = MAGHNSWBuilder(embeddings_dir, output_dir)
        manifest_path = builder.build_all_indices()
        
        logger.info("\n NEXT STEPS:")
        logger.info("1. Review the manifest file to verify all indices were built")
        logger.info("2. Run the neighbor finding script to build the graph")
        logger.info("3. Test the indices with sample queries")
        
        return True
        
    except Exception as e:
        logger.error(f" Failed to build HNSW indices: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

