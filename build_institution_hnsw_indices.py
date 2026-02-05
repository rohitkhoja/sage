#!/usr/bin/env python3
"""
Build HNSW Indices for Institution Nodes

This script generates embeddings and builds HNSW indices for institution nodes:
- Institution features: institution name only
- Uses the same embedding service as the main MAG pipeline
- Creates HNSW indices that point directly to institution nodes (not authors)

Input: Institution nodes from node_info.jsonl
Output: HNSW indices in /shared/khoja/CogComp/output/mag_hnsw_indices/
"""

import json
import os
import pickle
import time
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import numpy as np
import faiss
import gc
from loguru import logger

# Add src to path for embedding service
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.processors.embedding_service import EmbeddingService
from src.core.models import ProcessingConfig

class InstitutionHNSWBuilder:
    """Build HNSW indices for institution nodes"""
    
    def __init__(self, institution_nodes_file: str, output_dir: str):
        self.institution_nodes_file = Path(institution_nodes_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Institution feature definitions
        self.institution_features = [
            'institution_name_embedding'
        ]
        
        # HNSW parameters (using FAISS)
        self.embedding_dim = 384  # MiniLM-L6-v2 dimension
        self.hnsw_params = {
            'M': 64,              # Number of connections per element  
            'ef_construction': 2000, # Higher quality construction
            'ef_search': 1000,    # Higher quality search
        }
        
        # Initialize embedding service
        self.config = ProcessingConfig(
            use_faiss=True, 
            faiss_use_gpu=True,
            batch_size=4096,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedding_service = EmbeddingService(self.config)
        
        # Statistics
        self.stats = {
            'total_institution_nodes': 0,
            'processed_nodes': 0,
            'feature_counts': defaultdict(int),
            'empty_feature_counts': defaultdict(int)
        }
        
    def load_institution_nodes(self) -> List[Dict]:
        """Load institution nodes from JSON file"""
        logger.info(f"📖 Loading institution nodes from {self.institution_nodes_file}")
        
        with open(self.institution_nodes_file, 'r') as f:
            institution_nodes = json.load(f)
        
        self.stats['total_institution_nodes'] = len(institution_nodes)
        logger.info(f"✅ Loaded {len(institution_nodes):,} institution nodes")
        
        return institution_nodes
    
    def prepare_institution_texts(self, institution_nodes: List[Dict]) -> Dict[str, Dict[str, List]]:
        """Prepare texts from institution nodes for embedding generation"""
        logger.info("🔍 Preparing institution texts for embedding generation...")
        
        all_text_data = {
            'institution_name': {'texts': [], 'node_indices': []}
        }
        
        for node_idx, node in enumerate(institution_nodes):
            node_index = node['node_index']
            institution_name = node.get('DisplayName', '').strip()
            
            if not institution_name:
                continue
                
            # Institution name (primary institution name)
            all_text_data['institution_name']['texts'].append(institution_name)
            all_text_data['institution_name']['node_indices'].append(node_index)
            
            self.stats['processed_nodes'] += 1
        
        # Log statistics
        for feature, data in all_text_data.items():
            count = len(data['texts'])
            self.stats['feature_counts'][feature] = count
            logger.info(f"  {feature}: {count:,} texts prepared")
        
        return all_text_data
    
    def generate_institution_embeddings(self, institution_texts: Dict[str, Dict[str, List]]) -> Dict[str, Dict[int, np.ndarray]]:
        """Generate embeddings for all institution features"""
        logger.info("🔥 Generating embeddings for institution features...")
        
        institution_embeddings = {}
        
        for feature_name, data in institution_texts.items():
            if not data['texts']:
                logger.warning(f"⚠️  No texts found for {feature_name}, skipping")
                continue
            
            logger.info(f"🔨 Processing {feature_name}: {len(data['texts']):,} texts")
            feature_start = time.time()
            
            try:
                # Generate embeddings using the embedding service
                embeddings = self.embedding_service.generate_embeddings_bulk(data['texts'])
                
                # Create mapping from node_index to embedding
                node_embeddings = {}
                for node_index, embedding in zip(data['node_indices'], embeddings):
                    node_embeddings[node_index] = np.array(embedding, dtype=np.float32)
                
                institution_embeddings[feature_name] = node_embeddings
                
                feature_time = time.time() - feature_start
                logger.info(f"✅ {feature_name} completed in {feature_time:.2f}s")
                
                # Clear memory
                del embeddings
                gc.collect()
                
            except Exception as e:
                logger.error(f"❌ Failed to process {feature_name}: {e}")
                continue
        
        return institution_embeddings
    
    def build_hnsw_index(self, feature_name: str, embeddings_dict: Dict[int, np.ndarray]) -> str:
        """Build HNSW index for a specific institution feature using FAISS"""
        if not embeddings_dict:
            logger.warning(f"⚠️  No embeddings found for {feature_name}, skipping")
            return None
            
        logger.info(f"🔨 Building HNSW index for {feature_name} ({len(embeddings_dict):,} embeddings)")
        
        # Prepare data
        node_indices = list(embeddings_dict.keys())
        embeddings_matrix = np.array([embeddings_dict[nid] for nid in node_indices], dtype=np.float32)
        
        logger.info(f"  📏 Embedding dimension: {embeddings_matrix.shape[1]}")
        logger.info(f"  📥 Adding {len(node_indices):,} embeddings to index...")
        
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
        
        # Save index (using the same name as the current institution embedding)
        index_path = self.output_dir / f"institution_embedding_hnsw.faiss"
        faiss.write_index(hnsw_index, str(index_path))
        
        # Save node_index mapping  
        mapping_path = self.output_dir / f"institution_embedding_mapping.pkl"
        with open(mapping_path, 'wb') as f:
            pickle.dump(node_indices, f)
        
        logger.info(f"  ✅ Index built and saved in {build_time:.2f}s")
        logger.info(f"     Index: {index_path}")
        logger.info(f"     Mapping: {mapping_path}")
        
        # Clear memory
        del embeddings_matrix
        gc.collect()
        
        return str(index_path)
    
    def build_all_institution_indices(self):
        """Build HNSW indices for all institution features"""
        logger.info("🏛️ STARTING INSTITUTION HNSW INDEX BUILDING")
        logger.info("=" * 60)
        
        total_start_time = time.time()
        
        # Step 1: Load institution nodes
        institution_nodes = self.load_institution_nodes()
        
        # Step 2: Prepare texts for embedding
        institution_texts = self.prepare_institution_texts(institution_nodes)
        
        # Step 3: Generate embeddings
        institution_embeddings = self.generate_institution_embeddings(institution_texts)
        
        # Step 4: Build HNSW indices for each feature
        logger.info("\n🔨 Building HNSW indices...")
        
        built_indices = {}
        
        for i, feature_name in enumerate(self.institution_features):
            logger.info(f"\n--- Building index {i+1}/{len(self.institution_features)}: {feature_name} ---")
            
            # Map feature name to text data key
            text_key = feature_name.replace('_embedding', '')
            if text_key not in institution_embeddings:
                logger.warning(f"⚠️  No embeddings found for {text_key}, skipping")
                continue
            
            embeddings_dict = institution_embeddings[text_key]
            index_path = self.build_hnsw_index(text_key, embeddings_dict)
            
            if index_path:
                built_indices[feature_name] = {
                    'index_path': index_path,
                    'mapping_path': str(self.output_dir / f"institution_embedding_mapping.pkl"),
                    'count': len(embeddings_dict)
                }
        
        # Step 5: Save master manifest
        manifest = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_time_seconds': time.time() - total_start_time,
            'embedding_dimension': self.embedding_dim,
            'hnsw_parameters': self.hnsw_params,
            'statistics': dict(self.stats),
            'indices': built_indices,
            'institution_features': self.institution_features
        }
        
        manifest_path = self.output_dir / 'institution_hnsw_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        total_time = time.time() - total_start_time
        logger.info("\n🎉 ALL INSTITUTION HNSW INDICES BUILT SUCCESSFULLY!")
        logger.info("=" * 50)
        logger.info(f"⏱️  Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"📄 Manifest: {manifest_path}")
        logger.info(f"🔍 Built {len(built_indices)} institution HNSW indices")
        
        return manifest_path

def main():
    """Main function to build institution HNSW indices"""
    
    # Configuration
    institution_nodes_file = "/shared/khoja/CogComp/agent/institution_nodes.json"
    output_dir = "/shared/khoja/CogComp/output/mag_hnsw_indices"
    
    logger.info("🏛️ INSTITUTION HNSW INDEX BUILDER")
    logger.info("💡 Building HNSW indices for institution nodes")
    logger.info("")
    logger.info(f"📥 Input: {institution_nodes_file}")
    logger.info(f"📤 Output: {output_dir}")
    logger.info("")
    
    # Verify input file exists
    if not Path(institution_nodes_file).exists():
        logger.error(f"❌ Institution nodes file not found: {institution_nodes_file}")
        return False
    
    # Build indices
    try:
        builder = InstitutionHNSWBuilder(institution_nodes_file, output_dir)
        manifest_path = builder.build_all_institution_indices()
        
        logger.info("\n🎯 NEXT STEPS:")
        logger.info("1. Review the manifest file to verify institution indices were built")
        logger.info("2. Update HNSW manager to load institution manifest")
        logger.info("3. Test institution search with sample queries")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to build institution HNSW indices: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
