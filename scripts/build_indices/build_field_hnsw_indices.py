#!/usr/bin/env python3
"""
Build HNSW Indices for Field of Study Nodes

This script generates embeddings and builds HNSW indices for field of study nodes:
- Field features: display_name, level_context, paper_context
- Uses the same embedding service as the main MAG pipeline
- Creates separate HNSW indices for different field features

Input: Field nodes from node_info.jsonl
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.processors.embedding_service import EmbeddingService
from src.core.models import ProcessingConfig

class FieldHNSWBuilder:
    """Build HNSW indices for field of study nodes"""
    
    def __init__(self, field_nodes_file: str, output_dir: str):
        self.field_nodes_file = Path(field_nodes_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Field feature definitions
        self.field_features = [
            'display_name_embedding',
            'level_context_embedding', 
            'paper_context_embedding',
            'combined_embedding'
        ]
        
        # HNSW parameters (using FAISS)
        self.embedding_dim = 384 # MiniLM-L6-v2 dimension
        self.hnsw_params = {
            'M': 64, # Number of connections per element 
            'ef_construction': 2000, # Higher quality construction
            'ef_search': 1000, # Higher quality search
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
            'total_field_nodes': 0,
            'processed_nodes': 0,
            'feature_counts': defaultdict(int),
            'empty_feature_counts': defaultdict(int)
        }
        
    def load_field_nodes(self) -> List[Dict]:
        """Load field of study nodes from JSON file"""
        logger.info(f" Loading field nodes from {self.field_nodes_file}")
        
        with open(self.field_nodes_file, 'r') as f:
            field_nodes = json.load(f)
        
        self.stats['total_field_nodes'] = len(field_nodes)
        logger.info(f" Loaded {len(field_nodes):,} field nodes")
        
        return field_nodes
    
    def prepare_field_texts(self, field_nodes: List[Dict]) -> Dict[str, Dict[str, List]]:
        """Prepare texts from field nodes for embedding generation"""
        logger.info(" Preparing field texts for embedding generation...")
        
        all_text_data = {
            'display_name': {'texts': [], 'node_indices': []},
            'level_context': {'texts': [], 'node_indices': []},
            'paper_context': {'texts': [], 'node_indices': []},
            'combined': {'texts': [], 'node_indices': []}
        }
        
        for node_idx, node in enumerate(field_nodes):
            node_index = node['node_index']
            display_name = node.get('DisplayName', '').strip()
            level = node.get('Level', '')
            paper_count = node.get('PaperCount', 0)
            citation_count = node.get('CitationCount', 0)
            
            if not display_name:
                continue
                
            # 1. Display name (primary field name)
            all_text_data['display_name']['texts'].append(display_name)
            all_text_data['display_name']['node_indices'].append(node_index)
            
            # 2. Level context (field name + level information)
            level_context = f"{display_name}"
            if level != -1 and level != '':
                level_context += f" (Level {level})"
            all_text_data['level_context']['texts'].append(level_context)
            all_text_data['level_context']['node_indices'].append(node_index)
            
            # 3. Paper context (field name + paper/citation info)
            paper_context = f"{display_name}"
            if paper_count > 0:
                paper_context += f" ({paper_count} papers"
                if citation_count > 0:
                    paper_context += f", {citation_count} citations"
                paper_context += ")"
            all_text_data['paper_context']['texts'].append(paper_context)
            all_text_data['paper_context']['node_indices'].append(node_index)
            
            # 4. Combined context (all information)
            combined_context = f"{display_name}"
            if level != -1 and level != '':
                combined_context += f" (Level {level})"
            if paper_count > 0:
                combined_context += f" - {paper_count} papers"
                if citation_count > 0:
                    combined_context += f", {citation_count} citations"
            all_text_data['combined']['texts'].append(combined_context)
            all_text_data['combined']['node_indices'].append(node_index)
            
            self.stats['processed_nodes'] += 1
        
        # Log statistics
        for feature, data in all_text_data.items():
            count = len(data['texts'])
            self.stats['feature_counts'][feature] = count
            logger.info(f" {feature}: {count:,} texts prepared")
        
        return all_text_data
    
    def generate_field_embeddings(self, field_texts: Dict[str, Dict[str, List]]) -> Dict[str, Dict[int, np.ndarray]]:
        """Generate embeddings for all field features"""
        logger.info(" Generating embeddings for field features...")
        
        field_embeddings = {}
        
        for feature_name, data in field_texts.items():
            if not data['texts']:
                logger.warning(f" No texts found for {feature_name}, skipping")
                continue
            
            logger.info(f" Processing {feature_name}: {len(data['texts']):,} texts")
            feature_start = time.time()
            
            try:
                # Generate embeddings using the embedding service
                embeddings = self.embedding_service.generate_embeddings_bulk(data['texts'])
                
                # Create mapping from node_index to embedding
                node_embeddings = {}
                for node_index, embedding in zip(data['node_indices'], embeddings):
                    node_embeddings[node_index] = np.array(embedding, dtype=np.float32)
                
                field_embeddings[feature_name] = node_embeddings
                
                feature_time = time.time() - feature_start
                logger.info(f" {feature_name} completed in {feature_time:.2f}s")
                
                # Clear memory
                del embeddings
                gc.collect()
                
            except Exception as e:
                logger.error(f" Failed to process {feature_name}: {e}")
                continue
        
        return field_embeddings
    
    def build_hnsw_index(self, feature_name: str, embeddings_dict: Dict[int, np.ndarray]) -> str:
        """Build HNSW index for a specific field feature using FAISS"""
        if not embeddings_dict:
            logger.warning(f" No embeddings found for {feature_name}, skipping")
            return None
            
        logger.info(f" Building HNSW index for {feature_name} ({len(embeddings_dict):,} embeddings)")
        
        # Prepare data
        node_indices = list(embeddings_dict.keys())
        embeddings_matrix = np.array([embeddings_dict[nid] for nid in node_indices], dtype=np.float32)
        
        logger.info(f" Embedding dimension: {embeddings_matrix.shape[1]}")
        logger.info(f" Adding {len(node_indices):,} embeddings to index...")
        
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
        index_path = self.output_dir / f"field_{feature_name}_hnsw.faiss"
        faiss.write_index(hnsw_index, str(index_path))
        
        # Save node_index mapping 
        mapping_path = self.output_dir / f"field_{feature_name}_mapping.pkl"
        with open(mapping_path, 'wb') as f:
            pickle.dump(node_indices, f)
        
        logger.info(f" Index built and saved in {build_time:.2f}s")
        logger.info(f" Index: {index_path}")
        logger.info(f" Mapping: {mapping_path}")
        
        # Clear memory
        del embeddings_matrix
        gc.collect()
        
        return str(index_path)
    
    def build_all_field_indices(self):
        """Build HNSW indices for all field features"""
        logger.info(" STARTING FIELD OF STUDY HNSW INDEX BUILDING")
        logger.info("=" * 60)
        
        total_start_time = time.time()
        
        # Step 1: Load field nodes
        field_nodes = self.load_field_nodes()
        
        # Step 2: Prepare texts for embedding
        field_texts = self.prepare_field_texts(field_nodes)
        
        # Step 3: Generate embeddings
        field_embeddings = self.generate_field_embeddings(field_texts)
        
        # Step 4: Build HNSW indices for each feature
        logger.info("\n Building HNSW indices...")
        
        built_indices = {}
        
        for i, feature_name in enumerate(self.field_features):
            logger.info(f"\n--- Building index {i+1}/{len(self.field_features)}: field_{feature_name} ---")
            
            # Map feature name to text data key
            text_key = feature_name.replace('_embedding', '')
            if text_key not in field_embeddings:
                logger.warning(f" No embeddings found for {text_key}, skipping")
                continue
            
            embeddings_dict = field_embeddings[text_key]
            index_path = self.build_hnsw_index(text_key, embeddings_dict)
            
            if index_path:
                built_indices[feature_name] = {
                    'index_path': index_path,
                    'mapping_path': str(self.output_dir / f"field_{text_key}_mapping.pkl"),
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
            'field_features': self.field_features
        }
        
        manifest_path = self.output_dir / 'field_hnsw_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        total_time = time.time() - total_start_time
        logger.info("\n ALL FIELD HNSW INDICES BUILT SUCCESSFULLY!")
        logger.info("=" * 50)
        logger.info(f" Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        logger.info(f" Output directory: {self.output_dir}")
        logger.info(f" Manifest: {manifest_path}")
        logger.info(f" Built {len(built_indices)} field HNSW indices")
        
        return manifest_path

def main():
    """Main function to build field HNSW indices"""
    
    # Configuration
    field_nodes_file = "/shared/khoja/CogComp/agent/field_of_study_nodes.json"
    output_dir = "/shared/khoja/CogComp/output/mag_hnsw_indices"
    
    logger.info(" FIELD OF STUDY HNSW INDEX BUILDER")
    logger.info(" Building HNSW indices for field of study nodes")
    logger.info("")
    logger.info(f" Input: {field_nodes_file}")
    logger.info(f" Output: {output_dir}")
    logger.info("")
    
    # Verify input file exists
    if not Path(field_nodes_file).exists():
        logger.error(f" Field nodes file not found: {field_nodes_file}")
        return False
    
    # Build indices
    try:
        builder = FieldHNSWBuilder(field_nodes_file, output_dir)
        manifest_path = builder.build_all_field_indices()
        
        logger.info("\n NEXT STEPS:")
        logger.info("1. Review the manifest file to verify all field indices were built")
        logger.info("2. Integrate field search functionality into MAG agent")
        logger.info("3. Test field search with sample queries")
        
        return True
        
    except Exception as e:
        logger.error(f" Failed to build field HNSW indices: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
