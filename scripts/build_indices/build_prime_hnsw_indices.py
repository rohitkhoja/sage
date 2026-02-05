#!/usr/bin/env python3
"""
Phase 2: Build HNSW Indices for PRIME Dataset Features

This script builds separate HNSW indices for each embedding feature:
- Common features: content, entity_name
- Gene/Protein features: gene_summary, gene_full_name, gene_alias
- Disease features: disease_definition, disease_clinical, disease_symptoms
- Drug features: drug_description, drug_indication, drug_mechanism
- Pathway features: pathway_summation, pathway_go_terms

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

class PRIMEHNSWBuilder:
    """Build HNSW indices for all PRIME features"""
    
    def __init__(self, embeddings_dir: str, output_dir: str):
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature definitions based on PRIME entity types
        self.common_features = [
            'content_embedding',
            'entity_name_embedding'
        ]
        
        self.gene_features = [
            'gene_summary_embedding',
            'gene_full_name_embedding',
            'gene_alias_embedding'
        ]
        
        self.disease_features = [
            'disease_definition_embedding',
            'disease_clinical_embedding',
            'disease_symptoms_embedding'
        ]
        
        self.drug_features = [
            'drug_description_embedding',
            'drug_indication_embedding',
            'drug_mechanism_embedding'
        ]
        
        self.pathway_features = [
            'pathway_summation_embedding',
            'pathway_go_terms_embedding'
        ]
        
        # All features combined
        self.all_features = (
            self.common_features + 
            self.gene_features + 
            self.disease_features + 
            self.drug_features + 
            self.pathway_features
        )
        
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
            'entity_type_counts': defaultdict(int),
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
            feature: {} for feature in self.all_features
        }
        
        chunk_files = sorted(self.embeddings_dir.glob("chunk_*_embeddings.json"))
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
                
                # Count entity types
                self.stats['entity_type_counts'][node_type] += 1
                
                # Extract embeddings for all features
                for feature in self.all_features:
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
        
        logger.info("\n Entity Type Distribution:")
        for entity_type in sorted(self.stats['entity_type_counts'].keys()):
            count = self.stats['entity_type_counts'][entity_type]
            logger.info(f" {entity_type:20}: {count:8,}")
        
        logger.info("\n Feature Embedding Counts:")
        
        # Group by category
        feature_groups = [
            ("Common Features", self.common_features),
            ("Gene/Protein Features", self.gene_features),
            ("Disease Features", self.disease_features),
            ("Drug Features", self.drug_features),
            ("Pathway Features", self.pathway_features)
        ]
        
        for group_name, features in feature_groups:
            logger.info(f"\n {group_name}:")
            for feature in features:
                if feature in self.stats['feature_counts']:
                    valid_count = self.stats['feature_counts'][feature]
                    empty_count = self.stats['empty_feature_counts'][feature]
                    total = valid_count + empty_count
                    pct = (valid_count / total * 100) if total > 0 else 0
                    logger.info(f" {feature:30}: {valid_count:8,} valid / {total:8,} total ({pct:5.1f}%)")
    
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
        logger.info(" STARTING PRIME HNSW INDEX BUILDING")
        logger.info("=" * 60)
        
        total_start_time = time.time()
        
        # Step 1: Scan all chunks and collect embeddings
        feature_embeddings = self.scan_all_chunks()
        
        # Step 2: Build HNSW indices for each feature
        logger.info("\n Building HNSW indices...")
        
        built_indices = {}
        
        for i, feature in enumerate(self.all_features):
            logger.info(f"\n--- Building index {i+1}/{len(self.all_features)}: {feature} ---")
            
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
            'statistics': {
                'total_chunks': self.stats['total_chunks'],
                'total_nodes': self.stats['total_nodes'],
                'entity_type_counts': dict(self.stats['entity_type_counts']),
                'feature_counts': dict(self.stats['feature_counts']),
                'empty_feature_counts': dict(self.stats['empty_feature_counts'])
            },
            'indices': built_indices,
            'feature_categories': {
                'common_features': self.common_features,
                'gene_features': self.gene_features,
                'disease_features': self.disease_features,
                'drug_features': self.drug_features,
                'pathway_features': self.pathway_features
            }
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
        
        # Summary by category
        logger.info("\n Indices Built by Category:")
        for group_name, features in [
            ("Common", self.common_features),
            ("Gene/Protein", self.gene_features),
            ("Disease", self.disease_features),
            ("Drug", self.drug_features),
            ("Pathway", self.pathway_features)
        ]:
            count = sum(1 for f in features if f in built_indices)
            if count > 0:
                logger.info(f" {group_name:15}: {count} indices")
        
        return manifest_path

def main():
    """Main function to build all HNSW indices"""
    
    # Configuration
    embeddings_dir = "/shared/khoja/CogComp/output/prime_pipeline_cache/embeddings"
    output_dir = "/shared/khoja/CogComp/output/prime_hnsw_indices"
    
    logger.info(" PRIME HNSW INDEX BUILDER")
    logger.info(" Building separate HNSW indices for each embedding feature")
    logger.info("")
    logger.info(f" Input: {embeddings_dir}")
    logger.info(f" Output: {output_dir}")
    logger.info("")
    
    # Verify input directory exists
    if not Path(embeddings_dir).exists():
        logger.error(f" Embeddings directory not found: {embeddings_dir}")
        return False
    
    # Check for embedding files
    chunk_files = list(Path(embeddings_dir).glob("chunk_*_embeddings.json"))
    if not chunk_files:
        logger.error(f" No embedding chunk files found in {embeddings_dir}")
        logger.error(f" Make sure phase 1 (embedding generation) is complete")
        return False
    
    logger.info(f" Found {len(chunk_files)} embedding chunk files")
    
    # Build indices
    try:
        builder = PRIMEHNSWBuilder(embeddings_dir, output_dir)
        manifest_path = builder.build_all_indices()
        
        logger.info("\n NEXT STEPS:")
        logger.info("1. Review the manifest file to verify all indices were built")
        logger.info("2. Run the neighbor finding script (Phase 3) to build the graph")
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

