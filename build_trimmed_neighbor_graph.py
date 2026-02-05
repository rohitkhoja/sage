#!/usr/bin/env python3
"""
Build Trimmed MAG Neighbor Graph (200 neighbors per feature, compact JSON)

Optimized version that:
- Saves only 200 neighbors per feature (instead of 1000)
- Saves compact JSON (no indent, much smaller files)
- Processes ALL chunks (0-19)
"""#

import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import faiss
import gc
from loguru import logger

class TrimmedNeighborGraphBuilder:
    """Build trimmed neighbor graph using HNSW indices"""
    
    def __init__(self, embeddings_dir: str, hnsw_dir: str, output_dir: str):
        self.embeddings_dir = Path(embeddings_dir)
        self.hnsw_dir = Path(hnsw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load HNSW manifest
        self.manifest = self._load_manifest()
        self.indices = {}
        self.mappings = {}
        
        # Parameters - TRIMMED TO 200!
        self.k_neighbors = 200  # ← Changed from 1000 to 200
        self.ef_search = 200
        
        # Statistics
        self.stats = {
            'total_nodes_processed': 0,
            'paper_nodes_processed': 0,
            'author_nodes_processed': 0,
            'total_connections': 0,
            'cross_type_connections': 0,
            'feature_query_counts': defaultdict(int)
        }
        
    def _load_manifest(self) -> Dict:
        """Load HNSW manifest"""
        manifest_path = self.hnsw_dir / 'hnsw_manifest.json'
        with open(manifest_path, 'r') as f:
            return json.load(f)
    
    def load_hnsw_indices(self):
        """Load all HNSW indices and mappings"""
        logger.info("📥 Loading HNSW indices...")
        
        for feature, info in self.manifest['indices'].items():
            logger.info(f"  🔍 Loading {feature} ({info['count']:,} embeddings)")
            
            index = faiss.read_index(info['index_path'])
            
            if hasattr(index, 'hnsw'):
                index.hnsw.efSearch = self.ef_search
            
            with open(info['mapping_path'], 'rb') as f:
                object_ids = pickle.load(f)
            
            self.indices[feature] = index
            self.mappings[feature] = object_ids
            
        logger.info(f"✅ Loaded {len(self.indices)} HNSW indices")
    
    def get_node_embedding(self, node: Dict, feature: str) -> Optional[np.ndarray]:
        """Get embedding for a specific feature from node"""
        if feature in node and node[feature]:
            embedding = node[feature]
            if len(embedding) == self.manifest['embedding_dimension']:
                return np.array(embedding, dtype=np.float32).reshape(1, -1)
        return None
    
    def query_hnsw_neighbors(self, feature: str, query_embedding: np.ndarray, 
                           exclude_id: Optional[int] = None) -> List[Tuple[int, float]]:
        """Query FAISS HNSW index for neighbors"""
        if feature not in self.indices:
            return []
        
        try:
            query_normalized = query_embedding.copy()
            faiss.normalize_L2(query_normalized)
            
            k_search = min(self.k_neighbors + 10, self.indices[feature].ntotal)
            similarities, indices = self.indices[feature].search(query_normalized, k_search)
            
            neighbors = []
            object_ids = self.mappings[feature]
            
            for idx, similarity in zip(indices[0], similarities[0]):
                if idx == -1:
                    continue
                    
                object_id = object_ids[idx]
                
                if exclude_id is not None and object_id == exclude_id:
                    continue
                
                similarity_score = max(0.0, float(similarity))
                neighbors.append((object_id, similarity_score))
            
            neighbors.sort(key=lambda x: x[1], reverse=True)
            return neighbors[:self.k_neighbors]  # Return top 200
            
        except Exception as e:
            logger.warning(f"Error querying {feature}: {e}")
            return []
    
    def find_node_neighbors(self, node: Dict) -> Dict[str, List[Tuple[int, float]]]:
        """Find neighbors for a single node"""
        object_id = node['object_id']
        node_type = node['node_type']
        all_neighbors = {}
        
        if node_type == 'paper':
            paper_features = [
                'content_embedding',
                'original_title_embedding', 
                'abstract_embedding',
                'authors_embedding',
                'fields_of_study_embedding',
                'cites_embedding'
            ]
            
            for feature in paper_features:
                query_embedding = self.get_node_embedding(node, feature)
                if query_embedding is not None:
                    neighbors = self.query_hnsw_neighbors(feature, query_embedding, exclude_id=object_id)
                    if neighbors:
                        all_neighbors[feature] = neighbors
                        self.stats['feature_query_counts'][feature] += 1
            
            # Cross-type: Paper → Authors
            authors_embedding = self.get_node_embedding(node, 'authors_embedding')
            if authors_embedding is not None and 'display_name_embedding' in self.indices:
                author_neighbors = self.query_hnsw_neighbors('display_name_embedding', authors_embedding)
                if author_neighbors:
                    all_neighbors['cross_type_authors'] = author_neighbors
                    self.stats['cross_type_connections'] += len(author_neighbors)
                    self.stats['feature_query_counts']['cross_type_authors'] += 1
                    
        elif node_type == 'author':
            author_features = [
                'content_embedding',
                'display_name_embedding',
                'institution_embedding'
            ]
            
            for feature in author_features:
                query_embedding = self.get_node_embedding(node, feature)
                if query_embedding is not None:
                    neighbors = self.query_hnsw_neighbors(feature, query_embedding, exclude_id=object_id)
                    if neighbors:
                        all_neighbors[feature] = neighbors
                        self.stats['feature_query_counts'][feature] += 1
            
            # Cross-type: Author → Papers
            display_name_embedding = self.get_node_embedding(node, 'display_name_embedding')
            if display_name_embedding is not None and 'authors_embedding' in self.indices:
                paper_neighbors = self.query_hnsw_neighbors('authors_embedding', display_name_embedding)
                if paper_neighbors:
                    all_neighbors['cross_type_papers'] = paper_neighbors
                    self.stats['cross_type_connections'] += len(paper_neighbors)
                    self.stats['feature_query_counts']['cross_type_papers'] += 1
        
        total_connections = sum(len(neighbors) for neighbors in all_neighbors.values())
        self.stats['total_connections'] += total_connections
        
        return all_neighbors
    
    def process_all_chunks(self):
        """Process ALL chunks (0-19) to build trimmed neighbor graph"""
        logger.info("🔍 Processing ALL chunks to build trimmed neighbor graph...")
        
        chunk_files = sorted(self.embeddings_dir.glob("chunk_*.json"))
        logger.info(f"📁 Found {len(chunk_files)} chunk files")
        
        for i, chunk_file in enumerate(chunk_files):
            logger.info(f"📖 Processing {chunk_file.name} ({i+1}/{len(chunk_files)})")
            
            with open(chunk_file, 'r') as f:
                chunk_data = json.load(f)
            
            chunk_neighbors = {}
            
            for j, node in enumerate(chunk_data):
                if j % 10000 == 0 and j > 0:
                    logger.info(f"    Processed {j:,}/{len(chunk_data):,} nodes")
                
                object_id = node['object_id']
                node_type = node['node_type']
                
                neighbors = self.find_node_neighbors(node)
                chunk_neighbors[object_id] = {
                    'node_type': node_type,
                    'neighbors': neighbors
                }
                
                self.stats['total_nodes_processed'] += 1
                if node_type == 'paper':
                    self.stats['paper_nodes_processed'] += 1
                elif node_type == 'author':
                    self.stats['author_nodes_processed'] += 1
            
            # Save compact JSON (NO indent!)
            chunk_output_path = self.output_dir / f"neighbors_{chunk_file.stem}.json"
            with open(chunk_output_path, 'w') as f:
                json.dump(chunk_neighbors, f)  # ← COMPACT!
            
            logger.info(f"  💾 Saved neighbors for {len(chunk_neighbors):,} nodes (compact JSON)")
            
            del chunk_neighbors
            gc.collect()
        
        logger.info("✅ All chunks processed!")
    
    def save_manifest(self):
        """Save manifest with statistics"""
        manifest = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'k_neighbors': self.k_neighbors,
            'ef_search': self.ef_search,
            'statistics': dict(self.stats),
            'feature_query_counts': dict(self.stats['feature_query_counts'])
        }
        
        manifest_path = self.output_dir / 'trimmed_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"📄 Manifest saved: {manifest_path}")
    
    def build(self):
        """Main build function"""
        logger.info("🚀 STARTING TRIMMED NEIGHBOR GRAPH BUILDING")
        logger.info("=" * 60)
        logger.info(f"🎯 Neighbors per feature: {self.k_neighbors}")
        logger.info(f"💾 Output format: Compact JSON (no indent)")
        logger.info("")
        
        start_time = time.time()
        
        self.load_hnsw_indices()
        self.process_all_chunks()
        self.save_manifest()
        
        total_time = time.time() - start_time
        
        logger.info("\n📊 STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total nodes: {self.stats['total_nodes_processed']:,}")
        logger.info(f"Paper nodes: {self.stats['paper_nodes_processed']:,}")
        logger.info(f"Author nodes: {self.stats['author_nodes_processed']:,}")
        logger.info(f"Total connections: {self.stats['total_connections']:,}")
        logger.info(f"⏱️  Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        
        logger.info("\n✅ TRIMMED NEIGHBOR GRAPH COMPLETE!")

def main():
    embeddings_dir = "/shared/khoja/CogComp/output/mag_final_cache/embeddings"
    hnsw_dir = "/shared/khoja/CogComp/output/mag_hnsw_indices"
    output_dir = "/shared/khoja/CogComp/output/mag_neighbor_graph_trimmed"  # ← TRIMMED OUTPUT
    
    logger.info("🧬 TRIMMED MAG NEIGHBOR GRAPH BUILDER")
    logger.info("💡 Generating 200 neighbors per feature (compact JSON)")
    logger.info("")
    logger.info(f"📥 Embeddings: {embeddings_dir}")
    logger.info(f"🔍 HNSW indices: {hnsw_dir}")
    logger.info(f"📤 Output: {output_dir}")
    logger.info("")
    
    builder = TrimmedNeighborGraphBuilder(embeddings_dir, hnsw_dir, output_dir)
    builder.build()

if __name__ == "__main__":
    main()

