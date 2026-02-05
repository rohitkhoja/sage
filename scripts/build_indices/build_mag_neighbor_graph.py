#!/usr/bin/env python3
"""
Phase 2: Build MAG Neighbor Graph

This script finds neighbors for each node using all feature HNSW indices:
1. For each object_id, query all relevant HNSW indices to find 1000 neighbors
2. Store similarity scores along with neighbor IDs 
3. Handle cross-type connections (papers authors)
4. Save the complete neighbor graph

Cross-type connections:
- Papers query display_name HNSW (to find author neighbors via authors field)
- Authors query authors HNSW (to find paper neighbors via display_name)
"""

import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
import faiss
import gc
from loguru import logger

class MAGNeighborGraphBuilder:
    """Build neighbor graph using HNSW indices"""
    
    def __init__(self, embeddings_dir: str, hnsw_dir: str, output_dir: str):
        self.embeddings_dir = Path(embeddings_dir)
        self.hnsw_dir = Path(hnsw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load HNSW manifest
        self.manifest = self._load_manifest()
        self.indices = {}
        self.mappings = {}
        
        # Parameters
        self.k_neighbors = 1000 # Number of neighbors to find
        self.ef_search = 200 # Search width (should be >= k_neighbors)
        
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
        logger.info(" Loading HNSW indices...")
        
        for feature, info in self.manifest['indices'].items():
            logger.info(f" Loading {feature} ({info['count']:,} embeddings)")
            
            # Load FAISS HNSW index
            index = faiss.read_index(info['index_path'])
            
            # Set search parameters
            if hasattr(index, 'hnsw'):
                index.hnsw.efSearch = self.ef_search
            
            # Load object_id mapping
            with open(info['mapping_path'], 'rb') as f:
                object_ids = pickle.load(f)
            
            self.indices[feature] = index
            self.mappings[feature] = object_ids
            
        logger.info(f" Loaded {len(self.indices)} HNSW indices")
    
    def get_node_embedding(self, node: Dict, feature: str) -> Optional[np.ndarray]:
        """Get embedding for a specific feature from node"""
        if feature in node and node[feature]:
            embedding = node[feature]
            if len(embedding) == self.manifest['embedding_dimension']:
                return np.array(embedding, dtype=np.float32).reshape(1, -1)
        return None
    
    def query_hnsw_neighbors(self, feature: str, query_embedding: np.ndarray, 
                           exclude_id: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        Query FAISS HNSW index for neighbors
        Returns: List of (object_id, similarity_score) tuples
        """
        if feature not in self.indices:
            return []
        
        try:
            # Normalize query embedding for cosine similarity
            query_normalized = query_embedding.copy()
            faiss.normalize_L2(query_normalized)
            
            # Query the index
            k_search = min(self.k_neighbors + 10, self.indices[feature].ntotal)
            similarities, indices = self.indices[feature].search(query_normalized, k_search)
            
            # Convert to (object_id, similarity_score) pairs
            neighbors = []
            object_ids = self.mappings[feature]
            
            for idx, similarity in zip(indices[0], similarities[0]):
                if idx == -1: # Invalid index
                    continue
                    
                object_id = object_ids[idx]
                
                # Skip the query node itself
                if exclude_id is not None and object_id == exclude_id:
                    continue
                
                # FAISS returns cosine similarity directly (already normalized)
                similarity_score = max(0.0, float(similarity))
                neighbors.append((object_id, similarity_score))
            
            # Sort by similarity (highest first) and take top k
            neighbors.sort(key=lambda x: x[1], reverse=True)
            return neighbors[:self.k_neighbors]
            
        except Exception as e:
            logger.warning(f"Error querying {feature}: {e}")
            return []
    
    def find_node_neighbors(self, node: Dict) -> Dict[str, List[Tuple[int, float]]]:
        """
        Find all neighbors for a single node
        Returns: {feature_name: [(neighbor_id, similarity_score), ...]}
        """
        object_id = node['object_id']
        node_type = node['node_type']
        all_neighbors = {}
        
        if node_type == 'paper':
            # Paper features
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
            
            # Cross-type connection: Paper → Authors via display_name HNSW
            # Use the authors field embedding to query display_name HNSW
            authors_embedding = self.get_node_embedding(node, 'authors_embedding')
            if authors_embedding is not None and 'display_name_embedding' in self.indices:
                author_neighbors = self.query_hnsw_neighbors('display_name_embedding', authors_embedding)
                if author_neighbors:
                    all_neighbors['cross_type_authors'] = author_neighbors
                    self.stats['cross_type_connections'] += len(author_neighbors)
                    self.stats['feature_query_counts']['cross_type_authors'] += 1
                    
        elif node_type == 'author':
            # Author features
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
            
            # Cross-type connection: Author → Papers via authors HNSW 
            # Use display_name embedding to query authors HNSW
            display_name_embedding = self.get_node_embedding(node, 'display_name_embedding')
            if display_name_embedding is not None and 'authors_embedding' in self.indices:
                paper_neighbors = self.query_hnsw_neighbors('authors_embedding', display_name_embedding)
                if paper_neighbors:
                    all_neighbors['cross_type_papers'] = paper_neighbors
                    self.stats['cross_type_connections'] += len(paper_neighbors)
                    self.stats['feature_query_counts']['cross_type_papers'] += 1
        
        # Count total connections
        total_connections = sum(len(neighbors) for neighbors in all_neighbors.values())
        self.stats['total_connections'] += total_connections
        
        return all_neighbors
    
    def process_all_chunks(self):
        """Process all chunks to build the complete neighbor graph"""
        logger.info(" Processing all chunks to build neighbor graph...")
        
        chunk_files = sorted(self.embeddings_dir.glob("chunk_*.json"))
        logger.info(f" Found {len(chunk_files)} chunk files")
        
        for i, chunk_file in enumerate(chunk_files):
            # Skip already processed chunks
            chunk_output_path = self.output_dir / f"neighbors_{chunk_file.stem}.json"
            if chunk_output_path.exists():
                logger.info(f" Skipping {chunk_file.name} ({i+1}/{len(chunk_files)}) - already processed")
                continue
                
            logger.info(f" Processing {chunk_file.name} ({i+1}/{len(chunk_files)})")
            
            with open(chunk_file, 'r') as f:
                chunk_data = json.load(f)
            
            chunk_neighbors = {}
            
            for j, node in enumerate(chunk_data):
                if j % 10000 == 0 and j > 0:
                    logger.info(f" Processed {j:,}/{len(chunk_data):,} nodes")
                
                object_id = node['object_id']
                node_type = node['node_type']
                
                # Find neighbors for this node
                neighbors = self.find_node_neighbors(node)
                chunk_neighbors[object_id] = {
                    'node_type': node_type,
                    'neighbors': neighbors
                }
                
                # Update statistics
                self.stats['total_nodes_processed'] += 1
                if node_type == 'paper':
                    self.stats['paper_nodes_processed'] += 1
                elif node_type == 'author':
                    self.stats['author_nodes_processed'] += 1
            
            # Save chunk neighbors
            with open(chunk_output_path, 'w') as f:
                json.dump(chunk_neighbors, f, indent=2)
            
            logger.info(f" Saved neighbors for {len(chunk_neighbors):,} nodes")
            # Don't keep in memory - will load all later for final save
            del chunk_neighbors
            gc.collect()
        
        # Load all chunk neighbors at the end
        logger.info(" Loading all processed chunk neighbors for final save...")
        all_neighbors = {}
        for chunk_file in sorted(self.output_dir.glob("neighbors_chunk_*.json")):
            logger.info(f" Loading {chunk_file.name}...")
            with open(chunk_file, 'r') as f:
                chunk_data = json.load(f)
                all_neighbors.update(chunk_data)
        
        return all_neighbors
    
    def save_final_graph(self, all_neighbors: Dict):
        """Save the complete neighbor graph and statistics"""
        logger.info(" Saving final neighbor graph...")
        
        # Save complete graph
        graph_path = self.output_dir / 'mag_complete_neighbor_graph.json'
        with open(graph_path, 'w') as f:
            json.dump(all_neighbors, f, indent=2)
        
        # Save statistics and manifest
        final_stats = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_nodes': len(all_neighbors),
            'k_neighbors': self.k_neighbors,
            'ef_search': self.ef_search,
            'statistics': dict(self.stats),
            'feature_query_counts': dict(self.stats['feature_query_counts']),
            'hnsw_manifest_used': self.manifest
        }
        
        stats_path = self.output_dir / 'neighbor_graph_manifest.json'
        with open(stats_path, 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        logger.info(f" Complete graph: {graph_path}")
        logger.info(f" Manifest: {stats_path}")
        
        return graph_path, stats_path
    
    def _log_final_statistics(self):
        """Log final statistics"""
        logger.info("\n NEIGHBOR GRAPH STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total nodes processed: {self.stats['total_nodes_processed']:,}")
        logger.info(f"Paper nodes: {self.stats['paper_nodes_processed']:,}")
        logger.info(f"Author nodes: {self.stats['author_nodes_processed']:,}")
        logger.info(f"Total connections: {self.stats['total_connections']:,}")
        logger.info(f"Cross-type connections: {self.stats['cross_type_connections']:,}")
        logger.info("\n Feature Query Counts:")
        
        for feature in sorted(self.stats['feature_query_counts'].keys()):
            count = self.stats['feature_query_counts'][feature]
            logger.info(f" {feature:30}: {count:8,} queries")
    
    def build_neighbor_graph(self):
        """Main function to build the complete neighbor graph"""
        logger.info(" STARTING MAG NEIGHBOR GRAPH BUILDING")
        logger.info("=" * 60)
        logger.info(f" Target: {self.k_neighbors} neighbors per node")
        logger.info(f" Search width: {self.ef_search}")
        logger.info("")
        
        total_start_time = time.time()
        
        try:
            # Step 1: Load HNSW indices
            self.load_hnsw_indices()
            
            # Step 2: Process all chunks
            all_neighbors = self.process_all_chunks()
            
            # Step 3: Save final graph
            graph_path, stats_path = self.save_final_graph(all_neighbors)
            
            total_time = time.time() - total_start_time
            
            # Step 4: Log results
            self._log_final_statistics()
            
            logger.info("\n NEIGHBOR GRAPH BUILT SUCCESSFULLY!")
            logger.info("=" * 50)
            logger.info(f" Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
            logger.info(f" Output directory: {self.output_dir}")
            logger.info(f" Graph file: {graph_path}")
            logger.info(f" Statistics: {stats_path}")
            
            return graph_path
            
        except Exception as e:
            logger.error(f" Failed to build neighbor graph: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main function to build neighbor graph"""
    
    # Configuration
    embeddings_dir = "/shared/khoja/CogComp/output/mag_final_cache/embeddings"
    hnsw_dir = "/shared/khoja/CogComp/output/mag_hnsw_indices"
    output_dir = "/shared/khoja/CogComp/output/mag_neighbor_graph"
    
    logger.info(" MAG NEIGHBOR GRAPH BUILDER")
    logger.info(" Finding neighbors using HNSW indices for graph construction")
    logger.info("")
    logger.info(f" Embeddings: {embeddings_dir}")
    logger.info(f" HNSW indices: {hnsw_dir}")
    logger.info(f" Output: {output_dir}")
    logger.info("")
    
    # Verify input directories exist
    for path_name, path in [("Embeddings", embeddings_dir), ("HNSW", hnsw_dir)]:
        if not Path(path).exists():
            logger.error(f" {path_name} directory not found: {path}")
            return False
    
    # Build neighbor graph
    try:
        builder = MAGNeighborGraphBuilder(embeddings_dir, hnsw_dir, output_dir)
        graph_path = builder.build_neighbor_graph()
        
        if graph_path:
            logger.info("\n NEXT STEPS:")
            logger.info("1. Review the neighbor graph manifest")
            logger.info("2. Test graph connectivity and quality")
            logger.info("3. Use the graph for downstream tasks")
            return True
        else:
            return False
        
    except Exception as e:
        logger.error(f" Failed to build neighbor graph: {e}")
        import traceback
        traceback.print_exc()
        return False
 
if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)