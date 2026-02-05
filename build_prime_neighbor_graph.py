#!/usr/bin/env python3
"""
Phase 3: Build PRIME Neighbor Graph

This script finds neighbors for each node using all feature HNSW indices:
1. For each object_id, query all relevant HNSW indices to find top-K neighbors
2. Store similarity scores along with neighbor IDs  
3. Handle cross-entity type connections (gene ↔ disease, gene ↔ drug, etc.)
4. Save the complete neighbor graph

Cross-Entity Connections Strategy:
- Gene → Disease: gene_summary → disease_definition (molecular basis)
- Gene → Drug: gene_summary → drug_mechanism (drug targets)
- Disease → Drug: disease_definition → drug_indication (treatments)
- Disease → Gene: disease_clinical → gene_summary (genetic causes)
- Drug → Gene: drug_mechanism → gene_summary (drug targets)
- Pathway → All: pathway_summation → all entity features (pathway participation)
- All → Entity Names: cross-type discovery via entity_name_embedding
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

class PRIMENeighborGraphBuilder:
    """Build neighbor graph using HNSW indices with cross-entity connections"""
    
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
        self.k_neighbors = 1000  # Number of neighbors to find
        self.ef_search = 200     # Search width (should be >= k_neighbors)
        
        # Define cross-entity connection strategies
        self.cross_entity_connections = {
            # Gene connections
            'gene/protein': {
                'to_disease': [('gene_summary_embedding', 'disease_definition_embedding')],
                'to_drug': [('gene_summary_embedding', 'drug_mechanism_embedding')],
                'to_pathway': [('gene_summary_embedding', 'pathway_summation_embedding')]
            },
            # Disease connections
            'disease': {
                'to_gene': [('disease_definition_embedding', 'gene_summary_embedding'),
                           ('disease_clinical_embedding', 'gene_summary_embedding')],
                'to_drug': [('disease_definition_embedding', 'drug_indication_embedding'),
                           ('disease_symptoms_embedding', 'drug_indication_embedding')],
                'to_pathway': [('disease_definition_embedding', 'pathway_summation_embedding')]
            },
            # Drug connections
            'drug': {
                'to_gene': [('drug_mechanism_embedding', 'gene_summary_embedding')],
                'to_disease': [('drug_indication_embedding', 'disease_definition_embedding')],
                'to_pathway': [('drug_mechanism_embedding', 'pathway_summation_embedding')]
            },
            # Pathway connections
            'pathway': {
                'to_gene': [('pathway_summation_embedding', 'gene_summary_embedding')],
                'to_disease': [('pathway_summation_embedding', 'disease_definition_embedding')],
                'to_drug': [('pathway_summation_embedding', 'drug_mechanism_embedding')]
            }
        }
        
        # Statistics
        self.stats = {
            'total_nodes_processed': 0,
            'entity_type_counts': defaultdict(int),
            'total_connections': 0,
            'cross_type_connections': defaultdict(int),
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
                if idx == -1:  # Invalid index
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
        Find all neighbors for a single node (same-type and cross-type)
        Returns: {feature_name: [(neighbor_id, similarity_score), ...]}
        """
        object_id = node['object_id']
        node_type = node['node_type']
        all_neighbors = {}
        
        # Get same-type neighbors using available features
        same_type_features = self._get_features_for_type(node_type)
        
        for feature in same_type_features:
            query_embedding = self.get_node_embedding(node, feature)
            if query_embedding is not None:
                neighbors = self.query_hnsw_neighbors(feature, query_embedding, exclude_id=object_id)
                if neighbors:
                    all_neighbors[feature] = neighbors
                    self.stats['feature_query_counts'][feature] += 1
        
        # Get cross-type neighbors
        if node_type in self.cross_entity_connections:
            cross_type_neighbors = self._find_cross_type_neighbors(node, node_type)
            all_neighbors.update(cross_type_neighbors)
        
        # Count total connections
        total_connections = sum(len(neighbors) for neighbors in all_neighbors.values())
        self.stats['total_connections'] += total_connections
        
        return all_neighbors
    
    def _get_features_for_type(self, node_type: str) -> List[str]:
        """Get available features for a specific node type"""
        common_features = ['content_embedding', 'entity_name_embedding']
        
        type_specific = {
            'gene/protein': ['gene_summary_embedding', 'gene_full_name_embedding', 'gene_alias_embedding'],
            'disease': ['disease_definition_embedding', 'disease_clinical_embedding', 'disease_symptoms_embedding'],
            'drug': ['drug_description_embedding', 'drug_indication_embedding', 'drug_mechanism_embedding'],
            'pathway': ['pathway_summation_embedding', 'pathway_go_terms_embedding']
        }
        
        features = common_features.copy()
        if node_type in type_specific:
            features.extend(type_specific[node_type])
        
        return features
    
    def _find_cross_type_neighbors(self, node: Dict, node_type: str) -> Dict[str, List[Tuple[int, float]]]:
        """Find cross-entity type neighbors"""
        cross_neighbors = {}
        
        connections = self.cross_entity_connections.get(node_type, {})
        
        for target_type, feature_pairs in connections.items():
            for source_feature, target_feature in feature_pairs:
                # Get query embedding from source feature
                query_embedding = self.get_node_embedding(node, source_feature)
                
                if query_embedding is not None and target_feature in self.indices:
                    neighbors = self.query_hnsw_neighbors(target_feature, query_embedding)
                    
                    if neighbors:
                        connection_key = f"cross_{target_type}_{source_feature}_to_{target_feature}"
                        cross_neighbors[connection_key] = neighbors
                        self.stats['cross_type_connections'][connection_key] += len(neighbors)
                        self.stats['feature_query_counts'][connection_key] += 1
        
        return cross_neighbors
    
    def process_all_chunks(self):
        """Process all chunks to build the complete neighbor graph"""
        logger.info("🔍 Processing all chunks to build neighbor graph...")
        
        chunk_files = sorted(self.embeddings_dir.glob("chunk_*_embeddings.json"))
        logger.info(f"📁 Found {len(chunk_files)} chunk files")
        
        all_neighbors = {}
        
        for i, chunk_file in enumerate(chunk_files):
            logger.info(f"📖 Processing {chunk_file.name} ({i+1}/{len(chunk_files)})")
            
            with open(chunk_file, 'r') as f:
                chunk_data = json.load(f)
            
            chunk_neighbors = {}
            
            for j, node in enumerate(chunk_data):
                if j % 5000 == 0 and j > 0:
                    logger.info(f"    Processed {j:,}/{len(chunk_data):,} nodes")
                
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
                self.stats['entity_type_counts'][node_type] += 1
            
            # Merge with all_neighbors
            all_neighbors.update(chunk_neighbors)
            
            logger.info(f"  💾 Processed {len(chunk_neighbors):,} nodes from chunk")
            
            # Clear memory
            del chunk_data, chunk_neighbors
            gc.collect()
        
        return all_neighbors
    
    def save_final_graph(self, all_neighbors: Dict):
        """Save the complete neighbor graph and statistics"""
        logger.info("💾 Saving final neighbor graph...")
        
        # Save complete graph
        graph_path = self.output_dir / 'prime_complete_neighbor_graph.json'
        with open(graph_path, 'w') as f:
            json.dump(all_neighbors, f, indent=2)
        
        # Save statistics and manifest
        final_stats = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_nodes': len(all_neighbors),
            'k_neighbors': self.k_neighbors,
            'ef_search': self.ef_search,
            'statistics': {
                'total_nodes_processed': self.stats['total_nodes_processed'],
                'entity_type_counts': dict(self.stats['entity_type_counts']),
                'total_connections': self.stats['total_connections'],
                'cross_type_connections': dict(self.stats['cross_type_connections']),
                'feature_query_counts': dict(self.stats['feature_query_counts'])
            },
            'cross_entity_strategies': self.cross_entity_connections,
            'hnsw_manifest_used': self.manifest
        }
        
        stats_path = self.output_dir / 'neighbor_graph_manifest.json'
        with open(stats_path, 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        logger.info(f"📁 Complete graph: {graph_path}")
        logger.info(f"📄 Manifest: {stats_path}")
        
        return graph_path, stats_path
    
    def _log_final_statistics(self):
        """Log final statistics"""
        logger.info("\n📊 NEIGHBOR GRAPH STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total nodes processed: {self.stats['total_nodes_processed']:,}")
        
        logger.info("\n📈 Entity Type Distribution:")
        for entity_type in sorted(self.stats['entity_type_counts'].keys()):
            count = self.stats['entity_type_counts'][entity_type]
            logger.info(f"  {entity_type:20}: {count:8,}")
        
        logger.info(f"\nTotal connections: {self.stats['total_connections']:,}")
        
        logger.info("\n🔗 Cross-Type Connections:")
        total_cross = sum(self.stats['cross_type_connections'].values())
        logger.info(f"  Total cross-type: {total_cross:,}")
        
        # Group by target type
        cross_by_type = defaultdict(int)
        for key, count in self.stats['cross_type_connections'].items():
            target_type = key.split('_')[1]  # Extract target type from key
            cross_by_type[target_type] += count
        
        for target_type in sorted(cross_by_type.keys()):
            logger.info(f"  → {target_type:15}: {cross_by_type[target_type]:8,} connections")
        
        logger.info("\n📈 Top Feature Queries:")
        sorted_features = sorted(self.stats['feature_query_counts'].items(), 
                                key=lambda x: x[1], reverse=True)[:15]
        for feature, count in sorted_features:
            logger.info(f"  {feature:45}: {count:8,} queries")
    
    def build_neighbor_graph(self):
        """Main function to build the complete neighbor graph"""
        logger.info("🚀 STARTING PRIME NEIGHBOR GRAPH BUILDING")
        logger.info("=" * 60)
        logger.info(f"🎯 Target: {self.k_neighbors} neighbors per node")
        logger.info(f"🔍 Search width: {self.ef_search}")
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
            
            logger.info("\n🎉 NEIGHBOR GRAPH BUILT SUCCESSFULLY!")
            logger.info("=" * 50)
            logger.info(f"⏱️  Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
            logger.info(f"📁 Output directory: {self.output_dir}")
            logger.info(f"🔗 Graph file: {graph_path}")
            logger.info(f"📄 Statistics: {stats_path}")
            
            return graph_path
            
        except Exception as e:
            logger.error(f"❌ Failed to build neighbor graph: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main function to build neighbor graph"""
    
    # Configuration
    embeddings_dir = "/shared/khoja/CogComp/output/prime_pipeline_cache/embeddings"
    hnsw_dir = "/shared/khoja/CogComp/output/prime_hnsw_indices"
    output_dir = "/shared/khoja/CogComp/output/prime_neighbor_graph"
    
    logger.info("🧬 PRIME NEIGHBOR GRAPH BUILDER")
    logger.info("💡 Finding neighbors using HNSW indices with cross-entity connections")
    logger.info("")
    logger.info(f"📥 Embeddings: {embeddings_dir}")
    logger.info(f"🔍 HNSW indices: {hnsw_dir}")
    logger.info(f"📤 Output: {output_dir}")
    logger.info("")
    
    # Verify input directories exist
    for path_name, path in [("Embeddings", embeddings_dir), ("HNSW", hnsw_dir)]:
        if not Path(path).exists():
            logger.error(f"❌ {path_name} directory not found: {path}")
            return False
    
    # Build neighbor graph
    try:
        builder = PRIMENeighborGraphBuilder(embeddings_dir, hnsw_dir, output_dir)
        graph_path = builder.build_neighbor_graph()
        
        if graph_path:
            logger.info("\n🎯 NEXT STEPS:")
            logger.info("1. Review the neighbor graph manifest")
            logger.info("2. Test graph connectivity and quality")
            logger.info("3. Implement retrieval pipeline (Phase 4)")
            return True
        else:
            return False
        
    except Exception as e:
        logger.error(f"❌ Failed to build neighbor graph: {e}")
        import traceback
        traceback.print_exc()
        return False
 
if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

