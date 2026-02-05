#!/usr/bin/env python3
"""
HNSW Manager for MAG Dataset
Loads and manages existing HNSW indices with MAG Object IDs as labels
"""

import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger


class MAGHNSWManager:
    """Manager for HNSW indices with node_index support"""
    
    def __init__(self, indices_dir: str, graph_loader=None, neo4j_driver=None):
        self.indices_dir = Path(indices_dir)
        self.indices: Dict[str, faiss.IndexHNSWFlat] = {}
        self.mappings: Dict[str, List[int]] = {}
        self.manifest: Optional[Dict[str, Any]] = None
        self.graph_loader = graph_loader  # For node type mappings only
        self.neo4j_driver = neo4j_driver  # For fetching node metadata from Neo4j
        
        # Statistics
        self.stats = {
            'loaded_indices': 0,
            'total_embeddings': 0,
            'available_features': []
        }
    
    def load_manifest(self):
        """Load the HNSW manifest file"""
        manifest_path = self.indices_dir / 'hnsw_manifest.json'
        field_manifest_path = self.indices_dir / 'field_hnsw_manifest.json'
        
        # Load main manifest
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                self.manifest = json.load(f)
            logger.info("📄 Loaded main HNSW manifest")
        else:
            logger.warning(f"⚠️  Main manifest not found at {manifest_path}")
            self.manifest = {'indices': {}}
        
        # Load field manifest if it exists
        if field_manifest_path.exists():
            with open(field_manifest_path, 'r') as f:
                field_manifest = json.load(f)
            
            # Merge only field_display_name_embedding into main manifest
            if 'indices' not in self.manifest:
                self.manifest['indices'] = {}
            
            # Only add field_display_name_embedding
            if 'display_name_embedding' in field_manifest.get('indices', {}):
                field_info = field_manifest['indices']['display_name_embedding']
                # Update the paths to match the actual file names
                field_info['index_path'] = str(self.indices_dir / 'field_display_name_hnsw.faiss')
                field_info['mapping_path'] = str(self.indices_dir / 'field_display_name_mapping.pkl')
                self.manifest['indices']['field_display_name_embedding'] = field_info
                logger.info("📄 Loaded field_display_name_embedding from field manifest")
        
        # Load institution manifest if it exists
        institution_manifest_path = self.indices_dir / 'institution_hnsw_manifest.json'
        if institution_manifest_path.exists():
            with open(institution_manifest_path, 'r') as f:
                institution_manifest = json.load(f)
            
            # Merge institution indices into main manifest
            if 'indices' not in self.manifest:
                self.manifest['indices'] = {}
            
            # Add institution_name_embedding (which maps to institution_embedding)
            if 'institution_name_embedding' in institution_manifest.get('indices', {}):
                institution_info = institution_manifest['indices']['institution_name_embedding']
                # Update the paths to match the actual file names
                institution_info['index_path'] = str(self.indices_dir / 'institution_embedding_hnsw.faiss')
                institution_info['mapping_path'] = str(self.indices_dir / 'institution_embedding_mapping.pkl')
                self.manifest['indices']['institution_embedding'] = institution_info
                logger.info("📄 Loaded institution_embedding from institution manifest")
        
        logger.info(f"📊 Available features: {list(self.manifest.get('indices', {}).keys())}")
        
        return True
    
    def load_index(self, feature_name: str) -> bool:
        """Load a specific HNSW index"""
        # Handle field_display_name_embedding with different naming convention
        if feature_name == 'field_display_name_embedding':
            index_path = self.indices_dir / f"field_display_name_hnsw.faiss"
            mapping_path = self.indices_dir / f"field_display_name_mapping.pkl"
        elif feature_name == 'institution_embedding':
            index_path = self.indices_dir / f"institution_embedding_hnsw.faiss"
            mapping_path = self.indices_dir / f"institution_embedding_mapping.pkl"
        else:
            index_path = self.indices_dir / f"{feature_name}_hnsw.faiss"
            mapping_path = self.indices_dir / f"{feature_name}_mapping.pkl"
        
        if not index_path.exists():
            logger.warning(f"⚠️  Index not found: {index_path}")
            return False
        
        if not mapping_path.exists():
            logger.warning(f"⚠️  Mapping not found: {mapping_path}")
            return False
        
        try:
            # Load FAISS index
            index = faiss.read_index(str(index_path))
            self.indices[feature_name] = index
            
            # Load node_index mapping
            with open(mapping_path, 'rb') as f:
                node_indices = pickle.load(f)
            self.mappings[feature_name] = node_indices
            
            self.stats['loaded_indices'] += 1
            self.stats['total_embeddings'] += len(node_indices)
            self.stats['available_features'].append(feature_name)
            
            logger.info(f"✅ Loaded {feature_name}: {len(node_indices):,} embeddings")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load {feature_name}: {e}")
            return False
    
    def load_all_indices(self):
        """Load all available HNSW indices"""
        logger.info("🔍 Loading all HNSW indices...")
        
        # Load manifest first
        if not self.load_manifest():
            # Fallback: scan directory for available indices
            logger.info("🔍 Scanning directory for available indices...")
            index_files = list(self.indices_dir.glob("*_hnsw.faiss"))
            
            for index_file in index_files:
                feature_name = index_file.stem.replace("_hnsw", "")
                self.load_index(feature_name)
        else:
            # Load indices from manifest
            for feature_name in self.manifest.get('indices', {}).keys():
                self.load_index(feature_name)
        
        logger.info(f"✅ Loaded {self.stats['loaded_indices']} indices")
        logger.info(f"📊 Total embeddings: {self.stats['total_embeddings']:,}")
    
    def search(self, feature_name: str, query_embedding: np.ndarray, 
               include_scores: bool = True) -> List[Dict[str, Any]]:
        """
        Search HNSW index for similar embeddings with adaptive similarity-based stopping
        
        Uses percentile-based drop detection:
        1. Always searches for k=25 results initially
        2. Calculates percentage drops between consecutive results
        3. Finds the 80th percentile of all drops
        4. Stops at the first drop that falls in the top 80th percentile (sudden drop detection)
        5. If no sudden drop detected, returns all 25 results
        
        Args:
            feature_name: Name of the feature index to search
            query_embedding: Query embedding vector
            include_scores: Whether to include similarity scores
            
        Returns:
            List of results with node indices and optional scores (adaptively stopped based on similarity drops)
        """
        if feature_name not in self.indices:
            logger.error(f"❌ Index not loaded: {feature_name}")
            return []
        
        if feature_name not in self.mappings:
            logger.error(f"❌ Mapping not loaded: {feature_name}")
            return []
        
        index = self.indices[feature_name]
        node_indices = self.mappings[feature_name]
        
        # Prepare query
        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        
        # Always search for at least 25 results
        initial_k = min(25, len(node_indices))
        
        # Search with scores (required for similarity-based stopping)
        logger.info(f"🔍 HNSW search: feature={feature_name}, initial_k={initial_k}, query_shape={query.shape}")
        scores, indices = index.search(query, initial_k)
        logger.info(f"📊 Search returned {len(scores[0])} scores, {len(indices[0])} indices")
        
        if len(scores[0]) == 0:
            return []
        
        # Convert scores to float array (these are L2 distances - lower is better)
        score_values = [float(score) for score in scores[0]]
        
        # Calculate percentage increases and absolute differences between consecutive results
        # For L2 distance: score[i+1] > score[i] means worse match (distance increased)
        # For very small scores, use absolute difference instead of percentage
        increases = []
        abs_differences = []
        for i in range(len(score_values) - 1):
            abs_diff = score_values[i+1] - score_values[i]
            abs_differences.append(abs_diff)
            
            if score_values[i] > 0:
                # Calculate percentage increase: (new - old) / old * 100
                increase = (score_values[i+1] - score_values[i]) / score_values[i] * 100
                increases.append(increase)
            else:
                # If score[i] is 0, any positive score[i+1] is infinite increase
                increases.append(float('inf') if score_values[i+1] > 0 else 0.0)
        
        # Find 95th percentile of increases (higher threshold to avoid stopping on tiny differences)
        if len(increases) > 0:
            # Filter out infinite values for percentile calculation
            finite_increases = [inc for inc in increases if inc != float('inf')]
            if finite_increases:
                threshold = np.percentile(finite_increases, 95)  # Use 95th percentile instead of 80th
                abs_threshold = np.percentile(abs_differences, 95)  # Absolute difference threshold
                logger.info(f"📊 Calculated 95th percentile increase threshold: {threshold:.2f}%")
                logger.info(f"📊 Calculated 95th percentile absolute difference threshold: {abs_threshold:.6f}")
                
                # Find first large increase (sudden drop in quality)
                # Use BOTH percentage AND absolute difference to avoid false positives on tiny scores
                stop_index = len(score_values)  # Default: use all results
                for i, (increase, abs_diff) in enumerate(zip(increases, abs_differences)):
                    # Stop if: infinite increase OR (large percentage increase AND significant absolute difference)
                    # For very small scores (< 1e-10), require larger absolute difference (> 0.01)
                    is_large_percentage = increase == float('inf') or (finite_increases and increase >= threshold)
                    is_significant_absolute = abs_diff >= max(abs_threshold, 0.01)  # At least 0.01 absolute difference
                    
                    if is_large_percentage and is_significant_absolute:
                        stop_index = i + 1  # Include the result before the big increase
                        logger.info(f"⚠️ Large distance increase detected at position {i+1}: {increase:.2f}% increase, {abs_diff:.6f} absolute (thresholds: {threshold:.2f}%, {max(abs_threshold, 0.01):.6f}), stopping early")
                        break
                
                if stop_index == len(score_values):
                    logger.info(f"✅ No large increase detected, using all {len(score_values)} results")
            else:
                # All increases are infinite (score[i] = 0 for all)
                stop_index = 1
                logger.info(f"✅ All first scores are perfect (0), stopping after 1 result")
        else:
            stop_index = len(score_values)
        
        # Process results up to stop_index
        results = []
        for i in range(min(stop_index, len(indices[0]))):
            idx = indices[0][i]
            score = score_values[i]
            
            if idx < 0 or idx >= len(node_indices):  # Invalid index
                continue
            
            node_index = node_indices[idx]
            
            # Get node metadata from Neo4j if available
            node_metadata = None
            node_type = None
            
            node_index_int = int(node_index) if isinstance(node_index, str) else node_index
            
            # Get node type from graph_loader (lightweight, in-memory)
            if self.graph_loader:
                node_type = self.graph_loader.get_node_type(node_index_int)
            
            # Get node metadata from Neo4j (on-demand, fresh data)
            if self.neo4j_driver:
                node_metadata = self._get_node_metadata_from_neo4j(node_index_int, node_type)
            
            result = {
                'node_index': node_index,
                'score': score,
                'rank': i + 1,
                'feature': feature_name
            }
            
            if node_metadata:
                result['metadata'] = node_metadata
            if node_type:
                result['node_type'] = node_type
            
            results.append(result)
        
        stopped_early = len(results) < len(scores[0])
        logger.info(f"✅ Returned {len(results)} results (stopped early: {stopped_early})")
        return results
    
    def search_multiple_features(self, query_embeddings: Dict[str, np.ndarray]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search multiple feature indices simultaneously
        
        Args:
            query_embeddings: Dict mapping feature names to query embeddings
            
        Returns:
            Dict mapping feature names to search results
        """
        results = {}
        
        for feature_name, query_embedding in query_embeddings.items():
            if feature_name in self.indices:
                results[feature_name] = self.search(feature_name, query_embedding)
            else:
                logger.warning(f"⚠️  Index not available: {feature_name}")
                results[feature_name] = []
        
        return results
    
    def get_index_info(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific index"""
        if feature_name not in self.indices:
            return None
        
        index = self.indices[feature_name]
        node_indices = self.mappings[feature_name]
        
        return {
            'feature_name': feature_name,
            'embedding_count': len(node_indices),
            'embedding_dimension': index.d,
            'index_type': 'HNSW',
            'available': True
        }
    
    def get_all_index_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all loaded indices"""
        info = {}
        
        for feature_name in self.indices.keys():
            info[feature_name] = self.get_index_info(feature_name)
        
        return info
    
    def is_available(self, feature_name: str) -> bool:
        """Check if an index is available"""
        return feature_name in self.indices
    
    def get_available_features(self) -> List[str]:
        """Get list of available feature names"""
        return self.stats['available_features'].copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get HNSW manager statistics"""
        return self.stats.copy()
    
    def _get_node_metadata_from_neo4j(self, node_index: int, node_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch node metadata from Neo4j by node_index
        
        Args:
            node_index: The node index to fetch
            node_type: Optional node type (Paper, Author, FieldOfStudy, Institution)
            
        Returns:
            Dictionary of node properties or None if not found
        """
        if not self.neo4j_driver:
            return None
        
        try:
            # Map node type to Neo4j label
            label_map = {
                'paper': 'Paper',
                'author': 'Author',
                'field_of_study': 'FieldOfStudy',
                'institution': 'Institution'
            }
            
            # If we know the type, use it for faster query
            if node_type and node_type.lower() in label_map:
                label = label_map[node_type.lower()]
                cypher = f"MATCH (n:{label} {{paperId: $node_id}}) RETURN properties(n) AS props"
                id_field = 'paperId' if label == 'Paper' else f'{node_type.lower()}Id'
                
                # Adjust for correct field names
                if label == 'Paper':
                    cypher = f"MATCH (n:{label} {{paperId: $node_id}}) RETURN properties(n) AS props"
                elif label == 'Author':
                    cypher = f"MATCH (n:{label} {{authorId: $node_id}}) RETURN properties(n) AS props"
                elif label == 'FieldOfStudy':
                    cypher = f"MATCH (n:{label} {{fieldId: $node_id}}) RETURN properties(n) AS props"
                elif label == 'Institution':
                    cypher = f"MATCH (n:{label} {{institutionId: $node_id}}) RETURN properties(n) AS props"
            else:
                # Fallback: search all node types (slower)
                cypher = """
                MATCH (n)
                WHERE n.paperId = $node_id 
                   OR n.authorId = $node_id 
                   OR n.fieldId = $node_id 
                   OR n.institutionId = $node_id
                RETURN properties(n) AS props
                LIMIT 1
                """
            
            with self.neo4j_driver.session() as session:
                result = session.run(cypher, node_id=int(node_index))
                record = result.single()
                
                if record and record['props']:
                    return dict(record['props'])
                
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch metadata from Neo4j for node {node_index}: {e}")
            return None


def main():
    """Test the HNSW manager"""
    indices_dir = "/shared/khoja/CogComp/output/mag_hnsw_indices"
    
    logger.info("🧬 Testing MAG HNSW Manager")
    
    try:
        manager = MAGHNSWManager(indices_dir)
        manager.load_all_indices()
        
        logger.info("✅ HNSW Manager loaded successfully!")
        
        # Test search with dummy embedding
        dummy_embedding = np.random.randn(384).astype(np.float32)
        
        # Try searching a few features
        for feature in ['original_title_embedding', 'abstract_embedding', 'author_embedding']:
            if manager.is_available(feature):
                logger.info(f"🔍 Testing search on {feature}...")
                results = manager.search(feature, dummy_embedding, top_k=5)
                logger.info(f"  Results: {len(results)} found")
                if results:
                    logger.info(f"  Top result: node_index {results[0]['node_index']}, score {results[0]['score']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load HNSW manager: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
