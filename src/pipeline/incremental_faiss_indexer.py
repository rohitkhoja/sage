#!/usr/bin/env python3
"""
Incremental FAISS Indexing System
Monitors chunk cache and builds/updates FAISS index incrementally
"""

import os
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set
import numpy as np
import faiss
from loguru import logger
from dataclasses import dataclass
import pickle
from concurrent.futures import ThreadPoolExecutor
import hashlib

@dataclass
class FAISSIndexStatus:
    """Status information for FAISS index"""
    total_vectors: int
    last_update: str
    index_file: str
    dimension: int
    processed_chunks: Set[str]
    
    def to_dict(self) -> Dict:
        return {
            "total_vectors": self.total_vectors,
            "last_update": self.last_update,
            "index_file": self.index_file,
            "dimension": self.dimension,
            "processed_chunks": list(self.processed_chunks)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FAISSIndexStatus':
        data["processed_chunks"] = set(data.get("processed_chunks", []))
        return cls(**data)

class IncrementalFAISSIndexer:
    """Incremental FAISS indexing with monitoring and automatic updates"""
    
    def __init__(self, 
                 cache_dirs: List[str], 
                 index_dir: str = "output/faiss_index",
                 embedding_dim: int = 384,
                 max_neighbors: int = 200,
                 update_interval: int = 30):
        
        self.cache_dirs = [Path(d) for d in cache_dirs]
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_dim = embedding_dim
        self.max_neighbors = max_neighbors
        self.update_interval = update_interval
        
        # FAISS index files
        self.index_file = self.index_dir / "incremental_index.faiss"
        self.status_file = self.index_dir / "index_status.json"
        self.chunk_mapping_file = self.index_dir / "chunk_mapping.json"
        
        # Initialize index and status
        self.index = None
        self.chunk_id_to_index = {}  # Maps chunk_id to FAISS index position
        self.index_to_chunk_id = {}  # Maps FAISS index position to chunk_id
        self.status = self._load_or_create_status()
        
        self._load_existing_index()
        
        # Monitoring
        self.monitoring = False
        self.monitor_thread = None
        
        logger.info(f"Initialized FAISS indexer with {len(self.cache_dirs)} cache directories")
        logger.info(f"Current index size: {self.status.total_vectors} vectors")
    
    def _load_or_create_status(self) -> FAISSIndexStatus:
        """Load existing status or create new one"""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                data = json.load(f)
            return FAISSIndexStatus.from_dict(data)
        else:
            return FAISSIndexStatus(
                total_vectors=0,
                last_update=time.strftime("%Y%m%d_%H%M%S"),
                index_file=self.index_file.name,
                dimension=self.embedding_dim,
                processed_chunks=set()
            )
    
    def _save_status(self):
        """Save current status to disk"""
        with open(self.status_file, 'w') as f:
            json.dump(self.status.to_dict(), f, indent=2)
    
    def _load_existing_index(self):
        """Load existing FAISS index if available"""
        if self.index_file.exists():
            try:
                self.index = faiss.read_index(str(self.index_file))
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
                
                # Load chunk mappings
                if self.chunk_mapping_file.exists():
                    with open(self.chunk_mapping_file, 'r') as f:
                        mapping_data = json.load(f)
                        self.chunk_id_to_index = mapping_data.get("chunk_to_index", {})
                        # Rebuild reverse mapping
                        self.index_to_chunk_id = {v: k for k, v in self.chunk_id_to_index.items()}
                        
            except Exception as e:
                logger.error(f"Failed to load existing index: {e}")
                self.index = None
        
        if self.index is None:
            # Create new HNSW index
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 100
            logger.info("Created new HNSW FAISS index")
    
    def _save_index(self):
        """Save FAISS index and mappings to disk"""
        faiss.write_index(self.index, str(self.index_file))
        
        # Save chunk mappings
        mapping_data = {
            "chunk_to_index": self.chunk_id_to_index,
            "index_to_chunk": self.index_to_chunk_id
        }
        with open(self.chunk_mapping_file, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        
        self._save_status()
        logger.info(f"Saved FAISS index with {self.index.ntotal} vectors")
    
    def scan_for_new_chunks(self) -> List[Dict]:
        """Scan cache directories for new chunks"""
        new_chunks = []
        
        for cache_dir in self.cache_dirs:
            if not cache_dir.exists():
                continue
                
            # Load cache index
            cache_index_file = cache_dir / "chunk_index.json"
            if not cache_index_file.exists():
                continue
                
            with open(cache_index_file, 'r') as f:
                cache_index = json.load(f)
            
            # Check for new chunks
            for chunk_key, chunk_info in cache_index.items():
                chunk_id = chunk_info["chunk_id"]
                
                if chunk_id not in self.status.processed_chunks:
                    # Load chunk data
                    chunk_file = cache_dir / chunk_info["file"]
                    if chunk_file.exists():
                        with open(chunk_file, 'r') as f:
                            chunk_data = json.load(f)
                        
                        new_chunks.append({
                            "chunk_id": chunk_id,
                            "embedding": np.array(chunk_data["embedding"]),
                            "content": chunk_data["content"],
                            "source": chunk_data["source_document"]
                        })
        
        return new_chunks
    
    def add_chunks_to_index(self, chunks: List[Dict]) -> int:
        """Add new chunks to FAISS index"""
        if not chunks:
            return 0
        
        embeddings = np.stack([chunk["embedding"] for chunk in chunks]).astype(np.float32)
        
        # Add to FAISS index
        start_index = self.index.ntotal
        self.index.add(embeddings)
        
        # Update mappings
        for i, chunk in enumerate(chunks):
            chunk_id = chunk["chunk_id"]
            faiss_index = start_index + i
            
            self.chunk_id_to_index[chunk_id] = faiss_index
            self.index_to_chunk_id[faiss_index] = chunk_id
            self.status.processed_chunks.add(chunk_id)
        
        # Update status
        self.status.total_vectors = self.index.ntotal
        self.status.last_update = time.strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Added {len(chunks)} chunks to FAISS index. Total: {self.index.ntotal}")
        return len(chunks)
    
    def update_index(self) -> int:
        """Scan for new chunks and update index"""
        new_chunks = self.scan_for_new_chunks()
        
        if new_chunks:
            added_count = self.add_chunks_to_index(new_chunks)
            self._save_index()
            return added_count
        
        return 0
    
    def search_neighbors(self, chunk_ids: List[str], k: int = None) -> Dict[str, List[Dict]]:
        """Search for neighbors of given chunks"""
        if k is None:
            k = min(self.max_neighbors, self.index.ntotal - 1)
        
        results = {}
        
        for chunk_id in chunk_ids:
            if chunk_id not in self.chunk_id_to_index:
                logger.warning(f"Chunk {chunk_id} not found in index")
                continue
            
            faiss_index = self.chunk_id_to_index[chunk_id]
            
            # Get embedding for this chunk
            embedding = self.index.reconstruct(faiss_index).reshape(1, -1)
            
            # Search for neighbors
            distances, indices = self.index.search(embedding, k + 1)  # +1 to exclude self
            
            neighbors = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx != faiss_index and idx in self.index_to_chunk_id:  # Exclude self
                    neighbor_chunk_id = self.index_to_chunk_id[idx]
                    similarity = 1.0 - (dist / 2.0)  # Convert L2 distance to similarity
                    
                    neighbors.append({
                        "chunk_id": neighbor_chunk_id,
                        "similarity": float(similarity),
                        "distance": float(dist)
                    })
            
            results[chunk_id] = neighbors[:k]  # Limit to k neighbors
        
        return results
    
    def start_monitoring(self):
        """Start monitoring cache directories for new chunks"""
        if self.monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Started monitoring for new chunks")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Stopped monitoring")
    
    def _monitor_loop(self):
        """Monitoring loop that runs in background"""
        while self.monitoring:
            try:
                added_count = self.update_index()
                if added_count > 0:
                    logger.info(f"Monitoring: Added {added_count} new chunks to index")
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def get_index_stats(self) -> Dict:
        """Get comprehensive index statistics"""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.embedding_dim,
            "processed_chunks": len(self.status.processed_chunks),
            "last_update": self.status.last_update,
            "cache_directories": [str(d) for d in self.cache_dirs],
            "index_file_size": self.index_file.stat().st_size if self.index_file.exists() else 0
        }

class GraphBuilder:
    """Builds knowledge graph from FAISS neighbor results"""
    
    def __init__(self, 
                 faiss_indexer: IncrementalFAISSIndexer,
                 similarity_threshold: float = 0.3,
                 output_dir: str = "output/incremental_graphs"):
        
        self.faiss_indexer = faiss_indexer
        self.similarity_threshold = similarity_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def build_graph_from_neighbors(self, chunk_neighbors: Dict[str, List[Dict]]) -> Dict:
        """Build graph structure from neighbor relationships"""
        
        nodes = {}
        edges = {}
        
        # Create nodes
        for chunk_id in chunk_neighbors.keys():
            nodes[chunk_id] = {
                "id": chunk_id,
                "type": "chunk"
            }
        
        # Create edges based on similarity threshold
        edge_count = 0
        for source_chunk, neighbors in chunk_neighbors.items():
            for neighbor in neighbors:
                if neighbor["similarity"] >= self.similarity_threshold:
                    target_chunk = neighbor["chunk_id"]
                    
                    # Create bidirectional edge
                    edge_id = f"edge_{edge_count}"
                    edges[edge_id] = {
                        "id": edge_id,
                        "source": source_chunk,
                        "target": target_chunk,
                        "similarity": neighbor["similarity"],
                        "weight": neighbor["similarity"]
                    }
                    edge_count += 1
        
        graph = {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "similarity_threshold": self.similarity_threshold,
                "created_at": time.strftime("%Y%m%d_%H%M%S")
            }
        }
        
        return graph
    
    def build_incremental_graph(self, chunk_ids: List[str] = None) -> Dict:
        """Build graph incrementally for specified chunks or all chunks"""
        
        if chunk_ids is None:
            chunk_ids = list(self.faiss_indexer.status.processed_chunks)
        
        logger.info(f"Building graph for {len(chunk_ids)} chunks")
        
        # Get neighbors for all chunks
        chunk_neighbors = self.faiss_indexer.search_neighbors(chunk_ids)
        
        # Build graph
        graph = self.build_graph_from_neighbors(chunk_neighbors)
        
        # Save graph
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        graph_file = self.output_dir / f"incremental_graph_{timestamp}.json"
        
        with open(graph_file, 'w') as f:
            json.dump(graph, f, indent=2)
        
        logger.info(f"Saved incremental graph: {graph['metadata']['total_nodes']} nodes, "
                   f"{graph['metadata']['total_edges']} edges")
        
        return graph

def main():
    """Example usage of incremental FAISS indexing"""
    
    # Initialize indexer for multiple GPU cache directories
    cache_dirs = [
        "output/chunks_cache_gpu_0",
        "output/chunks_cache_gpu_1", 
        "output/chunks_cache_gpu_2",
        "output/chunks_cache_gpu_3"
    ]
    
    indexer = IncrementalFAISSIndexer(cache_dirs)
    
    try:
        # Start monitoring
        indexer.start_monitoring()
        
        # Let it run for a while to collect chunks
        logger.info("Monitoring for new chunks... (Press Ctrl+C to stop)")
        
        while True:
            time.sleep(60)  # Check every minute
            stats = indexer.get_index_stats()
            logger.info(f"Index stats: {stats['total_vectors']} vectors, "
                       f"{stats['processed_chunks']} chunks processed")
            
            # Build graph if we have enough chunks
            if stats['total_vectors'] >= 100:
                graph_builder = GraphBuilder(indexer)
                graph = graph_builder.build_incremental_graph()
                logger.info(f"Built graph with {graph['metadata']['total_nodes']} nodes")
                break
    
    except KeyboardInterrupt:
        logger.info("Stopping monitoring...")
    
    finally:
        indexer.stop_monitoring()

if __name__ == "__main__":
    main()
