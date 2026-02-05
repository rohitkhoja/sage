"""
Modular RAG Pipeline with separated chunking and graph building phases
"""

import os
import json
import time
import threading
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import numpy as np
import faiss
from loguru import logger

# Local imports to avoid serialization issues
from src.core.models import DocumentChunk, TableChunk, SourceInfo, ChunkType


@dataclass
class PipelineConfig:
    """Configuration for the modular pipeline"""
    num_gpus: int = 4
    chunk_batch_size: int = 32
    graph_build_threads: int = 64
    max_neighbors_per_chunk: int = 200
    similarity_threshold: float = 0.3
    output_dir: str = "output"
    faiss_index_type: str = "HNSW" # or "Flat"
    use_gpu_faiss: bool = True


class ChunkingPhase:
    """Phase 1: Build all chunks and save to output directory"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processing components lazily to avoid import issues
        self._document_processor = None
        self._table_processor = None
        
    def _get_document_processor(self):
        """Lazy initialization of document processor"""
        if self._document_processor is None:
            from src.pipeline.parallel_document_processor import DocumentProcessingOrchestrator
            self._document_processor = DocumentProcessingOrchestrator(
                num_gpus=self.config.num_gpus
            )
        return self._document_processor
    
    def _get_table_processor(self):
        """Lazy initialization of table processor"""
        if self._table_processor is None:
            from src.processors.table_processor import TableProcessor
            from src.processors.embedding_service import EmbeddingService
            from src.core.models import ProcessingConfig
            
            # Create minimal config for table processor
            proc_config = ProcessingConfig(batch_size=self.config.chunk_batch_size)
            embedding_service = EmbeddingService(proc_config)
            self._table_processor = TableProcessor(proc_config, embedding_service)
        return self._table_processor
    
    def process_json_content(self, json_file: str, start_index: int = 0, 
                           end_index: int = None) -> Dict[str, Any]:
        """Process mixed content from JSON file and save chunks"""
        logger.info(f"=== CHUNKING PHASE: Processing {json_file} ===")
        
        # Load content
        with open(json_file, 'r') as f:
            content_items = json.load(f)
        
        # Apply slicing
        if end_index is None:
            end_index = len(content_items)
        content_items = content_items[start_index:end_index]
        
        # Classify content types
        documents = []
        tables = []
        
        for item in content_items:
            if self._is_document(item):
                documents.append(item)
            elif self._is_table(item):
                tables.append(item)
        
        logger.info(f"Found {len(documents)} documents and {len(tables)} tables")
        
        results = {
            "documents_processed": 0,
            "tables_processed": 0,
            "total_chunks": 0,
            "chunk_cache_dirs": [],
            "processing_time": 0
        }
        
        start_time = time.time()
        
        try:
            # Process documents if any
            if documents:
                logger.info(f"Processing {len(documents)} documents...")
                doc_results = self._process_documents_batch(documents)
                results["documents_processed"] = len(documents)
                results["total_chunks"] += len(doc_results.get("processed_chunks", []))
                
                # Collect cache directories
                for gpu_id in range(self.config.num_gpus):
                    cache_dir = f"output/chunks_cache_gpu_{gpu_id}"
                    if os.path.exists(cache_dir):
                        results["chunk_cache_dirs"].append(cache_dir)
            
            # Process tables if any
            if tables:
                logger.info(f"Processing {len(tables)} tables...")
                table_results = self._process_tables_batch(tables)
                results["tables_processed"] = len(tables)
                results["total_chunks"] += len(table_results.get("processed_chunks", []))
            
            results["processing_time"] = time.time() - start_time
            results["success"] = True
            
            logger.info(f"=== CHUNKING PHASE COMPLETE ===")
            logger.info(f"Documents: {results['documents_processed']}")
            logger.info(f"Tables: {results['tables_processed']}")
            logger.info(f"Total chunks: {results['total_chunks']}")
            logger.info(f"Processing time: {results['processing_time']:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Chunking phase failed: {e}")
            return {"success": False, "error": str(e)}
        
        finally:
            # Cleanup processors
            self._cleanup_processors()
    
    def _is_document(self, item: Dict[str, Any]) -> bool:
        """Check if item is a document"""
        # Check metadata field first
        if 'metadata' in item:
            metadata = item['metadata']
            source_type = metadata.get('source_type', '').lower()
            file_path = metadata.get('file_path', '').lower()
        else:
            # Fallback to direct fields
            source_type = item.get('source_type', '').lower()
            file_path = item.get('file_path', '').lower()
        
        return (source_type == 'document' or 
                file_path.endswith(('.txt', '.md', '.doc', '.docx')) or
                'document' in source_type)
    
    def _is_table(self, item: Dict[str, Any]) -> bool:
        """Check if item is a table"""
        # Check metadata field first
        if 'metadata' in item:
            metadata = item['metadata']
            source_type = metadata.get('source_type', '').lower()
            file_path = metadata.get('file_path', '').lower()
        else:
            # Fallback to direct fields
            source_type = item.get('source_type', '').lower()
            file_path = item.get('file_path', '').lower()
        
        return (source_type == 'table' or 
                file_path.endswith(('.csv', '.xlsx', '.xls')) or
                'table' in source_type)
    
    def _process_documents_batch(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process documents using parallel document processor"""
        document_processor = self._get_document_processor()
        
        # Convert to format expected by document processor
        processed_docs = []
        for doc in documents:
            # Extract content and metadata
            if 'metadata' in doc:
                metadata = doc['metadata']
                content = doc.get('content', '')
                source = metadata.get('file_path', metadata.get('source', 'unknown'))
            else:
                content = doc.get('content', '')
                source = doc.get('file_path', doc.get('source', 'unknown'))
            
            processed_docs.append({
                'content': content,
                'source': source
            })
        
        # Process using parallel document processor
        results = document_processor.process_documents(
            processed_docs,
            similarity_threshold=0.8,
            batch_size=self.config.chunk_batch_size
        )
        
        return results
    
    def _process_tables_batch(self, tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process tables using table processor"""
        table_processor = self._get_table_processor()
        processed_chunks = []
        
        for table in tables:
            try:
                # Extract file path and metadata
                if 'metadata' in table:
                    metadata = table['metadata']
                    file_path = metadata.get('file_path', '')
                    source_id = metadata.get('id', '')
                    source_name = metadata.get('title', os.path.basename(file_path))
                else:
                    file_path = table.get('file_path', '')
                    source_id = table.get('id', '')
                    source_name = table.get('title', os.path.basename(file_path))
                
                if not os.path.exists(file_path):
                    logger.warning(f"Table file not found: {file_path}")
                    continue
                
                # Create SourceInfo
                source_info = SourceInfo(
                    source_id=source_id,
                    source_name=source_name,
                    source_type=ChunkType.TABLE,
                    file_path=file_path
                )
                
                # Process table
                chunks = table_processor.process(source_info)
                processed_chunks.extend([chunk.chunk_id for chunk in chunks])
                
            except Exception as e:
                logger.error(f"Error processing table: {e}")
                continue
        
        return {"processed_chunks": processed_chunks}
    
    def _cleanup_processors(self):
        """Cleanup processors"""
        if self._document_processor:
            try:
                self._document_processor.shutdown()
            except:
                pass
        self._document_processor = None
        self._table_processor = None


class GraphBuildingPhase:
    """Phase 2: Load all chunks into HNSW FAISS and build graph with parallel threads"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        
        # FAISS components
        self.faiss_index = None
        self.chunk_id_to_index = {}
        self.index_to_chunk_id = {}
        self.chunks_by_id = {}
        
        # Graph building
        self.graph_nodes = {}
        self.graph_edges = {}
        self.thread_lock = threading.Lock()
        
    def build_graph_from_cache(self, cache_dirs: List[str]) -> Dict[str, Any]:
        """Build graph from cached chunks with parallel processing"""
        logger.info(f"=== GRAPH BUILDING PHASE: Starting with {len(cache_dirs)} cache directories ===")
        
        start_time = time.time()
        
        try:
            # Step 1: Load all chunks into FAISS
            logger.info("Step 1: Loading chunks into FAISS...")
            load_time = time.time()
            chunks_loaded = self._load_chunks_to_faiss(cache_dirs)
            load_duration = time.time() - load_time
            
            if chunks_loaded == 0:
                logger.error("No chunks loaded into FAISS")
                return {"success": False, "error": "No chunks found"}
            
            logger.info(f"Loaded {chunks_loaded} chunks into FAISS in {load_duration:.2f}s")
            
            # Step 2: Build graph with parallel threads
            logger.info(f"Step 2: Building graph with {self.config.graph_build_threads} parallel threads...")
            graph_time = time.time()
            self._build_graph_parallel()
            graph_duration = time.time() - graph_time
            
            # Step 3: Save graph
            logger.info("Step 3: Saving graph...")
            save_time = time.time()
            graph_data = self._save_graph()
            save_duration = time.time() - save_time
            
            total_time = time.time() - start_time
            
            logger.info(f"=== GRAPH BUILDING PHASE COMPLETE ===")
            logger.info(f"Chunks loaded: {chunks_loaded}")
            logger.info(f"Nodes created: {len(self.graph_nodes)}")
            logger.info(f"Edges created: {len(self.graph_edges)}")
            logger.info(f"Load time: {load_duration:.2f}s")
            logger.info(f"Graph build time: {graph_duration:.2f}s")
            logger.info(f"Save time: {save_duration:.2f}s")
            logger.info(f"Total time: {total_time:.2f}s")
            
            return {
                "success": True,
                "chunks_loaded": chunks_loaded,
                "nodes_created": len(self.graph_nodes),
                "edges_created": len(self.graph_edges),
                "load_time": load_duration,
                "graph_build_time": graph_duration,
                "save_time": save_duration,
                "total_time": total_time,
                "graph_file": graph_data.get("graph_file")
            }
            
        except Exception as e:
            logger.error(f"Graph building phase failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _load_chunks_to_faiss(self, cache_dirs: List[str]) -> int:
        """Load all chunks from cache directories into FAISS index"""
        all_chunks = []
        embeddings = []
        
        # Load chunks from all cache directories
        for cache_dir in cache_dirs:
            cache_path = Path(cache_dir)
            if not cache_path.exists():
                logger.warning(f"Cache directory not found: {cache_dir}")
                continue
                
            # Load chunk index
            index_file = cache_path / "chunk_index.json"
            if not index_file.exists():
                logger.warning(f"No chunk index found in {cache_dir}")
                continue
            
            with open(index_file, 'r') as f:
                chunk_index = json.load(f)
            
            # Load individual chunks
            for chunk_key, chunk_info in chunk_index.items():
                chunk_file = cache_path / chunk_info["file"]
                if chunk_file.exists():
                    try:
                        with open(chunk_file, 'r') as f:
                            chunk_data = json.load(f)
                        
                        chunk_id = chunk_data["chunk_id"]
                        embedding = chunk_data["embedding"]
                        
                        if embedding and len(embedding) > 0:
                            all_chunks.append(chunk_data)
                            embeddings.append(embedding)
                            self.chunks_by_id[chunk_id] = chunk_data
                        
                    except Exception as e:
                        logger.warning(f"Error loading chunk from {chunk_file}: {e}")
                        continue
        
        if not embeddings:
            logger.error("No valid embeddings found")
            return 0
        
        # Build FAISS index
        embeddings_matrix = np.array(embeddings, dtype=np.float32)
        dimension = embeddings_matrix.shape[1]
        
        logger.info(f"Building FAISS index with {len(embeddings)} embeddings, dimension={dimension}")
        
        # Create HNSW index as requested
        if self.config.faiss_index_type == "HNSW":
            self.faiss_index = faiss.IndexHNSWFlat(dimension, 32) # M=32 for balanced performance
            self.faiss_index.hnsw.efConstruction = 200 # Higher quality construction
            self.faiss_index.hnsw.efSearch = 100 # Search quality
        else:
            self.faiss_index = faiss.IndexFlatIP(dimension) # Fallback to flat index
        
        # Move to GPU if requested and available
        if self.config.use_gpu_faiss and faiss.get_num_gpus() > 0:
            try:
                # Normalize for cosine similarity
                faiss.normalize_L2(embeddings_matrix)
                
                res = faiss.StandardGpuResources()
                self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
                logger.info("Using GPU FAISS index")
            except Exception as e:
                logger.warning(f"Failed to use GPU FAISS: {e}, using CPU")
        else:
            # Normalize for cosine similarity on CPU
            faiss.normalize_L2(embeddings_matrix)
        
        # Add embeddings to index
        self.faiss_index.add(embeddings_matrix)
        
        # Build mappings
        for i, chunk_data in enumerate(all_chunks):
            chunk_id = chunk_data["chunk_id"]
            self.chunk_id_to_index[chunk_id] = i
            self.index_to_chunk_id[i] = chunk_id
        
        logger.info(f"FAISS index built with {self.faiss_index.ntotal} vectors")
        return len(all_chunks)
    
    def _build_graph_parallel(self):
        """Build graph using parallel threads"""
        all_chunk_ids = list(self.chunks_by_id.keys())
        total_chunks = len(all_chunk_ids)
        
        # Create nodes for all chunks
        for chunk_id, chunk_data in self.chunks_by_id.items():
            with self.thread_lock:
                self.graph_nodes[chunk_id] = {
                    "id": chunk_id,
                    "content": chunk_data.get("content", "")[:200] + "..." if len(chunk_data.get("content", "")) > 200 else chunk_data.get("content", ""),
                    "source": chunk_data.get("source_document", "unknown"),
                    "type": chunk_data.get("chunk_type", "document"),
                    "keywords": chunk_data.get("keywords", [])
                }
        
        # Divide chunks among threads
        chunks_per_thread = max(1, total_chunks // self.config.graph_build_threads)
        thread_chunks = []
        
        for i in range(0, total_chunks, chunks_per_thread):
            end_idx = min(i + chunks_per_thread, total_chunks)
            thread_chunks.append(all_chunk_ids[i:end_idx])
        
        logger.info(f"Dividing {total_chunks} chunks among {len(thread_chunks)} threads")
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=self.config.graph_build_threads) as executor:
            futures = []
            for thread_id, chunk_batch in enumerate(thread_chunks):
                future = executor.submit(self._process_chunk_batch, thread_id, chunk_batch)
                futures.append(future)
            
            # Wait for all threads to complete
            completed = 0
            for future in as_completed(futures):
                try:
                    thread_id, edges_created = future.result()
                    completed += 1
                    logger.info(f"Thread {thread_id} completed: {edges_created} edges created "
                               f"({completed}/{len(futures)} threads done)")
                except Exception as e:
                    logger.error(f"Thread failed: {e}")
                    completed += 1
    
    def _process_chunk_batch(self, thread_id: int, chunk_ids: List[str]) -> tuple:
        """Process a batch of chunks in a single thread"""
        edges_created = 0
        
        for chunk_id in chunk_ids:
            try:
                # Find neighbors using FAISS
                neighbors = self._find_neighbors_faiss(chunk_id, self.config.max_neighbors_per_chunk)
                
                # Create edges for neighbors above threshold
                for neighbor_id, similarity in neighbors:
                    if similarity >= self.config.similarity_threshold:
                        edge_id = f"{chunk_id}_{neighbor_id}_{int(similarity*1000)}"
                        
                        # Thread-safe edge creation
                        with self.thread_lock:
                            if edge_id not in self.graph_edges:
                                self.graph_edges[edge_id] = {
                                    "id": edge_id,
                                    "source": chunk_id,
                                    "target": neighbor_id,
                                    "similarity": similarity,
                                    "weight": similarity
                                }
                                edges_created += 1
                
            except Exception as e:
                logger.warning(f"Error processing chunk {chunk_id}: {e}")
                continue
        
        return thread_id, edges_created
    
    def _find_neighbors_faiss(self, chunk_id: str, k: int) -> List[tuple]:
        """Find k nearest neighbors for a chunk using FAISS"""
        if chunk_id not in self.chunk_id_to_index:
            return []
        
        try:
            chunk_data = self.chunks_by_id[chunk_id]
            embedding = np.array([chunk_data["embedding"]], dtype=np.float32)
            
            # Search for k+1 neighbors (including self)
            search_k = min(k + 1, self.faiss_index.ntotal)
            similarities, indices = self.faiss_index.search(embedding, search_k)
            
            neighbors = []
            for sim, idx in zip(similarities[0], indices[0]):
                if idx == -1: # Invalid index
                    continue
                
                neighbor_chunk_id = self.index_to_chunk_id[idx]
                
                # Skip self-comparison
                if neighbor_chunk_id == chunk_id:
                    continue
                
                neighbors.append((neighbor_chunk_id, float(sim)))
            
            return neighbors[:k] # Return only k neighbors
            
        except Exception as e:
            logger.warning(f"Error finding neighbors for {chunk_id}: {e}")
            return []
    
    def _save_graph(self) -> Dict[str, Any]:
        """Save the built graph to file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        graph_file = self.output_dir / f"parallel_graph_{timestamp}.json"
        
        graph_data = {
            "nodes": self.graph_nodes,
            "edges": self.graph_edges,
            "metadata": {
                "total_nodes": len(self.graph_nodes),
                "total_edges": len(self.graph_edges),
                "similarity_threshold": self.config.similarity_threshold,
                "max_neighbors_per_chunk": self.config.max_neighbors_per_chunk,
                "graph_build_threads": self.config.graph_build_threads,
                "created_at": timestamp,
                "faiss_index_type": self.config.faiss_index_type
            }
        }
        
        with open(graph_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        logger.info(f"Graph saved to: {graph_file}")
        return {"graph_file": str(graph_file), "graph_data": graph_data}


class ModularPipeline:
    """Main modular pipeline orchestrator"""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.chunking_phase = ChunkingPhase(self.config)
        self.graph_building_phase = GraphBuildingPhase(self.config)
        
    def run_complete_pipeline(self, json_file: str, start_index: int = 0, 
                            end_index: int = None) -> Dict[str, Any]:
        """Run the complete two-phase pipeline"""
        logger.info(f"=== STARTING MODULAR PIPELINE ===")
        logger.info(f"Config: {self.config.graph_build_threads} threads, "
                   f"{self.config.max_neighbors_per_chunk} neighbors per chunk, "
                   f"threshold {self.config.similarity_threshold}")
        
        total_start_time = time.time()
        
        # Phase 1: Chunking
        logger.info("=== PHASE 1: CHUNKING ===")
        chunking_results = self.chunking_phase.process_json_content(
            json_file, start_index, end_index
        )
        
        if not chunking_results.get("success"):
            return chunking_results
        
        # Phase 2: Graph Building
        logger.info("=== PHASE 2: GRAPH BUILDING ===")
        cache_dirs = chunking_results.get("chunk_cache_dirs", [])
        
        if not cache_dirs:
            # Fallback to standard cache directories
            cache_dirs = []
            for gpu_id in range(self.config.num_gpus):
                cache_dir = f"output/chunks_cache_gpu_{gpu_id}"
                if os.path.exists(cache_dir):
                    cache_dirs.append(cache_dir)
        
        if not cache_dirs:
            logger.error("No cache directories found")
            return {"success": False, "error": "No cache directories found"}
        
        graph_results = self.graph_building_phase.build_graph_from_cache(cache_dirs)
        
        if not graph_results.get("success"):
            return graph_results
        
        total_time = time.time() - total_start_time
        
        # Combine results
        final_results = {
            "success": True,
            "chunking_phase": chunking_results,
            "graph_building_phase": graph_results,
            "total_pipeline_time": total_time,
            "summary": {
                "documents_processed": chunking_results.get("documents_processed", 0),
                "tables_processed": chunking_results.get("tables_processed", 0),
                "total_chunks": chunking_results.get("total_chunks", 0),
                "chunks_loaded_to_faiss": graph_results.get("chunks_loaded", 0),
                "nodes_created": graph_results.get("nodes_created", 0),
                "edges_created": graph_results.get("edges_created", 0),
                "graph_file": graph_results.get("graph_file")
            }
        }
        
        logger.info(f"=== MODULAR PIPELINE COMPLETE ===")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Documents: {final_results['summary']['documents_processed']}")
        logger.info(f"Tables: {final_results['summary']['tables_processed']}")
        logger.info(f"Total chunks: {final_results['summary']['total_chunks']}")
        logger.info(f"Graph nodes: {final_results['summary']['nodes_created']}")
        logger.info(f"Graph edges: {final_results['summary']['edges_created']}")
        logger.info(f"Graph saved to: {final_results['summary']['graph_file']}")
        
        return final_results
    
    def run_chunking_only(self, json_file: str, start_index: int = 0, 
                         end_index: int = None) -> Dict[str, Any]:
        """Run only the chunking phase"""
        return self.chunking_phase.process_json_content(json_file, start_index, end_index)
    
    def run_graph_building_only(self, cache_dirs: List[str] = None) -> Dict[str, Any]:
        """Run only the graph building phase"""
        if cache_dirs is None:
            # Auto-detect cache directories
            cache_dirs = []
            for gpu_id in range(self.config.num_gpus):
                cache_dir = f"output/chunks_cache_gpu_{gpu_id}"
                if os.path.exists(cache_dir):
                    cache_dirs.append(cache_dir)
        
        return self.graph_building_phase.build_graph_from_cache(cache_dirs)


def main():
    """Example usage of modular pipeline"""
    
    # Configure pipeline
    config = PipelineConfig(
        num_gpus=4,
        chunk_batch_size=32,
        graph_build_threads=64, # 64 parallel threads as requested
        max_neighbors_per_chunk=200, # 200 neighbors per chunk as requested
        similarity_threshold=0.3,
        output_dir="output",
        faiss_index_type="HNSW",
        use_gpu_faiss=True
    )
    
    # Initialize pipeline
    pipeline = ModularPipeline(config)
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(
        json_file="datasets/ott-qa.json",
        start_index=0,
        end_index=100 # Process first 100 items
    )
    
    if results.get("success"):
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results: {results['summary']}")
    else:
        logger.error(f"Pipeline failed: {results.get('error')}")


if __name__ == "__main__":
    main() 