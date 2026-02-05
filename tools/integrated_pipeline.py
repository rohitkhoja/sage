#!/usr/bin/env python3
"""
Integrated Pipeline: Document Processing → Table Processing → FAISS Indexing → Graph Building
This script orchestrates the entire pipeline with decoupled, fault-tolerant components for both documents and tables
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
from loguru import logger
import threading
import ray
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline.parallel_document_processor import DocumentProcessingOrchestrator
# from src.pipeline.incremental_faiss_indexer import IncrementalFAISSIndexer, GraphBuilder
from src.processors.table_processor import TableProcessor
from src.processors.embedding_service import EmbeddingService
from src.core.models import ProcessingConfig, SourceInfo, ChunkType

class IntegratedPipeline:
    """
    Integrated pipeline that coordinates:
    1. Parallel document processing with caching
    2. Table processing with chunking and embedding
    4. Knowledge graph construction from both document and table chunks
    """
    
    def __init__(self, 
                 num_gpus: int = None,
                 similarity_threshold: float = 0.8,
                 graph_similarity_threshold: float = 0.3,
                 faiss_update_interval: int = 30,
                 output_dir: str = "output"):
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store num_gpus as instance variable
        self.num_gpus = num_gpus or torch.cuda.device_count()
        
        # Initialize processing configuration
        self.config = ProcessingConfig(
            sentence_similarity_threshold=similarity_threshold,
            table_similarity_threshold=similarity_threshold,
            use_faiss=True,
            faiss_use_gpu=True if self.num_gpus and self.num_gpus > 0 else False
        )
        
        # Initialize embedding service for table processing
        self.embedding_service = EmbeddingService(self.config)
        
        # Initialize components
        self.document_processor = DocumentProcessingOrchestrator(self.num_gpus, cache_base_dir=output_dir)
        self.table_processor = TableProcessor(self.config, self.embedding_service)
        
        # NOTE: No longer using incremental FAISS indexer since we build FAISS directly in graph building
        # self.faiss_indexer = IncrementalFAISSIndexer(...)
        # self.graph_builder = GraphBuilder(...)
        
        self.similarity_threshold = similarity_threshold
        self.graph_similarity_threshold = graph_similarity_threshold
        self.processing_stats = {
            "start_time": None,
            "documents_processed": 0,
            "tables_processed": 0,
            "document_chunks_created": 0,
            "table_chunks_created": 0,
            "graphs_built": 0,
            "total_processing_time": 0
        }
        
        logger.info("Initialized integrated pipeline with document and table processing")

    def process_mixed_content_from_json(self, 
                                      json_file: str, 
                                      start_index: int = 0,
                                      end_index: int = None,
                                      batch_size: int = None) -> Dict[str, Any]:
        """
        Process mixed content (documents and tables) from JSON metadata file
        
        Args:
            json_file: Path to JSON file containing content metadata
            start_index: Starting item index
            end_index: Ending item index (None for all)
            batch_size: Processing batch size per GPU
        """
        
        self.processing_stats["start_time"] = time.time()
        
        # Load content items from JSON
        logger.info(f"Loading content items from {json_file}")
        content_items = self._load_content_from_json(json_file, start_index, end_index)
        
        if not content_items:
            logger.warning("No content items to process")
            return {"success": False, "message": "No content found"}
        
        # Separate documents and tables
        documents = [item for item in content_items if self._is_document(item)]
        tables = [item for item in content_items if self._is_table(item)]
        
        logger.info(f"Loaded {len(documents)} documents and {len(tables)} tables for processing")
        
        # Start FAISS monitoring
        # self.faiss_indexer.start_monitoring()
        
        all_chunks = []
        
        try:
            # Process documents
            if documents:
                logger.info(f"Processing {len(documents)} documents...")
                doc_results = self._process_documents_batch(documents, batch_size)
                all_chunks.extend(doc_results.get("processed_chunks", []))
                all_chunks.extend(doc_results.get("cached_chunks", [])) # Include cached chunks too
                self.processing_stats["documents_processed"] = len(documents)
                self.processing_stats["document_chunks_created"] = len(doc_results.get("processed_chunks", []))
            
            # Process tables 
            if tables:
                logger.info(f"Processing {len(tables)} tables...")
                table_results = self._process_tables_batch(tables)
                all_chunks.extend(table_results.get("processed_chunks", []))
                all_chunks.extend(table_results.get("cached_chunks", [])) # Include cached chunks too
                self.processing_stats["tables_processed"] = len(tables)
                self.processing_stats["table_chunks_created"] = len(table_results.get("processed_chunks", []))
            
            # Build graph from all chunks
            if all_chunks:
                logger.info(f"Building full KnowledgeGraph with HNSW FAISS and parallel processing...")
                graph_result = self._build_incremental_graph(all_chunks)
                self.processing_stats["graphs_built"] = 1
                
                # Save comprehensive results
                results = self._save_comprehensive_results({
                    "documents": doc_results if documents else {},
                    "tables": table_results if tables else {},
                    "graph": graph_result,
                    "total_chunks": len(all_chunks),
                    "stats": self.processing_stats
                })
                
                return results
            else:
                logger.warning("No chunks were created from the content")
                return {"success": False, "message": "No chunks created"}
                
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            return {"success": False, "error": str(e)}
        
        finally:
            # Stop FAISS monitoring
            # self.faiss_indexer.stop_monitoring()
            
            # Calculate total processing time
            if self.processing_stats["start_time"]:
                self.processing_stats["total_processing_time"] = time.time() - self.processing_stats["start_time"]

    def process_documents_from_json(self, 
                                  json_file: str, 
                                  start_index: int = 0,
                                  end_index: int = None,
                                  batch_size: int = None) -> Dict[str, Any]:
        """
        Process documents from JSON metadata file (legacy method for backward compatibility)
        """
        return self.process_mixed_content_from_json(json_file, start_index, end_index, batch_size)

    def _is_document(self, item: Dict[str, Any]) -> bool:
        """Check if content item is a document"""
        # Check in metadata first, then in item directly
        metadata = item.get("metadata", {})
        file_path = metadata.get("file_path", item.get("file_path", ""))
        source_type = metadata.get("source_type", item.get("source_type", ""))
        
        # Check by explicit source type
        if source_type.lower() == "document":
            return True
        
        # Check by file extension
        doc_extensions = ['.txt', '.md', '.pdf', '.doc', '.docx', '.html', '.htm']
        return any(file_path.lower().endswith(ext) for ext in doc_extensions)
    
    def _is_table(self, item: Dict[str, Any]) -> bool:
        """Check if content item is a table"""
        # Check in metadata first, then in item directly
        metadata = item.get("metadata", {})
        file_path = metadata.get("file_path", item.get("file_path", ""))
        source_type = metadata.get("source_type", item.get("source_type", ""))
        
        # Check by explicit source type
        if source_type.lower() == "table":
            return True
        
        # Check by file extension
        table_extensions = ['.csv', '.tsv', '.xlsx', '.xls']
        return any(file_path.lower().endswith(ext) for ext in table_extensions)

    def _process_documents_batch(self, documents: List[Dict[str, Any]], batch_size: int = None) -> Dict[str, Any]:
        """Process document batch using document processor"""
        # Convert to proper format for document processor
        doc_items = []
        for doc in documents:
            # Get source name from the already-parsed source field, or from metadata
            metadata = doc.get("metadata", {})
            source_name = doc.get("source", metadata.get("id", metadata.get("title", "unknown")))
            
            doc_items.append({
                "content": doc.get("content", ""),
                "source": source_name,
                "metadata": metadata # Pass through metadata for SourceInfo creation
            })
        
        # Process with document processor (returns chunk IDs)
        raw_results = self.document_processor.process_documents(
            doc_items, 
            self.similarity_threshold, 
            batch_size
        )
        
        # Convert chunk IDs to DocumentChunk objects for consistency
        processed_chunks = []
        for chunk_id in raw_results.get("processed_chunks", []):
            # Get DocumentChunk object from cache
            chunk_obj = self._get_chunk_from_cache(chunk_id)
            if chunk_obj:
                processed_chunks.append(chunk_obj)
        
        # Handle cached chunks
        cached_chunks = []
        for chunk_id in raw_results.get("cached_chunks", []):
            chunk_obj = self._get_chunk_from_cache(chunk_id)
            if chunk_obj:
                cached_chunks.append(chunk_obj)
        
        # Store processed document chunks for graph building
        if processed_chunks:
            self._store_document_chunks_for_graph_building(processed_chunks)
        if cached_chunks:
            self._store_document_chunks_for_graph_building(cached_chunks)
        
        return {
            "processed_chunks": processed_chunks,
            "cached_chunks": cached_chunks,
            "total_documents": raw_results.get("total_documents", len(documents))
        }
    
    def _store_document_chunks_for_graph_building(self, document_chunks: List['DocumentChunk']):
        """Store document chunks for later retrieval during graph building"""
        if not hasattr(self, '_processed_document_chunks'):
            self._processed_document_chunks = []
        self._processed_document_chunks.extend(document_chunks)

    def _process_tables_batch(self, tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process table batch using table processor with actual CSV files"""
        from loguru import logger
        
        processed_chunks = []
        cached_chunks = []
        
        for table_item in tables:
            try:
                # Get original data from metadata
                metadata = table_item.get("metadata", {})
                
                # Get table ID and file path
                table_id = metadata.get("id", f"table_{hash(str(table_item))}")
                file_path = metadata.get("file_path", "")
                
                if not file_path:
                    logger.warning(f"No file_path found for table {table_id}")
                    continue
                
                # Construct full path to CSV file in datasets directory
                full_file_path = f"datasets/{file_path}"
                
                # Check if file exists
                if not os.path.exists(full_file_path):
                    logger.warning(f"CSV file not found: {full_file_path}")
                    continue
                
                # Create SourceInfo for table with proper metadata (same as rag_pipeline.py)
                source_info = SourceInfo(
                    source_id=table_id,
                    source_name=metadata.get("title", f"Table {table_id}"),
                    source_type=ChunkType.TABLE,
                    file_path=full_file_path, # Use actual CSV file path
                    structural_link=metadata.get("structural_link", []),
                    original_source=metadata.get("original_source", ""),
                    additional_information=metadata.get("additional_information", ""),
                    content="" # No need for embedded content since we're reading from file
                )
                
                # Process table using the actual CSV file - returns proper TableChunk objects
                table_chunks = self.table_processor.process(source_info)
                
                # Store table chunks in the all_chunks list directly (they're already TableChunk objects)
                processed_chunks.extend(table_chunks)
                
                logger.info(f"Processed table {source_info.source_name}: {len(table_chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Failed to process table {table_item.get('metadata', {}).get('id', 'unknown')}: {e}")
                continue
        
        # Store processed table chunks for graph building
        if processed_chunks:
            self._store_table_chunks_for_graph_building(processed_chunks)
        
        return {
            "processed_chunks": processed_chunks,
            "cached_chunks": cached_chunks,
            "total_tables": len(tables)
        }
    
    def _store_table_chunks_for_graph_building(self, table_chunks: List['TableChunk']):
        """Store table chunks for later retrieval during graph building"""
        # Store table chunks in a way that _load_table_chunks can retrieve them
        if not hasattr(self, '_processed_table_chunks'):
            self._processed_table_chunks = []
        self._processed_table_chunks.extend(table_chunks)
    
    def _load_table_chunks(self) -> List['TableChunk']:
        """Load TableChunk objects from table processor results"""
        # Return table chunks that were processed and stored
        return getattr(self, '_processed_table_chunks', [])
    
    def _get_chunk_from_cache(self, chunk_id: str) -> Optional['DocumentChunk']:
        """Get DocumentChunk object from cache by ID using Ray remote calls"""
        from loguru import logger
        
        try:
            # Try to get from document processor cache using Ray remote calls
            for i, worker in enumerate(self.document_processor.workers):
                try:
                    chunk_future = worker.get_chunk_by_id.remote(chunk_id)
                    chunk = ray.get(chunk_future, timeout=5.0) # 5 second timeout
                if chunk:
                        logger.debug(f"Found chunk {chunk_id} in worker {i} cache")
                        return chunk # Returns DocumentChunk object directly
                except Exception as e:
                    logger.debug(f"Worker {i} cache lookup failed for {chunk_id}: {e}")
                    continue
            
            logger.debug(f"Chunk {chunk_id} not found in any worker cache")
            return None
        except Exception as e:
            logger.warning(f"Could not retrieve chunk {chunk_id} from cache: {e}")
            return None

    def _load_content_from_json(self, json_file: str, start_index: int = 0, end_index: int = None) -> List[Dict[str, Any]]:
        """Load content items (documents and tables) from JSON metadata file"""
        
        content_items = []
        
        try:
            with open(json_file, 'r') as f:
                metadata = json.load(f)
            
            # Extract content items within range
            if isinstance(metadata, list):
                # Direct list of content items
                selected_items = metadata[start_index:end_index]
            elif isinstance(metadata, dict) and "content" in metadata:
                # Content under "content" key
                selected_items = metadata["content"][start_index:end_index]
            else:
                # Try to find content in various formats
                for key in ["data", "items", "entries"]:
                    if key in metadata:
                        selected_items = metadata[key][start_index:end_index]
                        break
                else:
                    logger.error("Could not find content in JSON structure")
                    return []
            
            # Convert to standard format
            for i, item in enumerate(selected_items):
                if isinstance(item, dict):
                    # Ensure required fields
                    content = item.get("content", item.get("text", ""))
                    # Use id first, then title, then file_path as source name
                    source = item.get("id", item.get("title", item.get("file_path", f"item_{start_index + i}")))
                    
                    content_items.append({
                        "content": content,
                        "source": source,
                        "metadata": item
                    })
                else:
                    # Handle simple text content
                    content_items.append({
                        "content": str(item),
                        "source": f"item_{start_index + i}",
                        "metadata": {}
                    })
            
            logger.info(f"Successfully loaded {len(content_items)} content items")
            return content_items
            
        except Exception as e:
            logger.error(f"Failed to load content from {json_file}: {e}")
            return []
    
    def _build_incremental_graph(self, chunks: List[Union[str, Dict[str, Any], Any]]) -> Dict[str, Any]:
        """Build full KnowledgeGraph from all chunks using HNSW FAISS and parallel processing"""
        
        logger.info("Building full KnowledgeGraph with HNSW FAISS and parallel processing...")
        
        try:
            # Wait for all chunking to complete first
            logger.info("Waiting for all chunking to complete...")
            self._wait_for_chunking_completion()
            
            # Step 1: Load all chunks from cache directories as proper chunk objects
            logger.info("Step 1: Loading all chunks as proper DocumentChunk/TableChunk objects...")
            load_start = time.time()
            
            all_chunks = self._load_all_chunks_as_objects()
            if not all_chunks:
                logger.warning("No chunks found in cache")
                return {"success": False, "error": "No chunks found"}
            
            load_time = time.time() - load_start
            logger.info(f"Loaded {len(all_chunks)} chunk objects in {load_time:.2f}s")
            
            # Step 2: Build HNSW FAISS index from chunk embeddings
            logger.info("Step 2: Building HNSW FAISS index...")
            faiss_start = time.time()
            
            faiss_index, chunk_mappings = self._build_hnsw_faiss_index_from_objects(all_chunks)
            faiss_time = time.time() - faiss_start
            
            logger.info(f"Built HNSW FAISS index in {faiss_time:.2f}s")
            
            # Step 3: Build full KnowledgeGraph using parallel processing with proper edge types
            logger.info("Step 3: Building full KnowledgeGraph with 64 parallel threads...")
            graph_start = time.time()
            
            knowledge_graph = self._build_full_knowledge_graph_parallel(all_chunks, faiss_index, chunk_mappings)
            graph_time = time.time() - graph_start
            
            # Step 4: Save KnowledgeGraph in original format only
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            knowledge_graph_file = self.output_dir / f"knowledge_graph_{timestamp}.json"
            knowledge_graph.export_to_json(str(knowledge_graph_file))
            
            total_time = load_time + faiss_time + graph_time
            
            logger.info(f"Full KnowledgeGraph building completed successfully:")
            logger.info(f" - Chunks loaded: {len(all_chunks)}")
            logger.info(f" - Nodes: {len(knowledge_graph.nodes)}")
            logger.info(f" - Edges: {len(knowledge_graph.edges)}")
            logger.info(f" - Load time: {load_time:.2f}s")
            logger.info(f" - FAISS build time: {faiss_time:.2f}s")
            logger.info(f" - Graph build time: {graph_time:.2f}s")
            logger.info(f" - Total time: {total_time:.2f}s")
            logger.info(f" - KnowledgeGraph saved to: {knowledge_graph_file}")
            
            return {
                "success": True,
                "knowledge_graph": knowledge_graph,
                "load_time": load_time,
                "faiss_build_time": faiss_time,
                "graph_build_time": graph_time,
                "total_time": total_time,
                "chunks_loaded": len(all_chunks),
                "knowledge_graph_file": str(knowledge_graph_file)
            }
            
        except Exception as e:
            logger.error(f"Graph building failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _wait_for_chunking_completion(self):
        """Wait for all chunking operations to complete"""
        import time
        
        logger.info("Ensuring all chunking operations are complete...")
        
        # Give a moment for any final cache saves
        time.sleep(2)
        
        # Check if document processor is still active
        if hasattr(self, 'document_processor') and self.document_processor:
            try:
                # Get stats to ensure workers are idle
                stats = self.document_processor.get_worker_stats()
                logger.info(f"Document processor stats: {stats.get('global_stats', {})}")
            except Exception as e:
                logger.debug(f"Could not get worker stats: {e}")
        
        logger.info("Chunking completion check done")
    
    def _load_all_chunks_as_objects(self) -> List[Union['DocumentChunk', 'TableChunk']]:
        """Load all chunks from cache directories as proper DocumentChunk/TableChunk objects"""
        from src.core.models import DocumentChunk, TableChunk, SourceInfo, ChunkType
        
        all_chunks = []
        
        # Load document chunks from GPU cache directories
        for gpu_id in range(self.num_gpus):
            cache_dir = self.output_dir / f"chunks_cache_gpu_{gpu_id}"
            if cache_dir.exists():
                chunks_from_dir = self._load_document_chunks_from_cache_dir(str(cache_dir))
                all_chunks.extend(chunks_from_dir)
                logger.info(f"Loaded {len(chunks_from_dir)} document chunks from {cache_dir}")
            else:
                logger.info(f"Cache directory {cache_dir} does not exist, skipping")
        
        # Add document chunks from current processing run
        current_doc_chunks = getattr(self, '_processed_document_chunks', [])
        all_chunks.extend(current_doc_chunks)
        if current_doc_chunks:
            logger.info(f"Added {len(current_doc_chunks)} document chunks from current processing run")
        
        # Add table chunks from current processing run
        current_table_chunks = getattr(self, '_processed_table_chunks', [])
        all_chunks.extend(current_table_chunks)
        if current_table_chunks:
            logger.info(f"Added {len(current_table_chunks)} table chunks from current processing run")
        
        logger.info(f"Total chunks loaded: {len(all_chunks)}")
        return all_chunks
    
    def _load_document_chunks_from_cache_dir(self, cache_dir: str) -> List['DocumentChunk']:
        """Load DocumentChunk objects from a specific cache directory"""
        from pathlib import Path
        import json
        from src.core.models import DocumentChunk, SourceInfo, ChunkType
        
        chunks = []
        cache_path = Path(cache_dir)
        
        if not cache_path.exists():
            return chunks
        
        # Load chunk index
        index_file = cache_path / "chunk_index.json"
        if not index_file.exists():
            return chunks
        
        try:
            with open(index_file, 'r') as f:
                chunk_index = json.load(f)
            
            # Load individual chunks
            for chunk_key, chunk_info in chunk_index.items():
                chunk_file = cache_path / chunk_info["file"]
                if chunk_file.exists():
                    try:
                        with open(chunk_file, 'r') as f:
                            chunk_data = json.load(f)
                        
                        # Check if it's new format (DocumentChunk) or old format (ChunkEmbedding)
                        if chunk_data.get("chunk_type") == "document" and "source_info" in chunk_data:
                            # New format: Convert back to DocumentChunk
                            source_info_data = chunk_data["source_info"]
                            source_info = SourceInfo(**source_info_data)
                            
                            chunk = DocumentChunk(
                                chunk_id=chunk_data["chunk_id"],
                                content=chunk_data["content"],
                                source_info=source_info,
                                sentences=chunk_data.get("sentences", []),
                                keywords=chunk_data.get("keywords", []),
                                summary=chunk_data.get("summary", ""),
                                embedding=chunk_data.get("embedding"),
                                merged_sentence_count=chunk_data.get("merged_sentence_count", 1)
                            )
                            chunks.append(chunk)
                        elif chunk_data.get("embedding"):
                            # Old format: Convert ChunkEmbedding to DocumentChunk
                            source_info = SourceInfo(
                                source_id=chunk_data.get("source_document", "unknown"),
                                source_name=chunk_data.get("source_document", "unknown"),
                                source_type=ChunkType.DOCUMENT,
                                file_path="",
                                content=chunk_data.get("content", "")
                            )
                            
                            chunk = DocumentChunk(
                                chunk_id=chunk_data["chunk_id"],
                                content=chunk_data["content"],
                                source_info=source_info,
                                sentences=[], # Not available in old format
                                keywords=[], # Not available in old format
                                summary=chunk_data["content"][:200] + "..." if len(chunk_data["content"]) > 200 else chunk_data["content"],
                                embedding=chunk_data["embedding"],
                                merged_sentence_count=1
                            )
                            chunks.append(chunk)
                    except Exception as e:
                        logger.warning(f"Error loading chunk from {chunk_file}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error loading chunk index from {cache_dir}: {e}")
        
        return chunks
    
    def _build_hnsw_faiss_index_from_objects(self, chunks: List[Union['DocumentChunk', 'TableChunk']]) -> tuple:
        """Build HNSW FAISS index from chunk objects' embeddings"""
        import numpy as np
        import faiss
        
        embeddings = []
        chunk_id_to_index = {}
        index_to_chunk = {}
        
        # Extract embeddings and build mappings
        valid_chunks = []
        for chunk in chunks:
            if chunk.embedding:
                embeddings.append(chunk.embedding)
                chunk_id = chunk.chunk_id
                index = len(valid_chunks)
                chunk_id_to_index[chunk_id] = index
                index_to_chunk[index] = chunk
                valid_chunks.append(chunk)
        
        if not embeddings:
            raise ValueError("No valid embeddings found in chunks")
        
        # Convert to numpy array
        embeddings_matrix = np.array(embeddings, dtype=np.float32)
        dimension = embeddings_matrix.shape[1]
        
        logger.info(f"Building HNSW FAISS index with {len(embeddings)} embeddings, dimension={dimension}")
        
        # Create HNSW index
        hnsw_index = faiss.IndexHNSWFlat(dimension, 32) # M=32 for balanced performance
        hnsw_index.hnsw.efConstruction = 200 # Higher quality construction
        hnsw_index.hnsw.efSearch = 100 # Search quality
        
        # Use GPU if available
        try:
            if faiss.get_num_gpus() > 0:
                # Normalize for cosine similarity
                faiss.normalize_L2(embeddings_matrix)
                
                res = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(res, 0, hnsw_index)
                
                # Add embeddings to GPU index
                gpu_index.add(embeddings_matrix)
                
                # Move back to CPU for thread safety in parallel processing
                hnsw_index = faiss.index_gpu_to_cpu(gpu_index)
                logger.info("Built HNSW index on GPU and moved to CPU for parallel processing")
            else:
                # Normalize for cosine similarity on CPU
                faiss.normalize_L2(embeddings_matrix)
                hnsw_index.add(embeddings_matrix)
                logger.info("Built HNSW index on CPU")
        except Exception as e:
            logger.warning(f"GPU FAISS failed: {e}, using CPU")
            faiss.normalize_L2(embeddings_matrix)
            hnsw_index.add(embeddings_matrix)
            logger.info("Built HNSW index on CPU")
        
        chunk_mappings = {
            "chunk_id_to_index": chunk_id_to_index,
            "index_to_chunk": index_to_chunk,
            "valid_chunks": valid_chunks
        }
        
        return hnsw_index, chunk_mappings
    
    def _build_full_knowledge_graph_parallel(self, chunks: List[Union['DocumentChunk', 'TableChunk']], 
                                           faiss_index, chunk_mappings: Dict) -> 'KnowledgeGraph':
        """Build full KnowledgeGraph using optimized batch operations with parallel edge discovery"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from src.core.graph import KnowledgeGraph
        from src.core.models import GraphNode, ChunkType
        
        # Configuration
        num_threads = 64 # As requested
        max_neighbors = 200 # As requested
        
        # Initialize KnowledgeGraph
        knowledge_graph = KnowledgeGraph()
        
        # Step 1: Add all chunks as nodes using optimized batch operation (no locks needed!)
        logger.info(f"Adding {len(chunks)} chunks as nodes to KnowledgeGraph using optimized batch operation...")
        
        # Create all GraphNode objects at once
        all_nodes = []
        for chunk in chunks:
            node = GraphNode(
                node_id=chunk.chunk_id,
                chunk=chunk,
                connections=[] # Will be populated by edges
            )
            all_nodes.append(node)
        
        # Single optimized batch operation - no thread locks needed!
        total_nodes_added = knowledge_graph.add_nodes_batch_optimized(all_nodes)
        logger.info(f"Optimized batch node addition completed: {total_nodes_added} total nodes added")
        
        # Step 2: Build edges using parallel processing to discover edges, then batch add them
        valid_chunks = chunk_mappings["valid_chunks"]
        total_chunks = len(valid_chunks)
        chunks_per_thread = max(1, total_chunks // num_threads)
        
        thread_batches = []
        for i in range(0, total_chunks, chunks_per_thread):
            end_idx = min(i + chunks_per_thread, total_chunks)
            thread_batches.append(valid_chunks[i:end_idx])
        
        logger.info(f"Dividing {total_chunks} chunks among {len(thread_batches)} threads for parallel edge discovery")
        
        def discover_edges_batch(thread_id: int, chunk_batch: List[Union['DocumentChunk', 'TableChunk']]) -> List:
            """Discover edges for a batch of chunks (returns edge metadata list, no graph modification)"""
            discovered_edges = []
            
            for chunk in chunk_batch:
                try:
                    # Find neighbors using HNSW FAISS
                    neighbors = self._find_neighbors_hnsw_for_chunk_objects(
                        chunk, faiss_index, chunk_mappings, max_neighbors
                    )
                    
                    # For each neighbor, check if we should create an edge using graph_builder logic
                    for neighbor_chunk, similarity in neighbors:
                        if self._should_create_edge(chunk, neighbor_chunk, similarity):
                            # Create edge using graph_builder logic
                            edge_metadata = self._create_edge_metadata_from_graph_builder(chunk, neighbor_chunk, similarity)
                            if edge_metadata:
                                discovered_edges.append(edge_metadata)
                
                except Exception as e:
                    logger.warning(f"Error processing chunk {chunk.chunk_id}: {e}")
                    continue
            
            return discovered_edges
        
        # Execute parallel edge discovery (no graph modifications in threads)
        all_discovered_edges = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for thread_id, chunk_batch in enumerate(thread_batches):
                future = executor.submit(discover_edges_batch, thread_id, chunk_batch)
                futures.append(future)
            
            # Collect all discovered edges
            completed = 0
            for future in as_completed(futures):
                try:
                    thread_edges = future.result()
                    all_discovered_edges.extend(thread_edges)
                    completed += 1
                    logger.info(f"Edge discovery thread completed: {len(thread_edges)} edges discovered "
                               f"({completed}/{len(futures)} threads done)")
                except Exception as e:
                    logger.error(f"Thread failed: {e}")
                    completed += 1
        
        logger.info(f"Parallel edge discovery completed: {len(all_discovered_edges)} total edges discovered")
        
        # Step 3: Add all edges using optimized batch operation (no locks needed!)
        logger.info(f"Adding {len(all_discovered_edges)} edges using optimized batch operation...")
        total_edges_added = knowledge_graph.add_edges_batch_optimized(all_discovered_edges)
        logger.info(f"Optimized batch edge addition completed: {total_edges_added} total edges added")
        
        return knowledge_graph
    
    def _find_neighbors_hnsw_for_chunk_objects(self, chunk: Union['DocumentChunk', 'TableChunk'], 
                                              faiss_index, chunk_mappings: Dict, k: int) -> List[tuple]:
        """Find k nearest neighbors for a chunk using HNSW FAISS"""
        import numpy as np
        import faiss
        
        chunk_id = chunk.chunk_id
        if chunk_id not in chunk_mappings["chunk_id_to_index"]:
            return []
        
        try:
            embedding = np.array([chunk.embedding], dtype=np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embedding)
            
            # Search for k+1 neighbors (including self)
            search_k = min(k + 1, faiss_index.ntotal)
            similarities, indices = faiss_index.search(embedding, search_k)
            
            neighbors = []
            for sim, idx in zip(similarities[0], indices[0]):
                if idx == -1: # Invalid index
                    continue
                
                neighbor_chunk = chunk_mappings["index_to_chunk"][idx]
                
                # Skip self-comparison
                if neighbor_chunk.chunk_id == chunk_id:
                    continue
                
                neighbors.append((neighbor_chunk, float(sim)))
            
            return neighbors[:k] # Return only k neighbors
            
        except Exception as e:
            logger.warning(f"Error finding neighbors for {chunk_id}: {e}")
            return []
    
    def _should_create_edge(self, chunk1: Union['DocumentChunk', 'TableChunk'], 
                          chunk2: Union['DocumentChunk', 'TableChunk'], similarity: float) -> bool:
        """Determine if an edge should be created based on chunk types and similarity thresholds"""
        from src.core.models import ChunkType
        
        # Get chunk types
        type1 = chunk1.source_info.source_type
        type2 = chunk2.source_info.source_type
        
        # Determine appropriate threshold based on chunk types (same logic as graph_builder.py)
        if type1 == ChunkType.TABLE and type2 == ChunkType.TABLE:
            threshold = self.config.table_similarity_threshold
        elif (type1 == ChunkType.TABLE and type2 == ChunkType.DOCUMENT) or \
             (type1 == ChunkType.DOCUMENT and type2 == ChunkType.TABLE):
            threshold = self.config.table_similarity_threshold # Use table threshold for mixed connections
        elif type1 == ChunkType.DOCUMENT and type2 == ChunkType.DOCUMENT:
            threshold = self.config.sentence_similarity_threshold
        else:
            # Fallback
            threshold = self.config.sentence_similarity_threshold
        
        return similarity >= threshold
    
    def _create_edge_metadata_from_graph_builder(self, chunk1: Union['DocumentChunk', 'TableChunk'], 
                                                chunk2: Union['DocumentChunk', 'TableChunk'], 
                                                similarity: float):
        """Create edge metadata using the same logic as graph_builder.py"""
        import uuid
        from src.core.models import (
            ChunkType, BaseEdgeMetadata, DocumentToDocumentEdgeMetadata,
            TableToTableEdgeMetadata, TableToDocumentEdgeMetadata, EdgeType
        )
        
        # Determine edge type
        type1 = chunk1.source_info.source_type
        type2 = chunk2.source_info.source_type
        
        # Get entity matches from precomputed data (more efficient than recalculating)
        shared_keywords = []
        # Note: In a real implementation, we would have access to precomputed entity matches
        # For now, fallback to keywords intersection
        shared_keywords = list(set(chunk1.keywords) & set(chunk2.keywords))
        
        if type1 == ChunkType.TABLE and type2 == ChunkType.TABLE:
            return self._create_table_to_table_edge_metadata(chunk1, chunk2, similarity, shared_keywords)
        elif (type1 == ChunkType.TABLE and type2 == ChunkType.DOCUMENT) or \
             (type1 == ChunkType.DOCUMENT and type2 == ChunkType.TABLE):
            return self._create_table_to_document_edge_metadata(chunk1, chunk2, similarity, shared_keywords)
        elif type1 == ChunkType.DOCUMENT and type2 == ChunkType.DOCUMENT:
            return self._create_document_to_document_edge_metadata(chunk1, chunk2, similarity, shared_keywords)
        
        return None
    
    def _create_table_to_table_edge_metadata(self, chunk1, chunk2, similarity: float, shared_keywords: List[str]):
        """Create table-to-table edge metadata (same logic as graph_builder.py)"""
        import uuid
        from src.core.models import TableToTableEdgeMetadata
        
        # Calculate column similarity
        columns1 = set(chunk1.column_headers)
        columns2 = set(chunk2.column_headers)
        column_similarity = len(columns1 & columns2) / len(columns1 | columns2) if (columns1 | columns2) else 0
        
        # Calculate row-level overlap
        row_overlap = self._calculate_row_overlap(chunk1, chunk2)
        
        # Create schema context
        schema_context = {
            "chunk1_columns": chunk1.column_headers,
            "chunk2_columns": chunk2.column_headers,
            "common_columns": list(columns1 & columns2),
            "chunk1_table": chunk1.source_info.source_name,
            "chunk2_table": chunk2.source_info.source_name,
            "threshold_used": self.config.table_similarity_threshold
        }
        
        return TableToTableEdgeMetadata(
            edge_id=str(uuid.uuid4()),
            source_chunk_id=chunk1.chunk_id,
            target_chunk_id=chunk2.chunk_id,
            semantic_similarity=similarity,
            shared_keywords=shared_keywords,
            column_similarity=column_similarity,
            row_overlap=row_overlap,
            schema_context=schema_context,
            title_similarity=0.0, # TODO: Calculate actual title similarity
            description_similarity=0.0 # TODO: Calculate actual description similarity
        )
    
    def _create_table_to_document_edge_metadata(self, chunk1, chunk2, similarity: float, shared_keywords: List[str]):
        """Create table-to-document edge metadata (same logic as graph_builder.py)"""
        import uuid
        from src.core.models import TableToDocumentEdgeMetadata, ChunkType
        
        # Ensure chunk1 is table and chunk2 is document
        if chunk1.source_info.source_type == ChunkType.DOCUMENT:
            chunk1, chunk2 = chunk2, chunk1
        
        table_chunk = chunk1
        doc_chunk = chunk2
        
        # Find row references in document text
        row_refs = self._find_row_references_in_text(table_chunk, doc_chunk.content)
        
        # Find column references in document text
        column_refs = self._find_column_references_in_text(table_chunk, doc_chunk.content)
        
        return TableToDocumentEdgeMetadata(
            edge_id=str(uuid.uuid4()),
            source_chunk_id=chunk1.chunk_id,
            target_chunk_id=chunk2.chunk_id,
            semantic_similarity=similarity,
            shared_keywords=shared_keywords,
            row_references=row_refs,
            column_references=column_refs,
            topic_title_similarity=0.0, # TODO: Calculate actual topic-title similarity
            topic_summary_similarity=0.0 # TODO: Calculate actual topic-summary similarity
        )
    
    def _create_document_to_document_edge_metadata(self, chunk1, chunk2, similarity: float, shared_keywords: List[str]):
        """Create document-to-document edge metadata (same logic as graph_builder.py)"""
        import uuid
        from src.core.models import DocumentToDocumentEdgeMetadata
        
        # Calculate topic overlap as similarity for now
        topic_overlap = similarity
        
        return DocumentToDocumentEdgeMetadata(
            edge_id=str(uuid.uuid4()),
            source_chunk_id=chunk1.chunk_id,
            target_chunk_id=chunk2.chunk_id,
            semantic_similarity=similarity,
            shared_keywords=shared_keywords,
            topic_overlap=topic_overlap
        )
    
    def _calculate_row_overlap(self, chunk1, chunk2) -> float:
        """Calculate overlap between rows of two table chunks (same logic as graph_builder.py)"""
        
        if not hasattr(chunk1, 'rows_with_headers') or not hasattr(chunk2, 'rows_with_headers'):
            return 0.0
        
        if not chunk1.rows_with_headers or not chunk2.rows_with_headers:
            return 0.0
        
        # Convert rows to comparable format
        rows1 = [str(row) for row in chunk1.rows_with_headers]
        rows2 = [str(row) for row in chunk2.rows_with_headers]
        
        # Calculate Jaccard similarity
        set1 = set(rows1)
        set2 = set(rows2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _find_row_references_in_text(self, table_chunk, text: str) -> List[str]:
        """Find references to table rows in document text (same logic as graph_builder.py)"""
        
        references = []
        text_lower = text.lower()
        
        if not hasattr(table_chunk, 'rows_with_headers'):
            return references
        
        # Look for specific values that appear in table rows
        for row in table_chunk.rows_with_headers:
            for key, value in row.items():
                value_str = str(value).lower()
                if len(value_str) > 2 and value_str in text_lower:
                    references.append(f"{key}: {value}")
        
        return references[:10] # Limit to top 10 references
    
    def _find_column_references_in_text(self, table_chunk, text: str) -> List[str]:
        """Find references to table columns in document text (same logic as graph_builder.py)"""
        
        references = []
        text_lower = text.lower()
        
        if not hasattr(table_chunk, 'column_headers'):
            return references
        
        # Look for column headers mentioned in text
        for header in table_chunk.column_headers:
            header_lower = header.lower()
            if header_lower in text_lower:
                references.append(header)
        
        return references
    
    def _build_proper_knowledge_graph(self, all_chunks_data: List[Dict[str, Any]], simplified_graph: Dict[str, Any]) -> 'KnowledgeGraph':
        """DEPRECATED: This method is no longer used as we build the full KnowledgeGraph directly"""
        # This method is deprecated and replaced by _build_full_knowledge_graph_parallel
        pass
    
    def _save_comprehensive_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Save comprehensive results including document, table, and graph data"""
        
        output_file = self.output_dir / f"pipeline_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            # Save to JSON file
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Comprehensive results saved to: {output_file}")
            
            return {
                "success": True,
                "file_path": str(output_file),
                "data": data
            }
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return {"success": False, "error": str(e)}
    
    def shutdown(self):
        """Gracefully shutdown all pipeline components"""
        from loguru import logger
        
        logger.info("Shutting down integrated pipeline...")
        
        try:
            # self.faiss_indexer.stop_monitoring()
            self.document_processor.shutdown()
            logger.info("Pipeline shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

def main():
    """Main entry point for the integrated pipeline"""
    
    parser = argparse.ArgumentParser(description="Integrated Document and Table Processing Pipeline")
    parser.add_argument("--json-file", required=True, help="JSON file containing documents and tables")
    parser.add_argument("--start-index", type=int, default=0, help="Starting item index")
    parser.add_argument("--end-index", type=int, help="Ending item index")
    parser.add_argument("--num-gpus", type=int, help="Number of GPUs to use")
    parser.add_argument("--batch-size", type=int, help="Batch size per GPU")
    parser.add_argument("--similarity-threshold", type=float, default=0.8, 
                       help="Similarity threshold for chunk merging")
    parser.add_argument("--graph-threshold", type=float, default=0.3,
                       help="Similarity threshold for graph edges")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level=args.log_level)
    logger.add(f"{args.output_dir}/pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log", 
               level="DEBUG", rotation="100 MB")
    
    # Initialize pipeline
    pipeline = IntegratedPipeline(
        num_gpus=args.num_gpus,
        similarity_threshold=args.similarity_threshold,
        graph_similarity_threshold=args.graph_threshold,
        output_dir=args.output_dir
    )
    
    try:
        # Run pipeline
        results = pipeline.process_mixed_content_from_json(
            json_file=args.json_file,
            start_index=args.start_index,
            end_index=args.end_index,
            batch_size=args.batch_size
        )
        
        if results.get("success"):
            logger.info("Pipeline completed successfully!")
            return 0
        else:
            logger.error("Pipeline failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        return 1
        
    finally:
        pipeline.shutdown()

if __name__ == "__main__":
    exit(main())
