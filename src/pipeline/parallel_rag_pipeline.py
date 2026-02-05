"""
Parallel RAG Pipeline with multi-threading for document processing
"""

import os
import json
from typing import List, Dict, Any, Union
from pathlib import Path
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue

from src.core.models import ProcessingConfig, DocumentChunk, TableChunk, SourceInfo, ChunkType, DatasetConfig
from src.core.graph import KnowledgeGraph
from src.processors import DocumentProcessor, TableProcessor, EmbeddingService, GraphBuilder
from src.processors.graph_builder_faiss import FAISSGraphBuilder


class ParallelRAGPipeline:
    """
    Parallel RAG Pipeline that processes documents/tables in parallel threads
    """
    
    def __init__(self, config: ProcessingConfig, max_workers: int = 8):
        self.config = config
        self.max_workers = max_workers
        
        # Create shared embedding service
        self.embedding_service = EmbeddingService(config)
        
        # Create thread-local processors (each thread gets its own)
        self.local = threading.local()
        
        # Choose graph builder based on configuration
        if config.use_faiss:
            self.graph_builder = FAISSGraphBuilder(config, self.embedding_service, 
                                                 max_neighbors=config.max_neighbors,
                                                 use_gpu=config.faiss_use_gpu)
            logger.info("Using FAISS-accelerated graph builder")
        else:
            self.graph_builder = GraphBuilder(config, self.embedding_service)
            logger.info("Using standard graph builder")
            
        self.knowledge_graph: KnowledgeGraph = None
        
        logger.info(f"Parallel RAG Pipeline initialized with {max_workers} workers")
    
    def _get_thread_local_processors(self):
        """Get thread-local processors for the current thread"""
        if not hasattr(self.local, 'processors'):
            # Each thread gets its own processors but shares the embedding service
            self.local.processors = {
                'document': DocumentProcessor(self.config, self.embedding_service),
                'table': TableProcessor(self.config, self.embedding_service)
            }
        return self.local.processors
    
    def _process_single_item(self, metadata: Dict[str, Any], dataset_config: DatasetConfig) -> List[Union[DocumentChunk, TableChunk]]:
        """
        Process a single metadata item (document or table)
        This function runs in parallel threads
        """
        try:
            processors = self._get_thread_local_processors()
            
            # Create SourceInfo from metadata
            source_info = SourceInfo(
                source_id=metadata['id'],
                source_name=metadata['title'],
                source_type=ChunkType(metadata['source_type']),
                file_path=os.path.join(dataset_config.dataset_path, metadata['file_path']),
                structural_link=metadata.get('structural_link', []),
                original_source=metadata.get('original_source', ''),
                additional_information=metadata.get('additional_information', ''),
                content=metadata.get('content', None)
            )
            
            # Process based on type
            if source_info.source_type == ChunkType.DOCUMENT:
                if source_info.content or os.path.exists(source_info.file_path):
                    chunks = processors['document'].process(source_info)
                    logger.info(f"[Thread] Processed document {source_info.source_name}: {len(chunks)} chunks")
                    return chunks
                else:
                    logger.warning(f"[Thread] No content/file for document: {source_info.source_name}")
                    return []
            elif source_info.source_type == ChunkType.TABLE:
                if os.path.exists(source_info.file_path):
                    chunks = processors['table'].process(source_info)
                    logger.info(f"[Thread] Processed table {source_info.source_name}: {len(chunks)} chunks")
                    return chunks
                else:
                    logger.warning(f"[Thread] File not found: {source_info.file_path}")
                    return []
            else:
                logger.warning(f"[Thread] Unknown source type: {source_info.source_type}")
                return []
                
        except Exception as e:
            logger.error(f"[Thread] Error processing {metadata.get('id', 'unknown')}: {e}")
            return []
    
    def process_from_json_metadata(self, dataset_config: DatasetConfig) -> KnowledgeGraph:
        """
        Process data from JSON metadata file using parallel processing
        
        Args:
            dataset_config: Configuration for dataset processing
            
        Returns:
            KnowledgeGraph built from processed chunks
        """
        logger.info(f"Processing dataset: {dataset_config.dataset_name} with {self.max_workers} parallel workers")
        
        # Load metadata
        with open(dataset_config.metadata_file, 'r', encoding='utf-8') as f:
            metadata_list = json.load(f)
        
        logger.info(f"Loaded {len(metadata_list)} metadata entries")
        
        # Apply filtering and slicing
        if dataset_config.filter_source_type:
            metadata_list = [
                item for item in metadata_list 
                if item.get('source_type') == dataset_config.filter_source_type.value
            ]
            logger.info(f"Filtered to {len(metadata_list)} entries of type {dataset_config.filter_source_type.value}")
        
        # Apply index slicing
        start_idx = dataset_config.start_index or 0
        end_idx = dataset_config.end_index or len(metadata_list)
        metadata_list = metadata_list[start_idx:end_idx]
        
        logger.info(f"Processing entries {start_idx} to {end_idx-1} ({len(metadata_list)} total)")
        
        # Process items in parallel
        all_chunks = []
        processed_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_metadata = {
                executor.submit(self._process_single_item, metadata, dataset_config): metadata
                for metadata in metadata_list
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_metadata):
                metadata = future_to_metadata[future]
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                    processed_count += 1
                    
                    if processed_count % 50 == 0:
                        logger.info(f"Parallel processing: {processed_count}/{len(metadata_list)} items completed, {len(all_chunks)} total chunks")
                    
                    # Check chunk size limit
                    if (dataset_config.chunk_size and 
                        len(all_chunks) >= dataset_config.chunk_size):
                        logger.info(f"Reached chunk size limit of {dataset_config.chunk_size}, stopping")
                        # Cancel remaining futures
                        for f in future_to_metadata:
                            f.cancel()
                        break
                        
                except Exception as e:
                    logger.error(f"Error processing {metadata.get('id', 'unknown')} in parallel: {e}")
                    processed_count += 1
        
        if not all_chunks:
            logger.warning("No chunks were created from the dataset")
            return KnowledgeGraph()
        
        logger.info(f"Parallel processing completed: {len(all_chunks)} total chunks from {processed_count} items")
        
        # Build knowledge graph (this is still sequential but much faster with FAISS)
        logger.info(f"Building knowledge graph from {len(all_chunks)} total chunks")
        self.knowledge_graph = self.graph_builder.build_graph(all_chunks)
        
        return self.knowledge_graph
