"""
Main RAG Pipeline orchestrator that coordinates all components
"""

import os
import json
from typing import List, Dict, Any, Union
from pathlib import Path
from loguru import logger

from src.core.models import ProcessingConfig, DocumentChunk, TableChunk, SourceInfo, ChunkType, DatasetConfig
from src.core.graph import KnowledgeGraph
from src.processors import DocumentProcessor, TableProcessor, EmbeddingService, GraphBuilder
from src.processors.graph_builder_faiss import FAISSGraphBuilder


class RAGPipeline:
    """
    Main pipeline orchestrator that coordinates:
    1. Document and table processing
    2. Graph building 
    3. Provides interface for querying
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.embedding_service = EmbeddingService(config)
        self.document_processor = DocumentProcessor(config, self.embedding_service)
        self.table_processor = TableProcessor(config, self.embedding_service)
        
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
        
        logger.info("RAG Pipeline initialized")
    
    def process_from_json_metadata(self, dataset_config: DatasetConfig) -> KnowledgeGraph:
        """
        Process data from JSON metadata file (like OTT-QA dataset)
        
        Args:
            dataset_config: Configuration for dataset processing
            
        Returns:
            KnowledgeGraph built from processed chunks
        """
        logger.info(f"Processing dataset: {dataset_config.dataset_name}")
        
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
        
        all_chunks = []
        
        for i, metadata in enumerate(metadata_list):
            try:
                # Create SourceInfo from metadata
                source_info = SourceInfo(
                    source_id=metadata['id'],
                    source_name=metadata['title'],
                    source_type=ChunkType(metadata['source_type']),
                    file_path=os.path.join(dataset_config.dataset_path, metadata['file_path']),
                    structural_link=metadata.get('structural_link', []),
                    original_source=metadata.get('original_source', ''),
                    additional_information=metadata.get('additional_information', ''),
                    content=metadata.get('content', None) # Add embedded content
                )
                
                # For documents, check if content is embedded; for tables, check file existence
                if source_info.source_type == ChunkType.DOCUMENT:
                    if source_info.content:
                        logger.info(f"Processing document with embedded content: {source_info.source_name}")
                        chunks = self.document_processor.process(source_info)
                    elif os.path.exists(source_info.file_path):
                        logger.info(f"Processing document from file: {source_info.file_path}")
                        chunks = self.document_processor.process(source_info)
                    else:
                        logger.warning(f"No content or file found for document: {source_info.source_name}, skipping")
                        continue
                elif source_info.source_type == ChunkType.TABLE:
                    if not os.path.exists(source_info.file_path):
                        logger.warning(f"File not found: {source_info.file_path}, skipping")
                        continue
                    chunks = self.table_processor.process(source_info)
                else:
                    logger.warning(f"Unknown source type: {source_info.source_type}")
                    continue
                
                all_chunks.extend(chunks)
                logger.info(f"Processed {source_info.source_name}: {len(chunks)} chunks (item {i+1}/{len(metadata_list)})")
                
                # Process in batches if chunk_size is specified
                if (dataset_config.chunk_size and 
                    len(all_chunks) >= dataset_config.chunk_size):
                    logger.info(f"Reached chunk size limit of {dataset_config.chunk_size}, stopping")
                    break
                
            except Exception as e:
                logger.error(f"Error processing {metadata.get('id', 'unknown')}: {e}")
                continue
        
        if not all_chunks:
            logger.warning("No chunks were created from the dataset")
            return KnowledgeGraph()
        
        # Build knowledge graph
        logger.info(f"Building knowledge graph from {len(all_chunks)} total chunks")
        self.knowledge_graph = self.graph_builder.build_graph(all_chunks)
        
        return self.knowledge_graph
    
    def process_directory(self, data_directory: str) -> KnowledgeGraph:
        """
        Process all documents and tables in a directory
        
        Args:
            data_directory: Path to directory containing .txt and .csv files
            
        Returns:
            KnowledgeGraph built from processed chunks
        """
        data_path = Path(data_directory)
        if not data_path.exists():
            raise ValueError(f"Directory {data_directory} does not exist")
        
        logger.info(f"Processing directory: {data_directory}")
        
        all_chunks = []
        
        # Process text documents
        txt_files = list(data_path.glob("*.txt"))
        logger.info(f"Found {len(txt_files)} text files")
        
        for txt_file in txt_files:
            try:
                # Create SourceInfo for backward compatibility
                source_info = SourceInfo(
                    source_id=str(txt_file.stem),
                    source_name=txt_file.name,
                    source_type=ChunkType.DOCUMENT,
                    file_path=str(txt_file)
                )
                chunks = self.document_processor.process(source_info)
                all_chunks.extend(chunks)
                logger.info(f"Processed {txt_file.name}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Error processing {txt_file.name}: {e}")
        
        # Process CSV tables
        csv_files = list(data_path.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            try:
                # Create SourceInfo for backward compatibility
                source_info = SourceInfo(
                    source_id=str(csv_file.stem),
                    source_name=csv_file.name,
                    source_type=ChunkType.TABLE,
                    file_path=str(csv_file)
                )
                chunks = self.table_processor.process(source_info)
                all_chunks.extend(chunks)
                logger.info(f"Processed {csv_file.name}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Error processing {csv_file.name}: {e}")
        
        if not all_chunks:
            logger.warning("No chunks were created from the input directory")
            return KnowledgeGraph()
        
        # Build knowledge graph
        logger.info(f"Building knowledge graph from {len(all_chunks)} total chunks")
        self.knowledge_graph = self.graph_builder.build_graph(all_chunks)
        
        return self.knowledge_graph
    
    def process_files(self, file_paths: List[str]) -> KnowledgeGraph:
        """
        Process specific files
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            KnowledgeGraph built from processed chunks
        """
        logger.info(f"Processing {len(file_paths)} files")
        
        all_chunks = []
        
        for file_path in file_paths:
            file_path_obj = Path(file_path)
            
            if not file_path_obj.exists():
                logger.warning(f"File {file_path} does not exist, skipping")
                continue
            
            try:
                if file_path_obj.suffix.lower() == '.txt':
                    source_info = SourceInfo(
                        source_id=str(file_path_obj.stem),
                        source_name=file_path_obj.name,
                        source_type=ChunkType.DOCUMENT,
                        file_path=file_path
                    )
                    chunks = self.document_processor.process(source_info)
                    all_chunks.extend(chunks)
                elif file_path_obj.suffix.lower() == '.csv':
                    source_info = SourceInfo(
                        source_id=str(file_path_obj.stem),
                        source_name=file_path_obj.name,
                        source_type=ChunkType.TABLE,
                        file_path=file_path
                    )
                    chunks = self.table_processor.process(source_info)
                    all_chunks.extend(chunks)
                else:
                    logger.warning(f"Unsupported file type: {file_path}")
                    continue
                
                logger.info(f"Processed {file_path_obj.name}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        if not all_chunks:
            logger.warning("No chunks were created from the input files")
            return KnowledgeGraph()
        
        # Build knowledge graph
        logger.info(f"Building knowledge graph from {len(all_chunks)} total chunks")
        self.knowledge_graph = self.graph_builder.build_graph(all_chunks)
        
        return self.knowledge_graph
    
    def query_by_keywords(self, keywords: List[str], min_matches: int = 1) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph by keywords
        
        Args:
            keywords: List of keywords to search for
            min_matches: Minimum number of keyword matches required
            
        Returns:
            List of matching node information
        """
        if not self.knowledge_graph:
            raise ValueError("No knowledge graph available. Process data first.")
        
        matching_node_ids = self.knowledge_graph.query_by_keywords(keywords, min_matches)
        
        results = []
        for node_id in matching_node_ids:
            node = self.knowledge_graph.get_node_metadata(node_id)
            if node:
                results.append({
                    "node_id": node_id,
                    "chunk_type": node.chunk.source_info.source_type.value,
                    "source_name": node.chunk.source_info.source_name,
                    "keywords": node.chunk.keywords,
                    "summary": node.chunk.summary,
                    "connections": len(node.connections)
                })
        
        return results
    
    def query_by_text(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph by text similarity
        
        Args:
            query_text: Text query
            top_k: Number of top results to return
            
        Returns:
            List of most similar node information with similarity scores
        """
        if not self.knowledge_graph:
            raise ValueError("No knowledge graph available. Process data first.")
        
        # Generate embedding for query
        query_embedding = self.embedding_service.generate_single_embedding(query_text)
        
        # Find similar nodes
        similar_nodes = self.knowledge_graph.query_by_similarity(query_embedding, top_k)
        
        results = []
        for node_id, similarity in similar_nodes:
            node = self.knowledge_graph.get_node_metadata(node_id)
            if node:
                results.append({
                    "node_id": node_id,
                    "similarity": similarity,
                    "chunk_type": node.chunk.source_info.source_type.value,
                    "source_name": node.chunk.source_info.source_name,
                    "content_preview": (node.chunk.content[:200] + "..." 
                                      if hasattr(node.chunk, 'content') and len(node.chunk.content) > 200
                                      else getattr(node.chunk, 'content', node.chunk.summary)),
                    "keywords": node.chunk.keywords,
                    "connections": len(node.connections)
                })
        
        return results
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        if not self.knowledge_graph:
            return {"message": "No knowledge graph available"}
        
        return self.knowledge_graph.get_graph_statistics()
    
    def visualize_graph(self, **kwargs) -> Any:
        """Create interactive visualization of the knowledge graph"""
        if not self.knowledge_graph:
            raise ValueError("No knowledge graph available. Process data first.")
        
        return self.knowledge_graph.create_interactive_visualization(**kwargs)
    
    def export_graph(self, filepath: str):
        """Export the knowledge graph to JSON"""
        if not self.knowledge_graph:
            raise ValueError("No knowledge graph available. Process data first.")
        
        self.knowledge_graph.export_to_json(filepath)
        logger.info(f"Graph exported to {filepath}")
    
    def get_node_details(self, node_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific node"""
        if not self.knowledge_graph:
            raise ValueError("No knowledge graph available. Process data first.")
        
        node = self.knowledge_graph.get_node_metadata(node_id)
        if not node:
            return {"error": f"Node {node_id} not found"}
        
        chunk = node.chunk
        details = {
            "node_id": node_id,
            "chunk_type": chunk.source_info.source_type.value,
            "source_info": chunk.source_info.dict(),
            "keywords": chunk.keywords,
            "summary": chunk.summary,
            "connections": node.connections,
            "neighbor_count": len(node.connections)
        }
        
        # Add type-specific details
        if hasattr(chunk, 'content'): # Document chunk
            details["content"] = chunk.content
            details["sentences"] = chunk.sentences
        else: # Table chunk
            details["rows"] = chunk.rows
            details["column_headers"] = chunk.column_headers
            details["table_description"] = chunk.table_description
            details["column_descriptions"] = chunk.column_descriptions
        
        return details
    
    def find_path(self, source_node_id: str, target_node_id: str) -> Dict[str, Any]:
        """Find path between two nodes in the graph"""
        if not self.knowledge_graph:
            raise ValueError("No knowledge graph available. Process data first.")
        
        path = self.knowledge_graph.find_shortest_path(source_node_id, target_node_id)
        
        if path is None:
            return {"message": f"No path found between {source_node_id} and {target_node_id}"}
        
        # Get details for each node in the path
        path_details = []
        for node_id in path:
            node = self.knowledge_graph.get_node_metadata(node_id)
            if node:
                path_details.append({
                    "node_id": node_id,
                    "source_name": node.chunk.source_info.source_name,
                    "chunk_type": node.chunk.source_info.source_type.value,
                    "summary": node.chunk.summary
                })
        
        return {
            "path_length": len(path),
            "path": path,
            "path_details": path_details
        }