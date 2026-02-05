"""
FAISS-accelerated graph builder for constructing knowledge graph from chunks
This version uses GPU-accelerated similarity search to avoid O(N²) comparisons
"""

import uuid
import re
import faiss
import numpy as np
from typing import List, Union, Tuple, Dict, Any, Optional
from loguru import logger
import os
import json

from .embedding_service import EmbeddingService
from src.core.models import (
    DocumentChunk, TableChunk, GraphNode, ProcessingConfig,
    BaseEdgeMetadata, TableToTableEdgeMetadata, 
    TableToDocumentEdgeMetadata, DocumentToDocumentEdgeMetadata,
    EdgeType, ChunkType
)
from src.core.graph import KnowledgeGraph


class FAISSGraphBuilder:
    """
    FAISS-accelerated graph builder that uses GPU-based similarity search
    to efficiently build knowledge graphs from large numbers of chunks
    """
    
    def __init__(self, config: ProcessingConfig, embedding_service: EmbeddingService, 
                 max_neighbors: int = 150, use_gpu: bool = True):
        self.config = config
        self.embedding_service = embedding_service
        self.graph = KnowledgeGraph()
        self.max_neighbors = max_neighbors  # Maximum neighbors to consider per node
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        
        # FAISS index for similarity search
        self.faiss_index = None
        self.chunk_id_to_index = {}  # Map chunk_id to FAISS index position
        self.index_to_chunk_id = {}  # Map FAISS index position to chunk_id
        self.chunks_by_id = {}       # Map chunk_id to chunk object
        
        # Initialize similarity tracking
        self.similarity_log_file = "output/chunk_similarities.json"
        self.similarity_data = []
        self._initialize_json_file()
        
        logger.info(f"FAISSGraphBuilder initialized with max_neighbors={max_neighbors}, use_gpu={self.use_gpu}")
    
    def _initialize_json_file(self):
        """Initialize JSON file for logging chunk similarities"""
        os.makedirs("output", exist_ok=True)
        with open(self.similarity_log_file, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2)
    
    def _build_faiss_index(self, chunks: List[Union[DocumentChunk, TableChunk]]) -> bool:
        """
        Build FAISS index from chunk embeddings for fast similarity search
        
        Args:
            chunks: List of chunks with embeddings
            
        Returns:
            True if index built successfully, False otherwise
        """
        try:
            # Extract embeddings and build mappings
            embeddings = []
            valid_chunks = []
            
            for i, chunk in enumerate(chunks):
                if chunk.embedding is not None and len(chunk.embedding) > 0:
                    embeddings.append(chunk.embedding)
                    valid_chunks.append(chunk)
                    self.chunk_id_to_index[chunk.chunk_id] = len(embeddings) - 1
                    self.index_to_chunk_id[len(embeddings) - 1] = chunk.chunk_id
                    self.chunks_by_id[chunk.chunk_id] = chunk
                else:
                    logger.warning(f"Chunk {chunk.chunk_id} has no embedding, skipping")
            
            if not embeddings:
                logger.error("No valid embeddings found for FAISS index")
                return False
            
            # Convert to numpy array
            embeddings_matrix = np.array(embeddings, dtype=np.float32)
            dimension = embeddings_matrix.shape[1]
            
            logger.info(f"Building FAISS index with {len(embeddings)} embeddings, dimension={dimension}")
            
            # Choose FAISS index type based on size and GPU availability
            if self.use_gpu:
                try:
                    # Use GPU index for better performance
                    cpu_index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity with normalized vectors)
                    
                    # Normalize embeddings for cosine similarity
                    faiss.normalize_L2(embeddings_matrix)
                    
                    # Move to GPU
                    res = faiss.StandardGpuResources()
                    self.faiss_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                    logger.info("Created GPU FAISS index (IndexFlatIP)")
                except Exception as gpu_e:
                    logger.warning(f"Failed to create GPU FAISS index: {gpu_e}")
                    logger.info("Falling back to CPU FAISS index")
                    # Fall back to CPU
                    self.faiss_index = faiss.IndexFlatIP(dimension)
                    faiss.normalize_L2(embeddings_matrix)
                    logger.info("Created CPU FAISS index (IndexFlatIP)")
            else:
                # Use CPU index
                self.faiss_index = faiss.IndexFlatIP(dimension)
                faiss.normalize_L2(embeddings_matrix)
                logger.info("Created CPU FAISS index (IndexFlatIP)")
            
            # Add embeddings to index
            self.faiss_index.add(embeddings_matrix)
            
            logger.info(f"FAISS index built successfully with {self.faiss_index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            return False
    
    def _find_neighbors_faiss(self, chunk_id: str, k: int = None) -> List[Tuple[str, float]]:
        """
        Find k nearest neighbors for a given chunk using FAISS
        
        Args:
            chunk_id: ID of the chunk to find neighbors for
            k: Number of neighbors to find (defaults to max_neighbors)
            
        Returns:
            List of (neighbor_chunk_id, similarity_score) tuples
        """
        if k is None:
            k = self.max_neighbors
            
        if self.faiss_index is None:
            logger.error("FAISS index not built")
            return []
        
        if chunk_id not in self.chunk_id_to_index:
            logger.warning(f"Chunk {chunk_id} not found in FAISS index")
            return []
        
        try:
            # Get the embedding for this chunk
            chunk_index = self.chunk_id_to_index[chunk_id]
            
            # Search for k+1 neighbors (including the chunk itself)
            search_k = min(k + 1, self.faiss_index.ntotal)
            similarities, indices = self.faiss_index.search(
                np.array([self.chunks_by_id[chunk_id].embedding], dtype=np.float32), 
                search_k
            )
            
            neighbors = []
            for sim, idx in zip(similarities[0], indices[0]):
                if idx == -1:  # Invalid index
                    continue
                    
                neighbor_chunk_id = self.index_to_chunk_id[idx]
                
                # Skip self-comparison
                if neighbor_chunk_id == chunk_id:
                    continue
                
                # Convert inner product back to cosine similarity (should be same for normalized vectors)
                neighbors.append((neighbor_chunk_id, float(sim)))
            
            return neighbors[:k]  # Return only k neighbors (excluding self)
            
        except Exception as e:
            logger.error(f"Error finding neighbors for chunk {chunk_id}: {e}")
            return []
    
    def _log_chunk_similarity(self, chunk1: Union[DocumentChunk, TableChunk], 
                             chunk2: Union[DocumentChunk, TableChunk], 
                             similarity: float, threshold: float):
        """Log similarity between two chunks"""
        chunk1_preview = chunk1.content[:200] + "..." if len(chunk1.content) > 200 else chunk1.content
        chunk2_preview = chunk2.content[:200] + "..." if len(chunk2.content) > 200 else chunk2.content
        
        similarity_entry = {
            "source_name_1": chunk1.source_info.source_name,
            "source_name_2": chunk2.source_info.source_name,
            "chunk_id_1": chunk1.chunk_id,
            "chunk_id_2": chunk2.chunk_id,
            "chunk_type_1": chunk1.source_info.source_type.value,
            "chunk_type_2": chunk2.source_info.source_type.value,
            "chunk_1_preview": chunk1_preview,
            "chunk_2_preview": chunk2_preview,
            "similarity_value": float(round(similarity, 4)),
            "above_threshold": bool(similarity >= threshold),
            "threshold_used": float(round(threshold, 4)),
            "shared_keywords": list(set(chunk1.keywords) & set(chunk2.keywords))
        }
        self.similarity_data.append(similarity_entry)
    
    def _save_similarity_data(self):
        """Save accumulated similarity data to JSON file"""
        if self.similarity_data:
            try:
                # Read existing data or initialize empty list
                try:
                    with open(self.similarity_log_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    existing_data = []
                
                # Append new data
                existing_data.extend(self.similarity_data)
                
                # Write back to file
                with open(self.similarity_log_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
                
                # Clear the buffer
                self.similarity_data = []
            except Exception as e:
                logger.error(f"Error saving similarity data: {e}")
    
    def _analyze_chunk_similarity_distribution(self):
        """Analyze and log chunk similarity distribution for the entire graph"""
        if not self.similarity_data:
            return
        
        similarities = np.array([entry["similarity_value"] for entry in self.similarity_data])
        
        logger.info(f"=== FAISS-Accelerated Chunk Similarity Distribution ===")
        logger.info(f"Total chunk comparisons (limited by max_neighbors={self.max_neighbors}): {len(similarities)}")
        logger.info(f"Mean similarity: {similarities.mean():.4f}")
        logger.info(f"Std deviation: {similarities.std():.4f}")
        logger.info(f"Min similarity: {similarities.min():.4f}")
        logger.info(f"Max similarity: {similarities.max():.4f}")
        logger.info(f"Median similarity: {np.median(similarities):.4f}")
        
        # Distribution ranges
        ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        for low, high in ranges:
            count = np.sum((similarities >= low) & (similarities < high))
            percentage = (count / len(similarities)) * 100
            logger.info(f"Range [{low:.1f}-{high:.1f}): {count} ({percentage:.1f}%)")
        
        # Save data to JSON file
        self._save_similarity_data()
    
    def build_graph(self, chunks: List[Union[DocumentChunk, TableChunk]]) -> KnowledgeGraph:
        """
        Build knowledge graph from list of chunks using FAISS-accelerated similarity search
        
        Args:
            chunks: List of DocumentChunk and TableChunk objects
            
        Returns:
            KnowledgeGraph object
        """
        logger.info(f"Building FAISS-accelerated graph from {len(chunks)} chunks")
        
        # Step 1: Build FAISS index for fast similarity search
        if not self._build_faiss_index(chunks):
            logger.error("Failed to build FAISS index, falling back to standard method")
            # TODO: Fallback to original method if needed
            return self.graph
        
        # Step 2: Add all chunks as nodes
        for chunk in chunks:
            node = GraphNode(
                node_id=chunk.chunk_id,
                chunk=chunk
            )
            self.graph.add_node(node)
        
        # Step 3: Build edges using FAISS-accelerated neighbor search
        self._build_edges_faiss(chunks)
        
        # Step 4: Analyze similarity distribution
        self._analyze_chunk_similarity_distribution()
        
        stats = self.graph.get_graph_statistics()
        logger.info(f"FAISS-accelerated graph built with {stats['nodes']['total']} nodes and {stats['edges']['total']} edges")
        
        return self.graph
    
    def _build_edges_faiss(self, chunks: List[Union[DocumentChunk, TableChunk]]):
        """Build edges between chunks using FAISS-accelerated neighbor search"""
        
        total_comparisons = 0
        edges_created = 0
        
        for i, chunk in enumerate(chunks):
            if i % 100 == 0:
                logger.info(f"Processing chunk {i+1}/{len(chunks)} for edge building")
            
            # Find nearest neighbors using FAISS
            neighbors = self._find_neighbors_faiss(chunk.chunk_id, self.max_neighbors)
            
            for neighbor_chunk_id, faiss_similarity in neighbors:
                neighbor_chunk = self.chunks_by_id.get(neighbor_chunk_id)
                if neighbor_chunk is None:
                    continue
                
                total_comparisons += 1
                
                # Use FAISS similarity directly (already computed cosine similarity)
                similarity = faiss_similarity
                
                # Determine appropriate threshold based on chunk types
                threshold = self._get_threshold_for_chunk_types(chunk, neighbor_chunk)
                
                # Log similarity for analysis
                self._log_chunk_similarity(chunk, neighbor_chunk, similarity, threshold)
                
                # Only create edge if similarity exceeds threshold
                if similarity >= threshold:
                    edge_metadata = self._create_edge_metadata(chunk, neighbor_chunk, similarity)
                    if edge_metadata:
                        self.graph.add_edge(edge_metadata)
                        edges_created += 1
        
        logger.info(f"FAISS edge building completed: {total_comparisons} comparisons, {edges_created} edges created")
        logger.info(f"Comparison reduction: {len(chunks) * (len(chunks) - 1) // 2} -> {total_comparisons} " +
                   f"({total_comparisons / (len(chunks) * (len(chunks) - 1) // 2) * 100:.1f}% of full O(N²))")
    
    # Include all the helper methods from the original GraphBuilder
    def _get_threshold_for_chunk_types(self, 
                                     chunk1: Union[DocumentChunk, TableChunk], 
                                     chunk2: Union[DocumentChunk, TableChunk]) -> float:
        """Get the appropriate threshold based on the types of chunks being connected"""
        
        type1 = chunk1.source_info.source_type
        type2 = chunk2.source_info.source_type
        
        if type1 == ChunkType.TABLE and type2 == ChunkType.TABLE:
            return self.config.table_similarity_threshold
        elif (type1 == ChunkType.TABLE and type2 == ChunkType.DOCUMENT) or \
             (type1 == ChunkType.DOCUMENT and type2 == ChunkType.TABLE):
            return self.config.table_similarity_threshold  # Use table threshold for mixed connections
        elif type1 == ChunkType.DOCUMENT and type2 == ChunkType.DOCUMENT:
            return self.config.sentence_similarity_threshold
        else:
            # Fallback
            return self.config.sentence_similarity_threshold
    
    def _create_edge_metadata(self, 
                            chunk1: Union[DocumentChunk, TableChunk], 
                            chunk2: Union[DocumentChunk, TableChunk], 
                            similarity: float) -> BaseEdgeMetadata:
        """Create appropriate edge metadata based on chunk types"""
        
        # Determine edge type
        type1 = chunk1.source_info.source_type
        type2 = chunk2.source_info.source_type
        
        # Find shared keywords
        shared_keywords = list(set(chunk1.keywords) & set(chunk2.keywords))
        
        if type1 == ChunkType.TABLE and type2 == ChunkType.TABLE:
            return self._create_table_to_table_edge(chunk1, chunk2, similarity, shared_keywords)
        elif (type1 == ChunkType.TABLE and type2 == ChunkType.DOCUMENT) or \
             (type1 == ChunkType.DOCUMENT and type2 == ChunkType.TABLE):
            return self._create_table_to_document_edge(chunk1, chunk2, similarity, shared_keywords)
        elif type1 == ChunkType.DOCUMENT and type2 == ChunkType.DOCUMENT:
            return self._create_document_to_document_edge(chunk1, chunk2, similarity, shared_keywords)
        
        return None
    
    def _create_table_to_table_edge(self, 
                                   chunk1: TableChunk, 
                                   chunk2: TableChunk, 
                                   similarity: float, 
                                   shared_keywords: List[str]) -> TableToTableEdgeMetadata:
        """Create edge metadata for table-to-table connections"""
        
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
            schema_context=schema_context
        )
    
    def _create_table_to_document_edge(self, 
                                     chunk1: Union[TableChunk, DocumentChunk], 
                                     chunk2: Union[TableChunk, DocumentChunk], 
                                     similarity: float, 
                                     shared_keywords: List[str]) -> TableToDocumentEdgeMetadata:
        """Create edge metadata for table-to-document connections"""
        
        # Ensure chunk1 is table and chunk2 is document
        if isinstance(chunk1, DocumentChunk):
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
            column_references=column_refs
        )
    
    def _create_document_to_document_edge(self, 
                                        chunk1: DocumentChunk, 
                                        chunk2: DocumentChunk, 
                                        similarity: float, 
                                        shared_keywords: List[str]) -> DocumentToDocumentEdgeMetadata:
        """Create edge metadata for document-to-document connections"""
        
        # Calculate topic overlap as similarity for now
        # In a more sophisticated system, this could use topic modeling
        topic_overlap = similarity
        
        return DocumentToDocumentEdgeMetadata(
            edge_id=str(uuid.uuid4()),
            source_chunk_id=chunk1.chunk_id,
            target_chunk_id=chunk2.chunk_id,
            semantic_similarity=similarity,
            shared_keywords=shared_keywords,
            topic_overlap=topic_overlap
        )
    
    def _calculate_row_overlap(self, chunk1: TableChunk, chunk2: TableChunk) -> float:
        """Calculate overlap between rows of two table chunks"""
        
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
    
    def _find_row_references_in_text(self, table_chunk: TableChunk, text: str) -> List[str]:
        """Find references to table rows in document text"""
        
        references = []
        text_lower = text.lower()
        
        # Look for specific values that appear in table rows
        for row in table_chunk.rows_with_headers:
            for key, value in row.items():
                value_str = str(value).lower()
                if len(value_str) > 2 and value_str in text_lower:
                    references.append(f"{key}: {value}")
        
        return references[:10]  # Limit to top 10 references
    
    def _find_column_references_in_text(self, table_chunk: TableChunk, text: str) -> List[str]:
        """Find references to table columns in document text"""
        
        references = []
        text_lower = text.lower()
        
        # Look for column headers mentioned in text
        for header in table_chunk.column_headers:
            header_lower = header.lower()
            if header_lower in text_lower:
                references.append(header)
        
        return references
