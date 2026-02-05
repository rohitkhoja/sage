"""
Graph builder for constructing knowledge graph from chunks with different edge types
"""

import uuid
import re
from typing import List, Union, Tuple, Dict, Any
import numpy as np
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


class GraphBuilder:
    """
    Builds knowledge graph from document and table chunks with sophisticated edge relationships
    """
    
    def __init__(self, config: ProcessingConfig, embedding_service: EmbeddingService):
        self.config = config
        self.embedding_service = embedding_service
        self.graph = KnowledgeGraph()
        
        # Initialize similarity tracking
        self.similarity_log_file = "output/chunk_similarities.json"
        self.similarity_data = []
        self._initialize_json_file()
    
    def _initialize_json_file(self):
        """Initialize JSON file for logging chunk similarities"""
        os.makedirs("output", exist_ok=True)
        with open(self.similarity_log_file, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2)
    
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
        
        logger.info(f"=== Chunk Similarity Distribution (Graph Level) ===")
        logger.info(f"Total chunk comparisons: {len(similarities)}")
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
        
        # Above threshold analysis for different chunk type combinations
        doc_doc_sims = []
        table_table_sims = []
        table_doc_sims = []
        
        for entry in self.similarity_data:
            sim_val = entry["similarity_value"]
            type1, type2 = entry["chunk_type_1"], entry["chunk_type_2"]
            
            if type1 == 'document' and type2 == 'document':
                doc_doc_sims.append(sim_val)
            elif type1 == 'table' and type2 == 'table':
                table_table_sims.append(sim_val)
            elif (type1 == 'table' and type2 == 'document') or (type1 == 'document' and type2 == 'table'):
                table_doc_sims.append(sim_val)
        
        # Type-specific analysis
        for sim_type, sims in [("Document-Document", doc_doc_sims), 
                              ("Table-Table", table_table_sims), 
                              ("Table-Document", table_doc_sims)]:
            if sims:
                sims_array = np.array(sims)
                logger.info(f"--- {sim_type} Similarities ---")
                logger.info(f"Count: {len(sims)}")
                logger.info(f"Mean: {sims_array.mean():.4f}")
                logger.info(f"Std: {sims_array.std():.4f}")
        
        # Save data to JSON file
        self._save_similarity_data()
    
    def build_graph(self, chunks: List[Union[DocumentChunk, TableChunk]]) -> KnowledgeGraph:
        """
        Build knowledge graph from list of chunks
        
        Args:
            chunks: List of DocumentChunk and TableChunk objects
            
        Returns:
            KnowledgeGraph object
        """
        logger.info(f"Building graph from {len(chunks)} chunks")
        
        # Step 1: Add all chunks as nodes
        for chunk in chunks:
            node = GraphNode(
                node_id=chunk.chunk_id,
                chunk=chunk
            )
            self.graph.add_node(node)
        
        # Step 2: Build edges between chunks
        self._build_edges(chunks)
        
        # Step 3: Analyze similarity distribution
        self._analyze_chunk_similarity_distribution()
        
        stats = self.graph.get_graph_statistics()
        logger.info(f"Graph built with {stats['nodes']['total']} nodes and {stats['edges']['total']} edges")
        
        return self.graph
    
    def _build_edges(self, chunks: List[Union[DocumentChunk, TableChunk]]):
        """Build edges between chunks based on similarity and relationships"""
        
        for i, chunk1 in enumerate(chunks):
            for j, chunk2 in enumerate(chunks[i+1:], i+1):
                
                # Calculate semantic similarity
                if chunk1.embedding and chunk2.embedding:
                    similarity = self.embedding_service.compute_similarity(
                        chunk1.embedding, chunk2.embedding
                    )
                else:
                    similarity = 0.0
                
                # Determine appropriate threshold based on chunk types
                threshold = self._get_threshold_for_chunk_types(chunk1, chunk2)
                
                # Log similarity for analysis
                self._log_chunk_similarity(chunk1, chunk2, similarity, threshold)
                
                # Only create edge if similarity exceeds threshold
                if similarity >= threshold:
                    edge_metadata = self._create_edge_metadata(chunk1, chunk2, similarity)
                    if edge_metadata:
                        self.graph.add_edge(edge_metadata)
    
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