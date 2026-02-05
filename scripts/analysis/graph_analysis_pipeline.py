#!/usr/bin/env python3
"""
Graph Analysis Pipeline: Comprehensive analysis of document and table chunks
for optimal graph construction with detailed similarity metrics and visualizations.
"""

import os
import sys
import json
import time
# Removed numpy import - no longer needed with GPU-only implementation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
import pickle
import faiss
from sklearn.metrics.pairwise import cosine_similarity
# Removed SequenceMatcher import - using GPU-accelerated fuzzy matching instead
import torch
from loguru import logger
import re
import networkx as nx
try:
    from cdlib import algorithms
    CDLIB_AVAILABLE = True
except ImportError:
    CDLIB_AVAILABLE = False
    logger.warning("cdlib not available. Community detection will be skipped. Install with: pip install cdlib")

def make_hashable(item):
    """
    Recursively convert any nested structure to be hashable by converting lists to tuples
    """
    if isinstance(item, list):
        return tuple(make_hashable(x) for x in item)
    elif isinstance(item, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in item.items()))
    elif isinstance(item, set):
        return tuple(sorted(make_hashable(x) for x in item))
    else:
        return item

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.processors.embedding_service import EmbeddingService
from src.core.models import ProcessingConfig, ChunkType
from src.core.graph import KnowledgeGraph
from src.core.models import (
    GraphNode, EdgeType, BaseEdgeMetadata, DocumentToDocumentEdgeMetadata,
    TableToTableEdgeMetadata, TableToDocumentEdgeMetadata, DocumentChunk, TableChunk, SourceInfo
)

@dataclass
class SimilarityMetrics:
    """Container for all similarity metrics between two chunks"""
    # Core similarities (existing)
    topic_similarity: float
    content_similarity: float
    column_similarity: float
    edge_type: str
    source_chunk_id: str
    target_chunk_id: str
    
    # New features from CPU implementation
    # Doc-Doc specific
    entity_relationship_overlap: float = 0.0 # For doc-doc edges
    entity_count: int = 0 # Fuzzy matched entity count
    event_count: int = 0 # For doc-doc edges
    
    # Doc-Table specific 
    topic_title_similarity: float = 0.0 # Doc topic vs table title
    topic_summary_similarity: float = 0.0 # Doc topic vs table summary
    
    # Table-Table specific
    title_similarity: float = 0.0 # Table title vs table title
    description_similarity: float = 0.0 # Table description vs table description

@dataclass 
class ChunkData:
    """Enhanced chunk data with computed embeddings"""
    chunk_id: str
    chunk_type: str # "document" or "table"
    content: str
    content_embedding: List[float]
    entities: Dict[str, Any]
    entity_embeddings: Dict[str, List[float]] # entity_name -> embedding
    topic: Optional[str] = None
    topic_embedding: Optional[List[float]] = None
    column_descriptions: Optional[Dict[str, str]] = None
    column_embeddings: Optional[Dict[str, List[float]]] = None
    metadata: Optional[Dict[str, Any]] = None

class GPUAcceleratedSimilarityCalculator:
    """
    GPU-accelerated similarity calculator for massive parallel processing
    Uses precomputed embeddings and batch operations for optimal performance
    """
    
    def __init__(self, chunks: List[ChunkData], embedding_service: EmbeddingService, 
                 batch_size: int = 10000, gpu_id: int = 0):
        self.chunks = chunks
        self.embedding_service = embedding_service
        self.batch_size = batch_size
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        
        # Precomputed GPU tensors
        self.gpu_content_embeddings = None
        self.gpu_topic_embeddings = None
        self.gpu_column_embeddings = {} # chunk_id -> {column_name -> tensor}
        
        # Additional precomputed tensors for efficiency
        self.gpu_table_titles = None # Table titles embeddings
        self.gpu_table_summaries = None # Table summaries embeddings 
        self.gpu_table_descriptions = None # Table descriptions embeddings
        self.table_title_mapping = {} # chunk_id -> index in gpu_table_titles
        self.table_summary_mapping = {} # chunk_id -> index in gpu_table_summaries
        self.table_desc_mapping = {} # chunk_id -> index in gpu_table_descriptions
        
        # Precomputed similarity lookup tables for maximum efficiency
        self.precomputed_similarities = {
            'doc_topic_to_table_title': {}, # (doc_chunk_id, table_chunk_id) -> similarity
            'doc_topic_to_table_summary': {},
            'table_title_to_table_title': {},
            'table_desc_to_table_desc': {},
            'entity_fuzzy_matches': {}, # (chunk1_id, chunk2_id) -> (count, relationship_overlap)
            'event_fuzzy_matches': {} # (chunk1_id, chunk2_id) -> count
        }
        
        # Chunk mappings for fast lookups
        self.chunk_id_to_idx = {chunk.chunk_id: i for i, chunk in enumerate(chunks)}
        self.chunk_types = [chunk.chunk_type for chunk in chunks]
        
        # Batch processing tracking
        self.batch_results_dir = Path("/shared/khoja/CogComp/output/batch_results")
        self.batch_results_dir.mkdir(exist_ok=True)
        
        logger.info(f"GPU Similarity Calculator initialized on {self.device}")
        logger.info(f"Batch size: {batch_size}, Processing {len(chunks)} chunks")
        
        # Calculate optimal batch size based on GPU memory
        if torch.cuda.is_available():
            self.optimal_batch_size = self._calculate_optimal_batch_size()
            logger.info(f"Optimal batch size based on GPU memory: {self.optimal_batch_size}")
        else:
            self.optimal_batch_size = batch_size
    
    def precompute_embeddings(self):
        """Precompute all embeddings and move to GPU for batch processing"""
        logger.info("Precomputing embeddings for GPU processing...")
        
        # Extract content embeddings
        content_embeddings = []
        topic_embeddings = []
        
        for chunk in self.chunks:
            # Content embeddings
            if chunk.content_embedding:
                content_embeddings.append(chunk.content_embedding)
            else:
                content_embeddings.append([0.0] * 768) # Default embedding size
            
            # Topic embeddings
            if chunk.topic_embedding:
                topic_embeddings.append(chunk.topic_embedding)
            else:
                topic_embeddings.append([0.0] * 768)
        
        # Move to GPU
        self.gpu_content_embeddings = torch.tensor(content_embeddings, device=self.device, dtype=torch.float32)
        self.gpu_topic_embeddings = torch.tensor(topic_embeddings, device=self.device, dtype=torch.float32)
        
        # Normalize for cosine similarity
        self.gpu_content_embeddings = torch.nn.functional.normalize(self.gpu_content_embeddings, p=2, dim=1)
        self.gpu_topic_embeddings = torch.nn.functional.normalize(self.gpu_topic_embeddings, p=2, dim=1)
        
        # Skip entity embeddings - using fuzzy matching instead
        
        # Precompute column embeddings
        self._precompute_column_embeddings()
        
        # Precompute table metadata embeddings for efficient text similarity
        self._precompute_table_metadata_embeddings()
        
        # Skip precomputing all similarities here - will be done after getting unique pairs
        
        logger.info(f"Precomputed embeddings on GPU: {self.gpu_content_embeddings.shape}")
        logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
    
    def _precompute_entity_embeddings(self):
        """Precompute entity embeddings for each chunk"""
        logger.info("Precomputing entity embeddings...")
        
        for chunk in self.chunks:
            if chunk.entity_embeddings:
                chunk_entity_tensors = {}
                for entity_name, embedding in chunk.entity_embeddings.items():
                    if embedding:
                        tensor = torch.tensor([embedding], device=self.device, dtype=torch.float32)
                        chunk_entity_tensors[entity_name] = torch.nn.functional.normalize(tensor, p=2, dim=1)
                
                if chunk_entity_tensors:
                    self.gpu_entity_embeddings[chunk.chunk_id] = chunk_entity_tensors
    
    def _precompute_column_embeddings(self):
        """Precompute column embeddings for each chunk"""
        logger.info("Precomputing column embeddings...")
        
        for chunk in self.chunks:
            if chunk.column_embeddings:
                chunk_column_tensors = {}
                for column_name, embedding in chunk.column_embeddings.items():
                    if embedding:
                        tensor = torch.tensor([embedding], device=self.device, dtype=torch.float32)
                        chunk_column_tensors[column_name] = torch.nn.functional.normalize(tensor, p=2, dim=1)
                
                if chunk_column_tensors:
                    self.gpu_column_embeddings[chunk.chunk_id] = chunk_column_tensors
    
    def _precompute_table_metadata_embeddings(self):
        """Precompute table titles, summaries, and descriptions for efficient text similarity"""
        logger.info("Precomputing table metadata embeddings...")
        
        table_titles = []
        table_summaries = []
        table_descriptions = []
        
        title_chunks = []
        summary_chunks = [] 
        desc_chunks = []
        
        for chunk in self.chunks:
            if chunk.chunk_type == "table" and chunk.metadata:
                metadata = chunk.metadata
                
                # Table titles
                title = metadata.get('table_title', '')
                if title:
                    table_titles.append(title)
                    self.table_title_mapping[chunk.chunk_id] = len(table_titles) - 1
                    title_chunks.append(chunk.chunk_id)
                
                # Table summaries
                summary = metadata.get('table_summary', '')
                if summary:
                    table_summaries.append(summary)
                    self.table_summary_mapping[chunk.chunk_id] = len(table_summaries) - 1
                    summary_chunks.append(chunk.chunk_id)
                
                # Table descriptions (same as topic for tables)
                desc = metadata.get('table_description', '')
                if desc:
                    table_descriptions.append(desc)
                    self.table_desc_mapping[chunk.chunk_id] = len(table_descriptions) - 1
                    desc_chunks.append(chunk.chunk_id)
        
        # Generate embeddings and move to GPU
        if table_titles:
            logger.info(f"Computing {len(table_titles)} table title embeddings...")
            title_embeddings = []
            for title in table_titles:
                emb = self.embedding_service.generate_embeddings([title])[0]
                title_embeddings.append(emb)
            self.gpu_table_titles = torch.tensor(title_embeddings, device=self.device, dtype=torch.float32)
            self.gpu_table_titles = torch.nn.functional.normalize(self.gpu_table_titles, p=2, dim=1)
        
        if table_summaries:
            logger.info(f"Computing {len(table_summaries)} table summary embeddings...")
            summary_embeddings = []
            for summary in table_summaries:
                emb = self.embedding_service.generate_embeddings([summary])[0]
                summary_embeddings.append(emb)
            self.gpu_table_summaries = torch.tensor(summary_embeddings, device=self.device, dtype=torch.float32)
            self.gpu_table_summaries = torch.nn.functional.normalize(self.gpu_table_summaries, p=2, dim=1)
        
        if table_descriptions:
            logger.info(f"Computing {len(table_descriptions)} table description embeddings...")
            desc_embeddings = []
            for desc in table_descriptions:
                emb = self.embedding_service.generate_embeddings([desc])[0]
                desc_embeddings.append(emb)
            self.gpu_table_descriptions = torch.tensor(desc_embeddings, device=self.device, dtype=torch.float32)
            self.gpu_table_descriptions = torch.nn.functional.normalize(self.gpu_table_descriptions, p=2, dim=1)
        
        logger.info(f"Precomputed table metadata: {len(table_titles)} titles, {len(table_summaries)} summaries, {len(table_descriptions)} descriptions")
    
    def _precompute_all_similarities(self, unique_pairs: List[Tuple[str, str]]):
        """Precompute all similarities for the given unique pairs only (not all possible combinations)"""
        logger.info(f"Precomputing similarities for {len(unique_pairs)} unique pairs...")
        
        # Process pairs in batches of 50,000
        batch_size = 50000
        total_batches = (len(unique_pairs) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(unique_pairs))
            batch_pairs = unique_pairs[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches}: {len(batch_pairs)} pairs")
            
            # Process this batch on GPU
            self._precompute_similarities_for_batch(batch_pairs)
        
        logger.info(f"Precomputed similarities: "
                   f"{len(self.precomputed_similarities['doc_topic_to_table_title'])} doc-title, "
                   f"{len(self.precomputed_similarities['doc_topic_to_table_summary'])} doc-summary, "
                   f"{len(self.precomputed_similarities['table_title_to_table_title'])} title-title, "
                   f"{len(self.precomputed_similarities['table_desc_to_table_desc'])} desc-desc, "
                   f"{len(self.precomputed_similarities['entity_fuzzy_matches'])} entity matches, "
                   f"{len(self.precomputed_similarities['event_fuzzy_matches'])} event matches")
    
    def _precompute_similarities_for_batch(self, batch_pairs: List[Tuple[str, str]]):
        """Process a batch of pairs and compute all 5 similarity types for each pair"""
        
        for chunk1_id, chunk2_id in batch_pairs:
            # Get chunk objects
            chunk1_idx = self.chunk_id_to_idx.get(chunk1_id)
            chunk2_idx = self.chunk_id_to_idx.get(chunk2_id)
            
            if chunk1_idx is None or chunk2_idx is None:
                continue
                
            chunk1 = self.chunks[chunk1_idx]
            chunk2 = self.chunks[chunk2_idx]
            
            # Calculate all 5 similarity types for this pair
            self._calculate_all_similarities_for_pair(chunk1, chunk2)
    
    def _calculate_all_similarities_for_pair(self, chunk1: ChunkData, chunk2: ChunkData):
        """Calculate all 5 similarity types for a single pair"""
        
        # Determine edge type
        edge_type = self._get_edge_type(chunk1.chunk_type, chunk2.chunk_type)
        
        # 1. Doc topic to table title similarity (for doc-table pairs)
        if edge_type == "doc-table":
            doc_chunk = chunk1 if chunk1.chunk_type == "document" else chunk2
            table_chunk = chunk2 if chunk1.chunk_type == "document" else chunk1
            
            # Doc topic to table title
            if (self.gpu_topic_embeddings is not None and 
                self.gpu_table_titles is not None and 
                table_chunk.chunk_id in self.table_title_mapping):
                
                doc_idx = self.chunk_id_to_idx[doc_chunk.chunk_id]
                doc_topic_emb = self.gpu_topic_embeddings[doc_idx:doc_idx+1]
                
                table_title_idx = self.table_title_mapping[table_chunk.chunk_id]
                # Fix variable name typo: table_titleIdx -> table_title_idx
                table_title_emb = self.gpu_table_titles[table_title_idx:table_title_idx+1]
                
                similarity = torch.mm(doc_topic_emb, table_title_emb.T).item()
                self.precomputed_similarities['doc_topic_to_table_title'][(doc_chunk.chunk_id, table_chunk.chunk_id)] = similarity
            
            # Doc topic to table summary
            if (self.gpu_topic_embeddings is not None and 
                self.gpu_table_summaries is not None and 
                table_chunk.chunk_id in self.table_summary_mapping):
                
                doc_idx = self.chunk_id_to_idx[doc_chunk.chunk_id]
                doc_topic_emb = self.gpu_topic_embeddings[doc_idx:doc_idx+1]
                
                table_summary_idx = self.table_summary_mapping[table_chunk.chunk_id]
                table_summary_emb = self.gpu_table_summaries[table_summary_idx:table_summary_idx+1]
                
                similarity = torch.mm(doc_topic_emb, table_summary_emb.T).item()
                self.precomputed_similarities['doc_topic_to_table_summary'][(doc_chunk.chunk_id, table_chunk.chunk_id)] = similarity
        
        # 2. Table title to table title similarity (for table-table pairs)
        if (edge_type == "table-table" and 
            self.gpu_table_titles is not None and 
            chunk1.chunk_id in self.table_title_mapping and 
            chunk2.chunk_id in self.table_title_mapping):
            
            table1_title_idx = self.table_title_mapping[chunk1.chunk_id]
            table1_title_emb = self.gpu_table_titles[table1_title_idx:table1_title_idx+1]
            
            table2_title_idx = self.table_title_mapping[chunk2.chunk_id]
            table2_title_emb = self.gpu_table_titles[table2_title_idx:table2_title_idx+1]
            
            similarity = torch.mm(table1_title_emb, table2_title_emb.T).item()
            self.precomputed_similarities['table_title_to_table_title'][(chunk1.chunk_id, chunk2.chunk_id)] = similarity
            self.precomputed_similarities['table_title_to_table_title'][(chunk2.chunk_id, chunk1.chunk_id)] = similarity
        
        # 3. Table description to table description similarity (for table-table pairs)
        if (edge_type == "table-table" and 
            self.gpu_table_descriptions is not None and 
            chunk1.chunk_id in self.table_desc_mapping and 
            chunk2.chunk_id in self.table_desc_mapping):
            
            table1_desc_idx = self.table_desc_mapping[chunk1.chunk_id]
            table1_desc_emb = self.gpu_table_descriptions[table1_desc_idx:table1_desc_idx+1]
            
            table2_desc_idx = self.table_desc_mapping[chunk2.chunk_id]
            table2_desc_emb = self.gpu_table_descriptions[table2_desc_idx:table2_desc_idx+1]
            
            similarity = torch.mm(table1_desc_emb, table2_desc_emb.T).item()
            self.precomputed_similarities['table_desc_to_table_desc'][(chunk1.chunk_id, chunk2.chunk_id)] = similarity
            self.precomputed_similarities['table_desc_to_table_desc'][(chunk2.chunk_id, chunk1.chunk_id)] = similarity
        
        # 4. Fuzzy entity matches (for all pairs)
        if chunk1.entities and chunk2.entities:
            entity_count, _, relationship_overlap = self._fuzzy_entity_match_gpu(
                chunk1.entities, chunk2.entities, threshold=0.75
            )
            self.precomputed_similarities['entity_fuzzy_matches'][(chunk1.chunk_id, chunk2.chunk_id)] = (entity_count, relationship_overlap)
            self.precomputed_similarities['entity_fuzzy_matches'][(chunk2.chunk_id, chunk1.chunk_id)] = (entity_count, relationship_overlap)
        
        # 5. Event fuzzy matches (for doc-doc pairs only)
        if (edge_type == "doc-doc" and 
            chunk1.metadata and chunk2.metadata):
            events1 = chunk1.metadata.get('events', {})
            events2 = chunk2.metadata.get('events', {})
            if events1 and events2:
                event_count, _ = self._calculate_event_overlap(events1, events2)
                self.precomputed_similarities['event_fuzzy_matches'][(chunk1.chunk_id, chunk2.chunk_id)] = event_count
                self.precomputed_similarities['event_fuzzy_matches'][(chunk2.chunk_id, chunk1.chunk_id)] = event_count
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available GPU memory"""
        try:
            # Get GPU memory info
            gpu_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**3 # GB
            gpu_memory_free = torch.cuda.memory_reserved(self.device) / 1024**3 # GB
            available_memory = gpu_memory - gpu_memory_free
            
            # Estimate memory per pair (conservative estimate)
            # Each pair needs: 2 embeddings (768 dim) + intermediate results
            memory_per_pair = (768 * 2 * 4) / 1024**3 # 4 bytes per float, convert to GB
            
            # Use 80% of available memory for batch processing
            max_pairs_per_batch = int((available_memory * 0.8) / memory_per_pair)
            
            # Cap between 1000 and 50000 pairs per batch
            optimal_batch_size = min(100000, max(1000, max_pairs_per_batch))
            # optimal_batch_size = max_pairs_per_batch
            
            logger.info(f"GPU Memory: {gpu_memory:.1f}GB total, {available_memory:.1f}GB available")
            logger.info(f"Estimated memory per pair: {memory_per_pair*1000:.2f}MB")
            
            return optimal_batch_size
        except Exception as e:
            logger.warning(f"Could not calculate optimal batch size: {e}")
            return self.batch_size
    
    def calculate_similarities_batch(self, unique_pairs: List[Tuple[str, str]]) -> List[SimilarityMetrics]:
        """Calculate similarities for all pairs using GPU batch processing"""
        # Use optimal batch size for better GPU utilization
        effective_batch_size = self.optimal_batch_size
        logger.info(f"Processing {len(unique_pairs)} pairs in batches of {effective_batch_size}")
        
        all_results = []
        batch_count = 0
        
        # Clear old batch results
        self.clear_batch_results()
        
        # Process in batches
        for i in range(0, len(unique_pairs), effective_batch_size):
            batch_pairs = unique_pairs[i:i + effective_batch_size]
            batch_count += 1
            
            logger.info(f"Processing batch {batch_count}/{(len(unique_pairs) + effective_batch_size - 1) // effective_batch_size}")
            
            # Calculate similarities for this batch
            batch_results = self._calculate_batch_similarities_gpu(batch_pairs)
            
            # Save batch results incrementally
            self._save_batch_results(batch_results, batch_count)
            
            # Add to overall results
            all_results.extend(batch_results)
            
            # Clear GPU cache periodically
            if batch_count % 10 == 0:
                torch.cuda.empty_cache()
            
            logger.info(f"Batch {batch_count} completed: {len(batch_results)} similarities calculated")
        
        logger.info(f"GPU batch processing completed: {len(all_results)} total similarities")
        return all_results
    
    def _calculate_batch_similarities_gpu(self, batch_pairs: List[Tuple[str, str]]) -> List[SimilarityMetrics]:
        """Calculate similarities for a batch of pairs using GPU acceleration"""
        batch_results = []
        
        # Extract indices for batch processing
        source_indices = []
        target_indices = []
        pair_info = []
        
        for source_id, target_id in batch_pairs:
            source_idx = self.chunk_id_to_idx.get(source_id)
            target_idx = self.chunk_id_to_idx.get(target_id)
            
            if source_idx is not None and target_idx is not None:
                source_indices.append(source_idx)
                target_indices.append(target_idx)
                pair_info.append((source_id, target_id, source_idx, target_idx))
        
        if not source_indices:
            return []
        
        # Convert to tensors
        source_tensor = torch.tensor(source_indices, device=self.device)
        target_tensor = torch.tensor(target_indices, device=self.device)
        
        # Batch compute content similarities
        source_content = self.gpu_content_embeddings[source_tensor]
        target_content = self.gpu_content_embeddings[target_tensor]
        content_similarities = torch.sum(source_content * target_content, dim=1)
        
        # Batch compute topic similarities
        source_topic = self.gpu_topic_embeddings[source_tensor]
        target_topic = self.gpu_topic_embeddings[target_tensor]
        topic_similarities = torch.sum(source_topic * target_topic, dim=1)
        
        # Convert to CPU for individual processing
        content_sims = content_similarities.cpu().numpy()
        topic_sims = topic_similarities.cpu().numpy()
        
        # Process each pair for entity and column similarities
        for i, (source_id, target_id, source_idx, target_idx) in enumerate(pair_info):
            source_chunk = self.chunks[source_idx]
            target_chunk = self.chunks[target_idx]
            
            # Calculate column similarity
            column_sim = self._calculate_column_similarity_gpu(source_chunk, target_chunk)
            
            # Determine edge type
            edge_type = self._get_edge_type(source_chunk.chunk_type, target_chunk.chunk_type)
            
            # Calculate additional features based on edge type
            additional_metrics = self._calculate_additional_features(source_chunk, target_chunk, edge_type)
            
            # Create similarity metrics with all features
            similarity_metrics = SimilarityMetrics(
                topic_similarity=max(0.0, float(topic_sims[i])),
                content_similarity=max(0.0, float(content_sims[i])),
                column_similarity=column_sim,
                edge_type=edge_type,
                source_chunk_id=source_id,
                target_chunk_id=target_id,
                **additional_metrics # Unpack additional features
            )
            
            batch_results.append(similarity_metrics)
        
        return batch_results
    
    def _calculate_column_similarity_gpu(self, chunk1: ChunkData, chunk2: ChunkData) -> float:
        """Calculate column similarity using GPU tensors"""
        if chunk1.chunk_type == "document" and chunk2.chunk_type == "document":
            return 0.0
        
        # For table-table edges
        if chunk1.chunk_type == "table" and chunk2.chunk_type == "table":
            return self._calculate_table_table_column_similarity_gpu(chunk1, chunk2)
        
        # For doc-table edges
        return self._calculate_doc_table_column_similarity_gpu(chunk1, chunk2)
    
    def _calculate_table_table_column_similarity_gpu(self, table1: ChunkData, table2: ChunkData) -> float:
        """Calculate table-table column similarity using GPU"""
        columns1 = self.gpu_column_embeddings.get(table1.chunk_id, {})
        columns2 = self.gpu_column_embeddings.get(table2.chunk_id, {})
        
        if not columns1 or not columns2:
            return 0.0
        
        similarities = []
        
        for col1_name, tensor1 in columns1.items():
            best_similarity = 0.0
            for col2_name, tensor2 in columns2.items():
                try:
                    sim = torch.mm(tensor1, tensor2.t()).item()
                    best_similarity = max(best_similarity, sim)
                except Exception:
                    continue
            similarities.append(best_similarity)
        
        if not similarities:
            return 0.0
        
        top_similarities = sorted(similarities, reverse=True)[:3]
        return sum(top_similarities) / len(top_similarities) if top_similarities else 0.0
    
    def _calculate_doc_table_column_similarity_gpu(self, chunk1: ChunkData, chunk2: ChunkData) -> float:
        """Calculate doc-table column similarity using GPU"""
        # Identify doc and table
        if chunk1.chunk_type == "document":
            doc_chunk, table_chunk = chunk1, chunk2
        else:
            doc_chunk, table_chunk = chunk2, chunk1
        
        # Get document content embedding
        doc_idx = self.chunk_id_to_idx[doc_chunk.chunk_id]
        doc_content_tensor = self.gpu_content_embeddings[doc_idx:doc_idx+1]
        
        # Get table column embeddings
        table_columns = self.gpu_column_embeddings.get(table_chunk.chunk_id, {})
        
        if not table_columns:
            return 0.0
        
        similarities = []
        
        for col_name, col_tensor in table_columns.items():
            try:
                sim = torch.mm(doc_content_tensor, col_tensor.t()).item()
                similarities.append(sim)
            except Exception:
                continue
        
        if not similarities:
            return 0.0
        
        top_similarities = sorted(similarities, reverse=True)[:3]
        return sum(top_similarities) / len(top_similarities) if top_similarities else 0.0
    
    def _get_edge_type(self, type1: str, type2: str) -> str:
        """Determine edge type from chunk types"""
        if type1 == "document" and type2 == "document":
            return "doc-doc"
        elif type1 == "table" and type2 == "table":
            return "table-table"
        else:
            return "doc-table"
    
    def _normalize_entity_name(self, entity_name: str) -> str:
        """Normalize entity name for fuzzy matching"""
        # Convert to lowercase, strip whitespace, remove extra spaces
        normalized = re.sub(r'\s+', ' ', entity_name.lower().strip())
        # Remove common punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return normalized
    
    def _gpu_fuzzy_similarity_batch(self, strings1: List[str], strings2: List[str], threshold: float = 0.75) -> torch.Tensor:
        """
        GPU-accelerated fuzzy string similarity using character overlap ratios
        Returns similarity matrix of shape (len(strings1), len(strings2))
        """
        try:
            # Normalize all strings
            norm_strings1 = [self._normalize_entity_name(s) for s in strings1]
            norm_strings2 = [self._normalize_entity_name(s) for s in strings2]
            
            # Convert to character sets and calculate Jaccard similarity
            similarities = []
            
            for s1 in norm_strings1:
                row_similarities = []
                chars1 = set(s1.replace(' ', '')) # Remove spaces for character comparison
                
                for s2 in norm_strings2:
                    chars2 = set(s2.replace(' ', ''))
                    
                    if not chars1 or not chars2:
                        row_similarities.append(0.0)
                        continue
                    
                    # Calculate Jaccard similarity (intersection / union)
                    intersection = len(chars1.intersection(chars2))
                    union = len(chars1.union(chars2))
                    jaccard_sim = intersection / union if union > 0 else 0.0
                    
                    # Also consider length similarity to avoid very different length matches
                    len_sim = min(len(s1), len(s2)) / max(len(s1), len(s2)) if max(len(s1), len(s2)) > 0 else 0.0
                    
                    # Combined similarity (weighted average)
                    combined_sim = 0.7 * jaccard_sim + 0.3 * len_sim
                    row_similarities.append(combined_sim)
                
                similarities.append(row_similarities)
            
            # Convert to GPU tensor
            similarity_tensor = torch.tensor(similarities, device=self.device, dtype=torch.float32)
            return similarity_tensor
            
        except Exception as e:
            logger.warning(f"Error in GPU fuzzy similarity: {e}")
            # Fallback to zero similarity matrix
            return torch.zeros(len(strings1), len(strings2), device=self.device, dtype=torch.float32)
    
    def _fuzzy_entity_match_gpu(self, entities1: Dict[str, Any], entities2: Dict[str, Any], 
                               threshold: float = 0.75) -> Tuple[int, Dict[str, Any], Dict[str, float]]:
        """
        GPU-accelerated fuzzy match entities between two chunks with 75% similarity threshold
        Returns: (match_count, combined_entities, relationship_overlap)
        """
        if not entities1 or not entities2:
            return 0, {}, {}
        
        # Extract entity names and data
        names1 = list(entities1.keys())
        names2 = list(entities2.keys())
        data1 = list(entities1.values())
        data2 = list(entities2.values())
        
        # Use GPU-accelerated fuzzy similarity
        similarity_matrix = self._gpu_fuzzy_similarity_batch(names1, names2, threshold)
        
        # Find best matches using GPU operations
        matched_pairs = []
        used_indices2 = set()
        
        for i, name1 in enumerate(names1):
            # Get similarities for this entity
            similarities = similarity_matrix[i]
            
            # Find best match above threshold
            valid_indices = [j for j in range(len(names2)) if j not in used_indices2]
            if not valid_indices:
                continue
            
            valid_similarities = similarities[valid_indices]
            max_sim, max_idx_in_valid = torch.max(valid_similarities, dim=0)
            
            if max_sim.item() >= threshold:
                actual_idx = valid_indices[max_idx_in_valid.item()]
                matched_pairs.append((i, actual_idx, max_sim.item()))
                used_indices2.add(actual_idx)
        
        # Create combined entities and calculate relationship overlap
        combined_entities = {}
        relationship_overlap = {}
        
        for i, j, similarity_score in matched_pairs:
            entity_key = names1[i]
            data1_item = data1[i]
            data2_item = data2[j]
            
            # Combine entity data based on format
            if isinstance(data1_item, dict) and isinstance(data2_item, dict):
                # Check if this is doc-doc, doc-table, or table-table format
                if "relationships" in data1_item or "relationships" in data2_item:
                    # Document format - handle mixed data types properly
                    
                    # Handle relationships (might be lists or strings)
                    relationships = []
                    for rel_list in [data1_item.get("relationships", []), data2_item.get("relationships", [])]:
                        if isinstance(rel_list, list):
                            for rel in rel_list:
                                rel_tuple = tuple(rel) if isinstance(rel, list) else rel
                                if rel_tuple not in relationships:
                                    relationships.append(rel_tuple)
                        elif rel_list:
                            if rel_list not in relationships:
                                relationships.append(rel_list)
                    
                    # Handle actions (might be lists or strings)
                    actions = []
                    for action_list in [data1_item.get("actions", []), data2_item.get("actions", [])]:
                        if isinstance(action_list, list):
                            for action in action_list:
                                if action and action not in actions:
                                    actions.append(action)
                        elif action_list and action_list not in actions:
                            actions.append(action_list)
                    
                    # Handle details (might be lists or strings)
                    details = []
                    for detail_list in [data1_item.get("details", []), data2_item.get("details", [])]:
                        if isinstance(detail_list, list):
                            for detail in detail_list:
                                if detail and detail not in details:
                                    details.append(detail)
                        elif detail_list and detail_list not in details:
                            details.append(detail_list)
                    
                    combined_entities[entity_key] = {
                        "relationships": relationships,
                        "actions": actions,
                        "details": details
                    }
                    
                    # Calculate relationship overlap with proper type handling using make_hashable
                    rel1 = set()
                    rel1_data = data1_item.get("relationships", [])
                    if isinstance(rel1_data, list):
                        for rel in rel1_data:
                            hashable_rel = make_hashable(rel)
                            rel1.add(hashable_rel)
                    elif rel1_data:
                        hashable_rel = make_hashable(rel1_data)
                        rel1.add(hashable_rel)
                    
                    rel2 = set()
                    rel2_data = data2_item.get("relationships", [])
                    if isinstance(rel2_data, list):
                        for rel in rel2_data:
                            hashable_rel = make_hashable(rel)
                            rel2.add(hashable_rel)
                    elif rel2_data:
                        hashable_rel = make_hashable(rel2_data)
                        rel2.add(hashable_rel)
                    rel_overlap = len(rel1.intersection(rel2)) / max(len(rel1), len(rel2), 1)
                    relationship_overlap[entity_key] = float(rel_overlap)
                else:
                    # Table format or mixed format - handle descriptions properly
                    descriptions = []
                    for desc in [data1_item.get("description", ""), data2_item.get("description", "")]:
                        if desc and desc not in descriptions:
                            descriptions.append(desc)
                    
                    combined_entities[entity_key] = {
                        "type": data1_item.get("type", data2_item.get("type", "")),
                        "category": data1_item.get("category", data2_item.get("category", "")),
                        "description": descriptions
                    }
                    relationship_overlap[entity_key] = 0.0
        
        return len(matched_pairs), combined_entities, relationship_overlap
    
    def _calculate_event_overlap(self, events1: Dict[str, Any], events2: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Calculate overlap between events using GPU-accelerated fuzzy matching (only for doc-doc comparisons)"""
        if not events1 or not events2:
            return 0, {}
        
        # Extract event names and data
        names1 = list(events1.keys())
        names2 = list(events2.keys())
        data1 = list(events1.values())
        data2 = list(events2.values())
        
        # Use GPU-accelerated fuzzy similarity for event names
        similarity_matrix = self._gpu_fuzzy_similarity_batch(names1, names2, threshold=0.75)
        
        # Find best matches using GPU operations
        matched_events = []
        used_indices2 = set()
        threshold = 0.75
        
        for i, name1 in enumerate(names1):
            # Get similarities for this event
            similarities = similarity_matrix[i]
            
            # Find best match above threshold
            valid_indices = [j for j in range(len(names2)) if j not in used_indices2]
            if not valid_indices:
                continue
            
            valid_similarities = similarities[valid_indices]
            max_sim, max_idx_in_valid = torch.max(valid_similarities, dim=0)
            
            if max_sim.item() >= threshold:
                actual_idx = valid_indices[max_idx_in_valid.item()]
                matched_events.append((i, actual_idx, max_sim.item()))
                used_indices2.add(actual_idx)
        
        # Create combined events
        combined_events = {}
        for i, j, similarity_score in matched_events:
            event_key = names1[i]
            data1_item = data1[i]
            data2_item = data2[j]
            
            # Handle dates (should be strings)
            dates = []
            for date_val in [data1_item.get("date"), data2_item.get("date")]:
                if date_val and date_val not in dates:
                    dates.append(date_val)
            
            # Handle actions (might be lists or strings)
            actions = []
            for action_list in [data1_item.get("actions", []), data2_item.get("actions", [])]:
                if isinstance(action_list, list):
                    for action in action_list:
                        if action and action not in actions:
                            actions.append(action)
                elif action_list and action_list not in actions:
                    actions.append(action_list)
            
            # Handle details (might be strings or lists)
            details = []
            for detail_val in [data1_item.get("details"), data2_item.get("details")]:
                if detail_val:
                    if isinstance(detail_val, list):
                        for detail in detail_val:
                            if detail and detail not in details:
                                details.append(detail)
                    elif detail_val not in details:
                        details.append(detail_val)
            
            combined_events[event_key] = {
                "dates": dates,
                "actions": actions,
                "details": details
            }
        
        return len(matched_events), combined_events
    
    def _calculate_additional_features(self, chunk1: ChunkData, chunk2: ChunkData, edge_type: str) -> Dict[str, Any]:
        """Calculate all additional features based on edge type"""
        additional_features = {}
        
        # Get metadata for both chunks
        meta1 = chunk1.metadata or {}
        meta2 = chunk2.metadata or {}
        
        if edge_type == "doc-doc":
            # Doc-Doc specific features
            additional_features.update(self._calculate_doc_doc_features(chunk1, chunk2, meta1, meta2))
        elif edge_type == "doc-table":
            # Doc-Table specific features
            additional_features.update(self._calculate_doc_table_features(chunk1, chunk2, meta1, meta2))
        elif edge_type == "table-table":
            # Table-Table specific features
            additional_features.update(self._calculate_table_table_features(chunk1, chunk2, meta1, meta2))
        
        return additional_features
    
    def _calculate_doc_doc_features(self, doc1: ChunkData, doc2: ChunkData, 
                                   meta1: Dict[str, Any], meta2: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate features specific to doc-doc edges"""
        features = {}
        
        # 1. Use precomputed fuzzy entity matching and relationship overlap
        entity_match_data = self.precomputed_similarities['entity_fuzzy_matches'].get(
            (doc1.chunk_id, doc2.chunk_id)
        )
        if entity_match_data:
            entity_count, relationship_overlap = entity_match_data
            features['entity_count'] = entity_count
            
            # Calculate average relationship overlap
            if isinstance(relationship_overlap, dict) and relationship_overlap:
                avg_overlap = sum(relationship_overlap.values()) / len(relationship_overlap)
                features['entity_relationship_overlap'] = float(avg_overlap)
            else:
                features['entity_relationship_overlap'] = 0.0
        else:
            features['entity_count'] = 0
            features['entity_relationship_overlap'] = 0.0
        
        # 2. Use precomputed event overlap
        event_count = self.precomputed_similarities['event_fuzzy_matches'].get(
            (doc1.chunk_id, doc2.chunk_id), 0
        )
        features['event_count'] = event_count
        
        # Initialize other features to 0 for doc-doc
        features.update({
            'topic_title_similarity': 0.0,
            'topic_summary_similarity': 0.0,
            'title_similarity': 0.0,
            'description_similarity': 0.0
        })
        
        return features
    
    def _calculate_doc_table_features(self, chunk1: ChunkData, chunk2: ChunkData,
                                     meta1: Dict[str, Any], meta2: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate features specific to doc-table edges"""
        features = {}
        
        # Determine which is doc and which is table
        if chunk1.chunk_type == "document":
            doc_chunk, table_chunk = chunk1, chunk2
            doc_meta, table_meta = meta1, meta2
        else:
            doc_chunk, table_chunk = chunk2, chunk1
            doc_meta, table_meta = meta2, meta1
        
        # 1. Use precomputed fuzzy entity matching (mixed format)
        entity_match_data = self.precomputed_similarities['entity_fuzzy_matches'].get(
            (doc_chunk.chunk_id, table_chunk.chunk_id)
        )
        if entity_match_data:
            entity_count, _ = entity_match_data
            features['entity_count'] = entity_count
        else:
            features['entity_count'] = 0
        
        # 2. Topic to title similarity (using precomputed tensors)
        features['topic_title_similarity'] = self._calculate_text_similarity_gpu_optimized(
            doc_chunk.chunk_id, table_chunk.chunk_id, 'title')
        
        # 3. Topic to summary similarity (using precomputed tensors)
        features['topic_summary_similarity'] = self._calculate_text_similarity_gpu_optimized(
            doc_chunk.chunk_id, table_chunk.chunk_id, 'summary')
        
        # Initialize other features to 0 for doc-table
        features.update({
            'entity_relationship_overlap': 0.0,
            'event_count': 0,
            'title_similarity': 0.0,
            'description_similarity': 0.0
        })
        
        return features
    
    def _calculate_table_table_features(self, table1: ChunkData, table2: ChunkData,
                                       meta1: Dict[str, Any], meta2: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate features specific to table-table edges"""
        features = {}
        
        # 1. Use precomputed fuzzy entity matching (table format)
        entity_match_data = self.precomputed_similarities['entity_fuzzy_matches'].get(
            (table1.chunk_id, table2.chunk_id)
        )
        if entity_match_data:
            entity_count, _ = entity_match_data
            features['entity_count'] = entity_count
        else:
            features['entity_count'] = 0
        
        # 2. Title similarity (using precomputed tensors)
        features['title_similarity'] = self._calculate_table_table_text_similarity_gpu(
            table1.chunk_id, table2.chunk_id, 'title')
        
        # 3. Description similarity (using precomputed tensors)
        features['description_similarity'] = self._calculate_table_table_text_similarity_gpu(
            table1.chunk_id, table2.chunk_id, 'description')
        
        # Initialize other features to 0 for table-table
        features.update({
            'entity_relationship_overlap': 0.0,
            'event_count': 0,
            'topic_title_similarity': 0.0,
            'topic_summary_similarity': 0.0
        })
        
        return features
    
    def _calculate_text_similarity_gpu_optimized(self, doc_chunk_id: str, table_chunk_id: str, 
                                                similarity_type: str) -> float:
        """
        Calculate text similarity using precomputed lookup tables for maximum efficiency
        similarity_type: 'title', 'summary', 'description'
        """
        try:
            if similarity_type == 'title':
                # Use precomputed doc-topic to table-title similarity
                return self.precomputed_similarities['doc_topic_to_table_title'].get(
                    (doc_chunk_id, table_chunk_id), 0.0
                )
            elif similarity_type == 'summary':
                # Use precomputed doc-topic to table-summary similarity
                return self.precomputed_similarities['doc_topic_to_table_summary'].get(
                    (doc_chunk_id, table_chunk_id), 0.0
                )
            else:
                # For other types, fallback to 0.0
                return 0.0
            
        except Exception as e:
            logger.warning(f"Error looking up text similarity: {e}")
            return 0.0
    
    def _calculate_table_table_text_similarity_gpu(self, table1_chunk_id: str, table2_chunk_id: str,
                                                  similarity_type: str) -> float:
        """
        Calculate text similarity between two tables using precomputed lookup tables
        similarity_type: 'title', 'description'
        """
        try:
            if similarity_type == 'title':
                # Use precomputed table-title to table-title similarity
                return self.precomputed_similarities['table_title_to_table_title'].get(
                    (table1_chunk_id, table2_chunk_id), 0.0
                )
            elif similarity_type == 'description':
                # Use precomputed table-description to table-description similarity
                return self.precomputed_similarities['table_desc_to_table_desc'].get(
                    (table1_chunk_id, table2_chunk_id), 0.0
                )
            else:
                return 0.0
            
        except Exception as e:
            logger.warning(f"Error looking up table-table text similarity: {e}")
            return 0.0
    

    
    def _save_batch_results(self, batch_results: List[SimilarityMetrics], batch_id: int):
        """Save batch results to CSV file"""
        if not batch_results:
            return
        
        # Convert to DataFrame
        data = []
        for result in batch_results:
            data.append({
                'source_chunk_id': result.source_chunk_id,
                'target_chunk_id': result.target_chunk_id,
                'topic_similarity': result.topic_similarity,
                'content_similarity': result.content_similarity,
                'column_similarity': result.column_similarity,
                'edge_type': result.edge_type,
                # New features
                'entity_relationship_overlap': result.entity_relationship_overlap,
                'entity_count': result.entity_count,
                'event_count': result.event_count,
                'topic_title_similarity': result.topic_title_similarity,
                'topic_summary_similarity': result.topic_summary_similarity,
                'title_similarity': result.title_similarity,
                'description_similarity': result.description_similarity
            })
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        batch_file = self.batch_results_dir / f"batch_{batch_id:06d}.csv"
        df.to_csv(batch_file, index=False)
        
        logger.debug(f"Saved batch {batch_id} results to {batch_file}")
    
    def load_all_batch_results(self) -> List[SimilarityMetrics]:
        """Load all batch results from CSV files"""
        logger.info("Loading batch results from CSV files...")
        
        all_results = []
        batch_files = sorted(self.batch_results_dir.glob("batch_*.csv"))
        
        for batch_file in batch_files:
            try:
                df = pd.read_csv(batch_file)
                for _, row in df.iterrows():
                    similarity_metrics = SimilarityMetrics(
                        source_chunk_id=row['source_chunk_id'],
                        target_chunk_id=row['target_chunk_id'],
                        topic_similarity=row['topic_similarity'],
                        content_similarity=row['content_similarity'],
                        column_similarity=row['column_similarity'],
                        edge_type=row['edge_type'],
                        # New features (with defaults for backward compatibility)
                        entity_relationship_overlap=row.get('entity_relationship_overlap', 0.0),
                        entity_count=int(row.get('entity_count', 0)),
                        event_count=int(row.get('event_count', 0)),
                        topic_title_similarity=row.get('topic_title_similarity', 0.0),
                        topic_summary_similarity=row.get('topic_summary_similarity', 0.0),
                        title_similarity=row.get('title_similarity', 0.0),
                        description_similarity=row.get('description_similarity', 0.0)
                    )
                    all_results.append(similarity_metrics)
            except Exception as e:
                logger.warning(f"Error loading batch file {batch_file}: {e}")
        
        logger.info(f"Loaded {len(all_results)} similarity results from {len(batch_files)} batch files")
        return all_results
    
    def clear_batch_results(self):
        """Clear all batch result files"""
        batch_files = list(self.batch_results_dir.glob("batch_*.csv"))
        for batch_file in batch_files:
            batch_file.unlink()
        logger.info(f"Cleared {len(batch_files)} batch result files")

class GraphAnalysisPipeline:
    """
    Comprehensive pipeline for analyzing chunk similarities and optimizing graph construction
    """
    
    def __init__(self, 
                 doc_chunks_dir: str = "/shared/khoja/CogComp/output/full_pipeline/docs_chunks_1",
                 table_chunks_file: str = "/shared/khoja/CogComp/output/full_pipeline/table_chunks_with_metadata.json",
                 cache_dir: str = "/shared/khoja/CogComp/output/analysis_cache",
                 use_gpu: bool = True,
                 num_threads: int = 128):
        
        self.doc_chunks_dir = Path(doc_chunks_dir)
        self.table_chunks_file = Path(table_chunks_file)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.num_threads = num_threads
        
        # Initialize embedding service
        self.config = ProcessingConfig(use_faiss=True, faiss_use_gpu=self.use_gpu)
        self.embedding_service = EmbeddingService(self.config)
        
        # Data containers
        self.chunks: List[ChunkData] = []
        self.chunk_index: Dict[str, int] = {} # chunk_id -> index in self.chunks
        self.faiss_index = None
        self.similarity_data: List[SimilarityMetrics] = []
        
        # Thread-safe lock for data updates
        self.data_lock = threading.Lock()
        
        # Store outlier edges for graph building (optimization)
        self.outlier_edges = [] # Will be populated during analysis
        
        logger.info("Initialized graph analysis pipeline")

    def load_chunks(self) -> None:
        """Load all document and table chunks with caching"""
        logger.info("Loading document and table chunks...")
        
        indexing_filtered = pd.read_csv("/shared/khoja/CogComp/output/indexing_filtered.csv")
        # Build a fast lookup set of allowed chunk_ids (as strings)
        try:
            allowed_chunk_ids = set(indexing_filtered['chunk_id'].astype(str).tolist())
        except Exception:
            # Fallback if column name differs or file is empty
            allowed_chunk_ids = set()

        # Load from cache if available
        cache_file = self.cache_dir / "processed_chunks.pkl"
        if cache_file.exists():
            logger.info("Loading chunks from cache...")
            with open(cache_file, 'rb') as f:
                self.chunks = pickle.load(f)
            self._build_chunk_index()
            logger.info(f"Loaded {len(self.chunks)} chunks from cache")
            return
        
        # Load document chunks
        doc_chunks = self._load_document_chunks(allowed_chunk_ids)
        logger.info(f"Loaded {len(doc_chunks)} document chunks")
        
        # Load table chunks
        table_chunks = self._load_table_chunks(allowed_chunk_ids)
        logger.info(f"Loaded {len(table_chunks)} table chunks")
        
        # Combine all chunks
        self.chunks = doc_chunks + table_chunks
        self._build_chunk_index()
        
        # Process embeddings for all chunks
        self._process_embeddings()
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        logger.info(f"Processed and cached {len(self.chunks)} total chunks")

    def _load_document_chunks(self, allowed_chunk_ids: set) -> List[ChunkData]:
        """Load document chunks from directory"""
        doc_chunks = []
        
        for json_file in self.doc_chunks_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract entity names for embedding
                entities = data.get('metadata', {}).get('entities', {})
                topic = data.get('metadata', {}).get('topic', '')
                
                chunk_data = ChunkData(
                    chunk_id=data['chunk_id'],
                    chunk_type="document",
                    content=data['content'],
                    content_embedding=data.get('embedding', []),
                    entities=entities,
                    entity_embeddings={},
                    topic=topic,
                    topic_embedding=None,
                    metadata=data.get('metadata', {})
                )
                # If a filter set is provided, keep only allowed ids
                if allowed_chunk_ids:
                    if str(chunk_data.chunk_id) in allowed_chunk_ids:
                        doc_chunks.append(chunk_data)
                else:
                    doc_chunks.append(chunk_data)
                
            except Exception as e:
                logger.warning(f"Error loading document chunk {json_file}: {e}")
                continue
        
        return doc_chunks

    def _load_table_chunks(self, allowed_chunk_ids: set) -> List[ChunkData]:
        """Load table chunks from JSON file"""
        table_chunks = []
        
        try:
            with open(self.table_chunks_file, 'r') as f:
                data = json.load(f)
            
            for chunk_info in data:
                entities = chunk_info.get('metadata', {}).get('entities', {})
                col_desc = chunk_info.get('metadata', {}).get('col_desc', {})
                
                # Use table_description as topic for tables
                table_description = chunk_info.get('metadata', {}).get('table_description', '')
                
                chunk_data = ChunkData(
                    chunk_id=chunk_info['chunk_id'],
                    chunk_type="table",
                    content=chunk_info['content'],
                    content_embedding=[], # Will be generated
                    entities=entities,
                    entity_embeddings={},
                    topic=table_description, # Use table_description as topic
                    topic_embedding=None,
                    column_descriptions=col_desc,
                    column_embeddings={},
                    metadata=chunk_info.get('metadata', {})
                )
                # If a filter set is provided, keep only allowed ids
                if allowed_chunk_ids:
                    if str(chunk_data.chunk_id) in allowed_chunk_ids:
                        table_chunks.append(chunk_data)
                else:
                    table_chunks.append(chunk_data)
                
        except Exception as e:
            logger.error(f"Error loading table chunks: {e}")
            return []
        
        return table_chunks

    def _build_chunk_index(self):
        """Build chunk ID to index mapping"""
        self.chunk_index = {chunk.chunk_id: i for i, chunk in enumerate(self.chunks)}

    def _process_embeddings(self):
        """Process all required embeddings with threading and GPU acceleration"""
        logger.info("Processing embeddings for all chunks...")
        
        # Collect all texts that need embeddings
        content_texts = []
        entity_texts = []
        topic_texts = []
        column_texts = []
        
        content_indices = []
        entity_indices = []
        topic_indices = []
        column_indices = []
        
        for i, chunk in enumerate(self.chunks):
            # Content embeddings (for table chunks that don't have them)
            if not chunk.content_embedding and chunk.content:
                content_texts.append(chunk.content)
                content_indices.append(i)
            
            # Entity embeddings (just entity names)
            for entity_name in chunk.entities.keys():
                if entity_name not in chunk.entity_embeddings:
                    entity_texts.append(entity_name)
                    entity_indices.append((i, entity_name))
            
            # Topic embeddings
            if chunk.topic and not chunk.topic_embedding:
                topic_texts.append(chunk.topic)
                topic_indices.append(i)
            
            # Column description embeddings (for tables)
            if chunk.column_descriptions:
                for col_name, col_desc in chunk.column_descriptions.items():
                    if col_name not in (chunk.column_embeddings or {}):
                        column_texts.append(col_desc)
                        column_indices.append((i, col_name))

        # Generate embeddings in batches
        batch_size = 64
        
        # Content embeddings
        if content_texts:
            logger.info(f"Generating {len(content_texts)} content embeddings...")
            content_embeddings = self.embedding_service.generate_embeddings(content_texts)
            for idx, embedding in zip(content_indices, content_embeddings):
                self.chunks[idx].content_embedding = embedding
        
        # Skip entity embeddings - using fuzzy matching instead
        logger.info("Skipping entity embeddings - using GPU fuzzy matching instead")
        
        # Topic embeddings
        if topic_texts:
            logger.info(f"Generating {len(topic_texts)} topic embeddings...")
            topic_embeddings = self.embedding_service.generate_embeddings(topic_texts)
            for chunk_idx, embedding in zip(topic_indices, topic_embeddings):
                self.chunks[chunk_idx].topic_embedding = embedding
        
        # Column embeddings
        if column_texts:
            logger.info(f"Generating {len(column_texts)} column embeddings...")
            column_embeddings = self.embedding_service.generate_embeddings(column_texts)
            for (chunk_idx, col_name), embedding in zip(column_indices, column_embeddings):
                if self.chunks[chunk_idx].column_embeddings is None:
                    self.chunks[chunk_idx].column_embeddings = {}
                self.chunks[chunk_idx].column_embeddings[col_name] = embedding

    def build_hnsw_index(self):
        """Build HNSW FAISS index from all chunk content embeddings"""
        logger.info("Building HNSW FAISS index...")
        
        # Collect embeddings
        embeddings = []
        valid_chunk_indices = []
        
        for i, chunk in enumerate(self.chunks):
            if chunk.content_embedding:
                embeddings.append(chunk.content_embedding)
                valid_chunk_indices.append(i)
        
        if not embeddings:
            raise ValueError("No embeddings found for HNSW index")
        
        # Convert to numpy array
        embeddings_matrix = np.array(embeddings, dtype=np.float32)
        dimension = embeddings_matrix.shape[1];
        
        logger.info(f"Building HNSW index with {len(embeddings)} embeddings, dimension={dimension}")
        
        # Create HNSW index
        self.faiss_index = faiss.IndexHNSWFlat(dimension, 32) # M=32
        self.faiss_index.hnsw.efConstruction = 1000
        self.faiss_index.hnsw.efSearch = 500
        
        # Use GPU if available
        if self.use_gpu and faiss.get_num_gpus() > 0:
            try:
                # Normalize for cosine similarity
                faiss.normalize_L2(embeddings_matrix)
                
                res = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
                gpu_index.add(embeddings_matrix)
                
                # Move back to CPU for thread safety
                self.faiss_index = faiss.index_gpu_to_cpu(gpu_index)
                logger.info("Built HNSW index on GPU and moved to CPU")
            except Exception as e:
                logger.warning(f"GPU FAISS failed: {e}, using CPU")
                faiss.normalize_L2(embeddings_matrix)
                self.faiss_index.add(embeddings_matrix)
        else:
            faiss.normalize_L2(embeddings_matrix)
            self.faiss_index.add(embeddings_matrix)
        
        # Store mapping from FAISS index to chunk index
        self.faiss_to_chunk_mapping = valid_chunk_indices
        
        logger.info("HNSW index built successfully")

    def analyze_chunk_similarities(self, k_neighbors: int = 1000):
        """Analyze similarities for all chunks with their k nearest neighbors using optimized 3-phase approach"""
        logger.info(f"Analyzing similarities for {len(self.chunks)} chunks with {k_neighbors} neighbors each...")
        
        # Check for batch results first (from GPU processing)
        batch_results_dir = Path("/shared/khoja/CogComp/output/batch_results")
        if batch_results_dir.exists() and list(batch_results_dir.glob("batch_*.csv")):
            logger.info("Loading similarity analysis from batch result files...")
            gpu_calculator = GPUAcceleratedSimilarityCalculator(
                chunks=self.chunks,
                embedding_service=self.embedding_service,
                batch_size=10000,
                gpu_id=0
            )
            self.similarity_data = gpu_calculator.load_all_batch_results()
            logger.info(f"Loaded {len(self.similarity_data)} similarity records from batch files")
            return
        
        # Load cached results if available
        cache_file = self.cache_dir / f"similarity_analysis_k{k_neighbors}.pkl"
        if cache_file.exists():
            logger.info("Loading similarity analysis from cache...")
            with open(cache_file, 'rb') as f:
                self.similarity_data = pickle.load(f)
            logger.info(f"Loaded {len(self.similarity_data)} similarity records from cache")
            return
        
        # Phase 1: Find all k-neighbors for all chunks (parallel)
        logger.info("Phase 1: Finding k-neighbors for all chunks...")
        all_neighbor_relationships = self._find_all_neighbors_parallel(k_neighbors)
        
        # Phase 2: Generate unique pairs from neighbor relationships
        logger.info("Phase 2: Generating unique pairs...")
        unique_pairs = self._generate_unique_pairs(all_neighbor_relationships)
        logger.info(f"Generated {len(unique_pairs)} unique pairs from neighbor relationships")
        
        # Phase 3: Calculate similarities for unique pairs (GPU accelerated)
        logger.info("Phase 3: Calculating similarities for unique pairs...")
        self._calculate_similarities_for_pairs(unique_pairs)
        
        # Save to cache for future use
        with open(cache_file, 'wb') as f:
            pickle.dump(self.similarity_data, f)
        
        logger.info(f"Similarity analysis completed: {len(self.similarity_data)} total records")

    def _find_all_neighbors_parallel(self, k_neighbors: int) -> Dict[str, List[str]]:
        """Phase 1: Find k-neighbors for all chunks in parallel, return chunk_id -> neighbor_chunk_ids mapping"""
        all_neighbor_relationships = {}
        
        # Create chunk batches for parallel processing
        chunk_batches = []
        batch_size = max(1, len(self.chunks) // self.num_threads)
        
        for i in range(0, len(self.chunks), batch_size):
            chunk_batches.append(self.chunks[i:i + batch_size])
        
        # Process batches in parallel to find neighbors
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for batch_id, chunk_batch in enumerate(chunk_batches):
                future = executor.submit(self._find_neighbors_for_batch, batch_id, chunk_batch, k_neighbors)
                futures.append(future)
            
            # Collect neighbor relationships
            for future in as_completed(futures):
                try:
                    batch_relationships = future.result()
                    all_neighbor_relationships.update(batch_relationships)
                    logger.debug(f"Completed neighbor discovery for batch with {len(batch_relationships)} chunks")
                except Exception as e:
                    logger.error(f"Neighbor discovery batch failed: {e}")
        
        logger.info(f"Found neighbors for {len(all_neighbor_relationships)} chunks")
        return all_neighbor_relationships

    def _find_neighbors_for_batch(self, batch_id: int, chunk_batch: List[ChunkData], k_neighbors: int) -> Dict[str, List[str]]:
        """Find k-neighbors for a batch of chunks, return chunk_id -> neighbor_chunk_ids mapping"""
        batch_relationships = {}
        
        for chunk in chunk_batch:
            try:
                # Find k nearest neighbors using FAISS
                neighbor_chunk_ids = self._find_k_neighbor_ids(chunk, k_neighbors)
                batch_relationships[chunk.chunk_id] = neighbor_chunk_ids
                
            except Exception as e:
                logger.warning(f"Error finding neighbors for chunk {chunk.chunk_id}: {e}")
                batch_relationships[chunk.chunk_id] = []
                continue
        
        logger.debug(f"Batch {batch_id} found neighbors for {len(batch_relationships)} chunks")
        return batch_relationships

    def _find_k_neighbor_ids(self, chunk: ChunkData, k: int) -> List[str]:
        """Find k nearest neighbor chunk IDs for a chunk using FAISS"""
        if not chunk.content_embedding:
            return []
        
        try:
            # Prepare query embedding
            query_embedding = np.array([chunk.content_embedding], dtype=np.float32)
            faiss.normalize_L2(query_embedding)
            
            # Search in FAISS index
            search_k = min(k + 1, self.faiss_index.ntotal) # +1 to account for self
            similarities, indices = self.faiss_index.search(query_embedding, search_k)
            
            neighbor_chunk_ids = []
            for sim, idx in zip(similarities[0], indices[0]):
                if idx == -1: # Invalid index
                    continue
                
                chunk_idx = self.faiss_to_chunk_mapping[idx]
                neighbor_chunk = self.chunks[chunk_idx]
                
                # Skip self
                if neighbor_chunk.chunk_id == chunk.chunk_id:
                    continue
                
                neighbor_chunk_ids.append(neighbor_chunk.chunk_id)
            
            return neighbor_chunk_ids[:k]
            
        except Exception as e:
            logger.warning(f"Error finding neighbors for {chunk.chunk_id}: {e}")
            return []

    def _generate_unique_pairs(self, all_neighbor_relationships: Dict[str, List[str]]) -> List[Tuple[str, str]]:
        """Phase 2: Generate unique pairs from neighbor relationships, avoiding duplicates"""
        unique_pairs_set = set()
        
        for source_chunk_id, neighbor_chunk_ids in all_neighbor_relationships.items():
            for target_chunk_id in neighbor_chunk_ids:
                # Create ordered pair using lexicographic ordering to ensure uniqueness
                if source_chunk_id < target_chunk_id:
                    pair = (source_chunk_id, target_chunk_id)
                else:
                    pair = (target_chunk_id, source_chunk_id)
                
                unique_pairs_set.add(pair)
        
        return list(unique_pairs_set)

    def _calculate_similarities_for_pairs(self, unique_pairs: List[Tuple[str, str]]):
        """Phase 3: Calculate similarities for unique pairs using GPU acceleration only"""
        
        # GPU-only processing - no CPU fallback
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not available! This pipeline requires CUDA-capable GPU.")
        
        logger.info("Using GPU-only similarity calculation")
        
        # Initialize GPU calculator
        gpu_calculator = GPUAcceleratedSimilarityCalculator(
            chunks=self.chunks,
            embedding_service=self.embedding_service,
            batch_size=10000, # Use 10k pairs per batch as requested
            gpu_id=0 # Use first GPU only
        )
        
        # Precompute embeddings on GPU
        gpu_calculator.precompute_embeddings()
        
        # Precompute all similarities for the unique pairs
        gpu_calculator._precompute_all_similarities(unique_pairs)
        
        # Calculate similarities using GPU batch processing
        self.similarity_data = gpu_calculator.calculate_similarities_batch(unique_pairs)
        
        logger.info(f"GPU-only processing completed: {len(self.similarity_data)} similarities calculated")

    def generate_analysis_reports(self):
        """Generate comprehensive analysis reports and visualizations by edge type sections"""
        logger.info("Generating analysis reports and visualizations...")
        
        if not self.similarity_data:
            logger.error("No similarity data available. Run analyze_chunk_similarities first.")
            return
        
        # Create output directory
        output_dir = self.cache_dir / "analysis_reports"
        output_dir.mkdir(exist_ok=True)
        
        # Convert to DataFrame for analysis
        df = self._create_similarity_dataframe()
        
        # Generate analysis by sections
        self._generate_section_1_doc_doc_analysis(df, output_dir)
        self._generate_section_2_doc_table_analysis(df, output_dir)
        self._generate_section_3_table_table_analysis(df, output_dir)
        
        # Generate overall summary
        self._generate_overall_summary(df, output_dir)
        
        # NEW: Generate outlier analysis
        self._generate_outlier_analysis(df, output_dir)
        
        logger.info(f"Analysis reports saved to {output_dir}")

    def _create_similarity_dataframe(self) -> pd.DataFrame:
        """Convert similarity data to pandas DataFrame"""
        data = []
        
        for sim in self.similarity_data:
            data.append({
                'topic_similarity': sim.topic_similarity,
                'content_similarity': sim.content_similarity,
                'column_similarity': sim.column_similarity,
                'edge_type': sim.edge_type,
                'source_chunk_id': sim.source_chunk_id,
                'target_chunk_id': sim.target_chunk_id,
                # New features
                'entity_relationship_overlap': sim.entity_relationship_overlap,
                'entity_count': sim.entity_count,
                'event_count': sim.event_count,
                'topic_title_similarity': sim.topic_title_similarity,
                'topic_summary_similarity': sim.topic_summary_similarity,
                'title_similarity': sim.title_similarity,
                'description_similarity': sim.description_similarity
            })
        
        return pd.DataFrame(data)

    def _generate_section_1_doc_doc_analysis(self, df: pd.DataFrame, output_dir: Path):
        """Generate Section 1: Doc-Doc edge type analysis"""
        logger.info("Generating Section 1: Doc-Doc Analysis...")
        
        # Filter for doc-doc edges only
        doc_doc_df = df[df['edge_type'] == 'doc-doc'].copy()
        
        if len(doc_doc_df) == 0:
            logger.warning("No doc-doc edges found for analysis")
            return
        
        section_dir = output_dir / "section_1_doc_doc"
        section_dir.mkdir(exist_ok=True)
        
        # Metrics relevant for doc-doc: existing + new features
        relevant_cols = [
            'topic_similarity', 'content_similarity',
            'entity_relationship_overlap', 'entity_count', 'event_count'
        ]
        
        # Correlation analysis
        correlation_matrix = doc_doc_df[relevant_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.3f')
        plt.title('Section 1: Doc-Doc Correlation Matrix')
        plt.tight_layout()
        plt.savefig(section_dir / 'doc_doc_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Distribution analysis
        n_cols = len(relevant_cols)
        cols_per_row = 3
        n_rows = (n_cols + cols_per_row - 1) // cols_per_row # Ceiling division
        fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(18, 6*n_rows))
        
        # Handle single row case
        if n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(relevant_cols):
            sns.histplot(data=doc_doc_df, x=col, kde=True, ax=axes[i], bins=30)
            axes[i].set_title(f'Doc-Doc: {col.replace("_", " ").title()}')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(relevant_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(section_dir / 'doc_doc_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Scatter plots
        for i, col1 in enumerate(relevant_cols):
            for j, col2 in enumerate(relevant_cols[i+1:], i+1):
                plt.figure(figsize=(10, 8))
                plt.scatter(doc_doc_df[col1], doc_doc_df[col2], alpha=0.6, s=20)
                plt.xlabel(col1.replace('_', ' ').title())
                plt.ylabel(col2.replace('_', ' ').title())
                plt.title(f'Doc-Doc: {col1.replace("_", " ").title()} vs {col2.replace("_", " ").title()}')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(section_dir / f'doc_doc_scatter_{col1}_vs_{col2}.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # Statistics
        stats = doc_doc_df[relevant_cols].describe()
        stats.to_csv(section_dir / 'doc_doc_statistics.csv')
        
        # Insights
        insights = [
            "SECTION 1: DOC-DOC EDGE ANALYSIS",
            f"Total doc-doc edges analyzed: {len(doc_doc_df)}",
            "",
            "FEATURES ANALYZED:",
            " - Topic Similarity: Document topic vs document topic",
            " - Content Similarity: Full content embeddings similarity",
            " - Entity Relationship Overlap: Relationship overlap for fuzzy matched entities",
            " - Entity Count: Fuzzy matched entities (75% threshold)",
            " - Event Count: Fuzzy matched events between documents",
            "",
            "SIMILARITY METRICS SUMMARY:",
        ]
        
        for col in relevant_cols:
            mean_val = doc_doc_df[col].mean()
            std_val = doc_doc_df[col].std()
            insights.append(f" {col}: Mean={mean_val:.3f}, Std={std_val:.3f}")
        
        insights.append("")
        insights.append("HIGH CORRELATIONS (|r| > 0.7):")
        for i, col1 in enumerate(relevant_cols):
            for j, col2 in enumerate(relevant_cols[i+1:], i+1):
                corr = correlation_matrix.loc[col1, col2]
                if abs(corr) > 0.7:
                    insights.append(f" {col1} - {col2}: {corr:.3f}")
        
        with open(section_dir / 'doc_doc_insights.txt', 'w') as f:
            f.write('\n'.join(insights))

    def _generate_section_2_doc_table_analysis(self, df: pd.DataFrame, output_dir: Path):
        """Generate Section 2: Doc-Table edge type analysis"""
        logger.info("Generating Section 2: Doc-Table Analysis...")
        
        # Filter for doc-table edges only
        doc_table_df = df[df['edge_type'] == 'doc-table'].copy()
        
        if len(doc_table_df) == 0:
            logger.warning("No doc-table edges found for analysis")
            return
        
        section_dir = output_dir / "section_2_doc_table"
        section_dir.mkdir(exist_ok=True)
        
        # Metrics relevant for doc-table: existing + new features
        # Note: column_similarity = table column descriptions vs document content/topic
        relevant_cols = [
            'column_similarity',
            'entity_count', 'topic_title_similarity', 'topic_summary_similarity'
        ]
        
        # Correlation analysis
        correlation_matrix = doc_table_df[relevant_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.3f')
        plt.title('Section 2: Doc-Table Correlation Matrix')
        plt.tight_layout()
        plt.savefig(section_dir / 'doc_table_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Distribution analysis
        n_cols = len(relevant_cols)
        cols_per_row = 3
        n_rows = (n_cols + cols_per_row - 1) // cols_per_row # Ceiling division
        fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(18, 6*n_rows))
        
        # Handle single row case
        if n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(relevant_cols):
            sns.histplot(data=doc_table_df, x=col, kde=True, ax=axes[i], bins=30)
            axes[i].set_title(f'Doc-Table: {col.replace("_", " ").title()}')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(relevant_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(section_dir / 'doc_table_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Scatter plots
        for i, col1 in enumerate(relevant_cols):
            for j, col2 in enumerate(relevant_cols[i+1:], i+1):
                plt.figure(figsize=(10, 8))
                plt.scatter(doc_table_df[col1], doc_table_df[col2], alpha=0.6, s=20)
                plt.xlabel(col1.replace('_', ' ').title())
                plt.ylabel(col2.replace('_', ' ').title())
                plt.title(f'Doc-Table: {col1.replace("_", " ").title()} vs {col2.replace("_", " ").title()}')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(section_dir / f'doc_table_scatter_{col1}_vs_{col2}.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # Statistics
        stats = doc_table_df[relevant_cols].describe()
        stats.to_csv(section_dir / 'doc_table_statistics.csv')
        
        # Insights
        insights = [
            "SECTION 2: DOC-TABLE EDGE ANALYSIS",
            f"Total doc-table edges analyzed: {len(doc_table_df)}",
            "",
            "FEATURES ANALYZED:",
            " - Column Similarity: Document content/topic vs table column descriptions", 
            " - Entity Count: Fuzzy matched entities (75% threshold, mixed format)",
            " - Topic-Title Similarity: Document topic vs table title",
            " - Topic-Summary Similarity: Document topic vs table summary",
            "",
            "SIMILARITY METRICS SUMMARY:",
        ]
        
        for col in relevant_cols:
            mean_val = doc_table_df[col].mean()
            std_val = doc_table_df[col].std()
            insights.append(f" {col}: Mean={mean_val:.3f}, Std={std_val:.3f}")
        
        insights.append("")
        insights.append("HIGH CORRELATIONS (|r| > 0.7):")
        for i, col1 in enumerate(relevant_cols):
            for j, col2 in enumerate(relevant_cols[i+1:], i+1):
                corr = correlation_matrix.loc[col1, col2]
                if abs(corr) > 0.7:
                    insights.append(f" {col1} - {col2}: {corr:.3f}")
        
        with open(section_dir / 'doc_table_insights.txt', 'w') as f:
            f.write('\n'.join(insights))

    def _generate_section_3_table_table_analysis(self, df: pd.DataFrame, output_dir: Path):
        """Generate Section 3: Table-Table edge type analysis"""
        logger.info("Generating Section 3: Table-Table Analysis...")
        
        # Filter for table-table edges only
        table_table_df = df[df['edge_type'] == 'table-table'].copy()
        
        if len(table_table_df) == 0:
            logger.warning("No table-table edges found for analysis")
            return
        
        section_dir = output_dir / "section_3_table_table"
        section_dir.mkdir(exist_ok=True)
        
        # Metrics relevant for table-table: existing + new features
        relevant_cols = [
            'column_similarity',
            'entity_count', 'title_similarity', 'description_similarity'
        ]
        
        # Correlation analysis
        correlation_matrix = table_table_df[relevant_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.3f')
        plt.title('Section 3: Table-Table Correlation Matrix')
        plt.tight_layout()
        plt.savefig(section_dir / 'table_table_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Distribution analysis
        n_cols = len(relevant_cols)
        cols_per_row = 3
        n_rows = (n_cols + cols_per_row - 1) // cols_per_row # Ceiling division
        fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(18, 6*n_rows))
        
        # Handle single row case
        if n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(relevant_cols):
            sns.histplot(data=table_table_df, x=col, kde=True, ax=axes[i], bins=30)
            axes[i].set_title(f'Table-Table: {col.replace("_", " ").title()}')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(relevant_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(section_dir / 'table_table_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Scatter plots
        for i, col1 in enumerate(relevant_cols):
            for j, col2 in enumerate(relevant_cols[i+1:], i+1):
                plt.figure(figsize=(10, 8))
                plt.scatter(table_table_df[col1], table_table_df[col2], alpha=0.6, s=20)
                plt.xlabel(col1.replace('_', ' ').title())
                plt.ylabel(col2.replace('_', ' ').title())
                plt.title(f'Table-Table: {col1.replace("_", " ").title()} vs {col2.replace("_", " ").title()}')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(section_dir / f'table_table_scatter_{col1}_vs_{col2}.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # Statistics
        stats = table_table_df[relevant_cols].describe()
        stats.to_csv(section_dir / 'table_table_statistics.csv')
        
        # Insights
        insights = [
            "SECTION 3: TABLE-TABLE EDGE ANALYSIS",
            f"Total table-table edges analyzed: {len(table_table_df)}",
            "",
            "FEATURES ANALYZED:",
            " - Column Similarity: Column description similarities",
            " - Entity Count: Fuzzy matched entities (75% threshold, table format)",
            " - Title Similarity: Table title vs table title",
            " - Description Similarity: Table description vs table description",
            "",
            "SIMILARITY METRICS SUMMARY:",
        ]
        
        for col in relevant_cols:
            mean_val = table_table_df[col].mean()
            std_val = table_table_df[col].std()
            insights.append(f" {col}: Mean={mean_val:.3f}, Std={std_val:.3f}")
        
        insights.append("")
        insights.append("HIGH CORRELATIONS (|r| > 0.7):")
        for i, col1 in enumerate(relevant_cols):
            for j, col2 in enumerate(relevant_cols[i+1:], i+1):
                corr = correlation_matrix.loc[col1, col2]
                if abs(corr) > 0.7:
                    insights.append(f" {col1} - {col2}: {corr:.3f}")
        
        with open(section_dir / 'table_table_insights.txt', 'w') as f:
            f.write('\n'.join(insights))

    def _generate_overall_summary(self, df: pd.DataFrame, output_dir: Path):
        """Generate overall summary across all edge types"""
        logger.info("Generating overall summary...")
        
        # Edge type distribution
        edge_counts = df['edge_type'].value_counts()
        
        plt.figure(figsize=(10, 6))
        edge_counts.plot(kind='bar')
        plt.title('Distribution of Edge Types')
        plt.xlabel('Edge Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'overall_edge_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Overall statistics - all numeric features
        numeric_cols = [
            'topic_similarity', 'content_similarity', 'column_similarity',
            'entity_relationship_overlap', 'entity_count', 'event_count',
            'topic_title_similarity', 'topic_summary_similarity',
            'title_similarity', 'description_similarity'
        ]
        overall_stats = df[numeric_cols].describe()
        overall_stats.to_csv(output_dir / 'overall_statistics.csv')
        
        # Edge type comparison
        edge_type_stats = df.groupby('edge_type')[numeric_cols].mean()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(edge_type_stats.T, annot=True, cmap='viridis', fmt='.3f')
        plt.title('Average Similarity Metrics by Edge Type')
        plt.ylabel('Similarity Metrics')
        plt.xlabel('Edge Type')
        plt.tight_layout()
        plt.savefig(output_dir / 'overall_comparison_by_edge_type.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Summary insights
        insights = [
            "OVERALL GRAPH ANALYSIS SUMMARY",
            "=" * 40,
            "",
            "EDGE TYPE DISTRIBUTION:",
        ]
        
        for edge_type, count in edge_counts.items():
            percentage = (count / len(df)) * 100
            insights.append(f" {edge_type}: {count:,} edges ({percentage:.1f}%)")
        
        insights.extend([
            "",
            "METHODOLOGY:",
            " CORE FEATURES:",
            " - Topic Similarity: Document topic vs document topic (doc-doc edges only)",
            " - Content Similarity: Cosine similarity of full content embeddings", 
            " - Column Similarity: Content vs column descriptions (doc-table) or column-column (table-table)",
            "",
            " NEW FEATURES (GPU-Accelerated):",
            " - Entity Count: Fuzzy matched entities (75% similarity threshold)",
            " - Entity Relationship Overlap: Average relationship overlap for matched entities (doc-doc)",
            " - Event Count: Fuzzy matched events between documents (doc-doc)",
            " - Topic-Title Similarity: Document topic vs table title (doc-table)",
            " - Topic-Summary Similarity: Document topic vs table summary (doc-table)",
            " - Title Similarity: Table title vs table title (table-table)",
            " - Description Similarity: Table description vs table description (table-table)",
            "",
            "SECTIONS GENERATED:",
            " 1. Section 1: Doc-Doc edge analysis",
            " 2. Section 2: Doc-Table edge analysis", 
            " 3. Section 3: Table-Table edge analysis",
            "",
            "RECOMMENDATIONS FOR GRAPH CONSTRUCTION:",
            " - Review correlation matrices in each section to identify independent metrics",
            " - Use weighted averages only for uncorrelated metrics (|r| < 0.7)",
            " - Consider edge-type specific weighting based on metric distributions",
        ])
        
        with open(output_dir / 'overall_summary.txt', 'w') as f:
            f.write('\n'.join(insights))

    def _generate_outlier_analysis(self, df: pd.DataFrame, output_dir: Path):
        """Generate outlier analysis for all sections - finding rows where one metric is high but paired metric is low"""
        logger.info("Generating outlier analysis for all sections...")
        
        # Create outlier analysis directory
        outlier_dir = output_dir / "outlier_analysis"
        outlier_dir.mkdir(exist_ok=True)
        
        # Initialize outlier edges storage
        self.outlier_edges = []
        
        # Section 1: Doc-Doc Analysis
        self._analyze_doc_doc_outliers(df, outlier_dir)
        
        # Section 2: Doc-Table Analysis 
        self._analyze_doc_table_outliers(df, outlier_dir)
        
        # Section 3: Table-Table Analysis
        self._analyze_table_table_outliers(df, outlier_dir)
        
        logger.info(f"Outlier analysis completed and saved to {outlier_dir}")
        logger.info(f"Stored {len(self.outlier_edges)} outlier edges for graph building")

    def _analyze_doc_doc_outliers(self, df: pd.DataFrame, outlier_dir: Path):
        """Analyze outliers for doc-doc edges: topic_similarity vs content_similarity"""
        logger.info("Analyzing doc-doc outliers...")
        
        # Filter for doc-doc edges
        doc_doc_df = df[df['edge_type'] == 'doc-doc'].copy()
        
        if len(doc_doc_df) == 0:
            logger.warning("No doc-doc edges found for outlier analysis")
            return
        
        section_dir = outlier_dir / "section_1_doc_doc_outliers"
        section_dir.mkdir(exist_ok=True)
        
        # Variables to analyze
        var1, var2 = 'topic_similarity', 'content_similarity'
        # Calculate 99th percentile thresholds
        threshold_1 = doc_doc_df[var1].quantile(0.99)
        threshold_2 = doc_doc_df[var2].quantile(0.99)

        logger.info(f"Doc-Doc thresholds: {var1}={threshold_1:.3f}, {var2}={threshold_2:.3f}")
        
        # Case 1: Both topic_similarity AND content_similarity high (AND logic)
        case_mask = (doc_doc_df[var2] > threshold_2) & (doc_doc_df[var1] > threshold_1) 
        case_outliers = doc_doc_df[case_mask].copy()
        
        # Store outliers for graph building
        self._store_outliers_for_graph_building(case_outliers, 'doc-doc', f'high_content_and_topic')
        
        # Also store edges with entity matches (regardless of similarity thresholds)
        entity_match_mask = doc_doc_df['entity_count'] > 0
        entity_match_outliers = doc_doc_df[entity_match_mask].copy()
        self._store_outliers_for_graph_building(entity_match_outliers, 'doc-doc', 'entity_match')
        
        # Save outliers with content - File 1: AND operation outliers
        self._save_outliers_with_content(
            case_outliers, 
            section_dir / f"high_{var2}_and_{var1}",
            f"High {var2.replace('_', ' ').title()} AND High {var1.replace('_', ' ').title()}",
            f"{var1}_and_{var2}"
        )
        
        # Save outliers with content - File 2: Entity count > 0
        self._save_outliers_with_content(
            entity_match_outliers,
            section_dir / f"entity_count_greater_than_0", 
            f"Entity Count Greater Than 0",
            f"entity_match"
        )
        
        # Save outliers with content
        # self._save_outliers_with_content(
        # case1_outliers, 
        # section_dir / f"case1_high_{var2}_low_{var1}",
        # f"High {var2.replace('_', ' ').title()}, Low {var1.replace('_', ' ').title()}",
        # f"{var1}_vs_{var2}"
        # )
        
        # self._save_outliers_with_content(
        # case2_outliers,
        # section_dir / f"case2_high_{var1}_low_{var2}", 
        # f"High {var1.replace('_', ' ').title()}, Low {var2.replace('_', ' ').title()}",
        # f"{var1}_vs_{var2}"
        # )
        
        # # Summary statistics
        # summary = [
        # "DOC-DOC OUTLIER ANALYSIS SUMMARY",
        # "=" * 40,
        # f"Variables analyzed: {var1} vs {var2}",
        # f"80th percentile thresholds: {var1}={threshold_1:.3f}, {var2}={threshold_2:.3f}",
        # f"Total doc-doc edges: {len(doc_doc_df)}",
        # "",
        # f"Case 1 - High {var2}, Low {var1}: {len(case1_outliers)} rows ({len(case1_outliers)/len(doc_doc_df)*100:.1f}%)",
        # f"Case 2 - High {var1}, Low {var2}: {len(case2_outliers)} rows ({len(case2_outliers)/len(doc_doc_df)*100:.1f}%)",
        # ]
        
        # with open(section_dir / "summary.txt", 'w') as f:
        # f.write('\n'.join(summary))

    def _analyze_doc_table_outliers(self, df: pd.DataFrame, outlier_dir: Path):
        """Analyze outliers for doc-table edges: 3 variable pairs"""
        logger.info("Analyzing doc-table outliers...")
        
        # Filter for doc-table edges
        doc_table_df = df[df['edge_type'] == 'doc-table'].copy()
        
        if len(doc_table_df) == 0:
            logger.warning("No doc-table edges found for outlier analysis")
            return
        
        section_dir = outlier_dir / "section_2_doc_table_outliers"
        section_dir.mkdir(exist_ok=True)
        
        # Variable pairs to analyze
        # variable_pairs = [
        # ('column_similarity', 'topic_summary_similarity'),
        # ('column_similarity', 'topic_title_similarity'),
        # ('topic_summary_similarity', 'topic_title_similarity')
        # ]
        
        # summary_lines = [
        # "DOC-TABLE OUTLIER ANALYSIS SUMMARY",
        # "=" * 40,
        # f"Total doc-table edges: {len(doc_table_df)}",
        # ""
        # ]
        
            # Calculate 99th percentile thresholds for all 3 variables
        threshold_col = doc_table_df['column_similarity'].quantile(0.99)
        threshold_title = doc_table_df['topic_title_similarity'].quantile(0.99)
        threshold_summary = doc_table_df['topic_summary_similarity'].quantile(0.99)

        # Case 1: All three variables high
        case1_mask = (doc_table_df['column_similarity'] > threshold_col) & \
                        (doc_table_df['topic_title_similarity'] > threshold_title) & \
                        (doc_table_df['topic_summary_similarity'] > threshold_summary)
        case1_outliers = doc_table_df[case1_mask].copy()

        # Store outliers for graph building
        self._store_outliers_for_graph_building(case1_outliers, 'doc-table', f'all_three_high')

            # Save outliers with content
            # self._save_outliers_with_content(
            # case1_outliers,
            # section_dir / f"all_three_high",
            # f"High Column Similarity, High Topic-Title Similarity, and High Topic-Summary Similarity",
            # f"column_title_summary"
            # )

            # # Add to summary
            # summary_lines.extend([
            # f"All 3 variables above thresholds:",
            # f" Column Similarity > {threshold_col:.3f}",
            # f" Topic-Title Similarity > {threshold_title:.3f}",
            # f" Topic-Summary Similarity > {threshold_summary:.3f}",
            # f" Edges found: {len(case1_outliers)} rows ({len(case1_outliers)/len(doc_table_df)*100:.1f}%)",
            # ""
            # ])
            # Create subdirectory for this pair
            # pair_dir = section_dir / f"pair_{pair_idx}_{var1}_vs_{var2}"
            # pair_dir.mkdir(exist_ok=True)
            
            # # Save outliers with content
            # self._save_outliers_with_content(
            # case1_outliers,
            # pair_dir / f"case1_high_{var2}_low_{var1}",
            # f"High {var2.replace('_', ' ').title()}, Low {var1.replace('_', ' ').title()}",
            # f"{var1}_vs_{var2}"
            # )
            
            # self._save_outliers_with_content(
            # case2_outliers,
            # pair_dir / f"case2_high_{var1}_low_{var2}",
            # f"High {var1.replace('_', ' ').title()}, Low {var2.replace('_', ' ').title()}",
            # f"{var1}_vs_{var2}"
            # )
            
            # # Add to summary
            # summary_lines.extend([
            # f"PAIR {pair_idx}: {var1} vs {var2}",
            # f" Thresholds: {var1}={threshold_1:.3f}, {var2}={threshold_2:.3f}",
            # f" Case 1 - High {var2}, Low {var1}: {len(case1_outliers)} rows ({len(case1_outliers)/len(doc_table_df)*100:.1f}%)",
            # f" Case 2 - High {var1}, Low {var2}: {len(case2_outliers)} rows ({len(case2_outliers)/len(doc_table_df)*100:.1f}%)",
            # ""
            # ])
        
        # Store edges with entity matches for doc-table (regardless of similarity thresholds)
        entity_match_mask = doc_table_df['entity_count'] > 0
        entity_match_outliers = doc_table_df[entity_match_mask].copy()
        self._store_outliers_for_graph_building(entity_match_outliers, 'doc-table', 'entity_match')
        
        # Save outliers with content - File 1: AND operation outliers
        self._save_outliers_with_content(
            case1_outliers,
            section_dir / f"high_column_and_title_and_summary",
            f"High Column Similarity AND High Topic-Title Similarity AND High Topic-Summary Similarity",
            f"column_title_summary_and"
        )
        
        # Save outliers with content - File 2: Entity count > 0
        self._save_outliers_with_content(
            entity_match_outliers,
            section_dir / f"entity_count_greater_than_0", 
            f"Entity Count Greater Than 0",
            f"entity_match"
        )
        
        # with open(section_dir / "summary.txt", 'w') as f:
        # f.write('\n'.join(summary_lines))

    def _analyze_table_table_outliers(self, df: pd.DataFrame, outlier_dir: Path):
        """Analyze outliers for table-table edges: 3 variable pairs"""
        logger.info("Analyzing table-table outliers...")
        
        # Filter for table-table edges
        table_table_df = df[df['edge_type'] == 'table-table'].copy()
        
        if len(table_table_df) == 0:
            logger.warning("No table-table edges found for outlier analysis")
            return
        
        section_dir = outlier_dir / "section_3_table_table_outliers"
        section_dir.mkdir(exist_ok=True)
        
        # Variable pairs to analyze
        # variable_pairs = [
        # ('column_similarity', 'title_similarity'),
        # ('column_similarity', 'description_similarity'),
        # ('title_similarity', 'description_similarity')
        # ]
        
        # summary_lines = [
        # "TABLE-TABLE OUTLIER ANALYSIS SUMMARY",
        # "=" * 40,
        # f"Total table-table edges: {len(table_table_df)}",
        # ""
        # ]

        threshold_col = table_table_df['column_similarity'].quantile(0.99)
        threshold_title = table_table_df['title_similarity'].quantile(0.99)
        threshold_summary = table_table_df['description_similarity'].quantile(0.99)

        logger.info(f"Table-Table thresholds: column={threshold_col:.3f}, title={threshold_title:.3f}, description={threshold_summary:.3f}")

        case1_mask = (table_table_df['column_similarity'] > threshold_col) & \
                        (table_table_df['title_similarity'] > threshold_title) & \
                        (table_table_df['description_similarity'] > threshold_summary)
        case1_outliers = table_table_df[case1_mask].copy()

        # Store outliers for graph building
        self._store_outliers_for_graph_building(case1_outliers, 'table-table', f'all_three_high')

        # for pair_idx, (var1, var2) in enumerate(variable_pairs, 1):
        # logger.info(f"Analyzing table-table pair {pair_idx}: {var1} vs {var2}")
            
        # # Calculate 80th percentile thresholds
        # threshold_1 = table_table_df[var1].quantile(0.95)
        # threshold_2 = table_table_df[var2].quantile(0.95)
            
        # # Case 1: var2 high, var1 low
        # case1_mask = (table_table_df[var2] > threshold_2) & (table_table_df[var1] < threshold_1)
        # case1_outliers = table_table_df[case1_mask].copy()
            
        # # Case 2: var1 high, var2 low
        # case2_mask = (table_table_df[var1] > threshold_1) & (table_table_df[var2] < threshold_2)
        # case2_outliers = table_table_df[case2_mask].copy()
            
        # # Store outliers for graph building
        # self._store_outliers_for_graph_building(case1_outliers, 'table-table', f'high_{var2}_low_{var1}')
        # self._store_outliers_for_graph_building(case2_outliers, 'table-table', f'high_{var1}_low_{var2}')
            
        # # Create subdirectory for this pair
        # pair_dir = section_dir / f"pair_{pair_idx}_{var1}_vs_{var2}"
        # pair_dir.mkdir(exist_ok=True)
            
        # # Save outliers with content
        # self._save_outliers_with_content(
        # case1_outliers,
        # pair_dir / f"case1_high_{var2}_low_{var1}",
        # f"High {var2.replace('_', ' ').title()}, Low {var1.replace('_', ' ').title()}",
        # f"{var1}_vs_{var2}"
        # )
            
        # self._save_outliers_with_content(
        # case2_outliers,
        # pair_dir / f"case2_high_{var1}_low_{var2}",
        # f"High {var1.replace('_', ' ').title()}, Low {var2.replace('_', ' ').title()}",
        # f"{var1}_vs_{var2}"
        # )
            
        # # Add to summary
        # summary_lines.extend([
        # f"PAIR {pair_idx}: {var1} vs {var2}",
        # f" Thresholds: {var1}={threshold_1:.3f}, {var2}={threshold_2:.3f}",
        # f" Case 1 - High {var2}, Low {var1}: {len(case1_outliers)} rows ({len(case1_outliers)/len(table_table_df)*100:.1f}%)",
        # f" Case 2 - High {var1}, Low {var2}: {len(case2_outliers)} rows ({len(case2_outliers)/len(table_table_df)*100:.1f}%)",
        # ""
        # ])
        
        # Store edges with entity matches for table-table (regardless of similarity thresholds)
        entity_match_mask = table_table_df['entity_count'] > 0
        entity_match_outliers = table_table_df[entity_match_mask].copy()
        self._store_outliers_for_graph_building(entity_match_outliers, 'table-table', 'entity_match')
        
        # Save outliers with content - File 1: AND operation outliers
        self._save_outliers_with_content(
            case1_outliers,
            section_dir / f"high_column_and_title_and_description",
            f"High Column Similarity AND High Title Similarity AND High Description Similarity",
            f"column_title_description_and"
        )
        
        # Save outliers with content - File 2: Entity count > 0
        self._save_outliers_with_content(
            entity_match_outliers,
            section_dir / f"entity_count_greater_than_0", 
            f"Entity Count Greater Than 0",
            f"entity_match"
        )
        
        # with open(section_dir / "summary.txt", 'w') as f:
        # f.write('\n'.join(summary_lines))

    def _save_outliers_with_content(self, outliers_df: pd.DataFrame, file_path: Path, description: str, comparison_type: str = ""):
        """Save outlier rows with their similarity metrics and relevant content in JSON format"""
        if len(outliers_df) == 0:
            logger.info(f"No outliers found for {description}")
            return
        
        logger.info(f"Saving {len(outliers_df)} outliers: {description}")
        
        # Prepare data with relevant content based on comparison type
        output_data = []
        
        for _, row in outliers_df.iterrows():
            source_chunk_id = row['source_chunk_id']
            target_chunk_id = row['target_chunk_id']
            edge_type = row['edge_type']
            
            # Get relevant content based on comparison type
            relevant_content = self._get_relevant_content_for_comparison(
                source_chunk_id, target_chunk_id, edge_type, comparison_type
            )
            
            # Create output row with all similarity metrics and relevant content
            output_row = {
                'source_chunk_id': source_chunk_id,
                'target_chunk_id': target_chunk_id,
                'edge_type': edge_type,
                
                # All similarity metrics
                'topic_similarity': float(row['topic_similarity']),
                'content_similarity': float(row['content_similarity']),
                'column_similarity': float(row['column_similarity']),
                'entity_relationship_overlap': float(row['entity_relationship_overlap']),
                'entity_count': int(row['entity_count']),
                'event_count': int(row['event_count']),
                'topic_title_similarity': float(row['topic_title_similarity']),
                'topic_summary_similarity': float(row['topic_summary_similarity']),
                'title_similarity': float(row['title_similarity']),
                'description_similarity': float(row['description_similarity']),
                
                # Relevant content for comparison
                **relevant_content,
                
                # Metadata
                'description': description,
                'comparison_type': comparison_type
            }
            
            output_data.append(output_row)
        
        # Change file extension to .json
        json_file_path = file_path.with_suffix('.json')
        
        # Save as JSON
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved outlier analysis to {json_file_path}")

    def _get_relevant_content_for_comparison(self, source_chunk_id: str, target_chunk_id: str, 
                                           edge_type: str, comparison_type: str) -> Dict[str, Any]:
        """Get relevant content based on what's being compared"""
        try:
            source_chunk = self.chunks[self.chunk_index[source_chunk_id]]
            target_chunk = self.chunks[self.chunk_index[target_chunk_id]]
            
            if edge_type == "doc-doc":
                # For doc-doc: topic_similarity vs content_similarity
                return {
                    'source_topic': source_chunk.topic or "",
                    'target_topic': target_chunk.topic or "",
                    'source_content': source_chunk.content,
                    'target_content': target_chunk.content
                }
            
            elif edge_type == "doc-table":
                # Determine which is doc and which is table
                if source_chunk.chunk_type == "document":
                    doc_chunk, table_chunk = source_chunk, target_chunk
                    doc_prefix, table_prefix = "source", "target"
                else:
                    doc_chunk, table_chunk = target_chunk, source_chunk
                    doc_prefix, table_prefix = "target", "source"
                
                result = {}
                
                if "column" in comparison_type:
                    # Column similarity involves table column descriptions vs document topic/content
                    result[f'{table_prefix}_column_descriptions'] = table_chunk.column_descriptions or {}
                    result[f'{doc_prefix}_topic'] = doc_chunk.topic or ""
                
                if "topic_title" in comparison_type:
                    # Topic-title similarity: document topic vs table title
                    result[f'{doc_prefix}_topic'] = doc_chunk.topic or ""
                    table_title = ""
                    if table_chunk.metadata:
                        table_title = table_chunk.metadata.get('table_title', '')
                    result[f'{table_prefix}_title'] = table_title
                
                if "topic_summary" in comparison_type:
                    # Topic-summary similarity: document topic vs table summary
                    result[f'{doc_prefix}_topic'] = doc_chunk.topic or ""
                    table_summary = ""
                    if table_chunk.metadata:
                        table_summary = table_chunk.metadata.get('table_summary', '')
                    result[f'{table_prefix}_summary'] = table_summary
                
                return result
            
            elif edge_type == "table-table":
                source_meta = source_chunk.metadata or {}
                target_meta = target_chunk.metadata or {}
                
                result = {}
                
                if "column" in comparison_type:
                    # Column similarity: table column descriptions
                    result['source_column_descriptions'] = source_chunk.column_descriptions or {}
                    result['target_column_descriptions'] = target_chunk.column_descriptions or {}
                
                if "title" in comparison_type:
                    # Title similarity: table titles
                    result['source_title'] = source_meta.get('table_title', '')
                    result['target_title'] = target_meta.get('table_title', '')
                
                if "description" in comparison_type:
                    # Description similarity: table descriptions 
                    result['source_description'] = source_meta.get('table_description', '')
                    result['target_description'] = target_meta.get('table_description', '')
                
                return result
            
            else:
                return {'error': f'Unknown edge type: {edge_type}'}
                
        except Exception as e:
            logger.warning(f"Error getting relevant content: {e}")
            return {'error': str(e)}

    def _get_chunk_content(self, chunk_id: str) -> str:
        """Get content for a chunk by its ID"""
        try:
            chunk_idx = self.chunk_index.get(chunk_id)
            if chunk_idx is not None:
                chunk = self.chunks[chunk_idx]
                return chunk.content
            else:
                return f"Content not found for chunk_id: {chunk_id}"
        except Exception as e:
            logger.warning(f"Error getting content for chunk {chunk_id}: {e}")
            return f"Error retrieving content: {str(e)}"

    def build_knowledge_graph(self):
        """Build knowledge graph from similarity analysis results"""
        logger.info("Building knowledge graph from similarity analysis...")
        
        if not self.similarity_data:
            logger.error("No similarity data available. Run analyze_chunk_similarities first.")
            return
        
        # Create output directory
        output_dir = self.cache_dir / "knowledge_graph"
        output_dir.mkdir(exist_ok=True)
        
        # Convert to DataFrame for analysis
        df = self._create_similarity_dataframe()
        
        # Step 1: Extract edges based on outlier analysis and entity matching
        logger.info("Step 1: Extracting edges based on 95th percentile thresholds and entity matching...")
        edges_to_create = self._extract_edges_for_graph(df)
        
        # Step 2: Community Detection using Leiden Algorithm
        logger.info("Step 2: Detecting communities using Leiden algorithm...")
        communities = self._detect_communities_with_leiden(edges_to_create)
        
        # Step 3: Create knowledge graph
        logger.info("Step 3: Creating knowledge graph structure...")
        knowledge_graph = self._create_knowledge_graph(edges_to_create)
        
        # Step 3: Save knowledge graph
        logger.info("Step 4: Saving knowledge graph...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        knowledge_graph_file = output_dir / f"knowledge_graph_{timestamp}.json"
        knowledge_graph.export_to_json(str(knowledge_graph_file))
        
        logger.info(f"Knowledge graph saved to: {knowledge_graph_file}")
        logger.info(f"Graph statistics: {len(knowledge_graph.nodes)} nodes, {len(knowledge_graph.edges)} edges")
        
        return knowledge_graph

    def analyze_connected_components(self) -> Dict[str, Any]:
        """
        Analyze connected components from outlier edges using BFS
        
        Returns:
            Dictionary with connected components and statistics
        """
        logger.info("Analyzing connected components from outlier edges...")
        
        if not self.outlier_edges:
            logger.warning("No outlier edges found. Run similarity analysis first.")
            return {}
        
        # Import here to avoid circular imports
        from collections import deque, defaultdict
        
        # Build adjacency list from outlier edges
        adjacency_list = defaultdict(set)
        for edge in self.outlier_edges:
            source_id = edge['source_chunk_id']
            target_id = edge['target_chunk_id']
            
            # Add bidirectional connections
            adjacency_list[source_id].add(target_id)
            adjacency_list[target_id].add(source_id)
        
        logger.info(f"Built adjacency list with {len(adjacency_list)} nodes from {len(self.outlier_edges)} edges")
        
        # Find connected components using BFS
        visited = set()
        connected_components = []
        
        def bfs_component(start_node: str) -> List[str]:
            """BFS to find connected component starting from start_node"""
            component = []
            queue = deque([start_node])
            
            while queue:
                current_node = queue.popleft()
                
                if current_node in visited:
                    continue
                
                visited.add(current_node)
                component.append(current_node)
                
                # Add neighbors to queue
                for neighbor in adjacency_list[current_node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            
            return component
        
        # Find all connected components
        for node in adjacency_list:
            if node not in visited:
                component = bfs_component(node)
                connected_components.append(sorted(component))
                logger.info(f"Found connected component with {len(component)} nodes")
        
        # Calculate statistics
        component_sizes = [len(comp) for comp in connected_components]
        
        stats = {
            'total_components': len(connected_components),
            'total_nodes': sum(component_sizes),
            'largest_component_size': max(component_sizes) if component_sizes else 0,
            'smallest_component_size': min(component_sizes) if component_sizes else 0,
            'average_component_size': sum(component_sizes) / len(component_sizes) if component_sizes else 0,
            'component_size_distribution': {}
        }
        
        # Component size distribution
        size_counts = defaultdict(int)
        for size in component_sizes:
            size_counts[size] += 1
        
        stats['component_size_distribution'] = dict(size_counts)
        
        # Save results
        output_data = {
            'connected_components': connected_components,
            'statistics': stats
        }
        
        output_file = self.cache_dir / "connected_components.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(connected_components)} connected components to {output_file}")
        
        # Log statistics
        logger.info("=== Connected Components Statistics ===")
        logger.info(f"Total components: {stats['total_components']}")
        logger.info(f"Total nodes: {stats['total_nodes']}")
        logger.info(f"Largest component: {stats['largest_component_size']} nodes")
        logger.info(f"Smallest component: {stats['smallest_component_size']} nodes")
        logger.info(f"Average component size: {stats['average_component_size']:.2f} nodes")
        
        logger.info("Component size distribution:")
        for size, count in sorted(stats['component_size_distribution'].items()):
            logger.info(f" Size {size}: {count} components")
        
        return output_data

    def _detect_communities_with_leiden(self, edges_to_create: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Detect communities using Leiden algorithm with composite edge weights
        
        Args:
            edges_to_create: List of edge dictionaries with similarity metrics
            
        Returns:
            Dictionary containing community information and statistics
        """
        if not CDLIB_AVAILABLE:
            logger.warning("cdlib not available. Skipping community detection.")
            return None
            
        if not edges_to_create:
            logger.warning("No edges provided for community detection.")
            return None
            
        logger.info(f"Starting community detection with {len(edges_to_create)} edges...")
        
        # Build NetworkX graph with weighted edges
        G = nx.Graph()
        
        # Add nodes and edges with computed weights
        edge_weights = []
        for edge in edges_to_create:
            source_id = edge['source_chunk_id']
            target_id = edge['target_chunk_id']
            edge_type = edge['edge_type']
            entity_count = edge.get('entity_count', 0)
            
            # Calculate composite weight based on edge type
            weight = self._calculate_composite_weight(edge, edge_type, entity_count)
            edge_weights.append(weight)
            
            # Add edge to graph
            G.add_edge(source_id, target_id, weight=weight)
            
        logger.info(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        logger.info(f"Edge weight range: {min(edge_weights):.3f} - {max(edge_weights):.3f}")
        
        # Apply Leiden algorithm with weights
        try:
            logger.info("Running Leiden algorithm...")
            # cdlib expects the edge weight attribute name via the 'weights' parameter
            communities = algorithms.leiden(G, weights='weight')
            
            # Extract community information
            community_data = {
                'communities': communities.communities,
                'num_communities': len(communities.communities),
                'modularity': communities.modularity(),
                'community_sizes': [len(comm) for comm in communities.communities]
            }
            
            # Calculate statistics
            community_sizes = community_data['community_sizes']
            community_data['statistics'] = {
                'largest_community': max(community_sizes) if community_sizes else 0,
                'smallest_community': min(community_sizes) if community_sizes else 0,
                'average_community_size': sum(community_sizes) / len(community_sizes) if community_sizes else 0,
                'total_nodes_in_communities': sum(community_sizes)
            }
            
            # Save community results
            output_dir = self.cache_dir / "knowledge_graph"
            output_dir.mkdir(exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            communities_file = output_dir / f"leiden_communities_{timestamp}.json"
            
            # Convert communities to JSON-serializable format
            communities_serializable = {
                'communities': [list(comm) for comm in communities.communities],
                'num_communities': community_data['num_communities'],
                'modularity': community_data['modularity'],
                'community_sizes': community_data['community_sizes'],
                'statistics': community_data['statistics'],
                'edge_weights_summary': {
                    'min_weight': float(min(edge_weights)),
                    'max_weight': float(max(edge_weights)),
                    'avg_weight': float(sum(edge_weights) / len(edge_weights)),
                    'total_edges': len(edge_weights)
                }
            }
            
            with open(communities_file, 'w', encoding='utf-8') as f:
                json.dump(communities_serializable, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Leiden community detection completed:")
            logger.info(f" - Communities found: {community_data['num_communities']}")
            logger.info(f" - Modularity: {community_data['modularity']:.3f}")
            logger.info(f" - Largest community: {community_data['statistics']['largest_community']} nodes")
            logger.info(f" - Average community size: {community_data['statistics']['average_community_size']:.1f} nodes")
            logger.info(f" - Results saved to: {communities_file}")
            
            return community_data
            
        except Exception as e:
            logger.error(f"Error during Leiden community detection: {e}")
            return None

    def _calculate_composite_weight(self, edge: Dict[str, Any], edge_type: str, entity_count: int) -> float:
        """
        Calculate composite weight for an edge based on its type and available metrics
        
        Args:
            edge: Edge dictionary with similarity metrics
            edge_type: Type of edge ('doc-doc', 'doc-table', 'table-table')
            entity_count: Number of matched entities
            
        Returns:
            Composite weight as float
        """
        weight = 0.0
        
        if edge_type == 'doc-doc':
            # For doc-doc: topic_similarity + content_similarity + entity_count
            topic_sim = edge.get('topic_similarity', 0.0)
            content_sim = edge.get('content_similarity', 0.0)
            weight = topic_sim + content_sim + entity_count
            
        elif edge_type == 'doc-table':
            # For doc-table: column_similarity + topic_title_similarity + topic_summary_similarity + entity_count
            column_sim = edge.get('column_similarity', 0.0)
            topic_title_sim = edge.get('topic_title_similarity', 0.0)
            topic_summary_sim = edge.get('topic_summary_similarity', 0.0)
            weight = column_sim + topic_title_sim + topic_summary_sim + entity_count
            
        elif edge_type == 'table-table':
            # For table-table: column_similarity + title_similarity + description_similarity + entity_count
            column_sim = edge.get('column_similarity', 0.0)
            title_sim = edge.get('title_similarity', 0.0)
            description_sim = edge.get('description_similarity', 0.0)
            weight = column_sim + title_sim + description_sim + entity_count
            
        else:
            logger.warning(f"Unknown edge type: {edge_type}, using default weight")
            weight = edge.get('semantic_similarity', 0.0) + entity_count
            
        return max(weight, 0.01) # Ensure positive weight (minimum 0.01)

    def _extract_edges_for_graph(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract edges from pre-computed outliers (optimized - no reprocessing)"""
        logger.info("Using pre-computed outlier edges for graph building...")
        
        if not hasattr(self, 'outlier_edges') or not self.outlier_edges:
            logger.warning("No outlier edges found! Make sure to run generate_analysis_reports first.")
            return []
        
        # Remove duplicates by edge_id
        unique_edges = {}
        for edge in self.outlier_edges:
            edge_id = edge['edge_id']
            if edge_id not in unique_edges:
                unique_edges[edge_id] = edge
        
        edges_to_create = list(unique_edges.values())

        # Filter edges to only include those whose endpoints exist in the loaded chunk index
        if hasattr(self, 'chunk_index') and isinstance(self.chunk_index, dict):
            valid_ids = set(self.chunk_index.keys())
            before_count = len(edges_to_create)
            edges_to_create = [
                e for e in edges_to_create
                if e.get('source_chunk_id') in valid_ids and e.get('target_chunk_id') in valid_ids
            ]
            logger.info(
                f"Filtered edges to loaded chunks: {before_count} -> {len(edges_to_create)}"
            )
        
        # Save unique edges to JSON file
        output_dir = self.cache_dir / "knowledge_graph"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        edges_file = output_dir / f"unique_edges_{timestamp}.json"
        
        with open(edges_file, 'w', encoding='utf-8') as f:
            json.dump(edges_to_create, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(edges_to_create)} unique edges to {edges_file}")
        
        logger.info(f"Found {len(edges_to_create)} unique edges from outlier analysis")
        return edges_to_create


    def _create_knowledge_graph(self, edges_to_create: List[Dict[str, Any]]) -> KnowledgeGraph:
        """Create knowledge graph from chunks and edges"""
        logger.info("Creating knowledge graph structure...")
        
        # Initialize KnowledgeGraph
        knowledge_graph = KnowledgeGraph()
        
        # Step 1: Add all chunks as nodes
        logger.info(f"Adding {len(self.chunks)} chunks as nodes...")
        all_nodes = []
        
        for chunk_data in self.chunks:
            # Convert ChunkData to proper chunk object
            chunk_obj = self._convert_chunk_data_to_chunk_object(chunk_data)
            
            node = GraphNode(
                node_id=chunk_data.chunk_id,
                chunk=chunk_obj,
                connections=[] # Will be populated by edges
            )
            all_nodes.append(node)
        
        # Batch add nodes
        total_nodes_added = knowledge_graph.add_nodes_batch_optimized(all_nodes)
        logger.info(f"Added {total_nodes_added} nodes to knowledge graph")
        
        # Step 2: Create edge metadata objects
        logger.info(f"Creating {len(edges_to_create)} edge metadata objects...")
        edge_metadata_objects = []
        
        for edge_data in edges_to_create:
            edge_metadata = self._create_edge_metadata_from_data(edge_data)
            if edge_metadata:
                edge_metadata_objects.append(edge_metadata)
        
        # Step 3: Batch add edges
        logger.info(f"Adding {len(edge_metadata_objects)} edges to knowledge graph...")
        total_edges_added = knowledge_graph.add_edges_batch_optimized(edge_metadata_objects)
        logger.info(f"Added {total_edges_added} edges to knowledge graph")
        
        return knowledge_graph

    def _convert_chunk_data_to_chunk_object(self, chunk_data: ChunkData) -> Union[DocumentChunk, TableChunk]:
        """Convert ChunkData to proper DocumentChunk or TableChunk object"""
        import uuid
        
        # Create SourceInfo
        source_info = SourceInfo(
            source_id=chunk_data.chunk_id,
            source_name=chunk_data.metadata.get('title', chunk_data.chunk_id) if chunk_data.metadata else chunk_data.chunk_id,
            source_type=ChunkType.DOCUMENT if chunk_data.chunk_type == "document" else ChunkType.TABLE,
            file_path=chunk_data.metadata.get('file_path', '') if chunk_data.metadata else '',
            structural_link=chunk_data.metadata.get('structural_link', []) if chunk_data.metadata else [],
            original_source=chunk_data.metadata.get('original_source', '') if chunk_data.metadata else '',
            additional_information=chunk_data.metadata.get('additional_information', '') if chunk_data.metadata else '',
            content=chunk_data.content if chunk_data.chunk_type == "document" else None
        )
        
        # Extract keywords from entities
        keywords = list(chunk_data.entities.keys()) if chunk_data.entities else []
        
        if chunk_data.chunk_type == "document":
            return DocumentChunk(
                chunk_id=chunk_data.chunk_id,
                content=chunk_data.content,
                source_info=source_info,
                sentences=[chunk_data.content], # Simplified
                keywords=keywords,
                summary=chunk_data.content[:200] + "..." if len(chunk_data.content) > 200 else chunk_data.content,
                embedding=chunk_data.content_embedding,
                merged_sentence_count=1
            )
        else: # table
            return TableChunk(
                chunk_id=chunk_data.chunk_id,
                content=chunk_data.content,
                source_info=source_info,
                column_headers=list(chunk_data.column_descriptions.keys()) if chunk_data.column_descriptions else [],
                column_descriptions=list(chunk_data.column_descriptions.values()) if chunk_data.column_descriptions else [],
                rows_with_headers=[], # Simplified
                keywords=keywords,
                summary=chunk_data.content[:200] + "..." if len(chunk_data.content) > 200 else chunk_data.content,
                embedding=chunk_data.content_embedding,
                merged_row_count=1
            )

    def _create_edge_metadata_from_data(self, edge_data: Dict[str, Any]) -> BaseEdgeMetadata:
        """Create appropriate edge metadata object from edge data"""
        import uuid
        
        edge_type = edge_data['edge_type']
        
        # Get entity data from precomputed similarities (more efficient)
        source_id = edge_data['source_chunk_id']
        target_id = edge_data['target_chunk_id']
        
        # Get shared entities from precomputed fuzzy matches
        shared_keywords = []
        if hasattr(self, 'precomputed_similarities') and 'entity_fuzzy_matches' in self.precomputed_similarities:
            entity_data = self.precomputed_similarities['entity_fuzzy_matches'].get((source_id, target_id))
            if entity_data:
                # entity_data is (entity_count, relationship_overlap)
                # For now, use entity names as shared keywords (could be enhanced)
                if source_id not in self.chunk_index or target_id not in self.chunk_index:
                    logger.warning(
                        f"Skipping edge with unknown chunks (precomputed): {source_id} -> {target_id}"
                    )
                    return None
                source_chunk = self.chunks[self.chunk_index[source_id]]
                target_chunk = self.chunks[self.chunk_index[target_id]]
                shared_keywords = list(set(source_chunk.entities.keys()) & set(target_chunk.entities.keys()))
        
        # Fallback to direct entity intersection if no precomputed data
        if not shared_keywords:
            if source_id not in self.chunk_index or target_id not in self.chunk_index:
                logger.warning(
                    f"Skipping edge with unknown chunks: {source_id} -> {target_id}"
                )
                return None
            source_chunk = self.chunks[self.chunk_index[source_id]]
            target_chunk = self.chunks[self.chunk_index[target_id]]
            shared_keywords = list(set(source_chunk.entities.keys()) & set(target_chunk.entities.keys()))
        
        if edge_type == 'doc-doc':
            return DocumentToDocumentEdgeMetadata(
                edge_id=edge_data['edge_id'],
                source_chunk_id=edge_data['source_chunk_id'],
                target_chunk_id=edge_data['target_chunk_id'],
                semantic_similarity=edge_data['semantic_similarity'],
                shared_keywords=shared_keywords,
                topic_overlap=edge_data['topic_overlap']
            )
        elif edge_type == 'doc-table':
            return TableToDocumentEdgeMetadata(
                edge_id=edge_data['edge_id'],
                source_chunk_id=edge_data['source_chunk_id'],
                target_chunk_id=edge_data['target_chunk_id'],
                semantic_similarity=edge_data['semantic_similarity'],
                shared_keywords=shared_keywords,
                row_references=[], # Simplified
                column_references=[], # Simplified
                topic_title_similarity=edge_data.get('topic_title_similarity', 0.0),
                topic_summary_similarity=edge_data.get('topic_summary_similarity', 0.0)
            )
        elif edge_type == 'table-table':
            return TableToTableEdgeMetadata(
                edge_id=edge_data['edge_id'],
                source_chunk_id=edge_data['source_chunk_id'],
                target_chunk_id=edge_data['target_chunk_id'],
                semantic_similarity=edge_data['semantic_similarity'],
                shared_keywords=shared_keywords,
                column_similarity=edge_data['column_similarity'],
                row_overlap=0.0, # Simplified
                schema_context={}, # Simplified
                title_similarity=edge_data.get('title_similarity', 0.0),
                description_similarity=edge_data.get('description_similarity', 0.0)
            )
        
        return None

    def _store_outliers_for_graph_building(self, outliers_df: pd.DataFrame, edge_type: str, reason: str):
        """Store outliers for efficient graph building"""
        edge_ids_seen = set()
        
        for _, row in outliers_df.iterrows():
            source_id = row['source_chunk_id']
            target_id = row['target_chunk_id']
            edge_id = f"{source_id}_{target_id}"
            
            # Skip duplicates
            if edge_id in edge_ids_seen:
                continue
            edge_ids_seen.add(edge_id)
            
            # Get entity count from precomputed data
            entity_count = 0
            if hasattr(self, 'precomputed_similarities') and 'entity_fuzzy_matches' in self.precomputed_similarities:
                entity_data = self.precomputed_similarities['entity_fuzzy_matches'].get((source_id, target_id))
                if entity_data:
                    entity_count = entity_data[0] # First element is entity count
            
            # Store edge data
            edge_data = {
                'edge_id': edge_id,
                'source_chunk_id': source_id,
                'target_chunk_id': target_id,
                'edge_type': edge_type,
                'reason': reason,
                'semantic_similarity': row['content_similarity'],
                'entity_count': entity_count,
                'topic_overlap': row['topic_similarity']
            }
            
            # Add type-specific fields for community detection
            if edge_type == 'doc-doc':
                edge_data.update({
                    'topic_similarity': row['topic_similarity'],
                    'content_similarity': row['content_similarity'],
                    'entity_relationship_overlap': row.get('entity_relationship_overlap', 0.0),
                    'event_count': row.get('event_count', 0)
                })
            elif edge_type == 'doc-table':
                edge_data.update({
                    'column_similarity': row.get('column_similarity', 0.0),
                    'topic_title_similarity': row['topic_title_similarity'],
                    'topic_summary_similarity': row['topic_summary_similarity']
                })
            elif edge_type == 'table-table':
                edge_data.update({
                    'column_similarity': row['column_similarity'],
                    'title_similarity': row['title_similarity'],
                    'description_similarity': row['description_similarity']
                })
            
            self.outlier_edges.append(edge_data)
def main():
    """Main execution function"""
    
    # Activate virtual environment message
    logger.info("Make sure to activate the virtual environment: source cogcomp_env/bin/activate")
    
    # Initialize pipeline
    pipeline = GraphAnalysisPipeline()
    
    try:
        # Step 1: Load all chunks
        logger.info("=== Step 1: Loading Chunks ===")
        pipeline.load_chunks()
        
        # Step 2: Build HNSW index
        logger.info("=== Step 2: Building HNSW Index ===")
        pipeline.build_hnsw_index()
        
        # Step 3: Analyze similarities
        logger.info("=== Step 3: Analyzing Similarities ===")
        pipeline.analyze_chunk_similarities(k_neighbors=200)
        
        # Step 4: Generate reports
        logger.info("=== Step 4: Generating Analysis Reports ===")
        pipeline.generate_analysis_reports()
        
        # Step 4.5: Analyze Connected Components
        logger.info("=== Step 4.5: Analyzing Connected Components ===")
        pipeline.analyze_connected_components()
        
        # Step 5: Build Knowledge Graph
        logger.info("=== Step 5: Building Knowledge Graph ===")
        pipeline.build_knowledge_graph()
        
        logger.info("Graph analysis pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()