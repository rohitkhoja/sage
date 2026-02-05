#!/usr/bin/env python3
"""
STARK Dataset Graph Analysis Pipeline: Specialized analysis for Amazon product dataset
Separates products (tables) and reviews (documents) for optimal graph construction.
"""

import os
import sys
import json
import time
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
import torch
from loguru import logger
import re
import networkx as nx

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.processors.embedding_service import EmbeddingService
from src.core.models import ProcessingConfig, ChunkType
from src.core.graph import KnowledgeGraph
from src.core.models import (
    GraphNode, EdgeType, BaseEdgeMetadata, DocumentToDocumentEdgeMetadata,
    TableToTableEdgeMetadata, DocumentChunk, TableChunk, SourceInfo
)

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

@dataclass
class STARKSimilarityMetrics:
    """Container for STARK-specific similarity metrics between two chunks"""
    # Core similarities
    content_similarity: float
    edge_type: str
    source_chunk_id: str
    target_chunk_id: str
    
    # STARK-specific features
    entity_count: int = 0 # Matched entities (brand, color)
    
    # Doc-Doc specific (product info to product info)
    # Using: title, feature, detail, description
    title_similarity: float = 0.0
    feature_similarity: float = 0.0
    detail_similarity: float = 0.0
    description_similarity: float = 0.0
    
    # Table-Table specific (reviews table to reviews table) 
    # Using: aggregated review summaries and texts
    reviews_summary_similarity: float = 0.0
    reviews_text_similarity: float = 0.0

@dataclass 
class STARKChunkData:
    """STARK-specific chunk data with computed embeddings"""
    chunk_id: str
    chunk_type: str # "document" (product info) or "table" (reviews table)
    content: str
    content_embedding: List[float]
    entities: List[str] # Brand and color as strings
    asin: str # Product identifier
    
    # Document-specific (product info excluding reviews)
    title: Optional[str] = None
    brand: Optional[str] = None
    feature: Optional[List[str]] = None
    detail: Optional[str] = None
    description: Optional[List[str]] = None
    
    # Table-specific (all reviews combined as structured data)
    reviews_data: Optional[pd.DataFrame] = None # DataFrame with all reviews
    review_count: Optional[int] = None
    
    # Embeddings for similarity calculations
    title_embedding: Optional[List[float]] = None
    feature_embedding: Optional[List[float]] = None
    detail_embedding: Optional[List[float]] = None
    description_embedding: Optional[List[float]] = None
    reviews_summary_embedding: Optional[List[float]] = None # Combined summaries
    reviews_text_embedding: Optional[List[float]] = None # Combined review texts
    
    metadata: Optional[Dict[str, Any]] = None

class STARKGPUAcceleratedSimilarityCalculator:
    """
    GPU-accelerated similarity calculator specialized for STARK dataset
    Handles only doc-doc and table-table similarities (no doc-table)
    """
    
    def __init__(self, chunks: List[STARKChunkData], embedding_service: EmbeddingService, 
                 batch_size: int = 10000, gpu_id: int = 0):
        self.chunks = chunks
        self.embedding_service = embedding_service
        self.batch_size = batch_size
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        
        # Precomputed GPU tensors
        self.gpu_content_embeddings = None
        self.gpu_title_embeddings = None
        self.gpu_feature_embeddings = None
        self.gpu_detail_embeddings = None
        self.gpu_description_embeddings = None
        self.gpu_reviews_summary_embeddings = None
        self.gpu_reviews_text_embeddings = None
        
        # Chunk mappings for fast lookups
        self.chunk_id_to_idx = {chunk.chunk_id: i for i, chunk in enumerate(chunks)}
        self.chunk_types = [chunk.chunk_type for chunk in chunks]
        
        # Batch processing tracking
        self.batch_results_dir = Path("/shared/khoja/CogComp/output/stark_batch_results")
        self.batch_results_dir.mkdir(exist_ok=True)
        
        logger.info(f"STARK GPU Similarity Calculator initialized on {self.device}")
        logger.info(f"Batch size: {batch_size}, Processing {len(chunks)} chunks")
        
        # Calculate optimal batch size based on GPU memory
        if torch.cuda.is_available():
            self.optimal_batch_size = self._calculate_optimal_batch_size()
            logger.info(f"Optimal batch size based on GPU memory: {self.optimal_batch_size}")
        else:
            self.optimal_batch_size = batch_size
    
    def precompute_embeddings(self):
        """Precompute all embeddings and move to GPU for batch processing"""
        logger.info("Precomputing STARK embeddings for GPU processing...")
        
        # Extract content embeddings
        content_embeddings = []
        title_embeddings = []
        feature_embeddings = []
        detail_embeddings = []
        description_embeddings = []
        reviews_summary_embeddings = []
        reviews_text_embeddings = []
        
        for chunk in self.chunks:
            # Content embeddings (always present)
            content_embeddings.append(chunk.content_embedding or [0.0] * 384) # Using MiniLM model
            
            # Document-specific embeddings (product info)
            if chunk.chunk_type == "document":
                title_embeddings.append(chunk.title_embedding or [0.0] * 384)
                feature_embeddings.append(chunk.feature_embedding or [0.0] * 384)
                detail_embeddings.append(chunk.detail_embedding or [0.0] * 384)
                description_embeddings.append(chunk.description_embedding or [0.0] * 384)
                
                reviews_summary_embeddings.append([0.0] * 384) # Not used for docs
                reviews_text_embeddings.append([0.0] * 384) # Not used for docs
                
            # Table-specific embeddings (reviews tables)
            else: # chunk.chunk_type == "table"
                title_embeddings.append([0.0] * 384) # Not used for tables
                feature_embeddings.append([0.0] * 384) # Not used for tables
                detail_embeddings.append([0.0] * 384) # Not used for tables
                description_embeddings.append([0.0] * 384) # Not used for tables
                
                reviews_summary_embeddings.append(chunk.reviews_summary_embedding or [0.0] * 384)
                reviews_text_embeddings.append(chunk.reviews_text_embedding or [0.0] * 384)
        
        # Move to GPU
        self.gpu_content_embeddings = torch.tensor(content_embeddings, device=self.device, dtype=torch.float32)
        self.gpu_title_embeddings = torch.tensor(title_embeddings, device=self.device, dtype=torch.float32)
        self.gpu_feature_embeddings = torch.tensor(feature_embeddings, device=self.device, dtype=torch.float32)
        self.gpu_detail_embeddings = torch.tensor(detail_embeddings, device=self.device, dtype=torch.float32)
        self.gpu_description_embeddings = torch.tensor(description_embeddings, device=self.device, dtype=torch.float32)
        self.gpu_reviews_summary_embeddings = torch.tensor(reviews_summary_embeddings, device=self.device, dtype=torch.float32)
        self.gpu_reviews_text_embeddings = torch.tensor(reviews_text_embeddings, device=self.device, dtype=torch.float32)
        
        # Normalize for cosine similarity
        self.gpu_content_embeddings = torch.nn.functional.normalize(self.gpu_content_embeddings, p=2, dim=1)
        self.gpu_title_embeddings = torch.nn.functional.normalize(self.gpu_title_embeddings, p=2, dim=1)
        self.gpu_feature_embeddings = torch.nn.functional.normalize(self.gpu_feature_embeddings, p=2, dim=1)
        self.gpu_detail_embeddings = torch.nn.functional.normalize(self.gpu_detail_embeddings, p=2, dim=1)
        self.gpu_description_embeddings = torch.nn.functional.normalize(self.gpu_description_embeddings, p=2, dim=1)
        self.gpu_reviews_summary_embeddings = torch.nn.functional.normalize(self.gpu_reviews_summary_embeddings, p=2, dim=1)
        self.gpu_reviews_text_embeddings = torch.nn.functional.normalize(self.gpu_reviews_text_embeddings, p=2, dim=1)
        
        logger.info(f"Precomputed STARK embeddings on GPU: {self.gpu_content_embeddings.shape}")
        logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")
    
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
            optimal_batch_size = min(50000, max(1000, max_pairs_per_batch))
            
            logger.info(f"GPU Memory: {gpu_memory:.1f}GB total, {available_memory:.1f}GB available")
            logger.info(f"Estimated memory per pair: {memory_per_pair*1000:.2f}MB")
            
            return optimal_batch_size
        except Exception as e:
            logger.warning(f"Could not calculate optimal batch size: {e}")
            return self.batch_size
    
    def calculate_similarities_batch(self, unique_pairs: List[Tuple[str, str]]) -> List[STARKSimilarityMetrics]:
        """Calculate similarities for all pairs using GPU batch processing"""
        # Filter pairs to only include same-type comparisons (no doc-table)
        filtered_pairs = []
        for source_id, target_id in unique_pairs:
            source_idx = self.chunk_id_to_idx.get(source_id)
            target_idx = self.chunk_id_to_idx.get(target_id)
            
            if source_idx is not None and target_idx is not None:
                source_type = self.chunks[source_idx].chunk_type
                target_type = self.chunks[target_idx].chunk_type
                
                # Only include same-type pairs (doc-doc or table-table)
                if source_type == target_type:
                    filtered_pairs.append((source_id, target_id))
        
        logger.info(f"Filtered {len(unique_pairs)} pairs to {len(filtered_pairs)} same-type pairs")
        
        # Use optimal batch size for better GPU utilization
        effective_batch_size = self.optimal_batch_size
        logger.info(f"Processing {len(filtered_pairs)} pairs in batches of {effective_batch_size}")
        
        all_results = []
        batch_count = 0
        
        # Clear old batch results
        self.clear_batch_results()
        
        # Process in batches
        for i in range(0, len(filtered_pairs), effective_batch_size):
            batch_pairs = filtered_pairs[i:i + effective_batch_size]
            batch_count += 1
            
            logger.info(f"Processing batch {batch_count}/{(len(filtered_pairs) + effective_batch_size - 1) // effective_batch_size}")
            
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
        
        logger.info(f"STARK GPU batch processing completed: {len(all_results)} total similarities")
        return all_results
    
    def _calculate_batch_similarities_gpu(self, batch_pairs: List[Tuple[str, str]]) -> List[STARKSimilarityMetrics]:
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
        
        # Convert to CPU for individual processing
        content_sims = content_similarities.cpu().numpy()
        
        # Process each pair for specific similarity calculations
        for i, (source_id, target_id, source_idx, target_idx) in enumerate(pair_info):
            source_chunk = self.chunks[source_idx]
            target_chunk = self.chunks[target_idx]
            
            # Determine edge type
            edge_type = self._get_edge_type(source_chunk.chunk_type, target_chunk.chunk_type)
            
            # Calculate additional features based on edge type
            additional_metrics = self._calculate_additional_features(source_chunk, target_chunk, edge_type, source_idx, target_idx)
            
            # Create similarity metrics with all features
            similarity_metrics = STARKSimilarityMetrics(
                content_similarity=max(0.0, float(content_sims[i])),
                edge_type=edge_type,
                source_chunk_id=source_id,
                target_chunk_id=target_id,
                **additional_metrics # Unpack additional features
            )
            
            batch_results.append(similarity_metrics)
        
        return batch_results
    
    def _get_edge_type(self, type1: str, type2: str) -> str:
        """Determine edge type from chunk types"""
        if type1 == "document" and type2 == "document":
            return "doc-doc"
        elif type1 == "table" and type2 == "table":
            return "table-table"
        else:
            # This should not happen in STARK pipeline (doc-table filtered out)
            return "doc-table"
    
    def _calculate_additional_features(self, chunk1: STARKChunkData, chunk2: STARKChunkData, 
                                     edge_type: str, idx1: int, idx2: int) -> Dict[str, Any]:
        """Calculate all additional features based on edge type"""
        additional_features = {}
        
        if edge_type == "doc-doc":
            # Doc-Doc specific features (reviews to reviews)
            additional_features.update(self._calculate_doc_doc_features(chunk1, chunk2, idx1, idx2))
        elif edge_type == "table-table":
            # Table-Table specific features (products to products)
            additional_features.update(self._calculate_table_table_features(chunk1, chunk2, idx1, idx2))
        
        return additional_features
    
    def _calculate_doc_doc_features(self, doc1: STARKChunkData, doc2: STARKChunkData, 
                                   idx1: int, idx2: int) -> Dict[str, Any]:
        """Calculate features specific to doc-doc edges (product info to product info)"""
        features = {}
        
        # Entity matching (brand and color)
        entity_count = self._calculate_entity_matches(doc1.entities, doc2.entities)
        features['entity_count'] = entity_count
        
        # Product info similarities: title, feature, detail, description
        features['title_similarity'] = self._calculate_field_similarity(
            self.gpu_title_embeddings, idx1, idx2)
        features['feature_similarity'] = self._calculate_field_similarity(
            self.gpu_feature_embeddings, idx1, idx2)
        features['detail_similarity'] = self._calculate_field_similarity(
            self.gpu_detail_embeddings, idx1, idx2)
        features['description_similarity'] = self._calculate_field_similarity(
            self.gpu_description_embeddings, idx1, idx2)
        
        # Initialize table-specific features to 0
        features.update({
            'reviews_summary_similarity': 0.0,
            'reviews_text_similarity': 0.0
        })
        
        return features
    
    def _calculate_table_table_features(self, table1: STARKChunkData, table2: STARKChunkData,
                                       idx1: int, idx2: int) -> Dict[str, Any]:
        """Calculate features specific to table-table edges (reviews table to reviews table)"""
        features = {}
        
        # Entity matching (brand and color)
        entity_count = self._calculate_entity_matches(table1.entities, table2.entities)
        features['entity_count'] = entity_count
        
        # Reviews table similarities: aggregated summaries and texts
        features['reviews_summary_similarity'] = self._calculate_field_similarity(
            self.gpu_reviews_summary_embeddings, idx1, idx2)
        features['reviews_text_similarity'] = self._calculate_field_similarity(
            self.gpu_reviews_text_embeddings, idx1, idx2)
        
        # Initialize product info features to 0
        features.update({
            'title_similarity': 0.0,
            'feature_similarity': 0.0,
            'detail_similarity': 0.0,
            'description_similarity': 0.0
        })
        
        return features
    
    def _calculate_field_similarity(self, embedding_tensor: torch.Tensor, idx1: int, idx2: int) -> float:
        """Calculate similarity between two embeddings using GPU tensors"""
        try:
            emb1 = embedding_tensor[idx1:idx1+1]
            emb2 = embedding_tensor[idx2:idx2+1]
            
            # Check if either embedding is all zeros (indicating empty feature)
            if torch.all(emb1 == 0) or torch.all(emb2 == 0):
                return 0.0
            
            similarity = torch.mm(emb1, emb2.T).item()
            return max(0.0, similarity)
        except Exception as e:
            logger.warning(f"Error calculating field similarity: {e}")
            return 0.0
    
    def _calculate_entity_matches(self, entities1: List[str], entities2: List[str]) -> int:
        """Calculate number of matching entities (brand and color)"""
        if not entities1 or not entities2:
            return 0
        
        # Simple exact string matching for brand and color
        matches = 0
        for entity1 in entities1:
            for entity2 in entities2:
                if entity1.lower().strip() == entity2.lower().strip():
                    matches += 1
                    break # Count each entity1 only once
        
        return matches
    
    def _save_batch_results(self, batch_results: List[STARKSimilarityMetrics], batch_id: int):
        """Save batch results to CSV file"""
        if not batch_results:
            return
        
        # Convert to DataFrame
        data = []
        for result in batch_results:
            data.append({
                'source_chunk_id': result.source_chunk_id,
                'target_chunk_id': result.target_chunk_id,
                'content_similarity': result.content_similarity,
                'edge_type': result.edge_type,
                'entity_count': result.entity_count,
                # Doc-doc features (product info to product info)
                'title_similarity': result.title_similarity,
                'feature_similarity': result.feature_similarity,
                'detail_similarity': result.detail_similarity,
                'description_similarity': result.description_similarity,
                # Table-table features (reviews table to reviews table)
                'reviews_summary_similarity': result.reviews_summary_similarity,
                'reviews_text_similarity': result.reviews_text_similarity
            })
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        batch_file = self.batch_results_dir / f"stark_batch_{batch_id:06d}.csv"
        df.to_csv(batch_file, index=False)
        
        logger.debug(f"Saved STARK batch {batch_id} results to {batch_file}")
    
    def clear_batch_results(self):
        """Clear all batch result files"""
        batch_files = list(self.batch_results_dir.glob("stark_batch_*.csv"))
        for batch_file in batch_files:
            batch_file.unlink()
        logger.info(f"Cleared {len(batch_files)} STARK batch result files")
    
    def load_all_batch_results(self) -> List[STARKSimilarityMetrics]:
        """Load all batch results from CSV files"""
        batch_files = sorted(list(self.batch_results_dir.glob("stark_batch_*.csv")))
        
        if not batch_files:
            logger.warning("No STARK batch result files found")
            return []
        
        all_results = []
        
        for batch_file in batch_files:
            try:
                df = pd.read_csv(batch_file)
                
                for _, row in df.iterrows():
                    similarity_metrics = STARKSimilarityMetrics(
                        content_similarity=float(row['content_similarity']),
                        edge_type=str(row['edge_type']),
                        source_chunk_id=str(row['source_chunk_id']),
                        target_chunk_id=str(row['target_chunk_id']),
                        entity_count=int(row['entity_count']),
                        # Doc-doc features
                        title_similarity=float(row['title_similarity']),
                        feature_similarity=float(row['feature_similarity']),
                        detail_similarity=float(row['detail_similarity']),
                        description_similarity=float(row['description_similarity']),
                        # Table-table features
                        reviews_summary_similarity=float(row['reviews_summary_similarity']),
                        reviews_text_similarity=float(row['reviews_text_similarity'])
                    )
                    all_results.append(similarity_metrics)
                
                logger.debug(f"Loaded {len(df)} similarities from {batch_file}")
                
            except Exception as e:
                logger.warning(f"Error loading batch file {batch_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(all_results)} total similarities from {len(batch_files)} batch files")
        return all_results

class STARKGraphAnalysisPipeline:
    """
    STARK-specific pipeline for analyzing Amazon product and review similarities
    Separates products (tables) and reviews (documents) for graph construction
    """
    
    def __init__(self, 
                 stark_dataset_file: str = "/shared/khoja/CogComp/datasets/STARK/node_info.json",
                 cache_dir: str = "/shared/khoja/CogComp/output/stark_analysis_cache",
                 use_gpu: bool = True,
                 num_threads: int = 64):
        
        self.stark_dataset_file = Path(stark_dataset_file)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.num_threads = num_threads
        
        # Initialize embedding service
        self.config = ProcessingConfig(use_faiss=True, faiss_use_gpu=self.use_gpu)
        self.embedding_service = EmbeddingService(self.config)
        
        # Data containers
        self.chunks: List[STARKChunkData] = []
        self.chunk_index: Dict[str, int] = {} # chunk_id -> index in self.chunks
        self.faiss_index = None
        self.similarity_data: List[STARKSimilarityMetrics] = []
        
        # Thread-safe lock for data updates
        self.data_lock = threading.Lock()
        
        # Store outlier edges for graph building (optimization)
        self.outlier_edges = [] # Will be populated during analysis
        
        logger.info("Initialized STARK graph analysis pipeline")

    def load_stark_chunks(self, max_products: Optional[int] = None) -> None:
        """Load STARK dataset and create 2 chunks per product: 1 document (product info) + 1 table (all reviews)"""
        logger.info("Loading STARK dataset...")
        
        # Load from cache if available
        cache_file = self.cache_dir / "stark_processed_chunks.pkl"
        if cache_file.exists():
            logger.info("Loading STARK chunks from cache...")
            with open(cache_file, 'rb') as f:
                self.chunks = pickle.load(f)
            self._build_chunk_index()
            logger.info(f"Loaded {len(self.chunks)} STARK chunks from cache")
            return
        
        # Load raw STARK data
        with open(self.stark_dataset_file, 'r') as f:
            stark_data = json.load(f)
        
        logger.info(f"Loaded {len(stark_data)} products from STARK dataset")
        
        # Limit products for testing if specified
        if max_products:
            product_keys = list(stark_data.keys())[:max_products]
            stark_data = {k: stark_data[k] for k in product_keys}
            logger.info(f"Limited to {len(stark_data)} products for testing")
        
        # Process each product into exactly 2 chunks: 1 document + 1 table
        all_chunks = []
        for product_id, product_data in stark_data.items():
            try:
                # Create document chunk (product info excluding reviews)
                doc_chunk = self._create_product_document_chunk(product_id, product_data)
                if doc_chunk:
                    all_chunks.append(doc_chunk)
                
                # Create table chunk (all reviews combined)
                table_chunk = self._create_product_reviews_table_chunk(product_id, product_data)
                if table_chunk:
                    all_chunks.append(table_chunk)
                    
            except Exception as e:
                logger.warning(f"Error processing product {product_id}: {e}")
                continue
        
        self.chunks = all_chunks
        self._build_chunk_index()
        
        # Process embeddings for all chunks
        self._process_stark_embeddings()
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        # Count chunk types
        doc_chunks = [c for c in self.chunks if c.chunk_type == "document"]
        table_chunks = [c for c in self.chunks if c.chunk_type == "table"]
        
        logger.info(f"Loaded {len(doc_chunks)} product document chunks")
        logger.info(f"Loaded {len(table_chunks)} review table chunks")
        logger.info(f"Processed and cached {len(self.chunks)} total STARK chunks")

    def _create_product_document_chunk(self, product_id: str, product_data: Dict[str, Any]) -> Optional[STARKChunkData]:
        """Create document chunk containing product information (excluding reviews)"""
        try:
            asin = product_data.get('asin', product_id)
            
            # Extract product information
            title = product_data.get('title', '')
            brand = product_data.get('brand', 'Unknown')
            feature = product_data.get('feature', [])
            description = product_data.get('description', [])
            details = product_data.get('details', '')
            
            # Handle details field (might be NaN or dict)
            if isinstance(details, float) and np.isnan(details):
                details = ''
            elif isinstance(details, dict):
                details = str(details)
            
            # Create content from key fields for embedding (filter empty values)
            content_parts = []
            
            # Add title if not empty
            if title and title.strip():
                content_parts.append(title.strip())
            
            # Add features if not empty
            if feature:
                if isinstance(feature, list):
                    valid_features = [str(f).strip() for f in feature 
                                    if str(f).strip() and str(f).strip().lower() not in ['nan', 'none', 'null']]
                    content_parts.extend(valid_features)
                else:
                    feature_str = str(feature).strip()
                    if feature_str and feature_str.lower() not in ['nan', 'none', 'null']:
                        content_parts.append(feature_str)
            
            # Add description if not empty
            if description:
                if isinstance(description, list):
                    valid_descriptions = [str(d).strip() for d in description 
                                        if str(d).strip() and str(d).strip().lower() not in ['nan', 'none', 'null']]
                    content_parts.extend(valid_descriptions)
                else:
                    desc_str = str(description).strip()
                    if desc_str and desc_str.lower() not in ['nan', 'none', 'null']:
                        content_parts.append(desc_str)
            
            # Add details if not empty
            if details and str(details).strip() and str(details).strip().lower() not in ['nan', 'none', 'null']:
                content_parts.append(str(details).strip())
            
            content = ' '.join(content_parts) if content_parts else ""
            
            # Extract entities (brand and color)
            entities = self._extract_entities(title, feature, description, brand)
            
            chunk_data = STARKChunkData(
                chunk_id=f"product_doc_{asin}_{product_id}",
                chunk_type="document",
                content=content,
                content_embedding=[], # Will be generated
                entities=entities,
                asin=asin,
                title=title,
                brand=brand,
                feature=feature,
                detail=str(details),
                description=description,
                metadata={
                    'global_category': product_data.get('global_category', ''),
                    'category': product_data.get('category', []),
                    'price': product_data.get('price', ''),
                    'brand': brand,
                    'rank': product_data.get('rank', ''),
                    'original_product_id': product_id
                }
            )
            
            return chunk_data
            
        except Exception as e:
            logger.warning(f"Error creating product document chunk for {product_id}: {e}")
            return None

    def _create_product_reviews_table_chunk(self, product_id: str, product_data: Dict[str, Any]) -> Optional[STARKChunkData]:
        """Create table chunk containing ALL reviews for a product as structured data"""
        try:
            asin = product_data.get('asin', product_id)
            reviews = product_data.get('review', [])
            
            # Skip if no reviews
            if not reviews:
                logger.debug(f"No reviews found for product {product_id}")
                return None
            
            # Extract product-level entities
            brand = product_data.get('brand', 'Unknown')
            title = product_data.get('title', '')
            feature = product_data.get('feature', [])
            description = product_data.get('description', [])
            product_entities = self._extract_entities(title, feature, description, brand)
            
            # Create DataFrame with all reviews
            reviews_data = []
            for review_idx, review in enumerate(reviews):
                try:
                    summary = review.get('summary', '')
                    style = review.get('style', '')
                    review_text = review.get('reviewText', '')
                    
                    # Handle NaN values
                    if isinstance(style, float) and np.isnan(style):
                        style = ''
                    
                    reviews_data.append({
                        'review_idx': review_idx,
                        'summary': summary,
                        'style': str(style),
                        'reviewText': review_text,
                        'reviewerID': review.get('reviewerID', ''),
                        'vote': review.get('vote', ''),
                        'overall': review.get('overall', 0),
                        'verified': review.get('verified', False),
                        'reviewTime': review.get('reviewTime', '')
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing review {review_idx} for product {product_id}: {e}")
                    continue
            
            # Create DataFrame
            reviews_df = pd.DataFrame(reviews_data)
            
            # Create content for embeddings from all review summaries and texts (filter empty values)
            valid_summaries = [s.strip() for s in reviews_df['summary'].fillna('').astype(str) 
                             if s.strip() and s.strip().lower() not in ['nan', 'none', 'null']]
            valid_review_texts = [r.strip() for r in reviews_df['reviewText'].fillna('').astype(str) 
                                if r.strip() and r.strip().lower() not in ['nan', 'none', 'null']]
            
            content_parts = valid_summaries + valid_review_texts
            content = ' '.join(content_parts).strip() if content_parts else ""
            
            chunk_data = STARKChunkData(
                chunk_id=f"reviews_table_{asin}_{product_id}",
                chunk_type="table",
                content=content,
                content_embedding=[], # Will be generated
                entities=product_entities,
                asin=asin,
                reviews_data=reviews_df,
                review_count=len(reviews_df),
                metadata={
                    'product_asin': asin,
                    'original_product_id': product_id,
                    'review_columns': list(reviews_df.columns),
                    'total_reviews': len(reviews_df)
                }
            )
            
            return chunk_data
            
        except Exception as e:
            logger.warning(f"Error creating reviews table chunk for product {product_id}: {e}")
            return None

    def _extract_entities(self, title: str, features: List[str], descriptions: List[str], brand: str) -> List[str]:
        """Extract brand and color entities from product/review text"""
        entities = []
        
        # Add brand as entity
        if brand and brand.lower() != 'unknown':
            entities.append(brand)
        
        # Common color words to look for
        colors = [
            'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'white',
            'gray', 'grey', 'navy', 'turquoise', 'teal', 'maroon', 'silver', 'gold', 'beige', 'tan',
            'khaki', 'olive', 'burgundy', 'coral', 'salmon', 'magenta', 'cyan', 'lime', 'indigo', 'violet'
        ]
        
        # Search for colors in all text fields
        all_text = ' '.join([str(title), str(features), str(descriptions)]).lower()
        
        for color in colors:
            if color in all_text:
                entities.append(color)
        
        return entities

    def _build_chunk_index(self):
        """Build chunk ID to index mapping"""
        self.chunk_index = {chunk.chunk_id: i for i, chunk in enumerate(self.chunks)}

    def _process_stark_embeddings(self):
        """Process all required embeddings for STARK chunks"""
        logger.info("Processing STARK embeddings for all chunks...")
        
        # Collect all texts that need embeddings
        content_texts = []
        title_texts = []
        feature_texts = []
        detail_texts = []
        description_texts = []
        reviews_summary_texts = []
        reviews_text_texts = []
        
        content_indices = []
        title_indices = []
        feature_indices = []
        detail_indices = []
        description_indices = []
        reviews_summary_indices = []
        reviews_text_indices = []
        
        for i, chunk in enumerate(self.chunks):
            # Content embeddings (always needed)
            if chunk.content:
                content_texts.append(chunk.content)
                content_indices.append(i)
            
            if chunk.chunk_type == "document": # Product info documents
                # Title embeddings - check if not empty
                if chunk.title and chunk.title.strip():
                    title_texts.append(chunk.title.strip())
                    title_indices.append(i)
                
                # Feature embeddings - check if not empty
                if chunk.feature:
                    feature_text = ' '.join(chunk.feature) if isinstance(chunk.feature, list) else str(chunk.feature)
                    feature_text = feature_text.strip()
                    if feature_text and feature_text.lower() not in ['nan', 'none', 'null']:
                        feature_texts.append(feature_text)
                        feature_indices.append(i)
                
                # Detail embeddings - check if not empty
                if chunk.detail and chunk.detail.strip() and chunk.detail.strip().lower() not in ['nan', 'none', 'null']:
                    detail_texts.append(chunk.detail.strip())
                    detail_indices.append(i)
                
                # Description embeddings - check if not empty
                if chunk.description:
                    desc_text = ' '.join(chunk.description) if isinstance(chunk.description, list) else str(chunk.description)
                    desc_text = desc_text.strip()
                    if desc_text and desc_text.lower() not in ['nan', 'none', 'null']:
                        description_texts.append(desc_text)
                        description_indices.append(i)
            
            elif chunk.chunk_type == "table": # Reviews tables
                # Reviews summary embeddings (combined summaries) - check if not empty
                if chunk.reviews_data is not None and 'summary' in chunk.reviews_data.columns:
                    # Filter out empty/NaN summaries
                    valid_summaries = chunk.reviews_data['summary'].fillna('').astype(str)
                    valid_summaries = [s.strip() for s in valid_summaries if s.strip() and s.strip().lower() not in ['nan', 'none', 'null']]
                    
                    if valid_summaries:
                        summaries = ' '.join(valid_summaries)
                        reviews_summary_texts.append(summaries)
                        reviews_summary_indices.append(i)
                
                # Reviews text embeddings (combined review texts) - check if not empty
                if chunk.reviews_data is not None and 'reviewText' in chunk.reviews_data.columns:
                    # Filter out empty/NaN review texts
                    valid_review_texts = chunk.reviews_data['reviewText'].fillna('').astype(str)
                    valid_review_texts = [r.strip() for r in valid_review_texts if r.strip() and r.strip().lower() not in ['nan', 'none', 'null']]
                    
                    if valid_review_texts:
                        review_texts = ' '.join(valid_review_texts)
                        reviews_text_texts.append(review_texts)
                        reviews_text_indices.append(i)

        # Generate embeddings in batches
        batch_size = 64
        
        # Log empty feature statistics
        total_chunks = len(self.chunks)
        logger.info(f"Empty feature statistics out of {total_chunks} chunks:")
        logger.info(f" - Title texts: {len(title_texts)} (skipped {total_chunks - len(title_texts)} empty)")
        logger.info(f" - Feature texts: {len(feature_texts)} (skipped {total_chunks - len(feature_texts)} empty)")
        logger.info(f" - Detail texts: {len(detail_texts)} (skipped {total_chunks - len(detail_texts)} empty)")
        logger.info(f" - Description texts: {len(description_texts)} (skipped {total_chunks - len(description_texts)} empty)")
        logger.info(f" - Reviews summary texts: {len(reviews_summary_texts)} (skipped {total_chunks - len(reviews_summary_texts)} empty)")
        logger.info(f" - Reviews text texts: {len(reviews_text_texts)} (skipped {total_chunks - len(reviews_text_texts)} empty)")
        
        # Content embeddings
        if content_texts:
            logger.info(f"Generating {len(content_texts)} content embeddings...")
            content_embeddings = self.embedding_service.generate_embeddings(content_texts)
            for idx, embedding in zip(content_indices, content_embeddings):
                self.chunks[idx].content_embedding = embedding
        
        # Title embeddings (product documents)
        if title_texts:
            logger.info(f"Generating {len(title_texts)} title embeddings...")
            title_embeddings = self.embedding_service.generate_embeddings(title_texts)
            for chunk_idx, embedding in zip(title_indices, title_embeddings):
                self.chunks[chunk_idx].title_embedding = embedding
        
        # Feature embeddings (product documents)
        if feature_texts:
            logger.info(f"Generating {len(feature_texts)} feature embeddings...")
            feature_embeddings = self.embedding_service.generate_embeddings(feature_texts)
            for chunk_idx, embedding in zip(feature_indices, feature_embeddings):
                self.chunks[chunk_idx].feature_embedding = embedding
        
        # Detail embeddings (product documents)
        if detail_texts:
            logger.info(f"Generating {len(detail_texts)} detail embeddings...")
            detail_embeddings = self.embedding_service.generate_embeddings(detail_texts)
            for chunk_idx, embedding in zip(detail_indices, detail_embeddings):
                self.chunks[chunk_idx].detail_embedding = embedding
        
        # Description embeddings (product documents)
        if description_texts:
            logger.info(f"Generating {len(description_texts)} description embeddings...")
            description_embeddings = self.embedding_service.generate_embeddings(description_texts)
            for chunk_idx, embedding in zip(description_indices, description_embeddings):
                self.chunks[chunk_idx].description_embedding = embedding
        
        # Reviews summary embeddings (reviews tables)
        if reviews_summary_texts:
            logger.info(f"Generating {len(reviews_summary_texts)} reviews summary embeddings...")
            reviews_summary_embeddings = self.embedding_service.generate_embeddings(reviews_summary_texts)
            for chunk_idx, embedding in zip(reviews_summary_indices, reviews_summary_embeddings):
                self.chunks[chunk_idx].reviews_summary_embedding = embedding
        
        # Reviews text embeddings (reviews tables)
        if reviews_text_texts:
            logger.info(f"Generating {len(reviews_text_texts)} reviews text embeddings...")
            reviews_text_embeddings = self.embedding_service.generate_embeddings(reviews_text_texts)
            for chunk_idx, embedding in zip(reviews_text_indices, reviews_text_embeddings):
                self.chunks[chunk_idx].reviews_text_embedding = embedding

    def build_hnsw_index(self):
        """Build HNSW FAISS index from all chunk content embeddings"""
        logger.info("Building HNSW FAISS index for STARK dataset...")
        
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
        dimension = embeddings_matrix.shape[1]
        
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
        
        logger.info("STARK HNSW index built successfully")

    def analyze_stark_similarities(self, k_neighbors: int = 200):
        """Analyze similarities for STARK chunks with their k nearest neighbors"""
        logger.info(f"Analyzing STARK similarities for {len(self.chunks)} chunks with {k_neighbors} neighbors each...")
        
        # Check for batch results first (from GPU processing)
        batch_results_dir = Path("/shared/khoja/CogComp/output/stark_batch_results")
        if batch_results_dir.exists() and list(batch_results_dir.glob("stark_batch_*.csv")):
            logger.info("Loading STARK similarity analysis from batch result files...")
            gpu_calculator = STARKGPUAcceleratedSimilarityCalculator(
                chunks=self.chunks,
                embedding_service=self.embedding_service,
                batch_size=10000,
                gpu_id=0
            )
            self.similarity_data = gpu_calculator.load_all_batch_results()
            logger.info(f"Loaded {len(self.similarity_data)} STARK similarity records from batch files")
            return
        
        # Load cached results if available
        cache_file = self.cache_dir / f"stark_similarity_analysis_k{k_neighbors}.pkl"
        if cache_file.exists():
            logger.info("Loading STARK similarity analysis from cache...")
            with open(cache_file, 'rb') as f:
                self.similarity_data = pickle.load(f)
            logger.info(f"Loaded {len(self.similarity_data)} STARK similarity records from cache")
            return
        
        # Phase 1: Find all k-neighbors for all chunks (parallel)
        logger.info("Phase 1: Finding k-neighbors for all STARK chunks...")
        all_neighbor_relationships = self._find_all_neighbors_parallel(k_neighbors)
        
        # Phase 2: Generate unique pairs from neighbor relationships
        logger.info("Phase 2: Generating unique pairs...")
        unique_pairs = self._generate_unique_pairs(all_neighbor_relationships)
        logger.info(f"Generated {len(unique_pairs)} unique pairs from neighbor relationships")
        
        # Phase 3: Calculate similarities for unique pairs (GPU accelerated)
        logger.info("Phase 3: Calculating STARK similarities for unique pairs...")
        self._calculate_stark_similarities_for_pairs(unique_pairs)
        
        # Save to cache for future use
        with open(cache_file, 'wb') as f:
            pickle.dump(self.similarity_data, f)
        
        logger.info(f"STARK similarity analysis completed: {len(self.similarity_data)} total records")

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
        
        logger.info(f"Found neighbors for {len(all_neighbor_relationships)} STARK chunks")
        return all_neighbor_relationships

    def _find_neighbors_for_batch(self, batch_id: int, chunk_batch: List[STARKChunkData], k_neighbors: int) -> Dict[str, List[str]]:
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

    def _find_k_neighbor_ids(self, chunk: STARKChunkData, k: int) -> List[str]:
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

    def _calculate_stark_similarities_for_pairs(self, unique_pairs: List[Tuple[str, str]]):
        """Phase 3: Calculate similarities for unique pairs using GPU acceleration or CPU fallback"""
        
        # Check GPU availability
        if not torch.cuda.is_available():
            logger.warning("GPU not available! Falling back to CPU processing (will be slower).")
            # Set use_gpu to False for CPU processing
            original_use_gpu = self.use_gpu
            self.use_gpu = False
        else:
            logger.info("Using GPU-accelerated STARK similarity calculation")
        
        # Initialize similarity calculator (works on both GPU and CPU)
        similarity_calculator = STARKGPUAcceleratedSimilarityCalculator(
            chunks=self.chunks,
            embedding_service=self.embedding_service,
            batch_size=10000, # Use 10k pairs per batch as requested
            gpu_id=0 # Use first GPU if available, otherwise CPU
        )
        
        # Precompute embeddings
        similarity_calculator.precompute_embeddings()
        
        # Calculate similarities using batch processing
        self.similarity_data = similarity_calculator.calculate_similarities_batch(unique_pairs)
        
        device_type = "GPU" if torch.cuda.is_available() and self.use_gpu else "CPU"
        logger.info(f"STARK {device_type} processing completed: {len(self.similarity_data)} similarities calculated")

    def generate_stark_analysis_reports(self):
        """Generate comprehensive analysis reports for STARK dataset by edge type"""
        logger.info("Generating STARK analysis reports and visualizations...")
        
        if not self.similarity_data:
            logger.error("No STARK similarity data available. Run analyze_stark_similarities first.")
            return
        
        # Create output directory
        output_dir = self.cache_dir / "stark_analysis_reports"
        output_dir.mkdir(exist_ok=True)
        
        # Convert to DataFrame for analysis
        df = self._create_stark_similarity_dataframe()
        
        # Generate analysis by sections
        self._generate_stark_doc_doc_analysis(df, output_dir)
        self._generate_stark_table_table_analysis(df, output_dir)
        
        # Generate overall summary
        self._generate_stark_overall_summary(df, output_dir)
        
        # Generate outlier analysis
        self._generate_stark_outlier_analysis(df, output_dir)
        
        logger.info(f"STARK analysis reports saved to {output_dir}")

    def _create_stark_similarity_dataframe(self) -> pd.DataFrame:
        """Convert STARK similarity data to pandas DataFrame"""
        data = []
        
        for sim in self.similarity_data:
            data.append({
                'content_similarity': sim.content_similarity,
                'edge_type': sim.edge_type,
                'source_chunk_id': sim.source_chunk_id,
                'target_chunk_id': sim.target_chunk_id,
                'entity_count': sim.entity_count,
                # Doc-doc features (product info to product info)
                'title_similarity': sim.title_similarity,
                'feature_similarity': sim.feature_similarity,
                'detail_similarity': sim.detail_similarity,
                'description_similarity': sim.description_similarity,
                # Table-table features (reviews table to reviews table)
                'reviews_summary_similarity': sim.reviews_summary_similarity,
                'reviews_text_similarity': sim.reviews_text_similarity
            })
        
        return pd.DataFrame(data)

    def _generate_stark_doc_doc_analysis(self, df: pd.DataFrame, output_dir: Path):
        """Generate STARK Doc-Doc edge type analysis (review to review)"""
        logger.info("Generating STARK Section 1: Doc-Doc Analysis (Review to Review)...")
        
        # Filter for doc-doc edges only
        doc_doc_df = df[df['edge_type'] == 'doc-doc'].copy()
        
        if len(doc_doc_df) == 0:
            logger.warning("No STARK doc-doc edges found for analysis")
            return
        
        section_dir = output_dir / "section_1_stark_doc_doc"
        section_dir.mkdir(exist_ok=True)
        
        # Metrics relevant for doc-doc: product info similarities
        relevant_cols = [
            'content_similarity', 'entity_count',
            'title_similarity', 'feature_similarity', 'detail_similarity', 'description_similarity'
        ]
        
        # Correlation analysis
        correlation_matrix = doc_doc_df[relevant_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.3f')
        plt.title('STARK Section 1: Doc-Doc (Review-Review) Correlation Matrix')
        plt.tight_layout()
        plt.savefig(section_dir / 'stark_doc_doc_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Distribution analysis
        n_cols = len(relevant_cols)
        cols_per_row = 3
        n_rows = (n_cols + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(18, 6*n_rows))
        
        if n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(relevant_cols):
            sns.histplot(data=doc_doc_df, x=col, kde=True, ax=axes[i], bins=30)
            axes[i].set_title(f'STARK Doc-Doc: {col.replace("_", " ").title()}')
            axes[i].grid(True, alpha=0.3)
        
        for i in range(len(relevant_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(section_dir / 'stark_doc_doc_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Statistics
        stats = doc_doc_df[relevant_cols].describe()
        stats.to_csv(section_dir / 'stark_doc_doc_statistics.csv')
        
        # Insights
        insights = [
            "STARK SECTION 1: DOC-DOC EDGE ANALYSIS (PRODUCT INFO TO PRODUCT INFO)",
            f"Total product info edges analyzed: {len(doc_doc_df)}",
            "",
            "FEATURES ANALYZED:",
            " - Content Similarity: Full product content embeddings similarity",
            " - Entity Count: Matched entities (brand, color)",
            " - Title Similarity: Product title vs product title",
            " - Feature Similarity: Product features vs product features", 
            " - Detail Similarity: Product details vs product details",
            " - Description Similarity: Product description vs product description",
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
        
        with open(section_dir / 'stark_doc_doc_insights.txt', 'w') as f:
            f.write('\n'.join(insights))

    def _generate_stark_table_table_analysis(self, df: pd.DataFrame, output_dir: Path):
        """Generate STARK Table-Table edge type analysis (product to product)"""
        logger.info("Generating STARK Section 2: Table-Table Analysis (Product to Product)...")
        
        # Filter for table-table edges only
        table_table_df = df[df['edge_type'] == 'table-table'].copy()
        
        if len(table_table_df) == 0:
            logger.warning("No STARK table-table edges found for analysis")
            return
        
        section_dir = output_dir / "section_2_stark_table_table"
        section_dir.mkdir(exist_ok=True)
        
        # Metrics relevant for table-table: reviews table similarities
        relevant_cols = [
            'content_similarity', 'entity_count',
            'reviews_summary_similarity', 'reviews_text_similarity'
        ]
        
        # Correlation analysis
        correlation_matrix = table_table_df[relevant_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.3f')
        plt.title('STARK Section 2: Table-Table (Product-Product) Correlation Matrix')
        plt.tight_layout()
        plt.savefig(section_dir / 'stark_table_table_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Distribution analysis
        n_cols = len(relevant_cols)
        cols_per_row = 3
        n_rows = (n_cols + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(18, 6*n_rows))
        
        if n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(relevant_cols):
            sns.histplot(data=table_table_df, x=col, kde=True, ax=axes[i], bins=30)
            axes[i].set_title(f'STARK Table-Table: {col.replace("_", " ").title()}')
            axes[i].grid(True, alpha=0.3)
        
        for i in range(len(relevant_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(section_dir / 'stark_table_table_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Statistics
        stats = table_table_df[relevant_cols].describe()
        stats.to_csv(section_dir / 'stark_table_table_statistics.csv')
        
        # Insights
        insights = [
            "STARK SECTION 2: TABLE-TABLE EDGE ANALYSIS (REVIEWS TABLE TO REVIEWS TABLE)",
            f"Total reviews table edges analyzed: {len(table_table_df)}",
            "",
            "FEATURES ANALYZED:",
            " - Content Similarity: Combined reviews content embeddings similarity",
            " - Entity Count: Matched entities (brand, color)",
            " - Reviews Summary Similarity: Aggregated review summaries vs aggregated review summaries",
            " - Reviews Text Similarity: Aggregated review texts vs aggregated review texts",
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
        
        with open(section_dir / 'stark_table_table_insights.txt', 'w') as f:
            f.write('\n'.join(insights))

    def _generate_stark_overall_summary(self, df: pd.DataFrame, output_dir: Path):
        """Generate overall summary for STARK dataset analysis"""
        logger.info("Generating STARK overall summary...")
        
        # Edge type distribution
        edge_counts = df['edge_type'].value_counts()
        
        plt.figure(figsize=(10, 6))
        edge_counts.plot(kind='bar')
        plt.title('STARK Distribution of Edge Types')
        plt.xlabel('Edge Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'stark_overall_edge_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Overall statistics - all numeric features
        numeric_cols = [
            'content_similarity', 'entity_count',
            'title_similarity', 'feature_similarity', 'detail_similarity', 'description_similarity',
            'reviews_summary_similarity', 'reviews_text_similarity'
        ]
        overall_stats = df[numeric_cols].describe()
        overall_stats.to_csv(output_dir / 'stark_overall_statistics.csv')
        
        # Summary insights
        insights = [
            "STARK OVERALL GRAPH ANALYSIS SUMMARY",
            "=" * 50,
            "",
            "DATASET: Amazon STARK Product Reviews",
            "",
            "EDGE TYPE DISTRIBUTION:",
        ]
        
        for edge_type, count in edge_counts.items():
            percentage = (count / len(df)) * 100
            insights.append(f" {edge_type}: {count:,} edges ({percentage:.1f}%)")
        
        insights.extend([
            "",
            "METHODOLOGY:",
            " CHUNK SEPARATION:",
            " - Product Info → Documents (using title, feature, detail, description)",
            " - All Reviews → Tables (aggregated summaries and texts per product)",
            "",
            " ENTITY EXTRACTION:",
            " - Brand and Color as string entities",
            " - Simple exact matching for entity similarity",
            "",
            " SIMILARITY CALCULATIONS:",
            " - Only doc-doc (product info to product info) and table-table (reviews table to reviews table)",
            " - No doc-table (product info to reviews table) similarities calculated",
            "",
            " FEATURES BY EDGE TYPE:",
            " Doc-Doc (Product Info): title, feature, detail, description similarities + entity_count",
            " Table-Table (Reviews Tables): aggregated summaries, aggregated texts similarities + entity_count",
            "",
            "RECOMMENDATIONS FOR GRAPH CONSTRUCTION:",
            " - Review correlation matrices in each section to identify independent metrics",
            " - Use weighted averages only for uncorrelated metrics (|r| < 0.7)",
            " - Consider edge-type specific weighting based on metric distributions",
            " - Entity matching provides additional signal for both edge types",
        ])
        
        with open(output_dir / 'stark_overall_summary.txt', 'w') as f:
            f.write('\n'.join(insights))

    def _generate_stark_outlier_analysis(self, df: pd.DataFrame, output_dir: Path):
        """Generate outlier analysis for STARK dataset - finding high similarity edges"""
        logger.info("Generating STARK outlier analysis...")
        
        # Create outlier analysis directory
        outlier_dir = output_dir / "stark_outlier_analysis"
        outlier_dir.mkdir(exist_ok=True)
        
        # Initialize outlier edges storage
        self.outlier_edges = []
        
        # Section 1: Doc-Doc Analysis (Review-Review)
        self._analyze_stark_doc_doc_outliers(df, outlier_dir)
        
        # Section 2: Table-Table Analysis (Product-Product)
        self._analyze_stark_table_table_outliers(df, outlier_dir)
        
        logger.info(f"STARK outlier analysis completed and saved to {outlier_dir}")
        logger.info(f"Stored {len(self.outlier_edges)} outlier edges for graph building")

    def _analyze_stark_doc_doc_outliers(self, df: pd.DataFrame, outlier_dir: Path):
        """Analyze outliers for STARK doc-doc edges (review-review)"""
        logger.info("Analyzing STARK doc-doc outliers (review-review)...")
        
        # Filter for doc-doc edges
        doc_doc_df = df[df['edge_type'] == 'doc-doc'].copy()
        
        if len(doc_doc_df) == 0:
            logger.warning("No STARK doc-doc edges found for outlier analysis")
            return
        
        section_dir = outlier_dir / "section_1_stark_doc_doc_outliers"
        section_dir.mkdir(exist_ok=True)
        
        # Calculate 95th percentile thresholds for product info features
        threshold_content = doc_doc_df['content_similarity'].quantile(0.95)
        threshold_title = doc_doc_df['title_similarity'].quantile(0.95)
        threshold_feature = doc_doc_df['feature_similarity'].quantile(0.95)
        threshold_detail = doc_doc_df['detail_similarity'].quantile(0.95)
        threshold_description = doc_doc_df['description_similarity'].quantile(0.95)

        logger.info(f"STARK Doc-Doc thresholds: content={threshold_content:.3f}, title={threshold_title:.3f}, feature={threshold_feature:.3f}, detail={threshold_detail:.3f}, description={threshold_description:.3f}")

        # Case 1: High content AND high product info similarities
        case1_mask = (doc_doc_df['content_similarity'] > threshold_content) & \
                     (doc_doc_df['title_similarity'] > threshold_title) & \
                     (doc_doc_df['feature_similarity'] > threshold_feature) & \
                     (doc_doc_df['detail_similarity'] > threshold_detail) & \
                     (doc_doc_df['description_similarity'] > threshold_description)
        case1_outliers = doc_doc_df[case1_mask].copy()

        # Store outliers for graph building
        self._store_stark_outliers_for_graph_building(case1_outliers, 'doc-doc', 'all_product_features_high')

        # Store edges with entity matches (regardless of similarity thresholds)
        entity_match_mask = doc_doc_df['entity_count'] > 0
        entity_match_outliers = doc_doc_df[entity_match_mask].copy()
        self._store_stark_outliers_for_graph_building(entity_match_outliers, 'doc-doc', 'entity_match')

        # Save outliers with content
        self._save_stark_outliers_with_content(
            case1_outliers,
            section_dir / "high_all_product_features",
            "High Content AND Title AND Feature AND Detail AND Description Similarities",
            "all_product_features"
        )

        self._save_stark_outliers_with_content(
            entity_match_outliers,
            section_dir / "entity_count_greater_than_0", 
            "Entity Count Greater Than 0",
            "entity_match"
        )

    def _analyze_stark_table_table_outliers(self, df: pd.DataFrame, outlier_dir: Path):
        """Analyze outliers for STARK table-table edges (product-product)"""
        logger.info("Analyzing STARK table-table outliers (product-product)...")
        
        # Filter for table-table edges
        table_table_df = df[df['edge_type'] == 'table-table'].copy()
        
        if len(table_table_df) == 0:
            logger.warning("No STARK table-table edges found for outlier analysis")
            return
        
        section_dir = outlier_dir / "section_2_stark_table_table_outliers"
        section_dir.mkdir(exist_ok=True)
        
        # Calculate 95th percentile thresholds for reviews table features
        threshold_content = table_table_df['content_similarity'].quantile(0.95)
        threshold_reviews_summary = table_table_df['reviews_summary_similarity'].quantile(0.95)
        threshold_reviews_text = table_table_df['reviews_text_similarity'].quantile(0.95)

        logger.info(f"STARK Table-Table thresholds: content={threshold_content:.3f}, reviews_summary={threshold_reviews_summary:.3f}, reviews_text={threshold_reviews_text:.3f}")

        # Case 1: High content AND high reviews table similarities
        case1_mask = (table_table_df['content_similarity'] > threshold_content) & \
                     (table_table_df['reviews_summary_similarity'] > threshold_reviews_summary) & \
                     (table_table_df['reviews_text_similarity'] > threshold_reviews_text)
        case1_outliers = table_table_df[case1_mask].copy()

        # Store outliers for graph building
        self._store_stark_outliers_for_graph_building(case1_outliers, 'table-table', 'all_reviews_features_high')

        # Store edges with entity matches (regardless of similarity thresholds)
        entity_match_mask = table_table_df['entity_count'] > 0
        entity_match_outliers = table_table_df[entity_match_mask].copy()
        self._store_stark_outliers_for_graph_building(entity_match_outliers, 'table-table', 'entity_match')

        # Save outliers with content
        self._save_stark_outliers_with_content(
            case1_outliers,
            section_dir / "high_all_reviews_features",
            "High Content AND Reviews Summary AND Reviews Text Similarities",
            "all_reviews_features"
        )

        self._save_stark_outliers_with_content(
            entity_match_outliers,
            section_dir / "entity_count_greater_than_0", 
            "Entity Count Greater Than 0",
            "entity_match"
        )

    def _save_stark_outliers_with_content(self, outliers_df: pd.DataFrame, file_path: Path, 
                                         description: str, comparison_type: str = ""):
        """Save STARK outlier rows with their similarity metrics and relevant content in JSON format"""
        if len(outliers_df) == 0:
            logger.info(f"No STARK outliers found for {description}")
            return
        
        logger.info(f"Saving {len(outliers_df)} STARK outliers: {description}")
        
        # Prepare data with relevant content based on comparison type
        output_data = []
        
        for _, row in outliers_df.iterrows():
            source_chunk_id = row['source_chunk_id']
            target_chunk_id = row['target_chunk_id']
            edge_type = row['edge_type']
            
            # Get relevant content based on comparison type
            relevant_content = self._get_stark_relevant_content_for_comparison(
                source_chunk_id, target_chunk_id, edge_type, comparison_type
            )
            
            # Create output row with all similarity metrics and relevant content
            output_row = {
                'source_chunk_id': source_chunk_id,
                'target_chunk_id': target_chunk_id,
                'edge_type': edge_type,
                
                # All similarity metrics
                'content_similarity': float(row['content_similarity']),
                'entity_count': int(row['entity_count']),
                # Doc-doc metrics (product info to product info)
                'title_similarity': float(row['title_similarity']),
                'feature_similarity': float(row['feature_similarity']),
                'detail_similarity': float(row['detail_similarity']),
                'description_similarity': float(row['description_similarity']),
                # Table-table metrics (reviews table to reviews table)
                'reviews_summary_similarity': float(row['reviews_summary_similarity']),
                'reviews_text_similarity': float(row['reviews_text_similarity']),
                
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
        
        logger.info(f"Saved STARK outlier analysis to {json_file_path}")

    def _get_stark_relevant_content_for_comparison(self, source_chunk_id: str, target_chunk_id: str, 
                                                  edge_type: str, comparison_type: str) -> Dict[str, Any]:
        """Get relevant content based on what's being compared in STARK dataset"""
        try:
            source_chunk = self.chunks[self.chunk_index[source_chunk_id]]
            target_chunk = self.chunks[self.chunk_index[target_chunk_id]]
            
            if edge_type == "doc-doc":
                # For doc-doc: product info similarities
                return {
                    'source_title': source_chunk.title or "",
                    'target_title': target_chunk.title or "",
                    'source_brand': source_chunk.brand or "",
                    'target_brand': target_chunk.brand or "",
                    'source_feature': source_chunk.feature or [],
                    'target_feature': target_chunk.feature or [],
                    'source_detail': source_chunk.detail or "",
                    'target_detail': target_chunk.detail or "",
                    'source_description': source_chunk.description or [],
                    'target_description': target_chunk.description or [],
                    'source_entities': source_chunk.entities,
                    'target_entities': target_chunk.entities,
                    'source_asin': source_chunk.asin,
                    'target_asin': target_chunk.asin
                }
            
            elif edge_type == "table-table":
                # For table-table: reviews table similarities
                return {
                    'source_review_count': source_chunk.review_count or 0,
                    'target_review_count': target_chunk.review_count or 0,
                    'source_reviews_sample': source_chunk.reviews_data.head(3).to_dict('records') if source_chunk.reviews_data is not None else [],
                    'target_reviews_sample': target_chunk.reviews_data.head(3).to_dict('records') if target_chunk.reviews_data is not None else [],
                    'source_entities': source_chunk.entities,
                    'target_entities': target_chunk.entities,
                    'source_asin': source_chunk.asin,
                    'target_asin': target_chunk.asin
                }
            
            else:
                return {'error': f'Unknown edge type: {edge_type}'}
                
        except Exception as e:
            logger.warning(f"Error getting STARK relevant content: {e}")
            return {'error': str(e)}

    def _store_stark_outliers_for_graph_building(self, outliers_df: pd.DataFrame, edge_type: str, reason: str):
        """Store STARK outliers for efficient graph building"""
        edge_ids_seen = set()
        
        for _, row in outliers_df.iterrows():
            source_id = row['source_chunk_id']
            target_id = row['target_chunk_id']
            edge_id = f"{source_id}_{target_id}"
            
            # Skip duplicates
            if edge_id in edge_ids_seen:
                continue
            edge_ids_seen.add(edge_id)
            
            # Store edge data
            edge_data = {
                'edge_id': edge_id,
                'source_chunk_id': source_id,
                'target_chunk_id': target_id,
                'edge_type': edge_type,
                'reason': reason,
                'semantic_similarity': row['content_similarity'],
                'entity_count': row['entity_count']
            }
            
            # Add type-specific fields
            if edge_type == 'doc-doc':
                edge_data.update({
                    'title_similarity': row['title_similarity'],
                    'feature_similarity': row['feature_similarity'],
                    'detail_similarity': row['detail_similarity'],
                    'description_similarity': row['description_similarity']
                })
            elif edge_type == 'table-table':
                edge_data.update({
                    'reviews_summary_similarity': row['reviews_summary_similarity'],
                    'reviews_text_similarity': row['reviews_text_similarity']
                })
            
            self.outlier_edges.append(edge_data)

    def extract_and_save_unique_edges(self):
        """Extract unique edges from outlier analysis and save to JSON (FINAL STEP)"""
        logger.info("Extracting and saving unique STARK edges...")
        
        if not hasattr(self, 'outlier_edges') or not self.outlier_edges:
            logger.warning("No outlier edges found! Make sure to run generate_stark_analysis_reports first.")
            return None
        
        # Remove duplicates by edge_id
        unique_edges = {}
        for edge in self.outlier_edges:
            edge_id = edge['edge_id']
            if edge_id not in unique_edges:
                unique_edges[edge_id] = edge
        
        edges_to_save = list(unique_edges.values())
        
        # Filter edges to only include those whose endpoints exist in the loaded chunk index
        if hasattr(self, 'chunk_index') and isinstance(self.chunk_index, dict):
            valid_ids = set(self.chunk_index.keys())
            before_count = len(edges_to_save)
            edges_to_save = [
                e for e in edges_to_save
                if e.get('source_chunk_id') in valid_ids and e.get('target_chunk_id') in valid_ids
            ]
            logger.info(
                f"Filtered STARK edges to loaded chunks: {before_count} -> {len(edges_to_save)}"
            )
        
        # Save unique edges to JSON file
        output_dir = self.cache_dir / "stark_knowledge_graph"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        edges_file = output_dir / f"stark_unique_edges_{timestamp}.json"
        
        with open(edges_file, 'w', encoding='utf-8') as f:
            json.dump(edges_to_save, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(edges_to_save)} unique STARK edges to {edges_file}")
        logger.info("STARK pipeline completed successfully - ready for graph building!")
        
        return edges_file

def main():
    """Main execution function for STARK dataset"""
    
    # Activate virtual environment message
    logger.info("Make sure to activate the virtual environment: source cogcomp_env/bin/activate")
    
    # Initialize STARK pipeline
    pipeline = STARKGraphAnalysisPipeline()
    
    try:
        # Step 1: Load STARK chunks (products and reviews)
        logger.info("=== Step 1: Loading STARK Chunks ===")
        pipeline.load_stark_chunks(max_products=1000) # Limit for testing
        
        # Step 2: Build HNSW index
        logger.info("=== Step 2: Building HNSW Index ===")
        pipeline.build_hnsw_index()
        
        # Step 3: Analyze similarities
        logger.info("=== Step 3: Analyzing STARK Similarities ===")
        pipeline.analyze_stark_similarities(k_neighbors=200)
        
        # Step 4: Generate reports
        logger.info("=== Step 4: Generating STARK Analysis Reports ===")
        pipeline.generate_stark_analysis_reports()
        
        # Step 5: Extract and save unique edges (FINAL STEP)
        logger.info("=== Step 5: Extracting and Saving Unique Edges ===")
        edges_file = pipeline.extract_and_save_unique_edges()
        
        logger.info("STARK graph analysis pipeline completed successfully!")
        logger.info(f"Final output: {edges_file}")
        
    except Exception as e:
        logger.error(f"STARK pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
