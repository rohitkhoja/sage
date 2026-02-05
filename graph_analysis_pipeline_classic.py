#!/usr/bin/env python3
"""
Graph Analysis Pipeline: Comprehensive analysis of document and table chunks
for optimal graph construction with detailed similarity metrics and visualizations.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
import pickle
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import torch
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.processors.embedding_service import EmbeddingService
from src.core.models import ProcessingConfig, ChunkType

@dataclass
class SimilarityMetrics:
    """Container for all similarity metrics between two chunks"""
    entity_similarity: float
    topic_similarity: float
    content_similarity: float
    column_similarity: float  # For table-doc or table-table edges
    edge_type: str  # "doc-doc", "doc-table", "table-table"
    source_chunk_id: str
    target_chunk_id: str

@dataclass 
class ChunkData:
    """Enhanced chunk data with computed embeddings"""
    chunk_id: str
    chunk_type: str  # "document" or "table"
    content: str
    content_embedding: List[float]
    entities: Dict[str, Any]
    entity_embeddings: Dict[str, List[float]]  # entity_name -> embedding
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
        self.gpu_entity_embeddings = {}  # chunk_id -> {entity_name -> tensor}
        self.gpu_column_embeddings = {}  # chunk_id -> {column_name -> tensor}
        
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
                content_embeddings.append([0.0] * 768)  # Default embedding size
            
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
        
        # Precompute entity embeddings
        self._precompute_entity_embeddings()
        
        # Precompute column embeddings
        self._precompute_column_embeddings()
        
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
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available GPU memory"""
        try:
            # Get GPU memory info
            gpu_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**3  # GB
            gpu_memory_free = torch.cuda.memory_reserved(self.device) / 1024**3  # GB
            available_memory = gpu_memory - gpu_memory_free
            
            # Estimate memory per pair (conservative estimate)
            # Each pair needs: 2 embeddings (768 dim) + intermediate results
            memory_per_pair = (768 * 2 * 4) / 1024**3  # 4 bytes per float, convert to GB
            
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
            
            # Calculate entity similarity
            entity_sim = self._calculate_entity_similarity_gpu(source_chunk, target_chunk)
            
            # Calculate column similarity
            column_sim = self._calculate_column_similarity_gpu(source_chunk, target_chunk)
            
            # Determine edge type
            edge_type = self._get_edge_type(source_chunk.chunk_type, target_chunk.chunk_type)
            
            # Create similarity metrics
            similarity_metrics = SimilarityMetrics(
                entity_similarity=entity_sim,
                topic_similarity=max(0.0, float(topic_sims[i])),
                content_similarity=max(0.0, float(content_sims[i])),
                column_similarity=column_sim,
                edge_type=edge_type,
                source_chunk_id=source_id,
                target_chunk_id=target_id
            )
            
            batch_results.append(similarity_metrics)
        
        return batch_results
    
    def _calculate_entity_similarity_gpu(self, chunk1: ChunkData, chunk2: ChunkData) -> float:
        """Calculate entity similarity using GPU tensors"""
        entities1 = self.gpu_entity_embeddings.get(chunk1.chunk_id, {})
        entities2 = self.gpu_entity_embeddings.get(chunk2.chunk_id, {})
        
        if not entities1 or not entities2:
            return 0.0
        
        similarities = []
        threshold = 0.0
        
        for entity1, tensor1 in entities1.items():
            best_similarity = 0.0
            for entity2, tensor2 in entities2.items():
                try:
                    # GPU tensor similarity
                    sim = torch.mm(tensor1, tensor2.t()).item()
                    if sim > threshold:
                        best_similarity = max(best_similarity, sim)
                except Exception:
                    continue
            
            if best_similarity > 0:
                similarities.append(best_similarity)
        
        if not similarities:
            return 0.0
        
        top_similarities = sorted(similarities, reverse=True)[:5]
        return np.mean(top_similarities)
    
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
        return np.mean(top_similarities)
    
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
        return np.mean(top_similarities)
    
    def _get_edge_type(self, type1: str, type2: str) -> str:
        """Determine edge type from chunk types"""
        if type1 == "document" and type2 == "document":
            return "doc-doc"
        elif type1 == "table" and type2 == "table":
            return "table-table"
        else:
            return "doc-table"
    
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
                'entity_similarity': result.entity_similarity,
                'topic_similarity': result.topic_similarity,
                'content_similarity': result.content_similarity,
                'column_similarity': result.column_similarity,
                'edge_type': result.edge_type
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
                        entity_similarity=row['entity_similarity'],
                        topic_similarity=row['topic_similarity'],
                        content_similarity=row['content_similarity'],
                        column_similarity=row['column_similarity'],
                        edge_type=row['edge_type']
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
        self.chunk_index: Dict[str, int] = {}  # chunk_id -> index in self.chunks
        self.faiss_index = None
        self.similarity_data: List[SimilarityMetrics] = []
        
        # Thread-safe lock for data updates
        self.data_lock = threading.Lock()
        
        logger.info(f"Initialized GraphAnalysisPipeline with GPU: {self.use_gpu}, Threads: {num_threads}")

    def load_chunks(self) -> None:
        """Load all document and table chunks with caching"""
        logger.info("Loading document and table chunks...")
        
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
        doc_chunks = self._load_document_chunks()
        logger.info(f"Loaded {len(doc_chunks)} document chunks")
        
        # Load table chunks
        table_chunks = self._load_table_chunks()
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

    def _load_document_chunks(self) -> List[ChunkData]:
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
                
                doc_chunks.append(chunk_data)
                
            except Exception as e:
                logger.warning(f"Error loading document chunk {json_file}: {e}")
                continue
        
        return doc_chunks

    def _load_table_chunks(self) -> List[ChunkData]:
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
                    content_embedding=[],  # Will be generated
                    entities=entities,
                    entity_embeddings={},
                    topic=table_description,  # Use table_description as topic
                    topic_embedding=None,
                    column_descriptions=col_desc,
                    column_embeddings={},
                    metadata=chunk_info.get('metadata', {})
                )
                
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
        
        # Entity embeddings
        if entity_texts:
            logger.info(f"Generating {len(entity_texts)} entity embeddings...")
            entity_embeddings = self.embedding_service.generate_embeddings(entity_texts)
            for (chunk_idx, entity_name), embedding in zip(entity_indices, entity_embeddings):
                self.chunks[chunk_idx].entity_embeddings[entity_name] = embedding
        
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
        dimension = embeddings_matrix.shape[1]
        
        logger.info(f"Building HNSW index with {len(embeddings)} embeddings, dimension={dimension}")
        
        # Create HNSW index
        self.faiss_index = faiss.IndexHNSWFlat(dimension, 32)  # M=32
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
            search_k = min(k + 1, self.faiss_index.ntotal)  # +1 to account for self
            similarities, indices = self.faiss_index.search(query_embedding, search_k)
            
            neighbor_chunk_ids = []
            for sim, idx in zip(similarities[0], indices[0]):
                if idx == -1:  # Invalid index
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
        """Phase 3: Calculate similarities for unique pairs using GPU acceleration"""
        
        # Check if we can use GPU acceleration
        if torch.cuda.is_available():
            logger.info("Using GPU-accelerated similarity calculation")
            
            # Initialize GPU calculator
            gpu_calculator = GPUAcceleratedSimilarityCalculator(
                chunks=self.chunks,
                embedding_service=self.embedding_service,
                batch_size=10000,  # Use 10k pairs per batch as requested
                gpu_id=0  # Use first GPU only
            )
            
            # Precompute embeddings on GPU
            gpu_calculator.precompute_embeddings()
            
            # Calculate similarities using GPU batch processing
            self.similarity_data = gpu_calculator.calculate_similarities_batch(unique_pairs)
            
            logger.info(f"GPU-accelerated processing completed: {len(self.similarity_data)} similarities calculated")
        else:
            logger.warning("GPU not available, falling back to CPU processing")
            self._calculate_similarities_for_pairs_cpu(unique_pairs)
    
    def _calculate_similarities_for_pairs_cpu(self, unique_pairs: List[Tuple[str, str]]):
        """CPU fallback for similarity calculation"""
        
        # Create pair batches for parallel processing
        pair_batches = []
        batch_size = max(1, len(unique_pairs) // self.num_threads)
        
        for i in range(0, len(unique_pairs), batch_size):
            pair_batches.append(unique_pairs[i:i + batch_size])
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for batch_id, pair_batch in enumerate(pair_batches):
                future = executor.submit(self._calculate_similarities_for_pair_batch, batch_id, pair_batch)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    batch_similarities = future.result()
                    with self.data_lock:
                        self.similarity_data.extend(batch_similarities)
                    logger.info(f"Completed similarity batch with {len(batch_similarities)} records")
                except Exception as e:
                    logger.error(f"Similarity calculation batch failed: {e}")

    def _calculate_similarities_for_pair_batch(self, batch_id: int, pair_batch: List[Tuple[str, str]]) -> List[SimilarityMetrics]:
        """Calculate similarities for a batch of unique pairs"""
        batch_similarities = []
        
        for source_chunk_id, target_chunk_id in pair_batch:
            try:
                # Get chunk objects from IDs
                source_chunk_idx = self.chunk_index.get(source_chunk_id)
                target_chunk_idx = self.chunk_index.get(target_chunk_id)
                
                if source_chunk_idx is None or target_chunk_idx is None:
                    logger.warning(f"Could not find chunks for pair ({source_chunk_id}, {target_chunk_id})")
                    continue
                
                source_chunk = self.chunks[source_chunk_idx]
                target_chunk = self.chunks[target_chunk_idx]
                
                # Calculate all similarities
                similarity_metrics = self._calculate_all_similarities(source_chunk, target_chunk)
                batch_similarities.append(similarity_metrics)
                
            except Exception as e:
                logger.warning(f"Error calculating similarities for pair ({source_chunk_id}, {target_chunk_id}): {e}")
                continue
        
        logger.debug(f"Batch {batch_id} calculated similarities for {len(batch_similarities)} pairs")
        return batch_similarities

    def _calculate_all_similarities(self, chunk1: ChunkData, chunk2: ChunkData) -> SimilarityMetrics:
        """Calculate all similarity metrics between two chunks"""
        
        # Determine edge type
        if chunk1.chunk_type == "document" and chunk2.chunk_type == "document":
            edge_type = "doc-doc"
        elif chunk1.chunk_type == "table" and chunk2.chunk_type == "table":
            edge_type = "table-table"
        else:
            edge_type = "doc-table"
        
        # Entity similarity (exact + fuzzy matching)
        entity_sim = self._calculate_entity_similarity(chunk1, chunk2)
        
        # Topic similarity (cosine similarity of topic embeddings)
        topic_sim = self._calculate_topic_similarity(chunk1, chunk2)
        
        # Content similarity (cosine similarity of content embeddings)
        content_sim = self._calculate_content_similarity(chunk1, chunk2)
        
        # Column similarity (for table-related edges)
        column_sim = self._calculate_column_similarity(chunk1, chunk2)
        
        return SimilarityMetrics(
            entity_similarity=entity_sim,
            topic_similarity=topic_sim,
            content_similarity=content_sim,
            column_similarity=column_sim,
            edge_type=edge_type,
            source_chunk_id=chunk1.chunk_id,
            target_chunk_id=chunk2.chunk_id
        )

    def _calculate_entity_similarity(self, chunk1: ChunkData, chunk2: ChunkData) -> float:
        """Calculate entity similarity using embedding-based similarity with threshold"""
        entities1 = chunk1.entities.keys()
        entities2 = chunk2.entities.keys()
        
        if not entities1 or not entities2:
            return 0.0
        
        # Use embeddings for entity similarity
        similarities = []
        threshold = 0.0
        
        for entity1 in entities1:
            if entity1 not in chunk1.entity_embeddings:
                continue
            
            best_similarity = 0.0
            for entity2 in entities2:
                if entity2 not in chunk2.entity_embeddings:
                    continue
                
                try:
                    # Calculate cosine similarity between entity embeddings
                    sim = cosine_similarity(
                        [chunk1.entity_embeddings[entity1]], 
                        [chunk2.entity_embeddings[entity2]]
                    )[0][0]
                    
                    if sim > threshold:  # Only consider similarities above threshold
                        best_similarity = max(best_similarity, sim)
                except Exception:
                    continue
            
            if best_similarity > 0:
                similarities.append(best_similarity)
        
        # Return average of all above-threshold similarities
        if not similarities:
            return 0.0
        top_3_similarities = sorted(similarities, reverse=True)[:5]
        return np.mean(top_3_similarities)

    def _calculate_topic_similarity(self, chunk1: ChunkData, chunk2: ChunkData) -> float:
        """Calculate topic similarity using topic embeddings"""
        if not chunk1.topic_embedding or not chunk2.topic_embedding:
            return 0.0
        
        try:
            similarity = cosine_similarity([chunk1.topic_embedding], [chunk2.topic_embedding])[0][0]
            return max(0.0, similarity)  # Ensure non-negative
        except Exception:
            return 0.0

    def _calculate_content_similarity(self, chunk1: ChunkData, chunk2: ChunkData) -> float:
        """Calculate content similarity using content embeddings"""
        if not chunk1.content_embedding or not chunk2.content_embedding:
            return 0.0
        
        try:
            similarity = cosine_similarity([chunk1.content_embedding], [chunk2.content_embedding])[0][0]
            return max(0.0, similarity)  # Ensure non-negative
        except Exception:
            return 0.0

    def _calculate_column_similarity(self, chunk1: ChunkData, chunk2: ChunkData) -> float:
        """Calculate column similarity for table-related edges"""
        
        # For doc-doc edges, no column similarity
        if chunk1.chunk_type == "document" and chunk2.chunk_type == "document":
            return 0.0
        
        # For table-table edges
        if chunk1.chunk_type == "table" and chunk2.chunk_type == "table":
            return self._calculate_table_table_column_similarity(chunk1, chunk2)
        
        # For doc-table edges
        return self._calculate_doc_table_column_similarity(chunk1, chunk2)

    def _calculate_table_table_column_similarity(self, table1: ChunkData, table2: ChunkData) -> float:
        """Calculate column similarity between two tables"""
        if not table1.column_embeddings or not table2.column_embeddings:
            return 0.0
        
        similarities = []
        
        # Compare each column in table1 with each column in table2
        for col1_name, col1_emb in table1.column_embeddings.items():
            best_similarity = 0.0
            for col2_name, col2_emb in table2.column_embeddings.items():
                try:
                    sim = cosine_similarity([col1_emb], [col2_emb])[0][0]
                    best_similarity = max(best_similarity, sim)
                except Exception:
                    continue
            similarities.append(best_similarity)

        if not similarities:
            return 0.0
        top_3_similarities = sorted(similarities, reverse=True)[:3]
        return np.mean(top_3_similarities)

    def _calculate_doc_table_column_similarity(self, chunk1: ChunkData, chunk2: ChunkData) -> float:
        """Calculate similarity between document and table columns"""
        
        # Identify which is doc and which is table
        if chunk1.chunk_type == "document":
            doc_chunk, table_chunk = chunk1, chunk2
        else:
            doc_chunk, table_chunk = chunk2, chunk1
        
        if not doc_chunk.content_embedding or not table_chunk.column_embeddings:
            return 0.0
        
        similarities = []
        
        # Compare document content with each column description
        for col_name, col_emb in table_chunk.column_embeddings.items():
            try:
                sim = cosine_similarity([doc_chunk.content_embedding], [col_emb])[0][0]
                similarities.append(sim)
            except Exception:
                continue
        
        if not similarities:
            return 0.0
        top_3_similarities = sorted(similarities, reverse=True)[:3]
        return np.mean(top_3_similarities)

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
        
        logger.info(f"Analysis reports saved to {output_dir}")

    def _create_similarity_dataframe(self) -> pd.DataFrame:
        """Convert similarity data to pandas DataFrame"""
        data = []
        
        for sim in self.similarity_data:
            data.append({
                'entity_similarity': sim.entity_similarity,
                'topic_similarity': sim.topic_similarity,
                'content_similarity': sim.content_similarity,
                'column_similarity': sim.column_similarity,
                'edge_type': sim.edge_type,
                'source_chunk_id': sim.source_chunk_id,
                'target_chunk_id': sim.target_chunk_id
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
        
        # Metrics relevant for doc-doc: entity_similarity, topic_similarity, content_similarity
        relevant_cols = ['entity_similarity', 'topic_similarity', 'content_similarity']
        
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
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for i, col in enumerate(relevant_cols):
            sns.histplot(data=doc_doc_df, x=col, kde=True, ax=axes[i], bins=30)
            axes[i].set_title(f'Doc-Doc: {col.replace("_", " ").title()}')
            axes[i].grid(True, alpha=0.3)
        
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
            "SIMILARITY METRICS SUMMARY:",
        ]
        
        for col in relevant_cols:
            mean_val = doc_doc_df[col].mean()
            std_val = doc_doc_df[col].std()
            insights.append(f"  {col}: Mean={mean_val:.3f}, Std={std_val:.3f}")
        
        insights.append("")
        insights.append("HIGH CORRELATIONS (|r| > 0.7):")
        for i, col1 in enumerate(relevant_cols):
            for j, col2 in enumerate(relevant_cols[i+1:], i+1):
                corr = correlation_matrix.loc[col1, col2]
                if abs(corr) > 0.7:
                    insights.append(f"  {col1} - {col2}: {corr:.3f}")
        
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
        
        # Metrics relevant for doc-table: entity_similarity, topic_similarity, column_similarity
        relevant_cols = ['entity_similarity', 'topic_similarity', 'column_similarity']
        
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
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for i, col in enumerate(relevant_cols):
            sns.histplot(data=doc_table_df, x=col, kde=True, ax=axes[i], bins=30)
            axes[i].set_title(f'Doc-Table: {col.replace("_", " ").title()}')
            axes[i].grid(True, alpha=0.3)
        
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
            "SIMILARITY METRICS SUMMARY:",
            "  - Entity similarity: Embedding-based with 0.4 threshold",
            "  - Topic similarity: Document topic vs Table description",
            "  - Column similarity: Document content vs Column descriptions",
            "",
        ]
        
        for col in relevant_cols:
            mean_val = doc_table_df[col].mean()
            std_val = doc_table_df[col].std()
            insights.append(f"  {col}: Mean={mean_val:.3f}, Std={std_val:.3f}")
        
        insights.append("")
        insights.append("HIGH CORRELATIONS (|r| > 0.7):")
        for i, col1 in enumerate(relevant_cols):
            for j, col2 in enumerate(relevant_cols[i+1:], i+1):
                corr = correlation_matrix.loc[col1, col2]
                if abs(corr) > 0.7:
                    insights.append(f"  {col1} - {col2}: {corr:.3f}")
        
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
        
        # Metrics relevant for table-table: entity_similarity, topic_similarity, column_similarity
        relevant_cols = ['entity_similarity', 'topic_similarity', 'column_similarity']
        
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
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for i, col in enumerate(relevant_cols):
            sns.histplot(data=table_table_df, x=col, kde=True, ax=axes[i], bins=30)
            axes[i].set_title(f'Table-Table: {col.replace("_", " ").title()}')
            axes[i].grid(True, alpha=0.3)
        
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
            "SIMILARITY METRICS SUMMARY:",
            "  - Entity similarity: Embedding-based with 0.4 threshold",
            "  - Topic similarity: Table description vs Table description",
            "  - Column similarity: Column description similarities",
            "",
        ]
        
        for col in relevant_cols:
            mean_val = table_table_df[col].mean()
            std_val = table_table_df[col].std()
            insights.append(f"  {col}: Mean={mean_val:.3f}, Std={std_val:.3f}")
        
        insights.append("")
        insights.append("HIGH CORRELATIONS (|r| > 0.7):")
        for i, col1 in enumerate(relevant_cols):
            for j, col2 in enumerate(relevant_cols[i+1:], i+1):
                corr = correlation_matrix.loc[col1, col2]
                if abs(corr) > 0.7:
                    insights.append(f"  {col1} - {col2}: {corr:.3f}")
        
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
        
        # Overall statistics
        numeric_cols = ['entity_similarity', 'topic_similarity', 'content_similarity', 'column_similarity']
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
            insights.append(f"  {edge_type}: {count:,} edges ({percentage:.1f}%)")
        
        insights.extend([
            "",
            "METHODOLOGY:",
            "  - Entity Similarity: Embedding-based with 0.4 threshold",
            "  - Topic Similarity: Cosine similarity of topic/table_description embeddings",
            "  - Content Similarity: Cosine similarity of full content embeddings",
            "  - Column Similarity: Content vs column descriptions (doc-table) or column-column (table-table)",
            "",
            "SECTIONS GENERATED:",
            "  1. Section 1: Doc-Doc edge analysis",
            "  2. Section 2: Doc-Table edge analysis", 
            "  3. Section 3: Table-Table edge analysis",
            "",
            "RECOMMENDATIONS FOR GRAPH CONSTRUCTION:",
            "  - Review correlation matrices in each section to identify independent metrics",
            "  - Use weighted averages only for uncorrelated metrics (|r| < 0.7)",
            "  - Consider edge-type specific weighting based on metric distributions",
        ])
        
        with open(output_dir / 'overall_summary.txt', 'w') as f:
            f.write('\n'.join(insights))

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
        pipeline.analyze_chunk_similarities(k_neighbors=1000)
        
        # Step 4: Generate reports
        logger.info("=== Step 4: Generating Analysis Reports ===")
        pipeline.generate_analysis_reports()
        
        logger.info("Graph analysis pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main() 