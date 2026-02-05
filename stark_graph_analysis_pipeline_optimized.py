#!/usr/bin/env python3
"""
ULTRA-FAST STARK Dataset Graph Analysis Pipeline
Optimized for maximum GPU utilization and bulk processing
Expected: 100x+ speedup over original implementation
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
import pickle
import faiss
import torch
from loguru import logger
import re
import gc
from tqdm import tqdm
import math

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.processors.embedding_service import EmbeddingService
from src.core.models import ProcessingConfig

def make_hashable(item):
    """Convert any nested structure to be hashable"""
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
    content_similarity: float
    edge_type: str
    source_chunk_id: str
    target_chunk_id: str
    entity_count: int = 0
    title_similarity: float = 0.0
    feature_similarity: float = 0.0
    detail_similarity: float = 0.0
    description_similarity: float = 0.0
    reviews_summary_similarity: float = 0.0
    reviews_text_similarity: float = 0.0

@dataclass 
class STARKChunkData:
    """STARK-specific chunk data with computed embeddings"""
    chunk_id: str
    chunk_type: str  # "document" or "table"
    content: str
    entities: List[str]
    asin: str
    
    # Document-specific fields
    title: Optional[str] = None
    brand: Optional[str] = None
    global_category: Optional[str] = None
    category: Optional[List[str]] = None
    feature: Optional[List[str]] = None
    detail: Optional[str] = None
    description: Optional[List[str]] = None
    
    # Table-specific fields
    reviews_data: Optional[pd.DataFrame] = None
    review_count: Optional[int] = None
    
    # All embeddings (will be loaded from cache)
    content_embedding: Optional[List[float]] = None
    title_embedding: Optional[List[float]] = None
    feature_embedding: Optional[List[float]] = None
    detail_embedding: Optional[List[float]] = None
    description_embedding: Optional[List[float]] = None
    reviews_summary_embedding: Optional[List[float]] = None
    reviews_text_embedding: Optional[List[float]] = None
    
    metadata: Optional[Dict[str, Any]] = None

class UltraFastSTARKPipeline:
    """
    Ultra-fast STARK pipeline optimized for maximum GPU utilization
    Expected 100x+ speedup through bulk processing and multi-GPU parallelization
    """
    
    def __init__(self, 
                 stark_dataset_file: str = "/shared/khoja/CogComp/datasets/STARK/node_info.json",
                 cache_dir: str = "/shared/khoja/CogComp/output/stark_ultrafast_cache",
                 use_gpu: bool = True,
                 num_threads: int = 64):
        
        self.stark_dataset_file = Path(stark_dataset_file)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.num_threads = num_threads
        self.num_gpus = torch.cuda.device_count() if self.use_gpu else 0
        
        # Initialize optimized embedding service
        self.config = ProcessingConfig(
            use_faiss=True, 
            faiss_use_gpu=self.use_gpu,
            batch_size=4096,  # Large batch size for optimization
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedding_service = EmbeddingService(self.config)
        
        # Data containers
        self.chunks: List[STARKChunkData] = []
        self.chunk_index: Dict[str, int] = {}
        self.faiss_index = None
        self.faiss_to_chunk_mapping = []
        self.similarity_data: List[STARKSimilarityMetrics] = []
        self.outlier_edges = []
        
        # Stage cache files
        self.stage_files = {
            'chunks': self.cache_dir / 'stage1_chunks.pkl',
            'all_embeddings': self.cache_dir / 'stage2_all_embeddings.pkl',
            'hnsw_index': self.cache_dir / 'stage3_hnsw_index.faiss',
            'hnsw_mapping': self.cache_dir / 'stage3_hnsw_mapping.pkl',
            'similarities': self.cache_dir / 'stage4_similarities.pkl',
            'analysis_complete': self.cache_dir / 'stage5_analysis_complete.flag'
        }
        
        logger.info(f"🚀 Initialized Ultra-Fast STARK Pipeline with {self.num_gpus} GPUs")
        logger.info(f"💡 Expected speedup: 100x+ through bulk processing")

    def run_full_pipeline(self, max_products: Optional[int] = None, k_neighbors: int = 200):
        """Run the complete ultra-fast pipeline"""
        logger.info("🚀 Starting Ultra-Fast STARK Pipeline")
        logger.info("=" * 60)
        
        total_start_time = time.time()
        
        try:
            # Stage 1: Load All Chunks (CPU Task)
            self.stage1_load_all_chunks(max_products)
            
            # Stage 2: Bulk Calculate ALL Embeddings (Optimized GPU Task)
            self.stage2_bulk_calculate_all_embeddings()
            
            # Stage 3: Build Optimized HNSW Index (GPU Task)
            self.stage3_build_optimized_hnsw_index()
            
            # Stage 4: Vectorized Similarity Calculations (GPU Task)
            self.stage4_vectorized_calculate_similarities(k_neighbors)
            
            # Stage 5: Generate Analysis & Reports (CPU Task)
            edges_file = self.stage5_generate_analysis()
            
            total_time = time.time() - total_start_time
            
            logger.info("🎉 Ultra-Fast STARK Pipeline completed successfully!")
            logger.info(f"⏱️  Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
            logger.info(f"📁 Final results: {edges_file}")
            
            return edges_file
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise

    def stage1_load_all_chunks(self, max_products: Optional[int] = None):
        """Stage 1: Load and cache ALL STARK chunks in memory"""
        if self.stage_files['chunks'].exists():
            logger.info("✅ Stage 1: Loading chunks from cache...")
            with open(self.stage_files['chunks'], 'rb') as f:
                self.chunks = pickle.load(f)
            self._build_chunk_index()
            logger.info(f"   Loaded {len(self.chunks):,} chunks from cache")
            return
        
        logger.info("🔄 Stage 1: Loading ALL STARK chunks from dataset...")
        start_time = time.time()
        
        # Load raw data
        with open(self.stark_dataset_file, 'r') as f:
            stark_data = json.load(f)
        
        logger.info(f"   Loaded {len(stark_data):,} products from dataset")
        
        # Limit products if specified
        if max_products:
            product_keys = list(stark_data.keys())[:max_products]
            stark_data = {k: stark_data[k] for k in product_keys}
            logger.info(f"   Limited to {max_products:,} products")
        
        # Process ALL products into chunks with multi-threading
        logger.info("   Processing products into chunks with multi-threading...")
        
        product_items = list(stark_data.items())
        all_chunks = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_product = {
                executor.submit(self._process_product_to_chunks, product_id, product_data): product_id
                for product_id, product_data in product_items
            }
            
            for future in tqdm(as_completed(future_to_product), total=len(future_to_product), desc="Processing products"):
                try:
                    chunks = future.result()
                    if chunks:
                        all_chunks.extend(chunks)
                except Exception as e:
                    product_id = future_to_product[future]
                    logger.warning(f"   Failed to process product {product_id}: {e}")
        
        self.chunks = all_chunks
        self._build_chunk_index()
        
        # Cache chunks
        with open(self.stage_files['chunks'], 'wb') as f:
            pickle.dump(self.chunks, f)
        
        doc_chunks = [c for c in self.chunks if c.chunk_type == "document"]
        table_chunks = [c for c in self.chunks if c.chunk_type == "table"]
        
        load_time = time.time() - start_time
        logger.info(f"✅ Stage 1 Complete: {len(doc_chunks):,} document + {len(table_chunks):,} table chunks in {load_time:.2f}s")

    def stage2_bulk_calculate_all_embeddings(self):
        """Stage 2: Bulk calculate ALL embeddings at once using optimized GPU processing"""
        if self.stage_files['all_embeddings'].exists():
            logger.info("✅ Stage 2: Loading all embeddings from cache...")
            with open(self.stage_files['all_embeddings'], 'rb') as f:
                self.chunks = pickle.load(f)
            self._build_chunk_index()
            logger.info(f"   Loaded embeddings for {len(self.chunks):,} chunks from cache")
            return
        
        logger.info("🔄 Stage 2: BULK calculating ALL embeddings with ultra-fast GPU processing...")
        start_time = time.time()
        
        # Load chunks if not already loaded
        if not self.chunks:
            self.stage1_load_all_chunks()
        
        # Prepare ALL texts at once
        logger.info("   Preparing ALL text content for bulk embedding...")
        text_preparation_start = time.time()
        
        all_text_data = self._prepare_all_texts_bulk()
        
        text_prep_time = time.time() - text_preparation_start
        logger.info(f"   Text preparation completed in {text_prep_time:.2f}s")
        
        # Calculate ALL embeddings in one massive operation
        logger.info("🔥 Starting MASSIVE BULK embedding generation...")
        embedding_start_time = time.time()
        
        total_texts = sum(len(texts) for texts in all_text_data.values())
        logger.info(f"   Total texts to embed: {total_texts:,}")
        
        # Process each field type separately with memory management
        for field_type, data in all_text_data.items():
            if not data['texts']:
                continue
                
            logger.info(f"   🔥 Processing {field_type}: {len(data['texts']):,} texts")
            field_start = time.time()
            
            # Clear GPU memory before each field
            if hasattr(self.embedding_service, '_clear_gpu_cache'):
                self.embedding_service._clear_gpu_cache()
            
            # Generate embeddings for this field
            try:
                embeddings = self.embedding_service.generate_embeddings_bulk(data['texts'])
                
                field_time = time.time() - field_start
                texts_per_sec = len(data['texts']) / field_time if field_time > 0 else 0
                logger.info(f"   ✅ {field_type} completed: {len(embeddings):,} embeddings in {field_time:.2f}s ({texts_per_sec:.0f} texts/sec)")
                
                # Assign embeddings back to chunks
                self._assign_embeddings_to_chunks(field_type, data['chunk_indices'], embeddings)
                
                # Clear memory after processing each field
                del embeddings
                if hasattr(self.embedding_service, '_clear_gpu_cache'):
                    self.embedding_service._clear_gpu_cache()
                    
            except Exception as e:
                logger.error(f"❌ Failed to process {field_type}: {e}")
                # Continue with other fields even if one fails
                continue
        
        # Assign zero vectors to chunks without embeddings
        self._assign_zero_vectors_to_missing_embeddings()
        
        embedding_time = time.time() - embedding_start_time
        total_time = time.time() - start_time
        
        # Save all embeddings
        with open(self.stage_files['all_embeddings'], 'wb') as f:
            pickle.dump(self.chunks, f)
        
        logger.info(f"✅ Stage 2 Complete: ALL embeddings calculated in {embedding_time:.2f}s (total: {total_time:.2f}s)")
        logger.info(f"🚀 Achieved ~{total_texts/embedding_time:.0f} texts/second processing rate!")

    def stage3_build_optimized_hnsw_index(self):
        """Stage 3: Build optimized HNSW index using ALL content embeddings at once"""
        if self.stage_files['hnsw_index'].exists() and self.stage_files['hnsw_mapping'].exists():
            logger.info("✅ Stage 3: Loading HNSW index from cache...")
            self.faiss_index = faiss.read_index(str(self.stage_files['hnsw_index']))
            with open(self.stage_files['hnsw_mapping'], 'rb') as f:
                self.faiss_to_chunk_mapping = pickle.load(f)
            logger.info(f"   Loaded HNSW index with {self.faiss_index.ntotal:,} embeddings")
            return
        
        logger.info("🔄 Stage 3: Building optimized HNSW index...")
        start_time = time.time()
        
        # Load chunks with embeddings if not loaded
        if not self.chunks or not self.chunks[0].content_embedding:
            self.stage2_bulk_calculate_all_embeddings()
        
        # Collect ALL embeddings at once
        logger.info("   Collecting ALL content embeddings...")
        embeddings = []
        valid_chunk_indices = []
        
        for i, chunk in enumerate(self.chunks):
            if chunk.content_embedding and any(x != 0.0 for x in chunk.content_embedding):
                embeddings.append(chunk.content_embedding)
                valid_chunk_indices.append(i)
        
        if not embeddings:
            raise ValueError("No valid embeddings found for HNSW index")
        
        # Convert to numpy array for optimized processing
        embeddings_matrix = np.array(embeddings, dtype=np.float32)
        dimension = embeddings_matrix.shape[1]
        
        logger.info(f"   Building optimized HNSW index: {len(embeddings):,} embeddings, dimension={dimension}")
        
        # Create optimized HNSW index with better parameters
        self.faiss_index = faiss.IndexHNSWFlat(dimension, 64)  # M=64 for better recall
        self.faiss_index.hnsw.efConstruction = 2000  # Higher for better quality
        self.faiss_index.hnsw.efSearch = 1000  # Higher for better search quality
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_matrix)
        
        # Use ALL available GPUs if possible
        if self.use_gpu and faiss.get_num_gpus() > 0:
            try:
                logger.info(f"   🔥 Using ALL {faiss.get_num_gpus()} GPUs for ultra-fast index building")
                
                # Use multiple GPUs for index building
                gpu_resources = [faiss.StandardGpuResources() for _ in range(faiss.get_num_gpus())]
                gpu_index = faiss.index_cpu_to_all_gpus(self.faiss_index)
                
                index_start = time.time()
                gpu_index.add(embeddings_matrix)
                index_time = time.time() - index_start
                
                self.faiss_index = faiss.index_gpu_to_cpu(gpu_index)
                
                # Clean up GPU resources
                for res in gpu_resources:
                    del res
                self._clear_gpu_memory()
                
                logger.info(f"   ✅ GPU index building completed in {index_time:.2f}s")
                
            except Exception as e:
                logger.warning(f"   GPU indexing failed: {e}, falling back to CPU")
                self.faiss_index.add(embeddings_matrix)
        else:
            self.faiss_index.add(embeddings_matrix)
        
        # Store mapping
        self.faiss_to_chunk_mapping = valid_chunk_indices
        
        # Save index and mapping
        faiss.write_index(self.faiss_index, str(self.stage_files['hnsw_index']))
        with open(self.stage_files['hnsw_mapping'], 'wb') as f:
            pickle.dump(self.faiss_to_chunk_mapping, f)
        
        total_time = time.time() - start_time
        logger.info(f"✅ Stage 3 Complete: Optimized HNSW index built in {total_time:.2f}s")

    def stage4_vectorized_calculate_similarities(self, k_neighbors: int = 200):
        """Stage 4: Vectorized similarity calculations using GPU tensor operations"""
        if self.stage_files['similarities'].exists():
            logger.info("✅ Stage 4: Loading similarities from cache...")
            with open(self.stage_files['similarities'], 'rb') as f:
                self.similarity_data = pickle.load(f)
            logger.info(f"   Loaded {len(self.similarity_data):,} similarities from cache")
            return
        
        logger.info("🔄 Stage 4: VECTORIZED similarity calculations with GPU acceleration...")
        start_time = time.time()
        
        # Load previous stages if not loaded
        if not self.chunks:
            self.stage2_bulk_calculate_all_embeddings()
        if not self.faiss_index:
            self.stage3_build_optimized_hnsw_index()
        
        # Get ALL content embeddings for vectorized operations
        logger.info("   Preparing embeddings for vectorized similarity search...")
        all_embeddings = []
        chunk_id_to_embedding_idx = {}
        
        for i, chunk in enumerate(self.chunks):
            if chunk.content_embedding and any(x != 0.0 for x in chunk.content_embedding):
                chunk_id_to_embedding_idx[chunk.chunk_id] = len(all_embeddings)
                all_embeddings.append(chunk.content_embedding)
        
        logger.info(f"   Using {len(all_embeddings):,} embeddings for similarity search")
        
        # Vectorized k-NN search for ALL chunks at once
        logger.info(f"   🔥 Performing vectorized k-NN search (k={k_neighbors}) for ALL chunks...")
        knn_start = time.time()
        
        # Use FAISS for ultra-fast k-NN search
        query_embeddings = np.array(all_embeddings, dtype=np.float32)
        faiss.normalize_L2(query_embeddings)
        
        # Search for k+1 to exclude self-matches
        search_k = min(k_neighbors + 1, self.faiss_index.ntotal)
        similarities, indices = self.faiss_index.search(query_embeddings, search_k)
        
        knn_time = time.time() - knn_start
        total_queries = len(query_embeddings) * search_k
        logger.info(f"   ✅ Vectorized k-NN completed: {total_queries:,} similarity computations in {knn_time:.2f}s")
        logger.info(f"   🚀 Achieved {total_queries/knn_time:.0f} similarities/second!")
        
        # Convert results to similarity metrics using vectorized operations
        logger.info("   Converting k-NN results to similarity metrics...")
        conversion_start = time.time()
        
        self.similarity_data = self._convert_knn_results_to_similarities_vectorized(
            similarities, indices, chunk_id_to_embedding_idx
        )
        
        conversion_time = time.time() - conversion_start
        logger.info(f"   ✅ Conversion completed in {conversion_time:.2f}s")
        
        # Save similarities
        with open(self.stage_files['similarities'], 'wb') as f:
            pickle.dump(self.similarity_data, f)
        
        total_time = time.time() - start_time
        logger.info(f"✅ Stage 4 Complete: {len(self.similarity_data):,} similarities calculated in {total_time:.2f}s")

    def stage5_generate_analysis(self):
        """Stage 5: Generate analysis and extract edges"""
        if self.stage_files['analysis_complete'].exists():
            logger.info("✅ Stage 5: Analysis already complete, loading results...")
            edges_files = list(self.cache_dir.glob("stark_ultrafast_edges_*.json"))
            if edges_files:
                latest_edges_file = max(edges_files, key=lambda x: x.stat().st_mtime)
                logger.info(f"   Found existing results: {latest_edges_file}")
                return latest_edges_file
        
        logger.info("🔄 Stage 5: Generating analysis and extracting edges...")
        start_time = time.time()
        
        # Load similarities if not loaded
        if not self.similarity_data:
            self.stage4_vectorized_calculate_similarities()
        
        # Generate analysis reports
        self._generate_analysis_reports()
        
        # Extract unique edges
        edges_file = self._extract_and_save_unique_edges()
        
        # Mark stage as complete
        with open(self.stage_files['analysis_complete'], 'w') as f:
            f.write(f"Ultra-fast analysis completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        total_time = time.time() - start_time
        logger.info(f"✅ Stage 5 Complete: Analysis completed in {total_time:.2f}s")
        
        return edges_file

    # Helper methods for optimized processing
    def _process_product_to_chunks(self, product_id: str, product_data: Dict[str, Any]) -> List[STARKChunkData]:
        """Process a single product into chunks (thread-safe)"""
        chunks = []
        try:
            # Create document chunk
            doc_chunk = self._create_product_document_chunk(product_id, product_data)
            if doc_chunk:
                chunks.append(doc_chunk)
            
            # Create table chunk
            table_chunk = self._create_product_reviews_table_chunk(product_id, product_data)
            if table_chunk:
                chunks.append(table_chunk)
        except Exception as e:
            logger.warning(f"Failed to process product {product_id}: {e}")
        
        return chunks

    def _prepare_all_texts_bulk(self) -> Dict[str, Dict[str, List]]:
        """Prepare ALL texts for bulk embedding generation with optimized parallel processing"""
        logger.info("   🚀 Using optimized parallel text preparation...")
        
        # Pre-allocate with estimated sizes to avoid memory reallocation
        estimated_size = len(self.chunks)
        all_text_data = {
            'content': {'texts': [], 'chunk_indices': []},
            'title': {'texts': [], 'chunk_indices': []},
            'feature': {'texts': [], 'chunk_indices': []},
            'detail': {'texts': [], 'chunk_indices': []},
            'description': {'texts': [], 'chunk_indices': []},
            'reviews_summary': {'texts': [], 'chunk_indices': []},
            'reviews_text': {'texts': [], 'chunk_indices': []}
        }
        
        # Use ThreadPoolExecutor for parallel text processing
        def process_chunk_batch(chunk_batch):
            """Process a batch of chunks in parallel"""
            batch_results = {
                'content': {'texts': [], 'chunk_indices': []},
                'title': {'texts': [], 'chunk_indices': []},
                'feature': {'texts': [], 'chunk_indices': []},
                'detail': {'texts': [], 'chunk_indices': []},
                'description': {'texts': [], 'chunk_indices': []},
                'reviews_summary': {'texts': [], 'chunk_indices': []},
                'reviews_text': {'texts': [], 'chunk_indices': []}
            }
            
            for chunk_idx, chunk in chunk_batch:
                # Content embedding (main content)
                content = self._build_chunk_content(chunk)
                if content.strip():
                    batch_results['content']['texts'].append(content.strip())
                    batch_results['content']['chunk_indices'].append(chunk_idx)
                
                # Field-specific embeddings
                if chunk.chunk_type == "document":
                    self._collect_document_field_texts_batch(chunk, chunk_idx, batch_results)
                elif chunk.chunk_type == "table":
                    self._collect_table_field_texts_batch(chunk, chunk_idx, batch_results)
            
            return batch_results
        
        # Process chunks in parallel batches
        batch_size = 10000  # Process 10K chunks per batch
        chunk_batches = []
        
        for i in range(0, len(self.chunks), batch_size):
            batch = [(j, self.chunks[j]) for j in range(i, min(i + batch_size, len(self.chunks)))]
            chunk_batches.append(batch)
        
        logger.info(f"   Processing {len(self.chunks):,} chunks in {len(chunk_batches)} parallel batches...")
        
        # Use parallel processing
        with ThreadPoolExecutor(max_workers=min(16, len(chunk_batches))) as executor:
            future_to_batch = {executor.submit(process_chunk_batch, batch): batch for batch in chunk_batches}
            
            for future in tqdm(as_completed(future_to_batch), total=len(future_to_batch), desc="Processing text batches"):
                try:
                    batch_results = future.result()
                    
                    # Merge batch results efficiently
                    for field_type in all_text_data:
                        all_text_data[field_type]['texts'].extend(batch_results[field_type]['texts'])
                        all_text_data[field_type]['chunk_indices'].extend(batch_results[field_type]['chunk_indices'])
                        
                except Exception as e:
                    logger.warning(f"   Failed to process text batch: {e}")
        
        # Log statistics
        for field_type, data in all_text_data.items():
            logger.info(f"   {field_type}: {len(data['texts']):,} texts prepared")
        
        return all_text_data

    def _build_chunk_content(self, chunk: STARKChunkData) -> str:
        """Build main content for a chunk"""
        if chunk.chunk_type == "document":
            content_parts = []
            if chunk.title:
                content_parts.append(chunk.title)
            if chunk.global_category:
                content_parts.append(chunk.global_category)
            if chunk.category:
                if isinstance(chunk.category, list):
                    content_parts.extend(chunk.category)
                else:
                    content_parts.append(str(chunk.category))
            if chunk.brand:
                content_parts.append(chunk.brand)
            if chunk.feature:
                if isinstance(chunk.feature, list):
                    content_parts.extend([str(f) for f in chunk.feature])
                else:
                    content_parts.append(str(chunk.feature))
            if chunk.detail:
                content_parts.append(chunk.detail)
            if chunk.description:
                if isinstance(chunk.description, list):
                    content_parts.extend([str(d) for d in chunk.description])
                else:
                    content_parts.append(str(chunk.description))
            
            return ' '.join([p.strip() for p in content_parts if p and str(p).strip()])
        
        else:  # table
            content_parts = []
            if chunk.reviews_data is not None:
                if 'summary' in chunk.reviews_data.columns:
                    summaries = chunk.reviews_data['summary'].fillna('').astype(str)
                    valid_summaries = [s.strip() for s in summaries if s.strip()]
                    content_parts.extend(valid_summaries)
                
                if 'reviewText' in chunk.reviews_data.columns:
                    review_texts = chunk.reviews_data['reviewText'].fillna('').astype(str)
                    valid_texts = [r.strip() for r in review_texts if r.strip()]
                    content_parts.extend(valid_texts)
            
            return ' '.join(content_parts)

    def _collect_document_field_texts(self, chunk: STARKChunkData, chunk_idx: int, all_text_data: Dict):
        """Collect field-specific texts for document chunks"""
        if chunk.title and chunk.title.strip():
            all_text_data['title']['texts'].append(chunk.title.strip())
            all_text_data['title']['chunk_indices'].append(chunk_idx)
        
        if chunk.feature:
            feature_text = ' '.join(chunk.feature) if isinstance(chunk.feature, list) else str(chunk.feature)
            if feature_text.strip():
                all_text_data['feature']['texts'].append(feature_text.strip())
                all_text_data['feature']['chunk_indices'].append(chunk_idx)
        
        if chunk.detail and chunk.detail.strip():
            all_text_data['detail']['texts'].append(chunk.detail.strip())
            all_text_data['detail']['chunk_indices'].append(chunk_idx)
        
        if chunk.description:
            desc_text = ' '.join(chunk.description) if isinstance(chunk.description, list) else str(chunk.description)
            if desc_text.strip():
                all_text_data['description']['texts'].append(desc_text.strip())
                all_text_data['description']['chunk_indices'].append(chunk_idx)

    def _collect_document_field_texts_batch(self, chunk: STARKChunkData, chunk_idx: int, batch_results: Dict):
        """Optimized batch version for document field text collection"""
        if chunk.title and chunk.title.strip():
            batch_results['title']['texts'].append(chunk.title.strip())
            batch_results['title']['chunk_indices'].append(chunk_idx)
        
        if chunk.feature:
            feature_text = ' '.join(chunk.feature) if isinstance(chunk.feature, list) else str(chunk.feature)
            if feature_text.strip():
                batch_results['feature']['texts'].append(feature_text.strip())
                batch_results['feature']['chunk_indices'].append(chunk_idx)
        
        if chunk.detail and chunk.detail.strip():
            batch_results['detail']['texts'].append(chunk.detail.strip())
            batch_results['detail']['chunk_indices'].append(chunk_idx)
        
        if chunk.description:
            desc_text = ' '.join(chunk.description) if isinstance(chunk.description, list) else str(chunk.description)
            if desc_text.strip():
                batch_results['description']['texts'].append(desc_text.strip())
                batch_results['description']['chunk_indices'].append(chunk_idx)

    def _collect_table_field_texts(self, chunk: STARKChunkData, chunk_idx: int, all_text_data: Dict):
        """Collect field-specific texts for table chunks"""
        if chunk.reviews_data is not None:
            if 'summary' in chunk.reviews_data.columns:
                summaries = chunk.reviews_data['summary'].fillna('').astype(str)
                valid_summaries = [s.strip() for s in summaries if s.strip()]
                if valid_summaries:
                    all_text_data['reviews_summary']['texts'].append(' '.join(valid_summaries))
                    all_text_data['reviews_summary']['chunk_indices'].append(chunk_idx)
            
            if 'reviewText' in chunk.reviews_data.columns:
                review_texts = chunk.reviews_data['reviewText'].fillna('').astype(str)
                valid_texts = [r.strip() for r in review_texts if r.strip()]
                if valid_texts:
                    all_text_data['reviews_text']['texts'].append(' '.join(valid_texts))
                    all_text_data['reviews_text']['chunk_indices'].append(chunk_idx)

    def _collect_table_field_texts_batch(self, chunk: STARKChunkData, chunk_idx: int, batch_results: Dict):
        """Optimized batch version for table field text collection"""
        if chunk.reviews_data is not None:
            if 'summary' in chunk.reviews_data.columns:
                summaries = chunk.reviews_data['summary'].fillna('').astype(str)
                valid_summaries = [s.strip() for s in summaries if s.strip()]
                if valid_summaries:
                    batch_results['reviews_summary']['texts'].append(' '.join(valid_summaries))
                    batch_results['reviews_summary']['chunk_indices'].append(chunk_idx)
            
            if 'reviewText' in chunk.reviews_data.columns:
                review_texts = chunk.reviews_data['reviewText'].fillna('').astype(str)
                valid_texts = [r.strip() for r in review_texts if r.strip()]
                if valid_texts:
                    batch_results['reviews_text']['texts'].append(' '.join(valid_texts))
                    batch_results['reviews_text']['chunk_indices'].append(chunk_idx)

    def _assign_embeddings_to_chunks(self, field_type: str, chunk_indices: List[int], embeddings: List[List[float]]):
        """Assign embeddings back to chunks efficiently"""
        embedding_attr_map = {
            'content': 'content_embedding',
            'title': 'title_embedding',
            'feature': 'feature_embedding',
            'detail': 'detail_embedding',
            'description': 'description_embedding',
            'reviews_summary': 'reviews_summary_embedding',
            'reviews_text': 'reviews_text_embedding'
        }
        
        embedding_attr = embedding_attr_map[field_type]
        
        for chunk_idx, embedding in zip(chunk_indices, embeddings):
            setattr(self.chunks[chunk_idx], embedding_attr, embedding)

    def _assign_zero_vectors_to_missing_embeddings(self):
        """Assign zero vectors to chunks without embeddings"""
        zero_vector = [0.0] * 384  # Default embedding dimension
        
        embedding_attrs = [
            'content_embedding', 'title_embedding', 'feature_embedding',
            'detail_embedding', 'description_embedding', 'reviews_summary_embedding',
            'reviews_text_embedding'
        ]
        
        for chunk in self.chunks:
            for attr in embedding_attrs:
                if getattr(chunk, attr, None) is None:
                    setattr(chunk, attr, zero_vector)

    def _convert_knn_results_to_similarities_vectorized(self, similarities: np.ndarray, 
                                                       indices: np.ndarray, 
                                                       chunk_id_to_embedding_idx: Dict[str, int]) -> List[STARKSimilarityMetrics]:
        """Convert k-NN results to similarity metrics using vectorized operations"""
        similarity_list = []
        
        # Create reverse mapping
        embedding_idx_to_chunk = {idx: chunk_id for chunk_id, idx in chunk_id_to_embedding_idx.items()}
        
        # Process all similarity pairs
        for query_idx in range(len(similarities)):
            query_chunk_idx = self.faiss_to_chunk_mapping[query_idx]
            query_chunk = self.chunks[query_chunk_idx]
            
            # Process all neighbors for this query
            for neighbor_rank, (sim_score, neighbor_faiss_idx) in enumerate(zip(similarities[query_idx], indices[query_idx])):
                if neighbor_faiss_idx == -1:  # Invalid index
                    continue
                
                neighbor_chunk_idx = self.faiss_to_chunk_mapping[neighbor_faiss_idx]
                neighbor_chunk = self.chunks[neighbor_chunk_idx]
                
                # Skip self-matches
                if query_chunk.chunk_id == neighbor_chunk.chunk_id:
                    continue
                
                # Only calculate same-type similarities
                if query_chunk.chunk_type != neighbor_chunk.chunk_type:
                    continue
                
                # Create similarity metric
                similarity_metric = self._create_similarity_metric_fast(query_chunk, neighbor_chunk, float(sim_score))
                if similarity_metric:
                    similarity_list.append(similarity_metric)
        
        return similarity_list

    def _create_similarity_metric_fast(self, chunk1: STARKChunkData, chunk2: STARKChunkData, 
                                     content_similarity: float) -> Optional[STARKSimilarityMetrics]:
        """Create similarity metric efficiently"""
        try:
            edge_type = "doc-doc" if chunk1.chunk_type == "document" else "table-table"
            entity_count = self._calculate_entity_matches(chunk1.entities, chunk2.entities)
            
            # Field-specific similarities (vectorized)
            additional_metrics = {}
            
            if edge_type == "doc-doc":
                additional_metrics.update({
                    'title_similarity': self._cosine_similarity_fast(chunk1.title_embedding, chunk2.title_embedding),
                    'feature_similarity': self._cosine_similarity_fast(chunk1.feature_embedding, chunk2.feature_embedding),
                    'detail_similarity': self._cosine_similarity_fast(chunk1.detail_embedding, chunk2.detail_embedding),
                    'description_similarity': self._cosine_similarity_fast(chunk1.description_embedding, chunk2.description_embedding),
                    'reviews_summary_similarity': 0.0,
                    'reviews_text_similarity': 0.0
                })
            else:  # table-table
                additional_metrics.update({
                    'reviews_summary_similarity': self._cosine_similarity_fast(chunk1.reviews_summary_embedding, chunk2.reviews_summary_embedding),
                    'reviews_text_similarity': self._cosine_similarity_fast(chunk1.reviews_text_embedding, chunk2.reviews_text_embedding),
                    'title_similarity': 0.0,
                    'feature_similarity': 0.0,
                    'detail_similarity': 0.0,
                    'description_similarity': 0.0
                })
            
            return STARKSimilarityMetrics(
                content_similarity=max(0.0, content_similarity),
                edge_type=edge_type,
                source_chunk_id=chunk1.chunk_id,
                target_chunk_id=chunk2.chunk_id,
                entity_count=entity_count,
                **additional_metrics
            )
            
                except Exception as e:
            logger.warning(f"Error creating similarity metric: {e}")
            return None

    def _cosine_similarity_fast(self, emb1: List[float], emb2: List[float]) -> float:
        """Fast cosine similarity calculation"""
        if not emb1 or not emb2:
            return 0.0
        
        try:
            # Use numpy for faster computation
            emb1_np = np.array(emb1, dtype=np.float32)
            emb2_np = np.array(emb2, dtype=np.float32)
            
            # Check for zero vectors
            if np.allclose(emb1_np, 0) or np.allclose(emb2_np, 0):
                return 0.0
            
            # Normalized dot product
            similarity = np.dot(emb1_np, emb2_np) / (np.linalg.norm(emb1_np) * np.linalg.norm(emb2_np))
            return max(0.0, float(similarity))
            
        except Exception:
            return 0.0

    # Include all other helper methods from original implementation
    def _create_product_document_chunk(self, product_id: str, product_data: Dict[str, Any]) -> Optional[STARKChunkData]:
        """Create document chunk containing product information"""
        try:
            asin = product_data.get('asin', product_id)
            title = product_data.get('title', '')
            brand = product_data.get('brand', 'Unknown')
            global_category = product_data.get('global_category', '')
            category = product_data.get('category', [])
            feature = product_data.get('feature', [])
            description = product_data.get('description', [])
            details = product_data.get('details', '')
            
            # Handle NaN values
            if isinstance(details, float) and np.isnan(details):
                details = ''
            
            # Extract entities
            entities = self._extract_entities(title, feature, description, brand)
            
            chunk_data = STARKChunkData(
                chunk_id=f"product_doc_{asin}_{product_id}",
                chunk_type="document",
                content="",  # Will be computed in stage 2
                entities=entities,
                asin=asin,
                title=title,
                brand=brand,
                global_category=global_category,
                category=category,
                feature=feature,
                detail=str(details),
                description=description,
                metadata={
                    'original_product_id': product_id,
                    'price': product_data.get('price', ''),
                    'rank': product_data.get('rank', '')
                }
            )
            
            return chunk_data
            
        except Exception as e:
            logger.warning(f"Error creating product document chunk for {product_id}: {e}")
            return None

    def _create_product_reviews_table_chunk(self, product_id: str, product_data: Dict[str, Any]) -> Optional[STARKChunkData]:
        """Create table chunk containing ALL reviews for a product"""
        try:
            asin = product_data.get('asin', product_id)
            reviews = product_data.get('review', [])
            
            if not reviews:
                return None
            
            # Extract product-level entities
            brand = product_data.get('brand', 'Unknown')
            title = product_data.get('title', '')
            feature = product_data.get('feature', [])
            description = product_data.get('description', [])
            product_entities = self._extract_entities(title, feature, description, brand)
            
            # Create DataFrame
            reviews_data = []
            for review_idx, review in enumerate(reviews):
                try:
                    summary = review.get('summary', '')
                    style = review.get('style', '')
                    review_text = review.get('reviewText', '')
                    
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
            
            reviews_df = pd.DataFrame(reviews_data)
            
            chunk_data = STARKChunkData(
                chunk_id=f"reviews_table_{asin}_{product_id}",
                chunk_type="table",
                content="",  # Will be computed in stage 2
                entities=product_entities,
                asin=asin,
                reviews_data=reviews_df,
                review_count=len(reviews_df),
                metadata={
                    'product_asin': asin,
                    'original_product_id': product_id,
                    'total_reviews': len(reviews_df)
                }
            )
            
            return chunk_data
            
        except Exception as e:
            logger.warning(f"Error creating reviews table chunk for {product_id}: {e}")
            return None

    def _extract_entities(self, title: str, features: List[str], descriptions: List[str], brand: str) -> List[str]:
        """Extract brand and color entities from product text"""
        entities = []
        
        # Add brand
        if brand and brand.lower() != 'unknown':
            entities.append(brand)
        
        # Color detection
        colors = {
            'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black', 'white',
            'gray', 'grey', 'navy', 'turquoise', 'teal', 'maroon', 'silver', 'gold', 'beige', 'tan',
            'khaki', 'olive', 'burgundy', 'coral', 'salmon', 'magenta', 'cyan', 'lime', 'indigo', 'violet'
        }
        
        all_text = ' '.join([str(title), str(features), str(descriptions)]).lower()
        
        for color in colors:
            if color in all_text:
                entities.append(color)
        
        return entities

    def _build_chunk_index(self):
        """Build chunk ID to index mapping"""
        self.chunk_index = {chunk.chunk_id: i for i, chunk in enumerate(self.chunks)}

    def _calculate_entity_matches(self, entities1: List[str], entities2: List[str]) -> int:
        """Calculate number of matching entities"""
        if not entities1 or not entities2:
            return 0
        
        matches = 0
        for entity1 in entities1:
            for entity2 in entities2:
                if entity1.lower().strip() == entity2.lower().strip():
                    matches += 1
                    break
        
        return matches

    def _generate_analysis_reports(self):
        """Generate analysis reports from similarity data"""
        logger.info("   Generating analysis reports...")
        
        # Convert to DataFrame
        df = self._create_similarity_dataframe()
        
        # Analyze outliers and store edges
        self.outlier_edges = []
        self._analyze_outliers(df)
        
        logger.info(f"   Found {len(self.outlier_edges):,} outlier edges")

    def _create_similarity_dataframe(self) -> pd.DataFrame:
        """Convert similarity data to DataFrame"""
        data = []
        
        for sim in self.similarity_data:
            data.append({
                'content_similarity': sim.content_similarity,
                'edge_type': sim.edge_type,
                'source_chunk_id': sim.source_chunk_id,
                'target_chunk_id': sim.target_chunk_id,
                'entity_count': sim.entity_count,
                'title_similarity': sim.title_similarity,
                'feature_similarity': sim.feature_similarity,
                'detail_similarity': sim.detail_similarity,
                'description_similarity': sim.description_similarity,
                'reviews_summary_similarity': sim.reviews_summary_similarity,
                'reviews_text_similarity': sim.reviews_text_similarity
            })
        
        return pd.DataFrame(data)

    def _analyze_outliers(self, df: pd.DataFrame):
        """Analyze outliers for both edge types"""
        # Doc-doc outliers
        doc_doc_df = df[df['edge_type'] == 'doc-doc'].copy()
        if len(doc_doc_df) > 0:
        threshold_content = doc_doc_df['content_similarity'].quantile(0.95)
        threshold_title = doc_doc_df['title_similarity'].quantile(0.95)
        threshold_feature = doc_doc_df['feature_similarity'].quantile(0.95)

            high_sim_mask = (doc_doc_df['content_similarity'] > threshold_content) & \
                     (doc_doc_df['title_similarity'] > threshold_title) & \
                           (doc_doc_df['feature_similarity'] > threshold_feature)
            high_sim_outliers = doc_doc_df[high_sim_mask].copy()
            
            entity_match_outliers = doc_doc_df[doc_doc_df['entity_count'] > 0].copy()
            
            self._store_outliers_for_graph_building(high_sim_outliers, 'doc-doc', 'all_product_features_high')
            self._store_outliers_for_graph_building(entity_match_outliers, 'doc-doc', 'entity_match')
        
        # Table-table outliers
        table_table_df = df[df['edge_type'] == 'table-table'].copy()
        if len(table_table_df) > 0:
        threshold_content = table_table_df['content_similarity'].quantile(0.95)
            threshold_summary = table_table_df['reviews_summary_similarity'].quantile(0.95)
            threshold_text = table_table_df['reviews_text_similarity'].quantile(0.95)
            
            high_sim_mask = (table_table_df['content_similarity'] > threshold_content) & \
                           (table_table_df['reviews_summary_similarity'] > threshold_summary) & \
                           (table_table_df['reviews_text_similarity'] > threshold_text)
            high_sim_outliers = table_table_df[high_sim_mask].copy()
            
            entity_match_outliers = table_table_df[table_table_df['entity_count'] > 0].copy()
            
            self._store_outliers_for_graph_building(high_sim_outliers, 'table-table', 'all_reviews_features_high')
            self._store_outliers_for_graph_building(entity_match_outliers, 'table-table', 'entity_match')

    def _store_outliers_for_graph_building(self, outliers_df: pd.DataFrame, edge_type: str, reason: str):
        """Store outliers for graph building"""
        edge_ids_seen = set()
        
        for _, row in outliers_df.iterrows():
            source_id = row['source_chunk_id']
            target_id = row['target_chunk_id']
            edge_id = f"{source_id}_{target_id}"
            
            if edge_id in edge_ids_seen:
                continue
            edge_ids_seen.add(edge_id)
            
            edge_data = {
                'edge_id': edge_id,
                'source_chunk_id': source_id,
                'target_chunk_id': target_id,
                'edge_type': edge_type,
                'reason': reason,
                'semantic_similarity': row['content_similarity'],
                'entity_count': row['entity_count']
            }
            
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

    def _extract_and_save_unique_edges(self):
        """Extract and save unique edges"""
        logger.info("   Extracting and saving unique edges...")
        
        if not self.outlier_edges:
            logger.warning("   No outlier edges found!")
            return None
        
        # Remove duplicates
        unique_edges = {}
        for edge in self.outlier_edges:
            edge_id = edge['edge_id']
            if edge_id not in unique_edges:
                unique_edges[edge_id] = edge
        
        edges_to_save = list(unique_edges.values())
        
        # Filter to valid chunks
        if self.chunk_index:
            valid_ids = set(self.chunk_index.keys())
            edges_to_save = [
                e for e in edges_to_save
                if e.get('source_chunk_id') in valid_ids and e.get('target_chunk_id') in valid_ids
            ]
        
        # Save edges
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        edges_file = self.cache_dir / f"stark_ultrafast_edges_{timestamp}.json"
        
        with open(edges_file, 'w', encoding='utf-8') as f:
            json.dump(edges_to_save, f, indent=2, ensure_ascii=False)
        
        logger.info(f"   Saved {len(edges_to_save):,} unique edges to {edges_file}")
        
        return edges_file

    def _clear_gpu_memory(self):
        """Clear GPU memory"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
        gc.collect()

def main():
    """Main execution function"""
    logger.info("🚀 Ultra-Fast STARK Pipeline - Multi-GPU Accelerated")
    logger.info("💡 Expected 100x+ speedup through optimized bulk processing")
    logger.info("=" * 60)
    
    pipeline = UltraFastSTARKPipeline()
    
    try:
        edges_file = pipeline.run_full_pipeline()
        
        logger.info("🎉 Ultra-Fast Pipeline completed successfully!")
        logger.info(f"📁 Final results: {edges_file}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
