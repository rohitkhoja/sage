#!/usr/bin/env python3
"""
RESUMABLE STARK Dataset Graph Analysis Pipeline
Each stage is independently cached and resumable
GPU tasks are properly isolated with memory management
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    chunk_type: str # "document" or "table"
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

class STARKResumablePipeline:
    """
    Resumable STARK pipeline with stage-based caching
    Each GPU task is isolated and can be resumed from any point
    """
    
    def __init__(self, 
                 stark_dataset_file: str = "/shared/khoja/CogComp/datasets/STARK/node_info.json",
                 cache_dir: str = "/shared/khoja/CogComp/output/stark_resumable_cache",
                 use_gpu: bool = True,
                 num_threads: int = 64,
                 gpu_batch_size: int = 100000): # Chunks per GPU
        
        self.stark_dataset_file = Path(stark_dataset_file)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.num_threads = num_threads
        self.gpu_batch_size = gpu_batch_size
        
        # Initialize embedding service
        self.config = ProcessingConfig(use_faiss=True, faiss_use_gpu=self.use_gpu)
        self.embedding_service = EmbeddingService(self.config)
        
        # Data containers
        self.chunks: List[STARKChunkData] = []
        self.chunk_index: Dict[str, int] = {}
        self.faiss_index = None
        self.faiss_to_chunk_mapping = []
        self.similarity_data: List[STARKSimilarityMetrics] = []
        self.outlier_edges = []
        
        # GPU configuration
        self.num_gpus = torch.cuda.device_count() if self.use_gpu else 0
        
        # Stage cache files
        self.stage_files = {
            'chunks': self.cache_dir / 'stage1_chunks.pkl',
            'field_embeddings': self.cache_dir / 'stage2_field_embeddings.pkl',
            'content_embeddings': self.cache_dir / 'stage3_content_embeddings.pkl',
            'hnsw_index': self.cache_dir / 'stage4_hnsw_index.faiss',
            'hnsw_mapping': self.cache_dir / 'stage4_hnsw_mapping.pkl',
            'similarities': self.cache_dir / 'stage5_similarities.pkl',
            'analysis_complete': self.cache_dir / 'stage6_analysis_complete.flag'
        }
        
        logger.info(f"Initialized STARK Resumable Pipeline with {self.num_gpus} GPUs")
        logger.info(f"GPU batch size: {gpu_batch_size} chunks per GPU")

    def run_full_pipeline(self, max_products: Optional[int] = None, k_neighbors: int = 200):
        """Run the complete pipeline with automatic resume from any stage"""
        logger.info(" Starting STARK Resumable Pipeline")
        logger.info("=" * 60)
        
        try:
            # Stage 1: Load Chunks
            self.stage1_load_chunks(max_products)
            
            # Stage 2: Calculate Field Embeddings (GPU Task 1)
            self.stage2_calculate_field_embeddings()
            
            # Stage 3: Calculate Content Embeddings (GPU Task 2) 
            self.stage3_calculate_content_embeddings()
            
            # Stage 4: Build HNSW Index (GPU Task 3)
            self.stage4_build_hnsw_index()
            
            # Stage 5: Calculate Similarities (GPU Task 4)
            self.stage5_calculate_similarities(k_neighbors)
            
            # Stage 6: Generate Analysis & Reports (CPU Task)
            edges_file = self.stage6_generate_analysis()
            
            logger.info(" STARK Resumable Pipeline completed successfully!")
            logger.info(f" Final results: {edges_file}")
            
            return edges_file
            
        except Exception as e:
            logger.error(f" Pipeline failed at current stage: {e}")
            raise

    def stage1_load_chunks(self, max_products: Optional[int] = None):
        """Stage 1: Load and cache STARK chunks"""
        if self.stage_files['chunks'].exists():
            logger.info(" Stage 1: Loading chunks from cache...")
            with open(self.stage_files['chunks'], 'rb') as f:
                self.chunks = pickle.load(f)
            self._build_chunk_index()
            logger.info(f" Loaded {len(self.chunks)} chunks from cache")
            return
        
        logger.info(" Stage 1: Loading STARK chunks from dataset...")
        
        # Load raw data
        with open(self.stark_dataset_file, 'r') as f:
            stark_data = json.load(f)
        
        logger.info(f" Loaded {len(stark_data)} products from dataset")
        
        # Limit products if specified
        if max_products:
            product_keys = list(stark_data.keys())[:max_products]
            stark_data = {k: stark_data[k] for k in product_keys}
            logger.info(f" Limited to {max_products} products")
        
        # Process products into chunks
        self.chunks = []
        for product_id, product_data in stark_data.items():
            try:
                # Create document chunk (product info)
                doc_chunk = self._create_product_document_chunk(product_id, product_data)
                if doc_chunk:
                    self.chunks.append(doc_chunk)
                
                # Create table chunk (reviews)
                table_chunk = self._create_product_reviews_table_chunk(product_id, product_data)
                if table_chunk:
                    self.chunks.append(table_chunk)
                    
            except Exception as e:
                logger.warning(f" Failed to process product {product_id}: {e}")
                continue
        
        # Build index and cache
        self._build_chunk_index()
        
        with open(self.stage_files['chunks'], 'wb') as f:
            pickle.dump(self.chunks, f)
        
        doc_chunks = [c for c in self.chunks if c.chunk_type == "document"]
        table_chunks = [c for c in self.chunks if c.chunk_type == "table"]
        
        logger.info(f" Stage 1 Complete: {len(doc_chunks)} document + {len(table_chunks)} table chunks")

    def stage2_calculate_field_embeddings(self):
        """Stage 2: Calculate individual field embeddings with GPU memory management"""
        if self.stage_files['field_embeddings'].exists():
            logger.info(" Stage 2: Loading field embeddings from cache...")
            self._load_field_embeddings_from_cache()
            return
        
        logger.info(" Stage 2: Calculating field embeddings with GPU batching...")
        
        # Load chunks if not already loaded
        if not self.chunks:
            self.stage1_load_chunks()
        
        # Process in GPU batches to manage memory
        total_chunks = len(self.chunks)
        chunks_per_gpu = self.gpu_batch_size
        
        for batch_start in range(0, total_chunks, chunks_per_gpu * self.num_gpus):
            batch_end = min(batch_start + chunks_per_gpu * self.num_gpus, total_chunks)
            batch_chunks = self.chunks[batch_start:batch_end]
            
            logger.info(f" Processing batch {batch_start//chunks_per_gpu + 1}: chunks {batch_start}-{batch_end}")
            
            # Calculate embeddings for this batch
            self._calculate_field_embeddings_batch(batch_chunks)
            
            # Clear GPU memory
            self._clear_gpu_memory()
            
            # Save progress
            self._save_field_embeddings_progress()
        
        # Save final embeddings
        with open(self.stage_files['field_embeddings'], 'wb') as f:
            pickle.dump(self.chunks, f)
        
        logger.info(" Stage 2 Complete: All field embeddings calculated and cached")

    def stage3_calculate_content_embeddings(self):
        """Stage 3: Calculate content embeddings with GPU memory management"""
        if self.stage_files['content_embeddings'].exists():
            logger.info(" Stage 3: Loading content embeddings from cache...")
            self._load_content_embeddings_from_cache()
            return
        
        logger.info(" Stage 3: Calculating content embeddings with GPU batching...")
        
        # Load chunks with field embeddings if not loaded
        if not self.chunks or not self.chunks[0].title_embedding:
            self.stage2_calculate_field_embeddings()
        
        # Prepare content texts
        content_texts = []
        chunk_indices = []
        
        for i, chunk in enumerate(self.chunks):
            if chunk.chunk_type == "document":
                # Document content: title + global_category + category + brand + feature + details + description
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
                
                content = ' '.join([p.strip() for p in content_parts if p and str(p).strip()])
                
            else: # table
                # Table content: summary + reviewText (no style)
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
                
                content = ' '.join(content_parts)
            
            if content.strip():
                content_texts.append(content.strip())
                chunk_indices.append(i)
        
        # Process in GPU batches
        batch_size = 256 # Embedding batch size
        total_batches = (len(content_texts) + batch_size - 1) // batch_size
        
        all_embeddings = []
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(content_texts))
            batch_texts = content_texts[start_idx:end_idx]
            
            logger.info(f" Processing embedding batch {batch_idx + 1}/{total_batches}")
            
            # Generate embeddings
            batch_embeddings = self.embedding_service.generate_embeddings(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            # Clear GPU memory periodically
            if (batch_idx + 1) % 10 == 0:
                self._clear_gpu_memory()
        
        # Assign embeddings back to chunks
        for chunk_idx, embedding in zip(chunk_indices, all_embeddings):
            self.chunks[chunk_idx].content_embedding = embedding
        
        # Assign zero vectors to chunks without content
        zero_vector = [0.0] * 384 # Default embedding dimension
        for chunk in self.chunks:
            if chunk.content_embedding is None:
                chunk.content_embedding = zero_vector
        
        # Save content embeddings
        with open(self.stage_files['content_embeddings'], 'wb') as f:
            pickle.dump(self.chunks, f)
        
        logger.info(" Stage 3 Complete: All content embeddings calculated and cached")

    def stage4_build_hnsw_index(self):
        """Stage 4: Build HNSW index using content embeddings"""
        if self.stage_files['hnsw_index'].exists() and self.stage_files['hnsw_mapping'].exists():
            logger.info(" Stage 4: Loading HNSW index from cache...")
            self.faiss_index = faiss.read_index(str(self.stage_files['hnsw_index']))
            with open(self.stage_files['hnsw_mapping'], 'rb') as f:
                self.faiss_to_chunk_mapping = pickle.load(f)
            logger.info(f" Loaded HNSW index with {self.faiss_index.ntotal} embeddings")
            return
        
        logger.info(" Stage 4: Building HNSW index...")
        
        # Load chunks with content embeddings if not loaded
        if not self.chunks or not self.chunks[0].content_embedding:
            self.stage3_calculate_content_embeddings()
        
        # Collect embeddings
        embeddings = []
        valid_chunk_indices = []
        
        for i, chunk in enumerate(self.chunks):
            if chunk.content_embedding and any(x != 0.0 for x in chunk.content_embedding):
                embeddings.append(chunk.content_embedding)
                valid_chunk_indices.append(i)
        
        if not embeddings:
            raise ValueError("No valid embeddings found for HNSW index")
        
        # Convert to numpy array
        embeddings_matrix = np.array(embeddings, dtype=np.float32)
        dimension = embeddings_matrix.shape[1]
        
        logger.info(f" Building HNSW index: {len(embeddings)} embeddings, dimension={dimension}")
        
        # Create and build HNSW index
        self.faiss_index = faiss.IndexHNSWFlat(dimension, 64)
        self.faiss_index.hnsw.efConstruction = 2000
        self.faiss_index.hnsw.efSearch = 1000
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_matrix)
        
        # Use GPU if available
        if self.use_gpu and faiss.get_num_gpus() > 0:
            try:
                logger.info(f" Using {faiss.get_num_gpus()} GPUs for index building")
                gpu_index = faiss.index_cpu_to_all_gpus(self.faiss_index)
                gpu_index.add(embeddings_matrix)
                self.faiss_index = faiss.index_gpu_to_cpu(gpu_index)
                self._clear_gpu_memory()
            except Exception as e:
                logger.warning(f" GPU indexing failed: {e}, using CPU")
                self.faiss_index.add(embeddings_matrix)
        else:
            self.faiss_index.add(embeddings_matrix)
        
        # Store mapping
        self.faiss_to_chunk_mapping = valid_chunk_indices
        
        # Save index and mapping
        faiss.write_index(self.faiss_index, str(self.stage_files['hnsw_index']))
        with open(self.stage_files['hnsw_mapping'], 'wb') as f:
            pickle.dump(self.faiss_to_chunk_mapping, f)
        
        logger.info(" Stage 4 Complete: HNSW index built and cached")

    def stage5_calculate_similarities(self, k_neighbors: int = 200):
        """Stage 5: Calculate similarities with resume capability"""
        if self.stage_files['similarities'].exists():
            logger.info(" Stage 5: Loading similarities from cache...")
            with open(self.stage_files['similarities'], 'rb') as f:
                self.similarity_data = pickle.load(f)
            logger.info(f" Loaded {len(self.similarity_data)} similarities from cache")
            return
        
        logger.info(" Stage 5: Calculating similarities with GPU batching...")
        
        # Load previous stages if not loaded
        if not self.chunks:
            self.stage3_calculate_content_embeddings()
        if not self.faiss_index:
            self.stage4_build_hnsw_index()
        
        # Check for partial progress
        progress_file = self.cache_dir / 'stage5_progress.pkl'
        completed_chunks = set()
        partial_similarities = []
        
        if progress_file.exists():
            logger.info(" Resuming from previous progress...")
            with open(progress_file, 'rb') as f:
                progress_data = pickle.load(f)
                completed_chunks = progress_data.get('completed_chunks', set())
                partial_similarities = progress_data.get('similarities', [])
            logger.info(f" Resuming: {len(completed_chunks)} chunks already processed")
        
        # Find remaining chunks to process
        remaining_chunks = [
            chunk for chunk in self.chunks 
            if chunk.chunk_id not in completed_chunks
        ]
        
        logger.info(f" Processing {len(remaining_chunks)} remaining chunks")
        
        # Process chunks in batches
        batch_size = self.gpu_batch_size // self.num_gpus if self.num_gpus > 0 else 1000
        
        for batch_start in range(0, len(remaining_chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(remaining_chunks))
            batch_chunks = remaining_chunks[batch_start:batch_end]
            
            logger.info(f" Processing batch {batch_start//batch_size + 1}: {len(batch_chunks)} chunks")
            
            # Calculate similarities for this batch
            batch_similarities = self._calculate_similarities_batch(batch_chunks, k_neighbors)
            partial_similarities.extend(batch_similarities)
            
            # Update completed chunks
            for chunk in batch_chunks:
                completed_chunks.add(chunk.chunk_id)
            
            # Save progress
            progress_data = {
                'completed_chunks': completed_chunks,
                'similarities': partial_similarities
            }
            with open(progress_file, 'wb') as f:
                pickle.dump(progress_data, f)
            
            # Clear GPU memory
            self._clear_gpu_memory()
        
        # Save final similarities
        self.similarity_data = partial_similarities
        with open(self.stage_files['similarities'], 'wb') as f:
            pickle.dump(self.similarity_data, f)
        
        # Clean up progress file
        if progress_file.exists():
            progress_file.unlink()
        
        logger.info(" Stage 5 Complete: All similarities calculated and cached")

    def stage6_generate_analysis(self):
        """Stage 6: Generate analysis and extract edges"""
        if self.stage_files['analysis_complete'].exists():
            logger.info(" Stage 6: Analysis already complete, loading results...")
            # Find the most recent edges file
            edges_files = list(self.cache_dir.glob("stark_unique_edges_*.json"))
            if edges_files:
                latest_edges_file = max(edges_files, key=lambda x: x.stat().st_mtime)
                logger.info(f" Found existing results: {latest_edges_file}")
                return latest_edges_file
        
        logger.info(" Stage 6: Generating analysis and extracting edges...")
        
        # Load similarities if not loaded
        if not self.similarity_data:
            self.stage5_calculate_similarities()
        
        # Generate analysis reports
        self._generate_analysis_reports()
        
        # Extract unique edges
        edges_file = self._extract_and_save_unique_edges()
        
        # Mark stage as complete
        with open(self.stage_files['analysis_complete'], 'w') as f:
            f.write(f"Analysis completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logger.info(" Stage 6 Complete: Analysis and edge extraction finished")
        
        return edges_file

    # Helper methods
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
                content="", # Will be computed in stage 3
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
                content="", # Will be computed in stage 3
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
            logger.warning(f"Error creating reviews table chunk for product {product_id}: {e}")
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

    def _calculate_field_embeddings_batch(self, batch_chunks: List[STARKChunkData]):
        """Calculate field embeddings for a batch of chunks"""
        # Collect texts by field type
        field_texts = {
            'title': {'texts': [], 'indices': []},
            'feature': {'texts': [], 'indices': []},
            'detail': {'texts': [], 'indices': []},
            'description': {'texts': [], 'indices': []},
            'reviews_summary': {'texts': [], 'indices': []},
            'reviews_text': {'texts': [], 'indices': []}
        }
        
        # Collect texts
        for i, chunk in enumerate(batch_chunks):
            if chunk.chunk_type == "document":
                if chunk.title and chunk.title.strip():
                    field_texts['title']['texts'].append(chunk.title.strip())
                    field_texts['title']['indices'].append(i)
                
                if chunk.feature:
                    feature_text = ' '.join(chunk.feature) if isinstance(chunk.feature, list) else str(chunk.feature)
                    if feature_text.strip():
                        field_texts['feature']['texts'].append(feature_text.strip())
                        field_texts['feature']['indices'].append(i)
                
                if chunk.detail and chunk.detail.strip():
                    field_texts['detail']['texts'].append(chunk.detail.strip())
                    field_texts['detail']['indices'].append(i)
                
                if chunk.description:
                    desc_text = ' '.join(chunk.description) if isinstance(chunk.description, list) else str(chunk.description)
                    if desc_text.strip():
                        field_texts['description']['texts'].append(desc_text.strip())
                        field_texts['description']['indices'].append(i)
            
            elif chunk.chunk_type == "table":
                if chunk.reviews_data is not None:
                    if 'summary' in chunk.reviews_data.columns:
                        summaries = chunk.reviews_data['summary'].fillna('').astype(str)
                        valid_summaries = [s.strip() for s in summaries if s.strip()]
                        if valid_summaries:
                            field_texts['reviews_summary']['texts'].append(' '.join(valid_summaries))
                            field_texts['reviews_summary']['indices'].append(i)
                    
                    if 'reviewText' in chunk.reviews_data.columns:
                        review_texts = chunk.reviews_data['reviewText'].fillna('').astype(str)
                        valid_texts = [r.strip() for r in review_texts if r.strip()]
                        if valid_texts:
                            field_texts['reviews_text']['texts'].append(' '.join(valid_texts))
                            field_texts['reviews_text']['indices'].append(i)
        
        # Generate embeddings for each field type
        embedding_mapping = {
            'title': 'title_embedding',
            'feature': 'feature_embedding',
            'detail': 'detail_embedding',
            'description': 'description_embedding',
            'reviews_summary': 'reviews_summary_embedding',
            'reviews_text': 'reviews_text_embedding'
        }
        
        zero_vector = [0.0] * 384
        
        for field_type, embedding_attr in embedding_mapping.items():
            field_data = field_texts[field_type]
            
            if field_data['texts']:
                # Generate embeddings
                embeddings = self.embedding_service.generate_embeddings(field_data['texts'])
                
                # Assign to chunks
                for chunk_idx, embedding in zip(field_data['indices'], embeddings):
                    chunk = batch_chunks[chunk_idx]
                    # Find the actual chunk in self.chunks
                    actual_chunk_idx = self.chunk_index[chunk.chunk_id]
                    setattr(self.chunks[actual_chunk_idx], embedding_attr, embedding)
            
            # Assign zero vectors to chunks without this field
            for chunk in batch_chunks:
                actual_chunk_idx = self.chunk_index[chunk.chunk_id]
                actual_chunk = self.chunks[actual_chunk_idx]
                if getattr(actual_chunk, embedding_attr, None) is None:
                    setattr(actual_chunk, embedding_attr, zero_vector)

    def _clear_gpu_memory(self):
        """Clear GPU memory"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
        gc.collect()

    def _save_field_embeddings_progress(self):
        """Save field embeddings progress"""
        progress_file = self.cache_dir / 'stage2_progress.pkl'
        with open(progress_file, 'wb') as f:
            pickle.dump(self.chunks, f)

    def _load_field_embeddings_from_cache(self):
        """Load field embeddings from cache"""
        with open(self.stage_files['field_embeddings'], 'rb') as f:
            self.chunks = pickle.load(f)
        self._build_chunk_index()

    def _load_content_embeddings_from_cache(self):
        """Load content embeddings from cache"""
        with open(self.stage_files['content_embeddings'], 'rb') as f:
            self.chunks = pickle.load(f)
        self._build_chunk_index()

    def _calculate_similarities_batch(self, batch_chunks: List[STARKChunkData], k_neighbors: int) -> List[STARKSimilarityMetrics]:
        """Calculate similarities for a batch of chunks"""
        batch_similarities = []
        
        for chunk in batch_chunks:
            try:
                # Find neighbors using FAISS
                neighbor_chunk_ids = self._find_k_neighbor_ids(chunk, k_neighbors)
                
                # Calculate similarities with neighbors
                for neighbor_id in neighbor_chunk_ids:
                    if neighbor_id == chunk.chunk_id:
                        continue
                    
                    neighbor_idx = self.chunk_index.get(neighbor_id)
                    if neighbor_idx is None:
                        continue
                    
                    neighbor_chunk = self.chunks[neighbor_idx]
                    
                    # Only calculate same-type similarities
                    if chunk.chunk_type != neighbor_chunk.chunk_type:
                        continue
                    
                    # Calculate similarity metrics
                    similarity = self._calculate_chunk_similarity(chunk, neighbor_chunk)
                    if similarity:
                        batch_similarities.append(similarity)
                        
            except Exception as e:
                logger.warning(f"Error calculating similarities for chunk {chunk.chunk_id}: {e}")
                continue
        
        return batch_similarities

    def _find_k_neighbor_ids(self, chunk: STARKChunkData, k: int) -> List[str]:
        """Find k nearest neighbor chunk IDs using FAISS"""
        if not chunk.content_embedding:
            return []
        
        try:
            query_embedding = np.array([chunk.content_embedding], dtype=np.float32)
            faiss.normalize_L2(query_embedding)
            
            search_k = min(k + 1, self.faiss_index.ntotal)
            similarities, indices = self.faiss_index.search(query_embedding, search_k)
            
            neighbor_chunk_ids = []
            for sim, idx in zip(similarities[0], indices[0]):
                if idx == -1:
                    continue
                
                chunk_idx = self.faiss_to_chunk_mapping[idx]
                neighbor_chunk = self.chunks[chunk_idx]
                
                if neighbor_chunk.chunk_id == chunk.chunk_id:
                    continue
                
                neighbor_chunk_ids.append(neighbor_chunk.chunk_id)
            
            return neighbor_chunk_ids[:k]
            
        except Exception as e:
            logger.warning(f"Error finding neighbors for {chunk.chunk_id}: {e}")
            return []

    def _calculate_chunk_similarity(self, chunk1: STARKChunkData, chunk2: STARKChunkData) -> Optional[STARKSimilarityMetrics]:
        """Calculate similarity between two chunks"""
        try:
            # Content similarity
            content_sim = self._cosine_similarity(chunk1.content_embedding, chunk2.content_embedding)
            
            # Edge type
            edge_type = "doc-doc" if chunk1.chunk_type == "document" else "table-table"
            
            # Entity matching
            entity_count = self._calculate_entity_matches(chunk1.entities, chunk2.entities)
            
            # Field-specific similarities
            additional_metrics = {}
            
            if edge_type == "doc-doc":
                additional_metrics.update({
                    'title_similarity': self._cosine_similarity(chunk1.title_embedding, chunk2.title_embedding),
                    'feature_similarity': self._cosine_similarity(chunk1.feature_embedding, chunk2.feature_embedding),
                    'detail_similarity': self._cosine_similarity(chunk1.detail_embedding, chunk2.detail_embedding),
                    'description_similarity': self._cosine_similarity(chunk1.description_embedding, chunk2.description_embedding),
                    'reviews_summary_similarity': 0.0,
                    'reviews_text_similarity': 0.0
                })
            else: # table-table
                additional_metrics.update({
                    'reviews_summary_similarity': self._cosine_similarity(chunk1.reviews_summary_embedding, chunk2.reviews_summary_embedding),
                    'reviews_text_similarity': self._cosine_similarity(chunk1.reviews_text_embedding, chunk2.reviews_text_embedding),
                    'title_similarity': 0.0,
                    'feature_similarity': 0.0,
                    'detail_similarity': 0.0,
                    'description_similarity': 0.0
                })
            
            return STARKSimilarityMetrics(
                content_similarity=max(0.0, content_sim),
                edge_type=edge_type,
                source_chunk_id=chunk1.chunk_id,
                target_chunk_id=chunk2.chunk_id,
                entity_count=entity_count,
                **additional_metrics
            )
            
        except Exception as e:
            logger.warning(f"Error calculating similarity between {chunk1.chunk_id} and {chunk2.chunk_id}: {e}")
            return None

    def _cosine_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        if not emb1 or not emb2:
            return 0.0
        
        try:
            emb1_np = np.array(emb1)
            emb2_np = np.array(emb2)
            
            # Check if either embedding is all zeros
            if np.all(emb1_np == 0) or np.all(emb2_np == 0):
                return 0.0
            
            # Calculate cosine similarity
            dot_product = np.dot(emb1_np, emb2_np)
            norm1 = np.linalg.norm(emb1_np)
            norm2 = np.linalg.norm(emb2_np)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, float(similarity))
            
        except Exception as e:
            logger.warning(f"Error calculating cosine similarity: {e}")
            return 0.0

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
        logger.info(" Generating analysis reports...")
        
        # Convert to DataFrame
        df = self._create_similarity_dataframe()
        
        # Analyze outliers and store edges
        self.outlier_edges = []
        self._analyze_outliers(df)
        
        logger.info(f" Found {len(self.outlier_edges)} outlier edges")

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
            # High similarity outliers
            threshold_content = doc_doc_df['content_similarity'].quantile(0.95)
            threshold_title = doc_doc_df['title_similarity'].quantile(0.95)
            threshold_feature = doc_doc_df['feature_similarity'].quantile(0.95)
            
            high_sim_mask = (doc_doc_df['content_similarity'] > threshold_content) & \
                           (doc_doc_df['title_similarity'] > threshold_title) & \
                           (doc_doc_df['feature_similarity'] > threshold_feature)
            high_sim_outliers = doc_doc_df[high_sim_mask].copy()
            
            # Entity match outliers
            entity_match_outliers = doc_doc_df[doc_doc_df['entity_count'] > 0].copy()
            
            # Store outliers
            self._store_outliers_for_graph_building(high_sim_outliers, 'doc-doc', 'all_product_features_high')
            self._store_outliers_for_graph_building(entity_match_outliers, 'doc-doc', 'entity_match')
        
        # Table-table outliers
        table_table_df = df[df['edge_type'] == 'table-table'].copy()
        if len(table_table_df) > 0:
            # High similarity outliers
            threshold_content = table_table_df['content_similarity'].quantile(0.95)
            threshold_summary = table_table_df['reviews_summary_similarity'].quantile(0.95)
            threshold_text = table_table_df['reviews_text_similarity'].quantile(0.95)
            
            high_sim_mask = (table_table_df['content_similarity'] > threshold_content) & \
                           (table_table_df['reviews_summary_similarity'] > threshold_summary) & \
                           (table_table_df['reviews_text_similarity'] > threshold_text)
            high_sim_outliers = table_table_df[high_sim_mask].copy()
            
            # Entity match outliers
            entity_match_outliers = table_table_df[table_table_df['entity_count'] > 0].copy()
            
            # Store outliers
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
        logger.info(" Extracting and saving unique edges...")
        
        if not self.outlier_edges:
            logger.warning(" No outlier edges found!")
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
        edges_file = self.cache_dir / f"stark_unique_edges_{timestamp}.json"
        
        with open(edges_file, 'w', encoding='utf-8') as f:
            json.dump(edges_to_save, f, indent=2, ensure_ascii=False)
        
        logger.info(f" Saved {len(edges_to_save)} unique edges to {edges_file}")
        
        return edges_file

def main():
    """Main execution function"""
    logger.info(" STARK Resumable Pipeline - Multi-GPU Accelerated")
    logger.info("=" * 60)
    
    pipeline = STARKResumablePipeline()
    
    try:
        edges_file = pipeline.run_full_pipeline()
        
        logger.info(" Pipeline completed successfully!")
        logger.info(f" Final results: {edges_file}")
        
    except Exception as e:
        logger.error(f" Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
