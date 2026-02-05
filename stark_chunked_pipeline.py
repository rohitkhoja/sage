#!/usr/bin/env python3
"""
STARK Dataset Chunked Pipeline - Fault-Tolerant Architecture
Processes data in manageable chunks to avoid memory issues and enable resumability
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

@dataclass
class ChunkInfo:
    """Information about a processing chunk"""
    chunk_id: int
    start_idx: int
    end_idx: int
    product_count: int
    status: str  # 'pending', 'processing', 'completed', 'failed'
    embeddings_file: Optional[str] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None

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

class STARKChunkedPipeline:
    """
    Fault-tolerant STARK pipeline that processes data in manageable chunks
    Designed to prevent memory issues and enable complete resumability
    """
    
    def __init__(self, 
                 stark_dataset_file: str = "/shared/khoja/CogComp/datasets/STARK/node_info.json",
                 cache_dir: str = "/shared/khoja/CogComp/output/stark_chunked_cache",
                 chunk_size: int = 50000,  # 50k products per chunk
                 use_gpu: bool = True,
                 num_threads: int = 32):
        
        self.stark_dataset_file = Path(stark_dataset_file)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.chunk_size = chunk_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.num_threads = num_threads
        self.num_gpus = torch.cuda.device_count() if self.use_gpu else 0
        
        # Create subdirectories for organized storage
        self.chunks_dir = self.cache_dir / "chunks"
        self.embeddings_dir = self.cache_dir / "embeddings"
        self.final_dir = self.cache_dir / "final"
        
        for dir_path in [self.chunks_dir, self.embeddings_dir, self.final_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimized embedding service
        self.config = ProcessingConfig(
            use_faiss=True, 
            faiss_use_gpu=self.use_gpu,
            batch_size=4096,  # Large batch size for optimization
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedding_service = EmbeddingService(self.config)
        
        # Pipeline state tracking
        self.chunk_manifest_file = self.cache_dir / "chunk_manifest.json"
        self.chunks_info: List[ChunkInfo] = []
        self.total_products = 0
        
        logger.info(f"🚀 Initialized STARK Chunked Pipeline")
        logger.info(f"   📦 Chunk size: {self.chunk_size:,} products")
        logger.info(f"   💾 Cache directory: {self.cache_dir}")
        logger.info(f"   🔥 GPUs available: {self.num_gpus}")

    def run_chunked_pipeline(self, max_products: Optional[int] = None, k_neighbors: int = 200):
        """Run the complete chunked pipeline"""
        logger.info("🚀 Starting STARK Chunked Pipeline")
        logger.info("=" * 60)
        
        total_start_time = time.time()
        
        try:
            # Phase 1: Analyze and create chunks
            self.phase1_analyze_and_create_chunks(max_products)
            
            # Phase 2: Process each chunk (embeddings)
            self.phase2_process_chunks()
            
            # Phase 3: Merge all embeddings and build HNSW
            self.phase3_merge_and_build_index()
            
            # Phase 4: Calculate similarities
            self.phase4_calculate_similarities(k_neighbors)
            
            # Phase 5: Generate final analysis
            edges_file = self.phase5_generate_analysis()
            
            total_time = time.time() - total_start_time
            
            logger.info("🎉 STARK Chunked Pipeline completed successfully!")
            logger.info(f"⏱️  Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
            logger.info(f"📁 Final results: {edges_file}")
            
            return edges_file
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise

    def phase1_analyze_and_create_chunks(self, max_products: Optional[int] = None):
        """Phase 1: Analyze dataset and create chunk plan"""
        logger.info("🔄 Phase 1: Analyzing dataset and creating chunk plan...")
        start_time = time.time()
        
        # Load dataset metadata
        logger.info("   Loading dataset metadata...")
        with open(self.stark_dataset_file, 'r') as f:
            stark_data = json.load(f)
        
        self.total_products = len(stark_data)
        if max_products:
            self.total_products = min(self.total_products, max_products)
        
        logger.info(f"   Total products to process: {self.total_products:,}")
        
        # Calculate chunks
        total_chunks = math.ceil(self.total_products / self.chunk_size)
        logger.info(f"   Creating {total_chunks} chunks of ~{self.chunk_size:,} products each")
        
        # Create chunk info
        self.chunks_info = []
        for chunk_id in range(total_chunks):
            start_idx = chunk_id * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, self.total_products)
            
            chunk_info = ChunkInfo(
                chunk_id=chunk_id,
                start_idx=start_idx,
                end_idx=end_idx,
                product_count=end_idx - start_idx,
                status='pending'
            )
            self.chunks_info.append(chunk_info)
        
        # Save chunk manifest
        self._save_chunk_manifest()
        
        phase_time = time.time() - start_time
        logger.info(f"✅ Phase 1 Complete: {total_chunks} chunks planned in {phase_time:.2f}s")
        
        # Display chunk summary
        for chunk in self.chunks_info:
            logger.info(f"   Chunk {chunk.chunk_id}: products {chunk.start_idx:,}-{chunk.end_idx-1:,} ({chunk.product_count:,} products)")

    def phase2_process_chunks(self):
        """Phase 2: Process each chunk to generate embeddings"""
        logger.info("🔄 Phase 2: Processing chunks to generate embeddings...")
        start_time = time.time()
        
        # Load existing progress if available
        self._load_chunk_manifest()
        
        # Get chunks that need processing
        pending_chunks = [c for c in self.chunks_info if c.status in ['pending', 'failed']]
        completed_chunks = [c for c in self.chunks_info if c.status == 'completed']
        
        logger.info(f"   Chunks to process: {len(pending_chunks)}")
        logger.info(f"   Already completed: {len(completed_chunks)}")
        
        if not pending_chunks:
            logger.info("   All chunks already processed!")
            return
        
        # Process each chunk
        for chunk_info in pending_chunks:
            self._process_single_chunk(chunk_info)
            self._save_chunk_manifest()  # Save progress after each chunk
        
        phase_time = time.time() - start_time
        successful_chunks = len([c for c in self.chunks_info if c.status == 'completed'])
        logger.info(f"✅ Phase 2 Complete: {successful_chunks}/{len(self.chunks_info)} chunks processed in {phase_time:.2f}s")

    def _process_single_chunk(self, chunk_info: ChunkInfo):
        """Process a single chunk to generate embeddings"""
        logger.info(f"🔥 Processing Chunk {chunk_info.chunk_id}: products {chunk_info.start_idx:,}-{chunk_info.end_idx-1:,}")
        chunk_start_time = time.time()
        
        try:
            chunk_info.status = 'processing'
            
            # 1. Load chunk data
            logger.info(f"   📥 Loading {chunk_info.product_count:,} products...")
            chunk_data = self._load_chunk_data(chunk_info)
            
            if not chunk_data:
                raise ValueError("No valid chunks created from products")
            
            logger.info(f"   📊 Created {len(chunk_data):,} chunks ({sum(1 for c in chunk_data if c.chunk_type == 'document'):,} docs, {sum(1 for c in chunk_data if c.chunk_type == 'table'):,} tables)")
            
            # 2. Generate embeddings
            logger.info(f"   🔥 Generating embeddings...")
            self._generate_chunk_embeddings(chunk_data)
            
            # 3. Save embeddings
            embeddings_file = self.embeddings_dir / f"chunk_{chunk_info.chunk_id:03d}_embeddings.json"
            logger.info(f"   💾 Saving embeddings to {embeddings_file.name}...")
            self._save_chunk_embeddings(chunk_data, embeddings_file)
            
            # 4. Update chunk info
            chunk_info.embeddings_file = str(embeddings_file.name)
            chunk_info.status = 'completed'
            chunk_info.processing_time = time.time() - chunk_start_time
            
            # 5. Clear memory
            del chunk_data
            self._clear_gpu_memory()
            gc.collect()
            
            logger.info(f"   ✅ Chunk {chunk_info.chunk_id} completed in {chunk_info.processing_time:.2f}s")
            
        except Exception as e:
            chunk_info.status = 'failed'
            chunk_info.error_message = str(e)
            chunk_info.processing_time = time.time() - chunk_start_time
            logger.error(f"   ❌ Chunk {chunk_info.chunk_id} failed: {e}")
            
            # Clear memory on failure too
            self._clear_gpu_memory()
            gc.collect()

    def _load_chunk_data(self, chunk_info: ChunkInfo) -> List[STARKChunkData]:
        """Load and process products for a specific chunk"""
        # Load dataset
        with open(self.stark_dataset_file, 'r') as f:
            stark_data = json.load(f)
        
        # Get products for this chunk
        product_items = list(stark_data.items())[chunk_info.start_idx:chunk_info.end_idx]
        
        all_chunks = []
        
        # Process products in smaller batches within the chunk
        batch_size = 5000  # Process 5k products at a time
        for i in range(0, len(product_items), batch_size):
            batch = product_items[i:i+batch_size]
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=min(16, len(batch))) as executor:
                future_to_product = {
                    executor.submit(self._process_product_to_chunks, product_id, product_data): product_id
                    for product_id, product_data in batch
                }
                
                for future in tqdm(as_completed(future_to_product), total=len(future_to_product), desc=f"   Processing batch"):
                    try:
                        chunks = future.result()
                        if chunks:
                            all_chunks.extend(chunks)
                    except Exception as e:
                        product_id = future_to_product[future]
                        logger.warning(f"     Failed to process product {product_id}: {e}")
        
        return all_chunks

    def _generate_chunk_embeddings(self, chunk_data: List[STARKChunkData]):
        """Generate embeddings for a chunk of data"""
        # Prepare all texts
        logger.info("     Preparing texts for embedding...")
        all_text_data = self._prepare_chunk_texts(chunk_data)
        
        # Generate embeddings field by field (memory management)
        for field_type, data in all_text_data.items():
            if not data['texts']:
                continue
                
            logger.info(f"     🔥 Processing {field_type}: {len(data['texts']):,} texts")
            field_start = time.time()
            
            # Clear GPU memory before each field
            if hasattr(self.embedding_service, '_clear_gpu_cache'):
                self.embedding_service._clear_gpu_cache()
            
            try:
                embeddings = self.embedding_service.generate_embeddings_bulk(data['texts'])
                
                field_time = time.time() - field_start
                logger.info(f"     ✅ {field_type} completed in {field_time:.2f}s")
                
                # Assign embeddings back to chunks
                self._assign_embeddings_to_chunks(field_type, data['chunk_indices'], embeddings, chunk_data)
                
                # Clear memory after processing each field
                del embeddings
                if hasattr(self.embedding_service, '_clear_gpu_cache'):
                    self.embedding_service._clear_gpu_cache()
                    
            except Exception as e:
                logger.error(f"     ❌ Failed to process {field_type}: {e}")
                continue

    def _prepare_chunk_texts(self, chunk_data: List[STARKChunkData]) -> Dict[str, Dict[str, List]]:
        """Prepare texts from chunk data"""
        all_text_data = {
            'content': {'texts': [], 'chunk_indices': []},
            'title': {'texts': [], 'chunk_indices': []},
            'feature': {'texts': [], 'chunk_indices': []},
            'detail': {'texts': [], 'chunk_indices': []},
            'description': {'texts': [], 'chunk_indices': []},
            'reviews_summary': {'texts': [], 'chunk_indices': []},
            'reviews_text': {'texts': [], 'chunk_indices': []}
        }
        
        for chunk_idx, chunk in enumerate(chunk_data):
            # Content embedding (main content)
            content = self._build_chunk_content(chunk)
            if content.strip():
                all_text_data['content']['texts'].append(content.strip())
                all_text_data['content']['chunk_indices'].append(chunk_idx)
            
            # Field-specific embeddings
            if chunk.chunk_type == "document":
                self._collect_document_field_texts(chunk, chunk_idx, all_text_data)
            elif chunk.chunk_type == "table":
                self._collect_table_field_texts(chunk, chunk_idx, all_text_data)
        
        return all_text_data

    def _save_chunk_embeddings(self, chunk_data: List[STARKChunkData], embeddings_file: Path):
        """Save chunk embeddings to JSON file"""
        embeddings_data = []
        
        for chunk in chunk_data:
            chunk_embeddings = {
                'chunk_id': chunk.chunk_id,
                'chunk_type': chunk.chunk_type,
                'asin': chunk.asin,
                'content_embedding': chunk.content_embedding,
                'title_embedding': chunk.title_embedding,
                'feature_embedding': chunk.feature_embedding,
                'detail_embedding': chunk.detail_embedding,
                'description_embedding': chunk.description_embedding,
                'reviews_summary_embedding': chunk.reviews_summary_embedding,
                'reviews_text_embedding': chunk.reviews_text_embedding,
                'entities': chunk.entities,
                'metadata': chunk.metadata
            }
            embeddings_data.append(chunk_embeddings)
        
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f, indent=2, ensure_ascii=False)

    def phase3_merge_and_build_index(self):
        """Phase 3: Merge all embeddings and build HNSW index"""
        logger.info("🔄 Phase 3: Merging embeddings and building HNSW index...")
        start_time = time.time()
        
        # Check if already completed
        hnsw_index_file = self.final_dir / 'hnsw_index.faiss'
        hnsw_mapping_file = self.final_dir / 'hnsw_mapping.pkl'
        
        if hnsw_index_file.exists() and hnsw_mapping_file.exists():
            logger.info("   ✅ HNSW index already exists, skipping...")
            return
        
        # Load all chunk embeddings
        logger.info("   📥 Loading all chunk embeddings...")
        all_embeddings = []
        chunk_to_embedding_mapping = []
        
        completed_chunks = [c for c in self.chunks_info if c.status == 'completed']
        
        for chunk_info in tqdm(completed_chunks, desc="   Loading embeddings"):
            embeddings_file = self.embeddings_dir / chunk_info.embeddings_file
            
            with open(embeddings_file, 'r') as f:
                chunk_embeddings = json.load(f)
            
            for chunk_emb in chunk_embeddings:
                if chunk_emb['content_embedding'] and any(x != 0.0 for x in chunk_emb['content_embedding']):
                    all_embeddings.append(chunk_emb['content_embedding'])
                    chunk_to_embedding_mapping.append({
                        'chunk_id': chunk_emb['chunk_id'],
                        'chunk_type': chunk_emb['chunk_type'],
                        'asin': chunk_emb['asin']
                    })
        
        logger.info(f"   📊 Collected {len(all_embeddings):,} valid embeddings")
        
        # Build HNSW index
        if not all_embeddings:
            raise ValueError("No valid embeddings found for HNSW index")
        
        embeddings_matrix = np.array(all_embeddings, dtype=np.float32)
        dimension = embeddings_matrix.shape[1]
        
        logger.info(f"   🔧 Building HNSW index: {len(all_embeddings):,} embeddings, dimension={dimension}")
        
        # Create optimized HNSW index
        faiss_index = faiss.IndexHNSWFlat(dimension, 64)  
        faiss_index.hnsw.efConstruction = 2000  
        faiss_index.hnsw.efSearch = 1000  
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_matrix)
        
        # Build index (use GPU if available)
        if self.use_gpu and faiss.get_num_gpus() > 0:
            try:
                logger.info(f"   🔥 Using {faiss.get_num_gpus()} GPUs for index building")
                gpu_index = faiss.index_cpu_to_all_gpus(faiss_index)
                gpu_index.add(embeddings_matrix)
                faiss_index = faiss.index_gpu_to_cpu(gpu_index)
                del gpu_index
                self._clear_gpu_memory()
            except Exception as e:
                logger.warning(f"   GPU indexing failed: {e}, using CPU")
                faiss_index.add(embeddings_matrix)
        else:
            faiss_index.add(embeddings_matrix)
        
        # Save index and mapping
        faiss.write_index(faiss_index, str(hnsw_index_file))
        with open(hnsw_mapping_file, 'wb') as f:
            pickle.dump(chunk_to_embedding_mapping, f)
        
        phase_time = time.time() - start_time
        logger.info(f"✅ Phase 3 Complete: HNSW index built in {phase_time:.2f}s")

    def phase4_calculate_similarities(self, k_neighbors: int = 200):
        """Phase 4: Calculate similarities using optimized GPU-accelerated method"""
        logger.info("🔄 Phase 4: Preparing for optimized similarity calculation...")
        start_time = time.time()
        
        # Check if optimized edges already exist
        existing_edges = list(self.final_dir.glob("stark_optimized_edges_*.json"))
        if existing_edges:
            logger.info(f"   ✅ Optimized similarities already calculated: {existing_edges[-1].name}")
            return
        
        # Check if basic similarities exist (legacy)
        basic_similarities_file = self.final_dir / 'similarities.pkl'
        if basic_similarities_file.exists():
            logger.info("   📊 Basic similarities found, but optimized calculation is recommended")
            logger.info("   💡 Run 'python run_optimized_similarities.py' for GPU-accelerated calculation")
            return
        
        logger.info("   💡 For optimal performance with ~2M nodes, use the optimized calculator:")
        logger.info("   🚀 python run_optimized_similarities.py")
        logger.info("   📦 Features: GPU batching, deduplication, 95th percentile filtering")
        
        # Create a placeholder to indicate phase completion
        placeholder_file = self.final_dir / 'similarity_calculation_ready.txt'
        with open(placeholder_file, 'w') as f:
            f.write(f"Chunked pipeline completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Ready for optimized similarity calculation with {k_neighbors} neighbors\n")
            f.write("Run: python run_optimized_similarities.py\n")
        
        phase_time = time.time() - start_time
        logger.info(f"✅ Phase 4 Complete: Ready for optimized calculation in {phase_time:.2f}s")

    def phase5_generate_analysis(self):
        """Phase 5: Generate final analysis and edges"""
        logger.info("🔄 Phase 5: Generating final analysis...")
        start_time = time.time()
        
        # Simple analysis for now - just save the results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        analysis_file = self.final_dir / f"stark_chunked_analysis_{timestamp}.json"
        
        analysis_data = {
            'timestamp': timestamp,
            'total_chunks': len(self.chunks_info),
            'completed_chunks': len([c for c in self.chunks_info if c.status == 'completed']),
            'total_processing_time': sum(c.processing_time or 0 for c in self.chunks_info),
            'chunk_details': [
                {
                    'chunk_id': c.chunk_id,
                    'product_count': c.product_count,
                    'status': c.status,
                    'processing_time': c.processing_time,
                    'embeddings_file': c.embeddings_file
                }
                for c in self.chunks_info
            ]
        }
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        phase_time = time.time() - start_time
        logger.info(f"✅ Phase 5 Complete: Analysis saved to {analysis_file.name} in {phase_time:.2f}s")
        
        return analysis_file

    # Helper methods (reusing from original pipeline)
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

    def _assign_embeddings_to_chunks(self, field_type: str, chunk_indices: List[int], embeddings: List[List[float]], chunk_data: List[STARKChunkData]):
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
            setattr(chunk_data[chunk_idx], embedding_attr, embedding)

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

    def _save_chunk_manifest(self):
        """Save chunk manifest to disk"""
        manifest_data = {
            'total_products': self.total_products,
            'chunk_size': self.chunk_size,
            'total_chunks': len(self.chunks_info),
            'chunks': [
                {
                    'chunk_id': c.chunk_id,
                    'start_idx': c.start_idx,
                    'end_idx': c.end_idx,
                    'product_count': c.product_count,
                    'status': c.status,
                    'embeddings_file': c.embeddings_file,
                    'processing_time': c.processing_time,
                    'error_message': c.error_message
                }
                for c in self.chunks_info
            ]
        }
        
        with open(self.chunk_manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    def _load_chunk_manifest(self):
        """Load chunk manifest from disk"""
        if not self.chunk_manifest_file.exists():
            return
        
        with open(self.chunk_manifest_file, 'r') as f:
            manifest_data = json.load(f)
        
        self.total_products = manifest_data['total_products']
        self.chunks_info = []
        
        for chunk_data in manifest_data['chunks']:
            chunk_info = ChunkInfo(
                chunk_id=chunk_data['chunk_id'],
                start_idx=chunk_data['start_idx'],
                end_idx=chunk_data['end_idx'],
                product_count=chunk_data['product_count'],
                status=chunk_data['status'],
                embeddings_file=chunk_data.get('embeddings_file'),
                processing_time=chunk_data.get('processing_time'),
                error_message=chunk_data.get('error_message')
            )
            self.chunks_info.append(chunk_info)

    def _clear_gpu_memory(self):
        """Clear GPU memory"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
        gc.collect()

def main():
    """Main execution function"""
    logger.info("🚀 STARK Chunked Pipeline - Fault-Tolerant Architecture")
    logger.info("💡 Process data in manageable chunks to avoid memory issues")
    logger.info("=" * 60)
    
    pipeline = STARKChunkedPipeline(chunk_size=50000)  # 50k products per chunk
    
    try:
        analysis_file = pipeline.run_chunked_pipeline()
        
        logger.info("🎉 Chunked Pipeline completed successfully!")
        logger.info(f"📁 Final results: {analysis_file}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
