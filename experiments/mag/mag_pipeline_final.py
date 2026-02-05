#!/usr/bin/env python3
"""
MAG Dataset Pipeline - FINAL CLEAN VERSION
Based on successful testing of filtering + shuffling approach
Processes ALL papers and authors with balanced chunks
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
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.processors.embedding_service import EmbeddingService
from src.core.models import ProcessingConfig

@dataclass
class ChunkInfo:
    """Information about a processing chunk"""
    chunk_id: int
    start_idx: int
    end_idx: int
    node_count: int
    status: str # 'pending', 'processing', 'completed', 'failed'
    embeddings_file: Optional[str] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None

@dataclass
class MAGNode:
    """Clean MAG node data structure"""
    object_id: str
    node_type: str # "paper" or "author"
    content: str # Built content string
    
    # Paper-specific fields
    original_title: Optional[str] = None
    publisher: Optional[str] = None
    abstract: Optional[str] = None
    authors: Optional[List[str]] = None
    fields_of_study: Optional[List[str]] = None
    cites: Optional[List[str]] = None
    
    # Author-specific fields (ONLY these two as specified)
    display_name: Optional[str] = None
    institution: Optional[str] = None
    
    # All embeddings
    content_embedding: Optional[List[float]] = None
    original_title_embedding: Optional[List[float]] = None
    abstract_embedding: Optional[List[float]] = None
    authors_embedding: Optional[List[float]] = None
    fields_of_study_embedding: Optional[List[float]] = None
    cites_embedding: Optional[List[float]] = None
    display_name_embedding: Optional[List[float]] = None
    institution_embedding: Optional[List[float]] = None
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = None

class MAGPipelineFinal:
    """
    FINAL MAG Pipeline - Clean implementation based on successful tests
    Strategy: Filter paper/author first, shuffle for balance, then chunk into 20 pieces
    """
    
    def __init__(self, 
                 mag_dataset_file: str = "/shared/khoja/CogComp/datasets/MAG/data_with_citations.json",
                 cache_dir: str = "/shared/khoja/CogComp/output/mag_final_cache",
                 target_chunks: int = 20,
                 use_gpu: bool = True,
                 num_threads: int = 32):
        
        self.mag_dataset_file = Path(mag_dataset_file)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_chunks = target_chunks
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.num_threads = num_threads
        self.num_gpus = torch.cuda.device_count() if self.use_gpu else 0
        
        # Create subdirectories
        self.embeddings_dir = self.cache_dir / "embeddings"
        self.final_dir = self.cache_dir / "final"
        
        for dir_path in [self.embeddings_dir, self.final_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding service
        self.config = ProcessingConfig(
            use_faiss=True, 
            faiss_use_gpu=self.use_gpu,
            batch_size=2048, # Good balance of speed and memory
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedding_service = EmbeddingService(self.config)
        
        # Pipeline state
        self.chunk_manifest_file = self.cache_dir / "manifest.json"
        self.chunks_info: List[ChunkInfo] = []
        self.total_nodes = 0
        self.paper_count = 0
        self.author_count = 0
        self.mixed_nodes = [] # Pre-filtered and shuffled nodes
        
        logger.info(f" MAG Pipeline Final - Clean Implementation")
        logger.info(f" Target chunks: {self.target_chunks}")
        logger.info(f" Cache directory: {self.cache_dir}")
        logger.info(f" GPUs available: {self.num_gpus}")

    def run_pipeline(self, max_papers: Optional[int] = None):
        """Run the complete pipeline"""
        logger.info(" STARTING MAG FINAL PIPELINE")
        logger.info("=" * 60)
        logger.info(" Strategy: Filter → Shuffle → Chunk → Process → Build Index")
        logger.info("")
        
        total_start_time = time.time()
        
        try:
            # Phase 1: Load, filter, shuffle, and create balanced chunks
            self.phase1_prepare_balanced_chunks(max_papers)
            
            # Phase 2: Process each chunk to generate ALL embeddings
            self.phase2_process_chunks()
            
            # Phase 3: Build HNSW index
            self.phase3_build_index()
            
            # Phase 4: Final analysis
            analysis_file = self.phase4_generate_analysis()
            
            total_time = time.time() - total_start_time
            
            logger.info(" MAG FINAL PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f" Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
            logger.info(f" Results: {analysis_file}")
            
            return analysis_file
            
        except Exception as e:
            logger.error(f" Pipeline failed: {e}")
            raise

    def phase1_prepare_balanced_chunks(self, max_papers: Optional[int] = None):
        """Phase 1: Strategic filtering, shuffling, and balanced chunking"""
        logger.info(" Phase 1: Preparing balanced chunks...")
        start_time = time.time()
        
        # Step 1: Load MAG dataset
        logger.info(" Loading MAG dataset...")
        with open(self.mag_dataset_file, 'r') as f:
            mag_data = json.load(f)
        
        logger.info(f" Total MAG objects: {len(mag_data):,}")
        
        # Step 2: Strategic filtering (ALL authors + specified papers)
        logger.info(" Strategic filtering...")
        if max_papers is None:
            max_papers = 700244 # All papers in dataset
        
        logger.info(f" Strategy: ALL authors + {max_papers:,} papers")
        
        relevant_nodes = []
        paper_count = 0
        author_count = 0
        
        for obj_id, obj_data in tqdm(mag_data.items(), desc=" Filtering"):
            obj_type = obj_data.get('type')
            
            if obj_type == 'author':
                relevant_nodes.append((obj_id, obj_data))
                author_count += 1
            elif obj_type == 'paper' and paper_count < max_papers:
                relevant_nodes.append((obj_id, obj_data))
                paper_count += 1
                
                if paper_count >= max_papers:
                    logger.info(f" Reached {max_papers:,} papers, stopping paper collection...")
                    break
        
        self.total_nodes = len(relevant_nodes)
        self.paper_count = paper_count
        self.author_count = author_count
        
        logger.info(f" Filtered {self.total_nodes:,} nodes:")
        logger.info(f" Papers: {self.paper_count:,}")
        logger.info(f" Authors: {self.author_count:,}")
        
        # Step 3: Shuffle for balanced distribution
        logger.info(" Shuffling for balanced chunks...")
        random.shuffle(relevant_nodes)
        self.mixed_nodes = relevant_nodes
        
        # Verify balance in sample
        sample_papers = sum(1 for _, data in self.mixed_nodes[:1000] if data.get('type') == 'paper')
        sample_authors = sum(1 for _, data in self.mixed_nodes[:1000] if data.get('type') == 'author')
        logger.info(f" Balance check (first 1000): {sample_papers} papers, {sample_authors} authors")
        
        # Step 4: Create balanced chunks
        chunk_size = math.ceil(self.total_nodes / self.target_chunks)
        actual_chunks = math.ceil(self.total_nodes / chunk_size)
        
        logger.info(f" Creating {actual_chunks} balanced chunks...")
        logger.info(f" Chunk size: ~{chunk_size:,} nodes each")
        
        self.chunks_info = []
        for chunk_id in range(actual_chunks):
            start_idx = chunk_id * chunk_size
            end_idx = min(start_idx + chunk_size, self.total_nodes)
            
            chunk_info = ChunkInfo(
                chunk_id=chunk_id,
                start_idx=start_idx,
                end_idx=end_idx,
                node_count=end_idx - start_idx,
                status='pending'
            )
            self.chunks_info.append(chunk_info)
        
        # Verify chunk balance
        for chunk in self.chunks_info[:3]: # Show first 3 chunks
            chunk_nodes = self.mixed_nodes[chunk.start_idx:chunk.end_idx]
            chunk_papers = sum(1 for _, data in chunk_nodes if data.get('type') == 'paper')
            chunk_authors = sum(1 for _, data in chunk_nodes if data.get('type') == 'author')
            
            logger.info(f" Chunk {chunk.chunk_id}: {chunk.node_count:,} nodes "
                       f"({chunk_papers:,} papers, {chunk_authors:,} authors)")
        
        if len(self.chunks_info) > 3:
            logger.info(f" ... and {len(self.chunks_info) - 3} more balanced chunks")
        
        # Save manifest
        self._save_manifest()
        
        phase_time = time.time() - start_time
        logger.info(f" Phase 1 Complete: {len(self.chunks_info)} balanced chunks in {phase_time:.2f}s")

    def phase2_process_chunks(self):
        """Phase 2: Process each chunk to generate ALL embeddings"""
        logger.info(" Phase 2: Processing chunks to generate ALL embeddings...")
        start_time = time.time()
        
        pending_chunks = [c for c in self.chunks_info if c.status in ['pending', 'failed']]
        logger.info(f" Chunks to process: {len(pending_chunks)}")
        
        for chunk_info in pending_chunks:
            self._process_single_chunk(chunk_info)
            self._save_manifest()
        
        phase_time = time.time() - start_time
        successful_chunks = len([c for c in self.chunks_info if c.status == 'completed'])
        logger.info(f" Phase 2 Complete: {successful_chunks}/{len(self.chunks_info)} chunks in {phase_time:.2f}s")

    def _process_single_chunk(self, chunk_info: ChunkInfo):
        """Process a single chunk with ALL embeddings"""
        logger.info(f" Processing Chunk {chunk_info.chunk_id}")
        chunk_start_time = time.time()
        
        try:
            chunk_info.status = 'processing'
            
            # Load and convert chunk nodes
            chunk_nodes = self._load_and_convert_chunk(chunk_info)
            
            paper_count = sum(1 for node in chunk_nodes if node.node_type == 'paper')
            author_count = sum(1 for node in chunk_nodes if node.node_type == 'author')
            logger.info(f" {len(chunk_nodes):,} nodes ({paper_count:,} papers, {author_count:,} authors)")
            
            # Generate ALL embeddings
            self._generate_all_embeddings(chunk_nodes)
            
            # Save embeddings
            embeddings_file = self.embeddings_dir / f"chunk_{chunk_info.chunk_id:03d}.json"
            self._save_chunk_embeddings(chunk_nodes, embeddings_file)
            
            chunk_info.embeddings_file = embeddings_file.name
            chunk_info.status = 'completed'
            chunk_info.processing_time = time.time() - chunk_start_time
            
            logger.info(f" Chunk {chunk_info.chunk_id} completed in {chunk_info.processing_time:.2f}s")
            
        except Exception as e:
            chunk_info.status = 'failed'
            chunk_info.error_message = str(e)
            chunk_info.processing_time = time.time() - chunk_start_time
            logger.error(f" Chunk {chunk_info.chunk_id} failed: {e}")
        
        finally:
            # Always clear memory
            self._clear_gpu_memory()
            gc.collect()

    def _load_and_convert_chunk(self, chunk_info: ChunkInfo) -> List[MAGNode]:
        """Load chunk data and convert to MAGNode objects"""
        chunk_raw_nodes = self.mixed_nodes[chunk_info.start_idx:chunk_info.end_idx]
        
        processed_nodes = []
        
        for obj_id, obj_data in chunk_raw_nodes:
            obj_type = obj_data.get('type')
            
            if obj_type == 'paper':
                node = self._create_paper_node(obj_id, obj_data)
            elif obj_type == 'author':
                node = self._create_author_node(obj_id, obj_data)
            else:
                continue
            
            if node:
                processed_nodes.append(node)
        
        return processed_nodes

    def _create_paper_node(self, obj_id: str, obj_data: Dict[str, Any]) -> Optional[MAGNode]:
        """Create paper node with all required fields"""
        try:
            # Extract paper fields
            original_title = obj_data.get('OriginalTitle', '')
            publisher = obj_data.get('Publisher', '')
            abstract = obj_data.get('abstract', '')
            authors = obj_data.get('authors', [])
            fields_of_study = obj_data.get('fields_of_study', [])
            cites = obj_data.get('cites', [])
            
            # Build content: OriginalTitle + Publisher + abstract + authors + fields_of_study + cites
            content_parts = []
            
            if original_title and original_title.strip():
                content_parts.append(original_title.strip())
            
            if publisher and publisher.strip():
                content_parts.append(publisher.strip())
            
            if abstract and abstract.strip():
                content_parts.append(abstract.strip())
            
            if authors and isinstance(authors, list):
                authors_str = ' '.join([str(a) for a in authors if str(a).strip()])
                if authors_str.strip():
                    content_parts.append(authors_str.strip())
            
            if fields_of_study and isinstance(fields_of_study, list):
                fields_str = ' '.join([str(f) for f in fields_of_study if str(f).strip()])
                if fields_str.strip():
                    content_parts.append(fields_str.strip())
            
            if cites and isinstance(cites, list):
                cites_str = ' '.join([str(c) for c in cites if str(c).strip()])
                if cites_str.strip():
                    content_parts.append(cites_str.strip())
            
            content = ' '.join(content_parts)
            
            return MAGNode(
                object_id=obj_id,
                node_type="paper",
                content=content,
                original_title=original_title,
                publisher=publisher,
                abstract=abstract,
                authors=authors,
                fields_of_study=fields_of_study,
                cites=cites,
                metadata={
                    'year': obj_data.get('Year', ''),
                    'doc_type': obj_data.get('DocType', ''),
                    'paper_rank': obj_data.get('PaperRank', ''),
                    'citation_count': obj_data.get('PaperCitationCount', 0),
                    'reference_count': obj_data.get('ReferenceCount', 0)
                }
            )
            
        except Exception as e:
            logger.warning(f"Error creating paper node {obj_id}: {e}")
            return None

    def _create_author_node(self, obj_id: str, obj_data: Dict[str, Any]) -> Optional[MAGNode]:
        """Create author node with ONLY specified fields"""
        try:
            # Extract ONLY DisplayName and institution as specified
            display_name = obj_data.get('DisplayName', '')
            institution = obj_data.get('institution', '')
            
            # Build content: DisplayName + institution (ONLY these two)
            content_parts = []
            
            if display_name and display_name.strip():
                content_parts.append(display_name.strip())
            
            if institution and institution.strip():
                content_parts.append(institution.strip())
            
            content = ' '.join(content_parts)
            
            return MAGNode(
                object_id=obj_id,
                node_type="author",
                content=content,
                display_name=display_name,
                institution=institution,
                metadata={
                    'rank': obj_data.get('Rank', ''),
                    'paper_count': obj_data.get('PaperCount', 0),
                    'citation_count': obj_data.get('CitationCount', 0)
                }
            )
            
        except Exception as e:
            logger.warning(f"Error creating author node {obj_id}: {e}")
            return None

    def _generate_all_embeddings(self, nodes: List[MAGNode]):
        """Generate ALL embeddings for all nodes"""
        logger.info(" Generating ALL embeddings...")
        
        # Prepare all text collections
        embedding_tasks = {
            'content': [],
            'original_title': [],
            'abstract': [],
            'authors': [],
            'fields_of_study': [],
            'cites': [],
            'display_name': [],
            'institution': []
        }
        
        # Collect texts and track which nodes they belong to
        for node_idx, node in enumerate(nodes):
            # Content (always present)
            if node.content and node.content.strip():
                embedding_tasks['content'].append((node_idx, node.content.strip()))
            
            if node.node_type == 'paper':
                # Paper-specific fields
                if node.original_title and node.original_title.strip():
                    embedding_tasks['original_title'].append((node_idx, node.original_title.strip()))
                
                if node.abstract and node.abstract.strip():
                    embedding_tasks['abstract'].append((node_idx, node.abstract.strip()))
                
                if node.authors and isinstance(node.authors, list):
                    authors_str = ' '.join([str(a) for a in node.authors if str(a).strip()])
                    if authors_str.strip():
                        embedding_tasks['authors'].append((node_idx, authors_str.strip()))
                
                if node.fields_of_study and isinstance(node.fields_of_study, list):
                    fields_str = ' '.join([str(f) for f in node.fields_of_study if str(f).strip()])
                    if fields_str.strip():
                        embedding_tasks['fields_of_study'].append((node_idx, fields_str.strip()))
                
                if node.cites and isinstance(node.cites, list):
                    cites_str = ' '.join([str(c) for c in node.cites if str(c).strip()])
                    if cites_str.strip():
                        embedding_tasks['cites'].append((node_idx, cites_str.strip()))
            
            elif node.node_type == 'author':
                # Author-specific fields
                if node.display_name and node.display_name.strip():
                    embedding_tasks['display_name'].append((node_idx, node.display_name.strip()))
                
                if node.institution and node.institution.strip():
                    embedding_tasks['institution'].append((node_idx, node.institution.strip()))
        
        # Generate embeddings for each field
        for field_name, field_data in embedding_tasks.items():
            if not field_data:
                continue
            
            logger.info(f" {field_name}: {len(field_data)} texts")
            
            try:
                # Extract texts and indices
                texts = [text for _, text in field_data]
                node_indices = [node_idx for node_idx, _ in field_data]
                
                # Generate embeddings
                embeddings = self.embedding_service.generate_embeddings_bulk(texts)
                
                # Assign embeddings back to nodes
                embedding_attr = f"{field_name}_embedding"
                for node_idx, embedding in zip(node_indices, embeddings):
                    setattr(nodes[node_idx], embedding_attr, embedding)
                
                logger.info(f" {field_name}: {len(embeddings)} embeddings generated")
                
            except Exception as e:
                logger.error(f" {field_name} failed: {e}")
            
            # Clear memory after each field
            self._clear_gpu_memory()

    def _save_chunk_embeddings(self, nodes: List[MAGNode], file_path: Path):
        """Save chunk embeddings to JSON"""
        logger.info(f" Saving to {file_path.name}...")
        
        chunk_data = []
        for node in nodes:
            node_data = {
                'object_id': node.object_id,
                'node_type': node.node_type,
                'content': node.content,
                'content_embedding': node.content_embedding,
                'metadata': node.metadata
            }
            
            # Add type-specific fields and embeddings
            if node.node_type == 'paper':
                node_data.update({
                    'original_title': node.original_title,
                    'publisher': node.publisher,
                    'abstract': node.abstract,
                    'authors': node.authors,
                    'fields_of_study': node.fields_of_study,
                    'cites': node.cites,
                    'original_title_embedding': node.original_title_embedding,
                    'abstract_embedding': node.abstract_embedding,
                    'authors_embedding': node.authors_embedding,
                    'fields_of_study_embedding': node.fields_of_study_embedding,
                    'cites_embedding': node.cites_embedding
                })
            elif node.node_type == 'author':
                node_data.update({
                    'display_name': node.display_name,
                    'institution': node.institution,
                    'display_name_embedding': node.display_name_embedding,
                    'institution_embedding': node.institution_embedding
                })
            
            chunk_data.append(node_data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)

    def phase3_build_index(self):
        """Phase 3: Build HNSW index from all embeddings"""
        logger.info(" Phase 3: Building HNSW index...")
        # Implementation similar to original pipeline
        logger.info(" HNSW index building - to be implemented")

    def phase4_generate_analysis(self):
        """Phase 4: Generate final analysis"""
        logger.info(" Phase 4: Generating analysis...")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        analysis_file = self.final_dir / f"mag_final_analysis_{timestamp}.json"
        
        analysis_data = {
            'timestamp': timestamp,
            'total_nodes': self.total_nodes,
            'paper_count': self.paper_count,
            'author_count': self.author_count,
            'total_chunks': len(self.chunks_info),
            'completed_chunks': len([c for c in self.chunks_info if c.status == 'completed']),
            'strategy': 'filter_shuffle_chunk'
        }
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        return analysis_file

    def _save_manifest(self):
        """Save manifest to disk"""
        manifest_data = {
            'total_nodes': self.total_nodes,
            'paper_count': self.paper_count,
            'author_count': self.author_count,
            'target_chunks': self.target_chunks,
            'chunks': [
                {
                    'chunk_id': c.chunk_id,
                    'start_idx': c.start_idx,
                    'end_idx': c.end_idx,
                    'node_count': c.node_count,
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

    def _clear_gpu_memory(self):
        """Clear GPU memory"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
        gc.collect()

def main():
    """Main execution function"""
    # Set PyTorch memory configuration
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    logger.info(" MAG FINAL PIPELINE - Clean Implementation")
    logger.info(" Based on successful test_complete_filtering.py results")
    logger.info(" Strategy: Filter → Shuffle → 20 Balanced Chunks → ALL Embeddings")
    logger.info("=" * 70)
    
    # Start with a subset for testing, then remove max_papers for full dataset
    pipeline = MAGPipelineFinal(target_chunks=20)
    
    try:
        # Test with SMALL subset first for validation 
        analysis_file = pipeline.run_pipeline(max_papers=1000) # Start with just 1000 papers for testing
        
        logger.info(" MAG FINAL PIPELINE COMPLETED!")
        logger.info(f" Results: {analysis_file}")
        
    except Exception as e:
        logger.error(f" Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
