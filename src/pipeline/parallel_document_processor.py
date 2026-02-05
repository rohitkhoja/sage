#!/usr/bin/env python3
"""
Parallel Document Processor with GPU acceleration and intelligent caching
Supports multi-GPU processing with Ray/Dask orchestration
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import time
from torch.utils.data import DataLoader, Dataset
import ray
import psutil
import re
from loguru import logger
from .data.stopwords_loader import ENGLISH_STOPWORDS
from src.core.models import DocumentChunk, SourceInfo, ChunkType

@dataclass
class ChunkEmbedding:
    """Data structure for storing chunk embeddings with metadata - DEPRECATED, keeping for backward compatibility"""
    chunk_id: str
    content: str
    embedding: List[float]
    source_document: str
    chunk_index: int
    token_count: int
    created_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkEmbedding':
        return cls(**data)

class ChunkCache:
    """Enhanced caching system with dictionary-based chunk key lookup and thread safety"""
    
    def __init__(self, cache_dir: str = "output/chunks_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index_file = self.cache_dir / "chunk_index.json"
        self.chunk_keys_file = self.cache_dir / "chunk_keys_dict.json" 
        self._lock = None
        self.load_index()
        self.load_chunk_keys_dict()
    
    @property
    def lock(self):
        """Lazy initialization of lock for Ray compatibility"""
        if self._lock is None:
            from threading import Lock
            self._lock = Lock()
        return self._lock
    
    def load_index(self):
        """Load existing chunk index with thread safety"""
        
        
        with self.lock:
            if self.cache_index_file.exists():
                try:
                    with open(self.cache_index_file, 'r') as f:
                        self.index = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    logger.warning("Corrupted cache index, creating new one")
                    self.index = {}
            else:
                self.index = {}
            logger.info(f"Loaded chunk cache with {len(self.index)} existing chunks")
    
    def load_chunk_keys_dict(self):
        """Load chunk keys dictionary for fast O(1) lookup"""
        
        
        with self.lock:
            if self.chunk_keys_file.exists():
                try:
                    with open(self.chunk_keys_file, 'r') as f:
                        self.chunk_keys_dict = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    logger.warning("Corrupted chunk keys dict, creating new one")
                    self.chunk_keys_dict = {}
            else:
                self.chunk_keys_dict = {}
            logger.info(f"Loaded chunk keys dictionary with {len(self.chunk_keys_dict)} entries")
    
    def save_index(self):
        """Save chunk index to disk with thread safety"""
        
        
        with self.lock:
            try:
                # Atomic write using temporary file
                temp_file = self.cache_index_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(self.index, f, indent=2)
                temp_file.replace(self.cache_index_file)
            except Exception as e:
                logger.error(f"Failed to save cache index: {e}")
    
    def get_chunk_key(self, content: str, source_doc: str) -> str:
        """Generate unique key for chunk content - now deterministic"""
        # Use SHA256 for better uniqueness with large datasets
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        doc_hash = hashlib.md5(source_doc.encode()).hexdigest()[:8]
        return f"{doc_hash}_{content_hash}"
    
    def get_chunk_by_content_hash(self, content: str, source_doc: str) -> Optional[DocumentChunk]:
        """Get chunk by content hash - for deterministic retrieval"""
        key = self.get_chunk_key(content, source_doc)
        with self.lock:
            if key in self.index:
                chunk_file = self.cache_dir / f"{key}.json"
                if chunk_file.exists():
                    try:
                        with open(chunk_file, 'r') as f:
                            data = json.load(f)
                        # Convert back to DocumentChunk
                        return self._dict_to_document_chunk(data)
                    except (json.JSONDecodeError, FileNotFoundError):
                        
                        logger.warning(f"Corrupted chunk file: {chunk_file}")
                        # Remove corrupted entry
                        del self.index[key]
        return None
    
    def chunk_exists(self, content: str, source_doc: str) -> bool:
        """Check if chunk already processed with thread safety"""
        key = self.get_chunk_key(content, source_doc)
        with self.lock:
            return key in self.index
    
    def get_chunk(self, content: str, source_doc: str) -> Optional[DocumentChunk]:
        """Retrieve cached chunk with thread safety"""
        
        
        key = self.get_chunk_key(content, source_doc)
        with self.lock:
            if key in self.index:
                chunk_file = self.cache_dir / f"{key}.json"
                if chunk_file.exists():
                    try:
                        with open(chunk_file, 'r') as f:
                            data = json.load(f)
                        return self._dict_to_document_chunk(data)
                    except (json.JSONDecodeError, FileNotFoundError):
                        logger.warning(f"Corrupted chunk file: {chunk_file}")
                        # Remove corrupted entry
                        del self.index[key]
        return None
    
    def store_chunk(self, chunk: DocumentChunk) -> str:
        """Store DocumentChunk to cache with thread safety"""
        
        
        key = self.get_chunk_key(chunk.content, chunk.source_info.source_name)
        chunk_file = self.cache_dir / f"{key}.json"
        
        try:
            # Convert DocumentChunk to dict for storage
            chunk_dict = self._document_chunk_to_dict(chunk)
            
            # Save chunk data atomically
            temp_file = chunk_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(chunk_dict, f, indent=2)
            temp_file.replace(chunk_file)
            
            # Update index
            with self.lock:
                self.index[key] = {
                    "file": chunk_file.name,
                    "source_document": chunk.source_info.source_name,
                    "created_at": time.strftime("%Y%m%d_%H%M%S"),
                    "chunk_id": chunk.chunk_id,
                    "content_hash": key
                }
            
            return key
        except Exception as e:
            logger.error(f"Failed to store chunk: {e}")
            return None
    
    def _document_chunk_to_dict(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """Convert DocumentChunk to dictionary for storage"""
        return {
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "source_info": chunk.source_info.dict(),
            "sentences": chunk.sentences,
            "keywords": chunk.keywords,
            "summary": chunk.summary,
            "embedding": chunk.embedding,
            "merged_sentence_count": chunk.merged_sentence_count,
            "chunk_type": "document" # For type identification
        }
    
    def _dict_to_document_chunk(self, data: Dict[str, Any]) -> DocumentChunk:
        """Convert dictionary back to DocumentChunk"""
        # Handle SourceInfo conversion
        source_info_data = data["source_info"]
        source_info = SourceInfo(**source_info_data)
        
        return DocumentChunk(
            chunk_id=data["chunk_id"],
            content=data["content"],
            source_info=source_info,
            sentences=data.get("sentences", []),
            keywords=data.get("keywords", []),
            summary=data.get("summary", ""),
            embedding=data.get("embedding"),
            merged_sentence_count=data.get("merged_sentence_count", 1)
        )
    
    def get_new_chunks_since(self, timestamp: str) -> List[str]:
        """Get chunks created since timestamp"""
        with self.lock:
            new_chunks = []
            for key, info in self.index.items():
                if info["created_at"] > timestamp:
                    new_chunks.append(key)
            return new_chunks
    
    def get_all_chunk_ids(self) -> List[str]:
        """Get all cached chunk IDs"""
        with self.lock:
            return [info["chunk_id"] for info in self.index.values()]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with self.lock:
            total_chunks = len(self.index)
            cache_size = sum(
                (self.cache_dir / info["file"]).stat().st_size 
                for info in self.index.values()
                if (self.cache_dir / info["file"]).exists()
            )
            
            return {
                "total_chunks": total_chunks,
                "cache_size_mb": cache_size / (1024 * 1024),
                "cache_directory": str(self.cache_dir),
                "index_file": str(self.cache_index_file)
            }
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve chunk by chunk ID with thread safety"""
        
        
        with self.lock:
            for key, info in self.index.items():
                if info.get("chunk_id") == chunk_id:
                    chunk_file = self.cache_dir / f"{key}.json"
                    if chunk_file.exists():
                        try:
                            with open(chunk_file, 'r') as f:
                                data = json.load(f)
                            return self._dict_to_document_chunk(data)
                        except (json.JSONDecodeError, FileNotFoundError):
                            logger.warning(f"Corrupted chunk file: {chunk_file}")
                            # Remove corrupted entry
                            del self.index[key]
        return None

class GPUEmbeddingDataset(Dataset):
    """Dataset for efficient GPU embedding computation"""
    
    def __init__(self, texts: List[str]):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]

@ray.remote(num_gpus=1)
class ParallelDocumentProcessor:
    """Ray actor for parallel document processing with optimized GPU acceleration"""
    
    def __init__(self, gpu_id: int, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cache_dir: str = None):
        # Import logger locally to avoid serialization issues
        
        
        # Set environment variable to suppress tokenizer warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.model_name = model_name
        self.cache_dir = cache_dir or f"output/chunks_cache_gpu_{gpu_id}"
        self._cache = None # Will be initialized lazily
        
        # Set GPU for this process
        torch.cuda.set_device(gpu_id)
        
        # Initialize model on specific GPU with optimization
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Enable mixed precision for faster inference
        self.model.half() # Use FP16 for speed
        
        # Performance tracking
        self.processing_stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "cache_hits": 0,
            "gpu_memory_peak": 0,
            "processing_time": 0
        }
        
        logger.info(f"Initialized optimized processor on GPU {gpu_id}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.1f} GB")
    
    @property 
    def cache(self):
        """Lazy initialization of cache for Ray compatibility"""
        if self._cache is None:
            self._cache = ChunkCache(self.cache_dir)
        return self._cache
    
    def process_document_batch(self, documents: List[Dict[str, Any]], 
                             similarity_threshold: float = 0.8,
                             batch_size: int = None) -> Dict[str, Any]:
        """Process a batch of documents with optimized GPU acceleration"""
        
        
        if batch_size is None:
            # Auto-determine batch size based on GPU memory
            gpu_memory_gb = torch.cuda.get_device_properties(self.gpu_id).total_memory / 1e9
            batch_size = min(64, max(16, int(gpu_memory_gb * 8))) # Heuristic
        
        results = {
            "processed_chunks": [],
            "cached_chunks": [],
            "total_documents": len(documents),
            "processing_time": 0,
            "cache_hit_ratio": 0,
            "gpu_utilization": 0
        }
        
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats(self.gpu_id)
        
        try:
            for doc in documents:
                doc_chunks = self._process_single_document(
                    doc, similarity_threshold, batch_size
                )
                results["processed_chunks"].extend(doc_chunks["new_chunks"])
                results["cached_chunks"].extend(doc_chunks["cached_chunks"])
                
                # Update stats
                self.processing_stats["total_documents"] += 1
                self.processing_stats["cache_hits"] += len(doc_chunks["cached_chunks"])
        
        except Exception as e:
            logger.error(f"Error processing batch on GPU {self.gpu_id}: {e}")
            raise
        
        finally:
            # Calculate performance metrics
            results["processing_time"] = time.time() - start_time
            results["gpu_utilization"] = torch.cuda.max_memory_allocated(self.gpu_id) / torch.cuda.get_device_properties(self.gpu_id).total_memory
            
            total_chunks = len(results["processed_chunks"]) + len(results["cached_chunks"])
            results["cache_hit_ratio"] = len(results["cached_chunks"]) / max(1, total_chunks)
            
            # Always save cache index after processing (not just periodically)
                self.cache.save_index()
            
            # Clear GPU cache
            torch.cuda.empty_cache()
        
        return results
    
    def _create_source_info(self, document: Dict[str, Any]) -> SourceInfo:
        """Create proper SourceInfo object from document metadata"""
        # Extract metadata from document or use defaults
        metadata = document.get("metadata", {})
        source_name = document.get("source", metadata.get("title", "unknown"))
        
        return SourceInfo(
            source_id=metadata.get("id", source_name),
            source_name=source_name,
            source_type=ChunkType.DOCUMENT,
            file_path=metadata.get("file_path", ""),
            structural_link=metadata.get("structural_link", []),
            original_source=metadata.get("original_source", ""),
            additional_information=metadata.get("additional_information", ""),
            content=document.get("content", "")
        )
    
    def _generate_deterministic_chunk_id(self, content: str, source_doc: str, chunk_index: int) -> str:
        """Generate deterministic chunk ID based on content hash"""
        # Create a hash of the content for deterministic ID generation
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]
        # Clean source doc name for ID
        clean_source = re.sub(r'[^\w\-_]', '_', source_doc)
        return f"{clean_source}_chunk_{chunk_index}_{content_hash}"
    
    def _find_existing_document_chunks(self, content: str, source_doc: str) -> List[str]:
        """Find all existing chunks for a document by checking cache index for matching source"""
        
        
        existing_chunk_ids = []
        
        try:
            # Look through cache index for chunks from this source document
            with self.cache.lock:
                for chunk_key, chunk_info in self.cache.index.items():
                    if chunk_info.get("source_document") == source_doc:
                        chunk_id = chunk_info.get("chunk_id")
                        if chunk_id:
                            existing_chunk_ids.append(chunk_id)
            
            if existing_chunk_ids:
                logger.info(f"Found {len(existing_chunk_ids)} existing chunks for document {source_doc}")
                return existing_chunk_ids
            
            # Fallback: Check if the exact document content exists as a single chunk
            cached_chunk = self.cache.get_chunk_by_content_hash(content, source_doc)
            if cached_chunk:
                logger.info(f"Found single cached chunk for entire document {source_doc}")
                return [cached_chunk.chunk_id]
                
        except Exception as e:
            logger.warning(f"Error checking for existing chunks: {e}")
        
        return []
    
    def _process_single_document(self, document: Dict[str, Any], 
                               similarity_threshold: float,
                               batch_size: int) -> Dict[str, Any]:
        """Process a single document using sequential chunking approach"""
        
        
        content = document.get("content", "")
        source_doc = document.get("source", "unknown")
        
        
        if not content.strip():
            return {"new_chunks": [], "cached_chunks": []}
        
        # Create proper SourceInfo
        source_info = self._create_source_info(document)
        
        # NEW APPROACH: Check if document chunks already exist by looking for any chunk from this document
        existing_chunks = self._find_existing_document_chunks(content, source_doc)
        if existing_chunks:
            logger.info(f"Found {len(existing_chunks)} existing chunks for document {source_doc}")
            return {"new_chunks": [], "cached_chunks": existing_chunks}
        
        try:
            # Sequential chunking approach - now returns chunks AND their embeddings
            text_chunks, chunk_embeddings = self._chunk_text_sequential(content, source_doc, batch_size)
           
            if not text_chunks:
                logger.warning(f"No chunks created for {source_doc}")
            return {"new_chunks": [], "cached_chunks": []}
        
            new_chunks = []
                
            # Store new chunks with deterministic IDs as DocumentChunk objects
            # Use the pre-computed embeddings from chunking process
            for i, (chunk_content, chunk_embedding) in enumerate(zip(text_chunks, chunk_embeddings)):
                # Generate deterministic chunk ID
                chunk_id = self._generate_deterministic_chunk_id(chunk_content, source_doc, i)
                
                # Extract sentences from chunk
                chunk_sentences = self._split_into_sentences(chunk_content)
                
                # Extract keywords
                keywords = self._extract_keywords(chunk_content)
                
                # Create DocumentChunk object
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                        content=chunk_content,
                    source_info=source_info,
                    sentences=chunk_sentences,
                    keywords=keywords,
                    summary=chunk_content[:200] + "..." if len(chunk_content) > 200 else chunk_content,
                        embedding=chunk_embedding.tolist(),
                    merged_sentence_count=len(chunk_sentences)
                    )
                    
                    chunk_key = self.cache.store_chunk(chunk)
                    if chunk_key:
                        new_chunks.append(chunk.chunk_id)
            
            except Exception as e:
            logger.error(f"Error processing document on GPU {self.gpu_id}: {e}")
            return {"new_chunks": [], "cached_chunks": []}
        
        return {
            "new_chunks": new_chunks,
            "cached_chunks": []
        }
    
    def _extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract keywords from text"""
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Simple stopwords (can be enhanced)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'shall', 'this', 'that', 'these', 'those'}
        
        # Remove stopwords and short words
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count frequency
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top_k
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_k]]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using proper regex pattern with smart merging for short sentences"""
        import re
        
        # Use the same pattern as document processor for consistency
        sentences = re.split(r'(?<=[.?!])\s+', text)
        
        
        # Clean sentences first
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence: # Only add non-empty sentences
                cleaned_sentences.append(sentence)
        
        if not cleaned_sentences:
            return []
        
        # Merge short sentences with adjacent ones
        merged_sentences = []
        i = 0
        while i < len(cleaned_sentences):
            current_sentence = cleaned_sentences[i]
            
            # If sentence is shorter than 25 characters, merge it
            if len(current_sentence) < 25:
                if i < len(cleaned_sentences) - 1:
                    # Merge with next sentence
                    next_sentence = cleaned_sentences[i + 1]
                    merged_sentence = current_sentence + " " + next_sentence
                    merged_sentences.append(merged_sentence)
                    i += 2 # Skip the next sentence since we merged it
                elif merged_sentences:
                    # Last sentence is short, merge with previous sentence
                    previous_sentence = merged_sentences.pop()
                    merged_sentence = previous_sentence + " " + current_sentence
                    merged_sentences.append(merged_sentence)
                    i += 1
                else:
                    # Only one short sentence, keep it as is
                    merged_sentences.append(current_sentence)
                    i += 1
            else:
                # Sentence is long enough, keep as is
                merged_sentences.append(current_sentence)
                i += 1
        
        
        return merged_sentences
    
    def _combine_sentences(self, sentences: List[str]) -> List[str]:
        """Combine each sentence with its preceding and following sentences for context"""
        combined_sentences = []
        for i in range(len(sentences)):
            combined_sentence = sentences[i]
            
            # Add previous sentence for context
            if i > 0:
                combined_sentence = sentences[i-1] + ' ' + combined_sentence
            
            # Add next sentence for context
            if i < len(sentences) - 1:
                combined_sentence += ' ' + sentences[i+1]
            
            combined_sentences.append(combined_sentence)
        
        return combined_sentences
    
    def _calculate_cosine_distances(self, embeddings: List[List[float]]) -> List[float]:
        """Calculate cosine distances between adjacent sentence embeddings"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            # Fallback to numpy implementation if sklearn not available
            def cosine_similarity(a, b):
                a = np.array(a)
                b = np.array(b)
                return np.dot(a, b.T) / (np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1, keepdims=True))
        
        distances = []
        for i in range(len(embeddings) - 1):
            similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            distance = 1 - similarity
            distances.append(distance)
        return distances
    
    def _chunk_text_sequential(self, text: str, source_name: str, batch_size: int,
                              breakpoint_percentile_threshold: int = 80) -> Tuple[List[str], np.ndarray]:
        """
        Chunk text using sequential approach with percentile-based breakpoints
        
        Args:
            text: Input text to chunk
            source_name: Name of source for logging
            batch_size: Batch size for embedding computation
            breakpoint_percentile_threshold: Percentile threshold for identifying breakpoints (default: 80)
            
        Returns:
            Tuple of (text_chunks, final_embeddings) with post-processing applied
        """
        
        
       
        
        # Step 1: Split text into sentences
        single_sentences_list = self._split_into_sentences(text)
        logger.info(f"Split text into {len(single_sentences_list)} sentences")
        
        
        total_sentences_length = sum(len(s) for s in single_sentences_list)
        
        
        if len(single_sentences_list) <= 1:
            # Handle single sentence case - still need to compute embeddings
            if single_sentences_list:
                cleaned_content = [self._remove_stopwords_from_content(single_sentences_list[0])]
                embeddings = self._compute_embeddings_batch(cleaned_content, batch_size)
                return single_sentences_list, embeddings
            else:
                return [], np.array([])
        
        # Step 2: Create context windows
        combined_sentences = self._combine_sentences(single_sentences_list)
        logger.info(f"Created {len(combined_sentences)} context windows")
        
        
        
        # Step 3: Generate embeddings for context windows
        embeddings = self._compute_embeddings_batch(combined_sentences, batch_size)
        
        # Step 4: Calculate cosine distances between adjacent sentences
        distances = self._calculate_cosine_distances(embeddings.tolist())
        
        # Step 5: Determine breakpoint threshold using percentile
        breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
        
        # Step 6: Find breakpoint indices
        indices_above_thresh = [i for i, distance in enumerate(distances) 
                               if distance > breakpoint_distance_threshold]
        
        logger.info(f"Found {len(indices_above_thresh)} breakpoints using {breakpoint_percentile_threshold}th percentile threshold: {breakpoint_distance_threshold:.4f}")
        
        
        
        # Step 7: Create chunks based on breakpoints
        chunks = []
        start_index = 0
        
        for index in indices_above_thresh:
            chunk = ' '.join(single_sentences_list[start_index:index+1])
            chunks.append(chunk)
            start_index = index + 1
        
        # Add remaining sentences as final chunk
        if start_index < len(single_sentences_list):
            chunk = ' '.join(single_sentences_list[start_index:])
            chunks.append(chunk) 
        logger.info(f"Created {len(chunks)} initial chunks from {len(single_sentences_list)} sentences")
        
        
        for i, chunk in enumerate(chunks):
            chunk_sentences = self._split_into_sentences(chunk)
            
        
        # Step 8: Apply post-processing rules and get embeddings
        processed_chunks, chunk_embeddings = self._apply_chunk_post_processing(chunks, single_sentences_list, batch_size)
        
        logger.info(f"Final result: {len(processed_chunks)} chunks after post-processing")
        
        
        for i, chunk in enumerate(processed_chunks):
            chunk_sentences = self._split_into_sentences(chunk)
            
        
        return processed_chunks, chunk_embeddings
    
    def _apply_chunk_post_processing(self, chunks: List[str], all_sentences: List[str], 
                                   batch_size: int) -> Tuple[List[str], np.ndarray]:
        """Apply post-processing rules for chunk size validation and adjustment with efficient embedding reuse
        
        Returns:
            Tuple of (processed_chunks, final_embeddings)
        """
        
        
        
    
        processed_chunks = chunks.copy()
        changes_made = True
        iteration = 0
        
        # Compute embeddings once for all initial chunks to avoid recomputation
        logger.info(f"Computing embeddings for {len(processed_chunks)} initial chunks for post-processing")
        chunk_embeddings = {}
        
        # Initial embedding computation with stopword removal
        if processed_chunks:
            cleaned_chunks = [self._remove_stopwords_from_content(chunk) for chunk in processed_chunks]
            embeddings = self._compute_embeddings_batch(cleaned_chunks, batch_size)
            for i, chunk in enumerate(processed_chunks):
                chunk_embeddings[chunk] = embeddings[i]
        
        while changes_made and iteration < 2: # Limit iterations to prevent infinite loops
            changes_made = False
            iteration += 1
            new_chunks = []
            new_embeddings = {}
            i = 0
            
            
            while i < len(processed_chunks):
                current_chunk = processed_chunks[i]
                current_sentences = self._split_into_sentences(current_chunk)
                
                
                # Rule 1: If chunk has < 6 sentences, merge with most similar adjacent chunk
                if len(current_sentences) < 6:
                    merge_target = self._find_best_merge_target_with_cache_updated(
                        processed_chunks, new_chunks, chunk_embeddings, new_embeddings, i
                    )
                    if merge_target is not None:
                        # Merge chunks - FIXED: Use new_chunks instead of processed_chunks for merge_target content
                        if merge_target < i:
                            # Get the target chunk content from new_chunks (already processed)
                            target_chunk_content = new_chunks[merge_target]
                            merged_content = target_chunk_content + " " + current_chunk
                            new_chunks[merge_target] = merged_content # Update the target chunk in place
                            # Compute embedding for merged content with stopword removal
                            cleaned_merged_content = self._remove_stopwords_from_content(merged_content)
                            merged_embedding = self._compute_embeddings_batch([cleaned_merged_content], batch_size)[0]
                            new_embeddings[merged_content] = merged_embedding
                            # Remove old embedding for target chunk
                            old_target_content = target_chunk_content
                            if old_target_content in new_embeddings:
                                del new_embeddings[old_target_content]
                            logger.info(f"Merged small chunk ({len(current_sentences)} sentences) with previous chunk")
                        else:
                            # Merging with next chunk - use processed_chunks since next chunk hasn't been processed yet
                            next_chunk_content = processed_chunks[merge_target]
                            merged_content = current_chunk + " " + next_chunk_content
                            new_chunks.append(merged_content)
                            # Compute embedding for merged content with stopword removal
                            cleaned_merged_content = self._remove_stopwords_from_content(merged_content)
                            merged_embedding = self._compute_embeddings_batch([cleaned_merged_content], batch_size)[0]
                            new_embeddings[merged_content] = merged_embedding
                            i += 1 # Skip the next chunk as it's been merged
                            logger.info(f"Merged small chunk ({len(current_sentences)} sentences) with next chunk")
                        changes_made = True
                    else:
                        new_chunks.append(current_chunk)
                        new_embeddings[current_chunk] = chunk_embeddings[current_chunk]
                
                # Rule 2: If chunk has > 20 sentences, split into 2 chunks
                elif len(current_sentences) > 20:
                    split_point = len(current_sentences) // 2
                    chunk1 = ' '.join(current_sentences[:split_point])
                    chunk2 = ' '.join(current_sentences[split_point:])
                    new_chunks.extend([chunk1, chunk2])
                    
                    # Compute embeddings for split chunks with stopword removal
                    cleaned_chunk1 = self._remove_stopwords_from_content(chunk1)
                    cleaned_chunk2 = self._remove_stopwords_from_content(chunk2)
                    split_embeddings = self._compute_embeddings_batch([cleaned_chunk1, cleaned_chunk2], batch_size)
                    new_embeddings[chunk1] = split_embeddings[0]
                    new_embeddings[chunk2] = split_embeddings[1]
                    
                    logger.info(f"Split large chunk ({len(current_sentences)} sentences) into 2 chunks")
                    changes_made = True
                
                else:
                    new_chunks.append(current_chunk)
                    new_embeddings[current_chunk] = chunk_embeddings[current_chunk]
                
                i += 1
            
            processed_chunks = new_chunks
            chunk_embeddings = new_embeddings
            logger.info(f"Post-processing iteration {iteration}: {len(processed_chunks)} chunks")
        
        # Convert embeddings dict to numpy array in the same order as processed_chunks
        final_embeddings = np.array([chunk_embeddings[chunk] for chunk in processed_chunks])
        
        
        
        return processed_chunks, final_embeddings
    
    def _find_best_merge_target_with_cache_updated(self, original_chunks: List[str], current_new_chunks: List[str], 
                                                 chunk_embeddings: Dict[str, np.ndarray], 
                                                 new_embeddings: Dict[str, np.ndarray], 
                                                 chunk_index: int) -> Optional[int]:
        """Find the best adjacent chunk to merge with using cached embeddings, considering current processing state"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            # Fallback to numpy implementation if sklearn not available
            def cosine_similarity(a, b):
                a = np.array(a)
                b = np.array(b)
                return np.dot(a, b.T) / (np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1, keepdims=True))
        
        current_chunk = original_chunks[chunk_index]
        current_embedding = chunk_embeddings.get(current_chunk)
        
        if current_embedding is None:
            return None # Can't compute similarity without embedding
        
        best_similarity = -1
        best_target = None
        
        # Check previous chunk (use current_new_chunks since it's already processed)
        if chunk_index > 0 and len(current_new_chunks) > chunk_index - 1:
            prev_chunk_content = current_new_chunks[chunk_index - 1]
            prev_embedding = new_embeddings.get(prev_chunk_content)
            if prev_embedding is not None:
                try:
                    similarity = cosine_similarity([current_embedding], [prev_embedding])[0][0]
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_target = chunk_index - 1
                except Exception as e:
                    pass # Continue if similarity computation fails
        
        # Check next chunk (use original_chunks since it hasn't been processed yet)
        if chunk_index < len(original_chunks) - 1:
            next_chunk = original_chunks[chunk_index + 1]
            next_embedding = chunk_embeddings.get(next_chunk)
            if next_embedding is not None:
                try:
                    similarity = cosine_similarity([current_embedding], [next_embedding])[0][0]
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_target = chunk_index + 1
                except Exception as e:
                    pass # Continue if similarity computation fails
        
        return best_target

    def _find_best_merge_target_with_cache(self, chunks: List[str], chunk_embeddings: Dict[str, np.ndarray], 
                                          chunk_index: int) -> Optional[int]:
        """Find the best adjacent chunk to merge with using cached embeddings - DEPRECATED, use _find_best_merge_target_with_cache_updated"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            # Fallback to numpy implementation if sklearn not available
            def cosine_similarity(a, b):
                a = np.array(a)
                b = np.array(b)
                return np.dot(a, b.T) / (np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1, keepdims=True))
        
        current_chunk = chunks[chunk_index]
        current_embedding = chunk_embeddings.get(current_chunk)
        
        if current_embedding is None:
            return None # Can't compute similarity without embedding
        
        best_similarity = -1
        best_target = None
        
        # Check previous chunk
        if chunk_index > 0:
            prev_chunk = chunks[chunk_index - 1]
            prev_embedding = chunk_embeddings.get(prev_chunk)
            if prev_embedding is not None:
                try:
                    similarity = cosine_similarity([current_embedding], [prev_embedding])[0][0]
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_target = chunk_index - 1
                except Exception as e:
                    pass # Continue if similarity computation fails
        
        # Check next chunk
        if chunk_index < len(chunks) - 1:
            next_chunk = chunks[chunk_index + 1]
            next_embedding = chunk_embeddings.get(next_chunk)
            if next_embedding is not None:
                try:
                    similarity = cosine_similarity([current_embedding], [next_embedding])[0][0]
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_target = chunk_index + 1
                except Exception as e:
                    pass # Continue if similarity computation fails
        
        return best_target
    
    def _compute_embeddings_batch(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Optimized embedding computation with memory management"""
        
        
        if not texts:
            return np.array([])
        
        # Create optimized dataset and dataloader
        dataset = GPUEmbeddingDataset(texts)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            num_workers=0, # Use 0 to avoid forking issues with tokenizers
            pin_memory=True,
            shuffle=False
        )
        
        all_embeddings = []
        
        with torch.no_grad(), torch.amp.autocast('cuda'): # Mixed precision
            for batch_texts in dataloader:
                try:
                    embeddings = self.model.encode(
                        batch_texts,
                        convert_to_tensor=True,
                        show_progress_bar=False,
                        batch_size=len(batch_texts), # Process entire batch at once
                        normalize_embeddings=True # Normalize for better similarity computation
                    )
                    all_embeddings.append(embeddings.cpu().numpy())
                
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"GPU {self.gpu_id} OOM, reducing batch size")
                    torch.cuda.empty_cache()
                    # Process with smaller batches
                    for text in batch_texts:
                        embedding = self.model.encode(
                            [text],
                            convert_to_tensor=True,
                            show_progress_bar=False,
                            normalize_embeddings=True
                        )
                        all_embeddings.append(embedding.cpu().numpy())
        
        return np.vstack(all_embeddings) if all_embeddings else np.array([])
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        cache_stats = self.cache.get_cache_stats()
        
        return {
            **self.processing_stats,
            **cache_stats,
            "gpu_id": self.gpu_id,
            "gpu_memory_total": torch.cuda.get_device_properties(self.gpu_id).total_memory,
            "gpu_memory_allocated": torch.cuda.memory_allocated(self.gpu_id),
            "gpu_memory_cached": torch.cuda.memory_reserved(self.gpu_id)
        }
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Remote method to get chunk by ID from worker cache"""
        return self.cache.get_chunk_by_id(chunk_id)
    
    def _remove_stopwords_from_content(self, text: str) -> str:
        """
        Remove stopwords from chunk content before embedding generation.
        This is the final step before creating embeddings.
        
        Args:
            text: Original chunk content
            
        Returns:
            Content with stopwords removed
        """
        if not text.strip():
            return text
        
        stop_words = ENGLISH_STOPWORDS
    
        # Split into words while preserving sentence structure
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stopwords and very short words
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Join back with spaces
        cleaned_content = ' '.join(filtered_words)
        
        return cleaned_content if cleaned_content.strip() else text # Fallback to original if empty

class DocumentProcessingOrchestrator:
    """Advanced orchestrator for parallel document processing across multiple GPUs"""
    
    def __init__(self, num_gpus: int = None, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cache_base_dir: str = "output"):
        
        
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.model_name = model_name
        self.cache_base_dir = cache_base_dir
        
        # Initialize Ray with GPU allocation
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_gpus=self.num_gpus)
        
        # Initialize workers with proper GPU allocation
        self.workers = []
        self._initialize_workers()
        
        # Performance tracking
        self.total_stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "cache_hits": 0,
            "total_processing_time": 0,
            "gpu_utilization": {}
        }
        
        logger.info(f"Initialized orchestrator with {len(self.workers)} GPU workers")
    
    def _initialize_workers(self):
        """Initialize GPU workers with error handling"""
        
        
        for gpu_id in range(self.num_gpus):
            try:
                cache_dir = f"{self.cache_base_dir}/chunks_cache_gpu_{gpu_id}"
                worker = ParallelDocumentProcessor.remote(gpu_id, self.model_name, cache_dir)
                self.workers.append(worker)
                logger.info(f"Successfully initialized worker on GPU {gpu_id} with cache: {cache_dir}")
            except Exception as e:
                logger.error(f"Failed to initialize worker on GPU {gpu_id}: {e}")
        
        if not self.workers:
            raise RuntimeError("No GPU workers could be initialized")
    
    def process_documents(self, documents: List[Dict[str, Any]], 
                         similarity_threshold: float = 0.8,
                         batch_size: int = None,
                         max_retries: int = 3) -> Dict[str, Any]:
        """Process documents with intelligent load balancing and fault tolerance"""
        
        if not documents:
            return {"processed_chunks": [], "cached_chunks": [], "total_documents": 0}
        
        # Intelligent batch distribution based on GPU capabilities
        document_batches = self._distribute_documents_intelligently(documents)
        
        # Submit tasks with retry logic
        all_results = []
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                results = self._submit_and_collect_tasks(
                    document_batches, similarity_threshold, batch_size
                )
                all_results = results
                break
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"Processing attempt {retry_count} failed: {e}")
                if retry_count < max_retries:
                    logger.info("Retrying with reduced batch sizes...")
                    # Reduce batch sizes for retry
                    document_batches = self._redistribute_for_retry(document_batches)
                else:
                    logger.error("Max retries exceeded, processing failed")
                    raise
        
        # Aggregate and analyze results
        aggregated_results = self._aggregate_results(all_results, documents)
        
        # Update global stats
        self._update_global_stats(aggregated_results)
        
        return aggregated_results
    
    def _distribute_documents_intelligently(self, documents: List[Dict[str, Any]]) -> List[List[Dict]]:
        """Intelligently distribute documents based on GPU capabilities and content size"""
        
        
        if not self.workers:
            raise RuntimeError("No workers available")
        
        # Calculate document complexities (rough estimate based on content length)
        doc_complexities = []
        for doc in documents:
            content_length = len(doc.get("content", ""))
            complexity = min(10, max(1, content_length // 1000)) # Scale 1-10
            doc_complexities.append(complexity)
        
        # Distribute documents to balance total complexity across GPUs
        num_workers = len(self.workers)
        worker_loads = [0] * num_workers
        worker_batches = [[] for _ in range(num_workers)]
        
        # Sort documents by complexity (descending) for better load balancing
        sorted_docs = sorted(zip(documents, doc_complexities), key=lambda x: x[1], reverse=True)
        
        for doc, complexity in sorted_docs:
            # Assign to worker with minimum current load
            min_load_worker = min(range(num_workers), key=lambda i: worker_loads[i])
            worker_batches[min_load_worker].append(doc)
            worker_loads[min_load_worker] += complexity
        
        # Log distribution
        for i, batch in enumerate(worker_batches):
            if batch:
                logger.info(f"GPU {i}: {len(batch)} documents (complexity: {worker_loads[i]})")
        
        return worker_batches
    
    def _submit_and_collect_tasks(self, document_batches: List[List[Dict]], 
                                 similarity_threshold: float, 
                                 batch_size: int) -> List[Dict]:
        """Submit tasks and collect results with progress tracking"""
        
        
        # Submit tasks to workers
        futures = []
        for i, batch in enumerate(document_batches):
            if batch and i < len(self.workers):
                future = self.workers[i].process_document_batch.remote(
                    batch, similarity_threshold, batch_size
                )
                futures.append((i, future))
        
        # Collect results with progress tracking
        results = []
        completed = 0
        total_tasks = len(futures)
        
        logger.info(f"Submitted {total_tasks} tasks to GPU workers")
        
        while futures:
            # Check for completed tasks (non-blocking)
            ready_futures = []
            remaining_futures = []
            
            for worker_id, future in futures:
                try:
                    result = ray.get(future, timeout=0.1) # Quick check
                    ready_futures.append((worker_id, result))
                    completed += 1
                except ray.exceptions.GetTimeoutError:
                    remaining_futures.append((worker_id, future))
                except Exception as e:
                    logger.error(f"Worker {worker_id} failed: {e}")
                    completed += 1
            
            # Process ready results
            for worker_id, result in ready_futures:
                results.append(result)
                logger.info(f"GPU {worker_id} completed: {result['total_documents']} docs, "
                           f"{len(result['processed_chunks'])} new chunks")
            
            futures = remaining_futures
            
            # Progress update
            if ready_futures:
                progress = (completed / total_tasks) * 100
                logger.info(f"Processing progress: {progress:.1f}% ({completed}/{total_tasks})")
            
            time.sleep(0.5) # Brief pause to prevent busy waiting
        
        return results
    
    def _redistribute_for_retry(self, original_batches: List[List[Dict]]) -> List[List[Dict]]:
        """Redistribute documents with smaller batches for retry"""
        all_docs = []
        for batch in original_batches:
            all_docs.extend(batch)
        
        # Create smaller, more manageable batches
        smaller_batch_size = max(1, len(all_docs) // (len(self.workers) * 2))
        new_batches = []
        
        for i in range(len(self.workers)):
            start_idx = i * smaller_batch_size
            end_idx = min(start_idx + smaller_batch_size, len(all_docs))
            if start_idx < len(all_docs):
                new_batches.append(all_docs[start_idx:end_idx])
            else:
                new_batches.append([])
        
        return new_batches
    
    def _aggregate_results(self, results: List[Dict], original_documents: List[Dict]) -> Dict[str, Any]:
        """Aggregate results from all workers with comprehensive analysis"""
        
        total_results = {
            "processed_chunks": [],
            "cached_chunks": [],
            "total_documents": len(original_documents),
            "total_processing_time": 0,
            "average_processing_time": 0,
            "cache_hit_ratio": 0,
            "gpu_utilization": {},
            "performance_stats": {},
            "worker_results": results
        }
        
        if not results:
            return total_results
        
        # Aggregate basic stats
        for result in results:
            total_results["processed_chunks"].extend(result.get("processed_chunks", []))
            total_results["cached_chunks"].extend(result.get("cached_chunks", []))
            total_results["total_processing_time"] += result.get("processing_time", 0)
        
        # Calculate derived metrics
        total_chunks = len(total_results["processed_chunks"]) + len(total_results["cached_chunks"])
        if total_chunks > 0:
            total_results["cache_hit_ratio"] = len(total_results["cached_chunks"]) / total_chunks
        
        if results:
            total_results["average_processing_time"] = total_results["total_processing_time"] / len(results)
        
        # GPU utilization stats
        for i, result in enumerate(results):
            if "gpu_utilization" in result:
                total_results["gpu_utilization"][f"gpu_{i}"] = result["gpu_utilization"]
        
        # Performance analysis
        total_results["performance_stats"] = {
            "documents_per_second": len(original_documents) / max(1, total_results["total_processing_time"]),
            "chunks_per_second": total_chunks / max(1, total_results["total_processing_time"]),
            "cache_efficiency": total_results["cache_hit_ratio"],
            "parallel_efficiency": len(results) / max(1, total_results["total_processing_time"])
        }
        
        return total_results
    
    def _update_global_stats(self, results: Dict[str, Any]):
        """Update global processing statistics"""
        self.total_stats["documents_processed"] += results["total_documents"]
        self.total_stats["chunks_created"] += len(results["processed_chunks"])
        self.total_stats["cache_hits"] += len(results["cached_chunks"])
        self.total_stats["total_processing_time"] += results["total_processing_time"]
        
        # Update GPU utilization
        for gpu_key, utilization in results.get("gpu_utilization", {}).items():
            if gpu_key not in self.total_stats["gpu_utilization"]:
                self.total_stats["gpu_utilization"][gpu_key] = []
            self.total_stats["gpu_utilization"][gpu_key].append(utilization)
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get detailed statistics from all workers"""
        stats_futures = [worker.get_processing_stats.remote() for worker in self.workers]
        worker_stats = ray.get(stats_futures)
        
        return {
            "global_stats": self.total_stats,
            "worker_stats": worker_stats,
            "total_workers": len(self.workers)
        }
    
    def shutdown(self):
        """Gracefully shutdown the orchestrator"""
        
        
        logger.info("Shutting down document processing orchestrator...")
        
        try:
            # Get final stats from workers
            final_stats = self.get_worker_stats()
            logger.info(f"Final processing stats: {final_stats['global_stats']}")
            
            # Shutdown Ray
            ray.shutdown()
            logger.info("Orchestrator shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all workers"""
        
        
        health_status = {
            "healthy_workers": 0,
            "total_workers": len(self.workers),
            "gpu_memory_usage": {},
            "worker_status": []
        }
        
        try:
            stats_futures = [worker.get_processing_stats.remote() for worker in self.workers]
            worker_stats = ray.get(stats_futures, timeout=10)
            
            for i, stats in enumerate(worker_stats):
                is_healthy = stats.get("gpu_memory_allocated", 0) < stats.get("gpu_memory_total", 1) * 0.95
                
                worker_info = {
                    "worker_id": i,
                    "gpu_id": stats.get("gpu_id", i),
                    "healthy": is_healthy,
                    "memory_usage_percent": (stats.get("gpu_memory_allocated", 0) / 
                                           max(1, stats.get("gpu_memory_total", 1))) * 100,
                    "cache_size_mb": stats.get("cache_size_mb", 0)
                }
                
                health_status["worker_status"].append(worker_info)
                if is_healthy:
                    health_status["healthy_workers"] += 1
                
                health_status["gpu_memory_usage"][f"gpu_{i}"] = worker_info["memory_usage_percent"]
        
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status["error"] = str(e)
        
        return health_status

def main():
    """Example usage"""
    
    # Sample documents
    documents = [
        {"content": "This is a sample document about machine learning.", "source": "doc1"},
        {"content": "Natural language processing is fascinating.", "source": "doc2"},
        # Add more documents...
    ]
    
    # Initialize orchestrator
    orchestrator = DocumentProcessingOrchestrator()
    
    try:
        # Process documents
        results = orchestrator.process_documents(documents)
        
        logger.info(f"Processing complete:")
        logger.info(f"- New chunks: {len(results['processed_chunks'])}")
        logger.info(f"- Cached chunks: {len(results['cached_chunks'])}")
        logger.info(f"- Total time: {results['total_processing_time']:.2f}s")
        
    finally:
        orchestrator.shutdown()

if __name__ == "__main__":
    main()
