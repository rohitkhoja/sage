"""
Optimized Embedding service for generating vector representations of text
Multi-GPU support with massive batch processing for maximum performance
"""

from typing import List, Optional
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import numpy as np
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import math
import time

from src.core.models import ProcessingConfig


class OptimizedEmbeddingService:
    """
    High-performance embedding service with multi-GPU support and optimized batching
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.model: Optional[SentenceTransformer] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.use_multi_gpu = self.num_gpus > 1
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model with multi-GPU support"""
        try:
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self.model = SentenceTransformer(self.config.embedding_model)
            
            if self.use_multi_gpu:
                logger.info(f"Enabling DataParallel on {self.num_gpus} GPUs")
                self.model = nn.DataParallel(self.model)
                self.model = self.model.to(self.device)
                logger.info(f"Model loaded with DataParallel on {self.num_gpus} GPUs")
            else:
                self.model = self.model.to(self.device)
                logger.info(f"Model loaded on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def generate_embeddings_bulk(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for ALL texts at once with maximum GPU utilization
        
        Args:
            texts: List of all text strings to process
            
        Returns:
            List of embedding vectors (as lists of floats)
        """
        if not texts:
            return []
        
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")
        
        num_texts = len(texts)
        logger.info(f"🚀 BULK PROCESSING: {num_texts:,} texts with {self.num_gpus} GPUs")
        
        # Calculate optimal batch size based on GPU memory and number of GPUs
        optimal_batch_size = self._calculate_optimal_batch_size(num_texts)
        
        try:
            if num_texts <= optimal_batch_size:
                # Process all at once
                logger.info(f"Processing all {num_texts:,} texts in single massive batch")
                return self._generate_embeddings_massive_batch(texts)
            else:
                # Process in large optimized batches
                logger.info(f"Processing {num_texts:,} texts in batches of {optimal_batch_size:,}")
                return self._generate_embeddings_large_batches(texts, optimal_batch_size)
            
        except Exception as e:
            logger.error(f"Error in bulk embedding generation: {e}")
            raise
    
    def _calculate_optimal_batch_size(self, num_texts: int) -> int:
        """Calculate optimal batch size based on available GPU memory"""
        if not self.use_multi_gpu:
            # Single GPU: conservative batches
            return min(8192, num_texts)
        
        # Multi-GPU with TITAN RTX (24GB each): Very aggressive batches
        # TITAN RTX has ~22GB usable memory, much more than RTX 2080 Ti
        # Conservative batching to avoid OOM (GPU memory limited)
        
        if num_texts > 2000000:  # Massive datasets
            optimal_size = 5000
        elif num_texts > 1000000:  # Very large datasets
            optimal_size = 5000
        elif num_texts > 500000:  # Large datasets  
            optimal_size = 5000
        elif num_texts > 100000:  # Medium datasets
            optimal_size = 5000
        elif num_texts > 10000:  # Small-medium datasets
            optimal_size = 2000
        else:  # Small datasets
            optimal_size = min(1000, num_texts)
        
        logger.info(f"🚀 CONSERVATIVE BATCH: {optimal_size:,} (for {self.num_gpus} GPUs, {num_texts:,} texts)")
        return optimal_size
    
    def _generate_embeddings_massive_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for all texts in one massive batch"""
        logger.info(f"🔥 MASSIVE BATCH: Processing {len(texts):,} texts at once")
        
        # Get the actual model (unwrap DataParallel if needed)
        model_to_use = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        # Use conservative internal batching to avoid OOM (reduced from 2048/1024)
        internal_batch_size = 256 if self.use_multi_gpu else 128
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        
        embeddings = model_to_use.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            device=self.device,
            batch_size=internal_batch_size,
            normalize_embeddings=True  # Pre-normalize for cosine similarity
        )
        
        end_time.record()
        torch.cuda.synchronize()
        
        processing_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        
        # Convert to CPU and then to list
        if embeddings.is_cuda:
            embeddings = embeddings.cpu()
        
        embeddings_list = embeddings.numpy().tolist()
        
        texts_per_second = len(texts) / processing_time
        logger.info(f"✅ MASSIVE BATCH COMPLETE: {len(embeddings_list):,} embeddings in {processing_time:.2f}s ({texts_per_second:.0f} texts/sec)")
        
        return embeddings_list
    
    def _generate_embeddings_large_batches(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """Generate embeddings in large optimized batches with OOM recovery"""
        logger.info(f"🔥 LARGE BATCH PROCESSING: {len(texts):,} texts in batches of {batch_size:,}")
        
        all_embeddings = []
        total_batches = math.ceil(len(texts) / batch_size)
        current_batch_size = batch_size
        
        # Get the actual model (unwrap DataParallel if needed)
        model_to_use = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        # Conservative internal batch sizes to prevent OOM
        internal_batch_size = 1024 if self.use_multi_gpu else 512
        
        total_start_time = time.time()
        
        i = 0
        while i < len(texts):
            batch_texts = texts[i:i + current_batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts):,} texts, internal_batch: {internal_batch_size})")
            
            try:
                # Clear GPU cache before processing
                self._clear_gpu_cache()
                
                batch_embeddings = model_to_use.encode(
                    batch_texts,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    device=self.device,
                    batch_size=internal_batch_size,
                    normalize_embeddings=True
                )
                
                # Convert to CPU and extend results
                if batch_embeddings.is_cuda:
                    batch_embeddings = batch_embeddings.cpu()
                
                all_embeddings.extend(batch_embeddings.numpy().tolist())
                
                # Move to next batch
                i += current_batch_size
                
                # Clear GPU cache after processing
                self._clear_gpu_cache()
                
            except torch.cuda.OutOfMemoryError as e:
                logger.warning(f"⚠️  GPU OOM at batch {batch_num}, reducing batch size...")
                
                # Clear GPU memory
                self._clear_gpu_cache()
                
                # Reduce batch sizes
                current_batch_size = max(current_batch_size // 2, 1024)
                internal_batch_size = max(internal_batch_size // 2, 256)
                
                logger.info(f"🔄 Retrying with smaller batch: {current_batch_size}, internal: {internal_batch_size}")
                
                # Don't increment i, retry with smaller batch
                if current_batch_size < 1024:
                    logger.error(f"❌ Batch size too small ({current_batch_size}), cannot process")
                    raise RuntimeError(f"Cannot process batch due to GPU memory constraints: {e}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing batch {batch_num}: {e}")
                raise
        
        total_time = time.time() - total_start_time
        texts_per_second = len(texts) / total_time if total_time > 0 else 0
        
        logger.info(f"✅ LARGE BATCH COMPLETE: {len(all_embeddings):,} embeddings in {total_time:.2f}s ({texts_per_second:.0f} texts/sec)")
        return all_embeddings
    
    def generate_embeddings(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """
        Backward compatibility method - delegates to optimized bulk processing
        """
        return self.generate_embeddings_bulk(texts)
    
    def compute_batch_similarities_gpu(self, embeddings: List[List[float]], chunk_size: int = 20000) -> np.ndarray:
        """
        Compute all pairwise cosine similarities using optimized GPU operations
        
        Args:
            embeddings: List of embedding vectors (should be pre-normalized)
            chunk_size: Process in chunks to manage memory
            
        Returns:
            Similarity matrix as numpy array
        """
        try:
            if not embeddings:
                return np.array([])
            
            num_embeddings = len(embeddings)
            logger.info(f"🔥 Computing similarity matrix for {num_embeddings:,} embeddings")
            
            # Convert to tensor
            embeddings_tensor = torch.tensor(embeddings, device=self.device, dtype=torch.float32)
            
            # Embeddings should already be normalized, but ensure it
            embeddings_normalized = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)
            
            # For very large matrices, use chunked computation
            if num_embeddings > chunk_size:
                logger.info(f"Using chunked computation with chunks of {chunk_size:,}")
                return self._compute_chunked_similarities_optimized(embeddings_normalized, chunk_size)
            
            # Standard computation for smaller matrices
            logger.info("Computing full similarity matrix")
            similarity_matrix = torch.mm(embeddings_normalized, embeddings_normalized.t())
            
            # Convert back to CPU and numpy
            if similarity_matrix.is_cuda:
                similarity_matrix = similarity_matrix.cpu()
            
            logger.info(f"✅ Computed similarity matrix of shape {similarity_matrix.shape}")
            return similarity_matrix.numpy()
            
        except Exception as e:
            logger.error(f"Error computing batch similarities: {e}")
            raise
    
    def _compute_chunked_similarities_optimized(self, embeddings_normalized: torch.Tensor, chunk_size: int) -> np.ndarray:
        """Optimized chunked similarity computation"""
        num_embeddings = embeddings_normalized.size(0)
        similarity_matrix = np.zeros((num_embeddings, num_embeddings), dtype=np.float32)
        
        total_chunks = math.ceil(num_embeddings / chunk_size)
        logger.info(f"Computing {total_chunks}x{total_chunks} chunks")
        
        chunk_count = 0
        total_chunks_to_compute = total_chunks * total_chunks
        
        for i in range(0, num_embeddings, chunk_size):
            end_i = min(i + chunk_size, num_embeddings)
            chunk_i = embeddings_normalized[i:end_i]
            
            for j in range(0, num_embeddings, chunk_size):
                end_j = min(j + chunk_size, num_embeddings)
                chunk_j = embeddings_normalized[j:end_j]
                
                chunk_count += 1
                if chunk_count % 100 == 0:
                    logger.info(f"Processing chunk {chunk_count}/{total_chunks_to_compute}")
                
                # Compute chunk similarities
                chunk_sim = torch.mm(chunk_i, chunk_j.t())
                
                # Store in numpy array
                if chunk_sim.is_cuda:
                    chunk_sim = chunk_sim.cpu()
                
                similarity_matrix[i:end_i, j:end_j] = chunk_sim.numpy()
            
            # Clear GPU cache every few chunks
            if (i // chunk_size) % 5 == 0:
                self._clear_gpu_cache()
        
        return similarity_matrix
    
    def find_top_k_similar_vectorized(self, query_embeddings: List[List[float]], 
                                    all_embeddings: List[List[float]], k: int = 200) -> List[List[int]]:
        """
        Vectorized top-k similarity search using GPU
        
        Args:
            query_embeddings: Query embedding vectors
            all_embeddings: All embedding vectors to search against
            k: Number of top similar items to return
            
        Returns:
            List of lists containing indices of top-k similar items for each query
        """
        logger.info(f"🔥 Vectorized top-k search: {len(query_embeddings):,} queries x {len(all_embeddings):,} targets")
        
        # Convert to tensors
        query_tensor = torch.tensor(query_embeddings, device=self.device, dtype=torch.float32)
        target_tensor = torch.tensor(all_embeddings, device=self.device, dtype=torch.float32)
        
        # Normalize for cosine similarity
        query_normalized = torch.nn.functional.normalize(query_tensor, p=2, dim=1)
        target_normalized = torch.nn.functional.normalize(target_tensor, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(query_normalized, target_normalized.t())
        
        # Get top-k indices
        _, top_k_indices = torch.topk(similarity_matrix, k, dim=1)
        
        # Convert to CPU and numpy
        if top_k_indices.is_cuda:
            top_k_indices = top_k_indices.cpu()
        
        result = top_k_indices.numpy().tolist()
        logger.info(f"✅ Completed vectorized top-k search")
        
        return result
    
    def _clear_gpu_cache(self):
        """Aggressive GPU cache clearing for large dataset processing"""
        if torch.cuda.is_available():
            # Synchronize all GPUs first
            torch.cuda.synchronize()
            
            # Aggressive cache clearing on all GPUs
            for i in range(self.num_gpus):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            
            # Multiple rounds of garbage collection for large datasets
            for _ in range(3):
                gc.collect()
                time.sleep(0.05)
            
            # Reset memory stats if available
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                for i in range(self.num_gpus):
                    with torch.cuda.device(i):
                        torch.cuda.reset_peak_memory_stats()
            
            # Longer delay for memory stabilization in large batches
            time.sleep(0.2)
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        if self.model is None:
            return {"status": "No model loaded"}
        
        # Get the actual model (unwrap DataParallel if needed)
        actual_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        return {
            "model_name": self.config.embedding_model,
            "device": self.device,
            "num_gpus": self.num_gpus,
            "multi_gpu": self.use_multi_gpu,
            "max_seq_length": getattr(actual_model, 'max_seq_length', 'Unknown'),
            "embedding_dimension": actual_model.get_sentence_embedding_dimension()
        }


# Maintain backward compatibility
class EmbeddingService(OptimizedEmbeddingService):
    """Backward compatibility alias"""
    pass
