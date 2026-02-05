"""
Embedding service for generating vector representations of text
Now uses optimized multi-GPU implementation for maximum performance
"""

from src.processors.embedding_service_optimized import OptimizedEmbeddingService

# Use the optimized version as the main service
class EmbeddingService(OptimizedEmbeddingService):
    """
    Centralized service for generating embeddings using SentenceTransformers
    Now inherits from OptimizedEmbeddingService for maximum performance with multi-GPU support
    """
    pass