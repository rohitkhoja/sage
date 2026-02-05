"""
Core data models for the RAG application using Pydantic for validation
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import numpy as np


class ChunkType(str, Enum):
    """Enum for different types of chunks"""
    DOCUMENT = "document"
    TABLE = "table"


class SourceInfo(BaseModel):
    """Information about the source of a chunk"""
    source_id: str
    source_name: str
    source_type: ChunkType
    file_path: str
    structural_link: List[str] = Field(default_factory=list, description="List of related source IDs")
    original_source: Optional[str] = Field(default="", description="Original source path")
    additional_information: Optional[str] = Field(default="", description="Additional metadata")
    content: Optional[str] = Field(default=None, description="Embedded content (for documents)")


class DatasetConfig(BaseModel):
    """Configuration for dataset processing"""
    dataset_name: str = Field(..., description="Name of the dataset (e.g., 'ott-qa')")
    dataset_path: str = Field(..., description="Path to the dataset directory")
    metadata_file: str = Field(..., description="Path to the metadata JSON file")
    start_index: Optional[int] = Field(default=0, description="Starting index for processing")
    end_index: Optional[int] = Field(default=None, description="Ending index for processing (None means all)")
    chunk_size: Optional[int] = Field(default=None, description="Number of items to process in one batch")
    filter_source_type: Optional[ChunkType] = Field(default=None, description="Filter by source type (doc/table)")


class ProcessingConfig(BaseModel):
    """Configuration for processing parameters"""
    sentence_similarity_threshold: float = Field(default=0.8, ge=0, le=1)
    table_similarity_threshold: float = Field(default=0.8, ge=0, le=1)
    remove_stopwords: bool = False
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = Field(default=32, ge=1, description="Batch size for embedding generation")
    use_faiss: bool = Field(default=True, description="Use FAISS-accelerated graph building")
    faiss_use_gpu: bool = Field(default=False, description="Use GPU for FAISS index (may have compatibility issues)")
    max_neighbors: int = Field(default=150, ge=1, description="Maximum neighbors to consider per node in FAISS search")


class BaseChunk(BaseModel):
    """Base class for all chunk types"""
    chunk_id: str
    content: str
    source_info: SourceInfo
    keywords: List[str] = Field(default_factory=list)
    summary: str = ""
    embedding: Optional[List[float]] = None


class DocumentChunk(BaseChunk):
    """Document chunk with sentences"""
    sentences: List[str] = Field(default_factory=list)
    merged_sentence_count: int = 1


class TableChunk(BaseChunk):
    """Table chunk with rows and column information"""
    column_headers: List[str] = Field(default_factory=list)
    column_descriptions: List[str] = Field(default_factory=list)
    rows_with_headers: List[Dict[str, Any]] = Field(default_factory=list)
    merged_row_count: int = 1


class EdgeType(str, Enum):
    """Types of edges in the graph"""
    DOCUMENT_TO_DOCUMENT = "document_to_document"
    TABLE_TO_TABLE = "table_to_table"
    TABLE_TO_DOCUMENT = "table_to_document"


class BaseEdgeMetadata(BaseModel):
    """Base metadata for edges"""
    edge_id: str
    source_chunk_id: str
    target_chunk_id: str
    edge_type: EdgeType
    semantic_similarity: float = Field(ge=0)
    shared_keywords: List[str] = Field(default_factory=list)
    
    @field_validator('semantic_similarity')
    @classmethod
    def clamp_similarity(cls, v: float) -> float:
        """Clamp similarity values to [0, 1] to handle floating-point precision issues"""
        return max(0.0, min(1.0, float(v)))


class DocumentToDocumentEdgeMetadata(BaseEdgeMetadata):
    """Metadata for document-to-document edges"""
    edge_type: EdgeType = EdgeType.DOCUMENT_TO_DOCUMENT
    topic_overlap: float = Field(ge=0)
    
    @field_validator('semantic_similarity', 'topic_overlap')
    @classmethod
    def clamp_all_similarities(cls, v: float) -> float:
        """Clamp all similarity values to [0, 1] to handle floating-point precision issues"""
        return max(0.0, min(1.0, float(v)))


class TableToTableEdgeMetadata(BaseEdgeMetadata):
    """Metadata for table-to-table edges"""
    edge_type: EdgeType = EdgeType.TABLE_TO_TABLE
    column_similarity: float = Field(ge=0)
    row_overlap: float = Field(ge=0)
    schema_context: Dict[str, Any] = Field(default_factory=dict)
    title_similarity: float = Field(default=0.0, ge=0)
    description_similarity: float = Field(default=0.0, ge=0)
    
    @field_validator('semantic_similarity', 'column_similarity', 'row_overlap', 'title_similarity', 'description_similarity')
    @classmethod
    def clamp_values(cls, v: float) -> float:
        """Clamp values to [0, 1] to handle floating-point precision issues"""
        return max(0.0, min(1.0, float(v)))


class TableToDocumentEdgeMetadata(BaseEdgeMetadata):
    """Metadata for table-to-document edges"""
    edge_type: EdgeType = EdgeType.TABLE_TO_DOCUMENT
    row_references: List[str] = Field(default_factory=list)
    column_references: List[str] = Field(default_factory=list)
    topic_title_similarity: float = Field(default=0.0)
    topic_summary_similarity: float = Field(default=0.0)
    
    @field_validator('semantic_similarity', 'topic_title_similarity', 'topic_summary_similarity')
    @classmethod
    def clamp_values(cls, v: float) -> float:
        """Clamp values to [0, 1] to handle floating-point precision issues"""
        return max(0.0, min(1.0, float(v)))


class GraphNode(BaseModel):
    """Node in the knowledge graph"""
    node_id: str
    chunk: Union[DocumentChunk, TableChunk]
    connections: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True