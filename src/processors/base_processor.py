"""
Abstract base processor class for ensuring consistent interfaces
"""

from abc import ABC, abstractmethod
from typing import List, Union, Any
from src.core.models import DocumentChunk, TableChunk, ProcessingConfig, SourceInfo


class BaseProcessor(ABC):
    """Abstract base class for all processors"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    @abstractmethod
    def process(self, source_info: SourceInfo) -> Union[List[DocumentChunk], List[TableChunk]]:
        """Process input data using source info and return chunks"""
        pass
    
    @abstractmethod
    def validate_input(self, source_info: SourceInfo) -> bool:
        """Validate input data format using source info"""
        pass 