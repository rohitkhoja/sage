"""
Table processor for chunking CSV tables into fixed-size row chunks
"""

import uuid
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from loguru import logger
import os
import json
import hashlib

from .base_processor import BaseProcessor
from .embedding_service import EmbeddingService
from src.core.models import TableChunk, SourceInfo, ChunkType, ProcessingConfig


class TableProcessor(BaseProcessor):
    """
    Processes tables using simple row-based chunking:
    1. Break table into 10-row chunks
    2. Each chunk includes column headers
    3. Merge final chunk if it has less than 5 rows
    4. Generate embeddings for each chunk
    """
    
    def __init__(self, config: ProcessingConfig, embedding_service: EmbeddingService):
        super().__init__(config)
        self.embedding_service = embedding_service
        
        # Initialize tracking
        self.chunk_log_file = "output/table_chunks_info.json"
        self.chunks_content_file = "output/table_chunks.json"
        self.chunk_info_data = []
        self.chunks_data = []
        self._initialize_json_files()
    
    def _initialize_json_files(self):
        """Initialize JSON files for logging"""
        os.makedirs("output", exist_ok=True)
        
        # Initialize chunk info file
        with open(self.chunk_log_file, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2)
        
        # Initialize chunks content file
        with open(self.chunks_content_file, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2)
    
    def _log_chunk_info(self, source_name: str, chunk_index: int, row_count: int, 
                       start_row: int, end_row: int, was_merged: bool = False):
        """Log information about created chunks"""
        chunk_info_entry = {
            "source_name": source_name,
            "chunk_index": chunk_index,
            "row_count": row_count,
            "start_row": start_row,
            "end_row": end_row,
            "was_merged": was_merged
        }
        self.chunk_info_data.append(chunk_info_entry)
    
    def _log_chunk_content(self, chunk: TableChunk):
        """Log table chunk content and structure"""
        # Extract chunk index from deterministic ID format: source_chunk_index_hash
        chunk_index = 0
        try:
            chunk_id_parts = chunk.chunk_id.split('_')
            if len(chunk_id_parts) >= 3 and chunk_id_parts[1] == 'chunk':
                # Format: source_chunk_index_hash
                chunk_index = int(chunk_id_parts[2])
        except (ValueError, IndexError):
            # Fallback to 0 if parsing fails
            chunk_index = 0
        
        chunk_entry = {
            "chunk_id": chunk.chunk_id,
            "source_name": chunk.source_info.source_name,
            "source_id": chunk.source_info.source_id,
            "chunk_type": chunk.source_info.source_type.value,
            "content": chunk.content,
            "rows_with_headers": chunk.rows_with_headers,
            "row_count": len(chunk.rows_with_headers),
            "column_headers": chunk.column_headers,
            "column_descriptions": chunk.column_descriptions,
            "keywords": chunk.keywords,
            "summary": chunk.summary,
            "merged_row_count": chunk.merged_row_count,
            "content_length": len(chunk.content),
            "chunk_index": chunk_index
        }
        self.chunks_data.append(chunk_entry)
    
    def _save_chunk_info_data(self):
        """Save accumulated chunk info data to JSON file"""
        if self.chunk_info_data:
            try:
                # Read existing data or initialize empty list
                try:
                    with open(self.chunk_log_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    existing_data = []
                
                # Append new data
                existing_data.extend(self.chunk_info_data)
                
                # Write back to file
                with open(self.chunk_log_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
                
                # Clear the buffer
                self.chunk_info_data = []
            except Exception as e:
                logger.error(f"Error saving chunk info data: {e}")
    
    def _save_chunks_data(self):
        """Save accumulated chunks data to JSON file"""
        if self.chunks_data:
            try:
                # Read existing data or initialize empty list
                try:
                    with open(self.chunks_content_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    existing_data = []
                
                # Append new data
                existing_data.extend(self.chunks_data)
                
                # Write back to file
                with open(self.chunks_content_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
                
                # Clear the buffer
                self.chunks_data = []
            except Exception as e:
                logger.error(f"Error saving chunks data: {e}")
    
    def validate_input(self, source_info: SourceInfo) -> bool:
        """Validate that input is a valid CSV file path"""
        try:
            pd.read_csv(source_info.file_path, nrows=1)
            return True
        except Exception:
            return False
    
    def _rows_to_chunk_text_with_headers(self, rows: List[Dict[str, Any]], column_headers: List[str]) -> str:
        """Convert a group of rows to chunk text representation with headers"""
        chunk_text_parts = []
        
        # Add column headers context
        chunk_text_parts.append(f"Table columns: {', '.join(column_headers)}")
        
        # Add each row with headers
        for i, row_dict in enumerate(rows):
            row_text_parts = []
            for col in column_headers:
                if col in row_dict and pd.notna(row_dict[col]):
                    row_text_parts.append(f"{col}: {row_dict[col]}")
            
            if row_text_parts:
                chunk_text_parts.append(f"Row {i+1}: {' | '.join(row_text_parts)}")
        
        return "\n".join(chunk_text_parts)
    
    def _rows_to_markdown_table(self, rows: List[Dict[str, Any]], column_headers: List[str]) -> str:
        """Convert a group of rows to markdown table format for embedding generation"""
        if not rows or not column_headers:
            return ""
        
        markdown_parts = []
        
        # Create header row
        header_row = " ".join(column_headers)
        markdown_parts.append(header_row)
        
        # Create separator row (space for simplicity)
        separator_row = " ".join([" "] * len(column_headers))
        markdown_parts.append(separator_row)
        
        # Add data rows
        for row_dict in rows:
            row_values = []
            for col in column_headers:
                if col in row_dict and pd.notna(row_dict[col]):
                    row_values.append(str(row_dict[col]))
                else:
                    row_values.append("")
            row_text = " ".join(row_values)
            markdown_parts.append(row_text)
        
        return "\n".join(markdown_parts)
    
    def _extract_keywords_from_rows(self, rows: List[Dict[str, Any]], top_k: int = 10) -> List[str]:
        """Extract keywords from table rows"""
        text_values = []
        for row_dict in rows:
            for value in row_dict.values():
                if isinstance(value, str):
                    text_values.extend(value.lower().split())
                elif isinstance(value, (int, float)) and pd.notna(value):
                    text_values.append(str(value))
        
        # Count frequency and return top keywords
        word_freq = {}
        for word in text_values:
            if len(word) > 2:  # Filter short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_k]]
    
    def _generate_chunk_summary(self, rows: List[Dict[str, Any]], chunk_index: int) -> str:
        """Generate summary for a chunk of rows"""
        if not rows:
            return ""
        
        # Get column names
        columns = list(rows[0].keys()) if rows else []
        
        summary = f"Table chunk {chunk_index + 1} with {len(rows)} rows"
        if columns:
            summary += f" containing columns: {', '.join(columns[:3])}"
            if len(columns) > 3:
                summary += f" and {len(columns) - 3} more"
        
        return summary
    
    def _chunk_table_rows(self, df: pd.DataFrame, source_name: str, 
                         chunk_size: int = 10, min_final_chunk_size: int = 5) -> List[List[Dict[str, Any]]]:
        """
        Chunk table rows into fixed-size chunks
        
        Args:
            df: DataFrame to chunk
            source_name: Name of source for logging
            chunk_size: Number of rows per chunk (default: 10)
            min_final_chunk_size: Minimum size for final chunk, merge if smaller (default: 5)
            
        Returns:
            List of chunks, where each chunk is a list of row dictionaries
        """
        total_rows = len(df)
        chunks = []
        
        # Create chunks of chunk_size
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk_rows = []
            
            for idx in range(start_idx, end_idx):
                chunk_rows.append(df.iloc[idx].to_dict())
            
            chunks.append(chunk_rows)
            
            # Log chunk information
            self._log_chunk_info(
                source_name, 
                len(chunks) - 1, 
                len(chunk_rows), 
                start_idx, 
                end_idx - 1
            )
        
        # Handle final chunk merging
        if len(chunks) > 1 and len(chunks[-1]) < min_final_chunk_size:
            # Merge final chunk with previous chunk
            final_chunk = chunks.pop()
            chunks[-1].extend(final_chunk)
            
            # Log the merge
            self._log_chunk_info(
                source_name, 
                len(chunks) - 1, 
                len(chunks[-1]), 
                start_idx - chunk_size, 
                total_rows - 1, 
                was_merged=True
            )
            
            logger.info(f"Merged final chunk ({len(final_chunk)} rows) with previous chunk")
        
        # Save chunk info data
        self._save_chunk_info_data()
        
        logger.info(f"Created {len(chunks)} chunks from {total_rows} rows (chunk_size: {chunk_size})")
        return chunks
    
    def _infer_column_descriptions(self, df: pd.DataFrame) -> Dict[str, str]:
        """Infer basic descriptions for columns based on data types and content"""
        descriptions = {}
        
        for col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                descriptions[col] = "Empty column"
                continue
            
            dtype = col_data.dtype
            if dtype in ['int64', 'float64']:
                descriptions[col] = f"Numeric column with range {col_data.min()} to {col_data.max()}"
            elif dtype == 'object':
                unique_count = col_data.nunique()
                total_count = len(col_data)
                if unique_count < 10:
                    descriptions[col] = f"Categorical column with {unique_count} unique values"
                else:
                    descriptions[col] = f"Text column with {unique_count}/{total_count} unique values"
            else:
                descriptions[col] = f"Column of type {dtype}"
        
        return descriptions
    
    def _generate_deterministic_chunk_id(self, content: str, source_id: str, chunk_index: int) -> str:
        """Generate deterministic chunk ID based on content hash"""
        # Create a hash of the content for deterministic ID generation
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]
        return f"{source_id}_chunk_{chunk_index}_{content_hash}"
    
    def process(self, source_info: SourceInfo) -> List[TableChunk]:
        """
        Process a CSV table file using fixed-size row chunking
        
        Args:
            source_info: SourceInfo object containing metadata and file path
            
        Returns:
            List of TableChunk objects
        """
        try:
            # Read CSV file
            df = pd.read_csv(source_info.file_path)
            
            if df.empty:
                logger.warning(f"Empty table file: {source_info.file_path}")
                return []
            
            if not self.validate_input(source_info):
                raise ValueError(f"Invalid CSV file: {source_info.file_path}")
            
            # Get column headers and descriptions
            column_headers = df.columns.tolist()
            column_descriptions = self._infer_column_descriptions(df)
            
            logger.info(f"Processing table {source_info.file_path} with {len(df)} rows and {len(column_headers)} columns")
            
            # Chunk table into fixed-size chunks
            row_chunks = self._chunk_table_rows(df, source_info.source_name)
            
            if not row_chunks:
                return []
            
            # Create TableChunk objects with batch embedding generation for better GPU utilization
            chunk_texts = []
            chunk_metadata = []
            
            # First pass: prepare all chunk texts and metadata
            for i, chunk_rows in enumerate(row_chunks):
                chunk_text = self._rows_to_markdown_table(chunk_rows, column_headers)
                chunk_texts.append(chunk_text)
                
                # Store metadata for each chunk
                keywords = self._extract_keywords_from_rows(chunk_rows)
                column_desc_list = [column_descriptions.get(col, "") for col in column_headers]
                
                chunk_metadata.append({
                    'chunk_rows': chunk_rows,
                    'keywords': keywords,
                    'column_desc_list': column_desc_list,
                    'chunk_index': i
                })
            
            logger.info(f"Generating embeddings for {len(chunk_texts)} table chunks using GPU batch processing")
            
            # Generate all embeddings in parallel for better GPU utilization
            chunk_embeddings = self.embedding_service.generate_embeddings(chunk_texts, batch_size=self.config.batch_size)
            
            # Second pass: create TableChunk objects with pre-computed embeddings
            chunks = []
            for i, (chunk_text, chunk_embedding, metadata) in enumerate(zip(chunk_texts, chunk_embeddings, chunk_metadata)):
                chunk = TableChunk(
                    chunk_id=self._generate_deterministic_chunk_id(chunk_text, source_info.source_id, i),
                    content=chunk_text,
                    source_info=source_info,
                    rows_with_headers=metadata['chunk_rows'],
                    column_headers=column_headers,
                    column_descriptions=metadata['column_desc_list'],
                    keywords=metadata['keywords'],
                    summary=self._generate_chunk_summary(metadata['chunk_rows'], i),
                    embedding=chunk_embedding,
                    merged_row_count=len(metadata['chunk_rows'])
                )
                
                # Log chunk content
                self._log_chunk_content(chunk)
                
                chunks.append(chunk)
            
            # Save chunks data to JSON file
            self._save_chunks_data()
            
            logger.info(f"Created {len(chunks)} table chunks from {source_info.file_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing table {source_info.file_path}: {e}")
            raise 