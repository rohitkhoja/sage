"""
Document processor for chunking text documents using sequential chunking approach
"""

import re
import uuid
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
import os
import json
from .base_processor import BaseProcessor
from .embedding_service import EmbeddingService
from src.core.models import DocumentChunk, SourceInfo, ChunkType, ProcessingConfig
from .data.stopwords_loader import ENGLISH_STOPWORDS


class DocumentProcessor(BaseProcessor):
    """
    Processes documents using sequential chunking approach:
    1. Breaking into sentences
    2. Creating context windows (sentence + neighbors)
    3. Calculating semantic distances between adjacent sentences
    4. Identifying breakpoints using percentile threshold
    5. Creating chunks based on breakpoints
    """
    
    def __init__(self, config: ProcessingConfig, embedding_service: EmbeddingService):
        super().__init__(config)
        self.embedding_service = embedding_service
        
        # Load stopwords from local file
        if config.remove_stopwords:
            self.stop_words = ENGLISH_STOPWORDS
        else:
            self.stop_words = set()
        
        # Initialize tracking for sequential chunking
        self.distance_log_file = "output/sentence_distances.json"
        self.chunks_content_file = "output/document_chunks.json"
        self.distance_data = []
        self.chunks_data = []
        self._initialize_json_files()
    
    def _initialize_json_files(self):
        """Initialize JSON files for logging"""
        os.makedirs("output", exist_ok=True)
        
        # Initialize distance tracking file
        with open(self.distance_log_file, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2)
        
        # Initialize chunks content file
        with open(self.chunks_content_file, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2)
    
    def _log_distance(self, source_name: str, idx: int, sent_current: str, sent_next: str, 
                     distance: float, threshold: float):
        """Log distance between adjacent sentences"""
        distance_entry = {
            "source_name": source_name,
            "sentence_idx": int(idx),
            "sentence_current": sent_current[:200] + "..." if len(sent_current) > 200 else sent_current,
            "sentence_next": sent_next[:200] + "..." if len(sent_next) > 200 else sent_next,
            "cosine_distance": float(round(distance, 4)),
            "above_breakpoint_threshold": bool(distance > threshold),
            "breakpoint_threshold": float(round(threshold, 4))
        }
        self.distance_data.append(distance_entry)
    
    def _log_chunk_content(self, chunk: DocumentChunk):
        """Log document chunk content and structure"""
        chunk_entry = {
            "chunk_id": chunk.chunk_id,
            "source_name": chunk.source_info.source_name,
            "source_id": chunk.source_info.source_id,
            "chunk_type": chunk.source_info.source_type.value,
            "content": chunk.content,
            "sentences": chunk.sentences,
            "sentence_count": len(chunk.sentences),
            "keywords": chunk.keywords,
            "summary": chunk.summary,
            "merged_sentence_count": chunk.merged_sentence_count,
            "content_length": len(chunk.content),
            "chunk_index": int(chunk.chunk_id.split('_')[-1]) if '_' in chunk.chunk_id else 0
        }
        self.chunks_data.append(chunk_entry)
    
    def _save_distance_data(self):
        """Save accumulated distance data to JSON file"""
        if self.distance_data:
            try:
                # Read existing data or initialize empty list
                try:
                    with open(self.distance_log_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    existing_data = []
                
                # Append new data
                existing_data.extend(self.distance_data)
                
                # Write back to file
                with open(self.distance_log_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
                
                # Clear the buffer
                self.distance_data = []
            except Exception as e:
                logger.error(f"Error saving distance data: {e}")
    
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
    
    def _analyze_distance_distribution(self, source_name: str, breakpoint_threshold: float):
        """Analyze and log distance distribution for this document"""
        if not self.distance_data:
            return
        
        distances = np.array([entry["cosine_distance"] for entry in self.distance_data])
        
        logger.info(f"=== Sequential Chunking Distance Distribution for {source_name} ===")
        logger.info(f"Total sentence transitions: {len(distances)}")
        logger.info(f"Mean distance: {distances.mean():.4f}")
        logger.info(f"Std deviation: {distances.std():.4f}")
        logger.info(f"Min distance: {distances.min():.4f}")
        logger.info(f"Max distance: {distances.max():.4f}")
        logger.info(f"Median distance: {np.median(distances):.4f}")
        
        # Distribution ranges
        ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        for low, high in ranges:
            count = np.sum((distances >= low) & (distances < high))
            percentage = (count / len(distances)) * 100
            logger.info(f"Range [{low:.1f}-{high:.1f}): {count} ({percentage:.1f}%)")
        
        # Above breakpoint threshold analysis
        above_threshold = np.sum(distances > breakpoint_threshold)
        threshold_percentage = (above_threshold / len(distances)) * 100
        logger.info(f"Above breakpoint threshold ({breakpoint_threshold:.4f}): {above_threshold} ({threshold_percentage:.1f}%)")
        
        # Save data to JSON file
        self._save_distance_data()
    
    def validate_input(self, input_data: str) -> bool:
        """Validate that input is a string"""
        return isinstance(input_data, str) and len(input_data.strip()) > 0
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex for end-of-sentence punctuation"""
        # Use regex to split on punctuation followed by whitespace
        sentences = re.split(r'(?<=[.?!])\s+', text)
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
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
        distances = []
        for i in range(len(embeddings) - 1):
            similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            distance = 1 - similarity
            distances.append(distance)
        return distances
    
    def _extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract top keywords from text"""
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stopwords
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        # Count frequency
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top_k
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_k]]
    
    def _generate_summary(self, chunk_text: str, max_length: int = 200) -> str:
        """Generate a simple summary from chunk text"""
        if not chunk_text:
            return ""
        
        # Take first part up to max_length
        if len(chunk_text) <= max_length:
            return chunk_text
        else:
            return chunk_text[:max_length] + "..."
    
    def _chunk_text_sequential(self, text: str, source_name: str, 
                              breakpoint_percentile_threshold: int = 80) -> List[str]:
        """
        Chunk text using sequential approach with percentile-based breakpoints
        
        Args:
            text: Input text to chunk
            source_name: Name of source for logging
            breakpoint_percentile_threshold: Percentile threshold for identifying breakpoints (default: 80)
            
        Returns:
            List of text chunks
        """
        # Step 1: Split text into sentences
        single_sentences_list = self._split_sentences(text)
        logger.info(f"Split text into {len(single_sentences_list)} sentences")
        
        if len(single_sentences_list) <= 1:
            return single_sentences_list
        
        # Step 2: Create context windows
        combined_sentences = self._combine_sentences(single_sentences_list)
        logger.info(f"Created {len(combined_sentences)} context windows")
        
        # Step 3: Generate embeddings for context windows
        embeddings = self.embedding_service.generate_embeddings(combined_sentences)
        
        # Step 4: Calculate cosine distances between adjacent sentences
        distances = self._calculate_cosine_distances(embeddings)
        
        # Step 5: Determine breakpoint threshold using percentile
        breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
        
        # Step 6: Log distances for analysis
        for i, distance in enumerate(distances):
            self._log_distance(
                source_name, i, 
                single_sentences_list[i], 
                single_sentences_list[i+1], 
                distance, 
                breakpoint_distance_threshold
            )
        
        # Analyze distribution
        self._analyze_distance_distribution(source_name, breakpoint_distance_threshold)
        
        # Step 7: Find breakpoint indices
        indices_above_thresh = [i for i, distance in enumerate(distances) 
                               if distance > breakpoint_distance_threshold]
        
        logger.info(f"Found {len(indices_above_thresh)} breakpoints using {breakpoint_percentile_threshold}th percentile threshold: {breakpoint_distance_threshold:.4f}")
        
        # Step 8: Create chunks based on breakpoints
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
        
        logger.info(f"Created {len(chunks)} chunks from {len(single_sentences_list)} sentences")
        return chunks
    
    def _merge_small_chunks(self, chunks: List[DocumentChunk], min_sentences: int = 3) -> List[DocumentChunk]:
        """
        Merge chunks with fewer than min_sentences with their most similar neighbor
        
        Args:
            chunks: List of DocumentChunk objects
            min_sentences: Minimum number of sentences required per chunk
            
        Returns:
            List of DocumentChunk objects after merging small chunks
        """
        if len(chunks) <= 1:
            return chunks
        
        merged_chunks = chunks.copy()
        merge_occurred = True
        merge_count = 0
        
        while merge_occurred:
            merge_occurred = False
            i = 0
            
            while i < len(merged_chunks):
                current_chunk = merged_chunks[i]
                
                # Check if current chunk needs merging
                if len(current_chunk.sentences) < min_sentences:
                    merge_target_idx = None
                    best_similarity = -1
                    
                    # Check previous chunk (if exists)
                    if i > 0:
                        prev_chunk = merged_chunks[i - 1]
                        prev_similarity = self.embedding_service.compute_similarity(
                            current_chunk.embedding, prev_chunk.embedding
                        )
                        if prev_similarity > best_similarity:
                            best_similarity = prev_similarity
                            merge_target_idx = i - 1
                    
                    # Check next chunk (if exists)
                    if i < len(merged_chunks) - 1:
                        next_chunk = merged_chunks[i + 1]
                        next_similarity = self.embedding_service.compute_similarity(
                            current_chunk.embedding, next_chunk.embedding
                        )
                        if next_similarity > best_similarity:
                            best_similarity = next_similarity
                            merge_target_idx = i + 1
                    
                    # Perform merge if target found
                    if merge_target_idx is not None:
                        target_chunk = merged_chunks[merge_target_idx]
                        
                        # Determine merge order (smaller index first)
                        if merge_target_idx < i:
                            # Merge current into previous
                            merged_content = target_chunk.content + " " + current_chunk.content
                            merged_sentences = target_chunk.sentences + current_chunk.sentences
                            keep_idx, remove_idx = merge_target_idx, i
                        else:
                            # Merge next into current
                            merged_content = current_chunk.content + " " + target_chunk.content
                            merged_sentences = current_chunk.sentences + target_chunk.sentences
                            keep_idx, remove_idx = i, merge_target_idx
                        
                        # Generate new embedding for merged content
                        merged_embedding = self.embedding_service.generate_single_embedding(merged_content)
                        
                        # Create new merged chunk
                        merged_chunk = DocumentChunk(
                            chunk_id=f"{merged_chunks[keep_idx].source_info.source_id}_merged_{merge_count}",
                            source_info=merged_chunks[keep_idx].source_info,
                            content=merged_content,
                            sentences=merged_sentences,
                            keywords=self._extract_keywords(merged_content),
                            summary=self._generate_summary(merged_content),
                            embedding=merged_embedding,
                            merged_sentence_count=len(merged_sentences)
                        )
                        
                        # Replace the chunk at keep_idx and remove the other
                        merged_chunks[keep_idx] = merged_chunk
                        merged_chunks.pop(remove_idx)
                        
                        merge_count += 1
                        merge_occurred = True
                        
                        logger.info(f"Merged small chunk ({len(current_chunk.sentences)} sentences) with similarity {best_similarity:.4f}")
                        
                        # Restart from beginning since indices changed
                        break
                
                i += 1
        
        if merge_count > 0:
            logger.info(f"Small chunk merging completed: {merge_count} merges performed")
        
        return merged_chunks
    
    def process(self, source_info: SourceInfo) -> List[DocumentChunk]:
        """
        Process a document file using sequential chunking approach with small chunk merging
        
        Args:
            source_info: SourceInfo object containing metadata and file path or embedded content
            
        Returns:
            List of DocumentChunk objects
        """
        try:
            # Check if content is embedded in source_info, otherwise read from file
            if hasattr(source_info, 'content') and source_info.content:
                content = source_info.content
                logger.info(f"Using embedded content for {source_info.source_name}")
            else:
                # Read file
                with open(source_info.file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.info(f"Read content from file: {source_info.file_path}")
            
            if not self.validate_input(content):
                raise ValueError(f"Invalid input content for {source_info.source_name}")
            
            # Sequential chunking approach
            text_chunks = self._chunk_text_sequential(content, source_info.source_name)
            
            if not text_chunks:
                logger.warning(f"No chunks created for {source_info.file_path}")
                return []
            
            # Create initial DocumentChunk objects
            initial_chunks = []
            for i, chunk_text in enumerate(text_chunks):
                # Generate embedding for the entire chunk content
                chunk_embedding = self.embedding_service.generate_single_embedding(chunk_text)
                
                # Extract sentences from chunk for the sentences field
                chunk_sentences = self._split_sentences(chunk_text)
                
                chunk = DocumentChunk(
                    chunk_id=f"{source_info.source_id}_chunk_{i}",
                    source_info=source_info,
                    content=chunk_text,
                    sentences=chunk_sentences,
                    keywords=self._extract_keywords(chunk_text),
                    summary=self._generate_summary(chunk_text),
                    embedding=chunk_embedding,
                    merged_sentence_count=len(chunk_sentences)
                )
                
                initial_chunks.append(chunk)
            
            logger.info(f"Created {len(initial_chunks)} initial chunks from {source_info.file_path}")
            
            # Merge small chunks (fewer than 3 sentences)
            final_chunks = self._merge_small_chunks(initial_chunks, min_sentences=3)
            
            # Log final chunk content
            for chunk in final_chunks:
                self._log_chunk_content(chunk)
            
            # Save chunks data to JSON file
            self._save_chunks_data()
            
            logger.info(f"Final result: {len(final_chunks)} chunks after small chunk merging from {source_info.file_path}")
            return final_chunks
            
        except Exception as e:
            logger.error(f"Error processing document {source_info.file_path}: {e}")
            raise 