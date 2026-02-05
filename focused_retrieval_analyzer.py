#!/usr/bin/env python3
"""
Focused Retrieval Analysis System

This system analyzes the 10 retrieved chunks for each question to understand
how well neighbor-based filtering can identify gold documents.

Key approach:
1. For each question, analyze only the 10 retrieved chunks
2. Calculate type-specific similarities (doc vs table)
3. Get neighbors of retrieved chunks, filter by 95th percentile
4. Weight neighbors by frequency and similarity scores
5. Analyze gold document presence and margins
"""

import json
import os
import pandas as pd
import networkx as nx
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
import ast

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

@dataclass
class RetrievedChunkMetrics:
    """Metrics for a retrieved chunk relative to the question"""
    chunk_id: str
    chunk_type: str  # "document" or "table"
    rank: int  # 1-10 position in retrieved list
    
    # Content-first strategy results
    winning_approach: str = ""  # "main_question" or "sub_question"
    winning_sub_question: str = ""  # The specific sub-question that won (if applicable)
    
    # Winning similarity scores (only from winning approach)
    content_similarity: float = 0.0
    topic_similarity: float = 0.0  # For documents
    description_similarity: float = 0.0  # For tables
    max_column_similarity: float = 0.0  # For tables
    
    # Comparison scores for content (to determine winning approach)
    main_question_content_similarity: float = 0.0
    max_subquestion_content_similarity: float = 0.0
    
    # Entity/Event matching (from winning approach)
    entity_exact_matches: int = 0
    entity_substring_matches: int = 0
    event_exact_matches: int = 0
    event_substring_matches: int = 0
    
    # Derived metrics
    total_similarity: float = 0.0  # Sum of applicable similarities
    total_entity_matches: int = 0
    total_event_matches: int = 0

@dataclass
class NeighborCandidate:
    """A neighbor candidate with aggregated scores"""
    node_id: str
    node_type: str
    is_gold: bool
    appearance_count: int  # How many retrieved chunks had this as neighbor
    max_similarity: float  # Highest similarity score seen
    avg_similarity: float  # Average similarity across appearances
    
    # Content-first strategy results
    winning_approach: str = ""  # "main_question" or "sub_question"
    winning_sub_question: str = ""  # The specific sub-question that won (if applicable)
    
    # Best individual similarities (from winning approach only)
    best_content_similarity: float = 0.0
    best_topic_similarity: float = 0.0
    best_description_similarity: float = 0.0
    best_column_similarity: float = 0.0
    
    # Comparison scores for content (to determine winning approach)
    best_main_question_content_similarity: float = 0.0
    best_subquestion_content_similarity: float = 0.0
    
    # Entity/Event matching (from winning approach)
    entity_exact_matches: int = 0
    entity_substring_matches: int = 0
    event_exact_matches: int = 0
    event_substring_matches: int = 0
    
    # Gold chunk specific info (if is_gold=True)
    connected_to_retrieved_chunk: str = ""  # Which retrieved chunk this gold connects to
    distance_from_retrieved: int = 1  # Graph distance (usually 1)
    
    # Final weighted score
    weighted_score: float = 0.0

@dataclass 
class QuestionResult:
    """Complete analysis result for one question"""
    question_id: str
    question_text: str
    question_entities: List[str]
    question_events: List[str]
    question_sub_questions: List[str]  # NEW: Sub-questions for enhanced similarity
    
    # Retrieved chunks analysis
    retrieved_chunks: List[RetrievedChunkMetrics]
    
    # Neighbor analysis
    all_neighbor_candidates: List[NeighborCandidate]
    filtered_neighbors: List[NeighborCandidate]  # Final top-20 list
    
    # Gold analysis
    gold_docs: List[str]
    gold_in_neighbors: List[NeighborCandidate]
    gold_missed: List[str]
    gold_margin_analysis: Dict[str, Any]  # gold_id -> comprehensive analysis dict
    
    # Ranking insights
    chunks_ranking_above_gold: List[Dict[str, Any]]  # Chunks that outrank gold
    similarity_factor_analysis: Dict[str, float]  # Which factors help/hurt gold
    
    # Thresholds used
    similarity_95th_percentile: float
    frequency_threshold: int

class GPUEmbeddingService:
    """GPU-accelerated embedding generation service"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 1000, gpu_id: int = 0):
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name, device=self.device)
        self.batch_size = batch_size
        self.embedding_cache = {}
        
        logger.info(f"Initialized GPUEmbeddingService on {self.device}")
    
    def generate_embeddings_batch(self, texts: List[str], cache_key_prefix: str = "") -> torch.Tensor:
        """Generate embeddings for a batch of texts with caching"""
        # Check cache first
        cache_keys = [f"{cache_key_prefix}_{hash(text)}" for text in texts]
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, (text, cache_key) in enumerate(zip(texts, cache_keys)):
            if cache_key in self.embedding_cache:
                cached_embeddings.append((i, self.embedding_cache[cache_key]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        all_embeddings = [None] * len(texts)
        
        # Place cached embeddings
        for idx, embedding in cached_embeddings:
            all_embeddings[idx] = embedding
        
        # Generate new embeddings in batches
        if uncached_texts:
            logger.info(f"Generating {len(uncached_texts)} new embeddings...")
            
            for i in range(0, len(uncached_texts), self.batch_size):
                batch_texts = uncached_texts[i:i + self.batch_size]
                batch_indices = uncached_indices[i:i + self.batch_size]
                
                batch_embeddings = self.model.encode(
                    batch_texts, 
                    convert_to_tensor=True, 
                    show_progress_bar=True,
                    batch_size=min(32, len(batch_texts)),
                    normalize_embeddings=True
                )
                
                # Store in cache and results
                for j, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings)):
                    cache_key = f"{cache_key_prefix}_{hash(text)}"
                    self.embedding_cache[cache_key] = embedding
                    all_embeddings[batch_indices[j]] = embedding
        
        return torch.stack(all_embeddings)
    
    def calculate_similarities(self, texts1: List[str], texts2: List[str], cache_prefix: str) -> np.ndarray:
        """Calculate pairwise similarities between two lists of texts"""
        embeddings1 = self.generate_embeddings_batch(texts1, f"{cache_prefix}_1")
        embeddings2 = self.generate_embeddings_batch(texts2, f"{cache_prefix}_2")
        
        similarities = torch.nn.functional.cosine_similarity(
            embeddings1, embeddings2, dim=1
        ).cpu().numpy()
        
        return similarities
    
    def save_cache(self, cache_file: str):
        """Save embedding cache to file"""
        with open(cache_file, 'wb') as f:
            pickle.dump(self.embedding_cache, f)
        logger.info(f"Saved embedding cache to {cache_file}")
    
    def load_cache(self, cache_file: str):
        """Load embedding cache from file"""
        if Path(cache_file).exists():
            with open(cache_file, 'rb') as f:
                self.embedding_cache = pickle.load(f)
            logger.info(f"Loaded embedding cache from {cache_file} ({len(self.embedding_cache)} entries)")

class FocusedRetrievalAnalyzer:
    """Main analyzer for focused retrieval analysis"""
    
    def __init__(self, 
                 edges_file: str = "output/analysis_cache/knowledge_graph/unique_edges_20250731_173102.json",
                 csv_file: str = "output/dense_sparse_average_results_filtered.csv",
                 questions_metadata_file: str = "output/questions_metadata_output.json",
                 docs_chunks_dir: str = "output/full_pipeline/docs_chunks_1", 
                 table_chunks_file: str = "output/full_pipeline/table_chunks_with_metadata.json",
                 cache_dir: str = "output/focused_retrieval_cache",
                 gpu_id: int = 0):
        
        self.edges_file = edges_file
        self.csv_file = csv_file
        self.questions_metadata_file = questions_metadata_file
        self.docs_chunks_dir = Path(docs_chunks_dir)
        self.table_chunks_file = table_chunks_file
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize services
        self.embedding_service = GPUEmbeddingService(gpu_id=gpu_id)
        self.embedding_service.load_cache(str(self.cache_dir / "embeddings.pkl"))
        
        # Data containers
        self.graph = nx.Graph()
        self.chunk_metadata = {}  # chunk_id -> metadata dict
        self.questions_metadata = {}  # question -> {entities, events}
        self.question_results = []  # List[QuestionResult]
        
        logger.info("Initialized FocusedRetrievalAnalyzer")
    
    def load_all_data(self):
        """Load all required data"""
        logger.info("Loading all data...")
        
        # Load graph
        self._load_graph()
        
        # Load chunk metadata
        self._load_chunk_metadata()
        
        # Load questions metadata
        self._load_questions_metadata()
        
        logger.info("Data loading complete")
    
    def _load_graph(self):
        """Load the knowledge graph from edges file"""
        logger.info(f"Loading graph from: {self.edges_file}")
        
        with open(self.edges_file, 'r') as f:
            edges_data = json.load(f)
        
        # Build graph
        edges_to_add = []
        for edge in edges_data:
            source = edge['source_chunk_id']
            target = edge['target_chunk_id']
            edges_to_add.append((source, target))
        
        self.graph.add_edges_from(edges_to_add)
        
        logger.info(f"Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def _load_chunk_metadata(self):
        """Load metadata for all chunks"""
        logger.info("Loading chunk metadata...")
        
        # Load document chunks
        doc_count = 0
        for json_file in self.docs_chunks_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                chunk_id = data['chunk_id']
                self.chunk_metadata[chunk_id] = {
                    'type': 'document',
                    'content': data.get('content', ''),
                    'topic': data.get('metadata', {}).get('topic', ''),
                    'entities': data.get('metadata', {}).get('entities', {}),
                    'events': data.get('metadata', {}).get('events', {}),
                    'embedding': data.get('embedding', [])
                }
                doc_count += 1
                
            except Exception as e:
                logger.warning(f"Error loading document chunk {json_file}: {e}")
                continue
        
        # Load table chunks
        table_count = 0
        try:
            with open(self.table_chunks_file, 'r') as f:
                table_data = json.load(f)
            
            for chunk_info in table_data:
                chunk_id = chunk_info['chunk_id']
                metadata = chunk_info.get('metadata', {})
                
                self.chunk_metadata[chunk_id] = {
                    'type': 'table',
                    'content': chunk_info.get('content', ''),
                    'topic': metadata.get('table_description', ''),
                    'title': metadata.get('table_title', ''),
                    'description': metadata.get('table_description', ''),
                    'column_descriptions': metadata.get('col_desc', {}),
                    'entities': metadata.get('entities', {}),
                    'events': metadata.get('events', {}),
                    'embedding': chunk_info.get('embedding', [])
                }
                table_count += 1
                
        except Exception as e:
            logger.error(f"Error loading table chunks: {e}")
        
        logger.info(f"Loaded metadata for {doc_count} documents and {table_count} tables")
    
    def _load_questions_metadata(self):
        """Load questions metadata"""
        logger.info(f"Loading questions metadata from: {self.questions_metadata_file}")
        
        with open(self.questions_metadata_file, 'r') as f:
            self.questions_metadata = json.load(f)
        
        logger.info(f"Loaded metadata for {len(self.questions_metadata)} questions")
    
    def safe_parse_list(self, value) -> List[str]:
        """Safely parse a string representation of a list"""
        if pd.isna(value) or value == '':
            return []
        
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed]
                else:
                    return [str(parsed).strip()]
            except (ValueError, SyntaxError):
                return [value.strip()]
        
        if isinstance(value, list):
            return [str(item).strip() for item in value]
        
        return [str(value).strip()]
    
    def analyze_retrieved_chunks(self, question_text: str, retrieved_chunks: List[str], 
                               question_entities: List[str], question_events: List[str],
                               question_sub_questions: List[str]) -> List[RetrievedChunkMetrics]:
        """Analyze the 10 retrieved chunks for type-specific similarities"""
        logger.info(f"Analyzing {len(retrieved_chunks)} retrieved chunks...")
        
        # Separate by type for batch processing
        doc_chunks = []
        table_chunks = []
        chunk_types = {}
        
        for i, chunk_id in enumerate(retrieved_chunks):
            if chunk_id not in self.chunk_metadata:
                continue
                
            chunk_type = self.chunk_metadata[chunk_id]['type']
            chunk_types[chunk_id] = chunk_type
            
            if chunk_type == 'document':
                doc_chunks.append((i, chunk_id))
            else:
                table_chunks.append((i, chunk_id))
        
        # Initialize results
        chunk_metrics = [None] * len(retrieved_chunks)
        
        # Process document chunks
        if doc_chunks:
            self._process_document_chunks(question_text, doc_chunks, question_entities, 
                                        question_events, question_sub_questions, chunk_metrics)
        
        # Process table chunks  
        if table_chunks:
            self._process_table_chunks(question_text, table_chunks, question_entities,
                                     question_events, question_sub_questions, chunk_metrics)
        
        # Filter out None entries
        return [m for m in chunk_metrics if m is not None]
    
    def _process_document_chunks(self, question_text: str, doc_chunks: List[Tuple[int, str]],
                               question_entities: List[str], question_events: List[str],
                               question_sub_questions: List[str], chunk_metrics: List):
        """Process document chunks using content-first strategy"""
        
        # Prepare texts for batch processing
        question_texts = [question_text] * len(doc_chunks)
        content_texts = []
        topic_texts = []
        
        for rank, chunk_id in doc_chunks:
            metadata = self.chunk_metadata[chunk_id]
            content_texts.append(metadata['content'])
            topic_texts.append(metadata['topic'])
        
        # Step 1: Calculate content similarities for both approaches
        main_content_similarities = self.embedding_service.calculate_similarities(
            question_texts, content_texts, "doc_content_main"
        )
        
        # Calculate sub-question content similarities 
        max_subq_content_similarities = []
        winning_sub_questions = []  # Track which sub-question won for each chunk
        
        if question_sub_questions:
            for i, (rank, chunk_id) in enumerate(doc_chunks):
                metadata = self.chunk_metadata[chunk_id]
                chunk_content = metadata['content']
                
                # Calculate similarities for all sub-questions vs this chunk content
                subq_content_sims = self.embedding_service.calculate_similarities(
                    question_sub_questions, [chunk_content] * len(question_sub_questions), f"subq_doc_content_{i}"
                )
                
                # Find maximum similarity and which sub-question achieved it
                max_idx = np.argmax(subq_content_sims)
                max_subq_content_similarities.append(float(subq_content_sims[max_idx]))
                winning_sub_questions.append(question_sub_questions[max_idx])
        else:
            # No sub-questions available
            max_subq_content_similarities = [0.0] * len(doc_chunks)
            winning_sub_questions = [""] * len(doc_chunks)
        
        # Step 2: Determine winning approach per chunk and calculate remaining similarities
        for i, (rank, chunk_id) in enumerate(doc_chunks):
            metadata = self.chunk_metadata[chunk_id]
            
            # Determine winning approach based on content similarity
            main_content_sim = float(main_content_similarities[i])
            subq_content_sim = max_subq_content_similarities[i]
            
            if main_content_sim >= subq_content_sim:
                # Main question wins
                winning_approach = "main_question"
                winning_sub_question = ""
                content_sim = main_content_sim
                
                # Calculate other similarities using main question
                topic_sim = self.embedding_service.calculate_similarities(
                    [question_text], [metadata['topic']], f"doc_topic_main_{i}"
                )[0]
                
                # Entity/event matching using main question entities/events
                entity_exact, entity_substring = self._match_entities_events(
                    question_entities, list(metadata['entities'].keys())
                )
                event_exact, event_substring = self._match_entities_events(
                    question_events, list(metadata['events'].keys())
                )
                
            else:
                # Sub-question wins
                winning_approach = "sub_question"
                winning_sub_question = winning_sub_questions[i]
                content_sim = subq_content_sim
                
                # Calculate other similarities using winning sub-question
                topic_sim = self.embedding_service.calculate_similarities(
                    [winning_sub_question], [metadata['topic']], f"doc_topic_subq_{i}"
                )[0]
                
                # For sub-questions, extract entities/events from the sub-question text
                # (For now, use main question entities/events as proxy)
                entity_exact, entity_substring = self._match_entities_events(
                    question_entities, list(metadata['entities'].keys())
                )
                event_exact, event_substring = self._match_entities_events(
                    question_events, list(metadata['events'].keys())
                )
            
            metrics = RetrievedChunkMetrics(
                chunk_id=chunk_id,
                chunk_type='document',
                rank=rank + 1,
                winning_approach=winning_approach,
                winning_sub_question=winning_sub_question,
                content_similarity=content_sim,
                topic_similarity=float(topic_sim),
                main_question_content_similarity=main_content_sim,
                max_subquestion_content_similarity=subq_content_sim,
                entity_exact_matches=entity_exact,
                entity_substring_matches=entity_substring,
                event_exact_matches=event_exact,
                event_substring_matches=event_substring,
                total_entity_matches=entity_exact + entity_substring,
                total_event_matches=event_exact + event_substring
            )
            
            # Calculate total similarity for documents
            metrics.total_similarity = metrics.content_similarity + metrics.topic_similarity
            
            chunk_metrics[rank] = metrics
    
    def _process_table_chunks(self, question_text: str, table_chunks: List[Tuple[int, str]],
                            question_entities: List[str], question_events: List[str],
                            question_sub_questions: List[str], chunk_metrics: List):
        """Process table chunks using content-first strategy"""
        
        # Prepare texts for batch processing
        question_texts = [question_text] * len(table_chunks)
        content_texts = []
        description_texts = []
        column_texts = []
        
        for rank, chunk_id in table_chunks:
            metadata = self.chunk_metadata[chunk_id]
            content_texts.append(metadata['content'])
            description_texts.append(metadata['description'])
            
            # Find best column description match
            column_descs = metadata.get('column_descriptions', {})
            if column_descs:
                column_text = ' '.join(column_descs.values())
            else:
                column_text = ""
            column_texts.append(column_text)
        
        # Step 1: Calculate content similarities for both approaches
        main_content_similarities = self.embedding_service.calculate_similarities(
            question_texts, content_texts, "table_content_main"
        )
        
        # Calculate sub-question content similarities
        max_subq_content_similarities = []
        winning_sub_questions = []  # Track which sub-question won for each chunk
        
        if question_sub_questions:
            for i, (rank, chunk_id) in enumerate(table_chunks):
                metadata = self.chunk_metadata[chunk_id]
                chunk_content = metadata['content']
                
                # Calculate similarities for all sub-questions vs this chunk content
                subq_content_sims = self.embedding_service.calculate_similarities(
                    question_sub_questions, [chunk_content] * len(question_sub_questions), f"subq_table_content_{i}"
                )
                
                # Find maximum similarity and which sub-question achieved it
                max_idx = np.argmax(subq_content_sims)
                max_subq_content_similarities.append(float(subq_content_sims[max_idx]))
                winning_sub_questions.append(question_sub_questions[max_idx])
        else:
            # No sub-questions available
            max_subq_content_similarities = [0.0] * len(table_chunks)
            winning_sub_questions = [""] * len(table_chunks)
        
        # Step 2: Determine winning approach per chunk and calculate remaining similarities
        for i, (rank, chunk_id) in enumerate(table_chunks):
            metadata = self.chunk_metadata[chunk_id]
            
            # Determine winning approach based on content similarity
            main_content_sim = float(main_content_similarities[i])
            subq_content_sim = max_subq_content_similarities[i]
            
            # Find best column description match
            column_descs = metadata.get('column_descriptions', {})
            if column_descs:
                chunk_columns = ' '.join(column_descs.values())
            else:
                chunk_columns = ""
            
            if main_content_sim >= subq_content_sim:
                # Main question wins
                winning_approach = "main_question"
                winning_sub_question = ""
                content_sim = main_content_sim
                
                # Calculate other similarities using main question
                desc_sim = self.embedding_service.calculate_similarities(
                    [question_text], [metadata['description']], f"table_desc_main_{i}"
                )[0]
                
                column_sim = self.embedding_service.calculate_similarities(
                    [question_text], [chunk_columns], f"table_columns_main_{i}"
                )[0] if chunk_columns else 0.0
                
                # Entity/event matching using main question entities/events
                entity_exact, entity_substring = self._match_entities_events(
                    question_entities, list(metadata['entities'].keys())
                )
                event_exact, event_substring = self._match_entities_events(
                    question_events, list(metadata['events'].keys())
                )
                
            else:
                # Sub-question wins
                winning_approach = "sub_question"
                winning_sub_question = winning_sub_questions[i]
                content_sim = subq_content_sim
                
                # Calculate other similarities using winning sub-question
                desc_sim = self.embedding_service.calculate_similarities(
                    [winning_sub_question], [metadata['description']], f"table_desc_subq_{i}"
                )[0]
                
                column_sim = self.embedding_service.calculate_similarities(
                    [winning_sub_question], [chunk_columns], f"table_columns_subq_{i}"
                )[0] if chunk_columns else 0.0
                
                # For sub-questions, use main question entities/events as proxy
                entity_exact, entity_substring = self._match_entities_events(
                    question_entities, list(metadata['entities'].keys())
                )
                event_exact, event_substring = self._match_entities_events(
                    question_events, list(metadata['events'].keys())
                )
            
            metrics = RetrievedChunkMetrics(
                chunk_id=chunk_id,
                chunk_type='table',
                rank=rank + 1,
                winning_approach=winning_approach,
                winning_sub_question=winning_sub_question,
                content_similarity=content_sim,
                description_similarity=float(desc_sim),
                max_column_similarity=float(column_sim),
                main_question_content_similarity=main_content_sim,
                max_subquestion_content_similarity=subq_content_sim,
                entity_exact_matches=entity_exact,
                entity_substring_matches=entity_substring,
                event_exact_matches=event_exact,
                event_substring_matches=event_substring,
                total_entity_matches=entity_exact + entity_substring,
                total_event_matches=event_exact + event_substring
            )
            
            # Calculate total similarity for tables
            metrics.total_similarity = (metrics.content_similarity + 
                                      metrics.description_similarity + 
                                      metrics.max_column_similarity)
            
            chunk_metrics[rank] = metrics
    
    def _match_entities_events(self, question_items: List[str], node_items: List[str]) -> Tuple[int, int]:
        """Match entities or events between question and node"""
        exact_matches = 0
        substring_matches = 0
        
        for q_item in question_items:
            q_item_lower = q_item.lower().strip()
            
            for n_item in node_items:
                n_item_lower = n_item.lower().strip()
                
                if q_item_lower == n_item_lower:
                    exact_matches += 1
                    break
                elif q_item_lower in n_item_lower:
                    substring_matches += 1
                    break
        
        return exact_matches, substring_matches
    
    def get_top_k_neighbors(self, retrieved_chunks: List[RetrievedChunkMetrics],
                          question_text: str, question_entities: List[str], 
                          question_events: List[str], question_sub_questions: List[str],
                          k_per_chunk: int = 100, final_k: int = 100) -> Tuple[List[NeighborCandidate], float]:
        """Get top-K neighbors per retrieved chunk, then create weighted final top-K list"""
        
        # Step 1: Get top-K neighbors for each retrieved chunk
        per_chunk_top_neighbors = {}  # chunk_id -> list of (neighbor_id, similarity)
        all_neighbor_scores = {}  # neighbor_id -> list of (similarity, source_chunk)
        retrieved_chunk_ids = set([chunk.chunk_id for chunk in retrieved_chunks])
        
        logger.info(f"Getting top {k_per_chunk} neighbors for each of {len(retrieved_chunks)} retrieved chunks...")
        
        for chunk_metrics in retrieved_chunks:
            chunk_id = chunk_metrics.chunk_id
            
            if chunk_id not in self.graph:
                continue
            
            # Get neighbors
            neighbors = set(self.graph.neighbors(chunk_id))
            
            # Remove retrieved chunks from neighbors (avoid analyzing retrieved chunks as neighbors)
            neighbors = neighbors - retrieved_chunk_ids
            
            # Filter neighbors that have metadata
            valid_neighbors = [n for n in neighbors if n in self.chunk_metadata]
            
            if not valid_neighbors:
                continue
            
            # Calculate similarities for these neighbors
            neighbor_similarities = self._calculate_neighbor_similarities(
                question_text, question_entities, question_events, question_sub_questions, valid_neighbors
            )
            
            # Get top-K neighbors for this chunk
            sorted_neighbors = sorted(neighbor_similarities.items(), key=lambda x: x[1], reverse=True)
            top_k_for_chunk = sorted_neighbors[:k_per_chunk]
            per_chunk_top_neighbors[chunk_id] = top_k_for_chunk
            
            # Store in global scores for final ranking
            for neighbor_id, similarity in top_k_for_chunk:
                if neighbor_id not in all_neighbor_scores:
                    all_neighbor_scores[neighbor_id] = []
                all_neighbor_scores[neighbor_id].append((similarity, chunk_id))
        
        # Step 2: Create weighted final ranking
        neighbor_candidates = []
        
        for neighbor_id, scores in all_neighbor_scores.items():
            # Calculate metrics
            appearance_count = len(scores)  # How many retrieved chunks had this as top-K neighbor
            max_similarity = max(score for score, _ in scores)
            avg_similarity = np.mean([score for score, _ in scores])
            
            # Get detailed similarities and entity/event matches
            detailed_metrics = self._get_detailed_neighbor_metrics(
                neighbor_id, question_text, question_entities, question_events, question_sub_questions
            )
            
            # Calculate weighted score based on similarity, entities, and events (content-first approach)
            # This focuses on content relevance rather than frequency
            entity_bonus = detailed_metrics.get('entity_exact_matches', 0) * 0.1 + detailed_metrics.get('entity_substring_matches', 0) * 0.05
            event_bonus = detailed_metrics.get('event_exact_matches', 0) * 0.1 + detailed_metrics.get('event_substring_matches', 0) * 0.05
            
            # Note: Sub-question similarities are now handled via content-first approach in detailed_metrics
            weighted_score = avg_similarity + entity_bonus + event_bonus
            
            candidate = NeighborCandidate(
                node_id=neighbor_id,
                node_type=self.chunk_metadata[neighbor_id]['type'],
                is_gold=False,  # Will be set later
                appearance_count=appearance_count,
                max_similarity=max_similarity,
                avg_similarity=avg_similarity,
                weighted_score=weighted_score,
                **detailed_metrics
            )
            
            neighbor_candidates.append(candidate)
        
        # Step 3: Sort by weighted score and take final top-K
        neighbor_candidates.sort(key=lambda x: x.weighted_score, reverse=True)
        final_top_k = neighbor_candidates[:final_k]
        
        # Calculate average threshold for reporting
        avg_threshold = np.mean([c.avg_similarity for c in final_top_k]) if final_top_k else 0.0
        
        logger.info(f"Selected final top {len(final_top_k)} neighbors from {len(neighbor_candidates)} candidates")
        logger.info(f"Average similarity threshold: {avg_threshold:.3f}")
        logger.info(f"New scoring: similarity + entity_bonus + event_bonus (removed appearance_count)")
        
        return final_top_k, avg_threshold
    
    def _calculate_neighbor_similarities(self, question_text: str, question_entities: List[str],
                                       question_events: List[str], question_sub_questions: List[str], 
                                       neighbor_ids: List[str]) -> Dict[str, float]:
        """Calculate total similarity for a list of neighbors"""
        
        if not neighbor_ids:
            return {}
        
        # Separate by type
        doc_neighbors = []
        table_neighbors = []
        
        for neighbor_id in neighbor_ids:
            neighbor_type = self.chunk_metadata[neighbor_id]['type']
            if neighbor_type == 'document':
                doc_neighbors.append(neighbor_id)
            else:
                table_neighbors.append(neighbor_id)
        
        similarities = {}
        
        # Process documents
        if doc_neighbors:
            doc_sims = self._calculate_doc_neighbor_similarities(question_text, doc_neighbors)
            similarities.update(doc_sims)
        
        # Process tables
        if table_neighbors:
            table_sims = self._calculate_table_neighbor_similarities(question_text, table_neighbors)
            similarities.update(table_sims)
        
        return similarities
    
    def _calculate_doc_neighbor_similarities(self, question_text: str, doc_neighbor_ids: List[str]) -> Dict[str, float]:
        """Calculate similarities for document neighbors"""
        
        question_texts = [question_text] * len(doc_neighbor_ids)
        content_texts = []
        topic_texts = []
        
        for neighbor_id in doc_neighbor_ids:
            metadata = self.chunk_metadata[neighbor_id]
            content_texts.append(metadata['content'])
            topic_texts.append(metadata['topic'])
        
        content_sims = self.embedding_service.calculate_similarities(
            question_texts, content_texts, "neighbor_doc_content"
        )
        topic_sims = self.embedding_service.calculate_similarities(
            question_texts, topic_texts, "neighbor_doc_topic"
        )
        
        similarities = {}
        for i, neighbor_id in enumerate(doc_neighbor_ids):
            # Total similarity for documents = content + topic
            similarities[neighbor_id] = float(content_sims[i] + topic_sims[i])
        
        return similarities
    
    def _calculate_table_neighbor_similarities(self, question_text: str, table_neighbor_ids: List[str]) -> Dict[str, float]:
        """Calculate similarities for table neighbors"""
        
        question_texts = [question_text] * len(table_neighbor_ids)
        content_texts = []
        description_texts = []
        column_texts = []
        
        for neighbor_id in table_neighbor_ids:
            metadata = self.chunk_metadata[neighbor_id]
            content_texts.append(metadata['content'])
            description_texts.append(metadata['description'])
            
            # Column descriptions
            column_descs = metadata.get('column_descriptions', {})
            if column_descs:
                column_text = ' '.join(column_descs.values())
            else:
                column_text = ""
            column_texts.append(column_text)
        
        content_sims = self.embedding_service.calculate_similarities(
            question_texts, content_texts, "neighbor_table_content"
        )
        desc_sims = self.embedding_service.calculate_similarities(
            question_texts, description_texts, "neighbor_table_desc"
        )
        column_sims = self.embedding_service.calculate_similarities(
            question_texts, column_texts, "neighbor_table_columns"
        )
        
        similarities = {}
        for i, neighbor_id in enumerate(table_neighbor_ids):
            # Total similarity for tables = content + description + max_column
            similarities[neighbor_id] = float(content_sims[i] + desc_sims[i] + column_sims[i])
        
        return similarities
    
    def _get_detailed_neighbor_metrics(self, neighbor_id: str, question_text: str,
                                     question_entities: List[str], question_events: List[str],
                                     question_sub_questions: List[str]) -> Dict[str, Any]:
        """Get detailed similarity and entity/event metrics for a neighbor using content-first strategy"""
        
        metadata = self.chunk_metadata[neighbor_id]
        neighbor_type = metadata['type']
        
        # Step 1: Calculate content similarities for both approaches
        main_content_sim = self.embedding_service.calculate_similarities(
            [question_text], [metadata['content']], f"detailed_content_main_{neighbor_id}"
        )[0]
        
        subq_content_sim = 0.0
        winning_sub_question = ""
        if question_sub_questions:
            subq_content_sims = self.embedding_service.calculate_similarities(
                question_sub_questions, [metadata['content']] * len(question_sub_questions), f"detailed_subq_content_{neighbor_id}"
            )
            max_idx = np.argmax(subq_content_sims)
            subq_content_sim = float(subq_content_sims[max_idx])
            winning_sub_question = question_sub_questions[max_idx]
        
        # Step 2: Determine winning approach and calculate other similarities
        if main_content_sim >= subq_content_sim:
            # Main question wins
            winning_approach = "main_question"
            winning_sub_question = ""
            content_sim = main_content_sim
            
            # Entity/event matching using main question
            entity_exact, entity_substring = self._match_entities_events(
                question_entities, list(metadata['entities'].keys())
            )
            event_exact, event_substring = self._match_entities_events(
                question_events, list(metadata['events'].keys())
            )
            
            # Calculate other similarities using main question
            if neighbor_type == 'document':
                topic_sim = self.embedding_service.calculate_similarities(
                    [question_text], [metadata['topic']], f"detailed_topic_main_{neighbor_id}"
                )[0]
                result = {
                    'winning_approach': winning_approach,
                    'winning_sub_question': "",
                    'best_content_similarity': content_sim,
                    'best_main_question_content_similarity': main_content_sim,
                    'best_subquestion_content_similarity': subq_content_sim,
                    'best_topic_similarity': float(topic_sim),
                    'entity_exact_matches': entity_exact,
                    'entity_substring_matches': entity_substring,
                    'event_exact_matches': event_exact,
                    'event_substring_matches': event_substring
                }
                
            elif neighbor_type == 'table':
                desc_sim = self.embedding_service.calculate_similarities(
                    [question_text], [metadata['description']], f"detailed_desc_main_{neighbor_id}"
                )[0]
                
                column_descs = metadata.get('column_descriptions', {})
                if column_descs:
                    column_text = ' '.join(column_descs.values())
                    column_sim = self.embedding_service.calculate_similarities(
                        [question_text], [column_text], f"detailed_columns_main_{neighbor_id}"
                    )[0]
                else:
                    column_sim = 0.0
                
                result = {
                    'winning_approach': winning_approach,
                    'winning_sub_question': "",
                    'best_content_similarity': content_sim,
                    'best_main_question_content_similarity': main_content_sim,
                    'best_subquestion_content_similarity': subq_content_sim,
                    'best_description_similarity': float(desc_sim),
                    'best_column_similarity': float(column_sim),
                    'entity_exact_matches': entity_exact,
                    'entity_substring_matches': entity_substring,
                    'event_exact_matches': event_exact,
                    'event_substring_matches': event_substring
                }
        else:
            # Sub-question wins
            winning_approach = "sub_question"
            content_sim = subq_content_sim
            
            # Entity/event matching using main question entities/events as proxy
            entity_exact, entity_substring = self._match_entities_events(
                question_entities, list(metadata['entities'].keys())
            )
            event_exact, event_substring = self._match_entities_events(
                question_events, list(metadata['events'].keys())
            )
            
            # Calculate other similarities using winning sub-question
            if neighbor_type == 'document':
                topic_sim = self.embedding_service.calculate_similarities(
                    [winning_sub_question], [metadata['topic']], f"detailed_topic_subq_{neighbor_id}"
                )[0]
                result = {
                    'winning_approach': winning_approach,
                    'winning_sub_question': winning_sub_question,
                    'best_content_similarity': content_sim,
                    'best_main_question_content_similarity': main_content_sim,
                    'best_subquestion_content_similarity': subq_content_sim,
                    'best_topic_similarity': float(topic_sim),
                    'entity_exact_matches': entity_exact,
                    'entity_substring_matches': entity_substring,
                    'event_exact_matches': event_exact,
                    'event_substring_matches': event_substring
                }
                
            elif neighbor_type == 'table':
                desc_sim = self.embedding_service.calculate_similarities(
                    [winning_sub_question], [metadata['description']], f"detailed_desc_subq_{neighbor_id}"
                )[0]
                
                column_descs = metadata.get('column_descriptions', {})
                if column_descs:
                    column_text = ' '.join(column_descs.values())
                    column_sim = self.embedding_service.calculate_similarities(
                        [winning_sub_question], [column_text], f"detailed_columns_subq_{neighbor_id}"
                    )[0]
                else:
                    column_sim = 0.0
                
                result = {
                    'winning_approach': winning_approach,
                    'winning_sub_question': winning_sub_question,
                    'best_content_similarity': content_sim,
                    'best_main_question_content_similarity': main_content_sim,
                    'best_subquestion_content_similarity': subq_content_sim,
                    'best_description_similarity': float(desc_sim),
                    'best_column_similarity': float(column_sim),
                    'entity_exact_matches': entity_exact,
                    'entity_substring_matches': entity_substring,
                    'event_exact_matches': event_exact,
                    'event_substring_matches': event_substring
                }
        
        return result
    
    def analyze_gold_presence_with_ranking(self, neighbor_candidates: List[NeighborCandidate], 
                                         gold_docs: List[str], avg_threshold: float) -> Tuple[List[NeighborCandidate], List[str], Dict[str, Any]]:
        """Analyze gold document presence and ranking in neighbor candidates"""
        
        # Mark gold neighbors and track their ranks
        gold_in_neighbors = []
        gold_missed = []
        gold_analysis = {}
        
        # Create a ranking lookup
        neighbor_ranking = {candidate.node_id: i+1 for i, candidate in enumerate(neighbor_candidates)}
        
        for gold_doc in gold_docs:
            if gold_doc in neighbor_ranking:
                # Find the neighbor candidate
                for candidate in neighbor_candidates:
                    if candidate.node_id == gold_doc:
                        candidate.is_gold = True
                        gold_in_neighbors.append(candidate)
                        
                        # Store comprehensive analysis
                        gold_analysis[gold_doc] = {
                            'rank': neighbor_ranking[gold_doc],
                            'weighted_score': candidate.weighted_score,
                            'avg_similarity': candidate.avg_similarity,
                            'appearance_count': candidate.appearance_count,
                            'margin_above_avg_threshold': candidate.avg_similarity - avg_threshold,
                            'status': 'found'
                        }
                        break
            else:
                gold_missed.append(gold_doc)
                
                # For missed gold docs, check if they exist in our metadata
                if gold_doc in self.chunk_metadata:
                    gold_analysis[gold_doc] = {
                        'rank': -1,  # Not in final list
                        'weighted_score': 0.0,
                        'avg_similarity': 0.0,
                        'appearance_count': 0,
                        'margin_above_avg_threshold': -999,
                        'status': 'missed_in_metadata'
                    }
                else:
                    gold_analysis[gold_doc] = {
                        'rank': -1,
                        'weighted_score': 0.0,
                        'avg_similarity': 0.0,
                        'appearance_count': 0,
                        'margin_above_avg_threshold': -999,
                        'status': 'not_in_metadata'
                    }
        
        return gold_in_neighbors, gold_missed, gold_analysis
    
    def analyze_ranking_factors(self, neighbor_candidates: List[NeighborCandidate], 
                               gold_in_neighbors: List[NeighborCandidate],
                               retrieved_chunks: List[RetrievedChunkMetrics]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Analyze what factors contribute to ranking and what outranks gold documents"""
        
        chunks_ranking_above_gold = []
        similarity_factor_analysis = {
            'gold_avg_content_similarity': 0.0,
            'gold_avg_topic_similarity': 0.0,
            'gold_avg_description_similarity': 0.0,
            'gold_avg_column_similarity': 0.0,
            'gold_avg_entity_matches': 0.0,
            'gold_avg_event_matches': 0.0,
            'gold_avg_appearance_count': 0.0,
            'gold_avg_weighted_score': 0.0,
            'non_gold_avg_content_similarity': 0.0,
            'non_gold_avg_topic_similarity': 0.0,
            'non_gold_avg_description_similarity': 0.0,
            'non_gold_avg_column_similarity': 0.0,
            'non_gold_avg_entity_matches': 0.0,
            'non_gold_avg_event_matches': 0.0,
            'non_gold_avg_appearance_count': 0.0,
            'non_gold_avg_weighted_score': 0.0
        }
        
        # Get retrieved chunk IDs for analysis
        retrieved_chunk_ids = set([chunk.chunk_id for chunk in retrieved_chunks])
        
        # Find chunks that rank above gold
        if gold_in_neighbors:
            worst_gold_rank = max([self._get_candidate_rank(gold, neighbor_candidates) for gold in gold_in_neighbors])
            
            for i, candidate in enumerate(neighbor_candidates):
                current_rank = i + 1
                if current_rank < worst_gold_rank and not candidate.is_gold:
                    
                    chunk_analysis = {
                        'chunk_id': candidate.node_id,
                        'rank': current_rank,
                        'node_type': candidate.node_type,
                        'is_retrieved_chunk': candidate.node_id in retrieved_chunk_ids,
                        'weighted_score': candidate.weighted_score,
                        'avg_similarity': candidate.avg_similarity,
                        'max_similarity': candidate.max_similarity,
                        'appearance_count': candidate.appearance_count,
                        'content_similarity': candidate.best_content_similarity,
                        'topic_similarity': getattr(candidate, 'best_topic_similarity', 0.0),
                        'description_similarity': getattr(candidate, 'best_description_similarity', 0.0),
                        'column_similarity': getattr(candidate, 'best_column_similarity', 0.0),
                        'entity_exact_matches': candidate.entity_exact_matches,
                        'entity_substring_matches': candidate.entity_substring_matches,
                        'event_exact_matches': candidate.event_exact_matches,
                        'event_substring_matches': candidate.event_substring_matches,
                        'total_entity_matches': candidate.entity_exact_matches + candidate.entity_substring_matches,
                        'total_event_matches': candidate.event_exact_matches + candidate.event_substring_matches
                    }
                    chunks_ranking_above_gold.append(chunk_analysis)
        
        # Calculate factor analysis
        if gold_in_neighbors:
            gold_metrics = []
            for gold in gold_in_neighbors:
                gold_metrics.append({
                    'content_similarity': gold.best_content_similarity,
                    'topic_similarity': getattr(gold, 'best_topic_similarity', 0.0),
                    'description_similarity': getattr(gold, 'best_description_similarity', 0.0),
                    'column_similarity': getattr(gold, 'best_column_similarity', 0.0),
                    'entity_matches': gold.entity_exact_matches + gold.entity_substring_matches,
                    'event_matches': gold.event_exact_matches + gold.event_substring_matches,
                    'appearance_count': gold.appearance_count,
                    'weighted_score': gold.weighted_score
                })
            
            # Calculate gold averages
            similarity_factor_analysis['gold_avg_content_similarity'] = np.mean([m['content_similarity'] for m in gold_metrics])
            similarity_factor_analysis['gold_avg_topic_similarity'] = np.mean([m['topic_similarity'] for m in gold_metrics])
            similarity_factor_analysis['gold_avg_description_similarity'] = np.mean([m['description_similarity'] for m in gold_metrics])
            similarity_factor_analysis['gold_avg_column_similarity'] = np.mean([m['column_similarity'] for m in gold_metrics])
            similarity_factor_analysis['gold_avg_entity_matches'] = np.mean([m['entity_matches'] for m in gold_metrics])
            similarity_factor_analysis['gold_avg_event_matches'] = np.mean([m['event_matches'] for m in gold_metrics])
            similarity_factor_analysis['gold_avg_appearance_count'] = np.mean([m['appearance_count'] for m in gold_metrics])
            similarity_factor_analysis['gold_avg_weighted_score'] = np.mean([m['weighted_score'] for m in gold_metrics])
        
        # Calculate non-gold averages (top-10 non-gold for fair comparison)
        non_gold_candidates = [c for c in neighbor_candidates[:10] if not c.is_gold]
        if non_gold_candidates:
            non_gold_metrics = []
            for candidate in non_gold_candidates:
                non_gold_metrics.append({
                    'content_similarity': candidate.best_content_similarity,
                    'topic_similarity': getattr(candidate, 'best_topic_similarity', 0.0),
                    'description_similarity': getattr(candidate, 'best_description_similarity', 0.0),
                    'column_similarity': getattr(candidate, 'best_column_similarity', 0.0),
                    'entity_matches': candidate.entity_exact_matches + candidate.entity_substring_matches,
                    'event_matches': candidate.event_exact_matches + candidate.event_substring_matches,
                    'appearance_count': candidate.appearance_count,
                    'weighted_score': candidate.weighted_score
                })
            
            # Calculate non-gold averages
            similarity_factor_analysis['non_gold_avg_content_similarity'] = np.mean([m['content_similarity'] for m in non_gold_metrics])
            similarity_factor_analysis['non_gold_avg_topic_similarity'] = np.mean([m['topic_similarity'] for m in non_gold_metrics])
            similarity_factor_analysis['non_gold_avg_description_similarity'] = np.mean([m['description_similarity'] for m in non_gold_metrics])
            similarity_factor_analysis['non_gold_avg_column_similarity'] = np.mean([m['column_similarity'] for m in non_gold_metrics])
            similarity_factor_analysis['non_gold_avg_entity_matches'] = np.mean([m['entity_matches'] for m in non_gold_metrics])
            similarity_factor_analysis['non_gold_avg_event_matches'] = np.mean([m['event_matches'] for m in non_gold_metrics])
            similarity_factor_analysis['non_gold_avg_appearance_count'] = np.mean([m['appearance_count'] for m in non_gold_metrics])
            similarity_factor_analysis['non_gold_avg_weighted_score'] = np.mean([m['weighted_score'] for m in non_gold_metrics])
        
        return chunks_ranking_above_gold, similarity_factor_analysis
    
    def _get_candidate_rank(self, candidate: NeighborCandidate, neighbor_candidates: List[NeighborCandidate]) -> int:
        """Get the rank of a candidate in the neighbor list"""
        for i, neighbor in enumerate(neighbor_candidates):
            if neighbor.node_id == candidate.node_id:
                return i + 1
        return -1
    
    def _save_detailed_question_analysis(self, question_result: QuestionResult):
        """Save detailed per-question analysis to individual folder"""
        
        question_id = question_result.question_id
        output_dir = f"output/focused_retrieval_analysis/question_{question_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save retrieved chunks analysis
        retrieved_analysis = {
            "question_text": question_result.question_text,
            "question_entities": question_result.question_entities,
            "question_events": question_result.question_events,
            "question_sub_questions": question_result.question_sub_questions,
            "retrieved_chunks": []
        }
        
        for chunk in question_result.retrieved_chunks:
            chunk_data = {
                "chunk_id": chunk.chunk_id,
                "chunk_type": chunk.chunk_type,
                "rank": chunk.rank,
                "winning_approach": chunk.winning_approach,
                "winning_sub_question": chunk.winning_sub_question,
                "content_similarity": chunk.content_similarity,
                "main_question_content_similarity": chunk.main_question_content_similarity,
                "max_subquestion_content_similarity": chunk.max_subquestion_content_similarity,
                "topic_similarity": chunk.topic_similarity,
                "description_similarity": chunk.description_similarity,
                "max_column_similarity": chunk.max_column_similarity,
                "entity_exact_matches": chunk.entity_exact_matches,
                "entity_substring_matches": chunk.entity_substring_matches,
                "event_exact_matches": chunk.event_exact_matches,
                "event_substring_matches": chunk.event_substring_matches,
                "total_similarity": chunk.total_similarity
            }
            retrieved_analysis["retrieved_chunks"].append(chunk_data)
        
        with open(f"{output_dir}/retrieved_chunks_analysis.json", 'w') as f:
            json.dump(retrieved_analysis, f, indent=2, cls=NumpyEncoder)
        
        # 2. Save gold chunks analysis
        gold_analysis = {
            "gold_docs": question_result.gold_docs,
            "gold_found_in_neighbors": [],
            "gold_missed": question_result.gold_missed,
            "gold_margin_analysis": question_result.gold_margin_analysis
        }
        
        for gold_candidate in question_result.gold_in_neighbors:
            gold_data = {
                "chunk_id": gold_candidate.node_id,
                "chunk_type": gold_candidate.node_type,
                "connected_to_retrieved_chunk": gold_candidate.connected_to_retrieved_chunk,
                "distance_from_retrieved": gold_candidate.distance_from_retrieved,
                "winning_approach": gold_candidate.winning_approach,
                "winning_sub_question": gold_candidate.winning_sub_question,
                "content_similarity": gold_candidate.best_content_similarity,
                "main_question_content_similarity": gold_candidate.best_main_question_content_similarity,
                "max_subquestion_content_similarity": gold_candidate.best_subquestion_content_similarity,
                "topic_similarity": gold_candidate.best_topic_similarity,
                "description_similarity": gold_candidate.best_description_similarity,
                "column_similarity": gold_candidate.best_column_similarity,
                "entity_exact_matches": gold_candidate.entity_exact_matches,
                "entity_substring_matches": gold_candidate.entity_substring_matches,
                "event_exact_matches": gold_candidate.event_exact_matches,
                "event_substring_matches": gold_candidate.event_substring_matches,
                "weighted_score": gold_candidate.weighted_score,
                "rank_in_neighbors": self._get_candidate_rank(gold_candidate, question_result.all_neighbor_candidates)
            }
            gold_analysis["gold_found_in_neighbors"].append(gold_data)
        
        with open(f"{output_dir}/gold_chunks_analysis.json", 'w') as f:
            json.dump(gold_analysis, f, indent=2, cls=NumpyEncoder)
        
        # 3. Save top 100 neighbors analysis
        neighbors_analysis = {
            "total_neighbors_analyzed": len(question_result.all_neighbor_candidates),
            "similarity_95th_percentile": question_result.similarity_95th_percentile,
            "top_100_neighbors": []
        }
        
        for i, neighbor in enumerate(question_result.all_neighbor_candidates):
            neighbor_data = {
                "rank": i + 1,
                "chunk_id": neighbor.node_id,
                "chunk_type": neighbor.node_type,
                "is_gold": neighbor.is_gold,
                "winning_approach": neighbor.winning_approach,
                "winning_sub_question": neighbor.winning_sub_question,
                "content_similarity": neighbor.best_content_similarity,
                "main_question_content_similarity": neighbor.best_main_question_content_similarity,
                "max_subquestion_content_similarity": neighbor.best_subquestion_content_similarity,
                "topic_similarity": neighbor.best_topic_similarity,
                "description_similarity": neighbor.best_description_similarity,
                "column_similarity": neighbor.best_column_similarity,
                "entity_exact_matches": neighbor.entity_exact_matches,
                "entity_substring_matches": neighbor.entity_substring_matches,
                "event_exact_matches": neighbor.event_exact_matches,
                "event_substring_matches": neighbor.event_substring_matches,
                "appearance_count": neighbor.appearance_count,
                "weighted_score": neighbor.weighted_score
            }
            
            if neighbor.is_gold:
                neighbor_data.update({
                    "connected_to_retrieved_chunk": neighbor.connected_to_retrieved_chunk,
                    "distance_from_retrieved": neighbor.distance_from_retrieved
                })
            
            neighbors_analysis["top_100_neighbors"].append(neighbor_data)
        
        with open(f"{output_dir}/top_100_neighbors_analysis.json", 'w') as f:
            json.dump(neighbors_analysis, f, indent=2, cls=NumpyEncoder)
        
        # 4. Save similarity summary
        summary = {
            "question_analysis": {
                "question_id": question_result.question_id,
                "question_text": question_result.question_text,
                "has_sub_questions": len(question_result.question_sub_questions) > 0,
                "num_sub_questions": len(question_result.question_sub_questions),
                "num_entities": len(question_result.question_entities),
                "num_events": len(question_result.question_events)
            },
            "retrieval_performance": {
                "gold_docs_count": len(question_result.gold_docs),
                "gold_found_in_neighbors": len(question_result.gold_in_neighbors),
                "gold_missed": len(question_result.gold_missed),
                "success_rate": len(question_result.gold_in_neighbors) / len(question_result.gold_docs) if question_result.gold_docs else 0
            },
            "approach_statistics": {
                "retrieved_chunks_main_question_wins": sum(1 for c in question_result.retrieved_chunks if c.winning_approach == "main_question"),
                "retrieved_chunks_sub_question_wins": sum(1 for c in question_result.retrieved_chunks if c.winning_approach == "sub_question"),
                "neighbors_main_question_wins": sum(1 for n in question_result.all_neighbor_candidates if n.winning_approach == "main_question"),
                "neighbors_sub_question_wins": sum(1 for n in question_result.all_neighbor_candidates if n.winning_approach == "sub_question")
            },
            "similarity_factor_analysis": question_result.similarity_factor_analysis,
            "chunks_ranking_above_gold_count": len(question_result.chunks_ranking_above_gold)
        }
        
        with open(f"{output_dir}/similarity_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Saved detailed analysis for question {question_id} to {output_dir}")

    def analyze_single_question(self, question_data: Dict[str, Any]) -> QuestionResult:
        """Analyze a single question comprehensively"""
        
        question_text = question_data['question_text']
        question_id = question_data['question_id']
        gold_docs = question_data['gold_docs']
        retrieved_docs = question_data['retrieved_docs']
        
        # Get question entities, events, and sub-questions
        question_metadata = self.questions_metadata.get(question_text, {})
        question_entities = question_metadata.get('entities', [])
        question_events = question_metadata.get('events', [])
        question_sub_questions = question_metadata.get('sub_questions', [])  # NEW: Extract sub-questions
        
        logger.info(f"Analyzing question: {question_id}")
        if question_sub_questions:
            logger.info(f"Found {len(question_sub_questions)} sub-questions for enhanced similarity analysis")
        
        # Analyze retrieved chunks
        retrieved_metrics = self.analyze_retrieved_chunks(
            question_text, retrieved_docs, question_entities, question_events, question_sub_questions
        )
        
        # Get top-K neighbors (100 per chunk, top 100 overall)
        neighbor_candidates, avg_threshold = self.get_top_k_neighbors(
            retrieved_metrics, question_text, question_entities, question_events, question_sub_questions,
            k_per_chunk=100, final_k=100
        )
        
        # Analyze gold presence with ranking
        gold_in_neighbors, gold_missed, gold_analysis = self.analyze_gold_presence_with_ranking(
            neighbor_candidates, gold_docs, avg_threshold
        )
        
        # Analyze ranking factors and what beats gold
        chunks_ranking_above_gold, similarity_factor_analysis = self.analyze_ranking_factors(
            neighbor_candidates, gold_in_neighbors, retrieved_metrics
        )
        
        # Determine frequency threshold (could be based on retrieved chunk count)
        frequency_threshold = max(1, len(retrieved_docs) // 3)  # At least 1/3 of retrieved docs
        
        # Filter neighbors by frequency threshold
        filtered_neighbors = [n for n in neighbor_candidates if n.appearance_count >= frequency_threshold]
        
        result = QuestionResult(
            question_id=question_id,
            question_text=question_text,
            question_entities=question_entities,
            question_events=question_events,
            question_sub_questions=question_sub_questions,  # NEW: Include sub-questions
            retrieved_chunks=retrieved_metrics,
            all_neighbor_candidates=neighbor_candidates,
            filtered_neighbors=filtered_neighbors,
            gold_docs=gold_docs,
            gold_in_neighbors=gold_in_neighbors,
            gold_missed=gold_missed,
            gold_margin_analysis=gold_analysis,
            chunks_ranking_above_gold=chunks_ranking_above_gold,
            similarity_factor_analysis=similarity_factor_analysis,
            similarity_95th_percentile=avg_threshold,
            frequency_threshold=frequency_threshold
        )
        
        # Save detailed per-question analysis
        self._save_detailed_question_analysis(result)
        
        return result
    
    def run_focused_analysis(self, limit_questions: int = 100) -> List[QuestionResult]:
        """Run the complete focused retrieval analysis"""
        logger.info("Starting focused retrieval analysis...")
        
        # Load all data
        self.load_all_data()
        
        # Load questions from CSV
        df = pd.read_csv(self.csv_file)
        ranking_columns = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th']
        
        questions_to_analyze = []
        
        for idx, row in df.iterrows():
            # Parse gold and retrieved documents
            gold_docs = self.safe_parse_list(row['gold_docs'])
            retrieved_docs = []
            
            for col in ranking_columns:
                val = row[col]
                if pd.notna(val) and val != '':
                    retrieved_docs.append(str(val).strip())
            
            # Check if we have valid data
            if not gold_docs or not retrieved_docs:
                continue
            
            question_data = {
                'question_id': row.get('question_id', f'q_{idx}'),
                'question_text': row.get('question', ''),
                'gold_docs': gold_docs,
                'retrieved_docs': retrieved_docs
            }
            
            questions_to_analyze.append(question_data)
            
            if len(questions_to_analyze) >= limit_questions:
                break
        
        logger.info(f"Analyzing {len(questions_to_analyze)} questions...")
        
        # Analyze each question
        for i, question_data in enumerate(questions_to_analyze):
            try:
                result = self.analyze_single_question(question_data)
                self.question_results.append(result)
                
                logger.info(f"Completed question {i+1}/{len(questions_to_analyze)}: "
                          f"{len(result.gold_in_neighbors)} gold found, "
                          f"{len(result.gold_missed)} gold missed")
                
            except Exception as e:
                logger.error(f"Error analyzing question {question_data['question_id']}: {e}")
                continue
        
        # Save results and cache
        self._save_results()
        self.embedding_service.save_cache(str(self.cache_dir / "embeddings.pkl"))
        
        logger.info("Focused retrieval analysis completed!")
        return self.question_results
    
    def _save_results(self):
        """Save analysis results"""
        results_file = self.cache_dir / "focused_analysis_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(self.question_results, f)
        logger.info(f"Saved analysis results to {results_file}")

def main():
    """Main execution function"""
    analyzer = FocusedRetrievalAnalyzer()
    results = analyzer.run_focused_analysis(limit_questions=100)
    
    logger.info(f"Analysis complete! Processed {len(results)} questions")
    
    # Quick summary
    total_gold_found = sum(len(r.gold_in_neighbors) for r in results)
    total_gold_missed = sum(len(r.gold_missed) for r in results)
    
    logger.info(f"Summary: {total_gold_found} gold documents found, {total_gold_missed} missed")

if __name__ == "__main__":
    main()