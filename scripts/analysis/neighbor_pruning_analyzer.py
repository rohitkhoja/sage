#!/usr/bin/env python3
"""
Neighbor Pruning Analysis System

This system analyzes graph neighbors to identify distinguishing features of gold documents
for efficient neighbor pruning in knowledge graph traversal.

Key functionality:
1. Filter questions with 0-hop (direct) or 1-hop gold document connections
2. Extract all neighbors of retrieved documents
3. Calculate multiple similarity metrics between questions and neighbors
4. Analyze distinguishing features of gold documents vs other neighbors
5. Generate insights for optimal neighbor pruning strategies
"""

import json
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

@dataclass
class NodeMetrics:
    """Container for all metrics calculated for a node relative to a question"""
    node_id: str
    node_type: str # "document" or "table"
    is_gold: bool
    
    # Entity/Event matching (counts)
    entity_exact_matches: int
    entity_substring_matches: int
    event_exact_matches: int
    event_substring_matches: int
    
    # Additional metadata
    total_entities: int
    total_events: int
    hop_distance: int # 0 for direct neighbor, 1 for 1-hop
    
    # Text similarity metrics (cosine similarity) - with defaults
    content_similarity: float = 0.0
    topic_similarity: float = 0.0
    title_similarity: float = 0.0 # For tables
    description_similarity: float = 0.0 # For tables
    column_description_similarity: float = 0.0 # For tables

@dataclass
class QuestionAnalysis:
    """Container for complete analysis of a question"""
    question_id: str
    question_text: str
    question_entities: List[str]
    question_events: List[str]
    
    # Gold document information
    gold_docs: List[str]
    gold_docs_in_graph: List[str]
    
    # Retrieved documents analysis
    retrieved_docs: List[str]
    retrieved_with_gold_neighbors: List[Tuple[str, int]] # (doc_id, hop_distance)
    
    # Neighbor analysis results
    all_neighbors: List[NodeMetrics]
    gold_neighbors: List[NodeMetrics]
    non_gold_neighbors: List[NodeMetrics]
    
    # Summary statistics
    total_neighbors_analyzed: int
    gold_neighbor_count: int
    non_gold_neighbor_count: int

class GPUEmbeddingService:
    """GPU-accelerated embedding generation and similarity calculation service"""
    
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
                    batch_size=min(32, len(batch_texts)), # Use small internal batch size
                    normalize_embeddings=True
                )
                
                # Store in cache and results
                for j, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings)):
                    cache_key = f"{cache_key_prefix}_{hash(text)}"
                    self.embedding_cache[cache_key] = embedding
                    all_embeddings[batch_indices[j]] = embedding
        
        return torch.stack(all_embeddings)
    
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

class NeighborPruningAnalyzer:
    """Main analyzer for neighbor pruning optimization"""
    
    def __init__(self, 
                 edges_file: str = "output/analysis_cache/knowledge_graph/unique_edges_20250731_173102.json",
                 csv_file: str = "output/dense_sparse_average_results_filtered.csv",
                 questions_metadata_file: str = "output/questions_metadata_output.json",
                 docs_chunks_dir: str = "output/full_pipeline/docs_chunks_1",
                 table_chunks_file: str = "output/full_pipeline/table_chunks_with_metadata.json",
                 cache_dir: str = "output/neighbor_pruning_cache",
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
        self.chunk_metadata = {} # chunk_id -> metadata dict
        self.questions_metadata = {} # question -> {entities, events}
        self.analysis_results = [] # List[QuestionAnalysis]
        
        logger.info("Initialized NeighborPruningAnalyzer")
    
    def load_all_data(self):
        """Load all required data: graph, chunk metadata, questions metadata"""
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
        
        # Build graph with uniform edge weights
        edges_to_add = []
        for edge in edges_data:
            source = edge['source_chunk_id']
            target = edge['target_chunk_id']
            edges_to_add.append((source, target))
        
        self.graph.add_edges_from(edges_to_add)
        
        logger.info(f"Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def _load_chunk_metadata(self):
        """Load metadata for all chunks (documents and tables)"""
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
        """Load questions metadata (entities and events)"""
        logger.info(f"Loading questions metadata from: {self.questions_metadata_file}")
        
        with open(self.questions_metadata_file, 'r') as f:
            self.questions_metadata = json.load(f)
        
        logger.info(f"Loaded metadata for {len(self.questions_metadata)} questions")
    
    def safe_parse_list(self, value) -> List[str]:
        """Safely parse a string representation of a list into actual list"""
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
    
    def filter_0_1_hop_questions(self) -> List[Dict[str, Any]]:
        """Filter questions where gold docs are 0-hop or 1-hop neighbors of retrieved docs"""
        logger.info("Filtering questions with 0-hop and 1-hop gold document connections...")
        
        # Load the filtered CSV
        df = pd.read_csv(self.csv_file)
        ranking_columns = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th']
        
        filtered_questions = []
        
        for idx, row in df.iterrows():
            # Parse gold and retrieved documents
            gold_docs = self.safe_parse_list(row['gold_docs'])
            retrieved_docs = []
            
            for col in ranking_columns:
                val = row[col]
                if pd.notna(val) and val != '':
                    retrieved_docs.append(str(val).strip())
            
            # Check if any gold docs are in graph
            gold_docs_in_graph = [doc for doc in gold_docs if doc in self.graph]
            if not gold_docs_in_graph:
                continue
            
            # Check for 0-hop and 1-hop connections
            retrieved_with_gold = []
            
            for retrieved_doc in retrieved_docs:
                if retrieved_doc not in self.graph:
                    continue
                
                # Check 0-hop (direct neighbors)
                neighbors = set(self.graph.neighbors(retrieved_doc))
                for gold_doc in gold_docs_in_graph:
                    if gold_doc in neighbors:
                        retrieved_with_gold.append((retrieved_doc, 0, gold_doc))
                
                # Check 1-hop (neighbors of neighbors)
                for neighbor in neighbors:
                    second_neighbors = set(self.graph.neighbors(neighbor))
                    for gold_doc in gold_docs_in_graph:
                        if gold_doc in second_neighbors and gold_doc != retrieved_doc:
                            retrieved_with_gold.append((retrieved_doc, 1, gold_doc))
            
            if retrieved_with_gold:
                question_info = {
                    'question_id': row.get('question_id', f'q_{idx}'),
                    'question_text': row.get('question', ''),
                    'gold_docs': gold_docs,
                    'gold_docs_in_graph': gold_docs_in_graph,
                    'retrieved_docs': retrieved_docs,
                    'retrieved_with_gold': list(set(retrieved_with_gold)) # Remove duplicates
                }
                filtered_questions.append(question_info)
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} questions, found {len(filtered_questions)} with 0/1-hop connections")
        
        logger.info(f"Found {len(filtered_questions)} questions with 0-hop or 1-hop gold connections")
        return filtered_questions
    
    def analyze_question_neighbors(self, question_info: Dict[str, Any]) -> QuestionAnalysis:
        """Analyze all neighbors for a single question"""
        question_text = question_info['question_text']
        
        # Get question entities and events
        question_entities = self.questions_metadata.get(question_text, {}).get('entities', [])
        question_events = self.questions_metadata.get(question_text, {}).get('events', [])
        
        # Collect all unique neighbors to analyze
        all_neighbors = set()
        retrieved_with_gold_neighbors = []
        
        for retrieved_doc, hop_distance, gold_doc in question_info['retrieved_with_gold']:
            if retrieved_doc not in self.graph:
                continue
            
            retrieved_with_gold_neighbors.append((retrieved_doc, hop_distance))
            
            # Get direct neighbors
            direct_neighbors = set(self.graph.neighbors(retrieved_doc))
            all_neighbors.update(direct_neighbors)
            
            # Get 1-hop neighbors
            for neighbor in direct_neighbors:
                second_neighbors = set(self.graph.neighbors(neighbor))
                all_neighbors.update(second_neighbors)
        
        # Remove retrieved docs themselves from neighbors
        all_neighbors = all_neighbors - set(question_info['retrieved_docs'])
        
        # Filter neighbors that have metadata
        valid_neighbors = [n for n in all_neighbors if n in self.chunk_metadata]
        
        # Calculate metrics for all neighbors
        neighbor_metrics = []
        for neighbor_id in valid_neighbors:
            metrics = self._calculate_node_metrics(
                neighbor_id, 
                question_text, 
                question_entities, 
                question_events,
                question_info['gold_docs_in_graph']
            )
            neighbor_metrics.append(metrics)
        
        # Separate gold and non-gold neighbors
        gold_neighbors = [m for m in neighbor_metrics if m.is_gold]
        non_gold_neighbors = [m for m in neighbor_metrics if not m.is_gold]
        
        return QuestionAnalysis(
            question_id=question_info['question_id'],
            question_text=question_text,
            question_entities=question_entities,
            question_events=question_events,
            gold_docs=question_info['gold_docs'],
            gold_docs_in_graph=question_info['gold_docs_in_graph'],
            retrieved_docs=question_info['retrieved_docs'],
            retrieved_with_gold_neighbors=retrieved_with_gold_neighbors,
            all_neighbors=neighbor_metrics,
            gold_neighbors=gold_neighbors,
            non_gold_neighbors=non_gold_neighbors,
            total_neighbors_analyzed=len(neighbor_metrics),
            gold_neighbor_count=len(gold_neighbors),
            non_gold_neighbor_count=len(non_gold_neighbors)
        )
    
    def _calculate_node_metrics(self, node_id: str, question_text: str, 
                               question_entities: List[str], question_events: List[str],
                               gold_docs: List[str]) -> NodeMetrics:
        """Calculate all similarity metrics for a node relative to a question"""
        
        node_metadata = self.chunk_metadata[node_id]
        is_gold = node_id in gold_docs
        
        # Entity matching
        entity_exact, entity_substring = self._match_entities_events(
            question_entities, list(node_metadata['entities'].keys())
        )
        
        # Event matching
        event_exact, event_substring = self._match_entities_events(
            question_events, list(node_metadata['events'].keys())
        )
        
        # Text similarity calculations (will be computed in batch later)
        content_similarity = 0.0
        topic_similarity = 0.0
        title_similarity = 0.0
        description_similarity = 0.0
        column_description_similarity = 0.0
        
        return NodeMetrics(
            node_id=node_id,
            node_type=node_metadata['type'],
            is_gold=is_gold,
            entity_exact_matches=entity_exact,
            entity_substring_matches=entity_substring,
            event_exact_matches=event_exact,
            event_substring_matches=event_substring,
            content_similarity=content_similarity,
            topic_similarity=topic_similarity,
            title_similarity=title_similarity,
            description_similarity=description_similarity,
            column_description_similarity=column_description_similarity,
            total_entities=len(node_metadata['entities']),
            total_events=len(node_metadata['events']),
            hop_distance=0 # Will be set later based on graph analysis
        )
    
    def _match_entities_events(self, question_items: List[str], node_items: List[str]) -> Tuple[int, int]:
        """Match entities or events between question and node (exact + substring)"""
        exact_matches = 0
        substring_matches = 0
        
        for q_item in question_items:
            q_item_lower = q_item.lower().strip()
            
            for n_item in node_items:
                n_item_lower = n_item.lower().strip()
                
                # Exact match
                if q_item_lower == n_item_lower:
                    exact_matches += 1
                    break
                # Substring match (question item in node item)
                elif q_item_lower in n_item_lower:
                    substring_matches += 1
                    break
        
        return exact_matches, substring_matches
    
    def calculate_text_similarities_batch(self, analysis_results: List[QuestionAnalysis]):
        """Calculate all text similarities using GPU batch processing"""
        logger.info("Calculating text similarities using GPU batch processing...")
        
        # Collect all texts by type for separate processing
        content_pairs = []
        topic_pairs = []
        title_pairs = []
        description_pairs = []
        column_desc_pairs = []
        
        text_pair_indices = {} # Maps (question_idx, node_idx) to indices in each array
        
        for q_idx, analysis in enumerate(analysis_results):
            question_text = analysis.question_text
            
            for n_idx, node_metrics in enumerate(analysis.all_neighbors):
                node_metadata = self.chunk_metadata[node_metrics.node_id]
                
                # Content similarity pair
                content_idx = len(content_pairs)
                content_pairs.append((question_text, node_metadata['content']))
                
                # Topic similarity pair
                topic_idx = len(topic_pairs)
                topic_pairs.append((question_text, node_metadata['topic']))
                
                # Store indices for this question-node pair
                text_pair_indices[(q_idx, n_idx)] = {
                    'content_idx': content_idx,
                    'topic_idx': topic_idx
                }
                
                # Table-specific similarities
                if node_metadata['type'] == 'table':
                    title_idx = len(title_pairs)
                    title_pairs.append((question_text, node_metadata.get('title', '')))
                    
                    desc_idx = len(description_pairs)
                    description_pairs.append((question_text, node_metadata.get('description', '')))
                    
                    # Column descriptions (concatenated)
                    col_desc = ' '.join(node_metadata.get('column_descriptions', {}).values())
                    col_idx = len(column_desc_pairs)
                    column_desc_pairs.append((question_text, col_desc))
                    
                    text_pair_indices[(q_idx, n_idx)].update({
                        'title_idx': title_idx,
                        'description_idx': desc_idx,
                        'column_desc_idx': col_idx
                    })
        
        # Calculate similarities for each type
        content_similarities = self._calculate_pairwise_similarities(content_pairs, "content")
        topic_similarities = self._calculate_pairwise_similarities(topic_pairs, "topic")
        
        title_similarities = np.array([])
        description_similarities = np.array([])
        column_desc_similarities = np.array([])
        
        if title_pairs:
            title_similarities = self._calculate_pairwise_similarities(title_pairs, "title")
        if description_pairs:
            description_similarities = self._calculate_pairwise_similarities(description_pairs, "description")
        if column_desc_pairs:
            column_desc_similarities = self._calculate_pairwise_similarities(column_desc_pairs, "column_desc")
        
        # Update analysis results with calculated similarities
        logger.info("Updating analysis results with similarity scores...")
        for q_idx, analysis in enumerate(analysis_results):
            for n_idx, node_metrics in enumerate(analysis.all_neighbors):
                indices = text_pair_indices[(q_idx, n_idx)]
                
                # Update similarities
                node_metrics.content_similarity = float(content_similarities[indices['content_idx']])
                node_metrics.topic_similarity = float(topic_similarities[indices['topic_idx']])
                
                # Table-specific similarities
                if 'title_idx' in indices and len(title_similarities) > indices['title_idx']:
                    node_metrics.title_similarity = float(title_similarities[indices['title_idx']])
                if 'description_idx' in indices and len(description_similarities) > indices['description_idx']:
                    node_metrics.description_similarity = float(description_similarities[indices['description_idx']])
                if 'column_desc_idx' in indices and len(column_desc_similarities) > indices['column_desc_idx']:
                    node_metrics.column_description_similarity = float(column_desc_similarities[indices['column_desc_idx']])
        
        logger.info("Text similarity calculation completed")
    
    def _calculate_pairwise_similarities(self, text_pairs: List[Tuple[str, str]], pair_type: str) -> np.ndarray:
        """Calculate pairwise cosine similarities for a list of text pairs"""
        if not text_pairs:
            return np.array([])
        
        logger.info(f"Calculating {len(text_pairs)} {pair_type} similarities...")
        
        # Extract texts
        texts_1 = [pair[0] for pair in text_pairs]
        texts_2 = [pair[1] for pair in text_pairs]
        
        # Generate embeddings
        embeddings_1 = self.embedding_service.generate_embeddings_batch(
            texts_1, cache_key_prefix=f"{pair_type}_1"
        )
        embeddings_2 = self.embedding_service.generate_embeddings_batch(
            texts_2, cache_key_prefix=f"{pair_type}_2"
        )
        
        # Calculate pairwise similarities
        similarities = torch.nn.functional.cosine_similarity(
            embeddings_1, embeddings_2, dim=1
        ).cpu().numpy()
        
        return similarities
    
    def run_complete_analysis(self) -> List[QuestionAnalysis]:
        """Run the complete neighbor pruning analysis"""
        logger.info("Starting complete neighbor pruning analysis...")
        
        # Load all data
        self.load_all_data()
        
        # Filter questions with 0/1-hop connections
        filtered_questions = self.filter_0_1_hop_questions()
        
        # Analyze each question (limit for testing)
        logger.info(f"Analyzing {len(filtered_questions)} questions...")
        analysis_results = []
        
        # Limit to first 10 questions for initial testing due to memory constraints
        questions_to_analyze = filtered_questions[:10]
        
        for i, question_info in enumerate(questions_to_analyze):
            analysis = self.analyze_question_neighbors(question_info)
            analysis_results.append(analysis)
            
            if (i + 1) % 5 == 0:
                logger.info(f"Analyzed {i + 1}/{len(questions_to_analyze)} questions")
        
        # Calculate text similarities in batch
        self.calculate_text_similarities_batch(analysis_results)
        
        # Save results and cache
        self.analysis_results = analysis_results
        self._save_analysis_results()
        self.embedding_service.save_cache(str(self.cache_dir / "embeddings.pkl"))
        
        logger.info("Complete analysis finished")
        return analysis_results
    
    def _save_analysis_results(self):
        """Save analysis results to cache"""
        results_file = self.cache_dir / "analysis_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(self.analysis_results, f)
        logger.info(f"Saved analysis results to {results_file}")

def main():
    """Main execution function"""
    analyzer = NeighborPruningAnalyzer()
    results = analyzer.run_complete_analysis()
    
    logger.info(f"Analysis complete! Processed {len(results)} questions")
    for result in results[:3]: # Show first 3 as examples
        logger.info(f"Question: {result.question_text[:100]}...")
        logger.info(f" - Gold neighbors: {result.gold_neighbor_count}")
        logger.info(f" - Non-gold neighbors: {result.non_gold_neighbor_count}")
        logger.info(f" - Total neighbors analyzed: {result.total_neighbors_analyzed}")

if __name__ == "__main__":
    main()