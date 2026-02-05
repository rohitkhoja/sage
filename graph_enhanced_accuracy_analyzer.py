#!/usr/bin/env python3
"""
Graph Enhanced Accuracy Analyzer

This script analyzes how graph neighbors enhance retrieval accuracy compared to CSV-only results.
It tests various k-value combinations to determine where graph connectivity helps most.

Key Features:
- Pre-calculates all similarities for efficiency
- Tests 13 different k-value combinations
- Compares CSV-only vs CSV+Graph accuracy
- Uses only main questions (no sub-questions)
- Processes all available questions
"""

import pandas as pd
import numpy as np
import json
import torch
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import pickle
import ast
import re
import math
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_stopwords(language: str = 'english') -> Set[str]:
    """
    Load stopwords from local file
    
    Args:
        language: Language name (default: 'english')
        
    Returns:
        Set of stopwords
    """
    current_dir = Path(__file__).parent
    stopwords_file = current_dir / "src" / "pipeline" / "data" / "stopwords" / language
    
    if not stopwords_file.exists():
        # Fallback to basic English stopwords if file not found
        return {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        stopwords = set(line.strip() for line in f if line.strip())
    
    return stopwords

# Load English stopwords once at module level
ENGLISH_STOPWORDS = load_stopwords('english')

def preprocess_text_for_bm25(text: str) -> List[str]:
    """Preprocess text for BM25 - remove stopwords and clean"""
    try:
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Simple split instead of tokenization
        tokens = text.split()
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens if token not in ENGLISH_STOPWORDS and len(token) > 2 and token.isalpha()]
        
        return tokens
    except Exception as e:
        logger.warning(f"Error preprocessing text for BM25: {e}")
        return []

class BM25:
    """BM25 scoring algorithm implementation"""
    
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size if self.corpus_size > 0 else 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_lens = []
        
        # Calculate document frequencies and lengths
        nd = defaultdict(int)
        for doc in corpus:
            self.doc_lens.append(len(doc))
            frequencies = {}
            for word in doc:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.doc_freqs.append(frequencies)
            
            for word in frequencies.keys():
                nd[word] += 1
        
        # Calculate IDF values
        for word, freq in nd.items():
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
    
    def get_scores(self, query: List[str]) -> List[float]:
        """Calculate BM25 scores for a query against the corpus"""
        scores = []
        for i, doc in enumerate(self.corpus):
            score = 0
            doc_freqs = self.doc_freqs[i]
            doc_len = self.doc_lens[i]
            
            for word in query:
                if word in doc_freqs:
                    freq = doc_freqs[word]
                    idf = self.idf.get(word, 0)
                    score += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            
            scores.append(score)
        return scores

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

@dataclass
class KValueTest:
    """Configuration for a specific k-value test"""
    csv_k: int
    graph_k: int
    name: str

@dataclass
class QuestionAccuracyResult:
    """Accuracy result for a single question"""
    question_id: str
    gold_docs: List[str]
    csv_only_correct: bool
    csv_plus_graph_correct: bool
    graph_enhancement: bool  # True if graph helped find gold when CSV didn't
    csv_retrieved: List[str]
    graph_neighbors: List[str]
    found_gold_in_csv: List[str]
    found_gold_in_graph: List[str]

@dataclass
class CSVvsGraphComparison:
    """Comparison between CSV+Graph vs Extended CSV"""
    question_id: str
    question_text: str
    gold_docs: List[str]
    csv_k_retrieved: List[str]  # First k CSV results
    graph_neighbors: List[str]  # Graph neighbors
    extended_csv_retrieved: List[str]  # Next k CSV results (total 2k)
    
    # Results
    csv_plus_graph_found_gold: bool
    extended_csv_found_gold: bool
    scenario: str  # "graph_wins", "csv_wins", "both_win", "both_lose"
    
    # Analysis details
    gold_found_in_csv_k: List[str]
    gold_found_in_graph: List[str]
    gold_found_in_extended_csv: List[str]
    missed_gold_docs: List[str]

@dataclass
class KValueResult:
    """Results for a specific k-value combination"""
    test_config: KValueTest
    total_questions: int
    csv_only_accuracy: float
    csv_plus_graph_accuracy: float
    improvement: float
    questions_enhanced: int  # How many questions were improved by graph
    questions_results: List[QuestionAccuracyResult]

class GraphEnhancedAccuracyAnalyzer:
    """Main analyzer for graph enhancement of retrieval accuracy"""
    
    def __init__(self,
                 csv_file: str = "output/questions_dense_sparse_average_results1.csv",
                 edges_file: str = "output/analysis_cache/knowledge_graph/unique_edges_20250812_164429.json",
                 questions_metadata_file: str = "output/questions_metadata_output.json",
                 docs_chunks_dir: str = "output/full_pipeline/docs_chunks_1", 
                 table_chunks_file: str = "output/full_pipeline/table_chunks_with_metadata.json",
                 indexing_filtered_file: str = "output/indexing_filtered.csv",
                 cache_dir: str = "output/graph_enhancement_cache_60_40_weight",
                 gpu_id: int = 0):
        
        self.csv_file = csv_file
        self.edges_file = edges_file
        self.questions_metadata_file = questions_metadata_file
        self.docs_chunks_dir = Path(docs_chunks_dir)
        self.table_chunks_file = table_chunks_file
        self.indexing_filtered_file = indexing_filtered_file
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # GPU setup - using GPUEmbeddingService
        self.embedding_service = GPUEmbeddingService(gpu_id=gpu_id)
        
        # Load embedding cache if available
        cache_file = self.cache_dir / "embedding_cache.pkl"
        self.embedding_service.load_cache(cache_file)
        
        # Data containers
        self.df = None
        self.graph = None
        self.questions_metadata = {}
        self.chunk_metadata = {}
        
        # Pre-calculated similarities cache
        self.similarity_cache = {}
        
        # Define k-value test configurations
        self.k_value_tests = [
            KValueTest(1, 1, "k1_csv_k1_graph"),
            KValueTest(2, 3, "k2_csv_k3_graph"),
            KValueTest(5, 5, "k5_csv_k5_graph"),
            KValueTest(10, 10, "k10_csv_k10_graph"),
            KValueTest(20, 30, "k20_csv_k30_graph"),
            KValueTest(30, 20, "k30_csv_k20_graph"),
            KValueTest(40, 20, "k40_csv_k20_graph"),
            KValueTest(40, 40, "k40_csv_k40_graph"),
            KValueTest(50, 10, "k50_csv_k10_graph"),
            KValueTest(50, 20, "k50_csv_k20_graph"),
            KValueTest(50, 30, "k50_csv_k30_graph"),
            KValueTest(50, 40, "k50_csv_k40_graph"),
            KValueTest(50, 50, "k50_csv_k50_graph")
        ]
        
        logger.info(f"Initialized analyzer with {len(self.k_value_tests)} k-value tests")
    
    def load_all_data(self):
        """Load all required data files"""
        logger.info("Loading all data files...")
        
        self._load_csv_data()
        self._load_graph_data()
        self._load_indexing_filtered_data()
        self._load_questions_metadata() # this is the same as the questions_metadata_output.json file
        self._load_chunk_metadata()
        
        logger.info("All data loaded successfully")
    
    def _load_csv_data(self):
        """Load CSV retrieval results"""
        logger.info(f"Loading CSV data from: {self.csv_file}")
        self.df = pd.read_csv(self.csv_file)
        logger.info(f"Loaded {len(self.df)} questions from CSV")
    
    def _load_graph_data(self):
        """Load graph structure"""
        logger.info(f"Loading graph from: {self.edges_file}")
        
        with open(self.edges_file, 'r') as f:
            edges_data = json.load(f)
        
        self.graph = nx.Graph()
        
        for edge in edges_data:
            node1 = edge['source_chunk_id']
            node2 = edge['target_chunk_id']
            weight = edge.get('weight', 1.0)
            
            self.graph.add_edge(node1, node2, weight=weight)
        
        logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def _load_indexing_filtered_data(self):
        """Load indexing filtered data"""
        logger.info(f"Loading indexing filtered data from: {self.indexing_filtered_file}")
        self.indexing_filtered_data = pd.read_csv(self.indexing_filtered_file)
        logger.info(f"Loaded {len(self.indexing_filtered_data)} rows from indexing filtered data")
    
    def _load_questions_metadata(self):
        """Load questions metadata"""
        logger.info(f"Loading questions metadata from: {self.questions_metadata_file}")
        
        with open(self.questions_metadata_file, 'r') as f:
            self.questions_metadata = json.load(f)
        
        logger.info(f"Loaded metadata for {len(self.questions_metadata)} questions")
    
    def _load_chunk_metadata(self):
        """Load chunk metadata from documents and tables"""
        logger.info("Loading chunk metadata...")
        
        # Load document chunks
        doc_count = 0
        for json_file in self.docs_chunks_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                chunk_id = data.get('chunk_id', '')

                if chunk_id not in self.indexing_filtered_data['chunk_id'].values:
                    continue
                else:
                    print(f"Chunk {chunk_id} is in indexing filtered data")

                if chunk_id:
                    self.chunk_metadata[chunk_id] = {
                        'type': 'document',
                        'content': data.get('content', ''),
                        'topic': data.get('metadata', {}).get('topic', ''),
                        'entities': data.get('metadata', {}).get('entities', {}),
                        'events': data.get('metadata', {}).get('events', {})
                    }
                    doc_count += 1
                    
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
        
        # Load table chunks
        table_count = 0
        try:
            with open(self.table_chunks_file, 'r') as f:
                table_data = json.load(f)
            
            for chunk_info in table_data:
                chunk_id = chunk_info.get('chunk_id', '')

                if chunk_id not in self.indexing_filtered_data['chunk_id'].values:
                    continue
                else:
                    print(f"Chunk {chunk_id} is in indexing filtered data")

                if chunk_id:
                    metadata = chunk_info.get('metadata', {})
                    base_content = chunk_info.get('content', '')
                    title = metadata.get('table_title', '')
                    description = metadata.get('table_description', '')
                    
                    # Enhanced content: title + description + original content
                    enhanced_content = base_content
                    if title:
                        enhanced_content = title + " " + enhanced_content
                    if description:
                        enhanced_content = enhanced_content + " " + description if not title else title + " " + description + " " + base_content
                    
                    self.chunk_metadata[chunk_id] = {
                        'type': 'table',
                        'content': base_content,  # Keep original content
                        'enhanced_content': enhanced_content,  # Enhanced content for BM25
                        'title': title,
                        'description': description,
                        'column_descriptions': metadata.get('col_desc', {}),
                        'entities': metadata.get('entities', {}),
                        'events': metadata.get('events', {})
                    }
                    table_count += 1
                    
        except Exception as e:
            logger.error(f"Error loading table chunks: {e}")
        
        logger.info(f"Loaded metadata for {doc_count} documents and {table_count} tables")
    
    def extract_gold_docs(self, gold_str: str) -> List[str]:
        """Extract gold documents from string format (same as original function)"""
        if pd.isna(gold_str) or gold_str == '':
            return []
        
        # Clean the string
        gold_str = str(gold_str).strip()
        
        # If it's already a proper list string, use ast.literal_eval
        try:
            if gold_str.startswith('[') and gold_str.endswith(']'):
                gold_docs = ast.literal_eval(gold_str)
                return gold_docs if isinstance(gold_docs, list) else [gold_docs]
        except (ValueError, SyntaxError):
            pass
        
        # Try to extract from quoted string format
        try:
            # Remove outer quotes if present
            if gold_str.startswith('"') and gold_str.endswith('"'):
                gold_str = gold_str[1:-1]
            
            # Now try to parse as list
            if gold_str.startswith('[') and gold_str.endswith(']'):
                gold_docs = ast.literal_eval(gold_str)
                return gold_docs if isinstance(gold_docs, list) else [gold_docs]
        except (ValueError, SyntaxError):
            pass
        
        # If all else fails, try regex to extract items from list-like string
        match = re.search(r'\[(.*?)\]', gold_str)
        if match:
            items_str = match.group(1)
            # Split by comma and clean each item
            items = [item.strip().strip("'\"") for item in items_str.split(',')]
            return [item for item in items if item]
        
        # Last resort: return as single item if not empty
        return [gold_str] if gold_str else []
    
    def get_csv_retrieved_docs(self, row: pd.Series, k: int) -> List[str]:
        """Get top-k retrieved documents from CSV row"""
        retrieved_docs = []
        for i in range(1, min(k + 1, 101)):
            col_name = f"{i}st" if i == 1 else f"{i}nd" if i == 2 else f"{i}rd" if i == 3 else f"{i}th"
            if col_name in row and pd.notna(row[col_name]):
                retrieved_docs.append(row[col_name])
        
        return retrieved_docs[:k]
    
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
    
    def _calculate_hybrid_similarities(self, question_text: str, question_entities: List[str],
                                     question_events: List[str], neighbor_ids: List[str]) -> List[Tuple[str, float]]:
        """Calculate hybrid similarities: 50% embedding similarity + 50% BM25 score"""
        
        if not neighbor_ids:
            return []
        
        # Prepare content for BM25
        neighbor_contents = []
        neighbor_metadata = []
        
        for neighbor_id in neighbor_ids:
            if neighbor_id in self.chunk_metadata:
                metadata = self.chunk_metadata[neighbor_id]
                neighbor_metadata.append(metadata)
                
                # Use enhanced content for tables, regular content for documents
                if metadata['type'] == 'table':
                    content = metadata.get('enhanced_content', metadata['content'])
                else:
                    content = metadata['content']
                
                neighbor_contents.append(content)
            else:
                neighbor_metadata.append(None)
                neighbor_contents.append("")
        
        # Calculate embedding similarities (existing logic)
        embedding_similarities = self._calculate_neighbor_comprehensive_similarities(
            question_text, question_entities, question_events, neighbor_ids
        )
        
        # Calculate BM25 similarities
        bm25_similarities = self._calculate_bm25_similarities(question_text, neighbor_contents)
        
        # Combine similarities (50% each)
        hybrid_similarities = []
        embedding_dict = {neighbor_id: score for neighbor_id, score in embedding_similarities}
        
        for i, neighbor_id in enumerate(neighbor_ids):
            embedding_score = embedding_dict.get(neighbor_id, 0.0)
            bm25_score = bm25_similarities[i] if i < len(bm25_similarities) else 0.0

            # Combine with 45% weight for embedding and 55% for BM25
            hybrid_score = 0.40 * embedding_score + 0.60 * bm25_score
            hybrid_similarities.append((neighbor_id, hybrid_score))
        
        return hybrid_similarities
    
    def _calculate_bm25_similarities(self, question_text: str, neighbor_contents: List[str]) -> List[float]:
        """Calculate normalized BM25 similarities between question and neighbor contents"""
        
        # Preprocess question and contents
        question_tokens = preprocess_text_for_bm25(question_text)
        content_tokens = [preprocess_text_for_bm25(content) for content in neighbor_contents]
        
        if not question_tokens or not any(content_tokens):
            return [0.0] * len(neighbor_contents)
        
        # Build BM25 corpus
        bm25 = BM25(content_tokens)
        
        # Calculate BM25 scores
        bm25_scores = bm25.get_scores(question_tokens)
        
        # Normalize scores to [0, 1] range
        if bm25_scores:
            max_score = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
            normalized_scores = [score / max_score for score in bm25_scores]
        else:
            normalized_scores = [0.0] * len(neighbor_contents)
        
        return normalized_scores
    
    # Note: Old pre_calculate_all_similarities removed - now using GPUEmbeddingService for real-time calculations
    
    def get_graph_neighbors_for_retrieved_chunks(self, retrieved_chunks: List[str], 
                                                question_text: str, question_entities: List[str],
                                                question_events: List[str], top_k: int) -> List[str]:
        """Get top-k graph neighbors from retrieved chunks, ranked by comprehensive similarity"""
        
        # Collect all neighbors from retrieved chunks
        all_neighbors = set()
        
        for chunk_id in retrieved_chunks:
            if chunk_id in self.graph:
                neighbors = list(self.graph.neighbors(chunk_id))
                # Exclude retrieved chunks from neighbors
                neighbors = [n for n in neighbors if n not in retrieved_chunks]
                all_neighbors.update(neighbors)
        
        if not all_neighbors:
            return []
        
        # Calculate hybrid similarities for neighbors (50% embedding + 50% BM25)
        neighbor_similarities = self._calculate_hybrid_similarities(
            question_text, question_entities, question_events, list(all_neighbors)
        )
        
        # Sort by hybrid similarity (descending) and take top-k
        neighbor_similarities.sort(key=lambda x: x[1], reverse=True)
        top_neighbors = [neighbor_id for neighbor_id, _ in neighbor_similarities[:top_k]]
        
        return top_neighbors
    
    def _calculate_neighbor_comprehensive_similarities(self, question_text: str, 
                                                     question_entities: List[str], 
                                                     question_events: List[str],
                                                     neighbor_ids: List[str]) -> List[Tuple[str, float]]:
        """Calculate comprehensive similarities for neighbor candidates"""
        
        if not neighbor_ids:
            return []
        
        # Separate by type
        doc_neighbors = []
        table_neighbors = []
        
        for neighbor_id in neighbor_ids:
            if neighbor_id in self.chunk_metadata:
                neighbor_type = self.chunk_metadata[neighbor_id]['type']
                if neighbor_type == 'document':
                    doc_neighbors.append(neighbor_id)
                else:
                    table_neighbors.append(neighbor_id)
        
        similarities = []
        
        # Process documents with content + topic similarities
        if doc_neighbors:
            doc_similarities = self._calculate_document_similarities(
                question_text, question_entities, question_events, doc_neighbors
            )
            similarities.extend(doc_similarities)
        
        # Process tables with content + description + column similarities  
        if table_neighbors:
            table_similarities = self._calculate_table_similarities(
                question_text, question_entities, question_events, table_neighbors
            )
            similarities.extend(table_similarities)
        
        return similarities
    
    def _calculate_document_similarities(self, question_text: str, question_entities: List[str],
                                       question_events: List[str], doc_neighbor_ids: List[str]) -> List[Tuple[str, float]]:
        """Calculate comprehensive similarities for document neighbors"""
        
        question_texts = [question_text] * len(doc_neighbor_ids)
        content_texts = []
        topic_texts = []
        
        for neighbor_id in doc_neighbor_ids:
            metadata = self.chunk_metadata[neighbor_id]
            content_texts.append(metadata['content'])
            topic_texts.append(metadata['topic'])
        
        # Calculate similarities
        content_sims = self.embedding_service.calculate_similarities(
            question_texts, content_texts, "neighbor_doc_content"
        )
        topic_sims = self.embedding_service.calculate_similarities(
            question_texts, topic_texts, "neighbor_doc_topic"
        )
        
        similarities = []
        for i, neighbor_id in enumerate(doc_neighbor_ids):
            metadata = self.chunk_metadata[neighbor_id]
            
            # Calculate entity/event matches
            entity_exact, entity_substring = self._match_entities_events(
                question_entities, list(metadata['entities'].keys())
            )
            event_exact, event_substring = self._match_entities_events(
                question_events, list(metadata['events'].keys())
            )
            
            # Calculate bonuses
            entity_bonus = entity_exact * 0.1 + entity_substring * 0.05
            event_bonus = event_exact * 0.1 + event_substring * 0.05
            
            # Total similarity for documents = content + topic + bonuses
            total_similarity = float(content_sims[i] + topic_sims[i] + entity_bonus + event_bonus)
            similarities.append((neighbor_id, total_similarity))
        
        return similarities
    
    def _calculate_table_similarities(self, question_text: str, question_entities: List[str],
                                    question_events: List[str], table_neighbor_ids: List[str]) -> List[Tuple[str, float]]:
        """Calculate comprehensive similarities for table neighbors"""
        
        question_texts = [question_text] * len(table_neighbor_ids)
        enhanced_content_texts = []  # content + description + title
        description_texts = []
        column_texts = []
        
        for neighbor_id in table_neighbor_ids:
            metadata = self.chunk_metadata[neighbor_id]
            content = metadata['content']
            description = metadata['description']
            
            # Enhanced content: content + description + title (if available)
            enhanced_content = content
            if description:
                enhanced_content += " " + description
            
            enhanced_content_texts.append(enhanced_content)
            description_texts.append(description)
            
            # Column descriptions
            column_descs = metadata.get('column_descriptions', {})
            if column_descs:
                column_text = ' '.join(column_descs.values())
            else:
                column_text = ""
            column_texts.append(column_text)
        
        # Calculate similarities
        content_sims = self.embedding_service.calculate_similarities(
            question_texts, enhanced_content_texts, "neighbor_table_content"
        )
        desc_sims = self.embedding_service.calculate_similarities(
            question_texts, description_texts, "neighbor_table_desc"
        )
        column_sims = self.embedding_service.calculate_similarities(
            question_texts, column_texts, "neighbor_table_columns"
        )
        
        similarities = []
        for i, neighbor_id in enumerate(table_neighbor_ids):
            metadata = self.chunk_metadata[neighbor_id]
            
            # Calculate entity/event matches
            entity_exact, entity_substring = self._match_entities_events(
                question_entities, list(metadata['entities'].keys())
            )
            event_exact, event_substring = self._match_entities_events(
                question_events, list(metadata['events'].keys())
            )
            
            # Calculate bonuses
            entity_bonus = entity_exact * 0.1 + entity_substring * 0.05
            event_bonus = event_exact * 0.1 + event_substring * 0.05
            
            # Total similarity for tables = content + description + column + bonuses
            total_similarity = float(content_sims[i] + desc_sims[i] + column_sims[i] + entity_bonus + event_bonus)
            similarities.append((neighbor_id, total_similarity))
        
        return similarities
    
    def analyze_csv_vs_graph_comparison(self, row: pd.Series, csv_k: int, graph_k: int) -> CSVvsGraphComparison:
        """Compare CSV+Graph vs Extended CSV approach"""
        
        question_id = row['question_id']
        question_text = row['question']
        gold_docs = self.extract_gold_docs(row['gold_docs'])
        
        # Extract entities and events from questions metadata
        question_entities = []
        question_events = []
        
        if question_text in self.questions_metadata:
            metadata = self.questions_metadata[question_text]
            question_entities = metadata.get('entities', [])
            question_events = metadata.get('events', [])
        
        # Get first csv_k results
        csv_k_retrieved = self.get_csv_retrieved_docs(row, csv_k)
        
        # Get extended CSV results (total csv_k + graph_k)
        extended_csv_retrieved = self.get_csv_retrieved_docs(row, csv_k + graph_k)
        
        # Get graph neighbors
        graph_neighbors = self.get_graph_neighbors_for_retrieved_chunks(
            csv_k_retrieved, question_text, question_entities, question_events, graph_k
        )
        
        # Analyze gold document presence
        gold_found_in_csv_k = [doc for doc in gold_docs if doc in csv_k_retrieved]
        gold_found_in_graph = [doc for doc in gold_docs if doc in graph_neighbors]
        gold_found_in_extended_csv = [doc for doc in gold_docs if doc in extended_csv_retrieved]
        
        # Determine results
        csv_plus_graph_candidates = csv_k_retrieved + graph_neighbors
        csv_plus_graph_found_gold = any(doc in csv_plus_graph_candidates for doc in gold_docs)
        extended_csv_found_gold = len(gold_found_in_extended_csv) > 0
        
        # Determine scenario
        if csv_plus_graph_found_gold and extended_csv_found_gold:
            scenario = "both_win"
        elif csv_plus_graph_found_gold and not extended_csv_found_gold:
            scenario = "graph_wins"
        elif not csv_plus_graph_found_gold and extended_csv_found_gold:
            scenario = "csv_wins"
        else:
            scenario = "both_lose"
        
        # Find missed gold docs
        all_found_gold = set(gold_found_in_csv_k + gold_found_in_graph + gold_found_in_extended_csv)
        missed_gold_docs = [doc for doc in gold_docs if doc not in all_found_gold]
        
        return CSVvsGraphComparison(
            question_id=question_id,
            question_text=question_text,
            gold_docs=gold_docs,
            csv_k_retrieved=csv_k_retrieved,
            graph_neighbors=graph_neighbors,
            extended_csv_retrieved=extended_csv_retrieved,
            csv_plus_graph_found_gold=csv_plus_graph_found_gold,
            extended_csv_found_gold=extended_csv_found_gold,
            scenario=scenario,
            gold_found_in_csv_k=gold_found_in_csv_k,
            gold_found_in_graph=gold_found_in_graph,
            gold_found_in_extended_csv=gold_found_in_extended_csv,
            missed_gold_docs=missed_gold_docs
        )
    
    def run_csv_vs_graph_analysis(self) -> Dict[str, Any]:
        """Run comprehensive CSV vs Graph comparison analysis"""
        
        logger.info("Starting CSV vs Graph comparison analysis...")
        
        # Define comparison configurations
        comparisons = [
            {"csv_k": 1, "graph_k": 1, "name": "CSV1_vs_Graph1_vs_CSV2"},
            {"csv_k": 2, "graph_k": 3, "name": "CSV2_vs_Graph3_vs_CSV5"},
            {"csv_k": 5, "graph_k": 5, "name": "CSV5_vs_Graph5_vs_CSV10"},
            {"csv_k": 10, "graph_k": 10, "name": "CSV10_vs_Graph10_vs_CSV20"},
            {"csv_k": 20, "graph_k": 30, "name": "CSV20_vs_Graph30_vs_CSV50"},
            {"csv_k": 25, "graph_k": 25, "name": "CSV25_vs_Graph25_vs_CSV50"},
            {"csv_k": 30, "graph_k": 20, "name": "CSV30_vs_Graph20_vs_CSV50"},
            {"csv_k": 50, "graph_k": 50, "name": "CSV50_vs_Graph50_vs_CSV100"}
        ]
        
        all_comparison_results = {}
        
        for config in comparisons:
            csv_k = config["csv_k"]
            graph_k = config["graph_k"]
            name = config["name"]
            
            logger.info(f"Analyzing {name}...")
            
            comparison_results = []
            scenario_counts = {"graph_wins": 0, "csv_wins": 0, "both_win": 0, "both_lose": 0}
            
            for _, row in self.df.iterrows():
                gold_docs = self.extract_gold_docs(row['gold_docs'])
                if not gold_docs:  # Skip questions without gold docs
                    continue
                
                comparison = self.analyze_csv_vs_graph_comparison(row, csv_k, graph_k)
                comparison_results.append(comparison)
                scenario_counts[comparison.scenario] += 1
            
            # Calculate accuracies
            total_questions = len(comparison_results)
            csv_plus_graph_accuracy = sum(1 for r in comparison_results if r.csv_plus_graph_found_gold) / total_questions
            extended_csv_accuracy = sum(1 for r in comparison_results if r.extended_csv_found_gold) / total_questions
            
            all_comparison_results[name] = {
                "config": config,
                "results": comparison_results,
                "scenario_counts": scenario_counts,
                "total_questions": total_questions,
                "csv_plus_graph_accuracy": csv_plus_graph_accuracy,
                "extended_csv_accuracy": extended_csv_accuracy,
                "accuracy_difference": extended_csv_accuracy - csv_plus_graph_accuracy
            }
            
            logger.info(f"{name} complete: CSV+Graph: {csv_plus_graph_accuracy:.3f}, Extended CSV: {extended_csv_accuracy:.3f}")
        
        # Save detailed analysis
        self._save_csv_vs_graph_analysis(all_comparison_results)
        
        return all_comparison_results
    
    def _save_csv_vs_graph_analysis(self, all_results: Dict[str, Any]):
        """Save CSV vs Graph comparison analysis to files"""
        
        analysis_dir = self.cache_dir / "csv_vs_graph_analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        for name, data in all_results.items():
            config = data["config"]
            results = data["results"]
            
            # Create subdirectory for this comparison
            comp_dir = analysis_dir / name
            comp_dir.mkdir(exist_ok=True)
            
            # Save scenario-specific files
            graph_wins = [r for r in results if r.scenario == "graph_wins"]
            csv_wins = [r for r in results if r.scenario == "csv_wins"]
            both_win = [r for r in results if r.scenario == "both_win"]
            both_lose = [r for r in results if r.scenario == "both_lose"]
            
            # Save graph wins (where graph found gold but extended CSV didn't)
            if graph_wins:
                self._save_scenario_analysis(comp_dir / "graph_wins.json", graph_wins, 
                                           f"Questions where Graph k={config['graph_k']} found gold but CSV k={config['csv_k'] + config['graph_k']} didn't")
            
            # Save CSV wins (where extended CSV found gold but graph didn't)
            if csv_wins:
                self._save_scenario_analysis(comp_dir / "csv_wins.json", csv_wins,
                                           f"Questions where CSV k={config['csv_k'] + config['graph_k']} found gold but CSV k={config['csv_k']} + Graph k={config['graph_k']} didn't")
            
            # Save both win cases
            if both_win:
                self._save_scenario_analysis(comp_dir / "both_win.json", both_win,
                                           "Questions where both approaches found gold")
            
            # Save both lose cases
            if both_lose:
                self._save_scenario_analysis(comp_dir / "both_lose.json", both_lose,
                                           "Questions where neither approach found gold")
            
            # Save summary statistics
            summary = {
                "comparison_name": name,
                "configuration": config,
                "total_questions": data["total_questions"],
                "accuracy_comparison": {
                    "csv_plus_graph_accuracy": data["csv_plus_graph_accuracy"],
                    "extended_csv_accuracy": data["extended_csv_accuracy"],
                    "difference": data["accuracy_difference"],
                    "winner": "Extended CSV" if data["accuracy_difference"] > 0 else "CSV+Graph" if data["accuracy_difference"] < 0 else "Tie"
                },
                "scenario_breakdown": data["scenario_counts"],
                "scenario_percentages": {
                    scenario: count / data["total_questions"] * 100 
                    for scenario, count in data["scenario_counts"].items()
                }
            }
            
            with open(comp_dir / "summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
        
        # Save overall comparison summary
        overall_summary = {
            "analysis_type": "CSV vs Graph Comparison",
            "comparisons": {}
        }
        
        for name, data in all_results.items():
            overall_summary["comparisons"][name] = {
                "config": data["config"],
                "csv_plus_graph_accuracy": data["csv_plus_graph_accuracy"],
                "extended_csv_accuracy": data["extended_csv_accuracy"],
                "accuracy_difference": data["accuracy_difference"],
                "scenario_counts": data["scenario_counts"]
            }
        
        with open(analysis_dir / "overall_summary.json", 'w') as f:
            json.dump(overall_summary, f, indent=2)
        
        logger.info(f"CSV vs Graph analysis saved to {analysis_dir}")
    
    def _save_scenario_analysis(self, file_path: Path, scenario_results: List[CSVvsGraphComparison], description: str):
        """Save detailed analysis for a specific scenario"""
        
        analysis_data = {
            "description": description,
            "total_questions": len(scenario_results),
            "questions": []
        }
        
        for result in scenario_results:
            question_data = {
                "question_id": result.question_id,
                "question_text": result.question_text,
                "gold_docs": result.gold_docs,
                "analysis": {
                    "csv_k_retrieved": result.csv_k_retrieved,
                    "graph_neighbors": result.graph_neighbors,
                    "extended_csv_retrieved": result.extended_csv_retrieved,
                    "gold_found_in_csv_k": result.gold_found_in_csv_k,
                    "gold_found_in_graph": result.gold_found_in_graph,
                    "gold_found_in_extended_csv": result.gold_found_in_extended_csv,
                    "missed_gold_docs": result.missed_gold_docs
                },
                "retrieval_details": {
                    "csv_k_but_not_gold": [doc for doc in result.csv_k_retrieved if doc not in result.gold_docs],
                    "graph_but_not_gold": [doc for doc in result.graph_neighbors if doc not in result.gold_docs],
                    "extended_csv_but_not_gold": [doc for doc in result.extended_csv_retrieved if doc not in result.gold_docs]
                }
            }
            
            analysis_data["questions"].append(question_data)
        
        with open(file_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)

    def analyze_question_accuracy(self, row: pd.Series, test_config: KValueTest) -> QuestionAccuracyResult:
        """Analyze accuracy for a single question with given k-values"""
        
        question_id = row['question_id']
        question_text = row['question']
        gold_docs = self.extract_gold_docs(row['gold_docs'])
        
        # Extract entities and events from questions metadata
        question_entities = []
        question_events = []
        
        if question_text in self.questions_metadata:
            metadata = self.questions_metadata[question_text]
            question_entities = metadata.get('entities', [])
            question_events = metadata.get('events', [])
        
        if not gold_docs:
            # Skip questions without gold docs
            return QuestionAccuracyResult(
                question_id=question_id,
                gold_docs=[],
                csv_only_correct=False,
                csv_plus_graph_correct=False,
                graph_enhancement=False,
                csv_retrieved=[],
                graph_neighbors=[],
                found_gold_in_csv=[],
                found_gold_in_graph=[]
            )
        
        # Get CSV retrieved documents
        csv_retrieved = self.get_csv_retrieved_docs(row, test_config.csv_k)
        
        # Check CSV-only accuracy
        found_gold_in_csv = [doc for doc in gold_docs if doc in csv_retrieved]
        csv_only_correct = len(found_gold_in_csv) > 0
        
        # Get graph neighbors using comprehensive similarity
        graph_neighbors = self.get_graph_neighbors_for_retrieved_chunks(
            csv_retrieved, question_text, question_entities, question_events, test_config.graph_k
        )
        
        # Check combined CSV + Graph accuracy
        all_candidates = csv_retrieved + graph_neighbors
        found_gold_in_graph = [doc for doc in gold_docs if doc in graph_neighbors]
        found_gold_total = [doc for doc in gold_docs if doc in all_candidates]
        csv_plus_graph_correct = len(found_gold_total) > 0
        
        # Determine if graph provided enhancement
        graph_enhancement = (not csv_only_correct) and csv_plus_graph_correct
        
        return QuestionAccuracyResult(
            question_id=question_id,
            gold_docs=gold_docs,
            csv_only_correct=csv_only_correct,
            csv_plus_graph_correct=csv_plus_graph_correct,
            graph_enhancement=graph_enhancement,
            csv_retrieved=csv_retrieved,
            graph_neighbors=graph_neighbors,
            found_gold_in_csv=found_gold_in_csv,
            found_gold_in_graph=found_gold_in_graph
        )
    
    def run_k_value_test(self, test_config: KValueTest) -> KValueResult:
        """Run accuracy test for a specific k-value combination"""
        
        logger.info(f"Running test: {test_config.name} (CSV k={test_config.csv_k}, Graph k={test_config.graph_k})")
        
        question_results = []
        csv_only_correct = 0
        csv_plus_graph_correct = 0
        questions_enhanced = 0
        total_questions = 0
        
        for _, row in self.df.iterrows():
            result = self.analyze_question_accuracy(row, test_config)
            
            # Skip questions without gold docs
            if not result.gold_docs:
                continue
            
            question_results.append(result)
            total_questions += 1
            
            if result.csv_only_correct:
                csv_only_correct += 1
            
            if result.csv_plus_graph_correct:
                csv_plus_graph_correct += 1
            
            if result.graph_enhancement:
                questions_enhanced += 1
        
        # Calculate accuracies
        csv_only_accuracy = csv_only_correct / total_questions if total_questions > 0 else 0.0
        csv_plus_graph_accuracy = csv_plus_graph_correct / total_questions if total_questions > 0 else 0.0
        improvement = csv_plus_graph_accuracy - csv_only_accuracy
        
        result = KValueResult(
            test_config=test_config,
            total_questions=total_questions,
            csv_only_accuracy=csv_only_accuracy,
            csv_plus_graph_accuracy=csv_plus_graph_accuracy,
            improvement=improvement,
            questions_enhanced=questions_enhanced,
            questions_results=question_results
        )
        
        logger.info(f"Test {test_config.name} complete: "
                   f"CSV-only: {csv_only_accuracy:.3f}, "
                   f"CSV+Graph: {csv_plus_graph_accuracy:.3f}, "
                   f"Improvement: {improvement:.3f}, "
                   f"Enhanced: {questions_enhanced}")
        
        return result
    
    def run_all_tests(self) -> List[KValueResult]:
        """Run all k-value tests"""
        logger.info("Starting all k-value tests...")
        
        all_results = []
        
        for test_config in self.k_value_tests:
            result = self.run_k_value_test(test_config)
            all_results.append(result)
        
        # Save results
        self._save_results(all_results)
        
        logger.info(f"All tests complete! Results saved to {self.cache_dir}")
        return all_results
    
    def _save_results(self, all_results: List[KValueResult]):
        """Save all results to files"""
        
        # Save detailed results
        results_file = self.cache_dir / "k_value_test_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(all_results, f)
        
        # Save summary JSON
        summary = {
            'test_summary': [],
            'overall_statistics': {
                'total_tests': len(all_results),
                'avg_csv_only_accuracy': np.mean([r.csv_only_accuracy for r in all_results]),
                'avg_csv_plus_graph_accuracy': np.mean([r.csv_plus_graph_accuracy for r in all_results]),
                'avg_improvement': np.mean([r.improvement for r in all_results]),
                'total_questions_analyzed': all_results[0].total_questions if all_results else 0
            }
        }
        
        for result in all_results:
            summary['test_summary'].append({
                'test_name': result.test_config.name,
                'csv_k': result.test_config.csv_k,
                'graph_k': result.test_config.graph_k,
                'total_questions': result.total_questions,
                'csv_only_accuracy': result.csv_only_accuracy,
                'csv_plus_graph_accuracy': result.csv_plus_graph_accuracy,
                'improvement': result.improvement,
                'questions_enhanced': result.questions_enhanced,
                'enhancement_rate': result.questions_enhanced / result.total_questions if result.total_questions > 0 else 0.0
            })
        
        summary_file = self.cache_dir / "k_value_test_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {results_file} and {summary_file}")

def main():
    """Main execution function"""
    
    logger.info("Starting Graph Enhanced Accuracy Analysis...")
    
    # Initialize analyzer
    analyzer = GraphEnhancedAccuracyAnalyzer()
    
    # Load all data
    analyzer.load_all_data()
    
    # Run CSV vs Graph comparison analysis
    logger.info("\n" + "="*80)
    logger.info("RUNNING CSV vs GRAPH COMPARISON ANALYSIS")
    logger.info("="*80)
    
    comparison_results = analyzer.run_csv_vs_graph_analysis()
    
    # Print CSV vs Graph comparison summary
    print("\n" + "="*80)
    print("CSV vs GRAPH COMPARISON SUMMARY")
    print("="*80)
    
    for name, data in comparison_results.items():
        config = data["config"]
        csv_k = config["csv_k"]
        graph_k = config["graph_k"]
        total_k = csv_k + graph_k
        
        print(f"\n{name}:")
        print(f"  Comparison: CSV{csv_k} + Graph{graph_k} vs CSV{total_k}")
        print(f"  CSV+Graph accuracy: {data['csv_plus_graph_accuracy']:.3f} ({data['csv_plus_graph_accuracy']*100:.1f}%)")
        print(f"  Extended CSV accuracy: {data['extended_csv_accuracy']:.3f} ({data['extended_csv_accuracy']*100:.1f}%)")
        print(f"  Difference: {data['accuracy_difference']:.3f} ({data['accuracy_difference']*100:.1f}%)")
        
        winner = "Extended CSV" if data['accuracy_difference'] > 0 else "CSV+Graph" if data['accuracy_difference'] < 0 else "Tie"
        print(f"  Winner: {winner}")
        
        print(f"  Scenarios:")
        for scenario, count in data['scenario_counts'].items():
            percentage = count / data['total_questions'] * 100
            print(f"    {scenario}: {count} ({percentage:.1f}%)")
    
    # Run all k-value tests
    logger.info("\n" + "="*80)
    logger.info("RUNNING STANDARD K-VALUE TESTS")
    logger.info("="*80)
    
    results = analyzer.run_all_tests()
    
    # Save embedding cache
    cache_file = analyzer.cache_dir / "embedding_cache.pkl"
    analyzer.embedding_service.save_cache(cache_file)
    
    # Print k-value summary
    print("\n" + "="*80)
    print("STANDARD K-VALUE ANALYSIS SUMMARY")
    print("="*80)
    
    for result in results:
        improvement_pct = result.improvement * 100
        enhancement_rate = result.questions_enhanced / result.total_questions * 100 if result.total_questions > 0 else 0
        
        print(f"\n{result.test_config.name}:")
        print(f"  CSV k={result.test_config.csv_k}, Graph k={result.test_config.graph_k}")
        print(f"  CSV-only accuracy: {result.csv_only_accuracy:.3f} ({result.csv_only_accuracy*100:.1f}%)")
        print(f"  CSV+Graph accuracy: {result.csv_plus_graph_accuracy:.3f} ({result.csv_plus_graph_accuracy*100:.1f}%)")
        print(f"  Improvement: +{improvement_pct:.1f}% ({result.questions_enhanced} questions enhanced)")
        print(f"  Enhancement rate: {enhancement_rate:.1f}% of questions")
    
    logger.info("All analysis complete!")

if __name__ == "__main__":
    main()