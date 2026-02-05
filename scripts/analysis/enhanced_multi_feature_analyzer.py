#!/usr/bin/env python3
"""
Enhanced Multi-Feature Retrieval Analyzer
Uses all HNSW indices to enhance retrieval accuracy with ASIN-level merging
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
import torch
import faiss
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from tqdm import tqdm
import ast
import re
import math
from sentence_transformers import SentenceTransformer
import gc
from loguru import logger

# BM25 implementation
def load_stopwords(language: str = 'english') -> Set[str]:
    """Load stopwords from local file"""
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

@dataclass
class MultiFeatureRetrievalResult:
    """Result for multi-feature enhanced retrieval"""
    query_idx: int
    query: str
    answer_asins: List[str] # Gold ASIN IDs
    baseline_retrieved: List[str] # Baseline retrieved ASIN IDs
    
    # Enhanced retrieval results for different k values
    enhanced_results: Dict[int, Dict[str, Any]] # k -> results
    
    # Metrics comparison
    metrics_comparison: Dict[str, Dict[int, float]] # metric -> k -> value

class GPUEmbeddingService:
    """GPU-accelerated embedding generation service"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 1000, gpu_id: int = 0):
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name, device=self.device)
        self.batch_size = batch_size
        
        logger.info(f"Initialized GPUEmbeddingService on {self.device}")
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query"""
        embedding = self.model.encode([query], convert_to_tensor=True, normalize_embeddings=True)
        return embedding.cpu().numpy()[0]

class EnhancedMultiFeatureAnalyzer:
    """
    Enhanced analyzer using all HNSW feature indices for retrieval
    """
    
    def __init__(self,
                 bm25_results_file: str = "/shared/khoja/CogComp/output/bm25_results.csv",
                 hnsw_indices_dir: str = "/shared/khoja/CogComp/output/multi_feature_hnsw",
                 chunked_cache_dir: str = "/shared/khoja/CogComp/output/stark_chunked_cache",
                 stark_dataset_file: str = "/shared/khoja/CogComp/datasets/STARK/node_info.json",
                 output_dir: str = "/shared/khoja/CogComp/output/enhanced_multi_feature_analysis",
                 gpu_id: int = 0):
        
        self.bm25_results_file = bm25_results_file
        self.hnsw_indices_dir = Path(hnsw_indices_dir)
        self.chunked_cache_dir = Path(chunked_cache_dir)
        self.stark_dataset_file = stark_dataset_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # GPU embedding service
        self.embedding_service = GPUEmbeddingService(gpu_id=gpu_id)
        
        # Data containers
        self.bm25_df = None
        self.hnsw_indices = {} # feature_name -> faiss.Index
        self.hnsw_mappings = {} # feature_name -> list of mappings
        self.chunk_content_cache = {} # chunk_id -> content
        self.asin_to_chunks = {} # asin -> list of chunk_ids
        self.stark_id_to_asin = {} # stark_id -> asin mapping
        
        # Feature names
        self.embedding_features = [
            'content_embedding',
            'title_embedding', 
            'feature_embedding',
            'detail_embedding',
            'description_embedding',
            'reviews_summary_embedding',
            'reviews_text_embedding'
        ]
        
        # k values to test
        self.k_values = [1, 5, 10, 20, 50, 100]
        
        logger.info(f" Initialized Enhanced Multi-Feature Analyzer")
        logger.info(f" HNSW indices dir: {self.hnsw_indices_dir}")
        logger.info(f" Output dir: {self.output_dir}")
    
    def load_all_data(self):
        """Load all required data"""
        logger.info(" Loading all data...")
        
        self._load_stark_id_mapping()
        self._load_bm25_results()
        self._load_hnsw_indices()
        self._load_chunk_content_cache()
        self._build_asin_mapping()
        
        logger.info(" All data loaded successfully")
    
    def _load_stark_id_mapping(self):
        """Load STARK ID to ASIN mapping"""
        logger.info(f" Loading STARK ID to ASIN mapping from: {self.stark_dataset_file}")
        
        with open(self.stark_dataset_file, 'r') as f:
            stark_data = json.load(f)
        
        for stark_id, product_data in stark_data.items():
            asin = product_data.get('asin')
            if asin:
                self.stark_id_to_asin[stark_id] = asin
        
        logger.info(f" Loaded mapping for {len(self.stark_id_to_asin):,} STARK IDs")
    
    def _load_bm25_results(self):
        """Load BM25 baseline results"""
        logger.info(f" Loading BM25 results from: {self.bm25_results_file}")
        self.bm25_df = pd.read_csv(self.bm25_results_file)
        logger.info(f" Loaded {len(self.bm25_df)} queries")
    
    def _load_hnsw_indices(self):
        """Load all HNSW indices and mappings"""
        logger.info(" Loading HNSW indices...")
        
        for feature in self.embedding_features:
            index_file = self.hnsw_indices_dir / f"{feature}_hnsw_index.faiss"
            mapping_file = self.hnsw_indices_dir / f"{feature}_mapping.pkl"
            
            if index_file.exists() and mapping_file.exists():
                # Load index
                index = faiss.read_index(str(index_file))
                self.hnsw_indices[feature] = index
                
                # Load mapping
                with open(mapping_file, 'rb') as f:
                    mapping = pickle.load(f)
                self.hnsw_mappings[feature] = mapping
                
                logger.info(f" {feature}: {index.ntotal:,} embeddings")
            else:
                logger.warning(f" Missing files for {feature}")
    
    def _load_chunk_content_cache(self):
        """Load chunk content for BM25 scoring using reconstructed STARK content"""
        logger.info(" Loading chunk content cache...")
        
        # Load reconstructed content mapping
        content_mapping_file = Path('/shared/khoja/CogComp/output/asin_to_content_mapping.json')
        if not content_mapping_file.exists():
            logger.error(f" Content mapping not found: {content_mapping_file}")
            logger.error(" Run fix_content_reconstruction.py first!")
            return
        
        logger.info(f" Loading reconstructed content from: {content_mapping_file}")
        with open(content_mapping_file, 'r') as f:
            asin_to_content = json.load(f)
        
        logger.info(f" Loaded content for {len(asin_to_content):,} ASINs")
        
        # Now load chunk files to map chunk_id to content via ASIN
        embeddings_dir = self.chunked_cache_dir / "embeddings"
        chunk_files = sorted(embeddings_dir.glob("chunk_*_embeddings.json"))
        
        total_chunks = 0
        content_found = 0
        
        for chunk_file in tqdm(chunk_files, desc=" Mapping content"):
            with open(chunk_file, 'r') as f:
                chunks_data = json.load(f)
            
            for chunk_data in chunks_data:
                chunk_id = chunk_data.get('chunk_id', '')
                asin = chunk_data.get('asin', '')
                chunk_type = chunk_data.get('chunk_type', '')
                
                if chunk_id and asin:
                    # Use reconstructed content for document chunks
                    if chunk_type == 'document' and asin in asin_to_content:
                        content = asin_to_content[asin]
                        self.chunk_content_cache[chunk_id] = content
                        content_found += 1
                    else:
                        # Empty content for table chunks or missing ASINs
                        self.chunk_content_cache[chunk_id] = ''
                    
                    total_chunks += 1
        
        logger.info(f" Loaded content for {total_chunks:,} chunks ({content_found:,} with actual content)")
    
    def _build_asin_mapping(self):
        """Build mapping from ASIN to chunk IDs"""
        logger.info(" Building ASIN to chunk mapping...")
        
        for feature in self.hnsw_mappings:
            for mapping_entry in self.hnsw_mappings[feature]:
                asin = mapping_entry['asin']
                chunk_id = mapping_entry['chunk_id']
                
                if asin not in self.asin_to_chunks:
                    self.asin_to_chunks[asin] = []
                
                if chunk_id not in self.asin_to_chunks[asin]:
                    self.asin_to_chunks[asin].append(chunk_id)
        
        logger.info(f" Built mapping for {len(self.asin_to_chunks):,} ASINs")
    

    def extract_asin_list(self, id_str: str, convert_to_asin: bool = True) -> List[str]:
        """Extract ASIN list from string format, converting STARK IDs to ASINs if needed"""
        if pd.isna(id_str) or id_str == '':
            return []
        
        # Clean the string
        id_str = str(id_str).strip()
        
        # Handle list format
        stark_ids = []
        try:
            if id_str.startswith('[') and id_str.endswith(']'):
                id_list = ast.literal_eval(id_str)
                stark_ids = [str(id_val) for id_val in id_list]
        except (ValueError, SyntaxError):
            pass
        
        # Handle quoted list format
        if not stark_ids:
            try:
                if id_str.startswith('"') and id_str.endswith('"'):
                    id_str = id_str[1:-1]
                    if id_str.startswith('[') and id_str.endswith(']'):
                        id_list = ast.literal_eval(id_str)
                        stark_ids = [str(id_val) for id_val in id_list]
            except (ValueError, SyntaxError):
                pass
        
        if not convert_to_asin:
            return stark_ids
        
        # Convert STARK IDs to ASINs
        asins = []
        for stark_id in stark_ids:
            if stark_id in self.stark_id_to_asin:
                asins.append(self.stark_id_to_asin[stark_id])
        
        return asins
    
    def get_multi_feature_neighbors(self, retrieved_asins: List[str], query_embedding: np.ndarray, 
                                   k_neighbors: int = 100) -> Set[str]:
        """Get neighbors using efficient batched HNSW searches for retrieved ASINs"""
        
        all_neighbor_asins = set()
        
        # Collect all unique chunk IDs from retrieved ASINs
        all_chunk_ids = []
        for asin in retrieved_asins:
            if asin in self.asin_to_chunks:
                chunk_ids = self.asin_to_chunks[asin]
                all_chunk_ids.extend(chunk_ids)
        
        # Remove duplicates while preserving order
        unique_chunk_ids = list(dict.fromkeys(all_chunk_ids))
        
        if not unique_chunk_ids:
            return all_neighbor_asins
        
        # For each HNSW feature index, do batched search
        for feature_name, hnsw_index in self.hnsw_indices.items():
            mapping = self.hnsw_mappings[feature_name]
            
            # Find which chunks exist in this feature index
            chunk_to_idx = {}
            embeddings_to_search = []
            valid_chunk_ids = []
            
            for chunk_id in unique_chunk_ids:
                # Find this chunk in this feature index
                chunk_indices = [i for i, m in enumerate(mapping) if m['chunk_id'] == chunk_id]
                
                for chunk_idx in chunk_indices:
                    try:
                        # Get embedding for this chunk
                        embedding = hnsw_index.reconstruct(chunk_idx)
                        embeddings_to_search.append(embedding)
                        valid_chunk_ids.append(chunk_id)
                        chunk_to_idx[len(embeddings_to_search) - 1] = chunk_idx # Map batch index to original index
                    except Exception as e:
                        logger.warning(f"Error reconstructing embedding for {chunk_id} in {feature_name}: {e}")
                        continue
            
            if not embeddings_to_search:
                continue
            
            try:
                # Batch search for all chunks at once
                embeddings_matrix = np.array(embeddings_to_search).astype(np.float32)
                k_search = min(k_neighbors + 10, hnsw_index.ntotal) # Extra to account for self-exclusion
                
                distances, indices = hnsw_index.search(embeddings_matrix, k_search)
                
                # Process results for each query chunk
                for batch_idx, (chunk_distances, chunk_indices) in enumerate(zip(distances, indices)):
                    original_chunk_idx = chunk_to_idx[batch_idx]
                    
                    # Collect neighbor ASINs, excluding self
                    for neighbor_idx in chunk_indices:
                        if neighbor_idx < len(mapping) and neighbor_idx != original_chunk_idx: # Exclude self
                            neighbor_asin = mapping[neighbor_idx]['asin']
                            all_neighbor_asins.add(neighbor_asin)
                            
            except Exception as e:
                logger.warning(f"Error in batched HNSW search for {feature_name}: {e}")
                continue
        
        # Remove original retrieved ASINs from neighbors (exclude self)
        all_neighbor_asins = all_neighbor_asins - set(retrieved_asins)
        
        return all_neighbor_asins
    
    def calculate_hybrid_scores(self, query: str, candidate_asins: List[str]) -> List[Tuple[str, float]]:
        """Calculate hybrid scores (50% BM25 + 50% embedding similarity) with fallback to embedding-only"""
        
        if not candidate_asins:
            return []
        
        # Get content for BM25
        candidate_contents = []
        valid_asins = []
        has_content_count = 0
        
        for asin in candidate_asins:
            if asin in self.asin_to_chunks:
                chunk_ids = self.asin_to_chunks[asin]
                
                # Combine content from all chunks for this ASIN
                asin_content_parts = []
                for chunk_id in chunk_ids:
                    if chunk_id in self.chunk_content_cache:
                        content = self.chunk_content_cache[chunk_id]
                        if content.strip():
                            asin_content_parts.append(content.strip())
                
                combined_content = ' '.join(asin_content_parts)
                candidate_contents.append(combined_content)
                valid_asins.append(asin)
                
                if combined_content.strip():
                    has_content_count += 1
        
        if not valid_asins:
            return []
        
        # Log content availability but always use hybrid scoring
        logger.info(f" Content available for {has_content_count}/{len(valid_asins)} ASINs ({has_content_count/len(valid_asins)*100:.1f}%)")
        
        # Calculate BM25 scores
        query_tokens = preprocess_text_for_bm25(query)
        content_tokens = [preprocess_text_for_bm25(content) for content in candidate_contents]
        
        bm25_scores = [0.0] * len(valid_asins)
        if query_tokens and any(content_tokens):
            bm25 = BM25(content_tokens)
            bm25_scores = bm25.get_scores(query_tokens)
            
            # Normalize BM25 scores
            if bm25_scores and max(bm25_scores) > 0:
                max_bm25 = max(bm25_scores)
                bm25_scores = [score / max_bm25 for score in bm25_scores]
        
        # Calculate embedding similarities
        query_embedding = self.embedding_service.encode_query(query)
        embedding_scores = []
        
        for content in candidate_contents:
            if content.strip():
                content_embedding = self.embedding_service.encode_query(content)
                similarity = np.dot(query_embedding, content_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(content_embedding)
                )
                embedding_scores.append(float(similarity))
            else:
                embedding_scores.append(0.0)
        
        # Combine scores (50% each)
        hybrid_scores = []
        for i, asin in enumerate(valid_asins):
            hybrid_score = 0.5 * bm25_scores[i] + 0.5 * embedding_scores[i]
            hybrid_scores.append((asin, hybrid_score))
        
        return hybrid_scores
    
    def _calculate_embedding_only_scores(self, query: str, candidate_asins: List[str]) -> List[Tuple[str, float]]:
        """Calculate embedding-only scores by averaging embeddings across all features for each ASIN"""
        
        query_embedding = self.embedding_service.encode_query(query)
        embedding_scores = []
        
        for asin in candidate_asins:
            if asin not in self.asin_to_chunks:
                embedding_scores.append((asin, 0.0))
                continue
            
            chunk_ids = self.asin_to_chunks[asin]
            asin_similarities = []
            
            # Collect similarities from all available features for this ASIN
            for feature_name, hnsw_index in self.hnsw_indices.items():
                mapping = self.hnsw_mappings[feature_name]
                
                for chunk_id in chunk_ids:
                    # Find this chunk in this feature index
                    chunk_indices = [i for i, m in enumerate(mapping) if m['chunk_id'] == chunk_id]
                    
                    for chunk_idx in chunk_indices:
                        try:
                            # Get chunk embedding from HNSW index
                            chunk_embedding = hnsw_index.reconstruct(chunk_idx)
                            
                            # Calculate similarity
                            similarity = np.dot(query_embedding, chunk_embedding) / (
                                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                            )
                            asin_similarities.append(float(similarity))
                            
                        except Exception as e:
                            logger.warning(f"Error computing embedding similarity for {chunk_id} in {feature_name}: {e}")
                            continue
            
            # Average similarities for this ASIN
            if asin_similarities:
                avg_similarity = np.mean(asin_similarities)
                embedding_scores.append((asin, avg_similarity))
            else:
                embedding_scores.append((asin, 0.0))
        
        return embedding_scores
    
    def enhance_retrieval_for_query(self, row: pd.Series, k_retrieve: int = 100) -> Dict[str, Any]:
        """Enhance retrieval for a single query using multi-feature HNSW"""
        
        query = row['query']
        answer_asins = self.extract_asin_list(row['answer_ids'], convert_to_asin=True)
        baseline_retrieved = self.extract_asin_list(row['retrieved_docs'], convert_to_asin=True)
        
        # Take first k_retrieve from baseline
        baseline_k = baseline_retrieved[:k_retrieve]
        
        # Generate query embedding
        query_embedding = self.embedding_service.encode_query(query)
        
        # Get multi-feature neighbors
        neighbor_asins = self.get_multi_feature_neighbors(baseline_k, query_embedding, k_neighbors=100)
        
        # Score ONLY the neighbors (not baseline + neighbors)
        neighbor_list = list(neighbor_asins)
        scored_neighbors = self.calculate_hybrid_scores(query, neighbor_list)
        scored_neighbors.sort(key=lambda x: x[1], reverse=True)
        
        # Take top k neighbors and add to baseline
        top_neighbors = [asin for asin, _ in scored_neighbors[:k_retrieve]]
        enhanced_retrieved = baseline_k + top_neighbors # baseline (k) + top neighbors (k) = 2k total
        
        return {
            'query': query,
            'answer_asins': answer_asins,
            'baseline_retrieved': baseline_k,
            'neighbor_asins': list(neighbor_asins),
            'enhanced_retrieved': enhanced_retrieved,
            'top_neighbors': top_neighbors,
            'scored_neighbors': scored_neighbors[:10] # Top 10 scored neighbors for analysis
        }
    
    def calculate_metrics(self, answer_asins: List[str], retrieved_asins: List[str], k: int) -> Dict[str, float]:
        """Calculate flexible retrieval metrics based on k value"""
        
        if not answer_asins:
            return {'hit': 0.0, 'recall': 0.0, 'mrr': 0.0}
        
        answer_set = set(answer_asins)
        
        # Hit@k: Did we find at least one relevant document in top k?
        hit_k = 1.0 if any(asin in answer_set for asin in retrieved_asins[:k]) else 0.0
        
        # Recall@k: What percentage of relevant documents did we find in top k?
        recall_k_count = len([asin for asin in retrieved_asins[:k] if asin in answer_set])
        recall_k = recall_k_count / len(answer_asins) if answer_asins else 0.0
        
        # MRR: Mean Reciprocal Rank (position of first relevant document)
        mrr = 0.0
        for i, asin in enumerate(retrieved_asins):
            if asin in answer_set:
                mrr = 1.0 / (i + 1)
                break
        
        return {
            'hit': hit_k,
            'recall': recall_k,
            'mrr': mrr
        }
    
    def run_analysis(self, max_queries: Optional[int] = None) -> List[MultiFeatureRetrievalResult]:
        """Run complete multi-feature enhancement analysis"""
        
        logger.info(" Starting Enhanced Multi-Feature Analysis...")
        logger.info("=" * 60)
        
        results = []
        
        # Process queries
        queries_to_process = self.bm25_df.head(max_queries) if max_queries else self.bm25_df
        
        for idx, row in tqdm(queries_to_process.iterrows(), total=len(queries_to_process), desc="Processing queries"):
            
            query_results = {'enhanced_results': {}, 'metrics_comparison': {}}
            
            # Test different k values
            for k in self.k_values:
                enhanced_result = self.enhance_retrieval_for_query(row, k_retrieve=k)
                
                # Calculate metrics for baseline and enhanced
                baseline_metrics = self.calculate_metrics(
                    enhanced_result['answer_asins'], 
                    enhanced_result['baseline_retrieved'],
                    k # Compare at k level for baseline
                )
                enhanced_metrics = self.calculate_metrics(
                    enhanced_result['answer_asins'], 
                    enhanced_result['enhanced_retrieved'],
                    k * 2 # Compare at 2k level for enhanced (k baseline + k neighbors)
                )
                
                query_results['enhanced_results'][k] = enhanced_result
                query_results['metrics_comparison'][k] = {
                    'baseline': baseline_metrics,
                    'enhanced': enhanced_metrics
                }
            
            # Create result object
            result = MultiFeatureRetrievalResult(
                query_idx=int(row['query_idx']),
                query=row['query'],
                answer_asins=enhanced_result['answer_asins'],
                baseline_retrieved=enhanced_result['baseline_retrieved'],
                enhanced_results=query_results['enhanced_results'],
                metrics_comparison=query_results['metrics_comparison']
            )
            
            results.append(result)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: List[MultiFeatureRetrievalResult]):
        """Save analysis results"""
        
        # Save detailed results
        results_file = self.output_dir / "detailed_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        # Calculate and save summary
        summary = self._calculate_summary(results)
        
        summary_file = self.output_dir / "analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f" Results saved to: {self.output_dir}")
        logger.info(f" Detailed: {results_file.name}")
        logger.info(f" Summary: {summary_file.name}")
    
    def _calculate_summary(self, results: List[MultiFeatureRetrievalResult]) -> Dict[str, Any]:
        """Calculate analysis summary"""
        
        summary = {
            'total_queries': len(results),
            'k_values': self.k_values,
            'metrics_by_k': {}
        }
        
        for k in self.k_values:
            baseline_metrics = {'hit': [], 'recall': [], 'mrr': []}
            enhanced_metrics = {'hit': [], 'recall': [], 'mrr': []}
            
            for result in results:
                if k in result.metrics_comparison:
                    baseline = result.metrics_comparison[k]['baseline']
                    enhanced = result.metrics_comparison[k]['enhanced']
                    
                    for metric in baseline_metrics:
                        baseline_metrics[metric].append(baseline[metric])
                        enhanced_metrics[metric].append(enhanced[metric])
            
            # Calculate averages
            baseline_avg = {metric: np.mean(values) for metric, values in baseline_metrics.items()}
            enhanced_avg = {metric: np.mean(values) for metric, values in enhanced_metrics.items()}
            improvement = {metric: enhanced_avg[metric] - baseline_avg[metric] for metric in baseline_avg}
            
            summary['metrics_by_k'][k] = {
                'baseline': baseline_avg,
                'enhanced': enhanced_avg,
                'improvement': improvement
            }
        
        return summary

def main():
    """Main execution function"""
    logger.info(" Enhanced Multi-Feature Retrieval Analyzer")
    logger.info(" Using all HNSW feature indices for enhanced retrieval")
    logger.info("=" * 60)
    
    analyzer = EnhancedMultiFeatureAnalyzer()
    
    try:
        # Load all data
        analyzer.load_all_data()
        
        # Run analysis on all 81 queries
        results = analyzer.run_analysis() # No limit = all queries
        
        logger.info(" Enhanced Multi-Feature Analysis completed successfully!")
        logger.info(f" Processed {len(results)} queries")
        logger.info(f" Results saved to: {analyzer.output_dir}")
        
    except Exception as e:
        logger.error(f" Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
