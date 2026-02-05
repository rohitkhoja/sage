#!/usr/bin/env python3
"""
MAG Multi-Feature Graph-Enhanced Retrieval Analyzer

This script enhances BM25 baseline retrieval using the neighbor graph we built.
For each query, it:
1. Takes top-k from BM25 baseline
2. Splits into initial set + graph expansion slots
3. Gets 200 neighbors per feature from the graph
4. Reranks neighbors using hybrid scoring (50% embedding + 50% BM25)
5. Combines initial + top neighbors and evaluates
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, Counter
from tqdm import tqdm
from loguru import logger
import time
import math
from sentence_transformers import SentenceTransformer
import torch

class MAGMultiFeatureAnalyzer:
    """
    Multi-feature graph-enhanced retrieval analyzer for MAG dataset
    """
    
    def __init__(self,
                 bm25_results_file: str = "/shared/khoja/CogComp/output/BM25_stark_mag_test_rewritten.csv",
                 neighbor_graph_dir: str = "/shared/khoja/CogComp/output/mag_neighbor_graph_trimmed",
                 embeddings_dir: str = "/shared/khoja/CogComp/output/mag_embeddings_trimmed",
                 output_dir: str = "/shared/khoja/CogComp/output/mag_test-query_enhanced_analysis",
                 gpu_id: int = 0,
                 load_all_embeddings: bool = True):
        
        self.bm25_results_file = bm25_results_file
        self.neighbor_graph_dir = Path(neighbor_graph_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.load_all_embeddings = load_all_embeddings
        
        # GPU setup
        self.device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_model.to(self.device)
        
        # Data containers
        self.bm25_df = None
        self.neighbor_graph = {} # object_id -> neighbor data (lazy loaded)
        self.node_embeddings = {} # object_id -> content_embedding
        self.node_content = {} # object_id -> text content
        self.chunk_to_nodes = defaultdict(list) # chunk_file -> list of object_ids
        
        # K value configurations: (k_total, k_initial, k_from_graph)
        self.k_configs = {
            5: (5, 2, 3),
            10: (10, 5, 5),
            20: (20, 10, 10),
            40: (40, 20, 20),
            50: (50, 25, 25),
            80: (80, 40, 40),
            100: (100, 50, 50)
        }
        
        # Neighbors per feature to retrieve
        self.neighbors_per_feature = 200
        
        logger.info(f" Initialized MAG Multi-Feature Analyzer")
        logger.info(f" Neighbor graph: {self.neighbor_graph_dir}")
        logger.info(f" Embeddings: {self.embeddings_dir}")
        logger.info(f" Load all embeddings: {self.load_all_embeddings}")
    
    def load_data(self):
        """Load all necessary data"""
        logger.info(" Loading data...")
        
        # Load BM25 results
        logger.info(f" Loading BM25 results from: {self.bm25_results_file}")
        self.bm25_df = pd.read_csv(self.bm25_results_file)
        logger.info(f" Loaded {len(self.bm25_df)} queries")
        
        # Pre-load ALL embeddings and content
        logger.info(" Pre-loading ALL node content and embeddings...")
        self._load_all_embeddings()
        
        # Pre-load ALL neighbor graph data (first 200 neighbors per feature)
        logger.info(" Pre-loading ALL neighbor graph data...")
        self._load_all_neighbor_graphs()
        
        logger.info(" Data loading complete")
        
    
    def _load_all_embeddings(self):
        """Load all content and embeddings from trimmed files (much faster!)"""
        logger.info(" Loading ALL content and embeddings from trimmed files...")
        
        chunk_files = sorted(self.embeddings_dir.glob("chunk_*.json"))
        total_nodes = 0
        
        for chunk_file in tqdm(chunk_files, desc=" Loading embeddings"):
            with open(chunk_file, 'r') as f:
                chunk_data = json.load(f)
            
            for node in chunk_data:
                object_id = str(node.get('object_id', ''))
                content = node.get('content', '')
                content_embedding = node.get('content_embedding', [])
                
                if object_id:
                    if content_embedding:
                        self.node_embeddings[object_id] = np.array(content_embedding, dtype=np.float32)
                    if content:
                        self.node_content[object_id] = content
                    total_nodes += 1
        
        logger.info(f" Loaded {total_nodes:,} nodes")
        logger.info(f" Embeddings: {len(self.node_embeddings):,}")
        logger.info(f" Content: {len(self.node_content):,}")
    
    def _load_all_neighbor_graphs(self):
        """Pre-load ALL neighbor graph data from trimmed files"""
        neighbor_files = sorted(self.neighbor_graph_dir.glob("neighbors_chunk_*.json"))
        
        total_nodes = 0
        
        for neighbor_file in tqdm(neighbor_files, desc=" Loading neighbor graphs"):
            with open(neighbor_file, 'r') as f:
                chunk_neighbors = json.load(f)
            
            # Files are already trimmed to 200 neighbors per feature
            for object_id, node_data in chunk_neighbors.items():
                self.neighbor_graph[object_id] = node_data
                total_nodes += 1
        
        logger.info(f" Loaded neighbor graphs for {total_nodes:,} nodes")
    
    def _load_embeddings_for_nodes(self, node_ids: Set[str]):
        """Load content and embeddings only for specific nodes (memory efficient)"""
        # This method is not used when load_all_embeddings=True
        # Kept for compatibility but won't be called
        pass
    
    def _load_neighbor_graph_for_node(self, object_id: str) -> Dict:
        """Get neighbor data for a specific node (already pre-loaded)"""
        return self.neighbor_graph.get(object_id, {})
    
    def get_graph_neighbors(self, initial_node_ids: List[str]) -> Set[str]:
        """Get 200 neighbors per feature from the graph for initial nodes"""
        all_neighbors = set()
        initial_set = set(initial_node_ids)
        
        for node_id in initial_node_ids:
            node_neighbors = self._load_neighbor_graph_for_node(node_id)
            
            if 'neighbors' in node_neighbors:
                for feature_name, neighbor_list in node_neighbors['neighbors'].items():
                    # Take top 200 neighbors from this feature
                    for neighbor_entry in neighbor_list[:self.neighbors_per_feature]:
                        neighbor_id = str(neighbor_entry[0]) # (neighbor_id, similarity_score)
                        
                        # Exclude nodes already in initial set
                        if neighbor_id not in initial_set:
                            all_neighbors.add(neighbor_id)
        
        return all_neighbors

    def get_graph_neighbors_with_features(self, initial_node_ids: List[str]) -> Dict[str, Set[str]]:
        """Get neighbors with provenance features: neighbor_id -> set(features)"""
        neighbor_to_features: Dict[str, Set[str]] = {}
        initial_set = set(initial_node_ids)
        
        for node_id in initial_node_ids:
            node_neighbors = self._load_neighbor_graph_for_node(node_id)
            
            if 'neighbors' in node_neighbors:
                for feature_name, neighbor_list in node_neighbors['neighbors'].items():
                    for neighbor_entry in neighbor_list[:self.neighbors_per_feature]:
                        neighbor_id = str(neighbor_entry[0])
                        if neighbor_id in initial_set:
                            continue
                        if neighbor_id not in neighbor_to_features:
                            neighbor_to_features[neighbor_id] = set()
                        neighbor_to_features[neighbor_id].add(feature_name)
        
        return neighbor_to_features
    
    def calculate_hybrid_score(self, query: str, query_embedding: np.ndarray, 
                               candidate_ids: List[str]) -> List[Tuple[str, float]]:
        """Calculate hybrid score (50% embedding + 50% BM25) for candidates"""
        scores = []
        
        # Preprocess query for BM25
        query_tokens = self._preprocess_for_bm25(query)
        query_term_freq = Counter(query_tokens)
        
        for node_id in candidate_ids:
            # Get node embedding and content
            node_embedding = self.node_embeddings.get(node_id)
            node_content = self.node_content.get(node_id, '')
            
            if node_embedding is None:
                scores.append((node_id, 0.0))
                continue
            
            # 1. Embedding similarity
            embedding_sim = np.dot(query_embedding, node_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(node_embedding) + 1e-8
            )
            embedding_score = max(0.0, float(embedding_sim))
            
            # 2. BM25 similarity
            bm25_score = self._calculate_bm25_score(query_term_freq, node_content)
            
            # Hybrid score (50-50 weighted)
            hybrid_score = 0.4 * embedding_score + 0.6 * bm25_score
            scores.append((node_id, hybrid_score))
        
        return scores
    
    def _preprocess_for_bm25(self, text: str) -> List[str]:
        """Simple BM25 preprocessing"""
        import re
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        # Simple stopword removal
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'being'}
        tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
        return tokens
    
    def _calculate_bm25_score(self, query_term_freq: Counter, doc_text: str) -> float:
        """Simple BM25 scoring"""
        if not doc_text:
            return 0.0
        
        doc_tokens = self._preprocess_for_bm25(doc_text)
        doc_term_freq = Counter(doc_tokens)
        doc_length = len(doc_tokens)
        
        if doc_length == 0:
            return 0.0
        
        score = 0.0
        for term, qf in query_term_freq.items():
            if term in doc_term_freq:
                tf = doc_term_freq[term]
                # Simplified BM25 formula
                term_score = tf / (tf + 1.0)
                score += term_score
        
        # Normalize by document length
        score = score / math.sqrt(doc_length) if doc_length > 0 else 0.0
        
        return score
    
    def parse_id_list(self, id_str: str) -> List[str]:
        """Parse ID list from CSV string format"""
        if pd.isna(id_str) or id_str == '':
            return []
        
        import ast
        try:
            id_list = ast.literal_eval(str(id_str))
            return [str(id_val) for id_val in id_list]
        except:
            return []
    
    def calculate_metrics(self, gold_ids: List[str], retrieved_ids: List[str]) -> Dict[str, float]:
        """Calculate Hit, Recall, and MRR"""
        if not gold_ids:
            return {'hit': 0.0, 'recall': 0.0, 'mrr': 0.0}
        
        gold_set = set(gold_ids)
        
        # Hit: Did we find at least one relevant document?
        hit = 1.0 if any(rid in gold_set for rid in retrieved_ids) else 0.0
        
        # Recall: What fraction of relevant documents did we find?
        recall_count = sum(1 for rid in retrieved_ids if rid in gold_set)
        recall = recall_count / len(gold_ids)
        
        # MRR: Position of first relevant document
        mrr = 0.0
        for i, rid in enumerate(retrieved_ids):
            if rid in gold_set:
                mrr = 1.0 / (i + 1)
                break
        
        return {'hit': hit, 'recall': recall, 'mrr': mrr}
    
    def process_query(self, query: str, top_100: List[str], gold_docs: List[str]) -> Dict:
        """Process a single query for all k values"""
        
        results = {}
        
        # Generate query embedding once
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        query_embedding = query_embedding.astype(np.float32)
        
        for k_total, (_, k_initial, k_from_graph) in self.k_configs.items():
            # Step 1: Get top-k from baseline
            baseline_k = top_100[:k_total]
            
            # Step 2: Split into initial and graph portions
            initial_nodes = baseline_k[:k_initial]
            
            # Step 3: Get neighbors from graph (with provenance features)
            neighbor_feature_map = self.get_graph_neighbors_with_features(initial_nodes)
            neighbor_ids = set(neighbor_feature_map.keys())
            
            # Step 4: Rerank neighbors (embeddings already pre-loaded)
            neighbor_list = list(neighbor_ids)
            scored_neighbors = self.calculate_hybrid_score(query, query_embedding, neighbor_list)
            scored_neighbors.sort(key=lambda x: x[1], reverse=True)
            
            # Step 6: Take top k_from_graph neighbors
            top_neighbors = [node_id for node_id, _ in scored_neighbors[:k_from_graph]]
            top_neighbors_detailed = [
                {
                    'node_id': node_id,
                    'score': score,
                    'features': sorted(list(neighbor_feature_map.get(node_id, set())))
                }
                for node_id, score in scored_neighbors[:k_from_graph]
            ]
            
            # Step 7: Combine initial + top neighbors
            enhanced_retrieved = initial_nodes + top_neighbors
            
            # Step 8: Calculate metrics
            baseline_metrics = self.calculate_metrics(gold_docs, baseline_k)
            enhanced_metrics = self.calculate_metrics(gold_docs, enhanced_retrieved)

            # Additional analysis: gold ranks in full neighbor list and feature sources
            ranked_ids = [nid for nid, _ in scored_neighbors]
            gold_ranks: Dict[str, int] = {}
            gold_feature_sources: Dict[str, List[str]] = {}
            for gid in gold_docs:
                rank = (ranked_ids.index(gid) + 1) if gid in ranked_ids else 0
                gold_ranks[gid] = rank
                if gid in neighbor_feature_map:
                    gold_feature_sources[gid] = sorted(list(neighbor_feature_map[gid]))
                else:
                    gold_feature_sources[gid] = []

            # Per-feature gold presence before/after prune
            golds_in_neighbors_per_feature: Dict[str, Set[str]] = {}
            for gid, features in gold_feature_sources.items():
                for feat in features:
                    golds_in_neighbors_per_feature.setdefault(feat, set()).add(gid)

            golds_in_top_set = set(top_neighbors) & set(gold_docs)
            golds_in_top_per_feature: Dict[str, Set[str]] = {}
            for gid in golds_in_top_set:
                for feat in neighbor_feature_map.get(gid, set()):
                    golds_in_top_per_feature.setdefault(feat, set()).add(gid)
            
            # Gold multiplicity across features (how many features produced each gold)
            gold_feature_multiplicity: Dict[str, int] = {
                gid: len(neighbor_feature_map.get(gid, set())) for gid in gold_docs if gid in neighbor_feature_map
            }
            
            results[k_total] = {
                'baseline_retrieved': baseline_k,
                'initial_nodes': initial_nodes,
                'neighbor_count': len(neighbor_ids),
                'top_neighbors': top_neighbors,
                'top_neighbors_detailed': top_neighbors_detailed,
                'enhanced_retrieved': enhanced_retrieved,
                'baseline_metrics': baseline_metrics,
                'enhanced_metrics': enhanced_metrics,
                'gold_ranks_in_neighbors': gold_ranks,
                'gold_feature_sources': gold_feature_sources,
                'feature_analysis': {
                    'golds_in_neighbors_per_feature': {k: sorted(list(v)) for k, v in golds_in_neighbors_per_feature.items()},
                    'golds_in_top_per_feature': {k: sorted(list(v)) for k, v in golds_in_top_per_feature.items()},
                    'gold_feature_multiplicity': gold_feature_multiplicity,
                },
                'improvement': {
                    'hit': enhanced_metrics['hit'] - baseline_metrics['hit'],
                    'recall': enhanced_metrics['recall'] - baseline_metrics['recall'],
                    'mrr': enhanced_metrics['mrr'] - baseline_metrics['mrr']
                }
            }
        
        return results
    
    def run_analysis(self, max_queries: Optional[int] = None):
        """Run complete analysis"""
        logger.info(" Starting MAG Multi-Feature Graph-Enhanced Analysis")
        logger.info("=" * 70)
        
        queries_to_process = self.bm25_df.head(max_queries) if max_queries else self.bm25_df
        
        all_results = []
        # Aggregate feature contribution summary by k across all queries
        feature_contrib_by_k: Dict[int, Dict[str, Dict[str, int]]] = {}
        
        for idx, row in tqdm(queries_to_process.iterrows(), total=len(queries_to_process), desc="Processing queries"):
            query = row['query']
            top_100 = self.parse_id_list(row['top_100'])
            gold_docs = self.parse_id_list(row['gold_docs'])
            
            query_results = self.process_query(query, top_100, gold_docs)
            
            # Aggregate per-feature stats
            for k_total, k_result in query_results.items():
                fa = k_result.get('feature_analysis', {})
                in_neighbors = fa.get('golds_in_neighbors_per_feature', {})
                in_top = fa.get('golds_in_top_per_feature', {})
                multiplicity = fa.get('gold_feature_multiplicity', {})

                if k_total not in feature_contrib_by_k:
                    feature_contrib_by_k[k_total] = {}
                bucket = feature_contrib_by_k[k_total]

                # Count totals
                for feat, gids in in_neighbors.items():
                    bucket.setdefault(feat, {'gold_in_neighbors': 0, 'gold_in_top': 0, 'unique_gold': 0})
                    bucket[feat]['gold_in_neighbors'] += len(gids)
                for feat, gids in in_top.items():
                    bucket.setdefault(feat, {'gold_in_neighbors': 0, 'gold_in_top': 0, 'unique_gold': 0})
                    bucket[feat]['gold_in_top'] += len(gids)

                # Unique golds per feature (appeared in exactly one feature for this query)
                # Build per-gold feature list from in_neighbors
                per_gold_features: Dict[str, Set[str]] = {}
                for feat, gids in in_neighbors.items():
                    for gid in gids:
                        per_gold_features.setdefault(gid, set()).add(feat)
                for gid, feats in per_gold_features.items():
                    if len(feats) == 1:
                        only_feat = next(iter(feats))
                        bucket.setdefault(only_feat, {'gold_in_neighbors': 0, 'gold_in_top': 0, 'unique_gold': 0})
                        bucket[only_feat]['unique_gold'] += 1

            all_results.append({
                'query_idx': idx,
                'query': query,
                'gold_docs': gold_docs,
                'results_by_k': query_results
            })
        
        # Calculate summary statistics
        summary = self._calculate_summary(all_results)
        
        # Save results and feature contribution summary
        self._save_results(all_results, summary)
        feature_summary_file = self.output_dir / 'feature_contribution_summary.json'
        with open(feature_summary_file, 'w') as f:
            json.dump({str(k): v for k, v in feature_contrib_by_k.items()}, f, indent=2)
        logger.info(f" Feature contribution summary: {feature_summary_file.name}")
        
        # Display summary
        self._display_summary(summary)
        
        return all_results, summary
    
    def _calculate_summary(self, all_results: List[Dict]) -> Dict:
        """Calculate summary statistics across all queries"""
        summary = {'k_values': {}}
        
        for k_total in self.k_configs.keys():
            baseline_metrics = {'hit': [], 'recall': [], 'mrr': []}
            enhanced_metrics = {'hit': [], 'recall': [], 'mrr': []}
            
            for result in all_results:
                k_result = result['results_by_k'][k_total]
                
                for metric in ['hit', 'recall', 'mrr']:
                    baseline_metrics[metric].append(k_result['baseline_metrics'][metric])
                    enhanced_metrics[metric].append(k_result['enhanced_metrics'][metric])
            
            baseline_avg = {m: np.mean(v) for m, v in baseline_metrics.items()}
            enhanced_avg = {m: np.mean(v) for m, v in enhanced_metrics.items()}
            improvement_avg = {m: enhanced_avg[m] - baseline_avg[m] for m in baseline_avg}
            
            summary['k_values'][k_total] = {
                'baseline': baseline_avg,
                'enhanced': enhanced_avg,
                'improvement': improvement_avg
            }
        
        return summary
    
    def _save_results(self, all_results: List[Dict], summary: Dict):
        """Save results to files"""
        
        # Save detailed results
        results_file = self.output_dir / 'detailed_results.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Save summary
        summary_file = self.output_dir / 'analysis_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f" Results saved to: {self.output_dir}")
        logger.info(f" Detailed: {results_file.name}")
        logger.info(f" Summary: {summary_file.name}")
    
    def _display_summary(self, summary: Dict):
        """Display summary statistics"""
        logger.info("\n" + "=" * 70)
        logger.info(" ANALYSIS SUMMARY")
        logger.info("=" * 70)
        
        for k_total in sorted(summary['k_values'].keys()):
            k_data = summary['k_values'][k_total]
            k_initial, k_from_graph = self.k_configs[k_total][1], self.k_configs[k_total][2]
            
            logger.info(f"\n k={k_total} (Initial: {k_initial}, Graph: {k_from_graph})")
            logger.info(f" Baseline → Hit: {k_data['baseline']['hit']:.4f}, Recall: {k_data['baseline']['recall']:.4f}, MRR: {k_data['baseline']['mrr']:.4f}")
            logger.info(f" Enhanced → Hit: {k_data['enhanced']['hit']:.4f}, Recall: {k_data['enhanced']['recall']:.4f}, MRR: {k_data['enhanced']['mrr']:.4f}")
            logger.info(f" Improvement → Hit: {k_data['improvement']['hit']:+.4f}, Recall: {k_data['improvement']['recall']:+.4f}, MRR: {k_data['improvement']['mrr']:+.4f}")

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MAG Multi-Feature Graph-Enhanced Retrieval')
    parser.add_argument('--max-queries', type=int, default=None, help='Max queries to process (None = all)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--load-all', action='store_true', help='Load all embeddings upfront (faster but more RAM)')
    
    args = parser.parse_args()
    
    analyzer = MAGMultiFeatureAnalyzer(
        gpu_id=args.gpu,
        load_all_embeddings=args.load_all
    )
    
    try:
        analyzer.load_data()
        analyzer.run_analysis(max_queries=args.max_queries)
        
        logger.info(" Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f" Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

