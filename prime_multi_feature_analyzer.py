#!/usr/bin/env python3
"""
PRIME Multi-Feature Graph-Enhanced Retrieval Analyzer

This script enhances BM25 baseline retrieval using the neighbor graph for PRIME dataset.
For each query, it:
1. Takes top-k from BM25 baseline
2. Splits into initial set + graph expansion slots
3. Gets neighbors per feature from the graph (including cross-entity connections)
4. Reranks neighbors using hybrid scoring (embedding + BM25)
5. Combines initial + top neighbors and evaluates

Key differences from MAG:
- 10 entity types (vs 2 in MAG)
- 13 feature types (vs 9 in MAG)
- Cross-entity connections (gene↔disease, gene↔drug, disease↔drug, pathway↔all)
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

class PRIMEMultiFeatureAnalyzer:
    """
    Multi-feature graph-enhanced retrieval analyzer for PRIME dataset
    """
    
    def __init__(self,
                 bm25_results_file: str = "/shared/khoja/CogComp/datasets/PRIME/BM25/BM25_stark_prime_test_rewritten.csv",
                 neighbor_graph_file: str = "/shared/khoja/CogComp/output/prime_neighbor_graph/prime_complete_neighbor_graph.json",
                 embeddings_dir: str = "/shared/khoja/CogComp/output/prime_pipeline_cache/embeddings",
                 output_dir: str = "/shared/khoja/CogComp/output/prime_enhanced_analysis",
                 gpu_id: int = 0):
        
        self.bm25_results_file = bm25_results_file
        self.neighbor_graph_file = Path(neighbor_graph_file)
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # GPU setup
        self.device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_model.to(self.device)
        
        # Data containers
        self.bm25_df = None
        self.neighbor_graph = {}  # object_id -> neighbor data
        self.node_embeddings = {}  # object_id -> content_embedding
        self.node_content = {}  # object_id -> text content
        self.node_types = {}  # object_id -> entity type
        
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
        
        logger.info(f"🚀 Initialized PRIME Multi-Feature Analyzer")
        logger.info(f"   📁 Neighbor graph: {self.neighbor_graph_file}")
        logger.info(f"   📁 Embeddings: {self.embeddings_dir}")
        logger.info(f"   💻 Device: {self.device}")
    
    def load_data(self):
        """Load all necessary data"""
        logger.info("📥 Loading data...")
        
        # Load BM25 results
        logger.info(f"   Loading BM25 results from: {self.bm25_results_file}")
        self.bm25_df = pd.read_csv(self.bm25_results_file)
        logger.info(f"   ✅ Loaded {len(self.bm25_df)} queries")
        
        # Pre-load ALL embeddings and content
        logger.info("   💾 Pre-loading ALL node content and embeddings...")
        self._load_all_embeddings()
        
        # Pre-load neighbor graph
        logger.info("   🔗 Loading neighbor graph...")
        self._load_neighbor_graph()
        
        logger.info("✅ Data loading complete")
    
    def _load_all_embeddings(self):
        """Load all content and embeddings from embedding files"""
        logger.info("   💾 Loading ALL content and embeddings...")
        
        chunk_files = sorted(self.embeddings_dir.glob("chunk_*_embeddings.json"))
        total_nodes = 0
        
        for chunk_file in tqdm(chunk_files, desc="     Loading embeddings"):
            with open(chunk_file, 'r') as f:
                chunk_data = json.load(f)
            
            for node in chunk_data:
                object_id = str(node.get('object_id', ''))
                content = node.get('content', '')
                content_embedding = node.get('content_embedding', [])
                node_type = node.get('node_type', 'unknown')
                
                if object_id:
                    if content_embedding:
                        self.node_embeddings[object_id] = np.array(content_embedding, dtype=np.float32)
                    if content:
                        self.node_content[object_id] = content
                    self.node_types[object_id] = node_type
                    total_nodes += 1
        
        logger.info(f"     ✅ Loaded {total_nodes:,} nodes")
        logger.info(f"        📊 Embeddings: {len(self.node_embeddings):,}")
        logger.info(f"        📝 Content: {len(self.node_content):,}")
        logger.info(f"        🏷️  Types: {len(self.node_types):,}")
    
    def _load_neighbor_graph(self):
        """Load the complete neighbor graph"""
        logger.info("   🔗 Loading neighbor graph (this may take a few minutes)...")
        
        with open(self.neighbor_graph_file, 'r') as f:
            self.neighbor_graph = json.load(f)
        
        logger.info(f"     ✅ Loaded neighbor graph for {len(self.neighbor_graph):,} nodes")
    
    def get_graph_neighbors(self, initial_node_ids: List[str]) -> Set[str]:
        """Get neighbors per feature from the graph for initial nodes"""
        all_neighbors = set()
        initial_set = set(initial_node_ids)
        
        for node_id in initial_node_ids:
            node_neighbors = self.neighbor_graph.get(node_id, {})
            
            if 'neighbors' in node_neighbors:
                for feature_name, neighbor_list in node_neighbors['neighbors'].items():
                    # Take top neighbors_per_feature neighbors from this feature
                    for neighbor_entry in neighbor_list[:self.neighbors_per_feature]:
                        neighbor_id = str(neighbor_entry[0])  # (neighbor_id, similarity_score)
                        
                        # Exclude nodes already in initial set
                        if neighbor_id not in initial_set:
                            all_neighbors.add(neighbor_id)
        
        return all_neighbors

    def get_graph_neighbors_with_features(self, initial_node_ids: List[str]) -> Dict[str, Dict]:
        """
        Get neighbors with detailed provenance:
        neighbor_id -> {
            'features': set(feature_names),
            'is_cross_entity': bool,
            'source_types': set(entity_types),
            'target_type': entity_type
        }
        """
        neighbor_to_info: Dict[str, Dict] = {}
        initial_set = set(initial_node_ids)
        
        for node_id in initial_node_ids:
            node_neighbors = self.neighbor_graph.get(node_id, {})
            source_type = self.node_types.get(node_id, 'unknown')
            
            if 'neighbors' in node_neighbors:
                for feature_name, neighbor_list in node_neighbors['neighbors'].items():
                    # Check if this is a cross-entity feature
                    is_cross = feature_name.startswith('cross_')
                    
                    for neighbor_entry in neighbor_list[:self.neighbors_per_feature]:
                        neighbor_id = str(neighbor_entry[0])
                        similarity_score = neighbor_entry[1]
                        
                        if neighbor_id in initial_set:
                            continue
                        
                        target_type = self.node_types.get(neighbor_id, 'unknown')
                        
                        if neighbor_id not in neighbor_to_info:
                            neighbor_to_info[neighbor_id] = {
                                'features': set(),
                                'is_cross_entity': False,
                                'source_types': set(),
                                'target_type': target_type,
                                'max_score': 0.0
                            }
                        
                        neighbor_to_info[neighbor_id]['features'].add(feature_name)
                        neighbor_to_info[neighbor_id]['source_types'].add(source_type)
                        neighbor_to_info[neighbor_id]['max_score'] = max(
                            neighbor_to_info[neighbor_id]['max_score'], 
                            similarity_score
                        )
                        
                        if is_cross:
                            neighbor_to_info[neighbor_id]['is_cross_entity'] = True
        
        return neighbor_to_info
    
    def calculate_hybrid_score(self, query: str, query_embedding: np.ndarray, 
                               candidate_ids: List[str]) -> List[Tuple[str, float]]:
        """Calculate hybrid score (40% embedding + 60% BM25) for candidates"""
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
            
            # Hybrid score (40% embedding + 60% BM25)
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
            
            # Step 3: Get neighbors from graph with detailed provenance
            neighbor_info_map = self.get_graph_neighbors_with_features(initial_nodes)
            neighbor_ids = set(neighbor_info_map.keys())
            
            # Step 4: Rerank neighbors
            neighbor_list = list(neighbor_ids)
            scored_neighbors = self.calculate_hybrid_score(query, query_embedding, neighbor_list)
            scored_neighbors.sort(key=lambda x: x[1], reverse=True)
            
            # Step 5: Take top k_from_graph neighbors
            top_neighbors = [node_id for node_id, _ in scored_neighbors[:k_from_graph]]
            top_neighbors_detailed = [
                {
                    'node_id': node_id,
                    'score': score,
                    'features': sorted(list(neighbor_info_map.get(node_id, {}).get('features', set()))),
                    'is_cross_entity': neighbor_info_map.get(node_id, {}).get('is_cross_entity', False),
                    'entity_type': neighbor_info_map.get(node_id, {}).get('target_type', 'unknown')
                }
                for node_id, score in scored_neighbors[:k_from_graph]
            ]
            
            # Step 6: Combine initial + top neighbors
            enhanced_retrieved = initial_nodes + top_neighbors
            
            # Step 7: Calculate metrics
            baseline_metrics = self.calculate_metrics(gold_docs, baseline_k)
            enhanced_metrics = self.calculate_metrics(gold_docs, enhanced_retrieved)

            # Additional analysis: cross-entity statistics
            cross_entity_neighbors = sum(1 for n in top_neighbors_detailed if n['is_cross_entity'])
            
            # Entity type distribution in neighbors
            entity_type_dist = Counter([n['entity_type'] for n in top_neighbors_detailed])
            
            # Gold document analysis
            gold_in_initial = set(initial_nodes) & set(gold_docs)
            gold_in_neighbors = set(top_neighbors) & set(gold_docs)
            
            results[k_total] = {
                'baseline_retrieved': baseline_k,
                'initial_nodes': initial_nodes,
                'neighbor_count': len(neighbor_ids),
                'top_neighbors': top_neighbors,
                'top_neighbors_detailed': top_neighbors_detailed,
                'enhanced_retrieved': enhanced_retrieved,
                'baseline_metrics': baseline_metrics,
                'enhanced_metrics': enhanced_metrics,
                'cross_entity_analysis': {
                    'cross_entity_count': cross_entity_neighbors,
                    'cross_entity_percentage': cross_entity_neighbors / len(top_neighbors_detailed) * 100 if top_neighbors_detailed else 0,
                    'entity_type_distribution': dict(entity_type_dist)
                },
                'gold_analysis': {
                    'gold_in_initial': sorted(list(gold_in_initial)),
                    'gold_in_neighbors': sorted(list(gold_in_neighbors)),
                    'gold_types': {gid: self.node_types.get(gid, 'unknown') for gid in gold_docs}
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
        logger.info("🚀 Starting PRIME Multi-Feature Graph-Enhanced Analysis")
        logger.info("=" * 70)
        
        queries_to_process = self.bm25_df.head(max_queries) if max_queries else self.bm25_df
        
        all_results = []
        
        for idx, row in tqdm(queries_to_process.iterrows(), total=len(queries_to_process), desc="Processing queries"):
            query = row['query']
            top_100 = self.parse_id_list(row['top_100'])
            gold_docs = self.parse_id_list(row['gold_docs'])
            
            query_results = self.process_query(query, top_100, gold_docs)
            
            all_results.append({
                'query_idx': idx,
                'query': query,
                'gold_docs': gold_docs,
                'results_by_k': query_results
            })
        
        # Calculate summary statistics
        summary = self._calculate_summary(all_results)
        
        # Save results
        self._save_results(all_results, summary)
        
        # Display summary
        self._display_summary(summary)
        
        return all_results, summary
    
    def _calculate_summary(self, all_results: List[Dict]) -> Dict:
        """Calculate summary statistics across all queries"""
        summary = {'k_values': {}}
        
        for k_total in self.k_configs.keys():
            baseline_metrics = {'hit': [], 'recall': [], 'mrr': []}
            enhanced_metrics = {'hit': [], 'recall': [], 'mrr': []}
            cross_entity_stats = []
            
            for result in all_results:
                k_result = result['results_by_k'][k_total]
                
                for metric in ['hit', 'recall', 'mrr']:
                    baseline_metrics[metric].append(k_result['baseline_metrics'][metric])
                    enhanced_metrics[metric].append(k_result['enhanced_metrics'][metric])
                
                cross_entity_stats.append(k_result['cross_entity_analysis']['cross_entity_percentage'])
            
            baseline_avg = {m: np.mean(v) for m, v in baseline_metrics.items()}
            enhanced_avg = {m: np.mean(v) for m, v in enhanced_metrics.items()}
            improvement_avg = {m: enhanced_avg[m] - baseline_avg[m] for m in baseline_avg}
            
            summary['k_values'][k_total] = {
                'baseline': baseline_avg,
                'enhanced': enhanced_avg,
                'improvement': improvement_avg,
                'avg_cross_entity_percentage': np.mean(cross_entity_stats)
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
        
        logger.info(f"📁 Results saved to: {self.output_dir}")
        logger.info(f"   📊 Detailed: {results_file.name}")
        logger.info(f"   📋 Summary: {summary_file.name}")
    
    def _display_summary(self, summary: Dict):
        """Display summary statistics"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 ANALYSIS SUMMARY")
        logger.info("=" * 70)
        
        for k_total in sorted(summary['k_values'].keys()):
            k_data = summary['k_values'][k_total]
            k_initial, k_from_graph = self.k_configs[k_total][1], self.k_configs[k_total][2]
            
            logger.info(f"\n🔸 k={k_total} (Initial: {k_initial}, Graph: {k_from_graph})")
            logger.info(f"   Baseline  → Hit: {k_data['baseline']['hit']:.4f}, Recall: {k_data['baseline']['recall']:.4f}, MRR: {k_data['baseline']['mrr']:.4f}")
            logger.info(f"   Enhanced  → Hit: {k_data['enhanced']['hit']:.4f}, Recall: {k_data['enhanced']['recall']:.4f}, MRR: {k_data['enhanced']['mrr']:.4f}")
            logger.info(f"   Improvement → Hit: {k_data['improvement']['hit']:+.4f}, Recall: {k_data['improvement']['recall']:+.4f}, MRR: {k_data['improvement']['mrr']:+.4f}")
            logger.info(f"   Cross-Entity: {k_data['avg_cross_entity_percentage']:.1f}% of neighbors")

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PRIME Multi-Feature Graph-Enhanced Retrieval')
    parser.add_argument('--max-queries', type=int, default=None, help='Max queries to process (None = all)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    
    args = parser.parse_args()
    
    analyzer = PRIMEMultiFeatureAnalyzer(gpu_id=args.gpu)
    
    try:
        analyzer.load_data()
        analyzer.run_analysis(max_queries=args.max_queries)
        
        logger.info("✅ Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

