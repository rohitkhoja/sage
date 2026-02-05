#!/usr/bin/env python3
"""
Feature-wise Neighbor Analysis Tool
Analyzes which features contain gold documents and how neighbors are distributed across features
"""

import os
import sys
import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict, Counter
from tqdm import tqdm
import ast
from loguru import logger

# Import the fast analyzer to reuse its functionality
from enhanced_multi_feature_analyzer_fast import FastEnhancedMultiFeatureAnalyzer

class FeatureWiseNeighborAnalyzer:
    """
    Analyzes neighbor distribution across features and gold document locations
    """
    
    def __init__(self,
                 fast_analyzer: Optional[FastEnhancedMultiFeatureAnalyzer] = None,
                 output_dir: str = "/shared/khoja/CogComp/output/feature_analysis"):
        
        # Use existing analyzer or create new one
        if fast_analyzer is None:
            self.analyzer = FastEnhancedMultiFeatureAnalyzer()
            self.analyzer.load_all_data()
            logger.info(" Created new FastEnhancedMultiFeatureAnalyzer")
        else:
            self.analyzer = fast_analyzer
            logger.info(" Using existing FastEnhancedMultiFeatureAnalyzer")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature names including BM25
        self.all_features = self.analyzer.embedding_features + ['bm25_neighbors']
        
        logger.info(f" Initialized Feature-wise Neighbor Analyzer")
        logger.info(f" Output dir: {self.output_dir}")
        logger.info(f" Features to analyze: {self.all_features}")
    
    def analyze_feature_wise_neighbors(self, retrieved_asins: List[str], query_idx: int) -> Dict[str, Any]:
        """
        Analyze neighbors retrieved from each feature for given baseline ASINs
        
        Returns:
        - neighbors_by_feature: dict mapping feature -> list of neighbor ASINs
        - neighbor_counts: dict mapping feature -> count of neighbors
        - unique_neighbors: set of all unique neighbor ASINs
        - overlap_analysis: overlap between features
        """
        
        neighbors_by_feature = {}
        neighbor_counts = {}
        
        logger.info(f" Analyzing neighbors for {len(retrieved_asins)} baseline ASINs...")
        
        # Get neighbors from each embedding feature
        for feature_name in self.analyzer.embedding_features:
            feature_neighbors = set()
            neighbors_dict = self.analyzer.precomputed_neighbors.get(feature_name, {})
            
            # For each baseline ASIN, get its neighbors from this feature
            for asin in retrieved_asins:
                if asin not in self.analyzer.asin_to_chunks:
                    continue
                
                chunk_infos = self.analyzer.asin_to_chunks[asin]
                
                for chunk_info in chunk_infos:
                    precompute_key = chunk_info['precompute_key']
                    
                    if precompute_key in neighbors_dict:
                        neighbors = neighbors_dict[precompute_key]
                        
                        for neighbor in neighbors:
                            neighbor_asin = neighbor['asin']
                            real_neighbor_asin = self.analyzer._ensure_real_asin(neighbor_asin)
                            if real_neighbor_asin and real_neighbor_asin not in retrieved_asins:
                                feature_neighbors.add(real_neighbor_asin)
            
            neighbors_by_feature[feature_name] = list(feature_neighbors)
            neighbor_counts[feature_name] = len(feature_neighbors)
        
        # Get neighbors from BM25 feature
        bm25_neighbors = set()
        for asin in retrieved_asins:
            if asin not in self.analyzer.asin_to_chunks:
                continue
            
            chunk_infos = self.analyzer.asin_to_chunks[asin]
            
            for chunk_info in chunk_infos:
                precompute_key = chunk_info['precompute_key']
                
                if precompute_key in self.analyzer.bm25_neighbors:
                    bm25_neighbor_list = self.analyzer.bm25_neighbors[precompute_key]
                    
                    for neighbor in bm25_neighbor_list:
                        neighbor_asin = neighbor['asin']
                        real_neighbor_asin = self.analyzer._ensure_real_asin(neighbor_asin)
                        if real_neighbor_asin and real_neighbor_asin not in retrieved_asins:
                            bm25_neighbors.add(real_neighbor_asin)
        
        neighbors_by_feature['bm25_neighbors'] = list(bm25_neighbors)
        neighbor_counts['bm25_neighbors'] = len(bm25_neighbors)
        
        # Calculate unique neighbors across all features
        all_neighbors = set()
        for feature_neighbors in neighbors_by_feature.values():
            all_neighbors.update(feature_neighbors)
        
        # Calculate feature overlap
        overlap_analysis = self._calculate_feature_overlap(neighbors_by_feature)
        
        return {
            'query_idx': query_idx,
            'baseline_count': len(retrieved_asins),
            'neighbors_by_feature': neighbors_by_feature,
            'neighbor_counts': neighbor_counts,
            'unique_neighbors': list(all_neighbors),
            'total_unique_neighbors': len(all_neighbors),
            'overlap_analysis': overlap_analysis
        }
    
    def analyze_gold_distribution(self, answer_asins: List[str], neighbor_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze which features contain gold documents
        
        Returns:
        - gold_in_features: dict mapping feature -> list of gold ASINs found
        - gold_counts_by_feature: dict mapping feature -> count of gold docs
        - gold_not_found: list of gold ASINs not found in any feature
        - feature_recall: dict mapping feature -> recall (gold_found / total_gold)
        """
        
        answer_set = set(answer_asins)
        gold_in_features = {}
        gold_counts_by_feature = {}
        
        # Check each feature for gold documents
        for feature_name, feature_neighbors in neighbor_analysis['neighbors_by_feature'].items():
            gold_in_feature = [asin for asin in feature_neighbors if asin in answer_set]
            gold_in_features[feature_name] = gold_in_feature
            gold_counts_by_feature[feature_name] = len(gold_in_feature)
        
        # Find gold documents not found in any feature
        all_neighbors = set(neighbor_analysis['unique_neighbors'])
        gold_not_found = [asin for asin in answer_asins if asin not in all_neighbors]
        
        # Calculate recall per feature
        total_gold = len(answer_asins) if answer_asins else 1
        feature_recall = {feature: count / total_gold for feature, count in gold_counts_by_feature.items()}
        
        # Find which features each gold document appears in
        gold_feature_mapping = {}
        for gold_asin in answer_asins:
            features_containing_gold = []
            for feature_name, feature_neighbors in neighbor_analysis['neighbors_by_feature'].items():
                if gold_asin in feature_neighbors:
                    features_containing_gold.append(feature_name)
            gold_feature_mapping[gold_asin] = features_containing_gold
        
        return {
            'total_gold': len(answer_asins),
            'gold_in_features': gold_in_features,
            'gold_counts_by_feature': gold_counts_by_feature,
            'gold_not_found': gold_not_found,
            'gold_not_found_count': len(gold_not_found),
            'feature_recall': feature_recall,
            'gold_feature_mapping': gold_feature_mapping
        }
    
    def analyze_reranking_performance(self, query: str, neighbor_analysis: Dict[str, Any], 
                                    answer_asins: List[str], k_values: List[int] = [1, 5, 10, 20, 50, 100]) -> Dict[str, Any]:
        """
        Analyze how many gold documents survive reranking at different k values
        """
        
        # Get all unique neighbors
        all_neighbors = neighbor_analysis['unique_neighbors']
        
        if not all_neighbors:
            return {'k_values': k_values, 'gold_after_reranking': {k: 0 for k in k_values}}
        
        # Score all neighbors using hybrid scoring
        scored_neighbors = self.analyzer.calculate_hybrid_scores(query, all_neighbors)
        scored_neighbors.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate how many gold documents are in top-k after reranking
        answer_set = set(answer_asins)
        gold_after_reranking = {}
        
        for k in k_values:
            top_k_neighbors = [asin for asin, _ in scored_neighbors[:k]]
            gold_in_top_k = len([asin for asin in top_k_neighbors if asin in answer_set])
            gold_after_reranking[k] = gold_in_top_k
        
        return {
            'k_values': k_values,
            'total_neighbors_scored': len(scored_neighbors),
            'gold_after_reranking': gold_after_reranking,
            'top_scored_neighbors': scored_neighbors[:10] # Top 10 for analysis
        }
    
    def _calculate_feature_overlap(self, neighbors_by_feature: Dict[str, List[str]]) -> Dict[str, Any]:
        """Calculate overlap between different features"""
        
        feature_sets = {feature: set(neighbors) for feature, neighbors in neighbors_by_feature.items()}
        overlap_matrix = {}
        
        for feature1 in self.all_features:
            overlap_matrix[feature1] = {}
            for feature2 in self.all_features:
                if feature1 == feature2:
                    overlap_matrix[feature1][feature2] = len(feature_sets.get(feature1, set()))
                else:
                    set1 = feature_sets.get(feature1, set())
                    set2 = feature_sets.get(feature2, set())
                    overlap = len(set1 & set2)
                    overlap_matrix[feature1][feature2] = overlap
        
        return overlap_matrix
    
    def run_complete_analysis(self, max_queries: Optional[int] = None, k_baseline: int = 10) -> Dict[str, Any]:
        """
        Run complete feature-wise analysis on all queries
        """
        
        logger.info(" Starting Complete Feature-wise Neighbor Analysis...")
        logger.info("=" * 60)
        logger.info(f" Using k_baseline = {k_baseline}")
        
        # Get queries to process
        queries_to_process = self.analyzer.bm25_df.head(max_queries) if max_queries else self.analyzer.bm25_df
        
        all_results = []
        summary_stats = {
            'neighbor_counts_by_feature': defaultdict(list),
            'gold_counts_by_feature': defaultdict(list),
            'feature_recall': defaultdict(list),
            'gold_after_reranking': defaultdict(list),
            'overlap_stats': defaultdict(list)
        }
        
        for idx, row in tqdm(queries_to_process.iterrows(), total=len(queries_to_process), desc="Analyzing queries"):
            
            query = row['query']
            answer_asins = self.analyzer.extract_asin_list(row['answer_ids'], convert_to_asin=True)
            baseline_retrieved = self.analyzer.extract_asin_list(row['retrieved_docs'], convert_to_asin=True)
            
            # Take first k_baseline from baseline
            baseline_k = baseline_retrieved[:k_baseline]
            
            # Analyze neighbors by feature
            neighbor_analysis = self.analyze_feature_wise_neighbors(baseline_k, int(row['query_idx']))
            
            # Analyze gold distribution
            gold_analysis = self.analyze_gold_distribution(answer_asins, neighbor_analysis)
            
            # Analyze reranking performance
            reranking_analysis = self.analyze_reranking_performance(query, neighbor_analysis, answer_asins)
            
            # Combine results
            query_result = {
                'query_idx': int(row['query_idx']),
                'query': query,
                'answer_asins': answer_asins,
                'baseline_retrieved': baseline_k,
                'neighbor_analysis': neighbor_analysis,
                'gold_analysis': gold_analysis,
                'reranking_analysis': reranking_analysis
            }
            
            all_results.append(query_result)
            
            # Accumulate summary statistics
            for feature, count in neighbor_analysis['neighbor_counts'].items():
                summary_stats['neighbor_counts_by_feature'][feature].append(count)
            
            for feature, count in gold_analysis['gold_counts_by_feature'].items():
                summary_stats['gold_counts_by_feature'][feature].append(count)
            
            for feature, recall in gold_analysis['feature_recall'].items():
                summary_stats['feature_recall'][feature].append(recall)
            
            for k, gold_count in reranking_analysis['gold_after_reranking'].items():
                summary_stats['gold_after_reranking'][k].append(gold_count)
        
        # Calculate final summary
        final_summary = self._calculate_final_summary(summary_stats, len(queries_to_process))
        
        # Save results
        self._save_analysis_results(all_results, final_summary, k_baseline)
        
        logger.info(" Feature-wise analysis completed!")
        logger.info(f" Processed {len(all_results)} queries")
        logger.info(f" Results saved to: {self.output_dir}")
        
        return {
            'detailed_results': all_results,
            'summary': final_summary
        }
    
    def _calculate_final_summary(self, summary_stats: Dict, num_queries: int) -> Dict[str, Any]:
        """Calculate final summary statistics"""
        
        import numpy as np
        
        summary = {
            'total_queries': num_queries,
            'features_analyzed': self.all_features,
            'average_neighbors_by_feature': {},
            'average_gold_by_feature': {},
            'average_feature_recall': {},
            'average_gold_after_reranking': {}
        }
        
        # Average neighbors per feature
        for feature, counts in summary_stats['neighbor_counts_by_feature'].items():
            summary['average_neighbors_by_feature'][feature] = {
                'mean': np.mean(counts),
                'std': np.std(counts),
                'min': np.min(counts),
                'max': np.max(counts)
            }
        
        # Average gold documents per feature
        for feature, counts in summary_stats['gold_counts_by_feature'].items():
            summary['average_gold_by_feature'][feature] = {
                'mean': np.mean(counts),
                'std': np.std(counts),
                'total': np.sum(counts)
            }
        
        # Average recall per feature
        for feature, recalls in summary_stats['feature_recall'].items():
            summary['average_feature_recall'][feature] = {
                'mean': np.mean(recalls),
                'std': np.std(recalls)
            }
        
        # Average gold after reranking
        for k, gold_counts in summary_stats['gold_after_reranking'].items():
            summary['average_gold_after_reranking'][k] = {
                'mean': np.mean(gold_counts),
                'std': np.std(gold_counts),
                'total': np.sum(gold_counts)
            }
        
        return summary
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _save_analysis_results(self, results: List[Dict], summary: Dict, k_baseline: int):
        """Save analysis results to files"""
        
        # Save detailed results
        detailed_file = self.output_dir / f"feature_analysis_detailed_k{k_baseline}.pkl"
        with open(detailed_file, 'wb') as f:
            pickle.dump(results, f)
        
        # Save summary (convert numpy types to native Python types for JSON serialization)
        summary_file = self.output_dir / f"feature_analysis_summary_k{k_baseline}.json"
        json_compatible_summary = self._convert_numpy_types(summary)
        with open(summary_file, 'w') as f:
            json.dump(json_compatible_summary, f, indent=2)
        
        # Save human-readable report
        report_file = self.output_dir / f"feature_analysis_report_k{k_baseline}.txt"
        self._generate_human_readable_report(summary, report_file)
        
        logger.info(f" Feature analysis results saved:")
        logger.info(f" Detailed: {detailed_file.name}")
        logger.info(f" Summary: {summary_file.name}")
        logger.info(f" Report: {report_file.name}")
    
    def _generate_human_readable_report(self, summary: Dict, report_file: Path):
        """Generate human-readable analysis report"""
        
        with open(report_file, 'w') as f:
            f.write("FEATURE-WISE NEIGHBOR ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Queries Analyzed: {summary['total_queries']}\n")
            f.write(f"Features Analyzed: {', '.join(summary['features_analyzed'])}\n\n")
            
            f.write("AVERAGE NEIGHBORS PER FEATURE:\n")
            f.write("-" * 30 + "\n")
            for feature, stats in summary['average_neighbors_by_feature'].items():
                f.write(f"{feature:20s}: {stats['mean']:6.1f} ± {stats['std']:5.1f} (range: {stats['min']:3.0f}-{stats['max']:3.0f})\n")
            
            f.write("\nAVERAGE GOLD DOCUMENTS PER FEATURE:\n")
            f.write("-" * 40 + "\n")
            for feature, stats in summary['average_gold_by_feature'].items():
                f.write(f"{feature:20s}: {stats['mean']:5.2f} ± {stats['std']:4.2f} (total: {stats['total']:3.0f})\n")
            
            f.write("\nAVERAGE FEATURE RECALL:\n")
            f.write("-" * 25 + "\n")
            for feature, stats in summary['average_feature_recall'].items():
                f.write(f"{feature:20s}: {stats['mean']:5.3f} ± {stats['std']:5.3f}\n")
            
            f.write("\nGOLD DOCUMENTS AFTER RERANKING:\n")
            f.write("-" * 35 + "\n")
            for k, stats in summary['average_gold_after_reranking'].items():
                f.write(f"Top-{k:3d}: {stats['mean']:5.2f} ± {stats['std']:4.2f} (total: {stats['total']:3.0f})\n")

def main():
    """Main execution function"""
    logger.info(" Feature-wise Neighbor Analysis Tool")
    logger.info(" Analyzing neighbor distribution across features")
    logger.info("=" * 60)
    
    # Create analyzer (will reuse the fast analyzer's loaded data)
    analyzer = FeatureWiseNeighborAnalyzer()
    
    try:
        # Run complete analysis
        results = analyzer.run_complete_analysis(
            max_queries=None, # Analyze all queries
            k_baseline=10 # Use k=10 for baseline
        )
        
        logger.info(" Feature-wise analysis completed successfully!")
        
    except Exception as e:
        logger.error(f" Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
