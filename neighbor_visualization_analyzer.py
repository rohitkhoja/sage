#!/usr/bin/env python3
"""
Neighbor Visualization and Analysis System

This system creates comprehensive visualizations and statistical analysis
for neighbor pruning optimization in knowledge graph traversal.

Key functionality:
1. Generate distribution plots comparing gold vs non-gold neighbors
2. Create individual question analysis reports
3. Generate aggregated statistics and insights
4. Identify optimal thresholds for neighbor pruning
5. Analyze gold document distribution patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from collections import defaultdict, Counter
import pickle
import json
from dataclasses import asdict
from scipy import stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
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

# Import our data structures
from neighbor_pruning_analyzer import NodeMetrics, QuestionAnalysis, NeighborPruningAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NeighborVisualizationAnalyzer:
    """Comprehensive visualization and analysis system for neighbor pruning"""
    
    def __init__(self, output_dir: str = "output/neighbor_pruning_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "individual_questions").mkdir(exist_ok=True)
        (self.output_dir / "aggregated_analysis").mkdir(exist_ok=True)
        (self.output_dir / "distribution_plots").mkdir(exist_ok=True)
        
        # Analysis containers
        self.analysis_results = []
        self.aggregated_stats = {}
        self.threshold_analysis = {}
        
        logger.info(f"Initialized NeighborVisualizationAnalyzer with output dir: {self.output_dir}")
    
    def load_analysis_results(self, results_file: str = "output/neighbor_pruning_cache/analysis_results.pkl"):
        """Load analysis results from cache"""
        logger.info(f"Loading analysis results from: {results_file}")
        
        with open(results_file, 'rb') as f:
            self.analysis_results = pickle.load(f)
        
        logger.info(f"Loaded {len(self.analysis_results)} analysis results")
    
    def create_comparison_dataset(self) -> pd.DataFrame:
        """Create a comprehensive dataset comparing all neighbors"""
        logger.info("Creating comparison dataset...")
        
        data_rows = []
        
        for analysis in self.analysis_results:
            question_id = analysis.question_id
            question_text = analysis.question_text
            
            for node_metrics in analysis.all_neighbors:
                row = {
                    'question_id': question_id,
                    'question_text': question_text[:100] + "..." if len(question_text) > 100 else question_text,
                    'node_id': node_metrics.node_id,
                    'node_type': node_metrics.node_type,
                    'is_gold': node_metrics.is_gold,
                    'entity_exact_matches': node_metrics.entity_exact_matches,
                    'entity_substring_matches': node_metrics.entity_substring_matches,
                    'event_exact_matches': node_metrics.event_exact_matches,
                    'event_substring_matches': node_metrics.event_substring_matches,
                    'content_similarity': node_metrics.content_similarity,
                    'topic_similarity': node_metrics.topic_similarity,
                    'title_similarity': node_metrics.title_similarity,
                    'description_similarity': node_metrics.description_similarity,
                    'column_description_similarity': node_metrics.column_description_similarity,
                    'total_entities': node_metrics.total_entities,
                    'total_events': node_metrics.total_events,
                    'hop_distance': node_metrics.hop_distance,
                    # Derived metrics
                    'total_entity_matches': node_metrics.entity_exact_matches + node_metrics.entity_substring_matches,
                    'total_event_matches': node_metrics.event_exact_matches + node_metrics.event_substring_matches,
                    'entity_match_ratio': (node_metrics.entity_exact_matches + node_metrics.entity_substring_matches) / max(node_metrics.total_entities, 1),
                    'event_match_ratio': (node_metrics.event_exact_matches + node_metrics.event_substring_matches) / max(node_metrics.total_events, 1),
                    'max_text_similarity': max(node_metrics.content_similarity, node_metrics.topic_similarity,
                                             node_metrics.title_similarity, node_metrics.description_similarity,
                                             node_metrics.column_description_similarity)
                }
                data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        
        # Save dataset
        dataset_file = self.output_dir / "comparison_dataset.csv"
        df.to_csv(dataset_file, index=False)
        logger.info(f"Saved comparison dataset to {dataset_file} ({len(df)} rows)")
        
        return df
    
    def analyze_gold_distribution_patterns(self) -> Dict[str, Any]:
        """Analyze gold document distribution patterns across retrieved documents"""
        logger.info("Analyzing gold distribution patterns...")
        
        gold_distribution_stats = {
            'questions_analyzed': len(self.analysis_results),
            'total_gold_neighbors': 0,
            'total_non_gold_neighbors': 0,
            'questions_with_gold': 0,
            'retrieved_docs_with_gold': defaultdict(int),  # How many retrieved docs have gold neighbors
            'gold_neighbor_counts': [],  # Distribution of gold neighbor counts per question
            'non_gold_neighbor_counts': [],  # Distribution of non-gold neighbor counts per question
            'overlap_analysis': {
                'questions_with_multiple_gold': 0,
                'questions_with_frequent_non_gold': 0,
                'most_frequent_non_gold_neighbors': Counter()
            }
        }
        
        for analysis in self.analysis_results:
            gold_count = analysis.gold_neighbor_count
            non_gold_count = analysis.non_gold_neighbor_count
            
            gold_distribution_stats['total_gold_neighbors'] += gold_count
            gold_distribution_stats['total_non_gold_neighbors'] += non_gold_count
            gold_distribution_stats['gold_neighbor_counts'].append(gold_count)
            gold_distribution_stats['non_gold_neighbor_counts'].append(non_gold_count)
            
            if gold_count > 0:
                gold_distribution_stats['questions_with_gold'] += 1
            
            if gold_count > 1:
                gold_distribution_stats['overlap_analysis']['questions_with_multiple_gold'] += 1
            
            # Count retrieved docs with gold neighbors
            unique_retrieved_with_gold = len(set([doc for doc, _ in analysis.retrieved_with_gold_neighbors]))
            gold_distribution_stats['retrieved_docs_with_gold'][unique_retrieved_with_gold] += 1
            
            # Track frequent non-gold neighbors
            non_gold_neighbor_ids = [n.node_id for n in analysis.non_gold_neighbors]
            gold_distribution_stats['overlap_analysis']['most_frequent_non_gold_neighbors'].update(non_gold_neighbor_ids)
        
        # Calculate additional statistics
        gold_counts = gold_distribution_stats['gold_neighbor_counts']
        non_gold_counts = gold_distribution_stats['non_gold_neighbor_counts']
        
        gold_distribution_stats['average_gold_neighbors'] = np.mean(gold_counts) if gold_counts else 0
        gold_distribution_stats['average_non_gold_neighbors'] = np.mean(non_gold_counts) if non_gold_counts else 0
        gold_distribution_stats['median_gold_neighbors'] = np.median(gold_counts) if gold_counts else 0
        gold_distribution_stats['median_non_gold_neighbors'] = np.median(non_gold_counts) if non_gold_counts else 0
        
        # Save analysis
        distribution_file = self.output_dir / "aggregated_analysis" / "gold_distribution_analysis.json"
        with open(distribution_file, 'w') as f:
            # Convert Counter and defaultdict to regular dict for JSON serialization
            stats_for_json = dict(gold_distribution_stats)
            stats_for_json['retrieved_docs_with_gold'] = dict(stats_for_json['retrieved_docs_with_gold'])
            stats_for_json['overlap_analysis']['most_frequent_non_gold_neighbors'] = dict(
                stats_for_json['overlap_analysis']['most_frequent_non_gold_neighbors'].most_common(50)
            )
            json.dump(stats_for_json, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Saved gold distribution analysis to {distribution_file}")
        return gold_distribution_stats
    
    def perform_threshold_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform threshold analysis to identify optimal pruning criteria"""
        logger.info("Performing threshold analysis...")
        
        # Metrics to analyze for thresholding
        similarity_metrics = [
            'content_similarity', 'topic_similarity', 'title_similarity',
            'description_similarity', 'column_description_similarity', 'max_text_similarity'
        ]
        
        count_metrics = [
            'entity_exact_matches', 'entity_substring_matches', 'total_entity_matches',
            'event_exact_matches', 'event_substring_matches', 'total_event_matches',
            'entity_match_ratio', 'event_match_ratio'
        ]
        
        threshold_results = {}
        
        for metric in similarity_metrics + count_metrics:
            logger.info(f"Analyzing thresholds for {metric}")
            
            # Get values for gold and non-gold neighbors
            gold_values = df[df['is_gold'] == True][metric].values
            non_gold_values = df[df['is_gold'] == False][metric].values
            
            if len(gold_values) == 0 or len(non_gold_values) == 0:
                continue
            
            # Calculate basic statistics
            metric_stats = {
                'gold_mean': np.mean(gold_values),
                'gold_median': np.median(gold_values),
                'gold_std': np.std(gold_values),
                'non_gold_mean': np.mean(non_gold_values),
                'non_gold_median': np.median(non_gold_values),
                'non_gold_std': np.std(non_gold_values),
                'statistical_test': None
            }
            
            # Perform statistical test
            try:
                statistic, p_value = stats.mannwhitneyu(gold_values, non_gold_values, alternative='two-sided')
                metric_stats['statistical_test'] = {
                    'test': 'Mann-Whitney U',
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'significant': bool(p_value < 0.05)
                }
            except Exception as e:
                logger.warning(f"Statistical test failed for {metric}: {e}")
            
            # ROC analysis for optimal threshold
            try:
                y_true = df['is_gold'].astype(int).values
                y_scores = df[metric].values
                
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                roc_auc = auc(fpr, tpr)
                
                # Find optimal threshold (maximize TPR - FPR)
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                
                metric_stats['roc_analysis'] = {
                    'auc': float(roc_auc),
                    'optimal_threshold': float(optimal_threshold),
                    'optimal_tpr': float(tpr[optimal_idx]),
                    'optimal_fpr': float(fpr[optimal_idx])
                }
                
                # Precision-Recall analysis
                precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
                pr_auc = auc(recall, precision)
                
                # Find threshold that maximizes F1 score
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                best_f1_idx = np.argmax(f1_scores)
                
                metric_stats['precision_recall_analysis'] = {
                    'auc': float(pr_auc),
                    'best_f1_threshold': float(pr_thresholds[best_f1_idx]) if best_f1_idx < len(pr_thresholds) else float(pr_thresholds[-1]),
                    'best_f1_score': float(f1_scores[best_f1_idx]),
                    'best_precision': float(precision[best_f1_idx]),
                    'best_recall': float(recall[best_f1_idx])
                }
                
            except Exception as e:
                logger.warning(f"ROC/PR analysis failed for {metric}: {e}")
            
            threshold_results[metric] = metric_stats
        
        # Save threshold analysis
        threshold_file = self.output_dir / "aggregated_analysis" / "threshold_analysis.json"
        with open(threshold_file, 'w') as f:
            json.dump(threshold_results, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Saved threshold analysis to {threshold_file}")
        return threshold_results
    
    def create_distribution_plots(self, df: pd.DataFrame):
        """Create comprehensive distribution plots"""
        logger.info("Creating distribution plots...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Entity and Event Matching Distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Entity and Event Matching Distributions', fontsize=16)
        
        # Entity exact matches
        axes[0,0].hist([df[df['is_gold'] == True]['entity_exact_matches'], 
                       df[df['is_gold'] == False]['entity_exact_matches']], 
                      bins=20, alpha=0.7, label=['Gold', 'Non-Gold'], density=True)
        axes[0,0].set_title('Entity Exact Matches')
        axes[0,0].set_xlabel('Count')
        axes[0,0].set_ylabel('Density')
        axes[0,0].legend()
        
        # Entity substring matches
        axes[0,1].hist([df[df['is_gold'] == True]['entity_substring_matches'], 
                       df[df['is_gold'] == False]['entity_substring_matches']], 
                      bins=20, alpha=0.7, label=['Gold', 'Non-Gold'], density=True)
        axes[0,1].set_title('Entity Substring Matches')
        axes[0,1].set_xlabel('Count')
        axes[0,1].set_ylabel('Density')
        axes[0,1].legend()
        
        # Event exact matches
        axes[1,0].hist([df[df['is_gold'] == True]['event_exact_matches'], 
                       df[df['is_gold'] == False]['event_exact_matches']], 
                      bins=20, alpha=0.7, label=['Gold', 'Non-Gold'], density=True)
        axes[1,0].set_title('Event Exact Matches')
        axes[1,0].set_xlabel('Count')
        axes[1,0].set_ylabel('Density')
        axes[1,0].legend()
        
        # Event substring matches
        axes[1,1].hist([df[df['is_gold'] == True]['event_substring_matches'], 
                       df[df['is_gold'] == False]['event_substring_matches']], 
                      bins=20, alpha=0.7, label=['Gold', 'Non-Gold'], density=True)
        axes[1,1].set_title('Event Substring Matches')
        axes[1,1].set_xlabel('Count')
        axes[1,1].set_ylabel('Density')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "distribution_plots" / "entity_event_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Text Similarity Distributions
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Text Similarity Distributions', fontsize=16)
        
        similarity_metrics = [
            ('content_similarity', 'Content Similarity'),
            ('topic_similarity', 'Topic Similarity'),
            ('title_similarity', 'Title Similarity'),
            ('description_similarity', 'Description Similarity'),
            ('column_description_similarity', 'Column Description Similarity'),
            ('max_text_similarity', 'Max Text Similarity')
        ]
        
        for i, (metric, title) in enumerate(similarity_metrics):
            row, col = i // 3, i % 3
            
            gold_values = df[df['is_gold'] == True][metric]
            non_gold_values = df[df['is_gold'] == False][metric]
            
            axes[row, col].hist([gold_values, non_gold_values], 
                               bins=30, alpha=0.7, label=['Gold', 'Non-Gold'], density=True)
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel('Similarity Score')
            axes[row, col].set_ylabel('Density')
            axes[row, col].legend()
            
            # Add mean lines
            axes[row, col].axvline(gold_values.mean(), color='blue', linestyle='--', alpha=0.8)
            axes[row, col].axvline(non_gold_values.mean(), color='orange', linestyle='--', alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "distribution_plots" / "text_similarity_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Combined Metrics Box Plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Combined Metrics Comparison (Box Plots)', fontsize=16)
        
        # Entity match ratios
        entity_data = [df[df['is_gold'] == True]['entity_match_ratio'], 
                      df[df['is_gold'] == False]['entity_match_ratio']]
        axes[0,0].boxplot(entity_data, tick_labels=['Gold', 'Non-Gold'])
        axes[0,0].set_title('Entity Match Ratio')
        axes[0,0].set_ylabel('Ratio')
        
        # Event match ratios
        event_data = [df[df['is_gold'] == True]['event_match_ratio'], 
                     df[df['is_gold'] == False]['event_match_ratio']]
        axes[0,1].boxplot(event_data, tick_labels=['Gold', 'Non-Gold'])
        axes[0,1].set_title('Event Match Ratio')
        axes[0,1].set_ylabel('Ratio')
        
        # Max text similarity
        sim_data = [df[df['is_gold'] == True]['max_text_similarity'], 
                   df[df['is_gold'] == False]['max_text_similarity']]
        axes[1,0].boxplot(sim_data, tick_labels=['Gold', 'Non-Gold'])
        axes[1,0].set_title('Max Text Similarity')
        axes[1,0].set_ylabel('Similarity Score')
        
        # Total entity matches
        total_entity_data = [df[df['is_gold'] == True]['total_entity_matches'], 
                           df[df['is_gold'] == False]['total_entity_matches']]
        axes[1,1].boxplot(total_entity_data, tick_labels=['Gold', 'Non-Gold'])
        axes[1,1].set_title('Total Entity Matches')
        axes[1,1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "distribution_plots" / "combined_metrics_boxplots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Static distribution plots created successfully")
    
    def generate_individual_question_analysis(self):
        """Generate detailed analysis for each individual question"""
        logger.info("Generating individual question analysis...")
        
        individual_dir = self.output_dir / "individual_questions"
        
        for i, analysis in enumerate(self.analysis_results):
            question_id = analysis.question_id
            question_file = individual_dir / f"question_{question_id}_analysis.json"
            
            # Create detailed analysis for this question
            question_data = {
                'question_info': {
                    'question_id': analysis.question_id,
                    'question_text': analysis.question_text,
                    'question_entities': analysis.question_entities,
                    'question_events': analysis.question_events,
                    'gold_docs': analysis.gold_docs,
                    'retrieved_docs': analysis.retrieved_docs
                },
                'neighbor_summary': {
                    'total_neighbors': analysis.total_neighbors_analyzed,
                    'gold_neighbors': analysis.gold_neighbor_count,
                    'non_gold_neighbors': analysis.non_gold_neighbor_count,
                    'gold_ratio': analysis.gold_neighbor_count / max(analysis.total_neighbors_analyzed, 1)
                },
                'gold_neighbor_details': [asdict(node) for node in analysis.gold_neighbors],
                'top_non_gold_neighbors': [asdict(node) for node in 
                                         sorted(analysis.non_gold_neighbors, 
                                               key=lambda x: x.content_similarity + x.topic_similarity, reverse=True)[:10]],
                'metric_comparisons': self._compare_gold_vs_non_gold_for_question(analysis)
            }
            
            with open(question_file, 'w') as f:
                json.dump(question_data, f, indent=2, cls=NumpyEncoder)
            
            if (i + 1) % 5 == 0:
                logger.info(f"Generated analysis for {i + 1} questions")
        
        logger.info(f"Individual question analysis completed")
    
    def _compare_gold_vs_non_gold_for_question(self, analysis: QuestionAnalysis) -> Dict[str, Any]:
        """Compare gold vs non-gold neighbors for a single question"""
        if not analysis.gold_neighbors or not analysis.non_gold_neighbors:
            return {}
        
        metrics = [
            'entity_exact_matches', 'entity_substring_matches', 'event_exact_matches', 
            'event_substring_matches', 'content_similarity', 'topic_similarity',
            'title_similarity', 'description_similarity', 'column_description_similarity'
        ]
        
        comparison = {}
        
        for metric in metrics:
            gold_values = [getattr(node, metric) for node in analysis.gold_neighbors]
            non_gold_values = [getattr(node, metric) for node in analysis.non_gold_neighbors]
            
            comparison[metric] = {
                'gold_mean': np.mean(gold_values) if gold_values else 0,
                'gold_max': np.max(gold_values) if gold_values else 0,
                'non_gold_mean': np.mean(non_gold_values) if non_gold_values else 0,
                'non_gold_max': np.max(non_gold_values) if non_gold_values else 0,
                'gold_advantage': (np.mean(gold_values) - np.mean(non_gold_values)) if gold_values and non_gold_values else 0
            }
        
        return comparison
    
    def generate_aggregated_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive aggregated statistics"""
        logger.info("Generating aggregated statistics...")
        
        stats = {
            'dataset_overview': {
                'total_neighbors': len(df),
                'gold_neighbors': len(df[df['is_gold'] == True]),
                'non_gold_neighbors': len(df[df['is_gold'] == False]),
                'gold_ratio': len(df[df['is_gold'] == True]) / len(df),
                'questions_analyzed': len(self.analysis_results),
                'document_neighbors': len(df[df['node_type'] == 'document']),
                'table_neighbors': len(df[df['node_type'] == 'table'])
            },
            'entity_event_stats': {
                'entity_exact_matches': {
                    'gold_mean': df[df['is_gold'] == True]['entity_exact_matches'].mean(),
                    'non_gold_mean': df[df['is_gold'] == False]['entity_exact_matches'].mean(),
                    'overall_mean': df['entity_exact_matches'].mean()
                },
                'entity_substring_matches': {
                    'gold_mean': df[df['is_gold'] == True]['entity_substring_matches'].mean(),
                    'non_gold_mean': df[df['is_gold'] == False]['entity_substring_matches'].mean(),
                    'overall_mean': df['entity_substring_matches'].mean()
                },
                'event_exact_matches': {
                    'gold_mean': df[df['is_gold'] == True]['event_exact_matches'].mean(),
                    'non_gold_mean': df[df['is_gold'] == False]['event_exact_matches'].mean(),
                    'overall_mean': df['event_exact_matches'].mean()
                },
                'event_substring_matches': {
                    'gold_mean': df[df['is_gold'] == True]['event_substring_matches'].mean(),
                    'non_gold_mean': df[df['is_gold'] == False]['event_substring_matches'].mean(),
                    'overall_mean': df['event_substring_matches'].mean()
                }
            },
            'similarity_stats': {
                'content_similarity': {
                    'gold_mean': df[df['is_gold'] == True]['content_similarity'].mean(),
                    'non_gold_mean': df[df['is_gold'] == False]['content_similarity'].mean(),
                    'overall_mean': df['content_similarity'].mean()
                },
                'topic_similarity': {
                    'gold_mean': df[df['is_gold'] == True]['topic_similarity'].mean(),
                    'non_gold_mean': df[df['is_gold'] == False]['topic_similarity'].mean(),
                    'overall_mean': df['topic_similarity'].mean()
                },
                'max_text_similarity': {
                    'gold_mean': df[df['is_gold'] == True]['max_text_similarity'].mean(),
                    'non_gold_mean': df[df['is_gold'] == False]['max_text_similarity'].mean(),
                    'overall_mean': df['max_text_similarity'].mean()
                }
            }
        }
        
        # Save aggregated statistics
        stats_file = self.output_dir / "aggregated_analysis" / "aggregated_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Saved aggregated statistics to {stats_file}")
        return stats
    
    def run_complete_visualization_analysis(self):
        """Run the complete visualization and analysis pipeline"""
        logger.info("Starting complete visualization and analysis...")
        
        # Load analysis results
        self.load_analysis_results()
        
        # Create comparison dataset
        df = self.create_comparison_dataset()
        
        # Analyze gold distribution patterns
        gold_distribution = self.analyze_gold_distribution_patterns()
        
        # Perform threshold analysis
        threshold_analysis = self.perform_threshold_analysis(df)
        
        # Create visualizations
        self.create_distribution_plots(df)
        
        # Generate individual question analysis
        self.generate_individual_question_analysis()
        
        # Generate aggregated statistics
        aggregated_stats = self.generate_aggregated_statistics(df)
        
        # Generate summary report
        self._generate_summary_report(df, gold_distribution, threshold_analysis, aggregated_stats)
        
        logger.info("Complete visualization and analysis finished!")
    
    def _generate_summary_report(self, df: pd.DataFrame, gold_distribution: Dict, 
                                threshold_analysis: Dict, aggregated_stats: Dict):
        """Generate a comprehensive summary report"""
        logger.info("Generating summary report...")
        
        report_file = self.output_dir / "neighbor_pruning_summary_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("NEIGHBOR PRUNING ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Dataset Overview
            f.write("📊 DATASET OVERVIEW:\n")
            f.write(f"  • Total Questions Analyzed: {aggregated_stats['dataset_overview']['questions_analyzed']:,}\n")
            f.write(f"  • Total Neighbors Analyzed: {aggregated_stats['dataset_overview']['total_neighbors']:,}\n")
            f.write(f"  • Gold Neighbors: {aggregated_stats['dataset_overview']['gold_neighbors']:,}\n")
            f.write(f"  • Non-Gold Neighbors: {aggregated_stats['dataset_overview']['non_gold_neighbors']:,}\n")
            f.write(f"  • Gold Ratio: {aggregated_stats['dataset_overview']['gold_ratio']:.3f}\n\n")
            
            # Key Findings
            f.write("🔍 KEY FINDINGS:\n")
            
            # Entity/Event matching effectiveness
            entity_gold_mean = aggregated_stats['entity_event_stats']['entity_exact_matches']['gold_mean']
            entity_non_gold_mean = aggregated_stats['entity_event_stats']['entity_exact_matches']['non_gold_mean']
            f.write(f"  • Entity Exact Matches - Gold avg: {entity_gold_mean:.2f}, Non-Gold avg: {entity_non_gold_mean:.2f}\n")
            
            # Similarity effectiveness
            content_gold_mean = aggregated_stats['similarity_stats']['content_similarity']['gold_mean']
            content_non_gold_mean = aggregated_stats['similarity_stats']['content_similarity']['non_gold_mean']
            f.write(f"  • Content Similarity - Gold avg: {content_gold_mean:.3f}, Non-Gold avg: {content_non_gold_mean:.3f}\n")
            
            topic_gold_mean = aggregated_stats['similarity_stats']['topic_similarity']['gold_mean']
            topic_non_gold_mean = aggregated_stats['similarity_stats']['topic_similarity']['non_gold_mean']
            f.write(f"  • Topic Similarity - Gold avg: {topic_gold_mean:.3f}, Non-Gold avg: {topic_non_gold_mean:.3f}\n\n")
            
            # Best discriminating metrics
            f.write("🎯 BEST DISCRIMINATING METRICS:\n")
            
            # Find metrics with largest difference between gold and non-gold
            differences = {}
            for metric in ['entity_exact_matches', 'content_similarity', 'topic_similarity', 'max_text_similarity']:
                if metric in df.columns:
                    gold_vals = df[df['is_gold'] == True][metric].mean()
                    non_gold_vals = df[df['is_gold'] == False][metric].mean()
                    differences[metric] = gold_vals - non_gold_vals
            
            sorted_diffs = sorted(differences.items(), key=lambda x: abs(x[1]), reverse=True)
            for i, (metric, diff) in enumerate(sorted_diffs[:5], 1):
                f.write(f"  {i}. {metric}: {diff:+.4f} difference\n")
            
            f.write("\n")
            
            # Threshold recommendations
            f.write("⚡ PRUNING RECOMMENDATIONS:\n")
            if 'content_similarity' in threshold_analysis:
                content_threshold = threshold_analysis['content_similarity'].get('roc_analysis', {}).get('optimal_threshold', 0)
                f.write(f"  • Content Similarity Threshold: {content_threshold:.3f}\n")
            
            if 'entity_exact_matches' in threshold_analysis:
                entity_threshold = threshold_analysis['entity_exact_matches'].get('roc_analysis', {}).get('optimal_threshold', 0)
                f.write(f"  • Entity Exact Matches Threshold: {entity_threshold:.0f}\n")
            
            # Gold distribution insights
            f.write(f"\n📈 GOLD DISTRIBUTION:\n")
            f.write(f"  • Questions with Gold Neighbors: {gold_distribution['questions_with_gold']}/{gold_distribution['questions_analyzed']}\n")
            f.write(f"  • Average Gold Neighbors per Question: {gold_distribution['average_gold_neighbors']:.2f}\n")
            f.write(f"  • Average Non-Gold Neighbors per Question: {gold_distribution['average_non_gold_neighbors']:.2f}\n")
            
            f.write("\n📁 GENERATED FILES:\n")
            f.write("  • comparison_dataset.csv - Complete dataset for further analysis\n")
            f.write("  • distribution_plots/ - Static visualization plots\n")
            f.write("  • individual_questions/ - Per-question detailed analysis\n")
            f.write("  • aggregated_analysis/ - Summary statistics and threshold analysis\n")
        
        logger.info(f"Saved summary report to {report_file}")

def main():
    """Main execution function"""
    visualizer = NeighborVisualizationAnalyzer()
    visualizer.run_complete_visualization_analysis()

if __name__ == "__main__":
    main()