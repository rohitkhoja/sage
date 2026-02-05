#!/usr/bin/env python3
"""
Diagnostic Script: Analyze Graph Connectivity Issues

This script will help identify why all components are getting connected
even with high percentile thresholds (99th percentile).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import defaultdict, Counter
import networkx as nx
from loguru import logger
import sys

class ConnectivityAnalyzer:
    """Analyze why graph components are all connected"""
    
    def __init__(self, batch_results_dir: str, output_dir: str = None):
        self.batch_results_dir = Path(batch_results_dir)
        self.output_dir = Path(output_dir) if output_dir else Path("connectivity_analysis")
        self.output_dir.mkdir(exist_ok=True)
        
        # Data containers
        self.similarity_df = None
        self.edge_graph = None
        
        logger.info(f"Initialized connectivity analyzer")
        logger.info(f"Batch results dir: {self.batch_results_dir}")
        logger.info(f"Output dir: {self.output_dir}")
    
    def load_batch_results(self) -> pd.DataFrame:
        """Load all batch results from CSV files"""
        logger.info("Loading batch results...")
        
        if not self.batch_results_dir.exists():
            logger.error(f"Batch results directory does not exist: {self.batch_results_dir}")
            return pd.DataFrame()
        
        batch_files = list(self.batch_results_dir.glob("batch_*.csv"))
        if not batch_files:
            logger.error(f"No batch files found in {self.batch_results_dir}")
            return pd.DataFrame()
        
        logger.info(f"Found {len(batch_files)} batch files")
        
        all_data = []
        for i, batch_file in enumerate(batch_files):
            try:
                df = pd.read_csv(batch_file)
                all_data.append(df)
                if i % 10 == 0:
                    logger.info(f"Loaded {i+1}/{len(batch_files)} batch files")
            except Exception as e:
                logger.warning(f"Error loading {batch_file}: {e}")
        
        if not all_data:
            logger.error("No valid batch data loaded")
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Loaded {len(combined_df)} total similarity records")
        
        self.similarity_df = combined_df
        return combined_df
    
    def analyze_threshold_effects(self):
        """Analyze how different thresholds affect connectivity"""
        logger.info("Analyzing threshold effects on connectivity...")
        
        if self.similarity_df is None or len(self.similarity_df) == 0:
            logger.error("No similarity data available")
            return
        
        # Define different percentile thresholds to test
        percentiles = [50, 70, 80, 85, 90, 95, 97, 99, 99.5, 99.9]
        
        # Metrics to analyze by edge type
        metrics_by_type = {
            'doc-doc': ['topic_similarity', 'content_similarity', 'entity_count', 'event_count'],
            'doc-table': ['column_similarity', 'topic_title_similarity', 'topic_summary_similarity', 'entity_count'],
            'table-table': ['column_similarity', 'title_similarity', 'description_similarity', 'entity_count']
        }
        
        results = []
        
        for edge_type in ['doc-doc', 'doc-table', 'table-table']:
            edge_data = self.similarity_df[self.similarity_df['edge_type'] == edge_type].copy()
            
            if len(edge_data) == 0:
                logger.warning(f"No data for edge type: {edge_type}")
                continue
            
            logger.info(f"Analyzing {edge_type}: {len(edge_data)} edges")
            
            for percentile in percentiles:
                for metric in metrics_by_type[edge_type]:
                    if metric not in edge_data.columns:
                        continue
                    
                    threshold = edge_data[metric].quantile(percentile / 100.0)
                    above_threshold = (edge_data[metric] > threshold).sum()
                    percentage_above = (above_threshold / len(edge_data)) * 100
                    
                    results.append({
                        'edge_type': edge_type,
                        'metric': metric,
                        'percentile': percentile,
                        'threshold_value': threshold,
                        'edges_above_threshold': above_threshold,
                        'percentage_above': percentage_above,
                        'total_edges': len(edge_data)
                    })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / "threshold_analysis.csv", index=False)
        
        # Visualize threshold effects
        self._plot_threshold_effects(results_df)
        
        return results_df
    
    def _plot_threshold_effects(self, results_df: pd.DataFrame):
        """Plot threshold effects"""
        
        # Plot 1: Number of edges above threshold by percentile
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for i, edge_type in enumerate(['doc-doc', 'doc-table', 'table-table']):
            if i >= 3:
                break
            row, col = i // 2, i % 2
            
            edge_data = results_df[results_df['edge_type'] == edge_type]
            
            for metric in edge_data['metric'].unique():
                metric_data = edge_data[edge_data['metric'] == metric]
                axes[row, col].plot(metric_data['percentile'], metric_data['edges_above_threshold'], 
                                  marker='o', label=metric)
            
            axes[row, col].set_title(f'{edge_type} - Edges Above Threshold')
            axes[row, col].set_xlabel('Percentile')
            axes[row, col].set_ylabel('Number of Edges')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].set_yscale('log')
        
        # Remove unused subplot
        if len(['doc-doc', 'doc-table', 'table-table']) < 4:
            axes[1, 1].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "threshold_effects.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Threshold values by percentile
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for i, edge_type in enumerate(['doc-doc', 'doc-table', 'table-table']):
            if i >= 3:
                break
            row, col = i // 2, i % 2
            
            edge_data = results_df[results_df['edge_type'] == edge_type]
            
            for metric in edge_data['metric'].unique():
                metric_data = edge_data[edge_data['metric'] == metric]
                axes[row, col].plot(metric_data['percentile'], metric_data['threshold_value'], 
                                  marker='s', label=metric)
            
            axes[row, col].set_title(f'{edge_type} - Threshold Values')
            axes[row, col].set_xlabel('Percentile')
            axes[row, col].set_ylabel('Threshold Value')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        # Remove unused subplot
        if len(['doc-doc', 'doc-table', 'table-table']) < 4:
            axes[1, 1].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "threshold_values.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_connectivity_patterns(self):
        """Analyze connectivity patterns to understand why everything connects"""
        logger.info("Analyzing connectivity patterns...")
        
        if self.similarity_df is None:
            logger.error("No similarity data available")
            return
        
        # Test different threshold combinations
        threshold_combinations = [
            # Conservative (99.9th percentile)
            {'doc-doc': {'and_logic': True, 'thresholds': {'topic_similarity': 0.999, 'content_similarity': 0.999}},
             'doc-table': {'and_logic': True, 'thresholds': {'column_similarity': 0.999, 'topic_title_similarity': 0.999, 'topic_summary_similarity': 0.999}},
             'table-table': {'and_logic': True, 'thresholds': {'column_similarity': 0.999, 'title_similarity': 0.999, 'description_similarity': 0.999}}},
            
            # Very Conservative (99.5th percentile)
            {'doc-doc': {'and_logic': True, 'thresholds': {'topic_similarity': 0.995, 'content_similarity': 0.995}},
             'doc-table': {'and_logic': True, 'thresholds': {'column_similarity': 0.995, 'topic_title_similarity': 0.995, 'topic_summary_similarity': 0.995}},
             'table-table': {'and_logic': True, 'thresholds': {'column_similarity': 0.995, 'title_similarity': 0.995, 'description_similarity': 0.995}}},
            
            # Current approach (99th percentile)
            {'doc-doc': {'and_logic': True, 'thresholds': {'topic_similarity': 0.99, 'content_similarity': 0.99}},
             'doc-table': {'and_logic': True, 'thresholds': {'column_similarity': 0.99, 'topic_title_similarity': 0.99, 'topic_summary_similarity': 0.99}},
             'table-table': {'and_logic': True, 'thresholds': {'column_similarity': 0.99, 'title_similarity': 0.99, 'description_similarity': 0.99}}},
            
            # Entity-only approach
            {'doc-doc': {'and_logic': False, 'thresholds': {'entity_count': 1}},
             'doc-table': {'and_logic': False, 'thresholds': {'entity_count': 1}},
             'table-table': {'and_logic': False, 'thresholds': {'entity_count': 1}}},
        ]
        
        connectivity_results = []
        
        for i, threshold_config in enumerate(threshold_combinations):
            logger.info(f"Testing threshold configuration {i+1}/{len(threshold_combinations)}")
            
            selected_edges = self._select_edges_with_thresholds(threshold_config)
            
            if len(selected_edges) == 0:
                logger.warning(f"No edges selected with configuration {i+1}")
                connectivity_results.append({
                    'config_id': i+1,
                    'total_edges': 0,
                    'connected_components': 0,
                    'largest_component_size': 0,
                    'nodes_in_largest_component': 0
                })
                continue
            
            # Build graph and analyze connectivity
            component_analysis = self._analyze_graph_connectivity(selected_edges)
            component_analysis['config_id'] = i+1
            component_analysis['total_edges'] = len(selected_edges)
            
            connectivity_results.append(component_analysis)
            
            logger.info(f"Config {i+1}: {len(selected_edges)} edges, "
                       f"{component_analysis['connected_components']} components, "
                       f"largest: {component_analysis['largest_component_size']} nodes")
        
        # Save connectivity analysis
        connectivity_df = pd.DataFrame(connectivity_results)
        connectivity_df.to_csv(self.output_dir / "connectivity_analysis.csv", index=False)
        
        return connectivity_df
    
    def _select_edges_with_thresholds(self, threshold_config: dict) -> pd.DataFrame:
        """Select edges based on threshold configuration"""
        selected_edges = []
        
        for edge_type, config in threshold_config.items():
            edge_data = self.similarity_df[self.similarity_df['edge_type'] == edge_type].copy()
            
            if len(edge_data) == 0:
                continue
            
            and_logic = config.get('and_logic', True)
            thresholds = config['thresholds']
            
            if and_logic:
                # ALL conditions must be met (AND logic)
                mask = pd.Series([True] * len(edge_data))
                
                for metric, percentile in thresholds.items():
                    if metric not in edge_data.columns:
                        continue
                    
                    if metric == 'entity_count':
                        # For entity count, use direct threshold
                        threshold_value = percentile
                    else:
                        # For similarity metrics, use percentile
                        threshold_value = edge_data[metric].quantile(percentile)
                    
                    mask = mask & (edge_data[metric] > threshold_value)
                
                selected_edges.append(edge_data[mask])
            else:
                # ANY condition can be met (OR logic)
                mask = pd.Series([False] * len(edge_data))
                
                for metric, percentile in thresholds.items():
                    if metric not in edge_data.columns:
                        continue
                    
                    if metric == 'entity_count':
                        threshold_value = percentile
                    else:
                        threshold_value = edge_data[metric].quantile(percentile)
                    
                    mask = mask | (edge_data[metric] > threshold_value)
                
                selected_edges.append(edge_data[mask])
        
        if not selected_edges:
            return pd.DataFrame()
        
        return pd.concat(selected_edges, ignore_index=True)
    
    def _analyze_graph_connectivity(self, edges_df: pd.DataFrame) -> dict:
        """Analyze graph connectivity from edge DataFrame"""
        if len(edges_df) == 0:
            return {
                'connected_components': 0,
                'largest_component_size': 0,
                'nodes_in_largest_component': 0,
                'total_nodes': 0
            }
        
        # Build NetworkX graph
        G = nx.Graph()
        
        for _, edge in edges_df.iterrows():
            source = edge['source_chunk_id']
            target = edge['target_chunk_id']
            G.add_edge(source, target)
        
        # Analyze connectivity
        connected_components = list(nx.connected_components(G))
        component_sizes = [len(comp) for comp in connected_components]
        
        return {
            'connected_components': len(connected_components),
            'largest_component_size': max(component_sizes) if component_sizes else 0,
            'nodes_in_largest_component': max(component_sizes) if component_sizes else 0,
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges()
        }
    
    def analyze_similarity_distributions(self):
        """Analyze the distribution of similarity values to understand data characteristics"""
        logger.info("Analyzing similarity distributions...")
        
        if self.similarity_df is None:
            logger.error("No similarity data available")
            return
        
        # Analyze by edge type
        for edge_type in ['doc-doc', 'doc-table', 'table-table']:
            edge_data = self.similarity_df[self.similarity_df['edge_type'] == edge_type].copy()
            
            if len(edge_data) == 0:
                continue
            
            logger.info(f"\n=== {edge_type.upper()} ANALYSIS ===")
            logger.info(f"Total edges: {len(edge_data):,}")
            
            # Define relevant metrics for this edge type
            if edge_type == 'doc-doc':
                metrics = ['topic_similarity', 'content_similarity', 'entity_count', 'event_count']
            elif edge_type == 'doc-table':
                metrics = ['column_similarity', 'topic_title_similarity', 'topic_summary_similarity', 'entity_count']
            else:  # table-table
                metrics = ['column_similarity', 'title_similarity', 'description_similarity', 'entity_count']
            
            # Analyze each metric
            for metric in metrics:
                if metric not in edge_data.columns:
                    continue
                
                values = edge_data[metric].dropna()
                if len(values) == 0:
                    continue
                
                # Calculate statistics
                stats = {
                    'count': len(values),
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'p50': values.quantile(0.5),
                    'p90': values.quantile(0.9),
                    'p95': values.quantile(0.95),
                    'p99': values.quantile(0.99),
                    'p99.9': values.quantile(0.999),
                    'zeros': (values == 0).sum(),
                    'non_zeros': (values > 0).sum()
                }
                
                logger.info(f"\n{metric}:")
                logger.info(f"  Count: {stats['count']:,}")
                logger.info(f"  Mean: {stats['mean']:.4f}")
                logger.info(f"  Std: {stats['std']:.4f}")
                logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                logger.info(f"  Percentiles: 50%={stats['p50']:.4f}, 90%={stats['p90']:.4f}, 95%={stats['p95']:.4f}, 99%={stats['p99']:.4f}, 99.9%={stats['p99.9']:.4f}")
                logger.info(f"  Zero values: {stats['zeros']:,} ({stats['zeros']/stats['count']*100:.1f}%)")
                logger.info(f"  Non-zero values: {stats['non_zeros']:,} ({stats['non_zeros']/stats['count']*100:.1f}%)")
        
        # Generate distribution plots
        self._plot_similarity_distributions()
    
    def _plot_similarity_distributions(self):
        """Plot similarity distributions for all edge types and metrics"""
        
        for edge_type in ['doc-doc', 'doc-table', 'table-table']:
            edge_data = self.similarity_df[self.similarity_df['edge_type'] == edge_type].copy()
            
            if len(edge_data) == 0:
                continue
            
            # Define relevant metrics
            if edge_type == 'doc-doc':
                metrics = ['topic_similarity', 'content_similarity', 'entity_count', 'event_count']
            elif edge_type == 'doc-table':
                metrics = ['column_similarity', 'topic_title_similarity', 'topic_summary_similarity', 'entity_count']
            else:  # table-table
                metrics = ['column_similarity', 'title_similarity', 'description_similarity', 'entity_count']
            
            # Filter metrics that exist in data
            available_metrics = [m for m in metrics if m in edge_data.columns]
            
            if not available_metrics:
                continue
            
            # Create plots
            n_metrics = len(available_metrics)
            cols = 2
            rows = (n_metrics + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
            
            if rows == 1:
                axes = axes if n_metrics > 1 else [axes]
            else:
                axes = axes.flatten()
            
            for i, metric in enumerate(available_metrics):
                values = edge_data[metric].dropna()
                
                if len(values) == 0:
                    axes[i].text(0.5, 0.5, f'No data for {metric}', ha='center', va='center', transform=axes[i].transAxes)
                    continue
                
                # Plot histogram with percentile lines
                axes[i].hist(values, bins=50, alpha=0.7, edgecolor='black')
                
                # Add percentile lines
                percentiles = [90, 95, 99, 99.9]
                colors = ['orange', 'red', 'purple', 'black']
                
                for p, color in zip(percentiles, colors):
                    p_value = values.quantile(p/100)
                    axes[i].axvline(p_value, color=color, linestyle='--', alpha=0.8, 
                                   label=f'{p}th %ile: {p_value:.3f}')
                
                axes[i].set_title(f'{edge_type}: {metric}\n({len(values):,} values)')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(available_metrics), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f"distributions_{edge_type.replace('-', '_')}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def identify_connectivity_root_causes(self):
        """Identify root causes of high connectivity"""
        logger.info("Identifying connectivity root causes...")
        
        if self.similarity_df is None:
            logger.error("No similarity data available")
            return
        
        # Check for common issues
        issues = []
        
        # Issue 1: Too many high similarity values
        for edge_type in ['doc-doc', 'doc-table', 'table-table']:
            edge_data = self.similarity_df[self.similarity_df['edge_type'] == edge_type].copy()
            
            if len(edge_data) == 0:
                continue
            
            # Check similarity metrics
            similarity_cols = [col for col in edge_data.columns if 'similarity' in col and edge_data[col].dtype in ['float64', 'int64']]
            
            for col in similarity_cols:
                values = edge_data[col].dropna()
                if len(values) == 0:
                    continue
                
                # Check if too many values are high
                high_values = (values > 0.9).sum()
                very_high_values = (values > 0.95).sum()
                extremely_high_values = (values > 0.99).sum()
                
                high_pct = high_values / len(values) * 100
                very_high_pct = very_high_values / len(values) * 100
                extremely_high_pct = extremely_high_values / len(values) * 100
                
                if high_pct > 10:  # More than 10% of values are > 0.9
                    issues.append({
                        'issue_type': 'high_similarity_concentration',
                        'edge_type': edge_type,
                        'metric': col,
                        'high_values_pct': high_pct,
                        'very_high_values_pct': very_high_pct,
                        'extremely_high_values_pct': extremely_high_pct,
                        'description': f"{col} in {edge_type}: {high_pct:.1f}% of values > 0.9"
                    })
        
        # Issue 2: Entity count distribution
        entity_data = self.similarity_df[self.similarity_df['entity_count'] > 0]
        if len(entity_data) > 0:
            entity_pct = len(entity_data) / len(self.similarity_df) * 100
            
            if entity_pct > 5:  # More than 5% of edges have entity matches
                issues.append({
                    'issue_type': 'high_entity_matching',
                    'edge_type': 'all',
                    'metric': 'entity_count',
                    'entity_match_pct': entity_pct,
                    'description': f"{entity_pct:.1f}% of edges have entity matches"
                })
        
        # Issue 3: Check for duplicate content or highly similar embeddings
        content_sim_high = (self.similarity_df['content_similarity'] > 0.95).sum()
        content_sim_pct = content_sim_high / len(self.similarity_df) * 100
        
        if content_sim_pct > 1:
            issues.append({
                'issue_type': 'duplicate_content_suspected',
                'edge_type': 'all',
                'metric': 'content_similarity',
                'high_content_sim_pct': content_sim_pct,
                'description': f"{content_sim_pct:.1f}% of edges have content similarity > 0.95"
            })
        
        # Save issues to file
        with open(self.output_dir / "connectivity_issues.json", 'w') as f:
            json.dump(issues, f, indent=2)
        
        # Log issues
        logger.info(f"\n=== CONNECTIVITY ISSUES IDENTIFIED ===")
        if not issues:
            logger.info("No major connectivity issues detected.")
        else:
            for issue in issues:
                logger.warning(f"ISSUE: {issue['description']}")
        
        return issues
    
    def suggest_thresholds(self):
        """Suggest better thresholds to reduce connectivity"""
        logger.info("Suggesting improved thresholds...")
        
        if self.similarity_df is None:
            logger.error("No similarity data available")
            return
        
        suggestions = []
        
        # Target: Select top 0.1% to 1% of edges for each edge type
        target_percentages = [0.1, 0.5, 1.0]
        
        for edge_type in ['doc-doc', 'doc-table', 'table-table']:
            edge_data = self.similarity_df[self.similarity_df['edge_type'] == edge_type].copy()
            
            if len(edge_data) == 0:
                continue
            
            for target_pct in target_percentages:
                target_edges = int(len(edge_data) * target_pct / 100)
                
                if target_edges < 1:
                    continue
                
                # Calculate required percentile to get target number of edges
                required_percentile = 100 - target_pct
                
                # Define metrics for this edge type
                if edge_type == 'doc-doc':
                    metrics = ['topic_similarity', 'content_similarity']
                elif edge_type == 'doc-table':
                    metrics = ['column_similarity', 'topic_title_similarity', 'topic_summary_similarity']
                else:  # table-table
                    metrics = ['column_similarity', 'title_similarity', 'description_similarity']
                
                # Calculate thresholds for AND logic
                metric_thresholds = {}
                for metric in metrics:
                    if metric in edge_data.columns:
                        threshold = edge_data[metric].quantile(required_percentile / 100)
                        metric_thresholds[metric] = threshold
                
                # Test how many edges this would select
                mask = pd.Series([True] * len(edge_data))
                for metric, threshold in metric_thresholds.items():
                    mask = mask & (edge_data[metric] > threshold)
                
                selected_edges = mask.sum()
                actual_pct = selected_edges / len(edge_data) * 100
                
                suggestions.append({
                    'edge_type': edge_type,
                    'target_percentage': target_pct,
                    'target_edges': target_edges,
                    'actual_edges': selected_edges,
                    'actual_percentage': actual_pct,
                    'required_percentile': required_percentile,
                    'thresholds': metric_thresholds
                })
        
        # Save suggestions
        suggestions_df = pd.DataFrame(suggestions)
        suggestions_df.to_csv(self.output_dir / "threshold_suggestions.csv", index=False)
        
        # Log suggestions
        logger.info("\n=== THRESHOLD SUGGESTIONS ===")
        for suggestion in suggestions:
            logger.info(f"\n{suggestion['edge_type']} - Target: {suggestion['target_percentage']:.1f}% of edges")
            logger.info(f"  Actual: {suggestion['actual_edges']} edges ({suggestion['actual_percentage']:.2f}%)")
            logger.info(f"  Required percentile: {suggestion['required_percentile']:.1f}th")
            for metric, threshold in suggestion['thresholds'].items():
                logger.info(f"  {metric}: {threshold:.4f}")
        
        return suggestions
    
    def run_full_analysis(self):
        """Run complete connectivity analysis"""
        logger.info("Starting full connectivity analysis...")
        
        # Step 1: Load data
        self.load_batch_results()
        
        if self.similarity_df is None or len(self.similarity_df) == 0:
            logger.error("No data loaded. Cannot proceed with analysis.")
            return
        
        # Step 2: Analyze threshold effects
        logger.info("\n" + "="*50)
        threshold_results = self.analyze_threshold_effects()
        
        # Step 3: Analyze similarity distributions
        logger.info("\n" + "="*50)
        self.analyze_similarity_distributions()
        
        # Step 4: Analyze connectivity patterns
        logger.info("\n" + "="*50)
        connectivity_results = self.analyze_connectivity_patterns()
        
        # Step 5: Identify root causes
        logger.info("\n" + "="*50)
        issues = self.identify_connectivity_root_causes()
        
        # Step 6: Suggest better thresholds
        logger.info("\n" + "="*50)
        suggestions = self.suggest_thresholds()
        
        # Generate summary report
        self._generate_summary_report(threshold_results, connectivity_results, issues, suggestions)
        
        logger.info(f"\nFull analysis completed. Results saved to: {self.output_dir}")
    
    def _generate_summary_report(self, threshold_results, connectivity_results, issues, suggestions):
        """Generate a comprehensive summary report"""
        
        report_lines = [
            "GRAPH CONNECTIVITY ANALYSIS SUMMARY REPORT",
            "=" * 60,
            "",
            "PROBLEM: All graph components are getting connected even with 99th percentile thresholds",
            "",
            "DATA OVERVIEW:",
            f"  Total similarity records: {len(self.similarity_df):,}",
        ]
        
        # Edge type breakdown
        edge_counts = self.similarity_df['edge_type'].value_counts()
        for edge_type, count in edge_counts.items():
            pct = count / len(self.similarity_df) * 100
            report_lines.append(f"  {edge_type}: {count:,} ({pct:.1f}%)")
        
        report_lines.extend([
            "",
            "ROOT CAUSE ANALYSIS:",
        ])
        
        if issues:
            for issue in issues:
                report_lines.append(f"  ❌ {issue['description']}")
        else:
            report_lines.append("  ✅ No major issues detected in data quality")
        
        report_lines.extend([
            "",
            "CONNECTIVITY TEST RESULTS:",
        ])
        
        if connectivity_results is not None and len(connectivity_results) > 0:
            for _, result in connectivity_results.iterrows():
                report_lines.append(f"  Config {result['config_id']}: {result['total_edges']} edges → "
                                  f"{result['connected_components']} components "
                                  f"(largest: {result['largest_component_size']} nodes)")
        
        report_lines.extend([
            "",
            "RECOMMENDED SOLUTIONS:",
            "",
            "1. USE STRICTER THRESHOLDS:",
        ])
        
        if suggestions:
            # Show most restrictive suggestions (0.1% target)
            restrictive_suggestions = [s for s in suggestions if s['target_percentage'] == 0.1]
            for suggestion in restrictive_suggestions:
                report_lines.append(f"   {suggestion['edge_type']}: Use {suggestion['required_percentile']:.1f}th percentile")
                for metric, threshold in suggestion['thresholds'].items():
                    report_lines.append(f"     {metric} > {threshold:.4f}")
        
        report_lines.extend([
            "",
            "2. ALTERNATIVE APPROACHES:",
            "   - Use entity matching ONLY (ignore similarity scores)",
            "   - Implement topic modeling to group similar content first",
            "   - Use clustering to identify distinct content groups",
            "   - Apply minimum edge weight thresholds in graph construction",
            "",
            "3. DATA QUALITY IMPROVEMENTS:",
            "   - Check for duplicate or near-duplicate content",
            "   - Verify embedding quality and diversity",
            "   - Consider document preprocessing to increase diversity",
            "",
            "FILES GENERATED:",
            f"   - threshold_analysis.csv: Detailed threshold analysis",
            f"   - connectivity_analysis.csv: Connectivity test results", 
            f"   - threshold_suggestions.csv: Suggested threshold values",
            f"   - connectivity_issues.json: Identified data issues",
            f"   - Distribution plots: distributions_*.png",
            f"   - Threshold effect plots: threshold_*.png",
        ])
        
        # Save summary report
        with open(self.output_dir / "SUMMARY_REPORT.txt", 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Also log the summary
        logger.info("\n" + "\n".join(report_lines))

def main():
    """Main execution function"""
    
    # Configure paths
    batch_results_dir = "/shared/khoja/CogComp/output/batch_results"
    output_dir = "/shared/khoja/CogComp/connectivity_analysis"
    
    # Initialize analyzer
    analyzer = ConnectivityAnalyzer(batch_results_dir, output_dir)
    
    # Run full analysis
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main() 