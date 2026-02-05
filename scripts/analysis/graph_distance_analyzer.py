#!/usr/bin/env python3
"""
Graph Distance Analysis Tool

This script analyzes the distance between gold standard documents and retrieved documents
in a knowledge graph to understand retrieval performance.

Key functionality:
1. Load knowledge graph from edge data
2. For each query, calculate shortest paths from gold docs to retrieved docs
3. Generate comprehensive distance analysis and statistics
"""

import json
import pandas as pd
import networkx as nx
import numpy as np
import ast
import logging
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Set, Tuple, Optional
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphDistanceAnalyzer:
    """
    Analyzes shortest path distances between gold documents and retrieved documents
    in a knowledge graph to evaluate retrieval performance.
    """
    
    def __init__(self):
        self.graph = nx.Graph()
        self.nodes = set()
        self.edges_data = []
        self.distance_results = []
        self.stats = {}
        self.missing_nodes = {
            'gold_docs_missing': set(),
            'retrieved_docs_missing': set()
        }
        
    def load_edge_data(self, edges_file: str) -> bool:
        """
        Load edge data from JSON file and extract unique nodes.
        
        Args:
            edges_file: Path to JSON file containing edge data
            
        Returns:
            bool: Success status
        """
        logger.info(f"Loading edge data from: {edges_file}")
        
        try:
            with open(edges_file, 'r') as f:
                self.edges_data = json.load(f)
            
            logger.info(f"Loaded {len(self.edges_data):,} edges from JSON")
            
            # Extract all unique nodes
            for edge in self.edges_data:
                source = edge['source_chunk_id']
                target = edge['target_chunk_id']
                self.nodes.add(source)
                self.nodes.add(target)
            
            logger.info(f"Found {len(self.nodes):,} unique nodes in the graph")
            return True
            
        except Exception as e:
            logger.error(f"Error loading edge data: {e}")
            return False
    
    def build_graph(self) -> bool:
        """
        Build NetworkX graph from loaded edge data with uniform edge weights.
        
        Returns:
            bool: Success status
        """
        logger.info("Building NetworkX graph from edges...")
        
        try:
            # Add all nodes first
            self.graph.add_nodes_from(self.nodes)
            
            # Add edges with uniform weight of 1
            edges_to_add = []
            for edge in self.edges_data:
                source = edge['source_chunk_id']
                target = edge['target_chunk_id']
                # Use uniform weight of 1 for all edges
                edges_to_add.append((source, target))
            
            self.graph.add_edges_from(edges_to_add)
            
            # Graph statistics
            logger.info(f"Graph built successfully:")
            logger.info(f" - Nodes: {self.graph.number_of_nodes():,}")
            logger.info(f" - Edges: {self.graph.number_of_edges():,}")
            logger.info(f" - Connected components: {nx.number_connected_components(self.graph)}")
            
            # Find largest connected component
            largest_cc = max(nx.connected_components(self.graph), key=len)
            logger.info(f" - Largest component size: {len(largest_cc):,} nodes ({len(largest_cc)/len(self.nodes)*100:.1f}%)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error building graph: {e}")
            return False
    
    def safe_parse_list(self, value) -> List[str]:
        """
        Safely parse a string representation of a list into actual list.
        
        Args:
            value: String representation of list or actual list
            
        Returns:
            List of strings
        """
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
    
    def calculate_shortest_paths(self, source_nodes: List[str], target_nodes: List[str]) -> Dict:
        """
        Calculate shortest paths between source and target node sets.
        
        Args:
            source_nodes: List of source node IDs (gold docs)
            target_nodes: List of target node IDs (retrieved docs)
            
        Returns:
            Dictionary with path information
        """
        results = {
            'min_distance': float('inf'),
            'max_distance': 0,
            'avg_distance': 0,
            'paths_found': 0,
            'paths_missing': 0,
            'source_nodes_in_graph': 0,
            'target_nodes_in_graph': 0,
            'detailed_paths': [],
            'hop_analysis': [] # New: For 1-hop and 2-hop neighbor analysis
        }
        
        # Track missing nodes
        for node in source_nodes:
            if node not in self.graph:
                self.missing_nodes['gold_docs_missing'].add(node)
        
        for node in target_nodes:
            if node not in self.graph:
                self.missing_nodes['retrieved_docs_missing'].add(node)
        
        # Filter nodes that exist in graph
        valid_sources = [node for node in source_nodes if node in self.graph]
        valid_targets = [node for node in target_nodes if node in self.graph]
        
        results['source_nodes_in_graph'] = len(valid_sources)
        results['target_nodes_in_graph'] = len(valid_targets)
        
        if not valid_sources or not valid_targets:
            return results
        
        distances = []
        
        # Calculate shortest path from each source to each target
        for source in valid_sources:
            for target in valid_targets:
                try:
                    # Use unweighted shortest path (number of hops)
                    path_length = nx.shortest_path_length(self.graph, source, target)
                    distances.append(path_length)
                    
                    path_info = {
                        'source': source,
                        'target': target,
                        'distance': path_length
                    }
                    
                    # For 1-hop and 2-hop connections, add neighbor count analysis
                    if path_length <= 2:
                        source_neighbors = len(list(self.graph.neighbors(source)))
                        target_neighbors = len(list(self.graph.neighbors(target)))
                        
                        hop_analysis = {
                            'source': source,
                            'target': target,
                            'distance': path_length,
                            'source_neighbor_count': source_neighbors,
                            'target_neighbor_count': target_neighbors
                        }
                        results['hop_analysis'].append(hop_analysis)
                    
                    results['detailed_paths'].append(path_info)
                    results['paths_found'] += 1
                    
                except nx.NetworkXNoPath:
                    results['paths_missing'] += 1
                    continue
        
        if distances:
            results['min_distance'] = min(distances)
            results['max_distance'] = max(distances)
            results['avg_distance'] = np.mean(distances)
        
        return results
    
    def analyze_csv_questions(self, csv_file: str) -> bool:
        """
        Analyze distance for each question in the filtered CSV file.
        
        Args:
            csv_file: Path to filtered CSV file
            
        Returns:
            bool: Success status
        """
        logger.info(f"Loading filtered CSV: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded {len(df):,} questions from CSV")
            
            ranking_columns = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th']
            
            total_questions = len(df)
            
            for idx, row in df.iterrows():
                # Parse gold documents
                gold_docs = self.safe_parse_list(row['gold_docs'])
                
                # Parse retrieved documents
                retrieved_docs = []
                for col in ranking_columns:
                    val = row[col]
                    if pd.notna(val) and val != '':
                        retrieved_docs.append(str(val).strip())
                
                # Calculate shortest paths
                path_results = self.calculate_shortest_paths(gold_docs, retrieved_docs)
                
                # Store results
                question_result = {
                    'question_id': row.get('question_id', f'q_{idx}'),
                    'question': row.get('question', '')[:100] + '...' if len(str(row.get('question', ''))) > 100 else row.get('question', ''),
                    'gold_docs_count': len(gold_docs),
                    'retrieved_docs_count': len(retrieved_docs),
                    'gold_docs_in_graph': path_results['source_nodes_in_graph'],
                    'retrieved_docs_in_graph': path_results['target_nodes_in_graph'],
                    'min_distance': path_results['min_distance'] if path_results['min_distance'] != float('inf') else None,
                    'max_distance': path_results['max_distance'] if path_results['max_distance'] > 0 else None,
                    'avg_distance': path_results['avg_distance'] if path_results['avg_distance'] > 0 else None,
                    'paths_found': path_results['paths_found'],
                    'paths_missing': path_results['paths_missing'],
                    'detailed_paths': path_results['detailed_paths'],
                    'hop_analysis': path_results['hop_analysis'], # New: neighbor analysis for 1-2 hop connections
                    'gold_docs_not_in_graph': [doc for doc in gold_docs if doc not in self.graph],
                    'retrieved_docs_not_in_graph': [doc for doc in retrieved_docs if doc not in self.graph]
                }
                
                self.distance_results.append(question_result)
                
                # Progress logging
                if (idx + 1) % 50 == 0:
                    logger.info(f"Processed {idx + 1}/{total_questions} questions...")
            
            logger.info(f" Successfully analyzed {len(self.distance_results)} questions")
            return True
            
        except Exception as e:
            logger.error(f"Error analyzing CSV questions: {e}")
            return False
    
    def generate_statistics(self) -> Dict:
        """
        Generate comprehensive statistics from distance analysis.
        
        Returns:
            Dictionary containing various statistics
        """
        logger.info("Generating comprehensive statistics...")
        
        # Filter out questions with valid distances
        valid_results = [r for r in self.distance_results if r['min_distance'] is not None]
        
        if not valid_results:
            logger.warning("No valid distance results found!")
            return {}
        
        # Extract distance values
        min_distances = [r['min_distance'] for r in valid_results]
        max_distances = [r['max_distance'] for r in valid_results if r['max_distance'] is not None]
        avg_distances = [r['avg_distance'] for r in valid_results if r['avg_distance'] is not None]
        
        # Basic statistics
        stats = {
            'total_questions': len(self.distance_results),
            'questions_with_paths': len(valid_results),
            'questions_without_paths': len(self.distance_results) - len(valid_results),
            'coverage_percentage': len(valid_results) / len(self.distance_results) * 100,
            
            # Distance statistics
            'min_distance_stats': {
                'min': np.min(min_distances),
                'max': np.max(min_distances),
                'mean': np.mean(min_distances),
                'median': np.median(min_distances),
                'std': np.std(min_distances),
                'percentile_25': np.percentile(min_distances, 25),
                'percentile_75': np.percentile(min_distances, 75),
                'percentile_90': np.percentile(min_distances, 90),
                'percentile_95': np.percentile(min_distances, 95),
            }
        }
        
        # Distance distribution
        distance_counts = Counter(min_distances)
        stats['distance_distribution'] = dict(distance_counts)
        
        # Graph coverage statistics
        gold_in_graph = sum(r['gold_docs_in_graph'] for r in self.distance_results)
        total_gold = sum(r['gold_docs_count'] for r in self.distance_results)
        retrieved_in_graph = sum(r['retrieved_docs_in_graph'] for r in self.distance_results)
        total_retrieved = sum(r['retrieved_docs_count'] for r in self.distance_results)
        
        stats['graph_coverage'] = {
            'gold_docs_in_graph_percentage': gold_in_graph / max(total_gold, 1) * 100,
            'retrieved_docs_in_graph_percentage': retrieved_in_graph / max(total_retrieved, 1) * 100,
            'total_gold_docs': total_gold,
            'gold_docs_in_graph': gold_in_graph,
            'total_retrieved_docs': total_retrieved,
            'retrieved_docs_in_graph': retrieved_in_graph
        }
        
        # Path finding success rate
        total_paths_attempted = sum(r['paths_found'] + r['paths_missing'] for r in self.distance_results)
        total_paths_found = sum(r['paths_found'] for r in self.distance_results)
        
        stats['path_success_rate'] = {
            'paths_found': total_paths_found,
            'paths_attempted': total_paths_attempted,
            'success_percentage': total_paths_found / max(total_paths_attempted, 1) * 100
        }
        
        self.stats = stats
        return stats
    
    def save_results(self, output_dir: str) -> bool:
        """
        Save analysis results to files.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            bool: Success status
        """
        logger.info(f"Saving results to: {output_dir}")
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save detailed results
            results_df = pd.DataFrame(self.distance_results)
            results_file = output_path / "graph_distance_analysis_detailed.csv"
            results_df.to_csv(results_file, index=False)
            logger.info(f"Saved detailed results: {results_file}")
            
            # Save statistics
            stats_file = output_path / "graph_distance_analysis_statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2, default=str)
            logger.info(f"Saved statistics: {stats_file}")
            
            # Create summary report
            self._create_summary_report(output_path)
            
            # Save missing nodes analysis
            self._save_missing_nodes_analysis(output_path)
            
            # Save questions without paths analysis
            self._save_questions_without_paths_analysis(output_path)
            
            # Save hop neighbor analysis
            self._save_hop_neighbor_analysis(output_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False
    
    def _create_summary_report(self, output_path: Path):
        """Create a human-readable summary report."""
        report_file = output_path / "graph_distance_analysis_summary.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GRAPH DISTANCE ANALYSIS SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            if not self.stats:
                f.write("No statistics available.\n")
                return
            
            # Basic info
            f.write(f" OVERALL STATISTICS:\n")
            f.write(f" • Total Questions Analyzed: {self.stats['total_questions']:,}\n")
            f.write(f" • Questions with Valid Paths: {self.stats['questions_with_paths']:,}\n")
            f.write(f" • Questions without Paths: {self.stats['questions_without_paths']:,}\n")
            f.write(f" • Coverage Percentage: {self.stats['coverage_percentage']:.1f}%\n\n")
            
            # Distance statistics
            dist_stats = self.stats['min_distance_stats']
            f.write(f" MINIMUM DISTANCE STATISTICS:\n")
            f.write(f" • Minimum Distance: {dist_stats['min']:.0f} hops\n")
            f.write(f" • Maximum Distance: {dist_stats['max']:.0f} hops\n")
            f.write(f" • Average Distance: {dist_stats['mean']:.2f} hops\n")
            f.write(f" • Median Distance: {dist_stats['median']:.0f} hops\n")
            f.write(f" • Standard Deviation: {dist_stats['std']:.2f}\n")
            f.write(f" • 25th Percentile: {dist_stats['percentile_25']:.0f} hops\n")
            f.write(f" • 75th Percentile: {dist_stats['percentile_75']:.0f} hops\n")
            f.write(f" • 90th Percentile: {dist_stats['percentile_90']:.0f} hops\n")
            f.write(f" • 95th Percentile: {dist_stats['percentile_95']:.0f} hops\n\n")
            
            # Graph coverage
            coverage = self.stats['graph_coverage']
            f.write(f" GRAPH COVERAGE:\n")
            f.write(f" • Gold Docs in Graph: {coverage['gold_docs_in_graph']:,}/{coverage['total_gold_docs']:,} ({coverage['gold_docs_in_graph_percentage']:.1f}%)\n")
            f.write(f" • Retrieved Docs in Graph: {coverage['retrieved_docs_in_graph']:,}/{coverage['total_retrieved_docs']:,} ({coverage['retrieved_docs_in_graph_percentage']:.1f}%)\n\n")
            
            # Path success rate
            path_stats = self.stats['path_success_rate']
            f.write(f" PATH FINDING SUCCESS:\n")
            f.write(f" • Paths Found: {path_stats['paths_found']:,}/{path_stats['paths_attempted']:,}\n")
            f.write(f" • Success Rate: {path_stats['success_percentage']:.1f}%\n\n")
            
            # Distance distribution
            f.write(f" DISTANCE DISTRIBUTION:\n")
            dist_dist = self.stats['distance_distribution']
            for distance in sorted(dist_dist.keys()):
                count = dist_dist[distance]
                percentage = count / self.stats['questions_with_paths'] * 100
                f.write(f" • {int(distance)} hops: {count:,} questions ({percentage:.1f}%)\n")
        
        logger.info(f"Saved summary report: {report_file}")
    
    def _save_missing_nodes_analysis(self, output_path: Path):
        """Save analysis of missing nodes (Question 1)."""
        missing_file = output_path / "missing_nodes_analysis.txt"
        
        with open(missing_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MISSING NODES ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f" GOLD DOCUMENTS NOT IN GRAPH:\n")
            f.write(f" • Total missing gold docs: {len(self.missing_nodes['gold_docs_missing'])}\n")
            f.write(f" • Missing gold docs:\n")
            for doc in sorted(self.missing_nodes['gold_docs_missing']):
                f.write(f" - {doc}\n")
            
            f.write(f"\n RETRIEVED DOCUMENTS NOT IN GRAPH:\n")
            f.write(f" • Total missing retrieved docs: {len(self.missing_nodes['retrieved_docs_missing'])}\n")
            f.write(f" • Missing retrieved docs:\n")
            for doc in sorted(self.missing_nodes['retrieved_docs_missing']):
                f.write(f" - {doc}\n")
        
        # Also save as JSON for programmatic access
        missing_json = output_path / "missing_nodes_analysis.json"
        missing_data = {
            'gold_docs_missing': list(self.missing_nodes['gold_docs_missing']),
            'retrieved_docs_missing': list(self.missing_nodes['retrieved_docs_missing']),
            'gold_docs_missing_count': len(self.missing_nodes['gold_docs_missing']),
            'retrieved_docs_missing_count': len(self.missing_nodes['retrieved_docs_missing'])
        }
        with open(missing_json, 'w') as f:
            json.dump(missing_data, f, indent=2)
            
        logger.info(f"Saved missing nodes analysis: {missing_file}")
    
    def _save_questions_without_paths_analysis(self, output_path: Path):
        """Save analysis of questions without paths (Question 2)."""
        no_paths_file = output_path / "questions_without_paths_analysis.txt"
        
        # Find questions without paths
        questions_without_paths = [q for q in self.distance_results if q['min_distance'] is None]
        
        with open(no_paths_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("QUESTIONS WITHOUT PATHS ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f" SUMMARY:\n")
            f.write(f" • Total questions without paths: {len(questions_without_paths)}\n\n")
            
            # Categorize reasons
            gold_missing = []
            retrieved_missing = []
            both_present_no_path = []
            
            for q in questions_without_paths:
                if q['gold_docs_in_graph'] == 0:
                    gold_missing.append(q)
                elif q['retrieved_docs_in_graph'] == 0:
                    retrieved_missing.append(q)
                else:
                    both_present_no_path.append(q)
            
            f.write(f" BREAKDOWN BY REASON:\n")
            f.write(f" • Gold docs not in graph: {len(gold_missing)} questions\n")
            f.write(f" • Retrieved docs not in graph: {len(retrieved_missing)} questions\n")
            f.write(f" • Both present but no path: {len(both_present_no_path)} questions\n\n")
            
            # Details for each category
            f.write(f" DETAILED ANALYSIS:\n\n")
            
            f.write(f"1. QUESTIONS WITH GOLD DOCS NOT IN GRAPH ({len(gold_missing)}):\n")
            for i, q in enumerate(gold_missing[:10], 1): # Show first 10
                f.write(f" {i}. Question ID: {q['question_id']}\n")
                f.write(f" Question: {q['question']}\n")
                f.write(f" Gold docs not in graph: {q['gold_docs_not_in_graph']}\n\n")
            if len(gold_missing) > 10:
                f.write(f" ... and {len(gold_missing) - 10} more questions\n\n")
            
            f.write(f"2. QUESTIONS WITH RETRIEVED DOCS NOT IN GRAPH ({len(retrieved_missing)}):\n")
            for i, q in enumerate(retrieved_missing[:10], 1): # Show first 10
                f.write(f" {i}. Question ID: {q['question_id']}\n")
                f.write(f" Question: {q['question']}\n")
                f.write(f" Retrieved docs not in graph: {len(q['retrieved_docs_not_in_graph'])}\n\n")
            if len(retrieved_missing) > 10:
                f.write(f" ... and {len(retrieved_missing) - 10} more questions\n\n")
            
            f.write(f"3. QUESTIONS WITH BOTH PRESENT BUT NO PATH ({len(both_present_no_path)}):\n")
            for i, q in enumerate(both_present_no_path[:10], 1): # Show first 10
                f.write(f" {i}. Question ID: {q['question_id']}\n")
                f.write(f" Question: {q['question']}\n")
                f.write(f" Gold docs in graph: {q['gold_docs_in_graph']}\n")
                f.write(f" Retrieved docs in graph: {q['retrieved_docs_in_graph']}\n\n")
            if len(both_present_no_path) > 10:
                f.write(f" ... and {len(both_present_no_path) - 10} more questions\n\n")
        
        # Save detailed CSV
        no_paths_csv = output_path / "questions_without_paths_detailed.csv"
        if questions_without_paths:
            no_paths_df = pd.DataFrame(questions_without_paths)
            no_paths_df.to_csv(no_paths_csv, index=False)
            
        logger.info(f"Saved questions without paths analysis: {no_paths_file}")
    
    def _save_hop_neighbor_analysis(self, output_path: Path):
        """Save neighbor count analysis for 1-hop and 2-hop connections (Question 3)."""
        hop_file = output_path / "hop_neighbor_analysis.txt"
        
        # Collect all hop analysis data
        all_hop_data = []
        for q in self.distance_results:
            if q['hop_analysis']:
                for hop in q['hop_analysis']:
                    hop['question_id'] = q['question_id']
                    all_hop_data.append(hop)
        
        # Separate by distance
        one_hop_data = [h for h in all_hop_data if h['distance'] == 1]
        two_hop_data = [h for h in all_hop_data if h['distance'] == 2]
        
        with open(hop_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HOP NEIGHBOR ANALYSIS (1-HOP AND 2-HOP CONNECTIONS)\n")
            f.write("="*80 + "\n\n")
            
            # 1-hop analysis
            f.write(f" 1-HOP CONNECTIONS ANALYSIS:\n")
            f.write(f" • Total 1-hop connections: {len(one_hop_data)}\n")
            if one_hop_data:
                gold_neighbors = [h['source_neighbor_count'] for h in one_hop_data]
                retrieved_neighbors = [h['target_neighbor_count'] for h in one_hop_data]
                
                f.write(f" • Gold doc neighbors - Min: {min(gold_neighbors)}, Max: {max(gold_neighbors)}, Avg: {np.mean(gold_neighbors):.1f}\n")
                f.write(f" • Retrieved doc neighbors - Min: {min(retrieved_neighbors)}, Max: {max(retrieved_neighbors)}, Avg: {np.mean(retrieved_neighbors):.1f}\n\n")
                
                # Sample details
                f.write(f" Sample 1-hop connections:\n")
                for i, hop in enumerate(one_hop_data[:10], 1):
                    f.write(f" {i}. Gold: {hop['source']} ({hop['source_neighbor_count']} neighbors)\n")
                    f.write(f" → Retrieved: {hop['target']} ({hop['target_neighbor_count']} neighbors)\n")
                    f.write(f" Question ID: {hop['question_id']}\n\n")
            
            # 2-hop analysis
            f.write(f" 2-HOP CONNECTIONS ANALYSIS:\n")
            f.write(f" • Total 2-hop connections: {len(two_hop_data)}\n")
            if two_hop_data:
                gold_neighbors_2hop = [h['source_neighbor_count'] for h in two_hop_data]
                retrieved_neighbors_2hop = [h['target_neighbor_count'] for h in two_hop_data]
                
                f.write(f" • Gold doc neighbors - Min: {min(gold_neighbors_2hop)}, Max: {max(gold_neighbors_2hop)}, Avg: {np.mean(gold_neighbors_2hop):.1f}\n")
                f.write(f" • Retrieved doc neighbors - Min: {min(retrieved_neighbors_2hop)}, Max: {max(retrieved_neighbors_2hop)}, Avg: {np.mean(retrieved_neighbors_2hop):.1f}\n\n")
                
                # Sample details
                f.write(f" Sample 2-hop connections:\n")
                for i, hop in enumerate(two_hop_data[:10], 1):
                    f.write(f" {i}. Gold: {hop['source']} ({hop['source_neighbor_count']} neighbors)\n")
                    f.write(f" → Retrieved: {hop['target']} ({hop['target_neighbor_count']} neighbors)\n")
                    f.write(f" Question ID: {hop['question_id']}\n\n")
        
        # Save detailed CSV for programmatic analysis
        hop_csv = output_path / "hop_neighbor_analysis_detailed.csv"
        if all_hop_data:
            hop_df = pd.DataFrame(all_hop_data)
            hop_df.to_csv(hop_csv, index=False)
        
        logger.info(f"Saved hop neighbor analysis: {hop_file}")

def main():
    """Main execution function."""
    # File paths
    edges_file = "output/analysis_cache/knowledge_graph/unique_edges_20250731_173102.json"
    csv_file = "output/dense_sparse_average_results_filtered.csv"
    output_dir = "output/graph_distance_analysis"
    
    # Check if files exist
    if not Path(edges_file).exists():
        logger.error(f"Edge data file not found: {edges_file}")
        return
    
    if not Path(csv_file).exists():
        logger.error(f"CSV file not found: {csv_file}")
        return
    
    # Initialize analyzer
    analyzer = GraphDistanceAnalyzer()
    
    # Step 1: Load edge data
    logger.info(" Step 1: Loading edge data...")
    if not analyzer.load_edge_data(edges_file):
        logger.error("Failed to load edge data")
        return
    
    # Step 2: Build graph
    logger.info(" Step 2: Building knowledge graph...")
    if not analyzer.build_graph():
        logger.error("Failed to build graph")
        return
    
    # Step 3: Analyze questions
    logger.info(" Step 3: Analyzing distance for each question...")
    if not analyzer.analyze_csv_questions(csv_file):
        logger.error("Failed to analyze questions")
        return
    
    # Step 4: Generate statistics
    logger.info(" Step 4: Generating comprehensive statistics...")
    stats = analyzer.generate_statistics()
    
    if stats:
        # Quick preview of key statistics
        logger.info(f"\n QUICK STATISTICS PREVIEW:")
        logger.info(f" • Questions with paths: {stats['questions_with_paths']:,}/{stats['total_questions']:,} ({stats['coverage_percentage']:.1f}%)")
        if 'min_distance_stats' in stats:
            dist_stats = stats['min_distance_stats']
            logger.info(f" • Average minimum distance: {dist_stats['mean']:.2f} hops")
            logger.info(f" • Median minimum distance: {dist_stats['median']:.0f} hops")
            logger.info(f" • Distance range: {dist_stats['min']:.0f} - {dist_stats['max']:.0f} hops")
    
    # Step 5: Save results
    logger.info(" Step 5: Saving results...")
    if not analyzer.save_results(output_dir):
        logger.error("Failed to save results")
        return
    
    logger.info(f"\n Analysis completed successfully!")
    logger.info(f" Results saved to: {output_dir}")

if __name__ == "__main__":
    main()