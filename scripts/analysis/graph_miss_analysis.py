#!/usr/bin/env python3
"""
Graph Miss Analysis Tool

This script analyzes why graph neighbors miss certain gold documents that CSV retrieval finds.
It performs comprehensive analysis to understand the root causes of these misses.

Key Analyses:
1. Find common gold documents that CSV wins across all scenarios
2. Check if missed gold docs are present in the graph as nodes
3. Determine if misses are due to similarity-based pruning or lack of connectivity
4. Analyze 1-hop neighbors of retrieved CSV documents
5. Generate additional CSV vs Graph comparisons
"""

import pandas as pd
import numpy as np
import json
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MissAnalysisResult:
    """Analysis result for a missed gold document"""
    gold_doc: str
    question_id: str
    question_text: str
    scenarios_missed: List[str]
    
    # Graph connectivity analysis
    gold_in_graph: bool
    csv_nodes_with_gold_as_neighbor: List[str] # Which CSV nodes have this gold as 1-hop neighbor
    csv_nodes_without_gold_neighbor: List[str] # Which CSV nodes don't have this gold as neighbor
    
    # Similarity analysis (if gold is connected but not selected)
    gold_similarity_score: float # Similarity score if gold was in candidate neighbors
    gold_rank_among_neighbors: int # Rank of gold among all neighbors if it was considered
    
    # Pruning analysis
    miss_reason: str # "not_in_graph", "not_connected", "similarity_pruning", "rank_pruning"

@dataclass
class ScenarioComparison:
    """Comparison analysis for a specific scenario"""
    scenario_name: str
    csv_k: int
    graph_k: int
    extended_csv_k: int
    
    total_questions: int
    csv_wins_count: int
    csv_wins_questions: List[Dict[str, Any]]
    
    # Gold document analysis
    unique_gold_docs_missed: Set[str]
    common_gold_docs_missed: Set[str] # Common with other scenarios

class GraphMissAnalyzer:
    """Main analyzer for understanding graph misses"""
    
    def __init__(self,
                 csv_vs_graph_dir: str = "/shared/khoja/CogComp/output/graph_enhancement_cache/csv_vs_graph_analysis",
                 edges_file: str = "/shared/khoja/CogComp/output/analysis_cache/knowledge_graph/unique_edges_20250731_173102.json",
                 csv_file: str = "/shared/khoja/CogComp/output/dense_sparse_average_results (1).csv",
                 output_dir: str = "/shared/khoja/CogComp/output/graph_miss_analysis"):
        
        self.csv_vs_graph_dir = Path(csv_vs_graph_dir)
        self.edges_file = edges_file
        self.csv_file = csv_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.graph = None
        self.df = None
        self.scenario_analyses = {}
        
        logger.info("Initialized GraphMissAnalyzer")
    
    def load_data(self):
        """Load all required data"""
        logger.info("Loading graph data...")
        self._load_graph()
        
        logger.info("Loading CSV data...")
        self._load_csv_data()
        
        logger.info("Loading existing scenario analyses...")
        self._load_existing_scenarios()
        
        logger.info("Data loading complete")
    
    def _load_graph(self):
        """Load graph structure"""
        with open(self.edges_file, 'r') as f:
            edges_data = json.load(f)
        
        self.graph = nx.Graph()
        
        for edge in edges_data:
            node1 = edge['source_chunk_id']
            node2 = edge['target_chunk_id']
            weight = edge.get('weight', 1.0)
            self.graph.add_edge(node1, node2, weight=weight)
        
        logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def _load_csv_data(self):
        """Load CSV retrieval results"""
        self.df = pd.read_csv(self.csv_file)
        logger.info(f"Loaded {len(self.df)} questions from CSV")
    
    def _load_existing_scenarios(self):
        """Load existing CSV vs Graph scenario analyses"""
        scenarios = [
            ("CSV10_vs_Graph10_vs_CSV20", 10, 10, 20),
            ("CSV20_vs_Graph30_vs_CSV50", 20, 30, 50),
            ("CSV25_vs_Graph25_vs_CSV50", 25, 25, 50),
            ("CSV30_vs_Graph20_vs_CSV50", 30, 20, 50)
        ]
        
        for scenario_name, csv_k, graph_k, extended_csv_k in scenarios:
            scenario_path = self.csv_vs_graph_dir / scenario_name
            if scenario_path.exists():
                self._load_scenario_analysis(scenario_name, csv_k, graph_k, extended_csv_k, scenario_path)
    
    def _load_scenario_analysis(self, scenario_name: str, csv_k: int, graph_k: int, extended_csv_k: int, scenario_path: Path):
        """Load analysis for a specific scenario"""
        csv_wins_file = scenario_path / "csv_wins.json"
        
        if not csv_wins_file.exists():
            logger.warning(f"csv_wins.json not found for {scenario_name}")
            return
        
        with open(csv_wins_file, 'r') as f:
            csv_wins_data = json.load(f)
        
        # Extract gold documents that were missed by graph
        unique_gold_docs = set()
        for question in csv_wins_data['questions']:
            # Gold docs found in extended CSV but not in CSV+Graph
            extended_csv_gold = set(question['analysis']['gold_found_in_extended_csv'])
            csv_k_gold = set(question['analysis']['gold_found_in_csv_k'])
            graph_gold = set(question['analysis']['gold_found_in_graph'])
            
            # Gold docs that extended CSV found but CSV+Graph missed
            missed_by_graph = extended_csv_gold - csv_k_gold - graph_gold
            unique_gold_docs.update(missed_by_graph)
        
        scenario = ScenarioComparison(
            scenario_name=scenario_name,
            csv_k=csv_k,
            graph_k=graph_k,
            extended_csv_k=extended_csv_k,
            total_questions=csv_wins_data['total_questions'],
            csv_wins_count=len(csv_wins_data['questions']),
            csv_wins_questions=csv_wins_data['questions'],
            unique_gold_docs_missed=unique_gold_docs,
            common_gold_docs_missed=set() # Will be computed later
        )
        
        self.scenario_analyses[scenario_name] = scenario
        logger.info(f"Loaded {scenario_name}: {len(unique_gold_docs)} unique missed gold docs")
    
    def find_common_missed_golds(self) -> Set[str]:
        """Find gold documents that are commonly missed across scenarios"""
        if not self.scenario_analyses:
            return set()
        
        # Find intersection of all missed gold docs
        all_missed_sets = [scenario.unique_gold_docs_missed for scenario in self.scenario_analyses.values()]
        common_missed = set.intersection(*all_missed_sets) if all_missed_sets else set()
        
        # Update scenario analyses with common misses
        for scenario in self.scenario_analyses.values():
            scenario.common_gold_docs_missed = common_missed
        
        logger.info(f"Found {len(common_missed)} gold documents commonly missed across all scenarios")
        return common_missed
    
    def analyze_missed_gold_connectivity(self, gold_doc: str, csv_k_nodes: List[str]) -> Dict[str, Any]:
        """Analyze why a specific gold document was missed"""
        
        # Check if gold is in graph
        gold_in_graph = gold_doc in self.graph
        
        if not gold_in_graph:
            return {
                "gold_in_graph": False,
                "miss_reason": "not_in_graph",
                "csv_nodes_with_gold_as_neighbor": [],
                "csv_nodes_without_gold_neighbor": csv_k_nodes,
                "connectivity_details": "Gold document not present in the knowledge graph"
            }
        
        # Check which CSV nodes have this gold as 1-hop neighbor
        csv_nodes_with_gold_neighbor = []
        csv_nodes_without_gold_neighbor = []
        
        for csv_node in csv_k_nodes:
            if csv_node in self.graph and gold_doc in self.graph.neighbors(csv_node):
                csv_nodes_with_gold_neighbor.append(csv_node)
            else:
                csv_nodes_without_gold_neighbor.append(csv_node)
        
        # Determine miss reason
        if not csv_nodes_with_gold_neighbor:
            miss_reason = "not_connected"
            connectivity_details = "Gold document not connected to any retrieved CSV documents"
        else:
            miss_reason = "similarity_pruning" # Gold was connected but filtered out by similarity
            connectivity_details = f"Gold connected to {len(csv_nodes_with_gold_neighbor)} CSV nodes but filtered by similarity ranking"
        
        return {
            "gold_in_graph": True,
            "miss_reason": miss_reason,
            "csv_nodes_with_gold_as_neighbor": csv_nodes_with_gold_neighbor,
            "csv_nodes_without_gold_neighbor": csv_nodes_without_gold_neighbor,
            "connectivity_details": connectivity_details
        }
    
    def analyze_all_missed_golds(self) -> Dict[str, List[MissAnalysisResult]]:
        """Analyze all missed gold documents across scenarios"""
        logger.info("Starting comprehensive miss analysis...")
        
        common_missed = self.find_common_missed_golds()
        all_missed_golds = set()
        for scenario in self.scenario_analyses.values():
            all_missed_golds.update(scenario.unique_gold_docs_missed)
        
        results = {
            "common_missed": [],
            "scenario_specific": []
        }
        
        # Analyze each missed gold document
        for gold_doc in all_missed_golds:
            scenarios_missing_this_gold = []
            
            # Find which scenarios miss this gold and get question details
            for scenario_name, scenario in self.scenario_analyses.items():
                if gold_doc in scenario.unique_gold_docs_missed:
                    scenarios_missing_this_gold.append(scenario_name)
            
            # Get a representative question for analysis
            representative_question = None
            csv_k_nodes = []
            
            for scenario_name in scenarios_missing_this_gold:
                scenario = self.scenario_analyses[scenario_name]
                for question in scenario.csv_wins_questions:
                    extended_csv_gold = set(question['analysis']['gold_found_in_extended_csv'])
                    csv_k_gold = set(question['analysis']['gold_found_in_csv_k'])
                    graph_gold = set(question['analysis']['gold_found_in_graph'])
                    
                    missed_by_graph = extended_csv_gold - csv_k_gold - graph_gold
                    if gold_doc in missed_by_graph:
                        representative_question = question
                        csv_k_nodes = question['analysis']['csv_k_retrieved']
                        break
                if representative_question:
                    break
            
            if not representative_question:
                continue
            
            # Analyze connectivity
            connectivity_analysis = self.analyze_missed_gold_connectivity(gold_doc, csv_k_nodes)
            
            miss_result = MissAnalysisResult(
                gold_doc=gold_doc,
                question_id=representative_question['question_id'],
                question_text=representative_question['question_text'],
                scenarios_missed=scenarios_missing_this_gold,
                gold_in_graph=connectivity_analysis['gold_in_graph'],
                csv_nodes_with_gold_as_neighbor=connectivity_analysis['csv_nodes_with_gold_as_neighbor'],
                csv_nodes_without_gold_neighbor=connectivity_analysis['csv_nodes_without_gold_neighbor'],
                gold_similarity_score=0.0, # Would need similarity calculation
                gold_rank_among_neighbors=0, # Would need neighbor ranking
                miss_reason=connectivity_analysis['miss_reason']
            )
            
            if gold_doc in common_missed:
                results["common_missed"].append(miss_result)
            else:
                results["scenario_specific"].append(miss_result)
        
        logger.info(f"Analyzed {len(results['common_missed'])} common missed and {len(results['scenario_specific'])} scenario-specific missed gold docs")
        return results
    
    def generate_additional_scenarios(self):
        """Note: Additional scenarios should be generated separately to avoid GPU memory issues"""
        logger.info("Skipping additional scenario generation to avoid GPU memory issues.")
        logger.info("Run the updated graph_enhanced_accuracy_analyzer.py separately to generate additional scenarios.")
        return {}
    
    def save_analysis_results(self, miss_analysis_results: Dict[str, List[MissAnalysisResult]]):
        """Save all analysis results to files"""
        logger.info("Saving analysis results...")
        
        # Save miss analysis results
        miss_analysis_file = self.output_dir / "miss_analysis_results.json"
        
        # Convert to serializable format
        serializable_results = {}
        for category, results in miss_analysis_results.items():
            serializable_results[category] = [asdict(result) for result in results]
        
        with open(miss_analysis_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save summary statistics
        summary = {
            "analysis_overview": {
                "total_scenarios_analyzed": len(self.scenario_analyses),
                "common_missed_golds": len(miss_analysis_results["common_missed"]),
                "scenario_specific_missed": len(miss_analysis_results["scenario_specific"])
            },
            "miss_reasons_breakdown": {
                "not_in_graph": 0,
                "not_connected": 0,
                "similarity_pruning": 0
            },
            "scenarios_summary": {}
        }
        
        # Count miss reasons
        all_results = miss_analysis_results["common_missed"] + miss_analysis_results["scenario_specific"]
        for result in all_results:
            summary["miss_reasons_breakdown"][result.miss_reason] += 1
        
        # Scenario summaries
        for scenario_name, scenario in self.scenario_analyses.items():
            summary["scenarios_summary"][scenario_name] = {
                "csv_k": scenario.csv_k,
                "graph_k": scenario.graph_k,
                "extended_csv_k": scenario.extended_csv_k,
                "csv_wins_count": scenario.csv_wins_count,
                "unique_missed_golds": len(scenario.unique_gold_docs_missed),
                "common_missed_golds": len(scenario.common_gold_docs_missed)
            }
        
        summary_file = self.output_dir / "analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate detailed report
        self._generate_detailed_report(miss_analysis_results)
        
        logger.info(f"Analysis results saved to {self.output_dir}")
    
    def _generate_detailed_report(self, miss_analysis_results: Dict[str, List[MissAnalysisResult]]):
        """Generate a human-readable detailed report"""
        report_file = self.output_dir / "detailed_miss_analysis_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("GRAPH MISS ANALYSIS DETAILED REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Executive Summary
            f.write(" EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            common_missed = miss_analysis_results["common_missed"]
            scenario_specific = miss_analysis_results["scenario_specific"]
            
            f.write(f"• Total gold documents missed by graph: {len(common_missed) + len(scenario_specific)}\n")
            f.write(f"• Commonly missed across all scenarios: {len(common_missed)}\n")
            f.write(f"• Scenario-specific misses: {len(scenario_specific)}\n\n")
            
            # Miss Reasons Analysis
            all_results = common_missed + scenario_specific
            miss_reasons = Counter(result.miss_reason for result in all_results)
            
            f.write(" MISS REASONS BREAKDOWN\n")
            f.write("-" * 40 + "\n")
            for reason, count in miss_reasons.items():
                percentage = count / len(all_results) * 100
                f.write(f"• {reason}: {count} ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Common Missed Gold Documents
            if common_missed:
                f.write(" COMMONLY MISSED GOLD DOCUMENTS (All Scenarios)\n")
                f.write("-" * 60 + "\n")
                for i, result in enumerate(common_missed[:10], 1): # Show top 10
                    f.write(f"{i}. Gold Doc: {result.gold_doc}\n")
                    f.write(f" Question: {result.question_text[:100]}...\n")
                    f.write(f" Miss Reason: {result.miss_reason}\n")
                    f.write(f" In Graph: {result.gold_in_graph}\n")
                    f.write(f" Connected CSV Nodes: {len(result.csv_nodes_with_gold_as_neighbor)}\n\n")
                
                if len(common_missed) > 10:
                    f.write(f" ... and {len(common_missed) - 10} more commonly missed gold docs\n\n")
            
            # Scenario-Specific Analysis
            f.write(" SCENARIO-SPECIFIC ANALYSIS\n")
            f.write("-" * 40 + "\n")
            for scenario_name, scenario in self.scenario_analyses.items():
                f.write(f"\n{scenario_name}:\n")
                f.write(f" Configuration: CSV k={scenario.csv_k}, Graph k={scenario.graph_k}, Extended CSV k={scenario.extended_csv_k}\n")
                f.write(f" CSV Wins: {scenario.csv_wins_count} questions\n")
                f.write(f" Unique Missed Golds: {len(scenario.unique_gold_docs_missed)}\n")
                f.write(f" Common Missed Golds: {len(scenario.common_gold_docs_missed)}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        logger.info("Starting complete graph miss analysis...")
        
        # Load data
        self.load_data()
        
        # Analyze missed gold documents
        miss_analysis_results = self.analyze_all_missed_golds()
        
        # Generate additional scenarios
        additional_scenarios = self.generate_additional_scenarios()
        
        # Save results
        self.save_analysis_results(miss_analysis_results)
        
        # Print summary
        self._print_summary(miss_analysis_results)
        
        logger.info("Complete analysis finished!")
        return miss_analysis_results, additional_scenarios
    
    def _print_summary(self, miss_analysis_results: Dict[str, List[MissAnalysisResult]]):
        """Print a summary of the analysis"""
        print("\n" + "=" * 80)
        print("GRAPH MISS ANALYSIS SUMMARY")
        print("=" * 80)
        
        common_missed = miss_analysis_results["common_missed"]
        scenario_specific = miss_analysis_results["scenario_specific"]
        all_results = common_missed + scenario_specific
        
        print(f"\n OVERALL STATISTICS:")
        print(f" • Total scenarios analyzed: {len(self.scenario_analyses)}")
        print(f" • Total missed gold documents: {len(all_results)}")
        print(f" • Commonly missed (all scenarios): {len(common_missed)}")
        print(f" • Scenario-specific misses: {len(scenario_specific)}")
        
        print(f"\n MISS REASONS:")
        miss_reasons = Counter(result.miss_reason for result in all_results)
        for reason, count in miss_reasons.items():
            percentage = count / len(all_results) * 100
            print(f" • {reason}: {count} ({percentage:.1f}%)")
        
        print(f"\n SCENARIO BREAKDOWN:")
        for scenario_name, scenario in self.scenario_analyses.items():
            print(f" • {scenario_name}: {len(scenario.unique_gold_docs_missed)} missed golds")

def main():
    """Main execution function"""
    analyzer = GraphMissAnalyzer()
    miss_results, additional_scenarios = analyzer.run_complete_analysis()
    
    print(f"\n Analysis complete! Results saved to: {analyzer.output_dir}")
    return analyzer, miss_results, additional_scenarios

if __name__ == "__main__":
    main()