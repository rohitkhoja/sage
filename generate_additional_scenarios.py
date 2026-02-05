#!/usr/bin/env python3
"""
Generate Additional CSV vs Graph Scenarios

This script generates the additional CSV vs Graph comparisons requested:
- CSV1 + Graph1 vs CSV2
- CSV2 + Graph3 vs CSV5  
- CSV5 + Graph5 vs CSV10
- CSV50 + Graph50 vs CSV100
"""

import pandas as pd
import numpy as np
import json
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from dataclasses import dataclass, asdict
import ast
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

class SimpleCSVGraphAnalyzer:
    """Simple analyzer that doesn't use GPU-intensive embedding calculations"""
    
    def __init__(self,
                 csv_file: str = "/shared/khoja/CogComp/output/dense_sparse_average_results (1).csv",
                 edges_file: str = "/shared/khoja/CogComp/output/analysis_cache/knowledge_graph/unique_edges_20250731_173102.json",
                 output_dir: str = "/shared/khoja/CogComp/output/graph_enhancement_cache/csv_vs_graph_analysis"):
        
        self.csv_file = csv_file
        self.edges_file = edges_file
        self.output_dir = Path(output_dir)
        
        # Data containers
        self.df = None
        self.graph = None
        
        logger.info("Initialized SimpleCSVGraphAnalyzer")
    
    def load_data(self):
        """Load CSV and graph data"""
        logger.info("Loading CSV data...")
        self.df = pd.read_csv(self.csv_file)
        logger.info(f"Loaded {len(self.df)} questions from CSV")
        
        logger.info("Loading graph data...")
        with open(self.edges_file, 'r') as f:
            edges_data = json.load(f)
        
        self.graph = nx.Graph()
        for edge in edges_data:
            node1 = edge['source_chunk_id']
            node2 = edge['target_chunk_id']
            weight = edge.get('weight', 1.0)
            self.graph.add_edge(node1, node2, weight=weight)
        
        logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def extract_gold_docs(self, gold_str: str) -> List[str]:
        """Extract gold documents from string format"""
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
    
    def get_simple_graph_neighbors(self, csv_k_retrieved: List[str], graph_k: int) -> List[str]:
        """Get graph neighbors using simple connectivity (no similarity ranking)"""
        # Collect all neighbors from retrieved chunks
        all_neighbors = set()
        
        for chunk_id in csv_k_retrieved:
            if chunk_id in self.graph:
                neighbors = list(self.graph.neighbors(chunk_id))
                # Exclude retrieved chunks from neighbors
                neighbors = [n for n in neighbors if n not in csv_k_retrieved]
                all_neighbors.update(neighbors)
        
        if not all_neighbors:
            return []
        
        # For simplicity, just return the first graph_k neighbors (no similarity ranking)
        # This will help us understand if the issue is connectivity vs similarity
        return list(all_neighbors)[:graph_k]
    
    def analyze_csv_vs_graph_comparison(self, row: pd.Series, csv_k: int, graph_k: int) -> CSVvsGraphComparison:
        """Compare CSV+Graph vs Extended CSV approach (simplified version)"""
        
        question_id = row['question_id']
        question_text = row['question']
        gold_docs = self.extract_gold_docs(row['gold_docs'])
        
        # Get first csv_k results
        csv_k_retrieved = self.get_csv_retrieved_docs(row, csv_k)
        
        # Get extended CSV results (total csv_k + graph_k)
        extended_csv_retrieved = self.get_csv_retrieved_docs(row, csv_k + graph_k)
        
        # Get graph neighbors (simplified - no similarity ranking)
        graph_neighbors = self.get_simple_graph_neighbors(csv_k_retrieved, graph_k)
        
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
        """Run CSV vs Graph comparison analysis for additional scenarios"""
        
        logger.info("Starting additional CSV vs Graph comparison analysis...")
        
        # Define additional comparison configurations
        comparisons = [
            {"csv_k": 1, "graph_k": 1, "name": "CSV1_vs_Graph1_vs_CSV2"},
            {"csv_k": 2, "graph_k": 3, "name": "CSV2_vs_Graph3_vs_CSV5"},
            {"csv_k": 5, "graph_k": 5, "name": "CSV5_vs_Graph5_vs_CSV10"},
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
        
        analysis_dir = self.output_dir
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
        
        # Update overall comparison summary
        overall_summary_file = analysis_dir / "overall_summary.json"
        
        # Load existing summary if it exists
        overall_summary = {"analysis_type": "CSV vs Graph Comparison", "comparisons": {}}
        if overall_summary_file.exists():
            with open(overall_summary_file, 'r') as f:
                overall_summary = json.load(f)
        
        # Add new comparisons
        for name, data in all_results.items():
            overall_summary["comparisons"][name] = {
                "config": data["config"],
                "csv_plus_graph_accuracy": data["csv_plus_graph_accuracy"],
                "extended_csv_accuracy": data["extended_csv_accuracy"],
                "accuracy_difference": data["accuracy_difference"],
                "scenario_counts": data["scenario_counts"]
            }
        
        with open(overall_summary_file, 'w') as f:
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

def main():
    """Main execution function"""
    
    logger.info("Starting Additional CSV vs Graph Scenarios Generation...")
    
    # Initialize analyzer
    analyzer = SimpleCSVGraphAnalyzer()
    
    # Load data
    analyzer.load_data()
    
    # Run analysis
    results = analyzer.run_csv_vs_graph_analysis()
    
    # Print summary
    print("\n" + "="*80)
    print("ADDITIONAL CSV vs GRAPH COMPARISON SUMMARY")
    print("="*80)
    
    for name, data in results.items():
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
    
    logger.info("Additional scenarios generation complete!")
    return results

if __name__ == "__main__":
    main()