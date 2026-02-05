#!/usr/bin/env python3
"""
Clean Graph Accuracy Analyzer

This script performs the REAL comparison between CSV+Graph vs Extended CSV
after removing the 241 questions with similarity pruning issues.

Key difference from the previous incorrect approach:
- CSV+Graph: Uses actual graph neighbors (not just extended CSV)
- Extended CSV: Uses pure CSV retrieval at extended k-value
- Excludes 241 questions with similarity pruning issues
"""

import pandas as pd
import numpy as np
import json
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
import logging
from dataclasses import dataclass
import ast
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CleanComparisonResult:
    """Result for a single question comparison"""
    question_id: str
    gold_docs: List[str]
    csv_k_retrieved: List[str]
    graph_neighbors: List[str]
    extended_csv_retrieved: List[str]
    
    # Results
    csv_plus_graph_found_gold: bool
    extended_csv_found_gold: bool
    
    # Gold document locations
    gold_found_in_csv_k: List[str]
    gold_found_in_graph: List[str]
    gold_found_in_extended_csv: List[str]

class CleanGraphAccuracyAnalyzer:
    """Analyzer that performs the REAL CSV+Graph vs Extended CSV comparison"""
    
    def __init__(self,
                 csv_file: str = "/shared/khoja/CogComp/output/dense_sparse_average_results (1).csv",
                 edges_file: str = "/shared/khoja/CogComp/output/analysis_cache/knowledge_graph/unique_edges_20250731_173102.json",
                 similarity_analysis_file: str = "/shared/khoja/CogComp/output/similarity_pruning_analysis/similarity_pruning_comprehensive_report.json",
                 output_dir: str = "/shared/khoja/CogComp/output/clean_graph_accuracy_analysis"):
        
        self.csv_file = csv_file
        self.edges_file = edges_file
        self.similarity_analysis_file = similarity_analysis_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.df = None
        self.graph = None
        self.similarity_pruning_questions = set()
        
        # Define scenario configurations
        self.scenario_configs = [
            {"csv_k": 1, "graph_k": 1, "extended_csv_k": 2, "name": "CSV1+Graph1"},
            {"csv_k": 2, "graph_k": 3, "extended_csv_k": 5, "name": "CSV2+Graph3"},
            {"csv_k": 5, "graph_k": 5, "extended_csv_k": 10, "name": "CSV5+Graph5"},
            {"csv_k": 10, "graph_k": 10, "extended_csv_k": 20, "name": "CSV10+Graph10"},
            {"csv_k": 20, "graph_k": 30, "extended_csv_k": 50, "name": "CSV20+Graph30"},
            {"csv_k": 25, "graph_k": 25, "extended_csv_k": 50, "name": "CSV25+Graph25"},
            {"csv_k": 30, "graph_k": 20, "extended_csv_k": 50, "name": "CSV30+Graph20"},
            {"csv_k": 50, "graph_k": 50, "extended_csv_k": 100, "name": "CSV50+Graph50"}
        ]
        
        logger.info("Initialized CleanGraphAccuracyAnalyzer")
    
    def load_all_data(self):
        """Load all required data"""
        logger.info("Loading all data...")
        
        self._load_csv_data()
        self._load_graph_data()
        self._load_similarity_pruning_questions()
        
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
    
    def _load_similarity_pruning_questions(self):
        """Load questions with similarity pruning issues"""
        logger.info("Loading similarity pruning questions...")
        
        with open(self.similarity_analysis_file, 'r') as f:
            similarity_data = json.load(f)
        
        # Extract all unique question IDs with similarity pruning issues
        for scenario_data in similarity_data['detailed_questions_by_scenario'].values():
            for question in scenario_data:
                self.similarity_pruning_questions.add(question['question_id'])
        
        logger.info(f"Found {len(self.similarity_pruning_questions)} unique questions with similarity pruning issues")
    
    def extract_gold_docs(self, gold_str: str) -> List[str]:
        """Extract gold documents from string format"""
        if pd.isna(gold_str) or gold_str == '':
            return []
        
        gold_str = str(gold_str).strip()
        
        try:
            if gold_str.startswith('[') and gold_str.endswith(']'):
                gold_docs = ast.literal_eval(gold_str)
                return gold_docs if isinstance(gold_docs, list) else [gold_docs]
        except (ValueError, SyntaxError):
            pass
        
        try:
            if gold_str.startswith('"') and gold_str.endswith('"'):
                gold_str = gold_str[1:-1]
            if gold_str.startswith('[') and gold_str.endswith(']'):
                gold_docs = ast.literal_eval(gold_str)
                return gold_docs if isinstance(gold_docs, list) else [gold_docs]
        except (ValueError, SyntaxError):
            pass
        
        match = re.search(r'\[(.*?)\]', gold_str)
        if match:
            items_str = match.group(1)
            items = [item.strip().strip("'\"") for item in items_str.split(',')]
            return [item for item in items if item]
        
        return [gold_str] if gold_str else []
    
    def get_csv_retrieved_docs(self, row: pd.Series, k: int) -> List[str]:
        """Get top-k retrieved documents from CSV row"""
        retrieved_docs = []
        for i in range(1, min(k + 1, 101)):
            col_name = f"{i}st" if i == 1 else f"{i}nd" if i == 2 else f"{i}rd" if i == 3 else f"{i}th"
            if col_name in row and pd.notna(row[col_name]):
                retrieved_docs.append(row[col_name])
        return retrieved_docs[:k]
    
    def get_graph_neighbors(self, retrieved_chunks: List[str], top_k: int) -> List[str]:
        """Get top-k graph neighbors from retrieved chunks (simplified - no similarity ranking)"""
        
        # Collect all neighbors from retrieved chunks
        all_neighbors = set()
        
        for chunk_id in retrieved_chunks:
            if chunk_id in self.graph:
                neighbors = list(self.graph.neighbors(chunk_id))
                # Exclude retrieved chunks from neighbors
                neighbors = [n for n in neighbors if n not in retrieved_chunks]
                all_neighbors.update(neighbors)
        
        # For simplicity, just return first top_k neighbors (no similarity ranking)
        # This gives us a baseline comparison
        return list(all_neighbors)[:top_k]
    
    def analyze_single_question(self, row: pd.Series, csv_k: int, graph_k: int, extended_csv_k: int, exclude_pruning: bool = True) -> CleanComparisonResult:
        """Analyze a single question for CSV+Graph vs Extended CSV comparison"""
        
        question_id = row['question_id']
        
        # Skip questions with similarity pruning issues if requested
        if exclude_pruning and question_id in self.similarity_pruning_questions:
            return None
        
        gold_docs = self.extract_gold_docs(row['gold_docs'])
        if not gold_docs:  # Skip questions without gold docs
            return None
        
        # Get CSV k retrieved documents
        csv_k_retrieved = self.get_csv_retrieved_docs(row, csv_k)
        
        # Get graph neighbors from CSV k retrieved documents
        graph_neighbors = self.get_graph_neighbors(csv_k_retrieved, graph_k)
        
        # Get extended CSV retrieved documents (for fair comparison)
        extended_csv_retrieved = self.get_csv_retrieved_docs(row, extended_csv_k)
        
        # Analyze gold document presence
        gold_found_in_csv_k = [doc for doc in gold_docs if doc in csv_k_retrieved]
        gold_found_in_graph = [doc for doc in gold_docs if doc in graph_neighbors]
        gold_found_in_extended_csv = [doc for doc in gold_docs if doc in extended_csv_retrieved]
        
        # Determine results
        csv_plus_graph_candidates = csv_k_retrieved + graph_neighbors
        csv_plus_graph_found_gold = any(doc in csv_plus_graph_candidates for doc in gold_docs)
        extended_csv_found_gold = len(gold_found_in_extended_csv) > 0
        
        return CleanComparisonResult(
            question_id=question_id,
            gold_docs=gold_docs,
            csv_k_retrieved=csv_k_retrieved,
            graph_neighbors=graph_neighbors,
            extended_csv_retrieved=extended_csv_retrieved,
            csv_plus_graph_found_gold=csv_plus_graph_found_gold,
            extended_csv_found_gold=extended_csv_found_gold,
            gold_found_in_csv_k=gold_found_in_csv_k,
            gold_found_in_graph=gold_found_in_graph,
            gold_found_in_extended_csv=gold_found_in_extended_csv
        )
    
    def calculate_accuracy_for_scenario(self, csv_k: int, graph_k: int, extended_csv_k: int, exclude_pruning: bool = True) -> Tuple[float, float, int]:
        """Calculate CSV+Graph vs Extended CSV accuracy for a specific scenario"""
        
        csv_plus_graph_correct = 0
        extended_csv_correct = 0
        total_questions = 0
        
        for _, row in self.df.iterrows():
            result = self.analyze_single_question(row, csv_k, graph_k, extended_csv_k, exclude_pruning)
            
            if result is None:  # Skipped question
                continue
            
            total_questions += 1
            
            if result.csv_plus_graph_found_gold:
                csv_plus_graph_correct += 1
            
            if result.extended_csv_found_gold:
                extended_csv_correct += 1
        
        csv_plus_graph_accuracy = csv_plus_graph_correct / total_questions if total_questions > 0 else 0.0
        extended_csv_accuracy = extended_csv_correct / total_questions if total_questions > 0 else 0.0
        
        return csv_plus_graph_accuracy, extended_csv_accuracy, total_questions
    
    def run_analysis(self):
        """Run complete clean graph accuracy analysis"""
        logger.info("Starting clean graph accuracy analysis...")
        
        results = {
            'with_pruning_issues': [],
            'without_pruning_issues': [],
            'comparison_analysis': []
        }
        
        for config in self.scenario_configs:
            csv_k = config['csv_k']
            graph_k = config['graph_k']
            extended_csv_k = config['extended_csv_k']
            name = config['name']
            
            # Calculate with pruning issues (all questions)
            csv_graph_acc_with, ext_csv_acc_with, total_with = self.calculate_accuracy_for_scenario(
                csv_k, graph_k, extended_csv_k, exclude_pruning=False
            )
            
            # Calculate without pruning issues (excluding problematic questions)
            csv_graph_acc_without, ext_csv_acc_without, total_without = self.calculate_accuracy_for_scenario(
                csv_k, graph_k, extended_csv_k, exclude_pruning=True
            )
            
            results['with_pruning_issues'].append({
                'scenario': name,
                'csv_k': csv_k,
                'graph_k': graph_k,
                'extended_csv_k': extended_csv_k,
                'csv_plus_graph_accuracy': csv_graph_acc_with,
                'extended_csv_accuracy': ext_csv_acc_with,
                'total_questions': total_with
            })
            
            results['without_pruning_issues'].append({
                'scenario': name,
                'csv_k': csv_k,
                'graph_k': graph_k,
                'extended_csv_k': extended_csv_k,
                'csv_plus_graph_accuracy': csv_graph_acc_without,
                'extended_csv_accuracy': ext_csv_acc_without,
                'total_questions': total_without
            })
            
            # Calculate advantage
            advantage_before = csv_graph_acc_with - ext_csv_acc_with
            advantage_after = csv_graph_acc_without - ext_csv_acc_without
            
            results['comparison_analysis'].append({
                'scenario': name,
                'csv_k': csv_k,
                'graph_k': graph_k,
                'advantage_before': advantage_before,
                'advantage_after': advantage_after,
                'advantage_improvement': advantage_after - advantage_before,
                'questions_excluded': total_with - total_without
            })
            
            logger.info(f"{name}: CSV+Graph {csv_graph_acc_with:.3f}→{csv_graph_acc_without:.3f}, "
                       f"Extended CSV {ext_csv_acc_with:.3f}→{ext_csv_acc_without:.3f}, "
                       f"Advantage: {advantage_before:.3f}→{advantage_after:.3f}, "
                       f"Excluded {total_with - total_without} questions")
        
        # Save results
        self._save_results(results)
        self._print_summary(results)
        
        logger.info("Clean graph accuracy analysis complete!")
        return results
    
    def _save_results(self, results: Dict[str, List[Dict]]):
        """Save results to JSON file"""
        results_file = self.output_dir / "clean_graph_accuracy_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary report
        self._create_summary_report(results)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def _create_summary_report(self, results: Dict[str, List[Dict]]):
        """Create human-readable summary report"""
        report_file = self.output_dir / "clean_graph_accuracy_summary.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CLEAN GRAPH ACCURACY ANALYSIS REPORT\n")
            f.write("(Real CSV+Graph vs Extended CSV Comparison)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"📊 QUESTIONS EXCLUDED: {len(self.similarity_pruning_questions)} unique questions\n\n")
            
            f.write("📋 REAL COMPARISON: CSV+Graph vs Extended CSV\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Scenario':<20} {'CSV+Graph Before':<16} {'CSV+Graph After':<16} {'ExtCSV Before':<14} {'ExtCSV After':<14} {'Advantage After':<12}\n")
            f.write("-" * 110 + "\n")
            
            for i, scenario in enumerate(results['without_pruning_issues']):
                before = results['with_pruning_issues'][i]
                after = results['without_pruning_issues'][i]
                comparison = results['comparison_analysis'][i]
                
                f.write(f"{scenario['scenario']:<20} "
                       f"{before['csv_plus_graph_accuracy']:<16.3f} "
                       f"{after['csv_plus_graph_accuracy']:<16.3f} "
                       f"{before['extended_csv_accuracy']:<14.3f} "
                       f"{after['extended_csv_accuracy']:<14.3f} "
                       f"{comparison['advantage_after']:+<12.3f}\n")
            
            f.write("\n📋 INTERPRETATION:\n")
            f.write("• CSV+Graph: CSV k-value + actual graph neighbors\n")
            f.write("• Extended CSV: Pure CSV retrieval at k+graph_k value\n")
            f.write("• Advantage After: Positive = Graph helps, Negative = Pure CSV better\n")
            f.write("• This shows the REAL impact of graph neighbors vs just more CSV retrieval\n")
            
            f.write("\n" + "=" * 80 + "\n")
    
    def _print_summary(self, results: Dict[str, List[Dict]]):
        """Print summary to console"""
        print("\n" + "=" * 80)
        print("CLEAN GRAPH ACCURACY ANALYSIS SUMMARY")
        print("=" * 80)
        
        print(f"\n📊 EXCLUDED QUESTIONS: {len(self.similarity_pruning_questions)} questions with similarity pruning issues")
        
        print(f"\n📋 KEY FINDINGS (After Removing Pruning Issues):")
        
        # Calculate overall advantages
        advantages_after = [item['advantage_after'] for item in results['comparison_analysis']]
        avg_advantage = np.mean(advantages_after)
        
        positive_advantages = [adv for adv in advantages_after if adv > 0]
        negative_advantages = [adv for adv in advantages_after if adv < 0]
        
        print(f"  • Average graph advantage: {avg_advantage:.3f} ({avg_advantage*100:.1f}%)")
        print(f"  • Scenarios where graph helps: {len(positive_advantages)}/{len(advantages_after)}")
        print(f"  • Scenarios where CSV better: {len(negative_advantages)}/{len(advantages_after)}")
        
        if positive_advantages:
            print(f"  • Average advantage when graph helps: +{np.mean(positive_advantages):.3f}")
        if negative_advantages:
            print(f"  • Average disadvantage when CSV better: {np.mean(negative_advantages):.3f}")

def main():
    """Main execution function"""
    analyzer = CleanGraphAccuracyAnalyzer()
    analyzer.load_all_data()
    results = analyzer.run_analysis()
    
    print(f"\n✅ Analysis complete! Results saved to: {analyzer.output_dir}")
    return analyzer, results

if __name__ == "__main__":
    main()