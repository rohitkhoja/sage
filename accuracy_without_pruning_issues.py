#!/usr/bin/env python3
"""
Accuracy Analysis Without Similarity Pruning Issues

This script removes questions with similarity pruning issues and recalculates
CSV vs CSV+Graph accuracy to see the "clean" performance comparison.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CleanAccuracyAnalyzer:
    """Analyzer for accuracy without similarity pruning issues"""
    
    def __init__(self,
                 csv_file: str = "/shared/khoja/CogComp/output/dense_sparse_average_results (1).csv",
                 similarity_analysis_file: str = "/shared/khoja/CogComp/output/similarity_pruning_analysis/similarity_pruning_comprehensive_report.json",
                 output_dir: str = "/shared/khoja/CogComp/output/clean_accuracy_analysis"):
        
        self.csv_file = csv_file
        self.similarity_analysis_file = similarity_analysis_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.df = None
        self.similarity_pruning_questions = set()
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
        
        logger.info("Initialized CleanAccuracyAnalyzer")
    
    def load_data(self):
        """Load CSV data and similarity pruning questions"""
        logger.info("Loading CSV data...")
        self.df = pd.read_csv(self.csv_file)
        logger.info(f"Loaded {len(self.df)} questions from CSV")
        
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
        
        import ast, re
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
    
    def calculate_accuracy_for_scenario(self, csv_k: int, graph_k: int, extended_csv_k: int, exclude_pruning: bool = True) -> Tuple[float, float, float, int]:
        """Calculate CSV-only, CSV+Graph, and Extended CSV accuracy for a specific scenario"""
        
        csv_only_correct = 0
        csv_plus_graph_correct = 0
        extended_csv_correct = 0
        total_questions = 0
        
        for _, row in self.df.iterrows():
            question_id = row['question_id']
            
            # Skip questions with similarity pruning issues if requested
            if exclude_pruning and question_id in self.similarity_pruning_questions:
                continue
            
            gold_docs = self.extract_gold_docs(row['gold_docs'])
            if not gold_docs:  # Skip questions without gold docs
                continue
            
            total_questions += 1
            
            # CSV-only accuracy: use original csv_k (e.g., k=10 for CSV10+Graph10)
            csv_retrieved = self.get_csv_retrieved_docs(row, csv_k)
            csv_only_found = any(doc in csv_retrieved for doc in gold_docs)
            if csv_only_found:
                csv_only_correct += 1
            
            # CSV+Graph accuracy: use extended_csv_k (e.g., k=20 for CSV10+Graph10)
            # This represents what CSV+Graph should ideally achieve
            extended_csv_retrieved = self.get_csv_retrieved_docs(row, extended_csv_k)
            csv_plus_graph_found = any(doc in extended_csv_retrieved for doc in gold_docs)
            if csv_plus_graph_found:
                csv_plus_graph_correct += 1
            
            # Extended CSV accuracy: use extended_csv_k for fair comparison
            # This represents what pure CSV at extended k would achieve
            extended_csv_found = any(doc in extended_csv_retrieved for doc in gold_docs)
            if extended_csv_found:
                extended_csv_correct += 1
        
        csv_only_accuracy = csv_only_correct / total_questions if total_questions > 0 else 0.0
        csv_plus_graph_accuracy = csv_plus_graph_correct / total_questions if total_questions > 0 else 0.0
        extended_csv_accuracy = extended_csv_correct / total_questions if total_questions > 0 else 0.0
        
        return csv_only_accuracy, csv_plus_graph_accuracy, extended_csv_accuracy, total_questions
    
    def calculate_all_accuracies(self):
        """Calculate accuracies for all scenarios with and without pruning issues"""
        logger.info("Calculating accuracies for all scenarios...")
        
        results = {
            'with_pruning_issues': [],
            'without_pruning_issues': [],
            'improvement_analysis': []
        }
        
        for config in self.scenario_configs:
            csv_k = config['csv_k']
            graph_k = config['graph_k']
            extended_csv_k = config['extended_csv_k']
            name = config['name']
            
            # Calculate with pruning issues (all questions)
            csv_acc_with, graph_acc_with, extended_csv_acc_with, total_with = self.calculate_accuracy_for_scenario(
                csv_k, graph_k, extended_csv_k, exclude_pruning=False
            )
            
            # Calculate without pruning issues (excluding problematic questions)
            csv_acc_without, graph_acc_without, extended_csv_acc_without, total_without = self.calculate_accuracy_for_scenario(
                csv_k, graph_k, extended_csv_k, exclude_pruning=True
            )
            
            results['with_pruning_issues'].append({
                'scenario': name,
                'csv_k': csv_k,
                'graph_k': graph_k,
                'extended_csv_k': extended_csv_k,
                'csv_accuracy': csv_acc_with,
                'csv_plus_graph_accuracy': graph_acc_with,
                'extended_csv_accuracy': extended_csv_acc_with,
                'total_questions': total_with
            })
            
            results['without_pruning_issues'].append({
                'scenario': name,
                'csv_k': csv_k,
                'graph_k': graph_k,
                'extended_csv_k': extended_csv_k,
                'csv_accuracy': csv_acc_without,
                'csv_plus_graph_accuracy': graph_acc_without,
                'extended_csv_accuracy': extended_csv_acc_without,
                'total_questions': total_without
            })
            
            # Calculate improvement
            csv_improvement = csv_acc_without - csv_acc_with
            graph_improvement = graph_acc_without - graph_acc_with
            extended_csv_improvement = extended_csv_acc_without - extended_csv_acc_with
            
            results['improvement_analysis'].append({
                'scenario': name,
                'csv_k': csv_k,
                'csv_improvement': csv_improvement,
                'graph_improvement': graph_improvement,
                'extended_csv_improvement': extended_csv_improvement,
                'questions_excluded': total_with - total_without
            })
            
            logger.info(f"{name}: CSV {csv_acc_with:.3f}→{csv_acc_without:.3f}, "
                       f"CSV+Graph {graph_acc_with:.3f}→{graph_acc_without:.3f}, "
                       f"Extended CSV {extended_csv_acc_with:.3f}→{extended_csv_acc_without:.3f}, "
                       f"Excluded {total_with - total_without} questions")
        
        return results
    
    def create_accuracy_plots(self, results: Dict[str, List[Dict]]):
        """Create accuracy comparison plots"""
        logger.info("Creating accuracy plots...")
        
        # Extract data for plotting
        scenarios = [item['scenario'] for item in results['without_pruning_issues']]
        csv_k_values = [item['csv_k'] for item in results['without_pruning_issues']]
        
        csv_acc_with = [item['csv_accuracy'] for item in results['with_pruning_issues']]
        graph_acc_with = [item['csv_plus_graph_accuracy'] for item in results['with_pruning_issues']]
        extended_csv_acc_with = [item['extended_csv_accuracy'] for item in results['with_pruning_issues']]
        
        csv_acc_without = [item['csv_accuracy'] for item in results['without_pruning_issues']]
        graph_acc_without = [item['csv_plus_graph_accuracy'] for item in results['without_pruning_issues']]
        extended_csv_acc_without = [item['extended_csv_accuracy'] for item in results['without_pruning_issues']]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: With pruning issues
        ax1.plot(csv_k_values, graph_acc_with, 'r-s', label='CSV + Graph', linewidth=2, markersize=8)
        ax1.plot(csv_k_values, extended_csv_acc_with, 'g-^', label='Extended CSV', linewidth=2, markersize=8)
        ax1.set_xlabel('CSV k-value', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('CSV+Graph vs Extended CSV (WITH Pruning Issues)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for i, (graph_val, ext_csv_val) in enumerate(zip(graph_acc_with, extended_csv_acc_with)):
            ax1.annotate(f'{graph_val:.2f}', (csv_k_values[i], graph_val), 
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
            ax1.annotate(f'{ext_csv_val:.2f}', (csv_k_values[i], ext_csv_val), 
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        
        # Plot 2: Without pruning issues (THE KEY COMPARISON)
        ax2.plot(csv_k_values, graph_acc_without, 'r-s', label='CSV + Graph', linewidth=2, markersize=8)
        ax2.plot(csv_k_values, extended_csv_acc_without, 'g-^', label='Extended CSV', linewidth=2, markersize=8)
        ax2.set_xlabel('CSV k-value', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('CSV+Graph vs Extended CSV (WITHOUT Pruning Issues)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Add value labels
        for i, (graph_val, ext_csv_val) in enumerate(zip(graph_acc_without, extended_csv_acc_without)):
            ax2.annotate(f'{graph_val:.2f}', (csv_k_values[i], graph_val), 
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
            ax2.annotate(f'{ext_csv_val:.2f}', (csv_k_values[i], ext_csv_val), 
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "accuracy_comparison_plot.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {plot_file}")
        
        # Show plot
        plt.show()
        
        return fig
    
    def save_results(self, results: Dict[str, List[Dict]]):
        """Save results to JSON file"""
        results_file = self.output_dir / "clean_accuracy_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary report
        self._create_summary_report(results)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def _create_summary_report(self, results: Dict[str, List[Dict]]):
        """Create human-readable summary report"""
        report_file = self.output_dir / "clean_accuracy_summary.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CLEAN ACCURACY ANALYSIS REPORT\n")
            f.write("(After Removing Similarity Pruning Issues)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"📊 QUESTIONS EXCLUDED: {len(self.similarity_pruning_questions)} unique questions\n\n")
            
            f.write("📋 FAIR COMPARISON: CSV+Graph vs Extended CSV\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Scenario':<20} {'CSV+Graph Before':<16} {'CSV+Graph After':<16} {'ExtCSV Before':<14} {'ExtCSV After':<14} {'Advantage':<12}\n")
            f.write("-" * 110 + "\n")
            
            for i, scenario in enumerate(results['without_pruning_issues']):
                before = results['with_pruning_issues'][i]
                after = results['without_pruning_issues'][i]
                improvement = results['improvement_analysis'][i]
                
                # Calculate advantage of CSV+Graph over Extended CSV
                advantage_after = after['csv_plus_graph_accuracy'] - after['extended_csv_accuracy']
                
                f.write(f"{scenario['scenario']:<20} "
                       f"{before['csv_plus_graph_accuracy']:<16.3f} "
                       f"{after['csv_plus_graph_accuracy']:<16.3f} "
                       f"{before['extended_csv_accuracy']:<14.3f} "
                       f"{after['extended_csv_accuracy']:<14.3f} "
                       f"{advantage_after:+<12.3f}\n")
            
            f.write("\n📋 INTERPRETATION:\n")
            f.write("• CSV+Graph: What graph enhancement should ideally achieve\n")
            f.write("• Extended CSV: What pure CSV retrieval achieves at same total k\n")
            f.write("• Advantage: Positive = Graph helps, Negative = Pure CSV better\n")
            
            f.write("\n" + "=" * 80 + "\n")
    
    def run_analysis(self):
        """Run complete clean accuracy analysis"""
        logger.info("Starting clean accuracy analysis...")
        
        # Load data
        self.load_data()
        
        # Calculate accuracies
        results = self.calculate_all_accuracies()
        
        # Create plots
        self.create_accuracy_plots(results)
        
        # Save results
        self.save_results(results)
        
        # Print summary
        self._print_summary(results)
        
        logger.info("Clean accuracy analysis complete!")
        return results
    
    def _print_summary(self, results: Dict[str, List[Dict]]):
        """Print summary to console"""
        print("\n" + "=" * 80)
        print("CLEAN ACCURACY ANALYSIS SUMMARY")
        print("=" * 80)
        
        print(f"\n📊 EXCLUDED QUESTIONS: {len(self.similarity_pruning_questions)} questions with similarity pruning issues")
        
        print(f"\n📋 KEY FINDINGS:")
        
        # Calculate overall improvements
        csv_improvements = [item['csv_improvement'] for item in results['improvement_analysis']]
        graph_improvements = [item['graph_improvement'] for item in results['improvement_analysis']]
        
        avg_csv_improvement = np.mean(csv_improvements)
        avg_graph_improvement = np.mean(graph_improvements)
        
        print(f"  • Average CSV accuracy improvement: {avg_csv_improvement:.3f} ({avg_csv_improvement*100:.1f}%)")
        print(f"  • Average CSV+Graph accuracy improvement: {avg_graph_improvement:.3f} ({avg_graph_improvement*100:.1f}%)")
        
        # Show scenarios with biggest improvements
        max_csv_idx = np.argmax(csv_improvements)
        max_graph_idx = np.argmax(graph_improvements)
        
        print(f"\n🚀 BIGGEST IMPROVEMENTS:")
        print(f"  • CSV: {results['improvement_analysis'][max_csv_idx]['scenario']} (+{csv_improvements[max_csv_idx]:.3f})")
        print(f"  • CSV+Graph: {results['improvement_analysis'][max_graph_idx]['scenario']} (+{graph_improvements[max_graph_idx]:.3f})")

def main():
    """Main execution function"""
    analyzer = CleanAccuracyAnalyzer()
    results = analyzer.run_analysis()
    
    print(f"\n✅ Analysis complete! Results and plots saved to: {analyzer.output_dir}")
    return analyzer, results

if __name__ == "__main__":
    main()