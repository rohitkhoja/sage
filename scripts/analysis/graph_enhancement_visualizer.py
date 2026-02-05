#!/usr/bin/env python3
"""
Graph Enhancement Visualizer

Creates comprehensive visualizations and reports for graph enhancement analysis.
Shows how graph neighbors improve retrieval accuracy across different k-value combinations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
import json
import pickle
from dataclasses import asdict

# Import our data structures
from graph_enhanced_accuracy_analyzer import KValueResult, QuestionAccuracyResult, KValueTest

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphEnhancementVisualizer:
    """Comprehensive visualization system for graph enhancement analysis"""
    
    def __init__(self, cache_dir: str = "output/graph_enhancement_cache"):
        self.cache_dir = Path(cache_dir)
        self.output_dir = self.cache_dir / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results container
        self.all_results = []
        
        logger.info(f"Initialized visualizer with output dir: {self.output_dir}")
    
    def load_results(self, results_file: str = None):
        """Load analysis results from cache"""
        if results_file is None:
            results_file = self.cache_dir / "k_value_test_results.pkl"
        
        logger.info(f"Loading results from: {results_file}")
        
        with open(results_file, 'rb') as f:
            self.all_results = pickle.load(f)
        
        logger.info(f"Loaded results for {len(self.all_results)} k-value tests")
    
    def create_accuracy_comparison_plot(self):
        """Create main accuracy comparison plot across all k-value tests"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Prepare data
        test_names = []
        csv_only_accuracies = []
        csv_plus_graph_accuracies = []
        improvements = []
        csv_k_values = []
        graph_k_values = []
        
        for result in self.all_results:
            test_names.append(result.test_config.name)
            csv_only_accuracies.append(result.csv_only_accuracy * 100)
            csv_plus_graph_accuracies.append(result.csv_plus_graph_accuracy * 100)
            improvements.append(result.improvement * 100)
            csv_k_values.append(result.test_config.csv_k)
            graph_k_values.append(result.test_config.graph_k)
        
        # Plot 1: Accuracy Comparison
        x = np.arange(len(test_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, csv_only_accuracies, width, 
                       label='CSV Only', alpha=0.8, color='lightcoral')
        bars2 = ax1.bar(x + width/2, csv_plus_graph_accuracies, width,
                       label='CSV + Graph', alpha=0.8, color='lightblue')
        
        ax1.set_xlabel('K-Value Test Configuration')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Retrieval Accuracy: CSV-Only vs CSV+Graph')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"CSV k={csv_k}\nGraph k={graph_k}" 
                            for csv_k, graph_k in zip(csv_k_values, graph_k_values)], 
                           rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add improvement values on bars
        for i, (bar1, bar2, improvement) in enumerate(zip(bars1, bars2, improvements)):
            if improvement > 0:
                height = max(bar1.get_height(), bar2.get_height())
                ax1.text(i, height + 1, f'+{improvement:.1f}%', 
                        ha='center', va='bottom', fontweight='bold', color='green')
        
        # Plot 2: Improvement Analysis
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars3 = ax2.bar(x, improvements, color=colors, alpha=0.7)
        
        ax2.set_xlabel('K-Value Test Configuration')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('Graph Enhancement Improvement by Test')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"CSV k={csv_k}\nGraph k={graph_k}" 
                            for csv_k, graph_k in zip(csv_k_values, graph_k_values)], 
                           rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.grid(True, alpha=0.3)
        
        # Add improvement values on bars
        for bar, improvement in zip(bars3, improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., 
                    height + (0.1 if height >= 0 else -0.3),
                    f'{improvement:.1f}%', ha='center', 
                    va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "accuracy_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Created accuracy comparison plot")
    
    def create_enhancement_pattern_analysis(self):
        """Analyze patterns in graph enhancement effectiveness"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Graph Enhancement Pattern Analysis', fontsize=16)
        
        # Prepare data
        csv_k_values = [r.test_config.csv_k for r in self.all_results]
        graph_k_values = [r.test_config.graph_k for r in self.all_results]
        improvements = [r.improvement * 100 for r in self.all_results]
        enhancement_rates = [r.questions_enhanced / r.total_questions * 100 for r in self.all_results]
        
        # Plot 1: Improvement vs CSV k-value
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(csv_k_values, improvements, c=graph_k_values, 
                              cmap='viridis', s=100, alpha=0.7)
        ax1.set_xlabel('CSV k-value')
        ax1.set_ylabel('Improvement (%)')
        ax1.set_title('Improvement vs CSV k-value\n(colored by Graph k-value)')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Graph k-value')
        
        # Plot 2: Improvement vs Graph k-value
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(graph_k_values, improvements, c=csv_k_values, 
                              cmap='plasma', s=100, alpha=0.7)
        ax2.set_xlabel('Graph k-value')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('Improvement vs Graph k-value\n(colored by CSV k-value)')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='CSV k-value')
        
        # Plot 3: Enhancement Rate Distribution
        ax3 = axes[1, 0]
        ax3.hist(enhancement_rates, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_xlabel('Enhancement Rate (%)')
        ax3.set_ylabel('Number of Tests')
        ax3.set_title('Distribution of Enhancement Rates')
        ax3.axvline(np.mean(enhancement_rates), color='red', linestyle='--',
                   label=f'Mean: {np.mean(enhancement_rates):.1f}%')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: CSV vs Graph k-value heatmap
        ax4 = axes[1, 1]
        
        # Create heatmap data
        unique_csv_k = sorted(set(csv_k_values))
        unique_graph_k = sorted(set(graph_k_values))
        
        heatmap_data = np.zeros((len(unique_csv_k), len(unique_graph_k)))
        
        for i, csv_k in enumerate(unique_csv_k):
            for j, graph_k in enumerate(unique_graph_k):
                # Find matching result
                for result in self.all_results:
                    if result.test_config.csv_k == csv_k and result.test_config.graph_k == graph_k:
                        heatmap_data[i, j] = result.improvement * 100
                        break
        
        im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
        ax4.set_xticks(range(len(unique_graph_k)))
        ax4.set_yticks(range(len(unique_csv_k)))
        ax4.set_xticklabels(unique_graph_k)
        ax4.set_yticklabels(unique_csv_k)
        ax4.set_xlabel('Graph k-value')
        ax4.set_ylabel('CSV k-value')
        ax4.set_title('Improvement Heatmap')
        
        # Add text annotations
        for i in range(len(unique_csv_k)):
            for j in range(len(unique_graph_k)):
                value = heatmap_data[i, j]
                if value != 0:
                    ax4.text(j, i, f'{value:.1f}%', ha='center', va='center',
                            color='white' if abs(value) > 2 else 'black', fontweight='bold')
        
        plt.colorbar(im, ax=ax4, label='Improvement (%)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "enhancement_patterns.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Created enhancement pattern analysis")
    
    def create_detailed_question_analysis(self):
        """Create detailed analysis of which questions benefit most from graph enhancement"""
        
        # Collect data across all tests
        all_question_data = {}
        
        for result in self.all_results:
            test_name = result.test_config.name
            
            for q_result in result.questions_results:
                q_id = q_result.question_id
                
                if q_id not in all_question_data:
                    all_question_data[q_id] = {
                        'gold_count': len(q_result.gold_docs),
                        'enhancements': [],
                        'csv_only_successes': [],
                        'graph_successes': []
                    }
                
                all_question_data[q_id]['enhancements'].append(q_result.graph_enhancement)
                all_question_data[q_id]['csv_only_successes'].append(q_result.csv_only_correct)
                all_question_data[q_id]['graph_successes'].append(q_result.csv_plus_graph_correct)
        
        # Calculate statistics
        enhancement_counts = [sum(data['enhancements']) for data in all_question_data.values()]
        csv_success_counts = [sum(data['csv_only_successes']) for data in all_question_data.values()]
        graph_success_counts = [sum(data['graph_successes']) for data in all_question_data.values()]
        gold_counts = [data['gold_count'] for data in all_question_data.values()]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Question-Level Enhancement Analysis', fontsize=16)
        
        # Plot 1: Distribution of enhancement counts per question
        ax1 = axes[0, 0]
        ax1.hist(enhancement_counts, bins=range(0, max(enhancement_counts) + 2), 
                alpha=0.7, color='lightblue', edgecolor='black')
        ax1.set_xlabel('Number of Tests Enhanced by Graph')
        ax1.set_ylabel('Number of Questions')
        ax1.set_title('Distribution of Graph Enhancements per Question')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: CSV success vs Graph enhancement correlation
        ax2 = axes[0, 1]
        ax2.scatter(csv_success_counts, enhancement_counts, alpha=0.6, s=50)
        ax2.set_xlabel('Number of CSV-Only Successes')
        ax2.set_ylabel('Number of Graph Enhancements')
        ax2.set_title('CSV Success vs Graph Enhancement')
        ax2.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = np.corrcoef(csv_success_counts, enhancement_counts)[0, 1]
        ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 3: Gold count vs Enhancement relationship
        ax3 = axes[1, 0]
        # Group by gold count
        gold_count_groups = {}
        for i, gold_count in enumerate(gold_counts):
            if gold_count not in gold_count_groups:
                gold_count_groups[gold_count] = []
            gold_count_groups[gold_count].append(enhancement_counts[i])
        
        gold_counts_unique = sorted(gold_count_groups.keys())
        avg_enhancements = [np.mean(gold_count_groups[gc]) for gc in gold_counts_unique]
        
        ax3.bar(gold_counts_unique, avg_enhancements, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_xlabel('Number of Gold Documents')
        ax3.set_ylabel('Average Graph Enhancements')
        ax3.set_title('Average Graph Enhancement by Gold Document Count')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Question success rate analysis
        ax4 = axes[1, 1]
        
        # Calculate success rates
        total_tests = len(self.all_results)
        csv_success_rates = [count / total_tests for count in csv_success_counts]
        graph_success_rates = [count / total_tests for count in graph_success_counts]
        
        ax4.scatter(csv_success_rates, graph_success_rates, alpha=0.6, s=50)
        ax4.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='No Improvement Line')
        ax4.set_xlabel('CSV-Only Success Rate')
        ax4.set_ylabel('CSV+Graph Success Rate')
        ax4.set_title('Question Success Rate: CSV vs CSV+Graph')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "question_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Created detailed question analysis")
    
    def create_k_value_optimization_analysis(self):
        """Analyze optimal k-value combinations"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('K-Value Optimization Analysis', fontsize=16)
        
        # Prepare data
        results_df = []
        for result in self.all_results:
            results_df.append({
                'csv_k': result.test_config.csv_k,
                'graph_k': result.test_config.graph_k,
                'improvement': result.improvement * 100,
                'enhancement_rate': result.questions_enhanced / result.total_questions * 100,
                'csv_accuracy': result.csv_only_accuracy * 100,
                'graph_accuracy': result.csv_plus_graph_accuracy * 100
            })
        
        df = pd.DataFrame(results_df)
        
        # Plot 1: Improvement by CSV k-value
        ax1 = axes[0, 0]
        csv_k_groups = df.groupby('csv_k')['improvement'].agg(['mean', 'std', 'count'])
        
        ax1.errorbar(csv_k_groups.index, csv_k_groups['mean'], 
                    yerr=csv_k_groups['std'], fmt='o-', capsize=5, capthick=2)
        ax1.set_xlabel('CSV k-value')
        ax1.set_ylabel('Average Improvement (%)')
        ax1.set_title('Average Improvement by CSV k-value')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Improvement by Graph k-value
        ax2 = axes[0, 1]
        graph_k_groups = df.groupby('graph_k')['improvement'].agg(['mean', 'std', 'count'])
        
        ax2.errorbar(graph_k_groups.index, graph_k_groups['mean'], 
                    yerr=graph_k_groups['std'], fmt='o-', capsize=5, capthick=2, color='orange')
        ax2.set_xlabel('Graph k-value')
        ax2.set_ylabel('Average Improvement (%)')
        ax2.set_title('Average Improvement by Graph k-value')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Efficiency analysis (improvement per graph neighbor)
        ax3 = axes[1, 0]
        df['efficiency'] = df['improvement'] / df['graph_k']
        
        ax3.scatter(df['graph_k'], df['efficiency'], c=df['csv_k'], 
                   cmap='viridis', s=100, alpha=0.7)
        ax3.set_xlabel('Graph k-value')
        ax3.set_ylabel('Improvement per Graph Neighbor (%)')
        ax3.set_title('Graph Enhancement Efficiency')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(ax3.collections[0], ax=ax3, label='CSV k-value')
        
        # Plot 4: Best configurations
        ax4 = axes[1, 1]
        
        # Find top 5 configurations by improvement
        top_configs = df.nlargest(5, 'improvement')
        
        config_labels = [f"CSV={row['csv_k']}, Graph={row['graph_k']}" 
                        for _, row in top_configs.iterrows()]
        
        bars = ax4.barh(range(len(top_configs)), top_configs['improvement'], 
                       color='lightcoral', alpha=0.8)
        ax4.set_yticks(range(len(top_configs)))
        ax4.set_yticklabels(config_labels)
        ax4.set_xlabel('Improvement (%)')
        ax4.set_title('Top 5 K-Value Configurations')
        ax4.grid(True, alpha=0.3)
        
        # Add values on bars
        for i, (bar, improvement) in enumerate(zip(bars, top_configs['improvement'])):
            ax4.text(improvement + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{improvement:.1f}%', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "k_value_optimization.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Created k-value optimization analysis")
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive text report"""
        
        report_file = self.output_dir / "graph_enhancement_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("GRAPH ENHANCEMENT ANALYSIS - COMPREHENSIVE REPORT\n")
            f.write("=" * 100 + "\n\n")
            
            # Overall Summary
            f.write(" OVERALL SUMMARY:\n")
            f.write(f" • Total K-Value Tests: {len(self.all_results)}\n")
            if self.all_results:
                f.write(f" • Total Questions Analyzed: {self.all_results[0].total_questions}\n")
                
                avg_csv_accuracy = np.mean([r.csv_only_accuracy for r in self.all_results]) * 100
                avg_graph_accuracy = np.mean([r.csv_plus_graph_accuracy for r in self.all_results]) * 100
                avg_improvement = np.mean([r.improvement for r in self.all_results]) * 100
                
                f.write(f" • Average CSV-Only Accuracy: {avg_csv_accuracy:.2f}%\n")
                f.write(f" • Average CSV+Graph Accuracy: {avg_graph_accuracy:.2f}%\n")
                f.write(f" • Average Improvement: {avg_improvement:.2f}%\n\n")
            
            # Detailed Results by Test
            f.write(" DETAILED RESULTS BY TEST:\n")
            f.write("-" * 100 + "\n")
            
            for result in self.all_results:
                improvement_pct = result.improvement * 100
                enhancement_rate = result.questions_enhanced / result.total_questions * 100
                
                f.write(f"\n{result.test_config.name}:\n")
                f.write(f" Configuration: CSV k={result.test_config.csv_k}, Graph k={result.test_config.graph_k}\n")
                f.write(f" CSV-Only Accuracy: {result.csv_only_accuracy:.4f} ({result.csv_only_accuracy*100:.2f}%)\n")
                f.write(f" CSV+Graph Accuracy: {result.csv_plus_graph_accuracy:.4f} ({result.csv_plus_graph_accuracy*100:.2f}%)\n")
                f.write(f" Improvement: {improvement_pct:+.2f}%\n")
                f.write(f" Questions Enhanced: {result.questions_enhanced}/{result.total_questions} ({enhancement_rate:.1f}%)\n")
            
            # Best and Worst Performing Configurations
            f.write("\n" + "=" * 50 + "\n")
            f.write(" BEST PERFORMING CONFIGURATIONS:\n")
            
            sorted_results = sorted(self.all_results, key=lambda x: x.improvement, reverse=True)
            
            for i, result in enumerate(sorted_results[:3], 1):
                improvement_pct = result.improvement * 100
                f.write(f" {i}. {result.test_config.name}: {improvement_pct:+.2f}% improvement\n")
                f.write(f" CSV k={result.test_config.csv_k}, Graph k={result.test_config.graph_k}\n")
                f.write(f" Enhanced {result.questions_enhanced} questions\n")
            
            f.write("\n LEAST EFFECTIVE CONFIGURATIONS:\n")
            for i, result in enumerate(sorted_results[-3:], 1):
                improvement_pct = result.improvement * 100
                f.write(f" {i}. {result.test_config.name}: {improvement_pct:+.2f}% improvement\n")
                f.write(f" CSV k={result.test_config.csv_k}, Graph k={result.test_config.graph_k}\n")
                f.write(f" Enhanced {result.questions_enhanced} questions\n")
            
            # Key Insights
            f.write("\n" + "=" * 50 + "\n")
            f.write(" KEY INSIGHTS:\n")
            
            # Calculate insights
            positive_improvements = [r for r in self.all_results if r.improvement > 0]
            if positive_improvements:
                f.write(f" {len(positive_improvements)}/{len(self.all_results)} configurations show positive improvement\n")
                
                avg_positive_improvement = np.mean([r.improvement for r in positive_improvements]) * 100
                f.write(f" Average positive improvement: {avg_positive_improvement:.2f}%\n")
            
            # CSV k-value analysis
            csv_k_improvements = {}
            for result in self.all_results:
                csv_k = result.test_config.csv_k
                if csv_k not in csv_k_improvements:
                    csv_k_improvements[csv_k] = []
                csv_k_improvements[csv_k].append(result.improvement * 100)
            
            best_csv_k = max(csv_k_improvements.keys(), 
                           key=lambda k: np.mean(csv_k_improvements[k]))
            f.write(f" Best performing CSV k-value: {best_csv_k} "
                   f"(avg improvement: {np.mean(csv_k_improvements[best_csv_k]):.2f}%)\n")
            
            # Graph k-value analysis
            graph_k_improvements = {}
            for result in self.all_results:
                graph_k = result.test_config.graph_k
                if graph_k not in graph_k_improvements:
                    graph_k_improvements[graph_k] = []
                graph_k_improvements[graph_k].append(result.improvement * 100)
            
            best_graph_k = max(graph_k_improvements.keys(), 
                             key=lambda k: np.mean(graph_k_improvements[k]))
            f.write(f" Best performing Graph k-value: {best_graph_k} "
                   f"(avg improvement: {np.mean(graph_k_improvements[best_graph_k]):.2f}%)\n")
            
            f.write("\n GENERATED VISUALIZATIONS:\n")
            f.write(" • accuracy_comparison.png - Main accuracy comparison\n")
            f.write(" • enhancement_patterns.png - Pattern analysis\n")
            f.write(" • question_analysis.png - Question-level analysis\n")
            f.write(" • k_value_optimization.png - K-value optimization\n")
        
        logger.info(f"Generated comprehensive report: {report_file}")
    
    def run_complete_visualization(self):
        """Run the complete visualization pipeline"""
        logger.info("Starting complete visualization pipeline...")
        
        # Load results
        self.load_results()
        
        # Create all visualizations
        self.create_accuracy_comparison_plot()
        self.create_enhancement_pattern_analysis()
        self.create_detailed_question_analysis()
        self.create_k_value_optimization_analysis()
        
        # Generate comprehensive report
        self.generate_comprehensive_report()
        
        logger.info(f"Complete visualization pipeline finished! All outputs saved to {self.output_dir}")

def main():
    """Main execution function"""
    visualizer = GraphEnhancementVisualizer()
    visualizer.run_complete_visualization()

if __name__ == "__main__":
    main()