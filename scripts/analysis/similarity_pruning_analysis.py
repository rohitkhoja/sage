#!/usr/bin/env python3
"""
Similarity Pruning Analysis

This script analyzes the 84% similarity pruning cases to understand:
1. Total number of questions representing similarity pruning misses
2. Breakdown by scenario 
3. Common questions analysis across scenarios
4. Focus on higher k-value scenarios
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict, Counter
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimilarityPruningAnalyzer:
    """Analyzer for similarity pruning cases across scenarios"""
    
    def __init__(self,
                 csv_vs_graph_dir: str = "/shared/khoja/CogComp/output/graph_enhancement_cache/csv_vs_graph_analysis",
                 miss_analysis_file: str = "/shared/khoja/CogComp/output/graph_miss_analysis/miss_analysis_results.json",
                 output_dir: str = "/shared/khoja/CogComp/output/similarity_pruning_analysis"):
        
        self.csv_vs_graph_dir = Path(csv_vs_graph_dir)
        self.miss_analysis_file = miss_analysis_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define scenarios to analyze (including ALL scenarios as requested)
        self.scenarios_to_analyze = [
            "CSV1_vs_Graph1_vs_CSV2",
            "CSV2_vs_Graph3_vs_CSV5",
            "CSV5_vs_Graph5_vs_CSV10",
            "CSV10_vs_Graph10_vs_CSV20",
            "CSV20_vs_Graph30_vs_CSV50", 
            "CSV25_vs_Graph25_vs_CSV50",
            "CSV30_vs_Graph20_vs_CSV50",
            "CSV50_vs_Graph50_vs_CSV100"
        ]
        
        # Data containers
        self.scenario_data = {}
        self.similarity_pruning_questions = {}
        self.miss_analysis_data = {}
        
        logger.info("Initialized SimilarityPruningAnalyzer")
    
    def load_data(self):
        """Load all scenario data and miss analysis results"""
        logger.info("Loading scenario data...")
        
        # Load scenario data
        for scenario in self.scenarios_to_analyze:
            scenario_path = self.csv_vs_graph_dir / scenario
            if scenario_path.exists():
                self._load_scenario_data(scenario, scenario_path)
            else:
                logger.warning(f"Scenario path not found: {scenario_path}")
        
        # Load miss analysis data
        logger.info("Loading miss analysis data...")
        if Path(self.miss_analysis_file).exists():
            with open(self.miss_analysis_file, 'r') as f:
                self.miss_analysis_data = json.load(f)
        else:
            logger.warning(f"Miss analysis file not found: {self.miss_analysis_file}")
        
        logger.info("Data loading complete")
    
    def _load_scenario_data(self, scenario_name: str, scenario_path: Path):
        """Load data for a specific scenario"""
        csv_wins_file = scenario_path / "csv_wins.json"
        
        if not csv_wins_file.exists():
            logger.warning(f"csv_wins.json not found for {scenario_name}")
            return
        
        with open(csv_wins_file, 'r') as f:
            csv_wins_data = json.load(f)
        
        self.scenario_data[scenario_name] = csv_wins_data
        logger.info(f"Loaded {scenario_name}: {csv_wins_data['total_questions']} questions in csv_wins")
    
    def analyze_similarity_pruning_questions(self):
        """Analyze questions where gold docs were missed due to similarity pruning"""
        logger.info("Analyzing similarity pruning questions...")
        
        # Get similarity pruning cases from miss analysis
        similarity_pruning_cases = []
        
        # Check both common_missed and scenario_specific for similarity_pruning cases
        for category in ['common_missed', 'scenario_specific']:
            if category in self.miss_analysis_data:
                for case in self.miss_analysis_data[category]:
                    if case['miss_reason'] == 'similarity_pruning':
                        similarity_pruning_cases.append(case)
        
        logger.info(f"Found {len(similarity_pruning_cases)} gold documents missed due to similarity pruning")
        
        # Now find questions from csv_wins that contain these missed gold docs
        similarity_pruning_questions_by_scenario = {}
        
        for scenario_name, scenario_data in self.scenario_data.items():
            questions_with_sim_pruning = []
            
            for question in scenario_data['questions']:
                question_id = question['question_id']
                
                # Check if this question has gold docs that were missed due to similarity pruning
                # Get gold docs that extended CSV found but CSV+Graph missed
                extended_csv_gold = set(question['analysis']['gold_found_in_extended_csv'])
                csv_k_gold = set(question['analysis']['gold_found_in_csv_k'])
                graph_gold = set(question['analysis']['gold_found_in_graph'])
                
                missed_by_graph = extended_csv_gold - csv_k_gold - graph_gold
                
                # Check if any of these missed gold docs are due to similarity pruning
                has_similarity_pruning = False
                similarity_pruning_golds = []
                
                for sim_case in similarity_pruning_cases:
                    if sim_case['gold_doc'] in missed_by_graph:
                        has_similarity_pruning = True
                        similarity_pruning_golds.append(sim_case['gold_doc'])
                
                if has_similarity_pruning:
                    question_data = {
                        'question_id': question_id,
                        'question_text': question['question_text'],
                        'similarity_pruning_golds': similarity_pruning_golds,
                        'total_missed_golds': list(missed_by_graph),
                        'csv_k_retrieved': question['analysis']['csv_k_retrieved'],
                        'graph_neighbors': question['analysis']['graph_neighbors']
                    }
                    questions_with_sim_pruning.append(question_data)
            
            similarity_pruning_questions_by_scenario[scenario_name] = questions_with_sim_pruning
            logger.info(f"{scenario_name}: {len(questions_with_sim_pruning)} questions with similarity pruning issues")
        
        self.similarity_pruning_questions = similarity_pruning_questions_by_scenario
        return similarity_pruning_questions_by_scenario
    
    def analyze_common_questions(self):
        """Analyze questions that are common across scenarios"""
        logger.info("Analyzing common questions across scenarios...")
        
        # Create sets of question IDs for each scenario
        scenario_question_sets = {}
        for scenario_name, questions in self.similarity_pruning_questions.items():
            question_ids = {q['question_id'] for q in questions}
            scenario_question_sets[scenario_name] = question_ids
        
        # Analyze pairwise overlaps
        overlap_analysis = {}
        scenario_names = list(scenario_question_sets.keys())
        
        for i, scenario1 in enumerate(scenario_names):
            for j, scenario2 in enumerate(scenario_names):
                if i < j: # Avoid duplicate pairs
                    set1 = scenario_question_sets[scenario1]
                    set2 = scenario_question_sets[scenario2]
                    
                    overlap = set1.intersection(set2)
                    overlap_analysis[f"{scenario1}_vs_{scenario2}"] = {
                        'common_questions': len(overlap),
                        'scenario1_unique': len(set1 - set2),
                        'scenario2_unique': len(set2 - set1),
                        'scenario1_total': len(set1),
                        'scenario2_total': len(set2),
                        'overlap_percentage': len(overlap) / len(set1.union(set2)) * 100 if set1.union(set2) else 0,
                        'common_question_ids': list(overlap)
                    }
        
        # Find questions common to ALL scenarios
        all_question_sets = list(scenario_question_sets.values())
        if all_question_sets:
            common_to_all = set.intersection(*all_question_sets)
        else:
            common_to_all = set()
        
        return {
            'pairwise_overlaps': overlap_analysis,
            'common_to_all_scenarios': {
                'count': len(common_to_all),
                'question_ids': list(common_to_all)
            },
            'scenario_totals': {name: len(qset) for name, qset in scenario_question_sets.items()}
        }
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        logger.info("Generating comprehensive similarity pruning report...")
        
        # Analyze similarity pruning questions
        sim_pruning_questions = self.analyze_similarity_pruning_questions()
        
        # Analyze common questions
        common_analysis = self.analyze_common_questions()
        
        # Calculate overall statistics
        total_questions_with_sim_pruning = sum(len(questions) for questions in sim_pruning_questions.values())
        unique_questions_with_sim_pruning = len(set(
            q['question_id'] for questions in sim_pruning_questions.values() for q in questions
        ))
        
        # Create comprehensive report
        report = {
            'executive_summary': {
                'total_scenarios_analyzed': len(self.scenarios_to_analyze),
                'total_questions_with_similarity_pruning': total_questions_with_sim_pruning,
                'unique_questions_with_similarity_pruning': unique_questions_with_sim_pruning,
                'scenarios_included': self.scenarios_to_analyze
            },
            'scenario_breakdown': {},
            'common_questions_analysis': common_analysis,
            'detailed_questions_by_scenario': sim_pruning_questions
        }
        
        # Add scenario breakdown
        for scenario, questions in sim_pruning_questions.items():
            total_csv_wins = len(self.scenario_data[scenario]['questions']) if scenario in self.scenario_data else 0
            
            report['scenario_breakdown'][scenario] = {
                'questions_with_similarity_pruning': len(questions),
                'total_csv_wins_questions': total_csv_wins,
                'percentage_of_csv_wins': (len(questions) / total_csv_wins * 100) if total_csv_wins > 0 else 0,
                'unique_similarity_pruning_golds': len(set(
                    gold for q in questions for gold in q['similarity_pruning_golds']
                ))
            }
        
        # Save report
        self._save_report(report)
        
        return report
    
    def _save_report(self, report: Dict[str, Any]):
        """Save comprehensive report to files"""
        
        # Save JSON report
        json_file = self.output_dir / "similarity_pruning_comprehensive_report.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save human-readable report
        txt_file = self.output_dir / "similarity_pruning_analysis_report.txt"
        with open(txt_file, 'w') as f:
            self._write_human_readable_report(f, report)
        
        logger.info(f"Reports saved to {self.output_dir}")
    
    def _write_human_readable_report(self, f, report: Dict[str, Any]):
        """Write human-readable version of the report"""
        f.write("=" * 80 + "\n")
        f.write("SIMILARITY PRUNING ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Executive Summary
        exec_summary = report['executive_summary']
        f.write(" EXECUTIVE SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"• Scenarios analyzed: {exec_summary['total_scenarios_analyzed']}\n")
        f.write(f"• Total question instances with similarity pruning: {exec_summary['total_questions_with_similarity_pruning']}\n")
        f.write(f"• Unique questions with similarity pruning: {exec_summary['unique_questions_with_similarity_pruning']}\n")
        f.write(f"• Scenarios included: {', '.join(exec_summary['scenarios_included'])}\n\n")
        
        # Scenario Breakdown
        f.write(" SCENARIO BREAKDOWN\n")
        f.write("-" * 40 + "\n")
        for scenario, data in report['scenario_breakdown'].items():
            f.write(f"\n{scenario}:\n")
            f.write(f" • Questions with similarity pruning: {data['questions_with_similarity_pruning']}\n")
            f.write(f" • Total CSV wins questions: {data['total_csv_wins_questions']}\n")
            f.write(f" • Percentage of CSV wins: {data['percentage_of_csv_wins']:.1f}%\n")
            f.write(f" • Unique similarity pruning gold docs: {data['unique_similarity_pruning_golds']}\n")
        
        # Common Questions Analysis
        f.write(f"\n COMMON QUESTIONS ANALYSIS\n")
        f.write("-" * 40 + "\n")
        
        common_to_all = report['common_questions_analysis']['common_to_all_scenarios']
        f.write(f"• Questions common to ALL scenarios: {common_to_all['count']}\n")
        
        if common_to_all['count'] > 0:
            f.write(" Common question IDs:\n")
            for qid in common_to_all['question_ids'][:10]: # Show first 10
                f.write(f" - {qid}\n")
            if len(common_to_all['question_ids']) > 10:
                f.write(f" ... and {len(common_to_all['question_ids']) - 10} more\n")
        
        f.write(f"\n PAIRWISE OVERLAPS (Top 10):\n")
        pairwise = report['common_questions_analysis']['pairwise_overlaps']
        
        # Sort by overlap percentage
        sorted_pairs = sorted(pairwise.items(), key=lambda x: x[1]['overlap_percentage'], reverse=True)
        
        for pair_name, data in sorted_pairs[:10]:
            f.write(f"\n{pair_name}:\n")
            f.write(f" • Common questions: {data['common_questions']}\n")
            f.write(f" • Overlap percentage: {data['overlap_percentage']:.1f}%\n")
            f.write(f" • Scenario 1 unique: {data['scenario1_unique']}\n")
            f.write(f" • Scenario 2 unique: {data['scenario2_unique']}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    def print_summary(self, report: Dict[str, Any]):
        """Print summary to console"""
        print("\n" + "=" * 80)
        print("SIMILARITY PRUNING ANALYSIS SUMMARY")
        print("=" * 80)
        
        exec_summary = report['executive_summary']
        print(f"\n EXECUTIVE SUMMARY:")
        print(f" • Scenarios analyzed: {exec_summary['total_scenarios_analyzed']}")
        print(f" • Total question instances with similarity pruning: {exec_summary['total_questions_with_similarity_pruning']}")
        print(f" • Unique questions with similarity pruning: {exec_summary['unique_questions_with_similarity_pruning']}")
        
        print(f"\n SCENARIO BREAKDOWN:")
        for scenario, data in report['scenario_breakdown'].items():
            print(f" • {scenario}: {data['questions_with_similarity_pruning']} questions ({data['percentage_of_csv_wins']:.1f}% of CSV wins)")
        
        common_to_all = report['common_questions_analysis']['common_to_all_scenarios']
        print(f"\n COMMON QUESTIONS:")
        print(f" • Questions common to ALL scenarios: {common_to_all['count']}")
        
        # Show top 3 pairwise overlaps
        pairwise = report['common_questions_analysis']['pairwise_overlaps']
        sorted_pairs = sorted(pairwise.items(), key=lambda x: x[1]['overlap_percentage'], reverse=True)
        
        print(f"\n TOP PAIRWISE OVERLAPS:")
        for i, (pair_name, data) in enumerate(sorted_pairs[:3]):
            print(f" {i+1}. {pair_name}: {data['common_questions']} common ({data['overlap_percentage']:.1f}%)")

def main():
    """Main execution function"""
    
    analyzer = SimilarityPruningAnalyzer()
    
    # Load data
    analyzer.load_data()
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report()
    
    # Print summary
    analyzer.print_summary(report)
    
    print(f"\n Analysis complete! Detailed results saved to: {analyzer.output_dir}")
    
    return analyzer, report

if __name__ == "__main__":
    main()