#!/usr/bin/env python3
"""
Focused Visualization System

Creates comprehensive visualizations for the focused retrieval analysis:
1. Individual question plots showing gold margin analysis
2. Combined aggregated plots across all questions
3. Statistical analysis of neighbor filtering effectiveness
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

# Import our data structures
from focused_retrieval_analyzer import QuestionResult, NeighborCandidate, RetrievedChunkMetrics

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

class FocusedVisualizationSystem:
    """Comprehensive visualization system for focused retrieval analysis"""
    
    def __init__(self, output_dir: str = "output/focused_retrieval_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "individual_questions").mkdir(exist_ok=True)
        (self.output_dir / "aggregated_analysis").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        
        # Analysis containers
        self.question_results = []
        
        logger.info(f"Initialized FocusedVisualizationSystem with output dir: {self.output_dir}")
    
    def load_results(self, results_file: str = "output/focused_retrieval_cache/focused_analysis_results.pkl"):
        """Load analysis results from cache"""
        logger.info(f"Loading results from: {results_file}")
        
        with open(results_file, 'rb') as f:
            self.question_results = pickle.load(f)
        
        logger.info(f"Loaded {len(self.question_results)} question results")
    
    def create_individual_question_plot(self, question_result: QuestionResult):
        """Create detailed similarity analysis plot for a single question"""
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Similarity Analysis: {question_result.question_id}\n'
                    f'Gold Found: {len(question_result.gold_in_neighbors)}, Missed: {len(question_result.gold_missed)}', 
                    fontsize=16, fontweight='bold')
        
        # Prepare data for comprehensive similarity analysis
        all_chunks_data = []
        
        # Add retrieved chunks data
        for chunk in question_result.retrieved_chunks:
            chunk_data = {
                'chunk_id': chunk.chunk_id,
                'chunk_type': chunk.chunk_type,
                'source': 'retrieved',
                'retrieval_rank': chunk.rank,
                'neighbor_rank': None,
                'winning_approach': chunk.winning_approach,
                'winning_sub_question': chunk.winning_sub_question,
                'content_similarity': chunk.content_similarity,
                'topic_similarity': chunk.topic_similarity if chunk.chunk_type == 'document' else 0.0,
                'description_similarity': chunk.description_similarity if chunk.chunk_type == 'table' else 0.0,
                'column_similarity': chunk.max_column_similarity if chunk.chunk_type == 'table' else 0.0,
                'main_question_content_similarity': chunk.main_question_content_similarity,
                'max_subquestion_content_similarity': chunk.max_subquestion_content_similarity,
                'entity_matches': chunk.total_entity_matches,
                'event_matches': chunk.total_event_matches,
                'total_similarity': chunk.total_similarity,
                'weighted_score': 0.0, # Retrieved chunks don't have neighbor weighted scores
                'is_gold': any(chunk.chunk_id == gold_doc for gold_doc in question_result.gold_docs)
            }
            all_chunks_data.append(chunk_data)
        
        # Add neighbor candidates data
        for i, neighbor in enumerate(question_result.all_neighbor_candidates):
            chunk_data = {
                'chunk_id': neighbor.node_id,
                'chunk_type': neighbor.node_type,
                'source': 'neighbor',
                'retrieval_rank': None,
                'neighbor_rank': i + 1,
                'winning_approach': neighbor.winning_approach,
                'winning_sub_question': neighbor.winning_sub_question,
                'content_similarity': neighbor.best_content_similarity,
                'topic_similarity': getattr(neighbor, 'best_topic_similarity', 0.0),
                'description_similarity': getattr(neighbor, 'best_description_similarity', 0.0),
                'column_similarity': getattr(neighbor, 'best_column_similarity', 0.0),
                'main_question_content_similarity': getattr(neighbor, 'best_main_question_content_similarity', 0.0),
                'max_subquestion_content_similarity': getattr(neighbor, 'best_subquestion_content_similarity', 0.0),
                'entity_matches': neighbor.entity_exact_matches + neighbor.entity_substring_matches,
                'event_matches': neighbor.event_exact_matches + neighbor.event_substring_matches,
                'total_similarity': neighbor.avg_similarity,
                'weighted_score': neighbor.weighted_score,
                'is_gold': neighbor.is_gold
            }
            all_chunks_data.append(chunk_data)
        
        df = pd.DataFrame(all_chunks_data)
        
        # Define colors for different categories
        def get_color_and_marker(row):
            if row['is_gold']:
                return ('red', 'D', 150, 1.0) # Gold: Red diamonds, large, opaque
            elif row['source'] == 'retrieved':
                if row['chunk_type'] == 'document':
                    return ('darkblue', 'o', 100, 0.8) # Retrieved docs: Dark blue circles
                else:
                    return ('darkorange', 's', 100, 0.8) # Retrieved tables: Dark orange squares
            else: # neighbors
                if row['chunk_type'] == 'document':
                    return ('lightblue', 'o', 60, 0.6) # Neighbor docs: Light blue circles
                else:
                    return ('lightsalmon', 's', 60, 0.6) # Neighbor tables: Light salmon squares
        
        # Apply color and marker mapping
        df[['color', 'marker', 'size', 'alpha']] = df.apply(
            lambda row: pd.Series(get_color_and_marker(row)), axis=1
        )
        
        # 1. Content Similarity Analysis (Top Left)
        ax1 = axes[0, 0]
        for _, group in df.groupby(['color', 'marker', 'source']):
            if group['source'].iloc[0] == 'retrieved':
                x_vals = group['retrieval_rank']
                xlabel = 'Retrieval Rank (1-10)'
                title_suffix = 'vs Retrieval Rank'
            else:
                x_vals = group['neighbor_rank']
                xlabel = 'Neighbor Rank (1-30+)'
                title_suffix = 'vs Neighbor Rank'
                
            ax1.scatter(x_vals, group['content_similarity'], 
                       c=group['color'].iloc[0], marker=group['marker'].iloc[0],
                       s=group['size'].iloc[0], alpha=group['alpha'].iloc[0],
                       label=f"{group['source'].iloc[0].title()} {group.iloc[0]['chunk_type']}")
        
        ax1.set_xlabel('Rank Position')
        ax1.set_ylabel('Content Similarity')
        ax1.set_title('Content Similarity Distribution')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Topic/Description Similarity (Top Middle)
        ax2 = axes[0, 1]
        # Separate documents (topic) and tables (description)
        doc_mask = df['chunk_type'] == 'document'
        table_mask = df['chunk_type'] == 'table'
        
        if doc_mask.any():
            doc_df = df[doc_mask]
            for _, group in doc_df.groupby(['color', 'marker', 'source']):
                if group['source'].iloc[0] == 'retrieved':
                    x_vals = group['retrieval_rank']
                else:
                    x_vals = group['neighbor_rank']
                ax2.scatter(x_vals, group['topic_similarity'], 
                           c=group['color'].iloc[0], marker=group['marker'].iloc[0],
                           s=group['size'].iloc[0], alpha=group['alpha'].iloc[0])
        
        if table_mask.any():
            table_df = df[table_mask]
            for _, group in table_df.groupby(['color', 'marker', 'source']):
                if group['source'].iloc[0] == 'retrieved':
                    x_vals = group['retrieval_rank']
                else:
                    x_vals = group['neighbor_rank']
                ax2.scatter(x_vals, group['description_similarity'], 
                           c=group['color'].iloc[0], marker=group['marker'].iloc[0],
                           s=group['size'].iloc[0], alpha=group['alpha'].iloc[0])
        
        ax2.set_xlabel('Rank Position')
        ax2.set_ylabel('Topic/Description Similarity')
        ax2.set_title('Topic (Docs) / Description (Tables) Similarity')
        ax2.grid(True, alpha=0.3)
        
        # 3. Column Similarity (Top Right) - Tables only
        ax3 = axes[0, 2]
        if table_mask.any():
            table_df = df[table_mask]
            for _, group in table_df.groupby(['color', 'marker', 'source']):
                if group['source'].iloc[0] == 'retrieved':
                    x_vals = group['retrieval_rank']
                else:
                    x_vals = group['neighbor_rank']
                ax3.scatter(x_vals, group['column_similarity'], 
                           c=group['color'].iloc[0], marker=group['marker'].iloc[0],
                           s=group['size'].iloc[0], alpha=group['alpha'].iloc[0])
        
        ax3.set_xlabel('Rank Position')
        ax3.set_ylabel('Column Similarity')
        ax3.set_title('Column Similarity (Tables Only)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Entity and Event Matches (Bottom Left)
        ax4 = axes[1, 0]
        for _, group in df.groupby(['color', 'marker', 'source']):
            if group['source'].iloc[0] == 'retrieved':
                x_vals = group['retrieval_rank']
            else:
                x_vals = group['neighbor_rank']
            
            # Plot entity matches as main scatter
            ax4.scatter(x_vals, group['entity_matches'], 
                       c=group['color'].iloc[0], marker=group['marker'].iloc[0],
                       s=group['size'].iloc[0], alpha=group['alpha'].iloc[0])
        
        ax4.set_xlabel('Rank Position')
        ax4.set_ylabel('Entity Matches')
        ax4.set_title('Entity Matches Distribution')
        ax4.grid(True, alpha=0.3)
        
        # 5. Weighted Score Analysis (Bottom Middle) - Neighbors only
        ax5 = axes[1, 1]
        neighbor_df = df[df['source'] == 'neighbor']
        if not neighbor_df.empty:
            for _, group in neighbor_df.groupby(['color', 'marker']):
                ax5.scatter(group['neighbor_rank'], group['weighted_score'], 
                           c=group['color'].iloc[0], marker=group['marker'].iloc[0],
                           s=group['size'].iloc[0], alpha=group['alpha'].iloc[0])
            
            # Add top-30 cutoff line if applicable
            if len(neighbor_df) >= 30:
                ax5.axvline(x=30, color='red', linestyle='--', alpha=0.7, label='Top-30 Cutoff')
                ax5.legend()
        
        ax5.set_xlabel('Neighbor Rank')
        ax5.set_ylabel('Weighted Score')
        ax5.set_title('Weighted Score (Neighbors Only)')
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary Statistics (Bottom Right)
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Calculate summary statistics
        gold_count = len(question_result.gold_in_neighbors)
        total_gold = len(question_result.gold_docs)
        retrieved_count = len(question_result.retrieved_chunks)
        neighbor_count = len(question_result.all_neighbor_candidates)
        
        # Gold ranking info
        gold_ranks = []
        gold_scores = []
        for gold in question_result.gold_in_neighbors:
            analysis = question_result.gold_margin_analysis.get(gold.node_id, {})
            if isinstance(analysis, dict) and 'rank' in analysis:
                gold_ranks.append(analysis['rank'])
                gold_scores.append(gold.weighted_score)
        
        summary_text = f"""
SUMMARY STATISTICS

 Document Counts:
• Retrieved Chunks: {retrieved_count}
• Neighbor Candidates: {neighbor_count}
• Gold Documents: {total_gold}
• Gold Found: {gold_count}

 Gold Performance:
"""
        if gold_ranks:
            avg_rank = np.mean(gold_ranks)
            best_rank = min(gold_ranks)
            avg_score = np.mean(gold_scores)
            summary_text += f"""• Average Rank: {avg_rank:.1f}
• Best Rank: #{best_rank}
• Avg Weighted Score: {avg_score:.3f}
• Success Rate: {gold_count/total_gold:.1%}

 Top Similarities:
"""
            # Find highest similarities
            max_content = df['content_similarity'].max()
            max_entity = df['entity_matches'].max()
            summary_text += f"""• Max Content Sim: {max_content:.3f}
• Max Entity Matches: {int(max_entity)}
"""
        else:
            summary_text += """• No gold documents found
• Check similarity thresholds
• Review neighbor selection
"""
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Add comprehensive legend at the bottom
        legend_elements = [
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='red', markersize=12, label=' Gold Documents'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', markersize=10, label=' Retrieved Documents'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='darkorange', markersize=10, label=' Retrieved Tables'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=8, label=' Neighbor Documents'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightsalmon', markersize=8, label=' Neighbor Tables')
        ]
        
        fig.legend(handles=legend_elements, loc='lower center', ncol=5, 
                  bbox_to_anchor=(0.5, -0.02), fontsize=12)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save plot
        plot_file = self.output_dir / "individual_questions" / f"question_{question_result.question_id}_similarity_analysis.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created detailed similarity plot for question {question_result.question_id}")
    
    def create_aggregated_plots(self):
        """Create aggregated analysis plots across all questions"""
        
        # Create comprehensive aggregated analysis
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('Aggregated Focused Retrieval Analysis', fontsize=16)
        
        # 1. Gold Detection Success Rate (Top Left)
        ax1 = axes[0, 0]
        
        success_data = []
        for result in self.question_results:
            total_gold = len(result.gold_docs)
            found_gold = len(result.gold_in_neighbors)
            success_rate = found_gold / total_gold if total_gold > 0 else 0
            success_data.append(success_rate)
        
        ax1.hist(success_data, bins=10, alpha=0.7, color='green', edgecolor='black')
        ax1.set_xlabel('Gold Detection Success Rate')
        ax1.set_ylabel('Number of Questions')
        ax1.set_title('Gold Document Detection Success Rate Distribution')
        ax1.axvline(np.mean(success_data), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(success_data):.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Neighbor Count Distribution (Top Right)
        ax2 = axes[0, 1]
        
        neighbor_counts = [len(result.all_neighbor_candidates) for result in self.question_results]
        filtered_counts = [len(result.filtered_neighbors) for result in self.question_results]
        
        x = range(len(self.question_results))
        width = 0.35
        
        ax2.bar([i - width/2 for i in x], neighbor_counts, width, label='All Neighbors', alpha=0.7, color='lightblue')
        ax2.bar([i + width/2 for i in x], filtered_counts, width, label='Filtered Neighbors', alpha=0.7, color='darkblue')
        
        ax2.set_xlabel('Question Index')
        ax2.set_ylabel('Number of Neighbors')
        ax2.set_title('Neighbor Count Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Similarity Threshold Distribution (Middle Left)
        ax3 = axes[1, 0]
        
        thresholds = [result.similarity_95th_percentile for result in self.question_results]
        
        ax3.hist(thresholds, bins=15, alpha=0.7, color='purple', edgecolor='black')
        ax3.set_xlabel('95th Percentile Similarity Threshold')
        ax3.set_ylabel('Number of Questions')
        ax3.set_title('Similarity Threshold Distribution')
        ax3.axvline(np.mean(thresholds), color='red', linestyle='--',
                   label=f'Mean: {np.mean(thresholds):.3f}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Gold Ranking Distribution (Middle Right)
        ax4 = axes[1, 1]
        
        all_ranks = []
        rank_bins = [1, 6, 11, 16, 21] # Top 5, 6-10, 11-15, 16-20
        rank_labels = ['Top 5', 'Rank 6-10', 'Rank 11-15', 'Rank 16-20']
        rank_counts = [0, 0, 0, 0]
        
        for result in self.question_results:
            for gold_id, analysis in result.gold_margin_analysis.items():
                if isinstance(analysis, dict) and analysis.get('status') == 'found':
                    rank = analysis.get('rank', -1)
                    if rank > 0:
                        all_ranks.append(rank)
                        # Categorize rank
                        if rank <= 5:
                            rank_counts[0] += 1
                        elif rank <= 10:
                            rank_counts[1] += 1
                        elif rank <= 15:
                            rank_counts[2] += 1
                        else:
                            rank_counts[3] += 1
        
        if all_ranks:
            colors = ['green', 'lightgreen', 'gold', 'orange']
            bars = ax4.bar(rank_labels, rank_counts, color=colors, alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Rank Range in Top-20')
            ax4.set_ylabel('Number of Gold Documents')
            ax4.set_title('Gold Document Rank Distribution')
            ax4.grid(True, alpha=0.3)
            
            # Add count labels on bars
            for bar, count in zip(bars, rank_counts):
                if count > 0:
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            str(count), ha='center', va='bottom', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No Gold Documents\nFound in Top-20', 
                    transform=ax4.transAxes, ha='center', va='center',
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        # 5. Retrieval Type Analysis (Bottom Left)
        ax5 = axes[2, 0]
        
        doc_similarities = []
        table_similarities = []
        
        for result in self.question_results:
            for chunk in result.retrieved_chunks:
                if chunk.chunk_type == 'document':
                    doc_similarities.append(chunk.total_similarity)
                else:
                    table_similarities.append(chunk.total_similarity)
        
        if doc_similarities and table_similarities:
            ax5.hist(doc_similarities, bins=15, alpha=0.7, label='Documents', color='blue')
            ax5.hist(table_similarities, bins=15, alpha=0.7, label='Tables', color='orange')
            ax5.set_xlabel('Total Similarity Score')
            ax5.set_ylabel('Count')
            ax5.set_title('Retrieved Chunk Similarity by Type')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Summary Statistics (Bottom Right)
        ax6 = axes[2, 1]
        ax6.axis('off')
        
        # Calculate summary statistics
        total_questions = len(self.question_results)
        total_gold_docs = sum(len(r.gold_docs) for r in self.question_results)
        total_found_gold = sum(len(r.gold_in_neighbors) for r in self.question_results)
        avg_neighbors = np.mean([len(r.all_neighbor_candidates) for r in self.question_results])
        avg_filtered = np.mean([len(r.filtered_neighbors) for r in self.question_results])
        avg_threshold = np.mean([r.similarity_95th_percentile for r in self.question_results])
        
        stats_text = f"""
        SUMMARY STATISTICS
        
        Total Questions: {total_questions}
        Total Gold Documents: {total_gold_docs}
        Gold Documents Found: {total_found_gold}
        Overall Success Rate: {total_found_gold/total_gold_docs:.3f}
        
        Avg Neighbors per Question: {avg_neighbors:.1f}
        Avg Filtered Neighbors: {avg_filtered:.1f}
        Filtering Reduction: {(1-avg_filtered/avg_neighbors)*100:.1f}%
        
        Avg 95th Percentile Threshold: {avg_threshold:.3f}
        """
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=12, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save aggregated plot
        plot_file = self.output_dir / "plots" / "aggregated_analysis.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info("Created aggregated analysis plot")
    
    def create_detailed_statistics(self):
        """Create detailed statistical analysis"""
        
        stats = {
            'overall_summary': {
                'total_questions': len(self.question_results),
                'total_gold_documents': sum(len(r.gold_docs) for r in self.question_results),
                'total_gold_found': sum(len(r.gold_in_neighbors) for r in self.question_results),
                'total_gold_missed': sum(len(r.gold_missed) for r in self.question_results),
                'overall_success_rate': 0.0
            },
            'per_question_analysis': [],
            'neighbor_filtering_stats': {
                'avg_neighbors_before_filtering': 0.0,
                'avg_neighbors_after_filtering': 0.0,
                'avg_filtering_reduction_percent': 0.0
            },
            'threshold_analysis': {
                'avg_95th_percentile_threshold': 0.0,
                'std_95th_percentile_threshold': 0.0,
                'min_threshold': 0.0,
                'max_threshold': 0.0
            },
            'gold_ranking_analysis': {
                'avg_rank_when_found': 0.0,
                'best_rank': 0,
                'worst_rank': 0,
                'top_5_count': 0,
                'top_10_count': 0,
                'top_15_count': 0,
                'top_20_count': 0
            },
            'retrieval_type_analysis': {
                'document_chunks': 0,
                'table_chunks': 0,
                'avg_doc_similarity': 0.0,
                'avg_table_similarity': 0.0
            }
        }
        
        # Calculate overall statistics
        total_gold = stats['overall_summary']['total_gold_documents']
        total_found = stats['overall_summary']['total_gold_found']
        stats['overall_summary']['overall_success_rate'] = total_found / total_gold if total_gold > 0 else 0.0
        
        # Per-question analysis
        neighbor_counts_before = []
        neighbor_counts_after = []
        thresholds = []
        all_ranks = []
        doc_similarities = []
        table_similarities = []
        
        for result in self.question_results:
            # Question-specific stats
            question_stats = {
                'question_id': result.question_id,
                'gold_documents': len(result.gold_docs),
                'gold_found': len(result.gold_in_neighbors),
                'gold_missed': len(result.gold_missed),
                'success_rate': len(result.gold_in_neighbors) / len(result.gold_docs) if len(result.gold_docs) > 0 else 0.0,
                'total_neighbors': len(result.all_neighbor_candidates),
                'filtered_neighbors': len(result.filtered_neighbors),
                'filtering_reduction': (len(result.all_neighbor_candidates) - len(result.filtered_neighbors)) / len(result.all_neighbor_candidates) if len(result.all_neighbor_candidates) > 0 else 0.0,
                'similarity_threshold': result.similarity_95th_percentile
            }
            stats['per_question_analysis'].append(question_stats)
            
            # Aggregate data for overall stats
            neighbor_counts_before.append(len(result.all_neighbor_candidates))
            neighbor_counts_after.append(len(result.filtered_neighbors))
            thresholds.append(result.similarity_95th_percentile)
            
            # Ranking analysis
            for gold_id, analysis in result.gold_margin_analysis.items():
                if isinstance(analysis, dict) and analysis.get('status') == 'found':
                    rank = analysis.get('rank', -1)
                    if rank > 0:
                        all_ranks.append(rank)
            
            # Retrieval type analysis
            for chunk in result.retrieved_chunks:
                if chunk.chunk_type == 'document':
                    doc_similarities.append(chunk.total_similarity)
                else:
                    table_similarities.append(chunk.total_similarity)
        
        # Calculate aggregated statistics
        stats['neighbor_filtering_stats'] = {
            'avg_neighbors_before_filtering': np.mean(neighbor_counts_before),
            'avg_neighbors_after_filtering': np.mean(neighbor_counts_after),
            'avg_filtering_reduction_percent': np.mean([q['filtering_reduction'] for q in stats['per_question_analysis']]) * 100
        }
        
        stats['threshold_analysis'] = {
            'avg_95th_percentile_threshold': np.mean(thresholds),
            'std_95th_percentile_threshold': np.std(thresholds),
            'min_threshold': np.min(thresholds),
            'max_threshold': np.max(thresholds)
        }
        
        if all_ranks:
            stats['gold_ranking_analysis'] = {
                'avg_rank_when_found': np.mean(all_ranks),
                'best_rank': np.min(all_ranks),
                'worst_rank': np.max(all_ranks),
                'top_5_count': len([r for r in all_ranks if r <= 5]),
                'top_10_count': len([r for r in all_ranks if r <= 10]),
                'top_15_count': len([r for r in all_ranks if r <= 15]),
                'top_20_count': len(all_ranks) # All found ranks are within top 20
            }
        
        stats['retrieval_type_analysis'] = {
            'document_chunks': len(doc_similarities),
            'table_chunks': len(table_similarities),
            'avg_doc_similarity': np.mean(doc_similarities) if doc_similarities else 0.0,
            'avg_table_similarity': np.mean(table_similarities) if table_similarities else 0.0
        }
        
        # Save detailed statistics
        stats_file = self.output_dir / "aggregated_analysis" / "detailed_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Saved detailed statistics to {stats_file}")
        return stats
    
    def generate_summary_report(self, stats: Dict[str, Any]):
        """Generate a comprehensive summary report"""
        
        report_file = self.output_dir / "focused_retrieval_summary_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FOCUSED RETRIEVAL ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall Summary
            f.write(" OVERALL SUMMARY:\n")
            f.write(f" • Total Questions Analyzed: {stats['overall_summary']['total_questions']}\n")
            f.write(f" • Total Gold Documents: {stats['overall_summary']['total_gold_documents']}\n")
            f.write(f" • Gold Documents Found: {stats['overall_summary']['total_gold_found']}\n")
            f.write(f" • Gold Documents Missed: {stats['overall_summary']['total_gold_missed']}\n")
            f.write(f" • Overall Success Rate: {stats['overall_summary']['overall_success_rate']:.3f}\n\n")
            
            # Filtering Effectiveness
            f.write(" NEIGHBOR FILTERING EFFECTIVENESS:\n")
            f.write(f" • Avg Neighbors Before Filtering: {stats['neighbor_filtering_stats']['avg_neighbors_before_filtering']:.1f}\n")
            f.write(f" • Avg Neighbors After Filtering: {stats['neighbor_filtering_stats']['avg_neighbors_after_filtering']:.1f}\n")
            f.write(f" • Average Reduction: {stats['neighbor_filtering_stats']['avg_filtering_reduction_percent']:.1f}%\n\n")
            
            # Threshold Analysis
            f.write(" SIMILARITY THRESHOLD ANALYSIS:\n")
            f.write(f" • Average 95th Percentile Threshold: {stats['threshold_analysis']['avg_95th_percentile_threshold']:.3f}\n")
            f.write(f" • Standard Deviation: {stats['threshold_analysis']['std_95th_percentile_threshold']:.3f}\n")
            f.write(f" • Range: {stats['threshold_analysis']['min_threshold']:.3f} - {stats['threshold_analysis']['max_threshold']:.3f}\n\n")
            
            # Gold Ranking Analysis
            if 'avg_rank_when_found' in stats['gold_ranking_analysis']:
                f.write(" GOLD DOCUMENT RANKING ANALYSIS:\n")
                f.write(f" • Average Rank When Found: {stats['gold_ranking_analysis']['avg_rank_when_found']:.1f}\n")
                f.write(f" • Best Rank Achieved: {stats['gold_ranking_analysis']['best_rank']}\n")
                f.write(f" • Worst Rank: {stats['gold_ranking_analysis']['worst_rank']}\n")
                f.write(f" • Gold Docs in Top 5: {stats['gold_ranking_analysis']['top_5_count']}\n")
                f.write(f" • Gold Docs in Top 10: {stats['gold_ranking_analysis']['top_10_count']}\n")
                f.write(f" • Gold Docs in Top 15: {stats['gold_ranking_analysis']['top_15_count']}\n")
                f.write(f" • Total Gold Docs Found: {stats['gold_ranking_analysis']['top_20_count']}\n")
                f.write("\n")
            
            # Retrieval Type Analysis
            f.write(" RETRIEVAL TYPE ANALYSIS:\n")
            f.write(f" • Document Chunks: {stats['retrieval_type_analysis']['document_chunks']}\n")
            f.write(f" • Table Chunks: {stats['retrieval_type_analysis']['table_chunks']}\n")
            f.write(f" • Avg Document Similarity: {stats['retrieval_type_analysis']['avg_doc_similarity']:.3f}\n")
            f.write(f" • Avg Table Similarity: {stats['retrieval_type_analysis']['avg_table_similarity']:.3f}\n\n")
            
            # Top Performing Questions
            f.write(" TOP PERFORMING QUESTIONS:\n")
            top_questions = sorted(stats['per_question_analysis'], key=lambda x: x['success_rate'], reverse=True)[:5]
            for i, q in enumerate(top_questions, 1):
                f.write(f" {i}. Question {q['question_id']}: {q['success_rate']:.3f} success rate "
                       f"({q['gold_found']}/{q['gold_documents']} found)\n")
            f.write("\n")
            
            # Key Insights
            f.write(" KEY INSIGHTS:\n")
            avg_reduction = stats['neighbor_filtering_stats']['avg_filtering_reduction_percent']
            f.write(f" • Neighbor filtering reduces search space by {avg_reduction:.1f}% on average\n")
            
            success_rate = stats['overall_summary']['overall_success_rate']
            if success_rate > 0.8:
                f.write(f" • High success rate ({success_rate:.3f}) indicates effective neighbor filtering\n")
            elif success_rate > 0.5:
                f.write(f" • Moderate success rate ({success_rate:.3f}) suggests room for improvement\n")
            else:
                f.write(f" • Low success rate ({success_rate:.3f}) indicates need for strategy adjustment\n")
            
            f.write(f" • 95th percentile thresholds are consistent (std: {stats['threshold_analysis']['std_95th_percentile_threshold']:.3f})\n")
            
            f.write("\n GENERATED FILES:\n")
            f.write(" • individual_questions/ - Per-question detailed analysis plots\n")
            f.write(" • plots/aggregated_analysis.png - Combined analysis visualization\n")
            f.write(" • aggregated_analysis/detailed_statistics.json - Complete statistical analysis\n")
        
        logger.info(f"Saved summary report to {report_file}")
    
    def analyze_ranking_factors(self):
        """Analyze what factors contribute to gold document ranking success/failure"""
        
        logger.info("Analyzing ranking factors...")
        
        # Collect data for analysis
        all_chunks_above_gold = []
        all_similarity_factors = []
        gold_vs_non_gold_comparison = {
            'content_similarity': {'gold': [], 'non_gold': []},
            'topic_similarity': {'gold': [], 'non_gold': []},
            'description_similarity': {'gold': [], 'non_gold': []},
            'column_similarity': {'gold': [], 'non_gold': []},
            'entity_matches': {'gold': [], 'non_gold': []},
            'event_matches': {'gold': [], 'non_gold': []},
            'appearance_count': {'gold': [], 'non_gold': []},
            'weighted_score': {'gold': [], 'non_gold': []}
        }
        
        successful_questions = []
        failed_questions = []
        
        for result in self.question_results:
            # Collect chunks that rank above gold
            all_chunks_above_gold.extend(result.chunks_ranking_above_gold)
            
            # Collect similarity factor analysis
            all_similarity_factors.append(result.similarity_factor_analysis)
            
            # Categorize questions
            if result.gold_in_neighbors:
                successful_questions.append(result)
            else:
                failed_questions.append(result)
            
            # Collect comparison data
            factors = result.similarity_factor_analysis
            for metric in gold_vs_non_gold_comparison.keys():
                gold_key = f'gold_avg_{metric}'
                non_gold_key = f'non_gold_avg_{metric}'
                if gold_key in factors and factors[gold_key] > 0:
                    gold_vs_non_gold_comparison[metric]['gold'].append(factors[gold_key])
                if non_gold_key in factors and factors[non_gold_key] > 0:
                    gold_vs_non_gold_comparison[metric]['non_gold'].append(factors[non_gold_key])
        
        # Save chunks ranking above gold analysis
        chunks_above_gold_file = self.output_dir / "aggregated_analysis" / "chunks_ranking_above_gold.json"
        with open(chunks_above_gold_file, 'w') as f:
            json.dump({
                'total_chunks_ranking_above_gold': len(all_chunks_above_gold),
                'chunks_by_type': {
                    'document': len([c for c in all_chunks_above_gold if c['node_type'] == 'document']),
                    'table': len([c for c in all_chunks_above_gold if c['node_type'] == 'table'])
                },
                'retrieved_vs_neighbor_chunks': {
                    'retrieved_chunks': len([c for c in all_chunks_above_gold if c['is_retrieved_chunk']]),
                    'neighbor_chunks': len([c for c in all_chunks_above_gold if not c['is_retrieved_chunk']])
                },
                'avg_metrics_of_chunks_above_gold': {
                    'avg_weighted_score': np.mean([c['weighted_score'] for c in all_chunks_above_gold]) if all_chunks_above_gold else 0.0,
                    'avg_appearance_count': np.mean([c['appearance_count'] for c in all_chunks_above_gold]) if all_chunks_above_gold else 0.0,
                    'avg_content_similarity': np.mean([c['content_similarity'] for c in all_chunks_above_gold]) if all_chunks_above_gold else 0.0,
                    'avg_entity_matches': np.mean([c['total_entity_matches'] for c in all_chunks_above_gold]) if all_chunks_above_gold else 0.0,
                    'avg_event_matches': np.mean([c['total_event_matches'] for c in all_chunks_above_gold]) if all_chunks_above_gold else 0.0
                },
                'detailed_chunks': all_chunks_above_gold[:50] # Save top 50 for detailed analysis
            }, f, indent=2, cls=NumpyEncoder)
        
        # Generate ranking factor comparison analysis
        factor_comparison_file = self.output_dir / "aggregated_analysis" / "gold_vs_non_gold_factor_analysis.json"
        comparison_analysis = {}
        
        for metric, values in gold_vs_non_gold_comparison.items():
            if values['gold'] and values['non_gold']:
                gold_avg = np.mean(values['gold'])
                non_gold_avg = np.mean(values['non_gold'])
                
                comparison_analysis[metric] = {
                    'gold_average': gold_avg,
                    'non_gold_average': non_gold_avg,
                    'difference': gold_avg - non_gold_avg,
                    'ratio': gold_avg / non_gold_avg if non_gold_avg > 0 else 0.0,
                    'gold_advantage': gold_avg > non_gold_avg,
                    'sample_sizes': {
                        'gold_samples': len(values['gold']),
                        'non_gold_samples': len(values['non_gold'])
                    }
                }
        
        with open(factor_comparison_file, 'w') as f:
            json.dump(comparison_analysis, f, indent=2, cls=NumpyEncoder)
        
        # Generate insights report
        insights_file = self.output_dir / "ranking_factor_insights.txt"
        with open(insights_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("RANKING FACTOR ANALYSIS - DETAILED INSIGHTS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(" GOLD DOCUMENT RANKING PERFORMANCE:\n")
            f.write(f" • Questions with gold found: {len(successful_questions)}/100\n")
            f.write(f" • Questions with gold missed: {len(failed_questions)}/100\n")
            f.write(f" • Total chunks ranking above gold: {len(all_chunks_above_gold)}\n\n")
            
            f.write(" CHUNKS THAT OUTRANK GOLD DOCUMENTS:\n")
            if all_chunks_above_gold:
                retrieved_count = len([c for c in all_chunks_above_gold if c['is_retrieved_chunk']])
                neighbor_count = len([c for c in all_chunks_above_gold if not c['is_retrieved_chunk']])
                f.write(f" • Retrieved chunks outranking gold: {retrieved_count}\n")
                f.write(f" • Neighbor chunks outranking gold: {neighbor_count}\n")
                f.write(f" • Document chunks: {len([c for c in all_chunks_above_gold if c['node_type'] == 'document'])}\n")
                f.write(f" • Table chunks: {len([c for c in all_chunks_above_gold if c['node_type'] == 'table'])}\n\n")
                
                f.write(" AVERAGE METRICS OF CHUNKS OUTRANKING GOLD:\n")
                f.write(f" • Avg Weighted Score: {np.mean([c['weighted_score'] for c in all_chunks_above_gold]):.3f}\n")
                f.write(f" • Avg Appearance Count: {np.mean([c['appearance_count'] for c in all_chunks_above_gold]):.1f}\n")
                f.write(f" • Avg Content Similarity: {np.mean([c['content_similarity'] for c in all_chunks_above_gold]):.3f}\n")
                f.write(f" • Avg Entity Matches: {np.mean([c['total_entity_matches'] for c in all_chunks_above_gold]):.1f}\n")
                f.write(f" • Avg Event Matches: {np.mean([c['total_event_matches'] for c in all_chunks_above_gold]):.1f}\n\n")
            
            f.write(" SIMILARITY FACTOR COMPARISON (Gold vs Non-Gold):\n")
            for metric, analysis in comparison_analysis.items():
                if 'gold_average' in analysis:
                    advantage = "" if analysis['gold_advantage'] else ""
                    f.write(f" {advantage} {metric.replace('_', ' ').title()}:\n")
                    f.write(f" Gold: {analysis['gold_average']:.3f} | Non-Gold: {analysis['non_gold_average']:.3f}\n")
                    f.write(f" Difference: {analysis['difference']:+.3f} | Ratio: {analysis['ratio']:.2f}\n")
            
            f.write("\n KEY INSIGHTS FOR BOOSTING GOLD RANKING:\n")
            
            # Identify which factors favor gold vs non-gold
            gold_advantages = []
            gold_disadvantages = []
            
            for metric, analysis in comparison_analysis.items():
                if 'gold_advantage' in analysis and metric != 'appearance_count': # Skip appearance_count
                    if analysis['gold_advantage']:
                        gold_advantages.append(f"{metric} (Gold: {analysis['gold_average']:.3f} vs Non-Gold: {analysis['non_gold_average']:.3f})")
                    else:
                        gold_disadvantages.append(f"{metric} (Gold: {analysis['gold_average']:.3f} vs Non-Gold: {analysis['non_gold_average']:.3f})")
            
            if gold_advantages:
                f.write(" Gold documents perform BETTER in:\n")
                for advantage in gold_advantages:
                    f.write(f" - {advantage}\n")
            
            if gold_disadvantages:
                f.write(" Gold documents perform WORSE in:\n")
                for disadvantage in gold_disadvantages:
                    f.write(f" - {disadvantage}\n")
            
            f.write("\n RECOMMENDATIONS FOR IMPROVING GOLD RANKING:\n")
            if gold_disadvantages:
                f.write(" 1. Focus on improving gold performance in weak areas:\n")
                for disadvantage in gold_disadvantages[:3]: # Top 3
                    metric_name = disadvantage.split(' (')[0]
                    f.write(f" - Boost {metric_name.replace('_', ' ')} weighting in ranking formula\n")
            
            if all_chunks_above_gold:
                neighbor_ratio = neighbor_count / len(all_chunks_above_gold) if all_chunks_above_gold else 0
                if neighbor_ratio > 0.5:
                    f.write(" 2. Many neighbor chunks outrank gold - consider tighter neighbor filtering\n")
                
                retrieved_ratio = retrieved_count / len(all_chunks_above_gold) if all_chunks_above_gold else 0
                if retrieved_ratio > 0.3:
                    f.write(" 3. Retrieved chunks often outrank gold - investigate retrieval quality\n")
        
        logger.info(f"Saved ranking factor analysis to {chunks_above_gold_file}")
        logger.info(f"Saved factor comparison to {factor_comparison_file}")
        logger.info(f"Saved insights report to {insights_file}")
    
    def run_complete_visualization(self):
        """Run the complete visualization pipeline"""
        logger.info("Starting complete visualization pipeline...")
        
        # Load results
        self.load_results()
        
        # Create individual question plots
        logger.info("Creating individual question plots...")
        for result in self.question_results:
            try:
                self.create_individual_question_plot(result)
            except Exception as e:
                logger.error(f"Error creating plot for question {result.question_id}: {e}")
                continue
        
        # Create aggregated plots
        self.create_aggregated_plots()
        
        # Generate detailed statistics
        stats = self.create_detailed_statistics()
        
        # Generate summary report
        self.generate_summary_report(stats)
        
        # Generate ranking factor analysis
        self.analyze_ranking_factors()
        
        logger.info("Complete visualization pipeline finished!")

def main():
    """Main execution function"""
    visualizer = FocusedVisualizationSystem()
    visualizer.run_complete_visualization()

if __name__ == "__main__":
    main()