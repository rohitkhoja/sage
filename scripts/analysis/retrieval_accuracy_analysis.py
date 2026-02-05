#!/usr/bin/env python3
"""
Retrieval Accuracy Analysis

This script analyzes how retrieval accuracy improves as we increase k from 1 to 100.
It extracts gold documents from the CSV and checks if at least one gold doc appears 
in the top-k retrieved documents.
"""

import pandas as pd
import ast
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
import re

def extract_gold_docs(gold_str: str) -> List[str]:
    """
    Extract gold documents from string format.
    Handles both string representations of lists and actual lists.
    """
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

def calculate_accuracy_at_k(df: pd.DataFrame, k: int) -> float:
    """
    Calculate accuracy@k: percentage of questions where at least one gold doc
    appears in the top-k retrieved documents.
    """
    correct = 0
    total = 0
    
    for idx, row in df.iterrows():
        # Extract gold docs
        gold_docs = extract_gold_docs(row['gold_docs'])
        if not gold_docs:
            continue
        
        # Get top-k retrieved docs
        retrieved_docs = []
        for i in range(1, min(k + 1, 101)): # k is 1-indexed, columns are 1st, 2nd, etc.
            col_name = f"{i}st" if i == 1 else f"{i}nd" if i == 2 else f"{i}rd" if i == 3 else f"{i}th"
            if col_name in row and pd.notna(row[col_name]):
                retrieved_docs.append(row[col_name])
        
        # Check if any gold doc appears in top-k
        found = False
        for gold_doc in gold_docs:
            if gold_doc in retrieved_docs[:k]:
                found = True
                break
        
        if found:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0

def analyze_retrieval_accuracy(csv_path: str):
    """
    Main function to analyze retrieval accuracy from k=1 to k=100.
    """
    print("Loading CSV file...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} questions")
    
    # Test gold doc extraction on a few samples
    print("\nTesting gold doc extraction on first 5 rows:")
    for i in range(min(5, len(df))):
        gold_str = df.iloc[i]['gold_docs']
        extracted = extract_gold_docs(gold_str)
        print(f"Row {i+1}: {gold_str[:100]}... -> {len(extracted)} docs: {extracted}")
    
    # Calculate accuracy for k=1 to k=100
    print("\nCalculating accuracy@k for k=1 to k=100...")
    k_values = list(range(1, 101))
    accuracies = []
    
    for k in k_values:
        accuracy = calculate_accuracy_at_k(df, k)
        accuracies.append(accuracy)
        if k % 10 == 0 or k <= 5:
            print(f"Accuracy@{k}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Save results
    results = {
        'k_values': k_values,
        'accuracies': accuracies,
        'total_questions': len(df),
        'summary_stats': {
            'accuracy_at_1': accuracies[0],
            'accuracy_at_5': accuracies[4],
            'accuracy_at_10': accuracies[9],
            'accuracy_at_20': accuracies[19],
            'accuracy_at_50': accuracies[49],
            'accuracy_at_100': accuracies[99]
        }
    }
    
    with open('retrieval_accuracy_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to 'retrieval_accuracy_results.json'")
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.plot(k_values, [acc * 100 for acc in accuracies], 'b-', linewidth=2, marker='o', markersize=3)
    plt.xlabel('k (Number of Retrieved Documents)', fontsize=12)
    plt.ylabel('Accuracy@k (%)', fontsize=12)
    plt.title('Retrieval Accuracy vs k\n(Percentage of questions with at least one gold doc in top-k)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(1, 100)
    plt.ylim(0, 100)
    
    # Add annotations for key points
    key_points = [1, 5, 10, 20, 50, 100]
    for k in key_points:
        acc = accuracies[k-1] * 100
        plt.annotate(f'k={k}: {acc:.1f}%', 
                    xy=(k, acc), 
                    xytext=(k+5, acc+2),
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig('retrieval_accuracy_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Plot saved as 'retrieval_accuracy_plot.png'")
    
    # Print summary
    print("\n" + "="*60)
    print("RETRIEVAL ACCURACY ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total Questions Analyzed: {len(df)}")
    print(f"Accuracy@1: {results['summary_stats']['accuracy_at_1']:.4f} ({results['summary_stats']['accuracy_at_1']*100:.2f}%)")
    print(f"Accuracy@5: {results['summary_stats']['accuracy_at_5']:.4f} ({results['summary_stats']['accuracy_at_5']*100:.2f}%)")
    print(f"Accuracy@10: {results['summary_stats']['accuracy_at_10']:.4f} ({results['summary_stats']['accuracy_at_10']*100:.2f}%)")
    print(f"Accuracy@20: {results['summary_stats']['accuracy_at_20']:.4f} ({results['summary_stats']['accuracy_at_20']*100:.2f}%)")
    print(f"Accuracy@50: {results['summary_stats']['accuracy_at_50']:.4f} ({results['summary_stats']['accuracy_at_50']*100:.2f}%)")
    print(f"Accuracy@100: {results['summary_stats']['accuracy_at_100']:.4f} ({results['summary_stats']['accuracy_at_100']*100:.2f}%)")
    
    # Calculate improvement metrics
    improvement_5_to_1 = results['summary_stats']['accuracy_at_5'] - results['summary_stats']['accuracy_at_1']
    improvement_10_to_5 = results['summary_stats']['accuracy_at_10'] - results['summary_stats']['accuracy_at_5']
    improvement_100_to_10 = results['summary_stats']['accuracy_at_100'] - results['summary_stats']['accuracy_at_10']
    
    print(f"\nImprovement Analysis:")
    print(f"k=1 to k=5: +{improvement_5_to_1:.4f} (+{improvement_5_to_1*100:.2f}%)")
    print(f"k=5 to k=10: +{improvement_10_to_5:.4f} (+{improvement_10_to_5*100:.2f}%)")
    print(f"k=10 to k=100: +{improvement_100_to_10:.4f} (+{improvement_100_to_10*100:.2f}%)")
    
    return results

if __name__ == "__main__":
    csv_file = "/shared/khoja/CogComp/output/questions_dense_sparse_average_results1.csv"
    results = analyze_retrieval_accuracy(csv_file)
