#!/usr/bin/env python3
"""
PRIME Gold Documents Analysis Script
Analyzes which types of PRIME nodes (genes, diseases, drugs, pathways, etc.) are used as gold documents
"""

import json
import pandas as pd
import ast
from collections import defaultdict, Counter
from typing import Dict, List, Set, Any
import numpy as np

def load_prime_data():
    """Load PRIME dataset and create object-ID-to-type mappings"""
    print(" Loading PRIME dataset...")
    
    prime_file = "/shared/khoja/CogComp/datasets/PRIME/BM25/node_info.json"
    
    with open(prime_file, 'r', encoding='utf-8') as f:
        prime_data = json.load(f)
    
    # Create mappings from object ID to object info
    obj_id_to_info = {}
    type_counts = defaultdict(int)
    type_id_ranges = defaultdict(list)
    
    for obj_id, obj_data in prime_data.items():
        obj_type = obj_data.get('type')
        obj_id_int = int(obj_id)
        
        # Map all PRIME entity types
        if obj_type in ['gene/protein', 'disease', 'drug', 'pathway', 'anatomy', 
                       'biological_process', 'cellular_component', 'molecular_function',
                       'effect/phenotype', 'exposure']:
            
            obj_id_to_info[obj_id_int] = {
                'type': obj_type,
                'name': obj_data.get('name', ''),
                'source': obj_data.get('source', ''),
                'id': obj_data.get('id', obj_id_int)
            }
            
            # Add type-specific information if available
            if obj_type == 'gene/protein':
                details = obj_data.get('details', {})
                obj_id_to_info[obj_id_int].update({
                    'full_name': details.get('name', ''),
                    'summary': details.get('summary', ''),
                    'has_genomic_pos': 'genomic_pos' in details,
                    'aliases': details.get('alias', [])
                })
            
            elif obj_type == 'disease':
                details = obj_data.get('details', {})
                obj_id_to_info[obj_id_int].update({
                    'mondo_name': details.get('mondo_name', ''),
                    'mondo_definition': details.get('mondo_definition', ''),
                    'orphanet_prevalence': details.get('orphanet_prevalence', '')
                })
            
            elif obj_type == 'drug':
                details = obj_data.get('details', {})
                obj_id_to_info[obj_id_int].update({
                    'description': details.get('description', ''),
                    'indication': details.get('indication', ''),
                    'mechanism_of_action': details.get('mechanism_of_action', ''),
                    'molecular_weight': details.get('molecular_weight', 0)
                })
            
            elif obj_type == 'pathway':
                details = obj_data.get('details', {})
                obj_id_to_info[obj_id_int].update({
                    'display_name': details.get('displayName', ''),
                    'summation': details.get('summation', ''),
                    'species_name': details.get('speciesName', ''),
                    'has_diagram': details.get('hasDiagram', False)
                })
            
            type_counts[obj_type] += 1
            type_id_ranges[obj_type].append(obj_id_int)
    
    # Calculate ID ranges for each type
    id_ranges = {}
    for obj_type, ids in type_id_ranges.items():
        if ids:
            id_ranges[obj_type] = {
                'min': min(ids),
                'max': max(ids),
                'count': len(ids)
            }
    
    print(f" Loaded PRIME data with {len(obj_id_to_info):,} objects")
    for obj_type in sorted(type_counts.keys()):
        print(f" {obj_type:20}: {type_counts[obj_type]:6,}")
    
    return obj_id_to_info, id_ranges

def analyze_csv_file(csv_file: str, obj_id_to_info: Dict, id_ranges: Dict):
    """Analyze a single CSV file for gold document types"""
    print(f"\n Analyzing {csv_file}")
    print("-" * 60)
    
    # Load CSV
    df = pd.read_csv(csv_file)
    print(f" Total queries: {len(df):,}")
    
    # Statistics containers
    gold_doc_stats = defaultdict(list)
    type_distribution = Counter()
    missing_ids = []
    obj_id_type_mapping = {}
    
    # Sample queries for detailed analysis
    sample_queries = []
    
    # Analyze each query
    for idx, row in df.iterrows():
        gold_docs_str = row['gold_docs']
        query = row['query']
        query_id = row.get('query_id', idx)
        
        try:
            # Parse gold_docs string to list
            gold_docs = ast.literal_eval(gold_docs_str)
            
            if not isinstance(gold_docs, list):
                continue
            
            # Analyze each gold document
            query_types = []
            query_gold_info = []
            
            for obj_id in gold_docs:
                if obj_id in obj_id_to_info:
                    obj_info = obj_id_to_info[obj_id]
                    obj_type = obj_info['type']
                    query_types.append(obj_type)
                    type_distribution[obj_type] += 1
                    obj_id_type_mapping[obj_id] = obj_type
                    
                    # Collect detailed info for sampling
                    query_gold_info.append({
                        'id': obj_id,
                        'type': obj_type,
                        'name': obj_info.get('name', ''),
                        'source': obj_info.get('source', '')
                    })
                else:
                    missing_ids.append(obj_id)
            
            # Store query-level statistics
            if query_types:
                unique_types = list(set(query_types))
                gold_doc_stats['query_types'].append(unique_types)
                gold_doc_stats['num_gold_docs'].append(len(gold_docs))
                gold_doc_stats['num_unique_types'].append(len(unique_types))
                gold_doc_stats['is_mixed_types'].append(len(unique_types) > 1)
                
                # Collect sample queries (first 5 of each type)
                if len(sample_queries) < 20:
                    sample_queries.append({
                        'query_id': query_id,
                        'query': query,
                        'gold_docs': gold_docs,
                        'gold_types': unique_types,
                        'gold_info': query_gold_info
                    })
        
        except (ValueError, SyntaxError) as e:
            print(f"Warning: Could not parse gold_docs for query {idx}: {gold_docs_str}")
            continue
    
    # Calculate statistics
    total_gold_docs = sum(type_distribution.values())
    missing_count = len(missing_ids)
    
    print(f" GOLD DOCUMENT ANALYSIS")
    print(f" Total gold documents: {total_gold_docs:,}")
    print(f" Missing object IDs: {missing_count:,}")
    if total_gold_docs > 0:
        print(f" Coverage: {((total_gold_docs - missing_count) / total_gold_docs * 100):.1f}%")
    else:
        print(f" Coverage: 0.0% (no valid gold documents found)")
    
    print(f"\n TYPE DISTRIBUTION")
    if total_gold_docs > 0:
        for obj_type, count in type_distribution.most_common():
            percentage = (count / total_gold_docs) * 100
            print(f" {obj_type:20}: {count:6,} ({percentage:5.1f}%)")
    else:
        print(" No valid gold documents found")
    
    # Query-level statistics
    if gold_doc_stats['query_types']:
        print(f"\n QUERY-LEVEL STATISTICS")
        print(f" Average gold docs per query: {np.mean(gold_doc_stats['num_gold_docs']):.1f}")
        print(f" Average unique types per query: {np.mean(gold_doc_stats['num_unique_types']):.1f}")
        print(f" Queries with mixed types: {sum(gold_doc_stats['is_mixed_types']):,} ({np.mean(gold_doc_stats['is_mixed_types'])*100:.1f}%)")
        
        # Type combination analysis
        type_combinations = Counter()
        for query_types in gold_doc_stats['query_types']:
            if len(query_types) == 1:
                type_combinations[f"Only {query_types[0]}"] += 1
            else:
                type_combinations[f"Mixed: {', '.join(sorted(query_types))}"] += 1
        
        print(f"\n TYPE COMBINATIONS")
        for combo, count in type_combinations.most_common(10):
            percentage = (count / len(gold_doc_stats['query_types'])) * 100
            print(f" {combo:40}: {count:4,} ({percentage:5.1f}%)")
    
    # Missing IDs analysis
    if missing_ids:
        print(f"\n MISSING OBJECT IDS ANALYSIS")
        print(f" Total missing: {len(missing_ids):,}")
        
        # Check which type ranges the missing IDs fall into
        missing_by_type = defaultdict(int)
        for obj_id in missing_ids:
            for obj_type, range_info in id_ranges.items():
                if range_info['min'] <= obj_id <= range_info['max']:
                    missing_by_type[obj_type] += 1
                    break
            else:
                missing_by_type['unknown'] += 1
        
        print(f" Missing by type range:")
        for obj_type, count in missing_by_type.items():
            print(f" {obj_type}: {count:,}")
    
    return {
        'type_distribution': type_distribution,
        'missing_ids': missing_ids,
        'obj_id_type_mapping': obj_id_type_mapping,
        'query_stats': gold_doc_stats,
        'sample_queries': sample_queries
    }

def analyze_sample_queries(sample_queries: List[Dict], obj_id_to_info: Dict):
    """Analyze sample queries in detail"""
    print(f"\n SAMPLE QUERIES ANALYSIS")
    print("-" * 50)
    
    for i, query_info in enumerate(sample_queries[:10]): # Show first 10
        print(f"\n Sample Query {i+1} (ID: {query_info['query_id']}):")
        print(f" Query: {query_info['query'][:100]}...")
        print(f" Gold Types: {query_info['gold_types']}")
        print(f" Gold Documents ({len(query_info['gold_docs'])}):")
        
        for gold_info in query_info['gold_info']:
            obj_id = gold_info['id']
            if obj_id in obj_id_to_info:
                obj_data = obj_id_to_info[obj_id]
                print(f" ID {obj_id:6} ({gold_info['type']:15}): {gold_info['name'][:60]}")
                
                # Show additional details for specific types
                if gold_info['type'] == 'gene/protein':
                    summary = obj_data.get('summary', '')
                    if summary:
                        print(f" Summary: {summary[:80]}...")
                elif gold_info['type'] == 'disease':
                    mondo_def = obj_data.get('mondo_definition', '')
                    if mondo_def:
                        print(f" Definition: {mondo_def[:80]}...")
                elif gold_info['type'] == 'drug':
                    indication = obj_data.get('indication', '')
                    if indication and isinstance(indication, str):
                        print(f" Indication: {indication[:80]}...")
                elif gold_info['type'] == 'pathway':
                    summation = obj_data.get('summation', '')
                    if summation:
                        print(f" Description: {summation[:80]}...")
            else:
                print(f" ID {obj_id:6} (MISSING)")

def analyze_id_ranges(id_ranges: Dict):
    """Analyze the ID ranges for each object type"""
    print(f"\n PRIME OBJECT ID RANGES ANALYSIS")
    print("-" * 40)
    
    for obj_type, range_info in id_ranges.items():
        print(f"{obj_type:20}: {range_info['count']:6,} objects, IDs {range_info['min']:8,} - {range_info['max']:8,}")

def analyze_query_types(sample_queries: List[Dict]):
    """Analyze the types of questions asked"""
    print(f"\n QUERY TYPE ANALYSIS")
    print("-" * 40)
    
    query_patterns = Counter()
    question_words = Counter()
    
    for query_info in sample_queries:
        query = query_info['query'].lower()
        
        # Analyze question patterns
        if 'which' in query:
            query_patterns['Which questions'] += 1
        if 'what' in query:
            query_patterns['What questions'] += 1
        if 'how' in query:
            query_patterns['How questions'] += 1
        if 'why' in query:
            query_patterns['Why questions'] += 1
        if 'can you' in query or 'could you' in query:
            query_patterns['Request questions'] += 1
        
        # Analyze question words
        words = query.split()
        for word in words:
            if word in ['which', 'what', 'how', 'why', 'where', 'when', 'who']:
                question_words[word] += 1
    
    print("Question patterns:")
    for pattern, count in query_patterns.most_common():
        print(f" {pattern:20}: {count:3}")
    
    print("\nMost common question words:")
    for word, count in question_words.most_common(5):
        print(f" {word:10}: {count:3}")

def main():
    """Main execution function"""
    print(" PRIME Gold Documents Analysis")
    print("=" * 60)
    
    # Load PRIME data
    obj_id_to_info, id_ranges = load_prime_data()
    
    # Analyze ID ranges
    analyze_id_ranges(id_ranges)
    
    # Analyze both CSV files
    csv_files = [
        "/shared/khoja/CogComp/datasets/PRIME/BM25/BM25_stark_prime_human_rewritten.csv",
        "/shared/khoja/CogComp/datasets/PRIME/BM25/BM25_stark_prime_test_rewritten.csv"
    ]
    
    all_results = {}
    all_sample_queries = []
    
    for csv_file in csv_files:
        results = analyze_csv_file(csv_file, obj_id_to_info, id_ranges)
        all_results[csv_file.split('/')[-1]] = results
        all_sample_queries.extend(results['sample_queries'])
    
    # Combined analysis
    print(f"\n COMBINED ANALYSIS")
    print("-" * 40)
    
    combined_type_dist = Counter()
    combined_missing = []
    
    for results in all_results.values():
        combined_type_dist.update(results['type_distribution'])
        combined_missing.extend(results['missing_ids'])
    
    total_combined = sum(combined_type_dist.values())
    print(f"Total gold documents across both files: {total_combined:,}")
    print(f"Total missing object IDs: {len(combined_missing):,}")
    if total_combined > 0:
        print(f"Overall coverage: {((total_combined - len(combined_missing)) / total_combined * 100):.1f}%")
    else:
        print(f"Overall coverage: 0.0%")
    
    print(f"\n COMBINED TYPE DISTRIBUTION")
    for obj_type, count in combined_type_dist.most_common():
        percentage = (count / total_combined) * 100 if total_combined > 0 else 0
        print(f" {obj_type:20}: {count:6,} ({percentage:5.1f}%)")
    
    # Analyze sample queries
    if all_sample_queries:
        analyze_sample_queries(all_sample_queries, obj_id_to_info)
        analyze_query_types(all_sample_queries)
    
    # Save detailed results
    output_file = "/shared/khoja/CogComp/output/prime_gold_docs_analysis.json"
    
    analysis_summary = {
        'id_ranges': id_ranges,
        'combined_type_distribution': dict(combined_type_dist),
        'total_gold_docs': total_combined,
        'missing_ids_count': len(combined_missing),
        'coverage_percentage': ((total_combined - len(combined_missing)) / total_combined * 100) if total_combined > 0 else 0,
        'file_results': {}
    }
    
    for filename, results in all_results.items():
        analysis_summary['file_results'][filename] = {
            'type_distribution': dict(results['type_distribution']),
            'missing_ids_count': len(results['missing_ids']),
            'query_stats': {
                'avg_gold_docs': float(np.mean(results['query_stats']['num_gold_docs'])) if results['query_stats']['num_gold_docs'] else 0,
                'avg_unique_types': float(np.mean(results['query_stats']['num_unique_types'])) if results['query_stats']['num_unique_types'] else 0,
                'mixed_type_queries': sum(results['query_stats']['is_mixed_types'])
            }
        }
    
    with open(output_file, 'w') as f:
        json.dump(analysis_summary, f, indent=2)
    
    print(f"\n Detailed analysis saved to: {output_file}")

if __name__ == "__main__":
    main()
