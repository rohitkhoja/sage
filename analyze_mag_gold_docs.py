#!/usr/bin/env python3
"""
MAG Gold Documents Analysis Script
Analyzes which types of MAG nodes (authors, papers, institutions, fields) are used as gold documents
"""

import json
import pandas as pd
import ast
from collections import defaultdict, Counter
from typing import Dict, List, Set, Any
import numpy as np

def load_mag_data():
    """Load MAG dataset and create object-ID-to-type mappings"""
    print("📥 Loading MAG dataset...")
    
    mag_file = "/shared/khoja/CogComp/datasets/MAG/data_with_citations.json"
    
    with open(mag_file, 'r', encoding='utf-8') as f:
        mag_data = json.load(f)
    
    # Create mappings from object ID to object info (gold_docs are object IDs, not ranks)
    obj_id_to_info = {}
    type_counts = defaultdict(int)
    type_id_ranges = defaultdict(list)
    
    for obj_id, obj_data in mag_data.items():
        obj_type = obj_data.get('type')
        obj_id_int = int(obj_id)
        
        if obj_type == 'author':
            obj_id_to_info[obj_id_int] = {
                'type': 'author',
                'name': obj_data.get('DisplayName', ''),
                'institution': obj_data.get('institution', ''),
                'paper_count': obj_data.get('PaperCount', 0),
                'citation_count': obj_data.get('CitationCount', 0),
                'rank': obj_data.get('Rank', 0)
            }
            type_counts['author'] += 1
            type_id_ranges['author'].append(obj_id_int)
        
        elif obj_type == 'paper':
            obj_id_to_info[obj_id_int] = {
                'type': 'paper',
                'title': obj_data.get('title', ''),
                'year': obj_data.get('Year', 0),
                'doc_type': obj_data.get('DocType', ''),
                'citation_count': obj_data.get('PaperCitationCount', 0),
                'reference_count': obj_data.get('ReferenceCount', 0),
                'rank': obj_data.get('PaperRank', 0)
            }
            type_counts['paper'] += 1
            type_id_ranges['paper'].append(obj_id_int)
        
        elif obj_type == 'institution':
            obj_id_to_info[obj_id_int] = {
                'type': 'institution',
                'name': obj_data.get('DisplayName', ''),
                'paper_count': obj_data.get('PaperCount', 0),
                'citation_count': obj_data.get('CitationCount', 0),
                'rank': obj_data.get('Rank', 0)
            }
            type_counts['institution'] += 1
            type_id_ranges['institution'].append(obj_id_int)
        
        elif obj_type == 'field_of_study':
            obj_id_to_info[obj_id_int] = {
                'type': 'field_of_study',
                'name': obj_data.get('DisplayName', ''),
                'level': obj_data.get('Level', 0),
                'paper_count': obj_data.get('PaperCount', 0),
                'citation_count': obj_data.get('CitationCount', 0),
                'rank': obj_data.get('Rank', 0)
            }
            type_counts['field_of_study'] += 1
            type_id_ranges['field_of_study'].append(obj_id_int)
    
    # Calculate ID ranges for each type
    id_ranges = {}
    for obj_type, ids in type_id_ranges.items():
        if ids:
            id_ranges[obj_type] = {
                'min': min(ids),
                'max': max(ids),
                'count': len(ids)
            }
    
    print(f"✅ Loaded MAG data with {len(obj_id_to_info):,} objects")
    print(f"   Authors: {type_counts['author']:,}")
    print(f"   Papers: {type_counts['paper']:,}")
    print(f"   Institutions: {type_counts['institution']:,}")
    print(f"   Fields: {type_counts['field_of_study']:,}")
    
    return obj_id_to_info, id_ranges

def analyze_csv_file(csv_file: str, obj_id_to_info: Dict, id_ranges: Dict):
    """Analyze a single CSV file for gold document types"""
    print(f"\n📊 Analyzing {csv_file}")
    print("-" * 60)
    
    # Load CSV
    df = pd.read_csv(csv_file)
    print(f"📈 Total queries: {len(df):,}")
    
    # Statistics containers
    gold_doc_stats = defaultdict(list)
    type_distribution = Counter()
    missing_ids = []
    obj_id_type_mapping = {}
    
    # Analyze each query
    for idx, row in df.iterrows():
        gold_docs_str = row['gold_docs']
        query = row['query']
        
        try:
            # Parse gold_docs string to list
            gold_docs = ast.literal_eval(gold_docs_str)
            
            if not isinstance(gold_docs, list):
                continue
            
            # Analyze each gold document (these are object IDs, not ranks)
            query_types = []
            for obj_id in gold_docs:
                if obj_id in obj_id_to_info:
                    obj_info = obj_id_to_info[obj_id]
                    obj_type = obj_info['type']
                    query_types.append(obj_type)
                    type_distribution[obj_type] += 1
                    obj_id_type_mapping[obj_id] = obj_type
                else:
                    missing_ids.append(obj_id)
            
            # Store query-level statistics
            if query_types:
                unique_types = list(set(query_types))
                gold_doc_stats['query_types'].append(unique_types)
                gold_doc_stats['num_gold_docs'].append(len(gold_docs))
                gold_doc_stats['num_unique_types'].append(len(unique_types))
                gold_doc_stats['is_mixed_types'].append(len(unique_types) > 1)
        
        except (ValueError, SyntaxError) as e:
            print(f"Warning: Could not parse gold_docs for query {idx}: {gold_docs_str}")
            continue
    
    # Calculate statistics
    total_gold_docs = sum(type_distribution.values())
    missing_count = len(missing_ids)
    
    print(f"📋 GOLD DOCUMENT ANALYSIS")
    print(f"   Total gold documents: {total_gold_docs:,}")
    print(f"   Missing object IDs: {missing_count:,}")
    if total_gold_docs > 0:
        print(f"   Coverage: {((total_gold_docs - missing_count) / total_gold_docs * 100):.1f}%")
    else:
        print(f"   Coverage: 0.0% (no valid gold documents found)")
    
    print(f"\n📊 TYPE DISTRIBUTION")
    if total_gold_docs > 0:
        for obj_type, count in type_distribution.most_common():
            percentage = (count / total_gold_docs) * 100
            print(f"   {obj_type:15}: {count:6,} ({percentage:5.1f}%)")
    else:
        print("   No valid gold documents found")
    
    # Query-level statistics
    if gold_doc_stats['query_types']:
        print(f"\n📝 QUERY-LEVEL STATISTICS")
        print(f"   Average gold docs per query: {np.mean(gold_doc_stats['num_gold_docs']):.1f}")
        print(f"   Average unique types per query: {np.mean(gold_doc_stats['num_unique_types']):.1f}")
        print(f"   Queries with mixed types: {sum(gold_doc_stats['is_mixed_types']):,} ({np.mean(gold_doc_stats['is_mixed_types'])*100:.1f}%)")
        
        # Type combination analysis
        type_combinations = Counter()
        for query_types in gold_doc_stats['query_types']:
            if len(query_types) == 1:
                type_combinations[f"Only {query_types[0]}"] += 1
            else:
                type_combinations[f"Mixed: {', '.join(sorted(query_types))}"] += 1
        
        print(f"\n🔀 TYPE COMBINATIONS")
        for combo, count in type_combinations.most_common(10):
            percentage = (count / len(gold_doc_stats['query_types'])) * 100
            print(f"   {combo:30}: {count:4,} ({percentage:5.1f}%)")
    
    # Missing IDs analysis
    if missing_ids:
        print(f"\n❌ MISSING OBJECT IDS ANALYSIS")
        print(f"   Total missing: {len(missing_ids):,}")
        
        # Check which type ranges the missing IDs fall into
        missing_by_type = defaultdict(int)
        for obj_id in missing_ids:
            for obj_type, range_info in id_ranges.items():
                if range_info['min'] <= obj_id <= range_info['max']:
                    missing_by_type[obj_type] += 1
                    break
            else:
                missing_by_type['unknown'] += 1
        
        print(f"   Missing by type range:")
        for obj_type, count in missing_by_type.items():
            print(f"     {obj_type}: {count:,}")
    
    return {
        'type_distribution': type_distribution,
        'missing_ids': missing_ids,
        'obj_id_type_mapping': obj_id_type_mapping,
        'query_stats': gold_doc_stats
    }

def analyze_id_ranges(id_ranges: Dict):
    """Analyze the ID ranges for each object type"""
    print(f"\n📊 MAG OBJECT ID RANGES ANALYSIS")
    print("-" * 40)
    
    for obj_type, range_info in id_ranges.items():
        print(f"{obj_type:15}: {range_info['count']:6,} objects, IDs {range_info['min']:8,} - {range_info['max']:8,}")

def main():
    """Main execution function"""
    print("🔍 MAG Gold Documents Analysis")
    print("=" * 60)
    
    # Load MAG data
    obj_id_to_info, id_ranges = load_mag_data()
    
    # Analyze ID ranges
    analyze_id_ranges(id_ranges)
    
    # Analyze both CSV files
    csv_files = [
        "/shared/khoja/CogComp/output/BM25_stark_mag_human_rewritten.csv",
        "/shared/khoja/CogComp/output/BM25_stark_mag_test_rewritten.csv"
    ]
    
    all_results = {}
    
    for csv_file in csv_files:
        results = analyze_csv_file(csv_file, obj_id_to_info, id_ranges)
        all_results[csv_file.split('/')[-1]] = results
    
    # Combined analysis
    print(f"\n🎯 COMBINED ANALYSIS")
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
    
    print(f"\n📊 COMBINED TYPE DISTRIBUTION")
    for obj_type, count in combined_type_dist.most_common():
        percentage = (count / total_combined) * 100 if total_combined > 0 else 0
        print(f"   {obj_type:15}: {count:6,} ({percentage:5.1f}%)")
    
    # Save detailed results
    output_file = "/shared/khoja/CogComp/output/mag_gold_docs_analysis.json"
    
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
    
    print(f"\n💾 Detailed analysis saved to: {output_file}")

if __name__ == "__main__":
    main()
