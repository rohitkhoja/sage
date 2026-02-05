#!/usr/bin/env python3
"""
MAG Dataset Analysis Script
Analyzes the structure, content, and characteristics of the MAG dataset
"""

import json
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Any, Set
import statistics
import re

def analyze_mag_dataset(file_path: str, sample_size: int = 1000):
    """Analyze the MAG dataset structure and content"""
    
    print("🔍 Starting MAG Dataset Analysis")
    print("=" * 60)
    
    # Statistics containers
    type_counts = Counter()
    field_stats = defaultdict(list)
    string_lengths = defaultdict(list)
    numeric_stats = defaultdict(list)
    
    # Sample data for each type
    type_samples = defaultdict(list)
    
    # Field analysis
    all_fields = set()
    field_frequency = Counter()
    
    # String analysis
    string_fields = set()
    
    print(f"📥 Loading and analyzing dataset from: {file_path}")
    print(f"📊 Sample size per type: {sample_size}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_objects = len(data)
        print(f"📈 Total objects in dataset: {total_objects:,}")
        print()
        
        # Analyze each object
        for obj_id, obj_data in data.items():
            if not isinstance(obj_data, dict):
                continue
                
            obj_type = obj_data.get('type', 'unknown')
            type_counts[obj_type] += 1
            
            # Collect sample data
            if len(type_samples[obj_type]) < sample_size:
                type_samples[obj_type].append((obj_id, obj_data))
            
            # Analyze each field
            for field_name, field_value in obj_data.items():
                all_fields.add(field_name)
                field_frequency[field_name] += 1
                
                # Determine field type and collect statistics
                if isinstance(field_value, str):
                    string_fields.add(field_name)
                    string_lengths[field_name].append(len(field_value))
                    field_stats[field_name].append(field_value)
                elif isinstance(field_value, (int, float)):
                    numeric_stats[field_name].append(field_value)
                elif isinstance(field_value, list):
                    field_stats[field_name].append(len(field_value))  # List length
                else:
                    field_stats[field_name].append(type(field_value).__name__)
        
        # Print analysis results
        print_analysis_results(
            type_counts, all_fields, field_frequency, 
            string_lengths, numeric_stats, type_samples
        )
        
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")
        return

def print_analysis_results(type_counts, all_fields, field_frequency, 
                          string_lengths, numeric_stats, type_samples):
    """Print comprehensive analysis results"""
    
    print("📊 OBJECT TYPE DISTRIBUTION")
    print("-" * 40)
    total_objects = sum(type_counts.values())
    for obj_type, count in type_counts.most_common():
        percentage = (count / total_objects) * 100
        print(f"  {obj_type:20} {count:8,} ({percentage:5.1f}%)")
    
    print(f"\n📋 TOTAL OBJECTS: {total_objects:,}")
    print()
    
    print("🔧 FIELD ANALYSIS")
    print("-" * 40)
    print(f"Total unique fields: {len(all_fields)}")
    print(f"Most common fields:")
    for field, count in field_frequency.most_common(15):
        percentage = (count / total_objects) * 100
        print(f"  {field:30} {count:8,} ({percentage:5.1f}%)")
    
    print("\n📝 STRING FIELD ANALYSIS")
    print("-" * 40)
    for field in sorted(string_lengths.keys()):
        lengths = string_lengths[field]
        if lengths:
            avg_length = statistics.mean(lengths)
            median_length = statistics.median(lengths)
            max_length = max(lengths)
            min_length = min(lengths)
            print(f"  {field:30} avg: {avg_length:6.1f} median: {median_length:6.1f} range: {min_length}-{max_length}")
    
    print("\n🔢 NUMERIC FIELD ANALYSIS")
    print("-" * 40)
    for field in sorted(numeric_stats.keys()):
        values = numeric_stats[field]
        if values:
            avg_val = statistics.mean(values)
            median_val = statistics.median(values)
            max_val = max(values)
            min_val = min(values)
            print(f"  {field:30} avg: {avg_val:10.1f} median: {median_val:10.1f} range: {min_val}-{max_val}")
    
    print("\n📋 DETAILED OBJECT TYPE EXAMPLES")
    print("-" * 40)
    
    for obj_type, samples in type_samples.items():
        if not samples:
            continue
            
        print(f"\n🔸 {obj_type.upper()} OBJECTS ({type_counts[obj_type]:,} total)")
        print("-" * 30)
        
        # Show field structure
        if samples:
            sample_obj = samples[0][1]  # First sample
            print("  Fields:")
            for field_name, field_value in sample_obj.items():
                field_type = type(field_value).__name__
                if isinstance(field_value, str):
                    preview = field_value[:50] + "..." if len(field_value) > 50 else field_value
                    print(f"    {field_name:25} ({field_type:10}): {preview}")
                elif isinstance(field_value, list):
                    print(f"    {field_name:25} ({field_type:10}): [list with {len(field_value)} items]")
                else:
                    print(f"    {field_name:25} ({field_type:10}): {field_value}")
        
        # Show 2-3 examples
        print("\n  Examples:")
        for i, (obj_id, obj_data) in enumerate(samples[:3]):
            print(f"    Example {i+1} (ID: {obj_id}):")
            for field_name, field_value in obj_data.items():
                if isinstance(field_value, str) and len(field_value) > 100:
                    preview = field_value[:100] + "..."
                else:
                    preview = field_value
                print(f"      {field_name}: {preview}")

def analyze_citations_and_relationships(file_path: str):
    """Analyze citation relationships and connections"""
    
    print("\n🔗 CITATION AND RELATIONSHIP ANALYSIS")
    print("-" * 40)
    
    citation_fields = []
    relationship_fields = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Sample first 1000 objects to identify citation/relationship fields
            sample_count = 0
            for line in f:
                if sample_count >= 1000:
                    break
                
                # Parse JSON object
                try:
                    # Find the start and end of a JSON object
                    if line.strip().startswith('{'):
                        obj_data = json.loads(line.strip().rstrip(','))
                        
                        # Look for citation/relationship fields
                        for field_name, field_value in obj_data.items():
                            if 'citation' in field_name.lower():
                                citation_fields.append(field_name)
                            elif any(rel_word in field_name.lower() for rel_word in ['author', 'reference', 'cited', 'citing']):
                                relationship_fields.append(field_name)
                        
                        sample_count += 1
                except:
                    continue
        
        print(f"Citation-related fields found: {set(citation_fields)}")
        print(f"Relationship-related fields found: {set(relationship_fields)}")
        
    except Exception as e:
        print(f"❌ Error in relationship analysis: {e}")

def main():
    """Main execution function"""
    
    file_path = "/shared/khoja/CogComp/datasets/MAG/data_with_citations.json"
    
    # Basic structure analysis
    analyze_mag_dataset(file_path, sample_size=500)
    
    # Citation analysis
    analyze_citations_and_relationships(file_path)

if __name__ == "__main__":
    main()
