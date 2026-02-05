#!/usr/bin/env python3
"""
PRIME Dataset Analysis Script
Analyzes the structure, content, and characteristics of the PRIME dataset
"""

import json
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Any, Set
import statistics
import re

def analyze_prime_dataset(file_path: str, sample_size: int = 1000):
    """Analyze the PRIME dataset structure and content"""
    
    print("🔍 Starting PRIME Dataset Analysis")
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
    
    # Biological entity analysis
    source_counts = Counter()
    detail_field_analysis = defaultdict(lambda: defaultdict(list))
    
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
            
            # Track source distribution
            source = obj_data.get('source', 'unknown')
            source_counts[source] += 1
            
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
                elif isinstance(field_value, dict):
                    # Analyze details field specifically
                    if field_name == 'details':
                        for detail_key, detail_value in field_value.items():
                            detail_field_analysis[obj_type][detail_key].append(detail_value)
                else:
                    field_stats[field_name].append(type(field_value).__name__)
        
        # Print analysis results
        print_analysis_results(
            type_counts, all_fields, field_frequency, 
            string_lengths, numeric_stats, type_samples,
            source_counts, detail_field_analysis
        )
        
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")
        return

def print_analysis_results(type_counts, all_fields, field_frequency, 
                          string_lengths, numeric_stats, type_samples,
                          source_counts, detail_field_analysis):
    """Print comprehensive analysis results"""
    
    print("📊 OBJECT TYPE DISTRIBUTION")
    print("-" * 40)
    total_objects = sum(type_counts.values())
    for obj_type, count in type_counts.most_common():
        percentage = (count / total_objects) * 100
        print(f"  {obj_type:25} {count:8,} ({percentage:5.1f}%)")
    
    print(f"\n📋 TOTAL OBJECTS: {total_objects:,}")
    print()
    
    print("🏛️ SOURCE DISTRIBUTION")
    print("-" * 40)
    for source, count in source_counts.most_common():
        percentage = (count / total_objects) * 100
        print(f"  {source:25} {count:8,} ({percentage:5.1f}%)")
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
    
    print("\n🔬 DETAILS FIELD ANALYSIS BY TYPE")
    print("-" * 40)
    for obj_type, detail_fields in detail_field_analysis.items():
        print(f"\n🔸 {obj_type.upper()}")
        print("-" * 30)
        for detail_field, values in detail_fields.items():
            if values:
                # Count non-empty values
                non_empty = [v for v in values if v and str(v).strip()]
                if non_empty:
                    print(f"  {detail_field:35} {len(non_empty):6,} / {len(values):6,} ({len(non_empty)/len(values)*100:4.1f}%)")
                    
                    # Show sample values for string fields
                    if isinstance(non_empty[0], str) and len(non_empty[0]) < 100:
                        print(f"    Sample: {non_empty[0][:80]}...")
    
    print("\n📋 COMPLETE OBJECT TYPE STRUCTURES")
    print("=" * 60)
    
    # Define the order of object types for consistent output
    object_type_order = [
        'gene/protein', 'disease', 'drug', 'pathway', 'anatomy',
        'biological_process', 'cellular_component', 'molecular_function',
        'effect/phenotype', 'exposure'
    ]
    
    for obj_type in object_type_order:
        if obj_type in type_samples and type_samples[obj_type]:
            samples = type_samples[obj_type]
            print(f"\n🔸 {obj_type.upper()} OBJECTS ({type_counts[obj_type]:,} total)")
            print("-" * 50)
            
            # Show complete JSON structure for first sample
            if samples:
                sample_obj = samples[0][1]  # First sample
                print("  Complete JSON Structure:")
                print("  " + "{" * 1)
                
                for field_name, field_value in sample_obj.items():
                    if field_name == 'details' and isinstance(field_value, dict):
                        print(f'    "{field_name}": {{')
                        for detail_key, detail_val in field_value.items():
                            if isinstance(detail_val, str):
                                preview = detail_val[:80] + "..." if len(detail_val) > 80 else detail_val
                                print(f'      "{detail_key}": "{preview}",')
                            elif isinstance(detail_val, list):
                                if detail_val and isinstance(detail_val[0], str):
                                    print(f'      "{detail_key}": ["{detail_val[0][:30]}...", ...],')
                                else:
                                    print(f'      "{detail_key}": [list with {len(detail_val)} items],')
                            elif isinstance(detail_val, dict):
                                print(f'      "{detail_key}": {{...}},')
                            else:
                                print(f'      "{detail_key}": {detail_val},')
                        print("    },")
                    elif isinstance(field_value, str):
                        preview = field_value[:60] + "..." if len(field_value) > 60 else field_value
                        print(f'    "{field_name}": "{preview}",')
                    elif isinstance(field_value, list):
                        print(f'    "{field_name}": [list with {len(field_value)} items],')
                    else:
                        print(f'    "{field_name}": {field_value},')
                print("  }")
                
                # Show 2 additional examples
                print(f"\n  Additional Examples:")
                for i, (obj_id, obj_data) in enumerate(samples[1:3]):
                    print(f"    Example {i+2} (ID: {obj_id}):")
                    print(f"      name: {obj_data.get('name', 'N/A')[:60]}...")
                    if 'details' in obj_data:
                        details = obj_data['details']
                        key_details = []
                        for key in ['description', 'summary', 'indication', 'definition', 'summation']:
                            if key in details and isinstance(details[key], str):
                                key_details.append(f"{key}: {details[key][:40]}...")
                                break
                        if key_details:
                            print(f"      {key_details[0]}")
                    print()
    
    print("\n🔗 ENTITY RELATIONSHIP ANALYSIS")
    print("-" * 50)
    analyze_entity_relationships(type_samples)

def analyze_entity_relationships(type_samples):
    """Analyze relationships between different entity types"""
    
    print("🔍 Cross-Entity Relationship Patterns:")
    print()
    
    # Analyze common fields across entity types
    common_fields = defaultdict(set)
    entity_specific_fields = defaultdict(set)
    
    for obj_type, samples in type_samples.items():
        if not samples:
            continue
            
        sample_obj = samples[0][1]
        
        # Get all fields for this entity type
        all_fields = set(sample_obj.keys())
        if 'details' in sample_obj and isinstance(sample_obj['details'], dict):
            all_fields.update(sample_obj['details'].keys())
        
        entity_specific_fields[obj_type] = all_fields
        
        # Track common fields across all entity types
        for field in all_fields:
            common_fields[field].add(obj_type)
    
    print("📊 Field Usage Across Entity Types:")
    for field, entity_types in sorted(common_fields.items()):
        if len(entity_types) > 1:  # Fields used by multiple entity types
            print(f"  {field:30} -> {', '.join(sorted(entity_types))}")
    
    print(f"\n🔗 Entity Type Connections:")
    
    # Analyze specific relationship patterns
    relationships = {
        'gene-drug': [],
        'disease-gene': [],
        'pathway-gene': [],
        'drug-pathway': [],
        'disease-drug': [],
        'anatomy-gene': [],
        'process-function': []
    }
    
    for obj_type, samples in type_samples.items():
        if not samples:
            continue
            
        for obj_id, obj_data in samples[:10]:  # Analyze first 10 samples
            details = obj_data.get('details', {})
            
            # Look for cross-references in text fields
            text_fields = []
            for field, value in details.items():
                if isinstance(value, str):
                    text_fields.append(value.lower())
            
            combined_text = ' '.join(text_fields)
            
            # Identify potential relationships
            if obj_type == 'drug':
                if any(keyword in combined_text for keyword in ['gene', 'protein', 'target']):
                    relationships['gene-drug'].append(obj_id)
                if any(keyword in combined_text for keyword in ['pathway', 'metabolism']):
                    relationships['drug-pathway'].append(obj_id)
                if any(keyword in combined_text for keyword in ['disease', 'treatment', 'indication']):
                    relationships['disease-drug'].append(obj_id)
            
            elif obj_type == 'disease':
                if any(keyword in combined_text for keyword in ['gene', 'mutation', 'genetic']):
                    relationships['disease-gene'].append(obj_id)
            
            elif obj_type == 'gene/protein':
                if any(keyword in combined_text for keyword in ['pathway', 'signaling']):
                    relationships['pathway-gene'].append(obj_id)
                if any(keyword in combined_text for keyword in ['tissue', 'organ', 'anatomy']):
                    relationships['anatomy-gene'].append(obj_id)
            
            elif obj_type in ['biological_process', 'molecular_function']:
                if any(keyword in combined_text for keyword in ['function', 'process', 'activity']):
                    relationships['process-function'].append(obj_id)
    
    # Print relationship statistics
    for rel_type, entity_list in relationships.items():
        if entity_list:
            unique_count = len(set(entity_list))
            print(f"  {rel_type:20}: {unique_count:3} entities with {rel_type.replace('-', '-')} connections")
    
    print(f"\n🧬 Biological Domain Analysis:")
    
    # Group entities by biological domains
    domains = {
        'Genomics': ['gene/protein', 'biological_process', 'molecular_function', 'cellular_component'],
        'Clinical': ['disease', 'drug', 'effect/phenotype'],
        'Systems': ['pathway', 'anatomy'],
        'Environmental': ['exposure']
    }
    
    for domain, entity_types in domains.items():
        domain_entities = sum(len(type_samples.get(et, [])) for et in entity_types)
        print(f"  {domain:15}: {domain_entities:6,} entities ({', '.join(entity_types)})")
    
    print(f"\n🔍 Key Relationship Indicators:")
    
    # Analyze specific relationship indicators
    indicators = {
        'Genomic Information': 0,
        'Clinical Information': 0,
        'Pathway Information': 0,
        'Anatomical Information': 0,
        'Pharmacological Information': 0
    }
    
    for obj_type, samples in type_samples.items():
        if not samples:
            continue
            
        for obj_id, obj_data in samples[:5]:  # Check first 5 samples
            details = obj_data.get('details', {})
            
            if obj_type == 'gene/protein':
                if 'genomic_pos' in details:
                    indicators['Genomic Information'] += 1
            
            if obj_type in ['disease', 'drug']:
                clinical_fields = ['indication', 'symptoms', 'treatment', 'mechanism_of_action']
                if any(field in details for field in clinical_fields):
                    indicators['Clinical Information'] += 1
            
            if obj_type == 'pathway':
                if 'summation' in details or 'goBiologicalProcess' in details:
                    indicators['Pathway Information'] += 1
            
            if obj_type == 'anatomy':
                indicators['Anatomical Information'] += 1
            
            if obj_type == 'drug':
                if any(field in details for field in ['molecular_weight', 'pharmacodynamics', 'protein_binding']):
                    indicators['Pharmacological Information'] += 1
    
    for indicator, count in indicators.items():
        if count > 0:
            print(f"  {indicator:25}: {count:3} entities")

def analyze_biological_relationships(file_path: str):
    """Analyze biological relationships and connections"""
    
    print("\n🧬 BIOLOGICAL RELATIONSHIP ANALYSIS")
    print("-" * 50)
    
    relationship_patterns = defaultdict(int)
    cross_type_relationships = defaultdict(int)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Analyze relationships between different biological entity types
        type_pairs = []
        
        for obj_id, obj_data in data.items():
            if not isinstance(obj_data, dict):
                continue
                
            obj_type = obj_data.get('type', 'unknown')
            
            # Look for relationship indicators in details
            if 'details' in obj_data and isinstance(obj_data['details'], dict):
                details = obj_data['details']
                
                # Check for pathway relationships
                if 'pathway' in details:
                    relationship_patterns['pathway_relationship'] += 1
                
                # Check for protein interactions
                if 'protein' in str(details).lower():
                    relationship_patterns['protein_relationship'] += 1
                
                # Check for disease associations
                if 'disease' in str(details).lower() or 'mondo' in str(details).lower():
                    relationship_patterns['disease_relationship'] += 1
        
        print("Relationship patterns found:")
        for pattern, count in relationship_patterns.items():
            print(f"  {pattern:30} {count:8,}")
        
        # Analyze genomic positions for gene relationships
        genomic_entities = 0
        pathway_entities = 0
        
        for obj_id, obj_data in data.items():
            if obj_data.get('type') == 'gene/protein':
                if 'details' in obj_data and 'genomic_pos' in obj_data['details']:
                    genomic_entities += 1
            elif obj_data.get('type') == 'pathway':
                pathway_entities += 1
        
        print(f"\nGenomic entities: {genomic_entities:,}")
        print(f"Pathway entities: {pathway_entities:,}")
        
    except Exception as e:
        print(f"❌ Error in relationship analysis: {e}")

def analyze_text_content(file_path: str):
    """Analyze text content characteristics"""
    
    print("\n📝 TEXT CONTENT ANALYSIS")
    print("-" * 40)
    
    text_fields = ['name', 'summary', 'description', 'definition']
    field_stats = defaultdict(list)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for obj_id, obj_data in data.items():
            if not isinstance(obj_data, dict):
                continue
            
            # Analyze top-level text fields
            for field in text_fields:
                if field in obj_data and isinstance(obj_data[field], str):
                    text = obj_data[field].strip()
                    if text:
                        field_stats[field].append(text)
            
            # Analyze details text fields
            if 'details' in obj_data and isinstance(obj_data['details'], dict):
                details = obj_data['details']
                for field in text_fields:
                    if field in details and isinstance(details[field], str):
                        text = details[field].strip()
                        if text:
                            field_stats[f"details.{field}"].append(text)
        
        print("Text field statistics:")
        for field, texts in field_stats.items():
            if texts:
                lengths = [len(text) for text in texts]
                avg_length = statistics.mean(lengths)
                median_length = statistics.median(lengths)
                max_length = max(lengths)
                min_length = min(lengths)
                print(f"  {field:25} {len(texts):6,} texts, avg: {avg_length:6.1f}, median: {median_length:6.1f}, range: {min_length}-{max_length}")
        
    except Exception as e:
        print(f"❌ Error in text analysis: {e}")

def main():
    """Main execution function"""
    
    file_path = "/shared/khoja/CogComp/datasets/PRIME/BM25/node_info.json"
    
    # Basic structure analysis
    analyze_prime_dataset(file_path, sample_size=500)
    
    # Biological relationship analysis
    analyze_biological_relationships(file_path)
    
    # Text content analysis
    analyze_text_content(file_path)

if __name__ == "__main__":
    main()
