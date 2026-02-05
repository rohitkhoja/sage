#!/usr/bin/env python3
"""
Simple Connectivity Diagnosis: Quick analysis of why all components are connected
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import networkx as nx
from collections import Counter

def load_batch_results(batch_dir: str) -> pd.DataFrame:
    """Load all batch result files"""
    batch_path = Path(batch_dir)
    
    if not batch_path.exists():
        logger.error(f"Batch directory not found: {batch_dir}")
        return pd.DataFrame()
    
    batch_files = list(batch_path.glob("batch_*.csv"))
    logger.info(f"Found {len(batch_files)} batch files")
    
    if not batch_files:
        return pd.DataFrame()
    
    # Load first few files for quick analysis
    sample_files = batch_files[:5]  # Just load first 5 files for speed
    
    dfs = []
    for file in sample_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            logger.info(f"Loaded {file.name}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"Error loading {file}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total sample data: {len(combined)} rows")
    
    return combined

def analyze_thresholds(df: pd.DataFrame):
    """Quick threshold analysis"""
    logger.info("\n=== THRESHOLD ANALYSIS ===")
    
    for edge_type in ['doc-doc', 'doc-table', 'table-table']:
        edge_data = df[df['edge_type'] == edge_type]
        
        if len(edge_data) == 0:
            continue
        
        logger.info(f"\n{edge_type.upper()}: {len(edge_data)} edges")
        
        # Check relevant metrics
        if edge_type == 'doc-doc':
            metrics = ['topic_similarity', 'content_similarity', 'entity_count']
        elif edge_type == 'doc-table':
            metrics = ['column_similarity', 'topic_title_similarity', 'topic_summary_similarity', 'entity_count']
        else:
            metrics = ['column_similarity', 'title_similarity', 'description_similarity', 'entity_count']
        
        for metric in metrics:
            if metric not in edge_data.columns:
                continue
            
            values = edge_data[metric].dropna()
            if len(values) == 0:
                continue
            
            # Quick stats
            p95 = values.quantile(0.95)
            p99 = values.quantile(0.99)
            p999 = values.quantile(0.999)
            
            above_95 = (values > p95).sum()
            above_99 = (values > p99).sum()
            above_999 = (values > p999).sum()
            
            logger.info(f"  {metric}:")
            logger.info(f"    95th %ile: {p95:.4f} ({above_95} edges above)")
            logger.info(f"    99th %ile: {p99:.4f} ({above_99} edges above)")
            logger.info(f"    99.9th %ile: {p999:.4f} ({above_999} edges above)")

def test_connectivity(df: pd.DataFrame):
    """Test connectivity with different threshold strategies"""
    logger.info("\n=== CONNECTIVITY TEST ===")
    
    strategies = [
        ("Entity Only", {"entity_count": 1}),
        ("99.9th Percentile", 0.999),
        ("99.5th Percentile", 0.995),
        ("99th Percentile", 0.99),
        ("95th Percentile", 0.95)
    ]
    
    for strategy_name, config in strategies:
        logger.info(f"\nTesting: {strategy_name}")
        
        selected_edges = select_edges_with_strategy(df, config)
        
        if len(selected_edges) == 0:
            logger.info("  No edges selected")
            continue
        
        # Build graph
        G = nx.Graph()
        for _, edge in selected_edges.iterrows():
            G.add_edge(edge['source_chunk_id'], edge['target_chunk_id'])
        
        # Analyze connectivity
        components = list(nx.connected_components(G))
        component_sizes = [len(comp) for comp in components]
        
        logger.info(f"  Selected edges: {len(selected_edges)}")
        logger.info(f"  Nodes: {G.number_of_nodes()}")
        logger.info(f"  Connected components: {len(components)}")
        logger.info(f"  Largest component: {max(component_sizes) if component_sizes else 0} nodes")
        logger.info(f"  Component sizes: {Counter(component_sizes)}")

def select_edges_with_strategy(df: pd.DataFrame, config):
    """Select edges based on strategy"""
    if isinstance(config, dict):
        # Entity-only strategy
        entity_threshold = config.get("entity_count", 1)
        return df[df['entity_count'] >= entity_threshold]
    
    else:
        # Percentile strategy
        percentile = config
        selected_edges = []
        
        for edge_type in ['doc-doc', 'doc-table', 'table-table']:
            edge_data = df[df['edge_type'] == edge_type].copy()
            
            if len(edge_data) == 0:
                continue
            
            # Define metrics and apply AND logic
            if edge_type == 'doc-doc':
                metrics = ['topic_similarity', 'content_similarity']
            elif edge_type == 'doc-table':
                metrics = ['column_similarity', 'topic_title_similarity', 'topic_summary_similarity']
            else:
                metrics = ['column_similarity', 'title_similarity', 'description_similarity']
            
            # Apply AND logic: ALL metrics must be above threshold
            mask = pd.Series([True] * len(edge_data))
            
            for metric in metrics:
                if metric in edge_data.columns:
                    threshold = edge_data[metric].quantile(percentile)
                    mask = mask & (edge_data[metric] > threshold)
            
            if mask.any():
                selected_edges.append(edge_data[mask])
        
        return pd.concat(selected_edges, ignore_index=True) if selected_edges else pd.DataFrame()

def quick_data_check(df: pd.DataFrame):
    """Quick data quality check"""
    logger.info("\n=== DATA QUALITY CHECK ===")
    
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Edge types: {df['edge_type'].value_counts().to_dict()}")
    
    # Check for high similarity concentrations
    content_high = (df['content_similarity'] > 0.9).sum()
    content_very_high = (df['content_similarity'] > 0.95).sum()
    
    logger.info(f"Content similarity > 0.9: {content_high} ({content_high/len(df)*100:.1f}%)")
    logger.info(f"Content similarity > 0.95: {content_very_high} ({content_very_high/len(df)*100:.1f}%)")
    
    # Check entity matches
    entity_matches = (df['entity_count'] > 0).sum()
    logger.info(f"Edges with entity matches: {entity_matches} ({entity_matches/len(df)*100:.1f}%)")
    
    # Check for potential data issues
    if content_very_high / len(df) > 0.01:  # More than 1% very high similarity
        logger.warning("⚠️  HIGH SIMILARITY CONCENTRATION: Many pairs have content similarity > 0.95")
        logger.warning("    This suggests duplicate or very similar content in your dataset")
    
    if entity_matches / len(df) > 0.05:  # More than 5% entity matches
        logger.warning("⚠️  HIGH ENTITY MATCHING: Many pairs share entities")
        logger.warning("    This creates dense connectivity even with strict thresholds")

def main():
    """Main diagnosis function"""
    logger.info("Starting simple connectivity diagnosis...")
    
    # Load sample of batch results
    batch_dir = "/shared/khoja/CogComp/output/batch_results"
    df = load_batch_results(batch_dir)
    
    if len(df) == 0:
        logger.error("No data loaded. Check if batch results exist.")
        return
    
    # Run quick analysis
    quick_data_check(df)
    analyze_thresholds(df)
    test_connectivity(df)
    
    logger.info("\n=== RECOMMENDATIONS ===")
    logger.info("1. If content similarity is concentrated above 0.9:")
    logger.info("   → Check for duplicate documents in your dataset")
    logger.info("   → Consider using only entity-based connections")
    logger.info("")
    logger.info("2. If entity matching is too common:")
    logger.info("   → Use more specific entity types")
    logger.info("   → Require multiple entity matches (entity_count > 3)")
    logger.info("")
    logger.info("3. For better isolation:")
    logger.info("   → Use 99.9th percentile thresholds")
    logger.info("   → Consider using OR logic instead of AND logic")
    logger.info("   → Build multiple smaller graphs instead of one large graph")

if __name__ == "__main__":
    main() 