#!/usr/bin/env python3
"""
Preprocess MAG data for memory-efficient analysis

Step 1: Trim neighbor graphs to 200 neighbors per feature
Step 2: Trim embeddings to only object_id, content, content_embedding
"""

import json
from pathlib import Path
from tqdm import tqdm
from loguru import logger

def trim_neighbor_graphs(
    input_dir: str = "/shared/khoja/CogComp/output/mag_neighbor_graph",
    output_dir: str = "/shared/khoja/CogComp/output/mag_neighbor_graph_trimmed",
    neighbors_per_feature: int = 200
):
    """Trim neighbor graphs to save only top N neighbors per feature"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f" Step 1: Trimming neighbor graphs to {neighbors_per_feature} per feature")
    logger.info(f" Input: {input_dir}")
    logger.info(f" Output: {output_dir}")
    
    neighbor_files = sorted(input_path.glob("neighbors_chunk_*.json"))
    
    total_nodes = 0
    total_neighbors_before = 0
    total_neighbors_after = 0
    
    for neighbor_file in tqdm(neighbor_files, desc=" Trimming neighbor graphs"):
        # Load chunk
        with open(neighbor_file, 'r') as f:
            chunk_data = json.load(f)
        
        trimmed_chunk = {}
        
        for object_id, node_data in chunk_data.items():
            if 'neighbors' in node_data:
                trimmed_neighbors = {}
                
                for feature_name, neighbor_list in node_data['neighbors'].items():
                    # Count before
                    total_neighbors_before += len(neighbor_list)
                    
                    # Trim to first N neighbors
                    trimmed_neighbors[feature_name] = neighbor_list[:neighbors_per_feature]
                    
                    # Count after
                    total_neighbors_after += len(trimmed_neighbors[feature_name])
                
                trimmed_chunk[object_id] = {
                    'node_type': node_data.get('node_type'),
                    'neighbors': trimmed_neighbors
                }
                total_nodes += 1
        
        # Save trimmed chunk
        output_file = output_path / neighbor_file.name
        with open(output_file, 'w') as f:
            json.dump(trimmed_chunk, f)
    
    logger.info(f" Trimmed {total_nodes:,} nodes")
    logger.info(f" Neighbors before: {total_neighbors_before:,}")
    logger.info(f" Neighbors after: {total_neighbors_after:,}")
    logger.info(f" Space saved: {(1 - total_neighbors_after/total_neighbors_before)*100:.1f}%")

def trim_embeddings(
    input_dir: str = "/shared/khoja/CogComp/output/mag_final_cache/embeddings",
    output_dir: str = "/shared/khoja/CogComp/output/mag_embeddings_trimmed"
):
    """Trim embeddings to keep only object_id, content, content_embedding"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f" Step 2: Trimming embeddings to essential fields only")
    logger.info(f" Input: {input_dir}")
    logger.info(f" Output: {output_dir}")
    
    chunk_files = sorted(input_path.glob("chunk_*.json"))
    
    total_nodes = 0
    nodes_with_content = 0
    nodes_with_embedding = 0
    
    for chunk_file in tqdm(chunk_files, desc=" Trimming embeddings"):
        # Load chunk
        with open(chunk_file, 'r') as f:
            chunk_data = json.load(f)
        
        trimmed_chunk = []
        
        for node in chunk_data:
            object_id = node.get('object_id')
            content_embedding = node.get('content_embedding', [])
            content = node.get('content')
            
            
            
            
            # Create trimmed node with only essential fields
            trimmed_node = {
                'object_id': object_id,
                'content': content,
                'content_embedding': content_embedding
            }
            
            trimmed_chunk.append(trimmed_node)
            total_nodes += 1
            
            if content:
                nodes_with_content += 1
            if content_embedding:
                nodes_with_embedding += 1
        
        # Save trimmed chunk
        output_file = output_path / chunk_file.name
        with open(output_file, 'w') as f:
            json.dump(trimmed_chunk, f)
    
    logger.info(f" Trimmed {total_nodes:,} nodes")
    logger.info(f" Nodes with content: {nodes_with_content:,}")
    logger.info(f" Nodes with embeddings: {nodes_with_embedding:,}")

def main():
    logger.info(" MAG Data Preprocessing for Memory-Efficient Analysis")
    logger.info("=" * 70)
    
    # Step 2: Trim embeddings
    trim_embeddings()
    
    logger.info("\n Preprocessing complete!")
    logger.info(" Now use the trimmed files in mag_multi_feature_analyzer.py")

if __name__ == "__main__":
    main()

