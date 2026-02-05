#!/usr/bin/env python3
"""
MAG Graph Loader with PyTorch Geometric HeteroData
Loads the MAG dataset as a heterogeneous graph with canonical MAG Object IDs
"""

import json
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import torch_geometric
from torch_geometric.data import HeteroData
from loguru import logger


class MAGGraphLoader:
    """Load MAG dataset as PyG HeteroData with MAG Object ID mapping"""
    
    def __init__(self, processed_dir: str):
        self.processed_dir = Path(processed_dir)
        self.data = HeteroData()
        
        # ID mappings: node_index <-> Global Node Index
        self.node_index_to_global: Dict[int, int] = {}
        self.global_to_node_index: Dict[int, int] = {}
        self.node_index_to_type: Dict[int, str] = {}
        self.type_to_local: Dict[str, Dict[int, int]] = defaultdict(dict) # {type: {node_index: local_idx}}
        
        # Node attributes cache (node_index -> attributes)
        self.node_attrs: Dict[int, Dict[str, Any]] = {}
        
        # Statistics
        self.stats = {
            'total_nodes': 0,
            'total_edges': 0,
            'node_type_counts': defaultdict(int),
            'edge_type_counts': defaultdict(int)
        }
    
    def load_node_mappings(self):
        """Load node type mappings and create MAG Object ID mappings"""
        logger.info(" Loading node type mappings...")
        
        # Load node type dictionary
        with open(self.processed_dir / 'node_type_dict.json', 'r') as f:
            node_type_dict = json.load(f)
        
        # Load node types array
        with open(self.processed_dir / 'node_types.json', 'r') as f:
            node_types = json.load(f)
        
        logger.info(f" Loaded {len(node_types):,} node types")
        
        # Create MAG Object ID mappings
        current_local_indices = {str(i): 0 for i in range(4)} # 4 node types
        
        for global_idx, type_code in enumerate(node_types):
            node_type = node_type_dict[str(type_code)]
            
            # Load node attributes to get node_index
            # We'll do this in load_node_attributes, for now just track type
            self.node_index_to_type[global_idx] = node_type # Temporary: using global_idx as placeholder
            self.stats['node_type_counts'][node_type] += 1
        
        logger.info(" Node type mappings loaded")
    
    def load_node_attributes(self):
        """Load node attributes from node_info.jsonl and create node_index mappings"""
        logger.info(" Loading node attributes from node_info.jsonl...")
        
        node_info_path = self.processed_dir / 'node_info.jsonl'
        if not node_info_path.exists():
            raise FileNotFoundError(f"node_info.jsonl not found at {node_info_path}")
        
        current_local_indices = {str(i): 0 for i in range(4)} # 4 node types
        
        with open(node_info_path, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num % 100000 == 0:
                    logger.info(f" Processed {line_num:,} nodes...")
                
                node_data = json.loads(line.strip())
                global_idx = node_data['node_index']
                node_index = node_data.get('node_index', global_idx) # Use node_index from data
                node_type = node_data.get('type', 'unknown')
                
                # Update mappings
                self.node_index_to_global[node_index] = global_idx
                self.global_to_node_index[global_idx] = node_index
                self.node_index_to_type[node_index] = node_type
                
                # Track local indices per type
                type_code = str(self._get_type_code(node_type))
                local_idx = current_local_indices[type_code]
                self.type_to_local[node_type][node_index] = local_idx
                current_local_indices[type_code] += 1
                
                # Store node attributes
                self.node_attrs[node_index] = node_data
                self.stats['total_nodes'] += 1
        
        logger.info(f" Loaded {self.stats['total_nodes']:,} node attributes")
        logger.info(f" Node type distribution: {dict(self.stats['node_type_counts'])}")
    
    def _get_type_code(self, node_type: str) -> int:
        """Get numeric type code for node type"""
        type_map = {
            'author': 0,
            'institution': 1, 
            'field_of_study': 2,
            'paper': 3
        }
        return type_map.get(node_type, -1)
    
    def load_edge_data(self):
        """Load edge data and create heterogeneous edge indices"""
        logger.info(" Loading edge data...")
        
        # Load edge type dictionary
        with open(self.processed_dir / 'edge_type_dict.json', 'r') as f:
            edge_type_dict = json.load(f)
        
        # Load edge index and types
        with open(self.processed_dir / 'edge_index.json', 'r') as f:
            edge_index = json.load(f)
        
        with open(self.processed_dir / 'edge_types.json', 'r') as f:
            edge_types = json.load(f)
        
        logger.info(f" Loaded {len(edge_types):,} edges")
        
        # Group edges by type and create edge indices for each relation
        edge_groups = defaultdict(list)
        
        for i, (src_idx, tgt_idx) in enumerate(zip(edge_index[0], edge_index[1])):
            edge_type_code = edge_types[i]
            edge_type_name = edge_type_dict[str(edge_type_code)]
            
            # Convert to node_index
            src_node_index = self.global_to_node_index.get(src_idx)
            tgt_node_index = self.global_to_node_index.get(tgt_idx)
            
            if src_node_index is not None and tgt_node_index is not None:
                edge_groups[edge_type_name].append((src_node_index, tgt_node_index))
                self.stats['edge_type_counts'][edge_type_name] += 1
        
        # Create HeteroData edge indices
        for edge_type, edge_list in edge_groups.items():
            if not edge_list:
                continue
                
            # Parse edge type: "author___writes___paper"
            parts = edge_type.split('___')
            if len(parts) == 3:
                src_type, relation, tgt_type = parts
                
                # Convert node_index to local indices
                src_local = []
                tgt_local = []
                
                for src_node_index, tgt_node_index in edge_list:
                    if (src_node_index in self.type_to_local[src_type] and 
                        tgt_node_index in self.type_to_local[tgt_type]):
                        src_local.append(self.type_to_local[src_type][src_node_index])
                        tgt_local.append(self.type_to_local[tgt_type][tgt_node_index])
                
                if src_local and tgt_local:
                    # Create edge index tensor
                    edge_index_tensor = torch.tensor([src_local, tgt_local], dtype=torch.long)
                    
                    # Store in HeteroData
                    relation_key = (src_type, relation, tgt_type)
                    self.data[relation_key].edge_index = edge_index_tensor
                    
                    logger.info(f" {edge_type}: {len(src_local):,} edges")
        
        self.stats['total_edges'] = sum(self.stats['edge_type_counts'].values())
        logger.info(f" Loaded {self.stats['total_edges']:,} total edges")
    
    def set_node_counts(self):
        """Set node counts for each type in HeteroData"""
        for node_type, local_indices in self.type_to_local.items():
            self.data[node_type].num_nodes = len(local_indices)
            logger.info(f" {node_type}: {len(local_indices):,} nodes")
    
    def build_graph(self):
        """Build the complete heterogeneous graph"""
        logger.info(" Building MAG heterogeneous graph...")
        
        # Load all components
        self.load_node_mappings()
        self.load_node_attributes()
        self.load_edge_data()
        self.set_node_counts()
        
        logger.info(" Graph construction complete!")
        logger.info(f" Final stats: {self.stats['total_nodes']:,} nodes, {self.stats['total_edges']:,} edges")
        
        return self.data
    
    def get_node_attributes(self, node_index: int) -> Optional[Dict[str, Any]]:
        """Get node attributes by node_index"""
        return self.node_attrs.get(node_index)
    
    def get_node_type(self, node_index: int) -> Optional[str]:
        """Get node type by node_index"""
        return self.node_index_to_type.get(node_index)
    
    def get_global_index(self, node_index: int) -> Optional[int]:
        """Get global node index by node_index"""
        return self.node_index_to_global.get(node_index)
    
    def get_node_index(self, global_idx: int) -> Optional[int]:
        """Get node_index by global node index"""
        return self.global_to_node_index.get(global_idx)
    
    def get_local_index(self, node_index: int, node_type: str) -> Optional[int]:
        """Get local index within node type by node_index"""
        return self.type_to_local[node_type].get(node_index)
    
    def save_id_mappings(self, output_dir: str):
        """Save ID mappings for reproducibility"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save mappings
        mappings = {
            'node_index_to_global': self.node_index_to_global,
            'global_to_node_index': self.global_to_node_index,
            'node_index_to_type': self.node_index_to_type,
            'type_to_local': dict(self.type_to_local),
            'stats': self.stats
        }
        
        with open(output_path / 'id_mappings.json', 'w') as f:
            json.dump(mappings, f, indent=2)
        
        logger.info(f" Saved ID mappings to {output_path / 'id_mappings.json'}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        return self.stats.copy()


def main():
    """Test the graph loader"""
    processed_dir = "/shared/khoja/CogComp/datasets/MAG/processed"
    
    logger.info(" Testing MAG Graph Loader")
    
    try:
        loader = MAGGraphLoader(processed_dir)
        graph = loader.build_graph()
        
        logger.info(" Graph loaded successfully!")
        logger.info(f" Graph info: {graph}")
        
        # Save mappings
        loader.save_id_mappings("/shared/khoja/CogComp/agent/id_maps")
        
        return True
        
    except Exception as e:
        logger.error(f" Failed to load graph: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
