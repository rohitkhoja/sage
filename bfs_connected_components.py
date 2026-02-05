"""
BFS Connected Components Analysis
Find connected components from edge data using Breadth-First Search
"""

import json
from collections import deque, defaultdict
from typing import List, Dict, Set, Any
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConnectedComponentsAnalyzer:
    """Analyze connected components using BFS from edge data"""
    
    def __init__(self, edge_data: List[Dict[str, Any]], filter_by_entity_count: bool = False):
        """
        Initialize with edge data
        
        Args:
            edge_data: List of edge dictionaries with source_chunk_id and target_chunk_id
            filter_by_entity_count: If True, only keep edges where entity_count > 0
        """
        self.original_edge_data = edge_data
        self.filter_by_entity_count = filter_by_entity_count
        
        # Filter edges based on entity_count if requested
        if filter_by_entity_count:
            self.edge_data = self._filter_edges_by_entity_count()
        else:
            self.edge_data = edge_data
            
        self.adjacency_list = defaultdict(set)
        self.visited = set()
        self.connected_components = []
        
        # Build adjacency list
        self._build_adjacency_list()
    
    def _filter_edges_by_entity_count(self) -> List[Dict[str, Any]]:
        """
        Filter edges to keep only those where entity_count > 0
        
        Returns:
            Filtered list of edges
        """
        logger.info(f"Filtering {len(self.original_edge_data)} edges by entity_count > 0...")
        
        filtered_edges = []
        for edge in self.original_edge_data:
            entity_count = edge.get('entity_count', 0)
            if entity_count > 0:
                filtered_edges.append(edge)
        
        logger.info(f"Kept {len(filtered_edges)} edges with entity_count > 0 "
                   f"(filtered out {len(self.original_edge_data) - len(filtered_edges)} edges)")
        
        return filtered_edges
    
    def _build_adjacency_list(self):
        """Build adjacency list from edge data"""
        logger.info(f"Building adjacency list from {len(self.edge_data)} edges...")
        
        for edge in self.edge_data:
            source_id = edge['source_chunk_id']
            target_id = edge['target_chunk_id']
            
            # Add bidirectional connections
            self.adjacency_list[source_id].add(target_id)
            self.adjacency_list[target_id].add(source_id)
        
        total_nodes = len(self.adjacency_list)
        logger.info(f"Created adjacency list with {total_nodes} nodes")
    
    def _bfs_component(self, start_node: str) -> Set[str]:
        """
        Perform BFS to find all nodes in the connected component starting from start_node
        
        Args:
            start_node: Starting node for BFS
            
        Returns:
            Set of all chunk IDs in this connected component
        """
        component = set()
        queue = deque([start_node])
        
        while queue:
            current_node = queue.popleft()
            
            # Skip if already visited
            if current_node in self.visited:
                continue
            
            # Mark as visited and add to component
            self.visited.add(current_node)
            component.add(current_node)
            
            # Add all neighbors to queue
            for neighbor in self.adjacency_list[current_node]:
                if neighbor not in self.visited:
                    queue.append(neighbor)
        
        return component
    
    def find_connected_components(self) -> List[List[str]]:
        """
        Find all connected components using BFS
        
        Returns:
            List of connected components, where each component is a list of chunk IDs
        """
        logger.info("Finding connected components using BFS...")
        
        # Reset visited set and components list
        self.visited = set()
        self.connected_components = []
        
        # Iterate through all nodes
        for node in self.adjacency_list:
            if node not in self.visited:
                # Find connected component starting from this node
                component = self._bfs_component(node)
                
                # Convert to sorted list for consistency
                component_list = sorted(list(component))
                self.connected_components.append(component_list)
                
                logger.info(f"Found connected component with {len(component_list)} nodes")
        
        logger.info(f"Found {len(self.connected_components)} connected components total")
        return self.connected_components
    
    def get_component_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the connected components
        
        Returns:
            Dictionary with statistics
        """
        if not self.connected_components:
            return {}
        
        component_sizes = [len(comp) for comp in self.connected_components]
        
        stats = {
            'total_components': len(self.connected_components),
            'total_nodes': sum(component_sizes),
            'largest_component_size': max(component_sizes),
            'smallest_component_size': min(component_sizes),
            'average_component_size': sum(component_sizes) / len(component_sizes),
            'component_size_distribution': {}
        }
        
        # Component size distribution
        size_counts = defaultdict(int)
        for size in component_sizes:
            size_counts[size] += 1
        
        stats['component_size_distribution'] = dict(size_counts)
        
        return stats
    
    def save_components_to_file(self, output_file: str):
        """
        Save connected components to JSON file
        
        Args:
            output_file: Path to output file
        """
        output_data = {
            'connected_components': self.connected_components,
            'statistics': self.get_component_statistics()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.connected_components)} connected components to {output_file}")


def analyze_connected_components_from_file(input_file: str, output_file: str, filter_by_entity_count: bool = True):
    """
    Analyze connected components from edge data file
    
    Args:
        input_file: Path to JSON file containing edge data
        output_file: Path to output file for connected components
        filter_by_entity_count: If True, only keep edges where entity_count > 0
    """
    logger.info(f"Loading edge data from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        edge_data = json.load(f)
    
    logger.info(f"Loaded {len(edge_data)} edges")
    
    # Analyze connected components with filtering
    analyzer = ConnectedComponentsAnalyzer(edge_data, filter_by_entity_count=filter_by_entity_count)
    connected_components = analyzer.find_connected_components()
    
    # Print statistics
    stats = analyzer.get_component_statistics()
    
    if stats:
        logger.info("=== Connected Components Statistics ===")
        logger.info(f"Total components: {stats['total_components']}")
        logger.info(f"Total nodes: {stats['total_nodes']}")
        logger.info(f"Largest component: {stats['largest_component_size']} nodes")
        logger.info(f"Smallest component: {stats['smallest_component_size']} nodes")
        logger.info(f"Average component size: {stats['average_component_size']:.2f} nodes")
        
        logger.info("Component size distribution:")
        for size, count in sorted(stats['component_size_distribution'].items()):
            logger.info(f"  Size {size}: {count} components")
    else:
        logger.warning("No connected components found - no statistics to display")
        logger.warning("This might be because all edges were filtered out or no edges exist")
    
    # Save results
    analyzer.save_components_to_file(output_file)
    
    return connected_components, stats


def main():
    """Main function for testing"""
    # Example usage
    input_file = "/shared/khoja/CogComp/output/analysis_cache/analysis_reports/outlier_analysis/section_1_doc_doc_outliers/high_content_similarity_and_topic_similarity.json"  # Replace with your actual file
    output_file = "output/connected_components.json"
    
    if Path(input_file).exists():
        components, stats = analyze_connected_components_from_file(input_file, output_file, filter_by_entity_count=True)
        logger.info("Connected components analysis completed successfully!")
    else:
        logger.error(f"Input file {input_file} not found")
        
        # Example with sample data
        sample_edges = [
            {
                "edge_id": "edge1",
                "source_chunk_id": "chunk_A",
                "target_chunk_id": "chunk_B",
                "edge_type": "doc-doc",
                "entity_count": 2
            },
            {
                "edge_id": "edge2", 
                "source_chunk_id": "chunk_B",
                "target_chunk_id": "chunk_C",
                "edge_type": "doc-doc",
                "entity_count": 1
            },
            {
                "edge_id": "edge3",
                "source_chunk_id": "chunk_D",
                "target_chunk_id": "chunk_E",
                "edge_type": "doc-table",
                "entity_count": 0
            }
        ]
        
        logger.info("Running with sample data...")
        analyzer = ConnectedComponentsAnalyzer(sample_edges, filter_by_entity_count=True)
        components = analyzer.find_connected_components()
        
        logger.info(f"Sample results: {components}")
        # Expected: [['chunk_A', 'chunk_B', 'chunk_C']] (chunk_D and chunk_E filtered out due to entity_count=0)


if __name__ == "__main__":
    main()
