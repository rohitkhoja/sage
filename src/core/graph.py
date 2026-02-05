"""
Knowledge Graph implementation with visualization support and rich metadata access
"""

import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Union, Optional, Any, Tuple
import json
import numpy as np
import pandas as pd
from loguru import logger
import umap # Add UMAP import

from .models import (
    GraphNode, DocumentChunk, TableChunk, 
    BaseEdgeMetadata, TableToTableEdgeMetadata, 
    TableToDocumentEdgeMetadata, DocumentToDocumentEdgeMetadata,
    EdgeType, ChunkType
)


class KnowledgeGraph:
    """
    Main Knowledge Graph class that supports:
    - Rich metadata storage for nodes and edges
    - Interactive visualization with Plotly
    - Efficient querying and traversal
    - Export/Import functionality
    - Graph analytics
    """
    
    def __init__(self):
        self.graph = nx.Graph()
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, BaseEdgeMetadata] = {}
        self._node_positions: Optional[Dict[str, Tuple[float, float]]] = None
        
    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph with full metadata"""
        try:
            self.nodes[node.node_id] = node
            
            # Add to NetworkX graph with attributes
            node_attrs = {
                'chunk_type': node.chunk.source_info.source_type.value,
                'source_name': node.chunk.source_info.source_name,
                'keywords': node.chunk.keywords,
                'summary': node.chunk.summary
            }
            
            self.graph.add_node(node.node_id, **node_attrs)
            # Reduced logging for performance - only log errors and bulk summaries
            
        except Exception as e:
            logger.error(f"Error adding node {node.node_id}: {e}")
            raise
    
    def add_nodes_batch(self, nodes: List[GraphNode]) -> int:
        """Add multiple nodes to the graph in batch for better performance"""
        nodes_added = 0
        failed_nodes = []
        
        try:
            for node in nodes:
                try:
                    self.nodes[node.node_id] = node
                    
                    # Add to NetworkX graph with attributes
                    node_attrs = {
                        'chunk_type': node.chunk.source_info.source_type.value,
                        'source_name': node.chunk.source_info.source_name,
                        'keywords': node.chunk.keywords,
                        'summary': node.chunk.summary
                    }
                    
                    self.graph.add_node(node.node_id, **node_attrs)
                    nodes_added += 1
                    
                except Exception as e:
                    failed_nodes.append((node.node_id, str(e)))
                    continue
            
            if failed_nodes:
                logger.warning(f"Failed to add {len(failed_nodes)} nodes: {failed_nodes[:5]}...") # Show first 5 failures
            
            return nodes_added
            
        except Exception as e:
            logger.error(f"Error in batch node addition: {e}")
            raise
    
    def add_nodes_batch_optimized(self, nodes: List[GraphNode]) -> int:
        """
        Optimized batch operation that adds all nodes at once without individual operations.
        This method is thread-safe and much faster for large batches.
        """
        if not nodes:
            return 0
            
        try:
            # Prepare all data first (no locks needed)
            nodes_to_add = {}
            node_attrs_dict = {}
            failed_nodes = []
            
            for node in nodes:
                try:
                    # Validate node
                    if not node or not node.node_id:
                        failed_nodes.append((getattr(node, 'node_id', 'unknown'), "Invalid node or node_id"))
                        continue
                        
                    # Prepare node data
                    nodes_to_add[node.node_id] = node
                    
                    # Prepare NetworkX attributes
                    node_attrs_dict[node.node_id] = {
                        'chunk_type': node.chunk.source_info.source_type.value,
                        'source_name': node.chunk.source_info.source_name,
                        'keywords': node.chunk.keywords,
                        'summary': node.chunk.summary
                    }
                    
                except Exception as e:
                    failed_nodes.append((getattr(node, 'node_id', 'unknown'), str(e)))
                    continue
            
            # Single bulk operation to add all nodes (atomic operation)
            self.nodes.update(nodes_to_add)
            
            # Add all nodes to NetworkX graph in one operation
            node_list = [(node_id, attrs) for node_id, attrs in node_attrs_dict.items()]
            self.graph.add_nodes_from(node_list)
            
            nodes_added = len(nodes_to_add)
            
            if failed_nodes:
                logger.warning(f"Failed to add {len(failed_nodes)} nodes in batch operation")
            
            logger.info(f"Successfully added {nodes_added} nodes in optimized batch operation")
            return nodes_added
            
        except Exception as e:
            logger.error(f"Error in optimized batch node addition: {e}")
            raise
    
    def add_edge(self, edge_metadata: BaseEdgeMetadata) -> None:
        """Add an edge to the graph with full metadata"""
        try:
            # Validate that both nodes exist
            if edge_metadata.source_chunk_id not in self.nodes:
                raise ValueError(f"Source node {edge_metadata.source_chunk_id} not found")
            if edge_metadata.target_chunk_id not in self.nodes:
                raise ValueError(f"Target node {edge_metadata.target_chunk_id} not found")
            
            self.edges[edge_metadata.edge_id] = edge_metadata
            
            # Add to NetworkX graph with attributes
            edge_attrs = {
                'edge_type': edge_metadata.edge_type.value,
                'semantic_similarity': edge_metadata.semantic_similarity,
                'shared_keywords': edge_metadata.shared_keywords,
                'metadata': edge_metadata.dict()
            }
            
            self.graph.add_edge(
                edge_metadata.source_chunk_id,
                edge_metadata.target_chunk_id,
                **edge_attrs
            )
            
            # Update node connections
            self.nodes[edge_metadata.source_chunk_id].connections.append(edge_metadata.target_chunk_id)
            self.nodes[edge_metadata.target_chunk_id].connections.append(edge_metadata.source_chunk_id)
            
            logger.info(f"Added edge {edge_metadata.edge_id} between {edge_metadata.source_chunk_id} and {edge_metadata.target_chunk_id}")
            
        except Exception as e:
            logger.error(f"Error adding edge {edge_metadata.edge_id}: {e}")
            raise
    
    def add_edges_batch_optimized(self, edges: List[BaseEdgeMetadata]) -> int:
        """
        Optimized batch operation that adds all edges at once without individual operations.
        This method is thread-safe and much faster for large batches.
        """
        if not edges:
            return 0
            
        try:
            # Prepare all data first (no locks needed)
            edges_to_add = {}
            nx_edges_list = []
            connection_updates = {} # Track connection updates for nodes
            failed_edges = []
            
            for edge in edges:
                try:
                    # Validate edge and nodes
                    if not edge or not edge.edge_id:
                        failed_edges.append((getattr(edge, 'edge_id', 'unknown'), "Invalid edge or edge_id"))
                        continue
                        
                    if edge.source_chunk_id not in self.nodes:
                        failed_edges.append((edge.edge_id, f"Source node {edge.source_chunk_id} not found"))
                        continue
                        
                    if edge.target_chunk_id not in self.nodes:
                        failed_edges.append((edge.edge_id, f"Target node {edge.target_chunk_id} not found"))
                        continue
                    
                    # Prepare edge data
                    edges_to_add[edge.edge_id] = edge
                    
                    # Prepare NetworkX edge data
                    edge_attrs = {
                        'edge_type': edge.edge_type.value,
                        'semantic_similarity': edge.semantic_similarity,
                        'shared_keywords': edge.shared_keywords,
                        'metadata': edge.dict()
                    }
                    
                    nx_edges_list.append((edge.source_chunk_id, edge.target_chunk_id, edge_attrs))
                    
                    # Track connection updates
                    if edge.source_chunk_id not in connection_updates:
                        connection_updates[edge.source_chunk_id] = []
                    if edge.target_chunk_id not in connection_updates:
                        connection_updates[edge.target_chunk_id] = []
                        
                    connection_updates[edge.source_chunk_id].append(edge.target_chunk_id)
                    connection_updates[edge.target_chunk_id].append(edge.source_chunk_id)
                    
                except Exception as e:
                    failed_edges.append((getattr(edge, 'edge_id', 'unknown'), str(e)))
                    continue
            
            # Single bulk operations (atomic operations)
            self.edges.update(edges_to_add)
            
            # Add all edges to NetworkX graph in one operation
            self.graph.add_edges_from(nx_edges_list)
            
            # Update node connections in batch
            for node_id, new_connections in connection_updates.items():
                if node_id in self.nodes:
                    self.nodes[node_id].connections.extend(new_connections)
            
            edges_added = len(edges_to_add)
            
            if failed_edges:
                logger.warning(f"Failed to add {len(failed_edges)} edges in batch operation")
            
            logger.info(f"Successfully added {edges_added} edges in optimized batch operation")
            return edges_added
            
        except Exception as e:
            logger.error(f"Error in optimized batch edge addition: {e}")
            raise
    
    def get_node_metadata(self, node_id: str) -> Optional[GraphNode]:
        """Get complete metadata for a node"""
        return self.nodes.get(node_id)
    
    def get_edge_metadata(self, edge_id: str) -> Optional[BaseEdgeMetadata]:
        """Get complete metadata for an edge"""
        return self.edges.get(edge_id)
    
    def get_neighbors(self, node_id: str, edge_type: Optional[EdgeType] = None) -> List[str]:
        """Get neighboring nodes, optionally filtered by edge type"""
        if node_id not in self.nodes:
            return []
        
        neighbors = []
        for neighbor in self.graph.neighbors(node_id):
            if edge_type is None:
                neighbors.append(neighbor)
            else:
                edge_data = self.graph.get_edge_data(node_id, neighbor)
                if edge_data and edge_data.get('edge_type') == edge_type.value:
                    neighbors.append(neighbor)
        
        return neighbors
    
    def find_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest path between two nodes"""
        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return None
    
    def get_subgraph(self, node_ids: List[str]) -> 'KnowledgeGraph':
        """Extract a subgraph containing only specified nodes"""
        subgraph = KnowledgeGraph()
        
        # Add nodes
        for node_id in node_ids:
            if node_id in self.nodes:
                subgraph.add_node(self.nodes[node_id])
        
        # Add edges between included nodes
        for edge_id, edge_metadata in self.edges.items():
            if (edge_metadata.source_chunk_id in node_ids and 
                edge_metadata.target_chunk_id in node_ids):
                subgraph.add_edge(edge_metadata)
        
        return subgraph
    
    def create_interactive_visualization(self, 
                                       layout: str = "umap",
                                       title: str = "Knowledge Graph",
                                       show_edge_labels: bool = True,
                                       color_by: str = "chunk_type",
                                       use_3d: bool = False,
                                       max_nodes: int = 5000,
                                       max_edges: int = 10000,
                                       use_webgl: bool = True) -> go.Figure:
        """
        Create an interactive Plotly visualization of the graph
        
        Args:
            layout: Layout algorithm ('spring', 'circular', 'random', 'shell', 'similarity', 'umap')
            title: Title for the visualization
            show_edge_labels: Whether to show edge labels
            color_by: Attribute to color nodes by ('chunk_type', 'source_name')
            use_3d: Whether to create a 3D visualization (only works with UMAP layout)
            max_nodes: Maximum nodes to display (for performance)
            max_edges: Maximum edges to display (for performance) 
            use_webgl: Use WebGL for better performance with large graphs
        """
        if len(self.nodes) == 0:
            logger.warning("No nodes in graph to visualize")
            return go.Figure()
        
        # Performance optimization: Sample nodes/edges for large graphs
        nodes_to_visualize = list(self.nodes.keys())
        edges_to_visualize = list(self.edges.keys())
        
        if len(nodes_to_visualize) > max_nodes:
            logger.info(f"Sampling {max_nodes} nodes from {len(nodes_to_visualize)} for performance")
            # Sample highest-degree nodes for better representation
            node_degrees = [(node_id, len(list(self.graph.neighbors(node_id)))) 
                           for node_id in nodes_to_visualize]
            node_degrees.sort(key=lambda x: x[1], reverse=True)
            nodes_to_visualize = [node_id for node_id, _ in node_degrees[:max_nodes]]
        
        if len(edges_to_visualize) > max_edges:
            logger.info(f"Sampling {max_edges} edges from {len(edges_to_visualize)} for performance")
            # Sample edges with highest similarity scores
            edge_similarities = [(edge_id, self.edges[edge_id].semantic_similarity) 
                               for edge_id in edges_to_visualize 
                               if self.edges[edge_id].source_chunk_id in nodes_to_visualize 
                               and self.edges[edge_id].target_chunk_id in nodes_to_visualize]
            edge_similarities.sort(key=lambda x: x[1], reverse=True)
            edges_to_visualize = [edge_id for edge_id, _ in edge_similarities[:max_edges]]

        # Calculate layout positions (only for visible nodes)
        subgraph = self.graph.subgraph(nodes_to_visualize)
        if layout == "similarity":
            pos = self._create_similarity_layout_subgraph(subgraph)
        elif layout == "umap":
            pos = self._create_umap_layout_subgraph(subgraph, use_3d=use_3d)
        elif layout == "spring":
            pos = nx.spring_layout(subgraph, k=3, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(subgraph)
        elif layout == "random":
            pos = nx.random_layout(subgraph)
        elif layout == "shell":
            pos = nx.shell_layout(subgraph)
        else:
            pos = nx.spring_layout(subgraph)
        
        self._node_positions = pos
        
        # Prepare node data (optimized)
        node_data = self._prepare_node_data_optimized(nodes_to_visualize, pos, color_by, use_3d)
        
        # Prepare edge data (optimized)
        edge_data = self._prepare_edge_data_optimized(edges_to_visualize, pos, layout, use_3d)
        
        # Create figure with WebGL for better performance
        fig = go.Figure()
        
        # Add edges (with performance optimizations)
        if edge_data['x']:
            if use_webgl and not use_3d:
                # Use Scattergl for better performance with 2D
                fig.add_trace(go.Scattergl(
                    x=edge_data['x'], y=edge_data['y'],
                    line=dict(width=1, color='rgba(125,125,125,0.3)'),
                        hoverinfo='skip',
                        mode='lines',
                    name='Edges',
                        showlegend=False
                    ))
            elif use_3d:
                fig.add_trace(go.Scatter3d(
                    x=edge_data['x'], y=edge_data['y'], z=edge_data['z'],
                    line=dict(width=1, color='rgba(125,125,125,0.3)'),
                    hoverinfo='skip',
                    mode='lines',
                    name='Edges',
                    showlegend=False
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=edge_data['x'], y=edge_data['y'],
                    line=dict(width=1, color='rgba(125,125,125,0.3)'),
                    hoverinfo='skip',
                    mode='lines',
                    name='Edges',
                    showlegend=False
                ))
        
        # Add nodes (with performance optimizations)
        if use_webgl and not use_3d:
            # Use Scattergl for better performance with 2D
            fig.add_trace(go.Scattergl(
                x=node_data['x'], y=node_data['y'],
                mode='markers+text',
                marker=dict(
                    size=node_data['sizes'],
                    color=node_data['colors'],
                    line=dict(width=1, color='white'),
                    opacity=0.9
                ),
                text=node_data['text'],
                textposition="middle center",
                textfont=dict(size=8, color='black'),
                hovertext=node_data['hover_text'],
                hoverinfo='text',
                name='Nodes',
                showlegend=False
            ))
        elif use_3d:
            fig.add_trace(go.Scatter3d(
                x=node_data['x'], y=node_data['y'], z=node_data['z'],
                mode='markers+text',
                marker=dict(
                    size=node_data['sizes'],
                    color=node_data['colors'],
                    line=dict(width=1, color='white'),
                    opacity=0.9
                ),
                text=node_data['text'],
                textposition="middle center",
                textfont=dict(size=8, color='black'),
                hovertext=node_data['hover_text'],
                hoverinfo='text',
                name='Nodes',
                showlegend=False
            ))
        else:
            fig.add_trace(go.Scatter(
                x=node_data['x'], y=node_data['y'],
                mode='markers+text',
                marker=dict(
                    size=node_data['sizes'],
                    color=node_data['colors'],
                    line=dict(width=1, color='white'),
                    opacity=0.9
                ),
                text=node_data['text'],
                textposition="middle center",
                textfont=dict(size=8, color='black'),
                hovertext=node_data['hover_text'],
                hoverinfo='text',
                name='Nodes',
                showlegend=False
            ))
        
        # Update layout with performance optimizations
        layout_update = {
            'title': dict(
                text=f"{title} (Showing {len(nodes_to_visualize)}/{len(self.nodes)} nodes)",
                font=dict(size=16)
            ),
            'showlegend': False,
            'hovermode': 'closest',
            'margin': dict(b=20,l=5,r=5,t=40),
            'plot_bgcolor': 'white',
            'dragmode': 'pan',
            'clickmode': 'event+select'
        }
        
        if use_3d:
            layout_update.update({
                'scene': dict(
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    aspectmode='cube'
                )
            })
        else:
            layout_update.update({
                'xaxis': dict(showgrid=False, zeroline=False, showticklabels=False),
                'yaxis': dict(showgrid=False, zeroline=False, showticklabels=False)
            })
        
        fig.update_layout(**layout_update)
        return fig
    
    def _prepare_node_data_optimized(self, node_ids: List[str], pos: Dict, color_by: str, use_3d: bool) -> Dict:
        """Optimized node data preparation"""
        # Pre-allocate arrays for better performance
        n_nodes = len(node_ids)
        node_x = []
        node_y = []
        node_z = [] if use_3d else None
        node_text = []
        node_colors = []
        node_sizes = []
        hover_text = []
        
        # Color mapping (pre-computed)
        if color_by == "chunk_type":
            color_map = {"document": "lightblue", "table": "lightcoral"}
        else:
            unique_values = list(set(self.graph.nodes[node].get(color_by, "unknown") 
                                   for node in node_ids))
            colors = px.colors.qualitative.Set1[:len(unique_values)]
            color_map = dict(zip(unique_values, colors))
        
        # Batch process nodes
        for node_id in node_ids:
            if node_id not in pos:
                continue
                
            if use_3d:
                x, y, z = pos[node_id]
                node_x.append(x)
                node_y.append(y)
                node_z.append(z)
            else:
                x, y = pos[node_id]
                node_x.append(x)
                node_y.append(y)
            
            node_data = self.graph.nodes[node_id]
            node_text.append(node_id[:8] + "...") # Shorter text for performance
            
            # Color based on specified attribute
            attr_value = node_data.get(color_by, "unknown")
            node_colors.append(color_map.get(attr_value, "gray"))
            
            # Use constant size for all nodes
            node_sizes.append(20) # Fixed size for all nodes (increased for visibility)
            
            # Simplified hover text for performance
            chunk = self.nodes[node_id].chunk
            hover_info = f"<b>ID:</b> {node_id}<br><b>Type:</b> {chunk.source_info.source_type.value}<br><b>Source:</b> {chunk.source_info.source_name}"
            hover_text.append(hover_info)
        
        return {
            'x': node_x,
            'y': node_y,
            'z': node_z,
            'text': node_text,
            'colors': node_colors,
            'sizes': node_sizes,
            'hover_text': hover_text
        }
    
    def _prepare_edge_data_optimized(self, edge_ids: List[str], pos: Dict, layout: str, use_3d: bool) -> Dict:
        """Optimized edge data preparation"""
        edge_x = []
        edge_y = []
        edge_z = [] if use_3d else None
        
        for edge_id in edge_ids:
            edge_metadata = self.edges[edge_id]
            source_id = edge_metadata.source_chunk_id
            target_id = edge_metadata.target_chunk_id
            
            if source_id not in pos or target_id not in pos:
                continue
                
            if use_3d:
                x0, y0, z0 = pos[source_id]
                x1, y1, z1 = pos[target_id]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_z.extend([z0, z1, None])
            else:
                x0, y0 = pos[source_id]
                x1, y1 = pos[target_id]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
        
        return {
            'x': edge_x,
            'y': edge_y,
            'z': edge_z
        }
    
    def _create_umap_layout_subgraph(self, subgraph, use_3d: bool = False) -> Dict[str, Tuple[float, float]]:
        """Create UMAP layout for subgraph"""
        node_ids = list(subgraph.nodes())
        valid_nodes = [(node_id, self.nodes[node_id]) for node_id in node_ids 
                      if node_id in self.nodes and self.nodes[node_id].chunk.embedding is not None]
        
        if not valid_nodes:
            logger.warning("No embeddings available for UMAP layout. Using spring layout.")
            return nx.spring_layout(subgraph, k=3, iterations=50)
        
        # Extract embeddings for subgraph nodes
        node_ids = [node_id for node_id, _ in valid_nodes]
        embeddings = [node.chunk.embedding for _, node in valid_nodes]
        
        # Use simplified UMAP for subgraph
        import numpy as np
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        try:
            reducer = umap.UMAP(
                n_components=3 if use_3d else 2,
                n_neighbors=min(15, len(embeddings) // 2),
                min_dist=0.1,
                metric='cosine',
                random_state=42,
                init='random'
            )
            
            positions = reducer.fit_transform(embeddings_array)
            
            if use_3d:
                return {node_id: (float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2])) 
                       for i, node_id in enumerate(node_ids)}
            else:
                return {node_id: (float(positions[i, 0]), float(positions[i, 1])) 
                       for i, node_id in enumerate(node_ids)}
        except Exception as e:
            logger.warning(f"UMAP subgraph layout failed: {e}")
            return nx.spring_layout(subgraph, k=3, iterations=50)
    
    def _create_similarity_layout_subgraph(self, subgraph) -> Dict[str, Tuple[float, float]]:
        """Create similarity layout for subgraph"""
        return self._create_similarity_layout() # Can reuse existing logic
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the graph"""
        if len(self.nodes) == 0:
            return {"message": "Empty graph"}
        
        stats = {
            "nodes": {
                "total": len(self.nodes),
                "by_type": {},
                "by_source": {}
            },
            "edges": {
                "total": len(self.edges),
                "by_type": {}
            },
            "connectivity": {
                "density": nx.density(self.graph),
                "is_connected": nx.is_connected(self.graph),
                "number_of_components": nx.number_connected_components(self.graph)
            }
        }
        
        # Node statistics
        for node in self.nodes.values():
            chunk_type = node.chunk.source_info.source_type.value
            source_name = node.chunk.source_info.source_name
            
            stats["nodes"]["by_type"][chunk_type] = stats["nodes"]["by_type"].get(chunk_type, 0) + 1
            stats["nodes"]["by_source"][source_name] = stats["nodes"]["by_source"].get(source_name, 0) + 1
        
        # Edge statistics
        for edge in self.edges.values():
            edge_type = edge.edge_type.value
            stats["edges"]["by_type"][edge_type] = stats["edges"]["by_type"].get(edge_type, 0) + 1
        
        # Additional metrics if graph is connected
        if stats["connectivity"]["is_connected"]:
            stats["connectivity"]["diameter"] = nx.diameter(self.graph)
            stats["connectivity"]["average_clustering"] = nx.average_clustering(self.graph)
        
        return stats
    
    def export_to_json(self, filepath: str) -> None:
        """Export graph to JSON format"""
        export_data = {
            "nodes": {node_id: node.dict() for node_id, node in self.nodes.items()},
            "edges": {edge_id: edge.dict() for edge_id, edge in self.edges.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Graph exported to {filepath}")
    
    @classmethod
    def import_from_json(cls, filepath: str) -> 'KnowledgeGraph':
        """Import graph from JSON format (exported by KnowledgeGraph.export_to_json) - OPTIMIZED with batch operations"""
        from .models import GraphNode, DocumentChunk, TableChunk, SourceInfo, ChunkType
        from .models import BaseEdgeMetadata, DocumentToDocumentEdgeMetadata, TableToTableEdgeMetadata, TableToDocumentEdgeMetadata, EdgeType
        
        knowledge_graph = cls()
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Loading graph from {filepath}...")
            
            # Prepare all nodes for batch import
            nodes_to_import = []
            node_data_dict = data.get("nodes", {})
            
            logger.info(f"Preparing {len(node_data_dict)} nodes for batch import...")
            
            for node_id, node_data in node_data_dict.items():
                try:
                    # Reconstruct chunk from node data
                    chunk_data = node_data["chunk"]
                    
                    # Create SourceInfo
                    source_info = SourceInfo(**chunk_data["source_info"])
                    
                    # Create appropriate chunk type
                    if source_info.source_type == ChunkType.DOCUMENT:
                        chunk = DocumentChunk(
                            chunk_id=chunk_data["chunk_id"],
                            content=chunk_data.get("content", ""),
                            embedding=chunk_data.get("embedding"),
                            source_info=source_info,
                            sentences=chunk_data.get("sentences", []),
                            keywords=chunk_data.get("keywords", []),
                            summary=chunk_data.get("summary", ""),
                            merged_sentence_count=chunk_data.get("merged_sentence_count", 1)
                        )
                    else: # TABLE
                        # Handle TableChunk with proper field mapping
                        # Use content from chunk_data, fallback to summary if content is empty
                        content = chunk_data.get("content", "")
                        if not content:
                            content = chunk_data.get("summary", "")
                        
                        # Get table-specific fields with proper defaults
                        column_headers = chunk_data.get("column_headers", [])
                        column_descriptions = chunk_data.get("column_descriptions", [])
                        rows_with_headers = chunk_data.get("rows_with_headers", [])
                        merged_row_count = chunk_data.get("merged_row_count", 1)
                        
                        # Ensure column_descriptions is a list (handle both dict and list formats)
                        if isinstance(column_descriptions, dict):
                            # Convert dict to list, preserving order of column_headers
                            column_descriptions = [column_descriptions.get(col, "") for col in column_headers]
                        elif not isinstance(column_descriptions, list):
                            column_descriptions = []
                        
                        chunk = TableChunk(
                            chunk_id=chunk_data["chunk_id"],
                            content=content,
                            embedding=chunk_data.get("embedding"),
                            source_info=source_info,
                            column_headers=column_headers,
                            column_descriptions=column_descriptions,
                            rows_with_headers=rows_with_headers,
                            keywords=chunk_data.get("keywords", []),
                            summary=chunk_data.get("summary", ""),
                            merged_row_count=merged_row_count
                        )
                    
                    # Create node
                    node = GraphNode(
                        node_id=node_id,
                        chunk=chunk,
                        connections=node_data.get("connections", [])
                    )
                    nodes_to_import.append(node)
                    
                except Exception as e:
                    logger.warning(f"Failed to prepare node {node_id}: {e}")
                    continue
            
            # Batch import all nodes
            logger.info(f"Batch importing {len(nodes_to_import)} nodes...")
            nodes_added = knowledge_graph.add_nodes_batch_optimized(nodes_to_import)
            logger.info(f"Successfully imported {nodes_added} nodes")
            
            # Prepare all edges for batch import
            edges_to_import = []
            edge_data_dict = data.get("edges", {})
            
            logger.info(f"Preparing {len(edge_data_dict)} edges for batch import...")
            
            for edge_id, edge_data in edge_data_dict.items():
                try:
                    # Create appropriate edge metadata based on edge type
                    edge_type = EdgeType(edge_data["edge_type"])
                    
                    if edge_type == EdgeType.DOCUMENT_TO_DOCUMENT:
                        edge_metadata = DocumentToDocumentEdgeMetadata(**edge_data)
                    elif edge_type == EdgeType.TABLE_TO_TABLE:
                        edge_metadata = TableToTableEdgeMetadata(**edge_data)
                    elif edge_type == EdgeType.TABLE_TO_DOCUMENT:
                        edge_metadata = TableToDocumentEdgeMetadata(**edge_data)
                    else:
                        # Fallback to base edge metadata
                        edge_metadata = BaseEdgeMetadata(**edge_data)
                    
                    edges_to_import.append(edge_metadata)
                    
                except Exception as e:
                    logger.warning(f"Failed to prepare edge {edge_id}: {e}")
                    continue
            
            # Batch import all edges
            logger.info(f"Batch importing {len(edges_to_import)} edges...")
            edges_added = knowledge_graph.add_edges_batch_optimized(edges_to_import)
            logger.info(f"Successfully imported {edges_added} edges")
            
            logger.info(f"Successfully imported optimized graph from {filepath}: {len(knowledge_graph.nodes)} nodes, {len(knowledge_graph.edges)} edges")
            return knowledge_graph
            
        except Exception as e:
            logger.error(f"Failed to import graph from {filepath}: {e}")
            raise
    
    @classmethod
    def load_from_integrated_pipeline(cls, filepath: str) -> 'KnowledgeGraph':
        """Load graph from integrated pipeline JSON format (simplified format) - OPTIMIZED with batch operations"""
        from .models import GraphNode, DocumentChunk, TableChunk, SourceInfo, ChunkType
        from .models import BaseEdgeMetadata, EdgeType
        
        knowledge_graph = cls()
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Loading graph from integrated pipeline format: {filepath}...")
            
            # Handle different file formats
            graph_data = None
            if "graph" in data and isinstance(data["graph"], dict):
                # Format from _save_comprehensive_results
                graph_data = data["graph"]
            elif "nodes" in data and "edges" in data:
                # Direct graph format
                graph_data = data
            else:
                raise ValueError("Unrecognized graph file format")
            
            # Prepare all nodes for batch import
            nodes_to_import = []
            node_data_dict = graph_data.get("nodes", {})
            
            logger.info(f"Preparing {len(node_data_dict)} nodes for batch import...")
            
            for node_id, node_data in node_data_dict.items():
                try:
                    # Create minimal chunk from simplified node data
                    source_info = SourceInfo(
                        source_id=node_id,
                        source_name=node_data.get("source", "unknown"),
                        source_type=ChunkType.DOCUMENT if node_data.get("type") == "document" else ChunkType.TABLE,
                        file_path="",
                        content=node_data.get("content", "")
                    )
                    
                    # Create simplified chunk
                    if source_info.source_type == ChunkType.DOCUMENT:
                        chunk = DocumentChunk(
                            chunk_id=node_id,
                            content=node_data.get("content", ""),
                            embedding=None, # Not available in simplified format
                            source_info=source_info,
                            sentences=[],
                            keywords=node_data.get("keywords", []),
                            summary=node_data.get("content", "")[:200] + "..." if len(node_data.get("content", "")) > 200 else node_data.get("content", ""),
                            merged_sentence_count=1
                        )
                    else:
                        # Create TableChunk with correct field names to match the TableChunk model
                        chunk = TableChunk(
                            chunk_id=node_id,
                            content=node_data.get("content", ""), # Use content field, not table_description
                            embedding=None, # Not available in simplified format
                            source_info=source_info,
                            column_headers=[], # Empty list for simplified format
                            column_descriptions=[], # List, not dict
                            rows_with_headers=[], # Correct field name
                            keywords=node_data.get("keywords", []),
                            summary=node_data.get("content", "")[:200] + "..." if len(node_data.get("content", "")) > 200 else node_data.get("content", ""),
                            merged_row_count=1 # Default value
                        )
                    
                    # Create node
                    node = GraphNode(
                        node_id=node_id,
                        chunk=chunk,
                        connections=[] # Will be populated from edges
                    )
                    nodes_to_import.append(node)
                    
                except Exception as e:
                    logger.warning(f"Failed to prepare node {node_id}: {e}")
                    continue
            
            # Batch import all nodes
            logger.info(f"Batch importing {len(nodes_to_import)} nodes...")
            nodes_added = knowledge_graph.add_nodes_batch_optimized(nodes_to_import)
            logger.info(f"Successfully imported {nodes_added} nodes")
            
            # Prepare all edges for batch import
            edges_to_import = []
            edge_data_dict = graph_data.get("edges", {})
            
            logger.info(f"Preparing {len(edge_data_dict)} edges for batch import...")
            
            for edge_id, edge_data in edge_data_dict.items():
                try:
                    source_id = edge_data.get("source")
                    target_id = edge_data.get("target")
                    similarity = edge_data.get("similarity", 0.0)
                    
                    if source_id in knowledge_graph.nodes and target_id in knowledge_graph.nodes:
                        # Create basic edge metadata
                        edge_metadata = BaseEdgeMetadata(
                            edge_id=edge_id,
                            source_chunk_id=source_id,
                            target_chunk_id=target_id,
                            edge_type=EdgeType.DOCUMENT_TO_DOCUMENT, # Default type
                            semantic_similarity=similarity,
                            shared_keywords=[]
                        )
                        
                        edges_to_import.append(edge_metadata)
                        
                except Exception as e:
                    logger.warning(f"Failed to prepare edge {edge_id}: {e}")
                    continue
            
            # Batch import all edges
            logger.info(f"Batch importing {len(edges_to_import)} edges...")
            edges_added = knowledge_graph.add_edges_batch_optimized(edges_to_import)
            logger.info(f"Successfully imported {edges_added} edges")
            
            logger.info(f"Successfully loaded optimized graph from integrated pipeline format: {len(knowledge_graph.nodes)} nodes, {len(knowledge_graph.edges)} edges")
            return knowledge_graph
            
        except Exception as e:
            logger.error(f"Failed to load graph from integrated pipeline format {filepath}: {e}")
            raise
    
    def query_by_keywords(self, keywords: List[str], min_matches: int = 1) -> List[str]:
        """Find nodes that contain specified keywords"""
        matching_nodes = []
        
        for node_id, node in self.nodes.items():
            node_keywords = set(node.chunk.keywords)
            keyword_matches = len(node_keywords.intersection(set(keywords)))
            
            if keyword_matches >= min_matches:
                matching_nodes.append(node_id)
        
        return matching_nodes
    
    def query_by_similarity(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar nodes based on embedding similarity"""
        if not query_embedding:
            return []
        
        similarities = []
        query_array = np.array(query_embedding)
        
        for node_id, node in self.nodes.items():
            if node.chunk.embedding:
                node_array = np.array(node.chunk.embedding)
                similarity = np.dot(query_array, node_array) / (
                    np.linalg.norm(query_array) * np.linalg.norm(node_array)
                )
                similarities.append((node_id, float(similarity)))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _create_similarity_layout(self) -> Dict[str, Tuple[float, float]]:
        """
        Create a layout based on similarity scores where similar nodes are closer together.
        Uses edge similarity scores as weights, treating them as distances in the layout.
        """
        if len(self.nodes) == 0:
            return {}
        
        # Create a weighted graph where edge weights represent distances
        weighted_graph = nx.Graph()
        
        # Add all nodes
        for node_id in self.nodes.keys():
            weighted_graph.add_node(node_id)
        
        # Add edges with weights as distances (1 - similarity)
        for edge_id, edge_metadata in self.edges.items():
            source = edge_metadata.source_chunk_id
            target = edge_metadata.target_chunk_id
            similarity = edge_metadata.semantic_similarity
            
            # Convert similarity to distance: higher similarity = shorter distance
            # We use (1 - similarity) so that high similarity (close to 1) becomes low distance (close to 0)
            distance = max(0.01, 1.0 - similarity) # Ensure minimum distance to avoid zero weights
            
            weighted_graph.add_edge(source, target, weight=distance)
        
        try:
            # Use spring layout with weights as ideal distances
            # The weight parameter in spring_layout represents ideal edge length
            pos = nx.spring_layout(
                weighted_graph, 
                weight='weight', # Use our distance weights
                k=2.0, # Optimal distance between nodes
                iterations=100, # More iterations for better convergence
                threshold=1e-4, # Precision threshold
                dim=2
            )
            
            logger.info("Created similarity-based layout using weighted spring layout")
            return pos
            
        except Exception as e:
            logger.warning(f"Failed to create similarity layout: {e}. Falling back to spring layout.")
            # Fallback to regular spring layout
            return nx.spring_layout(self.graph, k=3, iterations=50)
    
    def _create_umap_layout(self, use_3d: bool = False) -> Dict[str, Tuple[float, float]]:
        """
        Create a layout using UMAP dimensionality reduction on node embeddings.
        Optimized for GPU acceleration and large datasets (10k+ nodes).
        
        Args:
            use_3d: Whether to create a 3D layout (3 components instead of 2)
        """
        if len(self.nodes) == 0:
            return {}
        
        try:
            # GPU Detection with improved logging
            use_gpu = False
            use_cuml = False
            cp = None
            
            try:
                import cupy as cp_module
                cp = cp_module
                if cp.cuda.is_available():
                    logger.info(" CuPy detected - GPU acceleration available")
                    try:
                        device_count = cp.cuda.runtime.getDeviceCount()
                        current_device = cp.cuda.get_device_id()
                        logger.info(f" - GPU Devices: {device_count}")
                        logger.info(f" - Current Device: {current_device}")
                    except Exception as e:
                        logger.info(f" - GPU info unavailable: {e}")
                    
                    try:
                        import cuml
                        use_cuml = True
                        use_gpu = True
                        logger.info(" cuML available - Full GPU acceleration enabled")
                    except ImportError:
                        use_gpu = True # Use CuPy for memory operations
                        logger.info(" Using CuPy optimization (cuML not available)")
                else:
                    logger.info(" CUDA not available on CuPy")
            except ImportError:
                logger.info(" CuPy not available, using CPU UMAP")
            except Exception as e:
                logger.info(f" CuPy initialization failed: {e}, using CPU UMAP")
            
            # Extract embeddings for specified nodes only
            logger.info(f" Processing embeddings for {len(self.nodes)} visualization nodes...")
            
            valid_nodes = []
            for node_id in self.nodes.keys():
                if node_id in self.nodes and self.nodes[node_id].chunk.embedding is not None:
                    valid_nodes.append((node_id, self.nodes[node_id]))
            
            if not valid_nodes:
                logger.warning("No embeddings available for visualization nodes. Falling back to spring layout.")
                # Fallback to spring layout for these specific nodes
                subgraph = self.graph.subgraph(self.nodes.keys())
                return nx.spring_layout(subgraph, k=3, iterations=50)
            
            # Extract embeddings efficiently
            node_ids_with_embeddings = [node_id for node_id, _ in valid_nodes]
            embeddings = [node.chunk.embedding for _, node in valid_nodes]
            
            # Convert to numpy array with optimal dtype
            embeddings_array = np.array(embeddings, dtype=np.float32)
            logger.info(f" Collected {len(embeddings)} embeddings with shape {embeddings_array.shape}")
            
            # Adaptive parameters for visualization performance
            n_nodes = len(embeddings)
            
            if n_nodes > 50000:
                n_neighbors = min(50, max(15, int(np.sqrt(n_nodes))))
                min_dist = 0.01
                spread = 2.0
            elif n_nodes > 10000:
                n_neighbors = min(30, max(15, int(np.sqrt(n_nodes) / 2)))
                min_dist = 0.05
                spread = 1.5
            else:
                n_neighbors = min(15, max(5, n_nodes // 3))
                min_dist = 0.1
                spread = 1.0
            
            logger.info(f" UMAP parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, nodes={n_nodes}")
            
            # GPU-accelerated computation
            if use_gpu and use_cuml:
                logger.info(" Using GPU-accelerated UMAP (cuML + CuPy)")
                
                # Move embeddings to GPU
                embeddings_gpu = cp.asarray(embeddings_array)
                
                import cuml
                reducer = cuml.UMAP(
                    n_components=3 if use_3d else 2,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    spread=spread,
                    metric='cosine',
                    random_state=42,
                    init='random',
                    verbose=False,
                    transform_seed=42
                )
                
                # Fit and transform on GPU
                logger.info(" Computing UMAP layout on GPU...")
                positions_gpu = reducer.fit_transform(embeddings_gpu)
                positions = cp.asnumpy(positions_gpu).astype(np.float64)
                logger.info(" GPU UMAP computation completed")
                
            elif use_gpu and cp is not None:
                logger.info(" Using CuPy memory optimization with CPU UMAP")
                
                # Use CuPy for memory operations
                embeddings_gpu = cp.asarray(embeddings_array)
                
                # Normalize embeddings on GPU for better performance
                norms = cp.linalg.norm(embeddings_gpu, axis=1, keepdims=True)
                embeddings_gpu_normalized = embeddings_gpu / norms
                
                # Move back to CPU for UMAP
                embeddings_array_normalized = cp.asnumpy(embeddings_gpu_normalized).astype(np.float32)
                
                reducer = umap.UMAP(
                    n_components=3 if use_3d else 2,
                            n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            spread=spread,
                            metric='cosine',
                            random_state=42,
                            init='random',
                            verbose=False,
                            low_memory=True,
                            transform_seed=42,
                            angular_rp_forest=True,
                            set_op_mix_ratio=1.0,
                            local_connectivity=1.0,
                            repulsion_strength=1.0,
                            negative_sample_rate=5,
                            transform_queue_size=4.0
                        )
                
                logger.info(" Computing UMAP layout with CuPy optimization...")
                positions = reducer.fit_transform(embeddings_array_normalized)
                logger.info(" CuPy-optimized UMAP computation completed")
                
            else:
                logger.info(" Using CPU UMAP with optimized parameters")
                
                reducer = umap.UMAP(
                    n_components=3 if use_3d else 2,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    spread=spread,
                    metric='cosine',
                    random_state=42,
                    init='random',
                    verbose=False,
                    low_memory=True,
                    transform_seed=42,
                    angular_rp_forest=True,
                    set_op_mix_ratio=1.0,
                    local_connectivity=1.0,
                    repulsion_strength=1.0,
                    negative_sample_rate=5,
                    transform_queue_size=4.0
                )
                
                logger.info(" Computing UMAP layout on CPU...")
                positions = reducer.fit_transform(embeddings_array)
                logger.info(" CPU UMAP computation completed")
            
            # Create position dictionary with explicit coordinates
            if use_3d:
                pos = {node_id: (float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2])) 
                      for i, node_id in enumerate(node_ids_with_embeddings)}
            else:
                pos = {node_id: (float(positions[i, 0]), float(positions[i, 1])) 
                      for i, node_id in enumerate(node_ids_with_embeddings)}
            
            # Handle nodes without embeddings using spring layout
            missing_nodes = set(node_ids) - set(node_ids_with_embeddings)
            if missing_nodes:
                logger.info(f" Adding spring layout positions for {len(missing_nodes)} nodes without embeddings")
                subgraph = self.graph.subgraph(missing_nodes)
                if len(missing_nodes) > 0:
                    missing_pos = nx.spring_layout(subgraph, k=1, iterations=20)
                pos.update(missing_pos)
            
            # Determine GPU mode for logging
            if use_gpu and use_cuml:
                gpu_mode = "GPU (cuML + CuPy)"
            elif use_gpu and cp is not None:
                gpu_mode = "GPU-optimized (CuPy + CPU UMAP)"
            else:
                gpu_mode = "CPU"
            
            logger.info(f" UMAP layout complete: {len(pos)} nodes positioned using {gpu_mode}")
            logger.info(f" Layout dimensions: {'3D' if use_3d else '2D'}")
            
            return pos
            
        except Exception as e:
            logger.warning(f" GPU UMAP layout failed: {e}. Falling back to spring layout.")
            # Fallback to spring layout
            subgraph = self.graph.subgraph(self.nodes.keys())
            return nx.spring_layout(subgraph, k=3, iterations=50)
    
    def generate_minimal_visualization_data(self, 
                                          layout: str = "umap",
                                          use_3d: bool = False,
                                          max_nodes: int = 10000) -> Dict:
        """
        Generate minimal data for fast dashboard visualization.
        Only includes node IDs, positions, and basic type info.
        Full node data is kept in memory for API access.
        
        Returns:
            {
                'nodes': [{'id': str, 'x': float, 'y': float, 'z': float, 'type': str, 'name': str}],
                'edges': [{'source': str, 'target': str, 'similarity': float}],
                'stats': {...}
            }
        """
        if len(self.nodes) == 0:
            logger.warning("No nodes in graph to visualize")
            return {
                'nodes': [], 
                'edges': [], 
                'stats': {
                    'total_nodes': 0,
                    'visible_nodes': 0,
                    'total_edges': 0,
                    'visible_edges': 0,
                    'layout': layout,
                    'use_3d': use_3d
                }
            }
        
        # Sample nodes if too many
        nodes_to_visualize = list(self.nodes.keys())
        if len(nodes_to_visualize) > max_nodes:
            logger.info(f"Sampling {max_nodes} highest-degree nodes from {len(nodes_to_visualize)} for visualization")
            # Sample highest-degree nodes for better representation
            node_degrees = [(node_id, len(list(self.graph.neighbors(node_id)))) 
                           for node_id in nodes_to_visualize]
            node_degrees.sort(key=lambda x: x[1], reverse=True)
            nodes_to_visualize = [node_id for node_id, _ in node_degrees[:max_nodes]]
        
        # Generate layout positions using GPU-optimized UMAP
        logger.info(f"Computing {layout} layout for {len(nodes_to_visualize)} nodes...")
        
        if layout == "umap":
            # Use GPU-accelerated UMAP for large datasets
            pos = self._create_gpu_optimized_umap_layout(nodes_to_visualize, use_3d=use_3d)
        elif layout == "similarity":
            subgraph = self.graph.subgraph(nodes_to_visualize)
            pos = self._create_similarity_layout_subgraph(subgraph)
        elif layout == "spring":
            subgraph = self.graph.subgraph(nodes_to_visualize)
            pos = nx.spring_layout(subgraph, k=3, iterations=50)
        elif layout == "circular":
            subgraph = self.graph.subgraph(nodes_to_visualize)
            pos = nx.circular_layout(subgraph)
        elif layout == "random":
            subgraph = self.graph.subgraph(nodes_to_visualize)
            pos = nx.random_layout(subgraph)
        else:
            subgraph = self.graph.subgraph(nodes_to_visualize)
            pos = nx.spring_layout(subgraph, k=3, iterations=50)
        
        # Store positions for API access
        self._node_positions = pos
        
        # Create minimal node data
        minimal_nodes = []
        for node_id in nodes_to_visualize:
            if node_id not in pos:
                continue
                
            node = self.nodes[node_id]
            chunk_type = node.chunk.source_info.source_type.value
            source_name = node.chunk.source_info.source_name
            
            # Extract position
            if use_3d:
                x, y, z = pos[node_id]
                node_data = {
                    'id': node_id,
                    'x': float(x),
                    'y': float(y), 
                    'z': float(z),
                    'type': chunk_type,
                    'name': node_id[:12] + "..." if len(node_id) > 12 else node_id,
                    'source': source_name[:20] + "..." if len(source_name) > 20 else source_name
                }
            else:
                x, y = pos[node_id]
                node_data = {
                    'id': node_id,
                    'x': float(x),
                    'y': float(y),
                    'type': chunk_type,
                    'name': node_id[:12] + "..." if len(node_id) > 12 else node_id,
                    'source': source_name[:20] + "..." if len(source_name) > 20 else source_name
                }
            
            minimal_nodes.append(node_data)
        
        logger.info(f" Created {len(minimal_nodes)} minimal node data entries")
        
        # Create minimal edge data (only high similarity edges for performance)
        minimal_edges = []
        edge_count = 0
        max_edges = 5000 # Limit edges for performance
        nodes_to_visualize_set = set(nodes_to_visualize) # Convert to set for O(1) lookup
        
        logger.info(f" Processing edges for visualization (sampling from {len(self.edges)} total edges)...")
        
        # Use a more efficient approach: collect high-quality edges without full sort
        edge_candidates = []
        processed_count = 0
        
        # Process edges in batches and only keep high-similarity ones
        for edge_id, edge in self.edges.items():
            processed_count += 1
            
            # Skip edges not involving visualization nodes
            if (edge.source_chunk_id not in nodes_to_visualize_set or 
                edge.target_chunk_id not in nodes_to_visualize_set):
                continue
            
            # Only consider edges with decent similarity (>= 0.3) to reduce processing
            if edge.semantic_similarity >= 0.3:
                edge_candidates.append((edge_id, edge.semantic_similarity))
            
            # Limit processing to avoid infinite loops with huge graphs
            if processed_count >= 100000: # Process max 100k edges for performance
                logger.info(f" Processed {processed_count} edges, found {len(edge_candidates)} candidates")
                break
        
        # Sort only the candidates (much smaller list)
        edge_candidates.sort(key=lambda x: x[1], reverse=True)
        logger.info(f" Found {len(edge_candidates)} edge candidates, selecting top {max_edges}")
        
        # Take top edges
        for edge_id, similarity in edge_candidates[:max_edges]:
            edge = self.edges[edge_id]
            minimal_edges.append({
                'source': edge.source_chunk_id,
                'target': edge.target_chunk_id,
                'similarity': float(similarity)
            })
            edge_count += 1
        
        # Generate basic stats
        stats = {
            'total_nodes': len(self.nodes),
            'visible_nodes': len(minimal_nodes),
            'total_edges': len(self.edges),
            'visible_edges': len(minimal_edges),
            'layout': layout,
            'use_3d': use_3d
        }
        
        logger.info(f" Generated minimal visualization data: {len(minimal_nodes)} nodes, {len(minimal_edges)} edges")
        logger.info(f" Edge processing completed: {edge_count} edges selected for visualization")
        
        # Store coordinates for potential export/reuse
        if hasattr(self, '_node_positions') and self._node_positions:
            logger.info(f" Node coordinates computed and stored for {len(self._node_positions)} nodes")
            stats['coordinates_computed'] = True
            stats['coordinate_dimensions'] = 3 if use_3d else 2
        else:
            stats['coordinates_computed'] = False
        
        return {
            'nodes': minimal_nodes,
            'edges': minimal_edges, 
            'stats': stats
        }
    
    def generate_sigma_visualization_data(self, 
                                        layout: str = "umap",
                                        use_3d: bool = False,
                                        max_nodes: int = 20000) -> Dict:
        """
        Generate Sigma.js-compatible visualization data for high-performance rendering.
        Optimized format for WebGL-based Sigma.js with thousands of nodes.
        
        Args:
            layout: Layout algorithm ('umap', 'similarity', 'spring', 'circular')
            use_3d: Whether to create 3D layout (not used for Sigma.js - 2D only)
            max_nodes: Maximum nodes to display (Sigma.js can handle more than Plotly)
        
        Returns:
            {
                'nodes': [{'id': str, 'x': float, 'y': float, 'size': int, 'color': str, 'name': str, 'type': str, 'source': str}],
                'edges': [{'source': str, 'target': str, 'similarity': float, 'size': float, 'color': str}],
                'stats': {...},
                'dataVersion': int # For change detection
            }
        """
        if len(self.nodes) == 0:
            logger.warning("No nodes in graph to visualize")
            return {
                'nodes': [], 
                'edges': [], 
                'stats': {
                    'total_nodes': 0,
                    'visible_nodes': 0,
                    'total_edges': 0,
                    'visible_edges': 0,
                    'layout': layout,
                    'renderer': 'sigma.js'
                },
                'dataVersion': 1
            }
        
        # Sample nodes if too many (Sigma.js can handle more than Plotly)
        nodes_to_visualize = list(self.nodes.keys())
        if len(nodes_to_visualize) > max_nodes:
            logger.info(f"Sampling {max_nodes} highest-degree nodes from {len(nodes_to_visualize)} for Sigma.js visualization")
            # Sample highest-degree nodes for better representation
            node_degrees = [(node_id, len(list(self.graph.neighbors(node_id)))) 
                           for node_id in nodes_to_visualize]
            node_degrees.sort(key=lambda x: x[1], reverse=True)
            nodes_to_visualize = [node_id for node_id, _ in node_degrees[:max_nodes]]
        
        # Generate layout positions using GPU-optimized UMAP
        logger.info(f"Computing {layout} layout for {len(nodes_to_visualize)} nodes (Sigma.js format)...")
        
        if layout == "umap":
            # Use GPU-accelerated UMAP for large datasets
            pos = self._create_gpu_optimized_umap_layout(nodes_to_visualize, use_3d=False) # Force 2D for Sigma.js
        elif layout == "similarity":
            subgraph = self.graph.subgraph(nodes_to_visualize)
            pos = self._create_similarity_layout_subgraph(subgraph)
        elif layout == "spring":
            subgraph = self.graph.subgraph(nodes_to_visualize)
            pos = nx.spring_layout(subgraph, k=3, iterations=50)
        elif layout == "circular":
            subgraph = self.graph.subgraph(nodes_to_visualize)
            pos = nx.circular_layout(subgraph)
        elif layout == "random":
            subgraph = self.graph.subgraph(nodes_to_visualize)
            pos = nx.random_layout(subgraph)
        else:
            subgraph = self.graph.subgraph(nodes_to_visualize)
            pos = nx.spring_layout(subgraph, k=3, iterations=50)
        
        # Store positions for API access
        self._node_positions = pos
        
        # Create Sigma.js-formatted node data
        sigma_nodes = []
        for node_id in nodes_to_visualize:
            if node_id not in pos:
                continue
                
            node = self.nodes[node_id]
            chunk_type = node.chunk.source_info.source_type.value
            source_name = node.chunk.source_info.source_name
            
            # Extract position (Sigma.js uses 2D)
            x, y = pos[node_id]
            
            # Determine node size based on connections (degree centrality)
            node_degree = len(list(self.graph.neighbors(node_id)))
            node_size = max(10, min(30, 10 + node_degree * 2)) # Scale size based on connections
            
            # Determine node color based on type
            if chunk_type == 'document':
                node_color = '#3498db' # Blue for documents
            else:
                node_color = '#e74c3c' # Red for tables
            
            # Create Sigma.js node format
            sigma_node = {
                'id': node_id,
                'x': float(x),
                'y': float(y),
                'size': node_size,
                'color': node_color,
                'label': node_id[:20] + "..." if len(node_id) > 20 else node_id, # Shorter labels for performance
                'name': node_id[:20] + "..." if len(node_id) > 20 else node_id,
                'type': chunk_type,
                'source': source_name[:30] + "..." if len(source_name) > 30 else source_name
            }
            
            sigma_nodes.append(sigma_node)
        
        logger.info(f" Created {len(sigma_nodes)} Sigma.js node data entries")
        
        # Create Sigma.js-formatted edge data (only high similarity edges for performance)
        sigma_edges = []
        edge_count = 0
        max_edges = 10000 # Sigma.js can handle more edges than Plotly
        nodes_to_visualize_set = set(nodes_to_visualize) # Convert to set for O(1) lookup
        
        logger.info(f" Processing edges for Sigma.js visualization (sampling from {len(self.edges)} total edges)...")
        
        # Use a more efficient approach: collect high-quality edges without full sort
        edge_candidates = []
        processed_count = 0
        
        # Process edges in batches and only keep high-similarity ones
        for edge_id, edge in self.edges.items():
            processed_count += 1
            
            # Skip edges not involving visualization nodes
            if (edge.source_chunk_id not in nodes_to_visualize_set or 
                edge.target_chunk_id not in nodes_to_visualize_set):
                continue
            
            # Only consider edges with decent similarity (>= 0.2) to reduce processing
            if edge.semantic_similarity >= 0.2:
                edge_candidates.append((edge_id, edge.semantic_similarity))
            
            # Limit processing to avoid infinite loops with huge graphs
            if processed_count >= 200000: # Process max 200k edges for performance
                logger.info(f" Processed {processed_count} edges, found {len(edge_candidates)} candidates")
                break
        
        # Sort only the candidates (much smaller list)
        edge_candidates.sort(key=lambda x: x[1], reverse=True)
        logger.info(f" Found {len(edge_candidates)} edge candidates, selecting top {max_edges}")
        
        # Take top edges and format for Sigma.js
        for edge_id, similarity in edge_candidates[:max_edges]:
            edge = self.edges[edge_id]
            
            # Calculate edge thickness based on similarity
            edge_size = max(1, similarity * 8) # Scale edge thickness
            
            # Determine edge color (subtle gray for all edges)
            edge_color = '#95a5a6' # Light gray
            
            sigma_edge = {
                'source': edge.source_chunk_id,
                'target': edge.target_chunk_id,
                'similarity': float(similarity),
                'size': float(edge_size),
                'color': edge_color,
                'type': 'line' # Sigma.js edge type
            }
            
            sigma_edges.append(sigma_edge)
            edge_count += 1
        
        # Generate basic stats
        stats = {
            'total_nodes': len(self.nodes),
            'visible_nodes': len(sigma_nodes),
            'total_edges': len(self.edges),
            'visible_edges': len(sigma_edges),
            'layout': layout,
            'renderer': 'sigma.js',
            'max_node_size': max([n['size'] for n in sigma_nodes]) if sigma_nodes else 0,
            'min_node_size': min([n['size'] for n in sigma_nodes]) if sigma_nodes else 0,
            'avg_similarity': sum([e['similarity'] for e in sigma_edges]) / len(sigma_edges) if sigma_edges else 0
        }
        
        # Generate data version for change detection
        import hashlib
        data_signature = f"{layout}_{len(sigma_nodes)}_{len(sigma_edges)}_{max_nodes}"
        data_version = int(hashlib.md5(data_signature.encode()).hexdigest()[:8], 16)
        
        logger.info(f" Generated Sigma.js visualization data: {len(sigma_nodes)} nodes, {len(sigma_edges)} edges")
        logger.info(f" Edge processing completed: {edge_count} edges selected for Sigma.js visualization")
        logger.info(f" Data version: {data_version}")
        
        return {
            'nodes': sigma_nodes,
            'edges': sigma_edges, 
            'stats': stats,
            'dataVersion': data_version
        }
    
    def export_node_coordinates(self, filepath: str) -> None:
        """
        Export computed node coordinates to JSON file for reuse.
        These coordinates can be loaded later for faster visualization.
        """
        if not hasattr(self, '_node_positions') or not self._node_positions:
            logger.warning("No node positions available to export")
            return
        
        coordinate_data = {
            'coordinates': self._node_positions,
            'total_nodes': len(self._node_positions),
            'computed_timestamp': str(pd.Timestamp.now()),
            'layout_type': 'umap_gpu_optimized' # or whatever layout was used
        }
        
        with open(filepath, 'w') as f:
            json.dump(coordinate_data, f, indent=2, default=str)
        
        logger.info(f" Exported {len(self._node_positions)} node coordinates to {filepath}")
    
    def import_node_coordinates(self, filepath: str) -> bool:
        """
        Import pre-computed node coordinates from JSON file.
        Returns True if successful, False otherwise.
        """
        try:
            with open(filepath, 'r') as f:
                coordinate_data = json.load(f)
            
            self._node_positions = coordinate_data.get('coordinates', {})
            logger.info(f" Imported {len(self._node_positions)} node coordinates from {filepath}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to import coordinates from {filepath}: {e}")
            return False
    
    def _create_gpu_optimized_umap_layout(self, node_ids: List[str], use_3d: bool = False) -> Dict[str, Tuple[float, float]]:
        """
        Create GPU-optimized UMAP layout for specified nodes only.
        This method specifically targets the visualization nodes for maximum performance.
        
        Args:
            node_ids: List of node IDs to compute layout for
            use_3d: Whether to create a 3D layout (3 components instead of 2)
        """
        if not node_ids:
            return {}
        
        try:
            # GPU Detection with improved logging
            use_gpu = False
            use_cuml = False
            cp = None
            
            try:
                import cupy as cp_module
                cp = cp_module
                if cp.cuda.is_available():
                    logger.info(" CuPy detected - GPU acceleration available")
                    try:
                        device_count = cp.cuda.runtime.getDeviceCount()
                        current_device = cp.cuda.get_device_id()
                        logger.info(f" - GPU Devices: {device_count}")
                        logger.info(f" - Current Device: {current_device}")
                    except Exception as e:
                        logger.info(f" - GPU info unavailable: {e}")
                    
                    try:
                        import cuml
                        use_cuml = True
                        use_gpu = True
                        logger.info(" cuML available - Full GPU acceleration enabled")
                    except ImportError:
                        use_gpu = True # Use CuPy for memory operations
                        logger.info(" Using CuPy optimization (cuML not available)")
                else:
                    logger.info(" CUDA not available on CuPy")
            except ImportError:
                logger.info(" CuPy not available, using CPU UMAP")
            except Exception as e:
                logger.info(f" CuPy initialization failed: {e}, using CPU UMAP")
            
            # Extract embeddings for specified nodes only
            logger.info(f" Processing embeddings for {len(node_ids)} visualization nodes...")
            
            valid_nodes = []
            for node_id in node_ids:
                if node_id in self.nodes and self.nodes[node_id].chunk.embedding is not None:
                    valid_nodes.append((node_id, self.nodes[node_id]))
            
            if not valid_nodes:
                logger.warning("No embeddings available for visualization nodes. Falling back to spring layout.")
                # Fallback to spring layout for these specific nodes
                subgraph = self.graph.subgraph(node_ids)
                return nx.spring_layout(subgraph, k=3, iterations=50)
            
            # Extract embeddings efficiently
            node_ids_with_embeddings = [node_id for node_id, _ in valid_nodes]
            embeddings = [node.chunk.embedding for _, node in valid_nodes]
            
            # Convert to numpy array with optimal dtype
            embeddings_array = np.array(embeddings, dtype=np.float32)
            logger.info(f" Collected {len(embeddings)} embeddings with shape {embeddings_array.shape}")
            
            # Adaptive parameters for visualization performance
            n_nodes = len(embeddings)
            
            if n_nodes > 50000:
                n_neighbors = min(50, max(15, int(np.sqrt(n_nodes))))
                min_dist = 0.01
                spread = 2.0
            elif n_nodes > 10000:
                n_neighbors = min(30, max(15, int(np.sqrt(n_nodes) / 2)))
                min_dist = 0.05
                spread = 1.5
            else:
                n_neighbors = min(15, max(5, n_nodes // 3))
                min_dist = 0.1
                spread = 1.0
            
            logger.info(f" UMAP parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, nodes={n_nodes}")
            
            # GPU-accelerated computation
            if use_gpu and use_cuml:
                logger.info(" Using GPU-accelerated UMAP (cuML + CuPy)")
                
                # Move embeddings to GPU
                embeddings_gpu = cp.asarray(embeddings_array)
                
                import cuml
                reducer = cuml.UMAP(
                    n_components=3 if use_3d else 2,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    spread=spread,
                    metric='cosine',
                    random_state=42,
                    init='random',
                    verbose=False,
                    transform_seed=42
                )
                
                # Fit and transform on GPU
                logger.info(" Computing UMAP layout on GPU...")
                positions_gpu = reducer.fit_transform(embeddings_gpu)
                positions = cp.asnumpy(positions_gpu).astype(np.float64)
                logger.info(" GPU UMAP computation completed")
                
            elif use_gpu and cp is not None:
                logger.info(" Using CuPy memory optimization with CPU UMAP")
                
                # Use CuPy for memory operations
                embeddings_gpu = cp.asarray(embeddings_array)
                
                # Normalize embeddings on GPU for better performance
                norms = cp.linalg.norm(embeddings_gpu, axis=1, keepdims=True)
                embeddings_gpu_normalized = embeddings_gpu / norms
                
                # Move back to CPU for UMAP
                embeddings_array_normalized = cp.asnumpy(embeddings_gpu_normalized).astype(np.float32)
                
                reducer = umap.UMAP(
                    n_components=3 if use_3d else 2,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    spread=spread,
                    metric='cosine',
                    random_state=42,
                    init='random',
                    verbose=False,
                    low_memory=True,
                    transform_seed=42,
                    angular_rp_forest=True,
                    set_op_mix_ratio=1.0,
                    local_connectivity=1.0,
                    repulsion_strength=1.0,
                    negative_sample_rate=5,
                    transform_queue_size=4.0
                )
                
                logger.info(" Computing UMAP layout with CuPy optimization...")
                positions = reducer.fit_transform(embeddings_array_normalized)
                logger.info(" CuPy-optimized UMAP computation completed")
                
            else:
                logger.info(" Using CPU UMAP with optimized parameters")
                
                reducer = umap.UMAP(
                    n_components=3 if use_3d else 2,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    spread=spread,
                    metric='cosine',
                    random_state=42,
                    init='random',
                    verbose=False,
                    low_memory=True,
                    transform_seed=42,
                    angular_rp_forest=True,
                    set_op_mix_ratio=1.0,
                    local_connectivity=1.0,
                    repulsion_strength=1.0,
                    negative_sample_rate=5,
                    transform_queue_size=4.0
                )
                
                logger.info(" Computing UMAP layout on CPU...")
                positions = reducer.fit_transform(embeddings_array)
                logger.info(" CPU UMAP computation completed")
            
            # Create position dictionary with explicit coordinates
            if use_3d:
                pos = {node_id: (float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2])) 
                      for i, node_id in enumerate(node_ids_with_embeddings)}
            else:
                pos = {node_id: (float(positions[i, 0]), float(positions[i, 1])) 
                      for i, node_id in enumerate(node_ids_with_embeddings)}
            
            # Handle nodes without embeddings using spring layout
            missing_nodes = set(node_ids) - set(node_ids_with_embeddings)
            if missing_nodes:
                logger.info(f" Adding spring layout positions for {len(missing_nodes)} nodes without embeddings")
                subgraph = self.graph.subgraph(missing_nodes)
                if len(missing_nodes) > 0:
                    missing_pos = nx.spring_layout(subgraph, k=1, iterations=20)
                    pos.update(missing_pos)
            
            # Determine GPU mode for logging
            if use_gpu and use_cuml:
                gpu_mode = "GPU (cuML + CuPy)"
            elif use_gpu and cp is not None:
                gpu_mode = "GPU-optimized (CuPy + CPU UMAP)"
            else:
                gpu_mode = "CPU"
            
            logger.info(f" UMAP layout complete: {len(pos)} nodes positioned using {gpu_mode}")
            logger.info(f" Layout dimensions: {'3D' if use_3d else '2D'}")
            
            return pos
            
        except Exception as e:
            logger.warning(f" GPU UMAP layout failed: {e}. Falling back to spring layout.")
            # Fallback to spring layout
            subgraph = self.graph.subgraph(node_ids)
            return nx.spring_layout(subgraph, k=3, iterations=50)
    
    def get_node_details(self, node_id: str) -> Optional[Dict]:
        """
        Get complete node details for API access.
        Returns all metadata for a specific node.
        """
        if node_id not in self.nodes:
            return None
        
        node = self.nodes[node_id]
        chunk = node.chunk
        
        # Get neighbor information
        neighbors = list(self.graph.neighbors(node_id))
        neighbor_details = []
        for neighbor_id in neighbors[:10]: # Limit to first 10 neighbors
            if neighbor_id in self.nodes:
                neighbor = self.nodes[neighbor_id]
                neighbor_details.append({
                    'id': neighbor_id,
                    'type': neighbor.chunk.source_info.source_type.value,
                    'source': neighbor.chunk.source_info.source_name
                })
        
        # Get edge information for this node
        connected_edges = []
        for edge_id, edge_metadata in self.edges.items():
            if edge_metadata.source_chunk_id == node_id or edge_metadata.target_chunk_id == node_id:
                other_node = (edge_metadata.target_chunk_id 
                            if edge_metadata.source_chunk_id == node_id 
                            else edge_metadata.source_chunk_id)
                connected_edges.append({
                    'edge_id': edge_id,
                    'other_node': other_node,
                    'similarity': float(edge_metadata.semantic_similarity),
                    'shared_keywords': edge_metadata.shared_keywords
                })
        
        # Sort edges by similarity
        connected_edges.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            'node_id': node_id,
            'chunk_id': chunk.chunk_id,
            'chunk_type': chunk.source_info.source_type.value,
            'source_info': {
                'source_id': chunk.source_info.source_id,
                'source_name': chunk.source_info.source_name,
                'source_type': chunk.source_info.source_type.value,
                'file_path': chunk.source_info.file_path
            },
            'content': chunk.content,
            'summary': chunk.summary,
            'keywords': chunk.keywords,
            'connections': {
                'total_neighbors': len(neighbors),
                'neighbors': neighbor_details,
                'total_edges': len(connected_edges),
                'top_edges': connected_edges[:10] # Top 10 most similar edges
            },
            'position': self._node_positions.get(node_id) if hasattr(self, '_node_positions') else None,
            'metadata': {
                'has_embedding': chunk.embedding is not None,
                'embedding_dim': len(chunk.embedding) if chunk.embedding else 0,
                'content_length': len(chunk.content)
            }
        } 