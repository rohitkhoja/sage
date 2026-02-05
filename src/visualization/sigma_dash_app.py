"""
High-Performance Sigma.js-based Dash Dashboard for Large Graph Visualization
Uses Sigma.js for efficient rendering of thousands of nodes and edges
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context, clientside_callback
import json
import pandas as pd
from typing import Optional
from flask import request, jsonify
from loguru import logger

# Global variables for the knowledge graph and minimal data
knowledge_graph = None
minimal_viz_data = None


def create_sigma_dashboard(kg: 'KnowledgeGraph', port: int = 8050) -> dash.Dash:
    """
    Create a high-performance Sigma.js-based Dash dashboard for large graph visualization.
    Uses Sigma.js with WebGL rendering for optimal performance with thousands of nodes.
    
    Args:
        kg: The KnowledgeGraph object to visualize
        port: Port number for the dashboard (default: 8050)
    
    Returns:
        Configured Dash app with Sigma.js visualization
    """
    global knowledge_graph, minimal_viz_data
    knowledge_graph = kg
    
    # Generate minimal visualization data optimized for Sigma.js format
    logger.info("🚀 Initializing Sigma.js dashboard...")
    logger.info(f"📊 Graph size: {len(knowledge_graph.nodes)} nodes, {len(knowledge_graph.edges)} edges")
    logger.info("🧠 Generating minimal visualization data with GPU-accelerated UMAP...")
    logger.info("⏳ This may take 30-60 seconds for large graphs...")
    
    minimal_viz_data = knowledge_graph.generate_sigma_visualization_data(
        layout="umap",
        use_3d=False,
        max_nodes=20000  # Sigma.js can handle more nodes than Plotly
    )
    logger.info(f"✅ Sigma.js data ready: {len(minimal_viz_data['nodes'])} nodes, {len(minimal_viz_data['edges'])} edges")
    logger.info("🎨 Creating Dash application with Sigma.js...")
    
    # Create app with enhanced configuration
    app = dash.Dash(
        __name__,
        suppress_callback_exceptions=True,
        serve_locally=True,
        external_stylesheets=[
            'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css',
            'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
        ],
        external_scripts=[
            'https://cdn.jsdelivr.net/npm/graphology@0.25.4/dist/graphology.umd.min.js',
            'https://cdn.jsdelivr.net/npm/sigma/build/sigma.min.js'
        ],
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"}
        ]
    )
    
    # Add API endpoint for node details
    @app.server.route('/api/node/<node_id>')
    def get_node_details(node_id):
        """API endpoint to fetch detailed node information"""
        try:
            details = knowledge_graph.get_node_details(node_id)
            if details:
                return jsonify(details)
            else:
                return jsonify({'error': 'Node not found'}), 404
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    # Enhanced CSS and Sigma.js integration
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                :root {
                    --primary: #2c3e50;
                    --accent: #3498db;
                    --success: #27ae60;
                    --light: #f8f9fa;
                    --border-radius: 12px;
                    --box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                }

                body {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                }

                .main-container {
                    background: rgba(255, 255, 255, 0.95);
                    backdrop-filter: blur(10px);
                    border-radius: var(--border-radius);
                    box-shadow: var(--box-shadow);
                    margin: 20px;
                    padding: 0;
                    overflow: hidden;
                }

                .app-header {
                    background: linear-gradient(135deg, var(--primary) 0%, #34495e 100%);
                    color: white;
                    padding: 1.5rem;
                    text-align: center;
                }

                .app-header h1 {
                    font-size: 2rem;
                    font-weight: 700;
                    margin: 0;
                }

                .control-panel {
                    background: white;
                    border-radius: var(--border-radius);
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                    padding: 1.5rem;
                    margin-bottom: 1rem;
                }

                .sigma-container {
                    background: white;
                    border-radius: var(--border-radius);
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                    padding: 1rem;
                    margin-bottom: 1rem;
                    position: relative;
                    overflow: hidden;
                }

                #sigma-graph {
                    width: 100%;
                    height: 600px;
                    border: 1px solid #e9ecef;
                    border-radius: 8px;
                    cursor: grab;
                    position: relative;
                    z-index: 1;
                }

                #sigma-graph:active {
                    cursor: grabbing;
                }

                #sigma-graph canvas {
                    pointer-events: auto !important;
                    cursor: pointer;
                }

                #sigma-graph .sigma-mouse {
                    pointer-events: auto !important;
                }

                .details-container {
                    background: white;
                    border-radius: var(--border-radius);
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                    padding: 1.5rem;
                    max-height: 400px;
                    overflow-y: auto;
                }

                .stat-card {
                    background: var(--light);
                    border-radius: 8px;
                    padding: 1rem;
                    margin: 0.5rem 0;
                    border-left: 4px solid var(--accent);
                }

                .btn {
                    border-radius: 8px;
                    padding: 0.75rem 1.5rem;
                    font-weight: 600;
                    border: none;
                }

                .btn-primary {
                    background: linear-gradient(135deg, var(--accent) 0%, #2980b9 100%);
                    color: white;
                }

                .loading-message {
                    text-align: center;
                    padding: 2rem;
                    color: #6c757d;
                }

                .node-details {
                    border: 1px solid #e9ecef;
                    border-radius: 8px;
                    padding: 1rem;
                    margin: 0.5rem 0;
                }

                .keyword-tag {
                    display: inline-block;
                    background: var(--accent);
                    color: white;
                    padding: 0.25rem 0.5rem;
                    border-radius: 4px;
                    font-size: 0.8rem;
                    margin: 0.2rem;
                }

                .connection-item {
                    padding: 0.5rem;
                    border-left: 3px solid var(--success);
                    margin: 0.5rem 0;
                    background: #f8f9fa;
                }

                .sigma-controls {
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    z-index: 1000;
                    background: rgba(255, 255, 255, 0.9);
                    border-radius: 8px;
                    padding: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }

                .sigma-controls button {
                    margin: 2px;
                    padding: 5px 10px;
                    border: none;
                    border-radius: 4px;
                    background: var(--accent);
                    color: white;
                    cursor: pointer;
                    font-size: 12px;
                }

                .sigma-controls button:hover {
                    background: #2980b9;
                }

                .search-highlight {
                    background: #fff3cd;
                    border-left: 4px solid #ffc107;
                    padding: 10px;
                    border-radius: 4px;
                    margin: 10px 0;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    # Dashboard layout using Sigma.js
    app.layout = html.Div([
        html.Div([
            # Header Section
            html.Div([
                html.H1([
                    html.I(className="fas fa-project-diagram me-3"),
                    "High-Performance RAG Knowledge Graph"
                ]),
                html.P("Powered by Sigma.js for Optimal Large-Scale Visualization")
            ], className="app-header"),
            
            # Main Content Container
            html.Div([
                # Left Panel: Controls and Stats
                html.Div([
                    html.Div([
                        html.H5([
                            html.I(className="fas fa-cog me-2"),
                            "Visualization Controls"
                        ]),
                        
                        html.Div([
                            html.Label("Layout Algorithm", className="form-label fw-bold"),
                            dcc.Dropdown(
                                id='layout-dropdown',
                                options=[
                                    {'label': '🧠 UMAP (GPU Accelerated)', 'value': 'umap'},
                                    {'label': '🎯 Similarity Clustering', 'value': 'similarity'},
                                    {'label': '🌸 Spring Layout', 'value': 'spring'},
                                    {'label': '🔄 Circular Layout', 'value': 'circular'}
                                ],
                                value='umap',
                                className='form-select mb-3'
                            ),
                            
                            html.Button([
                                html.I(className="fas fa-sync-alt me-2"),
                                "Regenerate Layout"
                            ], id='regenerate-button', className='btn btn-primary w-100 mb-3'),
                            
                            html.Div([
                                html.Label("Search Nodes", className="form-label fw-bold"),
                                dcc.Input(
                                    id='search-input',
                                    type='text',
                                    placeholder='Search by node ID or content...',
                                    className='form-control',
                                    debounce=True
                                )
                            ]),
                            
                            html.Div([
                                html.Label("Camera Controls", className="form-label fw-bold mt-3"),
                                html.Div([
                                    html.Button("Reset View", id='reset-camera-btn', className='btn btn-secondary btn-sm me-2 mb-2'),
                                    html.Button("Zoom In", id='zoom-in-btn', className='btn btn-secondary btn-sm me-2 mb-2'),
                                    html.Button("Zoom Out", id='zoom-out-btn', className='btn btn-secondary btn-sm mb-2')
                                ])
                            ])
                        ])
                    ], className="control-panel"),
                    
                    # Stats Panel
                    html.Div(id='stats-panel', className="control-panel")
                    
                ], className="col-12 col-lg-4"),
                
                # Right Panel: Sigma.js Graph Visualization
                html.Div([
                    html.Div([
                        html.H5([
                            html.I(className="fas fa-project-diagram me-2"),
                            "Interactive Graph (Sigma.js)"
                        ], className="mb-3"),
                        
                        # Sigma.js Container
                        html.Div([
                            # Sigma graph container
                            html.Div(id='sigma-graph'),
                            
                            # Embedded controls
                            html.Div([
                                html.Button("🔍+", id='sigma-zoom-in', title="Zoom In"),
                                html.Button("🔍-", id='sigma-zoom-out', title="Zoom Out"),
                                html.Button("🏠", id='sigma-reset-view', title="Reset View"),
                                html.Button("📷", id='sigma-screenshot', title="Take Screenshot")
                            ], className="sigma-controls")
                            
                        ], className="sigma-container")
                    ])
                ], className="col-12 col-lg-8")
                
            ], className="row"),
            
            # Search Results Panel
            html.Div(id='search-results', className="row mt-3", style={'display': 'none'}),
            
            # Bottom Panel: Node Details
            html.Div([
                html.Div([
                    html.H5([
                        html.I(className="fas fa-info-circle me-2"),
                        "Node Details"
                    ], className="mb-3"),
                    html.Div(id='node-details-content', children="Click on a node to see details")
                ], className="details-container")
            ], className="row mt-3"),
            
            # Hidden storage for state management
            dcc.Store(id='current-viz-data'),
            dcc.Store(id='selected-node-id'),
            dcc.Store(id='sigma-instance-state'),
            dcc.Store(id='search-state')
            
        ], className="container-fluid p-3")
    ], className="main-container")
    
    # Main callback to update visualization data
    @app.callback(
        [Output('current-viz-data', 'data'),
         Output('stats-panel', 'children')],
        [Input('regenerate-button', 'n_clicks'),
         Input('layout-dropdown', 'value')],
        prevent_initial_call=False
    )
    def update_visualization_data(n_clicks, layout):
        """Generate visualization data with selected layout"""
        global minimal_viz_data
        
        ctx = callback_context
        if ctx.triggered and 'regenerate-button' in ctx.triggered[0]['prop_id']:
            # Regenerate layout
            logger.info(f"🔄 Regenerating {layout} layout...")
            minimal_viz_data = knowledge_graph.generate_sigma_visualization_data(
                layout=layout,
                use_3d=False,
                max_nodes=20000
            )
            logger.info(f"✅ Regenerated: {len(minimal_viz_data['nodes'])} nodes")
        
        # Create stats panel
        stats = minimal_viz_data['stats']
        stats_panel = html.Div([
            html.H5([
                html.I(className="fas fa-chart-bar me-2"),
                "Graph Statistics"
            ]),
            
            html.Div([
                html.Strong("Total Nodes: "), f"{stats['total_nodes']:,}"
            ], className="stat-card"),
            
            html.Div([
                html.Strong("Visible Nodes: "), f"{stats['visible_nodes']:,}"
            ], className="stat-card"),
            
            html.Div([
                html.Strong("Total Edges: "), f"{stats['total_edges']:,}"
            ], className="stat-card"),
            
            html.Div([
                html.Strong("Visible Edges: "), f"{stats['visible_edges']:,}"
            ], className="stat-card"),
            
            html.Div([
                html.Strong("Layout: "), stats['layout'].upper()
            ], className="stat-card"),
            
            html.Div([
                html.Strong("Renderer: "), "Sigma.js WebGL"
            ], className="stat-card")
        ])
        
        return minimal_viz_data, stats_panel
    
    # Clientside callback to initialize and update Sigma.js graph
    clientside_callback(
        """
        function(vizData, resetClicks, zoomInClicks, zoomOutClicks, screenshotClicks) {
            if (!vizData || !vizData.nodes) {
                return '';
            }
            
            // Initialize Sigma.js if not already done
            if (!window.sigmaInstance || window.currentDataVersion !== vizData.dataVersion) {
                
                // Clear existing instance
                if (window.sigmaInstance) {
                    window.sigmaInstance.kill();
                }
                
                // Create new graph
                const graph = new graphology.Graph();
                
                // Add nodes
                vizData.nodes.forEach(node => {
                    graph.addNode(node.id, {
                        x: node.x,
                        y: node.y,
                        // Scale node size to ~5% of the original for better visibility
                        // Fallback default is adjusted proportionally as well
                        size: (node.size ? Math.max(0.2, node.size * 0.05) : 0.4),
                        color: node.color || (node.type === 'document' ? '#3498db' : '#e74c3c'),
                        label: node.name || node.id,
                        type: 'circle',
                        dataType: node.type || 'document',
                        source: node.source || 'unknown'
                    });
                });
                
                // Add edges
                vizData.edges.forEach(edge => {
                    if (graph.hasNode(edge.source) && graph.hasNode(edge.target)) {
                        // Avoid duplicate edges in simple graph
                        if (!graph.hasEdge(edge.source, edge.target) && !graph.hasEdge(edge.target, edge.source)) {
                            graph.addEdge(edge.source, edge.target, {
                                // Make edges significantly thinner (10× reduction)
                                size: Math.max(0.02, edge.similarity * 0.005),
                                color: '#95a5a6',
                                type: 'line'
                            });
                        }
                    }
                });
                
                // Create Sigma instance
                const container = document.getElementById('sigma-graph');
                if (!container) {
                    console.error('Sigma graph container not found');
                    return 'Sigma graph container not found';
                }
                
                // Resolve Sigma constructor for different UMD/global export names
                const SigmaClass = (window.sigma && (window.sigma.default || window.sigma.Sigma)) || window.Sigma || window.sigma || null;
                if (!SigmaClass) {
                    console.error('Sigma.js library not found on window');
                    console.log('Available on window:', Object.keys(window).filter(key => key.toLowerCase().includes('sigma')));
                    return 'Sigma.js library not found';
                }
                
                console.log('Creating Sigma instance with', graph.order, 'nodes and', graph.size, 'edges');

                window.sigmaInstance = new SigmaClass(graph, container, {
                    renderEdgeLabels: false,
                    defaultNodeType: 'circle',
                    defaultEdgeType: 'line',
                    labelFont: 'Arial',
                    labelSize: 12,
                    labelWeight: 'bold',
                    enableEdgeClickEvents: false,
                    enableEdgeWheelEvents: false,
                    enableEdgeHoverEvents: false,
                    enableNodeClickEvents: true,
                    enableNodeHoverEvents: true,
                    enableNodeWheelEvents: false,
                    allowInvalidContainer: false,
                    zIndex: 10
                });
                
                // Add hover handlers for visual feedback
                window.sigmaInstance.on('enterNode', function(event) {
                    const nodeId = event.node;
                    const graph = window.sigmaInstance.getGraph();
                    
                    // Change node size on hover
                    const originalSize = graph.getNodeAttribute(nodeId, 'size');
                    graph.setNodeAttribute(nodeId, 'size', originalSize * 1.5);
                    graph.setNodeAttribute(nodeId, 'originalSize', originalSize);
                    
                    // Change cursor
                    document.getElementById('sigma-graph').style.cursor = 'pointer';
                    
                    window.sigmaInstance.refresh();
                });
                
                window.sigmaInstance.on('leaveNode', function(event) {
                    const nodeId = event.node;
                    const graph = window.sigmaInstance.getGraph();
                    
                    // Restore original node size
                    const originalSize = graph.getNodeAttribute(nodeId, 'originalSize');
                    if (originalSize) {
                        graph.setNodeAttribute(nodeId, 'size', originalSize);
                    }
                    
                    // Restore cursor
                    document.getElementById('sigma-graph').style.cursor = 'grab';
                    
                    window.sigmaInstance.refresh();
                });
                
                // Add click handler for node selection
                window.sigmaInstance.on('clickNode', function(event) {
                    const nodeId = event.node;
                    window.selectedNodeId = nodeId;
                    console.log('Node clicked:', nodeId);
                    
                    // Highlight selected node
                    const graph = window.sigmaInstance.getGraph();
                    
                    // Reset all node borders
                    graph.forEachNode((node, attributes) => {
                        graph.setNodeAttribute(node, 'borderColor', 'white');
                        graph.setNodeAttribute(node, 'borderSize', 1);
                    });
                    
                    // Highlight selected node
                    graph.setNodeAttribute(nodeId, 'borderColor', '#f1c40f');
                    graph.setNodeAttribute(nodeId, 'borderSize', 3);
                    
                    window.sigmaInstance.refresh();
                    
                    // Update selected node store - this will trigger the server-side callback
                    window.dash_clientside.set_props('selected-node-id', {
                        data: nodeId
                    });
                    
                    console.log('Node selection updated, server callback will handle details');
                });
                
                // Camera controls
                window.sigmaControls = {
                    zoomIn: () => {
                        const camera = window.sigmaInstance.getCamera();
                        camera.animatedZoom({ duration: 300 });
                    },
                    zoomOut: () => {
                        const camera = window.sigmaInstance.getCamera();
                        camera.animatedUnzoom({ duration: 300 });
                    },
                    reset: () => {
                        const camera = window.sigmaInstance.getCamera();
                        camera.animatedReset({ duration: 500 });
                    },
                    screenshot: () => {
                        const canvas = window.sigmaInstance.getCanvases().nodes;
                        const link = document.createElement('a');
                        link.download = 'knowledge-graph.png';
                        link.href = canvas.toDataURL();
                        link.click();
                    }
                };
                
                window.currentDataVersion = vizData.dataVersion;
                console.log('Sigma.js graph initialized with', vizData.nodes.length, 'nodes and', vizData.edges.length, 'edges');
                console.log('Node click events enabled. Try clicking on a node!');
                
                // Add status message
                const statusDiv = document.createElement('div');
                statusDiv.id = 'sigma-status';
                statusDiv.style.cssText = 'position: absolute; top: 50px; right: 10px; background: rgba(46, 204, 113, 0.9); color: white; padding: 5px 10px; border-radius: 4px; font-size: 12px; z-index: 1001;';
                statusDiv.textContent = '✓ Nodes are clickable!';
                container.appendChild(statusDiv);
                
                // Remove status message after 3 seconds
                setTimeout(() => {
                    const status = document.getElementById('sigma-status');
                    if (status) status.remove();
                }, 3000);
            }
            
            // Handle control button clicks
            const ctx = window.dash_clientside.callback_context;
            if (ctx.triggered.length > 0) {
                const triggeredId = ctx.triggered[0].prop_id.split('.')[0];
                
                if (triggeredId === 'sigma-reset-view' && window.sigmaControls) {
                    window.sigmaControls.reset();
                } else if (triggeredId === 'sigma-zoom-in' && window.sigmaControls) {
                    window.sigmaControls.zoomIn();
                } else if (triggeredId === 'sigma-zoom-out' && window.sigmaControls) {
                    window.sigmaControls.zoomOut();
                } else if (triggeredId === 'sigma-screenshot' && window.sigmaControls) {
                    window.sigmaControls.screenshot();
                }
            }
            
            return 'Graph updated successfully';
        }
        """,
        Output('sigma-instance-state', 'data'),
        [Input('current-viz-data', 'data'),
         Input('sigma-reset-view', 'n_clicks'),
         Input('sigma-zoom-in', 'n_clicks'),
         Input('sigma-zoom-out', 'n_clicks'),
         Input('sigma-screenshot', 'n_clicks')],
        prevent_initial_call=False
    )
    
# Helper function initialization moved to client-side initialization to avoid duplicate callbacks
    
    # Search functionality
    @app.callback(
        [Output('search-results', 'children'),
         Output('search-results', 'style'),
         Output('search-state', 'data')],
        [Input('search-input', 'value')],
        [State('current-viz-data', 'data')],
        prevent_initial_call=True
    )
    def handle_search(search_value, viz_data):
        """Handle node search and highlighting"""
        if not search_value or not viz_data:
            return [], {'display': 'none'}, {}
        
        search_lower = search_value.lower()
        matching_nodes = []
        
        # Search through nodes
        for node in viz_data['nodes']:
            if (search_lower in node['id'].lower() or 
                search_lower in node['name'].lower() or 
                search_lower in node['source'].lower()):
                matching_nodes.append(node)
        
        if not matching_nodes:
            return [
                html.Div([
                    html.H6("No matches found"),
                    html.P(f"No nodes found matching '{search_value}'")
                ], className="search-highlight")
            ], {'display': 'block'}, {'matches': []}
        
        # Create search results display
        search_results = [
            html.Div([
                html.H6(f"Found {len(matching_nodes)} matches for '{search_value}'"),
                html.Div([
                    html.Div([
                        html.Strong(node['name']),
                        html.Br(),
                        html.Small(f"ID: {node['id']}, Type: {node['type']}, Source: {node['source']}")
                    ], className="connection-item", 
                       style={'cursor': 'pointer'},
                       id={'type': 'search-result-node', 'index': node['id']})
                    for node in matching_nodes[:10]  # Limit to first 10 results
                ])
            ], className="search-highlight")
        ]
        
        return search_results, {'display': 'block'}, {'matches': [n['id'] for n in matching_nodes]}
    
    # Server-side callback for node selection (fallback if client-side API fails)
    @app.callback(
        Output('node-details-content', 'children'),
        [Input('selected-node-id', 'data')],
        prevent_initial_call=True
    )
    def update_node_details(selected_node_id):
        """Update node details when a node is selected"""
        if not selected_node_id:
            return "Click on a node to see details"
        
        try:
            details = knowledge_graph.get_node_details(selected_node_id)
            if not details:
                return html.Div([
                    html.H6("Node Not Found"),
                    html.P(f"No details found for node: {selected_node_id}")
                ], className="alert alert-warning")
            
            return html.Div([
                html.Div([
                    html.H6([
                        html.I(className="fas fa-info-circle me-2"),
                        "Basic Information"
                    ]),
                    html.P(f"Node ID: {details['node_id']}"),
                    html.P(f"Type: {details['chunk_type']}"),
                    html.P(f"Source: {details['source_info']['source_name']}"),
                    html.P(f"File: {details['source_info']['file_path']}")
                ], className="node-details"),
                
                html.Div([
                    html.H6([
                        html.I(className="fas fa-file-text me-2"),
                        "Content"
                    ]),
                    html.P(f"Summary: {details['summary']}"),
                    html.Details([
                        html.Summary("View Full Content"),
                        html.Pre(
                            details['content'][:1000] + ("..." if len(details['content']) > 1000 else ""),
                            style={'whiteSpace': 'pre-wrap', 'maxHeight': '200px', 'overflowY': 'auto'}
                        )
                    ])
                ], className="node-details"),
                
                html.Div([
                    html.H6([
                        html.I(className="fas fa-tags me-2"),
                        "Keywords"
                    ]),
                    html.Div([
                        html.Span(keyword, className="keyword-tag")
                        for keyword in details['keywords'][:10]
                    ])
                ], className="node-details"),
                
                html.Div([
                    html.H6([
                        html.I(className="fas fa-link me-2"),
                        f"Connections ({details['connections']['total_neighbors']})"
                    ]),
                    html.Div([
                        html.Div([
                            html.Strong(conn['id'][:30] + "..."),
                            html.Br(),
                            html.Small(f"Type: {conn['type']}, Source: {conn['source'][:30]}...")
                        ], className="connection-item")
                        for conn in details['connections']['neighbors'][:5]
                    ])
                ], className="node-details")
            ])
            
        except Exception as e:
            logger.error(f"Error getting node details for {selected_node_id}: {e}")
            return html.Div([
                html.H6("Error"),
                html.P(f"Error loading details for node: {selected_node_id}"),
                html.P(f"Error: {str(e)}")
            ], className="alert alert-danger")
    
    # Clientside callback for search highlighting
    clientside_callback(
        """
        function(searchState) {
            if (!window.sigmaInstance || !searchState || !searchState.matches) {
                return '';
            }
            
            const graph = window.sigmaInstance.getGraph();
            
            // Reset all node colors first
            graph.forEachNode((node, attributes) => {
                const originalColor = attributes.type === 'document' ? '#3498db' : '#e74c3c';
                graph.setNodeAttribute(node, 'color', originalColor);
            });
            
            // Highlight matching nodes
            searchState.matches.forEach(nodeId => {
                if (graph.hasNode(nodeId)) {
                    graph.setNodeAttribute(nodeId, 'color', '#f1c40f');
                }
            });
            
            window.sigmaInstance.refresh();
            
            return 'Search highlighting applied';
        }
        """,
        Output('sigma-instance-state', 'data', allow_duplicate=True),
        Input('search-state', 'data'),
        prevent_initial_call=True
    )
    
    logger.info("✅ Sigma.js dashboard application configured successfully")
    logger.info("🌐 Ready to launch Sigma.js dashboard server...")
    logger.info(f"📊 Dashboard ready with {len(minimal_viz_data['nodes'])} nodes and {len(minimal_viz_data['edges'])} edges")
    return app


def run_sigma_dashboard(knowledge_graph: 'KnowledgeGraph', port: int = 8050, debug: bool = False):
    """
    Run the high-performance Sigma.js dashboard for large graph visualization
    """
    logger.info(f"🔧 Creating Sigma.js dashboard application...")
    app = create_sigma_dashboard(knowledge_graph, port)
    
    logger.info(f"🚀 Starting High-Performance Sigma.js Dashboard Server")
    logger.info(f"📊 Dashboard URL: http://127.0.0.1:{port}")
    logger.info(f"⚡ WebGL-accelerated Sigma.js visualization ready")
    
    try:
        app.run(debug=debug, port=port, host='127.0.0.1')
    except Exception as e:
        logger.error(f"❌ Error starting Sigma.js dashboard server: {e}")
        raise 