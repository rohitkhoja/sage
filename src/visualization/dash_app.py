"""
Optimized Dash web application for large graph visualization
Uses minimal data approach: only node positions + API calls for details
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import json
import pandas as pd
from typing import Optional
from flask import request, jsonify

# Global variables for the knowledge graph and minimal data
knowledge_graph = None
minimal_viz_data = None


def create_optimized_dashboard(kg: 'KnowledgeGraph', port: int = 8050) -> dash.Dash:
    """
    Create an optimized Dash dashboard for large graph visualization.
    Uses minimal data approach for better performance.
    
    Args:
        kg: The KnowledgeGraph object to visualize
        port: Port number for the dashboard (default: 8050)
    
    Returns:
        Configured Dash app with API endpoints
    """
    global knowledge_graph, minimal_viz_data
    knowledge_graph = kg
    
    # Generate minimal visualization data on startup (GPU-accelerated UMAP)
    print(" Initializing dashboard...")
    print(f" Graph size: {len(knowledge_graph.nodes)} nodes, {len(knowledge_graph.edges)} edges")
    print(" Generating minimal visualization data with GPU-accelerated UMAP...")
    print(" This may take 30-60 seconds for large graphs...")
    
    minimal_viz_data = knowledge_graph.generate_minimal_visualization_data(
        layout="umap",
        use_3d=False,
        max_nodes=10000
    )
    print(f" Visualization data ready: {len(minimal_viz_data['nodes'])} nodes, {len(minimal_viz_data['edges'])} edges")
    print(" Creating Dash application...")
    
    # Create app with enhanced configuration
    app = dash.Dash(
        __name__,
        suppress_callback_exceptions=True,
        serve_locally=True,
        external_stylesheets=[
            'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css',
            'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
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
    
    # Enhanced CSS styles for performance
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

                .graph-container {
                    background: white;
                    border-radius: var(--border-radius);
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                    padding: 1rem;
                    margin-bottom: 1rem;
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
    
    app.layout = html.Div([
        html.Div([
            # Header Section
            html.Div([
                html.H1([
                    html.I(className="fas fa-project-diagram me-3"),
                    "High-Performance RAG Knowledge Graph"
                ]),
                html.P("Optimized for Large-Scale Visualization with GPU Acceleration")
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
                                    {'label': ' UMAP (GPU Accelerated)', 'value': 'umap'},
                                    {'label': ' Similarity Clustering', 'value': 'similarity'},
                                    {'label': ' Spring Layout', 'value': 'spring'},
                                    {'label': ' Circular Layout', 'value': 'circular'}
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
                            ])
                        ])
                    ], className="control-panel"),
                    
                    # Stats Panel
                    html.Div(id='stats-panel', className="control-panel")
                    
                ], className="col-12 col-lg-4"),
                
                # Right Panel: Graph Visualization
                html.Div([
                    html.Div([
                        html.H5([
                            html.I(className="fas fa-project-diagram me-2"),
                            "Interactive Graph"
                        ], className="mb-3"),
                        dcc.Graph(
                            id='minimal-graph',
                            config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'modeBarButtonsToRemove': ['pan2d', 'lasso2d'],
                                'toImageButtonOptions': {
                                    'format': 'png',
                                    'filename': 'knowledge_graph',
                                    'height': 600,
                                    'width': 800,
                                    'scale': 1
                                }
                            },
                            style={'height': '600px'}
                        )
                    ], className="graph-container")
                ], className="col-12 col-lg-8")
                
            ], className="row"),
            
            # Bottom Panel: Node Details (Loaded via API)
            html.Div([
                html.Div([
                    html.H5([
                        html.I(className="fas fa-info-circle me-2"),
                        "Node Details"
                    ], className="mb-3"),
                    html.Div(id='node-details-content')
                ], className="details-container")
            ], className="row mt-3"),
            
            # Hidden storage for current data
            dcc.Store(id='current-viz-data'),
            dcc.Store(id='selected-node-id')
            
        ], className="container-fluid p-3")
    ], className="main-container")
    
    @app.callback(
        [Output('minimal-graph', 'figure'),
         Output('current-viz-data', 'data'),
         Output('stats-panel', 'children')],
        [Input('regenerate-button', 'n_clicks'),
         Input('layout-dropdown', 'value')],
        prevent_initial_call=False
    )
    def update_visualization(n_clicks, layout):
        """Generate minimal visualization with selected layout"""
        global minimal_viz_data
        
        ctx = callback_context
        if ctx.triggered and 'regenerate-button' in ctx.triggered[0]['prop_id']:
            # Regenerate layout
            print(f" Regenerating {layout} layout...")
            minimal_viz_data = knowledge_graph.generate_minimal_visualization_data(
                layout=layout,
                use_3d=False,
                max_nodes=10000
            )
            print(f" Regenerated: {len(minimal_viz_data['nodes'])} nodes")
        
        # Create minimal figure
        fig = create_minimal_graph_figure(minimal_viz_data)
        
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
            ], className="stat-card")
        ])
        
        return fig, minimal_viz_data, stats_panel
    
    @app.callback(
        [Output('node-details-content', 'children'),
         Output('selected-node-id', 'data')],
        [Input('minimal-graph', 'clickData')],
        prevent_initial_call=True
    )
    def handle_node_click(clickData):
        """Handle node click and fetch detailed data via API"""
        if not clickData:
            return "Click on a node to see details", None
        
        try:
            # Extract node ID from click data
            point = clickData['points'][0]
            node_id = point.get('customdata', {}).get('id') if 'customdata' in point else None
            
            if not node_id:
                return "No node data available", None
            
            # Fetch detailed data via knowledge graph
            details = knowledge_graph.get_node_details(node_id)
            
            if not details:
                return f"Node {node_id} not found", None
            
            # Create detailed display
            details_content = create_node_details_display(details)
            
            return details_content, node_id
            
        except Exception as e:
            return f"Error loading node details: {str(e)}", None
    
    @app.callback(
        Output('minimal-graph', 'figure', allow_duplicate=True),
        [Input('search-input', 'value')],
        [State('current-viz-data', 'data')],
        prevent_initial_call=True
    )
    def highlight_search_results(search_value, viz_data):
        """Highlight nodes based on search query"""
        if not search_value or not viz_data:
            return create_minimal_graph_figure(viz_data)
        
        # Create figure with search highlighting
        fig = create_minimal_graph_figure(viz_data, highlight_search=search_value.lower())
        return fig
    
    print(" Dashboard application configured successfully")
    print(" Ready to launch dashboard server...")
    print(f" Dashboard ready with {len(minimal_viz_data['nodes'])} nodes and {len(minimal_viz_data['edges'])} edges")
    return app


def create_minimal_graph_figure(viz_data, highlight_search=None):
    """Create a minimal, fast-loading graph figure"""
    if not viz_data or not viz_data['nodes']:
        return go.Figure().add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16, color="#6c757d")
        )
    
    nodes = viz_data['nodes']
    edges = viz_data['edges']
    
    # Create edge traces (minimal)
    edge_x = []
    edge_y = []
    
    # Create position lookup for faster edge creation
    pos_lookup = {node['id']: (node['x'], node['y']) for node in nodes}
    
    for edge in edges:
        if edge['source'] in pos_lookup and edge['target'] in pos_lookup:
            x0, y0 = pos_lookup[edge['source']]
            x1, y1 = pos_lookup[edge['target']]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
    
    # Create node data with minimal information
    node_x = [node['x'] for node in nodes]
    node_y = [node['y'] for node in nodes]
    node_text = [node['name'] for node in nodes]
    node_ids = [node['id'] for node in nodes]
    
    # Color based on type
    node_colors = []
    for node in nodes:
        if node['type'] == 'document':
            node_colors.append('#3498db') # Blue for documents
        else:
            node_colors.append('#e74c3c') # Red for tables
    
    # Highlight search results
    if highlight_search:
        highlighted_colors = []
        for i, node in enumerate(nodes):
            if (highlight_search in node['id'].lower() or 
                highlight_search in node['name'].lower() or
                highlight_search in node['source'].lower()):
                highlighted_colors.append('#f1c40f') # Yellow for matches
            else:
                highlighted_colors.append(node_colors[i])
        node_colors = highlighted_colors
    
    # Create figure
    fig = go.Figure()
    
    # Add edges (if not too many)
    if len(edges) < 5000:
        fig.add_trace(go.Scattergl(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='rgba(125,125,125,0.2)'),
            hoverinfo='skip',
            mode='lines',
            name='Connections',
            showlegend=False
        ))
    
    # Add nodes with minimal hover data
    fig.add_trace(go.Scattergl(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=15,
            color=node_colors,
            line=dict(width=1, color='white'),
            opacity=0.9
        ),
        text=node_text,
        textposition="middle center",
        textfont=dict(size=8, color='black'),
        customdata=[{'id': node_id} for node_id in node_ids],
        hovertemplate=(
            "<b>%{text}</b><br>" +
            "Click to see details<br>" +
            "<extra></extra>"
        ),
        name='Nodes',
        showlegend=False
    ))
    
    # Update layout for performance
    fig.update_layout(
        title=f"Knowledge Graph ({len(nodes)} nodes, {len(edges)} edges)",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        plot_bgcolor='white',
        dragmode='pan',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig


def create_node_details_display(details):
    """Create detailed node information display"""
    if not details:
        return html.Div("No details available", className="loading-message")
    
    return html.Div([
        # Basic Info Section
        html.Div([
            html.H6([
                html.I(className="fas fa-info-circle me-2"),
                "Basic Information"
            ]),
            html.P([html.Strong("Node ID: "), details['node_id']]),
            html.P([html.Strong("Type: "), details['chunk_type'].title()]),
            html.P([html.Strong("Source: "), details['source_info']['source_name']]),
            html.P([html.Strong("File: "), details['source_info']['file_path']])
        ], className="node-details"),
        
        # Content Section
        html.Div([
            html.H6([
                html.I(className="fas fa-file-text me-2"),
                "Content"
            ]),
            html.P([html.Strong("Summary: "), details['summary']]),
            html.Details([
                html.Summary("View Full Content"),
                html.Pre(details['content'][:1000] + ("..." if len(details['content']) > 1000 else ""),
                        style={'white-space': 'pre-wrap', 'max-height': '200px', 'overflow-y': 'auto'})
            ])
        ], className="node-details"),
        
        # Keywords Section
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
        
        # Connections Section
        html.Div([
            html.H6([
                html.I(className="fas fa-link me-2"),
                f"Connections ({details['connections']['total_neighbors']})"
            ]),
            html.Div([
                html.Div([
                    html.Strong(f"{conn['id'][:30]}..."),
                    html.Br(),
                    html.Small(f"Type: {conn['type']}, Source: {conn['source'][:30]}...")
                ], className="connection-item")
                for conn in details['connections']['neighbors'][:5]
            ])
        ], className="node-details"),
        
        # Top Edges Section
        html.Div([
            html.H6([
                html.I(className="fas fa-project-diagram me-2"),
                "Strongest Connections"
            ]),
            html.Div([
                html.Div([
                    html.Strong(f"→ {edge['other_node'][:30]}..."),
                    html.Br(),
                    html.Small(f"Similarity: {edge['similarity']:.3f}"),
                    html.Br(),
                    html.Small(f"Shared: {', '.join(edge['shared_keywords'][:3])}")
                ], className="connection-item")
                for edge in details['connections']['top_edges'][:3]
            ])
        ], className="node-details")
    ])


def run_optimized_dashboard(knowledge_graph: 'KnowledgeGraph', port: int = 8050, debug: bool = False):
    """
    Run the optimized dashboard for large graph visualization
    
    Args:
        knowledge_graph: The KnowledgeGraph object to visualize
        port: Port to run on
        debug: Whether to run in debug mode
    """
    print(f" Creating dashboard application...")
    app = create_optimized_dashboard(knowledge_graph, port)
    
    print(f" Starting High-Performance RAG Dashboard Server")
    print(f" Dashboard URL: http://127.0.0.1:{port}")
    print(f" GPU-accelerated UMAP visualization ready")
    print(f" Click any node to see full details")
    print(f" Starting Flask server on port {port}...")
    
    try:
        print(f" Starting Flask server on http://127.0.0.1:{port}")
        print(" Click the URL above to open the dashboard")
        print(" Press Ctrl+C to stop the server")
        app.run(debug=debug, port=port, host='127.0.0.1')
    except Exception as e:
        print(f" Error starting dashboard server: {e}")
        import traceback
        traceback.print_exc()
        raise