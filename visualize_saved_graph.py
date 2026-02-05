#!/usr/bin/env python3
"""
Visualize a saved knowledge graph with enhanced error handling and format detection

Usage Examples:
    # Create static HTML visualization
    python visualize_saved_graph.py --graph-path output/graph.json
    
    # Launch interactive dashboard 
    python visualize_saved_graph.py --graph-path output/graph.json --dashboard
    
    # Launch high-performance Sigma.js dashboard (recommended for large graphs)
    python visualize_saved_graph.py --graph-path output/graph.json --sigma
    
    # Dashboard with custom port and layout
    python visualize_saved_graph.py --graph-path output/graph.json --dashboard --port 8060 --layout similarity
    
    # Sigma.js dashboard with custom settings
    python visualize_saved_graph.py --graph-path output/graph.json --sigma --port 8060 --layout umap
    
    # 3D visualization with UMAP layout (static HTML only)
    python visualize_saved_graph.py --graph-path output/graph.json --layout umap --use-3d
    
    # Dashboard in debug mode
    python visualize_saved_graph.py --graph-path output/graph.json --dashboard --debug
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Fix SQLite3 import for NLTK compatibility 
try:
    import sqlite3
except ImportError:
    try:
        import pysqlite3 as sqlite3
        sys.modules['sqlite3'] = sqlite3
        print("Using pysqlite3 as sqlite3 replacement for NLTK compatibility")
    except ImportError:
        print("⚠️  Neither sqlite3 nor pysqlite3 available - NLTK may not work properly")

# Now import NLTK after sqlite3 fix
try:
    import nltk
    print("✅ NLTK successfully imported with sqlite3 fix")
except ImportError as e:
    print(f"⚠️  NLTK import failed: {e}")

from loguru import logger
import json

def main():
    """Main visualization function with enhanced error handling"""
    parser = argparse.ArgumentParser(description="Visualize saved knowledge graph")
    parser.add_argument("--graph-path", required=True, help="Path to saved knowledge graph JSON file")
    parser.add_argument("--output-file", default="knowledge_graph_visualization.html", 
                       help="Output HTML file name")
    parser.add_argument("--layout", default="spring", choices=["spring", "circular", "kamada_kawai", "similarity", "umap"],
                       help="Graph layout algorithm")
    parser.add_argument("--use-3d", action="store_true", help="Create 3D visualization")
    parser.add_argument("--port", type=int, default=8050, help="Port for Dash app (if applicable)")
    parser.add_argument("--dashboard", action="store_true", 
                       help="Launch interactive dashboard instead of creating static HTML")
    parser.add_argument("--sigma", action="store_true", 
                       help="Use Sigma.js dashboard for better performance with large graphs")
    parser.add_argument("--debug", action="store_true", help="Run dashboard in debug mode")
    
    args = parser.parse_args()
    
    logger.info(f"Loading knowledge graph from: {args.graph_path}")
    
    try:
        # Try to load with the corrected import method
        from src.core.graph import KnowledgeGraph
        
        # First, try the standard import method (for graphs saved with export_to_json)
        try:
            knowledge_graph = KnowledgeGraph.import_from_json(args.graph_path)
            logger.info(f"Successfully loaded graph using import_from_json: {len(knowledge_graph.nodes)} nodes, {len(knowledge_graph.edges)} edges")
        except Exception as e:
            logger.warning(f"Failed to load as original format: {e}")
            
            # Fallback to integrated pipeline format
            try:
                knowledge_graph = KnowledgeGraph.load_from_integrated_pipeline(args.graph_path)
                logger.info(f"Successfully loaded graph using load_from_integrated_pipeline: {len(knowledge_graph.nodes)} nodes, {len(knowledge_graph.edges)} edges")
            except Exception as e2:
                logger.error(f"Failed to load graph from integrated pipeline format {args.graph_path}: {e2}")
                logger.error(f"Failed to load graph in any supported format: {e2}")
                return 1
        
        # Generate statistics
        # stats = knowledge_graph.get_graph_statistics()
        # logger.info(f"Graph statistics: {stats}")
        
        if args.dashboard or args.sigma:
            # Launch interactive dashboard
            if args.sigma:
                logger.info("Launching Sigma.js high-performance dashboard...")
                try:
                    # Import Sigma.js dashboard
                    logger.info("🔧 Importing Sigma.js dashboard module...")
                    from src.visualization.sigma_dash_app import run_sigma_dashboard
                    logger.info("✅ Sigma.js dashboard module imported successfully")
                    
                    logger.info(f"🚀 Starting Sigma.js dashboard on port {args.port}...")
                    logger.info(f"📊 Dashboard URL: http://127.0.0.1:{args.port}")
                    logger.info("🎮 High-performance WebGL visualization with Sigma.js")
                    logger.info("Press Ctrl+C to stop the dashboard")
                    
                    # Run the Sigma.js dashboard
                    run_sigma_dashboard(knowledge_graph, port=args.port, debug=args.debug)
                    
                except KeyboardInterrupt:
                    logger.info("Sigma.js dashboard stopped by user")
                    return 0
                except ImportError as e:
                    logger.error(f"❌ Could not import Sigma.js dashboard module: {e}")
                    logger.error("Falling back to regular dashboard...")
                    args.sigma = False
                    args.dashboard = True  # Fall through to regular dashboard
                except Exception as e:
                    logger.error(f"❌ Error launching Sigma.js dashboard: {e}")
                    logger.error("Falling back to regular dashboard...")
                    args.sigma = False
                    args.dashboard = True  # Fall through to regular dashboard
            
            if args.dashboard and not args.sigma:
                logger.info("Launching interactive dashboard...")
                try:
                    # Import optimized dashboard
                    logger.info("🔧 Importing dashboard module...")
                    from src.visualization.dash_app import run_optimized_dashboard
                    logger.info("✅ Dashboard module imported successfully")
                    
                    logger.info(f"🚀 Starting dashboard on port {args.port}...")
                    logger.info(f"📊 Dashboard URL: http://127.0.0.1:{args.port}")
                    logger.info("Press Ctrl+C to stop the dashboard")
                    
                    # Run the optimized dashboard
                    run_optimized_dashboard(knowledge_graph, port=args.port, debug=args.debug)
                
                except KeyboardInterrupt:
                    logger.info("Dashboard stopped by user")
                    return 0
                except ImportError as e:
                    logger.error(f"❌ Could not import dashboard module: {e}")
                    logger.error(f"Full error: {type(e).__name__}: {str(e)}")
                    import traceback
                    logger.error(f"Traceback:\n{traceback.format_exc()}")
                    logger.info("Falling back to static HTML visualization...")
                    args.dashboard = False  # Fall through to static visualization
                    args.sigma = False
                except Exception as e:
                    logger.error(f"❌ Error launching dashboard: {e}")
                    logger.error(f"Full error: {type(e).__name__}: {str(e)}")
                    import traceback
                    logger.error(f"Traceback:\n{traceback.format_exc()}")
                    logger.info("Falling back to static HTML visualization...")
                    args.dashboard = False  # Fall through to static visualization
                    args.sigma = False
        
        if not args.dashboard and not args.sigma:
            # Create static visualization
            logger.info(f"Creating {args.layout} layout visualization...")
            fig = knowledge_graph.create_interactive_visualization(
                layout=args.layout,
                title=f"Knowledge Graph Visualization ({Path(args.graph_path).stem})",
                use_3d=args.use_3d,
                color_by="chunk_type"
            )
            
            # Save to HTML file
            output_path = Path(args.output_file)
            fig.write_html(str(output_path))
            logger.info(f"Visualization saved to: {output_path.absolute()}")
            
            # Try to open in browser
            try:
                import webbrowser
                webbrowser.open(f"file://{output_path.absolute()}")
                logger.info("Opened visualization in default browser")
            except Exception as e:
                logger.warning(f"Could not open browser automatically: {e}")
                logger.info(f"Please open {output_path.absolute()} manually in your browser")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error loading or visualizing graph: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 