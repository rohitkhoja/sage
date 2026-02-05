#!/usr/bin/env python3
"""
Simple HTTP Server for MAG Agent
Run on port 8080, call functions via HTTP requests
"""

import sys
import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from loguru import logger

# Add the agent directory to the path
sys.path.append('/shared/khoja/CogComp/agent')

from mag_agent import MAGAgent
from flexible_mag_agent import FlexibleMAGAgent

# Global variables
mag_agent = None
flex_agent = None
server = None

def load_system():
    """Load the system once"""
    global mag_agent, flex_agent
    
    if mag_agent is not None:
        logger.info("✅ System already loaded")
        return True
    
    logger.info("🚀 Loading MAG Agent System...")
    
    try:
        # Load MAG Agent
        mag_agent = MAGAgent("/shared/khoja/CogComp/datasets/MAG/processed", 
                            "/shared/khoja/CogComp/output/mag_hnsw_indices")
        mag_agent.load_all()
        logger.info("✅ MAG Agent loaded")
        
        # Load Flexible Agent
        flex_agent = FlexibleMAGAgent("/shared/khoja/CogComp/datasets/MAG/processed", 
                                    "/shared/khoja/CogComp/output/mag_hnsw_indices")
        flex_agent.load_all()
        logger.info("✅ Flexible Agent loaded")
        
        logger.info("🎉 System loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load system: {e}")
        return False

class MAGRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for MAG functions"""
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            # Parse URL
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            query_params = parse_qs(parsed_url.query)
            
            # Route requests
            if path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(self.get_help_page().encode())
                
            elif path == '/status':
                self.send_json_response({'status': 'running', 'loaded': mag_agent is not None})
                
            elif path == '/search_papers_by_title':
                query = query_params.get('query', [''])[0]
                top_k = int(query_params.get('top_k', ['10'])[0])
                result = mag_agent.search_papers_by_title(query, top_k)
                self.send_json_response(result)
                
            elif path == '/search_authors_by_name':
                query = query_params.get('query', [''])[0]
                top_k = int(query_params.get('top_k', ['10'])[0])
                result = mag_agent.search_authors_by_name(query, top_k)
                self.send_json_response(result)
                
            elif path == '/get_papers_by_year_range':
                start_year = int(query_params.get('start_year', ['2010'])[0])
                end_year = int(query_params.get('end_year', ['2020'])[0])
                result = mag_agent.get_papers_by_year_range(start_year, end_year)
                self.send_json_response(result)
                
            elif path == '/query_natural_language':
                query = query_params.get('query', [''])[0]
                result = mag_agent.query_natural_language(query)
                self.send_json_response(result)
                
            elif path == '/solve_query':
                query = query_params.get('query', [''])[0]
                session = query_params.get('session', ['default'])[0]
                result = flex_agent.solve_query(query, session)
                self.send_json_response(result)
                
            else:
                self.send_error(404, "Function not found")
                
        except Exception as e:
            self.send_error(500, f"Error: {str(e)}")
    
    def send_json_response(self, data):
        """Send JSON response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def get_help_page(self):
        """Get help page HTML"""
        return """
        <html>
        <head><title>MAG Agent HTTP Server</title></head>
        <body>
        <h1>🎯 MAG Agent HTTP Server</h1>
        <p>System Status: <strong>Running</strong></p>
        
        <h2>Available Functions:</h2>
        <ul>
        <li><strong>GET /search_papers_by_title?query=machine learning&top_k=5</strong></li>
        <li><strong>GET /search_authors_by_name?query=John Smith&top_k=3</strong></li>
        <li><strong>GET /get_papers_by_year_range?start_year=2010&end_year=2020</strong></li>
        <li><strong>GET /query_natural_language?query=papers about AI</strong></li>
        <li><strong>GET /solve_query?query=papers about deep learning&session=test</strong></li>
        <li><strong>GET /status</strong> - Check system status</li>
        </ul>
        
        <h2>Examples:</h2>
        <p><code>curl "http://localhost:8080/search_papers_by_title?query=machine learning&top_k=3"</code></p>
        <p><code>curl "http://localhost:8080/get_papers_by_year_range?start_year=2015&end_year=2020"</code></p>
        <p><code>curl "http://localhost:8080/solve_query?query=papers about AI&session=my_session"</code></p>
        </body>
        </html>
        """
    
    def log_message(self, format, *args):
        """Override to use logger instead of stderr"""
        logger.info(f"{self.address_string()} - {format % args}")

def start_server(port=8080):
    """Start the HTTP server"""
    global server
    
    # Load system first
    if not load_system():
        logger.error("❌ Failed to load system")
        return False
    
    try:
        server = HTTPServer(('localhost', port), MAGRequestHandler)
        logger.info(f"🚀 MAG Agent HTTP Server started on port {port}")
        logger.info(f"📡 Access at: http://localhost:{port}")
        logger.info("🔧 Available functions:")
        logger.info("  - /search_papers_by_title?query=...&top_k=...")
        logger.info("  - /search_authors_by_name?query=...&top_k=...")
        logger.info("  - /get_papers_by_year_range?start_year=...&end_year=...")
        logger.info("  - /query_natural_language?query=...")
        logger.info("  - /solve_query?query=...&session=...")
        logger.info("  - /status")
        logger.info("\n💡 Example: curl 'http://localhost:8080/search_papers_by_title?query=machine learning&top_k=3'")
        logger.info("Press Ctrl+C to stop the server")
        
        server.serve_forever()
        
    except KeyboardInterrupt:
        logger.info("👋 Shutting down server...")
        server.shutdown()
        return True
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MAG Agent HTTP Server')
    parser.add_argument('--port', type=int, default=8080, help='Port to run server on')
    args = parser.parse_args()
    
    start_server(args.port)

if __name__ == "__main__":
    main()
