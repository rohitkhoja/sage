#!/usr/bin/env python3
"""
MAG Agent Server - Loads once, runs on port, accepts function calls
Usage: 
1. Start server: python mag_server.py
2. Call functions: curl "http://localhost:8080/function_name?param1=value1&param2=value2"
"""

import sys
import os
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from loguru import logger

# Add the agent directory to the path
sys.path.append('/shared/khoja/CogComp/agent')

from mag_agent import MAGAgent
from flexible_mag_agent import FlexibleMAGAgent

# Global variables to keep loaded agents
mag_agent = None
flex_agent = None
server_start_time = None

def load_system():
    """Load the system once"""
    global mag_agent, flex_agent, server_start_time
    
    if mag_agent is not None:
        logger.info("✅ System already loaded")
        return True
    
    logger.info("🚀 Loading MAG Agent System...")
    server_start_time = time.time()
    
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

class MAGServerHandler(BaseHTTPRequestHandler):
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
                self.send_help_page()
                
            elif path == '/status':
                uptime = time.time() - server_start_time if server_start_time else 0
                self.send_json_response({
                    'status': 'running',
                    'loaded': mag_agent is not None,
                    'uptime': f"{uptime:.2f}s"
                })
                
            elif path == '/search_papers_by_title':
                query = query_params.get('query', [''])[0]
                result = mag_agent.search_papers_by_title(query)
                self.send_json_response(result)
                
            elif path == '/search_authors_by_name':
                query = query_params.get('query', [''])[0]
                result = mag_agent.search_authors_by_name(query)
                self.send_json_response(result)
                
            elif path == '/search_institutions_by_name':
                query = query_params.get('query', [''])[0]
                result = mag_agent.search_institutions_by_name(query)
                self.send_json_response(result)
                
            elif path == '/search_fields_by_name':
                query = query_params.get('query', [''])[0]
                result = mag_agent.search_fields_by_name(query)
                self.send_json_response(result)
            elif path == '/search_papers_by_abstract':
                query = query_params.get('query', [''])[0]
                result = mag_agent.search_papers_by_abstract(query)
                self.send_json_response(result)
                
            elif path == '/execute_cypher':
                # Execute Cypher query directly
                cypher_query = query_params.get('query', [''])[0]
                if not cypher_query:
                    self.send_json_response({'error': 'No query provided'})
                else:
                    try:
                        result = mag_agent.execute_cypher_query(cypher_query)
                        self.send_json_response(result)
                    except Exception as e:
                        self.send_json_response({'error': str(e)})
            
            elif path == '/get_papers_with_field':
                # Get all papers tagged with a specific field of study
                field_id = int(query_params.get('field_id', ['0'])[0])
                result = mag_agent.get_papers_with_field(field_id)
                self.send_json_response(result)
                
            elif path == '/query_natural_language':
                query = query_params.get('query', [''])[0]
                result = mag_agent.query_natural_language(query)
                self.send_json_response(result)
                
                
            elif path == '/get_authors_of_paper':
                paper_id = query_params.get('paper_id', ['0'])[0]
                # Check if it's a file path or single ID
                if paper_id.endswith('.json') and os.path.exists(paper_id):
                    # Load IDs from file
                    with open(paper_id, 'r') as f:
                        file_data = json.load(f)
                    if isinstance(file_data, list):
                        paper_ids = [item.get('node_index', item) if isinstance(item, dict) else item for item in file_data]
                    else:
                        paper_ids = [paper_id]
                    result = []
                    for pid in paper_ids:
                        authors = mag_agent.get_authors_of_paper(int(pid))
                        result.extend(authors)
                    self.send_json_response(result)
                else:
                    # Single ID
                    result = mag_agent.get_authors_of_paper(int(paper_id))
                    self.send_json_response(result)
                
            elif path == '/get_authors_affiliated_with':
                institution_id = int(query_params.get('institution_id', ['0'])[0])
                result = mag_agent.get_authors_affiliated_with(institution_id)
                self.send_json_response({'results': [{'node_index': aid} for aid in result], 'count': len(result)})
                
            elif path == '/get_papers_by_author':
                author_ids = query_params.get('author_ids', [''])[0]
                # Check if it's a file path or comma-separated IDs
                if author_ids.endswith('.json') and os.path.exists(author_ids):
                    # Load IDs from file
                    with open(author_ids, 'r') as f:
                        file_data = json.load(f)
                    if isinstance(file_data, list):
                        ids = [item.get('node_index', item) if isinstance(item, dict) else item for item in file_data]
                    else:
                        ids = [author_ids]
                    result = []
                    for aid in ids:
                        papers = mag_agent.get_papers_by_author([int(aid)])
                        result.extend(papers)
                    self.send_json_response(result)
                else:
                    # Comma-separated IDs
                    ids = [int(x.strip()) for x in author_ids.split(',') if x.strip()]
                    result = mag_agent.get_papers_by_author(ids)
                    self.send_json_response(result)
                
            elif path == '/get_papers_citing':
                paper_id = query_params.get('paper_id', ['0'])[0]
                # Check if it's a file path or single ID
                if paper_id.endswith('.json') and os.path.exists(paper_id):
                    # Load IDs from file
                    with open(paper_id, 'r') as f:
                        file_data = json.load(f)
                    if isinstance(file_data, list):
                        paper_ids = [item.get('node_index', item) if isinstance(item, dict) else item for item in file_data]
                    else:
                        paper_ids = [paper_id]
                    result = []
                    for pid in paper_ids:
                        citing = mag_agent.get_papers_citing(int(pid))
                        result.extend(citing)
                    self.send_json_response(result)
                else:
                    # Single ID
                    result = mag_agent.get_papers_citing(int(paper_id))
                    self.send_json_response(result)
                
            elif path == '/get_papers_cited_by':
                paper_id = query_params.get('paper_id', ['0'])[0]
                # Check if it's a file path or single ID
                if paper_id.endswith('.json') and os.path.exists(paper_id):
                    # Load IDs from file
                    with open(paper_id, 'r') as f:
                        file_data = json.load(f)
                    if isinstance(file_data, list):
                        paper_ids = [item.get('node_index', item) if isinstance(item, dict) else item for item in file_data]
                    else:
                        paper_ids = [paper_id]
                    result = []
                    for pid in paper_ids:
                        cited = mag_agent.get_papers_cited_by(int(pid))
                        result.extend(cited)
                    self.send_json_response(result)
                else:
                    # Single ID
                    result = mag_agent.get_papers_cited_by(int(paper_id))
                    self.send_json_response(result)
                
            elif path == '/get_paper_metadata':
                paper_id = query_params.get('paper_id', ['0'])[0]
                # Check if it's a file path or single ID
                if paper_id.endswith('.json') and os.path.exists(paper_id):
                    # Load IDs from file
                    with open(paper_id, 'r') as f:
                        file_data = json.load(f)
                    if isinstance(file_data, list):
                        paper_ids = [item.get('node_index', item) if isinstance(item, dict) else item for item in file_data]
                    else:
                        paper_ids = [paper_id]
                    result = []
                    for pid in paper_ids:
                        metadata = mag_agent.get_paper_metadata(int(pid))
                        result.append(metadata)
                    self.send_json_response(result)
                else:
                    # Single ID
                    result = mag_agent.get_paper_metadata(int(paper_id))
                    self.send_json_response(result)
                
            elif path == '/get_author_metadata':
                author_id = query_params.get('author_id', ['0'])[0]
                # Check if it's a file path or single ID
                if author_id.endswith('.json') and os.path.exists(author_id):
                    # Load IDs from file
                    with open(author_id, 'r') as f:
                        file_data = json.load(f)
                    if isinstance(file_data, list):
                        author_ids = [item.get('node_index', item) if isinstance(item, dict) else item for item in file_data]
                    else:
                        author_ids = [author_id]
                    result = []
                    for aid in author_ids:
                        metadata = mag_agent.get_author_metadata(int(aid))
                        result.append(metadata)
                    self.send_json_response(result)
                else:
                    # Single ID
                    result = mag_agent.get_author_metadata(int(author_id))
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
    
    def send_help_page(self):
        """Send help page"""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        html = """
        <html>
        <head><title>MAG Agent Server</title></head>
        <body>
        <h1>🎯 MAG Agent Server</h1>
        <p>System Status: <strong>Running</strong></p>
        
        <h2>Available Functions:</h2>
        <ul>
        <li><strong>GET /search_papers_by_title?query=machine learning</strong></li>
        <li><strong>GET /search_authors_by_name?query=John Smith</strong></li>
        <li><strong>GET /search_institutions_by_name?query=MIT</strong></li>
                <li><strong>GET /search_fields_by_name?query=computer science</strong></li>
                <li><strong>GET /search_papers_by_abstract?query=machine learning algorithms</strong></li>
        <li><strong>GET /get_papers_by_year_range?start_year=2010&end_year=2020</strong></li>
        <li><strong>GET /query_natural_language?query=papers about AI</strong></li>
        <li><strong>GET /get_authors_of_paper?paper_id=12345</strong> or <strong>?paper_id=/path/to/file.json</strong></li>
        <li><strong>GET /get_papers_by_author?author_ids=123,456,789</strong> or <strong>?author_ids=/path/to/file.json</strong></li>
        <li><strong>GET /get_papers_citing?paper_id=12345</strong> or <strong>?paper_id=/path/to/file.json</strong></li>
        <li><strong>GET /get_papers_cited_by?paper_id=12345</strong> or <strong>?paper_id=/path/to/file.json</strong></li>
        <li><strong>GET /get_paper_metadata?paper_id=12345</strong> or <strong>?paper_id=/path/to/file.json</strong></li>
        <li><strong>GET /get_author_metadata?author_id=12345</strong> or <strong>?author_id=/path/to/file.json</strong></li>
        <li><strong>GET /status</strong> - Check system status</li>
        </ul>
        
        <h3>📁 File Input Support:</h3>
        <p>Graph traversal functions now support file inputs for multiple IDs:</p>
        <ul>
        <li><strong>Single ID:</strong> <code>?paper_id=12345</code></li>
        <li><strong>File Input:</strong> <code>?paper_id=/path/to/results.json</code></li>
        <li><strong>File Format:</strong> JSON array of objects with 'node_index' field or simple array of IDs</li>
        </ul>
        
        <h2>Examples:</h2>
        <p><code>curl "http://localhost:8080/search_papers_by_title?query=machine learning"</code></p>
        <p><code>curl "http://localhost:8080/get_papers_by_year_range?start_year=2015&end_year=2020"</code></p>
        <p><code>curl "http://localhost:8080/solve_query?query=papers about AI&session=my_session"</code></p>
        </body>
        </html>
        """
        
        self.wfile.write(html.encode())
    
    def log_message(self, format, *args):
        """Override to use logger instead of stderr"""
        logger.info(f"{self.address_string()} - {format % args}")

def start_server(port=8080):
    """Start the HTTP server"""
    global server_start_time
    
    # Load system first
    if not load_system():
        logger.error("❌ Failed to load system")
        return False
    
    try:
        server = HTTPServer(('localhost', port), MAGServerHandler)
        logger.info(f"🚀 MAG Agent Server started on port {port}")
        logger.info(f"📡 Access at: http://localhost:{port}")
        logger.info("🔧 Available functions:")
        logger.info("  - /search_papers_by_title?query=...")
        logger.info("  - /search_authors_by_name?query=...")
        logger.info("  - /get_papers_by_year_range?start_year=...&end_year=...")
        logger.info("  - /query_natural_language?query=...")
        logger.info("  - /status")
        logger.info("")
        logger.info("📝 INTERMEDIATE RESULT STORAGE ENABLED:")
        logger.info("  - All mag_call results are automatically stored as step1_, step2_, etc.")
        logger.info("  - Use get_intersection to find common results between steps")
        logger.info("  - Use save_results_to_md to save all results to markdown")
        logger.info("  - See agent_prompt_template.md for detailed usage guide")
        logger.info("\n💡 Example: curl 'http://localhost:8080/search_papers_by_title?query=machine learning'")
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
    
    parser = argparse.ArgumentParser(description='MAG Agent Server')
    parser.add_argument('--port', type=int, default=8080, help='Port to run server on')
    args = parser.parse_args()
    
    start_server(args.port)

if __name__ == "__main__":
    main()
