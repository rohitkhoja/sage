#!/usr/bin/env python3
"""
Persistent MAG Agent Server
Loads the system once and keeps it running in memory
Allows multiple connections without reloading
"""

import sys
import os
import json
import time
import signal
import threading
from typing import Dict, List, Any, Optional
from loguru import logger

# Add the agent directory to the path
sys.path.append('/shared/khoja/CogComp/agent')

from graph_loader import MAGGraphLoader
from hnsw_manager import MAGHNSWManager
from traversal_utils import MAGTraversalUtils
from query_orchestrator import MAGQueryOrchestrator
from mag_agent import MAGAgent
from flexible_mag_agent import FlexibleMAGAgent


class PersistentMAGServer:
    """Persistent server that loads once and stays in memory"""
    
    def __init__(self, processed_dir: str, indices_dir: str):
        self.processed_dir = processed_dir
        self.indices_dir = indices_dir
        
        # System components
        self.graph_loader = None
        self.graph = None
        self.traversal_utils = None
        self.hnsw_manager = None
        self.query_orchestrator = None
        self.mag_agent = None
        self.flexible_agent = None
        
        # System state
        self.is_loaded = False
        self.load_time = None
        self.start_time = time.time()
        
        # Server state
        self.running = True
        self.connections = 0
        
    def load_system(self) -> bool:
        """Load the entire MAG system once"""
        logger.info(" Loading Persistent MAG Agent System...")
        start_time = time.time()
        
        try:
            # 1. Load graph
            logger.info(" Loading graph...")
            self.graph_loader = MAGGraphLoader(self.processed_dir)
            self.graph = self.graph_loader.build_graph()
            logger.info(" Graph loaded")
            
            # 2. Load HNSW indices
            logger.info(" Loading HNSW indices...")
            self.hnsw_manager = MAGHNSWManager(self.indices_dir, self.graph_loader)
            self.hnsw_manager.load_all_indices()
            logger.info(" HNSW indices loaded")
            
            # 3. Load traversal utilities
            logger.info(" Loading traversal utilities...")
            self.traversal_utils = MAGTraversalUtils(self.graph, self.graph_loader)
            logger.info(" Traversal utilities loaded")
            
            # 4. Load query orchestrator
            logger.info(" Loading query orchestrator...")
            self.query_orchestrator = MAGQueryOrchestrator(self.hnsw_manager, self.traversal_utils)
            logger.info(" Query orchestrator loaded")
            
            # 5. Load MAG agent
            logger.info(" Loading MAG agent...")
            self.mag_agent = MAGAgent(self.processed_dir, self.indices_dir)
            self.mag_agent.load_all()
            logger.info(" MAG agent loaded")
            
            # 6. Load flexible agent
            logger.info(" Loading flexible agent...")
            self.flexible_agent = FlexibleMAGAgent(self.processed_dir, self.indices_dir)
            self.flexible_agent.load_all()
            logger.info(" Flexible agent loaded")
            
            self.is_loaded = True
            self.load_time = time.time() - start_time
            
            logger.info(f" Persistent system loaded successfully in {self.load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f" Failed to load system: {e}")
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        if not self.is_loaded:
            return {"error": "System not loaded"}
        
        uptime = time.time() - self.start_time
        return {
            "is_loaded": self.is_loaded,
            "load_time": self.load_time,
            "uptime": uptime,
            "connections": self.connections,
            "graph_stats": self.graph_loader.get_stats() if self.graph_loader else None,
            "hnsw_stats": self.hnsw_manager.get_stats() if self.hnsw_manager else None,
            "available_functions": self._get_available_functions()
        }
    
    def _get_available_functions(self) -> List[str]:
        """Get list of available functions"""
        functions = []
        
        # Graph operations
        if self.traversal_utils:
            functions.extend([
                "authors_of_paper(node_index)",
                "papers_by_author(node_indices)",
                "papers_with_field(field_index)",
                "papers_citing(paper_index)",
                "papers_cited_by(paper_index)",
                "papers_by_year_range(start_year, end_year)",
                "papers_by_institution(institution_index)",
                "authors_affiliated_with(institution_index)",
                "get_paper_metadata(node_index)",
                "get_author_metadata(node_index)",
                "get_institution_metadata(node_index)",
                "get_field_metadata(node_index)"
            ])
        
        # HNSW search operations
        if self.hnsw_manager:
            functions.extend([
                "search_title(query, top_k=10)",
                "search_abstract(query, top_k=10)",
                "search_content(query, top_k=10)",
                "search_author_name(query, top_k=10)",
                "search_institution_name(query, top_k=10)",
                "search_field_name(query, top_k=10)"
            ])
        
        # Agent operations
        if self.mag_agent:
            functions.extend([
                "search_papers_by_title(query, top_k=10)",
                "search_papers_by_abstract(query, top_k=10)",
                "search_papers_by_content(query, top_k=10)",
                "search_authors_by_name(query, top_k=10)",
                "get_authors_of_paper(node_index)",
                "get_papers_by_author(node_indices)",
                "get_papers_citing(node_index)",
                "get_papers_cited_by(node_index)",
                "get_papers_by_year_range(start_year, end_year)",
                "get_paper_metadata(node_index)",
                "query_natural_language(query)"
            ])
        
        # Flexible agent operations
        if self.flexible_agent:
            functions.extend([
                "solve_query(query, session_id)",
                "execute_dynamic_code(code)",
                "get_system_info()"
            ])
        
        return functions
    
    def execute_function(self, function_call: str) -> Any:
        """Execute a function call"""
        if not self.is_loaded:
            return {"error": "System not loaded. Call load_system() first."}
        
        try:
            # Parse function call
            if '(' not in function_call or ')' not in function_call:
                return {"error": "Invalid function call format. Use: function_name(args)"}
            
            func_name = function_call.split('(')[0].strip()
            args_str = function_call.split('(')[1].split(')')[0].strip()
            
            # Parse arguments
            args = []
            if args_str:
                # Simple argument parsing (can be improved)
                for arg in args_str.split(','):
                    arg = arg.strip()
                    if arg.startswith('"') and arg.endswith('"'):
                        args.append(arg[1:-1]) # Remove quotes
                    elif arg.isdigit():
                        args.append(int(arg))
                    elif arg.replace('.', '').isdigit():
                        args.append(float(arg))
                    else:
                        args.append(arg)
            
            # Execute function based on name
            result = self._execute_by_name(func_name, args)
            return result
            
        except Exception as e:
            return {"error": f"Failed to execute function: {e}"}
    
    def _execute_by_name(self, func_name: str, args: List[Any]) -> Any:
        """Execute function by name"""
        
        # Graph traversal functions
        if func_name == "authors_of_paper":
            return self.traversal_utils.authors_of_paper(args[0])
        elif func_name == "papers_by_author":
            return self.traversal_utils.papers_by_author(args[0])
        elif func_name == "papers_with_field":
            return self.traversal_utils.papers_with_field(args[0])
        elif func_name == "papers_citing":
            return self.traversal_utils.papers_citing(args[0])
        elif func_name == "papers_cited_by":
            return self.traversal_utils.papers_cited_by(args[0])
        elif func_name == "papers_by_year_range":
            return self.traversal_utils.papers_by_year_range(args[0], args[1])
        elif func_name == "papers_by_institution":
            return self.traversal_utils.papers_by_institution(args[0])
        elif func_name == "authors_affiliated_with":
            return self.traversal_utils.authors_affiliated_with(args[0])
        elif func_name == "get_paper_metadata":
            return self.traversal_utils.get_paper_metadata(args[0])
        elif func_name == "get_author_metadata":
            return self.traversal_utils.get_author_metadata(args[0])
        elif func_name == "get_institution_metadata":
            return self.traversal_utils.get_institution_metadata(args[0])
        elif func_name == "get_field_metadata":
            return self.traversal_utils.get_field_metadata(args[0])
        
        # HNSW search functions
        elif func_name == "search_title":
            return self.hnsw_manager.search("original_title_embedding", 
                                           self.query_orchestrator.encode_query(args[0]), 
                                           top_k=args[1] if len(args) > 1 else 10)
        elif func_name == "search_abstract":
            return self.hnsw_manager.search("abstract_embedding", 
                                           self.query_orchestrator.encode_query(args[0]), 
                                           top_k=args[1] if len(args) > 1 else 10)
        elif func_name == "search_content":
            return self.hnsw_manager.search("abstract_embedding", 
                                           self.query_orchestrator.encode_query(args[0]), 
                                           top_k=args[1] if len(args) > 1 else 10)
        elif func_name == "search_author_name":
            return self.hnsw_manager.search("author_embedding", 
                                           self.query_orchestrator.encode_query(args[0]), 
                                           top_k=args[1] if len(args) > 1 else 10)
        elif func_name == "search_institution_name":
            return self.hnsw_manager.search("institution_embedding", 
                                           self.query_orchestrator.encode_query(args[0]), 
                                           top_k=args[1] if len(args) > 1 else 10)
        elif func_name == "search_field_name":
            return self.hnsw_manager.search("fields_of_study_embedding", 
                                           self.query_orchestrator.encode_query(args[0]), 
                                           top_k=args[1] if len(args) > 1 else 10)
        
        # MAG Agent functions
        elif func_name == "search_papers_by_title":
            return self.mag_agent.search_papers_by_title(args[0], top_k=args[1] if len(args) > 1 else 10)
        elif func_name == "search_papers_by_abstract":
            return self.mag_agent.search_papers_by_abstract(args[0], top_k=args[1] if len(args) > 1 else 10)
        elif func_name == "search_papers_by_content":
            return self.mag_agent.search_papers_by_content(args[0], top_k=args[1] if len(args) > 1 else 10)
        elif func_name == "search_authors_by_name":
            return self.mag_agent.search_authors_by_name(args[0], top_k=args[1] if len(args) > 1 else 10)
        elif func_name == "get_authors_of_paper":
            return self.mag_agent.get_authors_of_paper(args[0])
        elif func_name == "get_papers_by_author":
            return self.mag_agent.get_papers_by_author(args[0])
        elif func_name == "get_papers_citing":
            return self.mag_agent.get_papers_citing(args[0])
        elif func_name == "get_papers_cited_by":
            return self.mag_agent.get_papers_cited_by(args[0])
        elif func_name == "get_papers_by_year_range":
            return self.mag_agent.get_papers_by_year_range(args[0], args[1])
        elif func_name == "query_natural_language":
            return self.mag_agent.query_natural_language(args[0])
        
        # Flexible agent functions
        elif func_name == "solve_query":
            return self.flexible_agent.solve_query(args[0], args[1] if len(args) > 1 else "interactive")
        elif func_name == "execute_dynamic_code":
            return self.flexible_agent.execute_dynamic_code(args[0])
        elif func_name == "get_system_info":
            return self.flexible_agent.get_system_info()
        
        else:
            return {"error": f"Unknown function: {func_name}"}
    
    def run_persistent_server(self):
        """Run persistent server that loads once and stays running"""
        print(" Starting Persistent MAG Agent Server")
        print("=" * 50)
        
        # Load system once
        if not self.load_system():
            print(" Failed to load system. Exiting.")
            return
        
        print(f" System loaded in {self.load_time:.2f}s")
        print(" Server is now running persistently...")
        print(" System will stay loaded in memory")
        print(" Multiple connections can use the same loaded system")
        print("\nTo connect to this server, run:")
        print(" python client.py")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 50)
        
        # Keep server running
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n Shutting down persistent server...")
            self.running = False


def main():
    """Main function"""
    processed_dir = "/shared/khoja/CogComp/datasets/MAG/processed"
    indices_dir = "/shared/khoja/CogComp/output/mag_hnsw_indices"
    
    server = PersistentMAGServer(processed_dir, indices_dir)
    server.run_persistent_server()


if __name__ == "__main__":
    main()
