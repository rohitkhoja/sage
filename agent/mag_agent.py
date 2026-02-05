#!/usr/bin/env python3
"""
MAG Agent - Main Interface
Provides unified access to all MAG query capabilities including HNSW search and graph traversal
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from loguru import logger

from graph_loader import MAGGraphLoader
from hnsw_manager import MAGHNSWManager
from neo4j_traversal import Neo4jTraversalUtils
from query_orchestrator import MAGQueryOrchestrator


class MAGAgent:
    """Main agent interface for MAG dataset queries"""
    
    def __init__(self, processed_dir: str, indices_dir: str):
        self.processed_dir = Path(processed_dir)
        self.indices_dir = Path(indices_dir)
        
        # Core components
        self.graph_loader: Optional[MAGGraphLoader] = None
        self.hnsw_manager: Optional[MAGHNSWManager] = None
        self.traversal_utils: Optional[MAGTraversalUtils] = None
        self.query_orchestrator: Optional[MAGQueryOrchestrator] = None
        
        # State
        self.is_loaded = False
        self.load_time = 0.0
        
        # Statistics
        self.stats = {
            'queries_executed': 0,
            'total_query_time': 0.0,
            'successful_queries': 0,
            'failed_queries': 0
        }
    
    def load_all(self):
        """Load all components (graph, HNSW indices, orchestrator)"""
        start_time = time.time()
        
        logger.info(" Loading MAG Agent components...")
        
        try:
            # Initialize Neo4j traversal utilities first (for graph traversal)
            logger.info(" Initializing Neo4j traversal utilities...")
            import os
            from neo4j import GraphDatabase, basic_auth
            neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            neo4j_user = os.getenv("NEO4J_USER", "neo4j")
            neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4j123") # Updated to match Neo4j password
            neo4j_db = os.getenv("NEO4J_DATABASE") or None
            driver = GraphDatabase.driver(neo4j_uri, auth=basic_auth(neo4j_user, neo4j_password))
            self.traversal_utils = Neo4jTraversalUtils(driver, database=neo4j_db)
            logger.info(" Neo4j traversal utilities initialized")
            
            # Load minimal graph loader for HNSW (only node type mappings, not full attributes)
            logger.info(" Loading node type mappings for HNSW indices...")
            self.graph_loader = MAGGraphLoader(self.processed_dir)
            # Only load node type mappings (lightweight), skip full attributes and graph building
            # Neo4j has all node attributes and handles graph traversal
            self.graph_loader.load_node_mappings()
            # Skip load_node_attributes() - it loads 1.8M nodes into memory
            # If metadata is needed, query Neo4j instead
            logger.info(" Node type mappings loaded (lightweight)")
            
            # Load HNSW indices (with Neo4j for metadata)
            logger.info(" Loading HNSW indices...")
            self.hnsw_manager = MAGHNSWManager(self.indices_dir, self.graph_loader, driver)
            self.hnsw_manager.load_all_indices()
            logger.info(" HNSW indices loaded (metadata from Neo4j)")
            
            # Initialize query orchestrator
            logger.info(" Initializing query orchestrator...")
            self.query_orchestrator = MAGQueryOrchestrator(self.hnsw_manager, self.traversal_utils)
            logger.info(" Query orchestrator initialized")
            
            self.is_loaded = True
            self.load_time = time.time() - start_time
            
            logger.info(f" MAG Agent loaded successfully in {self.load_time:.2f}s (using Neo4j for graph traversal)")
            
            return True
            
        except Exception as e:
            logger.error(f" Failed to load MAG Agent: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def search_papers_by_title(self, query: str) -> List[Dict[str, Any]]:
        """Search papers by title similarity"""
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load_all() first.")
        
        try:
            logger.info(f" MAGAgent.search_papers_by_title called with query='{query}'")
            result = self.query_orchestrator._find_paper_by_title(query)
            logger.info(f" Query orchestrator returned: {result}")
            
            results = result.get('results', [])
            logger.info(f" Returning {len(results)} results from orchestrator")
            logger.info(f" First result keys: {list(results[0].keys()) if results else 'No results'}")
            
            return results
        except Exception as e:
            logger.error(f"Error in search_papers_by_title: {e}")
            return []
    
    def search_papers_by_abstract(self, query: str) -> List[Dict[str, Any]]:
        """Search papers by abstract similarity"""
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load_all() first.")
        
        try:
            result = self.query_orchestrator._find_paper_by_abstract(query)
            return result.get('results', [])
        except Exception as e:
            logger.error(f"Error in search_papers_by_abstract: {e}")
            return []
    
    def search_papers_by_content(self, query: str) -> List[Dict[str, Any]]:
        """Search papers by content similarity"""
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load_all() first.")
        
        try:
            result = self.query_orchestrator._find_paper_by_content(query)
            return result.get('results', [])
        except Exception as e:
            logger.error(f"Error in search_papers_by_content: {e}")
            return []
    
    def search_authors_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Search authors by name similarity"""
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load_all() first.")
        
        try:
            result = self.query_orchestrator._find_author_by_name(name)
            return result.get('results', [])
        except Exception as e:
            logger.error(f"Error in search_authors_by_name: {e}")
            return []
    
    def search_institutions_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Search institutions by name similarity"""
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load_all() first.")
        
        try:
            result = self.query_orchestrator._find_institution_by_name(name)
            return result.get('results', [])
        except Exception as e:
            logger.error(f"Error in search_institutions_by_name: {e}")
            return []
    
    def search_fields_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Search fields of study by name similarity"""
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load_all() first.")
        
        try:
            result = self.query_orchestrator._find_field_by_name(name)
            return result.get('results', [])
        except Exception as e:
            logger.error(f"Error in search_fields_by_name: {e}")
            return []
    
    def get_authors_affiliated_with(self, institution_id: int) -> List[int]:
        """Get all authors affiliated with an institution"""
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load_all() first.")
        
        return self.traversal_utils.authors_affiliated_with(institution_id)
    
    def get_authors_of_paper(self, paper_id: int) -> List[int]:
        """Get all authors of a specific paper"""
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load_all() first.")
        
        return self.traversal_utils.authors_of_paper(paper_id)
    
    def get_papers_by_author(self, author_ids: List[int]) -> List[int]:
        """Get all papers written by given authors"""
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load_all() first.")
        
        return self.traversal_utils.papers_by_author(author_ids)
    
    def get_papers_with_field(self, field_id: int) -> List[int]:
        """Get all papers tagged with a specific field"""
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load_all() first.")
        
        return self.traversal_utils.papers_with_field(field_id)
    
    def get_papers_citing(self, paper_id: int) -> List[int]:
        """Get all papers that cite the given paper"""
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load_all() first.")
        
        return self.traversal_utils.papers_citing(paper_id)
    
    def get_papers_cited_by(self, paper_id: int) -> List[int]:
        """Get all papers cited by the given paper"""
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load_all() first.")
        
        return self.traversal_utils.papers_cited_by(paper_id)
    
    def get_papers_by_year_range(self, start_year: int, end_year: int) -> List[int]:
        """Get all papers published between start_year and end_year"""
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load_all() first.")
        
        return self.traversal_utils.papers_by_year_range(start_year, end_year)
    
    def get_papers_by_institution(self, institution_id: int) -> List[int]:
        """Get all papers from authors at an institution"""
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load_all() first.")
        
        return self.traversal_utils.papers_by_institution(institution_id)
    
    def get_authors_affiliated_with(self, institution_id: int) -> List[int]:
        """Get all authors affiliated with an institution"""
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load_all() first.")
        
        return self.traversal_utils.authors_affiliated_with(institution_id)
    
    def get_paper_metadata(self, paper_id: int) -> Optional[Dict[str, Any]]:
        """Get metadata for a paper"""
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load_all() first.")
        
        return self.traversal_utils.get_paper_metadata(paper_id)
    
    def get_author_metadata(self, author_id: int) -> Optional[Dict[str, Any]]:
        """Get metadata for an author"""
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load_all() first.")
        
        return self.traversal_utils.get_author_metadata(author_id)
    
    def execute_cypher_query(self, cypher_query: str) -> List[Dict[str, Any]]:
        """Execute a Cypher query directly and return results"""
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load_all() first.")
        
        from neo4j import GraphDatabase, basic_auth
        import os
        
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4j123")
        neo4j_db = os.getenv("NEO4J_DATABASE") or None
        
        driver = GraphDatabase.driver(neo4j_uri, auth=basic_auth(neo4j_user, neo4j_password))
        
        try:
            with driver.session(database=neo4j_db) as session:
                result = session.run(cypher_query)
                records = []
                for record in result:
                    records.append(dict(record))
                return records
        finally:
            driver.close()
    
    def get_institution_metadata(self, institution_id: int) -> Optional[Dict[str, Any]]:
        """Get metadata for an institution"""
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load_all() first.")
        
        return self.traversal_utils.get_institution_metadata(institution_id)
    
    def get_field_metadata(self, field_id: int) -> Optional[Dict[str, Any]]:
        """Get metadata for a field of study"""
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load_all() first.")
        
        return self.traversal_utils.get_field_metadata(field_id)
    
    def execute_query_plan(self, query_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a multi-step query plan"""
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load_all() first.")
        
        start_time = time.time()
        
        try:
            result = self.query_orchestrator.execute_query_plan(query_plan)
            
            # Update statistics
            self.stats['queries_executed'] += 1
            self.stats['total_query_time'] += result.get('execution_time_seconds', 0.0)
            
            if result.get('success', False):
                self.stats['successful_queries'] += 1
            else:
                self.stats['failed_queries'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing query plan: {e}")
            self.stats['queries_executed'] += 1
            self.stats['failed_queries'] += 1
            
            return {
                'success': False,
                'error': str(e),
                'execution_time_seconds': time.time() - start_time,
                'steps': [],
                'final_results': [],
                'final_count': 0
            }
    
    def parse_natural_language_query(self, query: str) -> List[Dict[str, Any]]:
        """Parse natural language query into execution plan"""
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load_all() first.")
        
        return self.query_orchestrator.parse_natural_language_query(query)
    
    def query_natural_language(self, query: str) -> Dict[str, Any]:
        """Execute a natural language query"""
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load_all() first.")
        
        # Parse query into plan
        plan = self.parse_natural_language_query(query)
        
        if not plan:
            return {
                'success': False,
                'error': 'Could not parse query into execution plan',
                'query': query,
                'final_results': [],
                'final_count': 0
            }
        
        # Execute plan
        result = self.execute_query_plan(plan)
        result['query'] = query
        result['parsed_plan'] = plan
        
        return result
    
    def get_available_features(self) -> List[str]:
        """Get list of available HNSW features"""
        if not self.is_loaded:
            return []
        
        return self.hnsw_manager.get_available_features()
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        if not self.is_loaded:
            return {}
        
        return self.graph_loader.get_stats()
    
    def get_hnsw_stats(self) -> Dict[str, Any]:
        """Get HNSW statistics"""
        if not self.is_loaded:
            return {}
        
        return self.hnsw_manager.get_stats()
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        stats = self.stats.copy()
        stats['is_loaded'] = self.is_loaded
        stats['load_time_seconds'] = self.load_time
        
        if self.stats['queries_executed'] > 0:
            stats['avg_query_time_seconds'] = self.stats['total_query_time'] / self.stats['queries_executed']
            stats['success_rate'] = self.stats['successful_queries'] / self.stats['queries_executed']
        else:
            stats['avg_query_time_seconds'] = 0.0
            stats['success_rate'] = 0.0
        
        return stats
    
    def save_query_evidence(self, question_id: str, query_result: Dict[str, Any], output_dir: str = "/shared/khoja/CogComp/agent/output"):
        """Save query evidence and results"""
        output_path = Path(output_dir) / "qa"
        output_path.mkdir(parents=True, exist_ok=True)
        
        evidence = {
            'question_id': question_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'index_version': {'dense': 'v1'}, # TODO: Get actual version
            'query_result': query_result,
            'agent_stats': self.get_agent_stats()
        }
        
        evidence_file = output_path / f"{question_id}.json"
        with open(evidence_file, 'w') as f:
            json.dump(evidence, f, indent=2)
        
        logger.info(f" Saved query evidence to {evidence_file}")
        return str(evidence_file)


def main():
    """Test the MAG Agent"""
    logger.info(" Testing MAG Agent")
    
    try:
        # Initialize agent
        agent = MAGAgent(
            processed_dir="/shared/khoja/CogComp/datasets/MAG/processed",
            indices_dir="/shared/khoja/CogComp/output/mag_hnsw_indices"
        )
        
        # Load all components
        if not agent.load_all():
            return False
        
        logger.info(" MAG Agent loaded successfully!")
        
        # Test basic functionality
        logger.info(" Testing search functionality...")
        
        # Test title search
        title_results = agent.search_papers_by_title("machine learning", top_k=5)
        logger.info(f"Title search results: {len(title_results)} papers found")
        
        if title_results:
            top_paper = title_results[0]
            logger.info(f"Top paper: MAG ID {top_paper['mag_object_id']}, score {top_paper['score']:.4f}")
            
            # Test getting authors of this paper
            paper_id = top_paper['mag_object_id']
            authors = agent.get_authors_of_paper(paper_id)
            logger.info(f"Authors of paper {paper_id}: {len(authors)} found")
            
            if authors:
                # Test getting papers by these authors
                author_papers = agent.get_papers_by_author(authors[:2]) # Limit to first 2 authors
                logger.info(f"Papers by first 2 authors: {len(author_papers)} found")
        
        # Test year range query
        recent_papers = agent.get_papers_by_year_range(2010, 2019)
        logger.info(f"Papers from 2010-2019: {len(recent_papers)} found")
        
        # Test natural language query
        nl_result = agent.query_natural_language("papers about machine learning from 2010 to 2020")
        logger.info(f"Natural language query: {nl_result['final_count']} papers found")
        
        # Print statistics
        logger.info(" Agent statistics:")
        stats = agent.get_agent_stats()
        for key, value in stats.items():
            logger.info(f" {key}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f" Failed to test MAG Agent: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
