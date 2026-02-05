#!/usr/bin/env python3
"""
Flexible MAG Agent - Dynamic Query Execution
An agent that can dynamically write and execute code to solve complex queries
"""

import json
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from loguru import logger
import faiss
from sentence_transformers import SentenceTransformer

from graph_loader import MAGGraphLoader
from hnsw_manager import MAGHNSWManager
from neo4j_traversal import Neo4jTraversalUtils


class FlexibleMAGAgent:
    """
    Flexible agent that can dynamically analyze queries and execute custom code
    to solve complex problems using available graph and HNSW data
    """
    
    def __init__(self, processed_dir: str, indices_dir: str):
        self.processed_dir = Path(processed_dir)
        self.indices_dir = Path(indices_dir)
        
        # Core components
        self.graph_loader: Optional[MAGGraphLoader] = None
        self.hnsw_manager: Optional[MAGHNSWManager] = None
        self.traversal_utils: Optional[Neo4jTraversalUtils] = None
        self.encoder: Optional[SentenceTransformer] = None
        
        # State
        self.is_loaded = False
        self.load_time = 0.0
        
        # Dynamic execution context
        self.execution_context = {
            'variables': {},
            'results': {},
            'step_count': 0
        }
        
        # Statistics
        self.stats = {
            'queries_executed': 0,
            'total_query_time': 0.0,
            'successful_queries': 0,
            'failed_queries': 0,
            'dynamic_code_executions': 0
        }
    
    def load_all(self):
        """Load all components"""
        start_time = time.time()
        
        logger.info("🚀 Loading Flexible MAG Agent components...")
        
        try:
            # Initialize Neo4j traversal utilities first (for graph traversal)
            logger.info("🔗 Initializing Neo4j traversal utilities...")
            import os
            from neo4j import GraphDatabase, basic_auth
            neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            neo4j_user = os.getenv("NEO4J_USER", "neo4j")
            neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4j123")  # Updated to match Neo4j password
            neo4j_db = os.getenv("NEO4J_DATABASE") or None
            driver = GraphDatabase.driver(neo4j_uri, auth=basic_auth(neo4j_user, neo4j_password))
            self.traversal_utils = Neo4jTraversalUtils(driver, database=neo4j_db)
            logger.info("✅ Neo4j traversal utilities initialized")
            
            # Load minimal graph loader for HNSW (only node type mappings, not full attributes)
            logger.info("📊 Loading node type mappings for HNSW indices...")
            self.graph_loader = MAGGraphLoader(self.processed_dir)
            # Only load node type mappings (lightweight), skip full attributes and graph building
            # Neo4j has all node attributes and handles graph traversal
            self.graph_loader.load_node_mappings()
            # Skip load_node_attributes() - it loads 1.8M nodes into memory
            # If metadata is needed, query Neo4j instead
            logger.info("✅ Node type mappings loaded (lightweight)")
            
            # Load HNSW indices (with Neo4j for metadata)
            logger.info("🔍 Loading HNSW indices...")
            self.hnsw_manager = MAGHNSWManager(self.indices_dir, self.graph_loader, driver)
            self.hnsw_manager.load_all_indices()
            logger.info("✅ HNSW indices loaded (metadata from Neo4j)")
            
            # Load sentence transformer (robust local + fallback strategy)
            logger.info("🧠 Loading sentence transformer...")
            self.encoder = None
            local_model_root = "/shared/khoja/CogComp/models/sentence_transformers"
            snapshot_glob = "models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
            try:
                # Prefer exact snapshot directory if present
                candidate = None
                try:
                    import glob
                    candidates = glob.glob(f"{local_model_root}/{snapshot_glob}/*")
                    if candidates:
                        candidate = candidates[0]
                except Exception:
                    candidate = None
                model_path = candidate if candidate else local_model_root
                self.encoder = SentenceTransformer(model_path)
                logger.info(f"✅ Sentence transformer loaded from local path: {model_path}")
            except Exception as e_local:
                logger.warning(f"⚠️ Local model load failed: {e_local}")
                # Try to repair local cache by removing invalid folder and re-downloading
                try:
                    import shutil
                    if os.path.isdir(local_model_root):
                        logger.info(f"🧹 Removing invalid local model directory: {local_model_root}")
                        shutil.rmtree(local_model_root, ignore_errors=True)
                except Exception as e_rm:
                    logger.warning(f"⚠️ Failed to remove local model dir: {e_rm}")
                # Attempt fresh download into cache
                try:
                    self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                    logger.info("✅ Sentence transformer re-downloaded from Hugging Face")
                except Exception as e_hf:
                    logger.error(f"❌ Failed to load sentence transformer from HF: {e_hf}")
                    self.encoder = None
            
            self.is_loaded = True
            self.load_time = time.time() - start_time
            
            logger.info(f"✅ Flexible MAG Agent loaded successfully in {self.load_time:.2f}s")
            
            # Save ID mappings
            self.graph_loader.save_id_mappings("/shared/khoja/CogComp/agent/id_maps")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Flexible MAG Agent: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information for the agent"""
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load_all() first.")
        
        return {
            'graph_info': {
                'total_nodes': self.graph_loader.stats['total_nodes'],
                'total_edges': self.graph_loader.stats['total_edges'],
                'node_type_counts': dict(self.graph_loader.stats['node_type_counts']),
                'edge_type_counts': dict(self.graph_loader.stats['edge_type_counts']),
                'available_node_types': list(self.graph_loader.stats['node_type_counts'].keys()),
                'available_edge_types': list(self.graph_loader.stats['edge_type_counts'].keys())
            },
            'hnsw_info': {
                'available_features': self.hnsw_manager.get_available_features(),
                'feature_stats': self.hnsw_manager.get_all_index_info(),
                'total_embeddings': self.hnsw_manager.stats['total_embeddings']
            },
            'data_schema': {
                'paper_fields': self._get_paper_schema(),
                'author_fields': self._get_author_schema(),
                'institution_fields': self._get_institution_schema(),
                'field_fields': self._get_field_schema()
            },
            'traversal_functions': [
                'authors_of_paper(paper_id)',
                'papers_by_author(author_ids)',
                'papers_with_field(field_id)',
                'papers_citing(paper_id)',
                'papers_cited_by(paper_id)',
                'papers_by_year_range(start_year, end_year)',
                'papers_by_institution(institution_id)',
                'authors_affiliated_with(institution_id)'
            ],
            'hnsw_functions': [
                'search_title(query, top_k)',
                'search_abstract(query, top_k)',
                'search_content(query, top_k)',
                'search_author_name(name, top_k)'
            ]
        }
    
    def _get_paper_schema(self) -> List[str]:
        """Get paper node schema"""
        sample_paper = None
        for mag_id, attrs in self.graph_loader.node_attrs.items():
            if self.graph_loader.get_node_type(mag_id) == 'paper':
                sample_paper = attrs
                break
        
        return list(sample_paper.keys()) if sample_paper else []
    
    def _get_author_schema(self) -> List[str]:
        """Get author node schema"""
        sample_author = None
        for mag_id, attrs in self.graph_loader.node_attrs.items():
            if self.graph_loader.get_node_type(mag_id) == 'author':
                sample_author = attrs
                break
        
        return list(sample_author.keys()) if sample_author else []
    
    def _get_institution_schema(self) -> List[str]:
        """Get institution node schema"""
        sample_institution = None
        for mag_id, attrs in self.graph_loader.node_attrs.items():
            if self.graph_loader.get_node_type(mag_id) == 'institution':
                sample_institution = attrs
                break
        
        return list(sample_institution.keys()) if sample_institution else []
    
    def _get_field_schema(self) -> List[str]:
        """Get field of study node schema"""
        sample_field = None
        for mag_id, attrs in self.graph_loader.node_attrs.items():
            if self.graph_loader.get_node_type(mag_id) == 'field_of_study':
                sample_field = attrs
                break
        
        return list(sample_field.keys()) if sample_field else []
    
    def execute_dynamic_code(self, code: str, context_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute dynamically generated code with access to all agent capabilities"""
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load_all() first.")
        
        start_time = time.time()
        
        try:
            # Prepare execution context
            exec_globals = {
                # Core components
                'agent': self,
                'graph_loader': self.graph_loader,
                'hnsw_manager': self.hnsw_manager,
                'traversal_utils': self.traversal_utils,
                'encoder': self.encoder,
                
                # Utility functions
                'search_title': self._search_title,
                'search_abstract': self._search_abstract,
                'search_content': self._search_content,
                'search_author_name': self._search_author_name,
                'get_authors_of_paper': self._get_authors_of_paper,
                'get_papers_by_author': self._get_papers_by_author,
                'get_papers_with_field': self._get_papers_with_field,
                'get_papers_citing': self._get_papers_citing,
                'get_papers_cited_by': self._get_papers_cited_by,
                'get_papers_by_year_range': self._get_papers_by_year_range,
                'get_papers_by_institution': self._get_papers_by_institution,
                'get_authors_affiliated_with': self._get_authors_affiliated_with,
                'get_paper_metadata': self._get_paper_metadata,
                'get_author_metadata': self._get_author_metadata,
                'get_institution_metadata': self._get_institution_metadata,
                'get_field_metadata': self._get_field_metadata,
                'intersect_lists': self._intersect_lists,
                'union_lists': self._union_lists,
                'filter_papers_by_criteria': self._filter_papers_by_criteria,
                'encode_query': self._encode_query,
                
                # Standard libraries
                'json': json,
                'time': time,
                'numpy': np,
                'torch': torch,
                'faiss': faiss,
                'set': set,
                'list': list,
                'dict': dict,
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'sorted': sorted,
                'min': min,
                'max': max,
                'sum': sum,
                'any': any,
                'all': all,
                'isinstance': isinstance,
                'type': type,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                
                # Context variables
                **(context_vars or {}),
                
                # Execution tracking
                'step_count': 0,
                'results': {}
            }
            
            exec_locals = {}
            
            # Execute the code
            exec(code, exec_globals, exec_locals)
            
            execution_time = time.time() - start_time
            self.stats['dynamic_code_executions'] += 1
            
            # Extract results
            results = {}
            for key in ['result', 'results', 'final_answer', 'final_results', 'answer', 'output']:
                if key in exec_locals:
                    results[key] = exec_locals[key]
            
            # If no explicit result, return all local variables
            if not results:
                results = {k: v for k, v in exec_locals.items() if not k.startswith('_')}
            
            return {
                'success': True,
                'execution_time': execution_time,
                'results': results,
                'locals': exec_locals,
                'code': code
            }
            
        except Exception as e:
            logger.error(f"Error executing dynamic code: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time,
                'code': code
            }
    
    def solve_query(self, query: str, question_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main method to solve a query using dynamic code generation
        The agent analyzes the query and generates appropriate code
        """
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load_all() first.")
        
        start_time = time.time()
        
        logger.info(f"🎯 Solving query: {query}")
        
        try:
            # Get system information
            system_info = self.get_system_info()
            
            # Generate code based on query analysis
            generated_code = self._generate_code_for_query(query, system_info)
            
            logger.info(f"📝 Generated code:\n{generated_code}")
            
            # Execute the generated code
            execution_result = self.execute_dynamic_code(generated_code)
            
            # Prepare final result
            final_result = {
                'query': query,
                'question_id': question_id,
                'generated_code': generated_code,
                'execution_result': execution_result,
                'system_info': system_info,
                'execution_time': time.time() - start_time,
                'success': execution_result['success']
            }
            
            # Update statistics
            self.stats['queries_executed'] += 1
            self.stats['total_query_time'] += final_result['execution_time']
            
            if final_result['success']:
                self.stats['successful_queries'] += 1
            else:
                self.stats['failed_queries'] += 1
            
            # Save evidence if question_id provided
            if question_id:
                self._save_query_evidence(question_id, final_result)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error solving query: {e}")
            import traceback
            traceback.print_exc()
            
            self.stats['queries_executed'] += 1
            self.stats['failed_queries'] += 1
            
            return {
                'query': query,
                'question_id': question_id,
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def _generate_code_for_query(self, query: str, system_info: Dict[str, Any]) -> str:
        """Generate code to solve the given query"""
        query_lower = query.lower()
        
        # Analyze query type and generate appropriate code
        if 'paper' in query_lower and 'author' in query_lower and ('from' in query_lower or 'by' in query_lower):
            return self._generate_author_paper_query_code(query, system_info)
        elif 'paper' in query_lower and ('year' in query_lower or 'between' in query_lower):
            return self._generate_year_range_query_code(query, system_info)
        elif 'paper' in query_lower and ('cite' in query_lower or 'citation' in query_lower):
            return self._generate_citation_query_code(query, system_info)
        elif 'paper' in query_lower and ('field' in query_lower or 'topic' in query_lower):
            return self._generate_field_query_code(query, system_info)
        elif 'institution' in query_lower and 'paper' in query_lower:
            return self._generate_institution_query_code(query, system_info)
        else:
            return self._generate_general_search_code(query, system_info)
    
    def _generate_author_paper_query_code(self, query: str, system_info: Dict[str, Any]) -> str:
        """Generate code for author-paper queries"""
        return f"""
# Query: {query}
# Strategy: Find papers by author(s) and potentially filter by topic/year

# Step 1: Extract key terms from query
query_terms = "{query}".lower()

# Step 2: Search for papers by title/content if topic mentioned
papers_by_topic = []
if any(word in query_terms for word in ['about', 'topic', 'regarding', 'concerning']):
    # Extract topic from query
    topic_part = query_terms
    if 'about' in topic_part:
        topic_part = topic_part.split('about')[-1].strip()
    elif 'topic' in topic_part:
        topic_part = topic_part.split('topic')[-1].strip()
    
    # Remove author-related terms
    for remove_word in ['author', 'from', 'by', 'written', 'papers']:
        topic_part = topic_part.replace(remove_word, '')
    
    topic_part = topic_part.strip()
    
    if topic_part:
        papers_by_topic = search_title(topic_part, top_k=100)
        papers_by_topic = [r['node_index'] for r in papers_by_topic]

# Step 3: Find authors if mentioned
authors = []
if any(word in query_terms for word in ['author', 'written', 'by']):
    # Extract author name from query
    author_part = query_terms
    if 'by' in author_part:
        author_part = author_part.split('by')[-1].strip()
    elif 'from' in author_part:
        author_part = author_part.split('from')[-1].strip()
    
    # Remove paper-related terms
    for remove_word in ['paper', 'papers', 'about', 'topic', 'the', 'a', 'an']:
        author_part = author_part.replace(remove_word, '')
    
    author_part = author_part.strip()
    
    if author_part:
        author_results = search_author_name(author_part, top_k=10)
        authors = [r['node_index'] for r in author_results]

# Step 4: Get papers by authors
papers_by_authors = []
if authors:
    papers_by_authors = get_papers_by_author(authors)

# Step 5: Combine results
if papers_by_topic and papers_by_authors:
    # Intersect topic papers with author papers
    result_papers = intersect_lists(papers_by_topic, papers_by_authors)
elif papers_by_authors:
    result_papers = papers_by_authors
elif papers_by_topic:
    result_papers = papers_by_topic
else:
    result_papers = []

# Step 6: Apply year filter if mentioned
if any(word in query_terms for word in ['year', 'between', 'from', 'to', 'after', 'before']):
    import re
    years = re.findall(r'\b(?:19|20)\d{2}\b', query_terms)
    if len(years) >= 1:
        year = int(years[0])
        # Get papers from that year or recent years
        year_papers = get_papers_by_year_range(max(1950, year-5), min(2024, year+5))
        if result_papers:
            result_papers = intersect_lists(result_papers, year_papers)
        else:
            result_papers = year_papers

# Final result
final_result = {{
    'papers': result_papers,
    'count': len(result_papers),
    'query': '{query}',
    'strategy': 'author_paper_search',
    'steps': [
        {{'step': 'topic_search', 'count': len(papers_by_topic)}},
        {{'step': 'author_search', 'count': len(authors)}},
        {{'step': 'author_papers', 'count': len(papers_by_authors)}},
        {{'step': 'final_intersection', 'count': len(result_papers)}}
    ]
}}

result = final_result
"""
    
    def _generate_year_range_query_code(self, query: str, system_info: Dict[str, Any]) -> str:
        """Generate code for year range queries"""
        return f"""
# Query: {query}
# Strategy: Find papers within year range, optionally filter by topic

import re

# Extract years from query
            years = re.findall(r'\b(19|20)\d{2}\b', "{query}")
start_year = 1950
end_year = 2024

if len(years) >= 2:
    start_year = min(int(years[0]), int(years[1]))
    end_year = max(int(years[0]), int(years[1]))
elif len(years) == 1:
    year = int(years[0])
    start_year = year - 5
    end_year = year + 5

# Get papers in year range
year_papers = get_papers_by_year_range(start_year, end_year)

# Check if topic filtering needed
query_terms = "{query}".lower()
topic_papers = []

if any(word in query_terms for word in ['about', 'topic', 'regarding', 'concerning']):
    # Extract topic
    topic_part = query_terms
    if 'about' in topic_part:
        topic_part = topic_part.split('about')[-1].strip()
    
    # Remove year-related terms
    for remove_word in ['year', 'years', 'between', 'from', 'to', 'after', 'before', 'papers', 'the']:
        topic_part = topic_part.replace(remove_word, '')
    
    topic_part = topic_part.strip()
    
    if topic_part:
        topic_results = search_title(topic_part, top_k=500)
        topic_papers = [r['node_index'] for r in topic_results]

# Combine results
if topic_papers:
    result_papers = intersect_lists(year_papers, topic_papers)
else:
    result_papers = year_papers

final_result = {{
    'papers': result_papers,
    'count': len(result_papers),
    'query': '{query}',
    'strategy': 'year_range_search',
    'year_range': [start_year, end_year],
    'steps': [
        {{'step': 'year_range', 'count': len(year_papers), 'range': [start_year, end_year]}},
        {{'step': 'topic_filter', 'count': len(topic_papers)}},
        {{'step': 'final_intersection', 'count': len(result_papers)}}
    ]
}}

result = final_result
"""
    
    def _generate_citation_query_code(self, query: str, system_info: Dict[str, Any]) -> str:
        """Generate code for citation queries"""
        return f"""
# Query: {query}
# Strategy: Find citation relationships

import re

# Extract paper title or ID from query
query_terms = "{query}".lower()

if 'cite' in query_terms or 'citation' in query_terms:
    # Find paper mentioned in query
    paper_results = search_title("{query}", top_k=10)
    
    if paper_results:
        target_paper = paper_results[0]['node_index']
        
        # Get papers citing this paper
        citing_papers = get_papers_citing(target_paper)
        
        # Get papers cited by this paper
        cited_papers = get_papers_cited_by(target_paper)
        
        final_result = {{
            'papers': citing_papers + cited_papers,
            'citing_papers': citing_papers,
            'cited_papers': cited_papers,
            'target_paper': target_paper,
            'count': len(citing_papers) + len(cited_papers),
            'query': '{query}',
            'strategy': 'citation_search',
            'steps': [
                {{'step': 'find_target_paper', 'paper_id': target_paper}},
                {{'step': 'citing_papers', 'count': len(citing_papers)}},
                {{'step': 'cited_papers', 'count': len(cited_papers)}}
            ]
        }}
    else:
        final_result = {{'papers': [], 'count': 0, 'error': 'No target paper found'}}
else:
    final_result = {{'papers': [], 'count': 0, 'error': 'No citation query detected'}}

result = final_result
"""
    
    def _generate_field_query_code(self, query: str, system_info: Dict[str, Any]) -> str:
        """Generate code for field/topic queries"""
        return f"""
# Query: {query}
# Strategy: Find papers by field/topic

query_terms = "{query}".lower()

# Extract topic/field from query
topic_part = "{query}"
if 'about' in topic_part.lower():
    topic_part = topic_part.split('about')[-1].strip()
elif 'topic' in topic_part.lower():
    topic_part = topic_part.split('topic')[-1].strip()
elif 'field' in topic_part.lower():
    topic_part = topic_part.split('field')[-1].strip()

# Clean up topic
for remove_word in ['papers', 'the', 'a', 'an', 'in', 'of', 'for']:
    topic_part = topic_part.replace(remove_word, '')

topic_part = topic_part.strip()

if topic_part:
    # Search by title
    title_results = search_title(topic_part, top_k=100)
    title_papers = [r['node_index'] for r in title_results]
    
    # Search by abstract
    abstract_results = search_abstract(topic_part, top_k=100)
    abstract_papers = [r['node_index'] for r in abstract_results]
    
    # Search by content
    content_results = search_content(topic_part, top_k=100)
    content_papers = [r['node_index'] for r in content_results]
    
    # Combine all results
    all_papers = union_lists([title_papers, abstract_papers, content_papers])
    
    # Remove duplicates while preserving order
    result_papers = list(dict.fromkeys(all_papers))
else:
    result_papers = []

final_result = {{
    'papers': result_papers,
    'count': len(result_papers),
    'query': '{query}',
    'strategy': 'field_topic_search',
    'topic': topic_part,
    'steps': [
        {{'step': 'title_search', 'count': len(title_papers)}},
        {{'step': 'abstract_search', 'count': len(abstract_papers)}},
        {{'step': 'content_search', 'count': len(content_papers)}},
        {{'step': 'union_all', 'count': len(result_papers)}}
    ]
}}

result = final_result
"""
    
    def _generate_institution_query_code(self, query: str, system_info: Dict[str, Any]) -> str:
        """Generate code for institution queries"""
        return f"""
# Query: {query}
# Strategy: Find papers from institution

query_terms = "{query}".lower()

# Extract institution name
institution_part = "{query}"
if 'from' in institution_part.lower():
    institution_part = institution_part.split('from')[-1].strip()
elif 'institution' in institution_part.lower():
    institution_part = institution_part.replace('institution', '').strip()

# Clean up
for remove_word in ['papers', 'the', 'a', 'an', 'in', 'of', 'for', 'at']:
    institution_part = institution_part.replace(remove_word, '')

institution_part = institution_part.strip()

# Search for institution (this would need institution search capability)
# For now, search by author affiliation in metadata
institution_papers = []

# Get all papers and filter by institution in author metadata
# This is a simplified approach - in practice you'd want institution search
all_papers = []
for mag_id, attrs in agent.graph_loader.node_attrs.items():
    if agent.graph_loader.get_node_type(mag_id) == 'paper':
        all_papers.append(mag_id)

# Sample a subset for performance
sample_size = min(10000, len(all_papers))
sample_papers = all_papers[:sample_size]

# Check author affiliations
institution_related_papers = []
for paper_id in sample_papers:
    authors = get_authors_of_paper(paper_id)
    for author_id in authors[:3]:  # Check first 3 authors
        author_meta = get_author_metadata(author_id)
        if author_meta and 'institution' in str(author_meta).lower():
            if institution_part.lower() in str(author_meta).lower():
                institution_related_papers.append(paper_id)
                break

final_result = {{
    'papers': institution_related_papers,
    'count': len(institution_related_papers),
    'query': '{query}',
    'strategy': 'institution_search',
    'institution': institution_part,
    'note': 'Limited to sample of papers for performance',
    'steps': [
        {{'step': 'sample_papers', 'count': sample_size}},
        {{'step': 'check_affiliations', 'count': len(institution_related_papers)}}
    ]
}}

result = final_result
"""
    
    def _generate_general_search_code(self, query: str, system_info: Dict[str, Any]) -> str:
        """Generate code for general search queries"""
        return f"""
# Query: {query}
# Strategy: General multi-modal search

query_terms = "{query}".lower()

# Search across all available features
title_results = search_title("{query}", top_k=50)
title_papers = [r['node_index'] for r in title_results]

abstract_results = search_abstract("{query}", top_k=50)
abstract_papers = [r['node_index'] for r in abstract_results]

content_results = search_content("{query}", top_k=50)
content_papers = [r['node_index'] for r in content_results]

# Combine results with scoring
all_results = {{}}

# Score by search type
for paper_id in title_results:
    node_index = paper_id['node_index']
    score = paper_id.get('score', 0.0)
    all_results[node_index] = all_results.get(node_index, 0) + score * 3  # Title gets 3x weight

for paper_id in abstract_results:
    node_index = paper_id['node_index']
    score = paper_id.get('score', 0.0)
    all_results[node_index] = all_results.get(node_index, 0) + score * 2  # Abstract gets 2x weight

for paper_id in content_results:
    node_index = paper_id['node_index']
    score = paper_id.get('score', 0.0)
    all_results[node_index] = all_results.get(node_index, 0) + score * 1  # Content gets 1x weight

# Sort by combined score
sorted_papers = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
result_papers = [node_index for node_index, score in sorted_papers]

final_result = {{
    'papers': result_papers,
    'count': len(result_papers),
    'query': '{query}',
    'strategy': 'general_multi_modal_search',
    'scores': dict(sorted_papers[:20]),  # Top 20 with scores
    'steps': [
        {{'step': 'title_search', 'count': len(title_papers)}},
        {{'step': 'abstract_search', 'count': len(abstract_papers)}},
        {{'step': 'content_search', 'count': len(content_papers)}},
        {{'step': 'score_combination', 'count': len(result_papers)}}
    ]
}}

result = final_result
"""
    
    # Helper methods for dynamic execution
    def _search_title(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search papers by title"""
        if not self.hnsw_manager.is_available('original_title_embedding'):
            return []
        
        query_embedding = self.encoder.encode([query])[0].astype(np.float32)
        return self.hnsw_manager.search('original_title_embedding', query_embedding, top_k)
    
    def _search_abstract(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search papers by abstract"""
        if not self.hnsw_manager.is_available('abstract_embedding'):
            return []
        
        query_embedding = self.encoder.encode([query])[0].astype(np.float32)
        return self.hnsw_manager.search('abstract_embedding', query_embedding, top_k)
    
    def _search_content(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search papers by content (using abstract since content_embedding is not available)"""
        if not self.hnsw_manager.is_available('abstract_embedding'):
            return []
        
        query_embedding = self.encoder.encode([query])[0].astype(np.float32)
        return self.hnsw_manager.search('abstract_embedding', query_embedding, top_k)
    
    def _search_author_name(self, name: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search authors by name"""
        if not self.hnsw_manager.is_available('author_embedding'):
            return []
        
        query_embedding = self.encoder.encode([name])[0].astype(np.float32)
        return self.hnsw_manager.search('author_embedding', query_embedding, top_k)
    
    def _get_authors_of_paper(self, paper_id: int) -> List[int]:
        """Get authors of a paper"""
        return self.traversal_utils.authors_of_paper(paper_id)
    
    def _get_papers_by_author(self, author_ids: List[int]) -> List[int]:
        """Get papers by authors"""
        return self.traversal_utils.papers_by_author(author_ids)
    
    def _get_papers_with_field(self, field_id: int) -> List[int]:
        """Get papers with field"""
        return self.traversal_utils.papers_with_field(field_id)
    
    def _get_papers_citing(self, paper_id: int) -> List[int]:
        """Get papers citing"""
        return self.traversal_utils.papers_citing(paper_id)
    
    def _get_papers_cited_by(self, paper_id: int) -> List[int]:
        """Get papers cited by"""
        return self.traversal_utils.papers_cited_by(paper_id)
    
    def _get_papers_by_year_range(self, start_year: int, end_year: int) -> List[int]:
        """Get papers by year range"""
        return self.traversal_utils.papers_by_year_range(start_year, end_year)
    
    def _get_papers_by_institution(self, institution_id: int) -> List[int]:
        """Get papers by institution"""
        return self.traversal_utils.papers_by_institution(institution_id)
    
    def _get_authors_affiliated_with(self, institution_id: int) -> List[int]:
        """Get authors affiliated with"""
        return self.traversal_utils.authors_affiliated_with(institution_id)
    
    def _get_paper_metadata(self, paper_id: int) -> Optional[Dict[str, Any]]:
        """Get paper metadata"""
        return self.traversal_utils.get_paper_metadata(paper_id)
    
    def _get_author_metadata(self, author_id: int) -> Optional[Dict[str, Any]]:
        """Get author metadata"""
        return self.traversal_utils.get_author_metadata(author_id)
    
    def _get_institution_metadata(self, institution_id: int) -> Optional[Dict[str, Any]]:
        """Get institution metadata"""
        return self.traversal_utils.get_institution_metadata(institution_id)
    
    def _get_field_metadata(self, field_id: int) -> Optional[Dict[str, Any]]:
        """Get field metadata"""
        return self.traversal_utils.get_field_metadata(field_id)
    
    def _intersect_lists(self, list1: List[int], list2: List[int]) -> List[int]:
        """Intersect two lists"""
        return list(set(list1) & set(list2))
    
    def _union_lists(self, lists: List[List[int]]) -> List[int]:
        """Union multiple lists"""
        result = set()
        for lst in lists:
            result.update(lst)
        return list(result)
    
    def _filter_papers_by_criteria(self, papers: List[int], criteria: Dict[str, Any]) -> List[int]:
        """Filter papers by metadata criteria"""
        filtered = []
        for paper_id in papers:
            metadata = self._get_paper_metadata(paper_id)
            if metadata and self._matches_criteria(metadata, criteria):
                filtered.append(paper_id)
        return filtered
    
    def _matches_criteria(self, metadata: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
        """Check if metadata matches criteria"""
        for key, expected_value in criteria.items():
            if key not in metadata:
                return False
            
            actual_value = metadata[key]
            
            if isinstance(expected_value, dict):
                if 'min' in expected_value and actual_value < expected_value['min']:
                    return False
                if 'max' in expected_value and actual_value > expected_value['max']:
                    return False
            elif actual_value != expected_value:
                return False
        
        return True
    
    def _encode_query(self, query: str) -> np.ndarray:
        """Encode query to embedding"""
        return self.encoder.encode([query])[0].astype(np.float32)
    
    def _save_query_evidence(self, question_id: str, result: Dict[str, Any]):
        """Save query evidence"""
        output_path = Path("/shared/khoja/CogComp/agent/output") / "qa"
        output_path.mkdir(parents=True, exist_ok=True)
        
        evidence = {
            'question_id': question_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'query': result['query'],
            'generated_code': result.get('generated_code', ''),
            'execution_result': result.get('execution_result', {}),
            'system_info': result.get('system_info', {}),
            'agent_stats': self.get_agent_stats()
        }
        
        evidence_file = output_path / f"{question_id}.json"
        with open(evidence_file, 'w') as f:
            json.dump(evidence, f, indent=2)
        
        logger.info(f"💾 Saved query evidence to {evidence_file}")
    
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


def main():
    """Test the flexible MAG agent"""
    logger.info("🧬 Testing Flexible MAG Agent")
    
    try:
        # Initialize agent
        agent = FlexibleMAGAgent(
            processed_dir="/shared/khoja/CogComp/datasets/MAG/processed",
            indices_dir="/shared/khoja/CogComp/output/mag_hnsw_indices"
        )
        
        # Load all components
        if not agent.load_all():
            return False
        
        logger.info("✅ Flexible MAG Agent loaded successfully!")
        
        # Test queries
        test_queries = [
            "papers about machine learning from 2010 to 2020",
            "papers by authors of machine learning papers",
            "papers citing quantum computing research",
            "papers from MIT about artificial intelligence"
        ]
        
        for i, query in enumerate(test_queries):
            logger.info(f"🔍 Testing query {i+1}: {query}")
            result = agent.solve_query(query, f"test_{i+1}")
            
            if result['success']:
                execution_result = result['execution_result']
                if 'results' in execution_result:
                    papers = execution_result['results'].get('papers', [])
                    logger.info(f"  ✅ Found {len(papers)} papers")
                else:
                    logger.info(f"  ✅ Execution completed")
            else:
                logger.error(f"  ❌ Query failed: {result.get('error', 'Unknown error')}")
        
        # Print statistics
        logger.info("📊 Agent statistics:")
        stats = agent.get_agent_stats()
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to test Flexible MAG Agent: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
