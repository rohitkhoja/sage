#!/usr/bin/env python3
"""
Query Orchestrator for MAG Dataset
Decomposes natural language questions into sub-steps combining HNSW retrieval and graph traversal
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from sentence_transformers import SentenceTransformer
from loguru import logger

from hnsw_manager import MAGHNSWManager
from traversal_utils import MAGTraversalUtils


class MAGQueryOrchestrator:
    """Orchestrates complex queries combining HNSW search and graph traversal"""
    
    def __init__(self, hnsw_manager: MAGHNSWManager, traversal_utils: MAGTraversalUtils):
        self.hnsw_manager = hnsw_manager
        self.traversal_utils = traversal_utils
        
        # Load sentence transformer for query encoding (robust local + fallback strategy)
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
                    # Use the first snapshot folder
                    candidate = candidates[0]
            except Exception:
                candidate = None
            model_path = candidate if candidate else local_model_root
            self.encoder = SentenceTransformer(model_path)
            logger.info(f" Sentence transformer loaded from local path: {model_path}")
        except Exception as e_local:
            logger.warning(f" Local model load failed: {e_local}")
            # Try to repair local cache by removing invalid folder and re-downloading
            try:
                import shutil
                if os.path.isdir(local_model_root):
                    logger.info(f" Removing invalid local model directory: {local_model_root}")
                    shutil.rmtree(local_model_root, ignore_errors=True)
            except Exception as e_rm:
                logger.warning(f" Failed to remove local model dir: {e_rm}")
            # Attempt fresh download into cache
            try:
                self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                logger.info(" Sentence transformer re-downloaded from Hugging Face")
            except Exception as e_hf:
                logger.error(f" Failed to load sentence transformer from HF: {e_hf}")
                self.encoder = None
        
        # Query step templates
        self.step_templates = {
            'find_paper_by_title': self._find_paper_by_title,
            'find_paper_by_abstract': self._find_paper_by_abstract,
            'find_paper_by_content': self._find_paper_by_content,
            'find_author_by_name': self._find_author_by_name,
            'find_field_by_name': self._find_field_by_name,
            'authors_of_paper': self._get_authors_of_paper,
            'papers_by_author': self._get_papers_by_author,
            'papers_with_field': self._get_papers_with_field,
            'papers_citing': self._get_papers_citing,
            'papers_cited_by': self._get_papers_cited_by,
            'papers_by_year_range': self._get_papers_by_year_range,
            'papers_by_institution': self._get_papers_by_institution,
            'intersect_candidates': self._intersect_candidates,
            'union_candidates': self._union_candidates,
            'filter_by_criteria': self._filter_by_criteria
        }
    
    def encode_query(self, query_text: str) -> np.ndarray:
        """Encode query text to embedding vector"""
        if self.encoder is None:
            raise RuntimeError("Sentence transformer not loaded")
        
        embedding = self.encoder.encode([query_text])[0]
        return embedding.astype(np.float32)
    
    def _find_paper_by_title(self, query_text: str) -> Dict[str, Any]:
        """Find papers by title similarity"""
        try:
            logger.info(f" _find_paper_by_title called with query='{query_text}'")
            query_embedding = self.encode_query(query_text)
            logger.info(f" Query embedding shape: {query_embedding.shape}")
            
            results = self.hnsw_manager.search('original_title_embedding', query_embedding)
            logger.info(f" HNSW search returned {len(results)} results")
            logger.info(f" First HNSW result: {results[0] if results else 'No results'}")
            
            # HNSW already returns metadata from Neo4j, just reformat for backward compatibility
            enriched_results = []
            for i, result in enumerate(results):
                enriched_result = result.copy()
                # Copy metadata to node_data field for backward compatibility
                if 'metadata' in result:
                    enriched_result['node_data'] = result['metadata']
                enriched_results.append(enriched_result)
            
            logger.info(f" Final enriched results count: {len(enriched_results)}")
            return {
                'step_name': 'find_paper_by_title',
                'query_text': query_text,
                'results': enriched_results,
                'count': len(enriched_results),
                'confidence': self._calculate_confidence(enriched_results)
            }
        except Exception as e:
            logger.error(f"Error in find_paper_by_title: {e}")
            return {'step_name': 'find_paper_by_title', 'error': str(e), 'results': []}
    
    def _find_paper_by_abstract(self, query_text: str) -> Dict[str, Any]:
        """Find papers by abstract similarity"""
        try:
            query_embedding = self.encode_query(query_text)
            results = self.hnsw_manager.search('abstract_embedding', query_embedding)
            
            # HNSW already returns metadata from Neo4j
            enriched_results = []
            for result in results:
                enriched_result = result.copy()
                if 'metadata' in result:
                    enriched_result['node_data'] = result['metadata']
                enriched_results.append(enriched_result)
            
            return {
                'step_name': 'find_paper_by_abstract',
                'query_text': query_text,
                'results': enriched_results,
                'count': len(enriched_results),
                'confidence': self._calculate_confidence(enriched_results)
            }
        except Exception as e:
            logger.error(f"Error in find_paper_by_abstract: {e}")
            return {'step_name': 'find_paper_by_abstract', 'error': str(e), 'results': []}
    
    def _find_paper_by_content(self, query_text: str) -> Dict[str, Any]:
        """Find papers by content similarity (using abstract as content is not available)"""
        try:
            query_embedding = self.encode_query(query_text)
            # Use abstract_embedding since content_embedding is not available
            results = self.hnsw_manager.search('abstract_embedding', query_embedding)
            
            # HNSW already returns metadata from Neo4j
            enriched_results = []
            for result in results:
                enriched_result = result.copy()
                if 'metadata' in result:
                    enriched_result['node_data'] = result['metadata']
                enriched_results.append(enriched_result)
            
            return {
                'step_name': 'find_paper_by_content',
                'query_text': query_text,
                'results': enriched_results,
                'count': len(enriched_results),
                'confidence': self._calculate_confidence(enriched_results),
                'note': 'Using abstract_embedding as content_embedding is not available'
            }
        except Exception as e:
            logger.error(f"Error in find_paper_by_content: {e}")
            return {'step_name': 'find_paper_by_content', 'error': str(e), 'results': []}
    
    def _find_author_by_name(self, author_name: str) -> Dict[str, Any]:
        """Find authors by name similarity"""
        try:
            query_embedding = self.encode_query(author_name)
            # Use author_embedding since display_name_embedding is not available
            results = self.hnsw_manager.search('author_embedding', query_embedding)
            
            # HNSW already returns metadata from Neo4j
            enriched_results = []
            for result in results:
                enriched_result = result.copy()
                if 'metadata' in result:
                    enriched_result['node_data'] = result['metadata']
                enriched_results.append(enriched_result)
            
            return {
                'step_name': 'find_author_by_name',
                'query_text': author_name,
                'results': enriched_results,
                'count': len(enriched_results),
                'confidence': self._calculate_confidence(enriched_results),
                'note': 'Using author_embedding as display_name_embedding is not available'
            }
        except Exception as e:
            logger.error(f"Error in find_author_by_name: {e}")
            return {'step_name': 'find_author_by_name', 'error': str(e), 'results': []}
    
    def _find_institution_by_name(self, institution_name: str) -> Dict[str, Any]:
        """Find institutions by name similarity"""
        try:
            logger.info(f" _find_institution_by_name called with query='{institution_name}'")
            query_embedding = self.encode_query(institution_name)
            logger.info(f" Query embedding shape: {query_embedding.shape}")
            
            results = self.hnsw_manager.search('institution_embedding', query_embedding)
            logger.info(f" HNSW search returned {len(results)} results")
            logger.info(f" First HNSW result: {results[0] if results else 'No results'}")
            
            # HNSW already returns metadata from Neo4j
            enriched_results = []
            for i, result in enumerate(results):
                enriched_result = result.copy()
                if 'metadata' in result:
                    enriched_result['node_data'] = result['metadata']
                enriched_results.append(enriched_result)
            
            logger.info(f" Final enriched results count: {len(enriched_results)}")
            return {
                'step_name': 'find_institution_by_name',
                'query_text': institution_name,
                'results': enriched_results,
                'count': len(enriched_results),
                'confidence': self._calculate_confidence(enriched_results)
            }
        except Exception as e:
            logger.error(f"Error in find_institution_by_name: {e}")
            return {'step_name': 'find_institution_by_name', 'error': str(e), 'results': []}
    
    def _find_field_by_name(self, field_name: str) -> Dict[str, Any]:
        """Find fields of study by name similarity"""
        try:
            logger.info(f" _find_field_by_name called with query='{field_name}'")
            query_embedding = self.encode_query(field_name)
            logger.info(f" Query embedding shape: {query_embedding.shape}")
            
            # Use only field_display_name_embedding
            if not self.hnsw_manager.is_available('field_display_name_embedding'):
                logger.warning(" Field display name embedding not available")
                return {'step_name': 'find_field_by_name', 'error': 'Field display name embedding not available', 'results': []}
            
            logger.info(f" Searching with field_display_name_embedding")
            results = self.hnsw_manager.search('field_display_name_embedding', query_embedding)
            logger.info(f" HNSW search returned {len(results)} results")
            
            if not results:
                logger.warning(" No field results found")
                return {'step_name': 'find_field_by_name', 'error': 'No field results found', 'results': []}
            
            logger.info(f" First HNSW result: {results[0] if results else 'No results'}")
            
            # HNSW already returns metadata from Neo4j
            enriched_results = []
            for i, result in enumerate(results):
                enriched_result = result.copy()
                if 'metadata' in result:
                    enriched_result['node_data'] = result['metadata']
                enriched_results.append(enriched_result)
            
            logger.info(f" Final enriched results count: {len(enriched_results)}")
            return {
                'step_name': 'find_field_by_name',
                'query_text': field_name,
                'results': enriched_results,
                'count': len(enriched_results),
                'confidence': self._calculate_confidence(enriched_results)
            }
        except Exception as e:
            logger.error(f"Error in find_field_by_name: {e}")
            return {'step_name': 'find_field_by_name', 'error': str(e), 'results': []}
    
    def _find_paper_by_abstract_duplicate(self, abstract_query: str) -> Dict[str, Any]:
        """Find papers by abstract similarity"""
        try:
            logger.info(f" _find_paper_by_abstract called with query='{abstract_query}'")
            query_embedding = self.encode_query(abstract_query)
            logger.info(f" Query embedding shape: {query_embedding.shape}")
            
            # Use abstract_embedding
            if not self.hnsw_manager.is_available('abstract_embedding'):
                logger.warning(" Abstract embedding not available")
                return {'step_name': 'find_paper_by_abstract', 'error': 'Abstract embedding not available', 'results': []}
            
            logger.info(f" Searching with abstract_embedding")
            results = self.hnsw_manager.search('abstract_embedding', query_embedding)
            logger.info(f" HNSW search returned {len(results)} results")
            
            if not results:
                logger.warning(" No abstract results found")
                return {'step_name': 'find_paper_by_abstract', 'error': 'No abstract results found', 'results': []}
            
            logger.info(f" First HNSW result: {results[0] if results else 'No results'}")
            
            # HNSW already returns metadata from Neo4j
            enriched_results = []
            for i, result in enumerate(results):
                enriched_result = result.copy()
                if 'metadata' in result:
                    enriched_result['node_data'] = result['metadata']
                enriched_results.append(enriched_result)
            
            logger.info(f" Final enriched results count: {len(enriched_results)}")
            return {
                'step_name': 'find_paper_by_abstract',
                'query_text': abstract_query,
                'results': enriched_results,
                'count': len(enriched_results),
                'confidence': self._calculate_confidence(enriched_results)
            }
        except Exception as e:
            logger.error(f"Error in find_paper_by_abstract: {e}")
            return {'step_name': 'find_paper_by_abstract', 'error': str(e), 'results': []}
    
    def _get_authors_of_paper(self, paper_id: int) -> Dict[str, Any]:
        """Get authors of a specific paper"""
        try:
            authors = self.traversal_utils.authors_of_paper(paper_id)
            
            return {
                'step_name': 'authors_of_paper',
                'paper_id': paper_id,
                'results': [{'node_index': aid} for aid in authors],
                'count': len(authors)
            }
        except Exception as e:
            logger.error(f"Error in get_authors_of_paper: {e}")
            return {'step_name': 'authors_of_paper', 'error': str(e), 'results': []}
    
    def _get_authors_affiliated_with(self, institution_id: int) -> Dict[str, Any]:
        """Get authors affiliated with an institution"""
        try:
            authors = self.traversal_utils.authors_affiliated_with(institution_id)
            
            return {
                'step_name': 'authors_affiliated_with',
                'institution_id': institution_id,
                'results': [{'node_index': aid} for aid in authors],
                'count': len(authors)
            }
        except Exception as e:
            logger.error(f"Error in get_authors_affiliated_with: {e}")
            return {'step_name': 'authors_affiliated_with', 'error': str(e), 'results': []}
        """Get authors of a specific paper"""
        try:
            authors = self.traversal_utils.authors_of_paper(paper_id)
            
            return {
                'step_name': 'authors_of_paper',
                'paper_id': paper_id,
                'results': [{'node_index': aid} for aid in authors],
                'count': len(authors)
            }
        except Exception as e:
            logger.error(f"Error in get_authors_of_paper: {e}")
            return {'step_name': 'authors_of_paper', 'error': str(e), 'results': []}
    
    def _get_papers_by_author(self, author_ids: List[int]) -> Dict[str, Any]:
        """Get papers written by specific authors"""
        try:
            papers = self.traversal_utils.papers_by_author(author_ids)
            
            return {
                'step_name': 'papers_by_author',
                'author_ids': author_ids,
                'results': [{'node_index': pid} for pid in papers],
                'count': len(papers)
            }
        except Exception as e:
            logger.error(f"Error in get_papers_by_author: {e}")
            return {'step_name': 'papers_by_author', 'error': str(e), 'results': []}
    
    def _get_papers_with_field(self, field_id: int) -> Dict[str, Any]:
        """Get papers tagged with a specific field"""
        try:
            papers = self.traversal_utils.papers_with_field(field_id)
            
            return {
                'step_name': 'papers_with_field',
                'field_id': field_id,
                'results': [{'node_index': pid} for pid in papers],
                'count': len(papers)
            }
        except Exception as e:
            logger.error(f"Error in get_papers_with_field: {e}")
            return {'step_name': 'papers_with_field', 'error': str(e), 'results': []}
    
    def _get_papers_citing(self, paper_id: int) -> Dict[str, Any]:
        """Get papers that cite a specific paper"""
        try:
            papers = self.traversal_utils.papers_citing(paper_id)
            
            return {
                'step_name': 'papers_citing',
                'cited_paper_id': paper_id,
                'results': [{'node_index': pid} for pid in papers],
                'count': len(papers)
            }
        except Exception as e:
            logger.error(f"Error in get_papers_citing: {e}")
            return {'step_name': 'papers_citing', 'error': str(e), 'results': []}
    
    def _get_papers_cited_by(self, paper_id: int) -> Dict[str, Any]:
        """Get papers cited by a specific paper"""
        try:
            papers = self.traversal_utils.papers_cited_by(paper_id)
            
            return {
                'step_name': 'papers_cited_by',
                'citing_paper_id': paper_id,
                'results': [{'node_index': pid} for pid in papers],
                'count': len(papers)
            }
        except Exception as e:
            logger.error(f"Error in get_papers_cited_by: {e}")
            return {'step_name': 'papers_cited_by', 'error': str(e), 'results': []}
    
    def _get_papers_by_year_range(self, start_year: int, end_year: int) -> Dict[str, Any]:
        """Get papers published in a year range"""
        try:
            papers = self.traversal_utils.papers_by_year_range(start_year, end_year)
            
            return {
                'step_name': 'papers_by_year_range',
                'start_year': start_year,
                'end_year': end_year,
                'results': [{'node_index': pid} for pid in papers],
                'count': len(papers)
            }
        except Exception as e:
            logger.error(f"Error in get_papers_by_year_range: {e}")
            return {'step_name': 'papers_by_year_range', 'error': str(e), 'results': []}
    
    def _get_papers_by_institution(self, institution_id: int) -> Dict[str, Any]:
        """Get papers from authors at an institution"""
        try:
            papers = self.traversal_utils.papers_by_institution(institution_id)
            
            return {
                'step_name': 'papers_by_institution',
                'institution_id': institution_id,
                'results': [{'node_index': pid} for pid in papers],
                'count': len(papers)
            }
        except Exception as e:
            logger.error(f"Error in get_papers_by_institution: {e}")
            return {'step_name': 'papers_by_institution', 'error': str(e), 'results': []}
    
    def _intersect_candidates(self, candidate_sets: List[List[int]]) -> Dict[str, Any]:
        """Intersect multiple candidate sets"""
        try:
            if not candidate_sets:
                return {'step_name': 'intersect_candidates', 'results': [], 'count': 0}
            
            # Start with first set
            intersection = set(candidate_sets[0])
            
            # Intersect with remaining sets
            for candidate_set in candidate_sets[1:]:
                intersection = intersection.intersection(set(candidate_set))
            
            result = list(intersection)
            
            return {
                'step_name': 'intersect_candidates',
                'input_sets_count': len(candidate_sets),
                'results': [{'node_index': pid} for pid in result],
                'count': len(result)
            }
        except Exception as e:
            logger.error(f"Error in intersect_candidates: {e}")
            return {'step_name': 'intersect_candidates', 'error': str(e), 'results': []}
    
    def _union_candidates(self, candidate_sets: List[List[int]]) -> Dict[str, Any]:
        """Union multiple candidate sets"""
        try:
            if not candidate_sets:
                return {'step_name': 'union_candidates', 'results': [], 'count': 0}
            
            union = set()
            for candidate_set in candidate_sets:
                union.update(candidate_set)
            
            result = list(union)
            
            return {
                'step_name': 'union_candidates',
                'input_sets_count': len(candidate_sets),
                'results': [{'node_index': pid} for pid in result],
                'count': len(result)
            }
        except Exception as e:
            logger.error(f"Error in union_candidates: {e}")
            return {'step_name': 'union_candidates', 'error': str(e), 'results': []}
    
    def _filter_by_criteria(self, candidates: List[int], criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Filter candidates by metadata criteria"""
        try:
            filtered = []
            
            for candidate_id in candidates:
                metadata = self.traversal_utils.get_paper_metadata(candidate_id)
                if metadata and self._matches_criteria(metadata, criteria):
                    filtered.append(candidate_id)
            
            return {
                'step_name': 'filter_by_criteria',
                'criteria': criteria,
                'input_count': len(candidates),
                'results': [{'node_index': pid} for pid in filtered],
                'count': len(filtered)
            }
        except Exception as e:
            logger.error(f"Error in filter_by_criteria: {e}")
            return {'step_name': 'filter_by_criteria', 'error': str(e), 'results': []}
    
    def _matches_criteria(self, metadata: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
        """Check if metadata matches given criteria"""
        for key, expected_value in criteria.items():
            if key not in metadata:
                return False
            
            actual_value = metadata[key]
            
            if isinstance(expected_value, dict):
                # Range criteria (e.g., {'min': 2010, 'max': 2020})
                if 'min' in expected_value and actual_value < expected_value['min']:
                    return False
                if 'max' in expected_value and actual_value > expected_value['max']:
                    return False
            elif actual_value != expected_value:
                return False
        
        return True
    
    def _calculate_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on result quality"""
        if not results:
            return 0.0
        
        # Simple confidence based on top score
        top_score = results[0].get('score', 0.0)
        
        # Normalize to 0-1 range (assuming scores are typically 0-1 for cosine similarity)
        confidence = min(1.0, max(0.0, top_score))
        
        return confidence
    
    def execute_query_plan(self, query_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a multi-step query plan"""
        start_time = time.time()
        
        steps = []
        final_results = []
        
        try:
            for i, step in enumerate(query_plan):
                logger.info(f"Executing step {i+1}: {step.get('action', 'unknown')}")
                
                action = step.get('action')
                params = step.get('params', {})
                
                if action in self.step_templates:
                    step_result = self.step_templates[action](**params)
                    steps.append(step_result)
                    
                    # Extract results for next steps
                    if 'results' in step_result:
                        results = step_result['results']
                        if results:
                            # Extract node_index from results
                            node_indices = [r.get('node_index') for r in results if r.get('node_index')]
                            if node_indices:
                                final_results.extend(node_indices)
                else:
                    logger.warning(f"Unknown action: {action}")
                    steps.append({
                        'step_name': action,
                        'error': f'Unknown action: {action}',
                        'results': []
                    })
            
            # Deduplicate final results
            final_results = list(set(final_results))
            
            execution_time = time.time() - start_time
            
            return {
                'success': True,
                'execution_time_seconds': execution_time,
                'steps': steps,
                'final_results': final_results,
                'final_count': len(final_results)
            }
            
        except Exception as e:
            logger.error(f"Error executing query plan: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time_seconds': time.time() - start_time,
                'steps': steps,
                'final_results': [],
                'final_count': 0
            }
    
    def parse_natural_language_query(self, query: str) -> List[Dict[str, Any]]:
        """Parse natural language query into execution plan (simplified)"""
        query_lower = query.lower()
        
        # Simple keyword-based parsing (can be enhanced with NLP)
        plan = []
        
        # Look for paper title mentions
        if 'paper' in query_lower and ('title' in query_lower or 'about' in query_lower):
            # Extract the title/topic part (simplified)
            if 'about' in query_lower:
                topic_part = query_lower.split('about')[-1].strip()
                plan.append({
                    'action': 'find_paper_by_content',
                    'params': {'query_text': topic_part, 'top_k': 50}
                })
        
        # Look for author mentions
        if 'author' in query_lower and 'papers' in query_lower:
            if 'from' in query_lower:
                author_part = query_lower.split('from')[-1].split('that')[0].strip()
                plan.append({
                    'action': 'find_author_by_name',
                    'params': {'author_name': author_part, 'top_k': 10}
                })
        
        # Look for year range
        if 'year' in query_lower or 'between' in query_lower:
            # Extract years (simplified)
            import re
            years = re.findall(r'\b(19|20)\d{2}\b', query)
            if len(years) >= 2:
                start_year, end_year = int(years[0]), int(years[1])
                plan.append({
                    'action': 'papers_by_year_range',
                    'params': {'start_year': start_year, 'end_year': end_year}
                })
        
        return plan


def main():
    """Test the query orchestrator"""
    from hnsw_manager import MAGHNSWManager
    from traversal_utils import MAGTraversalUtils
    from graph_loader import MAGGraphLoader
    
    logger.info(" Testing MAG Query Orchestrator")
    
    try:
        # Initialize components
        hnsw_manager = MAGHNSWManager("/shared/khoja/CogComp/output/mag_hnsw_indices")
        hnsw_manager.load_all_indices()
        
        graph_loader = MAGGraphLoader("/shared/khoja/CogComp/datasets/MAG/processed")
        graph = graph_loader.build_graph()
        
        traversal_utils = MAGTraversalUtils(graph, graph_loader)
        
        orchestrator = MAGQueryOrchestrator(hnsw_manager, traversal_utils)
        
        logger.info(" Query orchestrator initialized successfully!")
        
        # Test simple query
        test_query = "papers about machine learning from 2010 to 2020"
        plan = orchestrator.parse_natural_language_query(test_query)
        logger.info(f"Parsed plan for '{test_query}': {plan}")
        
        if plan:
            result = orchestrator.execute_query_plan(plan)
            logger.info(f"Query result: {result['final_count']} papers found")
        
        return True
        
    except Exception as e:
        logger.error(f" Failed to test query orchestrator: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
