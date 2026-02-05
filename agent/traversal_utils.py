#!/usr/bin/env python3
"""
Graph Traversal Utilities for MAG Dataset
Provides functions to traverse the heterogeneous graph and extract relationships
"""

import torch
from typing import Dict, List, Set, Tuple, Optional, Any
from torch_geometric.data import HeteroData
from loguru import logger


class MAGTraversalUtils:
    """Utilities for traversing the MAG heterogeneous graph"""
    
    def __init__(self, graph: HeteroData, graph_loader):
        self.graph = graph
        self.graph_loader = graph_loader
        
        # Edge type mappings for easy access
        self.edge_types = {
            'author_writes_paper': ('author', 'writes', 'paper'),
            'paper_cites_paper': ('paper', 'cites', 'paper'),
            'paper_has_topic_field': ('paper', 'has_topic', 'field_of_study'),
            'author_affiliated_with_institution': ('author', 'affiliated_with', 'institution')
        }
    
    def _get_local_indices(self, node_indices: List[int], node_type: str) -> List[int]:
        """Convert node_index to local indices within a node type"""
        local_indices = []
        for node_index in node_indices:
            local_idx = self.graph_loader.get_local_index(node_index, node_type)
            if local_idx is not None:
                local_indices.append(local_idx)
        return local_indices
    
    def _get_node_indices_from_local(self, local_indices: List[int], node_type: str) -> List[int]:
        """Convert local indices back to node_index"""
        node_indices = []
        type_to_local = self.graph_loader.type_to_local[node_type]
        
        # Create reverse mapping
        local_to_node_index = {local_idx: node_index for node_index, local_idx in type_to_local.items()}
        
        for local_idx in local_indices:
            node_index = local_to_node_index.get(local_idx)
            if node_index is not None:
                node_indices.append(node_index)
        
        return node_indices
    
    def authors_of_paper(self, paper_node_index: int) -> List[int]:
        """Get all authors of a paper"""
        try:
            # Get local index of the paper
            paper_local_idx = self.graph_loader.get_local_index(paper_node_index, 'paper')
            if paper_local_idx is None:
                logger.warning(f"Paper {paper_node_index} not found")
                return []
            
            # Find edges where this paper is the target (author -> paper)
            edge_key = self.edge_types['author_writes_paper']
            if edge_key in self.graph.edge_types:
                edge_index = self.graph[edge_key].edge_index
                
                # Find all authors (source nodes) connected to this paper (target node)
                target_mask = edge_index[1] == paper_local_idx
                author_local_indices = edge_index[0][target_mask].tolist()
                
                # Convert back to node_index
                author_node_indices = self._get_node_indices_from_local(author_local_indices, 'author')
                
                logger.debug(f"Found {len(author_node_indices)} authors for paper {paper_node_index}")
                return author_node_indices
            
            return []
            
        except Exception as e:
            logger.error(f"Error finding authors for paper {paper_node_index}: {e}")
            return []
    
    def papers_by_author(self, author_node_indices: List[int]) -> List[int]:
        """Get all papers written by given authors"""
        try:
            all_paper_node_indices = set()
            
            for author_node_index in author_node_indices:
                # Get local index of the author
                author_local_idx = self.graph_loader.get_local_index(author_node_index, 'author')
                if author_local_idx is None:
                    logger.warning(f"Author {author_node_index} not found")
                    continue
                
                # Find edges where this author is the source (author -> paper)
                edge_key = self.edge_types['author_writes_paper']
                if edge_key in self.graph.edge_types:
                    edge_index = self.graph[edge_key].edge_index
                    
                    # Find all papers (target nodes) connected to this author (source node)
                    source_mask = edge_index[0] == author_local_idx
                    paper_local_indices = edge_index[1][source_mask].tolist()
                    
                    # Convert back to node_index
                    paper_node_indices = self._get_node_indices_from_local(paper_local_indices, 'paper')
                    all_paper_node_indices.update(paper_node_indices)
            
            result = list(all_paper_node_indices)
            logger.debug(f"Found {len(result)} papers for {len(author_node_indices)} authors")
            return result
            
        except Exception as e:
            logger.error(f"Error finding papers for authors {author_node_indices}: {e}")
            return []
    
    def papers_with_field(self, field_node_index: int) -> List[int]:
        """Get all papers tagged with a specific field of study"""
        try:
            # Get local index of the field
            field_local_idx = self.graph_loader.get_local_index(field_node_index, 'field_of_study')
            if field_local_idx is None:
                logger.warning(f"Field {field_node_index} not found")
                return []
            
            # Find edges where this field is the target (paper -> field)
            edge_key = self.edge_types['paper_has_topic_field']
            if edge_key in self.graph.edge_types:
                edge_index = self.graph[edge_key].edge_index
                
                # Find all papers (source nodes) connected to this field (target node)
                target_mask = edge_index[1] == field_local_idx
                paper_local_indices = edge_index[0][target_mask].tolist()
                
                # Convert back to node_index
                paper_node_indices = self._get_node_indices_from_local(paper_local_indices, 'paper')
                
                logger.debug(f"Found {len(paper_node_indices)} papers for field {field_node_index}")
                return paper_node_indices
            
            return []
            
        except Exception as e:
            logger.error(f"Error finding papers for field {field_node_index}: {e}")
            return []
    
    def papers_citing(self, paper_node_index: int) -> List[int]:
        """Get all papers that cite the given paper"""
        try:
            # Get local index of the cited paper
            paper_local_idx = self.graph_loader.get_local_index(paper_node_index, 'paper')
            if paper_local_idx is None:
                logger.warning(f"Paper {paper_node_index} not found")
                return []
            
            # Find edges where this paper is cited (citing_paper -> cited_paper)
            edge_key = self.edge_types['paper_cites_paper']
            if edge_key in self.graph.edge_types:
                edge_index = self.graph[edge_key].edge_index
                
                # Find all papers (source nodes) that cite this paper (target node)
                target_mask = edge_index[1] == paper_local_idx
                citing_local_indices = edge_index[0][target_mask].tolist()
                
                # Convert back to node_index
                citing_node_indices = self._get_node_indices_from_local(citing_local_indices, 'paper')
                
                logger.debug(f"Found {len(citing_node_indices)} papers citing paper {paper_node_index}")
                return citing_node_indices
            
            return []
            
        except Exception as e:
            logger.error(f"Error finding papers citing {paper_node_index}: {e}")
            return []
    
    def papers_cited_by(self, paper_node_index: int) -> List[int]:
        """Get all papers cited by the given paper"""
        try:
            # Get local index of the citing paper
            paper_local_idx = self.graph_loader.get_local_index(paper_node_index, 'paper')
            if paper_local_idx is None:
                logger.warning(f"Paper {paper_node_index} not found")
                return []
            
            # Find edges where this paper is the source (citing_paper -> cited_paper)
            edge_key = self.edge_types['paper_cites_paper']
            if edge_key in self.graph.edge_types:
                edge_index = self.graph[edge_key].edge_index
                
                # Find all papers (target nodes) cited by this paper (source node)
                source_mask = edge_index[0] == paper_local_idx
                cited_local_indices = edge_index[1][source_mask].tolist()
                
                # Convert back to node_index
                cited_node_indices = self._get_node_indices_from_local(cited_local_indices, 'paper')
                
                logger.debug(f"Found {len(cited_node_indices)} papers cited by paper {paper_node_index}")
                return cited_node_indices
            
            return []
            
        except Exception as e:
            logger.error(f"Error finding papers cited by {paper_node_index}: {e}")
            return []
    
    def papers_by_year_range(self, start_year: int, end_year: int) -> List[int]:
        """Get all papers published between start_year and end_year (inclusive)"""
        try:
            papers_in_range = []
            
            # Iterate through all paper nodes
            for node_index, attrs in self.graph_loader.node_attrs.items():
                if self.graph_loader.get_node_type(node_index) == 'paper':
                    year = attrs.get('Year')
                    if year is not None and start_year <= year <= end_year:
                        papers_in_range.append(node_index)
            
            logger.debug(f"Found {len(papers_in_range)} papers between {start_year} and {end_year}")
            return papers_in_range
            
        except Exception as e:
            logger.error(f"Error finding papers by year range {start_year}-{end_year}: {e}")
            return []
    
    def papers_by_institution(self, institution_node_index: int) -> List[int]:
        """Get all papers from authors affiliated with an institution"""
        try:
            # Get all authors affiliated with this institution
            affiliated_authors = self.authors_affiliated_with(institution_node_index)
            
            # Get all papers by these authors
            all_papers = self.papers_by_author(affiliated_authors)
            
            logger.debug(f"Found {len(all_papers)} papers from institution {institution_node_index}")
            return all_papers
            
        except Exception as e:
            logger.error(f"Error finding papers for institution {institution_node_index}: {e}")
            return []
    
    def authors_affiliated_with(self, institution_node_index: int) -> List[int]:
        """Get all authors affiliated with an institution"""
        try:
            # Get local index of the institution
            institution_local_idx = self.graph_loader.get_local_index(institution_node_index, 'institution')
            if institution_local_idx is None:
                logger.warning(f"Institution {institution_node_index} not found")
                return []
            
            # Find edges where this institution is the target (author -> institution)
            edge_key = self.edge_types['author_affiliated_with_institution']
            if edge_key in self.graph.edge_types:
                edge_index = self.graph[edge_key].edge_index
                
                # Find all authors (source nodes) connected to this institution (target node)
                target_mask = edge_index[1] == institution_local_idx
                author_local_indices = edge_index[0][target_mask].tolist()
                
                # Convert back to node_index
                author_node_indices = self._get_node_indices_from_local(author_local_indices, 'author')
                
                logger.debug(f"Found {len(author_node_indices)} authors affiliated with institution {institution_node_index}")
                return author_node_indices
            
            return []
            
        except Exception as e:
            logger.error(f"Error finding authors for institution {institution_node_index}: {e}")
            return []
    
    def get_paper_metadata(self, paper_node_index: int) -> Optional[Dict[str, Any]]:
        """Get metadata for a paper"""
        return self.graph_loader.get_node_attributes(paper_node_index)
    
    def get_author_metadata(self, author_node_index: int) -> Optional[Dict[str, Any]]:
        """Get metadata for an author"""
        return self.graph_loader.get_node_attributes(author_node_index)
    
    def get_institution_metadata(self, institution_node_index: int) -> Optional[Dict[str, Any]]:
        """Get metadata for an institution"""
        return self.graph_loader.get_node_attributes(institution_node_index)
    
    def get_field_metadata(self, field_node_index: int) -> Optional[Dict[str, Any]]:
        """Get metadata for a field of study"""
        return self.graph_loader.get_node_attributes(field_node_index)


def main():
    """Test the traversal utilities"""
    from graph_loader import MAGGraphLoader
    
    logger.info(" Testing MAG Traversal Utilities")
    
    try:
        # Load graph first
        processed_dir = "/shared/khoja/CogComp/datasets/MAG/processed"
        loader = MAGGraphLoader(processed_dir)
        graph = loader.build_graph()
        
        # Create traversal utils
        traversal = MAGTraversalUtils(graph, loader)
        
        logger.info(" Traversal utilities initialized successfully!")
        
        # Test some traversals (using example node_index from the data)
        test_paper_node_index = 274683 # From the example in the markdown
        
        # Test finding authors of a paper
        authors = traversal.authors_of_paper(test_paper_node_index)
        logger.info(f"Authors of paper {test_paper_node_index}: {len(authors)} found")
        
        if authors:
            # Test finding papers by authors
            papers = traversal.papers_by_author(authors[:2]) # Limit to first 2 authors
            logger.info(f"Papers by first 2 authors: {len(papers)} found")
        
        # Test year range query
        recent_papers = traversal.papers_by_year_range(2010, 2019)
        logger.info(f"Papers from 2010-2019: {len(recent_papers)} found")
        
        return True
        
    except Exception as e:
        logger.error(f" Failed to test traversal utilities: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
