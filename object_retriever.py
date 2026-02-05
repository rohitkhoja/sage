#!/usr/bin/env python3
"""
MAG Data Object Retriever

This script allows you to retrieve objects from the MAG data_with_citations.json file by their ID.
The dataset contains four types of objects:
- authors: Research authors with rank, name, affiliation, paper/citation counts
- papers: Research papers with title, abstract, authors, fields of study
- institutions: Research institutions with rank, name, paper/citation counts  
- fields_of_study: Research fields with rank, name, level, paper/citation counts

Usage:
    python object_retriever.py <object_id>
    
Example:
    python object_retriever.py 0
    python object_retriever.py 1104555
"""

import json
import sys
import os
from typing import Dict, Any, Optional

class MAGObjectRetriever:
    def __init__(self, data_file_path: str):
        """
        Initialize the retriever with the path to the JSON data file.
        
        Args:
            data_file_path: Path to the data_with_citations.json file
        """
        self.data_file_path = data_file_path
        self.data = None
        
    def load_data(self) -> None:
        """Load the JSON data into memory."""
        if not os.path.exists(self.data_file_path):
            raise FileNotFoundError(f"Data file not found: {self.data_file_path}")
            
        print(f"Loading data from {self.data_file_path}...")
        with open(self.data_file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"Data loaded successfully. Total objects: {len(self.data)}")
    
    def get_object(self, object_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an object by its ID.
        
        Args:
            object_id: The ID of the object to retrieve
            
        Returns:
            Dictionary containing the object data, or None if not found
        """
        if self.data is None:
            self.load_data()
            
        return self.data.get(object_id)
    
    def get_object_info(self, object_id: str) -> Optional[str]:
        """
        Get a formatted string representation of an object.
        
        Args:
            object_id: The ID of the object to retrieve
            
        Returns:
            Formatted string with object information, or None if not found
        """
        obj = self.get_object(object_id)
        if obj is None:
            return None
            
        obj_type = obj.get('type', 'unknown')
        
        if obj_type == 'author':
            return self._format_author(obj)
        elif obj_type == 'paper':
            return self._format_paper(obj)
        elif obj_type == 'institution':
            return self._format_institution(obj)
        elif obj_type == 'field_of_study':
            return self._format_field_of_study(obj)
        else:
            return f"Unknown object type: {json.dumps(obj, indent=2)}"
    
    def _format_author(self, author: Dict[str, Any]) -> str:
        """Format author object for display."""
        lines = [
            f"=== AUTHOR (ID: {author.get('Rank', 'N/A')}) ===",
            f"Name: {author.get('DisplayName', 'N/A')}",
            f"Affiliation ID: {author.get('LastKnownAffiliationId', 'N/A')}",
            f"Institution: {author.get('institution', 'N/A')}",
            f"Paper Count: {author.get('PaperCount', 'N/A')}",
            f"Citation Count: {author.get('CitationCount', 'N/A')}",
            f"Rank: {author.get('Rank', 'N/A')}"
        ]
        return "\n".join(lines)
    
    def _format_paper(self, paper: Dict[str, Any]) -> str:
        """Format paper object for display."""
        lines = [
            f"=== PAPER ===",
            f"Title: {paper.get('title', 'N/A')}",
            f"Authors: {', '.join(paper.get('authors', []))}",
            f"Fields of Study: {', '.join(paper.get('fields_of_study', []))}",
        ]
        
        # Handle abstract (might be very long)
        abstract = paper.get('abstract', '')
        if abstract:
            if len(abstract) > 300:
                lines.append(f"Abstract: {abstract[:300]}...")
            else:
                lines.append(f"Abstract: {abstract}")
        
        # Add other fields that might exist
        for key, value in paper.items():
            if key not in ['title', 'authors', 'fields_of_study', 'abstract', 'type']:
                lines.append(f"{key}: {value}")
                
        return "\n".join(lines)
    
    def _format_institution(self, institution: Dict[str, Any]) -> str:
        """Format institution object for display."""
        lines = [
            f"=== INSTITUTION (ID: {institution.get('Rank', 'N/A')}) ===",
            f"Name: {institution.get('DisplayName', 'N/A')}",
            f"Paper Count: {institution.get('PaperCount', 'N/A')}",
            f"Citation Count: {institution.get('CitationCount', 'N/A')}",
            f"Rank: {institution.get('Rank', 'N/A')}"
        ]
        return "\n".join(lines)
    
    def _format_field_of_study(self, field: Dict[str, Any]) -> str:
        """Format field of study object for display."""
        lines = [
            f"=== FIELD OF STUDY (ID: {field.get('Rank', 'N/A')}) ===",
            f"Name: {field.get('DisplayName', 'N/A')}",
            f"Level: {field.get('Level', 'N/A')}",
            f"Paper Count: {field.get('PaperCount', 'N/A')}",
            f"Citation Count: {field.get('CitationCount', 'N/A')}",
            f"Rank: {field.get('Rank', 'N/A')}"
        ]
        return "\n".join(lines)
    
    def search_by_name(self, name: str, object_type: str = None) -> list:
        """
        Search for objects by name (case-insensitive partial match).
        
        Args:
            name: Name to search for
            object_type: Optional filter by object type ('author', 'paper', 'institution', 'field_of_study')
            
        Returns:
            List of matching object IDs
        """
        if self.data is None:
            self.load_data()
            
        matches = []
        name_lower = name.lower()
        
        for obj_id, obj in self.data.items():
            if object_type and obj.get('type') != object_type:
                continue
                
            display_name = obj.get('DisplayName', '')
            if name_lower in display_name.lower():
                matches.append((obj_id, obj.get('type', 'unknown'), display_name))
        
        return matches

def main():
    """Main function to run the script from command line."""
    if len(sys.argv) != 2:
        print("Usage: python object_retriever.py <object_id>")
        print("Example: python object_retriever.py 0")
        sys.exit(1)
    
    object_id = sys.argv[1]
    data_file = "/shared/khoja/CogComp/datasets/MAG/data_with_citations.json"
    
    try:
        retriever = MAGObjectRetriever(data_file)
        result = retriever.get_object_info(object_id)
        
        if result:
            print(result)
        else:
            print(f"No object found with ID: {object_id}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
