#!/usr/bin/env python3
"""
Solve a question using the MAG Agent system
Plan -> Execute -> Return Results
"""

import requests
import json
from typing import List, Dict, Any, Set

def make_request(path: str, params: Dict = None) -> Any:
    """Make HTTP request to the running server"""
    try:
        url = f"http://localhost:8080{path}"
        response = requests.get(url, params=params, timeout=600)
        return response.json()
    except Exception as e:
        print(f" Request failed: {e}")
        return None

def extract_node_indices(results: List[Dict]) -> List[int]:
    """Extract node indices from search results"""
    node_indices = []
    for item in results:
        if isinstance(item, dict):
            node_idx = item.get('node_index')
            if node_idx is not None:
                node_indices.append(int(node_idx))
    return node_indices

def solve_question(question: str):
    """Solve a question by planning and executing in one go"""
    print(f"\n{'='*80}")
    print(f" Question: {question}")
    print(f"{'='*80}\n")
    
    # Step 1: Analyze the question
    print(" Analyzing question...")
    
    # This question has multiple constraints:
    # 1. Papers published in 2010 (year filter)
    # 2. Discuss "anisotropic relativistic charged fluid" (topic search)
    # 3. In the field of "spacetime symmetries and electromagnetic fields" (field search)
    
    # Step 2: Plan the execution
    print("\n Plan:")
    print(" 1. Search for papers with 'anisotropic relativistic charged fluid' in title/abstract")
    print(" 2. Search for field 'spacetime symmetries' and 'electromagnetic fields'")
    print(" 3. Get papers from year 2010")
    print(" 4. Find intersection of all three sets")
    print(" 5. If intersection is zero, return last non-zero intersection result")
    print("\n Executing plan...\n")
    
    # Step 3: Execute the plan
    
    # Step 3a: Search for topic in title
    print("Step 1: Searching for 'anisotropic relativistic charged fluid' in paper titles...")
    topic_query = "anisotropic relativistic charged fluid"
    title_results = make_request("/search_papers_by_title", {"query": topic_query})
    topic_papers = extract_node_indices(title_results) if title_results else []
    print(f" Found {len(topic_papers)} papers from title search")
    
    # Step 3b: Search for topic in abstract
    print("\nStep 2: Searching for 'anisotropic relativistic charged fluid' in paper abstracts...")
    abstract_results = make_request("/search_papers_by_abstract", {"query": topic_query})
    abstract_papers = extract_node_indices(abstract_results) if abstract_results else []
    print(f" Found {len(abstract_papers)} papers from abstract search")
    
    # Combine topic papers
    all_topic_papers = list(set(topic_papers + abstract_papers))
    print(f" Total unique topic papers: {len(all_topic_papers)}")
    
    # Step 3c: Search for fields
    print("\nStep 3: Searching for field 'spacetime symmetries'...")
    field1_results = make_request("/search_fields_by_name", {"query": "spacetime symmetries"})
    field1_ids = extract_node_indices(field1_results) if field1_results else []
    print(f" Found {len(field1_ids)} fields")
    
    field1_papers = []
    if field1_ids:
        for field_id in field1_ids:
            papers = make_request("/get_papers_with_field", {"field_id": field_id})
            if papers:
                field1_papers.extend(papers if isinstance(papers, list) else [])
        field1_papers = list(set(field1_papers))
        print(f" Papers in spacetime symmetries field: {len(field1_papers)}")
    
    # Search for electromagnetic fields
    print("\nStep 4: Searching for field 'electromagnetic fields'...")
    field2_results = make_request("/search_fields_by_name", {"query": "electromagnetic fields"})
    field2_ids = extract_node_indices(field2_results) if field2_results else []
    print(f" Found {len(field2_ids)} fields")
    
    field2_papers = []
    if field2_ids:
        for field_id in field2_ids:
            papers = make_request("/get_papers_with_field", {"field_id": field_id})
            if papers:
                field2_papers.extend(papers if isinstance(papers, list) else [])
        field2_papers = list(set(field2_papers))
        print(f" Papers in electromagnetic fields field: {len(field2_papers)}")
    
    # Combine field papers
    all_field_papers = list(set(field1_papers + field2_papers))
    print(f" Total unique field papers: {len(all_field_papers)}")
    
    # Step 3d: Get papers from 2010
    print("\nStep 5: Getting papers from year 2010...")
    year_papers = make_request("/get_papers_by_year_range", {"start_year": 2010, "end_year": 2010})
    year_papers = year_papers if isinstance(year_papers, list) else []
    print(f" Found {len(year_papers)} papers from 2010")
    
    # Step 4: Find intersections
    print(f"\n{'='*60}")
    print(" Intersection Analysis:")
    print(f"{'='*60}\n")
    
    intersections = []
    
    # Intersection 1: Topic + Year
    print("Intersection 1: Topic papers ∩ Year 2010")
    intersection1 = list(set(all_topic_papers) & set(year_papers))
    print(f" Result: {len(intersection1)} papers")
    intersections.append(("Topic + Year", intersection1))
    
    # Intersection 2: Topic + Fields
    print("\nIntersection 2: Topic papers ∩ Field papers")
    intersection2 = list(set(all_topic_papers) & set(all_field_papers))
    print(f" Result: {len(intersection2)} papers")
    intersections.append(("Topic + Fields", intersection2))
    
    # Intersection 3: All three (Topic + Fields + Year)
    print("\nIntersection 3: Topic papers ∩ Field papers ∩ Year 2010")
    intersection3 = list(set(all_topic_papers) & set(all_field_papers) & set(year_papers))
    print(f" Result: {len(intersection3)} papers")
    intersections.append(("Topic + Fields + Year", intersection3))
    
    # Step 5: Find best result (non-zero intersection)
    print(f"\n{'='*60}")
    print(" Final Results:")
    print(f"{'='*60}\n")
    
    # Find last non-zero intersection
    best_intersection = None
    best_name = None
    for name, inter in reversed(intersections):
        if len(inter) > 0:
            best_intersection = inter
            best_name = name
            break
    
    if best_intersection:
        print(f" Found {len(best_intersection)} papers ({best_name})")
        
        # Get metadata for papers and filter for relevance
        print(f"\n Paper Details:")
        papers_with_metadata = []
        
        for paper_id in best_intersection[:10]: # Limit to first 10
            metadata = make_request("/get_paper_metadata", {"paper_id": paper_id})
            if metadata:
                papers_with_metadata.append({
                    'node_index': paper_id,
                    'metadata': metadata
                })
        
        # Filter papers that mention relevant field terms
        relevant_keywords = ['spacetime', 'symmetries', 'electromagnetic', 'field']
        filtered_papers = []
        
        for paper in papers_with_metadata:
            metadata = paper['metadata']
            title = (metadata.get('OriginalTitle', '') or '').lower()
            abstract = (metadata.get('Abstract', '') or '').lower()
            text = title + ' ' + abstract
            
            # Check if paper mentions field-related keywords
            keyword_matches = sum(1 for keyword in relevant_keywords if keyword in text)
            if keyword_matches >= 2: # At least 2 keywords should match
                filtered_papers.append(paper)
        
        if filtered_papers:
            print(f" Found {len(filtered_papers)} papers that match field keywords")
        
        # Display papers
        display_papers = filtered_papers if filtered_papers else papers_with_metadata
        
        for i, paper in enumerate(display_papers, 1):
            metadata = paper['metadata']
            print(f"\n {i}. Paper ID: {paper['node_index']}")
            print(f" Title: {metadata.get('OriginalTitle', 'N/A')}")
            print(f" Year: {metadata.get('Year', 'N/A')}")
            print(f" Venue: {metadata.get('Venue', 'N/A')}")
            print(f" Citations: {metadata.get('CitationCount', 'N/A')}")
    else:
        print(" No papers found matching all criteria")
        print("\n Showing all intermediate results:")
        for name, inter in intersections:
            if len(inter) > 0:
                print(f"\n{name}: {len(inter)} papers")
                # Show a few paper IDs
                for paper_id in list(inter)[:5]:
                    print(f" - Paper ID: {paper_id}")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    question = "I am looking for papers published in 2010 that discuss anisotropic relativistic charged fluid in the filed of spacetime symmetries and electromagnetic fields"
    solve_question(question)

