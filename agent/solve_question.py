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
        print(f"❌ Request failed: {e}")
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

def get_intersection(list1: List[int], list2: List[int]) -> List[int]:
    """Get intersection of two lists"""
    return list(set(list1) & set(list2))

def solve_question(question: str):
    """Solve a question by planning and executing in one go"""
    print(f"\n{'='*80}")
    print(f"📝 Question: {question}")
    print(f"{'='*80}\n")
    
    # Step 1: Analyze the question
    print("🔍 Analyzing question...")
    
    # The question asks for the "first paper from the ABRACADABRA project"
    # This suggests we need to:
    # 1. Search for papers related to "ABRACADABRA" (probably in title or abstract)
    # 2. Get metadata for those papers to find the "first" one (probably by date/year)
    
    # Step 2: Plan the execution
    print("\n📋 Plan:")
    print("  1. Search for papers with 'ABRACADABRA' in the title")
    print("  2. If not enough results, also search in abstract")
    print("  3. Get paper metadata for all results")
    print("  4. Sort by year/date and return the first one")
    print("\n🚀 Executing plan...\n")
    
    # Step 3: Execute the plan
    all_papers = []
    
    # Search in title
    print("Step 1: Searching for 'ABRACADABRA' in paper titles...")
    title_results = make_request("/search_papers_by_title", {"query": "ABRACADABRA"})
    if title_results:
        paper_ids = extract_node_indices(title_results)
        print(f"   ✅ Found {len(paper_ids)} papers from title search")
        all_papers.extend(paper_ids)
    else:
        print(f"   ⚠️ No results from title search")
    
    # Search in abstract
    print("\nStep 2: Searching for 'ABRACADABRA' in paper abstracts...")
    abstract_results = make_request("/search_papers_by_abstract", {"query": "ABRACADABRA"})
    if abstract_results:
        paper_ids = extract_node_indices(abstract_results)
        print(f"   ✅ Found {len(paper_ids)} papers from abstract search")
        all_papers.extend(paper_ids)
    else:
        print(f"   ⚠️ No results from abstract search")
    
    # Remove duplicates
    unique_papers = list(set(all_papers))
    print(f"\n📊 Total unique papers found: {len(unique_papers)}")
    
    if not unique_papers:
        print("\n❌ No papers found for ABRACADABRA project")
        return
    
    # Step 4: Get metadata for all papers and sort by year
    print(f"\nStep 3: Fetching metadata for {len(unique_papers)} papers...")
    papers_with_metadata = []
    
    for i, paper_id in enumerate(unique_papers):
        metadata = make_request("/get_paper_metadata", {"paper_id": paper_id})
        if metadata:
            papers_with_metadata.append({
                'node_index': paper_id,
                'metadata': metadata
            })
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(unique_papers)} papers...")
    
    print(f"   ✅ Retrieved metadata for {len(papers_with_metadata)} papers")
    
    # Step 5: Filter papers that actually contain "ABRACADABRA" in title or abstract
    print(f"\nStep 4: Filtering papers that actually mention 'ABRACADABRA'...")
    
    abracadabra_papers = []
    for paper in papers_with_metadata:
        metadata = paper.get('metadata', {})
        title = metadata.get('OriginalTitle', '').upper()
        abstract = metadata.get('Abstract', '').upper() if metadata.get('Abstract') else ''
        
        if 'ABRACADABRA' in title or 'ABRACADABRA' in abstract.upper():
            abracadabra_papers.append(paper)
    
    print(f"   ✅ Found {len(abracadabra_papers)} papers that mention 'ABRACADABRA'")
    
    if not abracadabra_papers:
        print("\n⚠️ No papers directly mention 'ABRACADABRA', showing all results...")
        abracadabra_papers = papers_with_metadata
    
    # Step 6: Sort by year and find the first paper
    print(f"\nStep 5: Sorting papers by year to find the first one...")
    
    def get_year(paper):
        metadata = paper.get('metadata', {})
        year = metadata.get('Year')
        if year:
            if isinstance(year, int):
                return year
            elif isinstance(year, str) and year.isdigit():
                return int(year)
        return 9999  # Put papers without year at the end
    
    abracadabra_papers.sort(key=get_year)
    
    # Display results
    print(f"\n{'='*80}")
    print(f"📄 Results: Found {len(abracadabra_papers)} papers from ABRACADABRA project")
    print(f"{'='*80}\n")
    
    if abracadabra_papers:
        print("🏆 First Paper (earliest year):")
        first_paper = abracadabra_papers[0]
        metadata = first_paper['metadata']
        
        print(f"\n  Paper ID: {first_paper['node_index']}")
        print(f"  Title: {metadata.get('OriginalTitle', 'N/A')}")
        print(f"  Year: {metadata.get('Year', 'N/A')}")
        print(f"  Venue: {metadata.get('Venue', 'N/A')}")
        print(f"  Citations: {metadata.get('CitationCount', 'N/A')}")
        
        # Show a few more papers if available
        if len(abracadabra_papers) > 1:
            print(f"\n📋 Other papers ({len(abracadabra_papers) - 1} more):")
            for i, paper in enumerate(abracadabra_papers[1:6], 2):  # Show next 5
                metadata = paper['metadata']
                print(f"\n  {i}. Paper ID: {paper['node_index']}")
                print(f"     Title: {metadata.get('OriginalTitle', 'N/A')[:80]}...")
                print(f"     Year: {metadata.get('Year', 'N/A')}")
    else:
        print("❌ No papers with metadata found")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    question = "Give me the first paper from the ABRACADABRA project"
    solve_question(question)
