#!/usr/bin/env python3
"""
End-to-end question solver - plans, executes, and returns results directly
"""

import requests
import json
import sys
from typing import List, Dict, Any, Set

SERVER_URL = "http://localhost:8080"
TIMEOUT = 600

def make_request(path: str, params: Dict = None) -> Any:
    """Make HTTP request to server"""
    try:
        url = f"{SERVER_URL}{path}"
        response = requests.get(url, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Request failed: {path} - {e}")
        return None

def extract_node_indices(results: List[Dict]) -> Set[int]:
    """Extract node indices from results"""
    indices = set()
    for item in results:
        if isinstance(item, dict):
            idx = item.get('node_index', item.get('node_index'))
            if idx:
                indices.add(int(idx))
        elif isinstance(item, int):
            indices.add(item)
    return indices

def solve_question(question: str):
    """Solve question end-to-end"""
    print(f"\n{'='*70}")
    print(f"QUESTION: {question}")
    print(f"{'='*70}\n")
    
    question_lower = question.lower()
    
    # Plan based on question
    if "cites" in question_lower or "cited by" in question_lower:
        # Pattern: "Find a paper that cites X"
        # Extract author name
        if "cites" in question_lower:
            author_part = question_lower.split("cites")[-1].strip()
        elif "cited by" in question_lower:
            author_part = question_lower.split("cited by")[-1].strip()
        else:
            author_part = ""
        
        # Remove common phrases
        author_part = author_part.replace("find a paper that", "").replace("find papers that", "").strip()
        author_part = author_part.replace("a paper", "").replace("papers", "").strip()
        
        print(f"📋 Plan:")
        print(f"  1. Search for author: '{author_part}'")
        print(f"  2. Get papers by this author")
        print(f"  3. Get papers that cite those papers")
        print(f"  4. Return results\n")
        
        # Step 1: Search for author
        print(f"🔍 Step 1: Searching for author '{author_part}'...")
        author_results = make_request('/search_authors_by_name', {
            'query': author_part,
            'top_k': 25
        })
        
        if not author_results:
            print("❌ Could not find author")
            return []
        
        author_list = author_results if isinstance(author_results, list) else author_results.get('results', [])
        
        if not author_list:
            print("❌ No authors found")
            return []
        
        # Smart author selection: check if top result matches exactly
        selected_authors = []
        original_name_clean = author_part.strip().lower()
        
        # Check first result for exact match
        first_result = author_list[0]
        first_name = first_result.get('metadata', {}).get('DisplayName', '') if isinstance(first_result, dict) else ''
        
        if first_name.lower() == original_name_clean:
            # Exact match found - check if it has papers with citations
            author_id = int(first_result.get('node_index', first_result.get('node_index')))
            test_papers = make_request('/get_papers_by_author', {'author_ids': str(author_id)})
            
            if test_papers and len(test_papers) > 0:
                # Check if any papers have citations
                has_citations = False
                for paper_id in test_papers[:5]:  # Check first 5 papers
                    citing = make_request('/get_papers_citing', {'paper_id': str(paper_id)})
                    if citing and len(citing) > 0:
                        has_citations = True
                        break
                
                if has_citations:
                    # Exact match has papers with citations - use only this one
                    selected_authors = [author_id]
                    print(f"✅ Exact match found: {first_name} (ID: {author_id}) - has {len(test_papers)} papers with citations")
                else:
                    # Exact match has papers but no citations - use top 2 results for broader search
                    print(f"⚠️ Exact match '{first_name}' has papers but no citations, considering top 2 authors...")
                    for i, author in enumerate(author_list[:2], 1):
                        author_id_val = int(author.get('node_index', author.get('node_index')))
                        author_name = author.get('metadata', {}).get('DisplayName', 'Unknown')
                        selected_authors.append(author_id_val)
                        print(f"✅ Author {i}: {author_name} (ID: {author_id_val})")
            else:
                # Exact match has no papers - use top 2 results
                print(f"⚠️ Exact match '{first_name}' has no papers, considering top 2 authors...")
                for i, author in enumerate(author_list[:2], 1):
                    author_id_val = int(author.get('node_index', author.get('node_index')))
                    author_name = author.get('metadata', {}).get('DisplayName', 'Unknown')
                    selected_authors.append(author_id_val)
                    print(f"✅ Author {i}: {author_name} (ID: {author_id_val})")
        else:
            # No exact match - use top 2 results
            print(f"No exact match for '{author_part}', using top 2 authors...")
            for i, author in enumerate(author_list[:2], 1):
                author_id = int(author.get('node_index', author.get('node_index')))
                author_name = author.get('metadata', {}).get('DisplayName', 'Unknown')
                selected_authors.append(author_id)
                print(f"✅ Author {i}: {author_name} (ID: {author_id})")
        
        if not selected_authors:
            print("❌ No authors selected")
            return []
        
        # Step 2: Get papers by author
        print(f"\n🔍 Step 2: Getting papers by {len(selected_authors)} author(s)...")
        all_papers = set()
        
        for author_id in selected_authors:
            paper_results = make_request('/get_papers_by_author', {
                'author_ids': str(author_id)
            })
            if paper_results:
                papers = extract_node_indices(paper_results if isinstance(paper_results, list) else [])
                all_papers.update(papers)
        
        if not all_papers:
            print("❌ No papers found by this author")
            return []
        
        print(f"✅ Found {len(all_papers)} papers by author(s)")
        
        # Step 3: Get papers that cite these papers
        print(f"\n🔍 Step 3: Finding papers that cite these papers...")
        citing_papers = set()
        
        for paper_id in list(all_papers)[:100]:  # Limit to first 100 papers
            cite_results = make_request('/get_papers_citing', {
                'paper_id': str(paper_id)
            })
            if cite_results:
                citing = extract_node_indices(cite_results if isinstance(cite_results, list) else [])
                citing_papers.update(citing)
        
        if not citing_papers:
            print("❌ No papers found that cite this author's work")
            return []
        
        print(f"✅ Found {len(citing_papers)} papers that cite {author_part}'s work\n")
        
        # Step 4: Get metadata for results
        print(f"📄 Getting metadata for top 10 papers...\n")
        results = []
        for paper_id in list(citing_papers)[:10]:
            metadata = make_request('/get_paper_metadata', {'paper_id': str(paper_id)})
            if metadata:
                results.append({
                    'paper_id': paper_id,
                    'title': metadata.get('title', 'N/A'),
                    'year': metadata.get('Year', 'N/A'),
                    'citations': metadata.get('PaperCitationCount', 0),
                    'venue': metadata.get('JournalDisplayName', metadata.get('OriginalVenue', 'N/A'))
                })
                print(f"  {len(results)}. {metadata.get('title', 'N/A')}")
                year = metadata.get('Year', 'N/A')
                if year == -1:
                    year = 'N/A'
                citations = metadata.get('PaperCitationCount', 0)
                if citations == -1:
                    citations = 0
                venue = metadata.get('JournalDisplayName', metadata.get('OriginalVenue', 'N/A'))
                print(f"     Year: {year} | Citations: {citations} | Venue: {venue}")
                print()
        
        return results
    
    else:
        print("❌ Question pattern not recognized")
        return []

if __name__ == "__main__":
    if len(sys.argv) < 2:
        question = input("Enter your question: ")
    else:
        question = " ".join(sys.argv[1:])
    
    results = solve_question(question)
    
    print(f"\n{'='*70}")
    print(f"✅ Found {len(results)} papers")
    print(f"{'='*70}\n")

