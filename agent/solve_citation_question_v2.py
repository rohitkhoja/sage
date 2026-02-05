#!/usr/bin/env python3
"""
Execute question: This paper is cited by at least two of the same papers that cites 
"Photon counting spectral CT versus conventional CT: comparative evaluation for breast imaging application."

Reasoning:
1. Get the target paper ID
2. Get all papers that cite this target paper
3. For each citing paper, get all papers they cite
4. Take pairwise intersections of citation lists (paper1∩paper2, paper1∩paper3, etc.)
5. Union all intersection results to get final answer
"""

import requests
import json
from typing import List, Dict, Any, Set
from itertools import combinations
from datetime import datetime

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
        print(f" Request failed: {path} - {e}")
        return None

def extract_node_indices(results: List[Dict]) -> Set[int]:
    """Extract node indices from results"""
    indices = set()
    if not results:
        return indices
    
    for item in results:
        if isinstance(item, dict):
            idx = item.get('node_index')
            if idx:
                indices.add(int(idx))
        elif isinstance(item, int):
            indices.add(item)
    return indices

print(f"\n{'='*80}")
print(f" Question: This paper is cited by at least two of the same papers that cites")
print(f" 'Photon counting spectral CT versus conventional CT: comparative evaluation")
print(f" for breast imaging application.'")
print(f"{'='*80}\n")

print(" Execution Plan:")
print(" 1. Find target paper by title search")
print(" 2. Get all papers that cite this target paper")
print(" 3. For each citing paper, get all papers they cite")
print(" 4. Take pairwise intersections of citation lists (1∩2, 1∩3, 2∩3, etc.)")
print(" 5. Union all intersection results to get final answer\n")

print(" Executing...\n")

# === Step 1: Find the target paper ===
print("Step 1: Searching for target paper...")
target_title = "Photon counting spectral CT versus conventional CT: comparative evaluation for breast imaging application"

# Search in both title and abstract for better matching
print(" Searching in titles...")
title_results = make_request("/search_papers_by_title", {"query": target_title})
print(" Searching in abstracts...")
abstract_results = make_request("/search_papers_by_abstract", {"query": target_title})

# Combine results
all_candidates = []
if title_results:
    all_candidates.extend(title_results)
if abstract_results:
    all_candidates.extend(abstract_results)

# Find the best matching paper
target_paper_id = None
target_metadata = None

print(f" Checking {len(all_candidates)} candidate papers...")
for candidate in all_candidates:
    paper_id = candidate.get('node_index') if isinstance(candidate, dict) else candidate
    if paper_id:
        metadata = make_request("/get_paper_metadata", {"paper_id": str(paper_id)})
        if metadata:
            title = metadata.get('OriginalTitle', '').lower()
            keywords = ['photon counting', 'spectral ct', 'conventional ct', 'breast imaging']
            matches = sum(1 for keyword in keywords if keyword in title)
            
            if matches >= 3: # At least 3 keywords match
                target_paper_id = int(paper_id)
                target_metadata = metadata
                print(f" Found matching paper: ID {target_paper_id}")
                print(f" Title: {metadata.get('OriginalTitle', 'N/A')[:80]}...")
                print(f" Year: {metadata.get('Year', 'N/A')}")
                break

# If still not found, use the first result
if not target_paper_id and all_candidates:
    candidate = all_candidates[0]
    target_paper_id = candidate.get('node_index') if isinstance(candidate, dict) else candidate
    if target_paper_id:
        target_paper_id = int(target_paper_id)
        target_metadata = make_request("/get_paper_metadata", {"paper_id": str(target_paper_id)})
        print(f" Using first result: ID {target_paper_id}")
        if target_metadata:
            print(f" Title: {target_metadata.get('OriginalTitle', 'N/A')[:80]}...")
            print(f" Year: {target_metadata.get('Year', 'N/A')}")

if not target_paper_id:
    print(" Could not find target paper")
    exit(1)

# === Step 2: Get papers that cite the target paper ===
print(f"\nStep 2: Finding papers that cite the target paper (ID: {target_paper_id})...")
citing_papers = make_request("/get_papers_citing", {"paper_id": str(target_paper_id)})

if not citing_papers:
    print(" No papers found that cite the target paper")
    exit(1)

citing_paper_ids = list(citing_papers) if isinstance(citing_papers, list) else []
print(f" Found {len(citing_paper_ids)} papers that cite the target paper")

if len(citing_paper_ids) == 0:
    print(" No citing papers found")
    exit(1)

# === Step 3: For each citing paper, get papers it cites ===
print(f"\nStep 3: Getting papers cited by each of the {len(citing_paper_ids)} citing papers...")
print(" (This may take a while...)")

# Dictionary: {citing_paper_id: set of papers it cites}
citing_paper_citations = {}

processed = 0
for citing_paper_id in citing_paper_ids:
    processed += 1
    if processed % 10 == 0:
        print(f" Processed {processed}/{len(citing_paper_ids)} citing papers...")
    
    # Get papers cited by this citing paper
    cited_papers = make_request("/get_papers_cited_by", {"paper_id": str(citing_paper_id)})
    
    if cited_papers and isinstance(cited_papers, list):
        citing_paper_citations[citing_paper_id] = set(cited_papers)
    else:
        citing_paper_citations[citing_paper_id] = set()

print(f" Processed all {len(citing_paper_ids)} citing papers")

# === Step 4: Take pairwise intersections ===
print(f"\nStep 4: Taking pairwise intersections of citation lists...")
print(f" Number of citing papers: {len(citing_paper_ids)}")
print(f" Number of pairs to check: {len(list(combinations(range(len(citing_paper_ids)), 2)))}")

# Store all intersection results
all_intersections = set()

# Get all pairs of citing papers
citing_ids_list = list(citing_paper_citations.keys())

pair_count = 0
for i, citing_id1 in enumerate(citing_ids_list):
    for citing_id2 in citing_ids_list[i+1:]: # Avoid duplicates: (A,B) but not (B,A)
        pair_count += 1
        
        # Get citation sets for both papers
        citations1 = citing_paper_citations[citing_id1]
        citations2 = citing_paper_citations[citing_id2]
        
        # Take intersection
        intersection = citations1 & citations2
        
        if len(intersection) > 0:
            # Exclude the target paper itself from intersections
            intersection = intersection - {target_paper_id}
            if len(intersection) > 0:
                all_intersections.update(intersection)
        
        if pair_count % 50 == 0:
            print(f" Processed {pair_count} pairs, found {len(all_intersections)} unique papers so far...")

print(f" Completed all pairwise intersections")
print(f" Total pairs processed: {pair_count}")
print(f" Unique papers found in at least one intersection: {len(all_intersections)}")

# === Step 5: Final results ===
final_paper_ids = list(all_intersections)

if not final_paper_ids:
    print(f"\n No papers found that are cited by at least two of the citing papers")
    exit(1)

print(f"\n{'='*80}")
print(f" Final Results: {len(final_paper_ids)} papers")
print(f" (Papers cited by at least two of the same papers that cite the target)")
print(f"{'='*80}\n")

# Get metadata for final papers
final_papers = []
for i, paper_id in enumerate(final_paper_ids[:50], 1): # Limit to top 50 for display
    metadata = make_request("/get_paper_metadata", {"paper_id": str(paper_id)})
    if metadata:
        # Count how many citing papers cite this paper
        citation_count = sum(1 for citing_id, citations in citing_paper_citations.items() 
                            if paper_id in citations)
        
        paper_data = {
            'node_index': paper_id,
            'title': metadata.get('OriginalTitle', 'N/A'),
            'year': metadata.get('Year', 'N/A'),
            'venue': metadata.get('JournalDisplayName', metadata.get('OriginalVenue', 'N/A')),
            'cited_by_count': citation_count
        }
        final_papers.append(paper_data)
        
        print(f" {i}. Paper ID: {paper_id}")
        print(f" Title: {metadata.get('OriginalTitle', 'N/A')}")
        print(f" Year: {metadata.get('Year', 'N/A')}")
        print(f" Venue: {metadata.get('JournalDisplayName', metadata.get('OriginalVenue', 'N/A'))}")
        print(f" Cited by {citation_count} of the {len(citing_paper_ids)} papers that cite the target")
        print()

if len(final_paper_ids) > 50:
    print(f" ... and {len(final_paper_ids) - 50} more papers\n")

# === Save Results ===
output = {
    'question': 'This paper is cited by at least two of the same papers that cites "Photon counting spectral CT versus conventional CT: comparative evaluation for breast imaging application."',
    'target_paper': {
        'paper_id': target_paper_id,
        'title': target_metadata.get('OriginalTitle', 'N/A') if target_metadata else 'N/A'
    },
    'citing_papers_count': len(citing_paper_ids),
    'pairwise_intersections': pair_count,
    'final_nodes': final_paper_ids,
    'result_count': len(final_paper_ids),
    'papers_with_metadata': final_papers[:50] # First 50 for reference
}

output_file = f"/shared/khoja/CogComp/agent/output/qa/question_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"{'='*80}")
print(f" Execution complete!")
print(f" - Target paper: {target_paper_id}")
print(f" - Citing papers: {len(citing_paper_ids)}")
print(f" - Pairwise intersections: {pair_count}")
print(f" - Final papers (union of all intersections): {len(final_paper_ids)}")
print(f" Results saved to: {output_file}")
print(f"{'='*80}\n")

