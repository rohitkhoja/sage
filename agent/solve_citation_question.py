#!/usr/bin/env python3
"""
Execute question: This paper is cited by at least two of the same papers that cites 
"Photon counting spectral CT versus conventional CT: comparative evaluation for breast imaging application."

Plan: 
1. Find the target paper by title search
2. Get all papers that cite this target paper
3. For each citing paper, get all papers it cites (papers_cited_by)
4. Count which papers appear in citations from at least 2 different citing papers
5. Return those papers that appear at least twice
"""

import requests
import json
from typing import List, Dict, Any, Set
from collections import defaultdict
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
        print(f"❌ Request failed: {path} - {e}")
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
print(f"📝 Question: This paper is cited by at least two of the same papers that cites")
print(f"   'Photon counting spectral CT versus conventional CT: comparative evaluation")
print(f"   for breast imaging application.'")
print(f"{'='*80}\n")

print("📋 Execution Plan:")
print("  1. Search for target paper: 'Photon counting spectral CT versus conventional CT...'")
print("  2. Get all papers that cite this target paper")
print("  3. For each citing paper, get all papers it cites (cited_by)")
print("  4. Count how many times each paper is cited by different citing papers")
print("  5. Return papers cited by at least 2 different citing papers\n")

print("🚀 Executing...\n")

# === Step 1: Find the target paper ===
print("Step 1: Searching for target paper...")
target_title = "Photon counting spectral CT versus conventional CT: comparative evaluation for breast imaging application"

# Search in both title and abstract for better matching
print("   Searching in titles...")
title_results = make_request("/search_papers_by_title", {"query": target_title})
print("   Searching in abstracts...")
abstract_results = make_request("/search_papers_by_abstract", {"query": target_title})

# Combine results
all_candidates = []
if title_results:
    all_candidates.extend(title_results)
if abstract_results:
    all_candidates.extend(abstract_results)

# Find the best matching paper by checking if the exact title appears
target_paper_id = None
target_metadata = None

print(f"   Checking {len(all_candidates)} candidate papers...")
for candidate in all_candidates:
    paper_id = candidate.get('node_index') if isinstance(candidate, dict) else candidate
    if paper_id:
        metadata = make_request("/get_paper_metadata", {"paper_id": str(paper_id)})
        if metadata:
            title = metadata.get('OriginalTitle', '').lower()
            # Check if target title keywords appear
            target_lower = target_title.lower()
            keywords = ['photon counting', 'spectral ct', 'conventional ct', 'breast imaging']
            matches = sum(1 for keyword in keywords if keyword in title)
            
            if matches >= 3:  # At least 3 keywords match
                target_paper_id = int(paper_id)
                target_metadata = metadata
                print(f"   ✅ Found matching paper: ID {target_paper_id}")
                print(f"   Title: {metadata.get('OriginalTitle', 'N/A')[:80]}...")
                print(f"   Year: {metadata.get('Year', 'N/A')}")
                break

# If still not found, use the first result
if not target_paper_id and all_candidates:
    candidate = all_candidates[0]
    target_paper_id = candidate.get('node_index') if isinstance(candidate, dict) else candidate
    if target_paper_id:
        target_paper_id = int(target_paper_id)
        target_metadata = make_request("/get_paper_metadata", {"paper_id": str(target_paper_id)})
        print(f"   ⚠️ Using first result: ID {target_paper_id}")
        if target_metadata:
            print(f"   Title: {target_metadata.get('OriginalTitle', 'N/A')[:80]}...")
            print(f"   Year: {target_metadata.get('Year', 'N/A')}")

if not target_paper_id:
    print("❌ Could not find target paper")
    exit(1)

# === Step 2: Get papers that cite the target paper ===
print(f"\nStep 2: Finding papers that cite the target paper (ID: {target_paper_id})...")
citing_papers = make_request("/get_papers_citing", {"paper_id": str(target_paper_id)})

if not citing_papers:
    print("❌ No papers found that cite the target paper")
    exit(1)

citing_paper_ids = list(citing_papers) if isinstance(citing_papers, list) else []
print(f"   ✅ Found {len(citing_paper_ids)} papers that cite the target paper")

if len(citing_paper_ids) == 0:
    print("❌ No citing papers found")
    exit(1)

# === Step 3: For each citing paper, get papers it cites ===
print(f"\nStep 3: Getting papers cited by each of the {len(citing_paper_ids)} citing papers...")
print("   (This may take a while...)")

# Dictionary to track which citing papers cite which papers
# Structure: {cited_paper_id: [list of citing_paper_ids]}
cited_by_count = defaultdict(set)

processed = 0
for citing_paper_id in citing_paper_ids:
    processed += 1
    if processed % 10 == 0:
        print(f"   Processed {processed}/{len(citing_paper_ids)} citing papers...")
    
    # Get papers cited by this citing paper
    cited_papers = make_request("/get_papers_cited_by", {"paper_id": str(citing_paper_id)})
    
    if cited_papers and isinstance(cited_papers, list):
        for cited_paper_id in cited_papers:
            cited_by_count[cited_paper_id].add(citing_paper_id)

print(f"   ✅ Processed all {len(citing_paper_ids)} citing papers")

# === Step 4: Find papers cited by at least 2 different citing papers ===
print(f"\nStep 4: Finding papers cited by at least 2 different citing papers...")
print(f"   (Excluding the target paper itself)")

papers_cited_at_least_twice = []
for cited_paper_id, citing_paper_set in cited_by_count.items():
    # Exclude the target paper itself
    if cited_paper_id == target_paper_id:
        continue
    
    if len(citing_paper_set) >= 2:
        papers_cited_at_least_twice.append({
            'paper_id': cited_paper_id,
            'cited_by_count': len(citing_paper_set),
            'citing_papers': list(citing_paper_set)[:5]  # Store first 5 for reference
        })

# Sort by citation count (descending)
papers_cited_at_least_twice.sort(key=lambda x: x['cited_by_count'], reverse=True)

print(f"   ✅ Found {len(papers_cited_at_least_twice)} papers cited by at least 2 different citing papers")

# === Step 5: Get metadata and display results ===
print(f"\n{'='*80}")
print(f"📄 Final Results: {len(papers_cited_at_least_twice)} papers")
print(f"{'='*80}\n")

final_papers = []
for i, paper_info in enumerate(papers_cited_at_least_twice[:20], 1):  # Limit to top 20 for display
    paper_id = paper_info['paper_id']
    count = paper_info['cited_by_count']
    
    metadata = make_request("/get_paper_metadata", {"paper_id": str(paper_id)})
    if metadata:
        paper_data = {
            'node_index': paper_id,
            'title': metadata.get('OriginalTitle', 'N/A'),
            'year': metadata.get('Year', 'N/A'),
            'venue': metadata.get('JournalDisplayName', metadata.get('OriginalVenue', 'N/A')),
            'cited_by_count': count,
            'citing_papers': paper_info['citing_papers']
        }
        final_papers.append(paper_data)
        
        print(f"  {i}. Paper ID: {paper_id}")
        print(f"     Title: {metadata.get('OriginalTitle', 'N/A')}")
        print(f"     Year: {metadata.get('Year', 'N/A')}")
        print(f"     Venue: {metadata.get('JournalDisplayName', metadata.get('OriginalVenue', 'N/A'))}")
        print(f"     Cited by {count} different papers that also cite the target paper")
        print()

if len(papers_cited_at_least_twice) > 20:
    print(f"  ... and {len(papers_cited_at_least_twice) - 20} more papers\n")

# === Save Results ===
output = {
    'question': 'This paper is cited by at least two of the same papers that cites "Photon counting spectral CT versus conventional CT: comparative evaluation for breast imaging application."',
    'target_paper': {
        'paper_id': target_paper_id,
        'title': target_metadata.get('OriginalTitle', 'N/A') if target_metadata else 'N/A'
    },
    'citing_papers_count': len(citing_paper_ids),
    'final_nodes': [p['paper_id'] for p in papers_cited_at_least_twice],
    'result_count': len(papers_cited_at_least_twice),
    'papers_with_metadata': final_papers
}

output_file = f"/shared/khoja/CogComp/agent/output/qa/question_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"{'='*80}")
print(f"✅ Execution complete!")
print(f"   - Target paper: {target_paper_id}")
print(f"   - Citing papers: {len(citing_paper_ids)}")
print(f"   - Final papers (cited by ≥2): {len(papers_cited_at_least_twice)}")
print(f"📁 Results saved to: {output_file}")
print(f"{'='*80}\n")

