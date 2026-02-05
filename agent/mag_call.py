#!/usr/bin/env python3
"""
MAG Agent Terminal Wrapper
Converts terminal calls to HTTP calls to the running server
Usage: python mag_call.py function_name "value"
"""

import sys
import requests
import json
import os
from datetime import datetime
from pathlib import Path

# File-based storage for intermediate results
STORAGE_FILE = "/shared/khoja/CogComp/agent/output/qa/intermediate_results.json"

def make_request(path, params=None):
    """Make HTTP request to the running server"""
    try:
        url = f"http://localhost:8080{path}"
        response = requests.get(url, params=params, timeout=600)
        return response.json()
    except Exception as e:
        return {"error": f"Request failed: {e}"}

def load_storage():
    """Load intermediate results from file"""
    if os.path.exists(STORAGE_FILE):
        with open(STORAGE_FILE, 'r') as f:
            return json.load(f)
    return {"intermediate_results": {}, "current_question": None, "step_counter": 0}

def save_storage(data):
    """Save intermediate results to file"""
    os.makedirs(os.path.dirname(STORAGE_FILE), exist_ok=True)
    with open(STORAGE_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def save_results_to_file(step_key, results):
    """Save results to a JSON file for file-based input"""
    output_dir = Path("/shared/khoja/CogComp/agent/output/qa/files")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = output_dir / f"{step_key}.json"
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return str(file_path)

def store_step_result(step_name, result, query_params=None):
    """Store intermediate result for a step"""
    data = load_storage()
    
    if data["current_question"] is None:
        data["current_question"] = f"question_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    data["step_counter"] += 1
    step_key = f"step{data['step_counter']}_{step_name}"
    
    data["intermediate_results"][step_key] = {
        "step_number": data["step_counter"],
        "step_name": step_name,
        "query_params": query_params,
        "result": result,
        "timestamp": datetime.now().isoformat(),
        "result_count": len(result) if isinstance(result, list) else 1
    }
    
    # Auto-save to file if HNSW search or graph traversal function returns >5 results
    if (isinstance(result, list) and len(result) > 5 and 
        step_name in ['search_papers_by_title', 'search_authors_by_name', 'search_institutions_by_name', 
                     'search_fields_by_name', 'search_papers_by_abstract', 'get_authors_affiliated_with',
                     'get_papers_cited_by', 'get_papers_citing', 'get_papers_by_author', 'get_authors_of_paper']):
        file_path = save_results_to_file(step_key, result)
        data["intermediate_results"][step_key]["file_path"] = file_path
        print(f"📁 Auto-saved {len(result)} results to file: {file_path}")
    
    save_storage(data)
    print(f"📝 Stored {step_key}: {data['intermediate_results'][step_key]['result_count']} results")
    return result

def get_intersection(step1_key, step2_key):
    """Get intersection between two steps"""
    data = load_storage()
    intermediate_results = data["intermediate_results"]
    
    if step1_key not in intermediate_results or step2_key not in intermediate_results:
        print(f"❌ Steps {step1_key} or {step2_key} not found")
        print(f"Available steps: {list(intermediate_results.keys())}")
        return []
    
    step1_results = intermediate_results[step1_key]["result"]
    step2_results = intermediate_results[step2_key]["result"]
    
    # Extract node indices for intersection
    step1_indices = set()
    step2_indices = set()
    
    if isinstance(step1_results, list):
        for item in step1_results:
            if isinstance(item, dict) and 'node_index' in item:
                step1_indices.add(item['node_index'])
            elif isinstance(item, (int, str)):
                step1_indices.add(str(item))
    
    if isinstance(step2_results, list):
        for item in step2_results:
            if isinstance(item, dict) and 'node_index' in item:
                step2_indices.add(item['node_index'])
            elif isinstance(item, (int, str)):
                step2_indices.add(str(item))
    
    intersection = list(step1_indices.intersection(step2_indices))
    print(f"🔍 Intersection of {step1_key} and {step2_key}: {len(intersection)} items")
    return intersection

def save_results_to_md(question_text=None):
    """Save all intermediate results to markdown file"""
    data = load_storage()
    intermediate_results = data["intermediate_results"]
    current_question = data["current_question"]
    
    if not intermediate_results:
        print("❌ No results to save")
        return
    
    # Create output directory
    output_dir = Path("/shared/khoja/CogComp/agent/output/qa")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate markdown content
    md_content = f"""# Query Results: {current_question}

**Question:** {question_text or "Multi-step query"}
**Generated:** {datetime.now().isoformat()}

## Steps Executed

"""
    
    for step_key, step_data in intermediate_results.items():
        md_content += f"""### {step_key.upper()}
- **Step Name:** {step_data['step_name']}
- **Query Params:** {step_data['query_params']}
- **Result Count:** {step_data['result_count']}
- **Timestamp:** {step_data['timestamp']}

**Results:**
```json
{json.dumps(step_data['result'], indent=2)[:1000]}{'...' if len(json.dumps(step_data['result'])) > 1000 else ''}
```

---

"""
    
    # Save to file
    md_file = output_dir / f"{current_question}.md"
    with open(md_file, 'w') as f:
        f.write(md_content)
    
    print(f"💾 Results saved to: {md_file}")
    return str(md_file)

def reset_session():
    """Reset the session for a new question"""
    data = {"intermediate_results": {}, "current_question": None, "step_counter": 0}
    save_storage(data)
    print("🔄 Session reset for new question")

def get_step_file(step_key):
    """Get the file path for a step if it exists"""
    data = load_storage()
    intermediate_results = data["intermediate_results"]
    
    if step_key not in intermediate_results:
        print(f"❌ Step {step_key} not found")
        print(f"Available steps: {list(intermediate_results.keys())}")
        return None
    
    step_data = intermediate_results[step_key]
    if "file_path" in step_data:
        print(f"📁 File path for {step_key}: {step_data['file_path']}")
        return step_data["file_path"]
    else:
        print(f"❌ No file saved for {step_key}")
        return None

def search_papers_by_title(query, top_k=10):
    """Search papers by title"""
    result = make_request("/search_papers_by_title", {"query": query, "top_k": top_k})
    return store_step_result("search_papers_by_title", result, {"query": query, "top_k": top_k})

def search_authors_by_name(query, top_k=10):
    """Search authors by name"""
    result = make_request("/search_authors_by_name", {"query": query, "top_k": top_k})
    return store_step_result("search_authors_by_name", result, {"query": query, "top_k": top_k})

def search_institutions_by_name(query, top_k=10):
    """Search institutions by name"""
    result = make_request("/search_institutions_by_name", {"query": query, "top_k": top_k})
    return store_step_result("search_institutions_by_name", result, {"query": query, "top_k": top_k})

def search_fields_by_name(query, top_k=10):
    """Search fields of study by name"""
    result = make_request("/search_fields_by_name", {"query": query, "top_k": top_k})
    return store_step_result("search_fields_by_name", result, {"query": query, "top_k": top_k})

def search_papers_by_abstract(query, top_k=10):
    """Search papers by abstract similarity"""
    result = make_request("/search_papers_by_abstract", {"query": query, "top_k": top_k})
    return store_step_result("search_papers_by_abstract", result, {"query": query, "top_k": top_k})

def get_papers_by_year_range(start_year, end_year):
    """Get papers by year range"""
    result = make_request("/get_papers_by_year_range", {"start_year": start_year, "end_year": end_year})
    return store_step_result("get_papers_by_year_range", result, {"start_year": start_year, "end_year": end_year})

def query_natural_language(query):
    """Natural language query"""
    result = make_request("/query_natural_language", {"query": query})
    return store_step_result("query_natural_language", result, {"query": query})

def get_authors_of_paper(paper_id):
    """Get authors of a paper"""
    result = make_request("/get_authors_of_paper", {"paper_id": paper_id})
    return store_step_result("get_authors_of_paper", result, {"paper_id": paper_id})

def get_authors_affiliated_with(institution_id):
    """Get authors affiliated with an institution"""
    result = make_request("/get_authors_affiliated_with", {"institution_id": institution_id})
    # Extract the results list from the server response
    if isinstance(result, dict) and 'results' in result:
        authors_list = result['results']
    else:
        authors_list = result
    return store_step_result("get_authors_affiliated_with", authors_list, {"institution_id": institution_id})

def get_papers_by_author(author_ids):
    """Get papers by author IDs"""
    result = make_request("/get_papers_by_author", {"author_ids": author_ids})
    return store_step_result("get_papers_by_author", result, {"author_ids": author_ids})

def get_papers_citing(paper_id):
    """Get papers citing a paper"""
    result = make_request("/get_papers_citing", {"paper_id": paper_id})
    return store_step_result("get_papers_citing", result, {"paper_id": paper_id})

def get_papers_cited_by(paper_id):
    """Get papers cited by a paper"""
    result = make_request("/get_papers_cited_by", {"paper_id": paper_id})
    return store_step_result("get_papers_cited_by", result, {"paper_id": paper_id})

def get_paper_metadata(paper_id):
    """Get paper metadata"""
    result = make_request("/get_paper_metadata", {"paper_id": paper_id})
    return store_step_result("get_paper_metadata", result, {"paper_id": paper_id})

def get_author_metadata(author_id):
    """Get author metadata"""
    result = make_request("/get_author_metadata", {"author_id": author_id})
    return store_step_result("get_author_metadata", result, {"author_id": author_id})

def get_status():
    """Get server status"""
    return make_request("/status")

def help_functions():
    """Show available functions"""
    functions = [
        "search_papers_by_title 'query' [top_k]",
        "search_authors_by_name 'query' [top_k]",
        "search_institutions_by_name 'query' [top_k]",
        "search_fields_by_name 'query' [top_k]",
        "search_papers_by_abstract 'query' [top_k]",
        "get_papers_by_year_range start_year end_year",
        "query_natural_language 'query'",
        "get_authors_of_paper paper_id",
        "get_authors_affiliated_with institution_id",
        "get_papers_by_author 'id1,id2,id3'",
        "get_papers_citing paper_id",
        "get_papers_cited_by paper_id",
        "get_paper_metadata paper_id",
        "get_author_metadata author_id",
        "get_intersection step1_key step2_key",
        "get_step_file step_key",
        "save_results_to_md [question_text]",
        "reset_session",
        "status"
    ]
    
    print("🎯 Available Functions:")
    print("=" * 40)
    for func in functions:
        print(f"  python mag_call.py {func}")
    print("\n💡 Examples:")
    print("  python mag_call.py search_papers_by_title 'machine learning' 5")
    print("  python mag_call.py get_papers_by_year_range 2010 2020")
    print("  python mag_call.py get_step_file step1_search_papers_by_title")
    print("  python mag_call.py status")
    
    print("\n📁 File Input Support:")
    print("  - HNSW searches with >5 results are auto-saved to files")
    print("  - Use get_step_file to get file path for a step")
    print("  - Pass file path to graph traversal functions for multiple IDs")
    print("  - Example: get_authors_of_paper '/path/to/step1_search_papers_by_title.json'")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        help_functions()
        return 0
    
    function_name = sys.argv[1]
    args = sys.argv[2:] if len(sys.argv) > 2 else []
    
    # Map function names to functions
    functions = {
        'search_papers_by_title': search_papers_by_title,
        'search_authors_by_name': search_authors_by_name,
        'search_institutions_by_name': search_institutions_by_name,
        'search_fields_by_name': search_fields_by_name,
        'search_papers_by_abstract': search_papers_by_abstract,
        'get_papers_by_year_range': get_papers_by_year_range,
        'query_natural_language': query_natural_language,
        'get_authors_of_paper': get_authors_of_paper,
        'get_authors_affiliated_with': get_authors_affiliated_with,
        'get_papers_by_author': get_papers_by_author,
        'get_papers_citing': get_papers_citing,
        'get_papers_cited_by': get_papers_cited_by,
        'get_paper_metadata': get_paper_metadata,
        'get_author_metadata': get_author_metadata,
        'get_intersection': get_intersection,
        'get_step_file': get_step_file,
        'save_results_to_md': save_results_to_md,
        'reset_session': reset_session,
        'status': get_status,
        'help': help_functions
    }
    
    if function_name not in functions:
        print(f"❌ Unknown function: {function_name}")
        help_functions()
        return 1
    
    try:
        # Call the function
        result = functions[function_name](*args)
        
        if isinstance(result, dict) and 'error' in result:
            print(f"❌ {result['error']}")
            return 1
        
        # Print result
        if isinstance(result, list):
            print(f"✅ Found {len(result)} results")
            if len(result) <= 5:
                for i, item in enumerate(result):
                    print(f"  {i+1}. {item}")
            else:
                for i, item in enumerate(result[:3]):
                    print(f"  {i+1}. {item}")
                print(f"  ... and {len(result) - 3} more")
        elif isinstance(result, dict):
            print("✅ Result:")
            for key, value in result.items():
                if isinstance(value, list) and len(value) > 3:
                    print(f"  {key}: {len(value)} items")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"✅ {result}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
