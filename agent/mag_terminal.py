#!/usr/bin/env python3
"""
MAG Agent Terminal Interface
Usage: python mag_terminal.py function_name "value"
"""

import sys
import os
import json
from loguru import logger

# Add the agent directory to the path
sys.path.append('/shared/khoja/CogComp/agent')

from mag_agent import MAGAgent
from flexible_mag_agent import FlexibleMAGAgent

# Global variables to keep loaded agents
mag_agent = None
flex_agent = None

def load_system():
    """Load the system once"""
    global mag_agent, flex_agent
    
    if mag_agent is not None:
        return True
    
    logger.info("🚀 Loading MAG Agent System...")
    
    try:
        # Load MAG Agent
        mag_agent = MAGAgent("/shared/khoja/CogComp/datasets/MAG/processed", 
                            "/shared/khoja/CogComp/output/mag_hnsw_indices")
        mag_agent.load_all()
        logger.info("✅ MAG Agent loaded")
        
        # Load Flexible Agent
        flex_agent = FlexibleMAGAgent("/shared/khoja/CogComp/datasets/MAG/processed", 
                                    "/shared/khoja/CogComp/output/mag_hnsw_indices")
        flex_agent.load_all()
        logger.info("✅ Flexible Agent loaded")
        
        logger.info("🎉 System loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load system: {e}")
        return False

def search_papers_by_title(query, top_k=10):
    """Search papers by title"""
    if not load_system():
        return None
    return mag_agent.search_papers_by_title(query, int(top_k))

def search_authors_by_name(query, top_k=10):
    """Search authors by name"""
    if not load_system():
        return None
    return mag_agent.search_authors_by_name(query, int(top_k))

def get_papers_by_year_range(start_year, end_year):
    """Get papers by year range"""
    if not load_system():
        return None
    return mag_agent.get_papers_by_year_range(int(start_year), int(end_year))

def query_natural_language(query):
    """Natural language query"""
    if not load_system():
        return None
    return mag_agent.query_natural_language(query)

def solve_query(query, session="default"):
    """Solve query using flexible agent"""
    if not load_system():
        return None
    return flex_agent.solve_query(query, session)

def get_authors_of_paper(paper_id):
    """Get authors of a paper"""
    if not load_system():
        return None
    return mag_agent.get_authors_of_paper(int(paper_id))

def get_papers_by_author(author_ids):
    """Get papers by author IDs"""
    if not load_system():
        return None
    # Parse comma-separated IDs
    ids = [int(x.strip()) for x in author_ids.split(',') if x.strip()]
    return mag_agent.get_papers_by_author(ids)

def get_papers_citing(paper_id):
    """Get papers citing a paper"""
    if not load_system():
        return None
    return mag_agent.get_papers_citing(int(paper_id))

def get_papers_cited_by(paper_id):
    """Get papers cited by a paper"""
    if not load_system():
        return None
    return mag_agent.get_papers_cited_by(int(paper_id))

def get_paper_metadata(paper_id):
    """Get paper metadata"""
    if not load_system():
        return None
    return mag_agent.get_paper_metadata(int(paper_id))

def get_author_metadata(author_id):
    """Get author metadata"""
    if not load_system():
        return None
    return mag_agent.get_author_metadata(int(author_id))

def help_functions():
    """Show available functions"""
    functions = [
        "search_papers_by_title 'query' [top_k]",
        "search_authors_by_name 'query' [top_k]", 
        "get_papers_by_year_range start_year end_year",
        "query_natural_language 'query'",
        "solve_query 'query' [session]",
        "get_authors_of_paper paper_id",
        "get_papers_by_author 'id1,id2,id3'",
        "get_papers_citing paper_id",
        "get_papers_cited_by paper_id",
        "get_paper_metadata paper_id",
        "get_author_metadata author_id"
    ]
    
    print("🎯 Available Functions:")
    print("=" * 40)
    for func in functions:
        print(f"  python mag_terminal.py {func}")
    print("\n💡 Examples:")
    print("  python mag_terminal.py search_papers_by_title 'machine learning' 5")
    print("  python mag_terminal.py get_papers_by_year_range 2010 2020")
    print("  python mag_terminal.py solve_query 'papers about AI' my_session")

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
        'get_papers_by_year_range': get_papers_by_year_range,
        'query_natural_language': query_natural_language,
        'solve_query': solve_query,
        'get_authors_of_paper': get_authors_of_paper,
        'get_papers_by_author': get_papers_by_author,
        'get_papers_citing': get_papers_citing,
        'get_papers_cited_by': get_papers_cited_by,
        'get_paper_metadata': get_paper_metadata,
        'get_author_metadata': get_author_metadata,
        'help': help_functions
    }
    
    if function_name not in functions:
        print(f"❌ Unknown function: {function_name}")
        help_functions()
        return 1
    
    try:
        # Call the function
        result = functions[function_name](*args)
        
        if result is None:
            print("❌ Function failed")
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
