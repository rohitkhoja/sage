#!/usr/bin/env python3
"""
Simple script to run Cypher queries against Neo4j.

Usage:
    python agent/run_cypher_query.py "MATCH (p:Paper) RETURN count(p) AS count"
    python agent/run_cypher_query.py "MATCH (a:Author {name: 'John Smith'}) RETURN a"
"""

import sys
import os
from pathlib import Path
from neo4j import GraphDatabase, basic_auth

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_query(cypher_query: str, uri: str = None, user: str = None, password: str = None):
    """Run a Cypher query and print results."""
    uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = user or os.getenv("NEO4J_USER", "neo4j")
    password = password or os.getenv("NEO4J_PASSWORD", "neo4j123")
    
    driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
    
    try:
        with driver.session() as session:
            result = session.run(cypher_query)
            
            # Get column names
            keys = result.keys()
            print(f"Columns: {', '.join(keys)}")
            print("-" * 80)
            
            # Print results
            count = 0
            for record in result:
                count += 1
                row = {key: record[key] for key in keys}
                print(f"Row {count}: {row}")
            
            print("-" * 80)
            print(f"Total rows: {count}")
            
    finally:
        driver.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_cypher_query.py '<CYPHER_QUERY>'")
        print("\nExamples:")
        print(' python run_cypher_query.py "MATCH (p:Paper) RETURN count(p) AS count"')
        print(' python run_cypher_query.py "MATCH (a:Author {name: \\"John Smith\\"}) RETURN a.authorId, a.name"')
        print(' python run_cypher_query.py "MATCH (p:Paper)-[:AUTHORED]->(a:Author) RETURN p.title, a.name LIMIT 10"')
        sys.exit(1)
    
    query = sys.argv[1]
    run_query(query)

