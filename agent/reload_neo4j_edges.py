#!/usr/bin/env python3
"""
Script to reload only edges (connections) into Neo4j database
This will:
1. Clear all existing relationships/edges
2. Reload edges from edge_index.json and edge_types.json
3. Keep all existing nodes intact
"""

import sys
import os
from pathlib import Path

# Add agent directory to path
sys.path.append(str(Path(__file__).parent))

from neo4j_graph import Neo4jGraphClient
from neo4j import GraphDatabase, basic_auth
from loguru import logger

def clear_edges_only(uri="bolt://localhost:7687", user="neo4j", password="neo4j123", database=None):
    """Clear only relationships/edges from Neo4j database, keeping all nodes"""
    logger.info("🗑️  Clearing existing relationships/edges (keeping all nodes)...")
    
    try:
        driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
        db_kwargs = {"database": database} if database else {}
        
        with driver.session(**db_kwargs) as session:
            # Delete relationships in batches to avoid memory issues
            logger.info("   Deleting relationships in batches...")
            total_deleted = 0
            batch_size = 100000  # Delete 100k at a time
            
            while True:
                # Delete a batch of relationships
                result = session.run(
                    f"MATCH ()-[r]->() WITH r LIMIT {batch_size} DELETE r RETURN count(r) AS deleted"
                )
                record = result.single()
                if record:
                    deleted = record["deleted"]
                    total_deleted += deleted
                    logger.info(f"   Deleted {total_deleted:,} relationships so far...")
                    if deleted == 0:
                        break
                else:
                    break
            
            logger.info(f"   ✅ Deleted {total_deleted:,} relationships total")
        
        driver.close()
        logger.info("✅ All relationships cleared successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to clear relationships: {e}")
        import traceback
        traceback.print_exc()
        return False

def reload_edges_only(processed_dir, uri="bolt://localhost:7687", user="neo4j", password="neo4j123", database=None, clear_first=True):
    """Reload only edges into Neo4j, keeping all existing nodes"""
    
    processed_path = Path(processed_dir)
    if not processed_path.exists():
        logger.error(f"❌ Processed directory not found: {processed_dir}")
        return False
    
    # Check if edge files exist
    edge_index_path = processed_path / 'edge_index.json'
    edge_types_path = processed_path / 'edge_types.json'
    edge_type_dict_path = processed_path / 'edge_type_dict.json'
    
    if not edge_index_path.exists():
        logger.error(f"❌ edge_index.json not found at {edge_index_path}")
        return False
    if not edge_types_path.exists():
        logger.error(f"❌ edge_types.json not found at {edge_types_path}")
        return False
    if not edge_type_dict_path.exists():
        logger.error(f"❌ edge_type_dict.json not found at {edge_type_dict_path}")
        return False
    
    logger.info("=" * 80)
    logger.info("🔄 RELOADING NEO4J EDGES (CONNECTIONS)")
    logger.info("=" * 80)
    logger.info(f"Processed directory: {processed_dir}")
    logger.info(f"Edge files:")
    logger.info(f"  - {edge_index_path}")
    logger.info(f"  - {edge_types_path}")
    logger.info(f"  - {edge_type_dict_path}")
    logger.info(f"Neo4j URI: {uri}")
    logger.info(f"Database: {database or 'default'}")
    logger.info("=" * 80)
    
    # Step 1: Clear existing edges if requested
    if clear_first:
        if not clear_edges_only(uri, user, password, database):
            logger.error("❌ Failed to clear relationships. Exiting.")
            return False
        logger.info("")
    else:
        logger.info("⚠️  Skipping edge clear (will add new edges to existing ones)")
        logger.info("")
    
    # Step 2: Load edges
    logger.info("=" * 80)
    logger.info("🔗 LOADING EDGES")
    logger.info("=" * 80)
    try:
        client = Neo4jGraphClient(uri, user, password, database)
        edge_counts = client.load_edges_from_processed(processed_dir)
        logger.info(f"✅ Edge loading complete: {edge_counts}")
        
        # Verify edge counts
        with client._driver.session(**client._db_kwargs) as session:
            authored_count = session.run("MATCH ()-[r:AUTHORED]->() RETURN count(r) AS count").single()["count"]
            cites_count = session.run("MATCH ()-[r:CITES]->() RETURN count(r) AS count").single()["count"]
            has_field_count = session.run("MATCH ()-[r:HAS_FIELD]->() RETURN count(r) AS count").single()["count"]
            affiliated_count = session.run("MATCH ()-[r:AFFILIATED_WITH]->() RETURN count(r) AS count").single()["count"]
            
            logger.info(f"\n📈 Final Neo4j edge counts:")
            logger.info(f"   AUTHORED: {authored_count:,}")
            logger.info(f"   CITES: {cites_count:,}")
            logger.info(f"   HAS_FIELD: {has_field_count:,}")
            logger.info(f"   AFFILIATED_WITH: {affiliated_count:,}")
            
            # Show total
            total_edges = authored_count + cites_count + has_field_count + affiliated_count
            logger.info(f"   TOTAL: {total_edges:,}")
        
        client.close()
    except Exception as e:
        logger.error(f"❌ Failed to load edges: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("🎉 EDGE RELOAD COMPLETE!")
    logger.info("=" * 80)
    logger.info("✅ All nodes remain intact, only edges were updated")
    logger.info("=" * 80)
    return True

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Reload only edges (connections) into Neo4j")
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="/shared/khoja/CogComp/datasets/MAG/processed",
        help="Path to processed MAG directory (default: /shared/khoja/CogComp/datasets/MAG/processed)"
    )
    parser.add_argument(
        "--uri",
        type=str,
        default=None,
        help="Neo4j URI (default: bolt://localhost:7687)"
    )
    parser.add_argument(
        "--user",
        type=str,
        default=None,
        help="Neo4j username (default: neo4j)"
    )
    parser.add_argument(
        "--password",
        type=str,
        default=None,
        help="Neo4j password (default: neo4j123)"
    )
    parser.add_argument(
        "--database",
        type=str,
        default=None,
        help="Neo4j database name (default: None, uses default database)"
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Skip clearing existing edges (will add new edges to existing ones)"
    )
    
    args = parser.parse_args()
    
    # Use environment variables or defaults
    uri = args.uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = args.user or os.getenv("NEO4J_USER", "neo4j")
    password = args.password or os.getenv("NEO4J_PASSWORD", "neo4j123")
    database = args.database or os.getenv("NEO4J_DATABASE") or None
    
    success = reload_edges_only(
        processed_dir=args.processed_dir,
        uri=uri,
        user=user,
        password=password,
        database=database,
        clear_first=not args.no_clear
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

