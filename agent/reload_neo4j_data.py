#!/usr/bin/env python3
"""
Script to reload MAG data into Neo4j database
This will:
1. Clear all existing nodes and relationships
2. Reload nodes from node_info.jsonl
3. Reload edges from edge_index.json and edge_types.json
"""

import sys
import os
from pathlib import Path

# Add agent directory to path
sys.path.append(str(Path(__file__).parent))

from neo4j_graph import Neo4jGraphClient
from neo4j import GraphDatabase, basic_auth
from loguru import logger

def clear_database(uri="bolt://localhost:7687", user="neo4j", password="neo4j123", database=None):
    """Clear all nodes and relationships from Neo4j database"""
    logger.info("🗑️  Clearing existing Neo4j database...")
    
    try:
        driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
        db_kwargs = {"database": database} if database else {}
        
        with driver.session(**db_kwargs) as session:
            # Delete all relationships first in batches (required before deleting nodes)
            logger.info("   Deleting all relationships in batches...")
            total_deleted_rels = 0
            batch_size = 100000  # Delete 100k at a time
            
            while True:
                # Delete a batch of relationships
                result = session.run(
                    f"MATCH ()-[r]->() WITH r LIMIT {batch_size} DELETE r RETURN count(r) AS deleted"
                )
                record = result.single()
                if record:
                    deleted = record["deleted"]
                    total_deleted_rels += deleted
                    if total_deleted_rels % 1000000 == 0 or deleted == 0:
                        logger.info(f"   Deleted {total_deleted_rels:,} relationships so far...")
                    if deleted == 0:
                        break
                else:
                    break
            
            logger.info(f"   ✅ Deleted {total_deleted_rels:,} relationships total")
            
            # Delete all nodes in batches
            logger.info("   Deleting all nodes in batches...")
            total_deleted_nodes = 0
            batch_size = 100000  # Delete 100k at a time
            
            while True:
                # Delete a batch of nodes
                result = session.run(
                    f"MATCH (n) WITH n LIMIT {batch_size} DETACH DELETE n RETURN count(n) AS deleted"
                )
                record = result.single()
                if record:
                    deleted = record["deleted"]
                    total_deleted_nodes += deleted
                    if total_deleted_nodes % 1000000 == 0 or deleted == 0:
                        logger.info(f"   Deleted {total_deleted_nodes:,} nodes so far...")
                    if deleted == 0:
                        break
                else:
                    break
            
            logger.info(f"   ✅ Deleted {total_deleted_nodes:,} nodes total")
        
        driver.close()
        logger.info("✅ Database cleared successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to clear database: {e}")
        import traceback
        traceback.print_exc()
        return False

def reload_data(processed_dir, uri="bolt://localhost:7687", user="neo4j", password="neo4j123", database=None, clear_first=True):
    """Reload all data into Neo4j"""
    
    processed_path = Path(processed_dir)
    if not processed_path.exists():
        logger.error(f"❌ Processed directory not found: {processed_dir}")
        return False
    
    logger.info("=" * 80)
    logger.info("🔄 RELOADING NEO4J DATABASE")
    logger.info("=" * 80)
    logger.info(f"Processed directory: {processed_dir}")
    logger.info(f"Neo4j URI: {uri}")
    logger.info(f"Database: {database or 'default'}")
    logger.info("=" * 80)
    
    # Step 1: Clear database if requested
    if clear_first:
        if not clear_database(uri, user, password, database):
            logger.error("❌ Failed to clear database. Exiting.")
            return False
        logger.info("")
    else:
        logger.info("⚠️  Skipping database clear (will upsert existing data)")
        logger.info("")
    
    # Step 2: Load nodes
    logger.info("=" * 80)
    logger.info("📊 LOADING NODES")
    logger.info("=" * 80)
    try:
        client = Neo4jGraphClient(uri, user, password, database)
        node_counts = client.load_from_processed(processed_dir)
        logger.info(f"✅ Node loading complete: {node_counts}")
        
        # Verify node counts
        with client._driver.session(**client._db_kwargs) as session:
            paper_count = session.run("MATCH (p:Paper) RETURN count(p) AS count").single()["count"]
            author_count = session.run("MATCH (a:Author) RETURN count(a) AS count").single()["count"]
            field_count = session.run("MATCH (f:Field) RETURN count(f) AS count").single()["count"]
            inst_count = session.run("MATCH (i:Institution) RETURN count(i) AS count").single()["count"]
            
            logger.info(f"\n📈 Final Neo4j node counts:")
            logger.info(f"   Papers: {paper_count:,}")
            logger.info(f"   Authors: {author_count:,}")
            logger.info(f"   Fields: {field_count:,}")
            logger.info(f"   Institutions: {inst_count:,}")
        
        client.close()
    except Exception as e:
        logger.error(f"❌ Failed to load nodes: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Load edges
    logger.info("")
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
        
        client.close()
    except Exception as e:
        logger.error(f"❌ Failed to load edges: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("🎉 DATABASE RELOAD COMPLETE!")
    logger.info("=" * 80)
    return True

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Reload MAG data into Neo4j")
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
        help="Skip clearing database (will upsert instead of full reload)"
    )
    
    args = parser.parse_args()
    
    # Use environment variables or defaults
    uri = args.uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = args.user or os.getenv("NEO4J_USER", "neo4j")
    password = args.password or os.getenv("NEO4J_PASSWORD", "neo4j123")
    database = args.database or os.getenv("NEO4J_DATABASE") or None
    
    success = reload_data(
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

