#!/usr/bin/env python3
"""
Neo4j Graph Loader/Client for MAG

Provides:
- Connection management to Neo4j
- Schema setup (constraints, indexes, optional full-text/vector placeholders)
- Bulk upsert helpers to load nodes and edges from processed MAG files

Note: This module focuses on enabling graph traversal via Neo4j. Vector/HNSW search
remains in the existing FAISS/HNSW components per current architecture.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from loguru import logger
from neo4j import GraphDatabase, basic_auth


class Neo4jGraphClient:
    """Thin client around Neo4j driver with MAG-specific helpers."""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: Optional[str] = None,
    ) -> None:
        self.uri = uri
        self.user = user
        self.database = database
        self._driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
        self._db_kwargs = {"database": database} if database else {}

    def close(self) -> None:
        try:
            self._driver.close()
        except Exception:
            pass

    # -----------------------------
    # Schema setup
    # -----------------------------
    def ensure_constraints_and_indexes(self) -> None:
        """Create idempotent constraints and indexes."""
        stmts = [
            # Unique IDs
            "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.paperId IS UNIQUE",
            "CREATE CONSTRAINT author_id IF NOT EXISTS FOR (a:Author) REQUIRE a.authorId IS UNIQUE",
            "CREATE CONSTRAINT field_id IF NOT EXISTS FOR (f:Field) REQUIRE f.fieldId IS UNIQUE",
            "CREATE CONSTRAINT inst_id IF NOT EXISTS FOR (i:Institution) REQUIRE i.institutionId IS UNIQUE",
            # Lookups
            "CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.year)",
            "CREATE INDEX paper_mag_id IF NOT EXISTS FOR (p:Paper) ON (p.magId)",
            "CREATE INDEX author_name IF NOT EXISTS FOR (a:Author) ON (a.name)",
            "CREATE INDEX author_mag_id IF NOT EXISTS FOR (a:Author) ON (a.magId)",
            "CREATE INDEX field_name IF NOT EXISTS FOR (f:Field) ON (f.name)",
            "CREATE INDEX field_mag_id IF NOT EXISTS FOR (f:Field) ON (f.magId)",
            "CREATE INDEX field_level IF NOT EXISTS FOR (f:Field) ON (f.level)",
            "CREATE INDEX inst_name IF NOT EXISTS FOR (i:Institution) ON (i.name)",
            "CREATE INDEX inst_mag_id IF NOT EXISTS FOR (i:Institution) ON (i.magId)",
        ]

        def _run(tx, cypher: str) -> None:
            tx.run(cypher)

        with self._driver.session(**self._db_kwargs) as session:
            for stmt in stmts:
                try:
                    session.execute_write(_run, stmt)
                    logger.debug(f"Applied schema: {stmt}")
                except Exception as e:
                    # Constraint/index might already exist in some Neo4j versions
                    logger.warning(f"Schema statement may have failed (might already exist): {e}")

    # -----------------------------
    # Upsert primitives
    # -----------------------------
    def upsert_papers(self, papers: Iterable[Dict]) -> int:
        """Upsert Paper nodes.
        
        Expected keys: paperId (int), plus any other paper properties.
        The function will store all provided keys as properties.
        """
        # Use simple property setting (works without APOC)
        cypher = (
            "UNWIND $rows AS row "
            "MERGE (p:Paper {paperId: row.paperId}) "
            "SET p = row"
        )
        rows = []
        count = 0
        with self._driver.session(**self._db_kwargs) as session:
            for item in papers:
                if "paperId" not in item:
                    continue
                rows.append(item)
                if len(rows) >= 1000:
                    session.run(cypher, rows=rows)
                    count += len(rows)
                    rows = []
            if rows:
                session.run(cypher, rows=rows)
                count += len(rows)
        return count

    def upsert_authors(self, authors: Iterable[Dict]) -> int:
        """Upsert Author nodes with all provided properties."""
        cypher = (
            "UNWIND $rows AS row "
            "MERGE (a:Author {authorId: row.authorId}) "
            "SET a = row"
        )
        return self._batch_run(cypher, authors)

    def upsert_fields(self, fields: Iterable[Dict]) -> int:
        """Upsert Field nodes with all provided properties."""
        cypher = (
            "UNWIND $rows AS row "
            "MERGE (f:Field {fieldId: row.fieldId}) "
            "SET f = row"
        )
        return self._batch_run(cypher, fields)

    def upsert_institutions(self, institutions: Iterable[Dict]) -> int:
        """Upsert Institution nodes with all provided properties."""
        cypher = (
            "UNWIND $rows AS row "
            "MERGE (i:Institution {institutionId: row.institutionId}) "
            "SET i = row"
        )
        return self._batch_run(cypher, institutions)

    def _batch_run(self, cypher: str, rows_iter: Iterable[Dict]) -> int:
        rows: List[Dict] = []
        count = 0
        with self._driver.session(**self._db_kwargs) as session:
            for item in rows_iter:
                rows.append(item)
                if len(rows) >= 1000:
                    session.run(cypher, rows=rows)
                    count += len(rows)
                    rows = []
            if rows:
                session.run(cypher, rows=rows)
                count += len(rows)
        return count

    # -----------------------------
    # Relationship upserts
    # -----------------------------
    def upsert_writes(self, pairs: Iterable[Tuple[int, int]]) -> int:
        # MATCH only processes rows where both nodes exist
        # If nodes don't exist, the row is silently filtered out (which is what we want)
        cypher = (
            "UNWIND $rows AS row "
            "MATCH (a:Author {authorId: row.a}), (p:Paper {paperId: row.p}) "
            "MERGE (a)-[:AUTHORED]->(p)"
        )
        return self._batch_pairs(cypher, pairs)

    def upsert_cites(self, pairs: Iterable[Tuple[int, int]]) -> int:
        cypher = (
            "UNWIND $rows AS row "
            "MATCH (p:Paper {paperId: row.src}), (q:Paper {paperId: row.tgt}) "
            "MERGE (p)-[:CITES]->(q)"
        )
        return self._batch_pairs(cypher, pairs, row_keys=("src", "tgt"))

    def upsert_has_field(self, pairs: Iterable[Tuple[int, int]]) -> int:
        # MATCH only processes rows where both nodes exist
        # If nodes don't exist, the row is silently filtered out (which is what we want)
        cypher = (
            "UNWIND $rows AS row "
            "MATCH (p:Paper {paperId: row.p}), (f:Field {fieldId: row.f}) "
            "MERGE (p)-[:HAS_FIELD]->(f)"
        )
        return self._batch_pairs(cypher, pairs, row_keys=("p", "f"))

    def upsert_affiliated_with(self, pairs: Iterable[Tuple[int, int]]) -> int:
        # MATCH only processes rows where both nodes exist
        # If nodes don't exist, the row is silently filtered out (which is what we want)
        cypher = (
            "UNWIND $rows AS row "
            "MATCH (a:Author {authorId: row.a}), (i:Institution {institutionId: row.i}) "
            "MERGE (a)-[:AFFILIATED_WITH]->(i)"
        )
        return self._batch_pairs(cypher, pairs, row_keys=("a", "i"))

    def _batch_pairs(self, cypher: str, pairs: Iterable[Tuple[int, int]], row_keys: Tuple[str, str] = ("a", "p"), return_failed: bool = False) -> int:
        """Batch process pairs and create relationships.
        
        Args:
            cypher: Cypher query to execute
            pairs: Iterable of (source, target) pairs
            row_keys: Tuple of keys for source and target in the query
            return_failed: If True, return tuple of (created_count, failed_pairs), else just created_count
        
        Returns:
            If return_failed=False: int (created count)
            If return_failed=True: tuple (created_count, list of failed pairs)
        """
        key1, key2 = row_keys
        rows: List[Dict] = []
        total_processed = 0
        total_created = 0
        batch_num = 0
        failed_pairs = []
        
        with self._driver.session(**self._db_kwargs) as session:
            current_batch_pairs = []
            for u, v in pairs:
                rows.append({key1: u, key2: v})
                current_batch_pairs.append((u, v))
                
                if len(rows) >= 2000:
                    batch_num += 1
                    result = session.run(cypher, rows=rows)
                    summary = result.consume()
                    created = summary.counters.relationships_created or 0
                    total_created += created
                    total_processed += len(rows)
                    
                    # Track failed pairs if requested
                    if return_failed and created < len(rows):
                        # Find which pairs failed (nodes don't exist)
                        # We can't know exactly which ones failed, so we check each pair
                        for pair in current_batch_pairs:
                            # Check if nodes exist by trying to match them
                            check_query = cypher.replace("MERGE", "OPTIONAL MATCH").replace("MATCH", "OPTIONAL MATCH")
                            # This is approximate - we'll use a simpler approach
                            pass # Will handle differently
                    
                    if batch_num % 100 == 0: # Log every 100 batches (200k edges)
                        logger.debug(f" Processed {total_processed:,} edges, created {total_created:,} relationships")
                    rows = []
                    current_batch_pairs = []
                    
            if rows:
                batch_num += 1
                result = session.run(cypher, rows=rows)
                summary = result.consume()
                created = summary.counters.relationships_created or 0
                total_created += created
                total_processed += len(rows)
        
        # Log warning if many relationships weren't created (likely missing nodes)
        if total_processed > 0 and total_created < total_processed * 0.9:
            logger.warning(f" Only {total_created:,} out of {total_processed:,} relationships were created ({total_created/total_processed*100:.1f}%). Some nodes may be missing.")
        
        logger.debug(f" Final: processed {total_processed:,}, created {total_created:,} relationships")
        
        if return_failed:
            return total_created, failed_pairs
        return total_created

    # -----------------------------
    # Loading from processed files
    # -----------------------------
    def _normalize_value(self, value):
        """Normalize values: convert -1 to None, keep other values as-is."""
        if value == -1 or value == "-1" or value == -1.0:
            return None
        return value
    
    def _prepare_paper_node(self, obj: Dict) -> Dict:
        """Convert raw paper node to Neo4j format."""
        paper = {
            'paperId': int(obj['node_index']),
            'magId': int(obj.get('mag_id', 0)),
            'title': obj.get('title') or obj.get('OriginalTitle'),
            'originalTitle': obj.get('OriginalTitle'),
            'abstract': obj.get('abstract'),
            'year': obj.get('Year') or obj.get('year'),
            'date': self._normalize_value(obj.get('Date')),
            'docType': self._normalize_value(obj.get('DocType')),
            'publisher': self._normalize_value(obj.get('Publisher')),
            'journalId': self._normalize_value(obj.get('JournalId')),
            'journalDisplayName': self._normalize_value(obj.get('JournalDisplayName')),
            'journalRank': self._normalize_value(obj.get('JournalRank')),
            'journalPaperCount': self._normalize_value(obj.get('JournalPaperCount')),
            'journalCitationCount': self._normalize_value(obj.get('JournalCitationCount')),
            'conferenceSeriesId': self._normalize_value(obj.get('ConferenceSeriesId')),
            'conferenceSeriesDisplayName': self._normalize_value(obj.get('ConferenceSeriesDisplayName')),
            'conferenceInstanceId': self._normalize_value(obj.get('ConferenceInstanceId')),
            'referenceCount': self._normalize_value(obj.get('ReferenceCount')),
            'paperCitationCount': self._normalize_value(obj.get('PaperCitationCount')),
            'estimatedCitationCount': self._normalize_value(obj.get('EstimatedCitationCount')),
            'originalVenue': self._normalize_value(obj.get('OriginalVenue')),
            'paperRank': self._normalize_value(obj.get('PaperRank')),
            'bookTitle': self._normalize_value(obj.get('BookTitle')),
        }
        # Remove None values
        return {k: v for k, v in paper.items() if v is not None}
    
    def _prepare_author_node(self, obj: Dict) -> Dict:
        """Convert raw author node to Neo4j format."""
        author = {
            'authorId': int(obj['node_index']),
            'magId': int(obj.get('mag_id', 0)),
            'name': obj.get('DisplayName'),
            'displayName': obj.get('DisplayName'),
            'rank': self._normalize_value(obj.get('Rank')),
            'paperCount': self._normalize_value(obj.get('PaperCount')),
            'citationCount': self._normalize_value(obj.get('CitationCount')),
            'lastKnownAffiliationId': self._normalize_value(obj.get('LastKnownAffiliationId')),
        }
        return {k: v for k, v in author.items() if v is not None}
    
    def _prepare_field_node(self, obj: Dict) -> Dict:
        """Convert raw field node to Neo4j format."""
        field = {
            'fieldId': int(obj['node_index']),
            'magId': int(obj.get('mag_id', 0)),
            'name': obj.get('DisplayName'),
            'displayName': obj.get('DisplayName'),
            'level': self._normalize_value(obj.get('Level')),
            'rank': self._normalize_value(obj.get('Rank')),
            'paperCount': self._normalize_value(obj.get('PaperCount')),
            'citationCount': self._normalize_value(obj.get('CitationCount')),
        }
        return {k: v for k, v in field.items() if v is not None}
    
    def _prepare_institution_node(self, obj: Dict) -> Dict:
        """Convert raw institution node to Neo4j format."""
        inst = {
            'institutionId': int(obj['node_index']),
            'magId': int(obj.get('mag_id', 0)),
            'name': obj.get('DisplayName'),
            'displayName': obj.get('DisplayName'),
            'rank': self._normalize_value(obj.get('Rank')),
            'paperCount': self._normalize_value(obj.get('PaperCount')),
            'citationCount': self._normalize_value(obj.get('CitationCount')),
        }
        return {k: v for k, v in inst.items() if v is not None}
    
    def load_from_processed(self, processed_dir: str, node_types: Optional[List[str]] = None) -> Dict[str, int]:
        """Load nodes from processed MAG files.
        
        Args:
            processed_dir: Path to processed MAG directory
            node_types: List of node types to load (default: all ['paper', 'author', 'field_of_study', 'institution'])
        
        Returns:
            Dict mapping node type to count of loaded nodes
        """
        base = Path(processed_dir)
        self.ensure_constraints_and_indexes()
        
        if node_types is None:
            node_types = ['paper', 'author', 'field_of_study', 'institution']
        
        node_info = base / 'node_info.jsonl'
        if not node_info.exists():
            logger.warning(f"node_info.jsonl not found at {node_info}")
            return {}
        
        counts = {}
        
        # Load papers
        if 'paper' in node_types:
            logger.info("Loading Paper nodes from node_info.jsonl...")
            def _iter_papers():
                with node_info.open('r') as f:
                    for line in f:
                        obj = json.loads(line)
                        if obj.get('type') == 'paper':
                            yield self._prepare_paper_node(obj)
            counts['paper'] = self.upsert_papers(_iter_papers())
            logger.info(f" Upserted {counts['paper']:,} Paper nodes")
        
        # Load authors
        if 'author' in node_types:
            logger.info("Loading Author nodes from node_info.jsonl...")
            def _iter_authors():
                with node_info.open('r') as f:
                    for line in f:
                        obj = json.loads(line)
                        if obj.get('type') == 'author':
                            yield self._prepare_author_node(obj)
            counts['author'] = self.upsert_authors(_iter_authors())
            logger.info(f" Upserted {counts['author']:,} Author nodes")
        
        # Load fields
        if 'field_of_study' in node_types:
            logger.info("Loading Field nodes from node_info.jsonl...")
            def _iter_fields():
                with node_info.open('r') as f:
                    for line in f:
                        obj = json.loads(line)
                        if obj.get('type') == 'field_of_study':
                            yield self._prepare_field_node(obj)
            counts['field_of_study'] = self.upsert_fields(_iter_fields())
            logger.info(f" Upserted {counts['field_of_study']:,} Field nodes")
        
        # Load institutions
        if 'institution' in node_types:
            logger.info("Loading Institution nodes from node_info.jsonl...")
            def _iter_institutions():
                with node_info.open('r') as f:
                    for line in f:
                        obj = json.loads(line)
                        if obj.get('type') == 'institution':
                            yield self._prepare_institution_node(obj)
            counts['institution'] = self.upsert_institutions(_iter_institutions())
            logger.info(f" Upserted {counts['institution']:,} Institution nodes")
        
        logger.info(f" Node loading complete: {counts}")
        return counts
    
    def _detect_and_fix_edge_direction(self, src_id: int, tgt_id: int, expected_src_type: str, expected_tgt_type: str) -> Tuple[int, int]:
        """Detect if edge direction is reversed and fix it.
        
        Node ID ranges (approximate):
        - Authors: 0 to ~1,104,553
        - Institutions: ~1,104,554 to ~1,113,254
        - Fields: ~1,113,255 to ~1,172,723
        - Papers: ~1,172,724 to ~1,872,967
        
        Args:
            src_id: Source node ID
            tgt_id: Target node ID
            expected_src_type: Expected source node type ('author', 'paper', 'institution', 'field_of_study')
            expected_tgt_type: Expected target node type
        
        Returns:
            Tuple of (corrected_source_id, corrected_target_id)
        """
        # Define ID ranges for each node type (based on actual data)
        # These are approximate boundaries - we'll use them as heuristics
        AUTHOR_MAX = 1104553
        INSTITUTION_MIN = 1104554
        INSTITUTION_MAX = 1113254
        FIELD_MIN = 1113255
        FIELD_MAX = 1172723
        PAPER_MIN = 1172724
        
        def _get_node_type_range(node_id: int) -> str:
            """Determine node type based on ID range"""
            if node_id <= AUTHOR_MAX:
                return 'author'
            elif INSTITUTION_MIN <= node_id <= INSTITUTION_MAX:
                return 'institution'
            elif FIELD_MIN <= node_id <= FIELD_MAX:
                return 'field_of_study'
            elif node_id >= PAPER_MIN:
                return 'paper'
            else:
                return 'unknown'
        
        src_actual_type = _get_node_type_range(src_id)
        tgt_actual_type = _get_node_type_range(tgt_id)
        
        # Check if direction matches expected
        src_matches = src_actual_type == expected_src_type
        tgt_matches = tgt_actual_type == expected_tgt_type
        
        # Check if reversed
        src_matches_reversed = src_actual_type == expected_tgt_type
        tgt_matches_reversed = tgt_actual_type == expected_src_type
        
        # If reversed, swap
        if not src_matches and not tgt_matches and src_matches_reversed and tgt_matches_reversed:
            return (tgt_id, src_id)
        
        # If direction is correct, return as-is
        return (src_id, tgt_id)
    
    def load_edges_from_processed(self, processed_dir: str) -> Dict[str, int]:
        """Load edges from processed MAG files.
        
        Args:
            processed_dir: Path to processed MAG directory
        
        Returns:
            Dict mapping edge type to count of loaded edges
        """
        base = Path(processed_dir)
        
        # Load edge type dictionary
        edge_type_dict_path = base / 'edge_type_dict.json'
        if not edge_type_dict_path.exists():
            logger.warning(f"edge_type_dict.json not found at {edge_type_dict_path}")
            return {}
        
        with open(edge_type_dict_path, 'r') as f:
            edge_type_dict = json.load(f)
        
        # Load edge index and types
        edge_index_path = base / 'edge_index.json'
        edge_types_path = base / 'edge_types.json'
        
        if not edge_index_path.exists() or not edge_types_path.exists():
            logger.warning(f"Edge files not found")
            return {}
        
        logger.info("Loading edge index and types...")
        with open(edge_index_path, 'r') as f:
            edge_index = json.load(f)
        
        with open(edge_types_path, 'r') as f:
            edge_types = json.load(f)
        
        logger.info(f" Loaded {len(edge_types):,} edges from files")
        
        # Group edges by type
        edge_groups = defaultdict(list)
        for i, (src_global_idx, tgt_global_idx) in enumerate(zip(edge_index[0], edge_index[1])):
            edge_type_code = edge_types[i]
            edge_type_name = edge_type_dict[str(edge_type_code)]
            edge_groups[edge_type_name].append((src_global_idx, tgt_global_idx))
        
        counts = {}
        
        # Load each edge type
        for edge_type_name, edge_list in edge_groups.items():
            logger.info(f"Loading {edge_type_name} edges ({len(edge_list):,} edges)...")
            
            # Parse edge type: "author___writes___paper"
            parts = edge_type_name.split('___')
            if len(parts) != 3:
                logger.warning(f"Skipping edge type with unexpected format: {edge_type_name}")
                continue
            
            src_type, relation, tgt_type = parts
            
            # Two-pass approach: Try original direction first, then flip failed ones
            logger.info(f" Pass 1: Attempting to create relationships with original edge directions...")
            
            # First pass: Try all edges as-is
            first_pass_created = 0
            if edge_type_name == "author___writes___paper":
                first_pass_created = self.upsert_writes(edge_list)
            elif edge_type_name == "paper___cites___paper":
                first_pass_created = self.upsert_cites(edge_list)
            elif edge_type_name == "paper___has_topic___field_of_study":
                first_pass_created = self.upsert_has_field(edge_list)
            elif edge_type_name == "author___affiliated_with___institution":
                first_pass_created = self.upsert_affiliated_with(edge_list)
            
            logger.info(f" Pass 1: Created {first_pass_created:,} out of {len(edge_list):,} relationships")
            
            # If all created, we're done
            if first_pass_created == len(edge_list):
                counts[edge_type_name] = first_pass_created
                logger.info(f" Created all {counts[edge_type_name]:,} relationships in first pass")
                continue
            
            # Second pass: Identify failed edges, flip them, and check if flipped version already exists
            failed_count = len(edge_list) - first_pass_created
            logger.info(f" Pass 2: Identifying {failed_count:,} failed edges, flipping and checking if they already exist...")
            
            # Check which edges failed and prepare flipped pairs
            # But also check if the flipped relationship already exists (from a different edge)
            failed_pairs = []
            with self._driver.session(**self._db_kwargs) as session:
                batch_size = 10000
                for i in range(0, len(edge_list), batch_size):
                    batch = edge_list[i:i+batch_size]
                    
                    # Build query to check if relationship exists in ORIGINAL direction
                    # AND if flipped relationship exists
                    if edge_type_name == "author___writes___paper":
                        check_query = """
                        UNWIND $rows AS row
                        OPTIONAL MATCH (a1:Author {authorId: row.src})-[r1:AUTHORED]->(p1:Paper {paperId: row.tgt})
                        OPTIONAL MATCH (a2:Author {authorId: row.tgt})-[r2:AUTHORED]->(p2:Paper {paperId: row.src})
                        RETURN row.src AS src, row.tgt AS tgt, 
                               r1 IS NOT NULL AS exists_original,
                               r2 IS NOT NULL AS exists_flipped
                        """
                    elif edge_type_name == "paper___cites___paper":
                        check_query = """
                        UNWIND $rows AS row
                        OPTIONAL MATCH (p1:Paper {paperId: row.src})-[r1:CITES]->(q1:Paper {paperId: row.tgt})
                        OPTIONAL MATCH (p2:Paper {paperId: row.tgt})-[r2:CITES]->(q2:Paper {paperId: row.src})
                        RETURN row.src AS src, row.tgt AS tgt,
                               r1 IS NOT NULL AS exists_original,
                               r2 IS NOT NULL AS exists_flipped
                        """
                    elif edge_type_name == "paper___has_topic___field_of_study":
                        check_query = """
                        UNWIND $rows AS row
                        OPTIONAL MATCH (p1:Paper {paperId: row.src})-[r1:HAS_FIELD]->(f1:Field {fieldId: row.tgt})
                        OPTIONAL MATCH (p2:Paper {paperId: row.tgt})-[r2:HAS_FIELD]->(f2:Field {fieldId: row.src})
                        RETURN row.src AS src, row.tgt AS tgt,
                               r1 IS NOT NULL AS exists_original,
                               r2 IS NOT NULL AS exists_flipped
                        """
                    elif edge_type_name == "author___affiliated_with___institution":
                        check_query = """
                        UNWIND $rows AS row
                        OPTIONAL MATCH (a1:Author {authorId: row.src})-[r1:AFFILIATED_WITH]->(i1:Institution {institutionId: row.tgt})
                        OPTIONAL MATCH (a2:Author {authorId: row.tgt})-[r2:AFFILIATED_WITH]->(i2:Institution {institutionId: row.src})
                        RETURN row.src AS src, row.tgt AS tgt,
                               r1 IS NOT NULL AS exists_original,
                               r2 IS NOT NULL AS exists_flipped
                        """
                    else:
                        check_query = None
                    
                    if check_query:
                        rows = [{"src": src, "tgt": tgt} for src, tgt in batch]
                        result = session.run(check_query, rows=rows)
                        for record in result:
                            if not record["exists_original"]:
                                # Original relationship doesn't exist
                                if not record["exists_flipped"]:
                                    # Flipped relationship also doesn't exist - need to create it
                                    failed_pairs.append((record["tgt"], record["src"]))
                                # If flipped exists, skip it (already created from a different edge)
            
            logger.info(f" Pass 2: Found {len(failed_pairs):,} edges that need to be created (flipped and don't exist yet)...")
            
            # Try flipped pairs that don't already exist
            second_pass_created = 0
            if failed_pairs:
                if edge_type_name == "author___writes___paper":
                    second_pass_created = self.upsert_writes(failed_pairs)
                elif edge_type_name == "paper___cites___paper":
                    second_pass_created = self.upsert_cites(failed_pairs)
                elif edge_type_name == "paper___has_topic___field_of_study":
                    second_pass_created = self.upsert_has_field(failed_pairs)
                elif edge_type_name == "author___affiliated_with___institution":
                    second_pass_created = self.upsert_affiliated_with(failed_pairs)
            
            logger.info(f" Pass 2: Created {second_pass_created:,} additional relationships")
            
            total_created = first_pass_created + second_pass_created
            counts[edge_type_name] = total_created
            
            if edge_type_name == "author___writes___paper":
                logger.info(f" Created {total_created:,} AUTHORED relationships total (Pass 1: {first_pass_created:,}, Pass 2: {second_pass_created:,})")
            elif edge_type_name == "paper___cites___paper":
                logger.info(f" Created {total_created:,} CITES relationships total (Pass 1: {first_pass_created:,}, Pass 2: {second_pass_created:,})")
            elif edge_type_name == "paper___has_topic___field_of_study":
                logger.info(f" Created {total_created:,} HAS_FIELD relationships total (Pass 1: {first_pass_created:,}, Pass 2: {second_pass_created:,})")
            elif edge_type_name == "author___affiliated_with___institution":
                logger.info(f" Created {total_created:,} AFFILIATED_WITH relationships total (Pass 1: {first_pass_created:,}, Pass 2: {second_pass_created:,})")
            else:
                logger.warning(f"Unknown edge type: {edge_type_name}")
        
        logger.info(f" Edge loading complete: {counts}")
        return counts


__all__ = ["Neo4jGraphClient"]


