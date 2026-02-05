#!/usr/bin/env python3
"""
Neo4j-backed Traversal Utilities for MAG

Implements the same interface as MAGTraversalUtils but executes Cypher on Neo4j.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger
from neo4j import Driver


class Neo4jTraversalUtils:
    def __init__(self, driver: Driver, database: Optional[str] = None) -> None:
        self._driver = driver
        self._db_kwargs = {"database": database} if database else {}

    # -----------------------------
    # Helpers
    # -----------------------------
    def _values(self, result, key: str) -> List[int]:
        return [record[key] for record in result]

    # -----------------------------
    # API methods (match existing names)
    # -----------------------------
    def authors_of_paper(self, paper_node_index: int) -> List[int]:
        cypher = (
            "MATCH (a:Author)-[:AUTHORED]->(p:Paper {paperId: $pid}) "
            "RETURN DISTINCT a.authorId AS id"
        )
        with self._driver.session(**self._db_kwargs) as session:
            res = session.run(cypher, pid=int(paper_node_index))
            return self._values(res, "id")

    def papers_by_author(self, author_ids: List[int]) -> List[int]:
        if not author_ids:
            return []
        cypher = (
            "UNWIND $aids AS aid "
            "MATCH (:Author {authorId: aid})-[:AUTHORED]->(p:Paper) "
            "RETURN DISTINCT p.paperId AS id"
        )
        with self._driver.session(**self._db_kwargs) as session:
            res = session.run(cypher, aids=[int(a) for a in author_ids])
            return self._values(res, "id")

    def papers_with_field(self, field_id: int) -> List[int]:
        cypher = (
            "MATCH (p:Paper)-[:HAS_FIELD]->(f:Field {fieldId: $fid}) "
            "RETURN DISTINCT p.paperId AS id"
        )
        with self._driver.session(**self._db_kwargs) as session:
            res = session.run(cypher, fid=int(field_id))
            return self._values(res, "id")

    def authors_affiliated_with(self, institution_id: int) -> List[int]:
        cypher = (
            "MATCH (a:Author)-[:AFFILIATED_WITH]->(i:Institution {institutionId: $iid}) "
            "RETURN DISTINCT a.authorId AS id"
        )
        with self._driver.session(**self._db_kwargs) as session:
            res = session.run(cypher, iid=int(institution_id))
            return self._values(res, "id")

    def papers_citing(self, paper_node_index: int) -> List[int]:
        cypher = (
            "MATCH (q:Paper)-[:CITES]->(p:Paper {paperId: $pid}) "
            "RETURN DISTINCT q.paperId AS id"
        )
        with self._driver.session(**self._db_kwargs) as session:
            res = session.run(cypher, pid=int(paper_node_index))
            return self._values(res, "id")

    def papers_cited_by(self, paper_node_index: int) -> List[int]:
        cypher = (
            "MATCH (p:Paper {paperId: $pid})-[:CITES]->(q:Paper) "
            "RETURN DISTINCT q.paperId AS id"
        )
        with self._driver.session(**self._db_kwargs) as session:
            res = session.run(cypher, pid=int(paper_node_index))
            return self._values(res, "id")

    def papers_by_year_range(self, start_year: int, end_year: int) -> List[int]:
        cypher = (
            "MATCH (p:Paper) WHERE p.year >= $s AND p.year <= $e "
            "RETURN DISTINCT p.paperId AS id"
        )
        with self._driver.session(**self._db_kwargs) as session:
            res = session.run(cypher, s=int(start_year), e=int(end_year))
            return self._values(res, "id")

    def coauthors_of_author(self, author_id: int) -> List[int]:
        cypher = (
            "MATCH (:Author {authorId: $aid})-[:AUTHORED]->(:Paper)<-[:AUTHORED]-(co:Author) "
            "WHERE co.authorId <> $aid "
            "RETURN DISTINCT co.authorId AS id"
        )
        with self._driver.session(**self._db_kwargs) as session:
            res = session.run(cypher, aid=int(author_id))
            return self._values(res, "id")

    def cited_by_at_least_two_citers(self, target_paper_id: int) -> List[int]:
        cypher = (
            "MATCH (citer:Paper)-[:CITES]->(t:Paper {paperId: $pid}) "
            "MATCH (citer)-[:CITES]->(x:Paper) "
            "WITH x, COUNT(DISTINCT citer) AS k "
            "WHERE k >= 2 AND x.paperId <> $pid "
            "RETURN x.paperId AS id ORDER BY k DESC"
        )
        with self._driver.session(**self._db_kwargs) as session:
            res = session.run(cypher, pid=int(target_paper_id))
            return self._values(res, "id")

    # -----------------------------
    # Metadata retrieval methods
    # -----------------------------
    def get_paper_metadata(self, paper_id: int) -> Optional[Dict[str, Any]]:
        """Get all metadata for a paper node"""
        cypher = "MATCH (p:Paper {paperId: $pid}) RETURN properties(p) AS props"
        with self._driver.session(**self._db_kwargs) as session:
            result = session.run(cypher, pid=int(paper_id))
            record = result.single()
            return record["props"] if record else None

    def get_author_metadata(self, author_id: int) -> Optional[Dict[str, Any]]:
        """Get all metadata for an author node"""
        cypher = "MATCH (a:Author {authorId: $aid}) RETURN properties(a) AS props"
        with self._driver.session(**self._db_kwargs) as session:
            result = session.run(cypher, aid=int(author_id))
            record = result.single()
            return record["props"] if record else None

    def get_field_metadata(self, field_id: int) -> Optional[Dict[str, Any]]:
        """Get all metadata for a field node"""
        cypher = "MATCH (f:Field {fieldId: $fid}) RETURN properties(f) AS props"
        with self._driver.session(**self._db_kwargs) as session:
            result = session.run(cypher, fid=int(field_id))
            record = result.single()
            return record["props"] if record else None

    def get_institution_metadata(self, institution_id: int) -> Optional[Dict[str, Any]]:
        """Get all metadata for an institution node"""
        cypher = "MATCH (i:Institution {institutionId: $iid}) RETURN properties(i) AS props"
        with self._driver.session(**self._db_kwargs) as session:
            result = session.run(cypher, iid=int(institution_id))
            record = result.single()
            return record["props"] if record else None


__all__ = ["Neo4jTraversalUtils"]


