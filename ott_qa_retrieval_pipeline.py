#!/usr/bin/env python3
"""
OTT-QA Graph-Enhanced Hybrid Retrieval Pipeline

This pipeline combines:
1. HNSW dense search (using pre-computed embeddings)
2. BM25 sparse search (with source names included)
3. Graph-based 1-hop neighbor expansion
4. 80th percentile cutoff for filtering results

Author: Generated for CogComp project
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path("/shared/khoja/CogComp")
OUTPUT_DIR = BASE_DIR / "output"
DOCS_CHUNKS_DIR = OUTPUT_DIR / "full_pipeline" / "docs_chunks_1"
TABLE_CHUNKS_FILE = OUTPUT_DIR / "full_pipeline" / "table_chunks_with_metadata.json"
QUESTIONS_FILE = OUTPUT_DIR / "dense_sparse_average_results (1).csv"
SIMILARITY_EDGES_FILE = OUTPUT_DIR / "analysis_cache" / "knowledge_graph" / "unique_edges_20250812_164429.json"
STRUCTURAL_LINKS_FILE = BASE_DIR / "datasets" / "ott-qa-full-data.json"

# Output paths
PIPELINE_CACHE_DIR = OUTPUT_DIR / "ott_qa_pipeline_cache"
RESULTS_FILE = OUTPUT_DIR / "ott_qa_retrieval_results.csv"


class ChunkLoader:
    """Loads and manages document and table chunks."""
    
    def __init__(self):
        self.chunks: Dict[str, dict] = {}  # chunk_id -> chunk data
        self.embeddings: Dict[str, np.ndarray] = {}  # chunk_id -> embedding
        self.bm25_texts: Dict[str, str] = {}  # chunk_id -> enriched text for BM25
        
    def load_document_chunks(self, docs_dir: Path) -> int:
        """Load document chunks with embeddings from individual JSON files."""
        logger.info(f"Loading document chunks from {docs_dir}...")
        
        json_files = list(docs_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} document chunk files")
        
        loaded = 0
        for json_file in tqdm(json_files, desc="Loading doc chunks"):
            try:
                with open(json_file, 'r') as f:
                    chunk = json.load(f)
                
                chunk_id = chunk.get('chunk_id')
                if not chunk_id:
                    continue
                    
                self.chunks[chunk_id] = chunk
                
                # Extract embedding
                if 'embedding' in chunk:
                    self.embeddings[chunk_id] = np.array(chunk['embedding'], dtype=np.float32)
                
                # Build BM25 text with source name
                source_name = ""
                if 'source_info' in chunk and 'source_name' in chunk['source_info']:
                    source_name = chunk['source_info']['source_name']
                content = chunk.get('content', '')
                self.bm25_texts[chunk_id] = f"{source_name} | {content}" if source_name else content
                
                loaded += 1
                
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
                
        logger.info(f"Loaded {loaded} document chunks with {len(self.embeddings)} embeddings")
        return loaded
    
    def load_table_chunks(self, table_file: Path) -> int:
        """Load table chunks from JSON file."""
        logger.info(f"Loading table chunks from {table_file}...")
        
        with open(table_file, 'r') as f:
            table_chunks = json.load(f)
        
        loaded = 0
        for chunk in tqdm(table_chunks, desc="Loading table chunks"):
            chunk_id = chunk.get('chunk_id')
            if not chunk_id:
                continue
                
            self.chunks[chunk_id] = chunk
            
            # Build BM25 text with source name
            source_name = chunk.get('source_name', '')
            content = chunk.get('content', '')
            self.bm25_texts[chunk_id] = f"{source_name} | {content}" if source_name else content
            
            loaded += 1
            
        logger.info(f"Loaded {loaded} table chunks (need embeddings)")
        return loaded
    
    def generate_table_embeddings(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Generate embeddings for table chunks that don't have them."""
        from sentence_transformers import SentenceTransformer
        
        # Find chunks without embeddings
        chunks_needing_embeddings = [
            (chunk_id, self.chunks[chunk_id])
            for chunk_id in self.chunks
            if chunk_id not in self.embeddings
        ]
        
        if not chunks_needing_embeddings:
            logger.info("All chunks already have embeddings")
            return
            
        logger.info(f"Generating embeddings for {len(chunks_needing_embeddings)} chunks...")
        
        model = SentenceTransformer(model_name)
        
        # Prepare texts for embedding
        texts = []
        chunk_ids = []
        for chunk_id, chunk in chunks_needing_embeddings:
            content = chunk.get('content', '')
            if not content:
                content = chunk.get('summary', '')
            texts.append(content)
            chunk_ids.append(chunk_id)
        
        # Generate embeddings in batches
        batch_size = 64
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            batch_ids = chunk_ids[i:i+batch_size]
            
            embeddings = model.encode(batch_texts, convert_to_numpy=True)
            
            for chunk_id, emb in zip(batch_ids, embeddings):
                self.embeddings[chunk_id] = emb.astype(np.float32)
        
        logger.info(f"Generated {len(chunks_needing_embeddings)} embeddings")
    
    def get_all_chunk_ids(self) -> List[str]:
        """Get all chunk IDs in order."""
        return list(self.chunks.keys())
    
    def get_embedding_matrix(self, chunk_ids: List[str]) -> np.ndarray:
        """Get embedding matrix for given chunk IDs."""
        embeddings = []
        for chunk_id in chunk_ids:
            if chunk_id in self.embeddings:
                embeddings.append(self.embeddings[chunk_id])
            else:
                # Use zero vector as placeholder
                dim = next(iter(self.embeddings.values())).shape[0] if self.embeddings else 384
                embeddings.append(np.zeros(dim, dtype=np.float32))
        return np.vstack(embeddings)


class HNSWIndex:
    """FAISS HNSW index for dense search."""
    
    def __init__(self):
        self.index = None
        self.chunk_ids: List[str] = []
        self.id_to_idx: Dict[str, int] = {}
        
    def build(self, chunk_ids: List[str], embeddings: np.ndarray, M: int = 32, ef_construction: int = 200):
        """Build HNSW index from embeddings."""
        import faiss
        
        logger.info(f"Building HNSW index with {len(chunk_ids)} vectors...")
        
        self.chunk_ids = chunk_ids
        self.id_to_idx = {cid: idx for idx, cid in enumerate(chunk_ids)}
        
        dim = embeddings.shape[1]
        
        # Create HNSW index
        self.index = faiss.IndexHNSWFlat(dim, M)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = 100  # Can be adjusted at search time
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add vectors
        self.index.add(embeddings)
        
        logger.info(f"HNSW index built with {self.index.ntotal} vectors")
        
    def search(self, query_embedding: np.ndarray, k: int = 100) -> Tuple[List[str], np.ndarray]:
        """Search for k nearest neighbors."""
        import faiss
        
        # Normalize query
        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        
        # Search
        distances, indices = self.index.search(query, k)
        
        # Convert to chunk IDs and distances
        result_ids = [self.chunk_ids[idx] for idx in indices[0] if idx >= 0]
        result_distances = distances[0][:len(result_ids)]
        
        return result_ids, result_distances
    
    def save(self, path: Path):
        """Save index to disk."""
        import faiss
        
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "hnsw.index"))
        with open(path / "chunk_ids.pkl", 'wb') as f:
            pickle.dump(self.chunk_ids, f)
            
    def load(self, path: Path):
        """Load index from disk."""
        import faiss
        
        self.index = faiss.read_index(str(path / "hnsw.index"))
        with open(path / "chunk_ids.pkl", 'rb') as f:
            self.chunk_ids = pickle.load(f)
        self.id_to_idx = {cid: idx for idx, cid in enumerate(self.chunk_ids)}


class BM25Index:
    """BM25 index for sparse search."""
    
    def __init__(self):
        self.bm25 = None
        self.chunk_ids: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        
    def build(self, chunk_ids: List[str], texts: Dict[str, str]):
        """Build BM25 index from texts."""
        from rank_bm25 import BM25Okapi
        
        logger.info(f"Building BM25 index with {len(chunk_ids)} documents...")
        
        self.chunk_ids = chunk_ids
        
        # Tokenize corpus
        self.tokenized_corpus = []
        for chunk_id in tqdm(chunk_ids, desc="Tokenizing for BM25"):
            text = texts.get(chunk_id, "")
            tokens = self._tokenize(text)
            self.tokenized_corpus.append(tokens)
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        logger.info("BM25 index built")
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens
    
    def search(self, query: str, k: int = 100) -> Tuple[List[str], np.ndarray]:
        """Search for k most relevant documents."""
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        result_ids = [self.chunk_ids[idx] for idx in top_indices]
        result_scores = scores[top_indices]
        
        return result_ids, result_scores
    
    def save(self, path: Path):
        """Save index to disk."""
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "bm25.pkl", 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'chunk_ids': self.chunk_ids,
                'tokenized_corpus': self.tokenized_corpus
            }, f)
            
    def load(self, path: Path):
        """Load index from disk."""
        with open(path / "bm25.pkl", 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.chunk_ids = data['chunk_ids']
            self.tokenized_corpus = data['tokenized_corpus']


class NeighborGraph:
    """Unified neighbor graph combining similarity edges and structural links."""
    
    def __init__(self):
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        
    def load_similarity_edges(self, edges_file: Path):
        """Load similarity edges from JSON file."""
        logger.info(f"Loading similarity edges from {edges_file}...")
        
        with open(edges_file, 'r') as f:
            edges = json.load(f)
        
        for edge in tqdm(edges, desc="Loading similarity edges"):
            source = edge.get('source_chunk_id')
            target = edge.get('target_chunk_id')
            if source and target:
                self.adjacency[source].add(target)
                self.adjacency[target].add(source)  # Bidirectional
        
        logger.info(f"Loaded {len(edges)} similarity edges, {len(self.adjacency)} nodes")
        
    def load_structural_links(self, links_file: Path, chunk_ids: Set[str]):
        """Load structural links and convert to chunk-level edges."""
        logger.info(f"Loading structural links from {links_file}...")
        
        with open(links_file, 'r') as f:
            docs = json.load(f)
        
        # Build doc_id to chunk_ids mapping
        doc_to_chunks: Dict[str, List[str]] = defaultdict(list)
        for chunk_id in chunk_ids:
            # Extract doc_id from chunk_id (e.g., "ottaqa-xxx_chunk_0_hash" -> "ottaqa-xxx")
            parts = chunk_id.split('_chunk_')
            if parts:
                doc_id = parts[0]
                doc_to_chunks[doc_id].append(chunk_id)
        
        structural_edges = 0
        for doc in tqdm(docs, desc="Processing structural links"):
            doc_id = doc.get('id')
            structural_links = doc.get('structural_link', [])
            
            if not doc_id or not structural_links:
                continue
            
            # Get chunks for this doc
            source_chunks = doc_to_chunks.get(doc_id, [])
            
            for linked_doc_id in structural_links:
                target_chunks = doc_to_chunks.get(linked_doc_id, [])
                
                # Create edges between all chunk pairs
                for src_chunk in source_chunks:
                    for tgt_chunk in target_chunks:
                        self.adjacency[src_chunk].add(tgt_chunk)
                        self.adjacency[tgt_chunk].add(src_chunk)
                        structural_edges += 1
        
        logger.info(f"Added {structural_edges} structural edges")
        
    def get_neighbors(self, chunk_id: str) -> Set[str]:
        """Get 1-hop neighbors of a chunk."""
        return self.adjacency.get(chunk_id, set())
    
    def expand_with_neighbors(self, chunk_ids: Set[str]) -> Set[str]:
        """Expand a set of chunk IDs with their 1-hop neighbors."""
        expanded = set(chunk_ids)
        for chunk_id in chunk_ids:
            expanded.update(self.get_neighbors(chunk_id))
        return expanded
    
    def save(self, path: Path):
        """Save graph to disk."""
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "neighbor_graph.pkl", 'wb') as f:
            pickle.dump(dict(self.adjacency), f)
            
    def load(self, path: Path):
        """Load graph from disk."""
        with open(path / "neighbor_graph.pkl", 'rb') as f:
            self.adjacency = defaultdict(set, {k: set(v) for k, v in pickle.load(f).items()})


class HybridRetriever:
    """Hybrid retrieval combining HNSW, BM25, and graph expansion."""
    
    def __init__(self, 
                 chunk_loader: ChunkLoader,
                 hnsw_index: HNSWIndex,
                 bm25_index: BM25Index,
                 neighbor_graph: NeighborGraph,
                 embedding_model=None):
        self.chunk_loader = chunk_loader
        self.hnsw_index = hnsw_index
        self.bm25_index = bm25_index
        self.neighbor_graph = neighbor_graph
        self.embedding_model = embedding_model
        
    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query string."""
        if self.embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        return self.embedding_model.encode(query, convert_to_numpy=True).astype(np.float32)
    
    def _apply_percentile_cutoff_hnsw(self, chunk_ids: List[str], distances: np.ndarray, 
                                       percentile: float = 20) -> Tuple[List[str], np.ndarray]:
        """Apply percentile cutoff for HNSW (lower distance = better, keep bottom percentile)."""
        if len(distances) == 0:
            return [], np.array([])
        
        threshold = np.percentile(distances, percentile)
        mask = distances <= threshold
        
        filtered_ids = [cid for cid, keep in zip(chunk_ids, mask) if keep]
        filtered_distances = distances[mask]
        
        return filtered_ids, filtered_distances
    
    def _apply_percentile_cutoff_bm25(self, chunk_ids: List[str], scores: np.ndarray,
                                       percentile: float = 80) -> Tuple[List[str], np.ndarray]:
        """Apply percentile cutoff for BM25 (higher score = better, keep top percentile)."""
        if len(scores) == 0:
            return [], np.array([])
        
        # Filter out zero scores first
        nonzero_mask = scores > 0
        if not np.any(nonzero_mask):
            return [], np.array([])
        
        threshold = np.percentile(scores[nonzero_mask], percentile)
        mask = scores >= threshold
        
        filtered_ids = [cid for cid, keep in zip(chunk_ids, mask) if keep]
        filtered_scores = scores[mask]
        
        return filtered_ids, filtered_scores
    
    def retrieve(self, query: str, k: int = 20, initial_k: int = 100) -> List[Tuple[str, float]]:
        """
        Retrieve top-k chunks for a query using hybrid approach.
        
        Steps:
        1. HNSW search -> apply 80th percentile cutoff
        2. BM25 search -> apply 80th percentile cutoff
        3. Find common chunks
        4. Expand with 1-hop neighbors
        5. Rerank and return top-k
        """
        # Step 1: HNSW search
        query_embedding = self._embed_query(query)
        hnsw_ids, hnsw_distances = self.hnsw_index.search(query_embedding, k=initial_k)
        hnsw_filtered_ids, hnsw_filtered_distances = self._apply_percentile_cutoff_hnsw(
            hnsw_ids, hnsw_distances, percentile=20
        )
        
        # Step 2: BM25 search
        bm25_ids, bm25_scores = self.bm25_index.search(query, k=initial_k)
        bm25_filtered_ids, bm25_filtered_scores = self._apply_percentile_cutoff_bm25(
            bm25_ids, bm25_scores, percentile=80
        )
        
        # Step 3: Find common chunks
        hnsw_set = set(hnsw_filtered_ids)
        bm25_set = set(bm25_filtered_ids)
        common_chunks = hnsw_set & bm25_set
        
        # If no common chunks, use union with prioritization
        if not common_chunks:
            # Combine both sets, prioritizing chunks that appear in both original results
            all_hnsw = set(hnsw_ids)
            all_bm25 = set(bm25_ids)
            common_chunks = all_hnsw & all_bm25
            
            if not common_chunks:
                # If still empty, use top results from both
                common_chunks = set(hnsw_ids[:k//2]) | set(bm25_ids[:k//2])
        
        # Step 4: Expand with 1-hop neighbors
        expanded_chunks = self.neighbor_graph.expand_with_neighbors(common_chunks)
        
        # Step 5: Rerank
        # Score each chunk: combine HNSW similarity and BM25 score
        chunk_scores = {}
        
        # Create lookup dicts
        hnsw_score_dict = {cid: 1.0 / (1.0 + dist) for cid, dist in zip(hnsw_ids, hnsw_distances)}
        bm25_score_dict = {cid: score for cid, score in zip(bm25_ids, bm25_scores)}
        
        # Normalize BM25 scores
        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1.0
        if max_bm25 > 0:
            bm25_score_dict = {k: v / max_bm25 for k, v in bm25_score_dict.items()}
        
        for chunk_id in expanded_chunks:
            hnsw_score = hnsw_score_dict.get(chunk_id, 0.0)
            bm25_score = bm25_score_dict.get(chunk_id, 0.0)
            
            # Boost for common chunks
            boost = 1.5 if chunk_id in common_chunks else 1.0
            
            # Combined score
            combined_score = boost * (0.5 * hnsw_score + 0.5 * bm25_score)
            chunk_scores[chunk_id] = combined_score
        
        # Sort by score and return top-k
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        return sorted_chunks


def main():
    """Main pipeline execution."""
    
    # Create cache directory
    PIPELINE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # ========== Step 1: Load Document Chunks ==========
    logger.info("=" * 60)
    logger.info("Step 1: Loading Document Chunks")
    logger.info("=" * 60)
    
    chunk_loader = ChunkLoader()
    chunk_loader.load_document_chunks(DOCS_CHUNKS_DIR)
    
    # ========== Step 2: Load Table Chunks ==========
    logger.info("=" * 60)
    logger.info("Step 2: Loading Table Chunks")
    logger.info("=" * 60)
    
    chunk_loader.load_table_chunks(TABLE_CHUNKS_FILE)
    
    # Generate embeddings for table chunks
    chunk_loader.generate_table_embeddings()
    
    logger.info(f"Total chunks: {len(chunk_loader.chunks)}")
    logger.info(f"Total embeddings: {len(chunk_loader.embeddings)}")
    
    # ========== Step 3: Build HNSW Index ==========
    logger.info("=" * 60)
    logger.info("Step 3: Building HNSW Index")
    logger.info("=" * 60)
    
    hnsw_cache = PIPELINE_CACHE_DIR / "hnsw"
    hnsw_index = HNSWIndex()
    
    if (hnsw_cache / "hnsw.index").exists():
        logger.info("Loading cached HNSW index...")
        hnsw_index.load(hnsw_cache)
    else:
        chunk_ids = chunk_loader.get_all_chunk_ids()
        embeddings = chunk_loader.get_embedding_matrix(chunk_ids)
        hnsw_index.build(chunk_ids, embeddings)
        hnsw_index.save(hnsw_cache)
    
    # ========== Step 4: Build BM25 Index ==========
    logger.info("=" * 60)
    logger.info("Step 4: Building BM25 Index")
    logger.info("=" * 60)
    
    bm25_cache = PIPELINE_CACHE_DIR / "bm25"
    bm25_index = BM25Index()
    
    if (bm25_cache / "bm25.pkl").exists():
        logger.info("Loading cached BM25 index...")
        bm25_index.load(bm25_cache)
    else:
        chunk_ids = chunk_loader.get_all_chunk_ids()
        bm25_index.build(chunk_ids, chunk_loader.bm25_texts)
        bm25_index.save(bm25_cache)
    
    # ========== Step 5: Build Neighbor Graph ==========
    logger.info("=" * 60)
    logger.info("Step 5: Building Neighbor Graph")
    logger.info("=" * 60)
    
    graph_cache = PIPELINE_CACHE_DIR / "graph"
    neighbor_graph = NeighborGraph()
    
    if (graph_cache / "neighbor_graph.pkl").exists():
        logger.info("Loading cached neighbor graph...")
        neighbor_graph.load(graph_cache)
    else:
        neighbor_graph.load_similarity_edges(SIMILARITY_EDGES_FILE)
        neighbor_graph.load_structural_links(STRUCTURAL_LINKS_FILE, set(chunk_loader.chunks.keys()))
        neighbor_graph.save(graph_cache)
    
    logger.info(f"Neighbor graph has {len(neighbor_graph.adjacency)} nodes")
    
    # ========== Step 6 & 7: Retrieval Pipeline with Neighbor Expansion ==========
    logger.info("=" * 60)
    logger.info("Step 6 & 7: Initializing Hybrid Retriever")
    logger.info("=" * 60)
    
    retriever = HybridRetriever(
        chunk_loader=chunk_loader,
        hnsw_index=hnsw_index,
        bm25_index=bm25_index,
        neighbor_graph=neighbor_graph
    )
    
    # ========== Step 8: Run Evaluation ==========
    logger.info("=" * 60)
    logger.info("Step 8: Running Evaluation on All Questions")
    logger.info("=" * 60)
    
    # Load questions
    logger.info(f"Loading questions from {QUESTIONS_FILE}...")
    questions_df = pd.read_csv(QUESTIONS_FILE)
    logger.info(f"Loaded {len(questions_df)} questions")
    
    # Run retrieval for each question
    results = []
    
    for idx, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Processing questions"):
        question = row['question']
        question_id = row.get('question_id', idx)
        gold_docs = row.get('gold_docs', '')
        
        try:
            # Retrieve top-20 chunks
            retrieved = retriever.retrieve(question, k=20)
            
            # Extract chunk IDs and scores
            retrieved_ids = [chunk_id for chunk_id, score in retrieved]
            retrieved_scores = [score for chunk_id, score in retrieved]
            
            results.append({
                'question_id': question_id,
                'question': question,
                'gold_docs': gold_docs,
                'retrieved_chunk_ids': json.dumps(retrieved_ids),
                'retrieved_scores': json.dumps(retrieved_scores),
                'num_retrieved': len(retrieved_ids)
            })
            
        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}")
            results.append({
                'question_id': question_id,
                'question': question,
                'gold_docs': gold_docs,
                'retrieved_chunk_ids': json.dumps([]),
                'retrieved_scores': json.dumps([]),
                'num_retrieved': 0
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_FILE, index=False)
    logger.info(f"Results saved to {RESULTS_FILE}")
    
    # Print summary statistics
    logger.info("=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)
    logger.info(f"Total questions processed: {len(results_df)}")
    logger.info(f"Average chunks retrieved: {results_df['num_retrieved'].mean():.2f}")
    logger.info(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()


