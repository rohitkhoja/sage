#!/usr/bin/env python3
"""
OTT-QA API Server

Provides REST API endpoints for:
- HNSW semantic search
- BM25 keyword search  
- Hybrid search (HNSW + BM25)
- Graph neighbor lookup
- Chunk content retrieval

This server loads pre-built indices from the pipeline cache.
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict
from flask import Flask, request, jsonify
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path("/shared/khoja/CogComp")
OUTPUT_DIR = BASE_DIR / "output"
PIPELINE_CACHE_DIR = OUTPUT_DIR / "ott_qa_pipeline_cache"
DOCS_CHUNKS_DIR = OUTPUT_DIR / "full_pipeline" / "docs_chunks_1"
TABLE_CHUNKS_FILE = OUTPUT_DIR / "full_pipeline" / "table_chunks_with_metadata.json"

app = Flask(__name__)

# Global state
chunks_data: Dict[str, dict] = {}  # chunk_id -> chunk data
embeddings: Dict[str, np.ndarray] = {}  # chunk_id -> embedding
bm25_texts: Dict[str, str] = {}  # chunk_id -> BM25 text
hnsw_index = None
bm25_index = None
neighbor_graph: Dict[str, Set[str]] = {}
chunk_ids_list: List[str] = []
embedding_model = None


def load_chunks():
    """Load all chunks (document + table)"""
    global chunks_data, embeddings, bm25_texts
    
    logger.info("Loading document chunks...")
    json_files = list(DOCS_CHUNKS_DIR.glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                chunk = json.load(f)
            
            chunk_id = chunk.get('chunk_id')
            if not chunk_id:
                continue
                
            chunks_data[chunk_id] = chunk
            
            if 'embedding' in chunk:
                embeddings[chunk_id] = np.array(chunk['embedding'], dtype=np.float32)
            
            source_name = ""
            if 'source_info' in chunk and 'source_name' in chunk['source_info']:
                source_name = chunk['source_info']['source_name']
            content = chunk.get('content', '')
            bm25_texts[chunk_id] = f"{source_name} | {content}" if source_name else content
            
        except Exception as e:
            logger.warning(f"Error loading {json_file}: {e}")
    
    logger.info(f"Loaded {len(chunks_data)} document chunks")
    
    # Load table chunks
    logger.info("Loading table chunks...")
    with open(TABLE_CHUNKS_FILE, 'r') as f:
        table_chunks = json.load(f)
    
    for chunk in table_chunks:
        chunk_id = chunk.get('chunk_id')
        if not chunk_id:
            continue
            
        chunks_data[chunk_id] = chunk
        
        source_name = chunk.get('source_name', '')
        content = chunk.get('content', '')
        bm25_texts[chunk_id] = f"{source_name} | {content}" if source_name else content
    
    logger.info(f"Total chunks loaded: {len(chunks_data)}")


def load_hnsw_index():
    """Load HNSW index from cache"""
    global hnsw_index, chunk_ids_list
    import faiss
    
    hnsw_path = PIPELINE_CACHE_DIR / "hnsw"
    
    if not (hnsw_path / "hnsw.index").exists():
        logger.error("HNSW index not found!")
        return False
    
    logger.info("Loading HNSW index...")
    hnsw_index = faiss.read_index(str(hnsw_path / "hnsw.index"))
    
    with open(hnsw_path / "chunk_ids.pkl", 'rb') as f:
        chunk_ids_list = pickle.load(f)
    
    logger.info(f"HNSW index loaded with {hnsw_index.ntotal} vectors")
    return True


def load_bm25_index():
    """Load BM25 index from cache"""
    global bm25_index
    
    bm25_path = PIPELINE_CACHE_DIR / "bm25"
    
    if not (bm25_path / "bm25.pkl").exists():
        logger.error("BM25 index not found!")
        return False
    
    logger.info("Loading BM25 index...")
    with open(bm25_path / "bm25.pkl", 'rb') as f:
        data = pickle.load(f)
        bm25_index = data
    
    logger.info("BM25 index loaded")
    return True


def load_neighbor_graph():
    """Load neighbor graph from cache"""
    global neighbor_graph
    
    graph_path = PIPELINE_CACHE_DIR / "graph"
    
    if not (graph_path / "neighbor_graph.pkl").exists():
        logger.error("Neighbor graph not found!")
        return False
    
    logger.info("Loading neighbor graph...")
    with open(graph_path / "neighbor_graph.pkl", 'rb') as f:
        raw_graph = pickle.load(f)
        neighbor_graph = {k: set(v) for k, v in raw_graph.items()}
    
    logger.info(f"Neighbor graph loaded with {len(neighbor_graph)} nodes")
    return True


def get_embedding_model():
    """Load embedding model lazily"""
    global embedding_model
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return embedding_model


def embed_query(query: str) -> np.ndarray:
    """Embed a query string"""
    model = get_embedding_model()
    return model.encode(query, convert_to_numpy=True).astype(np.float32)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'chunks_loaded': len(chunks_data),
        'hnsw_ready': hnsw_index is not None,
        'bm25_ready': bm25_index is not None,
        'graph_ready': len(neighbor_graph) > 0
    })


@app.route('/hnsw_search', methods=['POST'])
def hnsw_search():
    """HNSW semantic search endpoint"""
    import faiss
    
    data = request.get_json()
    query = data.get('query', '')
    k = min(data.get('k', 50), 200)
    
    if not query:
        return jsonify([])
    
    try:
        # Embed query
        query_emb = embed_query(query).reshape(1, -1)
        faiss.normalize_L2(query_emb)
        
        # Search
        distances, indices = hnsw_index.search(query_emb, k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(chunk_ids_list):
                chunk_id = chunk_ids_list[idx]
                chunk = chunks_data.get(chunk_id, {})
                results.append({
                    'chunk_id': chunk_id,
                    'distance': float(dist),
                    'rank': i + 1,
                    'content': chunk.get('content', '')[:500],
                    'source_name': chunk.get('source_name', '') or chunk.get('source_info', {}).get('source_name', '')
                })
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"HNSW search error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/bm25_search', methods=['POST'])
def bm25_search():
    """BM25 keyword search endpoint"""
    import re
    
    data = request.get_json()
    query = data.get('query', '')
    k = min(data.get('k', 50), 200)
    
    if not query:
        return jsonify([])
    
    try:
        # Tokenize query
        query_tokens = re.findall(r'\w+', query.lower())
        
        # Get BM25 scores
        bm25_obj = bm25_index['bm25']
        bm25_chunk_ids = bm25_index['chunk_ids']
        
        scores = bm25_obj.get_scores(query_tokens)
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for i, idx in enumerate(top_indices):
            if scores[idx] > 0:
                chunk_id = bm25_chunk_ids[idx]
                chunk = chunks_data.get(chunk_id, {})
                results.append({
                    'chunk_id': chunk_id,
                    'score': float(scores[idx]),
                    'rank': i + 1,
                    'content': chunk.get('content', '')[:500],
                    'source_name': chunk.get('source_name', '') or chunk.get('source_info', {}).get('source_name', '')
                })
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"BM25 search error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/hybrid_search', methods=['POST'])
def hybrid_search():
    """Combined HNSW + BM25 search"""
    import faiss
    import re
    
    data = request.get_json()
    query = data.get('query', '')
    k = min(data.get('k', 50), 200)
    
    if not query:
        return jsonify([])
    
    try:
        # HNSW search
        query_emb = embed_query(query).reshape(1, -1)
        faiss.normalize_L2(query_emb)
        hnsw_distances, hnsw_indices = hnsw_index.search(query_emb, k * 2)
        
        hnsw_scores = {}
        for dist, idx in zip(hnsw_distances[0], hnsw_indices[0]):
            if idx >= 0 and idx < len(chunk_ids_list):
                chunk_id = chunk_ids_list[idx]
                hnsw_scores[chunk_id] = 1.0 / (1.0 + float(dist))
        
        # BM25 search
        query_tokens = re.findall(r'\w+', query.lower())
        bm25_obj = bm25_index['bm25']
        bm25_chunk_ids = bm25_index['chunk_ids']
        scores = bm25_obj.get_scores(query_tokens)
        
        # Normalize BM25 scores
        max_bm25 = max(scores) if len(scores) > 0 else 1.0
        bm25_scores = {}
        for idx, score in enumerate(scores):
            if score > 0:
                chunk_id = bm25_chunk_ids[idx]
                bm25_scores[chunk_id] = score / max_bm25 if max_bm25 > 0 else 0
        
        # Combine scores
        all_chunks = set(hnsw_scores.keys()) | set(bm25_scores.keys())
        combined = []
        
        for chunk_id in all_chunks:
            h_score = hnsw_scores.get(chunk_id, 0)
            b_score = bm25_scores.get(chunk_id, 0)
            
            # Boost if in both
            boost = 1.5 if chunk_id in hnsw_scores and chunk_id in bm25_scores else 1.0
            combined_score = boost * (0.5 * h_score + 0.5 * b_score)
            
            combined.append((chunk_id, combined_score, h_score, b_score))
        
        # Sort by combined score
        combined.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i, (chunk_id, c_score, h_score, b_score) in enumerate(combined[:k]):
            chunk = chunks_data.get(chunk_id, {})
            results.append({
                'chunk_id': chunk_id,
                'combined_score': c_score,
                'hnsw_score': h_score,
                'bm25_score': b_score,
                'rank': i + 1,
                'content': chunk.get('content', '')[:500],
                'source_name': chunk.get('source_name', '') or chunk.get('source_info', {}).get('source_name', '')
            })
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/get_neighbors', methods=['POST'])
def get_neighbors():
    """Get 1-hop neighbors of a chunk"""
    data = request.get_json()
    chunk_id = data.get('chunk_id', '')
    
    if not chunk_id:
        return jsonify([])
    
    neighbors = list(neighbor_graph.get(chunk_id, set()))
    return jsonify(neighbors)


@app.route('/get_chunks', methods=['POST'])
def get_chunks():
    """Get full content of specific chunks"""
    data = request.get_json()
    chunk_ids = data.get('chunk_ids', [])
    
    if not chunk_ids:
        return jsonify([])
    
    results = []
    for chunk_id in chunk_ids:
        chunk = chunks_data.get(chunk_id)
        if chunk:
            results.append({
                'chunk_id': chunk_id,
                'content': chunk.get('content', ''),
                'source_name': chunk.get('source_name', '') or chunk.get('source_info', {}).get('source_name', '')
            })
    
    return jsonify(results)


def initialize():
    """Initialize all components"""
    logger.info("=" * 60)
    logger.info("Initializing OTT-QA API Server")
    logger.info("=" * 60)
    
    load_chunks()
    
    if not load_hnsw_index():
        logger.error("Failed to load HNSW index!")
        return False
    
    if not load_bm25_index():
        logger.error("Failed to load BM25 index!")
        return False
    
    if not load_neighbor_graph():
        logger.error("Failed to load neighbor graph!")
        return False
    
    logger.info("=" * 60)
    logger.info("Server ready!")
    logger.info("=" * 60)
    return True


if __name__ == '__main__':
    if initialize():
        app.run(host='0.0.0.0', port=8082, threaded=True)
    else:
        logger.error("Failed to initialize server!")
        exit(1)

