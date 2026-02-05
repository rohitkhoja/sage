# SAGE: Structure Aware Graph Expansion 

A framework for **graph-enhanced retrieval-augmented generation (RAG)** over multi-modal knowledge sources (documents and tables). SAGE constructs knowledge graphs from heterogeneous data using embedding similarity, HNSW indexing, and FAISS-accelerated search, then evaluates how graph-based neighbor expansion improves retrieval accuracy compared to flat vector search.

## Overview

Standard dense retrieval methods treat each document independently. SAGE instead builds a knowledge graph that captures semantic relationships between document chunks, table chunks, and cross-modal connections. At query time, the system retrieves initial candidates via embedding similarity, then expands the candidate set by traversing graph neighbors -- improving recall for complex questions that require information from multiple related sources.

The framework is evaluated across four benchmark datasets:

| Dataset | Domain | Source Types | Scale |
|---------|--------|--------------|-------|
| **OTT-QA** | Open-domain QA | Wikipedia documents + tables | ~5K questions |
| **STARKbench (Amazon)** | Product QA | Product metadata + reviews | ~2K questions |
| **MAG** | Academic | Papers + authors + citations | ~700K papers, ~1.1M authors |
| **PRIME** | Biomedical | Genes, diseases, drugs, pathways | Multi-entity |

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended; CPU fallback available)
- Neo4j 5.x (required only for the MAG agent)

### Installation

```bash
# Clone the repository
git clone https://github.com/rohitkhoja/sage.git
cd sage

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For GPU-accelerated FAISS (recommended):
# pip install faiss-gpu instead of faiss-cpu
```

### Configuration

1. Copy the environment template and fill in your credentials:

```bash
cp .env.example .env
```

2. Edit `.env` with your API keys:
   - `OPENAI_API_KEY` -- required for the LLM-based agent and question processing
   - `NEO4J_*` -- required only for STARK dataset

3. Review `config.yaml` for pipeline parameters (similarity thresholds, FAISS settings, GPU configuration).


## Key Components

### Embedding Service

Multi-GPU SentenceTransformer service with automatic batch sizing, OOM recovery, and caching. Configured via `config.yaml`.

### Graph Builder

Two implementations:
- **Standard** (`graph_builder.py`): O(N^2) pairwise similarity computation. Suitable for small datasets.
- **FAISS-accelerated** (`graph_builder_faiss.py`): Uses FAISS HNSW indices for approximate nearest neighbor search. Scales to millions of chunks.

### HNSW Index Builder

Builds per-feature HNSW indices (title embeddings, abstract embeddings, author name embeddings, etc.) for efficient multi-feature retrieval. Indices are stored as pickle files and loaded at query time.

### Retrieval Analysis

Compares retrieval accuracy (Hit@K, Recall@K, MRR) between:
- **Flat retrieval**: Top-K by embedding similarity
- **Graph-enhanced retrieval**: Top-K expanded with graph neighbor candidates, re-ranked by hybrid scoring (embedding similarity + BM25)

## License

MIT License. See [LICENSE](LICENSE) for details.
