#!/usr/bin/env python3
"""
FastAPI Server for MAG Agent
Provides REST API endpoints for querying the MAG dataset
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Union
import time
import json
from pathlib import Path
from loguru import logger

from mag_agent import MAGAgent


# Request/Response models
class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    feature: Optional[str] = None

class QueryPlanRequest(BaseModel):
    query_plan: List[Dict[str, Any]]
    question_id: Optional[str] = None

class NaturalLanguageRequest(BaseModel):
    query: str
    question_id: Optional[str] = None

class YearRangeRequest(BaseModel):
    start_year: int
    end_year: int

class MetadataRequest(BaseModel):
    object_id: int
    object_type: str # 'paper', 'author', 'institution', 'field_of_study'

class Response(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    count: Optional[int] = None

class StatusResponse(BaseModel):
    status: str
    is_loaded: bool
    load_time: float
    available_features: List[str]
    graph_stats: Dict[str, Any]
    hnsw_stats: Dict[str, Any]
    agent_stats: Dict[str, Any]

# Initialize FastAPI app
app = FastAPI(
    title="MAG Agent API",
    description="REST API for querying Microsoft Academic Graph dataset",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent: Optional[MAGAgent] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the MAG Agent on startup"""
    global agent
    
    logger.info(" Starting MAG Agent API Server...")
    
    try:
        agent = MAGAgent(
            processed_dir="/shared/khoja/CogComp/datasets/MAG/processed",
            indices_dir="/shared/khoja/CogComp/output/mag_hnsw_indices"
        )
        
        logger.info(" Loading MAG Agent components...")
        if agent.load_all():
            logger.info(" MAG Agent loaded successfully!")
        else:
            logger.error(" Failed to load MAG Agent")
            
    except Exception as e:
        logger.error(f" Startup failed: {e}")
        import traceback
        traceback.print_exc()

@app.get("/", response_model=Response)
async def root():
    """Root endpoint"""
    return Response(
        success=True,
        data={
            "message": "MAG Agent API Server",
            "version": "1.0.0",
            "endpoints": [
                "/docs - API documentation",
                "/status - System status",
                "/search/title - Search papers by title",
                "/search/abstract - Search papers by abstract",
                "/search/content - Search papers by content",
                "/search/authors - Search authors by name",
                "/query/plan - Execute query plan",
                "/query/natural - Execute natural language query",
                "/graph/traverse - Graph traversal functions",
                "/metadata - Get object metadata"
            ]
        }
    )

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status and statistics"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        return StatusResponse(
            status="running" if agent.is_loaded else "loading",
            is_loaded=agent.is_loaded,
            load_time=agent.load_time,
            available_features=agent.get_available_features(),
            graph_stats=agent.get_graph_stats(),
            hnsw_stats=agent.get_hnsw_stats(),
            agent_stats=agent.get_agent_stats()
        )
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/title", response_model=Response)
async def search_papers_by_title(request: SearchRequest):
    """Search papers by title similarity"""
    if agent is None or not agent.is_loaded:
        raise HTTPException(status_code=503, detail="Agent not loaded")
    
    start_time = time.time()
    
    try:
        results = agent.search_papers_by_title(request.query, request.top_k)
        
        return Response(
            success=True,
            data=results,
            count=len(results),
            execution_time=time.time() - start_time
        )
    except Exception as e:
        logger.error(f"Error in title search: {e}")
        return Response(
            success=False,
            error=str(e),
            execution_time=time.time() - start_time
        )

@app.post("/search/abstract", response_model=Response)
async def search_papers_by_abstract(request: SearchRequest):
    """Search papers by abstract similarity"""
    if agent is None or not agent.is_loaded:
        raise HTTPException(status_code=503, detail="Agent not loaded")
    
    start_time = time.time()
    
    try:
        results = agent.search_papers_by_abstract(request.query, request.top_k)
        
        return Response(
            success=True,
            data=results,
            count=len(results),
            execution_time=time.time() - start_time
        )
    except Exception as e:
        logger.error(f"Error in abstract search: {e}")
        return Response(
            success=False,
            error=str(e),
            execution_time=time.time() - start_time
        )

@app.post("/search/content", response_model=Response)
async def search_papers_by_content(request: SearchRequest):
    """Search papers by content similarity"""
    if agent is None or not agent.is_loaded:
        raise HTTPException(status_code=503, detail="Agent not loaded")
    
    start_time = time.time()
    
    try:
        results = agent.search_papers_by_content(request.query, request.top_k)
        
        return Response(
            success=True,
            data=results,
            count=len(results),
            execution_time=time.time() - start_time
        )
    except Exception as e:
        logger.error(f"Error in content search: {e}")
        return Response(
            success=False,
            error=str(e),
            execution_time=time.time() - start_time
        )

@app.post("/search/authors", response_model=Response)
async def search_authors_by_name(request: SearchRequest):
    """Search authors by name similarity"""
    if agent is None or not agent.is_loaded:
        raise HTTPException(status_code=503, detail="Agent not loaded")
    
    start_time = time.time()
    
    try:
        results = agent.search_authors_by_name(request.query, request.top_k)
        
        return Response(
            success=True,
            data=results,
            count=len(results),
            execution_time=time.time() - start_time
        )
    except Exception as e:
        logger.error(f"Error in author search: {e}")
        return Response(
            success=False,
            error=str(e),
            execution_time=time.time() - start_time
        )

@app.post("/query/plan", response_model=Response)
async def execute_query_plan(request: QueryPlanRequest):
    """Execute a multi-step query plan"""
    if agent is None or not agent.is_loaded:
        raise HTTPException(status_code=503, detail="Agent not loaded")
    
    try:
        result = agent.execute_query_plan(request.query_plan)
        
        # Save evidence if question_id provided
        if request.question_id:
            agent.save_query_evidence(request.question_id, result)
        
        return Response(
            success=result.get('success', False),
            data=result,
            execution_time=result.get('execution_time_seconds', 0.0),
            count=result.get('final_count', 0)
        )
    except Exception as e:
        logger.error(f"Error executing query plan: {e}")
        return Response(
            success=False,
            error=str(e)
        )

@app.post("/query/natural", response_model=Response)
async def execute_natural_language_query(request: NaturalLanguageRequest):
    """Execute a natural language query"""
    if agent is None or not agent.is_loaded:
        raise HTTPException(status_code=503, detail="Agent not loaded")
    
    try:
        result = agent.query_natural_language(request.query)
        
        # Save evidence if question_id provided
        if request.question_id:
            agent.save_query_evidence(request.question_id, result)
        
        return Response(
            success=result.get('success', False),
            data=result,
            execution_time=result.get('execution_time_seconds', 0.0),
            count=result.get('final_count', 0)
        )
    except Exception as e:
        logger.error(f"Error executing natural language query: {e}")
        return Response(
            success=False,
            error=str(e)
        )

# Graph traversal endpoints
@app.get("/graph/authors_of_paper/{paper_id}", response_model=Response)
async def get_authors_of_paper(paper_id: int):
    """Get all authors of a paper"""
    if agent is None or not agent.is_loaded:
        raise HTTPException(status_code=503, detail="Agent not loaded")
    
    start_time = time.time()
    
    try:
        authors = agent.get_authors_of_paper(paper_id)
        
        return Response(
            success=True,
            data=authors,
            count=len(authors),
            execution_time=time.time() - start_time
        )
    except Exception as e:
        logger.error(f"Error getting authors of paper: {e}")
        return Response(
            success=False,
            error=str(e),
            execution_time=time.time() - start_time
        )

@app.post("/graph/papers_by_author", response_model=Response)
async def get_papers_by_author(author_ids: List[int]):
    """Get all papers written by given authors"""
    if agent is None or not agent.is_loaded:
        raise HTTPException(status_code=503, detail="Agent not loaded")
    
    start_time = time.time()
    
    try:
        papers = agent.get_papers_by_author(author_ids)
        
        return Response(
            success=True,
            data=papers,
            count=len(papers),
            execution_time=time.time() - start_time
        )
    except Exception as e:
        logger.error(f"Error getting papers by author: {e}")
        return Response(
            success=False,
            error=str(e),
            execution_time=time.time() - start_time
        )

@app.get("/graph/papers_by_year/{start_year}/{end_year}", response_model=Response)
async def get_papers_by_year_range(start_year: int, end_year: int):
    """Get papers published in a year range"""
    if agent is None or not agent.is_loaded:
        raise HTTPException(status_code=503, detail="Agent not loaded")
    
    start_time = time.time()
    
    try:
        papers = agent.get_papers_by_year_range(start_year, end_year)
        
        return Response(
            success=True,
            data=papers,
            count=len(papers),
            execution_time=time.time() - start_time
        )
    except Exception as e:
        logger.error(f"Error getting papers by year range: {e}")
        return Response(
            success=False,
            error=str(e),
            execution_time=time.time() - start_time
        )

@app.get("/graph/papers_citing/{paper_id}", response_model=Response)
async def get_papers_citing(paper_id: int):
    """Get papers that cite a specific paper"""
    if agent is None or not agent.is_loaded:
        raise HTTPException(status_code=503, detail="Agent not loaded")
    
    start_time = time.time()
    
    try:
        papers = agent.get_papers_citing(paper_id)
        
        return Response(
            success=True,
            data=papers,
            count=len(papers),
            execution_time=time.time() - start_time
        )
    except Exception as e:
        logger.error(f"Error getting papers citing: {e}")
        return Response(
            success=False,
            error=str(e),
            execution_time=time.time() - start_time
        )

@app.get("/graph/papers_cited_by/{paper_id}", response_model=Response)
async def get_papers_cited_by(paper_id: int):
    """Get papers cited by a specific paper"""
    if agent is None or not agent.is_loaded:
        raise HTTPException(status_code=503, detail="Agent not loaded")
    
    start_time = time.time()
    
    try:
        papers = agent.get_papers_cited_by(paper_id)
        
        return Response(
            success=True,
            data=papers,
            count=len(papers),
            execution_time=time.time() - start_time
        )
    except Exception as e:
        logger.error(f"Error getting papers cited by: {e}")
        return Response(
            success=False,
            error=str(e),
            execution_time=time.time() - start_time
        )

@app.post("/metadata", response_model=Response)
async def get_metadata(request: MetadataRequest):
    """Get metadata for an object"""
    if agent is None or not agent.is_loaded:
        raise HTTPException(status_code=503, detail="Agent not loaded")
    
    start_time = time.time()
    
    try:
        if request.object_type == 'paper':
            metadata = agent.get_paper_metadata(request.object_id)
        elif request.object_type == 'author':
            metadata = agent.get_author_metadata(request.object_id)
        elif request.object_type == 'institution':
            metadata = agent.get_institution_metadata(request.object_id)
        elif request.object_type == 'field_of_study':
            metadata = agent.get_field_metadata(request.object_id)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown object type: {request.object_type}")
        
        if metadata is None:
            raise HTTPException(status_code=404, detail=f"Object {request.object_id} not found")
        
        return Response(
            success=True,
            data=metadata,
            execution_time=time.time() - start_time
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metadata: {e}")
        return Response(
            success=False,
            error=str(e),
            execution_time=time.time() - start_time
        )

@app.get("/evidence/{question_id}", response_model=Response)
async def get_query_evidence(question_id: str):
    """Get saved query evidence"""
    try:
        evidence_file = Path("/shared/khoja/CogComp/agent/output/qa") / f"{question_id}.json"
        
        if not evidence_file.exists():
            raise HTTPException(status_code=404, detail=f"Evidence for question {question_id} not found")
        
        with open(evidence_file, 'r') as f:
            evidence = json.load(f)
        
        return Response(
            success=True,
            data=evidence
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting evidence: {e}")
        return Response(
            success=False,
            error=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    
    logger.info(" Starting MAG Agent API Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
