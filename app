from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cache import QueryCache
from logger import log_query
import time

app = FastAPI(
    title="RAG API System",
    description="Retrieval-Augmented Generation API using sentence-transformers + FAISS",
    version="1.0.0"
)

_retriever = None
cache = QueryCache()

def get_retriever():
    global _retriever
    if _retriever is None:
        print("Initializing Retriever (this may take a moment)...")
        try:
            from retriever import Retriever
            from classifier import classify_query
            _retriever = Retriever()
            print("Retriever initialized successfully!")
        except Exception as e:
            print(f"Error initializing retriever: {e}")
            raise
    return _retriever


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


@app.get("/")
def root():
    return {
        "retriever.py": "Handles document retrieval using FAISS and sentence-transformers.",
        "cache.py": "Implements a simple in-memory cache with TTL and hit/miss tracking.",
        "logger.py": "Logs queries and their results to a JSON file for analysis.",
        "run.py": "Entry point to start the FastAPI server."
    }


@app.get("/health")
def health():
    retriever = get_retriever()
    return {"status": "ok", "chunks_loaded": len(retriever.chunks)}


@app.post("/query")
def query(request: QueryRequest):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query must not be empty")

        cache_key = request.query.strip().lower()
        cached = cache.get(cache_key)

        if cached:
            log_query(request.query, source="cache", results=cached["results"])
            return {"source": "cache", "data": cached}

        retriever = get_retriever()
        from classifier import classify_query
        
        start = time.time()
        query_type = classify_query(request.query)
        results = retriever.retrieve(request.query, top_k=request.top_k)
        elapsed = round(time.time() - start, 4)

        response_data = {
            "query": request.query,
            "type": query_type,
            "latency_s": elapsed,
            "results": results
        }

        cache.set(cache_key, response_data)
        log_query(request.query, source="fresh", results=results)

        return {"source": "fresh", "data": response_data}
    except Exception as e:
        print(f"ERROR in /query: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache/stats")
def cache_stats():
    return cache.stats()


@app.get("/cache/clear")
def cache_clear():
    cache.clear()
    return {"message": "Cache cleared"}


@app.get("/logs")
def get_logs(limit: int = 20):
    from logger import get_logs as fetch_logs
    return {"logs": fetch_logs(limit)}

