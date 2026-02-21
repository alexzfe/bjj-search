#!/usr/bin/env python3
"""
BJJ Transcript Search - FastAPI Backend

Serves the API and static frontend for BJJ instructional video search.

Usage:
    python server.py
"""

import os
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from bjj_rag_v2 import BJJSearchRAGv2, classify_intent, INTENT_DESCRIPTIONS

load_dotenv()

profile = os.environ.get("PROFILE", "laptop")
db_path = "./data/bjj_search_db_openai" if profile == "homeserver" else "./data/bjj_search_db"

print("Initializing BJJ Search...")
rag = BJJSearchRAGv2(
    db_path=db_path,
    llm_model="llama3.1:8b",
    use_reranker=False,
    profile=profile,
)

app = FastAPI(title="BJJ Transcript Search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str


class SearchResult(BaseModel):
    video_title: str
    timestamp: str
    end_time: str
    relevance_score: int
    text: str


class SearchResponse(BaseModel):
    intent: str
    intent_description: str
    summary: str
    total_results: int
    instructor_count: int
    results: dict[str, list[SearchResult]]


class ErrorResponse(BaseModel):
    error: str


@app.post("/api/search")
async def search(req: SearchRequest):
    query = req.query.strip()
    if len(query) < 3:
        return ErrorResponse(error="Query must be at least 3 characters.")

    try:
        intent = classify_intent(query, rag.llm)
        intent_desc = INTENT_DESCRIPTIONS.get(intent, "understand this technique")
        summary, grouped = rag.search(query)

        total = sum(len(v) for v in grouped.values())

        return SearchResponse(
            intent=intent,
            intent_description=intent_desc,
            summary=summary,
            total_results=total,
            instructor_count=len(grouped),
            results=grouped,
        )
    except Exception as e:
        traceback.print_exc()
        return ErrorResponse(error=f"Search failed: {str(e)}")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
