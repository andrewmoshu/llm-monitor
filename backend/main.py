from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import json
from datetime import datetime
import os
# from sqlalchemy.orm import Session
# from backend.database import get_db
# from backend.models import LatencyRecord
# from backend.schemas import LatencyRecord as LatencyRecordSchema
# from backend.models import LatencyRecord as LatencyRecordModel

app = FastAPI()

# Enable CORS with more specific configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",           # Development
        "http://llm-monitor.local",        # Production domain
        "https://llm-monitor.local"        # If using HTTPS
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for latency data (in production, use a database)
latency_data = []

class LatencyRecord(BaseModel):
    model_name: str
    timestamp: datetime
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost: float
    arena_score: float | None = None
    context_window: int
    is_cloud: bool

@app.get("/")
async def read_root():
    return {"status": "LLM Latency Monitor is running"}

@app.get("/api/latency")
async def get_latency_data():
    return latency_data

@app.post("/api/latency")
async def add_latency_data(record: LatencyRecord):
    data = record.dict()
    print("Received record:", data)  # Debug logging
    latency_data.append(data)
    # Keep only last 1000 records
    if len(latency_data) > 1000:
        latency_data.pop(0)
    return {"status": "success"}

@app.get("/api/models")
async def get_models():
    # Get unique model names from latency data
    models = set(record["model_name"] for record in latency_data)
    return list(models)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 