from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import json
from datetime import datetime
import os
import logging
# from sqlalchemy.orm import Session
# from backend.database import get_db
# from backend.models import LatencyRecord
# from backend.schemas import LatencyRecord as LatencyRecordSchema
# from backend.models import LatencyRecord as LatencyRecordModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS with more specific configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

@app.get("/latency")
@app.get("/api/latency")
async def get_latency_data():
    return latency_data

@app.post("/latency")
@app.post("/api/latency")
async def add_latency_data(record: LatencyRecord):
    try:
        data = record.dict()
        logger.info(f"Received latency record: {data}")
        latency_data.append(data)
        # Keep only last 1000 records
        if len(latency_data) > 1000:
            latency_data.pop(0)
        return {"status": "success", "message": "Record added successfully"}
    except Exception as e:
        logger.error(f"Error processing latency record: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
@app.get("/api/models")
async def get_models():
    # Get unique model names from latency data
    models = set(record["model_name"] for record in latency_data)
    return list(models)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 