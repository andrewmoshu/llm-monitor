from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel

class LatencyRecordBase(BaseModel):
    model_name: str
    latency_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost: Optional[float] = None
    arena_score: Optional[float] = None
    context_window: Optional[int] = None
    is_cloud: Optional[bool] = None
    status: Optional[str] = None

class LatencyRecordCreate(LatencyRecordBase):
    pass

class LatencyRecord(LatencyRecordBase):
    id: int
    timestamp: datetime
    status: str

    class Config:
        orm_mode = True

class ModelInfo(BaseModel):
    # Add any necessary fields for the ModelInfo schema
    pass 