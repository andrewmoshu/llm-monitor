from typing import Optional
from datetime import datetime
from pydantic import BaseModel

class LatencyRecord(BaseModel):
    model_name: str
    timestamp: datetime
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost: float
    arena_score: Optional[float] = None
    context_window: int

    class Config:
        from_attributes = True 