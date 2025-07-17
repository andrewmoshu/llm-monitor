from typing import Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel

Environment = Literal["dev", "test", "qa", "prod"]

class LatencyRecordBase(BaseModel):
    model_name: str
    environment: Environment = "dev"
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

class UsageStats(BaseModel):
    daily_cost: float
    daily_input_tokens: int
    daily_output_tokens: int
    daily_prompts: int
    monthly_avg_cost: float
    monthly_avg_input_tokens: int
    monthly_avg_output_tokens: int

class ModelInfo(BaseModel):
    provider: str
    context_window: Optional[int] = None
    pricing: Optional[Dict[str, float]] = None  # {'input': x, 'output': y} per 1K tokens
    is_cloud: Optional[bool] = None
    arena_score: Optional[int] = None
    usage_stats: Optional[UsageStats] = None 