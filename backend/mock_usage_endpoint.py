"""
Mock endpoint for simulating LLM usage data endpoints.
This is for testing purposes only.
"""
from fastapi import FastAPI, Query
from datetime import date, datetime, timedelta
import random
from typing import List, Dict, Any, Optional
import uvicorn

app = FastAPI(title="Mock LLM Usage Endpoint")

# Mock model configurations with realistic data
MOCK_MODELS = {
    "gpt-4o-mini": {
        "inputTokenRate": 0.00015 / 1000,  # $0.15 per 1M tokens = $0.00015 per 1K tokens
        "outputTokenRate": 0.0006 / 1000,   # $0.60 per 1M tokens
        "baseUsage": {"input": 5000000, "output": 1000000, "prompts": 10000}
    },
    # "gpt-4": {
    #     "inputTokenRate": 0.03 / 1000,     # $30 per 1M tokens
    #     "outputTokenRate": 0.06 / 1000,    # $60 per 1M tokens
    #     "baseUsage": {"input": 500000, "output": 100000, "prompts": 1000}
    # },
    # "gpt-3.5-turbo": {
    #     "inputTokenRate": 0.0015 / 1000,   # $1.50 per 1M tokens
    #     "outputTokenRate": 0.002 / 1000,   # $2.00 per 1M tokens
    #     "baseUsage": {"input": 10000000, "output": 2000000, "prompts": 20000}
    # },
    # "claude-3-sonnet-20240229": {
    #     "inputTokenRate": 0.003 / 1000,    # $3 per 1M tokens
    #     "outputTokenRate": 0.015 / 1000,   # $15 per 1M tokens
    #     "baseUsage": {"input": 2000000, "output": 400000, "prompts": 5000}
    # },
    # "claude-3-opus-20240229": {
    #     "inputTokenRate": 0.015 / 1000,    # $15 per 1M tokens
    #     "outputTokenRate": 0.075 / 1000,   # $75 per 1M tokens
    #     "baseUsage": {"input": 300000, "output": 60000, "prompts": 500}
    # },
    # "ollama-llama3": {
    #     "inputTokenRate": 0.0,  # On-prem models have no per-token cost
    #     "outputTokenRate": 0.0,
    #     "baseUsage": {"input": 15000000, "output": 3000000, "prompts": 30000}
    # },
    # "ollama-mistral": {
    #     "inputTokenRate": 0.0,
    #     "outputTokenRate": 0.0,
    #     "baseUsage": {"input": 12000000, "output": 2400000, "prompts": 25000}
    # },
    # # Additional models that might be in production but not in base config
    # "gpt-4-turbo": {
    #     "inputTokenRate": 0.01 / 1000,     # $10 per 1M tokens
    #     "outputTokenRate": 0.03 / 1000,    # $30 per 1M tokens
    #     "baseUsage": {"input": 1000000, "output": 200000, "prompts": 2000}
    # },
    # "claude-3-haiku-20240307": {
    #     "inputTokenRate": 0.00025 / 1000,  # $0.25 per 1M tokens
    #     "outputTokenRate": 0.00125 / 1000, # $1.25 per 1M tokens
    #     "baseUsage": {"input": 8000000, "output": 1600000, "prompts": 15000}
    # }
}


def generate_usage_data(target_date: date) -> List[Dict[str, Any]]:
    """
    Generate mock usage data for all models for a specific date.
    
    Returns data in the format expected by the backend.
    """
    usage_data = []
    
    # Calculate days from start of month for monthly averages
    days_in_month = 30  # Simplified
    current_day_of_month = target_date.day
    
    for model_code, config in MOCK_MODELS.items():
        # Add some randomness to make data more realistic
        variance = random.uniform(0.8, 1.2)
        
        # Daily usage (with variance)
        daily_input_tokens = int(config["baseUsage"]["input"] * variance)
        daily_output_tokens = int(config["baseUsage"]["output"] * variance)
        daily_prompts = int(config["baseUsage"]["prompts"] * variance)
        
        # Calculate daily cost
        daily_cost = (
            daily_input_tokens * config["inputTokenRate"] +
            daily_output_tokens * config["outputTokenRate"]
        )
        
        # Monthly averages (accumulated over days)
        avg_cost_for_month = daily_cost * current_day_of_month / current_day_of_month
        avg_input_tokens_for_month = daily_input_tokens * current_day_of_month / current_day_of_month
        avg_output_tokens_for_month = daily_output_tokens * current_day_of_month / current_day_of_month
        avg_prompts_for_month = daily_prompts * current_day_of_month / current_day_of_month
        
        usage_data.append({
            "llmCode": model_code,
            "averageCostForMonth": round(avg_cost_for_month, 6),
            "averageInputTokensForMonth": int(avg_input_tokens_for_month),
            "averageOutputTokensForMonth": int(avg_output_tokens_for_month),
            "averagePromptCountForMonth": int(avg_prompts_for_month),
            "cost": round(daily_cost, 6),
            "inputTokens": daily_input_tokens,
            "outputTokens": daily_output_tokens,
            "inputTokenRate": config["inputTokenRate"],
            "outputTokenRate": config["outputTokenRate"],
            "promptCount": daily_prompts
        })
    
    return usage_data


@app.get("/api/llm-usage")
async def get_llm_usage(
    date: Optional[str] = Query(None, description="Date in ISO format (YYYY-MM-DD)")
) -> List[Dict[str, Any]]:
    """
    Get LLM usage data for a specific date.
    
    If no date is provided, returns data for today.
    """
    if date:
        try:
            target_date = datetime.fromisoformat(date).date()
        except ValueError:
            target_date = datetime.now().date()
    else:
        target_date = datetime.now().date()
    
    return generate_usage_data(target_date)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    # Run on port 8001 to avoid conflict with main app
    uvicorn.run(app, host="0.0.0.0", port=8001)