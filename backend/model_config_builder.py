"""
Dynamic model configuration builder that combines static base config with endpoint data.
"""
import logging
from typing import Dict, Any, List, Optional
from llm_manager import MAX_OUTPUT_TOKENS

logger = logging.getLogger(__name__)


# Minimal static configuration for models
# Only include what can't be fetched from endpoints
BASE_MODEL_CONFIG = {
    # Cloud models - minimal config, most data from endpoints
    "gpt-4o-mini": {
        "provider": "openai",
        "tokenizer_model": "gpt-4o",
        "context_window": 128000,
        "arena_score": 1274,
    },
    "gpt-4": {
        "provider": "openai",
        "tokenizer_model": "gpt-4",
        "context_window": 8192,
        "arena_score": 1250,
    },
    "gpt-3.5-turbo": {
        "provider": "openai",
        "tokenizer_model": "gpt-3.5-turbo",
        "context_window": 16385,
        "arena_score": 1068,
    },
    "claude-3-sonnet-20240229": {
        "provider": "anthropic",
        "tokenizer_model": "claude-3-sonnet-20240229",
        "context_window": 200000,
        "arena_score": 1268,
    },
    "claude-3-opus-20240229": {
        "provider": "anthropic",
        "tokenizer_model": "claude-3-opus-20240229",
        "context_window": 200000,
        "arena_score": 1300,
    },
    # On-prem models - need more static config
    "ollama-llama3": {
        "provider": "ollama",
        "tokenizer_model": "llama3",
        "context_window": 4096,
        "arena_score": None,
        "api_base": "http://localhost:11434",
    },
    "ollama-mistral": {
        "provider": "ollama",
        "tokenizer_model": "mistral",
        "context_window": 8192,
        "arena_score": None,
        "api_base": "http://localhost:11434",
    }
}

# Default values for all models
MODEL_DEFAULTS = {
    "temperature": 0,
    "max_output_tokens": MAX_OUTPUT_TOKENS,
    "is_cloud": True,  # Will be overridden for on-prem models
}


def infer_provider_from_model_code(model_code: str) -> str:
    """Infer the provider from the model code."""
    if model_code.startswith("gpt"):
        return "openai"
    elif model_code.startswith("claude"):
        return "anthropic"
    elif model_code.startswith("ollama"):
        return "ollama"
    else:
        return "unknown"


def build_dynamic_model_config(usage_data_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Build a complete model configuration by combining base config with endpoint data.
    
    Args:
        usage_data_list: List of usage data from the endpoint
        
    Returns:
        Complete model configuration dict
    """
    dynamic_config = {}
    
    # Process each model from the endpoint
    for usage_data in usage_data_list:
        model_code = usage_data.get("llmCode")
        if not model_code:
            logger.warning(f"Skipping usage data without llmCode: {usage_data}")
            continue
            
        # Start with base config if available, otherwise create new
        if model_code in BASE_MODEL_CONFIG:
            model_config = BASE_MODEL_CONFIG[model_code].copy()
        else:
            # For models not in base config, infer what we can
            model_config = {
                "provider": infer_provider_from_model_code(model_code),
                "tokenizer_model": model_code,  # Use model code as tokenizer model
            }
            logger.info(f"Creating dynamic config for new model: {model_code}")
        
        # Apply defaults
        for key, value in MODEL_DEFAULTS.items():
            if key not in model_config:
                model_config[key] = value
        
        # Add pricing data from endpoint
        model_config["pricing"] = {
            "input": usage_data.get("inputTokenRate", 0) * 1000,  # Convert to per 1K tokens
            "output": usage_data.get("outputTokenRate", 0) * 1000,  # Convert to per 1K tokens
        }
        
        # Add usage statistics (optional, for monitoring)
        model_config["usage_stats"] = {
            "daily_cost": usage_data.get("cost", 0),
            "daily_input_tokens": usage_data.get("inputTokens", 0),
            "daily_output_tokens": usage_data.get("outputTokens", 0),
            "daily_prompts": usage_data.get("promptCount", 0),
            "monthly_avg_cost": usage_data.get("averageCostForMonth", 0),
            "monthly_avg_input_tokens": usage_data.get("averageInputTokensForMonth", 0),
            "monthly_avg_output_tokens": usage_data.get("averageOutputTokensForMonth", 0),
        }
        
        # Override is_cloud for on-prem models
        if model_config["provider"] == "ollama" or model_config.get("api_base", "").startswith("http://localhost"):
            model_config["is_cloud"] = False
        
        dynamic_config[model_code] = model_config
    
    # Include any base models not in endpoint data (e.g., on-prem models)
    for model_code, base_config in BASE_MODEL_CONFIG.items():
        if model_code not in dynamic_config:
            config = base_config.copy()
            # Apply defaults
            for key, value in MODEL_DEFAULTS.items():
                if key not in config:
                    config[key] = value
            # Set zero pricing for on-prem models
            if config.get("provider") == "ollama":
                config["is_cloud"] = False
                config["pricing"] = {"input": 0, "output": 0}
            dynamic_config[model_code] = config
            logger.info(f"Added base config for model not in endpoint data: {model_code}")
    
    return dynamic_config


def get_minimal_static_config() -> Dict[str, Dict[str, Any]]:
    """
    Get minimal static configuration for initial LLMManager setup.
    This is used before endpoint data is available.
    """
    static_config = {}
    
    for model_code, base_config in BASE_MODEL_CONFIG.items():
        config = base_config.copy()
        # Apply defaults
        for key, value in MODEL_DEFAULTS.items():
            if key not in config:
                config[key] = value
        
        # Override is_cloud for on-prem models
        if config.get("provider") == "ollama":
            config["is_cloud"] = False
            
        static_config[model_code] = config
    
    return static_config