"""
Configuration for multi-environment monitoring.
Set MULTI_ENV_MODE to control which implementation to use.
"""
import os

# Multi-environment mode configuration
# Options: "simple" or "full"
# - "simple": Uses current endpoint for all environments with simulated variations
# - "full": Uses separate endpoints and managers for each environment
MULTI_ENV_MODE = os.getenv("MULTI_ENV_MODE", "simple")

# Environment configuration
ENVIRONMENTS = ["dev", "test", "qa", "prod"]

# Simulated environment variations (for simple mode)
# These percentages are added to the base measurements
ENVIRONMENT_VARIATIONS = {
    "dev": 0.0,    # No variation (baseline)
    "test": 0.05,  # 5% increase
    "qa": 0.10,    # 10% increase  
    "prod": 0.15   # 15% increase
}

def is_multi_env_enabled():
    """Check if multi-environment monitoring is enabled."""
    return MULTI_ENV_MODE in ["simple", "full"]

def is_full_multi_env():
    """Check if full multi-environment mode is enabled."""
    return MULTI_ENV_MODE == "full"