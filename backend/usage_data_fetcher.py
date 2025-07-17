"""
Module for fetching LLM usage and cost data from environment-specific endpoints.
"""
import os
import aiohttp
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, date
from enum import Enum

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Available environments"""
    DEV = "dev"
    TEST = "test"
    QA = "qa"
    PROD = "prod"


class UsageDataFetcher:
    """Fetches LLM usage data from environment-specific endpoints"""
    
    def __init__(self, environment: Optional[str] = None, timeout_seconds: int = 30):
        """
        Initialize the usage data fetcher.
        
        Args:
            environment: Environment name (dev, test, qa, prod). If None, reads from env var.
            timeout_seconds: Timeout for API requests
        """
        env_str = environment or os.getenv("ENVIRONMENT", "dev")
        try:
            self.environment = Environment(env_str.lower())
        except ValueError:
            logger.warning(f"Invalid environment '{env_str}', defaulting to 'dev'")
            self.environment = Environment.DEV
            
        self.timeout_seconds = timeout_seconds
        
        # Define endpoint URLs for each environment
        self.endpoints = {
            Environment.DEV: os.getenv("DEV_USAGE_ENDPOINT", "http://localhost:8001/api/llm-usage"),
            Environment.TEST: os.getenv("TEST_USAGE_ENDPOINT", "http://test-endpoint.example.com/api/llm-usage"),
            Environment.QA: os.getenv("QA_USAGE_ENDPOINT", "http://qa-endpoint.example.com/api/llm-usage"),
            Environment.PROD: os.getenv("PROD_USAGE_ENDPOINT", "http://prod-endpoint.example.com/api/llm-usage")
        }
        
        logger.info(f"UsageDataFetcher initialized for environment: {self.environment.value}")
        
    @property
    def endpoint_url(self) -> str:
        """Get the endpoint URL for the current environment"""
        return self.endpoints[self.environment]
    
    async def fetch_usage_data(self, target_date: Optional[date] = None) -> List[Dict[str, Any]]:
        """
        Fetch usage data from the environment-specific endpoint.
        
        Args:
            target_date: Date to fetch data for. If None, uses today.
            
        Returns:
            List of usage data dictionaries with the following structure:
            [{
                "llmCode": "openai-4o...",
                "averageCostForMonth": 14.074934,
                "averageInputTokensForMonth": 4234324324,
                "averageOutputTokensForMonth": 234243243,
                "averagePromptCountForMonth": 32434,
                "cost": 1.232434,  # Total cost for the day
                "inputTokens": 234324234,
                "outputTokens": 232443,
                "inputTokenRate": 0.000343,  # Cost per input token in $
                "outputTokenRate": 0.000024234,  # Cost per output token in $
                "promptCount": 234
            }]
        """
        if target_date is None:
            target_date = date.today()
            
        url = self.endpoint_url
        params = {"date": target_date.isoformat()}
        
        logger.info(f"Fetching usage data from {url} for date {target_date}")
        
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Successfully fetched usage data for {len(data)} models")
                        return data
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to fetch usage data. Status: {response.status}, Error: {error_text}")
                        return []
                        
        except aiohttp.ClientTimeout:
            logger.error(f"Timeout while fetching usage data from {url}")
            return []
        except aiohttp.ClientError as e:
            logger.error(f"Client error while fetching usage data: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error while fetching usage data: {str(e)}", exc_info=True)
            return []
    
    def calculate_daily_cost(self, usage_data: Dict[str, Any]) -> float:
        """
        Calculate the daily cost from usage data.
        
        The cost should already be calculated in the response, but this method
        verifies it and can recalculate if needed.
        
        Args:
            usage_data: Usage data dictionary from the endpoint
            
        Returns:
            Total cost for the day in dollars
        """
        # If cost is already provided, use it
        if "cost" in usage_data and usage_data["cost"] is not None:
            return float(usage_data["cost"])
        
        # Otherwise calculate from tokens and rates
        try:
            input_tokens = usage_data.get("inputTokens", 0)
            output_tokens = usage_data.get("outputTokens", 0)
            input_rate = usage_data.get("inputTokenRate", 0)
            output_rate = usage_data.get("outputTokenRate", 0)
            
            total_cost = (input_tokens * input_rate) + (output_tokens * output_rate)
            return total_cost
            
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Error calculating cost for {usage_data.get('llmCode', 'unknown')}: {e}")
            return 0.0
    
    def get_model_usage_map(self, usage_data_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Convert list of usage data to a map keyed by model code.
        
        Args:
            usage_data_list: List of usage data from the endpoint
            
        Returns:
            Dictionary mapping model code to usage data
        """
        usage_map = {}
        for data in usage_data_list:
            model_code = data.get("llmCode")
            if model_code:
                usage_map[model_code] = data
            else:
                logger.warning(f"Usage data missing llmCode: {data}")
                
        return usage_map