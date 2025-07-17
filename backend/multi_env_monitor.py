"""
Multi-environment monitoring system.
This module handles monitoring across all environments (dev, test, qa, prod).
"""
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone
import os
from concurrent.futures import ThreadPoolExecutor

from llm_manager import LLMManager
from usage_data_fetcher import UsageDataFetcher, Environment
from model_config_builder import build_dynamic_model_config, get_minimal_static_config

logger = logging.getLogger(__name__)

# Test prompt for latency measurements
TEST_PROMPT = """You are tasked with explaining a complex topic in a clear and concise manner.
Please explain the concept of quantum computing, focusing on the following aspects:

1. Basic principles of quantum computing
2. Key differences from classical computing
3. Current applications and limitations
4. Future potential and challenges

Please provide a comprehensive yet accessible explanation suitable for someone with a basic understanding of computer science but no prior knowledge of quantum mechanics.
Your explanation should be structured, using clear examples and analogies where appropriate."""


class MultiEnvironmentMonitor:
    """Monitors latency across multiple environments."""
    
    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds
        self.environments = ["dev", "test", "qa", "prod"]
        self.llm_managers: Dict[str, LLMManager] = {}
        self.usage_fetchers: Dict[str, UsageDataFetcher] = {}
        self.model_configs: Dict[str, dict] = {}
        
    async def initialize(self):
        """Initialize managers for all environments."""
        for env in self.environments:
            try:
                logger.info(f"Initializing environment: {env}")
                
                # For testing, reuse dev configuration for all environments
                # In production, each environment would have its own endpoint
                if env == "dev":
                    usage_fetcher = UsageDataFetcher(
                        environment=env,
                        timeout_seconds=self.timeout_seconds
                    )
                else:
                    # For test/qa/prod, temporarily use dev fetcher
                    # In production, these would have separate endpoints
                    usage_fetcher = UsageDataFetcher(
                        environment="dev",  # Temporary: use dev for all
                        timeout_seconds=self.timeout_seconds
                    )
                
                # Fetch usage data to build dynamic configuration
                usage_data_list = await usage_fetcher.fetch_usage_data()
                
                if usage_data_list:
                    model_config = build_dynamic_model_config(usage_data_list)
                    logger.info(f"Built dynamic configuration for {env} with {len(model_config)} models")
                else:
                    logger.warning(f"No usage data for {env}, using minimal static configuration")
                    model_config = get_minimal_static_config()
                
                # Initialize LLM Manager for this environment
                llm_manager = LLMManager(
                    model_configs=model_config,
                    timeout_seconds=self.timeout_seconds
                )
                
                self.llm_managers[env] = llm_manager
                self.usage_fetchers[env] = usage_fetcher
                self.model_configs[env] = model_config
                
                logger.info(f"Successfully initialized {env} environment")
                
            except Exception as e:
                logger.error(f"Failed to initialize {env} environment: {e}", exc_info=True)
    
    async def measure_latency_for_model(self, model_name: str, environment: str) -> Optional[dict]:
        """Measure latency for a specific model in a specific environment."""
        try:
            llm_manager = self.llm_managers.get(environment)
            usage_fetcher = self.usage_fetchers.get(environment)
            
            if not llm_manager or not usage_fetcher:
                logger.error(f"Missing manager/fetcher for {environment}")
                return None
            
            logger.info(f"Measuring latency for {model_name} in {environment}")
            
            # Get latency metrics
            metrics = await llm_manager.measure_latency(model_name, TEST_PROMPT)
            
            # Fetch usage data for cost calculation
            usage_data_list = await usage_fetcher.fetch_usage_data()
            usage_map = usage_fetcher.get_model_usage_map(usage_data_list)
            model_usage = usage_map.get(model_name, {})
            
            # Calculate cost
            if model_usage:
                input_rate = model_usage.get("inputTokenRate", 0)
                output_rate = model_usage.get("outputTokenRate", 0)
                measured_cost = (
                    metrics.get("input_tokens", 0) * input_rate +
                    metrics.get("output_tokens", 0) * output_rate
                )
            else:
                measured_cost = 0
                logger.warning(f"No usage data for {model_name} in {environment}")
            
            # Prepare record
            record = {
                "model_name": model_name,
                "environment": environment,
                "timestamp": datetime.now(timezone.utc),
                "latency_ms": metrics.get("latency_ms"),
                "input_tokens": metrics.get("input_tokens"),
                "output_tokens": metrics.get("output_tokens"),
                "cost": measured_cost,
                "context_window": metrics.get("context_window"),
                "is_cloud": metrics.get("is_cloud"),
                "status": metrics.get("status", "success")
            }
            
            return record
            
        except Exception as e:
            logger.error(f"Error measuring {model_name} in {environment}: {e}", exc_info=True)
            return {
                "model_name": model_name,
                "environment": environment,
                "timestamp": datetime.now(timezone.utc),
                "status": "error",
                "latency_ms": None,
                "input_tokens": None,
                "output_tokens": None,
                "cost": None,
                "context_window": None,
                "is_cloud": None
            }
    
    async def run_monitoring_cycle(self) -> List[dict]:
        """Run a single monitoring cycle across all environments and models."""
        all_records = []
        tasks = []
        
        # Create tasks for all environment/model combinations
        for env in self.environments:
            model_config = self.model_configs.get(env, {})
            for model_name in model_config.keys():
                task = self.measure_latency_for_model(model_name, env)
                tasks.append(task)
        
        # Run all measurements in parallel
        logger.info(f"Starting parallel measurement of {len(tasks)} model/environment combinations")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
            elif result is not None:
                all_records.append(result)
        
        logger.info(f"Completed monitoring cycle with {len(all_records)} successful measurements")
        return all_records


# Helper function to integrate with existing monitoring system
async def monitor_all_environments(
    multi_monitor: MultiEnvironmentMonitor,
    data_store: list,
    shutdown_event
):
    """Monitor all environments periodically."""
    monitor_interval = int(os.getenv("MONITOR_INTERVAL", 60))
    
    while not shutdown_event.is_set():
        try:
            # Run monitoring cycle
            records = await multi_monitor.run_monitoring_cycle()
            
            # Add records to data store
            for record in records:
                if record.get("latency_ms") is not None:
                    data_store.append(record)
            
            logger.info(f"Added {len(records)} records to data store")
            
            # Wait for next cycle
            await asyncio.sleep(monitor_interval)
            
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}", exc_info=True)
            await asyncio.sleep(5)  # Short delay before retry