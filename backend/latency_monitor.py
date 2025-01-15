import asyncio
import aiohttp
import time
from datetime import datetime
import json
import logging
from llm_manager import LLMManager, MODEL_CONFIGS
import os
import ssl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MONITOR_INTERVAL = 60  # 1 minute in seconds
TEST_INPUT_TOKENS = 500
API_ENDPOINT = 'http://localhost:8001/api/latency'
TIMEOUT = aiohttp.ClientTimeout(total=5)  # 5 seconds timeout

# Create a test prompt with approximately 500 tokens
TEST_PROMPT = """You are tasked with explaining a complex topic in a clear and concise manner. 
Please explain the concept of quantum computing, focusing on the following aspects:

1. Basic principles of quantum computing
2. Quantum bits (qubits) and superposition
3. Quantum entanglement and its role
4. Current challenges in quantum computing
5. Potential applications in:
   - Cryptography
   - Drug discovery
   - Optimization problems
   - Machine learning
   - Financial modeling

For each point, provide:
- Clear definition
- Key characteristics
- Practical implications
- Current limitations

Keep your response focused and technical, but accessible to someone with a basic understanding of computer science.
Format your response in a structured manner with clear sections.

Additional context to consider:
- Comparison with classical computing
- Recent breakthroughs in the field
- Major companies and research institutions involved
- Timeline of quantum computing development
- Future prospects and predictions

Technical specifications to address:
- Quantum gates and circuits
- Error correction methods
- Quantum algorithms
- Hardware implementations
- Scaling challenges

Please provide specific examples where relevant and cite any major developments from the past year.
Focus on practical applications and real-world implementations rather than theoretical concepts.
Consider both advantages and limitations of current quantum computing systems."""

async def measure_latency(session, model_name: str, llm_manager: LLMManager):
    """Measure latency for a specific model using the LLMManager."""
    try:
        # Get latency metrics from LLMManager
        metrics = await llm_manager.measure_latency(model_name, TEST_PROMPT)
        logger.info(f"Got metrics for {model_name}: {metrics}")
        
        # Prepare data for API
        data = {
            "model_name": model_name,
            "timestamp": datetime.utcnow().isoformat(),
            "latency_ms": metrics["latency_ms"],
            "input_tokens": metrics["input_tokens"],
            "output_tokens": metrics["output_tokens"],
            "cost": metrics["cost"],
            "arena_score": metrics["arena_score"],
            "context_window": metrics["context_window"],
            "is_cloud": True
        }
        
        # Send data to API with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Sending data to {API_ENDPOINT} (attempt {attempt + 1})")
                async with session.post(
                    API_ENDPOINT, 
                    json=data,
                    timeout=TIMEOUT,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        logger.error(f"Failed to send data: Status {response.status}, Response: {response_text}")
                    else:
                        response_data = await response.json()
                        logger.info(f"Successfully recorded latency: {metrics['latency_ms']:.2f}ms")
                        return
            except asyncio.TimeoutError:
                logger.error(f"Timeout on attempt {attempt + 1}")
                if attempt == max_retries - 1:
                    raise
            except aiohttp.ClientError as e:
                logger.error(f"Network error on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    raise
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    raise
            
            # Wait before retry
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                
    except Exception as e:
        logger.error(f"Error measuring latency for {model_name}: {str(e)}")

async def monitor_latencies():
    # Initialize LLMManager
    llm_manager = LLMManager()
    
    # Get supported models
    models = llm_manager.get_supported_models()
    logger.info(f"Starting monitoring for models: {', '.join(models)}")
    
    # Configure client session with timeout
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        while True:
            try:
                tasks = []
                logger.info(f"Running latency checks for models: {', '.join(models)}")
                for model in models:
                    # Create proper asyncio tasks
                    task = asyncio.create_task(measure_latency(session, model, llm_manager))
                    tasks.append(task)
                
                # Use wait instead of gather to handle individual task failures
                done, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
                
                # Handle results and exceptions
                for task in done:
                    try:
                        await task
                    except Exception as e:
                        logger.error(f"Task failed with error: {str(e)}")
                
                logger.info(f"Completed latency measurements. Waiting {MONITOR_INTERVAL} seconds...")
                await asyncio.sleep(MONITOR_INTERVAL)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(5)  # Wait a bit before retrying

if __name__ == "__main__":
    logger.info("Starting LLM Latency Monitor...")
    asyncio.run(monitor_latencies()) 