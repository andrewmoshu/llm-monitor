import asyncio
import aiohttp
import time
from datetime import datetime
import json
import logging
from llm_manager import LLMManager, MODEL_CONFIGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MONITOR_INTERVAL = 60  # 1 minute in seconds
TEST_INPUT_TOKENS = 500
API_ENDPOINT = "http://localhost:8000/api/latency"

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
    """
    Measure latency for a specific model using the LLMManager.
    """
    try:
        # Get latency metrics from LLMManager
        metrics = await llm_manager.measure_latency(model_name, TEST_PROMPT)
        
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
            "is_cloud": metrics["is_cloud"]
        }
        
        # Send data to API
        async with session.post(API_ENDPOINT, json=data) as response:
            if response.status != 200:
                logger.error(f"Failed to send data for {model_name}: {await response.text()}")
            else:
                logger.info(f"Successfully recorded latency for {model_name}: {metrics['latency_ms']:.2f}ms")
                
    except Exception as e:
        logger.error(f"Error measuring latency for {model_name}: {str(e)}")

async def monitor_latencies():
    # Initialize LLMManager
    llm_manager = LLMManager()
    
    # Get supported models
    models = llm_manager.get_supported_models()
    logger.info(f"Monitoring latency for models: {', '.join(models)}")
    
    async with aiohttp.ClientSession() as session:
        while True:
            tasks = []
            for model in models:
                task = measure_latency(session, model, llm_manager)
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            logger.info(f"Completed latency measurements. Waiting {MONITOR_INTERVAL} seconds...")
            await asyncio.sleep(MONITOR_INTERVAL)

if __name__ == "__main__":
    logger.info("Starting LLM Latency Monitor...")
    asyncio.run(monitor_latencies()) 