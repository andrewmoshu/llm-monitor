from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import anthropic
import openai
import time
import os
from dotenv import load_dotenv
import logging
from typing import Dict, Optional
import tiktoken
import asyncio

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 500

# Add context window sizes to your model configurations
MODEL_CONFIGS = {
        'gpt-4o-mini': {
        'context_window': 8192,
        'pricing': {'input': 0.01, 'output': 0.03},
        'is_cloud': True
    }
    # 'gpt-3.5-turbo': {
    #     'context_window': 16385,
    #     'pricing': {'input': 0.0015, 'output': 0.002},
    #     'is_cloud': True
    # },
    # 'gpt-4': {
    #     'context_window': 8192,
    #     'pricing': {'input': 0.03, 'output': 0.06},
    #     'is_cloud': True
    # },
    # 'claude-3-sonnet-20240229': {
    #     'context_window': 200000,
    #     'pricing': {'input': 0.015, 'output': 0.075},
    #     'is_cloud': True
    # }
}

class LLMManager:
    def __init__(self):
        # Initialize API clients
        self.openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Store model configurations
        self.model_configs = MODEL_CONFIGS
        
        # Initialize OpenAI models through LangChain
        self.openai_models = {
            "gpt-4o-mini": ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0,
                max_tokens=MAX_OUTPUT_TOKENS,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            # "gpt-4": ChatOpenAI(
            #     model_name="gpt-4",
            #     temperature=0,
            #     max_tokens=MAX_OUTPUT_TOKENS,
            #     openai_api_key=os.getenv("OPENAI_API_KEY")
            # ),
            # "gpt-3.5-turbo": ChatOpenAI(
            #     model_name="gpt-3.5-turbo",
            #     temperature=0,
            #     max_tokens=MAX_OUTPUT_TOKENS,
            #     openai_api_key=os.getenv("OPENAI_API_KEY")
            # ),
        }
        
        # Define supported Anthropic models
        # self.anthropic_models = [""]
        
        # Initialize tokenizers
        self.tokenizers = {
            # "gpt-4": tiktoken.encoding_for_model("gpt-4"),
            "gpt-4o-mini": tiktoken.encoding_for_model("gpt-4"),  # Uses same tokenizer as GPT-4
            # "gpt-3.5-turbo": tiktoken.encoding_for_model("gpt-3.5-turbo"),
            # "claude-3-sonnet-20240229": tiktoken.encoding_for_model("gpt-4")  # Approximation
        }
        
        # Cache for arena scores
        self.arena_scores = {
            # "gpt-4": 1250,
            "gpt-4o-mini": 1274,
            # "gpt-3.5-turbo": 1068,
            # "claude-3-sonnet-20240229": 1268
        }

    async def initialize_model_info(self):
        """Fetch and cache model information from APIs"""
        try:
            # Pricing information (per 1K tokens)
            pricing_info = {
                # "gpt-4": {
                #     "input": 0.03,
                #     "output": 0.06,
                #     "context_window": 8192,
                #     "training_data": "Up to 2023-04"
                # },
                "gpt-4o-mini": {
                    "input": 0.01,
                    "output": 0.03,
                    "context_window": 128000,
                    "training_data": "Up to 2024-01"
                },
                # "gpt-3.5-turbo": {
                #     "input": 0.001,
                #     "output": 0.002,
                #     "context_window": 16385,
                #     "training_data": "Up to 2023-09"
                # },
                # "claude-3-sonnet-20240229": {
                #     "input": 0.003,
                #     "output": 0.015,
                #     "context_window": 200000,
                #     "training_data": "Up to 2024-02"
                # }
            }

            # Fetch OpenAI models
            openai_models = await self.openai_client.models.list()
            for model in openai_models.data:
                if model.id in self.openai_models:
                    self.model_info[model.id] = {
                        "id": model.id,
                        "owned_by": model.owned_by,
                        "pricing": pricing_info[model.id],
                        "capabilities": {
                            "vision": model.id == "gpt-4-vision-preview",
                            "streaming": True,
                            "function_calling": True
                        }
                    }

            # Add Anthropic models
            for model_id in self.anthropic_models:
                self.model_info[model_id] = {
                    "id": model_id,
                    "owned_by": "anthropic",
                    "pricing": pricing_info[model_id],
                    "capabilities": {
                        "vision": True,
                        "streaming": True,
                        "function_calling": True
                    }
                }

            logger.info("Model information initialized successfully")
            logger.debug(f"Model info: {self.model_info}")

        except Exception as e:
            logger.error(f"Error initializing model information: {str(e)}")
            raise

    def count_tokens(self, text: str, model_name: str) -> int:
        """Count tokens for a given text and model."""
        tokenizer = self.tokenizers.get(model_name)
        if not tokenizer:
            return len(text.split())  # Fallback to word count
        return len(tokenizer.encode(text))

    async def _call_openai_model(self, model_name: str, prompt: str) -> str:
        """Call OpenAI model and return the response text."""
        try:
            response = await self.openai_models[model_name].ainvoke(
                [HumanMessage(content=prompt)]
            )
            return response.content
        except Exception as e:
            logger.error(f"Error calling OpenAI model {model_name}: {str(e)}")
            raise

    async def _call_anthropic_model(self, model_name: str, prompt: str) -> str:
        """Call Anthropic model and return the response text."""
        try:
            response = await self.anthropic_client.messages.create(
                model=model_name,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                system="You are a helpful AI assistant. Please provide concise responses."
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error calling Anthropic model {model_name}: {str(e)}")
            raise

    async def measure_latency(self, model_name: str, prompt: str) -> Dict:
        try:
            start_time = time.time()
            input_tokens = len(self.tokenizers[model_name].encode(prompt))
            
            if model_name in self.openai_models:
                response = await self._call_openai_model(model_name, prompt)
            # elif model_name in self.anthropic_models:
            #     response = await self._call_anthropic_model(model_name, prompt)
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            
            output_tokens = len(self.tokenizers[model_name].encode(response))
            
            # Get model config with default fallback
            model_config = self.model_configs.get(model_name, {})
            
            return {
                "latency_ms": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": self._calculate_cost(model_name, input_tokens, output_tokens),
                "arena_score": self.arena_scores.get(model_name),
                "context_window": model_config.get('context_window', 4096),
                "is_cloud": model_config.get('is_cloud', True)  # Add default value
            }
            
        except Exception as e:
            logger.error(f"Error measuring latency for {model_name}: {str(e)}")
            raise

    def _calculate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost based on token usage and model pricing."""
        pricing = self.model_configs[model_name]['pricing']
        input_cost = (input_tokens / 1000) * pricing['input']
        output_cost = (output_tokens / 1000) * pricing['output']
        return input_cost + output_cost

    def get_supported_models(self):
        """Return list of supported models"""
        return list(self.openai_models.keys())

    async def get_model_info(self, model_name: str) -> Dict:
        """Get detailed information about a specific model."""
        if not self.model_info:
            await self.initialize_model_info()
        return self.model_info.get(model_name, {})