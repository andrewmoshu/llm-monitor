from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import anthropic
import openai
import time
import os
from dotenv import load_dotenv
import logging
from typing import Dict, Any, Optional, List
import tiktoken
import asyncio
import aiohttp

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define default max output tokens, can be overridden by config
MAX_OUTPUT_TOKENS = 500

class LLMManager:
    # Accept model_configs and timeout in __init__
    def __init__(self, model_configs: Dict[str, Dict[str, Any]], timeout_seconds: int = 30):
        """
        Initializes the LLMManager with a unified configuration.

        Args:
            model_configs: A dictionary where keys are model names and values
                           are dictionaries containing configuration for each model
                           (provider, context_window, pricing, api_key_env, etc.).
            timeout_seconds: Timeout for underlying HTTP client sessions.
        """
        self.model_configs = model_configs
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.tokenizers = {}
        self.clients = {} # Store initialized clients (Langchain, Anthropic, etc.)

        # Initialize clients and tokenizers based on the provided config
        self._initialize_clients_and_tokenizers()

    def _initialize_clients_and_tokenizers(self):
        """Initializes API clients and tokenizers based on model_configs."""
        # Use a shared session for OpenAI if multiple models use it
        openai_api_key = os.getenv("OPENAI_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        # Initialize shared clients if keys are available
        shared_openai_client = None
        if openai_api_key:
             # Note: Langchain's ChatOpenAI manages its own async client handling.
             # If using openai SDK directly, initialize AsyncOpenAI here with timeout.
             # self.clients['openai_sdk'] = openai.AsyncOpenAI(api_key=openai_api_key, timeout=self.timeout.total)
             pass # Langchain handles this internally for now

        shared_anthropic_client = None
        if anthropic_api_key:
            try:
                # Anthropic client doesn't directly accept aiohttp timeout,
                # relies on httpx timeout which can be configured via env vars
                # or potentially by customizing the httpx client instance if needed.
                self.clients['anthropic_sdk'] = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
                shared_anthropic_client = self.clients['anthropic_sdk']
            except Exception as e:
                 logger.error(f"Failed to initialize Anthropic client: {e}")


        for model_name, config in self.model_configs.items():
            provider = config.get('provider', '').lower()
            tokenizer_model_name = config.get('tokenizer_model')

            # Initialize Tokenizer
            if tokenizer_model_name:
                try:
                    self.tokenizers[model_name] = tiktoken.encoding_for_model(tokenizer_model_name)
                    logger.info(f"Initialized tokenizer for {model_name} using '{tokenizer_model_name}'")
                except KeyError:
                    try:
                        # Fallback for custom names or models not directly known by tiktoken
                        self.tokenizers[model_name] = tiktoken.get_encoding(tokenizer_model_name)
                        logger.info(f"Initialized tokenizer for {model_name} using encoding '{tokenizer_model_name}'")
                    except Exception as e:
                        logger.warning(f"Could not initialize tokenizer '{tokenizer_model_name}' for model {model_name}: {e}. Token counts may be inaccurate.")
                        self.tokenizers[model_name] = None # Indicate failure
            else:
                 logger.warning(f"No 'tokenizer_model' specified for {model_name}. Token counts may be inaccurate.")
                 self.tokenizers[model_name] = None

            # Initialize Client/Model Instance
            try:
                if provider == 'openai':
                    if openai_api_key:
                        # Use Langchain ChatOpenAI
                        self.clients[model_name] = ChatOpenAI(
                            model_name=model_name,
                            temperature=config.get('temperature', 0),
                            max_tokens=config.get('max_output_tokens', MAX_OUTPUT_TOKENS),
                            openai_api_key=openai_api_key,
                            # Langchain's ChatOpenAI uses httpx internally. Timeout can be set via client_options
                            # or potentially environment variables like OPENAI_TIMEOUT_SECONDS.
                            # Example using client_options (check Langchain docs for specifics):
                            # model_kwargs={"timeout": self.timeout.total} # May vary based on langchain version
                        )
                        logger.info(f"Initialized Langchain OpenAI client for {model_name}")
                    else:
                        logger.warning(f"Skipping OpenAI model {model_name}: OPENAI_API_KEY not found.")

                elif provider == 'anthropic':
                    if shared_anthropic_client:
                        # We use the shared client but store the model name reference if needed
                        self.clients[model_name] = shared_anthropic_client # Reference the shared client
                        logger.info(f"Configured Anthropic model {model_name} (uses shared client)")
                    else:
                        logger.warning(f"Skipping Anthropic model {model_name}: ANTHROPIC_API_KEY not found or client init failed.")

                # --- Add other providers like Ollama ---
                elif provider == 'ollama':
                     # Example using Langchain's Ollama integration (requires `langchain-community`)
                     try:
                         from langchain_community.llms import Ollama # type: ignore
                         base_url = config.get('api_base', 'http://localhost:11434') # Default Ollama URL
                         self.clients[model_name] = Ollama(
                             model=model_name, # The specific ollama model name (e.g., 'llama3')
                             base_url=base_url,
                             temperature=config.get('temperature', 0),
                             # Ollama timeout might be configured differently, check langchain-community docs
                             # num_predict=config.get('max_output_tokens', MAX_OUTPUT_TOKENS), # Parameter name might differ
                         )
                         logger.info(f"Initialized Langchain Ollama client for {model_name} at {base_url}")
                     except ImportError:
                         logger.error(f"Failed to initialize Ollama model {model_name}: `langchain-community` not installed.")
                     except Exception as e:
                         logger.error(f"Failed to initialize Ollama model {model_name}: {e}")

                else:
                    logger.warning(f"Unsupported provider '{provider}' for model {model_name}.")

            except Exception as e:
                logger.error(f"Failed to initialize client for model {model_name}: {e}", exc_info=True)


    def count_tokens(self, text: str, model_name: str) -> int:
        """Count tokens for a given text and model using its configured tokenizer."""
        tokenizer = self.tokenizers.get(model_name)
        if not tokenizer or text is None:
             # Fallback to word count or estimate if tokenizer failed or text is None
             word_count = len(text.split()) if text else 0
             # Simple approximation: tokens often slightly more than words
             estimated_tokens = int(word_count * 1.3) if text else 0
             # logger.debug(f"Using fallback token count ({estimated_tokens}) for model {model_name}")
             return estimated_tokens
        try:
            return len(tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Tokenizer error for model {model_name}: {e}. Falling back to estimation.")
            word_count = len(text.split())
            return int(word_count * 1.3) # Fallback estimation


    async def _call_openai_model(self, model_name: str, prompt: str) -> str:
        """Call OpenAI model via Langchain and return the response text."""
        client = self.clients.get(model_name)
        if not client or not isinstance(client, ChatOpenAI):
            raise ValueError(f"OpenAI client not properly initialized for {model_name}")
        try:
            # Use ainvoke for async call
            response = await client.ainvoke(
                [HumanMessage(content=prompt)]
            )
            return response.content # type: ignore
        except Exception as e:
            logger.error(f"Error calling Langchain OpenAI model {model_name}: {str(e)}")
            raise

    async def _call_anthropic_model(self, model_name: str, prompt: str) -> str:
        """Call Anthropic model using the shared client and return the response text."""
        # The client stored under model_name is the shared AsyncAnthropic client
        client = self.clients.get(model_name)
        if not client or not isinstance(client, anthropic.AsyncAnthropic):
             # Check if the shared client exists if the specific model key wasn't found
             client = self.clients.get('anthropic_sdk')
             if not client or not isinstance(client, anthropic.AsyncAnthropic):
                 raise ValueError(f"Anthropic client not properly initialized.")

        config = self.model_configs.get(model_name, {})
        try:
            response = await client.messages.create(
                model=model_name, # Pass the specific model name here
                max_tokens=config.get('max_output_tokens', MAX_OUTPUT_TOKENS),
                temperature=config.get('temperature', 0),
                messages=[
                    {"role": "user", "content": prompt}
                ],
                # system="You are a helpful AI assistant..." # Optional system prompt
            )
            # Handle potential list response
            if response.content and isinstance(response.content, list) and hasattr(response.content[0], 'text'):
                 return response.content[0].text
            else:
                 logger.warning(f"Unexpected Anthropic response format for {model_name}: {response.content}")
                 return "" # Or raise an error
        except Exception as e:
            logger.error(f"Error calling Anthropic model {model_name}: {str(e)}")
            raise

    async def _call_ollama_model(self, model_name: str, prompt: str) -> str:
        """Call Ollama model via Langchain and return the response text."""
        client = self.clients.get(model_name)
        # Check the actual type based on how you initialized it (e.g., langchain_community.llms.Ollama)
        # For this example, let's assume it's the Ollama class from langchain_community
        try:
            from langchain_community.llms import Ollama # type: ignore
            if not client or not isinstance(client, Ollama):
                 raise ValueError(f"Ollama client not properly initialized for {model_name}")
        except ImportError:
             raise ValueError("Ollama client requires `langchain-community`.")

        try:
            # Use ainvoke for async call if available, otherwise invoke
            if hasattr(client, 'ainvoke'):
                response = await client.ainvoke(prompt)
            else:
                # Fallback to synchronous invoke if ainvoke is not available
                # This will block the event loop - consider running in executor if needed
                response = await asyncio.to_thread(client.invoke, prompt)
            return response # Assuming response is the string content
        except Exception as e:
            logger.error(f"Error calling Langchain Ollama model {model_name}: {str(e)}")
            raise


    async def measure_latency(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """
        Measures latency and gathers metrics for a specific model.

        Args:
            model_name: The name of the model to measure.
            prompt: The input prompt string.

        Returns:
            A dictionary containing latency, token counts, cost, and other model info.
            Returns None for essential fields like latency_ms if measurement fails.
        """
        if model_name not in self.model_configs:
            logger.error(f"Configuration for model '{model_name}' not found.")
            # Return dict with None values to avoid breaking caller expecting a dict
            return {
                "latency_ms": None, "input_tokens": None, "output_tokens": None,
                "cost": None, "arena_score": None, "context_window": None, "is_cloud": None
            }

        config = self.model_configs[model_name]
        provider = config.get('provider', '').lower()
        response_text = ""
        latency = None
        input_tokens = self.count_tokens(prompt, model_name) # Count tokens first

        try:
            start_time = time.time()

            if provider == 'openai':
                response_text = await self._call_openai_model(model_name, prompt)
            elif provider == 'anthropic':
                response_text = await self._call_anthropic_model(model_name, prompt)
            elif provider == 'ollama':
                response_text = await self._call_ollama_model(model_name, prompt)
            # Add elif for other providers here
            else:
                logger.error(f"Unsupported provider '{provider}' for latency measurement of {model_name}")
                raise ValueError(f"Unsupported provider: {provider}")

            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to milliseconds

            output_tokens = self.count_tokens(response_text, model_name)

            # Calculate cost using the unified config
            cost = self._calculate_cost(model_name, input_tokens, output_tokens)

            # Return all relevant info from the config
            return {
                "latency_ms": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "arena_score": config.get("arena_score"), # Get directly from config
                "context_window": config.get("context_window"), # Get directly from config
                "is_cloud": config.get("is_cloud") # Get directly from config
            }

        except Exception as e:
            logger.error(f"Failed to measure latency for {model_name}: {str(e)}", exc_info=True)
            # Return dict with None latency to indicate failure, but keep other known info
            return {
                "latency_ms": None, # Indicate failure
                "input_tokens": input_tokens, # We might have counted input tokens
                "output_tokens": self.count_tokens(response_text, model_name) if response_text else 0,
                "cost": None, # Cost calculation failed
                "arena_score": config.get("arena_score"),
                "context_window": config.get("context_window"),
                "is_cloud": config.get("is_cloud")
            }


    def _calculate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> Optional[float]:
        """Calculate the cost based on token usage and model pricing from config."""
        config = self.model_configs.get(model_name, {})
        pricing = config.get('pricing')

        if not pricing or 'input' not in pricing or 'output' not in pricing:
            # logger.debug(f"Pricing info missing or incomplete for model {model_name}. Cannot calculate cost.")
            return None # Return None if pricing info is missing

        try:
            # Ensure tokens are non-negative integers
            input_tokens = max(0, int(input_tokens)) if input_tokens is not None else 0
            output_tokens = max(0, int(output_tokens)) if output_tokens is not None else 0

            input_cost = (input_tokens / 1000) * float(pricing['input'])
            output_cost = (output_tokens / 1000) * float(pricing['output'])
            return input_cost + output_cost
        except (TypeError, ValueError) as e:
            logger.error(f"Error calculating cost for {model_name}: Invalid pricing or token values. Pricing: {pricing}, Tokens: ({input_tokens}, {output_tokens}). Error: {e}")
            return None


    def get_supported_models(self) -> List[str]:
        """Return list of model names loaded from the configuration."""
        # Return models for which clients were successfully initialized, or just all config keys?
        # Returning all config keys is simpler:
        return list(self.model_configs.keys())
        # Or return only models with initialized clients:
        # return [name for name, client in self.clients.items() if client is not None and not isinstance(client, anthropic.AsyncAnthropic)] \
        #      + [name for name, cfg in self.model_configs.items() if cfg.get('provider') == 'anthropic' and 'anthropic_sdk' in self.clients]


    # get_model_info is removed as config is external now.
    # If needed, a method could be added to return a filtered version
    # of the config for a specific model.
    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
         """Returns the configuration dictionary for a specific model."""
         return self.model_configs.get(model_name)