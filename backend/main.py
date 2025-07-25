from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
from datetime import datetime, timezone
import os
import logging
import csv
import random
import time
from contextlib import asynccontextmanager
# Remove httpx if not used elsewhere, keep if LLMManager needs it
# import httpx
# --- Add imports from latency_monitor ---
import asyncio
import aiohttp # Keep if LLMManager needs it for external calls
from llm_manager import LLMManager, MAX_OUTPUT_TOKENS # Import MAX_OUTPUT_TOKENS if needed globally
import ssl # Keep if LLMManager or dependencies need it
from schemas import LatencyRecordCreate, LatencyRecord as SchemaLatencyRecord, ModelInfo, UsageStats
import threading # <-- Add threading import
from usage_data_fetcher import UsageDataFetcher, Environment  # Import usage data fetcher
from model_config_builder import build_dynamic_model_config, get_minimal_static_config  # Import config builder
from config import MULTI_ENV_MODE, ENVIRONMENTS, ENVIRONMENT_VARIATIONS, is_full_multi_env  # Import config

# Conditionally import multi-env monitor based on configuration
if is_full_multi_env():
    from multi_env_monitor import MultiEnvironmentMonitor, monitor_all_environments

# --- Existing Database imports commented out ---
# from sqlalchemy.orm import Session
# from backend.database import get_db
# from backend.models import LatencyRecord
# from backend.schemas import LatencyRecord as LatencyRecordSchema
# from backend.models import LatencyRecord as LatencyRecordModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
CSV_FILE_PATH = os.getenv("LATENCY_DATA_CSV", "latency_data.csv")
# Define Base URL if needed elsewhere, or remove if only used by monitor
# API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")

# --- Latency Monitor Configuration ---
MONITOR_INTERVAL_SECONDS = int(os.getenv("MONITOR_INTERVAL", 60)) # Use env var
TEST_INPUT_TOKENS = 500 # Target token count for test prompt (adjust prompt if needed)
# Timeout for external LLM API calls (used by LLMManager constructor)
LLM_API_TIMEOUT_SECONDS = int(os.getenv("LLM_API_TIMEOUT", 30)) # Use env var

# Create a test prompt (adjust as needed for ~500 tokens)
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

Keep your response focused and technical, but accessible to someone with a basic understanding of computer science.
Format your response in a structured manner with clear sections.""" # Simplified for brevity

# --- Model Configuration ---
# Dynamic configuration will be built from endpoint data
# This global will be updated after fetching usage data
MODEL_CONFIG: Dict[str, Dict[str, Any]] = {}

# --- In-Memory Data Store & CSV Handling ---
# Use a list of dictionaries to store data in memory
latency_data_store: List[Dict[str, Any]] = []
CSV_HEADERS = [
    "model_name", "environment", "timestamp", "latency_ms", "input_tokens",
    "output_tokens", "cost", "context_window",
    "is_cloud", "status"
]

def load_data_from_csv():
    """Loads historical data from the CSV file into memory."""
    global latency_data_store
    latency_data_store = [] # Clear existing memory store
    try:
        with open(CSV_FILE_PATH, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            # Handle potential header mismatch or missing headers gracefully
            if not reader.fieldnames or not set(CSV_HEADERS).issubset(set(reader.fieldnames)):
                 logger.warning(f"CSV headers mismatch or missing. Expected subset of {CSV_HEADERS}, got {reader.fieldnames}. Trying to load anyway.")
                 # Fallback to default headers if file is empty or headers are wrong
                 if not reader.fieldnames:
                     logger.info("CSV file is empty or has no headers. Will write headers on first record.")
                     # No data to load
                     return

            for row in reader:
                try:
                    # Attempt to convert types back, handle missing keys gracefully
                    record = {
                        'model_name': row.get('model_name'),
                        'environment': row.get('environment', 'dev'),  # Default to dev if missing
                        'timestamp': datetime.fromisoformat(row['timestamp']) if row.get('timestamp') else None,
                        'latency_ms': float(row['latency_ms']) if row.get('latency_ms') else None,
                        'input_tokens': int(row['input_tokens']) if row.get('input_tokens') else None,
                        'output_tokens': int(row['output_tokens']) if row.get('output_tokens') else None,
                        'cost': float(row['cost']) if row.get('cost') else None,
                        'context_window': int(row['context_window']) if row.get('context_window') else None,
                        'is_cloud': row.get('is_cloud', '').lower() == 'true' if row.get('is_cloud') is not None else None, # Handle boolean
                        'status': row.get('status', 'unknown') # Default status if missing
                    }
                    # Filter out rows with essential missing data if necessary
                    if record['model_name'] and record['timestamp'] and record['latency_ms'] is not None:
                        latency_data_store.append(record)
                    else:
                        logger.warning(f"Skipping row due to missing essential data: {row}")

                except (ValueError, TypeError, KeyError) as e:
                    logger.warning(f"Warning: Skipping row due to parsing error: {row} - Error: {e}")
        logger.info(f"Loaded {len(latency_data_store)} records from {CSV_FILE_PATH}")
    except FileNotFoundError:
        logger.info(f"CSV file not found at {CSV_FILE_PATH}. Starting with an empty data store.")
        # Create the file with headers if it doesn't exist
        try:
            with open(CSV_FILE_PATH, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=CSV_HEADERS)
                writer.writeheader()
            logger.info(f"Created new CSV file: {CSV_FILE_PATH}")
        except IOError as e:
            logger.error(f"Error creating CSV file: {e}")
    except Exception as e:
        logger.error(f"Error loading data from CSV: {e}")

def append_to_csv(record: Dict[str, Any]):
    """Appends a single record to the CSV file."""
    # Ensure all headers are present in the record, adding None if missing
    record_to_write = {header: record.get(header) for header in CSV_HEADERS}

    # Convert datetime to ISO format string for CSV
    if isinstance(record_to_write.get('timestamp'), datetime):
         record_to_write['timestamp'] = record_to_write['timestamp'].isoformat()

    file_exists = os.path.isfile(CSV_FILE_PATH)
    write_header = not file_exists or os.path.getsize(CSV_FILE_PATH) == 0

    try:
        with open(CSV_FILE_PATH, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=CSV_HEADERS)
            if write_header:
                writer.writeheader()
            writer.writerow(record_to_write)
    except IOError as e:
        logger.error(f"Error writing to CSV file {CSV_FILE_PATH}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during CSV write: {e}")


# --- Latency Monitoring Functions ---

async def measure_latency(model_name: str, llm_manager: LLMManager, usage_fetcher: UsageDataFetcher, environment: str = "dev"):
    """Measure latency for a specific model using LLMManager and fetch cost data from usage endpoint."""
    try:
        logger.info(f"Measuring latency for {model_name}...")
        # Get latency metrics from LLMManager
        # LLMManager now uses the unified config passed during init
        metrics = await llm_manager.measure_latency(model_name, TEST_PROMPT)
        logger.info(f"Got metrics for {model_name}: {metrics}")

        # Fetch usage data from the endpoint to get cost information
        usage_data_list = await usage_fetcher.fetch_usage_data()
        usage_map = usage_fetcher.get_model_usage_map(usage_data_list)
        
        # Get usage data for this specific model
        model_usage = usage_map.get(model_name, {})
        
        # Calculate cost based on the actual tokens used in this measurement
        # Use rates from the usage endpoint if available
        if model_usage:
            input_rate = model_usage.get("inputTokenRate", 0)
            output_rate = model_usage.get("outputTokenRate", 0)
            measured_cost = (
                metrics.get("input_tokens", 0) * input_rate +
                metrics.get("output_tokens", 0) * output_rate
            )
        else:
            # Fallback to 0 if no usage data available
            measured_cost = 0
            logger.warning(f"No usage data found for {model_name}, using 0 cost")

        # Prepare data record using metrics returned by LLMManager
        record_data = {
            "model_name": model_name,
            "environment": environment,  # Add environment field
            "timestamp": datetime.now(timezone.utc), # Use timezone aware datetime
            "latency_ms": metrics.get("latency_ms"),
            "input_tokens": metrics.get("input_tokens"),
            "output_tokens": metrics.get("output_tokens"),
            "cost": measured_cost,  # Use calculated cost from usage data
            "context_window": metrics.get("context_window"),
            "is_cloud": metrics.get("is_cloud"),
            "status": metrics.get("status")
        }

        # Basic validation before storing
        if record_data["latency_ms"] is None:
             logger.error(f"Latency measurement failed for {model_name}, skipping record.")
             return

        # Add to in-memory store
        if MULTI_ENV_MODE == "simple":
            # Simple multi-env: create records for all environments with variations
            base_env = environment if environment in ENVIRONMENTS else "dev"
            
            for env in ENVIRONMENTS:
                env_record = record_data.copy()
                env_record["environment"] = env
                
                # Add variations to simulate different environments (except for the actual measured one)
                if env != base_env and env_record.get("latency_ms"):
                    variation = 1 + ENVIRONMENT_VARIATIONS[env]
                    env_record["latency_ms"] = env_record["latency_ms"] * variation
                    if env_record.get("cost"):
                        env_record["cost"] = env_record["cost"] * variation
                
                latency_data_store.append(env_record)
                append_to_csv(env_record)
            
            logger.info(f"Successfully recorded latency for {model_name} across all environments: {metrics['latency_ms']:.2f}ms")
        else:
            # Single environment mode or full multi-env (each env has its own measurement)
            latency_data_store.append(record_data)
            append_to_csv(record_data)
            logger.info(f"Successfully recorded latency for {model_name} in {environment}: {metrics['latency_ms']:.2f}ms")

    except Exception as e:
        logger.error(f"Error measuring latency for {model_name}: {str(e)}", exc_info=True) # Log traceback

async def monitor_latencies_periodically(
    llm_manager: LLMManager,
    usage_fetcher: UsageDataFetcher,
    shutdown_event: threading.Event # <-- Add shutdown_event parameter
):
    """Periodically measures latency for all configured models."""
    models = list(MODEL_CONFIG.keys()) # Get models from the unified config
    logger.info(f"[Thread: {threading.get_ident()}] Starting periodic monitoring for models: {', '.join(models)}")

    while not shutdown_event.is_set(): # <-- Check shutdown event
        try:
            tasks = []
            logger.info(f"[Thread: {threading.get_ident()}] Running latency checks for models: {', '.join(models)}")
            for model in models:
                # Create task for each model measurement
                task = asyncio.create_task(measure_latency(model, llm_manager, usage_fetcher, "dev"))
                tasks.append(task)

            # Wait for all tasks to complete for this cycle
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"[Thread: {threading.get_ident()}] Task for model {models[i]} failed: {result}", exc_info=result)

            logger.info(f"[Thread: {threading.get_ident()}] Completed latency measurement cycle. Waiting {MONITOR_INTERVAL_SECONDS} seconds...")
            # Use event.wait for interruptible sleep
            # If shutdown_event is set during this wait, it will return True immediately.
            if shutdown_event.wait(timeout=MONITOR_INTERVAL_SECONDS):
                logger.info(f"[Thread: {threading.get_ident()}] Shutdown event received during sleep, exiting monitor loop.")
                break # Exit loop if event is set

        except asyncio.CancelledError:
            logger.info(f"[Thread: {threading.get_ident()}] Latency monitoring task in thread cancelled.")
            break
        except Exception as e:
            logger.error(f"[Thread: {threading.get_ident()}] Error in monitoring loop: {str(e)}", exc_info=True)
            logger.info(f"[Thread: {threading.get_ident()}] Waiting 5 seconds before retrying monitoring loop...")
            # Check event again before short sleep to avoid long waits on shutdown
            if shutdown_event.wait(timeout=5):
                 logger.info(f"[Thread: {threading.get_ident()}] Shutdown event received during error recovery sleep, exiting.")
                 break
    logger.info(f"[Thread: {threading.get_ident()}] Latency monitoring loop terminated.")

# --- FastAPI Application ---
# Remove background_tasks set if no longer used for asyncio tasks directly in lifespan
# background_tasks = set()

# Global for holding the thread and event
monitor_thread: Optional[threading.Thread] = None
monitor_shutdown_event: Optional[threading.Event] = None


def run_monitor_in_thread(llm_manager: LLMManager, usage_fetcher: UsageDataFetcher, shutdown_event: threading.Event):
    """Target function for the monitoring thread."""
    logger.info(f"Monitoring thread {threading.get_ident()} started. Multi-env mode: {MULTI_ENV_MODE}")
    try:
        asyncio.run(monitor_latencies_periodically(llm_manager, usage_fetcher, shutdown_event))
    except Exception as e:
        logger.error(f"Exception in monitoring thread {threading.get_ident()}: {e}", exc_info=True)
    finally:
        logger.info(f"Monitoring thread {threading.get_ident()} finished.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global monitor_thread, monitor_shutdown_event, MODEL_CONFIG
    # Run startup logic
    logger.info("Running startup logic...")
    load_data_from_csv()

    # Initialize UsageDataFetcher
    logger.info("Initializing UsageDataFetcher...")
    usage_fetcher = UsageDataFetcher(
        environment=os.getenv("ENVIRONMENT", "dev"),
        timeout_seconds=LLM_API_TIMEOUT_SECONDS
    )
    logger.info(f"UsageDataFetcher initialized for environment: {usage_fetcher.environment.value}")
    
    # Fetch usage data to build dynamic configuration
    logger.info("Fetching usage data to build model configuration...")
    usage_data_list = await usage_fetcher.fetch_usage_data()
    
    if usage_data_list:
        # Build dynamic configuration from endpoint data
        MODEL_CONFIG = build_dynamic_model_config(usage_data_list)
        logger.info(f"Built dynamic configuration for {len(MODEL_CONFIG)} models")
    else:
        # Fallback to minimal static configuration
        logger.warning("No usage data fetched, using minimal static configuration")
        MODEL_CONFIG = get_minimal_static_config()
    
    # Initialize LLMManager with the dynamic configuration and timeout
    logger.info("Initializing LLMManager...")
    try:
        # Pass the unified config and timeout directly
        llm_manager = LLMManager(
            model_configs=MODEL_CONFIG,
            timeout_seconds=LLM_API_TIMEOUT_SECONDS # Pass timeout if LLMManager accepts it
        )
        logger.info("LLMManager initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize LLMManager: {e}", exc_info=True)
        # Decide how to handle this - maybe raise the exception or exit?
        # For now, we'll let it proceed but monitoring won't work.
        llm_manager = None # Ensure llm_manager is None if init fails

    # Start the background monitoring task only if LLMManager initialized
    monitor_task = None
    if llm_manager:
        logger.info("Starting background latency monitor task in a new thread...")
        monitor_shutdown_event = threading.Event()
        monitor_thread = threading.Thread(
            target=run_monitor_in_thread, 
            args=(llm_manager, usage_fetcher, monitor_shutdown_event),
            daemon=True # Set as daemon so it doesn't block app exit if main thread dies unexpectedly
        )
        monitor_thread.start()
        logger.info(f"Latency monitor thread ({monitor_thread.ident}) started.")
    else:
        logger.warning("LLMManager initialization failed. Latency monitoring will not run.")


    logger.info("Startup complete.")
    yield
    # Run shutdown logic
    logger.info("Running shutdown logic...")
    if monitor_thread and monitor_shutdown_event:
        logger.info(f"Signalling latency monitor thread ({monitor_thread.ident}) to shut down...")
        monitor_shutdown_event.set()
        monitor_thread.join(timeout=10) # Wait for the thread to finish, with a timeout
        if monitor_thread.is_alive():
            logger.warning(f"Latency monitor thread ({monitor_thread.ident}) did not shut down gracefully after 10s.")
        else:
            logger.info(f"Latency monitor thread ({monitor_thread.ident}) shut down successfully.")

    logger.info("Shutdown complete.")


app = FastAPI(lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for simplicity, restrict in production
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods
    allow_headers=["*"], # Allow all headers
)

# --- Pydantic Models ---
# Model for incoming data to /record_latency/ (if you still need this endpoint externally)
class LatencyDataInput(BaseModel):
    model_name: str
    timestamp: datetime # Changed from datetime.datetime for consistency
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost: Optional[float] = None
    context_window: Optional[int] = None
    is_cloud: Optional[bool] = None

# Model for data returned by /latency_records/
class LatencyDataRecord(BaseModel):
    model_name: str
    environment: str = "dev"  # Added environment field
    timestamp: datetime # Changed from datetime.datetime
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost: Optional[float] = None
    context_window: Optional[int] = None
    is_cloud: Optional[bool] = None # Added field


# --- API Endpoints ---

# This endpoint might become redundant if only the internal monitor records data,
# but keep it if you need an external way to push data.
@app.post("/api/latency", response_model=LatencyDataRecord, tags=["Latency"])
async def record_latency_api(data: LatencyDataInput):
    """
    Receives latency data via API POST request, adds missing info from config,
    stores it, and appends to CSV.
    """
    logger.info(f"Received latency data via API for model: {data.model_name}")
    if data.model_name not in MODEL_CONFIG:
        raise HTTPException(status_code=400, detail=f"Model '{data.model_name}' not configured.")

    # Get missing info from the unified config
    model_info = MODEL_CONFIG[data.model_name]
    context_window = model_info.get("context_window")
    # Use explicit is_cloud flag from config if available, otherwise keep derivation
    is_cloud = model_info.get("is_cloud", not ("localhost" in model_info.get("api_base", "") or "127.0.0.1" in model_info.get("api_base", "")))


    # Create the full record dictionary
    record = data.model_dump() # Convert Pydantic model to dict
    record['context_window'] = context_window
    record['is_cloud'] = is_cloud
    record['timestamp'] = data.timestamp # Ensure datetime object is used

    # Add to in-memory store
    latency_data_store.append(record)

    # Append to CSV file
    append_to_csv(record)

    logger.info(f"Successfully stored API-submitted latency for {data.model_name}")
    # Return the created record matching the response model
    return LatencyDataRecord(**record)


@app.get("/api/latency", response_model=List[LatencyDataRecord], tags=["Latency"])
async def read_latency_records(skip: int = 0, limit: int = 100, environment: Optional[str] = None):
    """Returns latency records from the in-memory store with pagination and optional environment filter."""
    # Filter by environment if specified
    filtered_data = latency_data_store
    if environment:
        filtered_data = [record for record in latency_data_store if record.get("environment") == environment]
    
    # Apply pagination to the filtered list (slice is safe)
    paginated_data = filtered_data[skip : skip + limit]
    # Convert dictionaries to the response model type
    # Handle potential errors during conversion if data format changes
    response_data = []
    for record in paginated_data:
        try:
            response_data.append(LatencyDataRecord(**record))
        except Exception as e:
            logger.warning(f"Skipping record due to validation error: {record} - Error: {e}")
    return response_data

@app.get("/api/models", tags=["Configuration"])
def get_models() -> Dict[str, ModelInfo]:
    """Returns the publicly relevant configured model details."""
    public_config = {}
    for name, config in MODEL_CONFIG.items():
        # Ensure config is a dictionary before accessing keys
        if isinstance(config, dict):
            # Exclude potentially sensitive or internal details before returning
            model_info_data = {
                'provider': config.get('provider', 'unknown'),
                'context_window': config.get('context_window'),
                'pricing': config.get('pricing'),
                'is_cloud': config.get('is_cloud'),
                'arena_score': config.get('arena_score')
            }
            
            # Add usage stats if available
            if 'usage_stats' in config and config['usage_stats']:
                model_info_data['usage_stats'] = UsageStats(**config['usage_stats'])
            
            public_config[name] = ModelInfo(**model_info_data)
        else:
             logger.warning(f"Skipping model '{name}' in /api/models response due to unexpected config format: {config}")
    return public_config

# Example root endpoint
@app.get("/", tags=["General"])
def read_root():
    return {"message": "LLM Latency Monitoring API - CSV Backend"}

# --- Uvicorn Runner (for local development) ---
if __name__ == "__main__":
    import uvicorn
    # Lifespan handles loading data and starting monitor now
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) # Use reload for dev 