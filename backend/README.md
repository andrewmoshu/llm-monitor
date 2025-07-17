# LLM Monitor Backend

A FastAPI-based backend for monitoring LLM performance across multiple environments.

## Architecture

The backend monitors LLM latency, token usage, and costs across 4 environments: dev, test, qa, and prod.

### Core Files

- **`main.py`** - FastAPI application with API endpoints and monitoring loop
- **`llm_manager.py`** - Manages LLM API calls and latency measurements
- **`usage_data_fetcher.py`** - Fetches usage statistics from external endpoints
- **`model_config_builder.py`** - Builds dynamic model configurations from usage data
- **`schemas.py`** - Pydantic models for data validation
- **`config.py`** - Configuration for multi-environment modes

### Optional Files

- **`multi_env_monitor.py`** - Full multi-environment monitoring (for production use)
- **`mock_usage_endpoint.py`** - Mock endpoint for testing without external dependencies

### Data Storage

- **`latency_data.csv`** - Persistent storage for all measurements
- In-memory store for fast API responses

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file:
```env
# Multi-environment mode (simple or full)
MULTI_ENV_MODE=simple

# Monitoring settings
MONITOR_INTERVAL=60
LLM_API_TIMEOUT=30

# OpenAI settings (if using OpenAI models)
OPENAI_API_KEY=your-api-key
```

### 3. Start Mock Usage Endpoint (for testing)

```bash
python mock_usage_endpoint.py
```

### 4. Start the Backend

```bash
python main.py
```

The backend will:
- Start on http://localhost:8000
- Begin monitoring immediately
- Create data for all 4 environments (in simple mode)

## API Endpoints

### Get Latency Data
```bash
# All environments
GET /api/latency

# Specific environment
GET /api/latency?environment=prod

# With pagination
GET /api/latency?environment=dev&skip=0&limit=100
```

### Get Model Configuration
```bash
GET /api/models
```

### Health Check
```bash
GET /health
```

### API Documentation
```bash
GET /docs  # Swagger UI
GET /redoc # ReDoc
```

## Multi-Environment Modes

### Simple Mode (Default)
- Uses one set of endpoints
- Simulates all 4 environments with variations
- Perfect for testing and development

### Full Mode (Production)
- Requires separate endpoints for each environment
- Real measurements from each environment
- Configure with environment variables:

```bash
export MULTI_ENV_MODE=full
export DEV_USAGE_ENDPOINT=http://dev-api.example.com/api/llm-usage
export TEST_USAGE_ENDPOINT=http://test-api.example.com/api/llm-usage
export QA_USAGE_ENDPOINT=http://qa-api.example.com/api/llm-usage
export PROD_USAGE_ENDPOINT=http://prod-api.example.com/api/llm-usage
```

## Monitoring Flow

1. **Startup**: Fetches model configuration from usage endpoint
2. **Monitoring Loop** (every 60 seconds):
   - Measures latency for each model
   - In simple mode: Creates records for all 4 environments
   - In full mode: Queries each environment separately
3. **Data Storage**: Saves to CSV and in-memory store
4. **API**: Serves data with environment filtering

## Project Structure

```
backend/
├── main.py                    # FastAPI application
├── llm_manager.py             # LLM API management
├── usage_data_fetcher.py      # Usage data fetching
├── model_config_builder.py    # Dynamic configuration
├── schemas.py                 # Data models
├── config.py                  # Multi-env configuration
├── multi_env_monitor.py       # Full multi-env support
├── mock_usage_endpoint.py     # Testing endpoint
├── requirements.txt           # Dependencies
├── latency_data.csv          # Data storage
├── .env                      # Environment variables
└── README.md                 # This file
```

## Development

### Adding New Models

Models are configured dynamically from the usage endpoint. To add new models:

1. Update the usage endpoint to include the new model
2. The backend will automatically detect and monitor it

### Extending Functionality

1. **New Metrics**: Add fields to `schemas.py` and update `measure_latency` in `main.py`
2. **New Endpoints**: Add routes to `main.py`
3. **New Environments**: Update `ENVIRONMENTS` in `config.py`

## Deployment

### Docker

```bash
docker build -t llm-monitor-backend .
docker run -p 8000:8000 -v $(pwd)/latency_data.csv:/app/latency_data.csv llm-monitor-backend
```

### Production Checklist

- [ ] Set `MULTI_ENV_MODE=full`
- [ ] Configure all environment endpoints
- [ ] Set up proper OpenAI API keys
- [ ] Configure monitoring alerts
- [ ] Set up data backup for CSV file
- [ ] Use environment-specific secrets management

## Troubleshooting

### No data appearing
- Wait at least one monitoring cycle (60 seconds)
- Check logs for errors
- Verify usage endpoint is accessible

### High latency measurements
- Check LLM API rate limits
- Verify network connectivity
- Consider increasing `LLM_API_TIMEOUT`

### Memory usage growing
- CSV file may be getting large
- Consider implementing data rotation
- Limit in-memory store size

## Related Documentation

- [Multi-Environment Setup Guide](MULTI_ENV_SETUP_GUIDE.md)
- [Solution Summary](FINAL_SOLUTION_SUMMARY.md)