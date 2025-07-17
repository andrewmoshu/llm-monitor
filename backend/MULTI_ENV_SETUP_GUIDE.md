# Multi-Environment Setup Guide

This guide explains how to use the multi-environment monitoring feature in the LLM Monitor backend.

## Quick Start (Simple Mode)

By default, the backend runs in **simple multi-environment mode**, which simulates all 4 environments from a single measurement.

### 1. Start the Backend

```bash
cd backend
python main.py
```

The backend will automatically:
- Monitor models using your current configuration
- Create measurements for all 4 environments (dev, test, qa, prod)
- Apply realistic variations to simulate different environment behaviors

### 2. Environment Variations

In simple mode, measurements are varied to simulate realistic differences:
- **dev**: Baseline (no variation)
- **test**: +5% latency
- **qa**: +10% latency
- **prod**: +15% latency

### 3. Frontend Usage

The frontend will automatically:
- Show the environment switcher
- Filter data based on selected environment
- Display environment-specific metrics

## Configuration Options

### Environment Variables

```bash
# Set multi-environment mode (default: "simple")
export MULTI_ENV_MODE="simple"  # or "full"

# Set monitoring interval (default: 60 seconds)
export MONITOR_INTERVAL=60

# Set LLM API timeout (default: 30 seconds)
export LLM_API_TIMEOUT=30
```

### Simple Mode (Default)

```bash
# No additional configuration needed
export MULTI_ENV_MODE="simple"
python main.py
```

**Benefits:**
- ✅ Works out of the box
- ✅ No additional endpoints needed
- ✅ Great for testing and development
- ✅ Realistic data variations

### Full Mode (Production)

For production use with real separate environments:

```bash
# Enable full multi-environment mode
export MULTI_ENV_MODE="full"

# Configure environment-specific endpoints
export DEV_USAGE_ENDPOINT="http://dev-api.company.com/api/llm-usage"
export TEST_USAGE_ENDPOINT="http://test-api.company.com/api/llm-usage"
export QA_USAGE_ENDPOINT="http://qa-api.company.com/api/llm-usage"
export PROD_USAGE_ENDPOINT="http://prod-api.company.com/api/llm-usage"

python main.py
```

**Benefits:**
- ✅ Real measurements from each environment
- ✅ Accurate environment-specific data
- ✅ Parallel monitoring of all environments
- ✅ Production-ready

## API Usage

### Get All Latency Data
```bash
GET /api/latency
```

### Get Environment-Specific Data
```bash
GET /api/latency?environment=prod
GET /api/latency?environment=qa
GET /api/latency?environment=test
GET /api/latency?environment=dev
```

### Get Models Configuration
```bash
GET /api/models
```

## Troubleshooting

### Error: "'LLMManager' object has no attribute 'run_monitoring_cycle'"

This error occurs when the multi_env_monitor module is imported but not properly configured. Solutions:

1. **Use Simple Mode** (Recommended for testing):
   ```bash
   export MULTI_ENV_MODE="simple"
   python main.py
   ```

2. **Or Remove the Import**:
   - Comment out the multi_env_monitor import in main.py
   - The simple mode will work automatically

### No Data Showing

1. Check the backend logs for errors
2. Verify the usage endpoint is accessible
3. Wait at least one monitoring interval (default: 60 seconds)

### Frontend Not Updating

1. Ensure backend is running
2. Check browser console for errors
3. Verify API calls are returning data

## Upgrading from Simple to Full Mode

When ready for production:

1. Set up environment-specific usage endpoints
2. Configure endpoint URLs via environment variables
3. Change mode to "full":
   ```bash
   export MULTI_ENV_MODE="full"
   ```
4. Restart the backend

The frontend requires no changes - it will automatically work with the full multi-environment data.

## Architecture

### Simple Mode
```
┌─────────────┐
│   Backend   │
│  (1 instance)│
└──────┬──────┘
       │
       ├─→ Measure once
       │
       └─→ Create 4 records
           ├─→ dev (actual)
           ├─→ test (+5%)
           ├─→ qa (+10%)
           └─→ prod (+15%)
```

### Full Mode
```
┌─────────────┐
│   Backend   │
│  (1 instance)│
└──────┬──────┘
       │
       ├─→ Dev Manager ──→ Dev Endpoint
       ├─→ Test Manager ─→ Test Endpoint
       ├─→ QA Manager ───→ QA Endpoint
       └─→ Prod Manager ─→ Prod Endpoint
```

## Best Practices

1. **Start with Simple Mode** for development and testing
2. **Use Full Mode** for production deployments
3. **Monitor the logs** to ensure all environments are healthy
4. **Set appropriate timeouts** based on your LLM response times
5. **Configure alerts** for environment-specific failures