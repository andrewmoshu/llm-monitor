# Frontend-Backend Integration Guide

## Overview
The LLM Monitor system now supports dynamic model configuration with cost data fetched from environment-specific endpoints. The frontend has been updated to properly consume and visualize this data.

## Running the Complete System

### 1. Start the Mock Usage Endpoint (Terminal 1)
```bash
cd /Users/moshurenko/Documents/Work/llm-monitor
python backend/mock_usage_endpoint.py
```
- Runs on port 8001
- Provides usage data for `gpt-4o-mini` (other models commented out for testing)

### 2. Start the Backend (Terminal 2)
```bash
cd /Users/moshurenko/Documents/Work/llm-monitor
ENVIRONMENT=dev python backend/main.py
```
- Runs on port 8000
- Fetches usage data from mock endpoint
- Builds dynamic configuration
- Monitors latency periodically

### 3. Start the Frontend (Terminal 3)
```bash
cd /Users/moshurenko/Documents/Work/llm-monitor/frontend
npm install  # if not already done
npm start
```
- Runs on port 3000
- Fetches data from backend APIs
- Displays latency graphs, tables, and cost summary

## Frontend Updates

### 1. **Updated Type Definitions**
- `ModelInfo` interface now matches backend schema:
  ```typescript
  interface ModelInfo {
    provider: string;
    context_window?: number | null;
    pricing?: { input: number; output: number; } | null;
    is_cloud?: boolean | null;
    arena_score?: number | null;
  }
  ```

### 2. **New Cost Summary Component**
- Displays total daily costs
- Shows cloud vs on-premise costs
- Lists top models by cost

### 3. **Dashboard Layout**
- 4-column layout with models, latencies, and costs
- Real-time updates every 60 seconds
- Dark mode support

## Data Flow

1. **Usage Endpoint** → Returns model pricing and usage data
2. **Backend** → Fetches usage data, builds configuration, measures latency
3. **Frontend** → Fetches both model info and latency records
4. **Visualization** → Displays graphs, tables, and cost summaries

## API Endpoints

### Backend APIs
- `GET /api/models` - Returns model configurations
- `GET /api/latency` - Returns latency measurements
- `POST /api/latency` - Records new latency data

### Mock Usage Endpoint
- `GET /api/llm-usage?date=YYYY-MM-DD` - Returns usage data for specified date

## Environment Variables

### Backend
```bash
ENVIRONMENT=dev|test|qa|prod
DEV_USAGE_ENDPOINT=http://localhost:8001/api/llm-usage
MONITOR_INTERVAL=60  # seconds
LLM_API_TIMEOUT=30   # seconds
OPENAI_API_KEY=your-key  # if measuring real latency
```

### Frontend
- No special configuration needed
- Automatically detects backend at localhost:8000

## Testing

To verify the integration:
1. Check backend logs for "Built dynamic configuration for 1 models"
2. Visit http://localhost:3000 in browser
3. Verify you see `gpt-4o-mini` in the models list
4. Check that latency measurements appear after ~60 seconds
5. Verify cost data is displayed correctly

## Troubleshooting

### No models showing
- Check mock endpoint is running on port 8001
- Verify backend logs show successful fetching of usage data

### No latency data
- Ensure OPENAI_API_KEY is set if testing real API calls
- Check backend logs for latency measurement errors

### Frontend connection errors
- Verify backend is running on port 8000
- Check browser console for CORS or network errors