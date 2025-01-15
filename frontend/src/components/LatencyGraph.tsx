import React, { useState, useMemo, useEffect } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { 
  Box, 
  Typography, 
  ToggleButtonGroup,
  ToggleButton,
  useTheme,
  alpha,
  Checkbox,
  FormControlLabel,
  Stack,
  Chip,
} from '@mui/material';
import { LatencyRecord } from '../types/types';
import RefreshIcon from '@mui/icons-material/Refresh';
import TimelineIcon from '@mui/icons-material/Timeline';
import { IconButton, Tooltip as MuiTooltip, Fade, Badge } from '@mui/material';

interface Props {
  data: LatencyRecord[];
}

interface ModelData {
  count: number;
  sum: number;
}

interface DataPoint {
  timestamp: number;
  models: Map<string, ModelData>;
}

interface LiveDataPoint {
  timestamp: number;
  modelCounts: { [key: string]: number };
  modelSums: { [key: string]: number };
}

// Distinct colors for better visibility
const MODEL_COLORS = [
  '#2196f3', // Blue
  '#f44336', // Red
  '#4caf50', // Green
  '#ff9800', // Orange
  '#9c27b0', // Purple
  '#00bcd4', // Cyan
  '#ff4081', // Pink
  '#cddc39', // Lime
  '#673ab7', // Deep Purple
  '#009688', // Teal
  '#e91e63', // Pink
  '#3f51b5', // Indigo
];

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <Box
        sx={{
          bgcolor: 'background.paper',
          p: 2,
          borderRadius: 1,
          boxShadow: 3,
          border: '1px solid',
          borderColor: 'divider',
        }}
      >
        <Typography variant="body2" color="text.secondary" gutterBottom>
          {new Date(label).toLocaleString()}
        </Typography>
        {payload
          .sort((a: any, b: any) => b.value - a.value)
          .map((entry: any) => (
            <Box key={entry.name} sx={{ my: 0.5 }}>
              <Typography
                variant="body2"
                component="span"
                sx={{ 
                  color: entry.color, 
                  fontWeight: 'medium',
                  display: 'inline-flex',
                  alignItems: 'center',
                }}
              >
                <Box
                  component="span"
                  sx={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    bgcolor: entry.color,
                    mr: 1,
                    display: 'inline-block'
                  }}
                />
                {entry.name}:
              </Typography>
              <Typography variant="body2" component="span" sx={{ ml: 1 }}>
                {entry.value.toFixed(2)}s
              </Typography>
            </Box>
          ))}
      </Box>
    );
  }
  return null;
};

const LatencyGraph: React.FC<Props> = ({ data }) => {
  const theme = useTheme();
  const [timeRange, setTimeRange] = useState<string>('live');
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const [lastUpdate, setLastUpdate] = useState<number>(Date.now());
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [updateCount, setUpdateCount] = useState(0);

  // Update more frequently for live view
  useEffect(() => {
    if (timeRange === 'live') {
      // Initial update
      setLastUpdate(Date.now());
      
      const interval = setInterval(() => {
        setLastUpdate(Date.now());
      }, 10000); // Update every 10 seconds
      return () => clearInterval(interval);
    }
  }, [timeRange]);

  useEffect(() => {
    const allModels = new Set(data.map(record => record.model_name));
    setSelectedModels(allModels);
  }, [data]);

  // Add debug logging
  useEffect(() => {
    console.log('Data received:', data.length, 'records');
    console.log('Time range:', timeRange);
    console.log('Last update:', new Date(lastUpdate).toISOString());
  }, [data, timeRange, lastUpdate]);

  const processedData = useMemo(() => {
    // Sort data by timestamp
    const sortedData = [...data].sort((a, b) => 
      new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );

    // For live view, use the most recent timestamp as reference
    const mostRecentTimestamp = sortedData.length > 0 
      ? new Date(sortedData[sortedData.length - 1].timestamp).getTime()
      : Date.now();

    // Filter by time range
    const ranges = {
      'live': 30 * 60 * 1000, // 30 minutes instead of 15
      '6h': 6 * 60 * 60 * 1000,
      '24h': 24 * 60 * 60 * 1000,
    };
    
    const rangeMs = ranges[timeRange as keyof typeof ranges];
    const cutoffTime = mostRecentTimestamp - rangeMs;
    
    // Debug logging
    console.log('Filtering data:', {
      mostRecentTimestamp: new Date(mostRecentTimestamp).toISOString(),
      cutoffTime: new Date(cutoffTime).toISOString(),
      totalRecords: sortedData.length
    });

    const filteredData = sortedData.filter(record => {
      const recordTime = new Date(record.timestamp).getTime();
      const isInRange = recordTime > cutoffTime && recordTime <= mostRecentTimestamp;
      return isInRange;
    });

    console.log('Filtered data count:', filteredData.length);

    // If no data in range, return empty array
    if (filteredData.length === 0) {
      console.log('No data in range');
      return [];
    }

    // For live view, show individual data points
    if (timeRange === 'live') {
      const liveDataPoints = new Map<number, LiveDataPoint>();
      
      // Process each data point
      filteredData.forEach(record => {
        const timestamp = new Date(record.timestamp);
        // Round to nearest minute for live view
        timestamp.setSeconds(0, 0);
        const timeKey = timestamp.getTime();

        if (!liveDataPoints.has(timeKey)) {
          liveDataPoints.set(timeKey, { 
            timestamp: timeKey,
            modelCounts: {},
            modelSums: {}
          });
        }

        const point = liveDataPoints.get(timeKey)!;
        const modelName = record.model_name;
        
        if (!(modelName in point.modelCounts)) {
          point.modelCounts[modelName] = 0;
          point.modelSums[modelName] = 0;
        }
        
        point.modelCounts[modelName]++;
        point.modelSums[modelName] += record.latency_ms / 1000;
      });

      // Convert to final format with averages
      const result = Array.from(liveDataPoints.values())
        .map(point => {
          const result: Record<string, number> = { timestamp: point.timestamp };
          Object.keys(point.modelSums).forEach(model => {
            result[model] = point.modelSums[model] / point.modelCounts[model];
          });
          return result;
        })
        .sort((a, b) => a.timestamp - b.timestamp);

      console.log('Live data points:', result.length);
      return result;
    }

    // For other views, aggregate data points by interval
    const interval = timeRange === '6h' ? 5 : 15; // minutes
    const dataPoints = new Map<number, DataPoint>();
    
    filteredData.forEach(record => {
      const timestamp = new Date(record.timestamp).getTime();
      const intervalTime = timeRange === 'live' ? 
        timestamp : 
        Math.floor(timestamp / (interval * 60 * 1000)) * (interval * 60 * 1000);

      if (!dataPoints.has(intervalTime)) {
        dataPoints.set(intervalTime, { 
          timestamp: intervalTime,
          models: new Map<string, ModelData>()
        });
      }
      
      const point = dataPoints.get(intervalTime)!;
      const modelData = point.models.get(record.model_name) || {
        count: 0,
        sum: 0
      };
      
      modelData.count++;
      modelData.sum += record.latency_ms / 1000;
      point.models.set(record.model_name, modelData);
    });

    // Convert aggregated data to final format
    return Array.from(dataPoints.values()).map(point => {
      const result: Record<string, number> = { timestamp: point.timestamp };
      point.models.forEach((data: ModelData, model: string) => {
        result[model] = data.sum / data.count;
      });
      return result;
    }).sort((a, b) => a.timestamp - b.timestamp);
  }, [data, timeRange]);  // Remove lastUpdate from dependencies

  const models = useMemo(() => 
    Array.from(new Set(data.map(record => record.model_name))),
    [data]
  );

  const handleModelToggle = (model: string) => {
    setSelectedModels(prev => {
      const newSelected = new Set(prev);
      if (newSelected.has(model)) {
        newSelected.delete(model);
      } else {
        newSelected.add(model);
      }
      return newSelected;
    });
  };

  // Calculate Y-axis domain based on visible data
  const yAxisDomain = useMemo(() => {
    let maxLatency = 0;
    processedData.forEach(point => {
      models.forEach(model => {
        if (selectedModels.has(model) && point[model]) {
          maxLatency = Math.max(maxLatency, point[model]);
        }
      });
    });
    return [0, Math.ceil(maxLatency * 1.1)]; // Add 10% padding
  }, [processedData, models, selectedModels]);

  // Add refresh handler
  const handleRefresh = () => {
    setIsRefreshing(true);
    setTimeout(() => setIsRefreshing(false), 1000);
  };

  return (
    <Box>
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'flex-start',
        mb: 3,
        flexWrap: 'wrap',
        gap: 2,
      }}>
        <Box>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <TimelineIcon sx={{ mr: 1, color: 'primary.main' }} />
            <Typography variant="h6" sx={{ fontWeight: 500 }}>
              Latency Over Time
            </Typography>
            <MuiTooltip 
              title="Refresh data" 
              TransitionComponent={Fade}
              TransitionProps={{ timeout: 300 }}
            >
              <IconButton 
                size="small" 
                onClick={handleRefresh}
                sx={{ ml: 1, transform: isRefreshing ? 'rotate(180deg)' : 'none', transition: 'transform 0.5s' }}
              >
                <Badge 
                  badgeContent={updateCount} 
                  color="primary"
                  sx={{ 
                    '& .MuiBadge-badge': { 
                      display: timeRange === 'live' ? 'block' : 'none' 
                    } 
                  }}
                >
                  <RefreshIcon fontSize="small" />
                </Badge>
              </IconButton>
            </MuiTooltip>
          </Box>
          
          {/* Model selection with improved layout */}
          <Box 
            sx={{ 
              display: 'flex',
              flexWrap: 'wrap',
              gap: 1,
              maxWidth: '80vw',
              p: 1,
              borderRadius: 1,
              bgcolor: 'background.paper',
              border: '1px solid',
              borderColor: 'divider',
            }}
          >
            {models.map((model, index) => (
              <Chip
                key={model}
                label={model}
                size="small"
                variant={selectedModels.has(model) ? "filled" : "outlined"}
                onClick={() => handleModelToggle(model)}
                sx={{
                  bgcolor: selectedModels.has(model) 
                    ? `${MODEL_COLORS[index % MODEL_COLORS.length]}20`
                    : 'transparent',
                  color: selectedModels.has(model)
                    ? MODEL_COLORS[index % MODEL_COLORS.length]
                    : 'text.secondary',
                  borderColor: MODEL_COLORS[index % MODEL_COLORS.length],
                  '&:hover': {
                    bgcolor: `${MODEL_COLORS[index % MODEL_COLORS.length]}30`,
                  },
                }}
              />
            ))}
          </Box>
        </Box>

        {/* Improved time range selector */}
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
          <ToggleButtonGroup
            value={timeRange}
            exclusive
            onChange={(_, value) => value && setTimeRange(value)}
            size="small"
            sx={{
              bgcolor: 'background.paper',
              '& .MuiToggleButton-root': {
                px: 3,
                py: 0.5,
                border: '1px solid',
                borderColor: 'divider',
                '&.Mui-selected': {
                  bgcolor: 'primary.main',
                  color: 'white',
                  '&:hover': {
                    bgcolor: 'primary.dark',
                  },
                },
              },
            }}
          >
            <ToggleButton value="live" sx={{ color: timeRange === 'live' ? 'white' : 'success.main' }}>
              Live
            </ToggleButton>
            <ToggleButton value="6h">6H</ToggleButton>
            <ToggleButton value="24h">24H</ToggleButton>
          </ToggleButtonGroup>
        </Box>
      </Box>

      {/* Graph container with improved styling */}
      <Box 
        sx={{ 
          width: '100%', 
          height: 400,
          p: 2,
          borderRadius: 2,
          bgcolor: 'background.paper',
          border: '1px solid',
          borderColor: 'divider',
          boxShadow: theme.shadows[1],
        }}
      >
        <ResponsiveContainer>
          <AreaChart
            data={processedData}
            margin={{ top: 10, right: 30, left: 10, bottom: 40 }}
          >
            <defs>
              {models.map((model, index) => (
                <linearGradient key={model} id={`gradient-${model}`} x1="0" y1="0" x2="0" y2="1">
                  <stop 
                    offset="5%" 
                    stopColor={MODEL_COLORS[index % MODEL_COLORS.length]} 
                    stopOpacity={0.3}
                  />
                  <stop 
                    offset="95%" 
                    stopColor={MODEL_COLORS[index % MODEL_COLORS.length]} 
                    stopOpacity={0.05}
                  />
                </linearGradient>
              ))}
            </defs>
            <CartesianGrid 
              strokeDasharray="3 3" 
              vertical={false}
              stroke={alpha(theme.palette.divider, 0.5)}
              strokeWidth={0.5}
            />
            <XAxis
              dataKey="timestamp"
              type="number"
              domain={['dataMin', 'dataMax']}
              tickFormatter={(timestamp) => {
                const date = new Date(timestamp);
                if (timeRange === 'live') {
                  return date.toLocaleTimeString([], { 
                    hour: '2-digit', 
                    minute: '2-digit',
                    second: '2-digit'
                  });
                } else if (timeRange === '6h') {
                  return date.toLocaleTimeString([], { 
                    hour: '2-digit', 
                    minute: '2-digit'
                  });
                } else {
                  return date.toLocaleString([], {
                    month: 'short',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit'
                  });
                }
              }}
              interval={timeRange === 'live' ? 0 : 'preserveStartEnd'}
              minTickGap={timeRange === 'live' ? 30 : 50}
              tick={{ fontSize: 11, fill: theme.palette.text.secondary }}
              angle={-45}
              textAnchor="end"
              height={70}
              tickMargin={25}
              stroke={theme.palette.divider}
              strokeWidth={0.5}
            />
            <YAxis 
              domain={yAxisDomain}
              label={{ 
                value: 'Latency (s)', 
                angle: -90, 
                position: 'insideLeft',
                style: { 
                  textAnchor: 'middle',
                  fill: theme.palette.text.secondary,
                  fontSize: 12,
                }
              }}
              tick={{ fontSize: 11, fill: theme.palette.text.secondary }}
              stroke={theme.palette.divider}
              strokeWidth={0.5}
            />
            <Tooltip 
              content={<CustomTooltip />}
              cursor={{ 
                stroke: theme.palette.divider, 
                strokeWidth: 1,
                strokeDasharray: '3 3'
              }}
            />
            {models
              .filter(model => selectedModels.has(model))
              .map((model, index) => (
                <Area
                  key={model}
                  type="monotone"
                  dataKey={model}
                  name={model}
                  stroke={MODEL_COLORS[index % MODEL_COLORS.length]}
                  fill={`url(#gradient-${model})`}
                  strokeWidth={2}
                  dot={timeRange === 'live'}
                  activeDot={{
                    r: 6,
                    stroke: theme.palette.background.paper,
                    strokeWidth: 2,
                    fill: MODEL_COLORS[index % MODEL_COLORS.length]
                  }}
                  connectNulls
                  animationDuration={300}
                  isAnimationActive={true}
                />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </Box>
    </Box>
  );
};

export default LatencyGraph; 