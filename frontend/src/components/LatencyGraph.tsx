import React, { useState, useMemo, useEffect } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  Legend,
} from 'recharts';
import { 
  Box, 
  Typography, 
  useTheme,
  alpha,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material';
import { LatencyRecord, ModelInfo } from '../types/types';
import RefreshIcon from '@mui/icons-material/Refresh';
import TimelineIcon from '@mui/icons-material/Timeline';
import { IconButton, Tooltip as MuiTooltip, Fade, Badge } from '@mui/material';
import ModelSelector from './ModelSelector';

interface Props {
  data: LatencyRecord[];
  modelInfo: { [key: string]: ModelInfo };
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
  const theme = useTheme();
  if (active && payload && payload.length) {
    return (
      <Box
        sx={{
          bgcolor: theme.palette.mode === 'dark' 
            ? 'rgba(26, 26, 26, 0.95)' 
            : 'rgba(255, 255, 255, 0.95)',
          backdropFilter: 'blur(10px)',
          p: 2,
          borderRadius: 2,
          boxShadow: theme.palette.mode === 'dark'
            ? '0 4px 24px rgba(0, 0, 0, 0.6)'
            : '0 4px 24px rgba(0, 0, 0, 0.15)',
          border: '1px solid',
          borderColor: theme.palette.mode === 'dark'
            ? 'rgba(255, 255, 255, 0.1)'
            : 'rgba(0, 0, 0, 0.05)',
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

// Define the structure of aggregated data points for the graph
interface GraphPoint {
  timestamp: number; // Unix timestamp (milliseconds) for the interval start
  [model: string]: number; // Average latency for each model in seconds
}

// Define the structure for intermediate aggregation
interface AggregatedPointData {
  timestamp: number;
  modelCounts: { [model: string]: number };
  modelSums: { [model: string]: number }; // Sum of latencies in seconds
}

// Helper function to aggregate data by time interval
const aggregateDataByTime = (
  data: LatencyRecord[],
  intervalMinutes: number,
  selectedModelsSet: Set<string>
): GraphPoint[] => {
  const intervalMillis = intervalMinutes * 60 * 1000;
  const aggregated: Map<number, AggregatedPointData> = new Map();

  const filteredData = data.filter(record =>
    selectedModelsSet.has(record.model_name) && record.timestamp
  );

  filteredData.forEach(record => {
    const timestamp = new Date(record.timestamp).getTime();
    const intervalStart = Math.floor(timestamp / intervalMillis) * intervalMillis;

    if (!aggregated.has(intervalStart)) {
      aggregated.set(intervalStart, {
        timestamp: intervalStart,
        modelCounts: {},
        modelSums: {},
      });
    }

    const point = aggregated.get(intervalStart)!;
    const modelName = record.model_name;

    if (!point.modelCounts[modelName]) {
      point.modelCounts[modelName] = 0;
      point.modelSums[modelName] = 0;
    }

    if (record.latency_ms !== null) {
        point.modelCounts[modelName]++;
        point.modelSums[modelName] += record.latency_ms / 1000;
    }
  });

  const result: GraphPoint[] = Array.from(aggregated.values()).map(point => {
    const averages: { [model: string]: number } = {};
    for (const modelName in point.modelCounts) {
      if (point.modelCounts[modelName] > 0) {
        averages[modelName] = point.modelSums[modelName] / point.modelCounts[modelName];
      }
    }
    return {
      timestamp: point.timestamp,
      ...averages,
    };
  }).sort((a, b) => a.timestamp - b.timestamp);

  return result;
};

const LatencyGraph: React.FC<Props> = ({ data, modelInfo }) => {
  const theme = useTheme();
  const [timeRange, setTimeRange] = useState<string>('live');
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [lastUpdate, setLastUpdate] = useState<number>(Date.now());
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [updateCount, setUpdateCount] = useState(0);
  const [intervalMinutes, setIntervalMinutes] = useState<number>(60);
  const allModels = useMemo(() => Object.keys(modelInfo).sort(), [modelInfo]);

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
    setSelectedModels(Array.from(allModels));
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
        
        if (record.latency_ms !== null) {
          point.modelCounts[modelName]++;
          point.modelSums[modelName] += record.latency_ms / 1000;
        }
      });

      // Convert to final format with averages
      const result = Array.from(liveDataPoints.values())
        .map(point => {
          const result: Record<string, number> = { timestamp: point.timestamp };
          Object.keys(point.modelSums).forEach(model => {
            if (point.modelCounts[model] > 0) {
              result[model] = point.modelSums[model] / point.modelCounts[model];
            } else {
              result[model] = 0;
            }
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
      
      if (record.latency_ms !== null) {
        modelData.count++;
        modelData.sum += record.latency_ms / 1000;
      }
      point.models.set(record.model_name, modelData);
    });

    // Convert aggregated data to final format
    return Array.from(dataPoints.values()).map(point => {
      const result: Record<string, number> = { timestamp: point.timestamp };
      point.models.forEach((data: ModelData, model: string) => {
        if (data.count > 0) {
          result[model] = data.sum / data.count;
        } else {
          result[model] = 0;
        }
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
      const newSelected = prev.filter(m => m !== model);
      return newSelected;
    });
  };

  // Calculate Y-axis domain based on visible data
  const yAxisDomain = useMemo(() => {
    let maxLatency = 0;
    processedData.forEach(point => {
      models.forEach(model => {
        if (selectedModels.includes(model) && point[model]) {
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

  const handleIntervalChange = (event: any) => {
    setIntervalMinutes(Number(event.target.value));
  };


  // Memoize aggregated data calculation
  const aggregatedData = useMemo(() => {
    console.log(`Aggregating graph data for interval: ${intervalMinutes}m, models: ${selectedModels.join(', ')}`);
    const selectedModelsSet = new Set(selectedModels);
    return aggregateDataByTime(data, intervalMinutes, selectedModelsSet);
  }, [data, intervalMinutes, selectedModels]);

  // Enhanced color palette for better visibility with many models
  const lineColors = useMemo(() => {
    const baseColors = [
      theme.palette.primary.main,     // Pink
      theme.palette.secondary.main,   // Blue
      '#00ff88',                      // Green
      '#ff9f40',                      // Orange
      '#9966ff',                      // Purple
      '#4bc0c0',                      // Teal
      '#ff6384',                      // Red
      '#ffce56',                      // Yellow
      '#36a2eb',                      // Light Blue
      '#ff80cc',                      // Light Pink
      '#00d084',                      // Emerald
      '#ff5b5b',                      // Coral
    ];
    
    // Generate additional colors if needed
    if (selectedModels.length > baseColors.length) {
      const additionalColors = selectedModels.slice(baseColors.length).map((_, index) => {
        const hue = (index * 360 / (selectedModels.length - baseColors.length)) % 360;
        return `hsl(${hue}, 70%, 60%)`;
      });
      return [...baseColors, ...additionalColors];
    }
    
    return baseColors;
  }, [selectedModels.length, theme]);

  const formatXAxis = (tickItem: number) => {
    return new Date(tickItem).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <Box>
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box>
            <Typography variant="h5" component="h3" sx={{ fontWeight: 600, mb: 0.5 }}>Latency Trend</Typography>
            <Typography variant="body2" color="text.secondary">Real-time performance monitoring</Typography>
          </Box>
          
          {/* Interval Selector */}
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
           <FormControl sx={{ m: 1, minWidth: 120 }} size="small">
             <InputLabel id="interval-select-label">Interval</InputLabel>
             <Select
               labelId="interval-select-label"
               value={intervalMinutes}
               label="Interval"
               onChange={handleIntervalChange}
               sx={{
                 '& .MuiOutlinedInput-notchedOutline': {
                   borderColor: theme.palette.mode === 'dark' 
                     ? 'rgba(255, 255, 255, 0.1)' 
                     : 'rgba(0, 0, 0, 0.1)',
                 },
                 '&:hover .MuiOutlinedInput-notchedOutline': {
                   borderColor: 'primary.main',
                 },
               }}
             >
               <MenuItem value={5}>5 min</MenuItem>
               <MenuItem value={15}>15 min</MenuItem>
               <MenuItem value={30}>30 min</MenuItem>
               <MenuItem value={60}>1 hour</MenuItem>
               <MenuItem value={180}>3 hours</MenuItem>
               <MenuItem value={360}>6 hours</MenuItem>
             </Select>
           </FormControl>
          </Box>
        </Box>
        
        {/* Model Selector */}
        <ModelSelector
          models={allModels}
          selectedModels={selectedModels}
          onChange={setSelectedModels}
          modelInfo={modelInfo}
        />
      </Box>

      {aggregatedData.length === 0 && selectedModels.length > 0 && (
         <Typography sx={{ textAlign: 'center', my: 4 }} color="text.secondary">
             No latency data available for the selected models in this time range.
         </Typography>
      )}
       {selectedModels.length === 0 && (
         <Typography sx={{ textAlign: 'center', my: 4 }} color="text.secondary">
             Please select one or more models to display the graph.
         </Typography>
      )}

      {aggregatedData.length > 0 && selectedModels.length > 0 && (
        <ResponsiveContainer width="100%" height={400}>
          <LineChart
            data={aggregatedData}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid 
              strokeDasharray="3 3" 
              stroke={theme.palette.mode === 'dark' 
                ? 'rgba(255, 255, 255, 0.05)' 
                : 'rgba(0, 0, 0, 0.05)'} 
            />
            <XAxis
              dataKey="timestamp"
              tickFormatter={formatXAxis}
              stroke={theme.palette.text.secondary}
              dy={10}
            />
            <YAxis
              stroke={theme.palette.text.secondary}
              label={{ value: 'Avg Latency (s)', angle: -90, position: 'insideLeft', fill: theme.palette.text.secondary }}
              tickFormatter={(value) => value.toFixed(1)}
            />
            <Tooltip
              content={<CustomTooltip />}
              cursor={{ fill: theme.palette.mode === 'dark' 
                ? 'rgba(255, 255, 255, 0.02)' 
                : 'rgba(0, 0, 0, 0.02)' }}
            />
            <Legend 
              wrapperStyle={{
                paddingTop: '10px',
                fontSize: '12px',
              }}
              iconType="line"
              iconSize={18}
              verticalAlign="bottom"
              height={selectedModels.length > 6 ? 80 : 36}
            />
            {selectedModels.map((modelName, index) => (
              <Line
                key={modelName}
                type="monotone"
                dataKey={modelName}
                stroke={lineColors[index % lineColors.length]}
                strokeWidth={3}
                dot={false}
                connectNulls
                strokeLinecap="round"
                activeDot={{ r: 6, strokeWidth: 0 }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )}
    </Box>
  );
};

export default LatencyGraph; 