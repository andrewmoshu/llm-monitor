import React, { useState, useMemo, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart,
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
} from '@mui/material';
import { LatencyRecord } from '../types/types';

interface Props {
  data: LatencyRecord[];
}

interface ChartDataPoint {
  timestamp: string;
  [key: string]: number | string;
}

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
          {label}
        </Typography>
        {payload.map((entry: any) => (
          <Box key={entry.name} sx={{ my: 0.5 }}>
            <Typography
              variant="body2"
              component="span"
              sx={{ color: entry.color, fontWeight: 'medium' }}
            >
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
  const [timeRange, setTimeRange] = useState<string>('1h');
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());

  useEffect(() => {
    const allModels = new Set(data.map(record => record.model_name));
    setSelectedModels(allModels);
  }, [data]);

  const processedData = useMemo(() => {
    // Sort data by timestamp
    const sortedData = [...data].sort((a, b) => 
      new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );

    // Filter by time range
    const now = new Date().getTime();
    const ranges = {
      '1h': 60 * 60 * 1000,
      '6h': 6 * 60 * 60 * 1000,
      '24h': 24 * 60 * 60 * 1000,
    };
    
    const rangeMs = ranges[timeRange as keyof typeof ranges];
    const cutoffTime = now - rangeMs;
    
    const filteredData = sortedData.filter(record => 
      new Date(record.timestamp).getTime() > cutoffTime
    );

    // If no data in range, return empty array
    if (filteredData.length === 0) return [];

    // Group by timestamp and create data points
    const dataPoints = new Map<string, ChartDataPoint>();
    
    filteredData.forEach(record => {
      const date = new Date(record.timestamp);
      const timestamp = timeRange === '1h' 
        ? date.toLocaleTimeString()
        : date.toLocaleString();
      const latency = record.latency_ms / 1000; // Convert to seconds

      if (!dataPoints.has(timestamp)) {
        dataPoints.set(timestamp, { timestamp });
      }
      
      const point = dataPoints.get(timestamp)!;
      point[record.model_name] = latency;
    });

    return Array.from(dataPoints.values());
  }, [data, timeRange]);

  const models = useMemo(() => 
    Array.from(new Set(data.map(record => record.model_name))),
    [data]
  );

  const colors = [
    '#2196f3', // Blue
    '#00bcd4', // Cyan
    '#009688', // Teal
    '#4caf50', // Green
  ];

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
          <Typography variant="h6" sx={{ fontWeight: 500, mb: 1 }}>
            Latency Over Time
          </Typography>
          <Stack direction="row" spacing={1} sx={{ flexWrap: 'wrap', gap: 1 }}>
            {models.map((model, index) => (
              <FormControlLabel
                key={model}
                control={
                  <Checkbox
                    checked={selectedModels.has(model)}
                    onChange={() => handleModelToggle(model)}
                    sx={{
                      color: colors[index % colors.length],
                      '&.Mui-checked': {
                        color: colors[index % colors.length],
                      },
                    }}
                    size="small"
                  />
                }
                label={
                  <Typography 
                    variant="body2" 
                    sx={{ 
                      color: selectedModels.has(model) 
                        ? colors[index % colors.length]
                        : 'text.secondary',
                      fontWeight: selectedModels.has(model) ? 500 : 400,
                    }}
                  >
                    {model}
                  </Typography>
                }
              />
            ))}
          </Stack>
        </Box>
        <ToggleButtonGroup
          value={timeRange}
          exclusive
          onChange={(_, value) => value && setTimeRange(value)}
          size="small"
          sx={{
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
          <ToggleButton value="1h">1H</ToggleButton>
          <ToggleButton value="6h">6H</ToggleButton>
          <ToggleButton value="24h">24H</ToggleButton>
        </ToggleButtonGroup>
      </Box>
      <Box sx={{ width: '100%', height: 400 }}>
        <ResponsiveContainer>
          <AreaChart
            data={processedData}
            margin={{ top: 10, right: 30, left: 10, bottom: 40 }}
          >
            <defs>
              {models.map((model, index) => (
                <linearGradient key={model} id={`gradient-${model}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={colors[index % colors.length]} stopOpacity={0.3}/>
                  <stop offset="95%" stopColor={colors[index % colors.length]} stopOpacity={0.05}/>
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
              tick={{ fontSize: 11, fill: theme.palette.text.secondary }}
              angle={-45}
              textAnchor="end"
              height={70}
              tickMargin={25}
              stroke={theme.palette.divider}
              strokeWidth={0.5}
            />
            <YAxis 
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
              cursor={{ stroke: theme.palette.divider, strokeWidth: 1 }}
            />
            {models
              .filter(model => selectedModels.has(model))
              .map((model, index) => (
                <Area
                  key={model}
                  type="natural"
                  dataKey={model}
                  name={model}
                  stroke={colors[index % colors.length]}
                  fill={`url(#gradient-${model})`}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{
                    r: 4,
                    stroke: theme.palette.background.paper,
                    strokeWidth: 2,
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