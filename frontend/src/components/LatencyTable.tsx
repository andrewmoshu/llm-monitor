import React, { useState, useMemo } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Typography,
  TableSortLabel,
  Chip,
  Box,
  Tooltip as MuiTooltip,
  useTheme,
  Collapse,
  IconButton,
  Grid,
  Divider,
  Card,
  CardContent,
} from '@mui/material';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import TimerOffIcon from '@mui/icons-material/TimerOff';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
import KeyboardArrowUpIcon from '@mui/icons-material/KeyboardArrowUp';
import TokenIcon from '@mui/icons-material/Token';
import SpeedIcon from '@mui/icons-material/Speed';
import { StatusChip, LatencyChip, ContextChip, DeploymentChip } from './StyledChips';
import AttachMoneyIcon from '@mui/icons-material/AttachMoney';
import HistoryIcon from '@mui/icons-material/History';
import { LatencyRecord, ModelInfo } from '../types/types';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';

interface Props {
  data: LatencyRecord[];
  modelInfo: { [key: string]: ModelInfo };
}

type SortField = 'model_name' | 'latency_ms' | 'cost' | 'input_tokens' | 'output_tokens' | 'context_window' | 'status';
type SortOrder = 'asc' | 'desc';

const LatencyTable: React.FC<Props> = ({ data, modelInfo }) => {
  const theme = useTheme();
  const [sortField, setSortField] = useState<SortField>('model_name');
  const [sortOrder, setSortOrder] = useState<SortOrder>('asc');
  const [expandedRow, setExpandedRow] = useState<string | null>(null);

  // Define createPlaceholderRecord *before* it's used in useMemo
  const createPlaceholderRecord = (modelName: string, info: ModelInfo | undefined): LatencyRecord => {
    const placeholderTimestamp = new Date(0).toISOString();

    return {
        model_name: modelName,
        timestamp: placeholderTimestamp,
        latency_ms: null,
        input_tokens: null,
        output_tokens: null,
        cost: null,
        context_window: info?.context_window ?? null,
        is_cloud: info?.is_cloud ?? false,
        status: 'unavailable',
    };
  };

  const displayRecords = useMemo(() => {
    console.log("Recalculating display records...");

    const latestMeasurements: { [key: string]: LatencyRecord } = {};
    data.forEach(record => {
      if (!latestMeasurements[record.model_name] || new Date(record.timestamp) > new Date(latestMeasurements[record.model_name].timestamp)) {
        latestMeasurements[record.model_name] = record;
      }
    });

    const allRecordsMap: { [key: string]: LatencyRecord } = {};
    const configuredModels = Object.keys(modelInfo);

    configuredModels.forEach(modelName => {
      if (latestMeasurements[modelName]) {
        allRecordsMap[modelName] = latestMeasurements[modelName];
      } else {
        console.log(`Creating placeholder for missing model: ${modelName}`);
        allRecordsMap[modelName] = createPlaceholderRecord(modelName, modelInfo[modelName]);
      }
    });

    Object.keys(latestMeasurements).forEach(modelName => {
        if (!allRecordsMap[modelName]) {
            console.warn(`Model ${modelName} found in latency data but not in modelInfo config.`);
            allRecordsMap[modelName] = latestMeasurements[modelName];
        }
    });

    return Object.values(allRecordsMap);

  }, [data, modelInfo]);

  const handleSort = (field: SortField) => {
    if (field === sortField) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortOrder('asc');
    }
  };

  const sortedRecords = useMemo(() => {
     return [...displayRecords].sort((a, b) => {
       const multiplier = sortOrder === 'asc' ? 1 : -1;
       const valA = a[sortField as keyof LatencyRecord];
       const valB = b[sortField as keyof LatencyRecord];

       if (valA == null && valB == null) return 0;
       if (valA == null) return sortOrder === 'asc' ? 1 : -1;
       if (valB == null) return sortOrder === 'asc' ? -1 : 1;

       if (typeof valA === 'string' && typeof valB === 'string') {
         if (sortField === 'status') {
            const statusOrder = { 'active': 1, 'timeout': 2, 'unavailable': 3, 'connection_error': 4, 'unresponsive': 5, 'error': 6 };
            const orderA = statusOrder[valA as keyof typeof statusOrder] ?? 99;
            const orderB = statusOrder[valB as keyof typeof statusOrder] ?? 99;
            return multiplier * (orderA - orderB);
         }
         return multiplier * valA.localeCompare(valB);
       }
       if (typeof valA === 'number' && typeof valB === 'number') {
         return multiplier * (valA - valB);
       }
       if (typeof valA === 'boolean' && typeof valB === 'boolean') {
           return multiplier * (Number(valA) - Number(valB));
       }
       return 0;
     });
  }, [displayRecords, sortField, sortOrder]);

  const formatContextWindow = (tokens: number | null | undefined): string => {
    if (tokens === null || tokens === undefined) {
      return 'N/A';
    }
    if (tokens >= 1000) {
      return `${(tokens / 1000).toFixed(0)}k`;
    }
    return tokens.toString();
  };

  const getLatencyColor = (latency: number | null): 'success' | 'warning' | 'error' | 'default' => {
    if (latency === null) return 'default';
    if (latency < 1000) return 'success';
    if (latency < 5000) return 'warning';
    return 'error';
  };

  const getStatusChip = (status: string | null | undefined) => {
    switch (status?.toLowerCase()) {
      case 'active':
        return { icon: <CheckCircleOutlineIcon />, color: 'success', label: 'Active' };
      case 'unresponsive':
      case 'timeout':
      case 'error':
      case 'connection_error':
        return { icon: <ErrorOutlineIcon />, color: 'error', label: status };
      case 'unavailable':
        return { icon: <HelpOutlineIcon />, color: 'default', label: 'Unavailable' };
      default:
        return { icon: <HelpOutlineIcon />, color: 'warning', label: status || 'Unknown' };
    }
  };

  const handleRowClick = (modelName: string) => {
    setExpandedRow(expandedRow === modelName ? null : modelName);
  };

  // Calculate model statistics
  const getModelStats = (modelName: string) => {
    const modelRecords = data.filter(r => r.model_name === modelName);
    const validRecords = modelRecords.filter(r => r.latency_ms !== null && r.latency_ms >= 0);
    
    if (validRecords.length === 0) {
      return {
        avgLatency: 0,
        minLatency: 0,
        maxLatency: 0,
        totalRequests: 0,
        successfulRequests: 0,
        avgInputTokens: 0,
        avgOutputTokens: 0,
        totalCost: 0,
      };
    }

    const latencies = validRecords.map(r => r.latency_ms!);
    const inputTokens = validRecords.map(r => r.input_tokens || 0);
    const outputTokens = validRecords.map(r => r.output_tokens || 0);
    const costs = validRecords.map(r => r.cost || 0);

    return {
      avgLatency: latencies.reduce((a, b) => a + b, 0) / latencies.length,
      minLatency: Math.min(...latencies),
      maxLatency: Math.max(...latencies),
      totalRequests: modelRecords.length,
      successfulRequests: validRecords.length,
      avgInputTokens: inputTokens.reduce((a, b) => a + b, 0) / inputTokens.length,
      avgOutputTokens: outputTokens.reduce((a, b) => a + b, 0) / outputTokens.length,
      totalCost: costs.reduce((a, b) => a + b, 0),
    };
  };

  return (
    <Box>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" component="h3" sx={{ fontWeight: 600, mb: 0.5 }}>
          Model Performance Comparison
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Detailed metrics and performance analysis
        </Typography>
      </Box>
      <TableContainer 
        sx={{ 
          background: 'transparent',
          borderRadius: 2,
          overflow: 'hidden',
          '& .MuiTable-root': {
            background: 'transparent',
          },
        }}
      >
        <Table stickyHeader size="small" sx={{ 
          '& .MuiTableCell-root': {
            borderBottom: 'none',
          }
        }}>
          <TableHead>
            <TableRow 
              sx={{ 
                backgroundColor: theme.palette.mode === 'dark' 
                  ? 'rgba(255, 255, 255, 0.02)' 
                  : 'rgba(0, 0, 0, 0.01)',
                '& th': {
                  backgroundColor: 'transparent',
                }
              }}
            >
              <TableCell
                sx={{
                  fontWeight: 600,
                  borderBottom: `1px solid ${theme.palette.mode === 'dark' 
                    ? 'rgba(255, 255, 255, 0.08)' 
                    : 'rgba(0, 0, 0, 0.08)'}`,
                }}
              >
                <TableSortLabel
                  active={sortField === 'model_name'}
                  direction={sortField === 'model_name' ? sortOrder : 'asc'}
                  onClick={() => handleSort('model_name')}
                >
                  Model
                </TableSortLabel>
              </TableCell>
              <TableCell
                sx={{
                  fontWeight: 600,
                  borderBottom: `1px solid ${theme.palette.mode === 'dark' 
                    ? 'rgba(255, 255, 255, 0.08)' 
                    : 'rgba(0, 0, 0, 0.08)'}`,
                }}
              >
                <TableSortLabel
                  active={sortField === 'status'}
                  direction={sortField === 'status' ? sortOrder : 'asc'}
                  onClick={() => handleSort('status')}
                >
                  Status
                </TableSortLabel>
              </TableCell>
              <TableCell
                sx={{
                  fontWeight: 600,
                  borderBottom: `1px solid ${theme.palette.mode === 'dark' 
                    ? 'rgba(255, 255, 255, 0.08)' 
                    : 'rgba(0, 0, 0, 0.08)'}`,
                }}
              >
                <TableSortLabel
                  active={sortField === 'context_window'}
                  direction={sortField === 'context_window' ? sortOrder : 'asc'}
                  onClick={() => handleSort('context_window')}
                >
                  Context
                </TableSortLabel>
              </TableCell>
              <TableCell
                sx={{
                  fontWeight: 600,
                  borderBottom: `1px solid ${theme.palette.mode === 'dark' 
                    ? 'rgba(255, 255, 255, 0.08)' 
                    : 'rgba(0, 0, 0, 0.08)'}`,
                }}
              >
                <TableSortLabel
                  active={sortField === 'latency_ms'}
                  direction={sortField === 'latency_ms' ? sortOrder : 'asc'}
                  onClick={() => handleSort('latency_ms')}
                >
                  Latency (Avg)
                </TableSortLabel>
              </TableCell>
              <TableCell
                sx={{
                  fontWeight: 600,
                  borderBottom: `1px solid ${theme.palette.mode === 'dark' 
                    ? 'rgba(255, 255, 255, 0.08)' 
                    : 'rgba(0, 0, 0, 0.08)'}`,
                }}
              >
                <TableSortLabel
                  active={sortField === 'cost'}
                  direction={sortField === 'cost' ? sortOrder : 'asc'}
                  onClick={() => handleSort('cost')}
                >
                  Cost ($/1K tokens)
                </TableSortLabel>
              </TableCell>
              <TableCell
                sx={{
                  fontWeight: 600,
                  borderBottom: `1px solid ${theme.palette.mode === 'dark' 
                    ? 'rgba(255, 255, 255, 0.08)' 
                    : 'rgba(0, 0, 0, 0.08)'}`,
                }}
              >
                <TableSortLabel
                  active={sortField === 'input_tokens'}
                  direction={sortField === 'input_tokens' ? sortOrder : 'asc'}
                  onClick={() => handleSort('input_tokens')}
                >
                  Tokens (In/Out)
                </TableSortLabel>
              </TableCell>
              <TableCell
                sx={{
                  fontWeight: 600,
                  borderBottom: `1px solid ${theme.palette.mode === 'dark' 
                    ? 'rgba(255, 255, 255, 0.08)' 
                    : 'rgba(0, 0, 0, 0.08)'}`,
                }}
              >
                Type
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {sortedRecords.map((record) => {
              // Determine the effective status based on latency
              const hasValidLatency = record.latency_ms !== null && record.latency_ms >= 0;
              const effectiveStatus = hasValidLatency ? 'active' : record.status;
              
              // Determine the timestamp to display in the tooltip
              // Use the actual record timestamp if it exists and is not the placeholder timestamp
              const placeholderTimestamp = new Date(0).toISOString();
              const displayTimestamp = record.timestamp !== placeholderTimestamp ? record.timestamp : null;

              return (
                <React.Fragment key={record.model_name}>
                <TableRow 
                  hover
                  onClick={() => handleRowClick(record.model_name)}
                  sx={{
                    cursor: 'pointer',
                    borderBottom: `1px solid ${theme.palette.mode === 'dark' 
                      ? 'rgba(255, 255, 255, 0.03)' 
                      : 'rgba(0, 0, 0, 0.03)'}`,
                    transition: 'background-color 0.2s ease',
                    '&:last-child': {
                      borderBottom: 0,
                    },
                    '&:hover': {
                      bgcolor: theme.palette.mode === 'dark'
                        ? 'rgba(255, 77, 166, 0.05)'
                        : 'rgba(255, 77, 166, 0.02)',
                    },
                  }}
                >
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <IconButton size="small" sx={{ mr: 1 }}>
                        {expandedRow === record.model_name ? <KeyboardArrowUpIcon /> : <KeyboardArrowDownIcon />}
                      </IconButton>
                      <Typography variant="body2" fontWeight="medium">
                        {record.model_name}
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell>
                    <MuiTooltip 
                      title={`Last check: ${displayTimestamp ? new Date(displayTimestamp).toLocaleString() : 'N/A'}`}
                      arrow
                    >
                      <Box>
                        <StatusChip status={effectiveStatus || 'unknown'} timestamp={displayTimestamp} />
                      </Box>
                    </MuiTooltip>
                  </TableCell>
                  <TableCell>
                    <ContextChip contextWindow={record.context_window} />
                  </TableCell>
                  <TableCell>
                    {record.latency_ms !== null ? (
                      <MuiTooltip title={`${record.latency_ms.toFixed(0)}ms`}>
                        <Box>
                          <LatencyChip latencyMs={record.latency_ms} />
                        </Box>
                      </MuiTooltip>
                    ) : (
                      <Typography variant="caption" color="textSecondary">N/A</Typography>
                    )}
                  </TableCell>
                  <TableCell>
                    {record.cost !== null && record.input_tokens && record.output_tokens ? (
                       <Typography variant="body2" color="textSecondary">
                         ${((record.cost / ((record.input_tokens + record.output_tokens) / 1000)) || 0).toFixed(4)}
                       </Typography>
                    ) : (
                       <Typography variant="caption" color="textSecondary">N/A</Typography>
                    )}
                  </TableCell>
                  <TableCell>
                    {record.input_tokens !== null && record.output_tokens !== null ? (
                      <Typography variant="body2" color="textSecondary">
                        {record.input_tokens} / {record.output_tokens}
                      </Typography>
                    ) : (
                       <Typography variant="caption" color="textSecondary">N/A</Typography>
                    )}
                  </TableCell>
                  <TableCell>
                    <DeploymentChip isCloud={record.is_cloud} />
                  </TableCell>
                </TableRow>
                <TableRow sx={{ bgcolor: 'transparent' }}>
                  <TableCell style={{ paddingBottom: 0, paddingTop: 0, backgroundColor: 'transparent', border: 'none' }} colSpan={7}>
                    <Collapse in={expandedRow === record.model_name} timeout="auto" unmountOnExit>
                      <Box sx={{ 
                        margin: 2,
                        bgcolor: theme.palette.mode === 'dark' 
                          ? 'rgba(255, 255, 255, 0.02)' 
                          : 'rgba(0, 0, 0, 0.01)',
                        borderRadius: 2,
                        p: 3,
                        border: `1px solid ${theme.palette.mode === 'dark' 
                          ? 'rgba(255, 255, 255, 0.05)' 
                          : 'rgba(0, 0, 0, 0.05)'}`
                      }}>
                        <ModelDetailView 
                          modelName={record.model_name}
                          data={data}
                          modelInfo={modelInfo[record.model_name]}
                          stats={getModelStats(record.model_name)}
                        />
                      </Box>
                    </Collapse>
                  </TableCell>
                </TableRow>
                </React.Fragment>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
};

// Model Detail View Component
interface ModelDetailViewProps {
  modelName: string;
  data: LatencyRecord[];
  modelInfo?: ModelInfo;
  stats: {
    avgLatency: number;
    minLatency: number;
    maxLatency: number;
    totalRequests: number;
    successfulRequests: number;
    avgInputTokens: number;
    avgOutputTokens: number;
    totalCost: number;
  };
}

const ModelDetailView: React.FC<ModelDetailViewProps> = ({ modelName, data, modelInfo, stats }) => {
  const theme = useTheme();
  
  // Debug log to check what data we're receiving
  console.log('ModelDetailView data:', {
    modelName,
    modelInfo,
    hasUsageStats: !!modelInfo?.usage_stats,
    usageStats: modelInfo?.usage_stats
  });
  
  const modelHistory = useMemo(() => {
    return data
      .filter(r => r.model_name === modelName)
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, 10); // Show last 10 records
  }, [data, modelName]);

  return (
    <Grid container spacing={3}>
      {/* Model Statistics */}
      <Grid item xs={12}>
        <Typography variant="h6" gutterBottom sx={{ 
          display: 'flex', 
          alignItems: 'center',
          color: theme.palette.text.primary,
          mb: 3
        }}>
          <SpeedIcon sx={{ mr: 1, color: theme.palette.primary.main }} />
          Model Statistics
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={3}>
            <Card variant="outlined" sx={{ 
              bgcolor: theme.palette.mode === 'dark' 
                ? 'rgba(255, 255, 255, 0.03)' 
                : 'rgba(0, 0, 0, 0.02)',
              backdropFilter: 'blur(10px)',
              border: '1px solid',
              borderColor: theme.palette.mode === 'dark' 
                ? 'rgba(255, 255, 255, 0.05)' 
                : 'rgba(0, 0, 0, 0.05)',
              transition: 'border-color 0.2s ease, background-color 0.2s ease, box-shadow 0.2s ease',
              '&:hover': {
                borderColor: theme.palette.primary.main,
                bgcolor: theme.palette.mode === 'dark' 
                  ? 'rgba(255, 77, 166, 0.05)' 
                  : 'rgba(255, 77, 166, 0.02)',
                boxShadow: theme.palette.mode === 'dark' 
                  ? '0 4px 12px rgba(255, 77, 166, 0.2)' 
                  : '0 4px 12px rgba(255, 77, 166, 0.1)'
              }
            }}>
              <CardContent sx={{ p: 2.5 }}>
                <Typography variant="body2" gutterBottom sx={{ 
                  color: theme.palette.text.secondary,
                  fontSize: '0.75rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>
                  Average Latency
                </Typography>
                <Typography variant="h5" sx={{ 
                  color: theme.palette.text.primary,
                  fontWeight: 700,
                  fontSize: '1.5rem'
                }}>
                  {(stats.avgLatency / 1000).toFixed(2)}s
                </Typography>
                <Typography variant="caption" sx={{ 
                  color: theme.palette.text.secondary,
                  fontSize: '0.7rem',
                  opacity: 0.8
                }}>
                  Min: {(stats.minLatency / 1000).toFixed(2)}s | Max: {(stats.maxLatency / 1000).toFixed(2)}s
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card variant="outlined" sx={{ 
              bgcolor: theme.palette.mode === 'dark' 
                ? 'rgba(255, 255, 255, 0.03)' 
                : 'rgba(0, 0, 0, 0.02)',
              backdropFilter: 'blur(10px)',
              border: '1px solid',
              borderColor: theme.palette.mode === 'dark' 
                ? 'rgba(255, 255, 255, 0.05)' 
                : 'rgba(0, 0, 0, 0.05)',
              transition: 'border-color 0.2s ease, background-color 0.2s ease, box-shadow 0.2s ease',
              '&:hover': {
                borderColor: theme.palette.primary.main,
                bgcolor: theme.palette.mode === 'dark' 
                  ? 'rgba(255, 77, 166, 0.05)' 
                  : 'rgba(255, 77, 166, 0.02)',
                boxShadow: theme.palette.mode === 'dark' 
                  ? '0 4px 12px rgba(255, 77, 166, 0.2)' 
                  : '0 4px 12px rgba(255, 77, 166, 0.1)'
              }
            }}>
              <CardContent sx={{ p: 2.5 }}>
                <Typography variant="body2" gutterBottom sx={{ 
                  color: theme.palette.text.secondary,
                  fontSize: '0.75rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>
                  Total Requests
                </Typography>
                <Typography variant="h5" sx={{ 
                  color: theme.palette.text.primary,
                  fontWeight: 700,
                  fontSize: '1.5rem'
                }}>
                  {stats.totalRequests}
                </Typography>
                <Typography variant="caption" sx={{ 
                  color: theme.palette.text.secondary,
                  fontSize: '0.7rem',
                  opacity: 0.8
                }}>
                  {stats.successfulRequests} successful
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card variant="outlined" sx={{ 
              bgcolor: theme.palette.mode === 'dark' 
                ? 'rgba(255, 255, 255, 0.03)' 
                : 'rgba(0, 0, 0, 0.02)',
              backdropFilter: 'blur(10px)',
              border: '1px solid',
              borderColor: theme.palette.mode === 'dark' 
                ? 'rgba(255, 255, 255, 0.05)' 
                : 'rgba(0, 0, 0, 0.05)',
              transition: 'border-color 0.2s ease, background-color 0.2s ease, box-shadow 0.2s ease',
              '&:hover': {
                borderColor: theme.palette.primary.main,
                bgcolor: theme.palette.mode === 'dark' 
                  ? 'rgba(255, 77, 166, 0.05)' 
                  : 'rgba(255, 77, 166, 0.02)',
                boxShadow: theme.palette.mode === 'dark' 
                  ? '0 4px 12px rgba(255, 77, 166, 0.2)' 
                  : '0 4px 12px rgba(255, 77, 166, 0.1)'
              }
            }}>
              <CardContent sx={{ p: 2.5 }}>
                <Typography variant="body2" gutterBottom sx={{ 
                  color: theme.palette.text.secondary,
                  fontSize: '0.75rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>
                  {modelInfo?.is_cloud ? 'Daily Tokens' : 'Average Tokens'}
                </Typography>
                <Typography variant="h5" sx={{ 
                  color: theme.palette.text.primary,
                  fontWeight: 700,
                  fontSize: '1.5rem'
                }}>
                  {modelInfo?.is_cloud && modelInfo?.usage_stats 
                    ? (modelInfo.usage_stats.daily_input_tokens + modelInfo.usage_stats.daily_output_tokens).toLocaleString()
                    : Math.round(stats.avgInputTokens + stats.avgOutputTokens).toLocaleString()
                  }
                </Typography>
                <Typography variant="caption" sx={{ 
                  color: theme.palette.text.secondary,
                  fontSize: '0.7rem',
                  opacity: 0.8
                }}>
                  {modelInfo?.is_cloud && modelInfo?.usage_stats
                    ? `In: ${modelInfo.usage_stats.daily_input_tokens.toLocaleString()} | Out: ${modelInfo.usage_stats.daily_output_tokens.toLocaleString()}`
                    : `In: ${Math.round(stats.avgInputTokens)} | Out: ${Math.round(stats.avgOutputTokens)}`
                  }
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card variant="outlined" sx={{ 
              bgcolor: theme.palette.mode === 'dark' 
                ? 'rgba(255, 255, 255, 0.03)' 
                : 'rgba(0, 0, 0, 0.02)',
              backdropFilter: 'blur(10px)',
              border: '1px solid',
              borderColor: theme.palette.mode === 'dark' 
                ? 'rgba(255, 255, 255, 0.05)' 
                : 'rgba(0, 0, 0, 0.05)',
              transition: 'border-color 0.2s ease, background-color 0.2s ease, box-shadow 0.2s ease',
              '&:hover': {
                borderColor: theme.palette.primary.main,
                bgcolor: theme.palette.mode === 'dark' 
                  ? 'rgba(255, 77, 166, 0.05)' 
                  : 'rgba(255, 77, 166, 0.02)',
                boxShadow: theme.palette.mode === 'dark' 
                  ? '0 4px 12px rgba(255, 77, 166, 0.2)' 
                  : '0 4px 12px rgba(255, 77, 166, 0.1)'
              }
            }}>
              <CardContent sx={{ p: 2.5 }}>
                <Typography variant="body2" gutterBottom sx={{ 
                  color: theme.palette.text.secondary,
                  fontSize: '0.75rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>
                  Total Cost
                </Typography>
                <Typography variant="h5" sx={{ 
                  color: theme.palette.text.primary,
                  fontWeight: 700,
                  fontSize: '1.5rem'
                }}>
                  ${stats.totalCost.toFixed(4)}
                </Typography>
                <Typography variant="caption" sx={{ 
                  color: theme.palette.text.secondary,
                  fontSize: '0.7rem',
                  opacity: 0.8
                }}>
                  Provider: {modelInfo?.provider || 'Unknown'}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
        
        {/* Monthly Average Tokens for Cloud Models */}
        {modelInfo?.is_cloud && modelInfo?.usage_stats && (
          <Grid container spacing={2} sx={{ mt: 2 }}>
            <Grid item xs={12}>
              <Typography variant="subtitle2" sx={{ 
                color: theme.palette.text.secondary,
                mb: 1,
                fontWeight: 500
              }}>
                Monthly Averages
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6}>
              <Card variant="outlined" sx={{ 
                bgcolor: theme.palette.background.paper,
                borderColor: theme.palette.divider,
                transition: 'all 0.3s ease'
              }}>
                <CardContent sx={{ py: 2 }}>
                  <Typography variant="body2" gutterBottom sx={{ 
                    color: theme.palette.text.secondary 
                  }}>
                    Average Monthly Tokens
                  </Typography>
                  <Typography variant="h6" sx={{ 
                    color: theme.palette.text.primary,
                    fontWeight: 600 
                  }}>
                    {(modelInfo.usage_stats.monthly_avg_input_tokens + modelInfo.usage_stats.monthly_avg_output_tokens).toLocaleString()}
                  </Typography>
                  <Typography variant="caption" sx={{ 
                    color: theme.palette.text.secondary 
                  }}>
                    In: {modelInfo.usage_stats.monthly_avg_input_tokens.toLocaleString()} | Out: {modelInfo.usage_stats.monthly_avg_output_tokens.toLocaleString()}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6}>
              <Card variant="outlined" sx={{ 
                bgcolor: theme.palette.background.paper,
                borderColor: theme.palette.divider,
                transition: 'all 0.3s ease'
              }}>
                <CardContent sx={{ py: 2 }}>
                  <Typography variant="body2" gutterBottom sx={{ 
                    color: theme.palette.text.secondary 
                  }}>
                    Average Monthly Cost
                  </Typography>
                  <Typography variant="h6" sx={{ 
                    color: theme.palette.text.primary,
                    fontWeight: 600 
                  }}>
                    ${modelInfo.usage_stats.monthly_avg_cost.toFixed(2)}
                  </Typography>
                  <Typography variant="caption" sx={{ 
                    color: theme.palette.text.secondary 
                  }}>
                    Daily avg: ${(modelInfo.usage_stats.monthly_avg_cost / 30).toFixed(2)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        )}
      </Grid>

      {/* Latency History Graph */}
      <Grid item xs={12}>
        <Typography variant="h6" gutterBottom sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          mt: 4,
          mb: 3,
          color: theme.palette.text.primary
        }}>
          <HistoryIcon sx={{ mr: 1, color: theme.palette.primary.main }} />
          Recent History
        </Typography>
        <Box sx={{ 
          p: 3, 
          bgcolor: theme.palette.mode === 'dark' 
            ? 'rgba(255, 255, 255, 0.03)' 
            : 'rgba(0, 0, 0, 0.02)',
          backdropFilter: 'blur(10px)',
          border: '1px solid',
          borderColor: theme.palette.mode === 'dark' 
            ? 'rgba(255, 255, 255, 0.05)' 
            : 'rgba(0, 0, 0, 0.05)',
          borderRadius: 2,
        }}>
          <ModelHistoryGraph data={modelHistory} theme={theme} />
        </Box>
      </Grid>
    </Grid>
  );
};

// Model History Graph Component
interface ModelHistoryGraphProps {
  data: LatencyRecord[];
  theme: any;
}

const ModelHistoryGraph: React.FC<ModelHistoryGraphProps> = ({ data, theme }) => {
  const chartData = useMemo(() => {
    return data
      .filter(r => r.latency_ms !== null)
      .map(record => ({
        timestamp: new Date(record.timestamp).getTime(),
        latency: record.latency_ms ? record.latency_ms / 1000 : 0,
        tokens: (record.input_tokens || 0) + (record.output_tokens || 0),
      }))
      .sort((a, b) => a.timestamp - b.timestamp);
  }, [data]);

  const formatXAxis = (tickItem: number) => {
    return new Date(tickItem).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <Box
          sx={{
            bgcolor: 'background.paper',
            p: 1.5,
            borderRadius: 1,
            boxShadow: 2,
            border: '1px solid',
            borderColor: 'divider',
          }}
        >
          <Typography variant="caption" color="text.secondary">
            {new Date(label).toLocaleString()}
          </Typography>
          {payload.map((entry: any) => (
            <Box key={entry.name} sx={{ mt: 0.5 }}>
              <Typography variant="caption" sx={{ color: entry.color }}>
                Latency: {entry.value.toFixed(2)}s
              </Typography>
            </Box>
          ))}
        </Box>
      );
    }
    return null;
  };

  if (chartData.length === 0) {
    return (
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <Typography variant="body2" color="text.secondary">
          No history data available
        </Typography>
      </Box>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart
        data={chartData}
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
        />
        <YAxis
          stroke={theme.palette.primary.main}
          label={{ 
            value: 'Latency (s)', 
            angle: -90, 
            position: 'insideLeft',
            style: { fill: theme.palette.text.secondary }
          }}
        />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: theme.palette.mode === 'dark' 
              ? 'rgba(26, 26, 26, 0.95)' 
              : 'rgba(255, 255, 255, 0.95)',
            backdropFilter: 'blur(10px)',
            border: `1px solid ${theme.palette.mode === 'dark' 
              ? 'rgba(255, 255, 255, 0.1)' 
              : 'rgba(0, 0, 0, 0.05)'}`,
            borderRadius: 8,
            boxShadow: theme.palette.mode === 'dark'
              ? '0 4px 24px rgba(0, 0, 0, 0.4)'
              : '0 4px 24px rgba(0, 0, 0, 0.1)',
          }}
          labelFormatter={formatXAxis}
          formatter={(value: number) => [`${value.toFixed(2)}s`, 'Latency']}
          cursor={{ fill: theme.palette.mode === 'dark' 
            ? 'rgba(255, 255, 255, 0.02)' 
            : 'rgba(0, 0, 0, 0.02)' 
          }}
        />
        <Line
          type="monotone"
          dataKey="latency"
          stroke={theme.palette.primary.main}
          strokeWidth={3}
          dot={false}
          connectNulls
          strokeLinecap="round"
          activeDot={{ 
            r: 6, 
            strokeWidth: 0,
            fill: theme.palette.primary.main,
            filter: theme.palette.mode === 'dark' 
              ? 'drop-shadow(0 0 8px rgba(255, 77, 166, 0.5))' 
              : 'drop-shadow(0 0 8px rgba(255, 77, 166, 0.3))'
          }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default LatencyTable; 