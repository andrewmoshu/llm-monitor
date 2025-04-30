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
  ChipProps,
  Box,
  Tooltip,
  useTheme,
} from '@mui/material';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import TimerOffIcon from '@mui/icons-material/TimerOff';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import { LatencyRecord, ModelInfo } from '../types/types';

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

  return (
    <Box sx={{ width: '100%', overflow: 'hidden', mt: 2 }}>
      <TableContainer component={Paper} elevation={2} sx={{ borderRadius: 2 }}>
        <Table stickyHeader size="small">
          <TableHead>
            <TableRow>
              <TableCell
                sx={{
                  bgcolor: theme.palette.mode === 'dark' 
                    ? 'rgba(0, 0, 0, 0.3)'
                    : 'rgba(0, 0, 0, 0.02)',
                  fontWeight: 600,
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
                  bgcolor: theme.palette.mode === 'dark' 
                    ? 'rgba(0, 0, 0, 0.3)'
                    : 'rgba(0, 0, 0, 0.02)',
                  fontWeight: 600,
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
                  bgcolor: theme.palette.mode === 'dark' 
                    ? 'rgba(0, 0, 0, 0.3)'
                    : 'rgba(0, 0, 0, 0.02)',
                  fontWeight: 600,
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
                  bgcolor: theme.palette.mode === 'dark' 
                    ? 'rgba(0, 0, 0, 0.3)'
                    : 'rgba(0, 0, 0, 0.02)',
                  fontWeight: 600,
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
                  bgcolor: theme.palette.mode === 'dark' 
                    ? 'rgba(0, 0, 0, 0.3)'
                    : 'rgba(0, 0, 0, 0.02)',
                  fontWeight: 600,
                }}
              >
                <TableSortLabel
                  active={sortField === 'cost'}
                  direction={sortField === 'cost' ? sortOrder : 'asc'}
                  onClick={() => handleSort('cost')}
                >
                  Cost ($/req)
                </TableSortLabel>
              </TableCell>
              <TableCell
                sx={{
                  bgcolor: theme.palette.mode === 'dark' 
                    ? 'rgba(0, 0, 0, 0.3)'
                    : 'rgba(0, 0, 0, 0.02)',
                  fontWeight: 600,
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
                  bgcolor: theme.palette.mode === 'dark' 
                    ? 'rgba(0, 0, 0, 0.3)'
                    : 'rgba(0, 0, 0, 0.02)',
                  fontWeight: 600,
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
              const statusInfo = getStatusChip(effectiveStatus);
              
              // Determine the timestamp to display in the tooltip
              // Use the actual record timestamp if it exists and is not the placeholder timestamp
              const placeholderTimestamp = new Date(0).toISOString();
              const displayTimestamp = record.timestamp !== placeholderTimestamp ? record.timestamp : null;

              return (
                <TableRow 
                  key={record.model_name} 
                  hover
                  sx={{
                    '&:nth-of-type(even)': {
                      bgcolor: theme.palette.mode === 'dark' 
                        ? 'rgba(255, 255, 255, 0.03)'
                        : 'rgba(0, 0, 0, 0.02)',
                    },
                    '&:hover': {
                      bgcolor: theme.palette.mode === 'dark'
                        ? 'rgba(255, 255, 255, 0.1) !important'
                        : 'rgba(0, 0, 0, 0.04) !important',
                    },
                  }}
                >
                  <TableCell>
                    <Typography variant="body2" fontWeight="medium">
                      {record.model_name}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Tooltip 
                      title={`Last check: ${displayTimestamp ? new Date(displayTimestamp).toLocaleString() : 'N/A'}`}
                      arrow
                    >
                      <Chip
                        icon={statusInfo.icon}
                        label={statusInfo.label}
                        color={statusInfo.color as ChipProps['color']}
                        size="small"
                        sx={{ minWidth: 90, textTransform: 'capitalize' }}
                      />
                    </Tooltip>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={`${formatContextWindow(record.context_window)}`}
                      size="small"
                      color="info"
                      variant="outlined"
                      sx={{ minWidth: 60, fontWeight: 500 }}
                    />
                  </TableCell>
                  <TableCell>
                    {record.latency_ms !== null ? (
                      <Tooltip title={`${record.latency_ms.toFixed(0)}ms`}>
                        <Chip
                          label={`${(record.latency_ms / 1000).toFixed(2)}s`}
                          color={getLatencyColor(record.latency_ms)}
                          size="small"
                          sx={{ minWidth: 60, fontWeight: 500 }}
                        />
                      </Tooltip>
                    ) : (
                      <Typography variant="caption" color="textSecondary">N/A</Typography>
                    )}
                  </TableCell>
                  <TableCell>
                    {record.cost !== null ? (
                       <Typography variant="body2" color="textSecondary">
                         ${record.cost.toFixed(4)}
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
                    <Chip
                      label={record.is_cloud ? 'Cloud' : 'On-Prem'}
                      size="small"
                      color={record.is_cloud ? 'secondary' : 'primary'}
                      variant="outlined"
                      sx={{ minWidth: 70, fontWeight: 500 }}
                    />
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
};

export default LatencyTable; 