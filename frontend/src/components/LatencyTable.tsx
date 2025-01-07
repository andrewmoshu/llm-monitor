import React, { useState } from 'react';
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
  Tooltip,
  useTheme,
} from '@mui/material';
import { LatencyRecord } from '../types/types';

interface Props {
  data: LatencyRecord[];
}

type SortField = 'model_name' | 'latency_ms' | 'cost' | 'input_tokens' | 'output_tokens' | 'arena_score' | 'context_window';
type SortOrder = 'asc' | 'desc';

const LatencyTable: React.FC<Props> = ({ data }) => {
  const theme = useTheme();
  const [sortField, setSortField] = useState<SortField>('latency_ms');
  const [sortOrder, setSortOrder] = useState<SortOrder>('asc');

  const latestRecords = data.reduce((acc, record) => {
    if (!acc[record.model_name] || 
        new Date(record.timestamp) > new Date(acc[record.model_name].timestamp)) {
      acc[record.model_name] = record;
    }
    return acc;
  }, {} as Record<string, LatencyRecord>);

  console.log('Latest records:', latestRecords);

  const handleSort = (field: SortField) => {
    if (field === sortField) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortOrder('asc');
    }
  };

  const sortedRecords = Object.values(latestRecords).sort((a, b) => {
    const multiplier = sortOrder === 'asc' ? 1 : -1;
    if (sortField === 'model_name') {
      return multiplier * a.model_name.localeCompare(b.model_name);
    }
    return multiplier * ((a[sortField] || 0) - (b[sortField] || 0));
  });

  const getLatencyColor = (latency: number) => {
    if (latency < 2000) return 'success';
    if (latency < 5000) return 'warning';
    return 'error';
  };

  const formatContextWindow = (tokens: number | undefined) => {
    if (tokens === undefined) return 'N/A';
    
    if (tokens >= 1000000) {
      return `${(tokens / 1000000).toFixed(1)}M`;
    }
    if (tokens >= 1000) {
      return `${(tokens / 1000).toFixed(0)}K`;
    }
    return tokens.toString();
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom sx={{ fontWeight: 500 }}>
        Latest Model Statistics
      </Typography>
      <TableContainer>
        <Table>
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
                  direction={sortOrder}
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
                  active={sortField === 'context_window'}
                  direction={sortOrder}
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
                  direction={sortOrder}
                  onClick={() => handleSort('latency_ms')}
                >
                  Latency
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
                  direction={sortOrder}
                  onClick={() => handleSort('cost')}
                >
                  Cost per 1K tokens
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
                Tokens (In/Out)
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
                  active={sortField === 'arena_score'}
                  direction={sortOrder}
                  onClick={() => handleSort('arena_score')}
                >
                  Arena Score
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
            {sortedRecords.map((record, index) => {
              console.log('Record context window:', record.model_name, record.context_window);
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
                    <Chip
                      label={`${formatContextWindow(record.context_window)} tokens`}
                      size="small"
                      color="info"
                      variant="outlined"
                      sx={{ minWidth: 80 }}
                    />
                  </TableCell>
                  <TableCell>
                    <Tooltip title={`${record.latency_ms.toFixed(2)}ms`}>
                      <Chip
                        label={`${(record.latency_ms / 1000).toFixed(2)}s`}
                        color={getLatencyColor(record.latency_ms)}
                        size="small"
                        sx={{ minWidth: 70 }}
                      />
                    </Tooltip>
                  </TableCell>
                  <TableCell>
                    <Typography color="primary">
                      ${record.cost.toFixed(4)}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {record.input_tokens} / {record.output_tokens}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    {record.arena_score ? (
                      <Chip
                        label={record.arena_score.toFixed(1)}
                        color={record.arena_score > 8 ? 'success' : 'warning'}
                        size="small"
                        sx={{ minWidth: 45 }}
                      />
                    ) : (
                      'N/A'
                    )}
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={record.is_cloud ? 'Cloud' : 'On-Prem'}
                      size="small"
                      color={record.is_cloud ? 'primary' : 'success'}
                      variant="outlined"
                      sx={{ minWidth: 80 }}
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