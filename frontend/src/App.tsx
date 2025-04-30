import React, { useEffect, useState } from 'react';
import { 
  ThemeProvider, 
  createTheme, 
  CssBaseline,
  Container,
  Box,
  AppBar,
  Toolbar,
  Typography,
  Paper,
  Grid,
  CircularProgress,
  Alert,
  IconButton,
  Chip,
  Divider,
} from '@mui/material';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import SpeedIcon from '@mui/icons-material/Speed';
import QueryStatsIcon from '@mui/icons-material/QueryStats';
import LatencyGraph from './components/LatencyGraph';
import LatencyTable from './components/LatencyTable';
import { api } from './api';
import { LatencyRecord, ModelInfo } from './types/types';
import './App.css';
import StatCard from './components/StatCard';

const API_BASE = 'http://localhost:8001/api';

// Assume the backend monitor interval is 60 seconds for calculation
const MONITOR_INTERVAL_MS = 60 * 1000;
// Tolerance window around the latest timestamp (e.g., half the interval)
const TIMESTAMP_TOLERANCE_MS = MONITOR_INTERVAL_MS / 2;

function App() {
  const [latencyData, setLatencyData] = useState<LatencyRecord[]>([]);
  const [modelInfo, setModelInfo] = useState<{ [key: string]: ModelInfo }>({});
  const [loading, setLoading] = useState<boolean>(true);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [darkMode, setDarkMode] = useState(false);

  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: '#2196f3',
      },
      secondary: {
        main: '#f50057',
      },
      background: {
        default: darkMode ? '#0a1929' : '#f5f5f7',
        paper: darkMode ? '#0d2339' : '#ffffff',
      },
    },
    shape: {
      borderRadius: 12,
    },
    components: {
      MuiPaper: {
        styleOverrides: {
          root: {
            backgroundImage: 'none',
            boxShadow: darkMode 
              ? '0 4px 6px rgba(0, 0, 0, 0.2)'
              : '0 4px 6px rgba(0, 0, 0, 0.05)',
          },
        },
      },
    },
  });

  useEffect(() => {
    const fetchData = async (isInitialLoad = false) => {
      if (isInitialLoad) {
        setLoading(true);
      } else {
        setIsRefreshing(true);
      }
      setError(null);

      try {
        const [latencyJson, modelsJson] = await Promise.all([
          api.getLatencyData(),
          api.getModels()
        ]);

        console.log("Fetched Latency Data:", latencyJson);
        console.log("Fetched Model Info:", modelsJson);

        setLatencyData(latencyJson);
        setModelInfo(modelsJson);

      } catch (e: any) {
        console.error("Failed to fetch data:", e);
        if (isInitialLoad) {
          setError(`Failed to load initial data: ${e.message || 'Unknown error'}`);
        } else {
          console.error("Failed to refresh data:", e);
        }
      } finally {
        setLoading(false);
        setIsRefreshing(false);
      }
    };

    fetchData(true);
    const intervalId = setInterval(() => fetchData(false), MONITOR_INTERVAL_MS);

    return () => clearInterval(intervalId);
  }, []);

  const calculateStats = (data: LatencyRecord[]) => {
    if (!data || data.length === 0) {
      return { cloudLatency: 0, onPremLatency: 0, monitoredInLastCycleCount: 0, uniqueModelCount: 0 };
    }

    // Calculate unique model count
    const uniqueModelNames = new Set(data.map(d => d.model_name));
    const uniqueModelCount = uniqueModelNames.size;

    const validLatencyRecords = data.filter(d => d.latency_ms !== null && d.latency_ms >= 0);
    const cloudRecords = validLatencyRecords.filter(d => d.is_cloud);
    const cloudLatency = cloudRecords.length
      ? cloudRecords.reduce((sum, d) => sum + (d.latency_ms ?? 0), 0) / cloudRecords.length
      : 0;

    const onPremRecords = validLatencyRecords.filter(d => !d.is_cloud);
    const onPremLatency = onPremRecords.length
      ? onPremRecords.reduce((sum, d) => sum + (d.latency_ms ?? 0), 0) / onPremRecords.length
      : 0;

    let monitoredInLastCycleCount = 0;
    try {
        // 1. Find the absolute latest timestamp (as epoch milliseconds)
        const maxTimestampEpoch = Math.max(...data.map(r => new Date(r.timestamp).getTime()));

        // 2. Find the latest timestamp for each model
        const latestTimestampPerModel = new Map<string, number>();
        data.forEach(record => {
            const modelName = record.model_name;
            const recordTimestampEpoch = new Date(record.timestamp).getTime();
            if (!latestTimestampPerModel.has(modelName) || recordTimestampEpoch > (latestTimestampPerModel.get(modelName) ?? 0)) {
                latestTimestampPerModel.set(modelName, recordTimestampEpoch);
            }
        });

        // 3. Count models whose latest timestamp is within the tolerance window
        latestTimestampPerModel.forEach((modelTimestampEpoch) => {
            if (maxTimestampEpoch - modelTimestampEpoch <= TIMESTAMP_TOLERANCE_MS) {
                monitoredInLastCycleCount++;
            }
        });
         console.log(`Max Timestamp: ${new Date(maxTimestampEpoch).toISOString()}, Models in last cycle: ${monitoredInLastCycleCount}`);

    } catch (e) {
        console.error("Error calculating last monitored count:", e);
        // Fallback or default value if calculation fails
        monitoredInLastCycleCount = 0; // Or maybe use the previous method as fallback?
    }

    return {
      cloudLatency: cloudLatency / 1000, // Convert ms to s
      onPremLatency: onPremLatency / 1000, // Convert ms to s
      monitoredInLastCycleCount: monitoredInLastCycleCount, // Use the new count
      uniqueModelCount: uniqueModelCount
    };
  };

  const stats = calculateStats(latencyData);

  if (loading) return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh',
        bgcolor: 'background.default' 
      }}>
        <CircularProgress />
      </Box>
    </ThemeProvider>
  );

  if (error) return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ p: 3, bgcolor: 'background.default' }}>
        <Alert severity="error" variant="filled">{error}</Alert>
      </Box>
    </ThemeProvider>
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ 
        minHeight: '100vh',
        bgcolor: 'background.default',
        pb: 4
      }}>
        <AppBar 
          position="static" 
          color="transparent" 
          elevation={0}
          sx={{ 
            backdropFilter: 'blur(10px)',
            borderBottom: '1px solid',
            borderColor: 'divider',
          }}
        >
          <Toolbar>
            <SpeedIcon sx={{ mr: 2, color: 'primary.main' }} />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1, fontWeight: 600 }}>
              LLM Performance Monitor
            </Typography>
            {isRefreshing && <CircularProgress size={24} color="inherit" sx={{ mr: 2 }} />}
            <IconButton onClick={() => setDarkMode(!darkMode)} color="inherit">
              {darkMode ? <Brightness7Icon /> : <Brightness4Icon />}
            </IconButton>
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl" sx={{ mt: 4 }}>
          {/* Stats Overview */}
          <Grid container spacing={3} sx={{ mb: 4 }}>
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 3, height: '100%' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <QueryStatsIcon sx={{ color: 'primary.main', mr: 1 }} />
                  <Typography variant="h6" sx={{ fontWeight: 500 }}>Models Monitored</Typography>
                </Box>
                <Typography variant="h3" sx={{ fontWeight: 600 }}>
                  {stats.uniqueModelCount}
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 3, height: '100%' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <SpeedIcon sx={{ color: 'warning.main', mr: 1 }} />
                  <Typography variant="h6" sx={{ fontWeight: 500 }}>Cloud Latency</Typography>
                </Box>
                <Typography variant="h3" sx={{ fontWeight: 600 }}>
                  {stats.cloudLatency.toFixed(2)}
                  <Typography component="span" variant="h6" color="text.secondary"> s</Typography>
                </Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 3, height: '100%' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <SpeedIcon sx={{ color: 'success.main', mr: 1 }} />
                  <Typography variant="h6" sx={{ fontWeight: 500 }}>On-Prem Latency</Typography>
                </Box>
                <Typography variant="h3" sx={{ fontWeight: 600 }}>
                  {stats.onPremLatency.toFixed(2)}
                  <Typography component="span" variant="h6" color="text.secondary"> s</Typography>
                </Typography>
              </Paper>
            </Grid>
          </Grid>

          {/* Main Content */}
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Paper sx={{ p: 3 }}>
                <LatencyGraph data={latencyData} modelInfo={modelInfo} />
              </Paper>
            </Grid>
            <Grid item xs={12}>
              <Paper sx={{ p: 3 }}>
                <LatencyTable data={latencyData} modelInfo={modelInfo} />
              </Paper>
            </Grid>
          </Grid>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;
