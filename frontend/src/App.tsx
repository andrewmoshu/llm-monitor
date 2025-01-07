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
import { LatencyRecord } from './types/types';
import './App.css';

function App() {
  const [latencyData, setLatencyData] = useState<LatencyRecord[]>([]);
  const [loading, setLoading] = useState(true);
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
    const fetchData = async () => {
      try {
        const data = await api.getLatencyData();
        console.log('Received data:', data);
        setLatencyData(data);
        setLoading(false);
      } catch (err) {
        setError('Failed to fetch data');
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  const getStats = () => {
    if (!latencyData.length) return { 
      models: 0, 
      cloudLatency: 0,
      onPremLatency: 0 
    };
    
    const uniqueModels = new Set(latencyData.map(d => d.model_name));
    
    // Calculate average latency for cloud models
    const cloudRecords = latencyData.filter(d => d.is_cloud);
    const cloudLatency = cloudRecords.length 
      ? cloudRecords.reduce((sum, d) => sum + d.latency_ms, 0) / cloudRecords.length 
      : 0;

    // Calculate average latency for on-prem models
    const onPremRecords = latencyData.filter(d => !d.is_cloud);
    const onPremLatency = onPremRecords.length 
      ? onPremRecords.reduce((sum, d) => sum + d.latency_ms, 0) / onPremRecords.length 
      : 0;

    return {
      models: uniqueModels.size,
      cloudLatency,
      onPremLatency
    };
  };

  const stats = getStats();

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
                  {stats.models}
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
                  {(stats.cloudLatency / 1000).toFixed(2)}
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
                  {(stats.onPremLatency / 1000).toFixed(2)}
                  <Typography component="span" variant="h6" color="text.secondary"> s</Typography>
                </Typography>
              </Paper>
            </Grid>
          </Grid>

          {/* Main Content */}
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Paper sx={{ p: 3 }}>
                <LatencyGraph data={latencyData} />
              </Paper>
            </Grid>
            <Grid item xs={12}>
              <Paper sx={{ p: 3 }}>
                <LatencyTable data={latencyData} />
              </Paper>
            </Grid>
          </Grid>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;
