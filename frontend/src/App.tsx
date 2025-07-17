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
  InputBase,
  Button,
  alpha,
  Fade,
  Slide,
  Skeleton,
  Grow,
  keyframes,
} from '@mui/material';
import SpeedIcon from '@mui/icons-material/Speed';
import SearchIcon from '@mui/icons-material/Search';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import LatencyGraph from './components/LatencyGraph';
import LatencyTable from './components/LatencyTable';
import EnvironmentSwitcher from './components/EnvironmentSwitcherModern';
import { api } from './api';
import { LatencyRecord, ModelInfo, Environment } from './types/types';
import './App.css';

const API_BASE = 'http://localhost:8000/api';

// Assume the backend monitor interval is 60 seconds for calculation
const MONITOR_INTERVAL_MS = 60 * 1000;
// Tolerance window around the latest timestamp (e.g., half the interval)
const TIMESTAMP_TOLERANCE_MS = MONITOR_INTERVAL_MS / 2;

// Beautiful keyframe animations
const shimmer = keyframes`
  0% {
    background-position: -200% center;
  }
  100% {
    background-position: 200% center;
  }
`;

const pulse = keyframes`
  0% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(1.05);
    opacity: 0.8;
  }
  100% {
    transform: scale(1);
    opacity: 1;
  }
`;

const float = keyframes`
  0% {
    transform: translateY(0px);
  }
  50% {
    transform: translateY(-10px);
  }
  100% {
    transform: translateY(0px);
  }
`;

const glow = keyframes`
  0% {
    box-shadow: 0 0 5px rgba(255, 77, 166, 0.5);
  }
  50% {
    box-shadow: 0 0 20px rgba(255, 77, 166, 0.8), 0 0 40px rgba(255, 77, 166, 0.4);
  }
  100% {
    box-shadow: 0 0 5px rgba(255, 77, 166, 0.5);
  }
`;

function App() {
  const [latencyData, setLatencyData] = useState<LatencyRecord[]>([]);
  const [modelInfo, setModelInfo] = useState<{ [key: string]: ModelInfo }>({});
  const [loading, setLoading] = useState<boolean>(true);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const darkMode = true; // Always use dark mode
  const [currentEnvironment, setCurrentEnvironment] = useState<Environment>('dev');
  const [isTransitioning, setIsTransitioning] = useState<boolean>(false);

  const theme = createTheme({
    palette: {
      mode: 'dark',
      primary: {
        main: '#ff4da6',
        light: '#ff80cc',
        dark: '#cc0066',
      },
      secondary: {
        main: '#00bfff',
        light: '#66d9ff',
        dark: '#0099cc',
      },
      background: {
        default: darkMode ? '#0a0a0a' : '#f5f5f7',
        paper: darkMode ? '#1a1a1a' : '#ffffff',
      },
      text: {
        primary: darkMode ? '#ffffff' : '#000000',
        secondary: darkMode ? 'rgba(255, 255, 255, 0.7)' : 'rgba(0, 0, 0, 0.6)',
      },
    },
    shape: {
      borderRadius: 16,
    },
    components: {
      MuiPaper: {
        styleOverrides: {
          root: {
            backgroundImage: darkMode 
              ? 'linear-gradient(to bottom right, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02))'
              : 'none',
            backdropFilter: darkMode ? 'blur(20px)' : 'none',
            border: darkMode ? '1px solid rgba(255, 255, 255, 0.1)' : 'none',
            boxShadow: darkMode 
              ? '0 4px 12px rgba(0, 0, 0, 0.2)'
              : '0 2px 8px rgba(0, 0, 0, 0.05)',
          },
        },
      },
      MuiButton: {
        styleOverrides: {
          root: {
            textTransform: 'none',
            fontWeight: 600,
            borderRadius: 8,
          },
        },
      },
    },
  });

  useEffect(() => {
    const fetchData = async (isInitialLoad = false, isEnvironmentChange = false) => {
      if (isInitialLoad) {
        setLoading(true);
      } else if (isEnvironmentChange) {
        setIsTransitioning(true);
      } else {
        setIsRefreshing(true);
      }
      setError(null);

      try {
        // Add a small delay for environment changes to make the transition smoother
        if (isEnvironmentChange) {
          await new Promise(resolve => setTimeout(resolve, 300));
        }

        const [latencyJson, modelsJson] = await Promise.all([
          api.getLatencyData(currentEnvironment),
          api.getModels(currentEnvironment)
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
        if (isEnvironmentChange) {
          setTimeout(() => setIsTransitioning(false), 100);
        }
      }
    };

    // Check if this is an environment change (not initial load)
    const isEnvironmentChange = latencyData.length > 0 && !loading;
    
    fetchData(!latencyData.length, isEnvironmentChange);
    const intervalId = setInterval(() => fetchData(false, false), MONITOR_INTERVAL_MS);

    return () => clearInterval(intervalId);
  }, [currentEnvironment]);


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
        pb: 4,
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a0a1a 50%, #0a0a1a 100%)'
      }}>
        <AppBar 
          position="static" 
          color="transparent" 
          elevation={0}
          sx={{ 
            background: 'transparent',
            boxShadow: 'none',
            border: 'none',
            '&::before': {
              display: 'none',
            },
          }}
        >
          <Toolbar sx={{ py: 2, px: { xs: 2, sm: 3, md: 4 } }}>
            <Box sx={{ 
              display: 'flex', 
              alignItems: 'center',
              mr: 2,
            }}>
              <SpeedIcon sx={{ 
                color: 'primary.main', 
                fontSize: 32,
              }} />
            </Box>
            
            <Box sx={{ flexGrow: 1 }}>
              <Typography 
                variant="h5" 
                component="div" 
                sx={{ 
                  fontWeight: 700,
                  color: 'white',
                  letterSpacing: '-0.5px',
                }}
              >
                LLM Performance Monitor
              </Typography>
              <Typography 
                variant="caption" 
                sx={{ 
                  color: 'text.secondary',
                  display: 'block',
                  mt: -0.5,
                  opacity: 0.8,
                }}
              >
                Real-time model analytics and monitoring
              </Typography>
            </Box>
            
            <Box sx={{ ml: 'auto', display: 'flex', alignItems: 'center', gap: 3 }}>
              <EnvironmentSwitcher 
                currentEnvironment={currentEnvironment}
                onEnvironmentChange={setCurrentEnvironment}
              />
              
              {isRefreshing && (
                <Box sx={{ 
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1,
                }}>
                  <CircularProgress 
                    size={20} 
                    thickness={5}
                    sx={{ 
                      color: 'primary.main',
                      '& .MuiCircularProgress-circle': {
                        strokeLinecap: 'round',
                      },
                    }} 
                  />
                  <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                    Updating...
                  </Typography>
                </Box>
              )}
            </Box>
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl" sx={{ mt: 4 }}>
          {/* Environment Indicator */}
          <Box sx={{ 
            mb: 3,
            display: 'flex',
            alignItems: 'center',
            gap: 2,
            perspective: '1000px',
          }}>
            <Box sx={{
              position: 'relative',
              px: 2,
              py: 0.5,
              borderRadius: 2,
              background: currentEnvironment === 'prod' 
                ? 'linear-gradient(135deg, #f44336 0%, #ff6b6b 100%)'
                : currentEnvironment === 'qa'
                ? 'linear-gradient(135deg, #ff9800 0%, #ffc947 100%)'
                : currentEnvironment === 'test'
                ? 'linear-gradient(135deg, #2196f3 0%, #64b5f6 100%)'
                : 'linear-gradient(135deg, #4caf50 0%, #81c784 100%)',
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
              transition: 'opacity 0.2s ease',
              opacity: isTransitioning ? 0.7 : 1,
            }}>
              <Typography variant="caption" sx={{ 
                color: 'white',
                fontWeight: 600,
                textTransform: 'uppercase',
                letterSpacing: '1px',
                fontSize: '0.7rem',
                position: 'relative',
                zIndex: 1,
              }}>
                {currentEnvironment} Environment
              </Typography>
            </Box>
            <Fade in={!isTransitioning} timeout={300}>
              <Typography variant="body2" sx={{ 
                color: 'text.secondary',
                opacity: isTransitioning ? 0 : 1,
                transition: 'all 0.3s ease',
              }}>
                Monitoring {Object.keys(modelInfo).length} models
              </Typography>
            </Fade>
          </Box>
          
          <Box sx={{ position: 'relative' }}>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Fade in={!isTransitioning} timeout={300} style={{ transitionDelay: isTransitioning ? '0ms' : '50ms' }}>
                  <Paper 
                    sx={{ 
                      p: 3,
                      position: 'relative',
                      background: darkMode 
                        ? 'rgba(26, 26, 26, 0.6)'
                        : 'rgba(255, 255, 255, 0.9)',
                      backdropFilter: 'blur(20px)',
                      border: '1px solid',
                      borderColor: darkMode 
                        ? 'rgba(255, 255, 255, 0.1)' 
                        : 'rgba(0, 0, 0, 0.05)',
                      boxShadow: darkMode
                        ? '0 4px 12px rgba(0, 0, 0, 0.2)'
                        : '0 2px 8px rgba(0, 0, 0, 0.05)',
                      opacity: isTransitioning ? 0.7 : 1,
                      transition: 'opacity 0.2s ease',
                      overflow: 'hidden',
                      '&::before': {
                        content: '""',
                        position: 'absolute',
                        top: 0,
                        left: '-100%',
                        width: '100%',
                        height: '100%',
                        background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent)',
                        animation: isTransitioning ? 'none' : `${shimmer} 3s ease-in-out infinite`,
                      },
                      '&:hover': {
                        boxShadow: darkMode
                          ? '0 6px 16px rgba(0, 0, 0, 0.3)'
                          : '0 4px 12px rgba(0, 0, 0, 0.08)',
                      }
                    }}
                  >
                    {isTransitioning ? (
                      <Box>
                        <Skeleton 
                          variant="text" 
                          width="40%" 
                          height={32} 
                          sx={{ 
                            mb: 2,
                            background: 'linear-gradient(90deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.1) 50%, rgba(255,255,255,0.05) 100%)',
                            backgroundSize: '200% 100%',
                            animation: `${shimmer} 1.5s ease-in-out infinite`,
                          }} 
                        />
                        <Skeleton 
                          variant="rectangular" 
                          height={400} 
                          sx={{ 
                            borderRadius: 2,
                            background: 'linear-gradient(90deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.1) 50%, rgba(255,255,255,0.05) 100%)',
                            backgroundSize: '200% 100%',
                            animation: `${shimmer} 1.5s ease-in-out infinite`,
                          }} 
                        />
                      </Box>
                    ) : (
                      <LatencyGraph data={latencyData} modelInfo={modelInfo} />
                    )}
                  </Paper>
                </Fade>
              </Grid>
              <Grid item xs={12}>
                <Fade in={!isTransitioning} timeout={300} style={{ transitionDelay: isTransitioning ? '0ms' : '100ms' }}>
                  <Paper 
                    sx={{ 
                      p: 3,
                      position: 'relative',
                      background: darkMode 
                        ? 'rgba(26, 26, 26, 0.6)'
                        : 'rgba(255, 255, 255, 0.9)',
                      backdropFilter: 'blur(20px)',
                      border: '1px solid',
                      borderColor: darkMode 
                        ? 'rgba(255, 255, 255, 0.1)' 
                        : 'rgba(0, 0, 0, 0.05)',
                      boxShadow: darkMode
                        ? '0 4px 12px rgba(0, 0, 0, 0.2)'
                        : '0 2px 8px rgba(0, 0, 0, 0.05)',
                      opacity: isTransitioning ? 0.7 : 1,
                      transition: 'opacity 0.2s ease 0.1s',
                      overflow: 'hidden',
                      '&::before': {
                        content: '""',
                        position: 'absolute',
                        top: 0,
                        right: '-100%',
                        width: '100%',
                        height: '100%',
                        background: 'linear-gradient(-90deg, transparent, rgba(255, 255, 255, 0.08), transparent)',
                        animation: isTransitioning ? 'none' : `${shimmer} 3s ease-in-out infinite 1.5s`,
                      },
                      '&:hover': {
                        boxShadow: darkMode
                          ? '0 6px 16px rgba(0, 0, 0, 0.3)'
                          : '0 4px 12px rgba(0, 0, 0, 0.08)',
                      }
                    }}
                  >
                    {isTransitioning ? (
                      <Box>
                        <Skeleton 
                          variant="text" 
                          width="30%" 
                          height={32} 
                          sx={{ 
                            mb: 2,
                            background: 'linear-gradient(90deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.1) 50%, rgba(255,255,255,0.05) 100%)',
                            backgroundSize: '200% 100%',
                            animation: `${shimmer} 1.5s ease-in-out infinite`,
                          }}
                        />
                        <Skeleton 
                          variant="rectangular" 
                          height={300} 
                          sx={{ 
                            borderRadius: 2,
                            background: 'linear-gradient(90deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.1) 50%, rgba(255,255,255,0.05) 100%)',
                            backgroundSize: '200% 100%',
                            animation: `${shimmer} 1.5s ease-in-out infinite`,
                          }}
                        />
                      </Box>
                    ) : (
                      <LatencyTable data={latencyData} modelInfo={modelInfo} />
                    )}
                  </Paper>
                </Fade>
              </Grid>
            </Grid>
          </Box>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;
