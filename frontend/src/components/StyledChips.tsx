import React from 'react';
import { Box, Typography, useTheme, alpha, keyframes } from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import PauseCircleIcon from '@mui/icons-material/PauseCircle';
import HelpIcon from '@mui/icons-material/Help';
import SpeedIcon from '@mui/icons-material/Speed';
import DataUsageIcon from '@mui/icons-material/DataUsage';
import CloudIcon from '@mui/icons-material/Cloud';
import ComputerIcon from '@mui/icons-material/Computer';
import FiberManualRecordIcon from '@mui/icons-material/FiberManualRecord';

// Elegant animations
const shimmer = keyframes`
  0% {
    background-position: -100% 0;
  }
  100% {
    background-position: 100% 0;
  }
`;

const pulse = keyframes`
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.6;
  }
`;

interface StyledChipProps {
  label: string;
  icon?: React.ReactNode;
  variant?: 'status' | 'latency' | 'context' | 'deployment';
  color?: 'success' | 'warning' | 'error' | 'info' | 'primary' | 'secondary';
  size?: 'small' | 'medium';
  tooltip?: string;
}

export const StyledChip: React.FC<StyledChipProps> = ({
  label,
  icon,
  variant = 'status',
  color = 'info',
  size = 'small',
}) => {
  const theme = useTheme();

  const getColorStyles = () => {
    const colors = {
      success: {
        bg: `linear-gradient(135deg, ${alpha(theme.palette.success.main, 0.12)} 0%, ${alpha(theme.palette.success.main, 0.08)} 100%)`,
        border: `linear-gradient(135deg, ${alpha(theme.palette.success.main, 0.4)} 0%, ${alpha(theme.palette.success.main, 0.2)} 100%)`,
        text: theme.palette.success.main,
        hoverBg: alpha(theme.palette.success.main, 0.16),
        glow: alpha(theme.palette.success.main, 0.3),
      },
      warning: {
        bg: `linear-gradient(135deg, ${alpha(theme.palette.warning.main, 0.12)} 0%, ${alpha(theme.palette.warning.main, 0.08)} 100%)`,
        border: `linear-gradient(135deg, ${alpha(theme.palette.warning.main, 0.4)} 0%, ${alpha(theme.palette.warning.main, 0.2)} 100%)`,
        text: theme.palette.warning.main,
        hoverBg: alpha(theme.palette.warning.main, 0.16),
        glow: alpha(theme.palette.warning.main, 0.3),
      },
      error: {
        bg: `linear-gradient(135deg, ${alpha(theme.palette.error.main, 0.12)} 0%, ${alpha(theme.palette.error.main, 0.08)} 100%)`,
        border: `linear-gradient(135deg, ${alpha(theme.palette.error.main, 0.4)} 0%, ${alpha(theme.palette.error.main, 0.2)} 100%)`,
        text: theme.palette.error.main,
        hoverBg: alpha(theme.palette.error.main, 0.16),
        glow: alpha(theme.palette.error.main, 0.3),
      },
      info: {
        bg: `linear-gradient(135deg, ${alpha(theme.palette.info.main, 0.12)} 0%, ${alpha(theme.palette.info.main, 0.08)} 100%)`,
        border: `linear-gradient(135deg, ${alpha(theme.palette.info.main, 0.4)} 0%, ${alpha(theme.palette.info.main, 0.2)} 100%)`,
        text: theme.palette.info.main,
        hoverBg: alpha(theme.palette.info.main, 0.16),
        glow: alpha(theme.palette.info.main, 0.3),
      },
      primary: {
        bg: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.12)} 0%, ${alpha(theme.palette.primary.main, 0.08)} 100%)`,
        border: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.4)} 0%, ${alpha(theme.palette.primary.main, 0.2)} 100%)`,
        text: theme.palette.primary.main,
        hoverBg: alpha(theme.palette.primary.main, 0.16),
        glow: alpha(theme.palette.primary.main, 0.3),
      },
      secondary: {
        bg: `linear-gradient(135deg, ${alpha(theme.palette.secondary.main, 0.12)} 0%, ${alpha(theme.palette.secondary.main, 0.08)} 100%)`,
        border: `linear-gradient(135deg, ${alpha(theme.palette.secondary.main, 0.4)} 0%, ${alpha(theme.palette.secondary.main, 0.2)} 100%)`,
        text: theme.palette.secondary.main,
        hoverBg: alpha(theme.palette.secondary.main, 0.16),
        glow: alpha(theme.palette.secondary.main, 0.3),
      },
    };
    return colors[color];
  };

  const colorStyles = getColorStyles();
  const isSmall = size === 'small';

  return (
    <Box
      sx={{
        position: 'relative',
        display: 'inline-flex',
        alignItems: 'center',
        gap: isSmall ? 0.6 : 0.8,
        px: isSmall ? 1.5 : 1.75,
        py: isSmall ? 0.4 : 0.6,
        borderRadius: '9999px',
        background: colorStyles.bg,
        backdropFilter: 'blur(8px)',
        overflow: 'hidden',
        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        cursor: 'default',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          borderRadius: 'inherit',
          background: colorStyles.border,
          padding: '1px',
          WebkitMask: 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)',
          WebkitMaskComposite: 'xor',
          maskComposite: 'exclude',
        },
        '&:hover': {
          background: colorStyles.hoverBg,
          transform: 'translateY(-2px) scale(1.02)',
          boxShadow: `0 4px 12px ${colorStyles.glow}`,
          '&::after': {
            opacity: 1,
          },
        },
        '&::after': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: '-100%',
          width: '100%',
          height: '100%',
          background: `linear-gradient(90deg, transparent, ${alpha(colorStyles.text, 0.2)}, transparent)`,
          transition: 'opacity 0.3s',
          opacity: 0,
          animation: `${shimmer} 2s ease-in-out infinite`,
        },
      }}
    >
      {icon && (
        <Box
          component="span"
          sx={{
            display: 'flex',
            alignItems: 'center',
            color: colorStyles.text,
            fontSize: isSmall ? 14 : 16,
            transition: 'transform 0.3s ease',
            '& svg': {
              filter: `drop-shadow(0 0 4px ${colorStyles.glow})`,
            },
          }}
        >
          {icon}
        </Box>
      )}
      <Typography
        variant={isSmall ? 'caption' : 'body2'}
        sx={{
          color: colorStyles.text,
          fontWeight: 500,
          letterSpacing: '0.025em',
          fontSize: isSmall ? '0.7rem' : '0.8rem',
          textTransform: variant === 'status' ? 'uppercase' : 'none',
        }}
      >
        {label}
      </Typography>
    </Box>
  );
};

// Specialized chip components
export const StatusChip: React.FC<{ status: string; timestamp?: string | null }> = ({ status, timestamp }) => {
  const theme = useTheme();
  
  const getStatusConfig = () => {
    switch (status?.toLowerCase()) {
      case 'active':
        return { 
          icon: <Box sx={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
            <CheckCircleIcon sx={{ fontSize: 16 }} />
            <FiberManualRecordIcon sx={{ 
              fontSize: 6, 
              position: 'absolute',
              top: -2,
              right: -2,
              animation: `${pulse} 2s ease-in-out infinite`,
              color: theme.palette.success.light
            }} />
          </Box>, 
          color: 'success' as const, 
          label: 'Active' 
        };
      case 'error':
      case 'connection_error':
        return { 
          icon: <ErrorIcon sx={{ fontSize: 16 }} />, 
          color: 'error' as const, 
          label: status === 'connection_error' ? 'Connection Error' : 'Error' 
        };
      case 'timeout':
        return { 
          icon: <PauseCircleIcon sx={{ fontSize: 16 }} />, 
          color: 'warning' as const, 
          label: 'Timeout' 
        };
      case 'unavailable':
        return { 
          icon: <HelpIcon sx={{ fontSize: 16 }} />, 
          color: 'info' as const, 
          label: 'Unavailable' 
        };
      case 'unresponsive':
        return { 
          icon: <ErrorIcon sx={{ fontSize: 16 }} />, 
          color: 'error' as const, 
          label: 'Unresponsive' 
        };
      default:
        return { 
          icon: <HelpIcon sx={{ fontSize: 16 }} />, 
          color: 'info' as const, 
          label: status || 'Unknown' 
        };
    }
  };

  const config = getStatusConfig();
  
  return (
    <StyledChip
      label={config.label}
      icon={config.icon}
      color={config.color}
      variant="status"
    />
  );
};

export const LatencyChip: React.FC<{ latencyMs: number | null }> = ({ latencyMs }) => {
  const theme = useTheme();
  
  if (latencyMs === null) {
    return (
      <Box sx={{ 
        display: 'inline-flex',
        alignItems: 'center',
        px: 1.5,
        py: 0.4,
        borderRadius: '9999px',
        background: alpha(theme.palette.text.secondary, 0.08),
        border: `1px solid ${alpha(theme.palette.text.secondary, 0.2)}`,
      }}>
        <Typography variant="caption" sx={{ 
          color: theme.palette.text.secondary,
          fontSize: '0.7rem',
          fontWeight: 500,
        }}>
          N/A
        </Typography>
      </Box>
    );
  }
  
  const latencySec = latencyMs / 1000;
  const getLatencyConfig = () => {
    if (latencySec < 10) return { 
      color: 'success' as const, 
      icon: <SpeedIcon sx={{ fontSize: 16 }} />,
      bgIntensity: 0.12 
    };
    if (latencySec < 20) return { 
      color: 'warning' as const, 
      icon: <SpeedIcon sx={{ fontSize: 16 }} />,
      bgIntensity: 0.15 
    };
    return { 
      color: 'error' as const, 
      icon: <SpeedIcon sx={{ fontSize: 16 }} />,
      bgIntensity: 0.18 
    };
  };
  
  const config = getLatencyConfig();
  
  return (
    <StyledChip
      label={`${latencySec.toFixed(2)}s`}
      icon={config.icon}
      color={config.color}
      variant="latency"
    />
  );
};

export const ContextChip: React.FC<{ contextWindow: number | null }> = ({ contextWindow }) => {
  const theme = useTheme();
  
  if (!contextWindow) {
    return (
      <Box sx={{ 
        display: 'inline-flex',
        alignItems: 'center',
        px: 1.5,
        py: 0.4,
        borderRadius: '9999px',
        background: alpha(theme.palette.text.secondary, 0.08),
        border: `1px solid ${alpha(theme.palette.text.secondary, 0.2)}`,
      }}>
        <Typography variant="caption" sx={{ 
          color: theme.palette.text.secondary,
          fontSize: '0.7rem',
          fontWeight: 500,
        }}>
          N/A
        </Typography>
      </Box>
    );
  }
  
  const formatContext = (size: number) => {
    if (size >= 1000000) return `${(size / 1000000).toFixed(1)}M`;
    if (size >= 1000) return `${Math.floor(size / 1000)}K`;
    return size.toString();
  };
  
  // Determine color based on context window size
  const getContextColor = () => {
    if (contextWindow >= 1000000) return 'primary'; // 1M+ tokens
    if (contextWindow >= 100000) return 'secondary'; // 100K+ tokens
    return 'info'; // Less than 100K
  };
  
  return (
    <StyledChip
      label={formatContext(contextWindow)}
      icon={<DataUsageIcon sx={{ fontSize: 16 }} />}
      color={getContextColor() as any}
      variant="context"
    />
  );
};

export const DeploymentChip: React.FC<{ isCloud: boolean }> = ({ isCloud }) => {
  return (
    <StyledChip
      label={isCloud ? 'Cloud' : 'On-Prem'}
      icon={isCloud ? 
        <CloudIcon sx={{ fontSize: 16 }} /> : 
        <ComputerIcon sx={{ fontSize: 16 }} />
      }
      color={isCloud ? 'secondary' : 'primary'}
      variant="deployment"
    />
  );
};