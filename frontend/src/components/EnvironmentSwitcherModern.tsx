import React from 'react';
import {
  Box,
  Typography,
  useTheme,
  alpha,
  Tooltip,
} from '@mui/material';
import { Environment } from '../types/types';

interface EnvironmentSwitcherProps {
  currentEnvironment: Environment;
  onEnvironmentChange: (env: Environment) => void;
}

const environments: {
  value: Environment;
  label: string;
  color: string;
  description: string;
}[] = [
  { value: 'dev', label: 'DEV', color: '#4caf50', description: 'Development' },
  { value: 'test', label: 'TEST', color: '#2196f3', description: 'Testing' },
  { value: 'qa', label: 'QA', color: '#ff9800', description: 'Quality Assurance' },
  { value: 'prod', label: 'PROD', color: '#f44336', description: 'Production' },
];

const EnvironmentSwitcherModern: React.FC<EnvironmentSwitcherProps> = ({
  currentEnvironment,
  onEnvironmentChange,
}) => {
  const theme = useTheme();

  return (
    <Box sx={{ 
      display: 'flex', 
      alignItems: 'center',
      gap: 1,
      p: 0.5,
      borderRadius: 3,
      background: theme.palette.mode === 'dark'
        ? 'rgba(255, 255, 255, 0.05)'
        : 'rgba(0, 0, 0, 0.05)',
      backdropFilter: 'blur(10px)',
      border: '1px solid',
      borderColor: theme.palette.mode === 'dark'
        ? 'rgba(255, 255, 255, 0.1)'
        : 'rgba(0, 0, 0, 0.1)',
    }}>
      {environments.map((env, index) => {
        const isActive = currentEnvironment === env.value;
        return (
          <React.Fragment key={env.value}>
            <Tooltip title={env.description} arrow placement="bottom">
              <Box
                onClick={() => onEnvironmentChange(env.value)}
                sx={{
                  position: 'relative',
                  px: 2.5,
                  py: 1,
                  borderRadius: 2.5,
                  cursor: 'pointer',
                  transition: 'background 0.2s ease, color 0.2s ease, box-shadow 0.2s ease',
                  background: isActive
                    ? `linear-gradient(135deg, ${env.color} 0%, ${alpha(env.color, 0.8)} 100%)`
                    : 'transparent',
                  color: isActive
                    ? 'white'
                    : theme.palette.text.secondary,
                  fontWeight: isActive ? 700 : 600,
                  fontSize: '0.875rem',
                  letterSpacing: '0.5px',
                  boxShadow: isActive
                    ? `0 2px 8px ${alpha(env.color, 0.3)}`
                    : 'none',
                  '&:hover': {
                    background: isActive
                      ? `linear-gradient(135deg, ${env.color} 0%, ${alpha(env.color, 0.8)} 100%)`
                      : theme.palette.mode === 'dark'
                        ? 'rgba(255, 255, 255, 0.08)'
                        : 'rgba(0, 0, 0, 0.08)',
                  },
                  '&::before': {
                    content: '""',
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    width: '100%',
                    height: '100%',
                    borderRadius: 2.5,
                    background: env.color,
                    opacity: 0,
                    transform: 'translate(-50%, -50%) scale(0)',
                    transition: 'opacity 0.3s ease, transform 0.3s ease',
                  },
                }}
              >
                <Typography
                  component="span"
                  sx={{
                    position: 'relative',
                    zIndex: 1,
                  }}
                >
                  {env.label}
                </Typography>
                <Box
                  sx={{
                    position: 'absolute',
                    bottom: -2,
                    left: '50%',
                    transform: 'translateX(-50%)',
                    width: isActive ? 4 : 0,
                    height: isActive ? 4 : 0,
                    borderRadius: '50%',
                    backgroundColor: 'white',
                    transition: 'background 0.2s ease, color 0.2s ease, box-shadow 0.2s ease',
                    opacity: isActive ? 1 : 0,
                  }}
                />
              </Box>
            </Tooltip>
            {index < environments.length - 1 && (
              <Box
                sx={{
                  width: 1,
                  height: 20,
                  backgroundColor: theme.palette.mode === 'dark'
                    ? 'rgba(255, 255, 255, 0.1)'
                    : 'rgba(0, 0, 0, 0.1)',
                }}
              />
            )}
          </React.Fragment>
        );
      })}
    </Box>
  );
};

export default EnvironmentSwitcherModern;