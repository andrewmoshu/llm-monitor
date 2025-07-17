import React from 'react';
import {
  ToggleButton,
  ToggleButtonGroup,
  Box,
  Typography,
  useTheme,
  alpha,
  Chip,
} from '@mui/material';
import { Environment } from '../types/types';
import DeveloperModeIcon from '@mui/icons-material/DeveloperMode';
import BugReportIcon from '@mui/icons-material/BugReport';
import ScienceIcon from '@mui/icons-material/Science';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';

interface EnvironmentSwitcherProps {
  currentEnvironment: Environment;
  onEnvironmentChange: (env: Environment) => void;
}

const environments: {
  value: Environment;
  label: string;
  icon: React.ReactElement;
  color: string;
}[] = [
  { value: 'dev', label: 'DEV', icon: <DeveloperModeIcon fontSize="small" />, color: '#4caf50' },
  { value: 'test', label: 'TEST', icon: <ScienceIcon fontSize="small" />, color: '#2196f3' },
  { value: 'qa', label: 'QA', icon: <BugReportIcon fontSize="small" />, color: '#ff9800' },
  { value: 'prod', label: 'PROD', icon: <RocketLaunchIcon fontSize="small" />, color: '#f44336' },
];

const EnvironmentSwitcher: React.FC<EnvironmentSwitcherProps> = ({
  currentEnvironment,
  onEnvironmentChange,
}) => {
  const theme = useTheme();

  const handleChange = (
    event: React.MouseEvent<HTMLElement>,
    newEnvironment: Environment | null,
  ) => {
    if (newEnvironment !== null) {
      onEnvironmentChange(newEnvironment);
    }
  };

  return (
    <Box sx={{ 
      display: 'flex', 
      alignItems: 'center',
      gap: 2,
    }}>
      <Typography 
        variant="body2" 
        sx={{ 
          color: 'text.secondary',
          fontWeight: 500,
          textTransform: 'uppercase',
          fontSize: '0.75rem',
          letterSpacing: '0.5px',
        }}
      >
        Environment
      </Typography>
      <ToggleButtonGroup
        value={currentEnvironment}
        exclusive
        onChange={handleChange}
        aria-label="environment"
        size="small"
        sx={{
          backgroundColor: theme.palette.mode === 'dark'
            ? alpha(theme.palette.common.white, 0.05)
            : alpha(theme.palette.common.black, 0.05),
          borderRadius: 3,
          '& .MuiToggleButton-root': {
            border: 'none',
            borderRadius: 3,
            px: 2,
            py: 0.75,
            textTransform: 'none',
            fontWeight: 600,
            gap: 1,
            transition: 'all 0.3s ease',
            color: theme.palette.text.secondary,
            '&:hover': {
              backgroundColor: theme.palette.mode === 'dark'
                ? alpha(theme.palette.common.white, 0.1)
                : alpha(theme.palette.common.black, 0.1),
            },
            '&.Mui-selected': {
              backgroundColor: 'transparent',
              color: theme.palette.common.white,
              '&:hover': {
                backgroundColor: 'transparent',
              },
            },
          },
        }}
      >
        {environments.map((env) => (
          <ToggleButton
            key={env.value}
            value={env.value}
            aria-label={env.label}
            sx={{
              '&.Mui-selected': {
                background: `linear-gradient(135deg, ${env.color} 0%, ${alpha(env.color, 0.7)} 100%)`,
                boxShadow: `0 4px 12px ${alpha(env.color, 0.3)}`,
                '& .MuiSvgIcon-root': {
                  filter: 'drop-shadow(0 2px 4px rgba(0, 0, 0, 0.2))',
                },
              },
            }}
          >
            {env.icon}
            <span>{env.label}</span>
          </ToggleButton>
        ))}
      </ToggleButtonGroup>
      
      {/* Current environment indicator */}
      <Chip
        size="small"
        label={environments.find(e => e.value === currentEnvironment)?.label}
        sx={{
          ml: 1,
          backgroundColor: environments.find(e => e.value === currentEnvironment)?.color,
          color: 'white',
          fontWeight: 600,
          fontSize: '0.7rem',
          height: 24,
          '& .MuiChip-label': {
            px: 1.5,
          },
        }}
      />
    </Box>
  );
};

export default EnvironmentSwitcher;