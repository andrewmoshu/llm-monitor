import React, { useState, useMemo } from 'react';
import {
  Box,
  Chip,
  TextField,
  Autocomplete,
  Typography,
  IconButton,
  Collapse,
  Button,
  useTheme,
  alpha,
  Paper,
  Checkbox,
  ListItemText,
  Tooltip,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import ClearAllIcon from '@mui/icons-material/ClearAll';
import DoneAllIcon from '@mui/icons-material/DoneAll';
import SearchIcon from '@mui/icons-material/Search';
import { ModelInfo } from '../types/types';

interface ModelSelectorProps {
  models: string[];
  selectedModels: string[];
  onChange: (selected: string[]) => void;
  modelInfo?: { [key: string]: ModelInfo };
}

const ModelSelector: React.FC<ModelSelectorProps> = ({
  models,
  selectedModels,
  onChange,
  modelInfo
}) => {
  const theme = useTheme();
  const [expanded, setExpanded] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  
  const maxChipsToShow = 3;
  const hasMany = selectedModels.length > maxChipsToShow;
  
  // Group models by provider
  const groupedModels = useMemo(() => {
    const groups: { [key: string]: string[] } = {};
    models.forEach(model => {
      const provider = model.split('-')[0] || 'other';
      if (!groups[provider]) groups[provider] = [];
      groups[provider].push(model);
    });
    return groups;
  }, [models]);

  const handleDelete = (modelToDelete: string) => {
    onChange(selectedModels.filter(model => model !== modelToDelete));
  };

  const handleToggleAll = () => {
    if (selectedModels.length === models.length) {
      onChange([]);
    } else {
      onChange(models);
    }
  };

  const visibleChips = hasMany && !expanded 
    ? selectedModels.slice(0, maxChipsToShow)
    : selectedModels;

  return (
    <Box sx={{ width: '100%' }}>
      {/* Compact Chip View */}
      <Paper
        elevation={0}
        sx={{
          p: 1.5,
          background: theme.palette.mode === 'dark'
            ? 'rgba(255, 255, 255, 0.02)'
            : 'rgba(0, 0, 0, 0.02)',
          border: '1px solid',
          borderColor: theme.palette.mode === 'dark'
            ? 'rgba(255, 255, 255, 0.1)'
            : 'rgba(0, 0, 0, 0.1)',
          borderRadius: 2,
          transition: 'all 0.2s ease',
          '&:hover': {
            borderColor: alpha(theme.palette.primary.main, 0.3),
            background: theme.palette.mode === 'dark'
              ? 'rgba(255, 255, 255, 0.03)'
              : 'rgba(0, 0, 0, 0.03)',
          }
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
          <Typography 
            variant="caption" 
            sx={{ 
              color: theme.palette.text.secondary,
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
              mr: 1
            }}
          >
            Models ({selectedModels.length})
          </Typography>
          
          {selectedModels.length === 0 ? (
            <Typography variant="body2" color="text.secondary">
              None selected
            </Typography>
          ) : (
            <>
              {visibleChips.map((model) => {
                const info = modelInfo?.[model];
                return (
                  <Chip
                    key={model}
                    label={model}
                    size="small"
                    onDelete={() => handleDelete(model)}
                    sx={{
                      height: 24,
                      fontSize: '0.75rem',
                      backgroundColor: info?.is_cloud 
                        ? alpha(theme.palette.secondary.main, 0.1)
                        : alpha(theme.palette.primary.main, 0.1),
                      color: info?.is_cloud 
                        ? theme.palette.secondary.main
                        : theme.palette.primary.main,
                      borderColor: info?.is_cloud 
                        ? alpha(theme.palette.secondary.main, 0.3)
                        : alpha(theme.palette.primary.main, 0.3),
                      border: '1px solid',
                      '& .MuiChip-deleteIcon': {
                        fontSize: 16,
                        color: 'inherit',
                        opacity: 0.6,
                        '&:hover': {
                          opacity: 1,
                        }
                      }
                    }}
                  />
                );
              })}
              
              {hasMany && !expanded && (
                <Chip
                  label={`+${selectedModels.length - maxChipsToShow} more`}
                  size="small"
                  sx={{
                    height: 24,
                    fontSize: '0.75rem',
                    backgroundColor: alpha(theme.palette.info.main, 0.1),
                    color: theme.palette.info.main,
                    border: '1px solid',
                    borderColor: alpha(theme.palette.info.main, 0.3),
                  }}
                />
              )}
            </>
          )}
          
          <Box sx={{ ml: 'auto', display: 'flex', gap: 0.5 }}>
            {selectedModels.length > maxChipsToShow && (
              <Tooltip title={expanded ? "Collapse" : "Expand"}>
                <IconButton 
                  size="small" 
                  onClick={() => setExpanded(!expanded)}
                  sx={{ 
                    p: 0.5,
                    color: theme.palette.text.secondary,
                    '&:hover': {
                      color: theme.palette.primary.main,
                    }
                  }}
                >
                  {expanded ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
                </IconButton>
              </Tooltip>
            )}
            
            <Tooltip title="Search models">
              <IconButton 
                size="small" 
                onClick={() => setSearchOpen(!searchOpen)}
                sx={{ 
                  p: 0.5,
                  color: searchOpen ? theme.palette.primary.main : theme.palette.text.secondary,
                  '&:hover': {
                    color: theme.palette.primary.main,
                  }
                }}
              >
                <SearchIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            
            <Tooltip title={selectedModels.length === models.length ? "Clear all" : "Select all"}>
              <IconButton 
                size="small" 
                onClick={handleToggleAll}
                sx={{ 
                  p: 0.5,
                  color: theme.palette.text.secondary,
                  '&:hover': {
                    color: theme.palette.primary.main,
                  }
                }}
              >
                {selectedModels.length === models.length ? 
                  <ClearAllIcon fontSize="small" /> : 
                  <DoneAllIcon fontSize="small" />
                }
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
      </Paper>

      {/* Search/Select Interface */}
      <Collapse in={searchOpen} timeout={200}>
        <Box sx={{ mt: 2 }}>
          <Autocomplete
            multiple
            options={models}
            value={selectedModels}
            onChange={(_, newValue) => onChange(newValue)}
            disableCloseOnSelect
            renderInput={(params) => (
              <TextField
                {...params}
                placeholder="Search and select models..."
                size="small"
                sx={{
                  '& .MuiOutlinedInput-root': {
                    backgroundColor: theme.palette.mode === 'dark'
                      ? 'rgba(255, 255, 255, 0.02)'
                      : 'rgba(0, 0, 0, 0.02)',
                    '& fieldset': {
                      borderColor: theme.palette.mode === 'dark' 
                        ? 'rgba(255, 255, 255, 0.1)' 
                        : 'rgba(0, 0, 0, 0.1)',
                    },
                    '&:hover fieldset': {
                      borderColor: alpha(theme.palette.primary.main, 0.5),
                    },
                    '&.Mui-focused fieldset': {
                      borderColor: theme.palette.primary.main,
                    },
                  },
                }}
              />
            )}
            renderOption={(props, option, { selected }) => {
              const info = modelInfo?.[option];
              return (
                <li {...props}>
                  <Checkbox
                    checked={selected}
                    size="small"
                    sx={{
                      '&.Mui-checked': {
                        color: theme.palette.primary.main,
                      },
                    }}
                  />
                  <ListItemText 
                    primary={option}
                    secondary={info?.is_cloud ? 'Cloud' : 'On-Premise'}
                    primaryTypographyProps={{
                      variant: 'body2',
                      sx: { fontWeight: selected ? 600 : 400 }
                    }}
                    secondaryTypographyProps={{
                      variant: 'caption',
                      sx: { 
                        color: info?.is_cloud 
                          ? theme.palette.secondary.main 
                          : theme.palette.text.secondary 
                      }
                    }}
                  />
                  <Chip
                    label={`${info?.context_window ? Math.floor(info.context_window / 1000) + 'K' : 'N/A'}`}
                    size="small"
                    sx={{
                      height: 20,
                      fontSize: '0.7rem',
                      ml: 'auto',
                      backgroundColor: 'transparent',
                      color: theme.palette.text.secondary,
                      border: '1px solid',
                      borderColor: theme.palette.divider,
                    }}
                  />
                </li>
              );
            }}
            groupBy={(option) => {
              const provider = option.split('-')[0] || 'other';
              return provider.charAt(0).toUpperCase() + provider.slice(1);
            }}
            sx={{ width: '100%' }}
          />
        </Box>
      </Collapse>
    </Box>
  );
};

export default ModelSelector;