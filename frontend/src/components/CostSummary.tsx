import React, { useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Chip,
  useTheme,
  Divider,
  alpha,
} from '@mui/material';
import AttachMoneyIcon from '@mui/icons-material/AttachMoney';
import CloudIcon from '@mui/icons-material/Cloud';
import StorageIcon from '@mui/icons-material/Storage';
import { LatencyRecord, ModelInfo } from '../types/types';

interface Props {
  data: LatencyRecord[];
  modelInfo: { [key: string]: ModelInfo };
}

const CostSummary: React.FC<Props> = ({ data, modelInfo }) => {
  const theme = useTheme();

  const summary = useMemo(() => {
    const result = {
      totalDailyCost: 0,
      cloudModelCost: 0,
      onPremModelCost: 0,
      totalRequests: 0,
      modelCosts: new Map<string, number>(),
    };

    // Group by model and get latest records
    const latestByModel = new Map<string, LatencyRecord>();
    data.forEach(record => {
      const existing = latestByModel.get(record.model_name);
      if (!existing || new Date(record.timestamp) > new Date(existing.timestamp)) {
        latestByModel.set(record.model_name, record);
      }
    });

    // Calculate costs
    latestByModel.forEach((record, modelName) => {
      if (record.cost !== null) {
        result.totalDailyCost += record.cost;
        if (record.is_cloud) {
          result.cloudModelCost += record.cost;
        } else {
          result.onPremModelCost += record.cost;
        }
        result.modelCosts.set(modelName, record.cost);
      }
    });

    result.totalRequests = data.length;

    return result;
  }, [data]);

  const topModels = useMemo(() => {
    return Array.from(summary.modelCosts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3);
  }, [summary]);

  return (
    <Card elevation={2} sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
          <AttachMoneyIcon sx={{ color: 'primary.main', mr: 1 }} />
          <Typography variant="h6" fontWeight={600}>
            Cost Summary
          </Typography>
        </Box>

        <Grid container spacing={2}>
          <Grid item xs={12}>
            <Box sx={{ p: 2, bgcolor: alpha(theme.palette.primary.main, 0.1), borderRadius: 2 }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Total Daily Cost
              </Typography>
              <Typography variant="h4" fontWeight={700} color="primary.main">
                ${summary.totalDailyCost.toFixed(2)}
              </Typography>
            </Box>
          </Grid>

          <Grid item xs={6}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <CloudIcon sx={{ fontSize: 16, mr: 0.5, color: 'info.main' }} />
              <Typography variant="body2" color="text.secondary">
                Cloud Models
              </Typography>
            </Box>
            <Typography variant="h6" fontWeight={600}>
              ${summary.cloudModelCost.toFixed(2)}
            </Typography>
          </Grid>

          <Grid item xs={6}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <StorageIcon sx={{ fontSize: 16, mr: 0.5, color: 'success.main' }} />
              <Typography variant="body2" color="text.secondary">
                On-Premise
              </Typography>
            </Box>
            <Typography variant="h6" fontWeight={600}>
              ${summary.onPremModelCost.toFixed(2)}
            </Typography>
          </Grid>

          {topModels.length > 0 && (
            <>
              <Grid item xs={12}>
                <Divider sx={{ my: 1 }} />
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  Top Models by Cost
                </Typography>
              </Grid>
              {topModels.map(([model, cost], index) => (
                <Grid item xs={12} key={model}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Chip
                      label={model}
                      size="small"
                      color={index === 0 ? 'primary' : 'default'}
                      variant={index === 0 ? 'filled' : 'outlined'}
                    />
                    <Typography variant="body2" fontWeight={500}>
                      ${cost.toFixed(4)}
                    </Typography>
                  </Box>
                </Grid>
              ))}
            </>
          )}
        </Grid>
      </CardContent>
    </Card>
  );
};

export default CostSummary;