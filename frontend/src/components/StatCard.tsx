import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';

interface Props {
  title: string;
  value: string | number;
  icon?: React.ReactElement; // Optional icon
}

const StatCard: React.FC<Props> = ({ title, value, icon }) => {
  return (
    <Card elevation={2} sx={{ borderRadius: 2, height: '100%' }}> {/* Ensure consistent height if needed */}
      <CardContent sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box>
          <Typography color="text.secondary" gutterBottom variant="body2">
            {title}
          </Typography>
          <Typography variant="h5" component="div" fontWeight="medium">
            {value}
          </Typography>
        </Box>
        {icon && (
          <Box sx={{ color: 'text.secondary' }}>
            {/* Cast the props object to 'any' to allow the 'sx' prop */}
            {React.cloneElement(icon, { sx: { fontSize: 40 } } as any)}
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default StatCard; 