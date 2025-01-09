import axios from 'axios';
import { LatencyRecord } from './types/types';

// In development, use localhost
// In production (kubernetes), the nginx config will handle the proxy
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? '/api'  // This will be proxied through nginx
  : 'http://localhost:8000/api';

export const api = {
  getLatencyData: async (): Promise<LatencyRecord[]> => {
    const response = await axios.get(`${API_BASE_URL}/latency`);
    return response.data;
  },

  getModels: async (): Promise<string[]> => {
    const response = await axios.get(`${API_BASE_URL}/models`);
    return response.data;
  }
}; 