import axios from 'axios';
import { LatencyRecord, ModelInfo } from './types/types';

// Base URL configuration
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? `${window.location.origin}/api`
  : 'http://localhost:8000/api';

// Add request interceptor for debugging
axios.interceptors.request.use(request => {
  console.log('API Request:', request.url);
  return request;
});

export const api = {
  getLatencyData: async (): Promise<LatencyRecord[]> => {
    try {
      const response = await axios.get<LatencyRecord[]>(`${API_BASE_URL}/latency`);
      return Array.isArray(response.data) ? response.data : [];
    } catch (error) {
      console.error('API Error fetching latency:', error);
      return [];
    }
  },

  getModels: async (): Promise<{ [key: string]: ModelInfo }> => {
    try {
      const response = await axios.get<{ [key: string]: ModelInfo }>(`${API_BASE_URL}/models`);
      return typeof response.data === 'object' && response.data !== null ? response.data : {};
    } catch (error) {
      console.error('API Error fetching models:', error);
      return {};
    }
  }
}; 