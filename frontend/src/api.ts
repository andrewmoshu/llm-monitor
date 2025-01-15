import axios from 'axios';
import { LatencyRecord } from './types/types';

// Base URL configuration
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? `${window.location.origin}/api`
  : 'http://localhost:8001/api';

// Add request interceptor for debugging
axios.interceptors.request.use(request => {
  console.log('API Request:', request.url);
  return request;
});

export const api = {
  getLatencyData: async (): Promise<LatencyRecord[]> => {
    try {
      const response = await axios.get(`${API_BASE_URL}/latency`);
      return Array.isArray(response.data) ? response.data : [];
    } catch (error) {
      console.error('API Error:', error);
      return [];
    }
  },

  getModels: async (): Promise<string[]> => {
    try {
      const response = await axios.get(`${API_BASE_URL}/models`);
      return Array.isArray(response.data) ? response.data : [];
    } catch (error) {
      console.error('API Error:', error);
      return [];
    }
  }
}; 