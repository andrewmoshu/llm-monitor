import axios from 'axios';
import { LatencyRecord, ModelInfo, Environment } from './types/types';

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
  getLatencyData: async (environment?: Environment): Promise<LatencyRecord[]> => {
    try {
      const url = environment 
        ? `${API_BASE_URL}/latency?environment=${environment}`
        : `${API_BASE_URL}/latency`;
      const response = await axios.get<LatencyRecord[]>(url);
      return Array.isArray(response.data) ? response.data : [];
    } catch (error) {
      console.error('API Error fetching latency:', error);
      return [];
    }
  },

  getModels: async (environment?: Environment): Promise<{ [key: string]: ModelInfo }> => {
    try {
      const url = environment 
        ? `${API_BASE_URL}/models?environment=${environment}`
        : `${API_BASE_URL}/models`;
      const response = await axios.get<{ [key: string]: ModelInfo }>(url);
      return typeof response.data === 'object' && response.data !== null ? response.data : {};
    } catch (error) {
      console.error('API Error fetching models:', error);
      return {};
    }
  }
}; 