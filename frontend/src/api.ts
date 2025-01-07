import axios from 'axios';
import { LatencyRecord } from './types/types';

const API_BASE_URL = 'http://localhost:8000/api';

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