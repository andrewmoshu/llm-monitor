export interface LatencyRecord {
  model_name: string;
  timestamp: string;
  latency_ms: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  cost: number | null;
  context_window: number | null;
  is_cloud: boolean;
  status: string;
}

export interface ModelInfo {
  pricing_per_1k: {
    input: number;
    output: number;
  };
  context_window: number;
  training_data: string;
  capabilities: {
    vision: boolean;
    streaming: boolean;
    function_calling: boolean;
  };
  is_cloud?: boolean | null;
}

export {};