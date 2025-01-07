export interface LatencyRecord {
  model_name: string;
  timestamp: string;
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  cost: number;
  arena_score: number | null;
  context_window: number;
  is_cloud: boolean;
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
}

export {};