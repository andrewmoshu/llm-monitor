export type Environment = 'dev' | 'test' | 'qa' | 'prod';

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
  environment?: Environment;
}

export interface ModelInfo {
  provider: string;
  context_window?: number | null;
  pricing?: {
    input: number;
    output: number;
  } | null;
  is_cloud?: boolean | null;
  arena_score?: number | null;
  usage_stats?: {
    daily_cost: number;
    daily_input_tokens: number;
    daily_output_tokens: number;
    daily_prompts: number;
    monthly_avg_cost: number;
    monthly_avg_input_tokens: number;
    monthly_avg_output_tokens: number;
  } | null;
}

export {};