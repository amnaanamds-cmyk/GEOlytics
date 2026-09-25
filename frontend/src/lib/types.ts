/** Shapes returned by the GEOlytics API, as the dashboard consumes them. */

export type Organization = {
  id: number;
  slug: string;
  name: string;
  plan: string;
  role?: string;
};

export type PageScore = {
  url: string;
  title: string | null;
  score: number;
  signals: Record<string, number>;
  contributions: Record<string, number>;
  recommendations: string[];
};

export type Audit = {
  audit_id: number;
  site_url: string;
  status: "pending" | "running" | "complete" | "failed";
  overall_score: number | null;
  weights_fitted: boolean;
  score_caveat: string | null;
  pages: PageScore[];
  summary: Record<string, unknown>;
  created_at: string | null;
};

export type AuditList = { items: Audit[]; limit: number; offset: number };

export type Usage = {
  period: string;
  plan: string;
  used: Record<string, number>;
  limits: Record<string, number>;
  remaining: Record<string, number>;
  sites: { used: number; limit: number };
  concurrent_audits: { used: number; limit: number };
};

export type ApiKey = {
  id: number;
  name: string;
  display_hint: string;
  scopes: string[];
  environment: string;
  created_at: string | null;
  last_used_at: string | null;
  expires_at: string | null;
  revoked_at: string | null;
};

export type Member = {
  user_id: number;
  email: string;
  full_name: string | null;
  role: string;
  joined_at: string | null;
};

export type Comparison = {
  name_a: string;
  name_b: string;
  metric: string;
  n: number;
  mean_a: number;
  mean_b: number;
  mean_diff: number;
  ci_low: number;
  ci_high: number;
  p_value: number;
  p_adjusted: number | null;
  effect_size: number;
  effect_label: string;
  significant: boolean;
};

export type ExperimentRun = {
  condition: string;
  chunker: Record<string, unknown>;
  retriever: Record<string, unknown>;
  index_stats: Record<string, number>;
  aggregate: Record<string, number>;
  n_queries: number;
};

export type Experiment = {
  experiment: string;
  metric: string;
  runs: ExperimentRun[];
  comparisons: Comparison[];
  methods_note: string;
};

export type ApiError = { error: string; message: string; request_id?: string };
