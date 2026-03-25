export interface GridStartRequest {
  prompt_a: string;
  prompt_b: string;
  prompt_c: string;
  grid_size: number;
  seed: number;
  seed_count?: number;
  height?: number;
  width?: number;
  steps?: number;
}

export interface GridStartResponse {
  job_id: string;
  total_cells: number;
  status: string;
}

export interface CellStatus {
  row: number;
  col: number;
  alpha: number;
  beta: number;
  status: string;
  sensitivity: number | null;
  cluster: number | null;
  thumbnail_url: string | null;
  hq_url: string | null;
  span: number;
}

export interface GridStatusResponse {
  job_id: string;
  status: string;
  phase: string;
  grid_size: number;
  cells_generated: number;
  cells_total: number;
  cells: CellStatus[];
  seed_cells: Record<string, CellStatus[]> | null;
  seeds: number[];
  prompt_a: string;
  prompt_b: string;
  prompt_c: string;
  heatmap_url: string | null;
  overlay_url: string | null;
  cluster_url: string | null;
  image_grid_url: string | null;
}

export interface SeedProbeRequest {
  alpha: number;
  beta: number;
  seed_start: number;
  seed_end: number;
}

export interface SeedProbeResponse {
  probe_id: string;
  total: number;
  status: string;
}

export interface SeedProbeStatus {
  probe_id: string;
  alpha: number;
  beta: number;
  seeds: number[];
  images: (string | null)[];
  complete: boolean;
}

export interface RefineRequest {
  tau: number;
  multiplier: number;
}

export interface RefineResponse {
  refine_job_id: string;
  parent_job_id: string;
  total_cells: number;
  status: string;
}
