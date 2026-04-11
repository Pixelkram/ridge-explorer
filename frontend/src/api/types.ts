export interface GridStartRequest {
  prompt_a: string;
  prompt_b: string;
  prompt_c: string;
  prompt_d?: string;
  dimensions?: number;
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
  dimensions: number;
  status: string;
}

export interface CellStatus {
  row: number;
  col: number;
  depth: number;
  alpha: number;
  beta: number;
  gamma: number;
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
  dimensions: number;
  grid_size: number;
  cells_generated: number;
  cells_total: number;
  cells: CellStatus[];
  seed_cells: Record<string, CellStatus[]> | null;
  seeds: number[];
  prompt_a: string;
  prompt_b: string;
  prompt_c: string;
  prompt_d: string;
  heatmap_url: string | null;
  overlay_url: string | null;
  cluster_url: string | null;
  image_grid_url: string | null;
  ridge_mesh_url: string | null;
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
  extra_positions?: [number, number][];  // manually selected (row, col) pairs
}

export interface RefineResponse {
  refine_job_id: string;
  parent_job_id: string;
  total_cells: number;
  status: string;
}

export interface FastScanRequest {
  prompt_a: string;
  prompt_b: string;
  prompt_c: string;
  prompt_d?: string;
  dimensions?: number;
  grid_size: number;
  seed: number;
  height?: number;
  width?: number;
  guidance_scale?: number;
}

export interface FastScanResponse {
  job_id: string;
  total_cells: number;
  status: string;
}

export interface GenerateSelectedRequest {
  tau: number;
  height: number;
  width: number;
  steps: number;
  guidance_scale?: number;
}

export interface GenerateSelectedResponse {
  status: string;
  total_cells: number;
}

export interface MFScanRequest {
  prompt_a: string;
  prompt_b: string;
  prompt_c: string;
  grid_size: number;
  seed: number;
  budget?: number;
  tau_mf?: number;
  height?: number;
  width?: number;
  steps?: number;
  guidance_scale?: number;
}

export interface MFScanResponse {
  job_id: string;
  total_cells: number;
  budget: number;
  status: string;
}
