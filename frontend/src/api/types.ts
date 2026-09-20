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
  error?: string | null;
}

export interface SeedProbeRequest {
  alpha: number;
  beta: number;
  seed_start: number;
  seed_end: number;
  steps?: number;
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

// ---- ridge graph: the ridge network as walkable arcs ----
export interface RidgeArc {
  id: number;
  stations: [number, number][];   // (alpha_idx, beta_idx) in walk order
  length: number;
  flank_pairs: [number, number][];
  itinerary: [number, number][];  // flanking pair sequence; usually length 1
  mean_sensitivity: number | null;
  s_percentile: number | null;
}

export interface RidgeGraph {
  params: { k: number; h_frac: number; min_arc: number };
  grid_size: number;
  n_basins: number;
  n_arcs: number;
  junctions: [number, number][];
  edges: RidgeArc[];
}

export interface RidgeGraphRequest { k?: number; h_frac?: number; min_arc?: number }
export interface RidgeGraphResponse {
  job_id: string; n_arcs: number; n_basins: number;
  n_junctions: number; params: Record<string, unknown>;
}

// ---- hikers: population search over prompt-simplex chains ----
export interface HikeStartRequest {
  prompt_a: string; prompt_b: string; prompt_c: string;
  beam?: number; budget?: number; grid_size?: number; seed?: number;
  height?: number; width?: number; steps?: number; guidance_scale?: number;
  k_basins?: number; h_frac?: number; prompt_pool?: string[];
  refine_rounds?: number; refine_cap?: number; margin?: number;
  seed_count?: number;   // >1 walks the seed-AVERAGED sensitivity field
}
export interface HikeStartResponse { hike_id: string; budget: number; beam: number; status: string }
export interface HikeChainRow {
  chain: number; hop: number; labels: string[];
  arc_len: number; cum_novel: number; n_arcs: number;
  // the backend explains a stalled or narrowed hop here; without these fields the UI
  // showed an idle-looking run with no reason given
  lineage?: string; note?: string | null;
  // the walked arc, for animating a marker over the simplex map
  grid?: number; margin?: number; cell?: number; approach?: number;
  frame_px?: number; n_frames?: number;
  stations?: [number, number][];
  basis?: string[]; coefs?: number[][];
}
export interface HikeStatus {
  hike_id: string; status: string; hop: number;
  grids_done: number; grids_dropped?: number; refined_cells?: number;
  budget: number; beam?: number;
  chains: HikeChainRow[]; notes?: string[] | null; error?: string | null;
}

// --- Discovery: the procedure the ablations support (backend/services/discover.py) ---

export interface DiscoverStartRequest {
  k: number;
  commitment: number;      // E[max weight]; hard floor is 1/k, checked server-side
  target_sim: number;
  batch: number;           // images per simplex before restarting
  total: number;
  seed?: number;
  steps?: number;
}

export interface DiscoverStartResponse {
  run_id: string;
  status: string;
  total: number;
  error?: string | null;
}

export interface DiscoverSimplex {
  index: number;
  prompts: string[];
  mean_sim: number;
  alpha: number;
  n_points: number;
  done: number;
  diversity: number | null;
}

export interface DiscoverStatus {
  run_id: string;
  status: string;
  generated: number;
  total: number;
  k: number;
  commitment: number;
  // Mean pairwise DINOv2 distance. Reference-free, which is why it is the live number.
  diversity: number | null;
  simplices: DiscoverSimplex[];
  notes: string[];
  error?: string | null;
}

export interface DiscoverDefaults {
  k: number;
  commitment: number;
  target_sim: number;
  batch: number;
  steps: number;
  provenance: Record<string, string>;
  alpha_table: Record<string, Record<string, number>>;
}

// ---- Cascade: fast boundary isolation + dense patch refinement ----
// Calibrated two-tier pipeline (search_problem E85/E86/E92): chords find crossings,
// coupled-seed B scores them against the run's own measured background, dense 5x5
// patches refine the top regions plus one exploration slot.

export interface CascadeStartRequest {
  k: number;
  prompts?: string[] | null;
  target_sim?: number;
  n_chords?: number;
  n_patches?: number;
  seed?: number;
  steps?: number;
  // focused exploration: confine the survey to a ball around this recipe
  // (requires pinned prompts so the weights refer to a known basis)
  focus?: number[] | null;
  focus_radius?: number;
  // tier-1 detection steps (default 4: gated at 93%/94% recall, ~2x faster probes)
  probe_steps?: number | null;
}

export interface CascadePointInfo {
  weights: number[] | null;
  div: number | null;
}

export interface CascadeStartResponse {
  run_id: string;
  status: string;
  error?: string | null;
}

export interface CascadeChord {
  // chord endpoints in weight space, for the map view
  a: number[];
  b: number[];
}

export interface CascadeCrossing {
  cid: number;
  weights: number[];
  // paired boundary strength B = mean over coupled seeds of (1 - cos); null until scored
  b: number | null;
  // true iff b beats the run's own background p95 (never a no-step null)
  significant: boolean;
  thumb: number;
  // crossings sharing a ridge_group portray the same ridge (free, from tier-1)
  ridge_group: number | null;
  // current bisection bracket width; ~0.012 = pinned (drives the lock-on reticle)
  bracket_w: number | null;
}

export interface CascadePatch {
  region: number;
  cid: number;
  b: number;
  significant: boolean;
  exploration: boolean;
  // 5x5 thumb indices; rows = tangent, cols = crossing the boundary; -1 = outside simplex
  grid: number[][];
  cols_with_crossing: number;
}

export interface CascadeStatus {
  run_id: string;
  status: string;
  phase: string;
  generated: number;
  phase_done: number;
  phase_total: number;
  prompts: string[];
  // most recent generated image indices -- the live ticker
  recent_thumbs: number[];
  // completed-generation positions (weight space) -- the live sampling cloud
  points: number[][];
  // local divergence per cloud point (aligned with points; null until measured)
  point_divs: (number | null)[];
  chords: CascadeChord[];
  crossings: CascadeCrossing[];
  bg_mean: number | null;
  bg_p95: number | null;
  // coverage certificate: distinct ridges, ridges crossed exactly once, and the
  // Good-Turing estimate of boundary area never crossed by this survey
  distinct_ridges: number | null;
  singleton_ridges: number | null;
  unexplored_share: number | null;
  patches: CascadePatch[];
  notes: string[];
  error?: string | null;
}

// ---- Ridge walk: human-in-the-loop traversal along one ridge ----

export interface WalkStartRequest {
  cid: number;
  direction: number;   // +1 / -1
  n_steps?: number;
}

export interface WalkStep {
  weights: number[];
  // single-seed local contrast across the boundary at this step (not the certified B)
  contrast: number;
  thumb: number;
}

export interface WalkStatus {
  walk_id: string;
  status: string;
  cid: number;
  steps: WalkStep[];
  // transversal probe lines (pairs of weight vectors) -- the walk's line bundle
  segs: number[][][];
  notes: string[];
  error?: string | null;
}
