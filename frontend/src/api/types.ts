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
  // full pipeline per detected crossing (rebracket + bisect + 4-seed score + patches), or
  // detection alone. OFF by default: ~6x cheaper per crossing, and the run stays uncertified
  // -- bracket-precision positions, no B, no patches, no walks
  certify?: boolean;
  seed?: number;
  steps?: number;
  // focused exploration: confine the survey to a ball around this recipe
  // (requires pinned prompts so the weights refer to a known basis)
  focus?: number[] | null;
  focus_radius?: number;
  // tier-1 detection steps (default 4: gated at 93%/94% recall, ~2x faster probes)
  probe_steps?: number | null;
  // chord probe spacing in weight space (default 0.025 = protocol of record, ~1 fine cell)
  stride?: number;
  // recursive chords: rays spawned from each selected crossing (0-6, 0 = off) and how many
  // generations of them (0-4). Roots stay fair area samples, children are exploratory
  branch?: number;
  depth?: number;
  // which crossings of a generation spawn the next: the top this-many % by probe-to-probe
  // divergence, at least one (1-100, default 20; 100 = every crossing branches)
  branch_top_pct?: number;
  // survey randomness apart from the image seed (null = seed)
  chord_seed?: number | null;
  // trace phase: walk every significant crossing both ways, certify, link crossings the walks reach
  trace?: boolean;
  trace_steps?: number;
  trace_certify?: boolean;
}

// one walk of the trace phase: origin crossing first, then the stations
export interface CascadeTrace {
  cid: number;
  direction: number;
  walk_id: string;
  points: number[][];
  cert?: boolean[] | null;   // per station: held on unseen seeds (null if not certified)
  end?: string;
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

// branching provenance of one chord, aligned with `chords`
export interface CascadeChordMeta {
  // 0 = root (fair sample); >= 1 = a child ray, one generation deeper
  gen: number;
  // chord it was spawned from and the crossing on it (-1 -1 for roots)
  parent: number;
  origin_cid: number;
  // angle spreading of a child ray: the line angle it achieved against the nearby chords it was
  // spread away from, and how many of those there were (null/0 for roots, which are not spread)
  min_angle_deg?: number | null;
  n_near?: number;
}

export interface CascadeCrossing {
  cid: number;
  weights: number[];
  // paired boundary strength B = mean over coupled seeds of (1 - cos); null until scored
  b: number | null;
  // true iff b beats the run's own background p95 (never a no-step null)
  significant: boolean;
  thumb: number;
  // generation of the chord this crossing sits on (0 = root, >= 1 = a child ray)
  gen?: number;
  // crossings sharing a ridge_group portray the same ridge (free, from tier-1)
  ridge_group: number | null;
  // current bisection bracket width; ~0.012 = pinned (drives the lock-on reticle)
  bracket_w: number | null;
  // ridge group with the trace phase's walk links added (null without a trace phase)
  traced_group?: number | null;
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
  // branching provenance per chord, aligned with `chords` (all gen 0 without branching)
  chords_meta?: CascadeChordMeta[];
  // the chord probe spacing this run was started with
  stride?: number;
  // the branching settings this run was started with: rays per selected crossing (0 = off),
  // generations of them, and the divergence percentile that picks the origins
  branch?: number;
  depth?: number;
  branch_top_pct?: number;
  // false = detection only: the crossings were never bisected or scored, so their positions
  // carry bracket precision (+-stride/2) and b/significant mean nothing
  certify?: boolean;
  crossings: CascadeCrossing[];
  bg_mean: number | null;
  bg_p95: number | null;
  // coverage certificate: distinct ridges, ridges crossed exactly once, and the
  // Good-Turing estimate of boundary area never crossed by this survey
  distinct_ridges: number | null;
  singleton_ridges: number | null;
  unexplored_share: number | null;
  // "all", or "roots" when branching was on: child rays start on a boundary, so they are
  // preferential samples and are left out of the fair-area certificate above
  stats_scope?: string;
  // trace phase: walked ridges, crossing pairs a walk joined, and the certificate recomputed with those links
  traces?: CascadeTrace[];
  trace_links?: number[][];
  traced_ridges?: number | null;
  traced_singletons?: number | null;
  traced_unexplored_share?: number | null;
  patches: CascadePatch[];
  notes: string[];
  error?: string | null;
}

// ---- Local boundary density: kernel-smoothed Crofton S_V over the run's own chords ----
// The survey's chords already sample boundary AREA fairly, so crossings per unit chord
// length convert to boundary area per unit volume; smoothing that ratio locally turns one
// global number into a map of WHERE the boundaries are dense. Calibration of record:
// search_problem/outputs/h23_local_sv_kde (verified independently). Two products, one
// estimator: the RANKING is reliable from ~20 chords, the VALUES need >= 80.

export interface LocalSvChord {
  // station positions in weight space, from chord endpoint a toward b
  points: number[][];
  // local S_V at each station (units only meaningful when calibrated_ok)
  values: number[];
  // percentile rank within the covered part of this map
  ranks: number[];
  // no local evidence here: kernel mass below 5% of its median (drawn grey, never coloured)
  uncovered: boolean[];
}

export interface LocalSvMap {
  run_id: string;
  k: number;
  h: number;                  // kernel width in orthonormal tangent coords (simplex edge = sqrt(2))
  mode: string;               // "all" | "certified" (numerator-only filter; a diagnostic)
  delta: number;
  c_d: number;
  n_chords: number;
  n_crossings: number;
  // false below 80 chords: the values are then worse than the global constant, ranking only
  calibrated_ok: boolean;
  // the global mean-of-ratios S_V from the same chords -- the constant the map has to beat
  s_global: number | null;
  chords: LocalSvChord[];
  // aligned by index with CascadeStatus.crossings
  crossing_cids: number[];
  crossing_values: number[];
  crossing_ranks: number[];
  crossing_uncovered: boolean[];
  cloud: number[][];
  cloud_values: number[];
  cloud_ranks: number[];
  cloud_uncovered: boolean[];
  error?: string | null;
}

// ---- JVP probe: the local crossing direction at one point of the simplex ----
// `normal_theta` is a unit direction in (alpha, beta[, gamma]) — the direction across
// prompt space along which the image changes fastest here. It is SIGN-FREE: a line,
// not an arrow, so the UI draws it as a segment through the point in both directions.

export interface JvpProbeRequest {
  alpha: number;
  beta: number;
  gamma?: number;
  seed?: number;
}

export interface JvpProbeStartResponse {
  probe_id: string;
  status: 'running' | 'error';
  error?: string | null;
}

/**
 * What every JVP probe reports, whichever surface launched it. The grid probe adds the
 * (alpha, beta, gamma) it was taken at; the cascade probe adds the k-vector of prompt
 * weights instead, because a cascade crossing has no grid coordinates.
 */
export interface JvpProbeCore {
  // the direction in barycentric prompt weights (length k, sums to 0) — the one readout
  // that means the same thing on both surfaces, and the only one a k-simplex map can draw
  normal_bary: number[];
  // unit vector in (alpha, beta[, gamma]); sign-free
  normal_theta: number[];
  normal_reading: string;
  sigma_whitened: number[];
  // comparable only between probes of the SAME job/run — never a ridge/not-ridge verdict
  sigma1_raw: number;
  // ~1 = a single front; participation_ratio ~1 = front, ~2 = corner
  rank1_share: number;
  participation_ratio: number;
  k: number;
  seed: number;
  steps: number;
  wall_s: number;
}

export interface JvpProbeResult extends JvpProbeCore {
  alpha: number;
  beta: number;
  gamma: number;
}

export interface JvpProbeStatus {
  probe_id: string;
  status: 'running' | 'done' | 'error';
  kind: 'jvp';
  error?: string | null;
  result: JvpProbeResult | null;
}

// ---- The same probe on a cascade run (k >= 3 prompts, no grid) ----
// Either name a crossing by `cid` or give an arbitrary point as barycentric `weights`.
// The status is read back through the SAME route as the grid probe (getJvpProbe), so the
// result differs only in how it says where it was taken.

export interface CascadeJvpProbeRequest {
  cid?: number;
  weights?: number[];
  seed?: number;
}

export interface CascadeJvpProbeResult extends JvpProbeCore {
  weights?: number[];
}

// ---- Token probe: which words the generation leans on at this point ----

export type TokenProbeWhich = 'a' | 'b' | 'c' | 'd';

export interface TokenProbeRequest {
  which: TokenProbeWhich;
  seed?: number;
}

export interface TokenProbeStartResponse {
  probe_id: string;
  status: 'running' | 'error';
  error?: string | null;
}

export interface TokenProbeRow {
  pos: number;
  cls: 'content' | 'template' | 'pad';
  text: string;
  // sensitivity of the generation to this token's embedding; bigger = more load-bearing
  sigma: number;
}

export interface TokenProbeResult {
  prompt: string;
  rows: TokenProbeRow[];          // sorted by position
  ranking: string[];              // content tokens, descending sigma
  seed: number;
  steps: number;
  n_jvp: number;
  wall_s: number;
}

export interface TokenProbeStatus {
  probe_id: string;
  status: 'running' | 'done' | 'error';
  kind: 'token';
  error?: string | null;
  result: TokenProbeResult | null;
}

// ---- Ridge walk: human-in-the-loop traversal along one ridge ----

export interface WalkStartRequest {
  cid: number;
  direction: number;   // +1 / -1
  n_steps?: number;
  use_jvp?: boolean;
  sig_mode?: 'relative' | 'absolute' | 'continuity';
  // fan = stations on one straight line (old); continuation = predictor-corrector along the secant (new)
  mode?: 'fan' | 'continuation';
}

export interface WalkStep {
  weights: number[];
  // single-seed local contrast across the boundary at this step (not the certified B)
  contrast: number;
  thumb: number;
  center?: number[] | null;   // best crossing estimate (centre of the final bisection bracket)
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
  mode?: string;
  images?: number;            // images the walk rendered
  cert?: { status: string; b?: (number | null)[]; significant?: boolean[]; threshold?: number | null; error?: string } | null;
  jvp?: { rank1_share: number; participation_ratio: number; cos_with_bracket?: number; wall_s: number } | null;
}
