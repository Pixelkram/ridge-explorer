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
  // final high-resolution pass: after the whole chord phase, the top hires_top_pct % of all
  // crossings get hires_factor - 1 extra cheap probes inside their bracket, re-detected at
  // stride/hires_factor (sharper positions; a bracket hiding several boundaries splits)
  hires?: boolean;
  hires_top_pct?: number;
  hires_factor?: number;
  // how those probes are spent: "bracket" (default) = finer along the chord, a sharper position;
  // "cloud" = hires_cloud_n random points in the tangent ball of radius hires_cloud_r around each
  // selected crossing, which measures the local normal, a chord-free position and junction hints
  hires_mode?: 'bracket' | 'cloud';
  hires_cloud_n?: number;
  hires_cloud_r?: number;
  // nearest neighbours each cloud point averages its divergence over (the cloud's stand-in for
  // the 4 grid neighbours of a lattice sensitivity)
  hires_cloud_k?: number;
  // survey randomness apart from the image seed (null = seed)
  chord_seed?: number | null;
  // trace phase: walk every significant crossing both ways, certify, link crossings the walks reach
  trace?: boolean;
  trace_steps?: number;
  trace_certify?: boolean;
  // how chord probes read their label: 'steps' (complete probe_steps-step images, the backend
  // default) or 'staged' (full schedule to step staged_t, x̂0 readout, flagged segments finished
  // to exact labels). The panel always sends it, from frontend/src/probeDefaults.ts
  probe_mode?: 'steps' | 'staged';
  staged_t?: number;
  staged_theta?: number;
}

/** A staged run's ledger (services/staged.py Ledger.report). */
export interface StagedStats {
  t: number;
  steps: number;
  readouts: number;
  resumed: number;               // finished from a cached latent
  from_scratch: number;          // finished from noise (latent evicted from the LRU)
  resumed_share: number | null;  // finished per readout
  segments: number;
  flagged: number;
  flagged_share: number | null;
  image_eq: number;              // measured: worker seconds over one full image's seconds
  image_eq_nominal: number;      // t/S per readout + (S-t)/S per finish
  image_eq_per_readout: number | null;
  decode_share: number | null;   // decode + DINOv2 as a share of one full image
  unit_s: number | null;
  rebracket_skipped: number;
  cache_misses: number;
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
  // staged runs: the image is still the x̂0 preview of an unfinished probe
  preview?: boolean;
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

// CLOUD mode of the high-resolution pass, around one crossing: the ball that was drawn, how its
// points split between the two sides, and what the two clusters say about the boundary here
export interface CascadeCloud {
  // points that arrived, and the ball's radius in tangent units
  n: number;
  r: number;
  // shares on side A, on side B, and in neither basin (further than the crossing threshold
  // from both side signatures)
  frac_a: number;
  frac_b: number;
  frac_other: number;
  // unit weight-space direction mean_B - mean_A and the midpoint of the two means; null unless
  // at least 2 points landed on each side. Where present they REPLACE the bracket chord as the
  // crossing's normal and position (so the scoring sides are taken across this normal)
  normal?: number[] | null;
  mid_est?: number[] | null;
  // a third basin turned up in the ball: possibly a junction, where one normal is a poor summary
  junction_hint?: boolean;
  // how many nearest neighbours each point's divergence averaged over (clipped to the pool), and
  // that divergence summarised: median, max, and the share of points above the crossing threshold
  // -- the share of the ball sitting ON a boundary rather than inside a basin
  k?: number;
  div_median?: number;
  div_max?: number;
  boundary_frac?: number;
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
  // final high-resolution pass: this bracket was re-probed at stride/hires_factor.
  // hires_width = the spacing the position is now quoted at (null when the pass found the change
  // spread over the bracket and kept it); split_from = the crossing this one was split out of
  hires?: boolean;
  hires_width?: number | null;
  split_from?: number | null;
  // cloud mode of that pass: which mode refined this crossing ("cloud"; null in bracket mode),
  // the ball's summary, and its points for the map -- [weights, side, image index, divergence]
  hires_mode?: string | null;
  cloud?: CascadeCloud | null;
  cloud_pts?: [number[], string, number, number][] | null;
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
  // the final high-resolution pass this run was started with: whether it ran, the divergence
  // percentile of crossings it refined, and how much finer it probed their brackets
  hires?: boolean;
  hires_top_pct?: number;
  hires_factor?: number;
  // which mode it ran in ("bracket" = finer along the chord, "cloud" = a ball around the
  // crossing), and for the cloud how many points per crossing at what radius
  hires_mode?: string;
  hires_cloud_n?: number;
  hires_cloud_r?: number;
  hires_cloud_k?: number;
  // false = detection only: the crossings were never bisected or scored, so their positions
  // carry bracket precision (+-stride/2) and b/significant mean nothing
  certify?: boolean;
  // the probe mode this run was started with, and (staged) its readout step, threshold, ledger
  probe_mode?: 'steps' | 'staged';
  staged_t?: number | null;
  staged_theta?: number | null;
  staged?: StagedStats | null;
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

// ---- AMR: nested barycentric lattices, refined around detected boundary edges ----
// Validated in search_problem h25f: at k=4 it reaches the full-lattice ceiling at 55 % of
// exhaustive cost (7x cheaper than chords for the same coverage), and it does NOT scale --
// the fan-out is factor^(k-1), so the backend supports k in {3, 4} and refuses k >= 5 with
// the projected probe count.

export interface AmrStartRequest {
  // 3 or 4 prompts; more is accepted by the schema only to be refused with the cost
  prompts: string[];
  // points per simplex edge at the coarsest level (evaluated in full)
  base?: number;
  // rungs in the ladder, including the coarsest
  levels?: number;
  // each rung is this many times finer than the last
  factor?: 2 | 3;
  // refinement radius around a detected boundary edge, in coarse cells (1 = of record)
  r_ref?: 1 | 2;
  seed?: number;
  steps?: number;
  // lattice probes on the cheap field (null = full fidelity, double the cost per cell)
  probe_steps?: number | null;
}

export interface AmrStartResponse {
  run_id: string;
  status: string;
  error?: string | null;
}

export interface AmrLevelStat {
  level: number;              // points per simplex edge
  cells: number;              // the full lattice at this level, C(level+k-1, k-1)
  // cells evaluated at this level: the refined set plus the ones carried over from a
  // coarser level (nested lattices, so those cost nothing again)
  n_candidates: number;
  n_evaluated: number;        // NEW probes rendered here -- what the cost is charged on
  n_edges: number;
  cost_image_eq: number;      // 0.5 per cheap probe, 1.0 at full steps
}

// Points and edges arrive as parallel arrays (as CascadeStatus serves its probe cloud): the
// map draws thousands of them and wants positions and readings in the same order.
export interface AmrStatus {
  run_id: string;
  status: string;
  phase: string;
  k: number;
  prompts: string[];
  schedule: number[];
  base: number;
  levels: number;
  factor: number;
  r_ref: number;
  probe_steps?: number | null;
  generated: number;
  phase_done: number;
  phase_total: number;
  recent_thumbs: number[];
  // evaluated cells: weights, the level that paid for each, its thumbnail index, and its
  // local divergence (max 1-cos over its measured lattice neighbours). A cell's index in
  // these arrays is the point id the edges refer to.
  points: number[][];
  point_levels: number[];
  point_images: number[];
  point_divs: (number | null)[];
  // boundary edges as point-id pairs, with the level they were found at and the cosine
  // distance across them (> 0.35, the Cascade's crossing threshold)
  edges: [number, number][];
  edge_levels: number[];
  edge_divs: number[];
  levels_stats: AmrLevelStat[];
  cost_image_eq: number;
  notes: string[];
  error?: string | null;
}

// ---- Metropolis ridge sampler (backend/services/metro.py) ----
// A chain whose stationary law is the sharpness field itself: pi ~ S(w)^beta, with
// S(w) = the mean divergence at the chord stride over m random tangent directions.
// Validated in search_problem h25a (replicated): beta = 1 reproduces the S-weighted law
// under single-seed noise (on-boundary 0.673 against the ideal 0.677, KS 0.083, a 2.3-2.6x
// enrichment over uniform sampling); beta = 2 does NOT (KS 0.19-0.28), because squaring an
// m-direction estimate of S amplifies the estimator noise -- hence the 1.5 cap and the
// warning above 1. It is a SAMPLER, not a search: at equal cost it finds 0.91-1.05x the
// distinct boundaries plain IUR chords do, so the Cascade stays the instrument for
// "where are the boundaries".

export interface MetroStartRequest {
  prompts: string[];                 // 3 or more
  beta?: number;                     // target exponent, 0.5..1.5 (1 = of record)
  sigma?: number;                    // random-walk step in weight-space units (0.03)
  m?: number;                        // tangent directions one energy averages (2)
  chains?: number;                   // C (60)
  chain_steps?: number;              // L -- the MCMC chain length, NOT the denoising steps
  seed_mode?: 'cascade' | 'iur';
  cascade_run_id?: string | null;
  n_seed_chords?: number;            // IUR mode: chords drawn to find the seed crossings
  seed?: number;
  steps?: number;                    // denoising steps of a full image
  probe_steps?: number | null;       // probes on the cheap field (null = full fidelity)
  render_full?: boolean;             // re-render every distinct accepted state at `steps`
}

export interface MetroStartResponse {
  run_id: string;
  status: string;
  error?: string | null;
}

export interface MetroSample {
  chain: number;
  step: number;                      // the round it was accepted at
  weights: number[];
  s: number;                         // its energy -- what the map and the gallery sort by
  image: number;                     // the cheap-field probe rendered at it
  full_image?: number | null;        // its full-fidelity re-render, when that pass ran
}

// Also the per-chain summary the live panel's chain table reads (id, steps done, current S,
// acceptance rate, stopped, moved) -- one list rather than a second one beside it.
export interface MetroChainStat {
  chain: number;
  weights: number[];
  s?: number | null;
  image: number;
  seed_weights: number[];
  seed_s?: number | null;
  seed_image: number;
  n_propose: number;
  n_accept: number;
  n_outside: number;                 // proposals that left the simplex (rejected for free)
  moved: boolean;
  stopped: boolean;                  // frozen from the panel: no more proposals, samples kept
  acc_rate?: number | null;          // n_accept / n_propose, served so the two cannot drift
  joined_round: number;              // 0 = seeded at the start; else the round it was added at
}

// One completed MCMC round. `n_proposed` counts every chain that proposed, out-of-simplex
// ones included, so `acc_rate` is the same ratio the summary's `acceptance` reports.
// `sigma`/`beta` are the kernel THIS round ran under: a run whose kernel was driven mid-flight
// has rows that differ, and that difference is where the law changed.
export interface MetroRoundLog {
  round: number;
  n_proposed: number;
  n_accepted: number;
  acc_rate?: number | null;
  mean_S_states?: number | null;     // over the chain set at the end of the round
  mean_S_accepted?: number | null;   // over just this round's accepted proposals
  cost_so_far: number;
  t_wall: number;
  sigma: number;
  beta: number;
}

// One proposal, which is what lets the map animate the chain rather than only show where it
// ended up. A proposal that left the simplex, and one whose probes never arrived, are both
// here with S_prop null and accepted false -- real rejections, drawn as such.
export interface MetroEvent {
  chain: number;
  w_from: number[];
  w_prop: number[];
  S_from?: number | null;
  S_prop?: number | null;
  accepted: boolean;
  image_idx_prop?: number | null;    // the image rendered AT the proposal, if any
}

export interface MetroRoundEvents {
  round: number;
  events: MetroEvent[];
}

/** The kernel in force right now -- what the NEXT round will run under. */
export interface MetroParams {
  sigma: number;
  beta: number;
}

export interface MetroParamsRequest {
  sigma?: number;                    // 0 < sigma <= 0.2, the start form's bound
  beta?: number;                     // 0.5..1.5, likewise
}

// Every field here is quoted over the ACCEPTED states (what the gallery shows), not over the
// chains' time average with repeats: a statistic of this sample, not an estimate of the law.
export interface MetroSummary {
  n_samples: number;
  mean_s?: number | null;
  seed_mean_s?: number | null;       // mean energy of the crossings the chains started on
  on_boundary_frac?: number | null;  // share past COS_T = 0.35 (h25a: 0.67 vs 0.26 uniform)
  acceptance?: number | null;
  n_propose: number;
  n_accept: number;
  n_outside: number;
  chains_never_moved: number;        // seed sheets the run kept (h25a: 1-5 % of chains)
  n_chains: number;
  n_chains_stopped: number;          // frozen by hand
  n_chains_moving: number;           // a starting energy and not stopped
  n_seed_crossings: number;
  n_chords: number;
  n_probes: number;
  n_full: number;
  rounds_done: number;
  cost_image_eq: number;
}

/** One probe of the seed survey, as it arrived. `div` is null until its chord neighbour lands. */
export interface MetroSeedProbe {
  w: number[];
  div?: number | null;
  image_idx?: number | null;
}

/** One seed crossing: a probe pair past COS_T, at its midpoint, with that pair's divergence. */
export interface MetroSeedCrossing {
  w: number[];
  div?: number | null;
}

export interface MetroStatus {
  run_id: string;
  status: string;
  phase: string;
  k: number;
  prompts: string[];
  beta: number;
  sigma: number;
  m: number;
  chains: number;
  chain_steps: number;
  delta: number;                     // the stride the energy is read at (0.025)
  seed_mode: string;
  cascade_run_id?: string | null;
  n_seed_chords: number;
  probe_steps?: number | null;
  render_full: boolean;
  seed: number;
  steps: number;
  generated: number;
  phase_done: number;
  phase_total: number;
  round_done: number;
  recent_thumbs: number[];
  seeds: number[][];                 // the crossings the chains were seeded at
  seed_divs: (number | null)[];      // divergence ACROSS each bracket (reported, not an S)
  // --- the seed phase, growing WHILE it runs: the chords the seeds come from, every probe that
  // has arrived (strided to 4,000 served), and the crossings found so far. Optional because a
  // backend that predates them serves a status without them, and the panel still has to draw.
  seed_chords?: number[][][];        // [[w_start, w_end], ...] in weight space
  seed_probes?: MetroSeedProbe[];
  seed_crossings?: MetroSeedCrossing[];
  chains_stats: MetroChainStat[];
  samples: MetroSample[];
  // --- the live view
  rounds: MetroRoundLog[];
  last_events: MetroEvent[];
  recent_events: MetroRoundEvents[]; // ring of the last 5 rounds, so a slow poll misses none
  // waiting between rounds. NOT a status: the run is still "running", it just is not
  // dispatching, so cancel (and every other status check) still means what it meant
  paused: boolean;
  // the kernel in force NOW, which is not necessarily `sigma`/`beta` above -- those are what
  // the run was STARTED with
  params: MetroParams;
  params_changed: boolean;
  summary: MetroSummary;
  notes: string[];
  error?: string | null;
}

export interface MetroCascadeRun {
  run_id: string;
  status: string;
  k: number;
  prompts: string[];
  n_crossings: number;
  certify: boolean;
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

// ---- Ridge microscope: an image lattice on a 2-D plane through a point (any k) ----

export type MicroMode = 'crossing' | 'prompt-swap' | 'random';

/** One body for /plan and /start (backend MicroscopeRequest). */
export interface MicroscopeRequest {
  mode: MicroMode;
  prompts?: string[] | null;          // crossing mode: taken from the cascade run
  centre?: number[] | null;           // default: the crossing midpoint / the barycentre
  cascade_run_id?: string | null;
  cid?: number | null;
  swap_a?: number[] | null;           // [i, j]: e1 = P_i up / P_j down
  swap_b?: number[] | null;           // [p, q]: e2 likewise, orthogonalised
  grid?: 3 | 5 | 7;
  s?: number;                         // half-width in tangent units
  plane_seed?: number | null;
  e2_redraw?: number;
  seed?: number;
  steps?: number;
}

export interface MicroCell {
  ia: number;                         // index along e1 (screen x)
  ib: number;                         // index along e2 (screen y, up)
  a: number;
  b: number;
  w: number[];
  inside: boolean;
  image: number;                      // -1 until rendered; outside cells are never rendered
  zoom_cost?: number | null;          // new images a click here renders (0 = level exists)
}

export interface MicroEdge {
  a: number[];                        // [ia, ib]
  b: number[];
  div: number;                        // 1 - cos between the two cells
  boundary: boolean;                  // div > COS_T
}

export interface MicroPlan {
  k: number;
  prompts: string[];
  mode: MicroMode;
  centre: number[];
  s: number;
  grid: number;
  e1: number[];
  e2: number[];
  plane_note: string;
  biplot: number[][];                 // per prompt: [e1[i], e2[i]]
  cells: MicroCell[];
  n_inside: number;
  cost_image_eq: number;
  max_image_eq: number;
  seed: number;
  steps: number;
  source_run?: string | null;
  source_cid?: number | null;
  notes: string[];
}

export interface MicroStartResponse {
  run_id: string;
  status: string;
  level: number;
  cost_image_eq: number;
}

export interface MicroLevel {
  level: number;
  parent: number | null;
  parent_cell: number[] | null;
  centre: number[];
  s: number;
  grid: number;
  status: string;
  cells: MicroCell[];
  edges: MicroEdge[];
  n_inside: number;
  n_new: number;
  n_reused: number;
  notes: string[];
  error?: string | null;
}

export interface MicroStatus {
  run_id: string;
  status: string;
  k: number;
  prompts: string[];
  mode: string;
  e1: number[];
  e2: number[];
  plane_note: string;
  biplot: number[][];
  grid: number;
  seed: number;
  steps: number;
  source_run?: string | null;
  source_cid?: number | null;
  levels: MicroLevel[];
  cost_image_eq: number;
  max_image_eq: number;
  s_min: number;
  cos_t: number;
  generated: number;
  notes: string[];
  error?: string | null;
}

export interface MicroZoomResponse {
  level: number;
  cached: boolean;
  status: string;
  cost_image_eq: number;
}

// ---- Mixing desk: one WeightLifter slider per prompt through the current mix (any k) ----

export interface DeskStartRequest {
  prompts: string[];
  w0?: number[] | null;               // null = the barycentre
  dalpha?: number;                    // sample spacing along each line (0.05)
  refine?: boolean;                   // bisect each change to dalpha/8 at full fidelity
  seed?: number;
  steps?: number;
  probe_steps?: number | null;        // cheap field (4); null = full fidelity
  probe_mode?: 'steps' | 'staged';    // see CascadeStartRequest
  staged_t?: number;
  staged_theta?: number;
}

/** A new current mix: outright (`w0`) or a released marker (`line` + `alpha`). */
export interface DeskMoveRequest {
  w0?: number[] | null;
  line?: number | null;
  alpha?: number | null;
  dalpha?: number | null;
  refine?: boolean | null;
}

export interface DeskStartResponse {
  desk_id: string;
  status: string;
  position: number;
  cost_image_eq: number;
  cached_lines: number;
}

export interface DeskFlip {
  lo: number;                         // sample indices of the bracket
  hi: number;
  alpha: number;                      // where the change is quoted
  div: number;                        // 1 - cos across the cheap bracket
  refined: boolean;
  width: number;                      // the bracket it is quoted at
  confirmed: boolean | null;          // full-fidelity ends still past COS_T (refined only)
  full_div: number | null;
  img_lo: number;
  img_hi: number;
}

export interface DeskSegment {
  lo: number;
  hi: number;
  alpha_lo: number;
  alpha_hi: number;
  thumb: number;
}

export interface DeskLine {
  i: number;
  prompt: string;
  alpha0: number;                     // the current mix's weight on this prompt
  degenerate: boolean;
  cached: boolean;
  status: string;
  alphas: number[];
  images: number[];
  divs: (number | null)[];
  flips: DeskFlip[];
  segments: DeskSegment[];
  // staged desks: per sample, finished (exact label)?; per segment, flagged by the readout?
  resumed?: boolean[];
  flagged?: boolean[];
}

export interface DeskFlipRef {
  dist: number;
  alpha: number;
  thumb: number;
  refined: boolean;
}

export interface DeskReadout {
  i: number;
  prompt: string;
  up: DeskFlipRef | null;
  down: DeskFlipRef | null;
  nearest: number | null;
}

export interface DeskPositionRef {
  pid: number;
  w0: number[];
  status: string;
  cost_image_eq: number;
}

export interface DeskStatus {
  desk_id: string;
  status: string;
  k: number;
  prompts: string[];
  seed: number;
  steps: number;
  probe_steps: number | null;
  probe_mode?: 'steps' | 'staged';
  staged_t?: number | null;
  staged_theta?: number | null;
  staged?: StagedStats | null;
  position: number;
  w0: number[];
  w0_image: number;
  dalpha: number;
  refine: boolean;
  lines: DeskLine[];
  readout: DeskReadout[];
  readout_label: string;
  positions: DeskPositionRef[];
  cost_image_eq: number;
  position_cost: number;
  max_request_image_eq: number;
  max_session_image_eq: number;
  cos_t: number;
  generated: number;
  notes: string[];
  error?: string | null;
}
