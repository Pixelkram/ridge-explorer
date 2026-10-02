from typing import Literal

from pydantic import BaseModel, Field, model_validator
from backend import config


def _check_simplex_prompts(req):
    """A 3-D grid mixes four prompt embeddings, so all four must exist.

    `dimensions=3` with an empty prompt_c reached `float * None` deep inside
    _process_generate: the task raised in the worker, emitted zero cells, and the job
    sat at 0/N in phase 'generating' forever. Reject it at the request boundary.
    """
    if req.dimensions == 3:
        missing = [n for n in ("prompt_c", "prompt_d") if not getattr(req, n, "").strip()]
        if missing:
            raise ValueError(f"dimensions=3 needs {' and '.join(missing)}")
    return req


class GridStartRequest(BaseModel):
    # Bounded because every one of these multiplies GPU cost: cells are grid_size**2
    # (grid_size**3 in 3-D) renders of height*width pixels, and all four were
    # unconstrained ints — grid_size=4096 or height=4096 was an OOM the server took.
    prompt_a: str
    prompt_b: str
    prompt_c: str = ""
    prompt_d: str = ""  # 4th prompt for 3D mode
    dimensions: int = Field(2, ge=2, le=3)
    grid_size: int = Field(15, ge=2, le=256)
    seed: int = 42
    seed_count: int = Field(1, ge=1, le=16)
    height: int = Field(config.DEFAULT_HEIGHT, ge=64, le=1024)
    width: int = Field(config.DEFAULT_WIDTH, ge=64, le=1024)
    steps: int = Field(config.DEFAULT_NUM_INFERENCE_STEPS, ge=1, le=50)
    guidance_scale: float = Field(4.0, ge=0.0, le=20.0)

    _prompts = model_validator(mode="after")(_check_simplex_prompts)


class GridStartResponse(BaseModel):
    job_id: str
    total_cells: int
    dimensions: int
    status: str


class RenderHQRequest(BaseModel):
    tau: float = 1.5


class SeedProbeRequest(BaseModel):
    alpha: float
    beta: float
    gamma: float = 0.0
    seed_start: int = 0
    seed_end: int = 19
    steps: int = 0  # 0 = inherit from parent job


class SeedProbeResponse(BaseModel):
    probe_id: str
    total: int
    status: str


class SeedProbeStatus(BaseModel):
    probe_id: str
    alpha: float
    beta: float
    seeds: list[int]
    images: list[str | None]
    complete: bool


class RefineRequest(BaseModel):
    tau: float = 1.5
    multiplier: int = 4
    extra_positions: list[tuple[int, int]] = []


class RefineResponse(BaseModel):
    refine_job_id: str
    parent_job_id: str
    total_cells: int
    status: str


class CellStatus(BaseModel):
    row: int
    col: int
    depth: int = 0  # z-index for 3D grids
    alpha: float
    beta: float
    gamma: float = 0.0
    status: str
    sensitivity: float | None = None
    cluster: int | None = None
    thumbnail_url: str | None = None
    hq_url: str | None = None
    span: int = 1


class GridStatusResponse(BaseModel):
    job_id: str
    status: str
    phase: str
    dimensions: int = 2
    grid_size: int
    cells_generated: int
    cells_total: int
    cells: list[CellStatus]
    seed_cells: dict[str, list[CellStatus]] | None = None
    seeds: list[int] = []
    prompt_a: str
    prompt_b: str
    prompt_c: str
    prompt_d: str = ""
    heatmap_url: str | None = None
    overlay_url: str | None = None
    cluster_url: str | None = None
    image_grid_url: str | None = None
    ridge_mesh_url: str | None = None  # 3D: marching cubes mesh as JSON
    # Set when a GPU worker failed the job's tasks. Without a field here the poller had
    # no way to tell "still generating" from "the work died", and the job sat at
    # phase 'generating' for the rest of the session.
    error: str | None = None


class FastScanRequest(BaseModel):
    prompt_a: str
    prompt_b: str
    prompt_c: str = ""
    prompt_d: str = ""  # 4th prompt for 3D mode
    dimensions: int = Field(2, ge=2, le=3)
    grid_size: int = Field(50, ge=2, le=256)
    seed: int = 42
    height: int = Field(config.DEFAULT_HEIGHT, ge=64, le=1024)
    width: int = Field(config.DEFAULT_WIDTH, ge=64, le=1024)
    guidance_scale: float = Field(1.0, ge=0.0, le=20.0)  # 1.0 = single forward pass (no CFG), fastest

    _prompts = model_validator(mode="after")(_check_simplex_prompts)


class FastScanResponse(BaseModel):
    job_id: str
    total_cells: int
    status: str


class GenerateSelectedRequest(BaseModel):
    tau: float = 1.5
    height: int = config.DEFAULT_HEIGHT
    width: int = config.DEFAULT_WIDTH
    steps: int = config.DEFAULT_NUM_INFERENCE_STEPS
    guidance_scale: float = 4.0


class GenerateSelectedResponse(BaseModel):
    status: str
    total_cells: int = 0


class MFScanRequest(BaseModel):
    prompt_a: str
    prompt_b: str
    prompt_c: str = ""
    grid_size: int = Field(50, ge=2, le=256)
    seed: int = 42
    budget: int = Field(80, ge=1, le=5000)
    tau_mf: float = Field(1.3, gt=0.0, le=10.0)
    height: int = Field(config.DEFAULT_HEIGHT, ge=64, le=1024)
    width: int = Field(config.DEFAULT_WIDTH, ge=64, le=1024)
    steps: int = Field(config.DEFAULT_NUM_INFERENCE_STEPS, ge=1, le=50)
    guidance_scale: float = Field(4.0, ge=0.0, le=20.0)


class MFScanResponse(BaseModel):
    job_id: str
    total_cells: int
    budget: int
    status: str


class HealthResponse(BaseModel):
    status: str          # ok | degraded | down, from live worker processes
    n_gpus: int
    workers_ready: int
    model: str
    recent_errors: list[str] = []


class RidgeGraphRequest(BaseModel):
    """Rebuild the ridge graph for a finished job. CPU only — no generation."""
    # bounded so a stray value cannot reach scipy: k < 2 has no boundary to trace and a
    # huge k is a long CPU stall on the event loop's threadpool
    k: int = Field(8, ge=2, le=64)        # number of semantic basins to cluster into
    h_frac: float = Field(0.10, gt=0.0, le=1.0)   # persistence depth, fraction of range
    min_arc: int = Field(3, ge=2, le=1000)        # discard arcs shorter than this


class RidgeGraphResponse(BaseModel):
    job_id: str
    n_arcs: int
    n_basins: int
    n_junctions: int
    params: dict


class HikeStartRequest(BaseModel):
    """Launch a population of hikers from a seed prompt triplet."""
    # bounded like RidgeGraphRequest, but here the cost is GPU time, not CPU: the job is
    # budget grids of grid_size*(grid_size+1)/2 images each, so an unbounded grid_size or
    # budget queues millions of generations. grid_size >= 2 because hiker.bary divides by
    # (gs - 1); beam >= 1 because beam <= 0 silently collapsed the population to width 1.
    prompt_a: str
    prompt_b: str
    prompt_c: str
    # Ceilings raised 2026-08-15 at the user's request; the UI no longer caps these.
    # They remain FINITE only so a typo cannot queue months of generation — a hike is
    # budget grids of grid_size*(grid_size+1)/2 images each, so grid_size=100 budget=1000
    # is already ~5M renders. Anything running can be stopped with POST /hike/{id}/cancel.
    beam: int = Field(4, ge=1, le=32)       # population width
    budget: int = Field(16, ge=1, le=1000)   # total grids to generate (this is the GPU cost)
    grid_size: int = Field(25, ge=2, le=100)
    # Ridge refinement, the hiker analogue of /refine. Each round doubles the lattice
    # (gs -> 2gs-1) and generates ONLY near the ridge, so cost is refine_cap per round
    # per chain rather than the full finer simplex. Bounded because it is GPU time.
    # Padding ring around the simplex. Hull cells otherwise keep 2.6 of 4 neighbours and
    # their sensitivity carries ~16% error on the alpha=0/beta=0 edges and ~30% on the
    # hypotenuse -- precisely where exits are chosen. The ring is measured, never walked to.
    # Seeds. seed_count > 1 generates every cell at seed, seed+1, ... and walks the
    # AVERAGED sensitivity field; the per-seed images are all kept so the UI can switch
    # between them at a fixed ridge. Cost is linear in seed_count.
    seed_count: int = Field(1, ge=1, le=8)
    margin: int = Field(0, ge=0, le=8)
    refine_rounds: int = Field(0, ge=0, le=2)
    refine_cap: int = Field(200, ge=10, le=20000)
    seed: int = 42
    height: int = Field(config.DEFAULT_HEIGHT, ge=64, le=1024)
    width: int = Field(config.DEFAULT_WIDTH, ge=64, le=1024)
    steps: int = Field(config.DEFAULT_NUM_INFERENCE_STEPS, ge=1, le=50)
    guidance_scale: float = Field(4.0, ge=0.0, le=20.0)
    k_basins: int = Field(6, ge=2, le=32)
    h_frac: float = Field(0.10, gt=0.0, le=1.0)
    prompt_pool: list[str] | None = None   # overrides config.HIKE_PROMPT_POOL


class HikeStartResponse(BaseModel):
    hike_id: str
    budget: int
    beam: int
    status: str


class HikeBranchRequest(HikeStartRequest):
    """Seed a new hike at one station of an existing walk.

    Inherits every sampler/budget field from HikeStartRequest; prompt_a/b/c are ignored
    because the seed triangle comes from the chosen station plus two freshly drawn
    prompts. They keep defaults so the client need not invent them.
    """
    prompt_a: str = ""
    prompt_b: str = ""
    prompt_c: str = ""
    hop: int = Field(..., ge=0)
    chain: int = Field(..., ge=0)
    station: int = Field(..., ge=0)


class HikeStatus(BaseModel):
    hike_id: str
    status: str
    hop: int
    grids_done: int
    # grids that were generated and charged but whose cells never arrived; they are
    # part of grids_done, so without this the spent budget looks like completed work
    grids_dropped: int = 0
    budget: int
    # extra cells generated by ridge refinement; not grids, so kept out of grids_done
    refined_cells: int = 0
    chains: list
    notes: list[str] = []
    error: str | None = None


# --- Discovery: the procedure the ablations support (services/discover.py) -----


class DiscoverStartRequest(BaseModel):
    """Bounds are the measured usable ranges, not arbitrary guards -- see
    services/discover.DEFAULTS_PROVENANCE for the experiment behind each."""
    # k: 3-4 is the default band. Higher k buys DIVERSITY at matched commitment
    # (partial rho +0.898) at a small coherence cost (-0.205) [E29]. Capped at 8
    # because that is the largest k we have measured the commitment optimum at.
    k: int = Field(3, ge=2, le=8)
    # commitment = E[max weight]. Coherence saturates ~0.60, diversity peaks ~0.50,
    # and the optimum is stable across k=3,4,5,8 [E28, E29]. The hard floor is 1/k
    # and is checked at request time, since it depends on k.
    commitment: float = Field(0.55, gt=0.0, lt=1.0)
    # Mean pairwise CLIP-text similarity of the drawn prompts. Novelty falls with
    # similarity (rho -0.817), coherence rises (+0.634), product peaks at 0.56 [E10].
    target_sim: float = Field(0.56, ge=0.2, le=0.95)
    # Images per simplex before restarting. A simplex exhausts: -0.040 diversity over
    # 960 images in one, none at 240 [E32, E34]. This is the loop's whole mechanism.
    batch: int = Field(300, ge=20, le=2000)
    total: int = Field(1200, ge=1, le=100000)
    seed: int = 42
    steps: int = Field(config.DEFAULT_NUM_INFERENCE_STEPS, ge=1, le=50)
    height: int = Field(config.DEFAULT_HEIGHT, ge=64, le=1024)
    width: int = Field(config.DEFAULT_WIDTH, ge=64, le=1024)
    guidance_scale: float = Field(config.DEFAULT_GUIDANCE_SCALE, ge=0.0, le=20.0)
    prompt_pool: list[str] | None = None
    # Reject vertex-dominated draws. E[max weight] is a MEAN: at 0.55 with k=3, 61% of
    # plain Dirichlet draws still have one prompt above 0.5 and only 21% are genuine
    # three-way mixtures, which in practice returned the source prompts rendered
    # separately. Banding max weight and flooring min weight makes a mixture the typical
    # sample. None = the old unbanded behaviour.
    commit_band: tuple[float, float] | None = None
    min_weight: float = Field(0.0, ge=0.0, le=0.5)
    # Pin the prompts instead of drawing them. Exists so a sampling change can be A/B'd
    # against the identical triple rather than confounded by a fresh draw.
    prompts: list[str] | None = None
    # Ridge targeting. scout_gs > 0 spends a triangular lattice of that size measuring the
    # sensitivity field, then aims the rest of the leg at its steepest cells. 0 = the
    # unguided sampler. Selectivity is the active ingredient: top-decile cells of a survey
    # beat balanced random sampling on 10/10 prompt triplets (redundancy 0.766 vs 0.912),
    # but a run that KEEPS the whole survey shows no benefit (E30/E32) -- which is why the
    # aimed range is reported separately.
    scout_gs: int = Field(0, ge=0, le=40)
    top_frac: float = Field(0.10, gt=0.0, le=1.0)
    # Jitter around an aimed cell, in cell widths. 0 generates at the measured coordinate.
    aim_jitter: float = Field(0.5, ge=0.0, le=2.0)
    # How far below zero a mixing weight may go. 0 keeps every point inside the prompt
    # hull, which caps quality: the sensitivity ridge sits OUTSIDE it (26/36 of the top
    # cells of a 400-cell grid), and a non-negative sampler cannot reach that at all.
    extrapolate: float = Field(0.0, ge=0.0, le=1.0)


class DiscoverStartResponse(BaseModel):
    run_id: str
    status: str
    total: int
    error: str | None = None


class DiscoverSimplex(BaseModel):
    index: int
    prompts: list[str]
    mean_sim: float
    alpha: float
    n_points: int
    done: int
    diversity: float | None = None
    # Fraction of this leg's draws that are genuine mixtures (max weight < 0.5) rather
    # than one prompt with a tint. Nothing measured this before, which is how a run of
    # source-prompt renders passed every check.
    blend_frac: float | None = None
    # Survey cost, and the index range of the AIMED images. The survey is a uniform
    # lattice -- keeping it mixed into the harvest is what wiped out the benefit of
    # aiming in E30/E32, so the UI needs to be able to show the aimed subset alone.
    scout_n: int = 0
    aim_lo: int = -1
    aim_hi: int = -1
    # Selectivity actually achieved. draws_per_cell near 1 and a high survey_ratio are
    # what distinguish an interesting harvest from a padded one.
    draws_per_cell: float | None = None
    survey_ratio: float | None = None


class DiscoverStatus(BaseModel):
    run_id: str
    status: str
    generated: int
    total: int
    k: int = 0
    commitment: float = 0.0
    # Mean pairwise DINOv2 distance over the harvest. Reference-free -- it depends only
    # on the generated images -- which is why it is the live number: the novelty metrics
    # depend on a prompt reference whose choice was worth 26 of 36 pp in E34b.
    diversity: float | None = None
    # Mean cosine to the nearest OTHER image. LOWER is better. Reported alongside
    # diversity because diversity alone cannot distinguish "three source prompts rendered
    # separately" (~0.74) from "many varied hybrids" (~0.75) -- a run of wheat fields and
    # red hearts scored 0.740 and passed every check until the images were looked at.
    redundancy: float | None = None
    simplices: list[DiscoverSimplex] = []
    notes: list[str] = []
    error: str | None = None


class DiscoverDefaults(BaseModel):
    k: int
    commitment: float
    target_sim: float
    batch: int
    steps: int
    provenance: dict
    alpha_table: dict


class SurpriseSampleRequest(BaseModel):
    # -1 = calm (argmin S), 0 = uniform, +1 = surprise (argmax S). See services/surprise.py.
    surprise: float = 1.0
    n: int = 12
    seed: int | None = None


class SurpriseSampleResponse(BaseModel):
    job_id: str
    surprise: float
    # (row, col) of the sampled cells, most surprising draw first by sampling order
    cells: list[tuple[int, int]]
    # sampling probability per eligible cell, keyed "row,col" — for heatmap overlays
    probabilities: dict[str, float]


# --- Cascade: fast isolation + dense refinement (services/cascade.py) ----------


class CascadeStartRequest(BaseModel):
    """Bounds follow the calibrated cascade of record (search_problem E85/E86/E92).
    Costs: roughly n_chords*20 images for isolation, (crossings+12)*8 for scoring,
    n_patches*25 for refinement -- ~1k images at the defaults."""
    # Number of prompts spanning the simplex. 3-9 measured; the cascade itself is
    # k-invariant in cost per chord (extents shrink but stride is fixed).
    # No upper bound: calibrations run to k=9; beyond that the machinery works but
    # verdicts are extrapolation (backgrounds self-calibrate per run either way).
    k: int = Field(4, ge=3)
    # Pin the prompt set; if omitted, k prompts are drawn from the pool at target_sim
    # (same band logic as Discover -- E10).
    prompts: list[str] | None = None
    target_sim: float = Field(0.56, ge=0.2, le=0.95)
    # Isolation budget. 24 chords ~ 500 images and ~15-25 crossings at k=4 [E92].
    n_chords: int = Field(24, ge=4)
    # Refined regions; one slot is ALWAYS the exploration floor (E77: ranking must
    # not decide what is never inspected).
    n_patches: int = Field(4, ge=2)
    # Run the full pipeline on every detected crossing (rebracket, bisection, coupled-seed B,
    # patches), or stop after detection. The default is OFF: detection alone is ~6x cheaper per
    # crossing, and leaves the run UNCERTIFIED -- positions at bracket precision (+-stride/2),
    # no B, no patches, no walks. Turn it on for certified boundaries.
    certify: bool = False
    # focused exploration: confine the survey to a ball around this recipe
    # (requires pinned prompts so the weights refer to a known basis)
    focus: list[float] | None = None
    focus_radius: float = Field(0.18, gt=0.02, le=1.0)
    # tier-1 detection steps (gated: 93%/94% recall at 4 vs 8; ~2x faster probes).
    # null = probe at full fidelity.
    probe_steps: int | None = Field(4, ge=1, le=50)
    # chord probe spacing in weight space; 0.025 (~1 fine cell) is the protocol of record.
    # Finer resolves boundaries closer together than one stride at a cost ~1/stride; coarser
    # merges them. Detection only -- the continuation walk's corrector spacing is unaffected.
    stride: float = Field(0.025, ge=0.01, le=0.2)
    # recursive ("branching") chords: out of each selected crossing of the previous generation,
    # spawn this many rays and repeat for `depth` generations. 0 = off (one fair generation, the
    # protocol of record). Root chords stay fair area samples; child rays are exploratory
    # (preferential), so the coverage certificate is quoted on the roots only. Total chords
    # are capped at the geometric sum n_chords*(1 + branch + ... + branch^depth).
    branch: int = Field(0, ge=0, le=6)
    depth: int = Field(0, ge=0, le=4)
    # which crossings of a generation spawn the next one: the top this-many % by probe-to-probe
    # divergence (always at least one). 100 = every crossing branches; 20 (default) spends the
    # branching budget on the fifth of the evidence that diverged most.
    branch_top_pct: int = Field(20, ge=1, le=100)
    # final high-resolution pass: after the WHOLE chord phase, the top hires_top_pct % of all
    # detected crossings get hires_factor - 1 extra cheap probes evenly spaced inside their
    # bracket, and the bracket is re-detected at stride/hires_factor. Sharpens the quoted
    # position by the factor, and a bracket that hid several boundaries splits into several
    # crossings. Off by default; costs ~0.5 image per extra probe (probes run on the cheap
    # field), and works with certify either way -- with it on, bisection starts from the
    # sharper bracket.
    hires: bool = False
    hires_top_pct: int = Field(20, ge=1, le=100)
    hires_factor: int = Field(4, ge=2, le=8)
    # which mode the pass spends its probes in. "bracket" (default) is the finer sweep along the
    # chord above: a sharper POSITION. "cloud" draws hires_cloud_n random points in the tangent
    # ball of radius hires_cloud_r around each selected crossing instead: side fractions, the
    # local NORMAL between the two side means, a chord-free position estimate, and a junction hint
    # when a third basin shows up -- all directions the chord says nothing about, at the price of
    # leaving the bracket as it was. hires_cloud_r is in tangent units (0.0236 = one fine cell);
    # the cloud costs ~0.5 image per point (cheap field). hires_cloud_k is how many nearest
    # neighbours (cloud points and the two bracket ends) each cloud point averages its divergence
    # over -- the cloud's stand-in for the 4 grid neighbours of a lattice sensitivity.
    hires_mode: Literal["bracket", "cloud"] = "bracket"
    hires_cloud_n: int = Field(12, ge=4, le=64)
    hires_cloud_r: float = Field(0.05, ge=0.01, le=0.30)
    hires_cloud_k: int = Field(4, ge=1, le=16)
    seed: int = 42
    # survey randomness (chords, background, patches) apart from the image seed; null = seed
    chord_seed: int | None = None
    # trace phase: after the survey, walk both ways from every significant crossing (continuation walk,
    # trace_steps x 0.05 each way), certify the stations, and link crossings a walk reaches
    trace: bool = False
    trace_steps: int = Field(8, ge=1, le=20)
    trace_certify: bool = True
    steps: int = Field(config.DEFAULT_NUM_INFERENCE_STEPS, ge=1, le=50)
    height: int = Field(config.DEFAULT_HEIGHT, ge=64, le=1024)
    width: int = Field(config.DEFAULT_WIDTH, ge=64, le=1024)
    guidance_scale: float = Field(config.DEFAULT_GUIDANCE_SCALE, ge=0.0, le=20.0)


class CascadePointInfo(BaseModel):
    # the recipe behind any generated thumbnail; null if the index is unknown
    weights: list[float] | None = None
    div: float | None = None


class CascadeStartResponse(BaseModel):
    run_id: str
    status: str
    error: str | None = None


class CascadeChord(BaseModel):
    # chord endpoints in weight space, for the map view
    a: list[float]
    b: list[float]


class CascadeChordMeta(BaseModel):
    """Branching provenance of one chord, aligned with `chords`."""
    # 0 = root (fair isotropic-uniform sample); >= 1 = a child ray, one generation deeper
    gen: int = 0
    # chord index this ray was spawned from, and the crossing on it (-1 -1 for roots)
    parent: int = -1
    origin_cid: int = -1
    # angle spreading of a child ray: the line angle it achieved against the nearby chords it was
    # spread away from, and how many of those there were (None/0 for roots, which are not spread)
    min_angle_deg: float | None = None
    n_near: int = 0


class CascadeCloud(BaseModel):
    """CLOUD mode of the high-resolution pass, around ONE crossing: n random points that arrived
    out of the ball of radius r (tangent units), what share of them matched each side, and what
    the two side clusters say about the boundary here."""
    n: int
    r: float
    # shares of the cloud on side A, side B, and in neither basin (further than the crossing
    # threshold from both signatures)
    frac_a: float = 0.0
    frac_b: float = 0.0
    frac_other: float = 0.0
    # unit weight-space direction mean_B - mean_A and the midpoint (mean_A + mean_B)/2; both null
    # unless at least 2 cloud points landed on each side. Where present they REPLACE the bracket
    # chord as the crossing's normal and position (the scoring sides read this normal).
    normal: list[float] | None = None
    mid_est: list[float] | None = None
    # a third basin showed up in the ball: this may be a junction, where one direction is a poor
    # summary of the boundary
    junction_hint: bool = False
    # how many nearest neighbours each point's divergence averaged over (clipped to the pool), and
    # that per-point divergence summarised: median, max, and the share of points above the crossing
    # threshold -- the share of the ball that sits ON a boundary rather than inside a basin
    k: int = 4
    div_median: float = 0.0
    div_max: float = 0.0
    boundary_frac: float = 0.0


class CascadeCrossing(BaseModel):
    cid: int
    weights: list[float]
    # Coupled-seed paired boundary strength B = mean_s(1-cos) at eps=0.047, m=4
    # [E84b: AUROC 1.000 on the exact k=4 field]. None until the score phase.
    b: float | None = None
    # True iff b > the run's own measured background p95 [E89: interior-referenced
    # thresholds, never a no-step null].
    significant: bool = False
    thumb: int = -1
    # generation of the chord this crossing sits on (0 = root; >= 1 = a branching child ray)
    gen: int = 0
    # crossings sharing a ridge_group portray the SAME ridge (side-signature match)
    ridge_group: int | None = None
    # current bisection bracket width (barycentric); ~0.012 = pinned. Drives the
    # lock-on reticle in the map during the pinning phase.
    bracket_w: float | None = None
    # ridge group with the trace phase's links added (None without a trace phase)
    traced_group: int | None = None
    # final high-resolution pass: this bracket was re-probed at stride/hires_factor.
    # hires_width is the sub-bracket spacing the position is now quoted at, or None when the
    # pass found the change spread over the whole bracket (no single sub-step cleared the
    # crossing threshold) and kept the original bracket. split_from names the crossing this one
    # was split out of -- a SECOND boundary inside one chord-stride bracket.
    hires: bool = False
    hires_width: float | None = None
    split_from: int | None = None
    # CLOUD mode of that pass: which mode refined this crossing ("cloud"; null in bracket mode),
    # the ball's summary, and its points for the map -- [[weights...], side, image index,
    # divergence] each, side being "A" / "B" / "other".
    hires_mode: str | None = None
    cloud: CascadeCloud | None = None
    cloud_pts: list | None = None


class CascadePatch(BaseModel):
    region: int
    cid: int
    b: float
    significant: bool
    exploration: bool
    # 5x5 thumb indices; rows = tangent direction, cols = crossing the boundary at
    # 1-cell steps; -1 where the point fell outside the simplex.
    grid: list[list[int]]
    cols_with_crossing: int


class WalkStartRequest(BaseModel):
    cid: int
    # Use the exact-JVP normal at the origin crossing for the transversal probe direction and the tangent
    # plane (one JVP, ~30 s) instead of the bracket chord that happened to cross (services/jvp_probe.py).
    use_jvp: bool = False
    # "absolute": both new sides within COS_T of the origin's sides (the coverage criterion);
    # "relative": each new side closer to its matching origin side than to the opposite one.
    # "continuity": relative + each new station's sides must continue the previous station's (PC_CONT_GAMMA)
    sig_mode: str = Field("relative", pattern="^(absolute|relative|continuity)$")   # relative won its A/B 24/24 (tests/walk_ab_*_relative.json)
    # "fan": stations on one straight line, all rendered at once; "continuation": sequential predictor-corrector
    # along the secant of the last two captured points (RESEARCH_ridge_following_k4.md, fix 1). Continuation won
    # its pre-registered A/B (tests/PREREG_walk_continuation.md, tests/walk_pc_results.json) and is the default.
    mode: str = Field("continuation", pattern="^(fan|continuation)$")
    # research override of the continuity threshold (PC_CONT_GAMMA); null = default
    cont_gamma: float | None = Field(None, gt=0.0, le=10.0)
    # +1 / -1: the two ways along the ridge from the crossing
    direction: int = Field(1, ge=-1, le=1)
    n_steps: int = Field(5, ge=1)


class RenderRequest(BaseModel):
    """Research tool: render arbitrary weight vectors through a run's own generation path; saves embeddings to
    tests/gt/<name>.npz. Used for dense ground-truth lattices."""
    weights: list[list[float]]
    seed: int | None = None
    name: str = Field(..., pattern=r"^[A-Za-z0-9_.-]{1,80}$")


class WalkStep(BaseModel):
    weights: list[float]
    # single-seed local contrast across the boundary at this step (NOT the certified B)
    contrast: float
    thumb: int
    # best estimate of the crossing: centre of the final bisection bracket (the thumb sits within a bracket of it)
    center: list[float] | None = None
    # continuation: largest side move since the previous station / that station's contrast (continuity statistic)
    cont: float | None = None


class WalkStatus(BaseModel):
    walk_id: str
    status: str
    cid: int = -1
    mode: str = "fan"
    images: int = 0              # images the walk rendered
    plane: dict | None = None    # {x0, t, n}: the walk plane span(t, n) through the origin crossing
    cert: dict | None = None     # held-out-seed certification per station (POST .../certify)
    trace: list = []             # continuation: per corrector attempt {j, h, offs, adj 1-cos, ends, chosen}
    jvp: dict | None = None      # {rank1_share, participation_ratio, cos_with_bracket, wall_s} when use_jvp
    steps: list[WalkStep] = []
    # transversal probe lines (pairs of weight vectors) -- the walk's line bundle
    segs: list[list[list[float]]] = []
    notes: list[str] = []
    error: str | None = None


class CascadeStatus(BaseModel):
    run_id: str
    status: str
    phase: str = ""
    generated: int = 0
    phase_done: int = 0
    phase_total: int = 0
    prompts: list[str] = []
    # most recent generated image indices -- the live 'just generated' ticker
    recent_thumbs: list[int] = []
    # completed-generation positions (weight space), capped -- the live map cloud
    points: list[list[float]] = []
    # local divergence per cloud point (aligned with points; null until measured)
    point_divs: list[float | None] = []
    chords: list[CascadeChord] = []
    # branching provenance per chord, aligned with `chords` (all gen 0 without branching)
    chords_meta: list[CascadeChordMeta] = []
    # the chord probe spacing this run was started with (0.025 = protocol of record)
    stride: float = 0.025
    # the branching settings this run was started with: rays per selected crossing (0 = off),
    # generations of them, and the divergence percentile that selects the origins
    branch: int = 0
    depth: int = 0
    branch_top_pct: int = 20
    # the final high-resolution pass this run was started with: whether it ran at all, the
    # divergence percentile of crossings it refined, and how much finer it probed their brackets
    hires: bool = False
    hires_top_pct: int = 20
    hires_factor: int = 4
    # which mode it ran in ("bracket" = finer along the chord, "cloud" = a ball around the
    # crossing) and, for the cloud, how many points per crossing at what radius, over how many
    # nearest neighbours each point's divergence was averaged
    hires_mode: str = "bracket"
    hires_cloud_n: int = 12
    hires_cloud_r: float = 0.05
    hires_cloud_k: int = 4
    # False = detection only: the crossings below were never bisected or scored, so their
    # positions carry bracket precision (+-stride/2) and b/significant mean nothing. The
    # default here is the back-compatible reading (a status without the field predates the
    # flag, so it ran the full pipeline); the REQUEST's default is off.
    certify: bool = True
    crossings: list[CascadeCrossing] = []
    bg_mean: float | None = None
    bg_p95: float | None = None
    # Coverage certificate (Good-Turing over ridge identities): distinct ridges found,
    # how many were crossed exactly once, and the estimated share of boundary area the
    # survey has never crossed. Unbiased chords make this estimable at all.
    distinct_ridges: int | None = None
    singleton_ridges: int | None = None
    unexplored_share: float | None = None
    # which crossings the certificate above was computed from: "all", or "roots" when
    # branching was on -- a child ray starts on a boundary, so it is a preferential sample
    # and would bias an estimate that reads chords as fair samples of boundary AREA.
    stats_scope: str = "all"
    # trace phase: walked ridges ({cid, direction, walk_id, points, cert, end}), crossing pairs a walk joined, and
    # the coverage certificate recomputed with those links (None without a trace phase)
    traces: list[dict] = []
    trace_links: list[list[int]] = []
    traced_ridges: int | None = None
    traced_singletons: int | None = None
    traced_unexplored_share: float | None = None
    patches: list[CascadePatch] = []
    notes: list[str] = []
    error: str | None = None


class LocalSvChord(BaseModel):
    """One chord's sample stations and the local boundary density read at each.

    Parallel arrays (as points/point_divs above): the map colours the polyline segment by
    segment, so it wants the positions and the readings in the same order, not objects.
    """
    # station positions in weight space, from chord endpoint a toward b
    points: list[list[float]] = []
    # local S_V at each station (units only meaningful when calibrated_ok)
    values: list[float] = []
    # percentile rank of each station within the covered part of this map
    ranks: list[float] = []
    # no local evidence here: kernel mass below 5% of its median (drawn grey, never coloured)
    uncovered: list[bool] = []


class LocalSvMap(BaseModel):
    """Kernel-smoothed local Crofton S_V over a finished cascade's own chords.

    Two products of one estimator, and the caller must not confuse them
    (search_problem/outputs/h23_local_sv_kde, replicated in verify/VERIFY.md): the RANKING
    is reliable from ~20 single-seed chords (Spearman 0.78-0.90 vs the converged map), while
    the VALUES in S_V units need >= 80 -- `calibrated_ok`. Calibrated at k=4 only.
    """
    run_id: str
    k: int = 0
    h: float = 0.0                  # kernel width in orthonormal tangent coords (simplex edge = sqrt(2))
    mode: str = "all"               # "all" | "certified" (numerator-only filter; a diagnostic)
    delta: float = 0.0              # chord sampling step; absolute S_V is Delta-dependent
    c_d: float = 0.0                # Crofton constant at d = k-1 (2 exactly at k=4)
    n_chords: int = 0
    n_crossings: int = 0
    # True iff n_chords >= 80: below that the values are worse than the global constant
    calibrated_ok: bool = False
    # the global mean-of-ratios S_V from the same chords -- the constant the map has to beat
    s_global: float | None = None
    # per chord, in the run's own chord order (aligned with CascadeStatus.chords)
    chords: list[LocalSvChord] = []
    # the map read at every crossing, aligned by index with CascadeStatus.crossings
    crossing_cids: list[int] = []
    crossing_values: list[float] = []
    crossing_ranks: list[float] = []
    crossing_uncovered: list[bool] = []
    # optional Dirichlet(1,...,1) sample of the space (?cloud=N). A research read-out only:
    # the map view deliberately draws no heat raster, because a 2-D shadow of a (k-1)-
    # dimensional field invents structure that is pure projection artefact.
    cloud: list[list[float]] = []
    cloud_values: list[float] = []
    cloud_ranks: list[float] = []
    cloud_uncovered: list[bool] = []
    error: str | None = None


# ---- Adaptive refinement (services/amr.py) ----
# Nested barycentric lattices refined only around detected boundary edges. Supported at
# k in {3, 4}; k >= 5 is refused by the router with the projected probe count, because the
# refinement fans out as factor^(k-1) (h25f: 4.8e9 probes at k=9).
class AmrStartRequest(BaseModel):
    """Bounds follow the h25f ladder of record (base 5, factor 2, 4 levels -> 5/10/20/40).

    `prompts` accepts up to 12 entries although only 3 and 4 are SUPPORTED: a 5-prompt
    request is answered with a 400 naming the projected cost, which is the useful answer,
    where a length limit here would only produce an anonymous 422.
    """
    prompts: list[str] = Field(..., min_length=3, max_length=12)
    # points per simplex edge at the coarsest level, which is evaluated in full
    base: int = Field(5, ge=3, le=10)
    # how many lattices the ladder has, including the coarsest
    levels: int = Field(4, ge=1, le=5)
    # each level is this many times finer than the last; 2 (of record) or 3
    factor: Literal[2, 3] = 2
    # refinement radius around a detected boundary edge, in COARSE cells. 1 is of record:
    # at k=4 it lost nothing against 2 (same pairs, same M2) for 22 % fewer probes.
    r_ref: Literal[1, 2] = 1
    seed: int = 42
    steps: int = Field(config.DEFAULT_NUM_INFERENCE_STEPS, ge=1, le=50)
    # lattice probes run on the cheap field, as the Cascade's chord probes do (gated at
    # 93 %/94 % recall); null probes at full fidelity and doubles the cost per cell
    probe_steps: int | None = Field(4, ge=1, le=50)
    height: int = Field(config.DEFAULT_HEIGHT, ge=64, le=1024)
    width: int = Field(config.DEFAULT_WIDTH, ge=64, le=1024)
    guidance_scale: float = Field(config.DEFAULT_GUIDANCE_SCALE, ge=0.0, le=20.0)


class AmrStartResponse(BaseModel):
    run_id: str
    status: str
    error: str | None = None


class AmrLevelStat(BaseModel):
    """One rung of the ladder."""
    level: int                   # points per simplex edge
    cells: int                   # the full lattice at this level, C(level+k-1, k-1)
    # cells this level evaluated: the refined set plus every cell carried over from a
    # coarser level (the lattices are nested, so those cost nothing again)
    n_candidates: int
    # NEW probes rendered at this level -- what the cost below is charged on
    n_evaluated: int
    n_edges: int                 # boundary edges detected among evaluated adjacent pairs
    cost_image_eq: float         # 0.5 per cheap probe (h25f convention), 1.0 at full steps


class AmrStatus(BaseModel):
    """A run's whole state. Points and edges are parallel arrays, as CascadeStatus serves
    its probe cloud: the map draws thousands of them and wants positions and readings in
    the same order, not objects."""
    run_id: str
    status: str
    phase: str = ""
    k: int = 0
    prompts: list[str] = []
    # the ladder this run was started with, and the settings behind it
    schedule: list[int] = []
    base: int = 5
    levels: int = 4
    factor: int = 2
    r_ref: int = 1
    probe_steps: int | None = 4
    generated: int = 0
    phase_done: int = 0
    phase_total: int = 0
    recent_thumbs: list[int] = []
    # evaluated cells: weight vectors, the level that paid for each, its thumbnail index,
    # and its local divergence (max 1-cos over its measured lattice neighbours; null where
    # no neighbour of it was ever evaluated). All four are aligned, and a cell's index in
    # them is the point id the edges below refer to.
    points: list[list[float]] = []
    point_levels: list[int] = []
    point_images: list[int] = []
    point_divs: list[float | None] = []
    # detected boundary edges as point-id pairs, with the level they were found at and the
    # cosine distance across them (> 0.35, the Cascade's crossing threshold)
    edges: list[list[int]] = []
    edge_levels: list[int] = []
    edge_divs: list[float] = []
    levels_stats: list[AmrLevelStat] = []
    cost_image_eq: float = 0.0
    notes: list[str] = []
    error: str | None = None


class AmrPointExport(BaseModel):
    id: int
    weights: list[float]
    level: int
    image: int
    div: float | None = None


class AmrEdgeExport(BaseModel):
    a: int
    b: int
    level: int
    divergence: float


class AmrExport(BaseModel):
    """points.json: the finished lattice as self-describing records, so another tool can
    reuse the survey without re-deriving which parallel array meant what."""
    run_id: str
    status: str
    k: int
    prompts: list[str] = []
    schedule: list[int] = []
    r_ref: int = 1
    seed: int = 42
    steps: int = 0
    probe_steps: int | None = None
    points: list[AmrPointExport] = []
    edges: list[AmrEdgeExport] = []
    levels_stats: list[AmrLevelStat] = []
    cost_image_eq: float = 0.0
    notes: list[str] = []


# ---- Metropolis ridge sampler (services/metro.py) ----
# A chain whose stationary law is the sharpness field itself: pi ~ S(w)^beta, with
# S(w) = mean over m random tangent directions of the divergence at the chord stride.
# Validated in search_problem h25a (replicated): beta = 1 reproduces the S-weighted law
# under single-seed noise (on-boundary 0.673 vs the ideal 0.677, KS 0.083); beta = 2 does
# NOT (KS 0.19-0.28), which is why beta is capped below 1.5 and the UI warns above 1.
class MetroStartRequest(BaseModel):
    """Defaults are the study's ERPT design: 60 chains x 50 steps, beta 1, sigma 0.03, m 2,
    seeded at the crossings of 20 fresh IUR chords.

    Note on `steps` vs `chain_steps`: `steps` is the denoising-step count of a full image,
    as in every other request here, and `chain_steps` is the MCMC chain length L.
    """
    prompts: list[str] = Field(..., min_length=3, max_length=12)
    # target exponent. 1.0 is of record; above BETA_WARN the single-seed KS degrades, and
    # 2.0 is known to fail, so the schema stops at 1.5 rather than at "whatever you type".
    beta: float = Field(1.0, ge=0.5, le=1.5)
    # random-walk step in tangent (weight-space) units. 0.03 is the best LAW cell with m=2;
    # bigger mixes faster and samples the law less tightly (h25a: sigma 0.12 best IAT).
    sigma: float = Field(0.03, gt=0.0, le=0.2)
    # tangent directions one energy averages. 2 is of record; 1 is cheaper and noisier.
    m: int = Field(2, ge=1, le=4)
    chains: int = Field(60, ge=1, le=200)
    chain_steps: int = Field(50, ge=1, le=200)
    # where the chains start: fresh IUR chords drawn by the sampler, or the crossings of a
    # finished Cascade run (its midpoints), which must have the same k
    seed_mode: Literal["cascade", "iur"] = "iur"
    cascade_run_id: str | None = None
    n_seed_chords: int = Field(20, ge=1, le=200)
    seed: int = 42
    steps: int = Field(config.DEFAULT_NUM_INFERENCE_STEPS, ge=1, le=50)
    # probes (states and their energy fan) run on the cheap field, as the Cascade's chord
    # probes do; null probes at full fidelity and doubles the cost per probe
    probe_steps: int | None = Field(4, ge=1, le=50)
    # optional post-pass: re-render every distinct accepted state at `steps`, 1 image each
    render_full: bool = False
    height: int = Field(config.DEFAULT_HEIGHT, ge=64, le=1024)
    width: int = Field(config.DEFAULT_WIDTH, ge=64, le=1024)
    guidance_scale: float = Field(config.DEFAULT_GUIDANCE_SCALE, ge=0.0, le=20.0)


class MetroStartResponse(BaseModel):
    run_id: str
    status: str
    error: str | None = None


class MetroSample(BaseModel):
    """One ACCEPTED state. `image` is the cheap-field probe rendered at it (the gallery
    thumbnail); `full_image` is its full-fidelity re-render when that pass ran."""
    chain: int
    step: int                    # the round it was accepted at
    weights: list[float]
    s: float                     # its energy, the quantity the map and the gallery sort by
    image: int
    full_image: int | None = None


class MetroChainStat(BaseModel):
    """One chain: where it started, where it is, and how it behaved."""
    chain: int
    weights: list[float]
    s: float | None = None
    image: int = -1
    seed_weights: list[float] = []
    seed_s: float | None = None
    seed_image: int = -1
    n_propose: int = 0
    n_accept: int = 0
    # proposals that left the simplex and were rejected without rendering anything
    n_outside: int = 0
    moved: bool = False


class MetroSummary(BaseModel):
    """Quoted over the ACCEPTED states (what the gallery shows), not over the chains' time
    average with repeats -- a statistic of this sample, not an estimate of the law."""
    n_samples: int = 0
    mean_s: float | None = None
    # the mean energy of the states the chains STARTED from, i.e. of their seed crossings
    seed_mean_s: float | None = None
    # share of samples past COS_T = 0.35: the "on-boundary proxy" (h25a reports 0.67 at
    # beta = 1 against 0.26 for uniform sampling)
    on_boundary_frac: float | None = None
    acceptance: float | None = None
    n_propose: int = 0
    n_accept: int = 0
    n_outside: int = 0
    # chains that never left their start: the seed sheets the run kept (h25a: 1-5 %)
    chains_never_moved: int = 0
    n_chains: int = 0
    n_seed_crossings: int = 0
    n_chords: int = 0
    n_probes: int = 0
    n_full: int = 0
    rounds_done: int = 0
    cost_image_eq: float = 0.0


class MetroStatus(BaseModel):
    run_id: str
    status: str
    phase: str = ""
    k: int = 0
    prompts: list[str] = []
    beta: float = 1.0
    sigma: float = 0.03
    m: int = 2
    chains: int = 60
    chain_steps: int = 50
    delta: float = 0.025
    seed_mode: str = "iur"
    cascade_run_id: str | None = None
    n_seed_chords: int = 20
    probe_steps: int | None = 4
    render_full: bool = False
    seed: int = 42
    steps: int = 0
    generated: int = 0
    phase_done: int = 0
    phase_total: int = 0
    round_done: int = 0
    recent_thumbs: list[int] = []
    # the crossings the chains were seeded at, as weight vectors + the divergence ACROSS the
    # bracket that found each (reported, never used as an energy)
    seeds: list[list[float]] = []
    seed_divs: list[float | None] = []
    chains_stats: list[MetroChainStat] = []
    samples: list[MetroSample] = []
    summary: MetroSummary = Field(default_factory=MetroSummary)
    notes: list[str] = []
    error: str | None = None


class MetroExport(BaseModel):
    """samples.json: the accepted states as self-describing records, with the settings they
    were drawn under, so a stimulus set can be reproduced outside this tool."""
    run_id: str
    status: str
    k: int
    prompts: list[str] = []
    beta: float = 1.0
    sigma: float = 0.03
    m: int = 2
    delta: float = 0.025
    chains: int = 60
    chain_steps: int = 50
    seed_mode: str = "iur"
    cascade_run_id: str | None = None
    seed: int = 42
    steps: int = 0
    probe_steps: int | None = None
    cos_t: float = 0.35
    seeds: list[list[float]] = []
    samples: list[MetroSample] = []
    chains_stats: list[MetroChainStat] = []
    summary: MetroSummary = Field(default_factory=MetroSummary)
    notes: list[str] = []


class MetroCascadeRun(BaseModel):
    """One candidate seed source for `seed_mode = "cascade"`."""
    run_id: str
    status: str
    k: int
    prompts: list[str] = []
    n_crossings: int = 0
    certify: bool = False


# ---- exact-JVP probes (services/jvp_probe.py) ----
class JvpProbeRequest(BaseModel):
    alpha: float = Field(..., ge=0.0, le=1.0)
    beta: float = Field(0.0, ge=0.0, le=1.0)
    gamma: float = Field(0.0, ge=0.0, le=1.0)
    seed: int | None = None


class ProbeStartResponse(BaseModel):
    probe_id: str
    status: str


class ProbeStatusResponse(BaseModel):
    probe_id: str
    status: str            # running | done | error
    kind: str
    error: str = ""
    result: dict | None = None


class TokenProbeRequest(BaseModel):
    which: str = Field("a", pattern="^[abcd]$")
    seed: int | None = None


class CascadeProbeRequest(BaseModel):
    cid: int | None = None
    weights: list[float] | None = None
    seed: int | None = None


class GenericProbeRequest(BaseModel):
    prompts: list[str] = Field(..., min_length=2, max_length=12)
    weights: list[float]
    seed: int = 42
    steps: int = Field(config.DEFAULT_NUM_INFERENCE_STEPS, ge=1, le=50)
    guidance_scale: float = Field(config.DEFAULT_GUIDANCE_SCALE, ge=0.0, le=20.0)
    height: int = Field(config.DEFAULT_HEIGHT, ge=64, le=1024)
    width: int = Field(config.DEFAULT_WIDTH, ge=64, le=1024)
    use_slerp: bool = False
