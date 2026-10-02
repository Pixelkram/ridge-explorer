import type { GridStartRequest, GridStartResponse, GridStatusResponse, RefineRequest, RefineResponse, SeedProbeRequest, SeedProbeResponse, SeedProbeStatus, FastScanRequest, FastScanResponse, GenerateSelectedRequest, GenerateSelectedResponse, MFScanRequest, MFScanResponse, RidgeGraph, RidgeGraphRequest, RidgeGraphResponse, HikeStartRequest, HikeStartResponse, HikeStatus, JvpProbeRequest, JvpProbeStartResponse, JvpProbeStatus, TokenProbeRequest, TokenProbeStartResponse, TokenProbeStatus } from './types';

const BASE = '';

// A bare `fetch(...).json()` resolves for a 422 or a 500 too, so a failed request came
// back as an object with none of the fields the caller reads — a store action then
// wrote `undefined` job ids into state instead of taking its error path. Throw instead.
// GET /status is deliberately NOT routed through this: a job the backend no longer
// knows answers 200 with phase 'unknown', which the poller handles explicitly.
// A backend restart takes ~3 minutes (seven FLUX workers load before the app accepts
// requests), and every call in that window fails as a bare "TypeError: Failed to fetch" —
// which tells a tester nothing and loses whatever they were doing. Retry transient
// network failures with backoff, and name the condition if it persists.
async function fetchRetry(url: string, init?: RequestInit, tries = 4): Promise<Response> {
  let last: any;
  for (let i = 0; i < tries; i++) {
    try {
      return await fetch(url, init);
    } catch (e) {
      last = e;                                  // TypeError = connection refused/reset
      if (i < tries - 1) await new Promise((r) => setTimeout(r, 800 * Math.pow(2, i)));
    }
  }
  throw new Error('cannot reach the server — it may be restarting (this takes about '
                  + '3 minutes while the GPUs load). Your run is not lost; it keeps going '
                  + 'on the server. ' + String(last?.message ?? last));
}

async function jsonOrThrow<T>(res: Response, what: string): Promise<T> {
  if (!res.ok) {
    let detail = '';
    try { detail = (await res.text()).slice(0, 200); } catch { /* body already consumed */ }
    throw new Error(`${what} failed: HTTP ${res.status}${detail ? ' — ' + detail : ''}`);
  }
  return res.json() as Promise<T>;
}

export async function startGrid(req: GridStartRequest): Promise<GridStartResponse> {
  const res = await fetchRetry(`${BASE}/api/grid/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return jsonOrThrow<GridStartResponse>(res, 'startGrid');
}

export async function getGridStatus(jobId: string): Promise<GridStatusResponse> {
  const res = await fetchRetry(`${BASE}/api/grid/${jobId}/status`);
  return res.json();
}

export async function refineGrid(jobId: string, req: RefineRequest): Promise<RefineResponse> {
  const res = await fetchRetry(`${BASE}/api/grid/${jobId}/refine`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return jsonOrThrow<RefineResponse>(res, 'refineGrid');
}

export async function startSeedProbe(jobId: string, req: SeedProbeRequest): Promise<SeedProbeResponse> {
  const res = await fetchRetry(`${BASE}/api/grid/${jobId}/seed-probe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return jsonOrThrow<SeedProbeResponse>(res, 'startSeedProbe');
}

export async function getSeedProbeStatus(probeId: string): Promise<SeedProbeStatus> {
  const res = await fetchRetry(`${BASE}/api/grid/probe/${probeId}/status`);
  return jsonOrThrow<SeedProbeStatus>(res, 'getSeedProbeStatus');
}

// ---- JVP / token probes ----------------------------------------------------
// Both start a GPU task and answer with an id to poll (a JVP probe is ~25 s, a token
// probe ~3 min). The two GETs use a bare fetch on purpose, as the other pollers do:
// a poll that retries with backoff just delays the next one, and the store's poller
// already treats a failed read as transient and keeps going.

export async function startJvpProbe(jobId: string, req: JvpProbeRequest): Promise<JvpProbeStartResponse> {
  const res = await fetchRetry(`${BASE}/api/grid/${jobId}/jvp-probe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return jsonOrThrow<JvpProbeStartResponse>(res, 'startJvpProbe');
}

export async function getJvpProbe(probeId: string): Promise<JvpProbeStatus> {
  const res = await fetch(`${BASE}/api/grid/jvp-probe/${probeId}`);
  return jsonOrThrow<JvpProbeStatus>(res, 'getJvpProbe');
}

// The cascade's own launch route. Only the START differs — a cascade point is named by
// crossing id or by barycentric weights, not by (alpha, beta) — so the poll goes back
// through getJvpProbe above, which is the shared status route for both surfaces.
export async function startCascadeJvpProbe(
  runId: string, req: import('./types').CascadeJvpProbeRequest,
): Promise<JvpProbeStartResponse> {
  const res = await fetchRetry(`${BASE}/api/cascade/${runId}/jvp-probe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return jsonOrThrow<JvpProbeStartResponse>(res, 'startCascadeJvpProbe');
}

export async function startTokenProbe(jobId: string, req: TokenProbeRequest): Promise<TokenProbeStartResponse> {
  const res = await fetchRetry(`${BASE}/api/grid/${jobId}/token-probe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return jsonOrThrow<TokenProbeStartResponse>(res, 'startTokenProbe');
}

export async function getTokenProbe(probeId: string): Promise<TokenProbeStatus> {
  const res = await fetch(`${BASE}/api/grid/token-probe/${probeId}`);
  return jsonOrThrow<TokenProbeStatus>(res, 'getTokenProbe');
}

export async function cancelJob(): Promise<void> {
  await fetchRetry(`${BASE}/api/grid/cancel`, { method: 'POST' });
}

export async function startFastScan(req: FastScanRequest): Promise<FastScanResponse> {
  const res = await fetchRetry(`${BASE}/api/grid/fast-scan`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return jsonOrThrow<FastScanResponse>(res, 'startFastScan');
}

export async function generateSelected(jobId: string, req: GenerateSelectedRequest): Promise<GenerateSelectedResponse> {
  const res = await fetchRetry(`${BASE}/api/grid/${jobId}/generate-selected`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return jsonOrThrow<GenerateSelectedResponse>(res, 'generateSelected');
}

export async function startMFScan(req: MFScanRequest): Promise<MFScanResponse> {
  const res = await fetchRetry(`${BASE}/api/grid/mf-scan`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return jsonOrThrow<MFScanResponse>(res, 'startMFScan');
}

export async function buildRidgeGraph(jobId: string, req: RidgeGraphRequest): Promise<RidgeGraphResponse> {
  const res = await fetchRetry(`${BASE}/api/grid/${jobId}/graph`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return jsonOrThrow<RidgeGraphResponse>(res, 'buildRidgeGraph');
}

export async function getRidgeGraph(jobId: string): Promise<RidgeGraph> {
  const res = await fetchRetry(`${BASE}/api/grid/${jobId}/ridge_graph.json`);
  return jsonOrThrow<RidgeGraph>(res, 'getRidgeGraph');
}

export function itineraryUrl(jobId: string, edgeId: number): string {
  return `${BASE}/api/grid/${jobId}/itinerary/${edgeId}.jpg`;
}

export async function startHike(req: HikeStartRequest): Promise<HikeStartResponse> {
  const res = await fetchRetry(`${BASE}/api/grid/hike/start`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(req),
  });
  return jsonOrThrow<HikeStartResponse>(res, 'startHike');
}

export async function getHikeStatus(hikeId: string): Promise<HikeStatus> {
  const res = await fetchRetry(`${BASE}/api/grid/hike/${hikeId}/status`);
  return jsonOrThrow<HikeStatus>(res, 'getHikeStatus');
}

export function hikeFilmstripUrl(hikeId: string, chain: number, hop: number,
                                 w?: number, seed?: number): string {
  // `w` asks the server for a variant scaled to w px per tile. Strips are stored at the
  // hike's generated resolution (up to 512px/tile), which is heavy over mobile data.
  // `seed` is an INDEX into the hike's seed list, not a seed value: only a hike run with
  // seed_count > 1 has anything past 0, and every variant shows the SAME walk, because
  // the ridge was found on the field averaged over all of them.
  const q = [w ? `w=${w}` : '', seed ? `seed=${seed}` : ''].filter(Boolean).join('&');
  return `${BASE}/api/grid/hike/${hikeId}/chain/${chain}/hop/${hop}.jpg${q ? `?${q}` : ''}`;
}

export async function cancelHike(hikeId: string): Promise<any> {
  const r = await fetchRetry(`${BASE}/api/grid/hike/${hikeId}/cancel`, { method: 'POST' });
  return jsonOrThrow<any>(r, 'cancelHike');
}

export async function branchHike(hikeId: string, body: any): Promise<HikeStartResponse> {
  const r = await fetchRetry(`${BASE}/api/grid/hike/${hikeId}/branch`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return jsonOrThrow<HikeStartResponse>(r, 'branchHike');
}

export function hikeMapUrl(hikeId: string, chain: number, hop: number): string {
  return `${BASE}/api/grid/hike/${hikeId}/chain/${chain}/hop/${hop}/map.jpg`;
}

// --- Discovery ---------------------------------------------------------------

export async function discoverDefaults(): Promise<import('./types').DiscoverDefaults> {
  const r = await fetchRetry('/api/discover/defaults');
  if (!r.ok) throw new Error(`defaults failed (${r.status})`);
  return r.json();
}

export async function discoverStart(
  req: import('./types').DiscoverStartRequest,
): Promise<import('./types').DiscoverStartResponse> {
  const r = await fetchRetry('/api/discover/start', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(req),
  });
  if (!r.ok) {
    // 422 carries pydantic's field errors; surface the first rather than a bare status,
    // because the commitment floor (1/k) is easy to trip and needs naming.
    let detail = `start failed (${r.status})`;
    try {
      const j = await r.json();
      if (Array.isArray(j?.detail) && j.detail[0]?.msg) detail = j.detail[0].msg;
      else if (j?.error) detail = j.error;
    } catch { /* keep the status text */ }
    throw new Error(detail);
  }
  return r.json();
}

// Status is deliberately not routed through fetchRetry: an unknown run answers 200 with
// status 'unknown', which the poller handles, and retrying a poll just delays the next one.
export async function discoverStatus(runId: string): Promise<import('./types').DiscoverStatus> {
  const r = await fetch(`/api/discover/${runId}/status`);
  if (!r.ok) throw new Error(`status failed (${r.status})`);
  return r.json();
}

export async function discoverCancel(runId: string): Promise<void> {
  await fetch(`/api/discover/${runId}/cancel`, { method: 'POST' });
}

export interface SurpriseSampleResponse {
  job_id: string;
  surprise: number;
  cells: [number, number][];
  probabilities: Record<string, number>;
}

export async function surpriseSample(
  jobId: string, surprise: number, n = 12, seed?: number,
): Promise<SurpriseSampleResponse> {
  const res = await fetchRetry(`${BASE}/api/grid/${jobId}/surprise-sample`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ surprise, n, seed }),
  });
  return jsonOrThrow<SurpriseSampleResponse>(res, 'surpriseSample');
}

// ---- Cascade ----

export async function cascadeStart(
  req: import('./types').CascadeStartRequest,
): Promise<import('./types').CascadeStartResponse> {
  const r = await fetchRetry(`${BASE}/api/cascade/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!r.ok) {
    let detail = `cascadeStart failed: HTTP ${r.status}`;
    try {
      const j = await r.json();
      if (Array.isArray(j?.detail) && j.detail[0]?.msg) detail = j.detail[0].msg;
      else if (j?.error) detail = j.error;
    } catch { /* keep the HTTP line */ }
    throw new Error(detail);
  }
  return r.json();
}

export async function cascadeStatus(
  runId: string,
): Promise<import('./types').CascadeStatus> {
  // bare fetch on purpose: retrying a poll just delays the next one
  const r = await fetch(`${BASE}/api/cascade/${runId}/status`);
  return r.json();
}

export async function cascadeCancel(runId: string): Promise<void> {
  await fetch(`${BASE}/api/cascade/${runId}/cancel`, { method: 'POST' });
}

export function cascadeImageUrl(runId: string, index: number): string {
  return `${BASE}/api/cascade/${runId}/${index}.jpg`;
}

export async function cascadeWalkStart(
  runId: string, req: import('./types').WalkStartRequest,
): Promise<import('./types').WalkStatus> {
  const r = await fetchRetry(`${BASE}/api/cascade/${runId}/walk`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return jsonOrThrow(r, 'cascadeWalkStart');
}

export async function cascadeWalkStatus(
  runId: string, walkId: string,
): Promise<import('./types').WalkStatus> {
  const r = await fetch(`${BASE}/api/cascade/${runId}/walk/${walkId}`);
  return r.json();
}

// Re-measure a finished walk's stations on three seeds it never used; poll the walk for `cert`.
export async function cascadeWalkCertify(
  runId: string, walkId: string,
): Promise<import('./types').WalkStatus> {
  const r = await fetch(`${BASE}/api/cascade/${runId}/walk/${walkId}/certify`, { method: 'POST' });
  return jsonOrThrow(r, 'cascadeWalkCertify');
}

export async function cascadeWalkCancel(runId: string, walkId: string): Promise<void> {
  await fetch(`${BASE}/api/cascade/${runId}/walk/${walkId}/cancel`, { method: 'POST' });
}

export async function cascadePointInfo(
  runId: string, index: number,
): Promise<import('./types').CascadePointInfo> {
  const r = await fetch(`${BASE}/api/cascade/${runId}/point/${index}`);
  return r.json();
}

// ---- AMR (adaptive refinement) ----
// Same shape as the cascade's calls: one POST to start, a bare-fetch status poll, a cancel,
// an image by index, plus a points.json export. The start can legitimately fail with a 400
// (k >= 5, or a ladder past the cell ceiling) whose body carries the projected cost, so that
// detail is lifted out rather than reported as a bare status.
export async function amrStart(
  req: import('./types').AmrStartRequest,
): Promise<import('./types').AmrStartResponse> {
  const r = await fetchRetry(`${BASE}/api/amr/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!r.ok) {
    let detail = `amrStart failed: HTTP ${r.status}`;
    try {
      const j = await r.json();
      if (typeof j?.detail === 'string') detail = j.detail;
      else if (Array.isArray(j?.detail) && j.detail[0]?.msg) detail = j.detail[0].msg;
      else if (j?.error) detail = j.error;
    } catch { /* keep the HTTP line */ }
    throw new Error(detail);
  }
  return r.json();
}

export async function amrStatus(runId: string): Promise<import('./types').AmrStatus> {
  // bare fetch on purpose: retrying a poll just delays the next one
  const r = await fetch(`${BASE}/api/amr/${runId}/status`);
  return r.json();
}

export async function amrCancel(runId: string): Promise<void> {
  await fetch(`${BASE}/api/amr/${runId}/cancel`, { method: 'POST' });
}

export function amrImageUrl(runId: string, index: number): string {
  return `${BASE}/api/amr/${runId}/image/${index}`;
}

export function amrPointsUrl(runId: string): string {
  return `${BASE}/api/amr/${runId}/points.json`;
}

// ---- Metropolis ridge sampler ----
// The AMR shape again. `start` can fail with a 4xx that carries the reason: a projected cost
// past the ceiling (400), or a cascade seed source that cannot be honoured (404 unknown,
// 409 still running, 400 wrong k / no crossings). Those details are the useful answer, so
// they are lifted out rather than reported as a bare status.
export async function metroStart(
  req: import('./types').MetroStartRequest,
): Promise<import('./types').MetroStartResponse> {
  const r = await fetchRetry(`${BASE}/api/metro/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!r.ok) {
    let detail = `metroStart failed: HTTP ${r.status}`;
    try {
      const j = await r.json();
      if (typeof j?.detail === 'string') detail = j.detail;
      else if (Array.isArray(j?.detail) && j.detail[0]?.msg) detail = j.detail[0].msg;
      else if (j?.error) detail = j.error;
    } catch { /* keep the HTTP line */ }
    throw new Error(detail);
  }
  return r.json();
}

export async function metroStatus(runId: string): Promise<import('./types').MetroStatus> {
  // bare fetch on purpose: retrying a poll just delays the next one
  const r = await fetch(`${BASE}/api/metro/${runId}/status`);
  return r.json();
}

export async function metroCancel(runId: string): Promise<void> {
  await fetch(`${BASE}/api/metro/${runId}/cancel`, { method: 'POST' });
}

// Finished Cascade runs this sampler can seed from, optionally of one arity only. Geometry
// only -- it reads the runs already in memory and generates nothing.
export async function metroCascadeRuns(
  k?: number,
): Promise<import('./types').MetroCascadeRun[]> {
  const r = await fetchRetry(`${BASE}/api/metro/cascade-runs${k ? `?k=${k}` : ''}`);
  return jsonOrThrow(r, 'metroCascadeRuns');
}

export function metroImageUrl(runId: string, index: number): string {
  return `${BASE}/api/metro/${runId}/image/${index}`;
}

export function metroSamplesUrl(runId: string): string {
  return `${BASE}/api/metro/${runId}/samples.json`;
}

// Local boundary density over the run's own chords -- geometry only, no images, so it is
// cheap enough to ask for on a toggle. 404 while the survey has laid down no chords yet.
export async function cascadeLocalSv(
  runId: string, mode: 'all' | 'certified' = 'all',
): Promise<import('./types').LocalSvMap> {
  const r = await fetchRetry(`${BASE}/api/cascade/${runId}/local-sv?mode=${mode}`);
  return jsonOrThrow(r, 'cascadeLocalSv');
}
