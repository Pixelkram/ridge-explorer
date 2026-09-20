import { create } from 'zustand';
import { startGrid, getGridStatus, refineGrid, startFastScan, generateSelected, cancelJob, startMFScan } from '../api/client';
import type { CellStatus } from '../api/types';

type ViewMode = 'images' | 'heatmap' | 'overlay' | 'clusters';
type Phase = 'idle' | 'scanning' | 'scan_complete' | 'generating' | 'analyzing' | 'complete'
  | 'mf_scanning' | 'mf_jacobian_done' | 'mf_finalizing' | 'error';

interface RidgeState {
  promptA: string;
  promptB: string;
  promptC: string;
  promptD: string;
  dimensions: number;
  gridSize: number;
  seed: number;
  jobId: string | null;
  phase: Phase;
  cellsGenerated: number;
  cellsTotal: number;
  cells: CellStatus[];
  activeView: ViewMode;
  tau: number;
  multiplier: number;
  steps: number;
  resolution: number;
  seedCount: number;
  seeds: number[];
  seedCells: Record<string, CellStatus[]> | null;
  activeSeedIdx: number;  // which seed grid to show (0-based)
  heatmapUrl: string | null;
  overlayUrl: string | null;
  clusterUrl: string | null;
  imageGridUrl: string | null;
  pollInterval: ReturnType<typeof setInterval> | null;
  currentGridSize: number;
  shouldCenter: boolean;
  manualSelection: Set<string>;
  ridgeMeshUrl: string | null;
  sliceIndex: number;  // z-slice index for 3D image browsing
  error: string | null;

  setPromptA: (v: string) => void;
  setPromptB: (v: string) => void;
  setPromptC: (v: string) => void;
  setGridSize: (v: number) => void;
  setSeed: (v: number) => void;
  setActiveView: (v: ViewMode) => void;
  setTau: (v: number) => void;
  setMultiplier: (v: number) => void;
  setSteps: (v: number) => void;
  setResolution: (v: number) => void;
  setSeedCount: (v: number) => void;
  setActiveSeedIdx: (v: number) => void;
  toggleManualCell: (row: number, col: number) => void;
  setManualSelection: (keys: string[]) => void;
  clearManualSelection: () => void;
  setPromptD: (v: string) => void;
  setDimensions: (v: number) => void;
  setSliceIndex: (v: number) => void;
  generate: () => Promise<void>;
  fastScan: () => Promise<void>;
  mfScan: () => Promise<void>;
  generateSelectedImages: () => Promise<void>;
  submitRefine: () => Promise<void>;
  stopPolling: () => void;
  cancel: () => void;
}

// Bumped by every job launch and by cancel(). A launch that finds the counter has
// moved while it was awaiting POST /start has been superseded by a later click and
// must not adopt its job id or start polling for it.
let launchSeq = 0;

function startPolling(set: any, get: any) {
  // Immediate first poll
  const doPoll = async () => {
    const { jobId } = get();
    if (!jobId) return;
    try {
      const status = await getGridStatus(jobId);
      // Cancelled, or a different job was started, while awaiting the response:
      // writing this status would overwrite the new job's cells with the old one's.
      if (get().jobId !== jobId) return;
      // A job the backend no longer knows returns HTTP 200 with phase 'unknown' /
      // status 'not_found' and an empty cell list. Without this branch the store
      // overwrote a finished grid with zeros and then polled that dead id every
      // 1.5 s for the rest of the session. app.state.jobs is in-memory, so any
      // restart or /cancel puts every open tab into exactly this state.
      if (status.phase === 'unknown' || (status as any).status === 'not_found') {
        console.warn('[Ridge] job', jobId, 'is unknown to the backend; stopping poll');
        get().stopPolling();
        set({ phase: 'error', error: 'This job is no longer on the server (restart or cancel). Generate again.' });
        return;
      }
      console.log('[Ridge] Poll:', status.phase, status.cells_generated, '/', status.cells_total,
                  'cells:', status.cells.length, 'gs:', status.grid_size);
      const updates: any = {
        cellsGenerated: status.cells_generated,
        cellsTotal: status.cells_total,
        cells: status.cells,
        currentGridSize: status.grid_size,
        seeds: status.seeds || [],
        seedCells: status.seed_cells || null,
        heatmapUrl: status.heatmap_url,
        overlayUrl: status.overlay_url,
        clusterUrl: status.cluster_url,
        imageGridUrl: status.image_grid_url,
        ridgeMeshUrl: (status as any).ridge_mesh_url || null,
        dimensions: status.dimensions || 2,
      };
      if (status.phase === 'scanning') updates.phase = 'scanning';
      if (status.phase === 'mf_scanning') updates.phase = 'mf_scanning';
      if (status.phase === 'mf_jacobian_done') updates.phase = 'mf_jacobian_done';
      if (status.phase === 'mf_finalizing') updates.phase = 'mf_finalizing';
      if (status.phase === 'generating') updates.phase = 'generating';
      if (status.phase === 'analyzing') updates.phase = 'analyzing';
      if (status.phase === 'scan_complete') {
        updates.phase = 'scan_complete';
        get().stopPolling();
      }
      if (status.phase === 'complete') {
        updates.phase = 'complete';
        get().stopPolling();
      }
      // A GPU worker failed this job's tasks. Its cells can never arrive, so
      // cells_generated will never reach cells_total — without this the poller ran for
      // the rest of the session against a job that was already dead.
      if (status.phase === 'error') {
        updates.phase = 'error';
        updates.error = status.error || 'The GPU worker failed this job.';
        get().stopPolling();
      }
      set(updates);
    } catch (err) {
      console.error('[Ridge] Poll error:', err);
    }
  };

  // Poll immediately, then every 1.5s
  doPoll();
  // Guard: if cancelled during the first doPoll, don't start interval
  if (!get().jobId) return;
  // Only `pollInterval` is clearable, so an interval created while one is already
  // running would leak and poll for the rest of the session.
  get().stopPolling();
  const interval = setInterval(doPoll, 1500);
  set({ pollInterval: interval });
}

// submitRefine blanks the grid before it POSTs, so every way the refine can fail
// has to put the server's copy back — otherwise the user is left with an empty
// grid at the new size that still looks 'complete'. The backend job is untouched
// by a failed refine, so a plain status re-fetch is a full recovery.
async function restoreGridFromServer(set: any, jobId: string, error: string | null) {
  try {
    const status = await getGridStatus(jobId);
    set({
      phase: 'complete', cells: status.cells,
      cellsGenerated: status.cells_generated, cellsTotal: status.cells_total,
      currentGridSize: status.grid_size,
      heatmapUrl: status.heatmap_url, overlayUrl: status.overlay_url,
      clusterUrl: status.cluster_url, imageGridUrl: status.image_grid_url,
      ridgeMeshUrl: (status as any).ridge_mesh_url || null,
      error,
    });
  } catch (err) {
    console.error('[Ridge] Refine recovery failed:', err);
    set({ phase: 'error', error: (error ? error + ' ' : '') + 'Could not reload the grid from the server.' });
  }
}

export const useRidgeStore = create<RidgeState>((set, get) => ({
  promptA: 'a photo of a giraffe',
  promptB: 'a photo of an airplane',
  promptC: 'an oil painting of a unicorn',
  promptD: 'a photo of a sports car',
  dimensions: 2,
  gridSize: 12,
  seed: 42,
  jobId: null,
  phase: 'idle',
  cellsGenerated: 0,
  cellsTotal: 0,
  cells: [],
  activeView: 'images',
  tau: 1.5,
  multiplier: 2,
  steps: 4,
  resolution: 256,
  seedCount: 1,
  seeds: [],
  seedCells: null,
  activeSeedIdx: 0,
  heatmapUrl: null,
  overlayUrl: null,
  clusterUrl: null,
  imageGridUrl: null,
  pollInterval: null,
  currentGridSize: 12,
  shouldCenter: true,
  manualSelection: new Set<string>(),
  ridgeMeshUrl: null,
  sliceIndex: 0,
  error: null,

  setPromptA: (v) => set({ promptA: v }),
  setPromptB: (v) => set({ promptB: v }),
  setPromptC: (v) => set({ promptC: v }),
  setPromptD: (v) => set({ promptD: v }),
  setDimensions: (v) => set({ dimensions: v }),
  setSliceIndex: (v) => set({ sliceIndex: v }),
  setGridSize: (v) => set({ gridSize: v }),
  setSeed: (v) => set({ seed: v }),
  setActiveView: (v) => set({ activeView: v }),
  setTau: (v) => set({ tau: v }),
  setMultiplier: (v) => set({ multiplier: v }),
  setSteps: (v) => set({ steps: v }),
  setResolution: (v) => set({ resolution: v }),
  setSeedCount: (v) => set({ seedCount: v }),
  setActiveSeedIdx: (v) => set({ activeSeedIdx: v }),
  toggleManualCell: (row, col) => set(state => {
    const key = `${row},${col}`;
    const next = new Set(state.manualSelection);
    if (next.has(key)) next.delete(key); else next.add(key);
    return { manualSelection: next };
  }),
  clearManualSelection: () => set({ manualSelection: new Set<string>() }),
  // surprise slider: replace the manual selection with a sampled cell set
  setManualSelection: (keys) => set({ manualSelection: new Set(keys) }),

  stopPolling: () => {
    const { pollInterval } = get();
    if (pollInterval) clearInterval(pollInterval);
    set({ pollInterval: null });
  },

  cancel: () => {
    // Re-enables the buttons synchronously, so disown any launch still awaiting
    // its POST /start — otherwise it would come back and start a second poller.
    launchSeq++;
    get().stopPolling();
    set({ phase: 'idle', jobId: null, cellsGenerated: 0, cellsTotal: 0, error: null });
    // Tell backend to drain pending GPU tasks so next job starts immediately
    cancelJob().catch(() => {});
  },

  generate: async () => {
    const { promptA, promptB, promptC, promptD, dimensions, gridSize, seed, steps, resolution, seedCount, stopPolling, phase } = get();
    // The buttons are disabled off `phase`, but `phase` used to be set only after
    // `await cancelJob()` returned, so a second click inside that window started a
    // second job and a second poller. Claim the phase before any await.
    if (phase !== 'idle' && phase !== 'complete' && phase !== 'scan_complete' && phase !== 'error') return;
    const launch = ++launchSeq;
    set({
      phase: 'generating', cellsGenerated: 0, cellsTotal: 0, cells: [],
      heatmapUrl: null, overlayUrl: null, clusterUrl: null, imageGridUrl: null,
      jobId: null, currentGridSize: gridSize, activeView: 'images', shouldCenter: true,
      manualSelection: new Set<string>(), error: null,
    });
    stopPolling();
    await cancelJob().catch(() => {});
    try {
      const res = await startGrid({
        prompt_a: promptA, prompt_b: promptB, prompt_c: promptC,
        prompt_d: dimensions === 3 ? promptD : undefined,
        dimensions,
        grid_size: gridSize, seed, seed_count: seedCount,
        height: resolution, width: resolution, steps,
      });
      if (launchSeq !== launch) return;  // superseded or cancelled while /start was in flight
      set({ jobId: res.job_id, cellsTotal: res.total_cells });
      startPolling(set, get);
    } catch (err) {
      console.error('[Ridge] Generate error:', err);
      if (launchSeq !== launch) return;
      set({ phase: 'idle' });
    }
  },

  fastScan: async () => {
    const { promptA, promptB, promptC, promptD, dimensions, gridSize, seed, stopPolling, phase } = get();
    if (phase !== 'idle' && phase !== 'complete' && phase !== 'scan_complete' && phase !== 'error') return;
    const launch = ++launchSeq;
    set({
      phase: 'scanning', cellsGenerated: 0, cellsTotal: 0, cells: [],
      heatmapUrl: null, overlayUrl: null, clusterUrl: null, imageGridUrl: null,
      ridgeMeshUrl: null, jobId: null, currentGridSize: gridSize, activeView: 'images',
      shouldCenter: true, manualSelection: new Set<string>(), error: null,
    });
    stopPolling();
    await cancelJob().catch(() => {});
    try {
      const res = await startFastScan({
        prompt_a: promptA, prompt_b: promptB, prompt_c: promptC,
        prompt_d: dimensions === 3 ? promptD : undefined,
        dimensions,
        grid_size: gridSize, seed,
      });
      if (launchSeq !== launch) return;
      set({ jobId: res.job_id, cellsTotal: res.total_cells });
      startPolling(set, get);
    } catch (err) {
      console.error('[Ridge] Fast scan error:', err);
      if (launchSeq !== launch) return;
      set({ phase: 'idle' });
    }
  },

  mfScan: async () => {
    const { promptA, promptB, promptC, gridSize, seed, steps, resolution, stopPolling, phase } = get();
    if (phase !== 'idle' && phase !== 'complete' && phase !== 'scan_complete' && phase !== 'error') return;
    const launch = ++launchSeq;
    set({
      phase: 'mf_scanning', cellsGenerated: 0, cellsTotal: 0, cells: [],
      heatmapUrl: null, overlayUrl: null, clusterUrl: null, imageGridUrl: null,
      ridgeMeshUrl: null, jobId: null, currentGridSize: gridSize, activeView: 'heatmap',
      shouldCenter: true, manualSelection: new Set<string>(), error: null,
    });
    stopPolling();
    await cancelJob().catch(() => {});
    try {
      const res = await startMFScan({
        prompt_a: promptA, prompt_b: promptB, prompt_c: promptC,
        grid_size: gridSize, seed, budget: 80, tau_mf: 1.3,
        height: resolution, width: resolution, steps,
        guidance_scale: 4.0,
      });
      if (launchSeq !== launch) return;
      set({ jobId: res.job_id, cellsTotal: res.total_cells });
      startPolling(set, get);
    } catch (err) {
      console.error('[Ridge] MF scan error:', err);
      if (launchSeq !== launch) return;
      set({ phase: 'idle' });
    }
  },

  generateSelectedImages: async () => {
    const { jobId, tau, steps, resolution, stopPolling } = get();
    if (!jobId) return;
    stopPolling();
    set({ phase: 'generating', cellsGenerated: 0 });
    try {
      const res = await generateSelected(jobId, {
        tau, height: resolution, width: resolution, steps,
      });
      if (res.status === 'no_cells') {
        set({ phase: 'scan_complete' });
        return;
      }
      set({ cellsTotal: res.total_cells });
      startPolling(set, get);
    } catch (err) {
      console.error('[Ridge] Generate selected error:', err);
      set({ phase: 'scan_complete' });
    }
  },

  submitRefine: async () => {
    const { jobId, tau, multiplier, stopPolling, currentGridSize, manualSelection } = get();
    if (!jobId) return;
    stopPolling();

    // Convert manual selection set to position tuples
    const extraPositions: [number, number][] = [];
    manualSelection.forEach(key => {
      const [r, c] = key.split(',').map(Number);
      extraPositions.push([r, c]);
    });

    const newGs = currentGridSize * multiplier;
    set({
      phase: 'generating', activeView: 'images',
      heatmapUrl: null, overlayUrl: null, clusterUrl: null, imageGridUrl: null,
      cells: [], cellsGenerated: 0, currentGridSize: newGs, shouldCenter: false,
      manualSelection: new Set<string>(), error: null,
    });

    try {
      console.log('[Ridge] Refining...', { tau, multiplier, newGs, extraPositions: extraPositions.length });
      const res = await refineGrid(jobId, { tau, multiplier, extra_positions: extraPositions });
      console.log('[Ridge] Refine response:', res);
      if (res.status === 'error' || res.status === 'no_cells') {
        await restoreGridFromServer(set, jobId,
          res.status === 'no_cells' ? 'No cells above tau to refine.' : 'Refine failed on the server.');
        return;
      }
      startPolling(set, get);
    } catch (err) {
      console.error('[Ridge] Refine error:', err);
      // Transport/parse failure: the request never took effect, so restore rather
      // than leaving the blanked grid sitting at 'complete' with no way back.
      await restoreGridFromServer(set, jobId, 'Refine request failed: ' + String(err));
    }
  },
}));
