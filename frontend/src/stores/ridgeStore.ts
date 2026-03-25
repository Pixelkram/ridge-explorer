import { create } from 'zustand';
import { startGrid, getGridStatus, refineGrid } from '../api/client';
import type { CellStatus } from '../api/types';

type ViewMode = 'images' | 'heatmap' | 'overlay' | 'clusters';
type Phase = 'idle' | 'generating' | 'analyzing' | 'complete';

interface RidgeState {
  promptA: string;
  promptB: string;
  promptC: string;
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
  generate: () => Promise<void>;
  submitRefine: () => Promise<void>;
  stopPolling: () => void;
}

function startPolling(set: any, get: any) {
  // Immediate first poll
  const doPoll = async () => {
    const { jobId } = get();
    if (!jobId) return;
    try {
      const status = await getGridStatus(jobId);
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
      };
      if (status.phase === 'generating') updates.phase = 'generating';
      if (status.phase === 'analyzing') updates.phase = 'analyzing';
      if (status.phase === 'complete') {
        updates.phase = 'complete';
        get().stopPolling();
      }
      set(updates);
    } catch (err) {
      console.error('[Ridge] Poll error:', err);
    }
  };

  // Poll immediately, then every 1.5s
  doPoll();
  const interval = setInterval(doPoll, 1500);
  set({ pollInterval: interval });
}

export const useRidgeStore = create<RidgeState>((set, get) => ({
  promptA: 'a photo of a giraffe',
  promptB: 'a photo of an airplane',
  promptC: 'an oil painting of a unicorn',
  gridSize: 12,
  seed: 42,
  jobId: null,
  phase: 'idle',
  cellsGenerated: 0,
  cellsTotal: 0,
  cells: [],
  activeView: 'images',
  tau: 1.5,
  multiplier: 3,
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

  setPromptA: (v) => set({ promptA: v }),
  setPromptB: (v) => set({ promptB: v }),
  setPromptC: (v) => set({ promptC: v }),
  setGridSize: (v) => set({ gridSize: v }),
  setSeed: (v) => set({ seed: v }),
  setActiveView: (v) => set({ activeView: v }),
  setTau: (v) => set({ tau: v }),
  setMultiplier: (v) => set({ multiplier: v }),
  setSteps: (v) => set({ steps: v }),
  setResolution: (v) => set({ resolution: v }),
  setSeedCount: (v) => set({ seedCount: v }),
  setActiveSeedIdx: (v) => set({ activeSeedIdx: v }),

  stopPolling: () => {
    const { pollInterval } = get();
    if (pollInterval) clearInterval(pollInterval);
    set({ pollInterval: null });
  },

  generate: async () => {
    const { promptA, promptB, promptC, gridSize, seed, steps, resolution, seedCount, stopPolling } = get();
    stopPolling();
    set({
      phase: 'generating', cellsGenerated: 0, cellsTotal: 0, cells: [],
      heatmapUrl: null, overlayUrl: null, clusterUrl: null, imageGridUrl: null,
      jobId: null, currentGridSize: gridSize, activeView: 'images', shouldCenter: true,
    });
    try {
      const res = await startGrid({
        prompt_a: promptA, prompt_b: promptB, prompt_c: promptC,
        grid_size: gridSize, seed, seed_count: seedCount,
        height: resolution, width: resolution, steps,
      });
      set({ jobId: res.job_id, cellsTotal: res.total_cells });
      startPolling(set, get);
    } catch (err) {
      console.error('[Ridge] Generate error:', err);
      set({ phase: 'idle' });
    }
  },

  submitRefine: async () => {
    const { jobId, tau, multiplier, stopPolling, currentGridSize } = get();
    if (!jobId) return;
    stopPolling();

    const newGs = currentGridSize * multiplier;
    // Clear everything and switch to images view
    set({
      phase: 'generating', activeView: 'images',
      heatmapUrl: null, overlayUrl: null, clusterUrl: null, imageGridUrl: null,
      cells: [], cellsGenerated: 0, currentGridSize: newGs, shouldCenter: false,
    });

    try {
      console.log('[Ridge] Refining...', { tau, multiplier, newGs });
      const res = await refineGrid(jobId, { tau, multiplier });
      console.log('[Ridge] Refine response:', res);
      if (res.status === 'error' || res.status === 'no_cells') {
        // Reload current state
        const status = await getGridStatus(jobId);
        set({
          phase: 'complete', cells: status.cells,
          currentGridSize: status.grid_size,
          heatmapUrl: status.heatmap_url, overlayUrl: status.overlay_url,
          clusterUrl: status.cluster_url,
        });
        return;
      }
      startPolling(set, get);
    } catch (err) {
      console.error('[Ridge] Refine error:', err);
      set({ phase: 'complete' });
    }
  },
}));
