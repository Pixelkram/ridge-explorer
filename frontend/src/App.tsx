import { useMemo, useRef, useState, useCallback, useEffect, lazy, Suspense } from 'react';
import { useRidgeStore } from './stores/ridgeStore';
import { startSeedProbe, getSeedProbeStatus } from './api/client';

const Plot = lazy(() => import('react-plotly.js'));

const RESOLUTION_OPTIONS = [128, 256, 384, 512];
const STEPS_OPTIONS = [2, 4, 8, 12, 20, 50];

function PromptInput() {
  const { promptA, promptB, promptC, promptD, dimensions, gridSize, seed, steps, resolution, seedCount, phase,
          setPromptA, setPromptB, setPromptC, setPromptD, setDimensions, setGridSize, setSeed, setSteps, setResolution, setSeedCount, generate, fastScan, cancel } = useRidgeStore();
  const busy = phase !== 'idle' && phase !== 'complete' && phase !== 'scan_complete';

  const inputStyle = { width: '100%', padding: 6, background: '#0f3460', border: '1px solid #333',
                       color: '#fff', borderRadius: 4, fontSize: 12 };
  const selectStyle = { padding: 4, background: '#0f3460', border: '1px solid #333',
                        color: '#fff', borderRadius: 4, fontSize: 11 };

  return (
    <div style={{ display: 'flex', gap: 10, padding: 10, background: '#16213e',
                  alignItems: 'flex-end', flexWrap: 'wrap' }}>
      <div>
        <label style={{ fontSize: 10, color: '#888' }}>Mode</label>
        <select value={dimensions} onChange={e => setDimensions(+e.target.value)} style={selectStyle}>
          <option value={2}>2D</option>
          <option value={3}>3D</option>
        </select>
      </div>
      <div style={{ flex: 1, minWidth: 130 }}>
        <label style={{ fontSize: 10, color: '#888' }}>Prompt A (origin)</label>
        <input value={promptA} onChange={e => setPromptA(e.target.value)} style={inputStyle} />
      </div>
      <div style={{ flex: 1, minWidth: 130 }}>
        <label style={{ fontSize: 10, color: '#888' }}>Prompt B (x-axis)</label>
        <input value={promptB} onChange={e => setPromptB(e.target.value)} style={inputStyle} />
      </div>
      <div style={{ flex: 1, minWidth: 130 }}>
        <label style={{ fontSize: 10, color: '#888' }}>Prompt C (y-axis)</label>
        <input value={promptC} onChange={e => setPromptC(e.target.value)} style={inputStyle} />
      </div>
      {dimensions === 3 && (
        <div style={{ flex: 1, minWidth: 130 }}>
          <label style={{ fontSize: 10, color: '#888' }}>Prompt D (z-axis)</label>
          <input value={promptD} onChange={e => setPromptD(e.target.value)} style={inputStyle} />
        </div>
      )}
      <div>
        <label style={{ fontSize: 10, color: '#888' }}>Grid {gridSize}x{gridSize}</label>
        <input type="range" min={3} max={100} step={1} value={gridSize}
               onChange={e => setGridSize(+e.target.value)} style={{ display: 'block', width: 70 }} />
      </div>
      <div>
        <label style={{ fontSize: 10, color: '#888' }}>Resolution</label>
        <select value={resolution} onChange={e => setResolution(+e.target.value)} style={selectStyle}>
          {RESOLUTION_OPTIONS.map(r => <option key={r} value={r}>{r}px</option>)}
        </select>
      </div>
      <div>
        <label style={{ fontSize: 10, color: '#888' }}>Steps</label>
        <select value={steps} onChange={e => setSteps(+e.target.value)} style={selectStyle}>
          {STEPS_OPTIONS.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
      </div>
      <div>
        <label style={{ fontSize: 10, color: '#888' }}>Seed</label>
        <input type="number" value={seed} onChange={e => setSeed(+e.target.value)}
               style={{ width: 55, padding: 4, background: '#0f3460', border: '1px solid #333',
                        color: '#fff', borderRadius: 4, fontSize: 11 }} />
      </div>
      <div>
        <label style={{ fontSize: 10, color: '#888' }}>Seeds {seedCount > 1 ? `(${seed}-${seed+seedCount-1})` : ''}</label>
        <select value={seedCount} onChange={e => setSeedCount(+e.target.value)} style={selectStyle}>
          {[1, 2, 3, 4, 5].map(n => <option key={n} value={n}>{n}{n > 1 ? ' seeds' : ' seed'}</option>)}
        </select>
      </div>
      <button onClick={generate} disabled={busy}
              style={{ padding: '5px 18px', background: busy ? '#555' : '#e94560',
                       color: '#fff', border: 'none', borderRadius: 4, fontWeight: 'bold',
                       cursor: busy ? 'not-allowed' : 'pointer', fontSize: 12 }}>
        {busy ? 'Working...' : 'Explore'}
      </button>
      {busy && (
        <button onClick={cancel}
                style={{ padding: '5px 14px', background: '#333', color: '#e94560',
                         border: '1px solid #e94560', borderRadius: 4, fontWeight: 'bold',
                         cursor: 'pointer', fontSize: 12 }}>
          Cancel
        </button>
      )}
      <button onClick={fastScan} disabled={busy}
              style={{ padding: '5px 14px', background: busy ? '#555' : '#0f3460',
                       color: '#fff', border: '1px solid #4ecca3', borderRadius: 4, fontWeight: 'bold',
                       cursor: busy ? 'not-allowed' : 'pointer', fontSize: 12 }}>
        {busy ? '...' : 'Fast Scan'}
      </button>
    </div>
  );
}

function ProgressBar() {
  const { phase, cellsGenerated, cellsTotal, currentGridSize } = useRidgeStore();
  if (phase === 'idle') return null;
  const pct = cellsTotal > 0 ? (cellsGenerated / cellsTotal * 100) : 0;
  const label = phase === 'scanning' ? `Fast scan: ${cellsGenerated}/${cellsTotal} latents`
    : phase === 'generating' ? `Generating: ${cellsGenerated}/${cellsTotal}`
    : phase === 'analyzing' ? 'Computing ridges...'
    : phase === 'scan_complete' ? `Fast scan complete (${currentGridSize}×${currentGridSize})`
    : `${currentGridSize}×${currentGridSize} grid`;

  return (
    <div style={{ padding: '4px 12px', background: '#1a1a2e', display: 'flex', alignItems: 'center', gap: 8 }}>
      <span style={{ fontSize: 11, color: '#888', minWidth: 160 }}>{label}</span>
      {phase !== 'complete' && phase !== 'scan_complete' && (
        <div style={{ flex: 1, background: '#333', borderRadius: 3, height: 4 }}>
          <div style={{ width: `${pct}%`, background: phase === 'scanning' ? '#4ecca3' : '#e94560', borderRadius: 3, height: '100%', transition: 'width 0.3s' }} />
        </div>
      )}
    </div>
  );
}

function ScanCompletePanel() {
  const { phase, tau, setTau, cells, steps, resolution, setSteps, setResolution,
          generateSelectedImages } = useRidgeStore();
  if (phase !== 'scan_complete') return null;

  const median = computeRealMedian(cells);
  const threshold = median * tau;
  const aboveCount = cells.filter(c => c.sensitivity != null && c.sensitivity! >= threshold).length;
  const totalCells = cells.length;
  const selectStyle = { padding: 4, background: '#0f3460', border: '1px solid #333',
                        color: '#fff', borderRadius: 4, fontSize: 11 };

  return (
    <div style={{ display: 'flex', gap: 12, padding: '6px 12px', background: '#1a1a2e',
                  borderTop: '1px solid #333', alignItems: 'center', flexWrap: 'wrap' }}>
      <span style={{ fontSize: 11, color: '#4ecca3', fontWeight: 'bold' }}>Jacobian ridge map</span>
      <div>
        <label style={{ fontSize: 10, color: '#888' }}>τ={tau.toFixed(2)}</label>
        <input type="range" min={0} max={4} step={0.05} value={tau}
               onChange={e => setTau(+e.target.value)} style={{ display: 'block', width: 140 }} />
      </div>
      <span style={{ fontSize: 11, color: '#aaa' }}>
        <span style={{ color: '#4ecca3' }}>{aboveCount}</span>/{totalCells} cells above threshold
      </span>
      <div>
        <label style={{ fontSize: 10, color: '#888' }}>Resolution</label>
        <select value={resolution} onChange={e => setResolution(+e.target.value)} style={selectStyle}>
          {[128, 256, 384, 512].map(r => <option key={r} value={r}>{r}px</option>)}
        </select>
      </div>
      <div>
        <label style={{ fontSize: 10, color: '#888' }}>Steps</label>
        <select value={steps} onChange={e => setSteps(+e.target.value)} style={selectStyle}>
          {[2, 4, 8, 12, 20, 50].map(s => <option key={s} value={s}>{s}</option>)}
        </select>
      </div>
      <button onClick={generateSelectedImages} disabled={aboveCount === 0}
              style={{ padding: '5px 16px', background: aboveCount === 0 ? '#333' : '#e94560',
                       color: aboveCount === 0 ? '#888' : '#fff', border: 'none', borderRadius: 4,
                       fontWeight: 'bold', cursor: aboveCount === 0 ? 'not-allowed' : 'pointer', fontSize: 12 }}>
        Generate {aboveCount} Images
      </button>
    </div>
  );
}

function RefinePanel() {
  const { phase, tau, multiplier, setTau, setMultiplier, submitRefine,
          cells, currentGridSize, manualSelection } = useRidgeStore();
  if (phase !== 'complete') return null;

  const median = computeRealMedian(cells);
  const threshold = median * tau;
  const tauSelected = new Set<string>();
  cells.forEach(c => {
    if (c.span === 1 && c.sensitivity !== null && c.sensitivity! >= threshold)
      tauSelected.add(`${c.row},${c.col}`);
  });
  // Union of tau + manual
  const combined = new Set([...tauSelected, ...manualSelection]);
  const aboveCount = combined.size;
  const manualOnly = [...manualSelection].filter(k => !tauSelected.has(k)).length;
  const newGs = currentGridSize * multiplier;
  const newCells = aboveCount * multiplier * multiplier;

  return (
    <div style={{ display: 'flex', gap: 12, padding: '6px 12px', background: '#1a1a2e',
                  borderTop: '1px solid #333', alignItems: 'center', flexWrap: 'wrap' }}>
      <div>
        <label style={{ fontSize: 10, color: '#888' }}>tau={tau.toFixed(2)}</label>
        <input type="range" min={0} max={4} step={0.05} value={tau}
               onChange={e => setTau(+e.target.value)} style={{ display: 'block', width: 140 }} />
      </div>
      <div>
        <label style={{ fontSize: 10, color: '#888' }}>{multiplier}x → {newGs}x{newGs}</label>
        <input type="range" min={2} max={8} step={1} value={multiplier}
               onChange={e => setMultiplier(+e.target.value)} style={{ display: 'block', width: 80 }} />
      </div>
      <span style={{ fontSize: 11, color: '#aaa' }}>
        <span style={{ color: '#4ecca3' }}>{aboveCount}</span> cells
        {manualOnly > 0 && <span style={{ color: '#e94560' }}> (+{manualOnly} manual)</span>}
        {' → '}{newCells} new
      </span>
      <button onClick={submitRefine} disabled={aboveCount === 0}
              style={{ padding: '4px 16px', background: aboveCount === 0 ? '#333' : '#4ecca3',
                       color: aboveCount === 0 ? '#888' : '#000', border: 'none', borderRadius: 4,
                       fontWeight: 'bold', cursor: aboveCount === 0 ? 'not-allowed' : 'pointer', fontSize: 12 }}>
        Refine
      </button>
    </div>
  );
}

function RidgeViewer3D() {
  const { phase, cells, currentGridSize, ridgeMeshUrl, sliceIndex, setSliceIndex, dimensions, tau } = useRidgeStore();
  const plotRef = useRef<any>(null);
  const [selectedImg, setSelectedImg] = useState<{ url: string; alpha: number; beta: number; gamma: number } | null>(null);
  const [hoverImg, setHoverImg] = useState<{ url: string; x: number; y: number } | null>(null);
  const [showSurface, setShowSurface] = useState(true);

  // Compute threshold for filtering
  const realSens = useMemo(() =>
    cells.filter(c => c.sensitivity != null).map(c => c.sensitivity!), [cells]);
  const median3d = realSens.length > 0
    ? [...realSens].sort((a, b) => a - b)[Math.floor(realSens.length / 2)] : 0;
  const threshold3d = median3d * tau;

  // Build 3D scatter data — only points above tau threshold
  const scatterData = useMemo(() => {
    const x: number[] = [], y: number[] = [], z: number[] = [];
    const color: number[] = [], text: string[] = [];
    const customdata: any[] = [];

    cells.forEach(c => {
      if (c.sensitivity != null && c.sensitivity >= threshold3d) {
        x.push(c.alpha);
        y.push(c.beta);
        z.push(c.gamma);
        color.push(c.sensitivity);
        text.push(`α=${c.alpha.toFixed(2)} β=${c.beta.toFixed(2)} γ=${c.gamma.toFixed(2)}<br>sens=${c.sensitivity.toFixed(4)}`);
        customdata.push({ url: c.thumbnail_url, alpha: c.alpha, beta: c.beta, gamma: c.gamma });
      }
    });
    return { x, y, z, color, text, customdata, count: x.length, total: cells.length };
  }, [cells, threshold3d]);

  // Mesh data from ridge_mesh_url
  const [meshData, setMeshData] = useState<any>(null);
  useEffect(() => {
    if (!ridgeMeshUrl) { setMeshData(null); return; }
    fetch(ridgeMeshUrl).then(r => r.json()).then(setMeshData).catch(() => setMeshData(null));
  }, [ridgeMeshUrl]);

  // H hotkey for recentering
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.key === 'h' || e.key === 'H') && !['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement).tagName)) {
        // Reset Plotly camera to default
        const plotEl = document.querySelector('.js-plotly-plot') as any;
        if (plotEl && (window as any).Plotly) {
          (window as any).Plotly.relayout(plotEl, {
            'scene.camera': { eye: { x: 1.5, y: 1.5, z: 1.5 }, center: { x: 0, y: 0, z: 0 } },
          });
        }
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  // 2D slice of the 3D grid for image browsing
  const gs = currentGridSize;
  const sliceCells = useMemo(() =>
    cells.filter(c => c.depth === sliceIndex), [cells, sliceIndex]);

  if (phase === 'idle' || dimensions !== 3) return null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden' }}>
      {/* 3D Plot */}
      <div style={{ flex: 1, minHeight: 300 }}>
        <Suspense fallback={<div style={{ padding: 20, color: '#666' }}>Loading 3D viewer...</div>}>
          <Plot
            ref={plotRef}
            data={[
              {
                type: 'scatter3d' as const,
                mode: 'markers' as const,
                x: scatterData.x, y: scatterData.y, z: scatterData.z,
                marker: {
                  size: 4,
                  color: scatterData.color,
                  colorscale: 'Hot',
                  opacity: 0.7,
                  colorbar: { title: { text: 'Sensitivity' }, thickness: 15, len: 0.5 },
                },
                text: scatterData.text,
                customdata: scatterData.customdata as any,
                hoverinfo: 'text' as const,
                name: 'Grid points',
              },
              ...(meshData && showSurface ? [{
                type: 'mesh3d' as const,
                x: meshData.vertices.map((v: number[]) => v[0]),
                y: meshData.vertices.map((v: number[]) => v[1]),
                z: meshData.vertices.map((v: number[]) => v[2]),
                i: meshData.faces.map((f: number[]) => f[0]),
                j: meshData.faces.map((f: number[]) => f[1]),
                k: meshData.faces.map((f: number[]) => f[2]),
                opacity: 0.3,
                colorscale: [[0, '#e94560'], [1, '#e94560']] as any,
                name: 'Ridge surface',
                hoverinfo: 'skip' as const,
              }] : []),
            ]}
            layout={{
              paper_bgcolor: '#0a0a1a',
              plot_bgcolor: '#0a0a1a',
              font: { color: '#aaa', size: 10 },
              scene: {
                xaxis: { title: { text: 'α (→B)' }, color: '#666', gridcolor: '#222' },
                yaxis: { title: { text: 'β (→C)' }, color: '#666', gridcolor: '#222' },
                zaxis: { title: { text: 'γ (→D)' }, color: '#666', gridcolor: '#222' },
                bgcolor: '#0a0a1a',
                dragmode: 'orbit',
              },
              margin: { l: 0, r: 0, t: 30, b: 0 },
              title: { text: `${scatterData.count}/${scatterData.total} points above τ=${tau.toFixed(1)}${meshData ? ` | ridge surface` : ''}`, font: { size: 12, color: '#aaa' } },
              showlegend: false,
              autosize: true,
            }}
            useResizeHandler
            style={{ width: '100%', height: '100%' }}
            config={{
              responsive: true,
              scrollZoom: true,
              displayModeBar: true,
              modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'] as any,
            }}
            onClick={(event: any) => {
              if (selectedImg) return; // don't re-trigger while overlay is open
              const point = event.points?.[0];
              if (point?.customdata?.url) {
                setSelectedImg(point.customdata);
              }
            }}
            onHover={(event: any) => {
              const point = event.points?.[0];
              if (point?.customdata?.url && event.event) {
                setHoverImg({ url: point.customdata.url, x: event.event.clientX, y: event.event.clientY });
              }
            }}
            onUnhover={() => setHoverImg(null)}
          />
        </Suspense>
      </div>

      {/* Slice browser */}
      <div style={{ padding: '6px 12px', background: '#16213e', borderTop: '1px solid #333',
                    display: 'flex', gap: 12, alignItems: 'center' }}>
        <span style={{ fontSize: 11, color: '#888' }}>Z-slice: {sliceIndex}/{gs - 1}</span>
        <input type="range" min={0} max={Math.max(0, gs - 1)} step={1} value={sliceIndex}
               onChange={e => setSliceIndex(+e.target.value)}
               style={{ flex: 1, maxWidth: 300 }} />
        <span style={{ fontSize: 11, color: '#666' }}>
          γ={gs > 0 ? (sliceIndex / Math.max(gs - 1, 1)).toFixed(2) : '0'} | {sliceCells.length} cells
        </span>
        {meshData && (
          <button onClick={() => setShowSurface(s => !s)}
                  style={{ padding: '2px 10px', border: 'none', borderRadius: 3, fontSize: 10,
                           background: showSurface ? '#e94560' : '#333',
                           color: showSurface ? '#fff' : '#888', cursor: 'pointer' }}>
            {showSurface ? 'surface ON' : 'surface OFF'}
          </button>
        )}
        <span style={{ fontSize: 10, color: '#444' }}>
          click=image | H=recenter | scroll=zoom
        </span>
      </div>

      {/* Hover image preview */}
      {hoverImg && !selectedImg && (
        <div style={{
          position: 'fixed',
          left: Math.min(hoverImg.x + 15, window.innerWidth - 180),
          top: Math.min(hoverImg.y + 15, window.innerHeight - 180),
          pointerEvents: 'none', zIndex: 900,
        }}>
          <img src={hoverImg.url}
               style={{ width: 160, height: 160, borderRadius: 6,
                        border: '2px solid #444', boxShadow: '0 0 20px rgba(0,0,0,0.8)' }} />
        </div>
      )}

      {/* Full image overlay on click */}
      {selectedImg && (
        <div onMouseDown={(e) => { e.stopPropagation(); setSelectedImg(null); }}
             style={{
               position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.92)',
               display: 'flex', alignItems: 'center', justifyContent: 'center',
               zIndex: 1000, cursor: 'pointer',
             }}>
          <div style={{ textAlign: 'center' }} onMouseDown={e => e.stopPropagation()}>
            <img src={selectedImg.url}
                 style={{ maxWidth: '80vw', maxHeight: '70vh', borderRadius: 8,
                          boxShadow: '0 0 40px rgba(0,0,0,0.5)' }} />
            <div style={{ marginTop: 8, fontSize: 12, color: '#aaa' }}>
              α={selectedImg.alpha.toFixed(3)} β={selectedImg.beta.toFixed(3)} γ={selectedImg.gamma.toFixed(3)}
            </div>
            <button onClick={() => setSelectedImg(null)}
                    style={{ marginTop: 8, padding: '4px 16px', background: '#333', color: '#aaa',
                             border: '1px solid #555', borderRadius: 4, cursor: 'pointer', fontSize: 11 }}>
              Close (or click outside)
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

/** Compute median from only real (span=1) cells to match backend. */
function computeRealMedian(cells: { sensitivity: number | null; span: number }[]): number {
  const real = cells.filter(c => c.span === 1 && c.sensitivity !== null).map(c => c.sensitivity!);
  if (real.length === 0) {
    // Fallback to all cells
    const all = cells.filter(c => c.sensitivity !== null).map(c => c.sensitivity!);
    if (all.length === 0) return 0;
    return [...all].sort((a, b) => a - b)[Math.floor(all.length / 2)];
  }
  return [...real].sort((a, b) => a - b)[Math.floor(real.length / 2)];
}

type LayerToggle = { images: boolean; heatmap: boolean; tau: boolean };

function UnifiedViewport() {
  const { phase, cells, currentGridSize, tau, seeds, seedCells, activeSeedIdx, setActiveSeedIdx,
          manualSelection, toggleManualCell, clearManualSelection } = useRidgeStore();

  // Layer toggles
  const [layers, setLayers] = useState<LayerToggle>({ images: true, heatmap: false, tau: true });
  const toggle = (key: keyof LayerToggle) => setLayers(l => ({ ...l, [key]: !l[key] }));

  // Auto-enable heatmap layer when fast scan completes
  const prevPhase = useRef(phase);
  useEffect(() => {
    if (phase === 'scan_complete' && prevPhase.current !== 'scan_complete') {
      setLayers(l => ({ ...l, heatmap: true }));
    }
    prevPhase.current = phase;
  }, [phase]);

  // Full-res overlay + seed probe
  const [overlayImg, setOverlayImg] = useState<{ url: string; alpha: number; beta: number } | null>(null);
  const [probeSeedStart, setProbeSeedStart] = useState(0);
  const [probeSeedEnd, setProbeSeedEnd] = useState(3);
  const [probeImages, setProbeImages] = useState<{ seeds: number[]; images: (string | null)[] } | null>(null);
  const [probeLoading, setProbeLoading] = useState(false);

  // Pan/zoom state
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const isPanning = useRef(false);
  const lastMouse = useRef({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  // For multi-seed: show active seed's thumbnails but use averaged sensitivity
  const isMultiSeed = seeds.length > 1 && seedCells;
  const activeSeed = seeds[activeSeedIdx] ?? seeds[0];
  const displayCells = useMemo(() => {
    if (!isMultiSeed || !seedCells) return cells;
    const seedKey = String(activeSeed);
    const seedData = seedCells[seedKey];
    if (!seedData) return cells;
    // Merge: use seed's thumbnail_url, but primary cells' sensitivity
    const sensMap = new Map<string, number | null>();
    cells.forEach(c => sensMap.set(`${c.row},${c.col}`, c.sensitivity));
    return seedData.map(sc => ({
      ...sc,
      sensitivity: sensMap.get(`${sc.row},${sc.col}`) ?? sc.sensitivity,
    }));
  }, [cells, seedCells, activeSeed, isMultiSeed]);

  const cellMap = useMemo(() => {
    const m = new Map<string, typeof displayCells[0]>();
    displayCells.forEach(c => m.set(`${c.row},${c.col}`, c));
    return m;
  }, [cells]);

  const allSens = useMemo(() =>
    cells.filter(c => c.sensitivity !== null).map(c => c.sensitivity!), [cells]);
  const median = useMemo(() => computeRealMedian(cells), [cells]);
  const threshold = median * tau;
  const sensMax = allSens.length > 0 ? Math.max(...allSens) : 1;
  const sensMin = allSens.length > 0 ? Math.min(...allSens) : 0;

  // Only span=1 cells can be selected for refinement (matching backend)
  const selectedSet = useMemo(() => {
    const s = new Set<string>();
    cells.forEach(c => {
      if (c.span === 1 && c.sensitivity !== null && c.sensitivity >= threshold) {
        s.add(`${c.row},${c.col}`);
      }
    });
    return s;
  }, [cells, threshold]);

  // Track all grid positions covered by any cell (including spanned sub-positions)
  const coveredSet = useMemo(() => {
    const s = new Set<string>();
    cells.forEach(c => {
      const span = c.span || 1;
      for (let di = 0; di < span; di++) {
        for (let dj = 0; dj < span; dj++) {
          s.add(`${c.row + di},${c.col + dj}`);
        }
      }
    });
    return s;
  }, [cells]);

  // Recenter: fit the full grid in the viewport
  const recenter = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const gs = currentGridSize;
    const ts = Math.max(4, Math.min(64, Math.floor(800 / gs)));
    const gpx = gs * ts;
    // Fit: scale so the grid fills the viewport with some padding
    const fitZoom = Math.min(rect.width / gpx, rect.height / gpx) * 0.95;
    setZoom(fitZoom);
    setPan({ x: (rect.width - gpx * fitZoom) / 2, y: (rect.height - gpx * fitZoom) / 2 });
  }, [currentGridSize]);

  // "H" key to recenter
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'h' || e.key === 'H') {
        // Don't trigger if typing in an input
        const tag = (e.target as HTMLElement).tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
        recenter();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [recenter]);

  // Center the grid whenever grid size changes.
  const { shouldCenter } = useRidgeStore();
  const needsCenter = useRef(true);

  useEffect(() => { if (shouldCenter) needsCenter.current = true; }, [currentGridSize, shouldCenter]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const doCenter = () => {
      if (!needsCenter.current) return;
      recenter();
      needsCenter.current = false;
    };
    doCenter();
    const observer = new ResizeObserver(() => doCenter());
    observer.observe(el);
    return () => observer.disconnect();
  });

  // Track which tile the mouse is hovering over
  const hoverTile = useRef<{ row: number; col: number } | null>(null);

  // Zoom: center on the hovered tile
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const handler = (e: WheelEvent) => {
      e.preventDefault();
      const factor = e.deltaY > 0 ? 0.9 : 1.1;

      setZoom(oldZoom => {
        const newZoom = Math.max(0.2, Math.min(20, oldZoom * factor));

        if (hoverTile.current) {
          const gs = currentGridSize;
          const ts = Math.max(4, Math.min(64, Math.floor(800 / gs)));

          // World-space center of the hovered tile
          const tileWorldX = (hoverTile.current.col + 0.5) * ts;
          const tileWorldY = (hoverTile.current.row + 0.5) * ts;

          // Screen-space center of the container
          const rect = el.getBoundingClientRect();
          const screenCx = rect.width / 2;
          const screenCy = rect.height / 2;

          // New pan: place the tile's world center at the screen center
          // screen = pan + world * zoom => pan = screen - world * zoom
          // We interpolate: shift pan toward centering the tile
          const targetPanX = screenCx - tileWorldX * newZoom;
          const targetPanY = screenCy - tileWorldY * newZoom;

          setPan(p => ({
            x: p.x + (targetPanX - p.x) * 0.3,
            y: p.y + (targetPanY - p.y) * 0.3,
          }));
        }

        return newZoom;
      });
    };
    el.addEventListener('wheel', handler, { passive: false });
    return () => el.removeEventListener('wheel', handler);
  });

  if (phase === 'idle') return null;

  const gs = currentGridSize;
  const baseTileSize = Math.max(4, Math.min(64, Math.floor(800 / gs)));
  const ts = baseTileSize;
  const gridPx = gs * ts;

  // Sensitivity to color
  const sensToColor = (s: number): string => {
    const t = (s - sensMin) / (sensMax - sensMin + 1e-10);
    const r = Math.round(255 * Math.min(1, t * 2));
    const g = Math.round(255 * Math.max(0, t - 0.5) * 2);
    const b = 0;
    return `rgb(${r},${g},${b})`;
  };

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Layer toggle buttons */}
      <div style={{ display: 'flex', gap: 4, padding: '4px 12px', background: '#12121f' }}>
        {(['images', 'heatmap', 'tau'] as const).map(key => (
          <button key={key} onClick={() => toggle(key)}
                  style={{
                    padding: '3px 12px', border: 'none', borderRadius: 3, fontSize: 11,
                    background: layers[key] ? '#4ecca3' : '#333',
                    color: layers[key] ? '#000' : '#888',
                    cursor: 'pointer', fontWeight: layers[key] ? 'bold' : 'normal',
                  }}>
            {key}
          </button>
        ))}
        {isMultiSeed && (
          <>
            <span style={{ fontSize: 10, color: '#555', marginLeft: 8 }}>seed:</span>
            {seeds.map((s, idx) => (
              <button key={s} onClick={() => setActiveSeedIdx(idx)}
                      style={{
                        padding: '2px 8px', border: 'none', borderRadius: 3, fontSize: 10,
                        background: idx === activeSeedIdx ? '#e94560' : '#333',
                        color: idx === activeSeedIdx ? '#fff' : '#888',
                        cursor: 'pointer',
                      }}>
                {s}
              </button>
            ))}
          </>
        )}
        {manualSelection.size > 0 && (
          <button onClick={clearManualSelection}
                  style={{ padding: '2px 8px', border: 'none', borderRadius: 3, fontSize: 10,
                           background: '#e94560', color: '#fff', cursor: 'pointer', marginLeft: 4 }}>
            clear {manualSelection.size} selected
          </button>
        )}
        <span style={{ fontSize: 10, color: '#555', marginLeft: 8 }}>
          scroll=zoom | drag=pan | dblclick=fullres | rightclick=select | H=recenter
        </span>
      </div>

      {/* Viewport */}
      <div ref={containerRef}
           onContextMenu={(e) => {
             e.preventDefault();
             // Right-click: toggle manual selection of the cell under cursor
             const rect = containerRef.current?.getBoundingClientRect();
             if (!rect || phase !== 'complete') return;
             const mx = e.clientX - rect.left;
             const my = e.clientY - rect.top;
             const wx = (mx - pan.x) / zoom;
             const wy = (my - pan.y) / zoom;
             const col = Math.floor(wx / ts);
             const row = Math.floor(wy / ts);
             if (col >= 0 && col < gs && row >= 0 && row < gs) {
               const alphaIdx = col;
               const betaIdx = gs - 1 - row;
               // Find the cell that covers this position (any span)
               const cell = displayCells.find(c => {
                 const s = c.span || 1;
                 return alphaIdx >= c.row && alphaIdx < c.row + s &&
                        betaIdx >= c.col && betaIdx < c.col + s;
               });
               if (cell) {
                 toggleManualCell(cell.row, cell.col);
               }
             }
           }}
           onDoubleClick={(e) => {
             const rect = containerRef.current?.getBoundingClientRect();
             if (!rect) return;
             const mx = e.clientX - rect.left;
             const my = e.clientY - rect.top;
             const wx = (mx - pan.x) / zoom;
             const wy = (my - pan.y) / zoom;
             const col = Math.floor(wx / ts);
             const row = Math.floor(wy / ts);
             if (col >= 0 && col < gs && row >= 0 && row < gs) {
               const alphaIdx = col;
               const betaIdx = gs - 1 - row;
               // Find the cell that covers this position
               const cell = displayCells.find(c => {
                 const s = c.span || 1;
                 return alphaIdx >= c.row && alphaIdx < c.row + s &&
                        betaIdx >= c.col && betaIdx < c.col + s;
               });
               if (cell?.thumbnail_url) {
                 setOverlayImg({
                   url: cell.thumbnail_url,
                   alpha: cell.alpha,
                   beta: cell.beta,
                 });
               }
             }
           }}
           onPointerDown={(e) => {
             isPanning.current = true;
             lastMouse.current = { x: e.clientX, y: e.clientY };
             containerRef.current?.setPointerCapture(e.pointerId);
           }}
           onPointerMove={(e) => {
             // Track hovered tile from screen coords
             const rect = containerRef.current?.getBoundingClientRect();
             if (rect) {
               const mx = e.clientX - rect.left;
               const my = e.clientY - rect.top;
               // Convert screen to world: world = (screen - pan) / zoom
               const wx = (mx - pan.x) / zoom;
               const wy = (my - pan.y) / zoom;
               const gs = currentGridSize;
               const ts = Math.max(4, Math.min(64, Math.floor(800 / gs)));
               const col = Math.floor(wx / ts);
               const row = Math.floor(wy / ts);
               if (col >= 0 && col < gs && row >= 0 && row < gs) {
                 hoverTile.current = { row, col };
               } else {
                 hoverTile.current = null;
               }
             }

             if (!isPanning.current) return;
             const dx = e.clientX - lastMouse.current.x;
             const dy = e.clientY - lastMouse.current.y;
             lastMouse.current = { x: e.clientX, y: e.clientY };
             setPan(p => ({ x: p.x + dx, y: p.y + dy }));
           }}
           onPointerUp={() => { isPanning.current = false; }}
           onLostPointerCapture={() => { isPanning.current = false; }}
           style={{
             flex: 1, overflow: 'hidden', cursor: 'grab',
             background: '#0a0a12', position: 'relative', minHeight: 400,
             userSelect: 'none', touchAction: 'none',
           }}>
        <div style={{
          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
          transformOrigin: '0 0',
          position: 'absolute',
          left: 0, top: 0,
          width: gridPx, height: gridPx,
          pointerEvents: 'none',
        }}>
          {/* Render grid cells */}
          {displayCells.map((cell) => {
            const span = cell.span || 1;
            const alphaIdx = cell.row;
            const betaIdx = cell.col;
            const displayCol = alphaIdx;
            const displayRow = gs - 1 - betaIdx;

            // For spanned cells, displayRow is the TOP of the span block
            // (betaIdx is the bottom-left in data coords, display row is flipped)
            const topRow = displayRow - (span - 1);
            const x = displayCol * ts;
            const y = topRow * ts;
            const cellW = span * ts;
            const cellH = span * ts;

            const isAboveTau = selectedSet.has(`${alphaIdx},${betaIdx}`);
            const hasSens = cell.sensitivity != null;
            const sensVal = cell.sensitivity ?? 0;

            return (
              <div key={`${cell.row},${cell.col}`} style={{
                position: 'absolute', left: x, top: y, width: cellW, height: cellH,
                overflow: 'hidden',
              }}>
                {/* Layer: image */}
                {layers.images && cell.thumbnail_url && (
                  <img src={cell.thumbnail_url} width={cellW} height={cellH}
                       style={{ position: 'absolute', inset: 0, display: 'block' }}
                       draggable={false} />
                )}

                {/* Layer: heatmap */}
                {layers.heatmap && hasSens && (
                  <div style={{
                    position: 'absolute', inset: 0,
                    background: sensToColor(sensVal),
                    opacity: layers.images ? 0.6 : 1,
                  }} />
                )}

                {/* Layer: tau selection borders */}
                {layers.tau && (phase === 'complete' || phase === 'scan_complete') && isAboveTau && (
                  <div style={{
                    position: 'absolute', inset: 0,
                    border: '1px solid #4ecca3',
                    boxSizing: 'border-box',
                    pointerEvents: 'none',
                  }} />
                )}

                {/* Layer: manual selection borders */}
                {(phase === 'complete' || phase === 'scan_complete') && manualSelection.has(`${cell.row},${cell.col}`) && (
                  <div style={{
                    position: 'absolute', inset: 0,
                    border: '2px solid #e94560',
                    boxSizing: 'border-box',
                    pointerEvents: 'none',
                  }} />
                )}

                {/* Empty cell placeholder */}
                {!cell.thumbnail_url && !hasSens && (
                  <div style={{
                    position: 'absolute', inset: 0,
                    background: '#181828', border: '1px solid #222', boxSizing: 'border-box',
                  }} />
                )}
              </div>
            );
          })}

          {/* Fill remaining empty cells (not covered by any cell or span) */}
          {Array.from({ length: gs * gs }).map((_, idx) => {
            const displayRow = Math.floor(idx / gs);
            const displayCol = idx % gs;
            const alphaIdx = displayCol;
            const betaIdx = gs - 1 - displayRow;
            // Skip if covered by any cell (including spanned sub-positions)
            if (coveredSet.has(`${alphaIdx},${betaIdx}`)) return null;
            const x = displayCol * ts;
            const y = displayRow * ts;
            return (
              <div key={`empty-${idx}`} style={{
                position: 'absolute', left: x, top: y, width: ts, height: ts,
                background: '#181828', border: '1px solid #1a1a2a', boxSizing: 'border-box',
              }} />
            );
          })}
        </div>
      </div>

      {/* Full-res image overlay with seed probe */}
      {overlayImg && (
        <div onClick={() => { setOverlayImg(null); setProbeImages(null); }}
             style={{
               position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.9)',
               display: 'flex', alignItems: 'flex-start', justifyContent: 'center',
               zIndex: 1000, cursor: 'pointer', overflowY: 'auto', padding: '40px 20px',
             }}>
          <div style={{ textAlign: 'center', maxWidth: 900 }} onClick={e => e.stopPropagation()}>
            <img src={overlayImg.url}
                 style={{ maxWidth: '80vw', maxHeight: '50vh', borderRadius: 8,
                          boxShadow: '0 0 40px rgba(0,0,0,0.5)' }} />
            <div style={{ marginTop: 8, fontSize: 12, color: '#aaa' }}>
              α={overlayImg.alpha.toFixed(3)} β={overlayImg.beta.toFixed(3)}
            </div>

            {/* Seed probe controls */}
            <div style={{ marginTop: 16, padding: 12, background: '#1a1a2e', borderRadius: 8,
                          display: 'flex', gap: 12, alignItems: 'center', justifyContent: 'center',
                          flexWrap: 'wrap' }}>
              <span style={{ fontSize: 12, color: '#888' }}>Explore seeds:</span>
              <input type="number" value={probeSeedStart}
                     onChange={e => setProbeSeedStart(+e.target.value)}
                     style={{ width: 50, padding: 4, background: '#0f3460', border: '1px solid #333',
                              color: '#fff', borderRadius: 4, fontSize: 11 }} />
              <span style={{ color: '#555' }}>to</span>
              <input type="number" value={probeSeedEnd}
                     onChange={e => setProbeSeedEnd(+e.target.value)}
                     style={{ width: 50, padding: 4, background: '#0f3460', border: '1px solid #333',
                              color: '#fff', borderRadius: 4, fontSize: 11 }} />
              <span style={{ fontSize: 11, color: '#666' }}>
                ({probeSeedEnd - probeSeedStart + 1} seeds)
              </span>
              <button disabled={probeLoading}
                      onClick={async () => {
                        const { jobId } = useRidgeStore.getState();
                        if (!jobId) return;
                        setProbeLoading(true);
                        setProbeImages(null);
                        try {
                          const res = await startSeedProbe(jobId, {
                            alpha: overlayImg.alpha, beta: overlayImg.beta,
                            seed_start: probeSeedStart, seed_end: probeSeedEnd,
                          });
                          if (res.status === 'running') {
                            // Poll for completion
                            const poll = setInterval(async () => {
                              const st = await getSeedProbeStatus(res.probe_id);
                              setProbeImages({ seeds: st.seeds, images: st.images });
                              if (st.complete) {
                                clearInterval(poll);
                                setProbeLoading(false);
                              }
                            }, 1000);
                          }
                        } catch (err) {
                          console.error(err);
                          setProbeLoading(false);
                        }
                      }}
                      style={{ padding: '4px 14px', background: probeLoading ? '#555' : '#e94560',
                               color: '#fff', border: 'none', borderRadius: 4, fontSize: 11,
                               cursor: probeLoading ? 'not-allowed' : 'pointer' }}>
                {probeLoading ? 'Generating...' : 'Generate'}
              </button>
            </div>

            {/* Seed probe gallery */}
            {probeImages && (
              <div style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 4,
                            justifyContent: 'center' }}>
                {probeImages.seeds.map((seed, idx) => (
                  <div key={seed} style={{ textAlign: 'center' }}>
                    {probeImages.images[idx] ? (
                      <img src={probeImages.images[idx]!}
                           style={{ width: 128, height: 128, borderRadius: 4, display: 'block',
                                    border: '1px solid #333' }} />
                    ) : (
                      <div style={{ width: 128, height: 128, background: '#222', borderRadius: 4,
                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                    color: '#555', fontSize: 10 }}>...</div>
                    )}
                    <div style={{ fontSize: 9, color: '#666', marginTop: 2 }}>seed {seed}</div>
                  </div>
                ))}
              </div>
            )}

            <div style={{ marginTop: 12, fontSize: 10, color: '#444' }}>
              click outside to close
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function MainViewport() {
  const { dimensions } = useRidgeStore();

  if (dimensions === 3) {
    return <RidgeViewer3D />;
  }
  return <UnifiedViewport />;
}

export default function App() {
  return (
    <div style={{ background: '#0a0a1a', height: '100vh', color: '#fff', fontFamily: 'system-ui',
                  display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '8px 12px', background: '#16213e', borderBottom: '1px solid #333' }}>
        <h1 style={{ margin: 0, fontSize: 16 }}>Ridge Explorer
          <span style={{ fontSize: 11, color: '#666', marginLeft: 8 }}>FLUX.2 Klein 4B + DINOv2</span>
        </h1>
      </div>
      <PromptInput />
      <ProgressBar />
      <ScanCompletePanel />
      <RefinePanel />
      <MainViewport />
    </div>
  );
}
