import { useMemo, useRef, useState, useCallback, useEffect, lazy, Suspense } from 'react';
import { useRidgeStore } from './stores/ridgeStore';
import { useProbeStore, probeAt, probeLabel } from './stores/probeStore';
import { startSeedProbe, getSeedProbeStatus } from './api/client';
import * as api from './api/client';
import { loadResume, saveResume, onResumeVisible } from './resume';
import DiscoverPanel from './DiscoverPanel';
import CascadePanel from './CascadePanel';
import AmrPanel from './AmrPanel';
import MetroPanel from './MetroPanel';
import MicroscopePanel from './MicroscopePanel';
import SurpriseSlider from './components/SurpriseSlider';
import { feature } from './featureFlags';

const Plot = lazy(() => import('react-plotly.js'));

const RESOLUTION_OPTIONS = [128, 256, 384, 512];
const STEPS_OPTIONS = [2, 4, 8, 12, 20, 50];

// The app's accent. Probes reuse it so a measured direction reads as part of the
// tool's own instrumentation rather than as another data layer.
const ACCENT = '#4ecca3';
// Everything a probe draws sits on the heatmap, which runs black → red → yellow, so
// no single colour is legible everywhere; a halo in the viewport's own background
// colour goes down first.
const HALO = '#0a0a12';

function PromptInput() {
  const { promptA, promptB, promptC, promptD, dimensions, gridSize, seed, steps, resolution, seedCount, phase,
          setPromptA, setPromptB, setPromptC, setPromptD, setDimensions, setGridSize, setSeed, setSteps, setResolution, setSeedCount, generate, fastScan, mfScan, cancel } = useRidgeStore();
  const busy = phase !== 'idle' && phase !== 'complete' && phase !== 'scan_complete';

  // Hiding the mode selector must not strand the app in 3D: prompt D, the Plotly view
  // and the refine path all key off `dimensions`, and with the selector gone there is
  // no control left to get back out.
  useEffect(() => {
    if (!feature('threed') && dimensions !== 2) setDimensions(2);
  }, [dimensions, setDimensions]);

  const inputStyle = { width: '100%', padding: 6, background: '#0f3460', border: '1px solid #333',
                       color: '#fff', borderRadius: 4, fontSize: 12 };
  const selectStyle = { padding: 4, background: '#0f3460', border: '1px solid #333',
                        color: '#fff', borderRadius: 4, fontSize: 11 };

  return (
    <div style={{ display: 'flex', gap: 10, padding: 10, background: '#16213e',
                  alignItems: 'flex-end', flexWrap: 'wrap' }}>
      {feature('threed') && (
        <div>
          <label style={{ fontSize: 10, color: '#888' }}>Mode</label>
          <select value={dimensions} onChange={e => setDimensions(+e.target.value)} style={selectStyle}>
            <option value={2}>2D</option>
            <option value={3}>3D</option>
          </select>
        </div>
      )}
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
      {feature('seeds') && (
        <div>
          <label style={{ fontSize: 10, color: '#888' }}>Seeds {seedCount > 1 ? `(${seed}-${seed+seedCount-1})` : ''}</label>
          <select value={seedCount} onChange={e => setSeedCount(+e.target.value)} style={selectStyle}>
            {[1, 2, 3, 4, 5].map(n => <option key={n} value={n}>{n}{n > 1 ? ' seeds' : ' seed'}</option>)}
          </select>
        </div>
      )}
      {feature('explore') && (
        <button onClick={generate} disabled={busy}
                style={{ padding: '5px 18px', background: busy ? '#555' : '#e94560',
                         color: '#fff', border: 'none', borderRadius: 4, fontWeight: 'bold',
                         cursor: busy ? 'not-allowed' : 'pointer', fontSize: 12 }}>
          {busy ? 'Working...' : 'Explore'}
        </button>
      )}
      {busy && (
        <button onClick={cancel}
                style={{ padding: '5px 14px', background: '#333', color: '#e94560',
                         border: '1px solid #e94560', borderRadius: 4, fontWeight: 'bold',
                         cursor: 'pointer', fontSize: 12 }}>
          Cancel
        </button>
      )}
      {feature('fastscan') && (
        <button onClick={fastScan} disabled={busy}
                style={{ padding: '5px 14px', background: busy ? '#555' : '#0f3460',
                         color: '#fff', border: '1px solid #4ecca3', borderRadius: 4, fontWeight: 'bold',
                         cursor: busy ? 'not-allowed' : 'pointer', fontSize: 12 }}>
          {busy ? '...' : 'Fast Scan'}
        </button>
      )}
      {feature('mfscan') && (
        <button onClick={mfScan} disabled={busy}
                style={{ padding: '5px 14px', background: busy ? '#555' : '#1a5276',
                         color: '#fff', border: '1px solid #f39c12', borderRadius: 4, fontWeight: 'bold',
                         cursor: busy ? 'not-allowed' : 'pointer', fontSize: 12 }}>
          {busy ? '...' : 'MF Scan'}
        </button>
      )}
    </div>
  );
}

// ------------------------------------------------------------- Token probe
// Which words of a prompt the generation is actually leaning on. One probe is ~3 min
// of GPU, so this sits collapsed next to the prompts and is never started implicitly.
function TokenPanel() {
  const { jobId, promptA, promptB, promptC, promptD, dimensions, seed } = useRidgeStore();
  const { token, startToken, clearToken } = useProbeStore();
  const [open, setOpen] = useState(false);
  const [showTemplate, setShowTemplate] = useState(false);

  // Nothing to probe before a job exists: the backend needs the job's prompts.
  if (!jobId) return null;

  const which: { key: 'a' | 'b' | 'c' | 'd'; label: string; prompt: string }[] = [
    { key: 'a', label: 'A', prompt: promptA },
    { key: 'b', label: 'B', prompt: promptB },
    { key: 'c', label: 'C', prompt: promptC },
    ...(dimensions === 3 ? [{ key: 'd' as const, label: 'D', prompt: promptD }] : []),
  ];

  const running = token.status === 'running';
  const result = token.status === 'done' ? token.result : null;
  const content = result ? result.rows.filter(r => r.cls === 'content') : [];
  const other = result ? result.rows.filter(r => r.cls !== 'content') : [];
  const ranked = [...content].sort((a, b) => b.sigma - a.sigma);
  const maxSigma = ranked.length ? Math.max(...ranked.map(r => r.sigma)) : 0;

  const bar = (r: { text: string; sigma: number; cls: string }, key: string, muted: boolean) => (
    <div key={key} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 2 }}>
      <span style={{ width: 130, flexShrink: 0, fontSize: 11, textAlign: 'right',
                     color: muted ? '#555' : '#ddd', overflow: 'hidden',
                     textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
            title={`${r.text} (${r.cls})`}>
        {r.text}
      </span>
      <div style={{ flex: 1, minWidth: 60, height: 10, background: '#12121f', borderRadius: 2 }}>
        {/* clamped: the scale is set by the top CONTENT token, and a template or pad
            token occasionally beats it — which must not run the bar off its track */}
        <div style={{ width: maxSigma > 0
                        ? `${Math.min(100, Math.max(1, (r.sigma / maxSigma) * 100))}%` : '0%',
                      height: '100%', borderRadius: 2,
                      background: muted ? '#3a3a4e' : ACCENT }} />
      </div>
      <span style={{ width: 54, flexShrink: 0, fontSize: 10,
                     color: muted ? '#555' : '#8ab' }}>
        {r.sigma.toPrecision(3)}
      </span>
    </div>
  );

  return (
    <div style={{ background: '#151528', borderTop: '1px solid #222' }}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', padding: '4px 12px',
                    flexWrap: 'wrap' }}>
        <button className="rx-focus" onClick={() => setOpen(o => !o)}
                aria-expanded={open}
                style={{ padding: '2px 10px', background: '#0f3460', color: '#aaa',
                         border: '1px solid #333', borderRadius: 3, fontSize: 11 }}>
          {open ? '▾' : '▸'} Which words matter?
        </button>
        {open && which.map(w => (
          <button key={w.key} className="rx-focus"
                  disabled={running}
                  onClick={() => startToken(jobId, w.key, seed)}
                  title={w.prompt}
                  style={{ padding: '2px 10px', borderRadius: 3, fontSize: 11,
                           background: running ? '#333'
                             : token.which === w.key ? ACCENT : '#0f3460',
                           color: running ? '#777' : token.which === w.key ? '#000' : '#cfd',
                           border: `1px solid ${running ? '#333' : ACCENT}`,
                           cursor: running ? 'not-allowed' : 'pointer' }}>
            Which words matter? ({w.label})
          </button>
        ))}
        {open && running && (
          <span style={{ fontSize: 11, color: ACCENT }}>
            probing prompt {token.which?.toUpperCase()}… (~3 min)
          </span>
        )}
        {open && result && (
          <button className="rx-focus" onClick={clearToken}
                  style={{ padding: '2px 8px', background: '#222', color: '#888',
                           border: '1px solid #333', borderRadius: 3, fontSize: 10 }}>
            clear
          </button>
        )}
      </div>

      {open && token.status === 'error' && (
        <div style={{ padding: '0 12px 6px 12px', fontSize: 11, color: '#ff8a9c' }}>
          token probe failed: {token.error}
        </div>
      )}

      {open && result && (
        <div style={{ padding: '2px 12px 8px 12px', maxWidth: 620 }}>
          <div style={{ fontSize: 10, color: '#666', marginBottom: 4 }}>
            prompt {token.which?.toUpperCase()}: “{result.prompt}” · {result.n_jvp} JVPs ·{' '}
            {result.steps} steps · seed {result.seed} · {result.wall_s.toFixed(1)} s
          </div>
          {ranked.length === 0 && (
            <div style={{ fontSize: 11, color: '#888' }}>no content tokens in this prompt.</div>
          )}
          {ranked.map(r => bar(r, `c${r.pos}`, false))}

          {other.length > 0 && (
            <button className="rx-focus" onClick={() => setShowTemplate(s => !s)}
                    aria-expanded={showTemplate}
                    style={{ marginTop: 4, padding: '1px 8px', background: 'transparent',
                             color: '#666', border: '1px solid #2a2a3a', borderRadius: 3,
                             fontSize: 10 }}>
              {showTemplate ? 'hide' : 'show'} template/pad ({other.length})
            </button>
          )}
          {showTemplate && (
            <div style={{ marginTop: 4 }}>
              {other.map(r => bar(r, `o${r.pos}`, true))}
            </div>
          )}

          <div style={{ fontSize: 10, color: '#777', marginTop: 6, lineHeight: 1.4 }}>
            Rankings are seed-invariant and follow the word, not its position. They say
            which word the generation depends on here — not how the image would change.
          </div>
        </div>
      )}
    </div>
  );
}

function ProgressBar() {
  const { phase, cellsGenerated, cellsTotal, currentGridSize, cancel, error } = useRidgeStore();
  if (phase === 'idle') return null;
  const busy = phase === 'scanning' || phase === 'generating' || phase === 'analyzing'
    || phase === 'mf_scanning' || phase === 'mf_jacobian_done' || phase === 'mf_finalizing';
  const pct = cellsTotal > 0 ? (cellsGenerated / cellsTotal * 100) : 0;
  const label = phase === 'scanning' ? `Fast scan: ${cellsGenerated}/${cellsTotal} latents`
    : phase === 'mf_scanning' ? `MF scan: ${cellsGenerated}/${cellsTotal} latents`
    : phase === 'mf_jacobian_done' ? 'MF: selecting cells via GP...'
    : phase === 'mf_finalizing' ? 'MF: computing GP sensitivity map...'
    : phase === 'generating' ? `Generating: ${cellsGenerated}/${cellsTotal}`
    : phase === 'analyzing' ? 'Computing ridges...'
    : phase === 'scan_complete' ? `Fast scan complete (${currentGridSize}×${currentGridSize})`
    // phase 'error' used to fall through to the generic grid label, so a job the server
    // had lost or failed looked identical to a finished one
    : phase === 'error' ? 'Job failed'
    : `${currentGridSize}×${currentGridSize} grid`;

  return (
    <div style={{ padding: '4px 12px', background: '#1a1a2e', display: 'flex', alignItems: 'center', gap: 8 }}>
      <span style={{ fontSize: 11, color: phase === 'error' ? '#e94560' : '#888', minWidth: 160 }}>{label}</span>
      {phase === 'error' && error && (
        <span style={{ fontSize: 11, color: '#ff8a9c', flex: 1 }}>{error}</span>
      )}
      {busy && (
        <div style={{ flex: 1, background: '#333', borderRadius: 3, height: 4 }}>
          <div style={{ width: `${pct}%`, background: phase === 'scanning' || phase === 'mf_scanning' ? '#4ecca3' : phase.startsWith('mf_') ? '#f39c12' : '#e94560', borderRadius: 3, height: '100%', transition: 'width 0.3s' }} />
        </div>
      )}
      {busy && (
        <button onClick={cancel}
                style={{ padding: '2px 10px', background: '#333', color: '#e94560',
                         border: '1px solid #e94560', borderRadius: 3,
                         cursor: 'pointer', fontSize: 10, flexShrink: 0 }}>
          Cancel
        </button>
      )}
    </div>
  );
}

function ScanCompletePanel() {
  const { phase, tau, setTau, cells, steps, resolution, setSteps, setResolution,
          generateSelectedImages, jobId, setManualSelection } = useRidgeStore();
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
      {feature('surprise') && (
        <SurpriseSlider jobId={jobId}
          onSample={cells => setManualSelection(cells.map(([r, c]) => `${r},${c}`))} />
      )}
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
          cells, currentGridSize, manualSelection, dimensions } = useRidgeStore();
  if (phase !== 'complete') return null;

  const is3d = dimensions === 3;
  // _refine_3d thresholds on the median over measured cells; the 2D path additionally
  // excludes retained coarse blocks (span > 1), which do not exist in a 3D refine.
  // Mirroring computeRealMedian in 3D predicted a different cell set than the backend.
  const median = is3d ? computeMeasuredMedian(cells) : computeRealMedian(cells);
  const threshold = median * tau;
  const tauSelected = new Set<string>();
  cells.forEach(c => {
    if (c.span === 1 && c.sensitivity !== null && c.sensitivity! >= threshold)
      // in 3D the cell list is the flattened gs^3 grid; keying on row,col alone
      // collapsed every z-column onto one key and undercounted by up to gs×.
      tauSelected.add(is3d ? `${c.row},${c.col},${c.depth}` : `${c.row},${c.col}`);
  });
  // Union of tau + manual. 3D has no manual-selection path at all (right-click lives
  // in UnifiedViewport, and _refine_3d ignores extra_positions), so it is tau-only.
  const combined = is3d ? tauSelected : new Set([...tauSelected, ...manualSelection]);
  const aboveCount = combined.size;
  const manualOnly = is3d ? 0 : [...manualSelection].filter(k => !tauSelected.has(k)).length;
  const newGs = currentGridSize * multiplier;
  // each selected cell is split along every axis: mult^2 sub-cells in 2D, mult^3 in 3D
  const newCells = aboveCount * multiplier ** (is3d ? 3 : 2);

  return (
    <div style={{ display: 'flex', gap: 12, padding: '6px 12px', background: '#1a1a2e',
                  borderTop: '1px solid #333', alignItems: 'center', flexWrap: 'wrap' }}>
      <div>
        <label style={{ fontSize: 10, color: '#888' }}>tau={tau.toFixed(2)}</label>
        <input type="range" min={0} max={4} step={0.05} value={tau}
               onChange={e => setTau(+e.target.value)} style={{ display: 'block', width: 140 }} />
      </div>
      <div>
        <label style={{ fontSize: 10, color: '#888' }}>
          {multiplier}x → {newGs}x{newGs}{is3d ? `x${newGs}` : ''}
        </label>
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
        <button onClick={() => setShowSurface(s => !s)}
                style={{ padding: '2px 10px', border: 'none', borderRadius: 3, fontSize: 10,
                         background: showSurface ? '#e94560' : '#333',
                         color: showSurface ? '#fff' : '#888', cursor: 'pointer',
                         opacity: meshData ? 1 : 0.4 }}>
          {showSurface ? 'surface ON' : 'surface OFF'}
        </button>
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

/** Median over strictly positive sensitivities, matching _refine_3d's threshold rule. */
function computeMeasuredMedian(cells: { sensitivity: number | null }[]): number {
  // mirrors ridge_detector.measured_mask: 0 is the unmeasured sentinel, and a cell that
  // came back at ~-1e-7 (near-identical neighbours, float error) is a real measurement
  const s = cells.filter(c => c.sensitivity !== null && c.sensitivity !== 0).map(c => c.sensitivity!);
  if (s.length === 0) return 0;
  return [...s].sort((a, b) => a - b)[Math.floor(s.length / 2)];
}

/** Median over real (span=1), measured (non-zero) cells — matches refine_grid. */
function computeRealMedian(cells: { sensitivity: number | null; span: number }[]): number {
  const real = cells.filter(c => c.span === 1 && c.sensitivity).map(c => c.sensitivity!);
  if (real.length === 0) {
    // Fallback to all measured cells
    const all = cells.filter(c => c.sensitivity).map(c => c.sensitivity!);
    if (all.length === 0) return 0;
    return [...all].sort((a, b) => a - b)[Math.floor(all.length / 2)];
  }
  return [...real].sort((a, b) => a - b)[Math.floor(real.length / 2)];
}

type LayerToggle = { images: boolean; heatmap: boolean; tau: boolean };

const ExportOverlayContext = { show: (_msg: string) => {}, hide: () => {} };

function ExportOverlay() {
  const [msg, setMsg] = useState<string | null>(null);
  ExportOverlayContext.show = (m: string) => setMsg(m);
  ExportOverlayContext.hide = () => setMsg(null);
  if (!msg) return null;
  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 2000,
    }}>
      <div style={{ background: '#1a1a2e', padding: '24px 48px', borderRadius: 12,
                    border: '1px solid #333', textAlign: 'center' }}>
        <div style={{ fontSize: 14, color: '#e94560', marginBottom: 8 }}>{msg}</div>
        <div style={{ width: 40, height: 40, margin: '0 auto',
                      border: '3px solid #333', borderTop: '3px solid #e94560',
                      borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    </div>
  );
}

function ExportButton({ jobId, layer }: { jobId: string; layer: string }) {
  const [state, setState] = useState<'idle' | 'loading' | 'done' | 'error'>('idle');

  const fail = () => {
    ExportOverlayContext.hide();
    setState('error');
    setTimeout(() => setState('idle'), 2000);
  };

  const handleClick = async () => {
    setState('loading');
    ExportOverlayContext.show(`Exporting ${layer}...`);
    try {
      const resp = await fetch(`/api/grid/${jobId}/export/${layer}.jpg`);
      // the endpoint answers HTTP 200 with a JSON {"error": ...} body for a job the
      // backend no longer has, so resp.ok alone downloaded that JSON as a .jpg and
      // showed the green success tick.
      if (!resp.ok || !(resp.headers.get('content-type') || '').startsWith('image/')) {
        fail();
        return;
      }
      const blob = await resp.blob();
      ExportOverlayContext.hide();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `ridge_${layer}_${jobId}.jpg`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      setState('done');
      setTimeout(() => setState('idle'), 2000);
    } catch {
      fail();
    }
  };

  return (
    <button onClick={handleClick} disabled={state === 'loading'}
            title={state === 'error' ? 'export failed — the server has no data for this job' : layer}
            style={{ padding: '2px 8px', border: 'none', borderRadius: 3, fontSize: 10,
                     background: state === 'loading' ? '#e94560' : state === 'done' ? '#4ecca3'
                                 : state === 'error' ? '#5a1a2a' : '#0f3460',
                     color: state === 'loading' ? '#fff' : state === 'done' ? '#000'
                            : state === 'error' ? '#ff8a9c' : '#aaa',
                     cursor: state === 'loading' ? 'wait' : 'pointer',
                     transition: 'background 0.3s' }}>
      {state === 'loading' ? `${layer}...` : state === 'done' ? `${layer} ✓`
        : state === 'error' ? `${layer} ✗` : layer}
    </button>
  );
}

// ------------------------------------------------- Crossing direction (JVP probe)
// Shown inside the per-cell detail overlay. The measurement is local and takes ~25 s,
// so it is always an explicit click — never started by opening a cell.
function CrossingProbeBlock({ alpha, beta }: { alpha: number; beta: number }) {
  const jobId = useRidgeStore(s => s.jobId);
  const seed = useRidgeStore(s => s.seed);
  const probes = useProbeStore(s => s.probes);
  const pendingAt = useProbeStore(s => s.pendingAt);
  const jvpStatus = useProbeStore(s => s.jvpStatus);
  const jvpError = useProbeStore(s => s.jvpError);
  const startJvp = useProbeStore(s => s.startJvp);

  const here = probeAt(probes, alpha, beta);
  const runningHere = jvpStatus === 'running' && !!pendingAt
    && Math.abs(pendingAt.alpha - alpha) < 1e-6 && Math.abs(pendingAt.beta - beta) < 1e-6;
  const busy = jvpStatus === 'running';
  // The grid extends the simplex affinely — cells with alpha+beta > 1 carry a NEGATIVE
  // weight on prompt A. The probe serves them (the mixing is affine there, see
  // gpu_pool._nlerp), but the seed-invariance evidence was gathered on the simplex
  // proper, so say that the reading is less tested rather than hide the button.
  const outside = alpha + beta > 1 + 1e-6;

  return (
    <div style={{ marginTop: 12, padding: 12, background: '#1a1a2e', borderRadius: 8,
                  textAlign: 'left', maxWidth: 560, marginLeft: 'auto', marginRight: 'auto' }}>
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
        <button className="rx-focus"
                disabled={busy || !jobId}
                title={outside ? 'outside the prompt simplex (prompt A carries a negative weight): '
                               + 'the map is affine here, but the direction is less tested' : undefined}
                onClick={() => { if (jobId) startJvp(jobId, alpha, beta, seed); }}
                style={{ padding: '4px 14px', borderRadius: 4, fontSize: 11,
                         background: busy ? '#333' : '#0f3460',
                         color: busy ? '#777' : '#cfd',
                         border: `1px solid ${busy ? '#333' : ACCENT}`,
                         cursor: busy || !jobId ? 'not-allowed' : 'pointer' }}>
          {runningHere ? 'probing…' : here ? 'Probe again (~25 s)' : 'Probe crossing direction (~25 s)'}
        </button>
        {outside && (
          <span style={{ fontSize: 11, color: '#888' }}>
            outside the prompt simplex (α+β &gt; 1): affine extension of the map, less tested
          </span>
        )}
        {runningHere && (
          <span style={{ fontSize: 11, color: ACCENT }}>
            <span className="rx-spin" style={{ display: 'inline-block', marginRight: 6 }}>◐</span>
            measuring the direction the image changes fastest…
          </span>
        )}
        {busy && !runningHere && (
          <span style={{ fontSize: 11, color: '#888' }}>another probe is running…</span>
        )}
        {jvpStatus === 'error' && jvpError && !busy && (
          <span style={{ fontSize: 11, color: '#ff8a9c' }}>{jvpError}</span>
        )}
      </div>

      {here && (
        <div style={{ marginTop: 8, fontSize: 12, color: '#ddd', lineHeight: 1.5 }}>
          <div style={{ color: ACCENT }}>{here.result.normal_reading}</div>
          <div style={{ fontSize: 11, color: '#9ab', marginTop: 4 }}>
            rank-1 share {here.result.rank1_share.toFixed(2)} ·{' '}
            participation ratio {here.result.participation_ratio.toFixed(2)}{' '}
            <span style={{ color: '#667' }}>
              ({here.result.participation_ratio >= 1.5
                ? 'a corner — two fronts meet here, so one direction is a poor summary'
                : 'a single front'})
            </span>
          </div>
          <div style={{ fontSize: 10, color: '#777', marginTop: 4 }}>
            Direction is seed-invariant; magnitude comparable only within this job.
          </div>
          <div style={{ fontSize: 10, color: '#555', marginTop: 2 }}>
            drawn on the map as a line through this cell (no arrow: the direction has no sign)
          </div>
        </div>
      )}
    </div>
  );
}

function UnifiedViewport() {
  const { phase, cells, currentGridSize, tau, seeds, seedCells, activeSeedIdx, setActiveSeedIdx,
          manualSelection, toggleManualCell, clearManualSelection, jobId } = useRidgeStore();

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
  const [probeSteps, setProbeSteps] = useState(10);
  const [probeImages, setProbeImages] = useState<{ seeds: number[]; images: (string | null)[] } | null>(null);
  const [probeLoading, setProbeLoading] = useState(false);
  const [probeError, setProbeError] = useState<string | null>(null);
  // the poll handle used to live only in the launch closure, so nothing but a
  // 'complete' status could ever stop it — not closing the overlay, not unmounting
  const probePoll = useRef<ReturnType<typeof setInterval> | null>(null);
  const stopProbePoll = useCallback(() => {
    if (probePoll.current) { clearInterval(probePoll.current); probePoll.current = null; }
  }, []);
  useEffect(() => stopProbePoll, [stopProbePoll]);

  // Pan/zoom state
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const isPanning = useRef(false);
  const lastMouse = useRef({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  // --- JVP probes: measured crossing directions drawn over the grid ----------
  const probes = useProbeStore(s => s.probes);
  const pendingAt = useProbeStore(s => s.pendingAt);
  const jvpStatus = useProbeStore(s => s.jvpStatus);
  const clearProbes = useProbeStore(s => s.clearProbes);
  const syncProbeJob = useProbeStore(s => s.syncJob);
  // A probe is measured at (alpha, beta) of ONE prompt simplex. Carrying the list
  // across a new job would draw segments at coordinates that now mean something else,
  // so the probe store follows this store's job id (including to null on cancel).
  useEffect(() => { syncProbeJob(jobId); }, [jobId, syncProbeJob]);
  // Probe segments are sized as a fraction of the VIEWPORT, not of the grid, so they
  // stay readable at every zoom — which needs the container's pixel size. Keyed on
  // `phase` because the container does not exist while the app is idle, so a mount-only
  // effect would attach the observer to nothing.
  const [viewSize, setViewSize] = useState({ w: 0, h: 0 });
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const update = () => {
      const r = el.getBoundingClientRect();
      setViewSize(s => (Math.abs(s.w - r.width) < 0.5 && Math.abs(s.h - r.height) < 0.5)
        ? s : { w: r.width, h: r.height });
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [phase]);

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

  // Recenter: fit the full grid in the viewport. Returns false if it could not run,
  // so a pending centering is not consumed by a call that did nothing.
  const recenter = useCallback(() => {
    const el = containerRef.current;
    if (!el) return false;
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return false;
    const gs = currentGridSize;
    // gs is 0 whenever the backend reports a grid it no longer has; dividing by it
    // made fitZoom Infinity and pan NaN, and the resulting transform is invalid CSS
    // so the browser drops it and the viewport freezes.
    if (gs <= 0) return false;
    const ts = Math.max(4, Math.min(64, Math.floor(800 / gs)));
    const gpx = gs * ts;
    // Fit: scale so the grid fills the viewport with some padding
    const fitZoom = Math.min(rect.width / gpx, rect.height / gpx) * 0.95;
    setZoom(fitZoom);
    setPan({ x: (rect.width - gpx * fitZoom) / 2, y: (rect.height - gpx * fitZoom) / 2 });
    return true;
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
      if (recenter()) needsCenter.current = false;
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

  // (alpha, beta) in [0,1]² -> screen px, using the same projection the cells use:
  // alpha indexes columns left→right, beta indexes rows but is drawn flipped
  // (displayRow = gs-1-betaIdx), so beta grows UPWARD on screen.
  const toScreen = (alpha: number, beta: number) => {
    const worldX = (alpha * (gs - 1) + 0.5) * ts;
    const worldY = ((gs - 1) * (1 - beta) + 0.5) * ts;
    return { x: pan.x + worldX * zoom, y: pan.y + worldY * zoom };
  };
  // ~10% of the viewport's short side, with a floor so a probe is still visible in a
  // small pane.
  const segLen = Math.max(28, 0.10 * Math.min(viewSize.w, viewSize.h));

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
        {probes.length > 0 && (
          <button className="rx-focus" onClick={clearProbes}
                  title="remove the measured crossing directions from the map"
                  style={{ padding: '2px 8px', borderRadius: 3, fontSize: 10,
                           background: '#0f3460', color: ACCENT,
                           border: `1px solid ${ACCENT}`, cursor: 'pointer', marginLeft: 4 }}>
            clear {probes.length} probe{probes.length > 1 ? 's' : ''}
          </button>
        )}
        {jobId && (
          <>
            <span style={{ marginLeft: 'auto' }} />
            {(['images', 'heatmap', 'overlay'] as const).map(layer => (
              <ExportButton key={layer} jobId={jobId} layer={layer} />
            ))}
          </>
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

        {/* Probe overlay: one sign-free segment per measured crossing direction.
            It is a child of the pan/zoom CONTAINER, not of the transformed grid div,
            and projects its own coordinates: inside the transform, stroke width,
            label size and segment length would all scale with zoom, so a 2.5px line
            would either vanish or swallow the grid. pointer-events: none keeps
            drag/zoom/double-click behaviour on the grid underneath unchanged. */}
        {(probes.length > 0 || (pendingAt && jvpStatus === 'running')) && viewSize.w > 0 && (
          <svg width={viewSize.w} height={viewSize.h}
               style={{ position: 'absolute', left: 0, top: 0, pointerEvents: 'none' }}>
            {probes.map((p) => {
              const th = p.result.normal_theta;
              if (!th || th.length < 2) return null;
              // screen: +alpha is right, +beta is UP (the grid draws beta flipped)
              let dx = th[0], dy = -th[1];
              const n = Math.hypot(dx, dy);
              if (!(n > 1e-9)) return null;
              dx /= n; dy /= n;
              const { x, y } = toScreen(p.alpha, p.beta);
              if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
              const hx = (dx * segLen) / 2, hy = (dy * segLen) / 2;
              return (
                <g key={`probe-${p.alpha.toFixed(6)}-${p.beta.toFixed(6)}`}>
                  {/* halo first, then the accent stroke on top of it */}
                  <line x1={x - hx} y1={y - hy} x2={x + hx} y2={y + hy}
                        stroke={HALO} strokeWidth={4.5} strokeLinecap="round" opacity={0.85} />
                  <line x1={x - hx} y1={y - hy} x2={x + hx} y2={y + hy}
                        stroke={ACCENT} strokeWidth={2.5} strokeLinecap="round" />
                  <circle cx={x} cy={y} r={3.5} fill={ACCENT} stroke={HALO} strokeWidth={1} />
                  <text x={x + 7} y={y - 7} fontSize={11} fill={ACCENT}
                        stroke={HALO} strokeWidth={3} paintOrder="stroke"
                        style={{ fontFamily: 'system-ui' }}>
                    {probeLabel(p.result)}
                  </text>
                </g>
              );
            })}
            {pendingAt && jvpStatus === 'running' && (() => {
              const { x, y } = toScreen(pendingAt.alpha, pendingAt.beta);
              if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
              return (
                <g>
                  <circle cx={x} cy={y} r={9} fill="none" stroke={ACCENT} strokeWidth={1.5}
                          strokeDasharray="3 3" opacity={0.9} />
                  <text x={x + 13} y={y + 4} fontSize={10} fill={ACCENT}
                        stroke={HALO} strokeWidth={3} paintOrder="stroke"
                        style={{ fontFamily: 'system-ui' }}>
                    probing…
                  </text>
                </g>
              );
            })()}
          </svg>
        )}
      </div>

      {/* Full-res image overlay with seed probe */}
      {overlayImg && (
        <div onClick={() => { stopProbePoll(); setProbeLoading(false); setProbeError(null);
                             setOverlayImg(null); setProbeImages(null); }}
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

            {/* Which way does the image change here? (~25 s JVP probe) */}
            <CrossingProbeBlock alpha={overlayImg.alpha} beta={overlayImg.beta} />

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
              <span style={{ fontSize: 12, color: '#888', marginLeft: 8 }}>Steps:</span>
              <input type="number" value={probeSteps} min={1} max={50}
                     onChange={e => setProbeSteps(Math.max(1, +e.target.value))}
                     style={{ width: 40, padding: 4, background: '#0f3460', border: '1px solid #333',
                              color: '#fff', borderRadius: 4, fontSize: 11 }} />
              <button disabled={probeLoading}
                      onClick={async () => {
                        const { jobId } = useRidgeStore.getState();
                        if (!jobId) return;
                        stopProbePoll();
                        setProbeLoading(true);
                        setProbeImages(null);
                        setProbeError(null);
                        try {
                          const res = await startSeedProbe(jobId, {
                            alpha: overlayImg.alpha, beta: overlayImg.beta,
                            seed_start: probeSeedStart, seed_end: probeSeedEnd,
                            steps: probeSteps,
                          });
                          // the endpoint answers HTTP 200 with status 'error' when the
                          // backend no longer has the job (and 422 with no status at all
                          // for a bad payload); with no else the button stayed disabled
                          // and read 'Generating...' for the rest of the session.
                          if (res.status !== 'running' || !res.probe_id) {
                            setProbeError(res.status === 'error'
                              ? 'probe failed — the server no longer has this job'
                              : `probe failed (${res.status ?? 'bad response'})`);
                            setProbeLoading(false);
                            return;
                          }
                          // Poll for completion. Bounded: a probe whose GPU tasks were
                          // drained never reports complete, and a seed that errors never
                          // gets a thumbnail, so an unbounded poll runs forever.
                          let attempts = 0;
                          probePoll.current = setInterval(async () => {
                            attempts++;
                            try {
                              const st = await getSeedProbeStatus(res.probe_id);
                              setProbeImages({ seeds: st.seeds, images: st.images });
                              if (st.complete) {
                                stopProbePoll();
                                setProbeLoading(false);
                                return;
                              }
                            } catch (err) {
                              // transient failure: keep polling, but the attempt counted
                              console.error('[Ridge] Probe poll error:', err);
                            }
                            if (attempts >= 600) {
                              stopProbePoll();
                              setProbeLoading(false);
                              setProbeError('probe stalled — stopped polling');
                            }
                          }, 1000);
                        } catch (err) {
                          console.error(err);
                          setProbeError(String(err));
                          setProbeLoading(false);
                        }
                      }}
                      style={{ padding: '4px 14px', background: probeLoading ? '#555' : '#e94560',
                               color: '#fff', border: 'none', borderRadius: 4, fontSize: 11,
                               cursor: probeLoading ? 'not-allowed' : 'pointer' }}>
                {probeLoading ? 'Generating...' : 'Generate'}
              </button>
              {probeError && (
                <span style={{ fontSize: 11, color: '#e94560' }}>{probeError}</span>
              )}
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


// ---------------------------------------------------------------- Itinerary
// The ridge as a set of walkable arcs rather than a mask. Rebuilding the graph is
// CPU-only (it reads the stored sensitivity field and DINOv2 embeddings), so the
// K and persistence sliders are interactive — no regeneration.
function ItineraryPanel() {
  const jobId = useRidgeStore((s) => s.jobId);
  const phase = useRidgeStore((s) => s.phase);
  const [k, setK] = useState(8);
  const [hFrac, setHFrac] = useState(0.10);
  const [graph, setGraph] = useState<any>(null);
  const [sel, setSel] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // bumped on every successful build so the filmstrip <img> cannot show the previous
  // build's cached render under the new arc's id
  const [build_v, setBuildV] = useState(0);

  // nothing reset these on a new grid, so a fresh job opened showing the previous
  // job's arcs and its filmstrip
  useEffect(() => { setGraph(null); setSel(null); setErr(null); }, [jobId]);

  const build = useCallback(async () => {
    if (!jobId) return;
    setBusy(true); setErr(null);
    try {
      const r: any = await api.buildRidgeGraph(jobId, { k, h_frac: hFrac });
      if (r?.params?.error) { setErr(String(r.params.error)); setGraph(null); setSel(null); }
      else {
        const g = await api.getRidgeGraph(jobId);
        setGraph(g); setSel(g?.edges?.[0]?.id ?? null); setBuildV((v) => v + 1);
        if (r?.params?.note) setErr(String(r.params.note));
      }
    } catch (e: any) { setErr(String(e)); }
    setBusy(false);
  }, [jobId, k, hFrac]);

  if (!jobId || phase === 'idle') return null;
  const arc = graph?.edges?.find((e: any) => e.id === sel);

  return (
    <div style={{ padding: '6px 12px', background: '#141c33', borderBottom: '1px solid #333',
                  display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
        <b style={{ fontSize: 12 }}>Ridge itinerary</b>
        <label style={{ fontSize: 11 }}>basins K
          <input type="range" min={3} max={16} value={k} onChange={(e) => setK(+e.target.value)}
                 style={{ width: 90, marginLeft: 6, verticalAlign: 'middle' }} />
          <span style={{ marginLeft: 4, color: '#4ecca3' }}>{k}</span>
        </label>
        <label style={{ fontSize: 11 }}>persistence h
          <input type="range" min={2} max={40} value={Math.round(hFrac * 100)}
                 onChange={(e) => setHFrac(+e.target.value / 100)}
                 style={{ width: 90, marginLeft: 6, verticalAlign: 'middle' }} />
          <span style={{ marginLeft: 4, color: '#4ecca3' }}>{hFrac.toFixed(2)}</span>
        </label>
        <button onClick={build} disabled={busy}
                style={{ padding: '3px 10px', background: busy ? '#333' : '#e94560', color: '#fff',
                         border: 'none', borderRadius: 4, fontSize: 11, cursor: 'pointer' }}>
          {busy ? 'building…' : graph ? 'rebuild' : 'extract arcs'}
        </button>
        {graph && (
          <span style={{ fontSize: 11, color: '#888' }}>
            {graph.n_arcs} arcs · {graph.n_basins} basins · {graph.junctions.length} junctions
          </span>
        )}
        {err && <span style={{ fontSize: 11, color: '#e94560' }}>{err}</span>}
      </div>

      {graph && graph.edges.length > 0 && (
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {graph.edges.map((e: any) => (
            <button key={e.id} onClick={() => setSel(e.id)}
              title={`arc ${e.id}: ${e.length} stations, ${e.s_percentile?.toFixed(0)}th pct of S`}
              style={{ padding: '2px 8px', fontSize: 11, borderRadius: 4, cursor: 'pointer',
                       border: sel === e.id ? '1px solid #4ecca3' : '1px solid #444',
                       background: sel === e.id ? '#20304d' : '#1a1a2e', color: '#ddd' }}>
              {e.itinerary.map((p: number[]) => `[${p[0]}|${p[1]}]`).join(' → ')}
              <span style={{ color: '#777', marginLeft: 6 }}>{e.length}</span>
            </button>
          ))}
        </div>
      )}

      {arc && (
        <div>
          <div style={{ fontSize: 11, color: '#999', marginBottom: 3 }}>
            arc {arc.id}: {arc.length} stations · flanking basins{' '}
            {arc.itinerary.map((p: number[]) => `${p[0]} | ${p[1]}`).join('  →  ')} ·{' '}
            {arc.s_percentile?.toFixed(0)}th percentile of sensitivity
          </div>
          <img key={`${jobId}-${arc.id}-${build_v}`}
               src={`${api.itineraryUrl(jobId, arc.id)}${api.itineraryUrl(jobId, arc.id).includes('?') ? '&' : '?'}v=${build_v}`}
               alt={`itinerary ${arc.id}`}
               onError={(e) => { (e.target as HTMLImageElement).style.visibility = 'hidden'; }}
               onLoad={(e) => { (e.target as HTMLImageElement).style.visibility = 'visible'; }}
               style={{ maxWidth: '100%', borderRadius: 4, background: '#000' }} />
        </div>
      )}
    </div>
  );
}



// ------------------------------------------------------ animated hike journey
//
// A hiker is a CHAIN across triangles, not a row per hop. `chain` in the status is the
// index within that hop's live population and is not stable between hops; `lineage` is
// (root, then ">i,j" appended at each exit). Grouping by lineage reconstructs each
// hiker's whole journey, so it plays as one continuous filmstrip and the hand-off
// between triangles is visible: the last frame of a hop IS the exit, and that exit
// becomes the first vertex of the next triangle.
function parentLineage(l: string): string | null {
  if (!l || l === 'root') return null;
  const i = l.lastIndexOf('>');
  return i <= 0 ? 'root' : l.slice(0, i);
}

export function journeys(rows: any[]): any[][] {
  const walked = rows.filter((r) => (r.stations?.length ?? 0) > 0);
  const byLin = new Map<string, any>();
  for (const r of walked) if (!byLin.has(r.lineage ?? 'root')) byLin.set(r.lineage ?? 'root', r);
  const isParent = new Set<string>();
  for (const l of byLin.keys()) { const p = parentLineage(l); if (p) isParent.add(p); }
  const out: any[][] = [];
  for (const [l, r] of byLin) {
    if (isParent.has(l)) continue;           // not a leaf: a longer journey covers it
    const path = [r];
    let cur = parentLineage(l);
    while (cur && byLin.has(cur)) { path.unshift(byLin.get(cur)); cur = parentLineage(cur); }
    out.push(path);
  }
  out.sort((a, b) => (b.length - a.length) || (b[b.length - 1].hop - a[a.length - 1].hop));
  return out;
}


// Lineage strings are exit coordinates (">7,0>0,9>0,6") -- exact, and unreadable. Name
// each hiker by its position in the branch tree instead: the first split gives A, B, C;
// a split under A gives A1, A2; under A1 gives A1a, A1b. Siblings are ordered by their
// exit coordinate so a name is stable across polls rather than jumping as rows arrive.
const TIERS = ['ABCDEFGH', '12345678', 'abcdefgh'];
// one hue per top-level branch, so siblings read as a family at a glance
const BRANCH_HUES = ['#4ecca3', '#5aa9e6', '#e9a145', '#c77dff', '#f4737f', '#7fd1ae'];
export function branchColour(name: string): string {
  const i = name ? (name.charCodeAt(0) - 65) : 0;
  return BRANCH_HUES[((i % BRANCH_HUES.length) + BRANCH_HUES.length) % BRANCH_HUES.length];
}

export function lineageNames(rows: any[]): Map<string, string> {
  // Name from the SAME rows that get drawn. A hop that produced no arc has no stations,
  // so it is not a node in LineageTree and not a journey — but while it was still counted
  // here it took a sibling slot, and the names shifted: with one arcless sibling at hop 1
  // the surviving hiker was "B" (blue) on its journey card and "A" (green) in the tree,
  // for the same lineage. LineageTree already passes the walked subset, so filtering here
  // is a no-op for it and makes every other caller agree with it by construction.
  const walked = rows.filter((r) => (r.stations?.length ?? 0) > 0);
  const lins = new Set<string>();
  for (const r of walked) {
    let l = r.lineage ?? 'root';
    while (l && l !== 'root') { lins.add(l); l = parentLineage(l); }
  }
  // Same trap as in LineageTree: a null parent must NOT collapse onto 'root', or root
  // becomes its own child and walk() below recurses forever.
  const kids = new Map<string, string[]>();
  for (const l of lins) {
    const p = parentLineage(l);
    if (p === null || p === l) continue;
    if (!kids.has(p)) kids.set(p, []);
    kids.get(p)!.push(l);
  }
  for (const v of kids.values()) v.sort();
  const name = new Map<string, string>([['root', '']]);
  const walk = (node: string, depth: number) => {
    const cs = kids.get(node) ?? [];
    cs.forEach((c, i) => {
      const tier = TIERS[Math.min(depth, TIERS.length - 1)];
      // a lone child continues its parent's name: it is the same hiker, not a branch
      name.set(c, cs.length === 1 ? (name.get(node) || 'A')
                                  : (name.get(node) ?? '') + tier[i % tier.length]);
      walk(c, depth + 1);
    });
  };
  walk('root', 0);
  return name;
}

// Where did this branch split away from its neighbours, and is it still alive?
export function branchInfo(paths: any[][], maxHop: number) {
  return paths.map((p) => {
    const leaf = p[p.length - 1];
    return {
      leaf,
      alive: leaf.hop >= maxHop,
      // index of the first triangle this journey does not share with the one before it
      forkAt: (other: any[]) => {
        let k = 0;
        while (k < p.length && k < other.length &&
               (p[k].lineage ?? 'root') === (other[k].lineage ?? 'root')) k++;
        return k;
      },
    };
  });
}


// A layered view of the branch topology. Every one of these trees is tiny -- nodes is
// bounded by `budget` and width by `beam` (measured over 27 saved hikes: max 9 nodes,
// max width 4, max depth 8) -- so depth IS the hop number and the layout is a fixed
// grid, not a graph problem. Leaves take consecutive slots in DFS order and a parent
// sits at the mean of its children, which is exact and deterministic for trees this size.
//
// It is a NAVIGATOR, not a replacement for the journeys: it answers "what is the
// population doing" (e.g. all three survivors descending from one lineage means the beam
// collapsed), while the journeys answer "what did it find". It hides itself when there
// is no branching, because a beam-1 run is a line and a line shows nothing.
export function LineageTree({ hikeId, rows, vertical, selected, onSelect, node = 38 }:
    { hikeId: string; rows: any[]; vertical?: boolean; node?: number;
      selected: string | null; onSelect: (lineage: string) => void }) {
  const walked = rows.filter((r) => (r.stations?.length ?? 0) > 0);
  if (walked.length < 2) return null;
  const names = lineageNames(walked);
  const byLin = new Map<string, any>();
  for (const r of walked) if (!byLin.has(r.lineage ?? 'root')) byLin.set(r.lineage ?? 'root', r);

  const kids = new Map<string, string[]>();
  for (const l of byLin.keys()) {
    const p = parentLineage(l);
    // `?? 'root'` here made root its OWN child, because parentLineage('root') is null.
    // place() then recursed into itself forever and took the whole UI down with a
    // RangeError the moment hop 0 was on screen.
    if (p === null || p === l) continue;
    if (!kids.has(p)) kids.set(p, []);
    kids.get(p)!.push(l);
  }
  for (const v of kids.values()) v.sort();
  // The seed triangle is a node too. Rooting the layout at whichever lineages have no
  // parent IN the data keeps it correct whether or not hop 0 was recorded.
  const roots = byLin.has('root') ? ['root']
    : [...byLin.keys()].filter((l) => !byLin.has(parentLineage(l) ?? ''));
  const maxHop = Math.max(0, ...walked.map((r) => r.hop ?? 0));
  // Count nodes per depth with a Map, not a sparse array: the array version left holes
  // wherever a depth was empty, and spreading a hole into Math.max yields undefined ->
  // NaN, so the "is there any branching" test silently stopped working.
  const perDepth = new Map<number, number>();
  for (const l of byLin.keys()) {
    const d = byLin.get(l)?.hop ?? 0;
    perDepth.set(d, (perDepth.get(d) ?? 0) + 1);
  }
  const width = Math.max(1, ...perDepth.values());
  if (width < 2) return null;                       // a line: nothing to draw

  // tidy layout: leaves get consecutive slots, parents centre on their children
  const slot = new Map<string, number>();
  let next = 0;
  // `seen` is belt-and-braces: the parent map is now acyclic by construction, but a
  // layout routine must never be able to hang the interface.
  const seen = new Set<string>();
  const place = (l: string): number => {
    if (seen.has(l)) return slot.get(l) ?? 0;
    seen.add(l);
    const cs = (kids.get(l) ?? []).filter((c) => byLin.has(c) && c !== l);
    if (!cs.length) { slot.set(l, next); return next++; }
    const ys = cs.map((c) => place(c));
    const y = (Math.min(...ys) + Math.max(...ys)) / 2;
    slot.set(l, y);
    return y;
  };
  roots.forEach((l) => place(l));

  const NODE = node, GAP_D = Math.round(node * 0.8), GAP_S = Math.round(node * 0.32);
  // depth is the hop number, straight from the row -- deriving it from the lineage
  // string put 'root' at -1 and rendered the seed triangle off-canvas
  const depthOf = (l: string) => byLin.get(l)?.hop ?? 0;
  const minHop = Math.min(...[...byLin.keys()].map(depthOf));
  const pos = (l: string) => {
    const d = depthOf(l), sl = slot.get(l) ?? 0;
    const along = (d - minHop) * (NODE + GAP_D), across = sl * (NODE + GAP_S);
    return vertical ? { x: across, y: along } : { x: along, y: across };
  };
  const span = { along: (maxHop - minHop) * (NODE + GAP_D) + NODE,
                 across: Math.max(1, next) * (NODE + GAP_S) };
  const W = vertical ? span.across : span.along;
  const H = vertical ? span.along : span.across;

  const edges: JSX.Element[] = [];
  for (const l of byLin.keys()) {
    const p = parentLineage(l);
    if (!p || !byLin.has(p)) continue;
    const a = pos(p), b = pos(l);
    edges.push(<line key={`e${l}`} x1={a.x + NODE / 2} y1={a.y + NODE / 2}
                     x2={b.x + NODE / 2} y2={b.y + NODE / 2}
                     stroke={branchColour(names.get(l) ?? '')} strokeWidth={2}
                     strokeOpacity={0.55} />);
  }

  return (
    <div style={{ position: 'relative', width: W, height: H, margin: '2px 0 10px' }}>
      <svg width={W} height={H} style={{ position: 'absolute', inset: 0 }}>{edges}</svg>
      {[...byLin.keys()].map((l) => {
        const r = byLin.get(l), { x, y } = pos(l);
        const nm = names.get(l) ?? '';
        const isLeaf = !(kids.get(l) ?? []).some((c) => byLin.has(c));
        const pruned = isLeaf && (r.hop ?? 0) < maxHop;
        const on = selected === l;
        return (
          <div key={l} onClick={() => onSelect(l)}
               title={`${nm || 'A'} · hop ${r.hop}${pruned ? ' · pruned' : ''}`}
               style={{ position: 'absolute', left: x, top: y, width: NODE, height: NODE,
                        borderRadius: 6, overflow: 'hidden', cursor: 'pointer',
                        boxShadow: on ? `0 0 0 2px #fff` : 'none',
                        border: `2px solid ${branchColour(nm)}`,
                        opacity: pruned ? 0.5 : 1, background: '#000' }}>
            <img src={api.hikeMapUrl(hikeId, r.chain, r.hop)} alt={nm}
                 onError={(e) => { (e.target as HTMLImageElement).style.visibility = 'hidden'; }}
                 style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
            {isLeaf && (
              <span style={{ position: 'absolute', left: 0, bottom: 0, fontSize: 9,
                             padding: '0 3px', background: 'rgba(0,0,0,.72)',
                             color: branchColour(nm), fontWeight: 700 }}>
                {nm || 'A'}{pruned ? '✕' : ''}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

function HikeJourney({ hikeId, path, playing, speed, onBranch, branching }:
                     { hikeId: string; path: any[]; playing: boolean; speed: number;
                       onBranch: (row: any, station: number) => void; branching: boolean }) {
  // one flat sequence of (row, stationIndex) across every triangle this hiker crossed
  const seq: { r: any; s: [number, number]; k: number; last: boolean;
               approach: boolean }[] = [];
  for (const r of path) {
    const st: [number, number][] = r.stations ?? [];
    const ap = r.approach ?? 0;
    st.forEach((p, k) => seq.push({ r, s: p, k, last: k === st.length - 1,
                                    approach: k < ap }));
  }
  const [n, setN] = useState(0);
  const [box, setBox] = useState({ w: 0, h: 0 });
  // Which noise draw the filmstrip shows. A multi-seed hike walks the AVERAGED
  // sensitivity field, so every seed's strip follows the identical route — switching is a
  // controlled comparison of what the same itinerary looks like under a different draw,
  // not a different hike. Index, not seed value; the row carries the values for the label.
  const [seedIdx, setSeedIdx] = useState(0);
  const nSeeds = Math.max(1, ...path.map((r: any) => r.n_seeds ?? 1));
  const seedVals: number[] = path.find((r: any) => r.seeds?.length)?.seeds ?? [];
  useEffect(() => { if (seedIdx >= nSeeds) setSeedIdx(0); }, [nSeeds, seedIdx]);
  // Animate only a forward step of one. Looping from the last station back to 0, and a
  // scrub jump, are POSITION CHANGES, not travel: transitioning them made the strip race
  // backwards through every frame and the map marker fly across the simplex.
  // a user scrub pauses this journey; auto-play only resumes when they release the pin
  const [pinned, setPinned] = useState(false);
  const [drag, setDrag] = useState(false);
  const strip = useRef<HTMLDivElement | null>(null);
  useEffect(() => { setN(0); }, [path.length, seq.length]);
  useEffect(() => {
    if (!playing || pinned || drag || seq.length < 2) return;
    const t = window.setInterval(() => setN((p) => (p + 1) % seq.length), speed);
    return () => window.clearInterval(t);
  }, [playing, pinned, drag, speed, seq.length]);

  // map a pointer x within the strip onto a frame index
  const scrubTo = useCallback((clientX: number) => {
    const el = strip.current; if (!el) return;
    const r = el.getBoundingClientRect();
    const f = Math.min(1, Math.max(0, (clientX - r.left) / Math.max(r.width, 1)));
    setN(Math.min(seq.length - 1, Math.floor(f * seq.length)));
  }, [seq.length]);
  useEffect(() => {
    if (!drag) return;
    const mv = (e: PointerEvent) => scrubTo(e.clientX);
    const up = () => { setDrag(false); setPinned(true); };
    window.addEventListener('pointermove', mv);
    window.addEventListener('pointerup', up);
    return () => { window.removeEventListener('pointermove', mv);
                   window.removeEventListener('pointerup', up); };
  }, [drag, scrubTo]);
  if (!seq.length) return null;

  const cur = seq[Math.min(n, seq.length - 1)];
  const r = cur.r;
  const gs = r.grid ?? 20, m = r.margin ?? 0, dim = gs + 3 * m;
  const [i, j] = cur.s;
  const fx = ((i + 0.5) / dim) * box.w, fy = ((dim - 1 - j + 0.5) / dim) * box.h;
  const hopIdx = path.indexOf(r);
  // frames in THIS hop's strip; stations is authoritative, n_frames is the server's count
  const BOX = 192;
  // ONE CONTINUOUS TRACK across every triangle. Each hop is a separate strip jpg, so the
  // hand-off used to swap the img src and reset the offset — there was nothing to scroll
  // INTO, and it read as a cut however it was timed. Lay every hop's strip end to end in
  // a track and translate by the GLOBAL tile index: crossing a triangle is then just the
  // scroll continuing. Because the exit image of hop h is vertex A of hop h+1 (extend()
  // makes it so exactly), the two adjacent tiles are the same picture, so the seam reads
  // as a single held beat rather than a jump.
  // All widths are percentages of the TRACK, which is TOT container-widths wide, so one
  // tile is exactly one container width at any screen size — no pixel constants.
  const frames = path.map((row: any) => Math.max(1, row.n_frames ?? (row.stations?.length ?? 1)));
  const cumF: number[] = [];
  { let acc = 0; for (const f of frames) { cumF.push(acc); acc += f; } }
  const TOT = Math.max(1, cumF[cumF.length - 1] + frames[frames.length - 1]);
  const gpos = (cumF[hopIdx] ?? 0) + Math.min(cur.k, (frames[hopIdx] ?? 1) - 1);
  // Animate only a forward step of exactly one TILE. The loop wrap and a scrub are
  // position changes, not travel; transitioning them made the strip race backwards
  // through every frame. A triangle hand-off DOES advance gpos by one, so it animates.
  const prevG = useRef(-1);
  const jumped = gpos !== prevG.current + 1;
  // The map is a different picture per triangle, so its marker must never glide across
  // a hand-off even though the strip does.
  const hopId = `${r?.hop}-${r?.chain}`;
  const prevHop = useRef<string>('');
  const hopJump = jumped || hopId !== prevHop.current;
  useEffect(() => { prevG.current = gpos; prevHop.current = hopId; }, [gpos, hopId]);
  const glide = (prop: string) => (jumped ? 'none' : `${prop} ${speed}ms linear`);


  return (
    <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
      <div style={{ position: 'relative', width: 175, flexShrink: 0 }}>
        <img key={`${r.hop}-${r.chain}`} src={api.hikeMapUrl(hikeId, r.chain, r.hop)}
             alt={`triangle ${hopIdx + 1}`}
             onLoad={(e) => { const el = e.target as HTMLImageElement;
                              setBox({ w: el.clientWidth, h: el.clientHeight }); }}
             onError={(e) => { (e.target as HTMLImageElement).style.visibility = 'hidden'; }}
             style={{ width: 175, borderRadius: 3, background: '#000', display: 'block',
                      outline: cur.last ? '2px solid #e94560' : 'none' }} />
        {box.w > 0 && (
          <div style={{ position: 'absolute', left: fx - 6, top: fy - 6, width: 12, height: 12,
                        borderRadius: 6, border: '2px solid #fff', pointerEvents: 'none',
                        background: cur.last ? 'rgba(233,69,96,.9)'
                                    : cur.approach ? 'rgba(150,200,255,.9)'
                                    : 'rgba(80,255,180,.85)',
                        transition: hopJump ? 'none'
                                    : `left ${speed}ms linear, top ${speed}ms linear` }} />
        )}
        <div style={{ fontSize: 10, color: '#889', marginTop: 2 }}>
          triangle {hopIdx + 1}/{path.length}
          {cur.last && hopIdx < path.length - 1 &&
            <span style={{ color: '#e94560' }}> · exiting →</span>}
        </div>
      </div>
      <div>
        {/* Slide by a FRACTION of the strip's own width, never a pixel count: tiles are
            now rendered at the hike's generated resolution (128-512px), so any hard-coded
            frame size would desync. width = n*100% makes one tile exactly one box width. */}
        <div style={{ width: BOX, height: BOX, overflow: 'hidden',
                      borderRadius: 3, background: '#000', position: 'relative' }}>
          <div style={{ position: 'absolute', top: 0, left: 0, height: '100%',
                        width: `${TOT * 100}%`,
                        transform: `translateX(-${(gpos * 100) / TOT}%)`,
                        transformOrigin: 'top left',
                        transition: glide('transform') }}>
            {path.map((row: any, i: number) => (
              /* only the neighbouring hops carry a src: layout is exact from the frame
                 counts alone, so distant strips cost nothing until they are approached */
              <img key={`f-${row.hop}-${row.chain}`} alt={i === hopIdx ? `frame ${cur.k}` : ''}
                   src={Math.abs(i - hopIdx) <= 1
                        ? api.hikeFilmstripUrl(hikeId, row.chain, row.hop, undefined,
                                               seedIdx) : undefined}
                   /* a seed variant that 404s must not hide the tile permanently: the
                      element is reused when the user switches back to a seed that has one */
                   onLoad={(e) => { (e.target as HTMLImageElement).style.visibility = 'visible'; }}
                   onError={(e) => { (e.target as HTMLImageElement).style.visibility = 'hidden'; }}
                   style={{ position: 'absolute', top: 0, height: 'auto',
                            left: `${(cumF[i] * 100) / TOT}%`,
                            width: `${(frames[i] * 100) / TOT}%` }} />
            ))}
          </div>
        </div>
        {/* drag anywhere on the bar to scrub; each tick is one station */}
        <div ref={strip}
             onPointerDown={(e) => { e.preventDefault(); setDrag(true); setPinned(true);
                                     scrubTo(e.clientX); }}
             title="drag to scrub · click a frame to pin it"
             style={{ display: 'flex', gap: 1, height: 14, marginTop: 4,
                      cursor: drag ? 'grabbing' : 'grab', userSelect: 'none' }}>
          {seq.map((q, idx) => (
            <div key={idx} style={{ flex: 1, borderRadius: 1,
                 background: idx === n ? '#fff'
                             : q.approach ? 'rgba(150,200,255,.55)'
                             : q.last ? 'rgba(233,69,96,.7)' : 'rgba(80,255,180,.45)' }} />
          ))}
        </div>
        <div style={{ fontSize: 10, color: '#778', marginTop: 3 }}>
          step {n + 1}/{seq.length} · cell [{i},{j}]
          {cur.approach && <span style={{ color: '#96c8ff' }}> · walking in from the exit</span>}
        </div>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginTop: 4 }}>
          <button onClick={() => setPinned((p) => !p)}
                  style={{ fontSize: 10, padding: '2px 8px', borderRadius: 3,
                           cursor: 'pointer', border: '1px solid #444', color: '#ddd',
                           background: pinned ? '#20304d' : '#1a1a2e' }}>
            {pinned ? '▶ resume' : '❚❚ pin'}
          </button>
          <button onClick={() => onBranch(r, cur.k)} disabled={branching}
                  title="start a new hike whose first triangle is this image plus two fresh prompts"
                  style={{ fontSize: 10, padding: '2px 8px', borderRadius: 3,
                           cursor: 'pointer', border: 'none', color: '#fff',
                           background: branching ? '#333' : '#4ecca3' }}>
            {branching ? 'starting…' : 'explore from here'}
          </button>
        </div>
        {nSeeds > 1 && (
          <div style={{ display: 'flex', gap: 4, alignItems: 'center', marginTop: 4,
                        flexWrap: 'wrap' }}
               title="the same walk under a different noise draw — the ridge was found on the field averaged over all of these">
            <span style={{ fontSize: 10, color: '#889' }}>seed</span>
            {Array.from({ length: nSeeds }, (_, k) => (
              <button key={k} onClick={() => setSeedIdx(k)}
                      style={{ fontSize: 10, padding: '1px 6px', borderRadius: 3,
                               cursor: 'pointer', border: '1px solid #444',
                               color: k === seedIdx ? '#fff' : '#99a',
                               background: k === seedIdx ? '#20304d' : '#1a1a2e' }}>
                {seedVals[k] ?? k}
              </button>
            ))}
          </div>
        )}
        <div style={{ fontSize: 10, color: '#667', marginTop: 2, maxWidth: BOX }}>
          {(r.labels ?? []).map((l: string) => l.slice(0, 30)).join('  |  ')}
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ Hikers
// A population of chains walks ridges, hops to a new simplex at each boundary, and
// competes against one shared novelty archive. Long-running: start, then poll.
function HikePanel() {
  const promptA = useRidgeStore((s) => s.promptA);
  const promptB = useRidgeStore((s) => s.promptB);
  const promptC = useRidgeStore((s) => s.promptC);
  const [open, setOpen] = useState(false);
  const [beam, setBeam] = useState(4);
  const [budget, setBudget] = useState(16);
  const [gridSize, setGridSize] = useState(25);
  const [refineRounds, setRefineRounds] = useState(0);
  const [margin, setMargin] = useState(0);
  const [steps, setSteps] = useState(8);
  // Seed was fixed at the backend default of 42 and never exposed, so every hike any
  // tester ran was ONE noise draw. E25 measured the ridge field's seed-to-seed Spearman
  // at 0.69-0.79 -- seed moves the ridge more than steps or resolution do -- so which
  // seed you got was the single largest uncontrolled factor in a hike.
  const [seed, setSeed] = useState(42);
  const [seedCount, setSeedCount] = useState(1);
  const [cfg, setCfg] = useState(4.0);
  const [res, setRes] = useState(512);
  const [pool, setPool] = useState<string[] | null>(null);   // null = server default
  const [poolName, setPoolName] = useState<string>('');
  const [stopping, setStopping] = useState(false);
  const [animate, setAnimate] = useState(true);
  const [speed, setSpeed] = useState(450);
  const [hikeId, setHikeId] = useState<string | null>(() => loadResume().hikeId ?? null);
  const [status, setStatus] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  // Exactly ONE poll chain, and it belongs to the hike currently on screen.
  //
  // `poll` reschedules itself, so every entry point that called it -- start, branch, and
  // every tab foregrounding -- forked another self-perpetuating chain while `timer`
  // remembered only the most recent one. Ten alt-tabs left ten chains: ten /status GETs
  // every 3s, unmount cancelled one of the ten, and after a branch the parent hike's
  // chain went on writing its own status into the panel, so hop/grids alternated between
  // the two runs every few seconds. Worst of all, a chain still polling a dead id took
  // the 'unknown' branch on every tick and cleared the resume id of the run that was
  // actually going.
  //
  // Hence `watching`: the one id a chain is allowed to report on. A chain that no longer
  // matches it drops out silently, and a reschedule cancels the pending tick first, so
  // transient duplicates (a foreground event landing mid-request) collapse back to one.
  //
  // reschedule in `finally`: a single transient fetch/parse failure used to end the
  // poll chain for good, freezing the panel at "running" with the button disabled
  // for the rest of the session.
  const watching = useRef<string | null>(null);
  const poll = useCallback(async (id: string) => {
    if (watching.current !== id) return;          // superseded before this tick ran
    let again = true;
    try {
      const st = await api.getHikeStatus(id);
      if (watching.current !== id) return;        // superseded while the request was in flight
      if (st.status === 'unknown') {
        again = false;                            // nothing left to poll: do not respawn
        saveResume({ hikeId: null }); setHikeId(null); setStatus(null); setStopping(false);
        setErr('that run is no longer on the server'); return;
      }
      setStatus(st);
      setErr(null);
      again = st.status === 'running';
      if (!again) setStopping(false);
    } catch (e: any) {
      setErr(`${String(e)} — retrying`);
    } finally {
      if (again && watching.current === id) {
        if (timer.current) window.clearTimeout(timer.current);
        timer.current = window.setTimeout(() => poll(id), 3000);
      }
    }
  }, []);

  // Resume a run the browser forgot (tab evicted, laptop slept) and re-poll the instant
  // the tab is foregrounded, since background timers are throttled. The cleanup also
  // covers unmount, which is why there is no second unmount-only effect.
  useEffect(() => {
    watching.current = hikeId;
    if (hikeId) poll(hikeId);
    const off = onResumeVisible(() => { if (watching.current) poll(watching.current); });
    return () => {
      off();
      watching.current = null;                    // an in-flight reply must not reschedule
      if (timer.current) { window.clearTimeout(timer.current); timer.current = null; }
    };
  }, [hikeId, poll]);

  // One prompt per line. If the file has commas we take the first column, so a
  // one-column CSV and a plain text list both work. Quotes are stripped and a
  // header-looking first row ("prompt") is dropped.
  const loadCsv = useCallback(async (f: File) => {
    try {
      const text = await f.text();
      let rows = text.split(/\r?\n/)
        .map((l) => (l.includes(',') ? l.split(',')[0] : l).trim().replace(/^"|"$/g, ''))
        .filter((l) => l.length > 0);
      if (rows.length && /^(prompt|prompts|text)$/i.test(rows[0])) rows = rows.slice(1);
      const uniq = Array.from(new Set(rows));
      if (uniq.length < 3) { setErr(`${f.name}: need at least 3 prompts, found ${uniq.length}`); return; }
      setPool(uniq); setPoolName(f.name); setErr(null);
    } catch (e: any) { setErr(`could not read ${f.name}: ${String(e)}`); }
  }, []);

  const [branching, setBranching] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);

  // Start a new hike seeded at the station the user pinned. The station is a barycentric
  // point in that walk's simplex, so the backend rebuilds it as an embedding coordinate
  // and pairs it with two freshly drawn prompts -- the same move an edge-hop makes, with
  // the user choosing the point instead of the selector.
  const branch = useCallback(async (row: any, station: number) => {
    if (!hikeId) return;
    setBranching(true); setErr(null);
    try {
      const r = await api.branchHike(hikeId, {
        hop: row.hop, chain: row.chain, station,
        beam, budget, grid_size: gridSize, refine_rounds: refineRounds, margin,
        seed, seed_count: seedCount,
        ...(pool ? { prompt_pool: pool } : {}),
      });
      // no poll() here: the hikeId effect owns starting the chain, and calling it
      // directly would only fork a second one (or, now, no-op against `watching`).
      if (r.hike_id) { saveResume({ hikeId: r.hike_id }); setStatus(null);
                       setHikeId(r.hike_id); }
    } catch (e: any) {
      setErr(String(e));
    } finally { setBranching(false); }
  }, [hikeId, beam, budget, gridSize, refineRounds, margin, pool, seed, seedCount]);

  const stop = useCallback(async () => {
    if (!hikeId) return;
    setStopping(true);
    try { await api.cancelHike(hikeId); }
    catch (e: any) { setErr(String(e)); setStopping(false); }
    // leave `stopping` set: the poll clears it when the status leaves 'running'
  }, [hikeId]);

  const start = useCallback(async () => {
    setErr(null); setStatus(null); setStopping(false);
    if (!promptA || !promptB || !promptC) { setErr('needs three seed prompts'); return; }
    try {
      const r = await api.startHike({
        prompt_a: promptA, prompt_b: promptB, prompt_c: promptC,
        beam, budget, grid_size: gridSize,
        refine_rounds: refineRounds, margin,
        seed, seed_count: seedCount,
        steps, guidance_scale: cfg, height: res, width: res,
        ...(pool ? { prompt_pool: pool } : {}),
      });
      if (!r.hike_id) { setErr('start failed'); return; }
      saveResume({ hikeId: r.hike_id }); setHikeId(r.hike_id);   // the effect starts polling
    } catch (e: any) { setErr(String(e)); }
  }, [promptA, promptB, promptC, beam, budget, gridSize, refineRounds, margin, pool]);

  const pct = status ? Math.round((status.grids_done / Math.max(status.budget, 1)) * 100) : 0;
  // newest first; group rows by hop for display
  const rows: any[] = status?.chains ?? [];

  return (
    <div style={{ background: '#101a2e', borderBottom: '1px solid #333', padding: '6px 12px' }}>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
        <b style={{ fontSize: 12, cursor: 'pointer' }} onClick={() => setOpen(!open)}>
          {open ? '▾' : '▸'} Hikers
        </b>
        <span style={{ fontSize: 11, color: '#888' }}>
          population ridge-walk across chained simplices
        </span>
        {/* 'interrupted' (the server restarted under this hike) and 'cancelled' are not
            successes; in green they read as one. Amber for every terminal state that is
            neither a completed run nor a hard error. */}
        {status && (
          <span style={{ fontSize: 11, color: status.status === 'error' ? '#e94560'
                           : (status.status === 'interrupted' || status.status === 'cancelled')
                             ? '#e9a145' : '#4ecca3' }}>
            {status.status} · hop {status.hop} · {status.grids_done}/{status.budget} grids
            {status.refined_cells ? ` · +${status.refined_cells} refined` : ''}
          </span>
        )}
      </div>

      {open && (
        <div style={{ marginTop: 6 }}>
          <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
            {/* Bounds match models.py exactly. They used to be 8/40/30 against server
                ceilings of 32/1000/100, so the spinners stopped well short of what the
                API accepts -- and models.py claims in a comment that the UI no longer
                caps these. A value above the server bound is rejected by FastAPI with a
                422 the panel cannot explain, so these must not exceed it either. */}
            <label style={{ fontSize: 11 }}>hikers
              <input type="number" min={1} max={32} value={beam}
                     onChange={(e) => setBeam(+e.target.value)}
                     style={{ width: 46, marginLeft: 5 }} />
            </label>
            <label style={{ fontSize: 11 }}>grid budget
              <input type="number" min={1} max={1000} value={budget}
                     onChange={(e) => setBudget(+e.target.value)}
                     style={{ width: 52, marginLeft: 5 }} />
            </label>
            <label style={{ fontSize: 11 }}>grid size
              <input type="number" min={2} max={100} value={gridSize}
                     onChange={(e) => setGridSize(+e.target.value)}
                     style={{ width: 52, marginLeft: 5 }} />
            </label>
            <label style={{ fontSize: 11 }} title="denoising steps per image; fewer is faster">
              steps
              <input type="number" min={1} max={50} value={steps}
                     onChange={(e) => setSteps(Math.max(1, Math.min(50, +e.target.value)))}
                     style={{ width: 42, marginLeft: 5 }} />
            </label>
            <label style={{ fontSize: 11 }}
                   title="classifier-free guidance. 1.0 runs a single forward pass per step (about 2x faster); above 1.0 adds an unconditional branch">
              cfg
              <input type="number" min={0} max={20} step={0.5} value={cfg}
                     onChange={(e) => setCfg(Math.max(0, Math.min(20, +e.target.value)))}
                     style={{ width: 50, marginLeft: 5 }} />
            </label>
            <label style={{ fontSize: 11 }} title="pixels per generated image">
              res
              <select value={res} onChange={(e) => setRes(+e.target.value)}
                      style={{ marginLeft: 5, fontSize: 11 }}>
                {[128, 192, 256, 384, 512].map((r) => <option key={r} value={r}>{r}</option>)}
              </select>
            </label>
            <label style={{ fontSize: 11 }}
                   title="the noise draw. The ridge field is only 0.69-0.79 reproducible across seeds (E25), so this is the largest single lever on what a hike finds.">
              seed
              <input type="number" value={seed} min={0}
                     onChange={(e) => setSeed(Math.max(0, +e.target.value || 0))}
                     style={{ marginLeft: 5, width: 62, fontSize: 11 }} />
            </label>
            <label style={{ fontSize: 11 }}
                   title="generate every cell at this many consecutive seeds and walk the AVERAGED sensitivity field, then switch the display between seeds. Cost is linear in this number.">
              seeds averaged
              <select value={seedCount} onChange={(e) => setSeedCount(+e.target.value)}
                      style={{ marginLeft: 5, fontSize: 11 }}>
                {[1, 2, 3, 4, 6, 8].map((r) => <option key={r} value={r}>{r}</option>)}
              </select>
            </label>
            <label style={{ fontSize: 11 }}
                   title="pad the simplex so hull cells have a full neighbourhood; the ring is measured, never walked to">
              pad
              <select value={margin} onChange={(e) => setMargin(+e.target.value)}
                      style={{ marginLeft: 5, fontSize: 11 }}>
                <option value={0}>off</option>
                <option value={1}>1</option>
                <option value={2}>2</option>
                <option value={3}>3</option>
              </select>
            </label>
            <label style={{ fontSize: 11 }} title="each round doubles the lattice and generates only near the ridge">
              refine
              <select value={refineRounds} onChange={(e) => setRefineRounds(+e.target.value)}
                      style={{ marginLeft: 5, fontSize: 11 }}>
                <option value={0}>off</option>
                <option value={1}>1x</option>
                <option value={2}>2x</option>
              </select>
            </label>
            {/* seedCount multiplies EVERYTHING linearly -- base grid and refine rounds
                alike, since every seed refines the same targets -- so leaving it out
                understated a 3-seed hike threefold. */}
            <span style={{ fontSize: 10, color: '#777' }}
                  title={seedCount > 1 ? `${seedCount} seeds, so ${seedCount}x the images `
                                         + `of a single-seed hike at these settings` : undefined}>
              ≈ {(budget * (Math.round((gridSize * (gridSize + 1)) / 2)
                 + margin * (3 * gridSize + 3 * margin))
                 + budget * refineRounds * 200) * seedCount} images
              {seedCount > 1 && ` (${seedCount} seeds)`}
            </span>
            <button onClick={start} disabled={status?.status === 'running'}
              style={{ padding: '3px 12px', fontSize: 11, borderRadius: 4, border: 'none',
                       cursor: 'pointer', color: '#fff',
                       background: status?.status === 'running' ? '#333' : '#4ecca3' }}>
              {status?.status === 'running' ? 'hiking…' : 'send hikers'}
            </button>
            {status?.status === 'running' && hikeId && (
              <button onClick={stop} disabled={stopping}
                title="stops this hike only; other jobs keep running"
                style={{ padding: '3px 12px', fontSize: 11, borderRadius: 4, border: 'none',
                         cursor: 'pointer', color: '#fff',
                         background: stopping ? '#333' : '#e94560' }}>
                {stopping ? 'stopping…' : 'stop'}
              </button>
            )}
            {err && <span style={{ fontSize: 11, color: '#e94560' }}>{err}</span>}
            {status?.error && <span style={{ fontSize: 11, color: '#e94560' }}>{status.error}</span>}
          </div>

          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 5 }}>
            <label style={{ fontSize: 11, cursor: 'pointer', color: '#9ab' }}>
              prompt pool
              <input type="file" accept=".csv,.txt,text/csv,text/plain"
                     onChange={(e) => { const f = e.target.files?.[0]; if (f) loadCsv(f); }}
                     style={{ marginLeft: 6, fontSize: 10, width: 190 }} />
            </label>
            <span style={{ fontSize: 10, color: pool ? '#4ecca3' : '#777' }}>
              {pool ? `${poolName}: ${pool.length} prompts` : 'using the built-in pool'}
            </span>
            <label style={{ fontSize: 11, color: '#9ab', marginLeft: 4 }}>
              <input type="checkbox" checked={animate}
                     onChange={(e) => setAnimate(e.target.checked)}
                     style={{ verticalAlign: 'middle', marginRight: 3 }} />
              follow hikers
            </label>
            {animate && (
              <label style={{ fontSize: 10, color: '#778' }}>
                {(1000 / speed).toFixed(1)}/s
                <input type="range" min={120} max={1200} step={60} value={1320 - speed}
                       onChange={(e) => setSpeed(1320 - +e.target.value)}
                       style={{ width: 70, marginLeft: 4, verticalAlign: 'middle' }} />
              </label>
            )}
            {pool && (
              <button onClick={() => { setPool(null); setPoolName(''); }}
                      style={{ fontSize: 10, padding: '1px 6px', cursor: 'pointer',
                               background: '#1a1a2e', color: '#ddd', border: '1px solid #444',
                               borderRadius: 3 }}>reset</button>
            )}
          </div>

          {status && (
            <div style={{ background: '#333', height: 4, borderRadius: 3, margin: '6px 0' }}>
              <div style={{ width: `${pct}%`, background: '#4ecca3', height: '100%',
                            borderRadius: 3, transition: 'width .3s' }} />
            </div>
          )}

          {(status?.notes?.length ?? 0) > 0 && (
            <div style={{ fontSize: 10, color: '#e9a145', margin: '4px 0' }}>
              {status.notes.map((n: string, i: number) => <div key={i}>· {n}</div>)}
            </div>
          )}

          {rows.length > 0 && animate && (
            <div style={{ maxHeight: 400, overflowY: 'auto' }}>
              {hikeId && (
                <LineageTree hikeId={hikeId} rows={rows} selected={selected}
                             onSelect={(l) => setSelected(l === selected ? null : l)} />
              )}
              {(() => {
                const names = lineageNames(rows);
                const maxHop = Math.max(0, ...rows.map((r: any) => r.hop ?? 0));
                const nameOf = (r: any) => names.get(r.lineage ?? 'root') ?? '';
                // the parent BRANCH is the nearest ancestor with a different name
                const parentOf = (r: any) => {
                  let l = parentLineage(r.lineage ?? 'root');
                  const mine = nameOf(r);
                  while (l && (names.get(l) ?? '') === mine) l = parentLineage(l);
                  return l ? (names.get(l) ?? '') : '';
                };
                const forkOf = (r: any) => {
                  // triangles shared with the parent branch = its depth
                  let l = parentLineage(r.lineage ?? 'root');
                  const mine = nameOf(r);
                  let d = (r.lineage ?? 'root') === 'root' ? 0 : (r.lineage as string).split('>').length - 1;
                  while (l && (names.get(l) ?? '') === mine) { d--; l = parentLineage(l); }
                  return Math.max(0, d - 1);
                };
                return journeys(rows).map((path) => {
                const leaf = path[path.length - 1];
                const stations = path.reduce((n: number, r: any) => n + (r.stations?.length ?? 0), 0);
                return (
                  <div key={leaf.lineage ?? `${leaf.hop}-${leaf.chain}`}
                       style={{ borderTop: '1px solid #222', padding: '5px 0' }}>
                    <div style={{ fontSize: 11, color: '#9ab', display: 'flex',
                                  alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                      {/* one dot per triangle: hollow where this hiker was still
                          travelling with its parent, filled once it went its own way */}
                      <span style={{ display: 'inline-flex', gap: 3, alignItems: 'center' }}>
                        {path.map((_, k) => (
                          <span key={k} style={{ width: 8, height: 8, borderRadius: 4,
                            border: `1.5px solid ${branchColour(nameOf(leaf))}`,
                            background: k >= forkOf(leaf) ? branchColour(nameOf(leaf)) : 'transparent',
                            opacity: k >= forkOf(leaf) ? 1 : 0.45 }} />
                        ))}
                      </span>
                      <b style={{ color: branchColour(nameOf(leaf)), fontSize: 12 }}>
                        {nameOf(leaf) || 'A'}
                      </b>
                      {parentOf(leaf) && (
                        <span style={{ color: '#667' }}>
                          split from {parentOf(leaf)} at triangle {forkOf(leaf) + 1}
                        </span>
                      )}
                      <span>{path.length} triangle{path.length > 1 ? 's' : ''} · {stations} steps</span>
                      {leaf.hop < maxHop && (
                        <span style={{ color: '#e9a145' }}>· pruned after hop {leaf.hop}</span>
                      )}
                      {Number.isFinite(leaf.cum_novel) && (
                        <span style={{ color: leaf.cum_novel >= 0.5 ? '#4ecca3' : '#e9a145' }}>
                          · {Math.round(leaf.cum_novel * 100)}% new
                        </span>
                      )}
                    </div>
                    <HikeJourney hikeId={hikeId!} path={path} playing={animate}
                                 speed={speed} onBranch={branch} branching={branching} />
                  </div>
                );
                });
              })()}
            </div>
          )}

          {rows.length > 0 && !animate && (
            <div style={{ maxHeight: 330, overflowY: 'auto' }}>
              {rows.map((r) => (
                // key by hop+chain, not array index: rows are PREPENDED, so an
                // index key reassigns every row's identity on each poll and React
                // reused the DOM node -- including an <img> already hidden by a
                // 404 from an earlier, still-generating hop.
                <div key={`${r.hop}-${r.chain}`}
                     style={{ borderTop: '1px solid #222', padding: '4px 0' }}>
                  <div style={{ fontSize: 11, color: '#9ab' }}>
                    hop {r.hop} · hiker {r.chain} · arc {r.arc_len} stations
                    {r.grid ? ` @ ${r.grid}×${r.grid}` : ''} ·{' '}
                    {r.n_arcs} arcs available ·{' '}
                    <span style={{ color: r.cum_novel >= 0.5 ? '#4ecca3' : '#e9a145' }}>
                      {Number.isFinite(r.cum_novel) ? `${Math.round(r.cum_novel * 100)}% new` : 'first hop'}
                    </span>
                  </div>
                  <div style={{ fontSize: 10, color: '#667', marginBottom: 3 }}>
                    {r.labels.map((l: string) => l.slice(0, 38)).join('  |  ')}
                  </div>
                  {r.note && (
                    <div style={{ fontSize: 10, color: '#e9a145', marginBottom: 3 }}>{r.note}</div>
                  )}
                  {hikeId && r.arc_len > 0 && (
                    // map beside filmstrip: WHERE the walk ran (and what it passed over)
                    // next to WHAT it produced. Either alone is hard to judge.
                    <div style={{ display: 'flex', gap: 6, alignItems: 'flex-start' }}>
                      <img key={`${r.hop}-${r.chain}-map`}
                           src={api.hikeMapUrl(hikeId, r.chain, r.hop)}
                           alt={`simplex map, hop ${r.hop} hiker ${r.chain}`}
                           title="walked arc (green) · arcs not taken (grey) · junctions (amber) · exit (red)"
                           onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                           style={{ width: 150, height: 150, objectFit: 'contain',
                                    borderRadius: 3, background: '#000', flexShrink: 0 }} />
                      <img key={`${r.hop}-${r.chain}-img`}
                           src={api.hikeFilmstripUrl(hikeId, r.chain, r.hop)}
                           alt={`hop ${r.hop} hiker ${r.chain}`}
                           onError={(e) => { (e.target as HTMLImageElement).style.visibility = 'hidden'; }}
                           onLoad={(e) => { (e.target as HTMLImageElement).style.visibility = 'visible'; }}
                           style={{ minWidth: 0, maxWidth: 'calc(100% - 156px)',
                                    borderRadius: 3, background: '#000' }} />
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
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
      <TokenPanel />
      <ProgressBar />
      <ScanCompletePanel />
      {feature('refine') && <RefinePanel />}
      {feature('itinerary') && <ItineraryPanel />}
      {feature('hikers') && <HikePanel />}
      {feature('discovery') && <DiscoverPanel />}
      {feature('cascade') && <CascadePanel />}
      {/* the other way to spend a boundary budget on the same simplex: a refined lattice
          instead of chords. Sits next to the Cascade because they are alternatives. */}
      {feature('amr') && <AmrPanel />}
      {/* and the third way: not a search at all. The Metropolis sampler DRAWS recipes in
          proportion to the sharpness of the field, so its gallery is a fair sample of the
          ridges rather than a tour of the ones a survey happened to find (h25a). */}
      {feature('metro') && <MetroPanel />}
      {/* k-independent views: instead of covering the simplex, look closely at one place in
          it. The microscope lays a G x G image lattice on a plane through a recipe -- G^2
          images at any k -- and zooms 2x per click (Sequential Gallery). */}
      {feature('microscope') && <MicroscopePanel />}
      <MainViewport />
      <ExportOverlay />
    </div>
  );
}
