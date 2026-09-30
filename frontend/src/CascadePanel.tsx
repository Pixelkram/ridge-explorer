/**
 * CascadePanel -- the two-tier ridge cascade as an interactive run.
 *
 * Runs the validated recipe (E85/E86/E92) on user prompts: isolate crossings with
 * well-spread chords, score them with coupled seeds against the run's own measured
 * background, refine the strongest regions plus one exploration slot. The panel is
 * built to make the process legible while it runs: a phase bar, a live ticker of the
 * images just generated, the map filling in dot by dot, and findings that light up
 * as they are certified.
 */
import { useEffect, useRef, useState } from 'react';
import {
  cascadeStart, cascadeStatus, cascadeCancel, cascadeImageUrl,
  cascadeWalkStart, cascadeWalkStatus, cascadeWalkCancel, cascadeWalkCertify, cascadePointInfo,
} from './api/client';
import type { CascadeStatus, CascadePatch, WalkStatus } from './api/types';
import CascadeMap from './CascadeMap';
import { useProbeStore } from './stores/probeStore';

const BOX: React.CSSProperties = {
  background: '#16213e', border: '1px solid #333', borderRadius: 4,
  padding: '8px 12px', margin: '0 12px 8px', fontSize: 12,
};
const NUM: React.CSSProperties = {
  width: 66, background: '#0a0a1a', color: '#fff', border: '1px solid #444',
  borderRadius: 3, padding: '2px 4px', fontSize: 12,
};
const ACCENT = '#4ecca3';
const WARN = '#e94560';
const CELL = 0.0236;         // one fine patch cell -- the unit the stride slider reads out in
const STRIDE_REF = 0.025;    // chord probe spacing of record; the cost model below is quoted at it
const CERT_NO = '#8a93b8';   // the map's grey for "not significant", reused for stations that did not certify
// What a detection-only run cannot do, in the backend's own words (routers/cascade.py).
const UNCERT = 'run was detection-only: no bisected crossings — re-run with certify on';
const CERTIFY_TIP = 'off (default): detection only — bracket-precision positions, no '
  + 'certification, no patches; ~6× cheaper per crossing. on: rebracket + bisect + 4-seed '
  + 'score every detected crossing (~12 images each) → certified boundaries, patches, walks. '
  + 'The boundary-density overlay and the coverage count still work (uncertified)';

// Total chords a branching survey may draw: the geometric sum n*(1 + b + ... + b^depth),
// with branch=1 the ratio formula's removable singularity. A cap, not a target -- a
// generation only fills from parents that actually crossed something.
const chordCap = (n: number, b: number, d: number) =>
  b <= 0 ? n : b === 1 ? n * (d + 1) : Math.round((n * (b ** (d + 1) - 1)) / (b - 1));

// Presets k=3..16 from the measured yield model: chords = max(12, ceil(12k/7)) keeps
// ~3 crossings per patch slot at 4 patches; k>9 is beyond the calibrated range (the
// significance background still self-calibrates per run, so verdicts stay honest).
const PRESETS = Array.from({ length: 14 }, (_, i) => {
  const k = i + 3;
  const patches = 4;
  const chords = Math.max(12, Math.ceil((12 * k) / 7));
  const cross = Math.max(1, Math.round((chords * 7) / k));
  const imgs = Math.round((chords * 110) / k + cross * 2 + (cross + 12) * 8
                          + patches * 25);
  const mins = Math.max(1, Math.round(imgs / 8 / 60));
  return { k, chords, patches,
           label: `k=${k} · ${chords} chords · ~${imgs} img · ~${mins} min`
                  + (k > 9 ? ' · beyond calibrated range' : '') };
});

// The pipeline as the user should read it; keys match backend phase names.
const PHASES: [string, string][] = [
  ['chords', 'isolate'],
  ['bisect', 'pin'],
  ['score', 'certify'],
  ['patches', 'refine'],
];

function PhaseBar({ status }: { status: CascadeStatus }) {
  // a detection-only run stops after the chords: the phases it never runs must not light up
  const phases = status.certify === false ? PHASES.slice(0, 1) : PHASES;
  const idx = phases.findIndex(([p]) => p === status.phase);
  const done = status.phase === 'done' || status.status === 'complete';
  return (
    <div style={{ display: 'flex', gap: 4, margin: '8px 0 2px' }}>
      {phases.map(([key, label], i) => {
        const active = !done && i === idx;
        const past = done || i < idx;
        const frac = active && status.phase_total > 0
          ? Math.min(1, status.phase_done / status.phase_total) : past ? 1 : 0;
        return (
          <div key={key} style={{ flex: 1 }}>
            <div style={{ height: 5, background: '#0a0a1a', borderRadius: 3,
                          overflow: 'hidden', border: '1px solid #2a2a4a' }}>
              <div style={{ width: `${frac * 100}%`, height: '100%',
                            background: past ? ACCENT : '#7d84b5',
                            transition: 'width 0.6s' }} />
            </div>
            <div style={{ fontSize: 10, marginTop: 2, textAlign: 'center',
                          color: active ? '#fff' : past ? ACCENT : '#667' }}>
              {label}{active && status.phase_total > 0 &&
                ` ${status.phase_done}/${status.phase_total}`}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function CrossingCard({ c, runId, selected, uncertified, onClick }: {
  c: CascadeStatus['crossings'][number]; runId: string;
  selected: boolean; uncertified?: boolean; onClick: () => void;
}) {
  const scored = c.b !== null;
  return (
    <div onClick={onClick}
         style={{ cursor: 'pointer', width: 96,
                  border: selected ? '2px solid #fff'
                    : c.significant ? `2px solid ${ACCENT}` : '2px solid #2a2a4a',
                  borderRadius: 4, background: '#0d0d20', overflow: 'hidden' }}>
      {c.thumb >= 0 ? (
        <img src={cascadeImageUrl(runId, c.thumb)} width={92} height={92}
             loading="lazy" style={{ objectFit: 'cover', display: 'block' }} />
      ) : (
        <div style={{ width: 92, height: 92, display: 'flex', alignItems: 'center',
                      justifyContent: 'center', color: '#555' }}>…</div>
      )}
      <div style={{ padding: '3px 5px 5px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between',
                      fontSize: 10, color: c.significant ? ACCENT : '#8a93b8' }}>
          <span>{scored ? `B ${c.b!.toFixed(2)}`
                        : uncertified ? 'uncertified' : 'pinning…'}</span>
          {c.ridge_group !== null && (
            <span style={{ color: '#667' }}>r{c.ridge_group}</span>
          )}
        </div>
        <div style={{ height: 3, background: '#1a1a2e', borderRadius: 2,
                      marginTop: 2 }}>
          {scored && (
            <div style={{ width: `${Math.min(100, c.b! * 100)}%`, height: '100%',
                          background: c.significant ? ACCENT : '#8a93b8',
                          borderRadius: 2 }} />
          )}
        </div>
      </div>
    </div>
  );
}

function PatchGrid({ runId, patch, onCell, running }: {
  runId: string; patch: CascadePatch; running?: boolean;
  onCell?: (thumb: number, row: number, col: number) => void;
}) {
  const total = patch.grid.length * (patch.grid[0]?.length ?? 0);
  const done = patch.grid.flat().filter((t) => t >= 0).length;
  const filling = !!running && done < total;
  return (
    <div style={{ background: '#0d0d20', border: '1px solid #2a2a4a',
                  borderRadius: 4, padding: 8,
                  borderLeft: `3px solid ${patch.significant ? ACCENT : '#8a93b8'}` }}>
      <div style={{ marginBottom: 4, fontSize: 11,
                    color: patch.significant ? ACCENT : '#8a93b8' }}>
        {patch.exploration ? 'exploration slot' : `region ${patch.region}`}
        {' '}· B {patch.b.toFixed(2)}{patch.significant ? ' · certified' : ''}
        {filling
          ? <span style={{ color: WARN }}> · assembling {done}/{total}…</span>
          : <> · crossing in {patch.cols_with_crossing}/5 rows</>}
      </div>
      <div style={{ fontSize: 10, color: '#556', marginBottom: 3 }}>
        → stepping across the boundary
      </div>
      {patch.grid.map((row, i) => (
        <div key={i} style={{ display: 'flex', gap: 2, marginBottom: 2 }}>
          {row.map((t, j) => t >= 0 ? (
            <img key={j} src={cascadeImageUrl(runId, t)} width={56} height={56}
                 loading="lazy"
                 onClick={() => onCell?.(t, i, j)}
                 style={{ objectFit: 'cover', borderRadius: 2, cursor: 'pointer' }} />
          ) : (
            <div key={j} className={filling ? 'cs-shimmer' : undefined}
                 style={{ width: 56, height: 56, background: '#0a0a1a',
                          borderRadius: 2,
                          animationDelay: `${((i * 5 + j) % 7) * 0.16}s` }} />
          ))}
        </div>
      ))}
    </div>
  );
}

export default function CascadePanel() {
  const [open, setOpen] = useState(false);
  const [k, setK] = useState(4);
  const [nChords, setNChords] = useState(24);
  const [stride, setStride] = useState(STRIDE_REF);
  const [branch, setBranch] = useState(0);
  const [depth, setDepth] = useState(0);
  const [trace, setTrace] = useState(false);
  const [certify, setCertify] = useState(false);
  const [nPatches, setNPatches] = useState(4);
  const [seed, setSeed] = useState(42);
  const [promptText, setPromptText] = useState('');
  const [runId, setRunId] = useState<string | null>(null);
  const [status, setStatus] = useState<CascadeStatus | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [selCid, setSelCid] = useState<number | null>(null);
  const [walk, setWalk] = useState<WalkStatus | null>(null);
  const timer = useRef<number | null>(null);
  const walkTimer = useRef<number | null>(null);
  // large-view modal: any clicked image, its recipe, and "explore from here"
  const [detail, setDetail] = useState<{
    thumb: number; title: string; weights: number[] | null;
    rows: [string, string][]; loading: boolean;
  } | null>(null);

  const openDetail = async (thumb: number, title: string,
                            rows: [string, string][],
                            weights?: number[] | null) => {
    if (thumb < 0 || !runId) return;
    if (weights) {
      setDetail({ thumb, title, rows, weights, loading: false });
      return;
    }
    setDetail({ thumb, title, rows, weights: null, loading: true });
    try {
      const info = await cascadePointInfo(runId, thumb);
      setDetail((d) => d && d.thumb === thumb
        ? { ...d, weights: info.weights,
            rows: info.div !== null
              ? [...rows, ['local divergence', info.div.toFixed(2)]] : rows,
            loading: false }
        : d);
    } catch {
      setDetail((d) => d && d.thumb === thumb ? { ...d, loading: false } : d);
    }
  };

  const exploreFrom = async (weights: number[]) => {
    if (!status) return;
    setDetail(null);
    setErr(null);
    setSelCid(null);
    setWalk(null);
    try {
      const r = await cascadeStart({
        k: status.prompts.length,
        prompts: status.prompts,
        n_chords: nChords,
        n_patches: nPatches,
        certify,
        stride,
        branch,
        depth,
        seed,
        focus: weights,
        trace,
      });
      if (r.error) { setErr(r.error); return; }
      setStatus(null);
      setRunId(r.run_id);
    } catch (e: any) { setErr(String(e.message || e)); }
  };

  // One poll chain only (re-entering the effect must clear the previous timer or a
  // re-render forks a second chain).
  useEffect(() => {
    if (!runId) return;
    let live = true;
    const tick = async () => {
      try {
        const s = await cascadeStatus(runId);
        if (!live) return;
        setStatus(s);
        if (s.status === 'running') timer.current = window.setTimeout(tick, 1200);
      } catch {
        if (live) timer.current = window.setTimeout(tick, 4000);
      }
    };
    tick();
    return () => {
      live = false;
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [runId]);

  // walk poll chain -- also while a finished walk is being certified
  useEffect(() => {
    if (!runId || !walk || (walk.status !== 'running' && walk.cert?.status !== 'running')) return;
    let live = true;
    const tick = async () => {
      try {
        const w = await cascadeWalkStatus(runId, walk.walk_id);
        if (!live) return;
        setWalk(w);
        if (w.status === 'running' || w.cert?.status === 'running') walkTimer.current = window.setTimeout(tick, 1000);
      } catch {
        if (live) walkTimer.current = window.setTimeout(tick, 3000);
      }
    };
    walkTimer.current = window.setTimeout(tick, 1000);
    return () => {
      live = false;
      if (walkTimer.current) window.clearTimeout(walkTimer.current);
    };
  }, [runId, walk?.walk_id, walk?.status, walk?.cert?.status]);

  const start = async () => {
    setErr(null);
    setStatus(null);
    setSelCid(null);
    setWalk(null);
    const lines = promptText.split('\n').map((s) => s.trim()).filter(Boolean);
    if (lines.length > 0 && lines.length !== k) {
      setErr(`enter exactly ${k} prompts (one per line) or none to draw from the pool`);
      return;
    }
    try {
      const r = await cascadeStart({
        k,
        prompts: lines.length ? lines : null,
        n_chords: nChords,
        n_patches: nPatches,
        certify,
        stride,
        branch,
        depth,
        seed,
        trace,
      });
      if (r.error) { setErr(r.error); return; }
      setRunId(r.run_id);
    } catch (e: any) {
      setErr(String(e.message || e));
    }
  };

  const select = (cid: number | null) => {
    if (walk?.status === 'running' && runId) cascadeWalkCancel(runId, walk.walk_id);
    setSelCid(cid);
    setWalk(null);
  };

  const [useJvp, setUseJvp] = useState(false);
  const [sigMode, setSigMode] = useState<'relative' | 'absolute' | 'continuity'>('relative');
  const [walkMode, setWalkMode] = useState<'fan' | 'continuation'>('continuation');
  const startWalk = async (direction: number) => {
    if (!runId || selCid === null) return;
    try {
      const w = await cascadeWalkStart(runId, { cid: selCid, direction, n_steps: 5, use_jvp: useJvp, sig_mode: sigMode, mode: walkMode });
      if (w.error) { setErr(w.error); return; }
      setWalk(w);
    } catch (e: any) { setErr(String(e.message || e)); }
  };

  const certifyWalk = async () => {
    if (!runId || !walk) return;
    try {
      const w = await cascadeWalkCertify(runId, walk.walk_id);
      if (w.error) { setErr(w.error); return; }
      setWalk(w);
    } catch (e: any) { setErr(String(e.message || e)); }
  };
  // per-station verdict once certified: true = held on the unseen seeds
  const certOk = (i: number): boolean | null =>
    walk?.cert?.status === 'done' ? !!walk.cert.significant?.[i] : null;

  // Boundary-density overlay. The toggle and the read mode live here; the map draws them.
  // Off by default, and one fetch covers both read modes (they are views of one response).
  const svOn = useProbeStore((s) => s.localSvOn);
  const svView = useProbeStore((s) => s.localSvView);
  const svStatus = useProbeStore((s) => s.localSvStatus);
  const svError = useProbeStore((s) => s.localSvError);
  const setSvOn = useProbeStore((s) => s.setLocalSvOn);
  const setSvView = useProbeStore((s) => s.setLocalSvView);
  const sv = useProbeStore((s) => (runId ? s.localSv[runId] : undefined));
  const svCal = !!sv?.calibrated_ok;
  // A run that finishes after the overlay was switched on has new chords and crossings, so
  // the map it was read from is stale: re-read it once the survey stops.
  const fetchSv = useProbeStore((s) => s.fetchLocalSv);
  useEffect(() => {
    if (svOn && runId && status?.status === 'complete') void fetchSv(runId);
  }, [svOn, runId, status?.status, fetchSv]);

  const running = status?.status === 'running';
  // this RUN was started with certify off, whatever the checkbox says now
  const uncert = status?.certify === false;
  const sigCount = status?.crossings.filter((c) => c.significant).length ?? 0;
  const scored = status?.crossings.filter((c) => c.b !== null) ?? [];
  const sorted = [...(status?.crossings ?? [])]
    .sort((a, b) => (b.b ?? -1) - (a.b ?? -1));
  const walkPath = walk && status
    ? [status.crossings.find((c) => c.cid === walk.cid)?.weights ?? [],
       ...walk.steps.map((s) => s.weights)].filter((w) => w.length > 0)
    : null;
  const walkColors = walkPath && walk?.cert?.status === 'done'
    ? walkPath.map((_, i) => (i === 0 ? null : certOk(i - 1) ? ACCENT : CERT_NO))
    : null;

  return (
    <div style={BOX}>
      <div style={{ cursor: 'pointer', userSelect: 'none' }}
           onClick={() => setOpen(!open)}>
        {open ? '▾' : '▸'} Cascade — find & refine boundaries
      </div>
      {open && (
        <div style={{ marginTop: 8 }}>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap',
                        alignItems: 'center' }}>
            <select value="" onChange={(e) => {
                       const p = PRESETS[Number(e.target.value)];
                       if (p) { setK(p.k); setNChords(p.chords); setNPatches(p.patches); }
                     }}
                     style={{ background: '#0a0a1a', color: '#8a9',
                              border: '1px solid #444', borderRadius: 3,
                              padding: '2px 4px', fontSize: 12 }}>
              <option value="">presets…</option>
              {PRESETS.map((p, i) => (
                <option key={p.k} value={i}>{p.label}</option>
              ))}
            </select>
            <label title="3-4 = crisp certified boundaries (exact map, quiet background); 6-9 = exotic blends, softer verdicts. No upper bound; k>9 is beyond the calibrated range">
              k <input style={NUM} type="number" min={3} value={k}
                       onChange={(e) => setK(Number(e.target.value))} />
            </label>
            <label title="yield ~7/k crossings per chord at ~110/k images each (measured); probing runs at 4 denoising steps (gated: 93% crossing recall, 94% certified recall, ~2x faster) with brackets re-rendered at full fidelity. Rule of thumb: chords = 3 x patches x k/7; raise toward 40+ at k>=6">
              chords <input style={NUM} type="number" min={4} value={nChords}
                            onChange={(e) => setNChords(Number(e.target.value))} />
            </label>
            <label title="chord probe spacing (0.025 = protocol of record ≈ 1 fine cell); finer resolves close boundaries but costs ∝ 1/stride; coarser merges crossings closer than one stride"
                   style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
              stride
              <input type="range" min={0.01} max={0.2} step={0.005} value={stride}
                     onChange={(e) => setStride(Number(e.target.value))}
                     style={{ width: 78 }} />
              <span style={{ color: '#8a9' }}>
                {stride.toFixed(3)} · ≈{(stride / CELL).toFixed(1)} cells
              </span>
            </label>
            <label title="how many NEW chords (rays) are spawned at each chord's most divergent crossing; 0 = no branching. Root chords stay fair samples, child chords are exploratory">
              new chords / crossing <input style={NUM} type="number" min={0} max={6} value={branch}
                            onChange={(e) => {
                              const b = Math.max(0, Math.min(6, Number(e.target.value)));
                              setBranch(b);
                              // branch > 0 with depth 0 would spawn nothing; start at one generation
                              if (b > 0 && depth === 0) setDepth(1);
                            }} />
            </label>
            <label title="how many times the branching is executed recursively: generation 1 branches from the root chords' strongest crossings, generation 2 from generation 1's, and so on; 0 = none"
                   style={{ color: branch > 0 ? undefined : '#667' }}>
              recursion depth <input style={NUM} type="number" min={0} max={4} value={depth}
                           disabled={branch === 0}
                           onChange={(e) => setDepth(
                             Math.max(0, Math.min(4, Number(e.target.value))))} />
            </label>
            {branch > 0 && (
              <span style={{ color: '#8a9', fontSize: 11 }}
                    title="hard cap on the total chords: the geometric sum chords x (1 + branch + ... + branch^depth). Reached only where every chord keeps finding crossings.">
                ≤ {chordCap(nChords, branch, depth)} chords
              </span>
            )}
            <label className="rx-focus" title={CERTIFY_TIP}
                   style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
              <input type="checkbox" checked={certify}
                     onChange={(e) => {
                       setCertify(e.target.checked);
                       // the trace phase walks certified crossings; nothing to walk without them
                       if (!e.target.checked) setTrace(false);
                     }} /> certify crossings
            </label>
            <label title={'how many boundary walks you want to see (25 images each); one slot is ALWAYS the exploration floor'
                          + (certify ? '' : ' — off with "certify crossings" unchecked: a detection-only run renders no patches')}
                   style={{ color: certify ? undefined : '#667' }}>
              patches <input style={NUM} type="number" min={2} value={nPatches}
                             disabled={!certify}
                             onChange={(e) => setNPatches(Number(e.target.value))} />
            </label>
            <label title="a style lever, not a quality dial: same prompts + new seed = a genuinely different boundary map (measured). Fix it to compare settings; change it for another atlas">
              seed <input style={NUM} type="number" value={seed}
                          onChange={(e) => setSeed(Number(e.target.value))} />
            </label>
            <label className="rx-focus" title={'after the survey, walk both ways from every certified crossing (continuation walk), check the stations on unseen seeds, and link crossings a walk reaches: one ridge often shows up as several crossings whose side images no longer match. Draws the ridges on the map and recomputes the unexplored share. Adds a few minutes.'
                     + (certify ? '' : ' — needs certified crossings: ' + UNCERT)}
                   style={{ display: 'inline-flex', alignItems: 'center', gap: 4,
                            color: certify ? undefined : '#667' }}>
              <input type="checkbox" checked={trace} disabled={!certify}
                     onChange={(e) => setTrace(e.target.checked)} /> trace ridges
            </label>
            <button onClick={start} disabled={running}
                    style={{ background: running ? '#333' : '#0f3460',
                             color: '#fff', border: `1px solid ${ACCENT}`,
                             borderRadius: 3, padding: '3px 12px',
                             cursor: running ? 'default' : 'pointer' }}>
              {running ? 'running…' : 'Start'}
            </button>
            {running && runId && (
              <button onClick={() => cascadeCancel(runId)}
                      style={{ background: '#0a0a1a', color: WARN,
                               border: `1px solid ${WARN}`, borderRadius: 3,
                               padding: '3px 10px', cursor: 'pointer' }}>
                Cancel
              </button>
            )}
          </div>
          {(() => {
            // branching multiplies the chord budget by the geometric factor; an upper bound,
            // since child rays are one-armed (half a root's length) and only spawn where a
            // parent crossed something
            const chordTotal = chordCap(nChords, branch, depth);
            const cross = Math.max(1, Math.round((chordTotal * 7) / k));
            // 110/k probes per chord is measured AT the stride of record; the count is ~1/stride
            const probeImgs = ((chordTotal * 110) / k) * (STRIDE_REF / stride);
            // detection only: the probes are the whole bill -- nothing is rebracketed,
            // bisected, scored or refined
            const imgs = Math.round(certify
              ? probeImgs + cross * 4 + (cross + 12) * 8 + nPatches * 25
              : probeImgs);
            // probes run at 4 denoising steps (gated: 93%/94% recall) ~ half price
            const mins = Math.max(1, Math.round((imgs - probeImgs * 0.5) / 8 / 60));
            return (
              <div style={{ color: '#667', marginTop: 4, fontSize: 11 }}>
                estimate: ≤~{imgs} images · ≤~{mins} min · ~{cross} crossings
                {!certify && ' — detection only: uncertified, bracket-precision positions,'
                  + ' no patches and no walks'}
                {certify && cross < 3 * nPatches &&
                  ' — few crossings per patch; consider more chords'}
                {branch > 0 && ' — upper bound: child rays are half-length and only spawn'
                  + ' from chords that crossed something'}
                {k > 9 && ' — k>9: beyond calibrated range (verdicts still self-calibrated)'}
              </div>
            );
          })()}
          <textarea value={promptText}
                    onChange={(e) => setPromptText(e.target.value)}
                    placeholder={`optional: pin ${k} prompts, one per line (blank = draw from the pool)`}
                    rows={2}
                    style={{ width: '100%', marginTop: 6, background: '#0a0a1a',
                             color: '#fff', border: '1px solid #444', borderRadius: 3,
                             fontSize: 12, padding: 4, resize: 'vertical' }} />
          {err && <div style={{ color: '#f66', marginTop: 4 }}>{err}</div>}

          {status && (
            <div style={{ marginTop: 6 }}>
              <PhaseBar status={status} />
              <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap',
                            color: '#aaa', marginTop: 4 }}>
                <span style={{ color: status.status === 'error' ? '#f66' : '#6a6' }}>
                  {status.status}
                </span>
                <span>{status.generated} images</span>
                <span title="chord probe spacing this run was started with (0.025 = protocol of record ≈ 1 fine cell)">
                  stride {(status.stride ?? STRIDE_REF).toFixed(3)}
                </span>
                {status.bg_p95 !== null && (
                  <span title="background = the run's own measured drift at the same step size; certified means B beats its 95th percentile">
                    background p95 {status.bg_p95.toFixed(2)}
                  </span>
                )}
                {scored.length > 0 && (
                  <span style={{ color: ACCENT }}>
                    {sigCount}/{scored.length} certified
                  </span>
                )}
                {uncert && (
                  <span style={{ color: '#c9a227' }} title={CERTIFY_TIP}>
                    {status.crossings.length} crossings · detection only · uncertified
                  </span>
                )}
                {status.unexplored_share !== null && status.distinct_ridges !== null && (
                  <span title={'Good-Turing certificate: the share of distinct ridges crossed exactly once estimates the boundary area this survey never crossed. Low = coverage saturated.'
                    + (status.stats_scope === 'roots'
                       ? ' Computed from the ROOT chords alone: a child ray starts on a boundary, so it is a preferential sample and would bias an estimate that reads chords as fair samples of boundary area.'
                       : '')}
                        style={{ color: status.unexplored_share <= 0.15
                                   ? ACCENT : '#c9a227' }}>
                    {status.distinct_ridges} ridges · ≈{Math.round(status.unexplored_share * 100)}% unexplored
                    {status.stats_scope === 'roots' && ' (roots only)'}
                  </span>
                )}
                {status.traced_unexplored_share != null && status.traced_ridges != null && (
                  <span title={`EXPERIMENTAL -- recomputed after the trace phase: ${status.trace_links?.length ?? 0} crossing pairs were joined by a walk (${status.traces?.length ?? 0} walks). A development test against a dense ground truth showed no reliable improvement over the plain estimate, and walk links can merge different ridges; trust the map more than this number.`}
                        style={{ color: status.traced_unexplored_share <= 0.15 ? ACCENT : '#c9a227' }}>
                    → traced: {status.traced_ridges} ridges · ≈{Math.round(status.traced_unexplored_share * 100)}% unexplored
                    {status.stats_scope === 'roots' && ' (roots only)'}
                  </span>
                )}
                {status.phase === 'trace' && running && (
                  <span style={{ color: WARN }}>tracing ridges…</span>
                )}
              </div>
              {status.prompts.length > 0 && (
                <div style={{ color: '#666', marginTop: 4, fontSize: 11 }}>
                  {status.prompts.join('  •  ')}
                </div>
              )}

              {running && status.recent_thumbs.length > 0 && runId && (
                <div style={{ marginTop: 8 }}>
                  <div style={{ fontSize: 10, color: '#667', marginBottom: 3,
                                letterSpacing: '0.08em' }}>
                    JUST GENERATED
                  </div>
                  <div style={{ display: 'flex', gap: 3, overflow: 'hidden' }}>
                    {[...status.recent_thumbs].reverse().map((t) => (
                      <img key={t} src={cascadeImageUrl(runId, t)} width={44}
                           height={44}
                           onClick={() => openDetail(t, `image #${t}`, [])}
                           style={{ objectFit: 'cover', borderRadius: 3,
                                    opacity: 0.9, cursor: 'pointer' }} />
                    ))}
                  </div>
                </div>
              )}

              {status.chords.length > 0 && runId && (
                <div style={{ marginTop: 10, display: 'flex', gap: 10, fontSize: 11,
                              alignItems: 'center', flexWrap: 'wrap' }}>
                  <label className="rx-focus"
                         title="Re-read this run's own chords as a local boundary-density map: crossings per unit chord length, kernel-smoothed, converted to boundary area per unit volume (Crofton). Costs no images — the survey already paid for the geometry. Calibration of record: search_problem/outputs/h23_local_sv_kde."
                         style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                    <input type="checkbox" checked={svOn}
                           onChange={(e) => setSvOn(runId, e.target.checked)} />
                    boundary density
                  </label>
                  <label className="rx-focus"
                         title="ranking = colour by percentile, reliable from about 20 chords; calibrated = colour by S_V units, which only mean something once the survey has 80+ chords"
                         style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                    read as
                    <select value={svCal ? svView : 'ranking'} disabled={!svOn}
                            onChange={(e) => setSvView(e.target.value as 'ranking' | 'calibrated')}
                            style={{ fontSize: 11 }}>
                      <option value="ranking">ranking (percentile)</option>
                      <option value="calibrated" disabled={!svCal}
                              title="needs ≥ 80 chords (calibrated at k = 4)">
                        calibrated (S_V units)
                      </option>
                    </select>
                  </label>
                  {svOn && sv && (
                    <span style={{ color: svCal ? '#889' : '#c9a227' }}
                          title={svCal ? undefined : 'needs ≥ 80 chords (calibrated at k = 4)'}>
                      {sv.n_chords} chords · h={sv.h.toFixed(2)}
                      {svCal ? '' : ' · ranking only: needs ≥ 80 chords (calibrated at k = 4)'}
                    </span>
                  )}
                  {svOn && svStatus === 'running' && (
                    <span style={{ color: '#667' }}>reading the chords…</span>
                  )}
                  {svOn && svStatus === 'error' && svError && (
                    <span style={{ color: '#c9a227' }}>{svError}</span>
                  )}
                </div>
              )}

              {status.chords.length > 0 && runId && (
                <div style={{ marginTop: 6 }}>
                  <CascadeMap status={status} runId={runId}
                              imageUrl={cascadeImageUrl} size={640}
                              selectedCid={selCid}
                              onSelect={select}
                              onImage={openDetail}
                              walkPath={walkPath}
                              walkSegs={walk?.segs ?? null}
                              walkColors={walkColors} />
                </div>
              )}

              {selCid !== null && status.status === 'complete' && runId && (
                <div style={{ marginTop: 8, padding: '8px 10px', background: '#0d0d20',
                              border: '1px solid #333', borderRadius: 4 }}>
                  <span style={{ color: '#aaa', marginRight: 10 }}>
                    walk the boundary from crossing {selCid}
                    <span title="Continuation (default): each station is re-found across the line through the last two, so the walk bends with the ridge; stations appear as they are found (~6 images each). Pre-registered test: ~26% further along boundaries than the straight fan (2 of 3 prompt sets; the third was a tie), at ~1.8x the images. Honest caveat: ridge IDENTITY is soft at this fidelity -- near junctions the walk may slide onto an adjacent boundary; it stops with a note when it notices."
                          style={{ color: '#667', cursor: 'help' }}> ⓘ</span>
                  </span>
                  <label className="rx-focus" title="continuation = step along the secant of the last two captured points and re-find the ridge across it, one step at a time (follows curving ridges; stations appear as they are found); fan = all stations on one straight line, rendered at once (old)" style={{ fontSize: 11, display: "inline-flex", alignItems: "center", gap: 4, marginRight: 8 }}>
                    walk <select value={walkMode} onChange={(e) => setWalkMode(e.target.value as "fan" | "continuation")} style={{ fontSize: 11 }}><option value="continuation">continuation (new)</option><option value="fan">straight fan (old)</option></select>
                  </label>
                  <label className="rx-focus" title="how a station is accepted as the same ridge: relative = each new side closer to its matching origin side than to the opposite one (won its A/B 24/24); strict = also each side may change by at most half the previous station's contrast (dev test against a dense ground truth: about half the ridge switches, about a quarter less distance); absolute = both sides within cosine distance 0.35 of the origin sides (the old rule)" style={{ fontSize: 11, display: "inline-flex", alignItems: "center", gap: 4, marginRight: 8 }}>

                    ridge test <select value={sigMode} onChange={(e) => setSigMode(e.target.value as "relative" | "absolute" | "continuity")} style={{ fontSize: 11 }}><option value="relative">relative</option><option value="continuity">strict: must continue last station</option><option value="absolute">absolute (old)</option></select>

                  </label>
                  <label className="rx-focus" title="one exact JVP at the origin (~30 s) gives the true normal and tangent plane instead of the bracket chord" style={{ fontSize: 11, display: "inline-flex", alignItems: "center", gap: 4, marginRight: 8 }}>

                    <input type="checkbox" checked={useJvp} onChange={(e) => setUseJvp(e.target.checked)} /> JVP normal

                  </label>
                  <button onClick={() => startWalk(-1)}
                          disabled={walk?.status === 'running' || uncert}
                          title={uncert ? UNCERT : undefined}
                          style={{ background: '#0f3460', color: '#fff',
                                   border: `1px solid ${WARN}`, borderRadius: 3,
                                   padding: '2px 10px', marginRight: 6,
                                   opacity: uncert ? 0.5 : 1,
                                   cursor: uncert ? 'default' : 'pointer' }}>
                    ← 5 steps
                  </button>
                  <button onClick={() => startWalk(1)}
                          disabled={walk?.status === 'running' || uncert}
                          title={uncert ? UNCERT : undefined}
                          style={{ background: '#0f3460', color: '#fff',
                                   border: `1px solid ${WARN}`, borderRadius: 3,
                                   padding: '2px 10px', opacity: uncert ? 0.5 : 1,
                                   cursor: uncert ? 'default' : 'pointer' }}>
                    5 steps →
                  </button>
                  {uncert && (
                    <span style={{ color: '#c9a227', marginLeft: 10, fontSize: 11 }}>
                      {UNCERT}
                    </span>
                  )}
                  {walk && (
                    <span style={{ color: '#888', marginLeft: 10 }}>
                      {walk.status === 'running'
                        ? `walking… ${walk.steps.length} steps`
                        : `${walk.status} · ${walk.steps.length} steps`}
                      {(walk.images ?? 0) > 0 && ` · ${walk.images} images`}
                      {walk.notes.length > 0 && ` · ${walk.notes[walk.notes.length - 1]}`}
                    </span>
                  )}
                  {walk && walk.status === 'complete' && walk.steps.length > 0 && (
                    walk.cert?.status === 'done' ? (
                      <span style={{ marginLeft: 10, fontSize: 11 }}
                            title={`each station re-rendered on 3 seeds the walk never used; it holds if its boundary strength B beats 95% of random pairs at those seeds (threshold ${walk.cert.threshold?.toFixed(2) ?? '?'})`}>
                        <span style={{ color: ACCENT }}>
                          {walk.cert.significant?.filter(Boolean).length ?? 0}/{walk.steps.length}
                        </span>
                        <span style={{ color: '#888' }}> stations hold on unseen seeds</span>
                      </span>
                    ) : walk.cert?.status === 'error' ? (
                      <span style={{ color: WARN, marginLeft: 10, fontSize: 11 }}>
                        certification failed: {walk.cert.error}
                      </span>
                    ) : (
                      <button className="rx-focus" onClick={certifyWalk}
                              disabled={walk.cert?.status === 'running'}
                              title="Is this a real boundary or a one-seed accident? Re-renders every station on 3 seeds the walk never saw and compares its boundary strength with random pairs at those seeds. Stations turn green (holds) or grey (does not)."
                              style={{ background: '#0f3460', color: '#fff', marginLeft: 10,
                                       border: `1px solid ${ACCENT}`, borderRadius: 3,
                                       padding: '2px 10px', cursor: 'pointer', fontSize: 11 }}>
                        {walk.cert?.status === 'running' ? 'certifying…' : 'certify on 3 unseen seeds (~15 s)'}
                      </button>
                    )
                  )}
                  {walk && walk.steps.length > 0 && (
                    <div style={{ display: 'flex', gap: 4, marginTop: 8,
                                  overflowX: 'auto' }}>
                      {walk.steps.map((s, i) => s.thumb >= 0 && (
                        <div key={i} style={{ textAlign: 'center' }}>
                          <img src={cascadeImageUrl(runId, s.thumb)} width={72}
                               height={72}
                               onClick={() => openDetail(s.thumb,
                                 `walk step ${i + 1}`,
                                 [['local contrast (walk seed)', s.contrast.toFixed(3)],
                                  ...(certOk(i) === null ? [] : [
                                    ['B on 3 unseen seeds', walk.cert?.b?.[i] != null ? walk.cert.b[i]!.toFixed(3) : 'n/a'],
                                    ['holds on unseen seeds', certOk(i) ? 'yes' : 'no'],
                                  ] as [string, string][])],
                                 s.weights)}
                               style={{ objectFit: 'cover', borderRadius: 3,
                                        border: certOk(i) === null ? `1px solid ${WARN}`
                                          : `2px solid ${certOk(i) ? ACCENT : CERT_NO}`,
                                        opacity: certOk(i) === false ? 0.6 : 1,
                                        cursor: 'pointer' }} />
                          <div style={{ fontSize: 10, color: certOk(i) === null ? '#888' : certOk(i) ? ACCENT : CERT_NO }}>
                            {certOk(i) === null ? s.contrast.toFixed(2)
                              : `B ${walk.cert?.b?.[i] != null ? walk.cert.b[i]!.toFixed(2) : '–'}`}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {sorted.length > 0 && runId && (
                <div style={{ marginTop: 10 }}>
                  <div style={{ fontSize: 10, color: '#667', marginBottom: 4,
                                letterSpacing: '0.08em' }}>
                    BOUNDARIES FOUND — strongest first · click to locate on the map
                  </div>
                  <div style={{ display: 'grid', gap: 6,
                                gridTemplateColumns: 'repeat(auto-fill, minmax(96px, 1fr))',
                                maxHeight: 240, overflowY: 'auto' }}>
                    {sorted.map((c) => (
                      <CrossingCard key={c.cid} c={c} runId={runId}
                                    selected={selCid === c.cid}
                                    uncertified={uncert}
                                    onClick={() => {
                                      select(c.cid);
                                      openDetail(c.thumb, `crossing ${c.cid}`, [
                                        ['boundary strength B',
                                         c.b !== null ? c.b.toFixed(3) : 'unscored'],
                                        ['certified',
                                         c.significant ? 'yes — beats background p95' : 'no'],
                                        ...(c.ridge_group !== null
                                          ? [['ridge group', `r${c.ridge_group}`] as [string, string]]
                                          : []),
                                      ], c.weights);
                                    }} />
                    ))}
                  </div>
                </div>
              )}

              {status.patches.length > 0 && runId && (
                <div style={{ marginTop: 10 }}>
                  <div style={{ fontSize: 10, color: '#667', marginBottom: 4,
                                letterSpacing: '0.08em' }}>
                    REFINED BOUNDARIES — 5×5 walks across each
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
                    {status.patches.map((p) => (
                      <PatchGrid key={p.region} runId={runId} patch={p}
                                 running={status?.status === 'running'}
                                 onCell={(t, ri, ci) => openDetail(t,
                                   `patch ${p.exploration ? '(exploration)' : p.region} · row ${ri + 1}, step ${ci + 1}`,
                                   [['patch B', p.b.toFixed(3)],
                                    ['certified', p.significant ? 'yes' : 'no'],
                                    ['position', `${ci + 1}/5 across the boundary, row ${ri + 1}/5 along it`]])} />
                    ))}
                  </div>
                </div>
              )}

              {status.notes.length > 0 && (
                <div style={{ color: '#666', marginTop: 8, fontSize: 11 }}>
                  {status.notes[status.notes.length - 1]}
                </div>
              )}
            </div>
          )}
        </div>
      )}
      {detail && runId && status && (
        <div onClick={() => setDetail(null)}
             style={{ position: 'fixed', inset: 0, background: 'rgba(5,5,18,0.88)',
                      zIndex: 60, display: 'flex', alignItems: 'center',
                      justifyContent: 'center', padding: 20 }}>
          <div onClick={(e) => e.stopPropagation()}
               style={{ background: '#12122c', border: '1px solid #34346a',
                        borderRadius: 8, padding: 16, display: 'flex', gap: 16,
                        flexWrap: 'wrap', maxWidth: '90vw', maxHeight: '90vh',
                        overflow: 'auto' }}>
            <img src={cascadeImageUrl(runId, detail.thumb)}
                 style={{ width: 'min(60vh, 512px)', height: 'min(60vh, 512px)',
                          objectFit: 'cover', borderRadius: 6 }} />
            <div style={{ minWidth: 240, maxWidth: 320, fontSize: 12,
                          color: '#c9d1f0' }}>
              <div style={{ fontSize: 14, marginBottom: 8, color: '#fff' }}>
                {detail.title}
              </div>
              {detail.rows.map(([l, v]) => (
                <div key={l} style={{ display: 'flex', justifyContent: 'space-between',
                                      marginBottom: 3, gap: 12 }}>
                  <span style={{ color: '#8a93b8' }}>{l}</span>
                  <span>{v}</span>
                </div>
              ))}
              <div style={{ margin: '10px 0 4px', color: '#8a93b8' }}>recipe</div>
              {detail.loading && <div style={{ color: '#667' }}>looking up…</div>}
              {!detail.loading && !detail.weights && (
                <div style={{ color: '#667' }}>
                  recipe unavailable for this image
                </div>
              )}
              {detail.weights && status.prompts.map((pr, i) => (
                <div key={i} style={{ marginBottom: 4 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between',
                                gap: 10 }}>
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis',
                                   whiteSpace: 'nowrap', maxWidth: 230 }}>{pr}</span>
                    <span style={{ color: '#8a93b8',
                                   fontVariantNumeric: 'tabular-nums' }}>
                      {((detail.weights![i] ?? 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div style={{ height: 4, background: '#1a1a2e', borderRadius: 2 }}>
                    <div style={{ width: `${Math.min(100, (detail.weights![i] ?? 0) * 100)}%`,
                                  height: '100%', background: ACCENT,
                                  borderRadius: 2 }} />
                  </div>
                </div>
              ))}
              <div style={{ display: 'flex', gap: 8, marginTop: 14 }}>
                <button disabled={!detail.weights || running}
                        onClick={() => detail.weights && exploreFrom(detail.weights)}
                        title="starts a NEW focused run: same prompts, chords confined to a ball (radius 0.18) around this recipe -- a zoomed-in survey of this image's neighbourhood"
                        style={{ background: '#0f3460', color: '#fff',
                                 border: `1px solid ${ACCENT}`, borderRadius: 3,
                                 padding: '4px 12px',
                                 cursor: detail.weights && !running ? 'pointer' : 'default',
                                 opacity: detail.weights && !running ? 1 : 0.5 }}>
                  ⌖ explore from here
                </button>
                <button onClick={() => setDetail(null)}
                        style={{ background: '#0a0a1a', color: '#8a9',
                                 border: '1px solid #444', borderRadius: 3,
                                 padding: '4px 12px', cursor: 'pointer' }}>
                  close
                </button>
              </div>
              {running && (
                <div style={{ color: '#667', marginTop: 6 }}>
                  wait for the current run to finish before exploring from here
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
