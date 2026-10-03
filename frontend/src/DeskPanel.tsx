/**
 * DeskPanel -- the Mixing desk: "what happens if I turn prompt i up or down?"
 *
 * WeightLifter (Pajer et al., IEEE TVCG 2017, sections 4.2-4.5): one slider per criterion, each
 * a straight line through the current weighting along which that criterion is traded against
 * all the others with their proportions kept, and black lines inside the slider wherever the
 * best outcome switches. Here the criteria are the k prompts and the outcome is the image: each
 * slider is prompt i's line w_i = alpha, w_j = w0_j (1 - alpha)/(1 - w0_i), sampled every dalpha
 * on the cheap field (the Cascade's chord probes), cut wherever two consecutive samples are
 * further apart than the Cascade's COS_T. Each stretch between cuts shows one image.
 *
 * Dragging a marker previews the mix it would make; RELEASING it moves the current mix there
 * and recomputes the other k - 1 lines (the dragged prompt's own line is the same line, so it
 * comes from the cache, as does every line and mix visited before). Nothing is computed while
 * dragging.
 *
 * The readout -- for each prompt, how far it can be turned up or down before the image changes
 * identity -- is a single-seed reading of a margin whose calibration is the open h26a study, so
 * it is labelled "exploratory — calibration pending (h26a)" wherever it shows.
 */
import { useEffect, useRef, useState } from 'react';
import { deskStart, deskMove, deskStatus, deskCancel, deskImageUrl } from './api/client';
import type { DeskStatus, DeskLine } from './api/types';
import { divColor } from './CascadeMap';
import { useHandoff } from './stores/handoffStore';
import { feature } from './featureFlags';
import ProbeModeControl from './components/ProbeModeControl';
import { PROBE_DEFAULTS, type ProbeMode } from './probeDefaults';

const BOX: React.CSSProperties = {
  background: '#16213e', border: '1px solid #333', borderRadius: 4,
  padding: '8px 12px', margin: '0 12px 8px', fontSize: 12,
};
const NUM: React.CSSProperties = {
  width: 66, background: '#0a0a1a', color: '#fff', border: '1px solid #444',
  borderRadius: 3, padding: '2px 4px', fontSize: 12,
};
const SEL: React.CSSProperties = {
  background: '#0a0a1a', color: '#8a9', border: '1px solid #444', borderRadius: 3,
  padding: '2px 4px', fontSize: 12,
};
const ACCENT = '#4ecca3';
const WARN = '#e94560';
/** WeightLifter's black switch lines: the UI's existing darkest token */
const CUT = '#0a0a1a';
const GREY = '#8a93b8';
const POLL_MS = 1200;
const TRACK_W = 560;
const TRACK_H = 46;
/** desk.py's accounting, mirrored so the projection here is the one the backend refuses on */
const PROBE_COST = 0.5;
const REFINE_MAX_PER_LINE = 4;
const REFINE_IMAGES = 5;
const REFINE_ROUNDS = 3;
const MAX_REQUEST = 300;
const READOUT_LABEL = 'exploratory — calibration pending (h26a)';

const DALPHA_TIP = 'spacing of the samples along every slider (both ends always sampled). Each '
  + 'sample is one cheap probe (0.5 image-eq), so a slider costs 1/Δα + 1 probes.';
const REFINE_TIP = 'bisect every change (at most 4 per slider, strongest first) to Δα/8 with '
  + 'full-fidelity images: the two ends re-rendered at full steps, then three midpoints — the '
  + 'Cascade\'s rebracket + bisection. A change the full-fidelity ends do not confirm is drawn '
  + 'dashed.';

function short(p: string, n = 30) {
  return p.length > n ? p.slice(0, n - 1) + '…' : p;
}

/** WeightLifter's slider line (desk.line_point): prompt i at alpha, the others' ratios kept. */
function linePoint(w0: number[], i: number, a: number): number[] {
  const k = w0.length;
  const rest = 1 - w0[i];
  const w = rest > 1e-9 ? w0.map((v) => (v * (1 - a)) / rest) : w0.map(() => (1 - a) / (k - 1));
  w[i] = a;
  return w;
}

/** desk.projected_cost: k x samples cheap probes + the mix, + refine's worst case. Staged:
 *  every sample read out (t/S) and, worst case, finished ((S-t)/S), the mix read out and
 *  finished, refine = its REFINE_ROUNDS midpoints (the change's ends are finished already). */
function projection(k: number, dalpha: number, refine: boolean, cheap: boolean,
                    cachedLines = 0, staged: { t: number; steps: number } | null = null) {
  const n = Math.ceil(1 / dalpha - 1e-9) + 1;
  const lines = Math.max(0, k - cachedLines);
  if (staged) {
    const probes = lines * n + 1;     // t/S + (S-t)/S per sample, worst case, + the mix
    const ref = refine ? lines * REFINE_MAX_PER_LINE * REFINE_ROUNDS : 0;
    return { n, probes, ref, total: probes + ref,
             readouts: (lines * n * staged.t) / staged.steps };
  }
  const unit = cheap ? PROBE_COST : 1;
  const probes = unit * (lines * n + 1);
  const ref = refine ? lines * REFINE_MAX_PER_LINE * REFINE_IMAGES : 0;
  return { n, probes, ref, total: probes + ref, readouts: 0 };
}

function parseWeights(text: string, k: number): { w: number[] | null; err: string | null } {
  const t = text.trim();
  if (!t) return { w: null, err: null };
  const v = t.split(/[\s,;]+/).filter(Boolean).map(Number);
  if (v.length !== k || v.some((x) => !Number.isFinite(x))) {
    return { w: null, err: `mix needs ${k} numbers (got ${v.length})` };
  }
  if (v.some((x) => x < 0)) return { w: null, err: 'weights must be non-negative' };
  const s = v.reduce((a, b) => a + b, 0);
  if (s <= 0) return { w: null, err: 'weights sum to 0' };
  return { w: v.map((x) => x / s), err: null };
}

/** A press that moves less than this is a click, not a drag (the Cascade map's rule). */
const DRAG_PX = 4;

/**
 * One WeightLifter slider: the line's stretches (one image each), the cuts between them, a
 * strip of the sample-to-sample divergence underneath (the maps' ramp), and the marker.
 *
 * Pointer: a drag anywhere on the track previews the mix (nothing is computed) and RELEASING
 * moves there; a click on a stretch's image enlarges it; a click elsewhere on the track moves
 * the marker to that point, as a release would.
 */
function Slider({ line, deskId, drag, onDrag, onRelease, onImage }: {
  line: DeskLine; deskId: string; drag: number | null;
  onDrag: (a: number | null) => void; onRelease: (a: number) => void;
  onImage: (img: number, title: string) => void;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const press = useRef<{ x: number; img: number | null; title: string; moved: boolean } | null>(
    null);
  const alphaAt = (clientX: number) => {
    const r = ref.current?.getBoundingClientRect();
    if (!r) return line.alpha0;
    return Math.min(1, Math.max(0, (clientX - r.left) / r.width));
  };
  const x = (a: number) => a * TRACK_W;
  return (
    <div ref={ref} style={{ position: 'relative', width: TRACK_W, height: TRACK_H + 12,
                            flex: 'none', touchAction: 'none', cursor: 'ew-resize' }}
         onPointerDown={(e) => {
           const t = e.target as HTMLElement;
           const img = t.dataset?.img !== undefined ? Number(t.dataset.img) : null;
           press.current = { x: e.clientX, img, title: t.dataset?.title ?? '', moved: false };
           ref.current?.setPointerCapture?.(e.pointerId);
         }}
         onPointerMove={(e) => {
           const p = press.current;
           if (!p) return;
           if (!p.moved && Math.abs(e.clientX - p.x) < DRAG_PX) return;
           p.moved = true;
           onDrag(alphaAt(e.clientX));
         }}
         onPointerUp={(e) => {
           const p = press.current;
           press.current = null;
           if (!p) return;
           onDrag(null);
           if (!p.moved && p.img !== null) { onImage(p.img, p.title); return; }
           onRelease(alphaAt(e.clientX));
         }}
         onPointerCancel={() => { press.current = null; onDrag(null); }}>
      {/* the stretches between changes, one image each */}
      {line.segments.map((s, j) => {
        const w = Math.max(1, x(s.alpha_hi) - x(s.alpha_lo));
        const side = Math.min(TRACK_H - 6, w - 4);
        return (
          <div key={j}
               title={`α ${s.alpha_lo.toFixed(2)}–${s.alpha_hi.toFixed(2)}: one identity along `
                      + 'this stretch (click the image to enlarge)'}
               style={{ position: 'absolute', left: x(s.alpha_lo), top: 0, width: w,
                        height: TRACK_H, background: '#1a1a2e', borderRadius: 3,
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        overflow: 'hidden' }}>
            {s.thumb >= 0 && side >= 6 && (
              <img src={deskImageUrl(deskId, s.thumb)} width={side} height={side}
                   draggable={false} data-img={s.thumb}
                   data-title={`P${line.i + 1} · α ${s.alpha_lo.toFixed(2)}–`
                               + `${s.alpha_hi.toFixed(2)}`}
                   style={{ objectFit: 'cover', borderRadius: 2, cursor: 'zoom-in' }} />
            )}
          </div>
        );
      })}
      {/* sample-to-sample divergence, on the maps' own ramp */}
      {line.divs.map((d, t) => (
        <div key={`d${t}`}
             title={line.resumed && line.resumed.length
               ? (line.resumed[t] && line.resumed[t + 1] ? 'exact (both samples finished)'
                 : 'x̂0 readout only (not finished)') : undefined}
             style={{ position: 'absolute', left: x(line.alphas[t]), top: TRACK_H + 2,
                      width: Math.max(1, x(line.alphas[t + 1]) - x(line.alphas[t])), height: 4,
                      background: d === null ? '#1a1a2e' : divColor(d),
                      // staged: a faint bar is the x̂0 readout of unfinished samples
                      opacity: line.resumed && line.resumed.length
                        && !(line.resumed[t] && line.resumed[t + 1]) ? 0.45 : 1 }} />
      ))}
      {/* WeightLifter's switch lines */}
      {line.flips.map((f, j) => (
        <div key={`f${j}`}
             title={`change at α ${f.alpha.toFixed(3)} (±${(f.width / 2).toFixed(3)}): 1 − cos `
                    + `${f.div.toFixed(2)} ${line.resumed && line.resumed.length
                      ? 'between finished samples (exact labels)' : 'on the cheap field'}`
                    + (f.refined ? `; full fidelity ${f.full_div?.toFixed(2) ?? '—'}`
                      + (f.confirmed === false ? ' — NOT confirmed at full fidelity' : '') : '')}
             style={{ position: 'absolute', left: x(f.alpha) - 2, top: -2, width: 4,
                      height: TRACK_H + 4, background: f.confirmed === false ? 'transparent' : CUT,
                      borderLeft: f.confirmed === false ? `2px dashed ${CUT}` : undefined,
                      boxShadow: `0 0 0 1px ${GREY}`, borderRadius: 1, pointerEvents: 'none' }} />
      ))}
      {/* the current mix, and the ghost of a drag */}
      <div title={`current: P${line.i + 1} = ${line.alpha0.toFixed(3)} — drag and release to move`}
           style={{ position: 'absolute', left: x(line.alpha0) - 1, top: -6, width: 2,
                    height: TRACK_H + 12, background: ACCENT, pointerEvents: 'none' }} />
      <div style={{ position: 'absolute', left: x(line.alpha0) - 5, top: -10, width: 0,
                    height: 0, borderLeft: '5px solid transparent',
                    borderRight: '5px solid transparent', borderTop: `7px solid ${ACCENT}`,
                    pointerEvents: 'none' }} />
      {drag !== null && (
        <div style={{ position: 'absolute', left: x(drag) - 1, top: -6, width: 2,
                      height: TRACK_H + 12, background: '#fff', opacity: 0.8,
                      pointerEvents: 'none' }} />
      )}
    </div>
  );
}

export default function DeskPanel() {
  const [open, setOpen] = useState(false);
  const [k, setK] = useState(4);
  const [promptText, setPromptText] = useState(
    'a photograph of a lighthouse\na pencil sketch of a cathedral\n'
    + 'an oil painting of a forest\na neon city at night');
  const [mixText, setMixText] = useState('');
  const [dalpha, setDalpha] = useState(0.05);
  const [refine, setRefine] = useState(false);
  const [seed, setSeed] = useState(42);
  const [steps, setSteps] = useState(8);
  const [cheap, setCheap] = useState(true);
  const [probeMode, setProbeMode] = useState<ProbeMode>(PROBE_DEFAULTS.mode);
  const [stagedT, setStagedT] = useState(PROBE_DEFAULTS.stagedT);
  const [stagedTheta, setStagedTheta] = useState(PROBE_DEFAULTS.stagedTheta);
  const [deskId, setDeskId] = useState<string | null>(null);
  const [status, setStatus] = useState<DeskStatus | null>(null);
  const [kick, setKick] = useState(0);
  const [err, setErr] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [drag, setDrag] = useState<{ line: number; alpha: number } | null>(null);
  const [detail, setDetail] = useState<{ img: number; title: string } | null>(null);
  const timer = useRef<number | null>(null);
  const openInMicroscope = useHandoff((st) => st.openInMicroscope);

  useEffect(() => {
    if (!deskId) return;
    let live = true;
    const tick = async () => {
      try {
        const st = await deskStatus(deskId);
        if (!live) return;
        setStatus(st);
        if (st.status === 'running' || st.status === 'pending') {
          timer.current = window.setTimeout(tick, POLL_MS);
        }
      } catch {
        if (live) timer.current = window.setTimeout(tick, 4000);
      }
    };
    tick();
    return () => {
      live = false;
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [deskId, kick]);

  const prompts = promptText.split('\n').map((t) => t.trim()).filter(Boolean);
  const mix = parseWeights(mixText, k);
  const stagedNow = probeMode === 'staged'
    ? { t: Math.min(stagedT, Math.max(1, steps - 1)), steps } : null;
  const proj = projection(k, dalpha, refine, cheap, 0, stagedNow);
  const tooBig = proj.total > MAX_REQUEST;
  const running = status?.status === 'running' || status?.status === 'pending';

  const start = async () => {
    setErr(null);
    setMsg(null);
    if (prompts.length !== k) { setErr(`enter exactly ${k} prompts (got ${prompts.length})`); return; }
    if (mix.err) { setErr(mix.err); return; }
    try {
      const r = await deskStart({ prompts, w0: mix.w, dalpha, refine, seed, steps,
                                  probe_steps: cheap ? 4 : null, probe_mode: probeMode,
                                  staged_t: stagedNow ? stagedNow.t : stagedT,
                                  staged_theta: stagedTheta });
      setStatus(null);
      setDeskId(r.desk_id);
      setKick((x) => x + 1);
    } catch (e: any) {
      setErr(String(e.message || e));
    }
  };

  const move = async (body: { w0?: number[]; line?: number; alpha?: number }, why: string) => {
    if (!deskId) return;
    setErr(null);
    try {
      const r = await deskMove(deskId, { ...body, dalpha, refine });
      setMsg(r.status === 'complete' ? `${why}: visited before, served from the cache`
        : `${why}: ${r.cached_lines} of ${status?.k ?? k} lines from the cache, ≤ `
          + `${r.cost_image_eq.toFixed(1)} image-eq`);
      setKick((x) => x + 1);
    } catch (e: any) {
      setErr(String(e.message || e));
    }
  };

  const lines = status?.lines ?? [];
  const sk = status?.k ?? k;
  const runStaged = status
    ? (status.probe_mode === 'staged' ? { t: status.staged_t ?? stagedT, steps: status.steps } : null)
    : stagedNow;
  const moveProj = projection(sk, status?.dalpha ?? dalpha, refine,
                              status ? status.probe_steps !== null : cheap, 1, runStaged);
  const preview = drag && status ? linePoint(status.w0, drag.line, drag.alpha) : null;

  return (
    <div style={BOX}>
      <div style={{ cursor: 'pointer', userSelect: 'none' }} onClick={() => setOpen(!open)}>
        {open ? '▾' : '▸'} Mixing desk — turn one prompt up or down (any k)
      </div>
      {open && (
        <div style={{ marginTop: 8 }}>
          <div style={{ color: '#667', fontSize: 11, marginBottom: 6 }}>
            One slider per prompt, WeightLifter-style (Pajer et al. 2017): each is the straight
            line through the current mix along which that prompt is traded against all the others,
            their proportions kept. Dark cuts mark where the image changes identity (1 − cos &gt;
            0.35 between neighbouring samples, cheap field, single seed). Drag a marker and release
            to move the mix — the other lines are recomputed on release, never while dragging.
          </div>

          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
            <label title="prompts — any k ≥ 3: a desk costs k sliders, not a lattice">
              k
              <select value={k} style={SEL} onChange={(e) => setK(Number(e.target.value))}>
                {[3, 4, 5, 6, 7, 8].map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </label>
            <label title={DALPHA_TIP}>
              Δα <input style={NUM} type="number" min={0.01} max={0.25} step={0.01} value={dalpha}
                        onChange={(e) => setDalpha(Math.max(0.01, Math.min(0.25,
                          Number(e.target.value) || 0.05)))} />
            </label>
            <label className="rx-focus" title={REFINE_TIP}
                   style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
              <input type="checkbox" checked={refine}
                     onChange={(e) => setRefine(e.target.checked)} /> refine to Δα/8
            </label>
            <label title="image seed; same prompts + new seed = a different field">
              seed <input style={NUM} type="number" value={seed}
                          onChange={(e) => setSeed(Number(e.target.value))} />
            </label>
            <label title="denoising steps of a full image; the samples run at 4 unless cheap probes is off">
              steps <input style={NUM} type="number" min={1} max={50} value={steps}
                           onChange={(e) => setSteps(Math.max(1, Math.min(50,
                             Number(e.target.value))))} />
            </label>
            <label className="rx-focus" title={"samples on the cheap field (4 denoising steps), as the Cascade's chord probes are; off = full fidelity, double the cost"
                                               + (probeMode === 'staged' ? ' — unused in staged mode' : '')}
                   style={{ display: 'inline-flex', alignItems: 'center', gap: 4,
                            color: probeMode === 'staged' ? '#667' : undefined }}>
              <input type="checkbox" checked={cheap} disabled={probeMode === 'staged'}
                     onChange={(e) => setCheap(e.target.checked)} />
              cheap probes
            </label>
            <ProbeModeControl mode={probeMode} t={stagedT} theta={stagedTheta} steps={steps}
                              onMode={setProbeMode} onT={setStagedT} onTheta={setStagedTheta} />
            <button onClick={start} disabled={tooBig}
                    style={{ background: tooBig ? '#333' : '#0f3460', color: '#fff',
                             border: `1px solid ${ACCENT}`, borderRadius: 3, padding: '3px 12px',
                             cursor: tooBig ? 'default' : 'pointer' }}>
              {deskId ? 'Start new desk' : 'Start'}
            </button>
            {running && deskId && (
              <button onClick={() => deskCancel(deskId)}
                      style={{ background: '#0a0a1a', color: WARN, border: `1px solid ${WARN}`,
                               borderRadius: 3, padding: '3px 10px', cursor: 'pointer' }}>
                Cancel
              </button>
            )}
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 6 }}>
            <label title="the starting mix, one weight per prompt (normalised); empty = the barycentre">
              mix{' '}
              <input style={{ ...NUM, width: 260 }} value={mixText}
                     placeholder={`barycentre (${(1 / k).toFixed(3)} each)`}
                     onChange={(e) => setMixText(e.target.value)} />
            </label>
            {mix.err && <span style={{ color: '#c9a227', fontSize: 11 }}>{mix.err}</span>}
          </div>
          <textarea value={promptText} onChange={(e) => setPromptText(e.target.value)}
                    placeholder={`pin ${k} prompts, one per line`} rows={Math.min(k, 8)}
                    style={{ width: '100%', marginTop: 6, background: '#0a0a1a', color: '#fff',
                             border: '1px solid #444', borderRadius: 3, fontSize: 12, padding: 4,
                             resize: 'vertical' }} />
          {prompts.length !== k && (
            <div style={{ color: '#c9a227', fontSize: 11 }}>
              {prompts.length} prompts entered, {k} needed
            </div>
          )}
          <div style={{ color: '#667', marginTop: 4, fontSize: 11 }}>
            estimate: {k} sliders × {proj.n} samples + the mix ={' '}
            <span style={{ color: tooBig ? WARN : '#aeb6dd' }}>
              {stagedNow ? '≤' : '≈'}{proj.probes.toFixed(1)} image-eq
            </span>
            {stagedNow && ` (staged: readouts ≈${proj.readouts.toFixed(1)}, the rest only if every `
                         + 'segment were flagged)'}
            {refine && ` + refine ≤ ${proj.ref} full images (at most ${REFINE_MAX_PER_LINE} changes `
                       + `per slider × ${REFINE_IMAGES})`}
            {' '}· each later move ≤ {moveProj.total.toFixed(1)} (the dragged slider is kept)
            {tooBig && (
              <div style={{ color: WARN }}>
                past the {MAX_REQUEST} image-eq ceiling of one request: raise Δα, turn refine off,
                or use fewer prompts
              </div>
            )}
          </div>
          {err && <div style={{ color: '#f66', marginTop: 4 }}>{err}</div>}

          {status && deskId && (
            <div style={{ marginTop: 8 }}>
              <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', alignItems: 'center',
                            fontSize: 11, color: '#aeb6dd', fontVariantNumeric: 'tabular-nums' }}>
                <span style={{ color: status.status === 'error' ? '#f66'
                  : status.status === 'complete' ? '#6a6' : '#c9a227' }}>
                  {running ? 'computing…' : status.status}
                </span>
                <span title="image-eq this desk has spent, against its ceiling; and what the current mix cost">
                  cost {status.cost_image_eq.toFixed(1)} / {status.max_session_image_eq.toFixed(0)}
                  {' '}(this mix {status.position_cost.toFixed(1)})
                </span>
                <span>Δα {status.dalpha}{status.refine ? ' · refined to Δα/8' : ''}</span>
                {status.probe_mode === 'staged' && status.staged && (
                  <span title="staged readout: every sample read at step t of the full schedule; samples of segments whose readouts differ by ≥ θ (and the mix) finished from their cached latent — changes are claimed between finished samples only">
                    staged t {status.staged_t}/{status.steps} · θ {status.staged_theta}
                    {' · '}finished {status.staged.resumed + status.staged.from_scratch}/
                    {status.staged.readouts}
                    {status.staged.resumed_share !== null
                      && ` (${Math.round(100 * status.staged.resumed_share)} %)`}
                  </span>
                )}
                {msg && <span style={{ color: ACCENT }}>{msg}</span>}
                {feature('microscope') && status.w0.length > 0 && (
                  <button className="rx-focus"
                          onClick={() => openInMicroscope({
                            from: 'desk', prompts: status.prompts, weights: status.w0,
                            seed: status.seed, steps: status.steps })}
                          title="look at this mix in the Ridge microscope: a G × G image lattice on a plane through it"
                          style={{ background: '#0a0a1a', color: '#8a9', border: '1px solid #444',
                                   borderRadius: 3, padding: '1px 8px', cursor: 'pointer',
                                   marginLeft: 'auto' }}>
                    ⊞ open this mix in the microscope
                  </button>
                )}
              </div>

              {/* the current mix */}
              <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginTop: 6 }}>
                {status.w0_image >= 0 && (
                  <img src={deskImageUrl(deskId, status.w0_image)} width={56} height={56}
                       onClick={() => setDetail({ img: status.w0_image, title: 'the current mix' })}
                       style={{ objectFit: 'cover', borderRadius: 4, cursor: 'zoom-in',
                                border: `2px solid ${ACCENT}` }} />
                )}
                <div style={{ fontSize: 11, color: '#889', fontVariantNumeric: 'tabular-nums' }}>
                  current mix:{' '}
                  {status.w0.map((w, i) => `P${i + 1} ${w.toFixed(3)}`).join(' · ')}
                  {preview && drag && (
                    <div style={{ color: '#fff' }}>
                      release to move P{drag.line + 1} → {drag.alpha.toFixed(3)}:{' '}
                      {preview.map((w, i) => `P${i + 1} ${w.toFixed(3)}`).join(' · ')}
                      <span style={{ color: '#667' }}>
                        {' '}(≤ {moveProj.total.toFixed(1)} image-eq; slider {drag.line + 1} is kept)
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* the sliders */}
              <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 12 }}>
                {lines.map((ln) => (
                  <div key={ln.i} style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                    <div style={{ width: 170, flex: 'none', fontSize: 11 }} title={ln.prompt}>
                      <div style={{ color: '#c9d1f0' }}>P{ln.i + 1} {short(ln.prompt, 22)}</div>
                      <div style={{ color: '#667', fontVariantNumeric: 'tabular-nums' }}>
                        {ln.alpha0.toFixed(3)} · {ln.flips.length} change
                        {ln.flips.length === 1 ? '' : 's'}
                        {ln.cached ? ' · cached' : ''}
                        {ln.degenerate ? ' · vertex: others shared equally' : ''}
                      </div>
                    </div>
                    <span style={{ color: '#667', fontSize: 10 }}>0</span>
                    <Slider line={ln} deskId={deskId}
                            drag={drag && drag.line === ln.i ? drag.alpha : null}
                            onDrag={(a) => setDrag(a === null ? null : { line: ln.i, alpha: a })}
                            onRelease={(a) => {
                              if (Math.abs(a - ln.alpha0) > 1e-3) {
                                void move({ line: ln.i, alpha: a },
                                          `P${ln.i + 1} → ${a.toFixed(2)}`);
                              }
                            }}
                            onImage={(img, title) => setDetail({ img, title })} />
                    <span style={{ color: '#667', fontSize: 10 }}>1</span>
                  </div>
                ))}
              </div>

              {/* the readout: which prompt flips identity first */}
              {status.readout.length > 0 && (
                <div style={{ marginTop: 12, padding: '6px 8px', background: '#0d0d20',
                              border: '1px solid #2a2a4a', borderRadius: 4 }}>
                  <div style={{ color: '#aeb6dd', marginBottom: 4 }}>
                    nearest flip per prompt{' '}
                    <span style={{ color: '#c9a227', fontSize: 11 }}>
                      — {status.readout_label || READOUT_LABEL}
                    </span>
                  </div>
                  <div style={{ fontSize: 10, color: '#667', marginBottom: 4 }}>
                    how far each prompt can be turned up (↑) or down (↓) from the current mix before
                    the image changes identity, nearest first; the thumbnail is what lies beyond.
                    ±Δα/2 ({(status.dalpha / 2).toFixed(3)}){status.refine
                      ? `, ±Δα/16 (${(status.dalpha / 16).toFixed(4)}) where refined` : ''}.
                  </div>
                  {status.readout.map((r) => (
                    <div key={r.i} style={{ display: 'flex', gap: 12, alignItems: 'center',
                                            padding: '3px 0', borderTop: '1px solid #23234d',
                                            fontVariantNumeric: 'tabular-nums' }}>
                      <span style={{ width: 190, color: '#c9d1f0' }} title={r.prompt}>
                        P{r.i + 1} {short(r.prompt, 24)}
                      </span>
                      {(['up', 'down'] as const).map((way) => {
                        const f = r[way];
                        return (
                          <span key={way} style={{ display: 'inline-flex', gap: 6,
                                                   alignItems: 'center', width: 150 }}>
                            <span style={{ color: f ? '#c9d1f0' : '#555', width: 64 }}>
                              {way === 'up' ? '↑' : '↓'}{' '}
                              {f ? `${way === 'up' ? '+' : '−'}${f.dist.toFixed(3)}` : '—'}
                            </span>
                            {f && f.thumb >= 0 && (
                              <img src={deskImageUrl(deskId, f.thumb)} width={34} height={34}
                                   title={`beyond the change at α ${f.alpha.toFixed(3)}`
                                          + (f.refined ? ' (refined, full fidelity)' : ' (cheap field)')}
                                   onClick={() => setDetail({ img: f.thumb,
                                     title: `P${r.i + 1} ${way === 'up' ? 'up' : 'down'} past α `
                                            + f.alpha.toFixed(3) })}
                                   style={{ objectFit: 'cover', borderRadius: 3,
                                            cursor: 'zoom-in' }} />
                            )}
                          </span>
                        );
                      })}
                    </div>
                  ))}
                </div>
              )}

              {/* mixes visited: a revisit is served from the cache */}
              {status.positions.length > 1 && (
                <div style={{ marginTop: 8, display: 'flex', gap: 6, flexWrap: 'wrap',
                              alignItems: 'center', fontSize: 10 }}>
                  <span style={{ color: '#667' }}>mixes visited:</span>
                  {status.positions.map((p) => (
                    <button key={p.pid} disabled={p.pid === status.position || running}
                            onClick={() => move({ w0: p.w0 }, `back to mix #${p.pid}`)}
                            title={p.w0.map((w, i) => `P${i + 1} ${w.toFixed(3)}`).join(' · ')
                                   + ` · ${p.status} · ${p.cost_image_eq.toFixed(1)} image-eq`}
                            style={{ background: p.pid === status.position ? '#20304d' : '#0a0a1a',
                                     color: p.pid === status.position ? '#fff'
                                       : p.status === 'complete' ? '#8a9' : '#555',
                                     border: p.pid === status.position ? `1px solid ${ACCENT}`
                                       : '1px solid #444',
                                     borderRadius: 3, padding: '1px 6px', cursor: 'pointer' }}>
                      #{p.pid}
                    </button>
                  ))}
                </div>
              )}
              {status.notes.length > 0 && (
                <div style={{ color: '#666', marginTop: 6, fontSize: 11 }}>
                  {status.notes[status.notes.length - 1]}
                </div>
              )}
              {status.error && <div style={{ color: '#f66', marginTop: 4 }}>{status.error}</div>}
            </div>
          )}
        </div>
      )}
      {detail && deskId && (
        <div onClick={() => setDetail(null)}
             style={{ position: 'fixed', inset: 0, background: 'rgba(5,5,18,0.88)', zIndex: 60,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      padding: 20, flexDirection: 'column', gap: 10 }}>
          <img src={deskImageUrl(deskId, detail.img)}
               style={{ width: 'min(70vh, 512px)', height: 'min(70vh, 512px)', objectFit: 'cover',
                        borderRadius: 6 }} />
          <div style={{ color: '#c9d1f0', fontSize: 13 }}>{detail.title} · click to close</div>
        </div>
      )}
    </div>
  );
}
