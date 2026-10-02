/**
 * MicroscopePanel -- the Ridge microscope: an image lattice on a 2-D plane through a point.
 *
 * Every other surface either lives in the whole (k-1)-dimensional simplex, where a lattice costs
 * ~n^(k-1) images, or along lines a person cannot read as a picture. This one cuts a PLANE
 * through one recipe and lays a G x G lattice of full-fidelity images on it, so it costs G^2
 * images at every k. Two precedents shape it:
 *
 *   * Sequential Gallery (Koyama et al., ACM TOG 2020): a 2-D plane shown as a 5 x 5 (or 3 x 3)
 *     image grid; clicking a cell zooms in 2x around it. Here: a breadcrumb of levels, back, and
 *     every level cached (a revisit costs nothing; a zoom reuses the parent's coinciding cells).
 *   * Sohns, Garth & Leitte (CGF 2023): the plane through a boundary is spanned by the PCA of a
 *     BALANCED local set (half each side), so the boundary is in view, under a grey biplot of the
 *     original axes. Here: "crossing" mode -- e1 = the crossing's cloud normal (else its chord),
 *     e2 = the balanced cloud's first principal direction (else random) -- and the biplot shows
 *     which prompt grows in which screen direction.
 *
 * Boundaries between neighbouring cells are the Cascade's own rule (cosine distance past
 * COS_T = 0.35), drawn on the divergence ramp the maps already use, so no new colour means
 * anything new. Single seed: what is drawn here is the Cascade's uncertified kind of boundary.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  microscopePlan, microscopeStart, microscopeStatus, microscopeZoom, microscopeCancel,
  microscopeImageUrl,
} from './api/client';
import type {
  MicroscopeRequest, MicroPlan, MicroStatus, MicroLevel, MicroCell, MicroMode,
} from './api/types';
import { divColor, makeProjector } from './CascadeMap';
import { useHandoff } from './stores/handoffStore';

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
/** the biplot's grey (the cascade map's "not significant" neutral) */
const GREY = '#8a93b8';
const POLL_MS = 1200;
const PLAN_DEBOUNCE_MS = 300;
/** lattice width in px; cells shrink with G */
const LATTICE_PX = 520;
const GAP = 6;
const BAR = 4;

const MODE_TIP = 'how the plane is chosen. crossing: e1 = the crossing\'s hi-res cloud normal '
  + '(else the chord that found it), e2 = the first principal direction of a BALANCED set of its '
  + 'cloud points (half each side, as Sohns et al. 2023) — else random. prompt-swap: e1 trades '
  + 'one prompt for another, e2 another pair. random: two seeded random directions.';
const S_TIP = 'half-width of the lattice in tangent units: the corner cells sit s·√2 from the '
  + 'centre. 0.10 = four Cascade strides from the centre to an edge. Each zoom halves it.';
const GRID_TIP = 'G × G images. 5 × 5 is Sequential Gallery\'s default; 3 × 3 is cheaper, 7 × 7 '
  + 'resolves more. Cost = the cells inside the simplex, one full-fidelity image each.';
const REDRAW_TIP = 'draw a new random e2 (orthogonal to e1, which stays): a different slice '
  + 'through the same point and the same e1.';

function fmt(x: number | null | undefined, d = 3): string {
  return x === null || x === undefined || !Number.isFinite(x) ? '—' : x.toFixed(d);
}

function short(p: string, n = 28) {
  return p.length > n ? p.slice(0, n - 1) + '…' : p;
}

/** "0.2, 0.3, …" -> a recipe over k prompts, or null with nothing typed. */
function parseWeights(text: string, k: number): { w: number[] | null; err: string | null } {
  const t = text.trim();
  if (!t) return { w: null, err: null };
  const v = t.split(/[\s,;]+/).filter(Boolean).map(Number);
  if (v.length !== k || v.some((x) => !Number.isFinite(x))) {
    return { w: null, err: `centre needs ${k} numbers (got ${v.length})` };
  }
  if (v.some((x) => x < 0)) return { w: null, err: 'centre weights must be non-negative' };
  const s = v.reduce((a, b) => a + b, 0);
  if (s <= 0) return { w: null, err: 'centre weights sum to 0' };
  return { w: v.map((x) => x / s), err: null };
}

/** The level chain from the root to `level`, by parent links (the breadcrumb). */
function chain(levels: MicroLevel[], level: number): MicroLevel[] {
  const out: MicroLevel[] = [];
  let cur: MicroLevel | undefined = levels[level];
  while (cur) {
    out.unshift(cur);
    cur = cur.parent === null ? undefined : levels[cur.parent];
  }
  return out;
}

/**
 * The lattice itself: images at their (a, b) positions, e1 to the right and e2 up, outside
 * cells empty, and a bar on the divergence ramp between every two neighbours past COS_T.
 */
function Lattice({ cells, edges, grid, runId, onCell, hover, setHover, preview }: {
  cells: MicroCell[]; edges: MicroLevel['edges']; grid: number; runId: string | null;
  onCell?: (c: MicroCell) => void; hover: string | null; setHover: (k: string | null) => void;
  preview?: boolean;
}) {
  const cell = Math.floor((LATTICE_PX - (grid - 1) * GAP) / grid);
  const x = (ia: number) => ia * (cell + GAP);
  const y = (ib: number) => (grid - 1 - ib) * (cell + GAP);
  const side = grid * cell + (grid - 1) * GAP;
  return (
    <div style={{ position: 'relative', width: side, height: side, flex: 'none' }}>
      {cells.map((c) => {
        const key = `${c.ia},${c.ib}`;
        const centre = c.a === 0 && c.b === 0;
        const style: React.CSSProperties = {
          position: 'absolute', left: x(c.ia), top: y(c.ib), width: cell, height: cell,
          borderRadius: 3, boxSizing: 'border-box',
          outline: hover === key && c.inside && !preview ? '2px solid #fff'
            : centre ? `1px solid ${ACCENT}` : undefined,
          cursor: c.inside && onCell ? 'pointer' : 'default',
        };
        const tip = c.inside
          ? `a ${c.a.toFixed(2)}, b ${c.b.toFixed(2)}`
            + (c.zoom_cost === 0 ? ' · zoom: level already rendered (free)'
              : c.zoom_cost != null ? ` · zoom ×2: ${c.zoom_cost} new images` : '')
          : 'outside the simplex — not a recipe, not rendered, not charged';
        if (!c.inside) {
          return (
            <div key={key} title={tip}
                 style={{ ...style, background: '#0d0d20', border: '1px dashed #34346a',
                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                          color: '#667', fontSize: 10 }}>
              outside
            </div>
          );
        }
        return (
          <div key={key} title={tip} onClick={() => onCell?.(c)}
               onMouseEnter={() => setHover(key)} onMouseLeave={() => setHover(null)}
               style={{ ...style, background: '#1a1a2e', overflow: 'hidden' }}>
            {c.image >= 0 && runId && (
              <img src={microscopeImageUrl(runId, c.image)} width={cell} height={cell}
                   style={{ objectFit: 'cover', display: 'block' }} />
            )}
          </div>
        );
      })}
      {edges.filter((e) => e.boundary).map((e) => {
        const [ia, ib] = e.a;
        const [ja, jb] = e.b;
        const vertical = ja !== ia;        // neighbours along e1 -> a vertical bar between them
        const st: React.CSSProperties = vertical
          ? { left: x(ja) - GAP / 2 - BAR / 2, top: y(ib), width: BAR, height: cell }
          : { left: x(ia), top: y(jb) + cell + GAP / 2 - BAR / 2, width: cell, height: BAR };
        return (
          <div key={`${ia},${ib}-${ja},${jb}`}
               title={`boundary: 1 − cos ${e.div.toFixed(2)} > 0.35 between these two cells`}
               style={{ position: 'absolute', ...st, borderRadius: 2,
                        background: divColor(e.div), pointerEvents: 'none' }} />
        );
      })}
    </div>
  );
}

/** Grey biplot: per prompt, the screen direction in which its weight grows on this plane. */
function Biplot({ arrows, prompts }: { arrows: number[][]; prompts: string[] }) {
  const S = 160, c = S / 2;
  const mx = Math.max(1e-9, ...arrows.map(([a, b]) => Math.hypot(a, b)));
  const sc = (S / 2 - 22) / mx;
  return (
    <svg width={S} height={S} style={{ background: '#0d0d20', borderRadius: 6, flex: 'none' }}>
      <line x1={8} y1={c} x2={S - 8} y2={c} stroke="#34346a" />
      <line x1={c} y1={8} x2={c} y2={S - 8} stroke="#34346a" />
      <text x={S - 8} y={c - 4} textAnchor="end" fill="#667" fontSize={9}>e1</text>
      <text x={c + 4} y={14} fill="#667" fontSize={9}>e2</text>
      {arrows.map(([a, b], i) => {
        const x2 = c + a * sc, y2 = c - b * sc;
        const len = Math.hypot(x2 - c, y2 - c);
        const ux = len > 1e-6 ? (x2 - c) / len : 0, uy = len > 1e-6 ? (y2 - c) / len : 0;
        return (
          <g key={i}>
            <title>{`P${i + 1}: ${prompts[i] ?? ''} — its weight grows this way on the plane `
                    + `(arrow = the projection of e_${i + 1} − 1/k onto e1, e2)`}</title>
            <line x1={c} y1={c} x2={x2} y2={y2} stroke={GREY} strokeWidth={1.5} />
            {len > 4 && (
              <polygon fill={GREY}
                       points={`${x2},${y2} ${x2 - 6 * ux + 3 * uy},${y2 - 6 * uy - 3 * ux} `
                               + `${x2 - 6 * ux - 3 * uy},${y2 - 6 * uy + 3 * ux}`} />
            )}
            <text x={x2 + 8 * ux} y={y2 + 8 * uy + 3} textAnchor="middle" fill={GREY}
                  fontSize={10}>P{i + 1}</text>
          </g>
        );
      })}
    </svg>
  );
}

/** Convex hull of 2-D points (monotone chain): a k > 3 shadow's vertex ring can self-cross. */
function hull2d(pts: (readonly [number, number])[]): (readonly [number, number])[] {
  const p = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const cross = (o: readonly [number, number], a: readonly [number, number],
                 b: readonly [number, number]) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo: (readonly [number, number])[] = [];
  const up: (readonly [number, number])[] = [];
  for (const q of p) {
    while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop();
    lo.push(q);
  }
  for (const q of [...p].reverse()) {
    while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop();
    up.push(q);
  }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}

/** Where the plane sits: the simplex (k > 3: a fixed 2-D shadow) with every level's footprint. */
function Locator({ k, e1, e2, levels, current }: {
  k: number; e1: number[]; e2: number[];
  levels: { centre: number[]; s: number; level: number }[]; current: number;
}) {
  const S = 160;
  const { project } = makeProjector(k, 0, S);
  const verts = Array.from({ length: k }, (_, i) =>
    project(Array.from({ length: k }, (_, j) => (i === j ? 1 : 0))));
  const hull = hull2d(verts);
  const foot = (c: number[], s: number) => [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    .map(([a, b]) => project(c.map((w, j) => w + s * (a * e1[j] + b * e2[j]))))
    .map(([x, y]) => `${x},${y}`).join(' ');
  return (
    <svg width={S} height={S} style={{ background: '#0d0d20', borderRadius: 6, flex: 'none' }}>
      <polygon points={hull.map(([x, y]) => `${x},${y}`).join(' ')} fill="#151538"
               stroke="#34346a" />
      {verts.map(([x, y], i) => (
        <text key={i} x={x} y={y} dy={y < S / 2 ? -4 : 11} textAnchor="middle" fill="#7d84c8"
              fontSize={9}>P{i + 1}</text>
      ))}
      {levels.map((lv) => (
        <polygon key={lv.level} points={foot(lv.centre, lv.s)} fill="none"
                 stroke={lv.level === current ? ACCENT : '#4a4f8f'}
                 strokeWidth={lv.level === current ? 1.5 : 1} />
      ))}
      {levels[current] && (() => {
        const [x, y] = project(levels[current].centre);
        return <circle cx={x} cy={y} r={2.5} fill={ACCENT} />;
      })()}
      <title>{k > 3 ? 'where the lattice sits: a fixed 2-D shadow of the simplex, every level\'s '
                      + 'footprint outlined (current in green)'
                    : 'where the lattice sits in the triangle: every level\'s footprint '
                      + 'outlined (current in green)'}</title>
    </svg>
  );
}

export default function MicroscopePanel() {
  const [open, setOpen] = useState(false);
  const [k, setK] = useState(4);
  const [promptText, setPromptText] = useState(
    'a photograph of a lighthouse\na pencil sketch of a cathedral\n'
    + 'an oil painting of a forest\na neon city at night');
  const [mode, setMode] = useState<MicroMode>('random');
  const [centreText, setCentreText] = useState('');
  const [swapA, setSwapA] = useState<[number, number]>([0, 1]);
  const [swapB, setSwapB] = useState<[number, number]>([2, 3]);
  const [grid, setGrid] = useState<3 | 5 | 7>(5);
  const [s, setS] = useState(0.1);
  const [seed, setSeed] = useState(42);
  const [steps, setSteps] = useState(8);
  const [redraw, setRedraw] = useState(0);
  // a Cascade crossing handed over by the Cascade panel: crossing mode reads that run
  const [src, setSrc] = useState<{ runId: string; cid: number; prompts: string[] } | null>(null);
  const [plan, setPlan] = useState<MicroPlan | null>(null);
  const [planErr, setPlanErr] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [status, setStatus] = useState<MicroStatus | null>(null);
  const [view, setView] = useState(0);
  const [kick, setKick] = useState(0);
  const [err, setErr] = useState<string | null>(null);
  const [clickMode, setClickMode] = useState<'zoom' | 'enlarge'>('zoom');
  const [hover, setHover] = useState<string | null>(null);
  const [detail, setDetail] = useState<MicroCell | null>(null);
  const timer = useRef<number | null>(null);
  const viewRef = useRef(0);
  const rootRef = useRef<HTMLDivElement | null>(null);
  viewRef.current = view;

  // ---- a handoff from another panel: open, fill the form, scroll here
  const handoff = useHandoff((st) => st.scope);
  useEffect(() => {
    if (!handoff) return;
    setOpen(true);
    setErr(null);
    setRedraw(0);
    // a new place to look at: put its plan on screen rather than the previous run's lattice
    // (that run stays on the server; Start renders the new one)
    setRunId(null);
    setStatus(null);
    setView(0);
    setDetail(null);
    setK(handoff.prompts.length);
    setPromptText(handoff.prompts.join('\n'));
    setCentreText(handoff.weights.map((w) => w.toFixed(4)).join(', '));
    if (handoff.cascadeRunId && handoff.cid != null) {
      setSrc({ runId: handoff.cascadeRunId, cid: handoff.cid, prompts: handoff.prompts });
      setMode('crossing');
      setCentreText('');                // the crossing's own midpoint
    } else {
      setSrc(null);
      setMode((m) => (m === 'crossing' ? 'random' : m));
    }
    if (handoff.seed != null) setSeed(handoff.seed);
    if (handoff.steps != null) setSteps(handoff.steps);
    window.setTimeout(() => rootRef.current?.scrollIntoView({ behavior: 'smooth',
                                                              block: 'start' }), 50);
  }, [handoff?.nonce]);   // eslint-disable-line react-hooks/exhaustive-deps

  const prompts = promptText.split('\n').map((t) => t.trim()).filter(Boolean);
  const centre = parseWeights(centreText, k);
  const crossing = mode === 'crossing' && !!src;

  const body: MicroscopeRequest | null = useMemo(() => {
    if (crossing && src) {
      return { mode: 'crossing', cascade_run_id: src.runId, cid: src.cid,
               centre: centre.w, grid, s, e2_redraw: redraw };
    }
    if (mode === 'crossing') return null;
    if (prompts.length !== k || centre.err) return null;
    return { mode, prompts, centre: centre.w, grid, s, seed, steps, e2_redraw: redraw,
             ...(mode === 'prompt-swap' ? { swap_a: swapA, swap_b: swapB } : {}) };
  }, [crossing, src, mode, promptText, k, centreText, grid, s, seed, steps, redraw,
      swapA[0], swapA[1], swapB[0], swapB[1]]);   // eslint-disable-line react-hooks/exhaustive-deps

  // ---- the plan: geometry and cost, re-asked 300 ms after the form stops changing
  useEffect(() => {
    if (!open || !body) { setPlan(null); return; }
    let live = true;
    const t = window.setTimeout(async () => {
      try {
        const p = await microscopePlan(body);
        if (live) { setPlan(p); setPlanErr(null); }
      } catch (e: any) {
        if (live) { setPlan(null); setPlanErr(String(e.message || e)); }
      }
    }, PLAN_DEBOUNCE_MS);
    return () => { live = false; window.clearTimeout(t); };
  }, [open, JSON.stringify(body)]);   // eslint-disable-line react-hooks/exhaustive-deps

  // ---- one poll chain (the cascade panel's rule); `kick` restarts it after a zoom
  useEffect(() => {
    if (!runId) return;
    let live = true;
    const tick = async () => {
      try {
        const st = await microscopeStatus(runId, viewRef.current);
        if (!live) return;
        setStatus(st);
        if (st.status === 'running') timer.current = window.setTimeout(tick, POLL_MS);
      } catch {
        if (live) timer.current = window.setTimeout(tick, 4000);
      }
    };
    tick();
    return () => {
      live = false;
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [runId, kick]);

  const start = async () => {
    if (!body) return;
    setErr(null);
    setStatus(null);
    setDetail(null);
    try {
      const r = await microscopeStart(body);
      setView(r.level);
      setRunId(r.run_id);
      setKick((x) => x + 1);
    } catch (e: any) {
      setErr(String(e.message || e));
    }
  };

  const goto = (level: number) => {
    setView(level);
    setDetail(null);
    setKick((x) => x + 1);
  };

  const onCell = async (c: MicroCell) => {
    if (!runId || !c.inside) return;
    if (clickMode === 'enlarge') {
      if (c.image >= 0) setDetail(c);
      return;
    }
    setErr(null);
    try {
      const r = await microscopeZoom(runId, view, c.ia, c.ib);
      goto(r.level);
    } catch (e: any) {
      setErr(String(e.message || e));
    }
  };

  const running = status?.status === 'running';
  const lv = status?.levels[view] ?? null;
  const crumbs = status && lv ? chain(status.levels, view) : [];
  const nBoundary = lv ? lv.edges.filter((e) => e.boundary).length : 0;
  const tooBig = !!plan && plan.cost_image_eq > plan.max_image_eq;
  const kk = crossing && src ? src.prompts.length : k;
  const pairOpts = Array.from({ length: kk }, (_, i) => i);
  const showRedraw = mode === 'random' || (crossing && !!plan && plan.plane_note.includes('random'));

  return (
    <div style={BOX} ref={rootRef}>
      <div style={{ cursor: 'pointer', userSelect: 'none' }} onClick={() => setOpen(!open)}>
        {open ? '▾' : '▸'} Ridge microscope — an image lattice on a plane through a ridge (any k)
      </div>
      {open && (
        <div style={{ marginTop: 8 }}>
          <div style={{ color: '#667', fontSize: 11, marginBottom: 6 }}>
            A G × G lattice of full-fidelity images on a 2-D plane through one recipe — G² images
            at every k. Click a cell to zoom in 2× around it (Sequential Gallery); bars between
            cells mark a change past the Cascade's threshold (1 − cos &gt; 0.35, single seed);
            the grey arrows show which prompt grows in which direction (Sohns et al. 2023).
          </div>

          {/* the plane */}
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
            <label title={MODE_TIP}>
              plane
              <select value={mode} style={SEL}
                      onChange={(e) => { setMode(e.target.value as MicroMode); setRedraw(0); }}>
                {src && <option value="crossing">crossing {src.cid} (cascade {src.runId})</option>}
                <option value="prompt-swap">prompt-swap</option>
                <option value="random">random</option>
              </select>
            </label>
            {mode === 'prompt-swap' && (
              <>
                <label title="e1: the first prompt's weight grows to the right, the second's shrinks">
                  e1 = P
                  <select value={swapA[0]} style={SEL}
                          onChange={(e) => setSwapA([Number(e.target.value), swapA[1]])}>
                    {pairOpts.map((i) => <option key={i} value={i}>{i + 1}</option>)}
                  </select>
                  ↑ P
                  <select value={swapA[1]} style={SEL}
                          onChange={(e) => setSwapA([swapA[0], Number(e.target.value)])}>
                    {pairOpts.map((i) => <option key={i} value={i}>{i + 1}</option>)}
                  </select>↓
                </label>
                <label title="e2: the same for another pair, made orthogonal to e1 (up on screen)">
                  e2 = P
                  <select value={swapB[0]} style={SEL}
                          onChange={(e) => setSwapB([Number(e.target.value), swapB[1]])}>
                    {pairOpts.map((i) => <option key={i} value={i}>{i + 1}</option>)}
                  </select>
                  ↑ P
                  <select value={swapB[1]} style={SEL}
                          onChange={(e) => setSwapB([swapB[0], Number(e.target.value)])}>
                    {pairOpts.map((i) => <option key={i} value={i}>{i + 1}</option>)}
                  </select>↓
                </label>
              </>
            )}
            {showRedraw && (
              <button className="rx-focus" onClick={() => setRedraw((r) => r + 1)}
                      title={REDRAW_TIP}
                      style={{ background: '#0a0a1a', color: '#8a9', border: '1px solid #444',
                               borderRadius: 3, padding: '2px 8px', cursor: 'pointer' }}>
                ↻ re-draw e2{redraw > 0 ? ` (#${redraw})` : ''}
              </button>
            )}
            <label title={GRID_TIP}>
              G
              <select value={grid} style={SEL}
                      onChange={(e) => setGrid(Number(e.target.value) as 3 | 5 | 7)}>
                {[3, 5, 7].map((g) => <option key={g} value={g}>{g} × {g}</option>)}
              </select>
            </label>
            <label title={S_TIP}>
              s <input style={NUM} type="number" min={0.005} max={0.5} step={0.01} value={s}
                       onChange={(e) => setS(Math.max(0.005, Math.min(0.5,
                         Number(e.target.value) || 0.1)))} />
            </label>
            {!crossing && (
              <>
                <label title="prompts spanning the simplex — any k ≥ 3; the lattice costs the same at every k">
                  k
                  <select value={k} style={SEL} onChange={(e) => setK(Number(e.target.value))}>
                    {[3, 4, 5, 6, 7, 8].map((n) => <option key={n} value={n}>{n}</option>)}
                  </select>
                </label>
                <label title="the image seed (and the random plane's, unless re-drawn)">
                  seed <input style={NUM} type="number" value={seed}
                              onChange={(e) => setSeed(Number(e.target.value))} />
                </label>
                <label title="denoising steps: every cell is a full-fidelity image">
                  steps <input style={NUM} type="number" min={1} max={50} value={steps}
                               onChange={(e) => setSteps(Math.max(1, Math.min(50,
                                 Number(e.target.value))))} />
                </label>
              </>
            )}
            <button onClick={start} disabled={!plan || tooBig}
                    style={{ background: !plan || tooBig ? '#333' : '#0f3460', color: '#fff',
                             border: `1px solid ${ACCENT}`, borderRadius: 3,
                             padding: '3px 12px', cursor: !plan || tooBig ? 'default' : 'pointer' }}>
              {running ? 'Start new' : 'Start'}
            </button>
            {running && runId && (
              <button onClick={() => microscopeCancel(runId)}
                      style={{ background: '#0a0a1a', color: WARN, border: `1px solid ${WARN}`,
                               borderRadius: 3, padding: '3px 10px', cursor: 'pointer' }}>
                Cancel
              </button>
            )}
          </div>

          {/* the centre and the prompts (free modes); crossing mode reads the cascade run */}
          {crossing && src ? (
            <div style={{ color: '#889', fontSize: 11, marginTop: 6 }}>
              centre = crossing {src.cid}'s midpoint; prompts, seed and steps are cascade run{' '}
              {src.runId}'s, so the lattice is drawn from the field the crossing was found in.{' '}
              <span style={{ color: '#666' }}>{src.prompts.map((p, i) => `P${i + 1} ${short(p, 22)}`)
                .join(' · ')}</span>
            </div>
          ) : (
            <>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 6 }}>
                <label title="the recipe the plane goes through, one weight per prompt (normalised). Empty = the barycentre. A Cascade crossing or the Mixing desk can fill this in.">
                  centre{' '}
                  <input style={{ ...NUM, width: 260 }} value={centreText}
                         placeholder={`barycentre (${(1 / k).toFixed(3)} each)`}
                         onChange={(e) => setCentreText(e.target.value)} />
                </label>
                {centreText && (
                  <button onClick={() => setCentreText('')}
                          style={{ background: '#0a0a1a', color: '#8a9', border: '1px solid #444',
                                   borderRadius: 3, padding: '1px 8px', cursor: 'pointer' }}>
                    barycentre
                  </button>
                )}
                {centre.err && <span style={{ color: '#c9a227', fontSize: 11 }}>{centre.err}</span>}
              </div>
              <textarea value={promptText} onChange={(e) => setPromptText(e.target.value)}
                        placeholder={`pin ${k} prompts, one per line`} rows={Math.min(k, 8)}
                        style={{ width: '100%', marginTop: 6, background: '#0a0a1a', color: '#fff',
                                 border: '1px solid #444', borderRadius: 3, fontSize: 12,
                                 padding: 4, resize: 'vertical' }} />
              {prompts.length !== k && (
                <div style={{ color: '#c9a227', fontSize: 11 }}>
                  {prompts.length} prompts entered, {k} needed
                </div>
              )}
            </>
          )}

          {/* the plan: what Start would render, before anything is */}
          {plan && (
            <div style={{ color: '#667', marginTop: 4, fontSize: 11 }}>
              {plan.n_inside} of {plan.grid * plan.grid} cells inside the simplex = {' '}
              <span style={{ color: tooBig ? WARN : '#aeb6dd' }}>
                {plan.cost_image_eq.toFixed(0)} images
              </span>{' '}
              (full fidelity, {plan.steps} steps) · ceiling {plan.max_image_eq.toFixed(0)} per run,
              zooms included · {plan.plane_note}
            </div>
          )}
          {planErr && <div style={{ color: '#c9a227', fontSize: 11, marginTop: 4 }}>{planErr}</div>}
          {err && <div style={{ color: '#f66', marginTop: 4 }}>{err}</div>}

          {/* before a run: the plan's empty lattice, so the outside cells and the axes show */}
          {!status && plan && (
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 8,
                          alignItems: 'flex-start' }}>
              <div style={{ opacity: 0.6 }}>
                <Lattice cells={plan.cells} edges={[]} grid={plan.grid} runId={null}
                         hover={null} setHover={() => {}} preview />
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <Biplot arrows={plan.biplot} prompts={plan.prompts} />
                <Locator k={plan.k} e1={plan.e1} e2={plan.e2} current={0}
                         levels={[{ centre: plan.centre, s: plan.s, level: 0 }]} />
              </div>
            </div>
          )}

          {status && lv && runId && (
            <div style={{ marginTop: 8 }}>
              {/* breadcrumb + back */}
              <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap',
                            fontSize: 11 }}>
                <button disabled={lv.parent === null}
                        onClick={() => lv.parent !== null && goto(lv.parent)}
                        title="back to the level this one zoomed out of (cached: no images)"
                        style={{ background: '#0a0a1a', color: lv.parent === null ? '#555' : '#8a9',
                                 border: '1px solid #444', borderRadius: 3, padding: '1px 8px',
                                 cursor: lv.parent === null ? 'default' : 'pointer' }}>
                  ← back
                </button>
                {crumbs.map((c, i) => (
                  <span key={c.level}>
                    {i > 0 && <span style={{ color: '#555' }}> › </span>}
                    <span onClick={() => goto(c.level)}
                          style={{ cursor: 'pointer',
                                   color: c.level === view ? '#fff' : '#7d84c8',
                                   textDecoration: c.level === view ? 'none' : 'underline' }}>
                      L{c.level} · s {c.s < 0.01 ? c.s.toExponential(1) : c.s.toFixed(3)}
                    </span>
                  </span>
                ))}
                {status.levels.length > crumbs.length && (
                  <select value="" style={{ ...SEL, fontSize: 11 }}
                          onChange={(e) => e.target.value !== '' && goto(Number(e.target.value))}
                          title="every level this run has rendered, including other branches">
                    <option value="">all levels ({status.levels.length})…</option>
                    {status.levels.map((l) => (
                      <option key={l.level} value={l.level}>
                        L{l.level} · s {l.s.toFixed(4)}
                        {l.parent !== null ? ` · from L${l.parent}` : ''}
                      </option>
                    ))}
                  </select>
                )}
                <span style={{ marginLeft: 'auto', display: 'inline-flex', gap: 4,
                               alignItems: 'center' }}>
                  click =
                  {(['zoom', 'enlarge'] as const).map((m) => (
                    <button key={m} onClick={() => setClickMode(m)}
                            style={{ background: clickMode === m ? '#20304d' : '#0a0a1a',
                                     color: clickMode === m ? '#fff' : '#8a9',
                                     border: clickMode === m ? `1px solid ${ACCENT}`
                                       : '1px solid #444',
                                     borderRadius: 3, padding: '1px 8px', cursor: 'pointer' }}>
                      {m === 'zoom' ? 'zoom ×2' : 'enlarge'}
                    </button>
                  ))}
                </span>
              </div>

              <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', marginTop: 4,
                            fontSize: 11, color: '#aeb6dd', fontVariantNumeric: 'tabular-nums' }}>
                <span style={{ color: lv.status === 'error' ? '#f66'
                  : lv.status === 'complete' ? '#6a6' : '#c9a227' }}>
                  {lv.status === 'running' ? `rendering… ${lv.cells.filter((c) => c.inside
                    && c.image >= 0).length}/${lv.n_inside}` : lv.status}
                </span>
                <span title="cells this level rendered, and cells read back from a coarser level (the same recipe at the same seed is the same image)">
                  {lv.n_inside} cells · {lv.n_new} rendered · {lv.n_reused} reused
                  {lv.n_inside < lv.cells.length ? ` · ${lv.cells.length - lv.n_inside} outside` : ''}
                </span>
                <span title="neighbour pairs past the Cascade's crossing threshold, single seed">
                  {nBoundary} boundar{nBoundary === 1 ? 'y' : 'ies'}
                </span>
                <span title="images this run has rendered (all levels), against its ceiling">
                  cost {status.cost_image_eq.toFixed(0)} / {status.max_image_eq.toFixed(0)} images
                </span>
              </div>

              <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 8,
                            alignItems: 'flex-start' }}>
                <div>
                  <Lattice cells={lv.cells} edges={lv.edges} grid={lv.grid} runId={runId}
                           onCell={onCell} hover={hover} setHover={setHover} />
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10,
                                color: '#667', marginTop: 3, width: LATTICE_PX }}>
                    <span>a = −1 (−s along e1)</span>
                    <span>e1 →   ·   e2 ↑ (top row b = +1)</span>
                    <span>a = +1</span>
                  </div>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  <Biplot arrows={status.biplot} prompts={status.prompts} />
                  <Locator k={status.k} e1={status.e1} e2={status.e2} current={view}
                           levels={status.levels} />
                  <div style={{ maxWidth: 160, fontSize: 10, color: '#889' }}>
                    {status.prompts.map((p, i) => (
                      <div key={i} title={p}>P{i + 1} {short(p, 24)}</div>
                    ))}
                  </div>
                </div>
              </div>

              <div style={{ color: '#666', marginTop: 6, fontSize: 11 }}>
                {status.plane_note}
                {lv.notes.length > 0 && <div>{lv.notes[lv.notes.length - 1]}</div>}
                {lv.error && <div style={{ color: '#f66' }}>{lv.error}</div>}
              </div>
            </div>
          )}
        </div>
      )}

      {detail && runId && status && lv && (
        <div onClick={() => setDetail(null)}
             style={{ position: 'fixed', inset: 0, background: 'rgba(5,5,18,0.88)', zIndex: 60,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      padding: 20 }}>
          <div onClick={(e) => e.stopPropagation()}
               style={{ background: '#12122c', border: '1px solid #34346a', borderRadius: 8,
                        padding: 16, display: 'flex', gap: 16, flexWrap: 'wrap',
                        maxWidth: '90vw', maxHeight: '90vh', overflow: 'auto' }}>
            <img src={microscopeImageUrl(runId, detail.image)}
                 style={{ width: 'min(60vh, 512px)', height: 'min(60vh, 512px)',
                          objectFit: 'cover', borderRadius: 6 }} />
            <div style={{ minWidth: 240, maxWidth: 320, fontSize: 12, color: '#c9d1f0' }}>
              <div style={{ fontSize: 14, marginBottom: 8, color: '#fff' }}>
                level {lv.level} · cell a {detail.a.toFixed(2)}, b {detail.b.toFixed(2)}
              </div>
              {lv.edges.filter((e) => (e.a[0] === detail.ia && e.a[1] === detail.ib)
                || (e.b[0] === detail.ia && e.b[1] === detail.ib)).map((e, i) => {
                const o = e.a[0] === detail.ia && e.a[1] === detail.ib ? e.b : e.a;
                const dir = o[0] > detail.ia ? '→' : o[0] < detail.ia ? '←'
                  : o[1] > detail.ib ? '↑' : '↓';
                return (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between',
                                        gap: 12, marginBottom: 2 }}>
                    <span style={{ color: '#8a93b8' }}>1 − cos to the {dir} neighbour</span>
                    <span style={{ color: divColor(e.div) }}>{fmt(e.div, 2)}</span>
                  </div>
                );
              })}
              <div style={{ margin: '10px 0 4px', color: '#8a93b8' }}>recipe</div>
              {status.prompts.map((pr, i) => (
                <div key={i} style={{ marginBottom: 4 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis',
                                   whiteSpace: 'nowrap', maxWidth: 230 }}>P{i + 1} {pr}</span>
                    <span style={{ color: '#8a93b8', fontVariantNumeric: 'tabular-nums' }}>
                      {((detail.w[i] ?? 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div style={{ height: 4, background: '#1a1a2e', borderRadius: 2 }}>
                    <div style={{ width: `${Math.min(100, Math.max(0, (detail.w[i] ?? 0) * 100))}%`,
                                  height: '100%', background: ACCENT, borderRadius: 2 }} />
                  </div>
                </div>
              ))}
              <button onClick={() => setDetail(null)}
                      style={{ background: '#0a0a1a', color: '#8a9', border: '1px solid #444',
                               borderRadius: 3, padding: '4px 12px', marginTop: 14,
                               cursor: 'pointer' }}>
                close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
