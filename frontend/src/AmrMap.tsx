/**
 * AmrMap -- the adaptive lattice drawn on the space it refines.
 *
 * The picture IS the simplex, as in CascadeMap: prompt vertices around the hull, then the
 * cells the sampler actually evaluated as dots and the boundary EDGES it detected as short
 * segments between adjacent cells. Where the Cascade draws lines it searched ALONG, this
 * draws the grid it searched ACROSS, so the two surfaces read as two ways of spending the
 * same budget on the same space.
 *
 * Colour is the MEASUREMENT, as on the Cascade map: the default reads every mark on
 * `divColor`, the blue->red divergence ramp exported from CascadeMap, because an AMR cell's
 * divergence (max 1 - cos to its measured lattice neighbours, backend `pair_divergences`)
 * and a chord probe's divergence there (`_set_div`) are the SAME quantity, and one quantity
 * gets one ramp. The boundary EDGES go on it too, each at its own divergence, so a 0.4 edge
 * and a 1.2 edge stop looking alike. The ladder then has the channel colour is not spending:
 * dot SIZE, big = coarse.
 *
 * The secondary toggle keeps the other reading, which answers a different question (where
 * did the budget go, not where is the boundary):
 *
 *   * BY LEVEL -- which rung of the ladder paid for this cell. Ordinal, so a one-hue
 *     light->dark ramp, coarse (dark) to fine (light), validated on this map's own surface
 *     (#0d0d20): monotone lightness, adjacent dL >= 0.06, hue spread 3 deg, darkest step
 *     2.35:1. Four of its five steps are tokens the map already uses for structure, and the
 *     hue is deliberately the structural neutral-blue rather than a data hue -- the two data
 *     ramps in this tool already mean something (blue->red divergence, orange S_V). Dot SIZE
 *     carries the level here as well, so the level is never colour-alone, and the edges stay
 *     on the divergence ramp -- they are a measurement either way.
 *
 * k=3 is the exact static triangle. k=4 is a 2-D shadow of the 3-D space, drawn by
 * CascadeMap's own `makeProjector` and turned with the same bindings as that map (left-drag
 * rotates, middle-drag pans, wheel / Alt+right-drag zooms, h homes), via `viewTransform`.
 * There is no auto-rotation here on purpose: the lattice is thousands of static dots that
 * are hovered one at a time, and a drifting projection would move the target under the
 * pointer.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import type { AmrStatus } from './api/types';
import { divColor, makeProjector } from './CascadeMap';
import {
  HOME, dragFactor, isDrag, isHome, pan, reset, rotationDelta, svgTransform, toScreen,
  wheelFactor, wrapAngle, zoomAbout,
} from './viewTransform';
import type { View } from './viewTransform';

const WARN = '#e94560';
const HALO = '#0a0a12';
/**
 * Refinement level, coarse -> fine. Ordinal ramp; see the header for the checks it passed
 * and why the hue is the structural one. Steps 2-5 are existing map tokens (the "no local
 * evidence" grey, the unscored-crossing dot, the vertex label ink, the vertex dot); step 1
 * is that family stepped one darker and then snapped up from #3a3f6e, which missed the 2:1
 * ordinal floor against this surface at 1.93:1.
 */
const LEVEL_RAMP = ['#454b80', '#565e93', '#7d84c8', '#aeb6dd', '#dfe3ff'];

/** Colour for rung `i` of a ladder of `n`, spread over the whole ramp however long it is. */
function levelColor(i: number, n: number): string {
  if (n <= 1) return LEVEL_RAMP[LEVEL_RAMP.length - 1];
  const t = Math.min(1, Math.max(0, i / (n - 1)));
  return LEVEL_RAMP[Math.round(t * (LEVEL_RAMP.length - 1))];
}

function short(p: string, n = 26) {
  return p.length > n ? p.slice(0, n - 1) + '…' : p;
}

// Bounded node counts: a 40/edge k=4 run evaluates ~7k cells and detects ~20k edges, far
// more SVG than the eye gains from. Points stride uniformly; EDGES are sorted finest-level
// first before striding, so the sharpest boundary is the part that survives.
const MAX_DOTS = 4000;
const MAX_EDGES = 3000;

export type AmrColorBy = 'level' | 'divergence';

export default function AmrMap({
  status, runId, size = 560, imageUrl, onImage,
}: {
  status: AmrStatus;
  runId: string;
  size?: number;
  imageUrl: (runId: string, index: number) => string;
  /** click a cell: hand its thumbnail and recipe to the panel's large view */
  onImage?: (thumb: number, title: string, rows: [string, string][],
             weights?: number[] | null) => void;
}) {
  const k = status.k || status.prompts.length;
  // divergence by default: the same reading, on the same ramp, as the Cascade's chord probes
  const [colorBy, setColorBy] = useState<AmrColorBy>('divergence');
  const [finestOnly, setFinestOnly] = useState(false);
  const [theta, setTheta] = useState(0);
  const [view, setView] = useState<View>(HOME);
  const [hover, setHover] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const hovering = useRef(false);
  const gesture = useRef<{
    kind: 'pan' | 'zoom' | 'rotate';
    x: number; y: number; pid: number; view: View; theta: number; moved: boolean;
  } | null>(null);
  const dragged = useRef(false);
  const canRotate = k > 3;

  const at = (e: { clientX: number; clientY: number }) => {
    const r = svgRef.current?.getBoundingClientRect();
    return [e.clientX - (r?.left ?? 0), e.clientY - (r?.top ?? 0)] as const;
  };

  // React's onWheel is passive and so cannot stop the page scrolling underneath; the same
  // explicit non-passive listener CascadeMap uses.
  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      setView((v) => zoomAbout(v, wheelFactor(e.deltaY, e.deltaMode),
                               e.clientX - r.left, e.clientY - r.top));
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  // `h` homes the map under the pointer (or the focused one); the angle keeps its own
  // history, exactly as in the cascade map.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'h' && e.key !== 'H') return;
      if (e.altKey || e.ctrlKey || e.metaKey) return;
      const t = e.target as HTMLElement | null;
      const tag = t?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
          || t?.isContentEditable) return;
      const el = svgRef.current;
      if (!el || (!hovering.current && document.activeElement !== el)) return;
      setView(reset());
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const onPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if (e.pointerType !== 'mouse') return;      // every binding names a mouse button
    const kind = e.button === 1 ? 'pan'
      : e.button === 2 && e.altKey ? 'zoom'
        : e.button === 0 && canRotate ? 'rotate'
          : null;
    if (!kind) return;
    gesture.current = { kind, x: at(e)[0], y: at(e)[1], pid: e.pointerId, view, theta,
                        moved: false };
    if (kind !== 'rotate') {
      e.preventDefault();
      svgRef.current?.setPointerCapture(e.pointerId);
    }
  };

  const onPointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const g = gesture.current;
    if (!g || e.pointerId !== g.pid) return;
    const [x, y] = at(e);
    const dx = x - g.x, dy = y - g.y;
    if (!g.moved) {
      if (!isDrag(dx, dy)) return;
      g.moved = true;
      if (g.kind === 'rotate') svgRef.current?.setPointerCapture(g.pid);
    }
    if (g.kind === 'pan') setView(pan(g.view, dx, dy));
    else if (g.kind === 'zoom') setView(zoomAbout(g.view, dragFactor(dy), g.x, g.y));
    else setTheta(wrapAngle(g.theta + rotationDelta(dx)));
  };

  const endGesture = (e: React.PointerEvent<SVGSVGElement>) => {
    const g = gesture.current;
    if (!g || e.pointerId !== g.pid) return;
    dragged.current = g.moved;
    gesture.current = null;
  };

  const sched = status.schedule?.length ? status.schedule : [];
  const levelIdx = useMemo(() => {
    const m = new Map<number, number>();
    sched.forEach((n, i) => m.set(n, i));
    return m;
  }, [sched.join(',')]);
  const finest = sched.length ? sched[sched.length - 1] : 0;

  // Screen px -> user units inside the drawn group, so marks keep their size at any zoom
  // and zooming spreads them apart instead of inflating them (CascadeMap's `q`).
  const q = (v: number) => v / view.s;
  const cx = size / 2, cy = size / 2, r = size / 2 - 34;
  const proj = makeProjector(k, theta, size, 0);
  const vs = Array.from({ length: k }, (_, i) => {
    const e = new Array(k).fill(0);
    e[i] = 1;
    return proj.project(e);
  });
  const hullOrder = [...vs].sort(
    (a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));

  const shownPts = (() => {
    const n = status.points.length;
    const keep: number[] = [];
    const stride = n > MAX_DOTS ? Math.ceil(n / MAX_DOTS) : 1;
    for (let i = 0; i < n; i += stride) {
      if (finestOnly && status.point_levels[i] !== finest) continue;
      keep.push(i);
    }
    return keep;
  })();
  const shownEdges = (() => {
    const idx = status.edges.map((_e, i) => i);
    // finest level first: when a long run is strided, the sharpest edges are the ones kept
    idx.sort((a, b) => (status.edge_levels[b] ?? 0) - (status.edge_levels[a] ?? 0));
    const sel = finestOnly
      ? idx.filter((i) => status.edge_levels[i] === finest) : idx;
    const stride = sel.length > MAX_EDGES ? Math.ceil(sel.length / MAX_EDGES) : 1;
    return sel.filter((_v, j) => j % stride === 0);
  })();

  const dotR = (lvl: number) => {
    const i = levelIdx.get(lvl) ?? 0;
    const n = Math.max(sched.length, 1);
    // coarse cells are few and are the structure, so they are the big dots; the finest
    // level is thousands of small ones
    return 3.4 - 1.9 * (n > 1 ? i / (n - 1) : 0);
  };
  const dotFill = (i: number) => colorBy === 'divergence'
    ? divColor(status.point_divs[i] ?? null)
    : levelColor(levelIdx.get(status.point_levels[i]) ?? 0, Math.max(sched.length, 1));

  const hoverPt = hover !== null && hover < status.points.length ? hover : null;
  const select = (i: number) => {
    if (dragged.current) return;
    const img = status.point_images[i];
    if (img === undefined || img < 0) return;
    onImage?.(img, `cell ${i} · level ${status.point_levels[i]}`, [
      ['refinement level', `${status.point_levels[i]} points per edge`],
      ['local divergence', status.point_divs[i] != null
        ? status.point_divs[i]!.toFixed(3) : 'no measured neighbour'],
    ], status.points[i]);
  };

  if (k < 3) return null;

  return (
    <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'flex-start' }}>
      <svg width={size} height={size} ref={svgRef} className="rx-focus" tabIndex={0}
           onPointerDown={onPointerDown} onPointerMove={onPointerMove}
           onPointerUp={endGesture} onPointerCancel={endGesture}
           onLostPointerCapture={endGesture}
           onPointerEnter={() => { hovering.current = true; }}
           onPointerLeave={() => { hovering.current = false; setHover(null); }}
           onClickCapture={(e) => {
             if (dragged.current) e.stopPropagation();   // a drag is never a selection
             dragged.current = false;
           }}
           onContextMenu={(e) => e.preventDefault()}
           style={{ background: '#0d0d20', borderRadius: 8,
                    touchAction: 'manipulation', userSelect: 'none' }}>
        {/* one transform for every drawn layer; the labels outside it keep their size */}
        <g transform={svgTransform(view)}>
          <polygon points={hullOrder.map(([x, y]) => `${x},${y}`).join(' ')}
                   fill="#151538" stroke="#34346a" strokeWidth={q(1.5)} />
          {/* boundary edges UNDER the dots: an edge is a property of the pair of cells it
              joins, so the cells stay the clickable objects on top of it. Each one is drawn
              at ITS OWN divergence on the shared ramp -- every detected edge is past 0.35,
              but they run from there to past 1, and one flat warning colour said otherwise. */}
          <g style={{ pointerEvents: 'none' }}>
            {shownEdges.map((ei) => {
              const [a, b] = status.edges[ei];
              const pa = status.points[a], pb = status.points[b];
              if (!pa || !pb) return null;
              const [x1, y1] = proj.project(pa);
              const [x2, y2] = proj.project(pb);
              if (![x1, y1, x2, y2].every(Number.isFinite)) return null;
              return <line key={`e${ei}`} x1={x1} y1={y1} x2={x2} y2={y2}
                           stroke={divColor(status.edge_divs[ei] ?? null)}
                           strokeWidth={q(2)} strokeOpacity={0.85}
                           strokeLinecap="round" />;
            })}
          </g>
          {shownPts.map((i) => {
            const [x, y] = proj.project(status.points[i]);
            if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
            const rad = q(dotR(status.point_levels[i]));
            return (
              <circle key={`p${i}`} cx={x} cy={y} r={rad} fill={dotFill(i)}
                      fillOpacity={0.95}
                      stroke={hover === i ? '#fff' : 'none'} strokeWidth={q(1.4)}
                      style={{ cursor: 'pointer' }}
                      onPointerEnter={() => setHover(i)}
                      onClick={() => select(i)} />
            );
          })}
        </g>
        {/* Prompt vertices move with the view -- they are places -- but are drawn at screen
            size, so zooming spreads the corners rather than inflating them. */}
        {vs.map(([wx, wy], i) => {
          const dx = (wx - cx) / r, dy = (wy - cy) / r;
          const [x, y] = toScreen(view, wx, wy);
          const anchor = Math.abs(dx) < 0.35 ? 'middle' : dx > 0 ? 'start' : 'end';
          return (
            <g key={i}>
              <circle cx={x} cy={y} r={4.5} fill="#dfe3ff" />
              <text x={x + dx * 14} y={y + dy * 14 + 3} fill="#aeb6dd" fontSize={11}
                    textAnchor={anchor}>
                {short(status.prompts[i] ?? `P${i + 1}`)}
              </text>
            </g>
          );
        })}
        {k > 3 && (
          <text x={size - 12} y={20} textAnchor="end" fill="#7d84c8" fontSize={11}>
            3-D space · 2-D shadow
          </text>
        )}
        {!isHome(view) && (
          <text x={10} y={20} fill="#7d84c8" fontSize={10}
                style={{ fontVariantNumeric: 'tabular-nums' }}>
            {view.s.toFixed(2)}× · h: home
          </text>
        )}
        {(shownPts.length < status.points.length
          || shownEdges.length < status.edges.length) && (
          <text x={10} y={size - 10} fill="#7d84c8" fontSize={10}>
            drawing {shownPts.length}/{status.points.length} cells ·{' '}
            {shownEdges.length}/{status.edges.length} edges
          </text>
        )}
      </svg>
      <div style={{ fontSize: 11, color: '#889', maxWidth: 200 }}>
        <div title={canRotate
          ? 'left-drag turns the shadow plane (~0.5°/px); a press that moves less than 4 px'
            + ' is still a click. Middle-drag pans, the wheel zooms about the cursor,'
            + ' Alt+right-drag zooms by drag distance, h homes the hovered map.'
          : 'middle-drag pans, the wheel zooms about the cursor, Alt+right-drag zooms by'
            + ' drag distance, h homes the hovered map. The k=3 triangle has no shadow'
            + ' plane to rotate.'}
             style={{ color: '#667', marginBottom: 6 }}>
          {canRotate ? 'rotate: LMB drag · ' : ''}pan: MMB drag · zoom: wheel / Alt+RMB
          {' '}· h: home
        </div>
        <label className="rx-focus"
               title="divergence (the default) = the cell's own 1-cos to its measured lattice neighbours, on the same blue→red ramp the Cascade colours its probes with; dot size carries the level. level = which rung of the ladder paid for the cell (dark = coarse, light = fine), which answers where the budget went rather than where the boundary is. The edges stay on the divergence ramp either way."
               style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 6,
                        color: '#aeb6dd' }}>
          colour
          <select value={colorBy}
                  onChange={(e) => setColorBy(e.target.value as AmrColorBy)}
                  style={{ background: '#0a0a1a', color: '#8a9', border: '1px solid #444',
                           borderRadius: 3, padding: '1px 3px', fontSize: 11 }}>
            <option value="divergence">by divergence</option>
            <option value="level">by level</option>
          </select>
        </label>
        <label className="rx-focus"
               title="hide every cell and edge the coarser levels found, leaving the finest lattice alone — the sharpest statement the run makes about where the boundary is"
               style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 8,
                        color: '#aeb6dd', cursor: 'pointer' }}>
          <input type="checkbox" checked={finestOnly}
                 onChange={(e) => setFinestOnly(e.target.checked)} />
          finest level only
        </label>
        {/* The divergence ramp is always here: in the default mode it colours the dots AND
            the edges, and in level mode it still colours the edges. Same ramp, same wording
            and same top of scale as the Cascade's cloud, so the two maps read as one
            instrument. */}
        <div style={{ marginBottom: 8 }}>
          <div style={{ color: '#aeb6dd' }}>
            divergence <span style={{ color: '#667' }}>· 1 − cos to neighbours</span>
          </div>
          <div style={{ height: 8, borderRadius: 2, marginTop: 4,
                        border: '1px solid rgba(255,255,255,0.10)',
                        background: `linear-gradient(90deg, ${divColor(0)}, `
                                    + `${divColor(0.25)}, ${divColor(0.5)})` }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', color: '#889',
                        fontVariantNumeric: 'tabular-nums' }}>
            <span>0</span><span>0.50</span>
          </div>
          <div style={{ color: '#667', marginTop: 2 }}>
            <span style={{ color: divColor(0) }}>quiet</span> →{' '}
            <span style={{ color: WARN }}>hot</span>, the Cascade's probe ramp
          </div>
        </div>
        {colorBy === 'level' ? (
          <div style={{ marginBottom: 8 }}>
            <div style={{ color: '#aeb6dd' }}>
              level <span style={{ color: '#667' }}>· coarse → fine</span>
            </div>
            <div style={{ display: 'flex', gap: 2, marginTop: 4 }}>
              {sched.map((n, i) => (
                <div key={n} style={{ flex: 1, textAlign: 'center' }}>
                  <div style={{ height: 8, borderRadius: 2,
                                border: '1px solid rgba(255,255,255,0.10)',
                                background: levelColor(i, Math.max(sched.length, 1)) }} />
                  <div style={{ color: '#889', fontVariantNumeric: 'tabular-nums' }}>{n}</div>
                </div>
              ))}
            </div>
            <div style={{ color: '#667', marginTop: 2 }}>
              points per simplex edge; dot size says the same, big = coarse
            </div>
          </div>
        ) : (
          /* size = level: with colour spent on the measurement, the ladder is the radius,
             so the legend draws the radii themselves rather than describing them */
          <div style={{ marginBottom: 8 }}>
            <div style={{ color: '#aeb6dd' }}>
              size <span style={{ color: '#667' }}>= level, coarse → fine</span>
            </div>
            <div style={{ display: 'flex', gap: 2, marginTop: 2 }}>
              {sched.map((n) => (
                <div key={n} style={{ flex: 1, textAlign: 'center' }}>
                  <svg width={16} height={10} style={{ display: 'block', margin: '0 auto' }}>
                    <circle cx={8} cy={5} r={dotR(n)} fill="#aeb6dd" />
                  </svg>
                  <div style={{ color: '#889', fontVariantNumeric: 'tabular-nums' }}>{n}</div>
                </div>
              ))}
            </div>
            <div style={{ color: '#667', marginTop: 2 }}>
              points per simplex edge
            </div>
          </div>
        )}
        <div style={{ marginBottom: 6 }}>
          <span style={{ color: divColor(0.5) }}>―</span> boundary edge — two adjacent cells
          whose images differ by more than 0.35 (the Cascade's crossing threshold), drawn at
          its own divergence on the ramp above
          <br /><span style={{ color: divColor(0) }}>●</span>
          <span style={{ color: divColor(0.5) }}>●</span> evaluated cells; the lattices are
          nested, so a coarse cell is also a fine one and was probed once
        </div>
        {hoverPt !== null && status.point_images[hoverPt] >= 0 ? (
          <div>
            <img src={imageUrl(runId, status.point_images[hoverPt])} width={140}
                 height={140} title="click the dot to enlarge"
                 onClick={() => select(hoverPt)}
                 style={{ objectFit: 'cover', borderRadius: 6, cursor: 'pointer',
                          border: `2px solid ${dotFill(hoverPt)}` }} />
            <div style={{ color: '#aeb6dd', marginTop: 2 }}>
              cell {hoverPt} · level {status.point_levels[hoverPt]}
              {status.point_divs[hoverPt] != null
                && ` · div ${status.point_divs[hoverPt]!.toFixed(2)}`}
            </div>
          </div>
        ) : (
          <div style={{ color: '#667' }}>hover a cell to see its image</div>
        )}
      </div>
    </div>
  );
}
