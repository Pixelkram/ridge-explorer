/**
 * MetroMap -- the Metropolis chains drawn on the space they sample.
 *
 * Same picture as CascadeMap and AmrMap, by the same arithmetic (`makeProjector`,
 * `viewTransform`): prompt vertices around the hull, the k=3 triangle exact, k>=4 a 2-D
 * shadow of the (k-1)-dimensional space that left-drag turns. Three layers:
 *
 *   * the SEED crossings as hollow rings -- where the survey (or this run's own chords)
 *     handed the chains their starting sheets;
 *   * each chain's accepted states joined in order as a faint POLYLINE -- the path is
 *     structure, not data, so it takes the map's existing chord token at low opacity and
 *     stays recessive under the dots. (This is deliberately below the 3:1 contrast a data
 *     mark needs: nothing is encoded by it alone, the legend names it, and the sample's own
 *     energy is in its fill, its tooltip and the gallery beside the map.)
 *   * the accepted STATES as dots filled by `divColor(S)` -- the same exported blue->red
 *     ramp, the same 0..0.50 top of scale, that the Cascade colours a probe's divergence
 *     with and the AMR map a cell's. S is that quantity, averaged over m directions instead
 *     of read across one lattice edge, so it belongs on the same ramp and gets no new one.
 *
 * No auto-rotation, for AmrMap's reason: these are hundreds of static dots hovered one at a
 * time, and a drifting projection would move the target under the pointer.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import type { MetroStatus } from './api/types';
import { divColor, makeProjector } from './CascadeMap';
import {
  HOME, dragFactor, isDrag, isHome, pan, reset, rotationDelta, svgTransform, toScreen,
  wheelFactor, wrapAngle, zoomAbout,
} from './viewTransform';
import type { View } from './viewTransform';

/** the map's existing search-line token: a chain's path is a path, like a chord */
const PATH = '#4a4f8f';
/** the crossing-dot neutral, reused for the seeds a chain started from */
const SEED = '#8a93b8';
const WARN = '#e94560';

function short(p: string, n = 26) {
  return p.length > n ? p.slice(0, n - 1) + '…' : p;
}

// A 60 x 50 run accepts at most 3,000 states, which is well inside what SVG draws happily,
// so the only cap here is on the polylines: 200 chains x 50 segments is more line than the
// eye gains from, and the chains with the most accepted states are the ones worth drawing.
const MAX_PATHS = 80;

export default function MetroMap({
  status, runId, size = 560, imageUrl, onImage,
}: {
  status: MetroStatus;
  runId: string;
  size?: number;
  imageUrl: (runId: string, index: number) => string;
  /** click a sample: hand its thumbnail and recipe to the panel's large view */
  onImage?: (thumb: number, title: string, rows: [string, string][],
             weights?: number[] | null) => void;
}) {
  const k = status.k || status.prompts.length;
  const [showPaths, setShowPaths] = useState(true);
  const [showSeeds, setShowSeeds] = useState(true);
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
  // explicit non-passive listener the other two maps use.
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

  // Screen px -> user units inside the drawn group, so marks keep their size at any zoom
  // (CascadeMap's `q`).
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

  // One polyline per chain: its seed, then its accepted states in the order they were
  // accepted. The chains that accepted most are the ones kept when the cap bites.
  const paths = useMemo(() => {
    const by = new Map<number, number[]>();
    status.samples.forEach((s, i) => {
      const a = by.get(s.chain);
      if (a) a.push(i); else by.set(s.chain, [i]);
    });
    const seedOf = new Map<number, number[]>();
    for (const c of status.chains_stats) seedOf.set(c.chain, c.seed_weights);
    return [...by.entries()]
      .sort((a, b) => b[1].length - a[1].length)
      .slice(0, MAX_PATHS)
      .map(([chain, idx]) => ({ chain, idx, seed: seedOf.get(chain) ?? null }));
  }, [status.samples.length, status.chains_stats.length]);

  const hoverPt = hover !== null && hover < status.samples.length ? hover : null;
  const select = (i: number) => {
    if (dragged.current) return;
    const s = status.samples[i];
    if (!s || s.image < 0) return;
    onImage?.(s.full_image ?? s.image,
              `sample ${i} · chain ${s.chain} · round ${s.step}`, [
                ['energy S', s.s.toFixed(3)],
                ['past the 0.35 threshold', s.s > 0.35 ? 'yes' : 'no'],
                ['rendered at', s.full_image != null
                  ? `${status.steps} steps (full pass)`
                  : `${status.probe_steps ?? status.steps} steps (cheap field)`],
              ], s.weights);
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
        <g transform={svgTransform(view)}>
          <polygon points={hullOrder.map(([x, y]) => `${x},${y}`).join(' ')}
                   fill="#151538" stroke="#34346a" strokeWidth={q(1.5)} />
          {/* chain paths UNDER everything: structure, not data */}
          {showPaths && (
            <g style={{ pointerEvents: 'none' }}>
              {paths.map(({ chain, idx, seed }) => {
                const pts = (seed ? [seed] : [])
                  .concat(idx.map((i) => status.samples[i].weights))
                  .map((w) => proj.project(w))
                  .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
                if (pts.length < 2) return null;
                return <polyline key={`c${chain}`} fill="none" stroke={PATH}
                                 strokeWidth={q(1.2)} strokeOpacity={0.45}
                                 strokeLinejoin="round"
                                 points={pts.map(([x, y]) => `${x},${y}`).join(' ')} />;
              })}
            </g>
          )}
          {/* the seed crossings: hollow, so they never read as samples */}
          {showSeeds && (
            <g style={{ pointerEvents: 'none' }}>
              {status.seeds.map((w, i) => {
                const [x, y] = proj.project(w);
                if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
                return <circle key={`s${i}`} cx={x} cy={y} r={q(3.2)} fill="none"
                               stroke={SEED} strokeWidth={q(1.2)} strokeOpacity={0.8} />;
              })}
            </g>
          )}
          {status.samples.map((s, i) => {
            const [x, y] = proj.project(s.weights);
            if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
            return (
              <circle key={`p${i}`} cx={x} cy={y} r={q(3)} fill={divColor(s.s)}
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
            {k - 1}-D space · 2-D shadow
          </text>
        )}
        {!isHome(view) && (
          <text x={10} y={20} fill="#7d84c8" fontSize={10}
                style={{ fontVariantNumeric: 'tabular-nums' }}>
            {view.s.toFixed(2)}× · h: home
          </text>
        )}
        {paths.length < status.summary.n_chains && showPaths && (
          <text x={10} y={size - 10} fill="#7d84c8" fontSize={10}>
            drawing the {paths.length} busiest of {status.summary.n_chains} chains
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
        <div style={{ marginBottom: 8 }}>
          <div style={{ color: '#aeb6dd' }}>
            energy S <span style={{ color: '#667' }}>· mean 1 − cos over {status.m}{' '}
            direction{status.m === 1 ? '' : 's'}</span>
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
            <span style={{ color: WARN }}>hot</span>, the Cascade's probe ramp — read at the
            same stride ({status.delta.toFixed(3)})
          </div>
        </div>
        <label className="rx-focus"
               title="each chain's accepted states joined in the order it accepted them, starting at its seed crossing. The path is structure, so it is drawn recessively — nothing is encoded by it alone."
               style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 4,
                        color: '#aeb6dd', cursor: 'pointer' }}>
          <input type="checkbox" checked={showPaths}
                 onChange={(e) => setShowPaths(e.target.checked)} />
          chain paths
        </label>
        <label className="rx-focus"
               title="the crossings the chains were seeded at: either this run's own IUR chords or a finished Cascade's. Hollow, so a seed never reads as a sample."
               style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 8,
                        color: '#aeb6dd', cursor: 'pointer' }}>
          <input type="checkbox" checked={showSeeds}
                 onChange={(e) => setShowSeeds(e.target.checked)} />
          seed crossings
        </label>
        <div style={{ marginBottom: 6 }}>
          <span style={{ color: divColor(0.5) }}>●</span> accepted state — colour = its
          energy S on the ramp above
          <br /><span style={{ color: PATH }}>―</span> one chain's path, seed first
          <br /><span style={{ color: SEED }}>◦</span> seed crossing
        </div>
        {hoverPt !== null && status.samples[hoverPt].image >= 0 ? (
          <div>
            <img src={imageUrl(runId, status.samples[hoverPt].full_image
                                     ?? status.samples[hoverPt].image)}
                 width={140} height={140} title="click the dot to enlarge"
                 onClick={() => select(hoverPt)}
                 style={{ objectFit: 'cover', borderRadius: 6, cursor: 'pointer',
                          border: `2px solid ${divColor(status.samples[hoverPt].s)}` }} />
            <div style={{ color: '#aeb6dd', marginTop: 2 }}>
              chain {status.samples[hoverPt].chain} · round{' '}
              {status.samples[hoverPt].step} · S {status.samples[hoverPt].s.toFixed(3)}
            </div>
          </div>
        ) : (
          <div style={{ color: '#667' }}>hover a state to see its image</div>
        )}
      </div>
    </div>
  );
}
