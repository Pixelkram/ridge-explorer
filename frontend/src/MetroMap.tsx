/**
 * MetroMap -- the Metropolis chains drawn on the space they sample, moving while they sample it.
 *
 * Same picture as CascadeMap and AmrMap, by the same arithmetic (`makeProjector`,
 * `viewTransform`): prompt vertices around the hull, the k=3 triangle exact, k>=4 a 2-D
 * shadow of the (k-1)-dimensional space that left-drag turns. Seven layers, the first two of
 * which belong to a phase that used to have no picture at all:
 *
 *   * the SEED CHORDS as faint lines and the SEED PROBES as dots on the divergence ramp,
 *     arriving one by one while the survey runs -- the Cascade's chord phase, in the Cascade's
 *     own visual language (line #4a4f8f, dot filled by `divColor`, grey while a probe has no
 *     chord neighbour to be read against yet). Before these existed the first minute or two of
 *     a run was a progress bar: the map mounted only once the first chain had a state, so the
 *     phase that DECIDES where every chain starts was the one phase nobody could watch;
 *   * the SEED crossings as hollow rings -- where the survey (or this run's own chords)
 *     handed the chains their starting sheets. They appear as they are detected, so a ring
 *     lands on the map while the chords around it are still being probed;
 *   * each chain's accepted states joined in order as a faint TRACE polyline -- the path is
 *     structure, not data, so it takes the map's existing chord token at low opacity and
 *     stays recessive under the dots. (This is deliberately below the 3:1 contrast a data
 *     mark needs: nothing is encoded by it alone, the legend names it, and the sample's own
 *     energy is in its fill, its tooltip and the gallery beside the map.)
 *   * the accepted STATES as dots filled by `divColor(S)` -- the same exported blue->red
 *     ramp, the same 0..0.50 top of scale, that the Cascade colours a probe's divergence
 *     with and the AMR map a cell's. S is that quantity, averaged over m directions instead
 *     of read across one lattice edge, so it belongs on the same ramp and gets no new one;
 *   * each chain's current state as a HEAD: the same ramp fill, larger, ringed in the path
 *     token, so "where the chains are now" reads apart from "where they have been". A head
 *     whose energy has not been measured yet -- every chain during the "initial energies"
 *     batch, and a hand-placed one for the round it joins at -- is drawn HOLLOW: it has a
 *     position and no reading, and filling it from the ramp would invent one;
 *   * the last round's PROPOSALS, for as long as they take to resolve.
 *
 * Motion is information, as it is on the cascade map. Every poll that brings a new round
 * animates that round's events: a proposal flashes where it was made, an accepted one bursts
 * in the accent while its chain's head glides to it over ~400 ms (a CSS transition on cx/cy,
 * so the browser interpolates and no JS runs per frame), and a rejected one fades out where
 * it was. The head LEAVING a position is what turns that position into a plain sample dot --
 * no extra mark is drawn for it, because the sample list already has it. Rotating the shadow
 * plane moves every projected point at once, which would read as all 60 chains gliding, so
 * the glide is switched off for the duration of a rotate drag. All animation is CSS keyframes
 * (reliable start-on-insert, unlike SMIL in SPAs) and off under prefers-reduced-motion.
 *
 * Identity is deliberately NOT a colour. 60 chains cannot have 60 distinguishable hues, and a
 * cycled palette would make two unrelated chains read as the same one, so every chain's trace
 * and head ring take ONE path token and a chain is picked out by FOLLOWING it (click a head,
 * or a row in the panel's chain table): the followed chain keeps full strength and everything
 * else dims. The two ramps on screen keep their jobs -- energy on the fill, nothing else.
 *
 * No auto-rotation, for AmrMap's reason: these are hundreds of dots hovered one at a time,
 * and a drifting projection would move the target under the pointer.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import type { MetroEvent, MetroStatus } from './api/types';
import { divColor, makeProjector } from './CascadeMap';
import { baryFromPoint } from './metroUtil';
import {
  HOME, dragFactor, isDrag, isHome, pan, reset, rotationDelta, svgTransform, toScreen,
  toWorld, wheelFactor, wrapAngle, zoomAbout,
} from './viewTransform';
import type { View } from './viewTransform';

/** the map's existing search-line token: a chain's trace is a path, like a chord */
const PATH = '#4a4f8f';
/** the crossing-dot neutral: the seeds a chain started from, and a proposal not taken */
const SEED = '#8a93b8';
const ACCENT = '#4ecca3';
const WARN = '#e94560';

const REDUCED = typeof window !== 'undefined'
  && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

/**
 * One burst per accepted proposal, one fade per rejected one, and the head glide.
 *
 * The glide is a transition rather than a keyframe because its endpoints are data: the head's
 * cx/cy are wherever the chain's state now projects to, and the browser interpolates from
 * whatever they were. 400 ms is long enough to be a move rather than a jump and short enough
 * to finish well inside the 1.2 s poll, so a head is never still travelling when the next
 * round's position arrives.
 */
const KEYFRAMES = `
  .mx-head { transition: cx 400ms ease-out, cy 400ms ease-out; }
  .mx-burst { transform-box: fill-box; transform-origin: center;
              animation: mxBurst .9s ease-out forwards; }
  @keyframes mxBurst { from { transform: scale(1); opacity: .95; }
                       to { transform: scale(3); opacity: 0; } }
  .mx-fade { animation: mxFade 1.1s ease-out forwards; }
  @keyframes mxFade { from { opacity: .85; } to { opacity: 0; } }
  .mx-flash { transform-box: fill-box; transform-origin: center;
              animation: mxFlash .5s ease-out forwards; }
  @keyframes mxFlash { from { transform: scale(2.2); opacity: 1; }
                       to { transform: scale(1); opacity: .9; } }
  @media (prefers-reduced-motion: reduce) {
    .mx-head { transition: none; }
    .mx-burst, .mx-flash { display: none; }
    .mx-fade { animation: none; opacity: .5; }
  }
`;

function short(p: string, n = 26) {
  return p.length > n ? p.slice(0, n - 1) + '…' : p;
}

// A 60 x 50 run accepts at most 3,000 states, which is well inside what SVG draws happily,
// so the only cap here is on the traces: 200 chains x 50 segments is more line than the
// eye gains from, and the chains with the most accepted states are the ones worth drawing.
const MAX_PATHS = 80;

export default function MetroMap({
  status, runId, size = 560, imageUrl, onImage, followed = null, onFollow, onSeed,
  live = false,
}: {
  status: MetroStatus;
  runId: string;
  size?: number;
  imageUrl: (runId: string, index: number) => string;
  /** click a sample: hand its thumbnail and recipe to the panel's large view */
  onImage?: (thumb: number, title: string, rows: [string, string][],
             weights?: number[] | null) => void;
  /** the chain being followed (others dim), owned by the panel so its table agrees */
  followed?: number | null;
  onFollow?: (chain: number | null) => void;
  /** start a new chain at this recipe; absent = the map offers no seeding */
  onSeed?: (w: number[], why: string) => void;
  /** the run is still going round, so seeding and following a moving head mean something */
  live?: boolean;
}) {
  const k = status.k || status.prompts.length;
  const [showPaths, setShowPaths] = useState(true);
  const [showSeeds, setShowSeeds] = useState(true);
  const [showSurvey, setShowSurvey] = useState(true);
  const [showHeads, setShowHeads] = useState(true);
  const [showSamples, setShowSamples] = useState(true);
  const [showRejected, setShowRejected] = useState(true);
  const [addMode, setAddMode] = useState(false);
  const [theta, setTheta] = useState(0);
  const [view, setView] = useState<View>(HOME);
  const [hover, setHover] = useState<number | null>(null);
  const [sel, setSel] = useState<number | null>(null);
  const [rotating, setRotating] = useState(false);
  const [hint, setHint] = useState<string | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const hovering = useRef(false);
  const gesture = useRef<{
    kind: 'pan' | 'zoom' | 'rotate';
    x: number; y: number; pid: number; view: View; theta: number; moved: boolean;
  } | null>(null);
  const dragged = useRef(false);
  const canRotate = k > 3;
  const canSeed = live && !!onSeed;

  // The seed survey. Optional on the wire: a backend that predates these fields serves a
  // status without them, and this map still has to draw the rest of the run.
  const chords = status.seed_chords ?? [];
  const probes = status.seed_probes ?? [];
  const seeding = status.status === 'running' && status.phase.startsWith('seed chords');

  // The round being animated. Only ever ONE: the heads are already at their final positions
  // in `chains_stats`, so a poll that fell two rounds behind replays the newest round's
  // proposals rather than queueing a backlog that would lie about where the chains are.
  const shownRound = useRef(0);
  const [anim, setAnim] = useState<{ round: number; events: MetroEvent[] } | null>(null);
  useEffect(() => {
    const ring = status.recent_events ?? [];
    const newest = ring.length ? ring[ring.length - 1] : null;
    if (!newest || newest.round <= shownRound.current) return;
    shownRound.current = newest.round;
    setAnim({ round: newest.round, events: newest.events });
  }, [status.recent_events]);
  // a fresh run reuses this component, so the "already shown" mark has to go back with it
  useEffect(() => {
    shownRound.current = 0;
    setAnim(null);
    setSel(null);
  }, [runId]);

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
      if (g.kind === 'rotate') {
        svgRef.current?.setPointerCapture(g.pid);
        // turning the plane re-projects every point; without this the whole chain set would
        // read as gliding, which is the one motion on this map that means something
        setRotating(true);
      }
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
    setRotating(false);
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
  const glide = !REDUCED && !rotating;

  // One polyline per chain: its seed, then its accepted states in the order they were
  // accepted. The chains that accepted most are the ones kept when the cap bites; a FOLLOWED
  // chain is always kept, or following it would hide the very thing being looked at.
  const paths = useMemo(() => {
    const by = new Map<number, number[]>();
    status.samples.forEach((s, i) => {
      const a = by.get(s.chain);
      if (a) a.push(i); else by.set(s.chain, [i]);
    });
    const seedOf = new Map<number, number[]>();
    for (const c of status.chains_stats) seedOf.set(c.chain, c.seed_weights);
    const ranked = [...by.entries()].sort((a, b) => b[1].length - a[1].length);
    const keep = ranked.slice(0, MAX_PATHS);
    if (followed !== null && !keep.some(([ch]) => ch === followed)) {
      const extra = ranked.find(([ch]) => ch === followed);
      if (extra) keep.push(extra);
    }
    return keep.map(([chain, idx]) => ({ chain, idx, seed: seedOf.get(chain) ?? null }));
  }, [status.samples.length, status.chains_stats.length, followed]);

  const dim = (chain: number) => (followed === null || followed === chain ? 1 : 0.16);
  const hoverPt = hover !== null && hover < status.samples.length ? hover : null;

  const select = (i: number) => {
    if (dragged.current || addMode) return;    // in add-mode every click means "seed here"
    const s = status.samples[i];
    if (!s || s.image < 0) return;
    setSel(i);
    onImage?.(s.full_image ?? s.image,
              `sample ${i} · chain ${s.chain} · round ${s.step}`, [
                ['energy S', s.s.toFixed(3)],
                ['past the 0.35 threshold', s.s > 0.35 ? 'yes' : 'no'],
                ['rendered at', s.full_image != null
                  ? `${status.steps} steps (full pass)`
                  : `${status.probe_steps ?? status.steps} steps (cheap field)`],
              ], s.weights);
  };

  /** A click on the hull in add-mode: only k=3 has an inverse to use (see baryFromPoint). */
  const seedAtClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!addMode || !canSeed || dragged.current) return;
    if (k !== 3) {
      setHint('clicking cannot name a recipe at k ≥ 4 — use the two buttons below');
      return;
    }
    const [sx, sy] = at(e);
    const [wx, wy] = toWorld(view, sx, sy);
    const w = baryFromPoint(vs, wx, wy);
    if (!w) return;
    if (w.some((v) => v < 0)) {
      setHint('that point is outside the triangle — a recipe has no negative weights');
      return;
    }
    setHint(null);
    onSeed?.(w, 'clicked on the map');
  };

  const seedAtSample = () => {
    const i = sel ?? hoverPt;
    if (i === null || !status.samples[i]) {
      setHint('select a sample first (click one on the map)');
      return;
    }
    setHint(null);
    onSeed?.(status.samples[i].weights, `sample ${i}`);
  };

  const seedAtCrossing = () => {
    if (!status.seeds.length) {
      setHint('this run found no seed crossings to start from');
      return;
    }
    const i = Math.floor(Math.random() * status.seeds.length);
    setHint(null);
    onSeed?.(status.seeds[i], `seed crossing ${i}`);
  };

  if (k < 3) return null;

  const toggle = (on: boolean, set: (v: boolean) => void, label: string, tip: string) => (
    <label className="rx-focus" title={tip}
           style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 3,
                    color: '#aeb6dd', cursor: 'pointer' }}>
      <input type="checkbox" checked={on} onChange={(e) => set(e.target.checked)} />
      {label}
    </label>
  );

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
           onClick={seedAtClick}
           onContextMenu={(e) => e.preventDefault()}
           style={{ background: '#0d0d20', borderRadius: 8,
                    touchAction: 'manipulation', userSelect: 'none',
                    cursor: addMode && canSeed && k === 3 ? 'crosshair' : undefined,
                    outline: addMode && canSeed ? `1px solid ${ACCENT}` : undefined }}>
        <style>{KEYFRAMES}</style>
        <g transform={svgTransform(view)}>
          <polygon points={hullOrder.map(([x, y]) => `${x},${y}`).join(' ')}
                   fill="#151538" stroke="#34346a" strokeWidth={q(1.5)} />
          {/* The seed survey, UNDER everything the chains do: the chords it probed along and
              the probes themselves, which arrive one by one. Faint lines and small dots on
              purpose -- by the time the chains are running this is the ground they started
              from, and during the seed phase it is the only thing on the map. Pointer events
              off: there is nothing to click here, and these dots must never take a click
              meant for a sample or a head sitting on top of them. */}
          {showSurvey && (
            <g style={{ pointerEvents: 'none' }}>
              {chords.map((c, i) => {
                if (!c || c.length < 2) return null;
                const [x1, y1] = proj.project(c[0]);
                const [x2, y2] = proj.project(c[1]);
                if (![x1, y1, x2, y2].every((v) => Number.isFinite(v))) return null;
                return <line key={`sc${i}`} x1={x1} y1={y1} x2={x2} y2={y2} stroke={PATH}
                             strokeWidth={q(1)} strokeOpacity={0.5} />;
              })}
              {probes.map((p, i) => {
                const [x, y] = proj.project(p.w);
                if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
                const d = p.div ?? null;
                // the Cascade's own cloud rule: a measured probe grows with its reading, an
                // unpaired one stays small and grey -- "not read yet", not "quiet"
                const hot = d !== null ? Math.min(1, d / 0.5) : 0;
                return <circle key={`sp${i}`} cx={x} cy={y}
                               r={q(d !== null ? 1.8 + 1.4 * hot : 1.5)} fill={divColor(d)}
                               fillOpacity={d !== null ? 0.9 : 0.55} />;
              })}
            </g>
          )}
          {/* chain traces UNDER everything: structure, not data */}
          {showPaths && (
            <g style={{ pointerEvents: 'none' }}>
              {paths.map(({ chain, idx, seed }) => {
                const pts = (seed ? [seed] : [])
                  .concat(idx.map((i) => status.samples[i].weights))
                  .map((w) => proj.project(w))
                  .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
                if (pts.length < 2) return null;
                const on = followed === chain;
                return <polyline key={`c${chain}`} fill="none"
                                 stroke={on ? ACCENT : PATH}
                                 strokeWidth={q(on ? 1.8 : 1.2)}
                                 strokeOpacity={(on ? 0.9 : 0.45) * dim(chain)}
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
          {showSamples && status.samples.map((s, i) => {
            const [x, y] = proj.project(s.weights);
            if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
            return (
              <circle key={`p${i}`} cx={x} cy={y} r={q(sel === i ? 3.8 : 3)}
                      fill={divColor(s.s)}
                      fillOpacity={0.95 * dim(s.chain)}
                      stroke={hover === i || sel === i ? '#fff' : 'none'}
                      strokeWidth={q(1.4)} strokeOpacity={dim(s.chain)}
                      style={{ cursor: 'pointer' }}
                      onPointerEnter={() => setHover(i)}
                      onClick={() => select(i)} />
            );
          })}
          {/* the last round's proposals, resolving: a burst where one was accepted (the head
              is on its way there), a fading dot where one was not */}
          {anim && !REDUCED && (
            <g style={{ pointerEvents: 'none' }}>
              {anim.events.map((e, i) => {
                const [x, y] = proj.project(e.w_prop);
                if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
                if (followed !== null && followed !== e.chain) return null;
                if (e.accepted) {
                  return (
                    <g key={`a${anim.round}.${i}`}>
                      <circle className="mx-burst" cx={x} cy={y} r={q(5)} fill="none"
                              stroke={ACCENT} strokeWidth={q(1.8)} />
                      <circle className="mx-flash" cx={x} cy={y} r={q(2.2)}
                              fill={ACCENT} />
                    </g>
                  );
                }
                if (!showRejected) return null;
                return (
                  <circle key={`r${anim.round}.${i}`} className="mx-fade" cx={x} cy={y}
                          r={q(2.6)} fill={SEED} stroke={SEED} strokeWidth={q(0.8)}
                          fillOpacity={0.35} />
                );
              })}
            </g>
          )}
          {/* the heads: where every chain is NOW. A chain whose energy has not been measured
              yet (the whole set during "initial energies", a hand-placed one for the round it
              joins at) is hollow: it has a position and no reading. */}
          {showHeads && status.chains_stats.map((c) => {
            const [x, y] = proj.project(c.weights);
            if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
            const on = followed === c.chain;
            const s = c.s ?? null;
            const unread = s === null;
            return (
              <circle key={`h${c.chain}`} className={glide ? 'mx-head' : undefined}
                      cx={x} cy={y} r={q(on ? 5.6 : 4.6)}
                      fill={unread ? 'none' : divColor(s)}
                      fillOpacity={0.98 * dim(c.chain)}
                      stroke={on ? '#fff' : c.stopped ? SEED : PATH}
                      strokeWidth={q(on ? 2.2 : 1.8)}
                      strokeOpacity={(c.stopped ? 0.7 : 1) * dim(c.chain)}
                      strokeDasharray={c.stopped ? `${q(2)},${q(2)}` : undefined}
                      style={{ cursor: addMode ? 'crosshair' : 'pointer' }}
                      onClick={() => {
                        if (dragged.current || addMode) return;
                        onFollow?.(on ? null : c.chain);
                      }}>
                <title>
                  {`chain ${c.chain} · `
                   + (unread
                     ? (status.phase === 'initial energies'
                       ? 'S not measured yet — its energy lands in this batch'
                       : 'no starting energy: its probes never arrived, so it sits out the '
                         + 'run (the run\'s notes count these)')
                     : `S ${s.toFixed(3)} · ${c.n_accept}/${c.n_propose} accepted`)
                   + (c.stopped ? ' · frozen' : '')
                   + (c.joined_round ? ` · added at round ${c.joined_round}` : '')}
                </title>
              </circle>
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
        {/* during the seed phase the map IS the survey, so it says so rather than leaving a
            half-drawn picture to be read as a finished one */}
        {seeding && (
          <text x={10} y={size - 38} fill={SEED} fontSize={10}
                style={{ fontVariantNumeric: 'tabular-nums' }}>
            seeding: {chords.length} chord{chords.length === 1 ? '' : 's'} ·{' '}
            {probes.length} probes · {status.seeds.length} crossings so far
          </text>
        )}
        {followed !== null && (
          <text x={10} y={size - 24} fill={ACCENT} fontSize={10}>
            following chain {followed} — click its head again to release
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
        {toggle(showHeads, setShowHeads, 'chain heads',
                'where every chain is NOW: one larger dot per chain, ringed in the trace '
                + 'colour, filled by its own energy on the ramp above. A head with no fill '
                + 'has no energy yet — it is waiting for the batch that measures it. It '
                + 'glides to each accepted proposal, and a frozen chain\'s ring goes dashed.')}
        {toggle(showSamples, setShowSamples, 'accepted states',
                'every state any chain has accepted, filled by its energy. This is the '
                + 'sample set — the gallery below shows the same states as images.')}
        {toggle(showPaths, setShowPaths, 'chain traces',
                'each chain\'s accepted states joined in the order it accepted them, '
                + 'starting at its seed crossing. The trace is structure, so it is drawn '
                + 'recessively — nothing is encoded by it alone.')}
        {toggle(showRejected, setShowRejected, 'show rejected',
                'proposals the chain did not take, fading where they were made. Includes '
                + 'the ones that left the simplex (drawn outside the hull, which is where '
                + 'they are), so the acceptance rate on screen is explainable from the '
                + 'picture rather than only from the counter.')}
        {toggle(showSeeds, setShowSeeds, 'seed crossings',
                'the crossings the chains were seeded at: either this run\'s own IUR chords '
                + 'or a finished Cascade\'s. Hollow, so a seed never reads as a sample.')}
        {(chords.length > 0 || probes.length > 0) && toggle(
          showSurvey, setShowSurvey, 'seed survey',
          'the chords the seeds were looked for along, and every probe of them, on the same '
          + 'divergence ramp as everything else. Each pair of adjacent probes past 0.35 is a '
          + 'crossing, and that is where a chain starts — so this layer is the evidence '
          + 'behind the rings. A probe still waiting for its neighbour is small and grey: '
          + 'not read yet, rather than quiet. In cascade seed mode the chords are the SOURCE '
          + 'run\'s and there are no probes of this run\'s own.')}
        <div style={{ margin: '6px 0' }}>
          <span style={{ color: divColor(0.5) }}>●</span> accepted state — colour = its
          energy S
          <br /><span style={{ color: divColor(0.35) }}>◉</span> a chain's head, where it is
          now <span style={{ color: '#667' }}>(hollow = energy not measured yet)</span>
          <br /><span style={{ color: ACCENT }}>◌</span> proposal accepted (the head is on
          its way)
          <br /><span style={{ color: SEED }}>•</span> proposal rejected, fading out
          <br /><span style={{ color: PATH }}>―</span> one chain's trace, seed first
          <br /><span style={{ color: SEED }}>◦</span> seed crossing
          {(chords.length > 0 || probes.length > 0) && (
            <>
              <br /><span style={{ color: PATH }}>―</span> seed chord ·{' '}
              <span style={{ color: divColor(0.2) }}>·</span> its probes
            </>
          )}
        </div>
        {canSeed && (
          <div style={{ borderTop: '1px solid #2a2a4a', paddingTop: 5, marginBottom: 6 }}>
            {toggle(addMode, setAddMode, 'add chain here',
                    k === 3
                      ? 'on: a click inside the triangle starts a new chain at that recipe. '
                        + 'Its head appears at the next round, once its energy has been '
                        + 'measured in that round\'s batch.'
                      : 'at k ≥ 4 the map is a 2-D shadow of a (k−1)-dimensional space, so '
                        + 'one pixel is a whole fibre of recipes and a click cannot name '
                        + 'one. Use the two buttons below, which name a recipe exactly.')}
            {k !== 3 && (
              <div style={{ color: '#667', marginBottom: 4 }}>
                k ≥ 4: a click is ambiguous under the projection, so seed at a recipe that
                already exists.
              </div>
            )}
            <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
              <button onClick={seedAtSample}
                      title="start a chain at the sample selected on the map — a state this
run already drew, so the recipe is exact at any k"
                      style={{ background: '#0a0a1a', color: '#8a9',
                               border: '1px solid #444', borderRadius: 3, fontSize: 10,
                               padding: '2px 5px', cursor: 'pointer' }}>
                seed at selected sample
              </button>
              <button onClick={seedAtCrossing}
                      title="start a chain at one of this run's seed crossings, picked at
random — the same kind of start the run's own chains were given"
                      style={{ background: '#0a0a1a', color: '#8a9',
                               border: '1px solid #444', borderRadius: 3, fontSize: 10,
                               padding: '2px 5px', cursor: 'pointer' }}>
                seed at a random crossing
              </button>
            </div>
            {hint && (
              <div style={{ color: '#c9a227', marginTop: 3 }}>{hint}</div>
            )}
          </div>
        )}
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
          <div style={{ color: '#667' }}>
            {seeding
              ? 'seeding: the chords and their probes are going up as they come back — the '
                + 'chains appear once the survey has crossings to start them on'
              : 'hover a state to see its image · click a head to follow its chain'}
          </div>
        )}
      </div>
    </div>
  );
}
