/**
 * CascadeMap -- the cascade drawn on the space it searches, with the effort visible.
 *
 * The intuitive picture IS the simplex: prompt vertices around a polygon, chords as
 * search lines, crossings as dots that appear when found and turn green when certified,
 * patch diamonds where dense refinement happened, and a live cloud of every generated
 * image coloured by its measured local divergence (blue quiet -> red hot).
 *
 * Motion is information here, not decoration: chords DRAW themselves in as the survey
 * lays them down, each newly generated image lands with a sonar ping, unscored
 * crossings PULSE while the algorithm is still working on them, certification fires a
 * one-shot green burst, and an active walk marches its dashes while its probe bundle
 * shows the instrument sampling across the ridge. All animation is CSS keyframes
 * (reliable start-on-insert, unlike SMIL in SPAs) and disabled under
 * prefers-reduced-motion.
 *
 * k=3 is the exact static triangle. k>3 is a 2-D shadow of the (k-1)-dimensional
 * space: single view with a slowly rotating shadow plane (depth reads as parallax),
 * or SPLIT VIEW -- every axis-pair anchor rendered side by side, synchronised in
 * angle, rotation, and selection.
 */
import { useEffect, useRef, useState } from 'react';
import type { CascadeStatus } from './api/types';

const ACCENT = '#4ecca3';
const WARN = '#e94560';

const REDUCED = typeof window !== 'undefined'
  && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

// blue (quiet) -> violet -> red (hot) ramp for local divergence
function divColor(d: number | null): string {
  if (d === null) return '#565e93';
  const t = Math.min(1, Math.max(0, d / 0.5));
  const lerp = (a: number, b: number) => Math.round(a + (b - a) * t);
  return `rgb(${lerp(75, 233)},${lerp(90, 69)},${lerp(160, 96)})`;
}

function vertexPositions(k: number, r: number, cx: number, cy: number) {
  return Array.from({ length: k }, (_, i) => {
    const th = -Math.PI / 2 + (2 * Math.PI * i) / k;
    return [cx + r * Math.cos(th), cy + r * Math.sin(th)] as const;
  });
}

// Helmert basis: orthonormal, sum-zero rows -- the natural chart of the simplex's
// (k-1)-dimensional affine hull. Row i has i ones then -i, scaled.
function helmert(k: number): number[][] {
  const B: number[][] = [];
  for (let i = 1; i < k; i++) {
    const row = new Array(k).fill(0);
    for (let j = 0; j < i; j++) row[j] = 1 / Math.sqrt(i * (i + 1));
    row[i] = -i / Math.sqrt(i * (i + 1));
    B.push(row);
  }
  return B;
}

/**
 * A projector from the simplex to screen coordinates. For k=3, the familiar static
 * triangle. For k>3, a 2-D shadow plane anchored at Helmert axis pair (base, base+1)
 * and rotating toward (base+2, base+3); when the rotation partners would collide with
 * the anchors (small k), v stays fixed to keep u ⊥ v exact -- a sheared shadow would
 * lie about distances.
 */
function makeProjector(k: number, theta: number, size: number, base = 0) {
  const cx = size / 2, cy = size / 2, r = size / 2 - 34;
  if (k === 3) {
    const vs = vertexPositions(3, r, cx, cy);
    return {
      project: (w: number[]) => {
        let x = 0, y = 0;
        for (let i = 0; i < 3 && i < w.length; i++) {
          x += w[i] * vs[i][0];
          y += w[i] * vs[i][1];
        }
        return [x, y] as const;
      },
    };
  }
  const B = helmert(k);
  const L = B.length;
  const c = Math.cos(theta), s = Math.sin(theta);
  const u = new Array(k).fill(0);
  const v = new Array(k).fill(0);
  const p = base % L, q = (base + 1) % L;
  const rIdx = (base + 2) % L, sIdx = (base + 3) % L;
  const canRotV = L >= 4 && sIdx !== p && sIdx !== rIdx;
  for (let j = 0; j < k; j++) {
    u[j] = c * B[p][j] + s * B[rIdx][j];
    v[j] = canRotV ? c * B[q][j] + s * B[sIdx][j] : B[q][j];
  }
  let rmax = 1e-9;
  for (let i = 0; i < k; i++) {
    const d = Math.hypot(u[i], v[i]);
    if (d > rmax) rmax = d;
  }
  const scale = r / rmax;
  return {
    project: (w: number[]) => {
      let x = 0, y = 0;
      for (let j = 0; j < k && j < w.length; j++) {
        x += w[j] * u[j];
        y += w[j] * v[j];
      }
      return [cx + x * scale, cy + y * scale] as const;
    },
  };
}

function short(p: string, n = 26) {
  return p.length > n ? p.slice(0, n - 1) + '…' : p;
}

const KEYFRAMES = `
  .cs-ping { transform-box: fill-box; transform-origin: center;
             animation: csPing 1.1s ease-out forwards; }
  @keyframes csPing { from { transform: scale(1); opacity: .9; }
                      to { transform: scale(7); opacity: 0; } }
  .cs-draw { stroke-dasharray: 1; stroke-dashoffset: 1;
             animation: csDraw .9s ease-out forwards; }
  @keyframes csDraw { to { stroke-dashoffset: 0; } }
  .cs-pulse { animation: csPulse 1.4s ease-in-out infinite; }
  @keyframes csPulse { 0%,100% { fill-opacity: .35; } 50% { fill-opacity: .9; } }
  .cs-burst { transform-box: fill-box; transform-origin: center;
              animation: csBurst .9s ease-out forwards; }
  @keyframes csBurst { from { transform: scale(1); opacity: 1; }
                       to { transform: scale(3.2); opacity: 0; } }
  .cs-march { animation: csMarch .8s linear infinite; }
  @keyframes csMarch { from { stroke-dashoffset: 22; } to { stroke-dashoffset: 0; } }
  .cs-beat { animation: csBeat 1.2s ease-in-out infinite; }
  @keyframes csBeat { 0%,100% { fill-opacity: 1; } 50% { fill-opacity: .25; } }
  .cs-spulse { animation: csSPulse 1.1s ease-in-out infinite; }
  @keyframes csSPulse { 0%,100% { stroke-opacity: 1; } 50% { stroke-opacity: .3; } }
  .cs-shimmer { animation: csShimmer 1.3s ease-in-out infinite; }
  @keyframes csShimmer { 0%,100% { background: #0a0a1a; } 50% { background: #23234d; } }
  @media (prefers-reduced-motion: reduce) {
    .cs-ping, .cs-burst { display: none; }
    .cs-draw { stroke-dasharray: none; stroke-dashoffset: 0; animation: none; }
    .cs-pulse, .cs-march, .cs-beat, .cs-spulse, .cs-shimmer { animation: none; }
  }
`;

/** One rendered shadow of the space; used once in single view, per-anchor in split. */
function MapSvg({
  status, size, theta, base, sel, onPick, walkPath, walkSegs,
  caption, showBadge, showBeat, spin, compact,
}: {
  status: CascadeStatus;
  size: number;
  theta: number;
  base: number;
  sel: number | null;
  onPick: (cid: number | null) => void;
  walkPath?: number[][] | null;
  walkSegs?: number[][][] | null;
  caption?: string;
  showBadge?: boolean;
  showBeat?: boolean;
  spin?: boolean;
  compact?: boolean;
}) {
  const prevLen = useRef(0);
  const newFrom = prevLen.current;
  useEffect(() => { prevLen.current = status.points.length; });

  const k = status.prompts.length;
  const cx = size / 2, cy = size / 2, r = size / 2 - 34;
  const proj = makeProjector(k, theta, size, base);
  const vraw = Array.from({ length: k }, (_, i) => {
    const e = new Array(k).fill(0); e[i] = 1;
    return [i, proj.project(e)] as const;
  });
  const vs = vraw.map(([, p]) => p);
  const hullOrder = [...vraw].sort((a, b) => {
    const aa = Math.atan2(a[1][1] - cy, a[1][0] - cx);
    const bb = Math.atan2(b[1][1] - cy, b[1][0] - cx);
    return aa - bb;
  });
  const selected = status.crossings.find((c) => c.cid === sel) ?? null;
  const selGroup = selected?.ridge_group ?? null;
  const running = status.status === 'running';
  const vscale = vs.length > 1
    ? Math.hypot(vs[0][0] - vs[1][0], vs[0][1] - vs[1][1]) / Math.SQRT2 : size / 2;
  const walking = !!walkPath && walkPath.length > 0;
  const maxPts = compact ? 900 : 1800;

  return (
    <svg width={size} height={size} style={{ background: '#0d0d20', borderRadius: 8,
                                             touchAction: 'manipulation' }}>
      <style>{KEYFRAMES}</style>
      <polygon points={hullOrder.map(([, [x, y]]) => `${x},${y}`).join(' ')}
               fill="#151538" stroke="#34346a" strokeWidth={1.5} />
      {(() => {
        const n = status.points.length;
        const stride = n > maxPts ? Math.ceil(n / maxPts) : 1;
        const out = [];
        for (let i = 0; i < n; i += stride) {
          const [x, y] = proj.project(status.points[i]);
          const d = status.point_divs[i] ?? null;
          const hot = d !== null ? Math.min(1, d / 0.5) : 0;
          out.push(
            <circle key={`pt${i}`} cx={x} cy={y}
                    r={(d !== null ? 2.4 + 1.8 * hot : 2.1) * (compact ? 0.8 : 1)}
                    fill={divColor(d)} fillOpacity={d !== null ? 0.9 : 0.6} />);
        }
        return out;
      })()}
      {!REDUCED && running && status.points.slice(newFrom).map((w, i) => {
        const [x, y] = proj.project(w);
        return (
          <circle key={`ping${newFrom + i}`} className="cs-ping" cx={x} cy={y}
                  r={2.4} fill="none" stroke="#9aa3e0" strokeWidth={1.5} />
        );
      })}
      {status.chords.map((c, i) => {
        const [x1, y1] = proj.project(c.a);
        const [x2, y2] = proj.project(c.b);
        return (
          <line key={i} className="cs-draw" x1={x1} y1={y1} x2={x2} y2={y2}
                stroke="#4a4f8f" strokeWidth={compact ? 1.4 : 2} pathLength={1}
                style={{ animationDelay: `${(i % 8) * 0.12}s` }} />
        );
      })}
      {walkSegs && walkSegs.map((seg, i) => {
        const [x1, y1] = proj.project(seg[0]);
        const [x2, y2] = proj.project(seg[1]);
        return <line key={`ws${i}`} x1={x1} y1={y1} x2={x2} y2={y2}
                     stroke={WARN} strokeWidth={1.8} strokeOpacity={0.6} />;
      })}
      {status.patches.map((p) => {
        const c = status.crossings.find((x) => x.cid === p.cid);
        if (!c) return null;
        const [x, y] = proj.project(c.weights);
        const h = compact ? 8 : 11;
        const filling = running && p.grid.some((row) => row.some((t) => t < 0));
        return (
          <rect key={`p${p.region}`} x={x - h} y={y - h} width={2 * h} height={2 * h}
                fill="none" stroke={filling ? WARN
                                    : p.significant ? ACCENT : '#8a93b8'}
                strokeWidth={compact ? 1.6 : 2.2}
                strokeDasharray={filling || p.exploration ? '4,3' : undefined}
                className={filling ? 'cs-march cs-spulse' : undefined}
                transform={`rotate(45 ${x} ${y})`} />
        );
      })}
      {status.crossings.map((c) => {
        const [x, y] = proj.project(c.weights);
        const scored = c.b !== null;
        const rad = (scored ? 4.5 + 6 * Math.min(1, Math.max(0, c.b!)) : 4.5)
          * (compact ? 0.75 : 1);
        const open_ = running && !scored && c.bracket_w !== null
          && c.bracket_w > 0.014;
        return (
          <g key={c.cid}>
            {open_ && (
              <circle className="cs-march" cx={x} cy={y}
                      r={Math.max(9, c.bracket_w! * vscale * 4)}
                      fill="none" stroke={WARN} strokeWidth={1.4}
                      strokeDasharray="5,6" strokeOpacity={0.85} />
            )}
            {!REDUCED && c.significant && (
              <circle key={`burst-${c.cid}-sig`} className="cs-burst" cx={x} cy={y}
                      r={rad + 2} fill="none" stroke={ACCENT} strokeWidth={2} />
            )}
            <circle cx={x} cy={y} r={rad}
                    fill={!scored ? '#7d84c8' : c.significant ? ACCENT : '#8a93b8'}
                    fillOpacity={scored ? 0.95 : 0.7}
                    stroke={sel === c.cid ? '#fff'
                            : (selGroup !== null && c.ridge_group === selGroup)
                              ? WARN : '#0d0d20'}
                    strokeWidth={sel === c.cid ? 2.2 : 1.2}
                    className={!scored && running ? 'cs-pulse' : undefined}
                    style={{ cursor: 'pointer' }}
                    onClick={() => onPick(sel === c.cid ? null : c.cid)} />
          </g>
        );
      })}
      {walkPath && walkPath.length > 1 && (
        <g>
          <polyline className="cs-march"
                    points={walkPath.map((w) => proj.project(w).join(',')).join(' ')}
                    fill="none" stroke={WARN} strokeWidth={2.6}
                    strokeDasharray="6,5" />
          {walkPath.map((w, i) => {
            const [x, y] = proj.project(w);
            return <circle key={`wk${i}`}
                           cx={x} cy={y} r={i === walkPath.length - 1 ? 5 : 3.2}
                           fill={WARN} />;
          })}
        </g>
      )}
      {vs.map(([x, y], i) => {
        const outward = 14;
        const dx = (x - cx) / r, dy = (y - cy) / r;
        const anchor = Math.abs(dx) < 0.35 ? 'middle' : dx > 0 ? 'start' : 'end';
        return (
          <g key={i}>
            <circle cx={x} cy={y} r={compact ? 3.2 : 4.5} fill="#dfe3ff" />
            <text x={x + dx * outward} y={y + dy * outward + 3}
                  fill="#aeb6dd" fontSize={compact ? 9 : 11} textAnchor={anchor}>
              {short(status.prompts[i], compact ? 13 : 26)}
            </text>
          </g>
        );
      })}
      {showBadge && k > 3 && (
        <text x={size - 12} y={20} textAnchor="end" fill="#7d84c8" fontSize={11}>
          {k - 1}-D space · 2-D shadow{spin ? ' · rotating' : ' · paused'}
        </text>
      )}
      {caption && (
        <text x={10} y={size - 10} fill="#7d84c8" fontSize={10}>{caption}</text>
      )}
      {showBeat && !REDUCED && (running || walking) && (
        <circle className="cs-beat" cx={16} cy={16} r={5}
                fill={walking ? WARN : ACCENT} />
      )}
    </svg>
  );
}

export default function CascadeMap({
  status, runId, imageUrl, size = 340, onSelect, onImage, walkPath, walkSegs,
  selectedCid,
}: {
  status: CascadeStatus;
  runId: string;
  imageUrl: (runId: string, index: number) => string;
  size?: number;
  onSelect?: (cid: number | null) => void;
  // open the full detail view (same modal as the strips/grids below the map)
  onImage?: (thumb: number, title: string, rows: [string, string][]) => void;
  walkPath?: number[][] | null;
  walkSegs?: number[][][] | null;
  selectedCid?: number | null;
}) {
  const [internalSel, setInternalSel] = useState<number | null>(null);
  const sel = selectedCid !== undefined ? selectedCid : internalSel;
  const [theta, setTheta] = useState(0);
  const [spin, setSpin] = useState(!REDUCED);
  const [basePair, setBasePair] = useState(0);
  const [split, setSplit] = useState(false);
  // draggable view scale: the divider below the map sets this; persisted so the
  // preferred size survives reloads (same spirit as resume.ts, but a plain number)
  const [mapSize, setMapSize] = useState(() => {
    try {
      const v = Number(localStorage.getItem('ridge.cascadeMapSize.v1'));
      if (Number.isFinite(v) && v >= 280 && v <= 1200) return v;
    } catch { /* private mode etc. -- fall through */ }
    return size;
  });
  const drag = useRef<{ y: number; s: number } | null>(null);
  const saveSize = (v: number) => {
    try { localStorage.setItem('ridge.cascadeMapSize.v1', String(v)); }
    catch { /* best effort */ }
  };

  const k = status.prompts.length;
  const rotating = k > 3 && spin;
  useEffect(() => {
    if (!rotating) return;
    const iv = window.setInterval(
      () => setTheta((t) => (t + 0.012) % (Math.PI * 2)), 80);
    return () => window.clearInterval(iv);
  }, [rotating]);
  if (k < 3) return null;

  const pick = (cid: number | null) => {
    if (cid !== null) setSpin(false);
    setInternalSel(cid);
    onSelect?.(cid);
  };
  const patchByCid = new Map(status.patches.map((p) => [p.cid, p]));
  const selected = status.crossings.find((c) => c.cid === sel) ?? null;
  const nViews = k - 1;
  const tile = Math.max(180, Math.round(
    mapSize * (nViews <= 4 ? 0.47 : 0.36)));
  const splitOn = split && k > 3;
  const deg = Math.round((theta * 180) / Math.PI) % 360;

  return (
    <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'flex-start' }}>
      {splitOn ? (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8,
                      maxWidth: mapSize + 210 }}>
          {Array.from({ length: nViews }, (_, b) => (
            <MapSvg key={b} status={status} size={tile} theta={theta} base={b}
                    sel={sel} onPick={pick} walkPath={walkPath} walkSegs={walkSegs}
                    caption={`view ${b + 1}·${(b % nViews) + 2 > nViews ? 1 : b + 2}`}
                    showBeat={b === 0} compact />
          ))}
        </div>
      ) : (
        <MapSvg status={status} size={mapSize} theta={theta} base={basePair}
                sel={sel} onPick={pick} walkPath={walkPath} walkSegs={walkSegs}
                showBadge showBeat spin={spin} />
      )}
      <div style={{ fontSize: 11, color: '#889', maxWidth: 190 }}>
        <div style={{ marginBottom: 6 }}>
          <span style={{ color: '#7d84c8' }}>·</span> generated images — colour = local
          divergence (<span style={{ color: '#5a64b0' }}>quiet</span> → <span
          style={{ color: WARN }}>hot</span>), pings mark fresh ones
          <br /><span style={{ color: '#4a4f8f' }}>―</span> search chords (drawn as laid)
          <br /><span style={{ color: '#7d84c8' }}>●</span> boundary found (pulsing =
          still being worked on)
          <br /><span style={{ color: ACCENT }}>●</span> certified — beats the run's own
          background; size = strength
          <br /><span style={{ color: ACCENT }}>◇</span> refined patch
          {' '}<span style={{ color: '#8a93b8' }}>(dashed = exploration slot)</span>
          <br /><span style={{ color: WARN }}>◦</span> same ridge as selection ·{' '}
          <span style={{ color: WARN }}>‖</span> walk probe rungs
          {k > 3 && (
            <>
              <br /><span style={{ color: '#667' }}>
                k={k}: the space is {k - 1}-dimensional — the shadow plane rotates
                slowly so depth shows as parallax. Split view shows every axis anchor
                at once, synchronised in angle and selection
              </span>
            </>
          )}
        </div>
        {selected && selected.thumb >= 0 && (
          <div>
            <img src={imageUrl(runId, selected.thumb)} width={140} height={140}
                 title={onImage ? 'click to enlarge & explore from here' : undefined}
                 onClick={() => onImage?.(selected.thumb, `crossing ${selected.cid}`, [
                   ['boundary strength B',
                    selected.b !== null ? selected.b.toFixed(3) : 'unscored'],
                   ['certified',
                    selected.significant ? 'yes — beats background p95' : 'no'],
                   ...(selected.ridge_group !== null
                     ? [['ridge group', `r${selected.ridge_group}`] as [string, string]]
                     : []),
                 ])}
                 style={{ objectFit: 'cover', borderRadius: 6,
                          cursor: onImage ? 'pointer' : undefined,
                          border: `2px solid ${selected.significant ? ACCENT : '#8a93b8'}` }} />
            <div style={{ color: selected.significant ? ACCENT : '#8a93b8', marginTop: 2 }}>
              crossing {selected.cid}
              {selected.b !== null && ` · B=${selected.b.toFixed(2)}`}
              {selected.significant ? ' · certified' : ''}
              {patchByCid.has(selected.cid) ? ' · refined' : ''}
            </div>
          </div>
        )}
        {!selected && status.crossings.length > 0 && (
          <div style={{ color: '#667' }}>tap a dot to see its image</div>
        )}
      </div>
      <div role="separator" aria-orientation="horizontal"
           title="drag to resize the map"
           onPointerDown={(e) => {
             drag.current = { y: e.clientY, s: mapSize };
             (e.target as HTMLElement).setPointerCapture(e.pointerId);
           }}
           onPointerMove={(e) => {
             if (!drag.current) return;
             const next = Math.round(drag.current.s + (e.clientY - drag.current.y));
             setMapSize(Math.min(1200, Math.max(280, next)));
           }}
           onPointerUp={() => {
             if (drag.current) saveSize(mapSize);
             drag.current = null;
           }}
           onPointerCancel={() => {
             if (drag.current) saveSize(mapSize);
             drag.current = null;
           }}
           style={{ flexBasis: '100%', height: 12, cursor: 'ns-resize',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    touchAction: 'none', userSelect: 'none' }}>
        <div style={{ width: 64, height: 4, borderRadius: 2, background: '#3a3f6e' }} />
      </div>
      {k > 3 && (
        <div style={{ width: splitOn ? mapSize + 210 : mapSize, display: 'flex', gap: 8,
                      alignItems: 'center', flexBasis: '100%', marginTop: -4 }}>
          <button onClick={() => setSpin(!spin)}
                  style={{ background: '#0a0a1a', color: spin ? WARN : ACCENT,
                           border: '1px solid #444', borderRadius: 3,
                           padding: '3px 10px', fontSize: 12, cursor: 'pointer',
                           minWidth: 74 }}>
            {spin ? '⏸ pause' : '▶ rotate'}
          </button>
          <button onClick={() => setSplit(!split)}
                  title="show every axis-pair anchor at once, synchronised in angle, rotation, and selection"
                  style={{ background: split ? '#20304d' : '#0a0a1a',
                           color: split ? '#fff' : '#8a9',
                           border: split ? `1px solid ${ACCENT}` : '1px solid #444',
                           borderRadius: 3, padding: '3px 10px', fontSize: 12,
                           cursor: 'pointer' }}>
            ▦ split
          </button>
          {!splitOn && (
            <select value={basePair}
                    onChange={(e) => setBasePair(Number(e.target.value))}
                    title="which axes of the space the shadow plane is anchored to — different views expose different overlaps"
                    style={{ background: '#0a0a1a', color: '#8a9',
                             border: '1px solid #444', borderRadius: 3,
                             padding: '2px 4px', fontSize: 12 }}>
              {Array.from({ length: nViews }, (_, i) => (
                <option key={i} value={i}>view {i + 1}·{(i + 1) % nViews + 1}</option>
              ))}
            </select>
          )}
          <input type="range" min={0} max={359} step={1} value={deg}
                 onChange={(e) => {
                   setSpin(false);
                   setTheta((Number(e.target.value) * Math.PI) / 180);
                 }}
                 title="shadow angle — drag to pick a static projection"
                 style={{ flex: 1 }} />
          <span style={{ color: '#667', fontSize: 11, minWidth: 34,
                         fontVariantNumeric: 'tabular-nums' }}>
            {deg}°
          </span>
        </div>
      )}
    </div>
  );
}
