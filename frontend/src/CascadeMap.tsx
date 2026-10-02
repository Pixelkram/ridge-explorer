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
import { useEffect, useMemo, useRef, useState } from 'react';
import type { CascadeStatus, LocalSvMap } from './api/types';
import { useProbeStore, probeLabel, probeCodim } from './stores/probeStore';
import type { CascadeProbeMap, LocalSvView } from './stores/probeStore';

const ACCENT = '#4ecca3';
const WARN = '#e94560';
const CERT_NO = '#8a93b8';   // the grey of non-significant crossings, reused for stations that did not certify
// dark outline behind probe strokes and their labels, so a measured direction stays
// readable where it crosses the bright sampling cloud
const HALO = '#0a0a12';

const REDUCED = typeof window !== 'undefined'
  && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

// blue (quiet) -> violet -> red (hot) ramp for local divergence
function divColor(d: number | null): string {
  if (d === null) return '#565e93';
  const t = Math.min(1, Math.max(0, d / 0.5));
  const lerp = (a: number, b: number) => Math.round(a + (b - a) * t);
  return `rgb(${lerp(75, 233)},${lerp(90, 69)},${lerp(160, 96)})`;
}

/**
 * Boundary-density ramp: ONE hue, low -> high, anchored on the documented orange/yellow
 * family (OKLCH hue 75 deg, mid step beside the documented #eda100).
 *
 * The map already spends blue -> red on local divergence, so this second sequential
 * context takes the next hue rather than re-using a ramp that already means something
 * else. Checked, not eyeballed, against this map's own surface (#0d0d20): monotone
 * lightness, adjacent dL >= 0.06, single hue (spread 10 deg), darkest step 2.76:1. The
 * hue is the next one along on purpose -- plain orange's mid step sits dE 9.8 from WARN
 * under normal vision (below the 15 floor: confusable with the walk polylines), while
 * this one clears both floors against the map's other marks (dE 17.0 normal, 8.6 deutan).
 */
const SV_RAMP = ['#814e00', '#b06a00', '#d38500', '#eeac44', '#fed191'];
// no local evidence: the cloud's own "not measured" grey, which means the same thing there
const SV_NONE = '#565e93';

function mixHex(a: string, b: string, t: number): string {
  const p = (h: string, i: number) => parseInt(h.slice(1 + 2 * i, 3 + 2 * i), 16);
  const c = (i: number) => Math.round(p(a, i) + (p(b, i) - p(a, i)) * t);
  return `rgb(${c(0)},${c(1)},${c(2)})`;
}

/** Ramp colour for a normalised reading in [0, 1]; null (or uncovered) reads as grey. */
function svColor(t: number | null): string {
  if (t === null || !Number.isFinite(t)) return SV_NONE;
  const x = Math.min(1, Math.max(0, t)) * (SV_RAMP.length - 1);
  const i = Math.min(SV_RAMP.length - 2, Math.floor(x));
  return mixHex(SV_RAMP[i], SV_RAMP[i + 1], x - i);
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

/**
 * Everything a tile needs to colour by boundary density. `vmax` and `read` are built once
 * in CascadeMap so every split-view tile shares ONE scale -- per-tile scales would make
 * the same chord a different colour in two views of the same space.
 */
interface SvOverlay {
  map: LocalSvMap;
  view: LocalSvView;
  /** one reading -> [0, 1] for the ramp; null = uncovered, i.e. no local evidence */
  read: (value: number, rank: number, uncovered: boolean) => number | null;
  /** crossing id -> its index in the map's crossing arrays */
  byCid: Map<number, number>;
  /** top of the scale in S_V units (calibrated view only) */
  vmax: number;
}

/**
 * Endpoints of one probed direction, drawn as a sign-free segment through its crossing.
 *
 * ε is not a fixed number of barycentric units. The map's scale changes with k, with the
 * tile size in split view, and -- for k>3 -- with the rotation angle, because a normal
 * lying near the shadow plane projects long while one near its complement projects short.
 * So ε is solved for from the drawing rather than guessed: project the ε=1 offset once,
 * measure how many pixels it covers, then set ε = 0.04·size / thatLength so the FULL
 * segment (both halves) is ~8% of the map. Everything is recomputed from `project`, which
 * the caller rebuilds on every render, so the segment follows the rotation instead of
 * freezing at the angle it was measured at.
 *
 * A normal almost perpendicular to the current shadow plane covers nearly no pixels;
 * scaling it up to 8% anyway would invent a direction the viewer cannot actually see in
 * this projection, so it is reported as flat and drawn as a bare dot until the plane
 * rotates far enough to show it.
 */
function probeSegment(
  project: (w: number[]) => readonly [number, number],
  w: number[], n: number[], size: number,
): { cx: number; cy: number; ends: readonly [number, number, number, number] | null } | null {
  const [cx, cy] = project(w);
  if (!Number.isFinite(cx) || !Number.isFinite(cy)) return null;
  if (!n || n.length < w.length) return null;
  const [ux, uy] = project(w.map((wi, i) => wi + n[i]));
  const unitPx = Math.hypot(ux - cx, uy - cy);
  // relative cut: an in-plane unit normal covers roughly 0.4·size px, so 1.5% of the
  // map is a direction within ~2° of perpendicular to the shadow plane
  if (!Number.isFinite(unitPx) || unitPx < 0.015 * size) return { cx, cy, ends: null };
  const eps = (0.04 * size) / unitPx;
  const [x1, y1] = project(w.map((wi, i) => wi + eps * n[i]));
  const [x2, y2] = project(w.map((wi, i) => wi - eps * n[i]));
  if (![x1, y1, x2, y2].every(Number.isFinite)) return { cx, cy, ends: null };
  return { cx, cy, ends: [x1, y1, x2, y2] as const };
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
  status, size, theta, base, sel, onPick, walkPath, walkSegs, walkColors,
  caption, showBadge, showBeat, spin, compact, probes, pendingCid, sv,
}: {
  status: CascadeStatus;
  size: number;
  theta: number;
  base: number;
  sel: number | null;
  onPick: (cid: number | null) => void;
  walkPath?: number[][] | null;
  walkSegs?: number[][][] | null;
  // per walkPath point once the walk is certified (null = keep the walk colour)
  walkColors?: (string | null)[] | null;
  caption?: string;
  showBadge?: boolean;
  showBeat?: boolean;
  spin?: boolean;
  compact?: boolean;
  /** finished JVP probes of this run, by crossing id */
  probes?: CascadeProbeMap;
  /** crossing whose probe is in flight, if any */
  pendingCid?: number | null;
  /** boundary-density overlay, or null when it is off */
  sv?: SvOverlay | null;
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
  // bound the overlay's segment count the way maxPts bounds the cloud's dots: a 200-chord
  // survey samples ~24k stations, which is far more SVG nodes than the eye gains from
  const svStride = (() => {
    if (!sv) return 1;
    const n = sv.map.chords.reduce((a, l) => a + l.points.length, 0);
    const cap = compact ? 700 : 1600;
    return n > cap ? Math.ceil(n / cap) : 1;
  })();

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
      {/* Search chords. With the density overlay on, each one is drawn as its own station
          polyline instead of a single line, every segment coloured by the local reading —
          re-projected through `project` on every render, like the probe segments, so the
          colours follow the rotating shadow plane rather than freezing. No heat raster is
          painted into the projection: a 2-D shadow of a (k-1)-dimensional field would
          invent structure that is pure projection artefact, so only the 1-D objects the
          survey actually measured along get coloured. Striding keeps a long survey's
          segment count bounded, and segments stay contiguous because each one ends where
          the next begins. Branching runs fade the child generations back, so the fair root
          survey stays the figure and the exploratory rays the ground; no connector is drawn
          to the parent crossing, because a ray STARTS at it (up to the bisection shift of
          half a bracket) and the line would be sub-pixel. */}
      {status.chords.map((c, i) => {
        const gen = status.chords_meta?.[i]?.gen ?? 0;
        const op = gen === 0 ? 1 : gen === 1 ? 0.7 : 0.5;
        const lane = sv?.map.chords[i];
        if (sv && lane && lane.points.length > 1) {
          const out = [];
          for (let j = 0; j + 1 < lane.points.length; j += svStride) {
            const j2 = Math.min(lane.points.length - 1, j + svStride);
            const [ax, ay] = proj.project(lane.points[j]);
            const [bx, by] = proj.project(lane.points[j2]);
            out.push(
              <line key={`${i}.${j}`} x1={ax} y1={ay} x2={bx} y2={by}
                    stroke={svColor(sv.read(lane.values[j], lane.ranks[j],
                                            lane.uncovered[j]))}
                    strokeOpacity={op}
                    strokeWidth={compact ? 2 : 2.8} strokeLinecap="butt" />);
          }
          return <g key={i}>{out}</g>;
        }
        const [x1, y1] = proj.project(c.a);
        const [x2, y2] = proj.project(c.b);
        return (
          <line key={i} className="cs-draw" x1={x1} y1={y1} x2={x2} y2={y2}
                stroke="#4a4f8f" strokeOpacity={op}
                strokeWidth={compact ? (gen ? 1.1 : 1.4) : (gen ? 1.5 : 2)} pathLength={1}
                style={{ animationDelay: `${(i % 8) * 0.12}s` }} />
        );
      })}
      {(status.traces ?? []).map((tr, i) => (
        <g key={`tr${i}`} style={{ pointerEvents: 'none' }}>
          <polyline points={tr.points.map((w) => proj.project(w).join(',')).join(' ')}
                    fill="none" stroke={ACCENT} strokeWidth={compact ? 1.3 : 1.9}
                    strokeOpacity={0.7} strokeLinejoin="round" />
          {tr.points.slice(1).map((w, j) => {
            const [x, y] = proj.project(w);
            return <circle key={j} cx={x} cy={y} r={compact ? 1.5 : 2.1}
                           fill={tr.cert && tr.cert[j] === false ? CERT_NO : ACCENT} />;
          })}
        </g>
      ))}
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
        // density overlay: the dot joins the chords on the SAME scale, and certification
        // moves to the ring so the overlay does not spend the map's only certified/not
        // channel on a second meaning
        const svi = sv?.byCid.get(c.cid);
        const svFill = sv && svi !== undefined
          ? svColor(sv.read(sv.map.crossing_values[svi], sv.map.crossing_ranks[svi],
                            sv.map.crossing_uncovered[svi]))
          : null;
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
                    fill={svFill ?? (!scored ? '#7d84c8'
                                     : c.significant ? ACCENT : '#8a93b8')}
                    fillOpacity={scored ? 0.95 : 0.7}
                    stroke={sel === c.cid ? '#fff'
                            : (selGroup !== null && c.ridge_group === selGroup)
                              ? WARN
                              : svFill && c.significant ? ACCENT : '#0d0d20'}
                    strokeWidth={sel === c.cid ? 2.2 : 1.2}
                    className={!scored && running ? 'cs-pulse' : undefined}
                    style={{ cursor: 'pointer' }}
                    onClick={() => onPick(sel === c.cid ? null : c.cid)} />
            {/* refined by the final high-resolution pass: a hairline ring, not a colour, so
                the dot keeps saying what it said (density, certification, selection) and the
                refinement reads as a property of the MEASUREMENT on top of it. Split children
                are ordinary crossings and draw as such -- they carry this ring too. */}
            {c.hires && (
              <circle cx={x} cy={y} r={rad + 2.6} fill="none" stroke="#fff"
                      strokeWidth={compact ? 0.7 : 0.9} strokeOpacity={0.75}
                      style={{ pointerEvents: 'none' }} />
            )}
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
                           fill={walkColors?.[i] ?? WARN} />;
          })}
        </g>
      )}
      {/* Measured crossing directions. Sign-free, so a segment and never an arrow.
          Re-projected from the stored barycentric normal on EVERY render — caching
          screen coordinates would leave the lines behind as the shadow plane turns.
          pointer-events:none keeps the crossing dots underneath clickable. */}
      {(probes || pendingCid != null) && (
        <g style={{ pointerEvents: 'none' }}>
          {probes && status.crossings.map((c) => {
            const r = probes[c.cid];
            if (!r) return null;
            const seg = probeSegment(proj.project, c.weights, r.normal_bary, size);
            if (!seg) return null;
            const fs = compact ? 9 : 11;
            return (
              <g key={`probe-${c.cid}`}>
                {seg.ends && (
                  <>
                    {/* halo first, then the accent stroke on top of it */}
                    <line x1={seg.ends[0]} y1={seg.ends[1]}
                          x2={seg.ends[2]} y2={seg.ends[3]}
                          stroke={HALO} strokeWidth={4.5} strokeLinecap="round"
                          opacity={0.85} />
                    <line x1={seg.ends[0]} y1={seg.ends[1]}
                          x2={seg.ends[2]} y2={seg.ends[3]}
                          stroke={ACCENT} strokeWidth={2.5} strokeLinecap="round" />
                  </>
                )}
                <circle cx={seg.cx} cy={seg.cy} r={compact ? 2.6 : 3.5}
                        fill={ACCENT} stroke={HALO} strokeWidth={1} />
                <text x={seg.cx + 7} y={seg.cy - 7} fontSize={fs} fill={ACCENT}
                      stroke={HALO} strokeWidth={3} paintOrder="stroke"
                      style={{ fontFamily: 'system-ui' }}>
                  {seg.ends ? probeLabel(r) : `${probeLabel(r)} · edge-on`}
                </text>
              </g>
            );
          })}
          {pendingCid != null && (() => {
            const c = status.crossings.find((x) => x.cid === pendingCid);
            if (!c) return null;
            const [x, y] = proj.project(c.weights);
            if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
            return (
              <g>
                <circle className="cs-march" cx={x} cy={y} r={compact ? 7 : 9}
                        fill="none" stroke={ACCENT} strokeWidth={1.5}
                        strokeDasharray="3,3" opacity={0.9} />
                <text x={x + 13} y={y + 4} fontSize={compact ? 9 : 10} fill={ACCENT}
                      stroke={HALO} strokeWidth={3} paintOrder="stroke"
                      style={{ fontFamily: 'system-ui' }}>
                  probing…
                </text>
              </g>
            );
          })()}
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

/**
 * The crossing-direction probe for one selected crossing. The measurement is local and
 * takes ~25 s, so it is always an explicit click -- never started by selecting a dot.
 */
function CascadeProbeBlock({ runId, cid }: { runId: string; cid: number }) {
  const probes = useProbeStore((s) => s.cascadeProbes[runId]);
  const pendingCid = useProbeStore((s) => s.cascadePendingCid);
  const phase = useProbeStore((s) => s.cascadeStatus);
  const error = useProbeStore((s) => s.cascadeError);
  const start = useProbeStore((s) => s.startCascadeJvp);

  const here = probes?.[cid];
  const busy = phase === 'running';
  const runningHere = busy && pendingCid === cid;

  return (
    <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid #23234d' }}>
      <button className="rx-focus"
              disabled={busy}
              onClick={() => start(runId, cid)}
              title="measure the direction across prompt space along which the image changes fastest here"
              style={{ padding: '3px 10px', borderRadius: 3, fontSize: 11,
                       background: busy ? '#333' : '#0f3460',
                       color: busy ? '#777' : '#cfd',
                       border: `1px solid ${busy ? '#333' : ACCENT}`,
                       cursor: busy ? 'not-allowed' : 'pointer' }}>
        {runningHere ? 'probing…'
          : here ? 'Probe again (~25 s)' : 'Probe crossing direction (~25 s)'}
      </button>
      {busy && !runningHere && (
        <div style={{ fontSize: 10, color: '#888', marginTop: 4 }}>
          another probe is running…
        </div>
      )}
      {phase === 'error' && error && !busy && (
        <div style={{ fontSize: 10, color: '#ff8a9c', marginTop: 4 }}>{error}</div>
      )}
      {here && (
        <div style={{ marginTop: 6, fontSize: 11, lineHeight: 1.5 }}>
          <div style={{ color: ACCENT }}>{here.normal_reading}</div>
          <div style={{ color: '#9ab', marginTop: 3 }}>
            rank-1 share {here.rank1_share.toFixed(2)} · participation ratio{' '}
            {here.participation_ratio.toFixed(2)}
          </div>
          <div style={{ color: '#9ab' }}>
            {probeCodim(here)}
            <span style={{ color: '#667' }}>
              {probeCodim(here) === 'corner'
                ? ' — two fronts meet here, so one direction is a poor summary'
                : ' — a single front'}
            </span>
          </div>
          <div style={{ fontSize: 10, color: '#777', marginTop: 4 }}>
            Direction is seed-invariant; magnitude comparable only within this run.
          </div>
        </div>
      )}
    </div>
  );
}

export default function CascadeMap({
  status, runId, imageUrl, size = 340, onSelect, onImage, walkPath, walkSegs,
  walkColors, selectedCid,
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
  walkColors?: (string | null)[] | null;
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

  // Probes are filed per run: a crossing id belongs to the survey that issued it, so a
  // new run drops the map rather than drawing last run's directions on this one's dots.
  const syncCascadeRun = useProbeStore((s) => s.syncCascadeRun);
  useEffect(() => { syncCascadeRun(runId); }, [runId, syncCascadeRun]);
  const probes = useProbeStore((s) => s.cascadeProbes[runId]);
  const probePending = useProbeStore(
    (s) => (s.cascadeStatus === 'running' ? s.cascadePendingCid : null));

  // Boundary density: a re-reading of this run's own chords (no new measurement), filed per
  // run beside the probes and dropped with them. The panel owns the toggle and the view.
  const svOn = useProbeStore((s) => s.localSvOn);
  const svViewSel = useProbeStore((s) => s.localSvView);
  const svMap = useProbeStore((s) => s.localSv[runId]);
  const sv = useMemo<SvOverlay | null>(() => {
    if (!svOn || !svMap || svMap.chords.length === 0) return null;
    // an uncalibrated survey may only be READ as a ranking, whatever the selector holds
    const view: LocalSvView = svViewSel === 'calibrated' && svMap.calibrated_ok
      ? 'calibrated' : 'ranking';
    let vmax = 0;
    for (const l of svMap.chords) {
      for (let i = 0; i < l.values.length; i++) {
        if (!l.uncovered[i] && l.values[i] > vmax) vmax = l.values[i];
      }
    }
    svMap.crossing_values.forEach((v, i) => {
      if (!svMap.crossing_uncovered[i] && v > vmax) vmax = v;
    });
    const byCid = new Map(svMap.crossing_cids.map((cid, i) => [cid, i] as const));
    const read = (value: number, rank: number, uncovered: boolean) =>
      uncovered ? null
        : view === 'calibrated' ? (vmax > 0 ? value / vmax : 0) : rank / 100;
    return { map: svMap, view, read, byCid, vmax };
  }, [svOn, svMap, svViewSel]);

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
                    walkColors={walkColors}
                    caption={`view ${b + 1}·${(b % nViews) + 2 > nViews ? 1 : b + 2}`}
                    showBeat={b === 0} compact
                    probes={probes} pendingCid={probePending} sv={sv} />
          ))}
        </div>
      ) : (
        <MapSvg status={status} size={mapSize} theta={theta} base={basePair}
                sel={sel} onPick={pick} walkPath={walkPath} walkSegs={walkSegs}
                walkColors={walkColors}
                showBadge showBeat spin={spin}
                probes={probes} pendingCid={probePending} sv={sv} />
      )}
      <div style={{ fontSize: 11, color: '#889', maxWidth: 190 }}>
        {sv && (
          <div style={{ marginBottom: 8, paddingBottom: 8,
                        borderBottom: '1px solid #23234d' }}>
            <div style={{ color: '#aeb6dd' }}>
              boundary density{' '}
              <span style={{ color: '#667' }}>
                · {sv.view === 'calibrated' ? 'S_V units' : 'percentile'}
              </span>
            </div>
            {/* the ramp itself, with a hairline ring so the lightest step stays bounded
                rather than bleeding into whatever surface the panel is drawn on */}
            <div style={{ height: 8, borderRadius: 2, marginTop: 4,
                          border: '1px solid rgba(255,255,255,0.10)',
                          background: `linear-gradient(90deg, ${SV_RAMP.join(', ')})` }} />
            <div style={{ display: 'flex', justifyContent: 'space-between',
                          fontVariantNumeric: 'tabular-nums', color: '#889' }}>
              <span>{sv.view === 'calibrated' ? '0' : '0%'}</span>
              <span>{sv.view === 'calibrated'
                ? sv.vmax.toFixed(sv.vmax >= 10 ? 0 : 1) : '100%'}</span>
            </div>
            <div style={{ color: '#667' }}>
              <span style={{ color: SV_NONE }}>―</span> no local evidence (kernel mass
              under 5% of median)
            </div>
            <div style={{ color: '#667', marginTop: 2 }}>
              {sv.map.n_chords} chords · h={sv.map.h.toFixed(2)}
              {sv.map.s_global !== null
                && ` · global S_V ${sv.map.s_global.toFixed(1)}`}
            </div>
            {!sv.map.calibrated_ok && (
              <div style={{ color: '#c9a227', marginTop: 2 }}>
                ranking only — values need ≥ 80 chords
              </div>
            )}
          </div>
        )}
        <div style={{ marginBottom: 6 }}>
          <span style={{ color: '#7d84c8' }}>·</span> generated images — colour = local
          divergence (<span style={{ color: '#5a64b0' }}>quiet</span> → <span
          style={{ color: WARN }}>hot</span>), pings mark fresh ones
          <br /><span style={{ color: sv ? SV_RAMP[2] : '#4a4f8f' }}>―</span> search chords
          {sv ? ' — coloured by the density above; crossing dots share that scale and'
                + ' keep certification in their ring'
              : ' (drawn as laid)'}
          {(status.chords_meta ?? []).some((m) => m.gen > 0) && (
            <>
              <br /><span style={{ color: '#4a4f8f', opacity: 0.6 }}>―</span> faded = child
              rays, spawned from a parent chord's strongest crossing — exploratory, so they
              are left out of the coverage certificate
            </>
          )}
          <br /><span style={{ color: '#7d84c8' }}>●</span> boundary found (pulsing =
          still being worked on)
          <br /><span style={{ color: ACCENT }}>●</span> certified — beats the run's own
          background; size = strength
          {status.crossings.some((c) => c.hires) && (
            <>
              <br /><span style={{ color: '#fff' }}>◌</span> thin white ring = sharpened by the
              final high-res pass (position quoted at a {status.hires_factor ?? 4}× finer
              spacing); where one bracket hid two boundaries, the extra dots are the split
            </>
          )}
          <br /><span style={{ color: ACCENT }}>◇</span> refined patch
          {' '}<span style={{ color: '#8a93b8' }}>(dashed = exploration slot)</span>
          <br /><span style={{ color: WARN }}>◦</span> same ridge as selection ·{' '}
          <span style={{ color: WARN }}>‖</span> walk probe rungs · after certifying, stations turn{' '}
          <span style={{ color: ACCENT }}>●</span> held on unseen seeds /{' '}
          <span style={{ color: CERT_NO }}>●</span> did not · <span style={{ color: ACCENT }}>—</span> traced ridge
          {probes && Object.keys(probes).length > 0 && (
            <>
              <br /><span style={{ color: ACCENT }}>╱</span> probed crossing direction —
              sign-free, so a line and not an arrow; it turns with the shadow plane and
              shortens when it points out of it
            </>
          )}
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
        {selected && <CascadeProbeBlock runId={runId} cid={selected.cid} />}
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
