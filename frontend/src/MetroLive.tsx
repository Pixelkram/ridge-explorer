/**
 * MetroLive -- the small read-outs the Metropolis panel watches a run through.
 *
 * Three pieces, none of which knows anything about running a chain: a `Sparkline`, the
 * `ChainTable` that lists the chains and lets one be followed or frozen, and `ChainTrace`,
 * the followed chain's own trajectory and its last thumbnails.
 *
 * Why these charts look the way they do
 * ------------------------------------
 * Both summary sparklines are change-over-time with ONE series each, so there is no legend
 * box (the label beside the chart names what is plotted) and no second y-axis anywhere --
 * acceptance rate and mean energy are two different measures, so they are two charts, never
 * one with two scales.
 *
 * Colour is three tokens the tool already owns, in the roles the sparkline recipe asks for:
 * a recessive series ink (#7d84b5, the panel's own in-progress bar), the accent on the
 * CURRENT point (#4ecca3), and the panel's caution gold (#c9a227) for the kernel-change rule
 * -- an annotation, not a series, and a caveat, which is exactly what that token means
 * everywhere else here. Checked rather than eyeballed, against this surface (#0a0a1a): the
 * two marks that share a chart are dE 21.7 apart under normal vision and 17.0 under deutan,
 * the rule clears both against the series (23.5 / 16.5), and all three clear 3:1 on contrast.
 * The validator's two categorical-scope complaints are deliberate and do not apply: these are
 * not a categorical palette (one series per chart), the series ink is MEANT to read near-grey
 * (the recipe's "de-emphasis hue"), and the other two are the tool's existing accent and
 * caution tokens, which may not be re-stepped here without meaning something different in
 * this panel than in every other.
 *
 * One thing was changed BECAUSE of that check. The mean-S sparkline's current point wanted to
 * be `divColor(S)`, the ramp that already means energy on the maps -- but the ramp's cool end
 * sits dE 14.3 from the series ink, under the 15 normal-vision floor, so at low S the dot
 * would vanish into its own line. The accent carries the current point on both charts
 * instead, and the ramp keeps its job: the mean-S chart's y-axis is fixed to the ramp's own
 * 0..0.50 top of scale, so a point's HEIGHT here and a dot's COLOUR on the map are the same
 * reading of the same number.
 */
import { useState } from 'react';
import type { MetroChainStat, MetroRoundLog, MetroSample } from './api/types';
import { divColor } from './CascadeMap';

/** recessive series ink: the panel's in-progress bar, and the sparkline recipe's de-emphasis */
const SERIES = '#7d84b5';
/** the tool's accent, on the CURRENT point only */
const CURRENT = '#4ecca3';
/** the panel's caution gold, on the kernel-change rule -- an annotation, and a caveat */
const MARK = '#c9a227';
/** the tool's track/input ground: the sparkline's own surface, and its marker rings */
const SURFACE = '#0a0a1a';
const INK = '#c9d1f0';
const LABEL = '#aeb6dd';
const MUTED = '#667';
const WARN = '#e94560';

export interface SparkPoint {
  x: number;
  /** null breaks the line rather than being drawn as a zero */
  y: number | null;
}

export interface SparkMark {
  x: number;
  title: string;
}

function fmt(x: number | null | undefined, d = 3): string {
  return x === null || x === undefined || !Number.isFinite(x) ? '—' : x.toFixed(d);
}

/**
 * One tiny single-series line chart: label, current value, trend.
 *
 * `domain` fixes the y range when the quantity has one that means something (acceptance is a
 * share, so [0, 1]; energy shares the divergence ramp's [0, 0.50]) -- a sparkline that
 * rescaled itself every round would make a flat run look dramatic. `marks` are vertical
 * annotation rules. The hover layer is per-point hit rects a good deal wider than the marks,
 * which drive a crosshair and a one-line read-out under the chart, so no value needs a label
 * on it.
 */
export function Sparkline({
  label, title, points, domain, marks, unit = '', digits = 2, width = 150, height = 38,
  hint,
}: {
  label: string;
  title?: string;
  points: SparkPoint[];
  /** [lo, hi]; the hi is raised to fit the data, never lowered */
  domain: [number, number];
  marks?: SparkMark[];
  unit?: string;
  digits?: number;
  width?: number;
  height?: number;
  /** one line under the chart when nothing is hovered */
  hint?: string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const pad = 5;
  const vals = points.map((p) => p.y).filter((y): y is number => y !== null);
  const lo = domain[0];
  const hi = Math.max(domain[1], ...(vals.length ? [Math.max(...vals)] : []));
  const xs = points.map((p) => p.x);
  const x0 = xs.length ? Math.min(...xs) : 0;
  const x1 = xs.length ? Math.max(...xs) : 1;
  const px = (x: number) => pad + ((x - x0) / Math.max(1e-9, x1 - x0)) * (width - 2 * pad);
  const py = (y: number) =>
    height - pad - ((y - lo) / Math.max(1e-9, hi - lo)) * (height - 2 * pad);
  // runs of consecutive non-null points: a gap is a gap, not a line through it
  const runs: SparkPoint[][] = [];
  for (const p of points) {
    if (p.y === null) { runs.push([]); continue; }
    if (!runs.length) runs.push([]);
    runs[runs.length - 1].push(p);
  }
  const last = [...points].reverse().find((p) => p.y !== null) ?? null;
  const at = hover !== null && hover < points.length ? points[hover] : null;
  const step = Math.max(1, (width - 2 * pad) / Math.max(1, points.length));

  return (
    <div style={{ minWidth: width }} title={title}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8,
                    alignItems: 'baseline' }}>
        <span style={{ color: LABEL, fontSize: 11 }}>{label}</span>
        <span style={{ color: INK, fontSize: 12, fontVariantNumeric: 'tabular-nums' }}>
          {fmt(last?.y, digits)}{unit}
        </span>
      </div>
      <svg width={width} height={height} role="img"
           aria-label={`${label}: ${fmt(last?.y, digits)}${unit} over `
                       + `${points.length} rounds`}
           style={{ background: SURFACE, borderRadius: 3, display: 'block',
                    border: '1px solid #2a2a4a' }}>
        {(marks ?? []).map((m) => (
          <line key={`m${m.x}`} x1={px(m.x)} y1={1} x2={px(m.x)} y2={height - 1}
                stroke={MARK} strokeWidth={1} strokeOpacity={0.85}>
            <title>{m.title}</title>
          </line>
        ))}
        {runs.filter((r) => r.length > 1).map((r, i) => (
          <polyline key={`r${i}`} fill="none" stroke={SERIES} strokeWidth={2}
                    strokeLinejoin="round" strokeLinecap="round"
                    points={r.map((p) => `${px(p.x)},${py(p.y as number)}`).join(' ')} />
        ))}
        {runs.filter((r) => r.length === 1).map((r, i) => (
          <circle key={`s${i}`} cx={px(r[0].x)} cy={py(r[0].y as number)} r={2}
                  fill={SERIES} />
        ))}
        {at && at.y !== null && (
          <line x1={px(at.x)} y1={1} x2={px(at.x)} y2={height - 1} stroke={LABEL}
                strokeWidth={1} strokeOpacity={0.5} />
        )}
        {last && last.y !== null && (
          // the current point: >= 8px with a surface ring, so it stays legible where it sits
          // on its own line
          <circle cx={px(last.x)} cy={py(last.y)} r={4} fill={CURRENT} stroke={SURFACE}
                  strokeWidth={2} />
        )}
        {at && at.y !== null && (
          <circle cx={px(at.x)} cy={py(at.y)} r={3} fill={INK} stroke={SURFACE}
                  strokeWidth={2} />
        )}
        {/* hit targets, wider than the marks (interaction.md), drawn last so they win */}
        {points.map((p, i) => (
          <rect key={`h${i}`} x={px(p.x) - step / 2} y={0} width={step} height={height}
                fill="transparent" onPointerEnter={() => setHover(i)}
                onPointerLeave={() => setHover((h) => (h === i ? null : h))} />
        ))}
      </svg>
      <div style={{ color: MUTED, fontSize: 10, height: 13, overflow: 'hidden',
                    fontVariantNumeric: 'tabular-nums' }}>
        {at ? `round ${at.x} · ${fmt(at.y, digits)}${unit}` : (hint ?? '')}
      </div>
    </div>
  );
}

/** The rounds at which the kernel differs from the round before -- the backend's own rule. */
export function paramChangeRounds(rounds: MetroRoundLog[]): SparkMark[] {
  const out: SparkMark[] = [];
  for (let i = 1; i < rounds.length; i++) {
    const a = rounds[i - 1], b = rounds[i];
    if (a.sigma === b.sigma && a.beta === b.beta) continue;
    const bits: string[] = [];
    if (a.sigma !== b.sigma) bits.push(`σ ${a.sigma} → ${b.sigma}`);
    if (a.beta !== b.beta) bits.push(`β ${a.beta} → ${b.beta}`);
    out.push({ x: b.round,
               title: `round ${b.round}: ${bits.join(', ')} — the kernel changed here, so `
                      + 'the states before and after are draws from two different chains' });
  }
  return out;
}

/**
 * The chains, as a list that can be followed and frozen.
 *
 * Sorted by current energy, which is what the panel is for -- "which chain is sitting on the
 * sharpest thing right now" is the question the gallery beside it answers for states. Chain
 * identity is NOT carried by colour: 60 chains cannot have 60 distinguishable hues, and
 * inventing a cycled palette would make two unrelated chains read as the same one. It is
 * carried by this list, by the number in it, and by following one (which dims the rest).
 */
export function ChainTable({
  chains, followed, onFollow, onStop, canStop, maxHeight = 184, maxWidth = 380,
}: {
  chains: MetroChainStat[];
  followed: number | null;
  onFollow: (chain: number | null) => void;
  onStop?: (chain: number) => void;
  canStop?: boolean;
  maxHeight?: number;
  /** five narrow columns: let them sit together rather than spread across the panel */
  maxWidth?: number;
}) {
  const rows = [...chains].sort((a, b) => (b.s ?? -1) - (a.s ?? -1));
  const cell: React.CSSProperties = { padding: '1px 5px', textAlign: 'right' };
  return (
    <div style={{ maxHeight, maxWidth, overflowY: 'auto', border: '1px solid #2a2a4a',
                  borderRadius: 3, background: SURFACE }}>
      <table style={{ borderCollapse: 'collapse', fontSize: 11, width: '100%',
                      fontVariantNumeric: 'tabular-nums' }}>
        <thead>
          <tr style={{ color: MUTED, position: 'sticky', top: 0, background: SURFACE }}>
            <th style={{ ...cell, textAlign: 'left' }}>chain</th>
            <th style={cell} title="MCMC steps this chain has proposed">steps</th>
            <th style={cell} title="the energy of the state it is sitting on now">S</th>
            <th style={cell} title="share of its proposals it accepted">acc</th>
            <th style={cell} />
          </tr>
        </thead>
        <tbody>
          {rows.map((c) => {
            const on = followed === c.chain;
            return (
              <tr key={c.chain}
                  onClick={() => onFollow(on ? null : c.chain)}
                  title={c.stopped ? 'frozen: it proposes no more, and every state it '
                                     + 'accepted is kept'
                    : !c.moved ? 'has never accepted a move — it still sits on the seed '
                                 + 'sheet it started on'
                      : `${c.n_accept} of ${c.n_propose} proposals accepted`
                        + (c.n_outside ? `, ${c.n_outside} left the simplex` : '')}
                  style={{ cursor: 'pointer', color: on ? '#fff' : INK,
                           background: on ? '#23234d' : 'transparent',
                           opacity: c.stopped ? 0.55 : 1 }}>
                <td style={{ ...cell, textAlign: 'left' }}>
                  <span style={{ color: on ? CURRENT : MUTED }}>{on ? '▸' : '·'}</span>{' '}
                  {c.chain}
                  {c.joined_round > 0 && (
                    <span style={{ color: MUTED }} title={`added by hand at round `
                                                          + `${c.joined_round}`}> +</span>
                  )}
                </td>
                <td style={cell}>{c.n_propose}</td>
                <td style={cell}>
                  <span style={{ color: divColor(c.s ?? null) }}>●</span>{' '}
                  {fmt(c.s, 3)}
                </td>
                <td style={cell}>{fmt(c.acc_rate, 2)}</td>
                <td style={{ ...cell, width: 46 }}>
                  {c.stopped ? <span style={{ color: MUTED }}>frozen</span>
                    : canStop && onStop ? (
                      <button onClick={(e) => { e.stopPropagation(); onStop(c.chain); }}
                              title="freeze this chain: no more proposals, its samples kept"
                              style={{ background: 'transparent', color: MUTED,
                                       border: '1px solid #333', borderRadius: 2,
                                       fontSize: 9, padding: '0 3px', cursor: 'pointer' }}>
                        stop
                      </button>
                    ) : null}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/**
 * One chain, followed: the trajectory of the state it is sitting on, and its last thumbnails.
 *
 * The trajectory is the chain's STATE energy per round, carried forward across the rounds it
 * rejected -- not the list of accepted energies, which would hide how long it sat still. That
 * is the same x-axis (round) as the summary sparklines above it, so the two read together.
 */
export function ChainTrace({
  chain, samples, roundsDone, runId, imageUrl, onImage, marks,
}: {
  chain: MetroChainStat;
  samples: MetroSample[];
  roundsDone: number;
  runId: string;
  imageUrl: (runId: string, index: number) => string;
  onImage?: (sample: MetroSample, index: number) => void;
  marks?: SparkMark[];
}) {
  const mine = samples
    .map((s, i) => ({ s, i }))
    .filter((x) => x.s.chain === chain.chain);
  const byRound = new Map<number, number>();
  for (const { s } of mine) byRound.set(s.step, s.s);
  const pts: SparkPoint[] = [];
  let cur = chain.seed_s ?? null;
  // round 0 is the seed sheet it was handed, which is the comparison the summary quotes
  pts.push({ x: 0, y: cur });
  const until = chain.stopped ? Math.max(0, ...[...byRound.keys()]) : roundsDone;
  for (let r = 1; r <= until; r++) {
    if (byRound.has(r)) cur = byRound.get(r)!;
    pts.push({ x: r, y: cur });
  }
  const strip = mine.slice(-8).reverse();
  return (
    <div style={{ marginTop: 6, padding: '6px 8px', background: '#12122c', maxWidth: 560,
                  border: `1px solid ${CURRENT}`, borderRadius: 4 }}>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap',
                    alignItems: 'flex-start' }}>
        <Sparkline label={`chain ${chain.chain} · S of its state`}
                   title="the energy of the state this chain is sitting on, round by round.
Flat stretches are rounds it rejected — a chain that sat still for ten rounds looks like it
sat still, which a plot of only its accepted energies would hide."
                   points={pts} domain={[0, 0.5]} digits={3} marks={marks}
                   width={168}
                   hint={`round 0 = its seed sheet · ${chain.n_accept}/${chain.n_propose} `
                         + 'accepted'} />
        <div style={{ fontSize: 11, color: LABEL, minWidth: 128 }}>
          <div>
            now <span style={{ color: divColor(chain.s ?? null) }}>●</span>{' '}
            <span style={{ color: INK, fontVariantNumeric: 'tabular-nums' }}>
              {fmt(chain.s, 3)}
            </span>
            <span style={{ color: MUTED }}> · seed {fmt(chain.seed_s, 3)}</span>
          </div>
          <div style={{ color: MUTED }}>
            {chain.n_accept}/{chain.n_propose} accepted
            {chain.n_outside > 0 && ` · ${chain.n_outside} outside`}
          </div>
          {chain.stopped && <div style={{ color: WARN }}>frozen — proposes no more</div>}
          {chain.joined_round > 0 && (
            <div style={{ color: MARK }}>
              added by hand at round {chain.joined_round}
            </div>
          )}
          {!chain.moved && !chain.stopped && (
            <div style={{ color: MARK }}>still on its seed sheet</div>
          )}
        </div>
      </div>
      {strip.length > 0 && (
        <div style={{ display: 'flex', gap: 3, marginTop: 5, flexWrap: 'wrap' }}>
          {strip.map(({ s, i }) => (
            <img key={i} src={imageUrl(runId, s.full_image ?? s.image)} width={54}
                 height={54} onClick={() => onImage?.(s, i)}
                 title={`round ${s.step} · S ${s.s.toFixed(3)}`}
                 style={{ objectFit: 'cover', borderRadius: 3, cursor: 'pointer',
                          border: `2px solid ${divColor(s.s)}` }} />
          ))}
        </div>
      )}
      <div style={{ color: MUTED, fontSize: 10, marginTop: 3 }}>
        its last {Math.min(8, strip.length)} accepted states, newest first
      </div>
    </div>
  );
}
