/**
 * AmrPanel -- octree / adaptive-mesh refinement as an interactive run.
 *
 * The other way to spend a boundary budget on the same simplex. Where the Cascade lays
 * chords THROUGH the space and reads crossings along them, this tiles the space with a
 * barycentric lattice, evaluates all of the coarsest one, and then refines only the cells
 * next to a detected label change -- nested levels, so a coarse cell is never re-probed, and
 * one batched GPU round per level.
 *
 * Verdict of record (search_problem h25f): at k=4 it reaches the full-lattice ceiling at
 * 55 % of exhaustive cost and 7x cheaper than chords need for the same coverage -- and it
 * does NOT scale, because the fan-out is factor^(k-1). k=3 and k=4 only; the backend refuses
 * k>=5 with the projected cost, and this panel never offers it.
 */
import { useEffect, useRef, useState } from 'react';
import { amrStart, amrStatus, amrCancel, amrImageUrl, amrPointsUrl } from './api/client';
import type { AmrStatus } from './api/types';
import AmrMap from './AmrMap';

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
/** the backend's own ceiling on the finest lattice (backend/services/amr.py MAX_CELLS) */
const MAX_CELLS = 40000;

// Boundary-adjacent cell fraction measured at k=4 (h25f section 3). The same table and the
// same log interpolation the backend projects with, so the estimate below and the refusal a
// k>=5 request gets cannot disagree.
const PHI_K4: [number, number][] = [[5, 0.7679], [15, 0.5159], [30, 0.3783], [60, 0.2598]];

function phi(n: number): number {
  if (n <= PHI_K4[0][0]) return PHI_K4[0][1];
  const last = PHI_K4[PHI_K4.length - 1];
  if (n >= last[0]) return last[1];
  for (let i = 0; i + 1 < PHI_K4.length; i++) {
    const [a, pa] = PHI_K4[i], [b, pb] = PHI_K4[i + 1];
    if (n >= a && n <= b) {
      return pa + ((Math.log(n) - Math.log(a)) / (Math.log(b) - Math.log(a))) * (pb - pa);
    }
  }
  return last[1];
}

function comb(n: number, r: number): number {
  let out = 1;
  for (let i = 1; i <= r; i++) out = (out * (n - r + i)) / i;
  return Math.round(out);
}

const schedule = (base: number, levels: number, factor: number) =>
  Array.from({ length: Math.max(1, levels) }, (_, i) => base * factor ** i);

/** Projected probes per level: the coarse lattice in full, then phi x cells x fan-out,
 *  capped at each level's own cell count (refinement can never beat exhaustive). */
function projection(k: number, base: number, levels: number, factor: number) {
  const d = k - 1;
  const sched = schedule(base, levels, factor);
  let total = 0;
  const rows = sched.map((n, i) => {
    const cells = comb(n + d, d);
    const probes = i === 0
      ? cells
      : Math.min(cells, Math.round(phi(sched[i - 1]) * comb(sched[i - 1] + d, d)
                                   * factor ** d));
    total += probes;
    return { level: n, cells, probes };
  });
  return { sched, rows, total };
}

const BASE_TIP = 'points per simplex edge at the coarsest level, which is evaluated in '
  + 'FULL — every cell of it gets an image. 5 is the h25f ladder of record (56 cells at k=4)';
const LEVELS_TIP = 'how many lattices the ladder has, including the coarsest. Each one is '
  + 'only evaluated near a boundary edge the previous one found, so the levels get cheaper '
  + 'relative to their size as they get finer (h25f: 100 % → 99 % → 75 % → 55 % of the '
  + 'lattice at 5/15/30/60)';
const FACTOR_TIP = 'how much finer each level is than the last. The levels must NEST (a '
  + 'coarse cell is also a fine cell, so it is probed once): 2 is the setting of record, 3 '
  + 'jumps resolution faster for a factor^(k-1) larger refinement fan';
const RREF_TIP = 'how far around a detected boundary edge the next level is evaluated, in '
  + 'COARSE cells. 1 is of record: on the exact k=4 oracle it lost nothing against 2 (same '
  + '108 boundary pairs, same coverage) for 22 % fewer probes';
const PROBE_TIP = 'lattice probes run on the cheap field (4 of 8 denoising steps), as the '
  + 'Cascade\'s chord probes do — two probes for the price of one image. Uncheck to probe at '
  + 'full fidelity, which doubles the bill per cell';

/** One segment per level: past levels full, the running one filling by its probe count. */
function LevelBar({ status }: { status: AmrStatus }) {
  const sched = status.schedule ?? [];
  const done = status.status === 'complete' || status.phase === 'done';
  const cur = sched.findIndex((n) => status.phase === `level ${n}`);
  return (
    <div style={{ display: 'flex', gap: 4, margin: '8px 0 2px' }}>
      {sched.map((n, i) => {
        const active = !done && i === cur;
        const past = done || (cur >= 0 && i < cur) || i < status.levels_stats.length - 0.5;
        const frac = active && status.phase_total > 0
          ? Math.min(1, status.phase_done / status.phase_total) : past ? 1 : 0;
        return (
          <div key={n} style={{ flex: 1 }}>
            <div style={{ height: 5, background: '#0a0a1a', borderRadius: 3,
                          overflow: 'hidden', border: '1px solid #2a2a4a' }}>
              <div style={{ width: `${frac * 100}%`, height: '100%',
                            background: past && !active ? ACCENT : '#7d84b5',
                            transition: 'width 0.6s' }} />
            </div>
            <div style={{ fontSize: 10, marginTop: 2, textAlign: 'center',
                          color: active ? '#fff' : past ? ACCENT : '#667' }}>
              {n}{active && status.phase_total > 0
                && ` ${status.phase_done}/${status.phase_total}`}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function AmrPanel() {
  const [open, setOpen] = useState(false);
  const [k, setK] = useState(4);
  const [base, setBase] = useState(5);
  const [levels, setLevels] = useState(4);
  const [factor, setFactor] = useState<2 | 3>(2);
  const [rRef, setRRef] = useState<1 | 2>(1);
  const [seed, setSeed] = useState(42);
  const [steps, setSteps] = useState(8);
  const [cheapProbes, setCheapProbes] = useState(true);
  // AMR needs a pinned basis: unlike the Cascade it has no pool-draw path, because the
  // lattice is the whole measurement and there is nothing to draw it against. Prefilled so
  // Start works on the first click.
  const [promptText, setPromptText] = useState(
    'a photograph of a lighthouse\na pencil sketch of a cathedral\n'
    + 'an oil painting of a forest\na neon city at night');
  const [runId, setRunId] = useState<string | null>(null);
  const [status, setStatus] = useState<AmrStatus | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const timer = useRef<number | null>(null);
  const [detail, setDetail] = useState<{
    thumb: number; title: string; weights: number[] | null; rows: [string, string][];
  } | null>(null);

  // One poll chain only: re-entering the effect clears the previous timer, or a re-render
  // forks a second chain (the cascade panel's rule).
  useEffect(() => {
    if (!runId) return;
    let live = true;
    const tick = async () => {
      try {
        const s = await amrStatus(runId);
        if (!live) return;
        setStatus(s);
        if (s.status === 'running') timer.current = window.setTimeout(tick, 1500);
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

  const prompts = promptText.split('\n').map((s) => s.trim()).filter(Boolean);
  const proj = projection(k, base, levels, factor);
  const finestCells = proj.rows[proj.rows.length - 1].cells;
  const tooBig = finestCells > MAX_CELLS;
  const unit = cheapProbes ? 0.5 : 1.0;
  const running = status?.status === 'running';

  const start = async () => {
    setErr(null);
    setStatus(null);
    setDetail(null);
    if (prompts.length !== k) {
      setErr(`enter exactly ${k} prompts, one per line (got ${prompts.length})`);
      return;
    }
    try {
      const r = await amrStart({
        prompts, base, levels, factor, r_ref: rRef, seed, steps,
        probe_steps: cheapProbes ? 4 : null,
      });
      if (r.error) { setErr(r.error); return; }
      setRunId(r.run_id);
    } catch (e: any) {
      setErr(String(e.message || e));
    }
  };

  const ladder = proj.sched.join(' → ');
  const probesSoFar = status?.levels_stats.reduce((a, r) => a + r.n_evaluated, 0) ?? 0;

  return (
    <div style={BOX}>
      <div style={{ cursor: 'pointer', userSelect: 'none' }} onClick={() => setOpen(!open)}>
        {open ? '▾' : '▸'} Adaptive refinement — nested lattices, refined at the boundary
      </div>
      {open && (
        <div style={{ marginTop: 8 }}>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
            <label title="prompts spanning the simplex. 3 and 4 only: the refinement fans out as factor^(k-1), so k=5 already costs ~8x the budget of record and the backend refuses it with the projected number (use the Cascade's chords there)">
              k
              <select value={k} onChange={(e) => setK(Number(e.target.value))} style={SEL}>
                <option value={3}>3</option>
                <option value={4}>4</option>
              </select>
            </label>
            <label title={BASE_TIP}>
              base <input style={NUM} type="number" min={3} max={10} value={base}
                          onChange={(e) => setBase(
                            Math.max(3, Math.min(10, Number(e.target.value))))} />
            </label>
            <label title={LEVELS_TIP}>
              levels <input style={NUM} type="number" min={1} max={5} value={levels}
                            onChange={(e) => setLevels(
                              Math.max(1, Math.min(5, Number(e.target.value))))} />
            </label>
            <label title={FACTOR_TIP}>
              × finer
              <select value={factor} onChange={(e) => setFactor(Number(e.target.value) as 2 | 3)}
                      style={SEL}>
                <option value={2}>2</option>
                <option value={3}>3</option>
              </select>
            </label>
            <label title={RREF_TIP}>
              refine radius
              <select value={rRef} onChange={(e) => setRRef(Number(e.target.value) as 1 | 2)}
                      style={SEL}>
                <option value={1}>1 cell</option>
                <option value={2}>2 cells</option>
              </select>
            </label>
            <label title="a style lever, not a quality dial: same prompts + new seed = a genuinely different boundary map (measured). Fix it to compare settings">
              seed <input style={NUM} type="number" value={seed}
                          onChange={(e) => setSeed(Number(e.target.value))} />
            </label>
            <label title="denoising steps of a full image; the lattice probes run at 4 unless the box below is unchecked">
              steps <input style={NUM} type="number" min={1} max={50} value={steps}
                           onChange={(e) => setSteps(
                             Math.max(1, Math.min(50, Number(e.target.value))))} />
            </label>
            <label className="rx-focus" title={PROBE_TIP}
                   style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
              <input type="checkbox" checked={cheapProbes}
                     onChange={(e) => setCheapProbes(e.target.checked)} /> cheap probes
            </label>
            <button onClick={start} disabled={running || tooBig}
                    style={{ background: running || tooBig ? '#333' : '#0f3460',
                             color: '#fff', border: `1px solid ${ACCENT}`, borderRadius: 3,
                             padding: '3px 12px',
                             cursor: running || tooBig ? 'default' : 'pointer' }}>
              {running ? 'running…' : 'Start'}
            </button>
            {running && runId && (
              <button onClick={() => amrCancel(runId)}
                      style={{ background: '#0a0a1a', color: WARN,
                               border: `1px solid ${WARN}`, borderRadius: 3,
                               padding: '3px 10px', cursor: 'pointer' }}>
                Cancel
              </button>
            )}
          </div>

          {/* schedule preview + cost: the same projection the backend refuses k>=5 with */}
          <div style={{ color: '#667', marginTop: 4, fontSize: 11 }}>
            <span style={{ color: '#aeb6dd' }}>{ladder}</span> points per edge ·{' '}
            ≤ {finestCells.toLocaleString()} cells at the finest level
            {' · '}estimate: ≈{proj.total.toLocaleString()} probes ={' '}
            ≈{Math.round(proj.total * unit).toLocaleString()} image-eq
            {' · '}≈{Math.max(1, Math.round(proj.total * unit / 8 / 60))} min
            <div style={{ marginTop: 2 }}>
              from the boundary fractions measured at k=4 in h25f; the coarsest level is
              exhaustive and every refinement is capped at its own lattice. At k=4 the study
              hit the exhaustive ceiling at 55 % of its cost — this estimate is the upper
              branch, not a promise.
            </div>
            {tooBig && (
              <div style={{ color: WARN, marginTop: 2 }}>
                the finest level has {finestCells.toLocaleString()} cells, past the{' '}
                {MAX_CELLS.toLocaleString()} ceiling the backend accepts (k=4 at 60 points
                per edge, the finest ladder h25f validated). Lower levels, base or × finer.
              </div>
            )}
          </div>

          <textarea value={promptText} onChange={(e) => setPromptText(e.target.value)}
                    placeholder={`pin ${k} prompts, one per line`}
                    rows={k}
                    style={{ width: '100%', marginTop: 6, background: '#0a0a1a',
                             color: '#fff', border: '1px solid #444', borderRadius: 3,
                             fontSize: 12, padding: 4, resize: 'vertical' }} />
          {prompts.length !== k && (
            <div style={{ color: '#c9a227', fontSize: 11 }}>
              {prompts.length} prompts entered, {k} needed
            </div>
          )}
          {err && <div style={{ color: '#f66', marginTop: 4 }}>{err}</div>}

          {status && (
            <div style={{ marginTop: 6 }}>
              <LevelBar status={status} />
              <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', color: '#aaa',
                            marginTop: 4 }}>
                <span style={{ color: status.status === 'error' ? '#f66' : '#6a6' }}>
                  {status.status}
                </span>
                <span>{status.generated} images</span>
                <span title="refinement radius and the nesting factor this run was started with">
                  {(status.schedule ?? []).join(' → ')} · r_ref {status.r_ref} ·{' '}
                  ×{status.factor}
                </span>
                <span title="0.5 image-eq per cheap probe (the h25f accounting), 1.0 at full fidelity">
                  {probesSoFar.toLocaleString()} probes ={' '}
                  {status.cost_image_eq.toFixed(1)} image-eq
                </span>
                <span style={{ color: status.edges.length ? WARN : '#667' }}
                      title="pairs of adjacent evaluated cells whose images differ by more than 0.35 cosine distance — the Cascade's crossing threshold, read across the lattice">
                  {status.edges.length.toLocaleString()} boundary edges
                </span>
                {status.status === 'complete' && runId && (
                  <a href={amrPointsUrl(runId)} target="_blank" rel="noreferrer"
                     title="the evaluated cells and the detected edges as JSON, for reuse outside this map"
                     style={{ color: ACCENT }}>points.json ↗</a>
                )}
              </div>
              {status.prompts.length > 0 && (
                <div style={{ color: '#666', marginTop: 4, fontSize: 11 }}>
                  {status.prompts.join('  •  ')}
                </div>
              )}

              {status.levels_stats.length > 0 && (
                <div style={{ marginTop: 8, overflowX: 'auto' }}>
                  <table style={{ borderCollapse: 'collapse', fontSize: 11,
                                  fontVariantNumeric: 'tabular-nums' }}>
                    <thead>
                      <tr style={{ color: '#667', textAlign: 'right' }}>
                        <th style={{ textAlign: 'left', padding: '2px 8px 2px 0' }}>level</th>
                        <th style={{ padding: '2px 8px' }}>cells</th>
                        <th style={{ padding: '2px 8px' }}
                            title="cells this level evaluated: the refined set plus the ones carried over from a coarser level (nested lattices, so those cost nothing again)">
                          selected
                        </th>
                        <th style={{ padding: '2px 8px' }}
                            title="NEW probes rendered at this level — what the cost is charged on">
                          probes
                        </th>
                        <th style={{ padding: '2px 8px' }}>edges</th>
                        <th style={{ padding: '2px 8px' }}>image-eq</th>
                      </tr>
                    </thead>
                    <tbody>
                      {status.levels_stats.map((r) => (
                        <tr key={r.level} style={{ color: '#aeb6dd', textAlign: 'right' }}>
                          <td style={{ textAlign: 'left', padding: '2px 8px 2px 0' }}>
                            {r.level}
                          </td>
                          <td style={{ padding: '2px 8px', color: '#889' }}>
                            {r.cells.toLocaleString()}
                          </td>
                          <td style={{ padding: '2px 8px' }}>
                            {r.n_candidates.toLocaleString()}
                            <span style={{ color: '#667' }}>
                              {' '}({Math.round((100 * r.n_candidates) / Math.max(r.cells, 1))} %)
                            </span>
                          </td>
                          <td style={{ padding: '2px 8px' }}>
                            {r.n_evaluated.toLocaleString()}
                          </td>
                          <td style={{ padding: '2px 8px', color: r.n_edges ? WARN : '#667' }}>
                            {r.n_edges.toLocaleString()}
                          </td>
                          <td style={{ padding: '2px 8px' }}>
                            {r.cost_image_eq.toFixed(1)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {status.points.length > 0 && runId && (
                <div style={{ marginTop: 8 }}>
                  <AmrMap status={status} runId={runId} size={560}
                          imageUrl={amrImageUrl}
                          onImage={(thumb, title, rows, weights) =>
                            setDetail({ thumb, title, rows, weights: weights ?? null })} />
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
               style={{ background: '#12122c', border: '1px solid #34346a', borderRadius: 8,
                        padding: 16, display: 'flex', gap: 16, flexWrap: 'wrap',
                        maxWidth: '90vw', maxHeight: '90vh', overflow: 'auto' }}>
            <img src={amrImageUrl(runId, detail.thumb)}
                 style={{ width: 'min(60vh, 512px)', height: 'min(60vh, 512px)',
                          objectFit: 'cover', borderRadius: 6 }} />
            <div style={{ minWidth: 240, maxWidth: 320, fontSize: 12, color: '#c9d1f0' }}>
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
              {detail.weights && status.prompts.map((pr, i) => (
                <div key={i} style={{ marginBottom: 4 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis',
                                   whiteSpace: 'nowrap', maxWidth: 230 }}>{pr}</span>
                    <span style={{ color: '#8a93b8',
                                   fontVariantNumeric: 'tabular-nums' }}>
                      {((detail.weights![i] ?? 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div style={{ height: 4, background: '#1a1a2e', borderRadius: 2 }}>
                    <div style={{ width: `${Math.min(100, (detail.weights![i] ?? 0) * 100)}%`,
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
