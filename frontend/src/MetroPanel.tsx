/**
 * MetroPanel -- the Metropolis ridge sampler as an interactive run.
 *
 * Every other surface in this tool SEARCHES: chords and lattices look for boundaries and
 * then report where they are. This one SAMPLES. Its chain's stationary law is the sharpness
 * field itself -- pi(w) ~ S(w)^beta, where S is the tool's own divergence read at the chord
 * stride over m random tangent directions -- so the gallery below is a fair draw from that
 * law rather than a hand-picked tour of the sharpest places something happened to find. The
 * use of record: stimulus sets and galleries of ridge images at any k, at about one image
 * per energy evaluation, where certifying crossings (~12 images each) is not needed.
 *
 * Calibration of record, search_problem h25a (verdict independently replicated):
 *
 *   * beta = 1 reproduces the law under single-seed energies -- on-boundary fraction 0.673
 *     against the 0.677 the exact S^beta law gives, KS 0.083, a 2.3-2.6x enrichment over
 *     uniform sampling. beta = 2 does NOT (KS 0.19-0.28): squaring a 1-2-direction estimate
 *     of S amplifies the ESTIMATOR noise, so the chain locks onto cells whose estimate
 *     happened to be large. The backend caps beta at 1.5 and the control warns above 1.
 *   * ERPT-style short chains (60 x 50) seeded at crossings beat one long chain on the law
 *     at small sigma, because the seeds already sit on boundaries.
 *   * NOT competitive for DISCOVERY: 0.91-1.05x the distinct boundaries plain IUR chords
 *     find at equal cost. The Cascade stays the instrument for "where are the boundaries".
 *
 * Each MCMC step across ALL chains is one batched GPU round, so a C x L run is L rounds
 * plus one for the chains' initial energies -- which is what makes 3,000 evaluations a
 * matter of minutes on the pool rather than days.
 */
import { useEffect, useRef, useState } from 'react';
import {
  metroStart, metroStatus, metroCancel, metroImageUrl, metroSamplesUrl, metroCascadeRuns,
} from './api/client';
import type { MetroStatus, MetroCascadeRun } from './api/types';
import MetroMap from './MetroMap';
import { divColor } from './CascadeMap';

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
/** the crossing threshold the on-boundary proxy counts against (cascade.COS_T) */
const COS_T = 0.35;
/** the stride the energy is read at (cascade.STRIDE), for the sigma readout */
const STRIDE = 0.025;
/** backend ceiling on a projected run, in image-eq (metro.MAX_IMAGE_EQ) */
const MAX_IMAGE_EQ = 20000;
/** mean k=4 IUR chord length, h25b -- the same number the backend projects with */
const MEAN_CHORD_LEN = 0.573;
/** above this beta the single-seed law degrades (metro.BETA_WARN) */
const BETA_WARN = 1.0;

const BETA_TIP = 'the exponent of the target law pi ~ S^beta: 1 samples in proportion to '
  + 'sharpness, higher concentrates harder on the sharpest places. 1 is the setting of '
  + 'record — h25a measured beta 2 FAILING under single-seed energies (KS 0.19–0.28 vs '
  + '0.07–0.13 at beta 1), because squaring a 1–2-direction estimate of S amplifies the '
  + 'estimator noise rather than the signal. The backend stops at 1.5.';
const BETA_OVER_TIP = 'above 1, the chain starts chasing the NOISE in its own energy '
  + 'estimate: nothing between 1 and 1.5 was measured, and at 2 the sampled distribution is '
  + 'measurably the wrong shape. Raise m as well if you raise this.';
const SIGMA_TIP = 'random-walk step, in weight-space units (the simplex edge is √2). 0.03 '
  + 'is the best LAW setting with m = 2; larger steps mix faster and sample the law less '
  + 'tightly (h25a: sigma 0.12 had the best autocorrelation but a worse KS at the same m). '
  + 'A proposal that leaves the simplex is rejected, which costs nothing.';
const M_TIP = 'how many random tangent directions one energy averages. The cost of an '
  + 'energy evaluation is (1 + m) probes — the state\'s own image plus its fan — so m is the '
  + 'main price lever. 2 is of record; 1 is cheaper and noisier (and noise is exactly what '
  + 'beta > 1 amplifies).';
const CHAINS_TIP = 'ERPT-style: many SHORT chains rather than one long one. Each chain '
  + 'starts on a seed crossing, so the run begins on boundaries instead of walking to them; '
  + 'h25a measured 4.5–8.1 distinct basins visited per chain at 60 × 50.';
const STEPS_TIP = 'MCMC steps per chain (L). One step across ALL chains is one batched GPU '
  + 'round, so this is the number of rounds the run takes — not the denoising steps of an '
  + 'image.';
const SEEDMODE_TIP = 'where the chains start. "fresh IUR chords" draws its own isotropic '
  + 'chords and seeds at their crossings (uniform anchors, not the tool\'s Mitchell ones — '
  + 'h25b measured those sitting at the simplex faces). "a finished Cascade run" reuses that '
  + 'run\'s crossings (their midpoints), which costs no survey at all.';
const RENDER_TIP = 'after the chain finishes, re-render every DISTINCT accepted state at '
  + 'full denoising steps — one whole image each, on top of the cheap probes. Off by '
  + 'default: the gallery reads fine off the cheap field, and this is for when the samples '
  + 'are going to be shown to people.';

function fmt(x: number | null | undefined, d = 3): string {
  return x === null || x === undefined || !Number.isFinite(x) ? '—' : x.toFixed(d);
}

/** The projection the backend refuses a run with, recomputed here so the two agree. */
function projection(
  m: number, chains: number, steps: number, chords: number, mode: string, cheap: boolean,
  full: boolean,
) {
  const unit = cheap ? 0.5 : 1.0;
  const perChord = Math.floor(MEAN_CHORD_LEN / STRIDE) + 1;
  const rows: { what: string; probes: number; eq: number }[] = [];
  if (mode === 'iur') rows.push({ what: `${chords} seed chords`, probes: chords * perChord,
                                  eq: 0 });
  rows.push({ what: 'initial energies', probes: chains * (1 + m), eq: 0 });
  rows.push({ what: `${steps} rounds`, probes: chains * (1 + m) * steps, eq: 0 });
  for (const r of rows) r.eq = unit * r.probes;
  let total = rows.reduce((a, r) => a + r.eq, 0);
  if (full) {
    rows.push({ what: 'full-fidelity pass (worst case)', probes: chains * steps,
                eq: chains * steps });
    total += chains * steps;
  }
  return { rows, total };
}

/** One segment per round: the rounds already done full, the running one filling by probes. */
function RoundBar({ status }: { status: MetroStatus }) {
  const L = Math.max(1, status.chain_steps);
  const done = status.round_done;
  const frac = status.phase_total > 0
    ? Math.min(1, status.phase_done / status.phase_total) : 0;
  const pct = Math.min(100, (100 * (done + (status.status === 'running' ? frac : 0))) / L);
  return (
    <div style={{ margin: '8px 0 2px' }}>
      <div style={{ height: 6, background: '#0a0a1a', borderRadius: 3, overflow: 'hidden',
                    border: '1px solid #2a2a4a' }}>
        <div style={{ width: `${pct}%`, height: '100%',
                      background: status.status === 'complete' ? ACCENT : '#7d84b5',
                      transition: 'width 0.6s' }} />
      </div>
      <div style={{ fontSize: 10, marginTop: 2, color: '#889',
                    fontVariantNumeric: 'tabular-nums' }}>
        {status.phase || status.status}
        {status.phase_total > 0 && ` · ${status.phase_done}/${status.phase_total} probes`}
        {' · '}round {done}/{L}
      </div>
    </div>
  );
}

export default function MetroPanel() {
  const [open, setOpen] = useState(false);
  const [k, setK] = useState(4);
  const [beta, setBeta] = useState(1.0);
  const [sigma, setSigma] = useState(0.03);
  const [m, setM] = useState(2);
  const [chains, setChains] = useState(60);
  const [chainSteps, setChainSteps] = useState(50);
  const [seedMode, setSeedMode] = useState<'iur' | 'cascade'>('iur');
  const [cascadeRunId, setCascadeRunId] = useState('');
  const [nChords, setNChords] = useState(20);
  const [seed, setSeed] = useState(42);
  const [steps, setSteps] = useState(8);
  const [cheapProbes, setCheapProbes] = useState(true);
  const [renderFull, setRenderFull] = useState(false);
  // Like AmrPanel: a pinned basis, prefilled so Start works on the first click.
  const [promptText, setPromptText] = useState(
    'a photograph of a lighthouse\na pencil sketch of a cathedral\n'
    + 'an oil painting of a forest\na neon city at night');
  const [runId, setRunId] = useState<string | null>(null);
  const [status, setStatus] = useState<MetroStatus | null>(null);
  const [sources, setSources] = useState<MetroCascadeRun[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<'s' | 'order'>('s');
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
        const s = await metroStatus(runId);
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

  // The seed-source list, for this k only: a Cascade run's crossings are weight vectors over
  // ITS prompts, so only a run of the same arity can seed these chains.
  useEffect(() => {
    if (!open || seedMode !== 'cascade') return;
    let live = true;
    metroCascadeRuns(k)
      .then((rows) => { if (live) setSources(rows); })
      .catch(() => { if (live) setSources([]); });   // fall back to the text field
    return () => { live = false; };
  }, [open, seedMode, k]);

  const prompts = promptText.split('\n').map((s) => s.trim()).filter(Boolean);
  const proj = projection(m, chains, chainSteps, nChords, seedMode, cheapProbes, renderFull);
  const tooBig = proj.total > MAX_IMAGE_EQ;
  const running = status?.status === 'running';
  const sm = status?.summary;

  const start = async () => {
    setErr(null);
    setStatus(null);
    setDetail(null);
    if (prompts.length !== k) {
      setErr(`enter exactly ${k} prompts, one per line (got ${prompts.length})`);
      return;
    }
    if (seedMode === 'cascade' && !cascadeRunId.trim()) {
      setErr('pick a finished Cascade run to seed from, or switch to fresh IUR chords');
      return;
    }
    try {
      const r = await metroStart({
        prompts, beta, sigma, m, chains, chain_steps: chainSteps, seed_mode: seedMode,
        cascade_run_id: seedMode === 'cascade' ? cascadeRunId.trim() : null,
        n_seed_chords: nChords, seed, steps,
        probe_steps: cheapProbes ? 4 : null, render_full: renderFull,
      });
      if (r.error) { setErr(r.error); return; }
      setRunId(r.run_id);
    } catch (e: any) {
      setErr(String(e.message || e));
    }
  };

  // The gallery: accepted states, sharpest first by default. `full_image` when the optional
  // pass ran, the cheap probe otherwise -- both are images OF that recipe.
  const gallery = (() => {
    const idx = (status?.samples ?? []).map((_s, i) => i);
    if (sortBy === 's') {
      idx.sort((a, b) => (status!.samples[b].s - status!.samples[a].s));
    }
    return idx;
  })();

  return (
    <div style={BOX}>
      <div style={{ cursor: 'pointer', userSelect: 'none' }} onClick={() => setOpen(!open)}>
        {open ? '▾' : '▸'} Ridge sampler — Metropolis on the sharpness field
      </div>
      {open && (
        <div style={{ marginTop: 8 }}>
          <div style={{ color: '#667', fontSize: 11, marginBottom: 6 }}>
            Samples recipes in proportion to how sharp the field is there (π ∝ S<sup>β</sup>,
            S = mean 1−cos over {m} tangent direction{m === 1 ? '' : 's'} at the{' '}
            {STRIDE} stride) — a fair draw for a stimulus set, not a search. Validated in
            h25a: β = 1 reaches 0.99× the ideal on-boundary fraction under single-seed
            noise; β = 2 does not. For DISCOVERY the Cascade's chords remain better at
            equal cost.
          </div>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
            <label title="prompts spanning the simplex. Any k ≥ 3: unlike the lattice surfaces, a chain costs the same per step at every k — which is the point of it.">
              k
              <select value={k} onChange={(e) => setK(Number(e.target.value))} style={SEL}>
                {[3, 4, 5, 6, 7, 8].map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </label>
            <label title={BETA_TIP}>
              β <input style={NUM} type="number" min={0.5} max={1.5} step={0.1} value={beta}
                       onChange={(e) => setBeta(
                         Math.max(0.5, Math.min(1.5, Number(e.target.value))))} />
            </label>
            <label title={SIGMA_TIP}>
              σ <input style={NUM} type="number" min={0.01} max={0.2} step={0.01}
                       value={sigma}
                       onChange={(e) => setSigma(
                         Math.max(0.01, Math.min(0.2, Number(e.target.value))))} />
              <span style={{ color: '#667' }}>
                {' '}≈{(sigma / STRIDE).toFixed(1)} cells
              </span>
            </label>
            <label title={M_TIP}>
              m <input style={NUM} type="number" min={1} max={4} value={m}
                       onChange={(e) => setM(
                         Math.max(1, Math.min(4, Number(e.target.value))))} />
            </label>
            <label title={CHAINS_TIP}>
              chains <input style={NUM} type="number" min={1} max={200} value={chains}
                            onChange={(e) => setChains(
                              Math.max(1, Math.min(200, Number(e.target.value))))} />
            </label>
            <label title={STEPS_TIP}>
              × steps <input style={NUM} type="number" min={1} max={200} value={chainSteps}
                             onChange={(e) => setChainSteps(
                               Math.max(1, Math.min(200, Number(e.target.value))))} />
            </label>
            <label title="a style lever, not a quality dial: same prompts + new seed = a genuinely different field, and a different chain over it. Fix it to compare settings">
              seed <input style={NUM} type="number" value={seed}
                          onChange={(e) => setSeed(Number(e.target.value))} />
            </label>
            <label title="denoising steps of a full image; the probes run at 4 unless the box below is unchecked">
              steps <input style={NUM} type="number" min={1} max={50} value={steps}
                           onChange={(e) => setSteps(
                             Math.max(1, Math.min(50, Number(e.target.value))))} />
            </label>
            <label className="rx-focus"
                   title="probes run on the cheap field (4 of 8 denoising steps), as the Cascade's chord probes do — two probes for the price of one image. Uncheck to probe at full fidelity, which doubles the bill"
                   style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
              <input type="checkbox" checked={cheapProbes}
                     onChange={(e) => setCheapProbes(e.target.checked)} /> cheap probes
            </label>
            <label className="rx-focus" title={RENDER_TIP}
                   style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
              <input type="checkbox" checked={renderFull}
                     onChange={(e) => setRenderFull(e.target.checked)} /> render accepted at
              full steps
            </label>
            <button onClick={start} disabled={running || tooBig}
                    style={{ background: running || tooBig ? '#333' : '#0f3460',
                             color: '#fff', border: `1px solid ${ACCENT}`, borderRadius: 3,
                             padding: '3px 12px',
                             cursor: running || tooBig ? 'default' : 'pointer' }}>
              {running ? 'running…' : 'Start'}
            </button>
            {running && runId && (
              <button onClick={() => metroCancel(runId)}
                      style={{ background: '#0a0a1a', color: WARN,
                               border: `1px solid ${WARN}`, borderRadius: 3,
                               padding: '3px 10px', cursor: 'pointer' }}>
                Cancel
              </button>
            )}
          </div>

          {beta > BETA_WARN && (
            <div style={{ color: '#c9a227', fontSize: 11, marginTop: 4 }}
                 title={BETA_OVER_TIP}>
              ⚠ β {beta.toFixed(1)} is above the 1.0 of record — nothing between 1 and 1.5
              was measured, and at β = 2 the sampled distribution is the wrong shape under
              single-seed energies (KS 0.19–0.28). Raise m if you raise β.
            </div>
          )}

          {/* where the chains start */}
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center',
                        marginTop: 6 }}>
            <label title={SEEDMODE_TIP}>
              seeds
              <select value={seedMode} style={SEL}
                      onChange={(e) => setSeedMode(e.target.value as 'iur' | 'cascade')}>
                <option value="iur">fresh IUR chords</option>
                <option value="cascade">a finished Cascade run</option>
              </select>
            </label>
            {seedMode === 'iur' ? (
              <label title="isotropic chords through the simplex, probed every 0.025 on the cheap field; every adjacent pair past 0.35 is a crossing, and the crossings' midpoints are the chain starts. 20 is enough to seed 60 chains (several chains then share a sheet, which the run says).">
                chords <input style={NUM} type="number" min={1} max={200} value={nChords}
                              onChange={(e) => setNChords(
                                Math.max(1, Math.min(200, Number(e.target.value))))} />
              </label>
            ) : sources && sources.length > 0 ? (
              <label title="finished Cascade runs of this k, most crossings first. Their crossings' midpoints become the chain starts, so this seeding costs no images at all.">
                run
                <select value={cascadeRunId} style={SEL}
                        onChange={(e) => setCascadeRunId(e.target.value)}>
                  <option value="">pick a run…</option>
                  {sources.map((s) => (
                    <option key={s.run_id} value={s.run_id} disabled={!s.n_crossings}>
                      {s.run_id} · {s.n_crossings} crossings
                      {s.certify ? ' · certified' : ''}
                      {s.prompts.length ? ` · ${s.prompts[0].slice(0, 18)}…` : ''}
                    </option>
                  ))}
                </select>
              </label>
            ) : (
              <label title="the 8-hex id of a finished Cascade run of this k, whose crossings the chains start on">
                run id
                <input style={{ ...NUM, width: 110 }} value={cascadeRunId}
                       placeholder={sources ? `no finished k=${k} run` : 'loading…'}
                       onChange={(e) => setCascadeRunId(e.target.value)} />
              </label>
            )}
          </div>

          {/* cost estimate: the same projection the backend refuses a run with */}
          <div style={{ color: '#667', marginTop: 4, fontSize: 11 }}>
            estimate: ≈{Math.round(proj.total).toLocaleString()} image-eq ={' '}
            {proj.rows.map((r) => `${r.what} ${Math.round(r.eq).toLocaleString()}`)
              .join(' + ')}
            {' · '}≈{Math.max(1, Math.round(proj.total / 8 / 60))} min
            <div style={{ marginTop: 2 }}>
              an energy evaluation is (1 + {m}) probes at {cheapProbes ? '0.5' : '1.0'}{' '}
              image-eq each, and the current state's image is cached — so a round is{' '}
              {(1 + m) * chains} probes and the whole chain is {chainSteps} rounds. A
              proposal that leaves the simplex is rejected without rendering anything, so a
              real run comes in under this.
            </div>
            {tooBig && (
              <div style={{ color: WARN, marginTop: 2 }}>
                ≈{Math.round(proj.total).toLocaleString()} image-eq is past the{' '}
                {MAX_IMAGE_EQ.toLocaleString()} ceiling the backend accepts. Lower chains,
                steps or m (the design of record is 60 × 50 with m = 2).
              </div>
            )}
          </div>

          <textarea value={promptText} onChange={(e) => setPromptText(e.target.value)}
                    placeholder={`pin ${k} prompts, one per line`}
                    rows={Math.min(k, 8)}
                    style={{ width: '100%', marginTop: 6, background: '#0a0a1a',
                             color: '#fff', border: '1px solid #444', borderRadius: 3,
                             fontSize: 12, padding: 4, resize: 'vertical' }} />
          {prompts.length !== k && (
            <div style={{ color: '#c9a227', fontSize: 11 }}>
              {prompts.length} prompts entered, {k} needed
            </div>
          )}
          {err && <div style={{ color: '#f66', marginTop: 4 }}>{err}</div>}

          {status && sm && (
            <div style={{ marginTop: 6 }}>
              <RoundBar status={status} />
              <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', color: '#aaa',
                            marginTop: 4 }}>
                <span style={{ color: status.status === 'error' ? '#f66' : '#6a6' }}>
                  {status.status}
                </span>
                <span>{status.generated} images</span>
                <span title="β, σ and m this run was started with, and the stride its energy is read at">
                  β {status.beta} · σ {status.sigma} · m {status.m} · δ{' '}
                  {status.delta.toFixed(3)}
                </span>
                <span title="0.5 image-eq per cheap probe (the h25 accounting), 1.0 at full fidelity; a full-fidelity re-render of an accepted state is a whole image">
                  {sm.n_probes.toLocaleString()} probes
                  {sm.n_full > 0 && ` + ${sm.n_full} full`} ={' '}
                  {sm.cost_image_eq.toFixed(1)} image-eq
                </span>
                <span title="accepted states: the chain's distinct draws, which is what the gallery and the map show">
                  {sm.n_samples.toLocaleString()} samples
                </span>
                <span title="share of proposals accepted. h25a measured 0.59 at σ 0.03 / m 2; a very low rate means σ is too big for the field's scale, a very high one that the chain is wandering on a flat part of it">
                  acceptance {fmt(sm.acceptance, 2)}
                  {sm.n_outside > 0 && (
                    <span style={{ color: '#667' }}>
                      {' '}({sm.n_outside.toLocaleString()} left the simplex)
                    </span>
                  )}
                </span>
                {status.status === 'complete' && runId && (
                  <a href={metroSamplesUrl(runId)} target="_blank" rel="noreferrer"
                     title="the accepted states and the settings they were drawn under, as JSON — a reproducible stimulus set"
                     style={{ color: ACCENT }}>samples.json ↗</a>
                )}
              </div>

              {/* the three numbers that say whether the sampler did its job */}
              <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', marginTop: 6,
                            fontVariantNumeric: 'tabular-nums' }}>
                <span title="mean energy of the accepted states, against the mean energy of the crossings the chains STARTED from. Above 1 means the chain found sharper places than the survey handed it; near 1 means it held on to them (which is also a pass — the seeds are already on boundaries)">
                  <span style={{ color: '#aeb6dd' }}>mean S</span>{' '}
                  <span style={{ color: divColor(sm.mean_s ?? null) }}>
                    {fmt(sm.mean_s)}
                  </span>
                  <span style={{ color: '#667' }}>
                    {' '}vs seeds {fmt(sm.seed_mean_s)}
                    {sm.mean_s && sm.seed_mean_s
                      ? ` (${(sm.mean_s / sm.seed_mean_s).toFixed(2)}×)` : ''}
                  </span>
                </span>
                <span title={`share of accepted states whose S is past ${COS_T}, the Cascade's crossing threshold — the "on-boundary proxy". h25a: 0.67 at beta 1, against 0.26 for uniform sampling and 0.68 for the exact law`}>
                  <span style={{ color: '#aeb6dd' }}>on-boundary</span>{' '}
                  {fmt(sm.on_boundary_frac, 2)}
                  <span style={{ color: '#667' }}> (h25a 0.67 · uniform 0.26)</span>
                </span>
                <span title="chains that never accepted a single move, i.e. that kept the seed sheet they started on. h25a measured 1–5 % of chains; many more than that means sigma is too large for this field">
                  <span style={{ color: '#aeb6dd' }}>seed sheets kept</span>{' '}
                  {sm.chains_never_moved}/{sm.n_chains}
                </span>
                <span title="the seed crossings the chains were started from, and the chords that found them (none in cascade seed mode — it reuses a survey)">
                  <span style={{ color: '#aeb6dd' }}>seeds</span>{' '}
                  {sm.n_seed_crossings}
                  {sm.n_chords > 0 && (
                    <span style={{ color: '#667' }}> from {sm.n_chords} chords</span>
                  )}
                </span>
              </div>
              {status.prompts.length > 0 && (
                <div style={{ color: '#666', marginTop: 4, fontSize: 11 }}>
                  {status.prompts.join('  •  ')}
                </div>
              )}

              {status.samples.length > 0 && runId && (
                <div style={{ marginTop: 8 }}>
                  <MetroMap status={status} runId={runId} size={560}
                            imageUrl={metroImageUrl}
                            onImage={(thumb, title, rows, weights) =>
                              setDetail({ thumb, title, rows, weights: weights ?? null })} />
                </div>
              )}

              {/* the gallery: the sample set itself */}
              {status.samples.length > 0 && runId && (
                <div style={{ marginTop: 10 }}>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center',
                                color: '#aeb6dd', marginBottom: 4 }}>
                    <span>accepted states</span>
                    <select value={sortBy} style={SEL}
                            onChange={(e) => setSortBy(e.target.value as 's' | 'order')}
                            title="sharpest first reads the sample set as a gallery; chain order reads it as the chains' history">
                      <option value="s">sharpest first</option>
                      <option value="order">in the order accepted</option>
                    </select>
                    <span style={{ color: '#667', fontSize: 11 }}>
                      border = S on the map's ramp · click to enlarge
                      {status.render_full && sm.n_full > 0
                        ? ' · re-rendered at full steps' : ''}
                    </span>
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {gallery.map((i) => {
                      const s = status.samples[i];
                      const img = s.full_image ?? s.image;
                      if (img < 0) return null;
                      return (
                        <img key={i} src={metroImageUrl(runId, img)} width={72} height={72}
                             title={`chain ${s.chain} · round ${s.step} · S ${s.s.toFixed(3)}`}
                             onClick={() => setDetail({
                               thumb: img,
                               title: `sample ${i} · chain ${s.chain} · round ${s.step}`,
                               weights: s.weights,
                               rows: [['energy S', s.s.toFixed(3)],
                                      [`past ${COS_T}`, s.s > COS_T ? 'yes' : 'no'],
                                      ['rendered at', s.full_image != null
                                        ? `${status.steps} steps (full pass)`
                                        : `${status.probe_steps ?? status.steps} steps `
                                          + '(cheap field)']],
                             })}
                             style={{ objectFit: 'cover', borderRadius: 4,
                                      cursor: 'pointer',
                                      border: `2px solid ${divColor(s.s)}` }} />
                      );
                    })}
                  </div>
                </div>
              )}

              {status.notes.length > 0 && (
                <div style={{ color: '#666', marginTop: 8, fontSize: 11 }}>
                  {status.notes[status.notes.length - 1]}
                </div>
              )}
              {status.error && (
                <div style={{ color: '#f66', marginTop: 4 }}>{status.error}</div>
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
            <img src={metroImageUrl(runId, detail.thumb)}
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
