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
 *
 * The map is up from the moment a run exists, including for the phases that are not rounds: the
 * seed survey draws its chords and their probes into it as they come back and its crossings as
 * they are found, the chains then appear as heads at those crossings (hollow until the batch
 * that measures their energies lands), and only then does the round animation start. The counter
 * row names whichever of the three is running.
 *
 * It is also a run you can DRIVE while it goes round, which is what most of this file is: the
 * map animates each round's proposals as they resolve, the sparklines show whether the chain
 * is mixing, and five controls change the run itself -- sigma, beta, a new chain, a frozen
 * chain, pause/resume. Each one lands at the START of the next round, which the control says,
 * and the two that change what the sample MEANS say that too: moving sigma or beta mid-run
 * breaks stationarity (the states before and after are draws from two different chains), and
 * placing or freezing a chain by eye is a choice in the sample. Both are the right thing to do
 * while looking for a setting that mixes and the wrong thing to do for a stimulus set you will
 * quote a law for, so the run carries a note naming the round either way and samples.json
 * exports the per-round kernel alongside the states.
 */
import { useEffect, useRef, useState } from 'react';
import {
  metroStart, metroStatus, metroCancel, metroImageUrl, metroSamplesUrl, metroCascadeRuns,
  metroSetParams, metroAddChain, metroStopChain, metroPause, metroResume,
} from './api/client';
import type { MetroStatus, MetroCascadeRun, MetroSample } from './api/types';
import MetroMap from './MetroMap';
import { ChainTable, ChainTrace, Sparkline } from './MetroLive';
import { paramChangeRounds } from './metroUtil';
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
/** the Cascade's polling interval, reused: one round is about this long on the shared pool */
const POLL_MS = 1200;
/** how long a slider sits still before the kernel change is posted */
const PARAM_DEBOUNCE_MS = 300;
const LIVE_SIGMA_TIP = 'σ for the rounds from here on. The chain is re-kernelled BETWEEN '
  + 'rounds, never inside one — which is the right way to hunt for a σ that mixes, and the '
  + 'wrong way to produce a sample you will quote a law for: the states accepted before the '
  + 'change and the ones after are draws from two different chains. The run records the round '
  + 'it happened at, the sparklines mark it, and samples.json exports the σ every state was '
  + 'actually drawn under.';
const LIVE_BETA_TIP = 'β for the rounds from here on, with the same caveat as σ — and the '
  + 'h25a one on top of it: above 1 the chain starts chasing the noise in its own energy '
  + 'estimate, and at 2 the sampled distribution is measurably the wrong shape.';
const PAUSE_TIP = 'stop dispatching between rounds. A paused run holds no GPU worker and has '
  + 'no batch in flight — it simply stops asking for the next one — so the shared pool is '
  + 'free while it waits. Resume picks up at the round it stopped at.';
/** the newest gallery arrivals ring briefly, then settle; off under reduced motion */
const GALLERY_KEYFRAMES = `
  .mg-new { animation: mgNew 2.2s ease-out forwards; }
  @keyframes mgNew { 0% { box-shadow: 0 0 0 3px ${ACCENT}; }
                     100% { box-shadow: 0 0 0 0 rgba(78,204,163,0); } }
  @media (prefers-reduced-motion: reduce) { .mg-new { animation: none; } }
`;

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

/**
 * Where the run is, for the counter row: the phase, and the round only once there are rounds.
 *
 * A run has three phases with something to say and the first two are not rounds, so counting
 * "round 0 / 50" through a minute of chord probes said nothing about what was happening. The
 * seed phase counts its probes (the only progress it has), the initial-energy batch names
 * itself, and from the first MCMC round on the round count is the number that matters.
 */
function phaseLabel(status: MetroStatus): { text: string; tip: string } {
  const ph = status.phase || status.status;
  if (ph.startsWith('seed chords')) {
    return { text: status.phase_total > 0
      ? `seeding chords · ${status.phase_done}/${status.phase_total}` : 'seeding chords',
    tip: 'the seed survey: isotropic chords probed every 0.025 on the cheap field. Every '
         + 'adjacent pair past 0.35 is a crossing, and the crossings are where the chains '
         + 'start — so no chain exists until this phase has found some.' };
  }
  if (ph === 'initial energies') {
    return { text: 'initial energies',
             tip: 'one batched round measuring S at every chain\'s start state: (1 + m) '
                  + 'probes each. The chains are already on the map at their seeds, hollow '
                  + 'until this batch gives them an energy.' };
  }
  if (ph === 'full-fidelity pass') {
    return { text: 'full-fidelity pass',
             tip: 're-rendering every distinct accepted state at full denoising steps, one '
                  + 'image each. The chain is finished; this only replaces the thumbnails.' };
  }
  return { text: `round ${status.round_done} / ${status.chain_steps}`,
           tip: 'MCMC rounds completed. One round is one batched GPU pass across every chain '
                + 'that proposed.' };
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
  // --- live: the chain being followed, the kernel the controls hold, what the last one said
  const [followed, setFollowed] = useState<number | null>(null);
  const [liveSigma, setLiveSigma] = useState(sigma);
  const [liveBeta, setLiveBeta] = useState(beta);
  const [ctlMsg, setCtlMsg] = useState<string | null>(null);
  const [ctlErr, setCtlErr] = useState<string | null>(null);
  const paramTimer = useRef<number | null>(null);

  // One poll chain only: re-entering the effect clears the previous timer, or a re-render
  // forks a second chain (the cascade panel's rule). A PAUSED run is still running, so it
  // keeps being polled -- the controls have to stay live while it waits.
  useEffect(() => {
    if (!runId) return;
    let live = true;
    const tick = async () => {
      try {
        const s = await metroStatus(runId);
        if (!live) return;
        setStatus(s);
        if (s.status === 'running') timer.current = window.setTimeout(tick, POLL_MS);
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

  // A new run resets everything that belonged to the old one, including the queued kernel
  // change -- posting a stale sigma at a run that just started would be the worst kind of
  // silent surprise.
  useEffect(() => {
    setFollowed(null);
    setCtlMsg(null);
    setCtlErr(null);
    if (paramTimer.current) window.clearTimeout(paramTimer.current);
    return () => { if (paramTimer.current) window.clearTimeout(paramTimer.current); };
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

  // --- the five live controls. Each says when it lands, because each lands next round.
  /**
   * Post the kernel 300 ms after the slider stops. BOTH values go every time: a drag across
   * sigma followed by one across beta inside the window would otherwise lose the first, and
   * the backend treats "the value it already has" as no change, so sending both is free.
   */
  const pushParams = (s: number, b: number) => {
    if (!runId || !running) return;
    if (paramTimer.current) window.clearTimeout(paramTimer.current);
    paramTimer.current = window.setTimeout(async () => {
      try {
        const r = await metroSetParams(runId, { sigma: s, beta: b });
        setCtlErr(null);
        setCtlMsg(`σ ${s} · β ${b} queued — applies at round ${r.applies_at_round}`);
      } catch (e: any) {
        setCtlErr(String(e.message || e));
      }
    }, PARAM_DEBOUNCE_MS);
  };

  const seedChain = async (w: number[], why: string) => {
    if (!runId) return;
    try {
      const r = await metroAddChain(runId, w);
      setCtlErr(null);
      setCtlMsg(`chain ${r.chain} queued at ${why} — its head appears at round `
                + `${r.joins_at_round}, once that round's batch has measured its energy`);
    } catch (e: any) {
      setCtlMsg(null);
      setCtlErr(String(e.message || e));
    }
  };

  const freezeChain = async (chain: number) => {
    if (!runId) return;
    try {
      await metroStopChain(runId, chain);
      setCtlErr(null);
      setCtlMsg(`chain ${chain} frozen from the next round — its samples are kept`);
    } catch (e: any) {
      setCtlErr(String(e.message || e));
    }
  };

  const togglePause = async () => {
    if (!runId || !status) return;
    const next = !status.paused;
    setStatus({ ...status, paused: next });     // the poll confirms it 1.2 s later
    try {
      await (next ? metroPause(runId) : metroResume(runId));
      setCtlErr(null);
      setCtlMsg(next ? 'paused between rounds — no GPU worker is held' : 'resumed');
    } catch (e: any) {
      setCtlErr(String(e.message || e));
    }
  };

  const start = async () => {
    setErr(null);
    setStatus(null);
    setDetail(null);
    setFollowed(null);
    setLiveSigma(sigma);
    setLiveBeta(beta);
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

  /** One sample in the large view; shared by the gallery, the map and the followed chain. */
  const showSample = (s: MetroSample, i: number) => setDetail({
    thumb: s.full_image ?? s.image,
    title: `sample ${i} · chain ${s.chain} · round ${s.step}`,
    weights: s.weights,
    rows: [['energy S', s.s.toFixed(3)],
           [`past ${COS_T}`, s.s > COS_T ? 'yes' : 'no'],
           ['rendered at', s.full_image != null
             ? `${status?.steps} steps (full pass)`
             : `${status?.probe_steps ?? status?.steps} steps (cheap field)`]],
  });

  const rounds = status?.rounds ?? [];
  const marks = paramChangeRounds(rounds);
  const followedChain = followed === null ? null
    : (status?.chains_stats ?? []).find((c) => c.chain === followed) ?? null;
  // the budget the counter reads against: this run's own projection, recomputed with however
  // many chains it has now (a chain added by hand raises the bill, and the counter should say
  // so rather than quietly run past the number the start form showed)
  const liveBudget = status
    ? projection(status.m, Math.max(status.chains, status.summary.n_chains),
                 status.chain_steps, status.n_seed_chords, status.seed_mode,
                 status.probe_steps != null, status.render_full).total
    : 0;

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
              <style>{GALLERY_KEYFRAMES}</style>
              <RoundBar status={status} />

              {/* the live counter: one line that says where the run is and what it has spent */}
              <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 4,
                            fontSize: 11, fontVariantNumeric: 'tabular-nums',
                            color: '#aeb6dd' }}>
                <span title={phaseLabel(status).tip}>
                  {phaseLabel(status).text}
                </span>
                <span title="image-equivalents spent against this run's own projection, recomputed with the chains it has now — a chain added by hand raises the bill">
                  cost {Math.round(sm.cost_image_eq).toLocaleString()} /{' '}
                  {Math.round(liveBudget).toLocaleString()} images
                </span>
                <span title="chains with a starting energy that are not frozen: the ones that will propose next round">
                  {sm.n_chains_moving} of {sm.n_chains} chains moving
                  {sm.n_chains_stopped > 0 && (
                    <span style={{ color: '#667' }}> · {sm.n_chains_stopped} frozen</span>
                  )}
                </span>
                {status.paused && (
                  <span style={{ color: '#c9a227' }}>paused between rounds</span>
                )}
                {status.params_changed && (
                  <span style={{ color: '#c9a227' }}
                        title="σ or β moved while this run was going. The states accepted before the change and the ones after are draws from two different chains, so the pooled set is not a draw from one law — see the note at the bottom of this panel.">
                    ⚠ kernel changed mid-run
                  </span>
                )}
              </div>

              {/* the live controls: everything here lands at the START of the next round */}
              {running && runId && (
                <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap',
                              alignItems: 'center', marginTop: 6, padding: '5px 8px',
                              background: '#0f1430', border: '1px solid #2a2a4a',
                              borderRadius: 4 }}>
                  <button onClick={togglePause} title={PAUSE_TIP}
                          style={{ background: '#0a0a1a',
                                   color: status.paused ? ACCENT : '#8a9',
                                   border: `1px solid ${status.paused ? ACCENT : '#444'}`,
                                   borderRadius: 3, padding: '2px 10px',
                                   cursor: 'pointer' }}>
                    {status.paused ? '▶ Resume' : '⏸ Pause'}
                  </button>
                  <label title={LIVE_SIGMA_TIP}
                         style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
                    σ
                    <input type="range" min={0.01} max={0.2} step={0.01} value={liveSigma}
                           onChange={(e) => {
                             const v = Number(e.target.value);
                             setLiveSigma(v);
                             pushParams(v, liveBeta);
                           }}
                           style={{ width: 96 }} />
                    <span style={{ color: '#c9d1f0', width: 30,
                                   fontVariantNumeric: 'tabular-nums' }}>
                      {liveSigma.toFixed(2)}
                    </span>
                    <span style={{ color: '#667' }}>
                      ≈{(liveSigma / STRIDE).toFixed(1)} cells
                    </span>
                  </label>
                  <label title={LIVE_BETA_TIP}
                         style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
                    β
                    <input type="range" min={0.5} max={1.5} step={0.1} value={liveBeta}
                           onChange={(e) => {
                             const v = Number(e.target.value);
                             setLiveBeta(v);
                             pushParams(liveSigma, v);
                           }}
                           style={{ width: 70 }} />
                    <span style={{ color: liveBeta > BETA_WARN ? '#c9a227' : '#c9d1f0',
                                   width: 22, fontVariantNumeric: 'tabular-nums' }}>
                      {liveBeta.toFixed(1)}
                    </span>
                  </label>
                  <span style={{ color: '#667', fontSize: 11 }}>
                    applies next round (σ {status.params.sigma} · β {status.params.beta} now)
                  </span>
                </div>
              )}
              {(ctlMsg || ctlErr) && (
                <div style={{ fontSize: 11, marginTop: 3,
                              color: ctlErr ? '#f66' : ACCENT }}>
                  {ctlErr ?? ctlMsg}
                </div>
              )}

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
              {/* two single-series sparklines: whether the chain is mixing, and whether it
                  is sitting anywhere sharp. Two measures of different scale, so two charts
                  rather than one with two y-axes. */}
              {rounds.length > 1 && (
                <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginTop: 8 }}>
                  <Sparkline
                    label="acceptance / round"
                    title={'share of each round\'s proposals that were accepted, including '
                           + 'the ones that left the simplex (which is how the summary '
                           + 'counts them too). h25a measured 0.59 at σ 0.03 / m 2: near 0 '
                           + 'means σ is too big for the field\'s scale, near 1 that the '
                           + 'chain is wandering on a flat part of it.'}
                    points={rounds.map((r) => ({ x: r.round, y: r.acc_rate ?? null }))}
                    domain={[0, 1]} marks={marks} digits={2}
                    hint={marks.length ? 'gold rule = kernel changed here'
                      : 'h25a: 0.59 at σ 0.03'} />
                  <Sparkline
                    label="mean S of states / round"
                    title={'mean energy of the chain SET at the end of each round (stopped '
                           + 'chains included — they still have a state). The y-axis is the '
                           + 'map\'s own 0–0.50 ramp scale, so a point\'s height here and a '
                           + 'dot\'s colour there are the same reading of the same number.'}
                    points={rounds.map((r) => ({ x: r.round, y: r.mean_S_states ?? null }))}
                    domain={[0, 0.5]} marks={marks} digits={3}
                    hint={`crossing threshold ${COS_T}`} />
                </div>
              )}

              {status.prompts.length > 0 && (
                <div style={{ color: '#666', marginTop: 4, fontSize: 11 }}>
                  {status.prompts.join('  •  ')}
                </div>
              )}

              {/* The map is up from the moment there IS a run: the seed survey draws itself
                  into it (chords, probes, crossings), then the chains appear as heads, then
                  the rounds animate. Gating it on "a chain has a state" left the first one to
                  two minutes of every run — the phase that decides where all 60 chains start
                  — behind a progress bar. */}
              {runId && (
                <div style={{ marginTop: 8 }}>
                  <MetroMap status={status} runId={runId} size={560}
                            imageUrl={metroImageUrl} followed={followed}
                            onFollow={setFollowed} onSeed={seedChain} live={!!running}
                            onImage={(thumb, title, rows, weights) =>
                              setDetail({ thumb, title, rows, weights: weights ?? null })} />
                </div>
              )}

              {/* the chains themselves: click a row (or a head on the map) to follow one */}
              {status.chains_stats.length > 0 && runId && (
                <div style={{ marginTop: 8 }}>
                  <div style={{ color: '#aeb6dd', marginBottom: 3 }}>
                    chains
                    <span style={{ color: '#667', fontSize: 11 }}>
                      {' '}— sharpest state first · click one to follow it on the map
                      {running ? ' · stop freezes it (its samples are kept)' : ''}
                    </span>
                  </div>
                  <ChainTable chains={status.chains_stats} followed={followed}
                              onFollow={setFollowed} onStop={freezeChain}
                              canStop={!!running} />
                  {followedChain && (
                    <ChainTrace chain={followedChain} samples={status.samples}
                                roundsDone={status.round_done} runId={runId}
                                imageUrl={metroImageUrl} onImage={showSample}
                                marks={marks} />
                  )}
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
                      border = S on the map's ramp · click to enlarge · newest arrivals ring
                      briefly
                      {status.render_full && sm.n_full > 0
                        ? ' · re-rendered at full steps' : ''}
                    </span>
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {gallery.map((i) => {
                      const s = status.samples[i];
                      const img = s.full_image ?? s.image;
                      if (img < 0) return null;
                      // Sorted by S, so the states that just arrived are scattered through
                      // the gallery and the ring is the only thing that says which they are.
                      // "Just arrived" is the ROUND they were accepted at, not a count kept
                      // between renders: a count would make the ring survive exactly until
                      // the next re-render, whatever caused it, instead of for the round.
                      const fresh = running && status.round_done > 0
                        && s.step === status.round_done;
                      const dimmed = followed !== null && s.chain !== followed;
                      return (
                        <img key={i} src={metroImageUrl(runId, img)} width={72} height={72}
                             className={fresh ? 'mg-new' : undefined}
                             title={`chain ${s.chain} · round ${s.step} · S ${s.s.toFixed(3)}`
                                    + (fresh ? ' · just arrived' : '')}
                             onClick={() => showSample(s, i)}
                             style={{ objectFit: 'cover', borderRadius: 4,
                                      cursor: 'pointer', opacity: dimmed ? 0.3 : 1,
                                      border: `2px solid ${divColor(s.s)}` }} />
                      );
                    })}
                  </div>
                  {followed !== null && (
                    <div style={{ color: '#667', fontSize: 11, marginTop: 2 }}>
                      following chain {followed} — the other chains' states are dimmed here
                      and on the map
                    </div>
                  )}
                </div>
              )}

              {/* the last few notes, not just the last one: driving a run adds notes (a
                  kernel change, a chain placed by hand) that are exactly the caveats someone
                  reading the sample set afterwards needs, and the newest note would otherwise
                  push them off screen */}
              {status.notes.length > 0 && (
                <div style={{ color: '#666', marginTop: 8, fontSize: 11 }}>
                  {status.notes.slice(-3).map((n, i) => (
                    <div key={i} style={{ marginTop: i ? 3 : 0,
                                          color: /stationarity|by hand/.test(n)
                                            ? '#c9a227' : '#666' }}>
                      {n}
                    </div>
                  ))}
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
