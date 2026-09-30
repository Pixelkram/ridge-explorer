/**
 * Probes: the two local measurements that answer "why does the image change HERE?"
 *
 *  - JVP probe  (~25 s): the crossing direction at one point of the simplex — the
 *    direction across prompt space along which the image changes fastest. Sign-free,
 *    so it is a line, not an arrow, and the viewport draws it as a segment.
 *  - token probe (~3 min): which words of one prompt the generation leans on here.
 *
 * Plus one cascade overlay that measures nothing new: the boundary-density map is a
 * re-reading of the survey's own chords (no GPU, one GET), so it lives here for the same
 * reason the cascade probes do — it is per-run state with a lifecycle of its own, and
 * `syncCascadeRun` must drop it when the run changes.
 *
 * Kept out of ridgeStore because its poller owns the grid job's lifecycle and runs on
 * its own clock; a probe is a side measurement that outlives neither the job nor the
 * page. `syncJob` is the one coupling: probe coordinates mean nothing once the job
 * (and so the prompt simplex) has changed, so the list is dropped with the job.
 */
import { create } from 'zustand';
import { startJvpProbe, getJvpProbe, startTokenProbe, getTokenProbe,
         startCascadeJvpProbe, cascadeLocalSv } from '../api/client';
import type { CascadeJvpProbeResult, JvpProbeCore, JvpProbeResult, LocalSvMap,
              TokenProbeResult, TokenProbeWhich } from '../api/types';

export interface JvpProbeEntry {
  alpha: number;
  beta: number;
  result: JvpProbeResult;
}

/**
 * Cascade probes live beside the grid ones but on their own clock: a cascade run is not
 * a grid job, so neither lifecycle may clear the other. They are filed per run and per
 * crossing id — `cascadeProbes[runId][cid]` — because a cid means nothing outside the
 * run that issued it, the way (alpha, beta) means nothing outside its job.
 */
export type CascadeProbeMap = Record<number, CascadeJvpProbeResult>;

export type ProbePhase = 'idle' | 'running' | 'done' | 'error';

/**
 * How the boundary-density overlay is READ, which is not the same question as how it was
 * computed. One response carries both: `ranking` colours by percentile and is trustworthy
 * from ~20 chords; `calibrated` colours by S_V units and is only honest once the map says
 * `calibrated_ok` (>= 80 chords, and the calibration itself was established at k = 4).
 * Switching between them costs nothing -- no refetch.
 */
export type LocalSvView = 'ranking' | 'calibrated';

const POLL_MS = 1500;
// Bound both pollers. A probe whose GPU task was drained (cancel, restart) never
// reports done, and an unbounded interval then polls a dead id for the rest of the
// session — the seed probe shipped with exactly that bug.
const JVP_MAX_POLLS = 400;    // ~10 min for a ~25 s probe
const TOKEN_MAX_POLLS = 800;  // ~20 min for a ~3 min probe

const CASCADE_MAX_POLLS = 400;   // ~10 min, same budget as the grid JVP probe

// Bumped by every start, clear and job change. A launch that finds the counter has
// moved while it was awaiting its POST has been superseded and must not adopt its
// probe id or start a poller for it.
let jvpSeq = 0;
let tokenSeq = 0;
let cascadeSeq = 0;
let localSvSeq = 0;

interface TokenState {
  which: TokenProbeWhich | null;
  status: ProbePhase;
  result: TokenProbeResult | null;
  error: string | null;
}

interface ProbeState {
  /** The job the probes below belong to. */
  jobId: string | null;
  probes: JvpProbeEntry[];
  /** id of the probe in flight, and where it was launched (for the live marker). */
  pendingId: string | null;
  pendingAt: { alpha: number; beta: number } | null;
  jvpStatus: ProbePhase;
  jvpError: string | null;
  jvpPoll: ReturnType<typeof setInterval> | null;

  token: TokenState;
  tokenId: string | null;
  tokenPoll: ReturnType<typeof setInterval> | null;

  /** The cascade run the map below belongs to, and its finished probes by cid. */
  cascadeRunId: string | null;
  cascadeProbes: Record<string, CascadeProbeMap>;
  cascadePendingCid: number | null;
  cascadeStatus: ProbePhase;
  cascadeError: string | null;
  cascadePoll: ReturnType<typeof setInterval> | null;

  /** Boundary-density overlay: off by default, filed per run like the probes above. */
  localSvOn: boolean;
  localSvView: LocalSvView;
  localSv: Record<string, LocalSvMap>;
  localSvStatus: ProbePhase;
  localSvError: string | null;

  syncJob: (jobId: string | null) => void;
  startJvp: (jobId: string, alpha: number, beta: number, seed?: number) => Promise<void>;
  clearProbes: () => void;
  startToken: (jobId: string, which: TokenProbeWhich, seed?: number) => Promise<void>;
  clearToken: () => void;
  stopPolling: () => void;

  syncCascadeRun: (runId: string | null) => void;
  startCascadeJvp: (runId: string, cid: number, seed?: number) => Promise<void>;
  clearCascadeProbes: () => void;

  setLocalSvOn: (runId: string | null, on: boolean) => void;
  setLocalSvView: (view: LocalSvView) => void;
  fetchLocalSv: (runId: string) => Promise<void>;
}

const IDLE_TOKEN: TokenState = { which: null, status: 'idle', result: null, error: null };

export const useProbeStore = create<ProbeState>((set, get) => ({
  jobId: null,
  probes: [],
  pendingId: null,
  pendingAt: null,
  jvpStatus: 'idle',
  jvpError: null,
  jvpPoll: null,
  token: IDLE_TOKEN,
  tokenId: null,
  tokenPoll: null,
  cascadeRunId: null,
  cascadeProbes: {},
  cascadePendingCid: null,
  cascadeStatus: 'idle',
  cascadeError: null,
  cascadePoll: null,
  localSvOn: false,
  localSvView: 'ranking',
  localSv: {},
  localSvStatus: 'idle',
  localSvError: null,

  // Deliberately does NOT touch cascadePoll: this is called on every grid job change,
  // and a cascade probe is not part of that lifecycle.
  stopPolling: () => {
    const { jvpPoll, tokenPoll } = get();
    if (jvpPoll) clearInterval(jvpPoll);
    if (tokenPoll) clearInterval(tokenPoll);
    set({ jvpPoll: null, tokenPoll: null });
  },

  // Called from the viewport whenever the grid store's jobId changes (including to
  // null on cancel). A probe is measured at (alpha, beta) of ONE prompt simplex, so
  // carrying it over would draw a segment at coordinates that now mean something else.
  syncJob: (jobId) => {
    if (get().jobId === jobId) return;
    jvpSeq++; tokenSeq++;
    get().stopPolling();
    set({
      jobId, probes: [], pendingId: null, pendingAt: null,
      jvpStatus: 'idle', jvpError: null,
      token: IDLE_TOKEN, tokenId: null,
    });
  },

  clearProbes: () => {
    jvpSeq++;
    const { jvpPoll } = get();
    if (jvpPoll) clearInterval(jvpPoll);
    set({ probes: [], pendingId: null, pendingAt: null,
          jvpStatus: 'idle', jvpError: null, jvpPoll: null });
  },

  startJvp: async (jobId, alpha, beta, seed) => {
    const launch = ++jvpSeq;
    const { jvpPoll } = get();
    if (jvpPoll) clearInterval(jvpPoll);
    set({ jvpPoll: null, jvpStatus: 'running', jvpError: null,
          pendingId: null, pendingAt: { alpha, beta } });

    const fail = (msg: string) => {
      if (jvpSeq !== launch) return;
      set({ jvpStatus: 'error', jvpError: msg, jvpPoll: null,
            pendingId: null, pendingAt: null });
    };

    try {
      const res = await startJvpProbe(jobId, { alpha, beta, seed });
      if (jvpSeq !== launch) return;   // superseded while /jvp-probe was in flight
      // The endpoint answers HTTP 200 with status 'error' and an empty probe_id when
      // it refuses — the job is gone, or the point lies outside the prompt simplex.
      // Without this branch the button would sit at 'probing…' forever.
      if (res.status !== 'running' || !res.probe_id) {
        fail(res.error || 'the server refused this probe — the point may lie outside the '
                        + 'prompt simplex, or the job is no longer on the server');
        return;
      }
      set({ pendingId: res.probe_id });

      let attempts = 0;
      const poll = setInterval(async () => {
        attempts++;
        if (jvpSeq !== launch) { clearInterval(poll); return; }
        try {
          const st = await getJvpProbe(res.probe_id);
          if (jvpSeq !== launch) { clearInterval(poll); return; }
          const result = st.result;
          if (st.status === 'done' && result) {
            clearInterval(poll);
            set((s) => ({
              // re-probing a point replaces its segment rather than stacking a
              // second one on top of it
              probes: [...s.probes.filter(p => !samePoint(p, alpha, beta)),
                       { alpha, beta, result }],
              jvpStatus: 'done', jvpError: null, jvpPoll: null,
              pendingId: null, pendingAt: null,
            }));
            return;
          }
          if (st.status === 'error') {
            clearInterval(poll);
            fail(st.error || 'the probe failed on the server');
            return;
          }
        } catch (err) {
          // transient read failure (a backend restart takes ~3 min): keep polling,
          // but the attempt counted against the bound
          console.error('[Probe] JVP poll error:', err);
        }
        if (attempts >= JVP_MAX_POLLS) {
          clearInterval(poll);
          fail('probe stalled — stopped polling');
        }
      }, POLL_MS);
      set({ jvpPoll: poll });
    } catch (err) {
      fail(String(err));
    }
  },

  clearToken: () => {
    tokenSeq++;
    const { tokenPoll } = get();
    if (tokenPoll) clearInterval(tokenPoll);
    set({ token: IDLE_TOKEN, tokenId: null, tokenPoll: null });
  },

  startToken: async (jobId, which, seed) => {
    const launch = ++tokenSeq;
    const { tokenPoll } = get();
    if (tokenPoll) clearInterval(tokenPoll);
    set({ tokenPoll: null, tokenId: null,
          token: { which, status: 'running', result: null, error: null } });

    const fail = (msg: string) => {
      if (tokenSeq !== launch) return;
      set({ token: { which, status: 'error', result: null, error: msg },
            tokenPoll: null, tokenId: null });
    };

    try {
      const res = await startTokenProbe(jobId, { which, seed });
      if (tokenSeq !== launch) return;
      if (res.status !== 'running' || !res.probe_id) {
        fail(res.error || 'the server would not start a token probe — it may no longer have this job');
        return;
      }
      set({ tokenId: res.probe_id });

      let attempts = 0;
      const poll = setInterval(async () => {
        attempts++;
        if (tokenSeq !== launch) { clearInterval(poll); return; }
        try {
          const st = await getTokenProbe(res.probe_id);
          if (tokenSeq !== launch) { clearInterval(poll); return; }
          const result = st.result;
          if (st.status === 'done' && result) {
            clearInterval(poll);
            set({ token: { which, status: 'done', result, error: null },
                  tokenPoll: null, tokenId: null });
            return;
          }
          if (st.status === 'error') {
            clearInterval(poll);
            fail(st.error || 'the token probe failed on the server');
            return;
          }
        } catch (err) {
          console.error('[Probe] token poll error:', err);
        }
        if (attempts >= TOKEN_MAX_POLLS) {
          clearInterval(poll);
          fail('token probe stalled — stopped polling');
        }
      }, POLL_MS);
      set({ tokenPoll: poll });
    } catch (err) {
      fail(String(err));
    }
  },

  // ---- cascade ------------------------------------------------------------
  // Called from CascadeMap whenever its runId changes. Crossing ids are per run, so
  // the whole map is dropped rather than carried: a stale cid would otherwise draw a
  // measured direction on a crossing of a different survey.
  syncCascadeRun: (runId) => {
    if (get().cascadeRunId === runId) return;
    cascadeSeq++;
    localSvSeq++;
    const { cascadePoll } = get();
    if (cascadePoll) clearInterval(cascadePoll);
    // The density map is keyed by chord, and chords belong to one survey -- carrying it
    // over would paint this run's lines with the last one's readings.
    set({ cascadeRunId: runId, cascadeProbes: {}, cascadePendingCid: null,
          cascadeStatus: 'idle', cascadeError: null, cascadePoll: null,
          localSv: {}, localSvStatus: 'idle', localSvError: null });
  },

  clearCascadeProbes: () => {
    cascadeSeq++;
    const { cascadePoll } = get();
    if (cascadePoll) clearInterval(cascadePoll);
    set({ cascadeProbes: {}, cascadePendingCid: null,
          cascadeStatus: 'idle', cascadeError: null, cascadePoll: null });
  },

  startCascadeJvp: async (runId, cid, seed) => {
    const launch = ++cascadeSeq;
    const { cascadePoll } = get();
    if (cascadePoll) clearInterval(cascadePoll);
    set({ cascadePoll: null, cascadeStatus: 'running', cascadeError: null,
          cascadePendingCid: cid });

    const fail = (msg: string) => {
      if (cascadeSeq !== launch) return;
      set({ cascadeStatus: 'error', cascadeError: msg, cascadePoll: null,
            cascadePendingCid: null });
    };

    try {
      const res = await startCascadeJvpProbe(runId, { cid, seed });
      if (cascadeSeq !== launch) return;   // superseded while the POST was in flight
      if (res.status !== 'running' || !res.probe_id) {
        fail(res.error || 'the server refused this probe — the run may no longer be '
                        + 'on the server');
        return;
      }

      let attempts = 0;
      const poll = setInterval(async () => {
        attempts++;
        if (cascadeSeq !== launch) { clearInterval(poll); return; }
        try {
          const st = await getJvpProbe(res.probe_id);
          if (cascadeSeq !== launch) { clearInterval(poll); return; }
          const result = st.result;
          if (st.status === 'done' && result) {
            clearInterval(poll);
            set((s) => ({
              // file under the run that was live at LAUNCH, not at completion: the
              // seq guard above already dropped the result if the run changed
              cascadeProbes: {
                ...s.cascadeProbes,
                [runId]: { ...(s.cascadeProbes[runId] ?? {}), [cid]: result },
              },
              cascadeStatus: 'done', cascadeError: null,
              cascadePoll: null, cascadePendingCid: null,
            }));
            return;
          }
          if (st.status === 'error') {
            clearInterval(poll);
            fail(st.error || 'the probe failed on the server');
            return;
          }
        } catch (err) {
          // transient read failure (a backend restart takes ~3 min): keep polling,
          // but the attempt counted against the bound
          console.error('[Probe] cascade JVP poll error:', err);
        }
        if (attempts >= CASCADE_MAX_POLLS) {
          clearInterval(poll);
          fail('probe stalled — stopped polling');
        }
      }, POLL_MS);
      set({ cascadePoll: poll });
    } catch (err) {
      fail(String(err));
    }
  },

  // ---- boundary density ---------------------------------------------------
  // One GET, no GPU: the endpoint re-reads geometry the survey already paid for. Turning
  // the overlay on fetches it once; the read mode (ranking vs calibrated) is a view of the
  // SAME response, so switching it never refetches.
  setLocalSvOn: (runId, on) => {
    set({ localSvOn: on });
    if (on && runId && !get().localSv[runId]) void get().fetchLocalSv(runId);
  },

  setLocalSvView: (view) => set({ localSvView: view }),

  fetchLocalSv: async (runId) => {
    const launch = ++localSvSeq;
    set({ localSvStatus: 'running', localSvError: null });
    try {
      const m = await cascadeLocalSv(runId);
      if (localSvSeq !== launch) return;      // superseded (run changed, or a second toggle)
      set((s) => ({ localSv: { ...s.localSv, [runId]: m },
                    localSvStatus: 'done', localSvError: null }));
    } catch (err: any) {
      if (localSvSeq !== launch) return;
      // 404 while the survey is still laying down chords is the normal early answer, not a
      // fault: say so rather than showing a raw HTTP line.
      const msg = String(err?.message ?? err);
      set({ localSvStatus: 'error',
            localSvError: msg.includes('404')
              ? 'no chords yet — the density map appears once the survey has laid some down'
              : msg });
    }
  },
}));

function samePoint(p: JvpProbeEntry, alpha: number, beta: number): boolean {
  return Math.abs(p.alpha - alpha) < 1e-6 && Math.abs(p.beta - beta) < 1e-6;
}

/** Exported for the viewport: is there a finished probe at this cell? */
export function probeAt(probes: JvpProbeEntry[], alpha: number, beta: number): JvpProbeEntry | undefined {
  return probes.find(p => samePoint(p, alpha, beta));
}

/**
 * The one-line label a probe gets on the map. `participation_ratio` near 1 means the
 * local change is one front; near 2 means two fronts meet, and a single direction is
 * then a poor summary — which the label has to say, because the segment alone looks
 * just as confident either way.
 */
export function probeLabel(r: JvpProbeCore): string {
  if (r.participation_ratio >= 1.5) return `corner PR ${r.participation_ratio.toFixed(1)}`;
  return `front ${r.rank1_share.toFixed(2)}`;
}

/** Codimension in one word, on the same 1.5 threshold the label uses. */
export function probeCodim(r: JvpProbeCore): 'front' | 'corner' {
  return r.participation_ratio < 1.5 ? 'front' : 'corner';
}
