/**
 * How chord / line probes read their label -- the ONE place the default lives.
 *
 * Both the Cascade panel and the Mixing desk start from PROBE_DEFAULTS and always send the
 * mode explicitly, so flipping the default after the h27 study (search_problem/outputs/
 * h27_staged_probes/PREREG.md) is a one-line change here and a frontend rebuild -- no backend
 * restart. The backend's own request default stays "steps" (unchanged behaviour for any other
 * client).
 *
 *   'steps'  -- today's probes: a complete 4-step-schedule image per probe (probe_steps = 4).
 *   'staged' -- the full 8-step schedule up to step t, DINOv2 of the PREDICTED clean image x̂0;
 *               segments whose two readouts are >= theta apart (cosine distance) are resumed
 *               from the cached latent to the full image, so their labels are exact. Crossings
 *               are only ever claimed between two finished probes. theta = 0.10 is provisional
 *               until the h27 calibration replaces it.
 */
export type ProbeMode = 'steps' | 'staged';

export interface ProbeDefaults {
  mode: ProbeMode;
  /** readout step t of the full schedule (1 .. steps - 1) */
  stagedT: number;
  /** flag threshold on the x̂0 cosine distance between neighbouring probes */
  stagedTheta: number;
}

export const PROBE_DEFAULTS: ProbeDefaults = {
  mode: 'steps',
  stagedT: 4,
  stagedTheta: 0.10,
};

export const PROBE_MODES: { value: ProbeMode; label: string }[] = [
  { value: 'steps', label: '4-step probes (current)' },
  { value: 'staged', label: 'Staged readout — exact labels where flagged' },
];

/** The one-line explanation both panels show under the advanced settings. */
export const STAGED_EXPLAINER =
  'Staged: each probe runs the full schedule to step t and reads the predicted image; only '
  + 'neighbours whose readouts differ by ≥ θ are finished (exact full-fidelity labels), and '
  + 'crossings are claimed between finished probes only. θ is provisional (h27).';

/** Nominal image-eq of a staged probe: readout t/S, plus (S - t)/S when it is finished. */
export function stagedProbeCost(t: number, steps: number, resumedShare: number): number {
  return t / steps + resumedShare * (steps - t) / steps;
}

/** Clamp a user-typed t into the legal range 1 .. steps - 1. */
export function clampStagedT(t: number, steps: number): number {
  if (!Number.isFinite(t)) return PROBE_DEFAULTS.stagedT;
  return Math.max(1, Math.min(Math.max(1, steps - 1), Math.round(t)));
}

/** Clamp a user-typed theta into (0, 2). */
export function clampStagedTheta(theta: number): number {
  if (!Number.isFinite(theta) || theta <= 0) return PROBE_DEFAULTS.stagedTheta;
  return Math.min(1.99, theta);
}
