/**
 * metroUtil -- the Metropolis panel's pure helpers, in a module of their own.
 *
 * Nothing here renders. The reason it is a separate file is Fast Refresh: a component file that
 * also exports a plain function cannot be hot-swapped (vite: "Could not Fast Refresh
 * (\"paramChangeRounds\" export is incompatible)"), so every edit to MetroLive.tsx or
 * MetroMap.tsx used to invalidate the whole module graph and reload the page -- which, on a
 * panel whose whole point is watching a run that takes minutes, throws away the run's view.
 * Component files export components; these three live here.
 *
 *   * `SparkPoint` / `SparkMark` -- the Sparkline's two data shapes (types, so they cost
 *     nothing at runtime, but they belong beside the function that builds the marks);
 *   * `paramChangeRounds` -- the kernel-change marks, read off the round log;
 *   * `baryFromPoint` -- the k=3 inverse of the map's projection.
 */
import type { MetroRoundLog } from './api/types';

export interface SparkPoint {
  x: number;
  /** null breaks the line rather than being drawn as a zero */
  y: number | null;
}

export interface SparkMark {
  x: number;
  title: string;
}

/**
 * The rounds at which the kernel differs from the round before -- the backend's own rule
 * (`metro.param_change_rounds`), recomputed here from the same log it reads.
 *
 * One source of truth: every row carries the sigma/beta it ran under, so a change is a property
 * of the log rather than a second list beside it that could disagree.
 */
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
 * Barycentric coordinates of a point in the k=3 triangle, or null if the triangle is
 * degenerate. `vs` must be the vertices in PROMPT order, which is how the caller builds them,
 * so the result is a recipe over the same prompts and not a permutation of one.
 *
 * k=3 only on purpose. For k>=4 the map is a 2-D shadow of a (k-1)-dimensional space: a pixel
 * is a whole fibre of recipes, so there is no inverse to compute and a click cannot mean one
 * recipe. The panel offers the two unambiguous seeds instead (an existing sample, or an
 * existing crossing), which is the honest version of the same gesture.
 */
export function baryFromPoint(
  vs: readonly (readonly [number, number])[], x: number, y: number,
): number[] | null {
  if (vs.length < 3) return null;
  const [[x1, y1], [x2, y2], [x3, y3]] = vs;
  const det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
  if (!Number.isFinite(det) || Math.abs(det) < 1e-9) return null;
  const a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det;
  const b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det;
  const w = [a, b, 1 - a - b];
  return w.every((v) => Number.isFinite(v)) ? w : null;
}
