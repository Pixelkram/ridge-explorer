/**
 * Which surfaces the UI shows.
 *
 * The tool has accreted research surfaces — hikers, cascade, discovery, the ridge
 * itinerary, MF scan, surprise sampling — on top of the original loop the README
 * describes: pick prompts, scan, threshold on τ, generate the selected cells, refine.
 * Demoing that original loop means getting the later panels off screen without
 * deleting them, so every added surface is gated here and defaults to visible.
 *
 * Two sources. The env var is read at dev-server start; the URL query overrides it per
 * tab, so a running server can be re-framed between takes without a restart:
 *
 *   VITE_RIDGE_HIDE=og npm run dev          # every take starts on the OG surface
 *   ?hide=og                                # same, for this tab only
 *   ?hide=hikers,cascade                    # hand-picked
 *   ?hide=og&show=itinerary                 # preset minus one
 *   ?hide=none                              # everything back, even if the env hid it
 *
 * Unknown names are ignored with a console warning rather than throwing: a typo mid-
 * recording should cost a panel, not the whole page.
 */

export type Feature =
  | 'hikers'      // HikePanel — population ridge-walk across chained simplices
  | 'cascade'     // CascadePanel + CascadeMap
  | 'amr'         // AmrPanel + AmrMap — octree/AMR adaptive refinement (k<=4)
  | 'metro'       // MetroPanel + MetroMap — Metropolis sampler on the sharpness field
  | 'discovery'   // DiscoverPanel
  | 'itinerary'   // ItineraryPanel — ridge graph as walkable arcs
  | 'mfscan'      // the MF Scan button (multi-fidelity GP detection)
  | 'surprise'    // SurpriseSlider inside the scan-complete bar
  | 'explore'     // the Explore button (full DINOv2 pass)
  | 'fastscan'    // the Fast Scan button (1-step Jacobian)
  | 'refine'      // RefinePanel — hierarchical subdivision
  | 'seeds'       // the multi-seed count selector
  | 'threed';     // the 2D/3D mode selector

export const ALL_FEATURES: Feature[] = [
  'hikers', 'cascade', 'amr', 'metro', 'discovery', 'itinerary', 'mfscan',
  'surprise', 'explore', 'fastscan', 'refine', 'seeds', 'threed',
];

/** Spellings that reach for the same panel. Keys are already normalised. */
const ALIASES: Record<string, Feature> = {
  hiker: 'hikers', hike: 'hikers', hiking: 'hikers',
  discover: 'discovery', discoverpanel: 'discovery',
  cascadepanel: 'cascade',
  amrpanel: 'amr', octree: 'amr', refinement: 'amr', adaptive: 'amr',
  metropanel: 'metro', mcmc: 'metro', metropolis: 'metro', sampler: 'metro',
  mf: 'mfscan', mfscan: 'mfscan',
  ridgeitinerary: 'itinerary', arcs: 'itinerary',
  surpriseslider: 'surprise',
  fast: 'fastscan',
  '3d': 'threed', '2d3d': 'threed', mode: 'threed',
  seedcount: 'seeds',
};

/**
 * `og` is the README feature set: everything committed before the research surfaces
 * landed. It deliberately keeps Explore, Fast Scan, τ/generate and Refine — those are
 * the original tool. Add `explore` to the list by hand if a take wants Fast Scan alone.
 */
const PRESETS: Record<string, Feature[]> = {
  og: ['hikers', 'cascade', 'amr', 'metro', 'discovery', 'itinerary', 'mfscan',
       'surprise'],
  none: [],
  all: ALL_FEATURES,
};

function normalise(token: string): string {
  return token.trim().toLowerCase().replace(/[\s_-]/g, '');
}

/** Comma-separated features and preset names → the set of features they name. */
export function parseSpec(spec: string | null | undefined): Set<Feature> {
  const out = new Set<Feature>();
  if (!spec) return out;
  for (const raw of spec.split(',')) {
    const token = normalise(raw);
    if (!token) continue;
    if (token in PRESETS) { PRESETS[token].forEach(f => out.add(f)); continue; }
    let resolved: Feature | undefined = ALIASES[token];
    if (!resolved && (ALL_FEATURES as string[]).includes(token)) resolved = token as Feature;
    if (resolved) out.add(resolved);
    else console.warn(`[RidgeExplorer] unknown feature "${raw.trim()}" — ignored. `
      + `Known: ${ALL_FEATURES.join(', ')} + presets ${Object.keys(PRESETS).join(', ')}`);
  }
  return out;
}

/** Exported so the test can drive it with explicit inputs instead of a fake location. */
export function resolveHidden(search: string, envSpec: string | undefined): Set<Feature> {
  const params = new URLSearchParams(search);
  const urlHide = params.get('hide');
  // `?hide=` present but empty means "show everything", which is why this tests for
  // null rather than falsiness — otherwise an empty param would silently fall back to
  // the env var it was typed to override.
  const hidden = parseSpec(urlHide !== null ? urlHide : envSpec);
  for (const f of parseSpec(params.get('show'))) hidden.delete(f);
  return hidden;
}

// Optional chaining on both: outside a Vite build (the node test imports this file
// transpiled) there is no import.meta.env and no window, and neither absence should
// throw on import.
const hidden = resolveHidden(
  typeof window === 'undefined' ? '' : window.location.search,
  import.meta.env?.VITE_RIDGE_HIDE,
);

/** True when the surface should render. */
export function feature(f: Feature): boolean {
  return !hidden.has(f);
}

export const hiddenFeatures: ReadonlySet<Feature> = hidden;

// Recording sessions run without devtools open often enough that the resolved set is
// worth one line in the console and a handle on window to poke at.
if (typeof window !== 'undefined') {
  (window as any).__ridgeFeatures = {
    hidden: [...hidden], all: ALL_FEATURES, presets: PRESETS,
  };
  if (hidden.size) {
    console.info(`[RidgeExplorer] hiding: ${[...hidden].sort().join(', ')} `
      + `(?hide=none to restore)`);
  }
}
