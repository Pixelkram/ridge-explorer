/**
 * viewTransform -- the pan/zoom/rotate arithmetic of the cascade map.
 *
 * The view is one uniform 2-D similarity sitting ON TOP of the simplex projection:
 *
 *     screen = (tx, ty) + s · world
 *
 * which is exactly the SVG `translate(tx,ty) scale(s)` the map wraps its drawn layers
 * in. Keeping it separate from `makeProjector` is the point: the shadow plane keeps
 * rotating, the Helmert anchor keeps changing, and neither touches the view -- a
 * re-projection composes with the view instead of resetting it.
 *
 * The drag gestures that FEED those two (rotation angle, click-vs-drag) live here too,
 * for the same reason: they are arithmetic, and arithmetic is cheaper to check than a
 * DOM. Everything is pure; `test/viewTransform.test.mjs` runs this source.
 */

/** tx, ty in screen px; s is the uniform zoom (1 = the projection's own scale). */
export interface View {
  tx: number;
  ty: number;
  s: number;
}

export const MIN_SCALE = 0.25;
export const MAX_SCALE = 20;
/** one wheel notch, exponential so zooming in and back out returns where it started */
export const ZOOM_STEP = 1.1;

export const HOME: View = { tx: 0, ty: 0, s: 1 };

/** The home view: pan 0, zoom 1 -- the projection exactly as `makeProjector` laid it. */
export function reset(): View {
  return HOME;
}

export function clamp(s: number): number {
  if (!Number.isFinite(s)) return 1;
  return Math.min(MAX_SCALE, Math.max(MIN_SCALE, s));
}

/**
 * At home to within round-off. Not an exact comparison on purpose: a wheel notch in and
 * back out is ×1.1 then ×(1/1.1), which lands 2e-16 away from 1, and a readout that then
 * says "1.00×" forever instead of nothing would be reporting arithmetic, not a view.
 * The translation tolerance is far below one screen pixel.
 */
export function isHome(v: View): boolean {
  return Math.abs(v.s - 1) < 1e-9 && Math.abs(v.tx) < 0.01 && Math.abs(v.ty) < 0.01;
}

export function pan(v: View, dx: number, dy: number): View {
  return { tx: v.tx + dx, ty: v.ty + dy, s: v.s };
}

/**
 * Zoom by `factor` about the screen point (px, py), which keeps the world coordinate it
 * was over -- the map grows out of the cursor rather than out of the origin. Clamping
 * shortens the factor and re-solves the translation, so the anchor stays fixed even on
 * the notch that hits a limit.
 */
export function zoomAbout(v: View, factor: number, px: number, py: number): View {
  const want = Number.isFinite(factor) && factor > 0 ? factor : 1;
  const s = clamp(v.s * want);
  const f = s / v.s;
  return { tx: px - (px - v.tx) * f, ty: py - (py - v.ty) * f, s };
}

/** world (projection px) -> screen px; for the layers drawn outside the scaled group. */
export function toScreen(v: View, x: number, y: number): readonly [number, number] {
  return [v.tx + x * v.s, v.ty + y * v.s] as const;
}

/** screen px -> world (projection px); the inverse of the group's transform. */
export function toWorld(v: View, px: number, py: number): readonly [number, number] {
  return [(px - v.tx) / v.s, (py - v.ty) / v.s] as const;
}

export function svgTransform(v: View): string {
  return `translate(${v.tx},${v.ty}) scale(${v.s})`;
}

/**
 * Wheel delta -> zoom factor, deltaMode-aware (0 px, 1 lines, 2 pages): a line-mode
 * wheel reports ~3, a pixel-mode one ~100 for the same physical notch, so taking deltaY
 * raw would make Firefox's wheel 30× weaker than Chrome's.
 */
export function wheelFactor(deltaY: number, deltaMode = 0, step = ZOOM_STEP): number {
  if (!Number.isFinite(deltaY)) return 1;
  const px = deltaMode === 1 ? deltaY * 16 : deltaMode === 2 ? deltaY * 400 : deltaY;
  return Math.pow(step, -px / 100);
}

/**
 * Vertical drag distance -> zoom factor for the Alt+right-drag gesture: drag up to zoom
 * in, 160 px per doubling. Always applied to the view the drag STARTED from, so the
 * gesture is absolute (no drift) and the drag-start point stays pinned throughout.
 */
export function dragFactor(dy: number): number {
  if (!Number.isFinite(dy)) return 1;
  return Math.pow(2, -dy / 160);
}

/** Below this much pointer travel a press-and-release is still a click, not a drag. */
export const DRAG_PX = 4;
/** Left-drag sensitivity: degrees of shadow-plane rotation per screen pixel. */
export const DEG_PER_PX = 0.5;

/**
 * Has the pointer travelled far enough to be a drag? Under the threshold the press is
 * left alone to become a click, so dragging the shadow plane and selecting a crossing
 * share the left button without either stealing from the other.
 */
export function isDrag(dx: number, dy: number, threshold = DRAG_PX): boolean {
  return Math.hypot(dx, dy) >= threshold;
}

/**
 * Horizontal drag distance -> shadow-plane rotation, in radians. Added to the angle the
 * drag STARTED from, so the gesture is absolute and composes with (never resets) the
 * pan/zoom view, which is a separate transform entirely.
 */
export function rotationDelta(dx: number, degPerPx = DEG_PER_PX): number {
  if (!Number.isFinite(dx)) return 0;
  return (dx * degPerPx * Math.PI) / 180;
}

/** An angle wrapped into [0, 2π): a left-drag can run negative, `%` alone cannot. */
export function wrapAngle(t: number): number {
  if (!Number.isFinite(t)) return 0;
  const tau = Math.PI * 2;
  return ((t % tau) + tau) % tau;
}
