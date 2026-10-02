// Runs the REAL view arithmetic out of src/viewTransform.ts.
//
// Same rule as features.test.mjs: test the shipped source, not a paraphrase. esbuild is
// already present as a Vite dependency, so the TS is transpiled and imported rather than
// re-implemented here -- a constant that drifts in the source fails this test.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';
import { transform } from 'esbuild';

const SRC = new URL('../src/viewTransform.ts', import.meta.url).pathname;
const { code } = await transform(fs.readFileSync(SRC, 'utf8'), {
  loader: 'ts', format: 'esm', target: 'es2020',
});
const tmp = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'ridge-view-')), 'view.mjs');
fs.writeFileSync(tmp, code);
const {
  HOME, MIN_SCALE, MAX_SCALE, ZOOM_STEP, DRAG_PX, DEG_PER_PX,
  clamp, dragFactor, isDrag, isHome, pan, reset, rotationDelta, svgTransform,
  toScreen, toWorld, wheelFactor, wrapAngle, zoomAbout,
} = await import(tmp);

let failures = 0;
function check(name, actual, expected) {
  try {
    assert.deepEqual(actual, expected);
    console.log(`  ok   ${name}`);
  } catch {
    failures++;
    console.log(`  FAIL ${name}\n       got      ${JSON.stringify(actual)}`
      + `\n       expected ${JSON.stringify(expected)}`);
  }
}
function near(name, actual, expected, tol = 1e-9) {
  const ok = Array.isArray(expected)
    ? expected.length === actual.length
      && expected.every((e, i) => Math.abs(actual[i] - e) <= tol)
    : Math.abs(actual - expected) <= tol;
  if (ok) {
    console.log(`  ok   ${name}`);
  } else {
    failures++;
    console.log(`  FAIL ${name}\n       got      ${JSON.stringify(actual)}`
      + `\n       expected ${JSON.stringify(expected)} (±${tol})`);
  }
}

console.log('viewTransform');

// --- the invariant the whole gesture rests on: the anchor does not move ---------------
// Any view, any factor, any anchor: the world point under the anchor is the same before
// and after. Checked across a grid rather than at one convenient point.
for (const v of [HOME, { tx: 0, ty: 0, s: 1 }, { tx: -140, ty: 37, s: 2.5 },
                 { tx: 311, ty: -92, s: 0.4 }]) {
  for (const factor of [1, 1.1, 1 / 1.1, 3, 0.37]) {
    for (const [px, py] of [[0, 0], [170, 170], [339, 12], [-25, 400]]) {
      const before = toWorld(v, px, py);
      const after = toWorld(zoomAbout(v, factor, px, py), px, py);
      near(`anchor fixed: s=${v.s} ×${factor} at (${px},${py})`, after, before, 1e-9);
    }
  }
}

// round trip: screen -> world -> screen
const V = { tx: -140, ty: 37, s: 2.5 };
near('toScreen ∘ toWorld = id', toScreen(V, ...toWorld(V, 83, -11)), [83, -11], 1e-9);

// --- zoom composition and clamping ----------------------------------------------------
near('zoom scales by the factor', zoomAbout(HOME, 1.1, 10, 10).s, 1.1);
near('zoom about the origin leaves the origin alone',
  [zoomAbout(HOME, 4, 0, 0).tx, zoomAbout(HOME, 4, 0, 0).ty], [0, 0]);
near('two notches compose',
  zoomAbout(zoomAbout(HOME, ZOOM_STEP, 50, 50), ZOOM_STEP, 50, 50).s, ZOOM_STEP ** 2);
check('a notch in and back out returns home',
  isHome(zoomAbout(zoomAbout(HOME, ZOOM_STEP, 50, 50), 1 / ZOOM_STEP, 50, 50)), true);

near('zoom clamps at MAX', zoomAbout({ tx: 0, ty: 0, s: 19 }, 100, 5, 5).s, MAX_SCALE);
near('zoom clamps at MIN', zoomAbout({ tx: 0, ty: 0, s: 0.3 }, 0.01, 5, 5).s, MIN_SCALE);
// the clamped notch must still keep its anchor: the factor is shortened, not dropped
{
  const v = { tx: 12, ty: -8, s: 19 };
  const got = toWorld(zoomAbout(v, 100, 77, 123), 77, 123);
  near('clamped zoom still pins the anchor', got, toWorld(v, 77, 123), 1e-9);
}
near('clamp passes a scale in range', clamp(3.5), 3.5);
near('clamp floors', clamp(0), MIN_SCALE);
near('clamp ceils', clamp(1e9), MAX_SCALE);
near('clamp rejects NaN', clamp(NaN), 1);
near('a non-finite factor is a no-op', zoomAbout(V, NaN, 9, 9).s, V.s);
near('a zero factor is a no-op', zoomAbout(V, 0, 9, 9).s, V.s);

// --- pan -------------------------------------------------------------------------------
check('pan adds screen px, leaving the zoom alone', pan(V, 10, -4),
  { tx: -130, ty: 33, s: 2.5 });
check('pan is reversible', pan(pan(V, 10, -4), -10, 4), V);
// panning by one screen px moves the drawing one screen px at every zoom
near('pan is in screen px, not world px',
  toScreen(pan(V, 7, 0), 0, 0)[0] - toScreen(V, 0, 0)[0], 7);

// --- reset -----------------------------------------------------------------------------
check('reset is the home view', reset(), { tx: 0, ty: 0, s: 1 });
check('reset is home', isHome(reset()), true);
check('a panned, zoomed view is not home', isHome(V), false);
check('a pure pan is not home', isHome(pan(HOME, 3, 0)), false);
check('a pure zoom is not home', isHome(zoomAbout(HOME, 2, 0, 0)), false);
// isHome is tolerant to round-off, but not to anything a viewer could see
check('a sub-pixel pan still reads as home', isHome(pan(HOME, 0.001, 0)), true);
check('a one-pixel pan does not', isHome(pan(HOME, 1, 0)), false);

// --- wheel: deltaMode-aware, exponential ----------------------------------------------
near('one pixel-mode notch up zooms in by the step', wheelFactor(-100, 0), ZOOM_STEP);
near('one pixel-mode notch down zooms out by the step',
  wheelFactor(100, 0), 1 / ZOOM_STEP);
near('wheel is exponential, so two notches = step²', wheelFactor(-200, 0), ZOOM_STEP ** 2);
near('line mode counts 16 px a line', wheelFactor(-3, 1), ZOOM_STEP ** (48 / 100));
near('page mode counts 400 px a page', wheelFactor(-1, 2), ZOOM_STEP ** 4);
near('a zero delta is a no-op', wheelFactor(0, 0), 1);
near('a non-finite delta is a no-op', wheelFactor(NaN, 0), 1);

// --- alt+right drag -------------------------------------------------------------------
near('drag up 160 px doubles', dragFactor(-160), 2);
near('drag down 160 px halves', dragFactor(160), 0.5);
near('no drag, no zoom', dragFactor(0), 1);

// --- the attribute the SVG group actually gets -----------------------------------------
check('svgTransform', svgTransform({ tx: -1.5, ty: 2, s: 3 }),
  'translate(-1.5,2) scale(3)');
check('home transform is the identity', svgTransform(HOME), 'translate(0,0) scale(1)');

// --- left drag -> shadow-plane rotation -----------------------------------------------
near(`one px is ${DEG_PER_PX}°`, rotationDelta(1), (DEG_PER_PX * Math.PI) / 180);
near('a 180 px drag is 90°', rotationDelta(180), Math.PI / 2);
near('dragging left turns the other way', rotationDelta(-180), -Math.PI / 2);
near('rotation is linear in the drag', rotationDelta(40) * 3, rotationDelta(120), 1e-12);
near('no drag, no rotation', rotationDelta(0), 0);
near('a non-finite drag is a no-op', rotationDelta(NaN), 0);
near('sensitivity is overridable', rotationDelta(10, 1), (10 * Math.PI) / 180);

// --- click vs drag ---------------------------------------------------------------------
check(`a still press is a click (<${DRAG_PX} px)`, isDrag(0, 0), false);
check('3 px of jitter is still a click', isDrag(3, 0), false);
check('2+2 px diagonal is still a click', isDrag(2, 2), false);
check(`${DRAG_PX} px is a drag`, isDrag(DRAG_PX, 0), true);
check('a drag counts diagonally, not per axis', isDrag(3, 3), true);
check('direction does not matter', isDrag(-9, 0), true);
check('the threshold is overridable', isDrag(5, 0, 20), false);

// --- angle wrapping -------------------------------------------------------------------
near('an in-range angle is left alone', wrapAngle(1.5), 1.5);
near('a negative angle wraps up', wrapAngle(-0.5), Math.PI * 2 - 0.5);
near('a big negative drag wraps too', wrapAngle(-7 * Math.PI), Math.PI, 1e-12);
near('tau is zero', wrapAngle(Math.PI * 2), 0, 1e-12);
near('a non-finite angle is zero', wrapAngle(NaN), 0);

fs.rmSync(path.dirname(tmp), { recursive: true, force: true });
console.log(failures ? `\n${failures} failing` : '\nall passing');
process.exit(failures ? 1 : 0);
