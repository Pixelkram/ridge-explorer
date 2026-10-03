// Runs the REAL probe-mode defaults out of src/probeDefaults.ts (transpiled, as features.test.mjs
// does), and checks that both panels take their initial probe mode from that ONE constant --
// so flipping the default after h27 is a one-line change there and nowhere else.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';
import { transform } from 'esbuild';

const SRC = new URL('../src/probeDefaults.ts', import.meta.url).pathname;
const { code } = await transform(fs.readFileSync(SRC, 'utf8'), {
  loader: 'ts', format: 'esm', target: 'es2020',
});
const tmp = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'ridge-probe-')), 'probe.mjs');
fs.writeFileSync(tmp, code);
const { PROBE_DEFAULTS, PROBE_MODES, STAGED_EXPLAINER, stagedProbeCost, clampStagedT,
  clampStagedTheta } = await import(tmp);

let failures = 0;
function check(name, actual, expected) {
  try {
    assert.deepEqual(actual, expected);
    console.log(`  ok   ${name}`);
  } catch (e) {
    failures++;
    console.log(`  FAIL ${name}\n       got      ${JSON.stringify(actual)}`
      + `\n       expected ${JSON.stringify(expected)}`);
  }
}

console.log('probeDefaults');
const modes = PROBE_MODES.map((m) => m.value);
check('the two modes, in order', modes, ['steps', 'staged']);
check('labels as specified', PROBE_MODES.map((m) => m.label),
  ['4-step probes (current)', 'Staged readout — exact labels where flagged']);
check('the default mode is one of them', modes.includes(PROBE_DEFAULTS.mode), true);
check('default t is an integer in 1..7 of the 8-step schedule',
  Number.isInteger(PROBE_DEFAULTS.stagedT) && PROBE_DEFAULTS.stagedT >= 1
    && PROBE_DEFAULTS.stagedT <= 7, true);
check('default theta lies in (0, 2) (the backend bound)',
  PROBE_DEFAULTS.stagedTheta > 0 && PROBE_DEFAULTS.stagedTheta < 2, true);
check('one-line explanation present', STAGED_EXPLAINER.length > 40 && !STAGED_EXPLAINER.includes('\n'),
  true);
check('nominal cost: t/S + share (S-t)/S', stagedProbeCost(4, 8, 0.5), 0.75);
check('clamp t into 1..S-1', [clampStagedT(0, 8), clampStagedT(9, 8), clampStagedT(3.4, 8),
  clampStagedT(NaN, 8)], [1, 7, 3, PROBE_DEFAULTS.stagedT]);
check('clamp theta into (0, 2)', [clampStagedTheta(-1), clampStagedTheta(5), clampStagedTheta(0.2)],
  [PROBE_DEFAULTS.stagedTheta, 1.99, 0.2]);

// both panels must start from the constant, never from a literal of their own
for (const panel of ['CascadePanel.tsx', 'DeskPanel.tsx']) {
  const src = fs.readFileSync(new URL(`../src/${panel}`, import.meta.url), 'utf8');
  check(`${panel} initialises probe mode from PROBE_DEFAULTS`,
    /useState<ProbeMode>\(PROBE_DEFAULTS\.mode\)/.test(src)
      && /useState\(PROBE_DEFAULTS\.stagedT\)/.test(src)
      && /useState\(PROBE_DEFAULTS\.stagedTheta\)/.test(src), true);
  check(`${panel} sends the mode with every start`, /probe_mode: probeMode/.test(src), true);
}

fs.rmSync(path.dirname(tmp), { recursive: true, force: true });
console.log(failures ? `\n${failures} failing` : '\nall passing');
process.exit(failures ? 1 : 0);
