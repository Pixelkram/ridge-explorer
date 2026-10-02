// Runs the REAL flag resolution out of src/featureFlags.ts.
//
// Same rule as lineage.test.mjs: test the shipped source, not a paraphrase. esbuild is
// already present as a Vite dependency, so the TS is transpiled and imported rather
// than re-implemented here — a preset that drifts in the source fails this test.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';
import { transform } from 'esbuild';

const SRC = new URL('../src/featureFlags.ts', import.meta.url).pathname;
const { code } = await transform(fs.readFileSync(SRC, 'utf8'), {
  loader: 'ts', format: 'esm', target: 'es2020',
});
const tmp = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'ridge-flags-')), 'flags.mjs');
fs.writeFileSync(tmp, code);
const { resolveHidden, parseSpec, ALL_FEATURES } = await import(tmp);

const set = (s, env) => [...resolveHidden(s, env)].sort();
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

const OG = ['amr', 'cascade', 'discovery', 'hikers', 'itinerary', 'metro', 'mfscan',
  'surprise'];

console.log('featureFlags');
check('nothing set → nothing hidden', set('', undefined), []);
check('env preset', set('', 'og'), OG);
check('url preset', set('?hide=og', undefined), OG);
check('url overrides env', set('?hide=cascade', 'og'), ['cascade']);
check('?hide=none beats env preset', set('?hide=none', 'og'), []);
check('?hide= (empty) beats env preset', set('?hide=', 'og'), []);
check('show subtracts from preset', set('?hide=og&show=itinerary', undefined),
  OG.filter(f => f !== 'itinerary'));
check('show subtracts from env preset', set('?show=hikers,cascade', 'og'),
  OG.filter(f => f !== 'hikers' && f !== 'cascade'));
check('hand-picked list', set('?hide=hikers,cascade,discovery', undefined),
  ['cascade', 'discovery', 'hikers']);
check('aliases normalise', set('?hide=Hiker, MF-Scan ,discover,3d', undefined),
  ['discovery', 'hikers', 'mfscan', 'threed']);
check('unknown tokens are dropped, known ones survive',
  set('?hide=hikers,nonsense', undefined), ['hikers']);
check('preset + extra', set('?hide=og,explore', undefined), [...OG, 'explore'].sort());
check('other query params are ignored', set('?ui=desktop&hide=cascade', undefined),
  ['cascade']);
check('all preset covers every feature',
  [...parseSpec('all')].sort(), [...ALL_FEATURES].sort());

// The og preset must leave the original loop intact — that is the whole point of it.
const ogHidden = new Set(parseSpec('og'));
for (const kept of ['explore', 'fastscan', 'refine', 'threed', 'seeds']) {
  check(`og keeps ${kept}`, ogHidden.has(kept), false);
}

fs.rmSync(path.dirname(tmp), { recursive: true, force: true });
console.log(failures ? `\n${failures} failing` : '\nall passing');
process.exit(failures ? 1 : 0);
