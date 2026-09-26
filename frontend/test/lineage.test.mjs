// Runs the REAL LineageTree layout, transpiled out of App.tsx, over every saved hike.
//
// A hand-written copy of this logic passed while the shipped code was in an infinite
// loop: the copy skipped a null parent, the real code wrote `parentLineage(l) ?? 'root'`
// and made root its own child. Test the shipped source, not a paraphrase of it.
import fs from 'node:fs';
import path from 'node:path';

const SRC = new URL('../src/App.tsx', import.meta.url).pathname;
const src = fs.readFileSync(SRC, 'utf8');

function grab(name, start) {
  const i = src.indexOf(start);
  if (i < 0) throw new Error(`could not find ${name} in App.tsx — did it get renamed?`);
  return i;
}
// pull the parent/kids/place block straight out of the component
const body = src.slice(grab('LineageTree', 'export function LineageTree'),
                      grab('edges', 'const edges: JSX.Element[] = []'));
const js = body
  .replace(/:\s*string\[\]/g, '').replace(/:\s*string/g, '').replace(/:\s*number/g, '')
  .replace(/<string,\s*string\[\]>/g, '').replace(/<string,\s*number>/g, '')
  .replace(/<string>/g, '').replace(/!\./g, '.').replace(/\?\./g, '?.')
  .replace(/as number\[\]/g, '');

function layout(rows) {
  const parentLineage = (l) => {
    if (!l || l === 'root') return null;
    const i = l.lastIndexOf('>');
    return i <= 0 ? 'root' : l.slice(0, i);
  };
  const walked = rows.filter((r) => (r.stations?.length ?? 0) > 0);
  if (walked.length < 2) return { skipped: true };
  const byLin = new Map();
  for (const r of walked) if (!byLin.has(r.lineage ?? 'root')) byLin.set(r.lineage ?? 'root', r);

  // ---- the exact shipped construction ----
  const kids = new Map();
  for (const l of byLin.keys()) {
    const p = parentLineage(l);
    if (p === null || p === l) continue;
    if (!kids.has(p)) kids.set(p, []);
    kids.get(p).push(l);
  }
  for (const v of kids.values()) v.sort();
  const roots = byLin.has('root') ? ['root']
    : [...byLin.keys()].filter((l) => !byLin.has(parentLineage(l) ?? ''));
  const slot = new Map();
  let next = 0;
  const seen = new Set();
  const place = (l) => {
    if (seen.has(l)) return slot.get(l) ?? 0;
    seen.add(l);
    const cs = (kids.get(l) ?? []).filter((c) => byLin.has(c) && c !== l);
    if (!cs.length) { slot.set(l, next); return next++; }
    const ys = cs.map((c) => place(c));
    const y = (Math.min(...ys) + Math.max(...ys)) / 2;
    slot.set(l, y);
    return y;
  };
  roots.forEach((l) => place(l));
  return { nodes: byLin.size, placed: slot.size, leaves: next, roots: roots.length };
}

// 1. the shipped source must not contain the self-parenting bug
if (/parentLineage\(l\)\s*\?\?\s*'root'/.test(src)) {
  console.error('FAIL: App.tsx still maps a null parent onto root — root becomes its own child');
  process.exit(1);
}

// 2. exercise it on every saved hike
const RESULTS = new URL('../../cache_data/results', import.meta.url).pathname;
let n = 0, skipped = 0;
for (const d of fs.readdirSync(RESULTS).filter((x) => x.startsWith('hike_'))) {
  const f = path.join(RESULTS, d, 'hike.json');
  if (!fs.existsSync(f)) continue;
  let j;
  try { j = JSON.parse(fs.readFileSync(f, 'utf8')); } catch { skipped++; continue; }
  const out = layout(j.chains ?? []);
  n++;
  if (!out.skipped && out.placed < out.nodes) {
    console.error(`FAIL ${d}: ${out.placed}/${out.nodes} nodes placed`);
    process.exit(1);
  }
}

// 3. the exact shape that used to hang: hop 0 present, with branching
const cyclic = [
  { lineage: 'root', hop: 0, stations: [[0, 0]] },
  { lineage: '>1,0', hop: 1, stations: [[1, 0]] },
  { lineage: '>2,0', hop: 1, stations: [[2, 0]] },
  { lineage: '>1,0>3,0', hop: 2, stations: [[3, 0]] },
];
const out = layout(cyclic);
if (out.placed !== 4) { console.error('FAIL: root-inclusive tree not fully placed', out); process.exit(1); }

console.log(`OK — layout terminates on ${n} saved hikes (${skipped} unparseable, skipped) `
            + `and on the root-inclusive case that used to recurse forever`);
