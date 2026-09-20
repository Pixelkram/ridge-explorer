// Mobile Ridge Explorer.
//
// The desktop UI is a dense multi-panel tool built around a pan/zoom canvas, hover
// tooltips and side-by-side layers — none of which survive a 390px touch screen. Rather
// than retrofit media queries onto it (and risk the working desktop), this is a separate
// touch-first view over the SAME api client and store, offering the two things that are
// actually useful on a phone: run a grid and look at it, or send hikers and watch where
// they go.
//
// Conventions here that differ from the desktop deliberately:
//   * one column, nothing side-by-side
//   * every control is at least 44px tall (Apple's minimum touch target)
//   * no hover: anything the desktop puts in a title= is either shown or dropped
//   * tap a cell or a frame to open it full-screen, rather than zooming a canvas
import { useCallback, useEffect, useRef, useState } from 'react';
import { useRidgeStore } from './stores/ridgeStore';
import * as api from './api/client';
import { discoverDefaults, discoverStart, discoverStatus, discoverCancel } from './api/client';
import { cascadeStart, cascadeStatus, cascadeCancel, cascadeImageUrl } from './api/client';
import type { DiscoverDefaults, DiscoverStatus, CascadeStatus } from './api/types';
import CascadeMap from './CascadeMap';
import { journeys, lineageNames, branchColour, LineageTree } from './App';
import { loadResume, saveResume, onResumeVisible } from './resume';
import { feature, type Feature } from './featureFlags';

// Phones never show a frame larger than this, and a 512px strip is ~4x the bytes
// for pixels the screen cannot resolve.
const MOBILE_FRAME_PX = 256;

const ACCENT = '#4ecca3';
const BG = '#12121e';
const CARD = '#1a1a2e';

function Field({ label, value, onChange }:
               { label: string; value: string; onChange: (v: string) => void }) {
  return (
    <label style={{ display: 'block', marginBottom: 8 }}>
      <div style={{ fontSize: 11, color: '#8a9', marginBottom: 3 }}>{label}</div>
      <input value={value} onChange={(e) => onChange(e.target.value)}
             style={{ width: '100%', boxSizing: 'border-box', padding: '11px 10px',
                      fontSize: 16,  /* 16px or iOS zooms the page on focus */
                      background: '#0e0e18', color: '#eee', borderRadius: 8,
                      border: '1px solid #333' }} />
    </label>
  );
}

// `max` is optional: the hiking settings are uncapped in the UI. Stepping to a large
// value with +/- would be unusable, so the readout is a real numeric input — tap it and
// type. The server still enforces a finite ceiling and reports it if exceeded.
function Num({ label, value, set, min, max }:
             { label: string; value: number; set: (v: number) => void;
               min: number; max?: number }) {
  const clamp = (v: number) => {
    if (!Number.isFinite(v)) return min;
    return Math.max(min, max === undefined ? v : Math.min(max, v));
  };
  return (
    <div style={{ flex: 1, minWidth: 92 }}>
      <div style={{ fontSize: 11, color: '#8a9', marginBottom: 3 }}>{label}</div>
      <div style={{ display: 'flex', alignItems: 'stretch', gap: 4 }}>
        <button onClick={() => set(clamp(value - 1))} aria-label={`${label} down`}
                style={{ ...btn(false), width: 40, padding: 0, fontSize: 18 }}>−</button>
        <input type="number" inputMode="numeric" value={value}
               onChange={(e) => set(clamp(parseInt(e.target.value, 10)))}
               onFocus={(e) => e.target.select()}
               style={{ flex: 1, minWidth: 0, textAlign: 'center', fontSize: 16,
                        background: '#0e0e18', borderRadius: 8, border: '1px solid #333',
                        color: '#eee', padding: '0 4px' }} />
        <button onClick={() => set(clamp(value + 1))} aria-label={`${label} up`}
                style={{ ...btn(false), width: 40, padding: 0, fontSize: 18 }}>+</button>
      </div>
    </div>
  );
}

function btn(primary: boolean, danger = false): React.CSSProperties {
  return {
    minHeight: 44, padding: '0 16px', fontSize: 14, borderRadius: 8,
    border: primary || danger ? 'none' : '1px solid #3a3a52',
    background: danger ? '#e94560' : primary ? ACCENT : CARD,
    color: primary || danger ? '#fff' : '#ccd', cursor: 'pointer',
    WebkitTapHighlightColor: 'transparent',
  };
}

// Full-screen viewer. On a phone the grid thumbnails are far too small to judge, so a
// tap has to open something worth looking at.
function Lightbox({ src, caption, onClose }:
                  { src: string; caption?: string; onClose: () => void }) {
  useEffect(() => {
    const esc = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', esc);
    document.body.style.overflow = 'hidden';
    return () => { window.removeEventListener('keydown', esc);
                   document.body.style.overflow = ''; };
  }, [onClose]);
  return (
    <div onClick={onClose}
         style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,.94)', zIndex: 900,
                  display: 'flex', flexDirection: 'column', justifyContent: 'center',
                  alignItems: 'center', padding: 12 }}>
      <img src={src} alt={caption ?? ''}
           style={{ maxWidth: '100%', maxHeight: '82vh', objectFit: 'contain',
                    borderRadius: 8 }} />
      {caption && <div style={{ color: '#bbc', fontSize: 12, marginTop: 10,
                                textAlign: 'center' }}>{caption}</div>}
      <button onClick={onClose} style={{ ...btn(false), marginTop: 14 }}>close</button>
    </div>
  );
}

// ------------------------------------------------------------------ grid mode
// Discovery: the procedure the ablations support (see backend/services/discover.py).
// Not a port of the desktop panel -- that one hangs its evidence on `title=` hover
// tooltips, which do not exist on touch, so here the provenance is a tappable line.
// This pane deliberately ignores the prompt A/B/C fields above: discovery DRAWS its own
// prompts at a target spread, which is the point of it.
function DiscoverPane({ onOpen }: { onOpen: (s: string, c: string) => void }) {
  const [def, setDef] = useState<DiscoverDefaults | null>(null);
  const [k, setK] = useState(3);
  const [commitment, setCommitment] = useState(0.55);
  const [batch, setBatch] = useState(300);
  const [total, setTotal] = useState(1200);
  const [runId, setRunId] = useState<string | null>(null);
  const [st, setSt] = useState<DiscoverStatus | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [why, setWhy] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    discoverDefaults().then((d) => {
      setDef(d); setK(d.k); setCommitment(d.commitment); setBatch(d.batch);
    }).catch((e) => setErr(String(e.message || e)));
  }, []);

  // Single poll chain; clearing on unmount matters more here because switching tabs
  // unmounts the pane while the run keeps going server-side.
  useEffect(() => {
    if (!runId) return;
    let live = true;
    const tick = async () => {
      try {
        const s = await discoverStatus(runId);
        if (!live) return;
        setSt(s);
        if (s.status === 'running') timer.current = window.setTimeout(tick, 2500);
      } catch { if (live) timer.current = window.setTimeout(tick, 5000); }
    };
    tick();
    return () => { live = false; if (timer.current) window.clearTimeout(timer.current); };
  }, [runId]);

  const floor = 1 / k;
  const below = commitment <= floor;

  const go = async () => {
    setErr(null); setSt(null);
    try {
      const r = await discoverStart({ k, commitment, target_sim: def?.target_sim ?? 0.55,
                                      batch, total });
      if (r.error) { setErr(r.error); return; }
      setRunId(r.run_id);
    } catch (e: any) { setErr(String(e.message || e)); }
  };

  return (
    <div>
      <div style={{ background: CARD, borderRadius: 10, padding: 12, marginBottom: 10 }}>
        <div style={{ fontSize: 12, color: '#8a9', marginBottom: 8 }}>
          Draws its own prompts, samples at measured commitment, and restarts before the
          simplex runs dry.
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <Num label="prompts (k)" value={k} set={setK} min={2} max={8} />
          <Num label="images / simplex" value={batch} set={setBatch} min={20} max={2000} />
          <Num label="total images" value={total} set={setTotal} min={20} max={100000} />
        </div>
        <div style={{ marginTop: 10 }}>
          <div style={{ fontSize: 11, color: '#8a9', marginBottom: 3 }}>
            commitment E[max weight] — {commitment.toFixed(2)}
          </div>
          <input type="range" min={0.05} max={0.95} step={0.05} value={commitment}
                 onChange={(e) => setCommitment(parseFloat(e.target.value))}
                 style={{ width: '100%' }} />
          <div style={{ fontSize: 11, color: below ? '#e94560' : '#667', marginTop: 2 }}>
            {below
              ? `must exceed the k=${k} floor of ${floor.toFixed(2)}`
              : 'measured optimum 0.50–0.60'}
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
          <button onClick={go} disabled={below || st?.status === 'running'}
                  style={{ ...btn(!below), flex: 1,
                           opacity: below || st?.status === 'running' ? 0.5 : 1 }}>
            {st?.status === 'running' ? 'running…' : 'Start discovery'}
          </button>
          {st?.status === 'running' && runId && (
            <button onClick={() => discoverCancel(runId)} style={btn(false, true)}>stop</button>
          )}
        </div>
        <div onClick={() => setWhy(why ? null : (def?.provenance
               ? Object.values(def.provenance).join('\n\n') : 'defaults unavailable'))}
             style={{ marginTop: 10, fontSize: 11, color: ACCENT, cursor: 'pointer' }}>
          {why ? 'hide the evidence ▲' : 'why these defaults? ▼'}
        </div>
        {why && (
          <div style={{ marginTop: 6, fontSize: 11, color: '#99a', whiteSpace: 'pre-wrap',
                        lineHeight: 1.45 }}>{why}</div>
        )}
        {err && <div style={{ marginTop: 8, color: '#e94560', fontSize: 12 }}>{err}</div>}
      </div>

      {st && (
        <div style={{ background: CARD, borderRadius: 10, padding: 12 }}>
          <div style={{ display: 'flex', gap: 14, fontSize: 13, flexWrap: 'wrap' }}>
            <span>{st.generated}/{st.total}</span>
            <span style={{ color: ACCENT }}>
              diversity {st.diversity != null ? st.diversity.toFixed(3) : '—'}
            </span>
            <span style={{ color: '#889' }}>{st.simplices.length} simplices</span>
            <span style={{ color: st.status === 'error' ? '#e94560' : '#8a9' }}>{st.status}</span>
          </div>
          {st.error && <div style={{ color: '#e94560', fontSize: 12, marginTop: 6 }}>{st.error}</div>}
          {st.simplices.map((sx) => (
            <div key={sx.index} style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid #2a2a40',
                                         fontSize: 11, color: '#99a' }}>
              <div style={{ color: '#ccd' }}>
                #{sx.index} · {sx.done}/{sx.n_points}
                {sx.diversity != null && ` · div ${sx.diversity.toFixed(3)}`}
              </div>
              <div>{sx.prompts.join(' · ')}</div>
            </div>
          ))}
          {st.generated > 0 && runId && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 4,
                          marginTop: 10 }}>
              {Array.from({ length: Math.min(st.generated, 60) }, (_, i) => (
                <img key={i} src={`/api/discover/${runId}/${i}.jpg`} loading="lazy"
                     onClick={() => onOpen(`/api/discover/${runId}/${i}.jpg`, `image ${i}`)}
                     style={{ width: '100%', aspectRatio: '1', objectFit: 'cover',
                              borderRadius: 6 }} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function GridPane({ onOpen }: { onOpen: (src: string, cap: string) => void }) {
  const s = useRidgeStore();
  const busy = ['generating', 'analyzing', 'scanning'].includes(s.phase);
  const pct = s.cellsTotal ? Math.round((s.cellsGenerated / s.cellsTotal) * 100) : 0;
  return (
    <>
      <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
        <Num label="grid size" value={s.gridSize} set={s.setGridSize} min={4} max={24} />
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 11, color: '#8a9', marginBottom: 3 }}>images</div>
          <div style={{ minHeight: 44, display: 'grid', placeItems: 'center',
                        fontSize: 13, color: '#99a' }}>
            ≈ {Math.round((s.gridSize * (s.gridSize + 1)) / 2)}
          </div>
        </div>
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        <button onClick={() => s.generate()} disabled={busy}
                style={{ ...btn(!busy), flex: 1, opacity: busy ? 0.5 : 1 }}>
          {busy ? `generating ${pct}%` : 'explore'}
        </button>
        {busy && <button onClick={() => s.cancel()} style={btn(false, true)}>stop</button>}
      </div>
      {busy && (
        <div style={{ background: '#282840', height: 5, borderRadius: 3, marginTop: 10 }}>
          <div style={{ width: `${pct}%`, background: ACCENT, height: '100%',
                        borderRadius: 3, transition: 'width .3s' }} />
        </div>
      )}
      {s.error && <div style={{ color: '#e94560', fontSize: 12, marginTop: 8 }}>{s.error}</div>}

      {s.imageGridUrl && (
        <div style={{ marginTop: 12 }}>
          <div style={{ fontSize: 11, color: '#8a9', marginBottom: 5 }}>
            tap to open · {s.currentGridSize}×{s.currentGridSize}
          </div>
          {[['images', s.imageGridUrl], ['sensitivity', s.heatmapUrl],
            ['overlay', s.overlayUrl]].filter(([, u]) => u).map(([name, url]) => (
            <img key={name} src={url as string} alt={name as string}
                 onClick={() => onOpen(url as string, name as string)}
                 style={{ width: '100%', borderRadius: 8, marginBottom: 8,
                          background: '#000', display: 'block' }} />
          ))}
        </div>
      )}
    </>
  );
}

// ---------------------------------------------------------------- hikers mode
function MobileJourney({ hikeId, path, onOpen, onBranch, branching, name, parent, fork, pruned }:
                       { hikeId: string; path: any[]; onOpen: (s: string, c: string) => void;
                         onBranch: (row: any, station: number) => void; branching: boolean;
                         name: string; parent: string; fork: number; pruned: boolean }) {
  const seq: { r: any; k: number; s: [number, number]; approach: boolean; last: boolean }[] = [];
  for (const r of path) {
    const ap = r.approach ?? 0;
    (r.stations ?? []).forEach((p: [number, number], k: number) =>
      seq.push({ r, k, s: p, approach: k < ap, last: k === (r.stations.length - 1) }));
  }
  const [n, setN] = useState(0);
  const [playing, setPlaying] = useState(true);
  // see App.tsx: only a +1 step is travel; a wrap or a scrub is a jump and must not
  // animate, or the strip visibly scrolls backwards through the whole hop
  const bar = useRef<HTMLDivElement | null>(null);
  useEffect(() => { setN(0); }, [seq.length]);
  useEffect(() => {
    if (!playing || seq.length < 2) return;
    const t = window.setInterval(() => setN((p) => (p + 1) % seq.length), 500);
    return () => window.clearInterval(t);
  }, [playing, seq.length]);

  // touch scrubbing: the bar spans the whole journey, so a thumb drag crosses triangles
  const scrub = useCallback((x: number) => {
    const el = bar.current; if (!el) return;
    const r = el.getBoundingClientRect();
    const f = Math.min(1, Math.max(0, (x - r.left) / Math.max(r.width, 1)));
    setN(Math.min(seq.length - 1, Math.floor(f * seq.length)));
  }, [seq.length]);

  if (!seq.length) return null;
  const cur = seq[Math.min(n, seq.length - 1)];
  const r = cur.r;
  const leaf = path[path.length - 1];
  // frames in THIS hop's filmstrip; the strip is rendered per hop, not per journey
  const hopIdx = path.indexOf(r);
  // ONE CONTINUOUS TRACK across every triangle. Each hop is a separate strip jpg, so the
  // hand-off used to swap the img src and reset the offset — there was nothing to scroll
  // INTO, and it read as a cut however it was timed. Lay every hop's strip end to end in
  // a track and translate by the GLOBAL tile index: crossing a triangle is then just the
  // scroll continuing. Because the exit image of hop h is vertex A of hop h+1 (extend()
  // makes it so exactly), the two adjacent tiles are the same picture, so the seam reads
  // as a single held beat rather than a jump.
  // All widths are percentages of the TRACK, which is TOT container-widths wide, so one
  // tile is exactly one container width at any screen size — no pixel constants.
  const frames = path.map((row: any) => Math.max(1, row.n_frames ?? (row.stations?.length ?? 1)));
  const cumF: number[] = [];
  { let acc = 0; for (const f of frames) { cumF.push(acc); acc += f; } }
  const TOT = Math.max(1, cumF[cumF.length - 1] + frames[frames.length - 1]);
  const gpos = (cumF[hopIdx] ?? 0) + Math.min(cur.k, (frames[hopIdx] ?? 1) - 1);
  // see App.tsx: only a forward step of one TILE is travel. A hand-off advances gpos by
  // one, so it now scrolls; a wrap or a scrub still snaps.
  const prevG = useRef(-1);
  const jumped = gpos !== prevG.current + 1;
  useEffect(() => { prevG.current = gpos; }, [gpos]);
  // see App.tsx: a multi-seed hike walks the AVERAGED field, so every seed's strip follows
  // the identical route and switching compares noise draws, not hikes. Index, not value.
  const [seedIdx, setSeedIdx] = useState(0);
  const nSeeds = Math.max(1, ...path.map((row: any) => row.n_seeds ?? 1));
  const seedVals: number[] = path.find((row: any) => row.seeds?.length)?.seeds ?? [];
  useEffect(() => { if (seedIdx >= nSeeds) setSeedIdx(0); }, [nSeeds, seedIdx]);


  return (
    <div style={{ background: CARD, borderRadius: 10, padding: 10, marginBottom: 10 }}>
      <div style={{ fontSize: 12, color: '#9ab', marginBottom: 6, display: 'flex',
                    alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
        <span style={{ display: 'inline-flex', gap: 3 }}>
          {path.map((_, k) => (
            <span key={k} style={{ width: 9, height: 9, borderRadius: 5,
              border: `1.5px solid ${branchColour(name)}`,
              background: k >= fork ? branchColour(name) : 'transparent',
              opacity: k >= fork ? 1 : 0.45 }} />
          ))}
        </span>
        <b style={{ color: branchColour(name), fontSize: 14 }}>{name || 'A'}</b>
        {parent && <span style={{ color: '#667' }}>from {parent}</span>}
        <span>{path.length} tri · {seq.length} steps</span>
        {pruned && <span style={{ color: '#e9a145' }}>pruned</span>}
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        <div style={{ flex: '0 0 38%' }}>
          <img src={api.hikeMapUrl(hikeId, r.chain, r.hop)} alt="simplex"
               onClick={() => onOpen(api.hikeMapUrl(hikeId, r.chain, r.hop),
                                     `triangle ${path.indexOf(r) + 1}/${path.length}`)}
               onError={(e) => { (e.target as HTMLImageElement).style.visibility = 'hidden'; }}
               style={{ width: '100%', borderRadius: 6, background: '#000', display: 'block' }} />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div onClick={() => onOpen(api.hikeFilmstripUrl(hikeId, r.chain, r.hop,
                                                          MOBILE_FRAME_PX, seedIdx),
                                     `hop ${r.hop} · full walk`
                                     + (nSeeds > 1 ? ` · seed ${seedVals[seedIdx] ?? seedIdx}` : ''))}
               style={{ width: '100%', aspectRatio: '1', overflow: 'hidden', borderRadius: 6,
                        background: '#000', position: 'relative' }}>
            {/* Size the strip in FRACTIONS OF ITSELF, not pixels. `height: 100%` scaled
                the image by container/tile (~1.7x here) while translateX still moved the
                unscaled 96px, so a step advanced ~58% of a frame. Width = n*100% makes
                one tile exactly one container width, and a -100/n% shift is exactly one
                frame at any screen size. */}
            <div style={{ position: 'absolute', top: 0, left: 0, height: '100%',
                          width: `${TOT * 100}%`,
                          transform: `translateX(-${(gpos * 100) / TOT}%)`,
                          transformOrigin: 'top left',
                          transition: jumped ? 'none' : 'transform .2s linear' }}>
              {path.map((row: any, i: number) => (
                <img key={`f-${row.hop}-${row.chain}`} alt={i === hopIdx ? `frame ${cur.k}` : ''}
                     src={Math.abs(i - hopIdx) <= 1
                          ? api.hikeFilmstripUrl(hikeId, row.chain, row.hop,
                                                 MOBILE_FRAME_PX, seedIdx)
                          : undefined}
                     /* a 404 on one seed variant must not hide the tile for good */
                     onLoad={(e) => { (e.target as HTMLImageElement).style.visibility = 'visible'; }}
                     onError={(e) => { (e.target as HTMLImageElement).style.visibility = 'hidden'; }}
                     style={{ position: 'absolute', top: 0, height: 'auto',
                              left: `${(cumF[i] * 100) / TOT}%`,
                              width: `${(frames[i] * 100) / TOT}%` }} />
              ))}
            </div>
          </div>
        </div>
      </div>

      <div ref={bar}
           onPointerDown={(e) => { e.preventDefault(); setPlaying(false); scrub(e.clientX);
                                   (e.target as HTMLElement).setPointerCapture(e.pointerId); }}
           onPointerMove={(e) => { if (e.buttons) scrub(e.clientX); }}
           style={{ display: 'flex', gap: 1, height: 22, marginTop: 8, alignItems: 'stretch',
                    touchAction: 'none', cursor: 'grab' }}>
        {seq.map((q, idx) => (
          <div key={idx} style={{ flex: 1, borderRadius: 2,
               background: idx === n ? '#fff' : q.approach ? 'rgba(150,200,255,.5)'
                           : q.last ? 'rgba(233,69,96,.65)' : 'rgba(80,255,180,.4)' }} />
        ))}
      </div>
      <div style={{ fontSize: 11, color: '#889', marginTop: 5 }}>
        step {n + 1}/{seq.length} · cell [{cur.s[0]},{cur.s[1]}]
        {cur.approach && <span style={{ color: '#96c8ff' }}> · entering</span>}
        {cur.last && <span style={{ color: '#e94560' }}> · exit</span>}
      </div>
      {nSeeds > 1 && (
        <div style={{ display: 'flex', gap: 5, alignItems: 'center', marginTop: 6,
                      flexWrap: 'wrap' }}>
          <span style={{ fontSize: 11, color: '#889' }}>seed</span>
          {Array.from({ length: nSeeds }, (_, k) => (
            <button key={k} onClick={() => setSeedIdx(k)}
                    style={{ ...btn(k === seedIdx), flex: '0 0 auto', padding: '4px 10px',
                             fontSize: 11 }}>
              {seedVals[k] ?? k}
            </button>
          ))}
        </div>
      )}
      <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
        <button onClick={() => setPlaying((p) => !p)} style={{ ...btn(false), flex: '0 0 96px' }}>
          {playing ? '❚❚ pause' : '▶ play'}
        </button>
        <button onClick={() => onBranch(r, cur.k)} disabled={branching}
                style={{ ...btn(true), flex: 1, opacity: branching ? 0.5 : 1 }}>
          {branching ? 'starting…' : 'explore from here'}
        </button>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ cascade mode
// The two-tier cascade (isolate crossings, score with coupled seeds against the run's
// own measured background, refine top regions + one exploration slot). Like discovery
// it draws its own prompts; provenance lives in the pane text, not hover tooltips.
function CascadePane({ onOpen }: { onOpen: (s: string, c: string) => void }) {
  const [k, setK] = useState(4);
  const [nChords, setNChords] = useState(24);
  const [nPatches, setNPatches] = useState(4);
  const [runId, setRunId] = useState<string | null>(null);
  const [st, setSt] = useState<CascadeStatus | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  // Single poll chain; cleared on unmount because switching tabs unmounts the pane
  // while the run keeps going server-side.
  useEffect(() => {
    if (!runId) return;
    let live = true;
    const tick = async () => {
      try {
        const s = await cascadeStatus(runId);
        if (!live) return;
        setSt(s);
        if (s.status === 'running') timer.current = window.setTimeout(tick, 2500);
      } catch { if (live) timer.current = window.setTimeout(tick, 5000); }
    };
    tick();
    return () => { live = false; if (timer.current) window.clearTimeout(timer.current); };
  }, [runId]);

  const go = async () => {
    setErr(null); setSt(null);
    try {
      // no seed control on mobile: randomise per start, else the deterministic
      // default re-draws the identical prompts and map every run
      const r = await cascadeStart({ k, n_chords: nChords, n_patches: nPatches,
                                     seed: Math.floor(Math.random() * 1e6) });
      if (r.error) { setErr(r.error); return; }
      setRunId(r.run_id);
    } catch (e: any) { setErr(String(e.message || e)); }
  };

  const running = st?.status === 'running';
  const scored = (st?.crossings ?? []).filter((c) => c.b !== null)
    .sort((a, b) => (b.b ?? 0) - (a.b ?? 0));
  const nSig = scored.filter((c) => c.significant).length;
  const phase: Record<string, string> = {
    chords: 'isolating', bisect: 'pinning crossings', score: 'scoring',
    patches: 'refining', done: 'done',
  };

  return (
    <div>
      <div style={{ background: CARD, borderRadius: 10, padding: 12, marginBottom: 10 }}>
        <div style={{ fontSize: 12, color: '#8a9', marginBottom: 8 }}>
          Finds boundaries between its drawn prompts, scores each against the run's own
          background drift, and walks a 5x5 image patch across the strongest ones.
        </div>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8 }}>
          {[3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16].map((pk) => {
            const pc = Math.max(12, Math.ceil((12 * pk) / 7));
            return (
              <button key={pk}
                      onClick={() => { setK(pk); setNChords(pc); setNPatches(4); }}
                      style={{ ...btn(false), padding: '6px 10px', fontSize: 13,
                               border: k === pk ? `1px solid ${ACCENT}` : '1px solid #3a3a52',
                               color: k === pk ? '#fff' : '#889' }}>
                k{pk}
              </button>
            );
          })}
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <Num label="prompts (k)" value={k} set={setK} min={3} />
          <Num label="chords" value={nChords} set={setNChords} min={4} />
          <Num label="patches" value={nPatches} set={setNPatches} min={2} />
        </div>
        {(() => {
          const cross = Math.max(1, Math.round((nChords * 7) / k));
          const imgs = Math.round((nChords * 110) / k + cross * 2
                                  + (cross + 12) * 8 + nPatches * 25);
          const mins = Math.max(1, Math.round(imgs / 8 / 60));
          return (
            <div style={{ fontSize: 11, color: '#667', marginTop: 8 }}>
              ~{imgs} images · ~{mins} min · ~{cross} crossings expected.
              k 3–4 = crisp boundaries; higher k = softer blends. Every start draws a
              fresh seed — same settings, new map.{k > 9 &&
                ' k>9 is beyond the calibrated range; verdicts stay self-calibrated.'}
            </div>
          );
        })()}
        <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
          <button onClick={go} disabled={running}
                  style={{ ...btn(true), flex: 1, opacity: running ? 0.5 : 1 }}>
            {running ? 'running\u2026' : 'Start cascade'}
          </button>
          {running && runId && (
            <button onClick={() => cascadeCancel(runId)} style={btn(false, true)}>stop</button>
          )}
        </div>
        {err && <div style={{ marginTop: 8, color: '#e94560', fontSize: 12 }}>{err}</div>}
      </div>

      {st && (
        <div style={{ background: CARD, borderRadius: 10, padding: 12 }}>
          <div style={{ display: 'flex', gap: 14, fontSize: 13, flexWrap: 'wrap' }}>
            <span style={{ color: st.status === 'error' ? '#e94560' : '#8a9' }}>
              {st.status === 'running' ? (phase[st.phase] || st.phase) : st.status}
              {st.status === 'running' && st.phase_total > 0 &&
                ` ${st.phase_done}/${st.phase_total}`}
            </span>
            <span style={{ color: '#889' }}>{st.generated} images</span>
            {scored.length > 0 && (
              <span style={{ color: ACCENT }}>{nSig}/{scored.length} significant</span>
            )}
            {st.unexplored_share !== null && st.distinct_ridges !== null && (
              <span style={{ color: st.unexplored_share <= 0.15 ? ACCENT : '#c9a227' }}>
                {st.distinct_ridges} ridges · ≈{Math.round(st.unexplored_share * 100)}% unexplored
              </span>
            )}
          </div>
          {st.prompts.length > 0 && (
            <div style={{ fontSize: 11, color: '#99a', marginTop: 6 }}>
              {st.prompts.join(' \u00b7 ')}
            </div>
          )}
          {st.error && <div style={{ color: '#e94560', fontSize: 12, marginTop: 6 }}>{st.error}</div>}
          {st.status === 'running' && st.recent_thumbs.length > 0 && runId && (
            <div style={{ display: 'flex', gap: 3, marginTop: 8, overflow: 'hidden' }}>
              {[...st.recent_thumbs].reverse().map((t) => (
                <img key={t} src={cascadeImageUrl(runId, t)}
                     style={{ width: 40, height: 40, objectFit: 'cover',
                              borderRadius: 4 }} />
              ))}
            </div>
          )}
          {st.chords.length > 0 && runId && (
            <div style={{ marginTop: 10 }}>
              <CascadeMap status={st} runId={runId} imageUrl={cascadeImageUrl}
                          size={340} />
            </div>
          )}
          {scored.length > 0 && runId && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 4,
                          marginTop: 10 }}>
              {scored.map((c) => c.thumb >= 0 && (
                <div key={c.cid} onClick={() => onOpen(cascadeImageUrl(runId, c.thumb),
                       `crossing ${c.cid} \u00b7 B=${c.b?.toFixed(2)}${c.significant ? ' \u00b7 significant' : ''}`)}>
                  <img src={cascadeImageUrl(runId, c.thumb)} loading="lazy"
                       style={{ width: '100%', aspectRatio: '1', objectFit: 'cover',
                                borderRadius: 6,
                                border: c.significant ? `2px solid ${ACCENT}` : '2px solid transparent' }} />
                  <div style={{ fontSize: 10, textAlign: 'center',
                                color: c.significant ? ACCENT : '#889' }}>
                    {c.b?.toFixed(2)}
                  </div>
                </div>
              ))}
            </div>
          )}
          {(st.patches ?? []).map((p) => runId && (
            <div key={p.region} style={{ marginTop: 12, paddingTop: 10,
                                         borderTop: '1px solid #2a2a40' }}>
              <div style={{ fontSize: 11, color: p.significant ? ACCENT : '#889',
                            marginBottom: 4 }}>
                {p.exploration ? 'exploration slot' : `region ${p.region}`} · B={p.b.toFixed(2)}
                {p.significant ? ' · significant' : ''} · {p.cols_with_crossing}/5 rows cross
              </div>
              {p.grid.map((row, i) => (
                <div key={i} style={{ display: 'flex', gap: 2, marginBottom: 2 }}>
                  {row.map((t, j) => t >= 0 ? (
                    <img key={j} src={cascadeImageUrl(runId, t)} loading="lazy"
                         onClick={() => onOpen(cascadeImageUrl(runId, t),
                           `patch ${p.region} \u00b7 row ${i}, step ${j}`)}
                         style={{ flex: 1, aspectRatio: '1', minWidth: 0,
                                  objectFit: 'cover', borderRadius: 4 }} />
                  ) : (
                    <div key={j} style={{ flex: 1, aspectRatio: '1', minWidth: 0,
                                          background: '#12121e', borderRadius: 4 }} />
                  ))}
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function HikePane({ onOpen }: { onOpen: (s: string, c: string) => void }) {
  const { promptA, promptB, promptC } = useRidgeStore();
  const [beam, setBeam] = useState(4);
  const [budget, setBudget] = useState(16);
  const [gs, setGs] = useState(25);
  const [steps, setSteps] = useState(8);
  // see App.tsx: seed was the largest uncontrolled factor in a hike
  const [seed, setSeed] = useState(42);
  const [seedCount, setSeedCount] = useState(1);
  const [cfg, setCfg] = useState(4.0);
  const [res, setRes] = useState(512);
  const [hikeId, setHikeId] = useState<string | null>(() => loadResume().hikeId ?? null);
  const [status, setStatus] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [branching, setBranching] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  // Exactly ONE poll chain, and it belongs to the hike on screen — same fix, and the
  // same reasons, as HikePanel in App.tsx. `poll` reschedules itself, so start, branch
  // and every foregrounding forked another chain while `timer` held only the last one:
  // a phone that is locked and woken ten times ended up making ten /status GETs every
  // 3s, and after a branch the parent run's chain kept overwriting the child's status.
  // `watching` names the only id a chain may report on; everything else drops out.
  const watching = useRef<string | null>(null);
  const poll = useCallback(async (id: string) => {
    if (watching.current !== id) return;      // superseded before this tick ran
    let again = true;
    try {
      const st = await api.getHikeStatus(id);
      if (watching.current !== id) return;    // superseded while the request was in flight
      if (st.status === 'unknown') {          // server restarted, or the id aged out
        again = false;                        // nothing left to poll: do not respawn
        saveResume({ hikeId: null }); setHikeId(null); setStatus(null);
        setErr('that run is no longer on the server'); return;
      }
      setStatus(st); setErr(null); again = st.status === 'running';
    } catch (e: any) { setErr(String(e)); }
    finally {
      if (again && watching.current === id) {
        if (timer.current) window.clearTimeout(timer.current);
        timer.current = window.setTimeout(() => poll(id), 3000);
      }
    }
  }, []);

  // A phone lock evicts the tab; the reload lands here with the id restored from
  // storage, so pick the run back up instead of showing an empty form. Also poll the
  // moment the tab is foregrounded again — background timers are throttled or stopped,
  // so the first thing a returning user sees would otherwise be stale. The cleanup
  // covers unmount too, so there is no separate unmount-only effect.
  useEffect(() => {
    watching.current = hikeId;
    if (hikeId) poll(hikeId);
    const off = onResumeVisible(() => { if (watching.current) poll(watching.current); });
    return () => {
      off();
      watching.current = null;                // an in-flight reply must not reschedule
      if (timer.current) { window.clearTimeout(timer.current); timer.current = null; }
    };
  }, [hikeId, poll]);

  const start = useCallback(async () => {
    setErr(null); setStatus(null);
    if (!promptA || !promptB || !promptC) { setErr('needs three prompts'); return; }
    try {
      const r = await api.startHike({ prompt_a: promptA, prompt_b: promptB,
                                      prompt_c: promptC, beam, budget, grid_size: gs,
                                      seed, seed_count: seedCount,
                                      steps, guidance_scale: cfg, height: res, width: res });
      // no poll() here: the hikeId effect owns starting the chain
      if (r.hike_id) { saveResume({ hikeId: r.hike_id }); setHikeId(r.hike_id); }
    } catch (e: any) {
      const m = String(e);
      // the server keeps a finite ceiling even though the UI does not; say so plainly
      setErr(m.includes('422')
        ? 'above the server limit (hikers ≤ 32, grids ≤ 1000, grid size ≤ 100)'
        : m);
    }
  }, [promptA, promptB, promptC, beam, budget, gs, seed, seedCount]);

  const branch = useCallback(async (row: any, station: number) => {
    if (!hikeId) return;
    setBranching(true); setErr(null);
    try {
      const r = await api.branchHike(hikeId, { hop: row.hop, chain: row.chain, station,
                                               beam, budget, grid_size: gs, steps,
                                      seed, seed_count: seedCount,
                                               guidance_scale: cfg, height: res, width: res });
      if (r.hike_id) { saveResume({ hikeId: r.hike_id }); setStatus(null);
                       setHikeId(r.hike_id); }
    } catch (e: any) { setErr(String(e)); }
    finally { setBranching(false); }
  }, [hikeId, beam, budget, gs, seed, seedCount]);

  const running = status?.status === 'running';
  const rows: any[] = status?.chains ?? [];
  return (
    <>
      <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
        <Num label="hikers" value={beam} set={setBeam} min={1} />
        <Num label="grids" value={budget} set={setBudget} min={1} />
        <Num label="grid size" value={gs} set={setGs} min={2} />
      </div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
        <Num label="steps" value={steps} set={setSteps} min={1} max={50} />
        <div style={{ flex: 1, minWidth: 92 }}>
          <div style={{ fontSize: 11, color: '#8a9', marginBottom: 3 }}>cfg</div>
          <div style={{ display: 'flex', gap: 4 }}>
            <button onClick={() => setCfg(Math.max(0, +(cfg - 0.5).toFixed(1)))}
                    style={{ ...btn(false), width: 40, padding: 0, fontSize: 18 }}>−</button>
            <input type="number" inputMode="decimal" step={0.5} value={cfg}
                   onChange={(e) => setCfg(Math.max(0, Math.min(20, +e.target.value)))}
                   onFocus={(e) => e.target.select()}
                   style={{ flex: 1, minWidth: 0, textAlign: 'center', fontSize: 16,
                            background: '#0e0e18', borderRadius: 8, border: '1px solid #333',
                            color: '#eee', padding: '0 4px' }} />
            <button onClick={() => setCfg(Math.min(20, +(cfg + 0.5).toFixed(1)))}
                    style={{ ...btn(false), width: 40, padding: 0, fontSize: 18 }}>+</button>
          </div>
        </div>
        <div style={{ flex: 1, minWidth: 92 }}>
          <div style={{ fontSize: 11, color: '#8a9', marginBottom: 3 }}>resolution</div>
          <select value={res} onChange={(e) => setRes(+e.target.value)}
                  style={{ width: '100%', minHeight: 44, fontSize: 15, borderRadius: 8,
                           background: '#0e0e18', color: '#eee', border: '1px solid #333' }}>
            {[128, 192, 256, 384, 512].map((r) => <option key={r} value={r}>{r}px</option>)}
          </select>
        </div>
      </div>
      {/* Seed was reachable only through the API on mobile: the state existed and was sent,
          but nothing rendered it, so every mobile hike ran at seed 42 with no averaging.
          E25 makes the seed the largest single lever on what a hike finds. */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
        <Num label="seed" value={seed} set={setSeed} min={0} />
        <div style={{ flex: 1, minWidth: 92 }}>
          <div style={{ fontSize: 11, color: '#8a9', marginBottom: 3 }}>seeds averaged</div>
          <select value={seedCount} onChange={(e) => setSeedCount(+e.target.value)}
                  style={{ width: '100%', minHeight: 44, fontSize: 15, borderRadius: 8,
                           background: '#0e0e18', color: '#eee', border: '1px solid #333' }}>
            {[1, 2, 3, 4, 6, 8].map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
      </div>
      {cfg <= 1.0 && (
        <div style={{ fontSize: 11, color: '#4ecca3', marginBottom: 8 }}>
          cfg 1.0 skips the unconditional branch — about 2x faster, different images
        </div>
      )}
      {/* the measured payoff, so the
          linear cost above is a judgement the user can actually make (E26) */}
      {seedCount > 1 ? (
        <div style={{ fontSize: 11, color: '#4ecca3', marginBottom: 8 }}>
          walking the field averaged over seeds {seed}–{seed + seedCount - 1}
          {seedCount >= 3 ? ' — ridge agrees ~88% with an independent rerun (49% at 1 seed)'
                          : ' — 3 is the knee; 2 gets about half the gain'}
        </div>
      ) : (
        <div style={{ fontSize: 11, color: '#889', marginBottom: 8 }}>
          one seed: the walk agrees only ~49% with an independent rerun — raise
          “seeds averaged” to 3 for ~88%
        </div>
      )}
      {/* uncapped settings need the cost visible before you commit to them.
          seedCount multiplies images LINEARLY -- omitting it understated a 3-seed hike
          by 3x, which is the one setting here whose cost is easiest to underestimate. */}
      {(() => {
        const imgs = budget * Math.round((gs * (gs + 1)) / 2) * seedCount;
        return (
          <div style={{ fontSize: 11, color: imgs > 20000 ? '#e9a145' : '#778',
                        marginBottom: 8 }}>
            ≈ {imgs.toLocaleString()} images
            {seedCount > 1 && ` (${seedCount} seeds × ${(imgs / seedCount).toLocaleString()})`}
            {imgs > 20000 && ' — this will take a while; stop is available'}
          </div>
        );
      })()}
      <div style={{ display: 'flex', gap: 8 }}>
        <button onClick={start} disabled={running}
                style={{ ...btn(!running), flex: 1, opacity: running ? 0.5 : 1 }}>
          {running ? `hiking · hop ${status.hop}` : 'send hikers'}
        </button>
        {running && hikeId && (
          <button onClick={() => api.cancelHike(hikeId).catch(() => {})}
                  style={btn(false, true)}>stop</button>
        )}
      </div>
      {status && (
        <div style={{ fontSize: 12, color: '#99a', marginTop: 8 }}>
          {status.status} · {status.grids_done}/{status.budget} grids
          {status.refined_cells ? ` · +${status.refined_cells} refined` : ''}
        </div>
      )}
      {err && <div style={{ color: '#e94560', fontSize: 12, marginTop: 8 }}>{err}</div>}
      {(status?.notes ?? []).map((nt: string, i: number) => (
        <div key={i} style={{ fontSize: 11, color: '#e9a145', marginTop: 4 }}>· {nt}</div>
      ))}
      <div style={{ marginTop: 12 }}>
        {hikeId && (
          <div style={{ overflowX: 'auto', marginBottom: 8, WebkitOverflowScrolling: 'touch' }}>
            {/* 46px nodes: below 44 a node is not a reliable touch target */}
            <LineageTree hikeId={hikeId} rows={rows} vertical node={46} selected={selected}
                         onSelect={(l) => setSelected(l === selected ? null : l)} />
          </div>
        )}
        {hikeId && (() => {
          const names = lineageNames(rows);
          const maxHop = Math.max(0, ...rows.map((r: any) => r.hop ?? 0));
          const par = (l: string | null): string | null => {
            if (!l || l === 'root') return null;
            const i = l.lastIndexOf('>'); return i <= 0 ? 'root' : l.slice(0, i);
          };
          const passes = (path: any[]) =>
            !selected || path.some((r) => (r.lineage ?? 'root') === selected);
          return journeys(rows).filter(passes).map((path) => {
            const leaf = path[path.length - 1];
            const mine = names.get(leaf.lineage ?? 'root') ?? '';
            let l = par(leaf.lineage ?? 'root');
            let d = (leaf.lineage ?? 'root') === 'root'
                    ? 0 : (leaf.lineage as string).split('>').length - 1;
            while (l && (names.get(l) ?? '') === mine) { d--; l = par(l); }
            return (
              <MobileJourney key={leaf.lineage ?? 'root'} hikeId={hikeId} path={path}
                             onOpen={onOpen} onBranch={branch} branching={branching}
                             name={mine} parent={l ? (names.get(l) ?? '') : ''}
                             fork={Math.max(0, d - 1)} pruned={leaf.hop < maxHop} />
            );
          });
        })()}
      </div>
    </>
  );
}

// ------------------------------------------------------------------------ shell

// The same gating the desktop panels use (see featureFlags.ts), so a demo build looks
// the same from either surface. 'grid' is the original loop and carries no flag, which
// also guarantees the tab list is never empty.
type MobileTab = 'grid' | 'hike' | 'discover' | 'cascade';
const TAB_FLAG: Record<MobileTab, Feature | null> = {
  hike: 'hikers', grid: null, discover: 'discovery', cascade: 'cascade',
};
const TAB_LABEL: Record<MobileTab, string> = {
  hike: 'hikers', grid: 'grid', discover: 'discover', cascade: 'cascade',
};

export default function MobileApp() {
  const s = useRidgeStore();
  // restore the prompts once, then mirror every edit
  useEffect(() => {
    const r = loadResume();
    if (r.promptA) s.setPromptA(r.promptA);
    if (r.promptB) s.setPromptB(r.promptB);
    if (r.promptC) s.setPromptC(r.promptC);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    saveResume({ promptA: s.promptA, promptB: s.promptB, promptC: s.promptC });
  }, [s.promptA, s.promptB, s.promptC]);
  const tabs = (['hike', 'grid', 'discover', 'cascade'] as MobileTab[])
    .filter(t => { const f = TAB_FLAG[t]; return f === null || feature(f); });
  const [tab, setTab] = useState<MobileTab>(() => tabs[0]);
  const [box, setBox] = useState<{ src: string; cap: string } | null>(null);
  const open = useCallback((src: string, cap: string) => setBox({ src, cap }), []);
  return (
    <div style={{ background: BG, minHeight: '100vh', color: '#ddd', padding: 12,
                  fontFamily: 'system-ui, -apple-system, sans-serif',
                  maxWidth: 720, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 12 }}>
        <h1 style={{ fontSize: 17, margin: 0 }}>Ridge Explorer</h1>
        <span style={{ fontSize: 11, color: '#667' }}>mobile</span>
        <a href="?ui=desktop" style={{ marginLeft: 'auto', fontSize: 11, color: '#667' }}>
          desktop ↗
        </a>
      </div>

      <Field label="prompt A" value={s.promptA} onChange={s.setPromptA} />
      <Field label="prompt B" value={s.promptB} onChange={s.setPromptB} />
      <Field label="prompt C" value={s.promptC} onChange={s.setPromptC} />

      {/* one tab is not a choice — the bar is just noise on a demo build */}
      <div style={{ display: tabs.length > 1 ? 'flex' : 'none', gap: 6, margin: '14px 0 12px' }}>
        {tabs.map((t) => (
          <button key={t} onClick={() => setTab(t)}
                  style={{ ...btn(false), flex: 1,
                           background: tab === t ? '#20304d' : CARD,
                           color: tab === t ? '#fff' : '#889',
                           border: tab === t ? `1px solid ${ACCENT}` : '1px solid #3a3a52' }}>
            {TAB_LABEL[t]}
          </button>
        ))}
      </div>

      {tab === 'grid' ? <GridPane onOpen={open} />
        : tab === 'discover' ? <DiscoverPane onOpen={open} />
        : tab === 'cascade' ? <CascadePane onOpen={open} />
        : <HikePane onOpen={open} />}
      {box && <Lightbox src={box.src} caption={box.cap} onClose={() => setBox(null)} />}
    </div>
  );
}
