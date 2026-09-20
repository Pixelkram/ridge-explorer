/**
 * Discovery panel — the procedure our ablations actually support.
 *
 * Deliberately not a ridge UI. The ridge graph, the guided survey and the exit-carrying
 * edge-hop were each ablated in Aug 2026 and none beat its control, so this panel exposes
 * only what survived: pick prompts at the right spread, sample the simplex at the right
 * COMMITMENT, and restart before the simplex exhausts.
 *
 * Every control shows the evidence behind its default on hover, because the whole point of
 * this section is that these are measured values rather than taste.
 */
import { useEffect, useRef, useState } from 'react';
import { discoverDefaults, discoverStart, discoverStatus, discoverCancel } from './api/client';
import type { DiscoverDefaults, DiscoverStatus } from './api/types';

const BOX: React.CSSProperties = {
  background: '#16213e', border: '1px solid #333', borderRadius: 4,
  padding: '8px 12px', margin: '0 12px 8px', fontSize: 12,
};
const NUM: React.CSSProperties = {
  width: 66, background: '#0a0a1a', color: '#fff', border: '1px solid #444',
  borderRadius: 3, padding: '2px 4px', fontSize: 12,
};

export default function DiscoverPanel() {
  const [open, setOpen] = useState(false);
  const [def, setDef] = useState<DiscoverDefaults | null>(null);
  const [k, setK] = useState(3);
  const [commitment, setCommitment] = useState(0.55);
  const [targetSim, setTargetSim] = useState(0.56);
  const [batch, setBatch] = useState(300);
  const [total, setTotal] = useState(1200);
  const [runId, setRunId] = useState<string | null>(null);
  const [status, setStatus] = useState<DiscoverStatus | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    if (!open || def) return;
    discoverDefaults().then((d) => {
      setDef(d);
      setK(d.k); setCommitment(d.commitment); setTargetSim(d.target_sim); setBatch(d.batch);
    }).catch((e) => setErr(String(e.message || e)));
  }, [open, def]);

  // One poll chain only. Re-entering this effect must clear the previous timer or a
  // re-render forks a second chain, which is how the hike panel once ended up issuing
  // one /status per alt-tab.
  useEffect(() => {
    if (!runId) return;
    let live = true;
    const tick = async () => {
      try {
        const s = await discoverStatus(runId);
        if (!live) return;
        setStatus(s);
        if (s.status === 'running') timer.current = window.setTimeout(tick, 2000);
      } catch {
        if (live) timer.current = window.setTimeout(tick, 4000);
      }
    };
    tick();
    return () => {
      live = false;
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [runId]);

  const floor = 1 / k;
  const belowFloor = commitment <= floor;
  const alpha = def?.alpha_table?.[String(k)]?.[commitment.toFixed(2)];

  const start = async () => {
    setErr(null); setStatus(null);
    try {
      const r = await discoverStart({ k, commitment, target_sim: targetSim, batch, total });
      if (r.error) { setErr(r.error); return; }
      setRunId(r.run_id);
    } catch (e: any) { setErr(String(e.message || e)); }
  };

  const tip = (key: string) => def?.provenance?.[key] || '';

  return (
    <div style={BOX}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}
           onClick={() => setOpen(!open)}>
        <span style={{ color: '#888' }}>{open ? '▾' : '▸'}</span>
        <strong>Discovery</strong>
        <span style={{ color: '#666', fontSize: 11 }}>
          sample at measured commitment, restart before the simplex exhausts
        </span>
      </div>

      {open && (
        <div style={{ marginTop: 8 }}>
          <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', alignItems: 'flex-end' }}>
            <label title={tip('k')}>
              <div style={{ color: '#888', fontSize: 10 }}>prompts (k)</div>
              <input style={NUM} type="number" min={2} max={8} value={k}
                     onChange={(e) => setK(+e.target.value)} />
            </label>
            <label title={tip('commitment')}>
              <div style={{ color: '#888', fontSize: 10 }}>commitment E[max w]</div>
              <input style={{ ...NUM, borderColor: belowFloor ? '#c33' : '#444' }}
                     type="number" step={0.05} min={0.05} max={0.95} value={commitment}
                     onChange={(e) => setCommitment(+e.target.value)} />
            </label>
            <label title={tip('target_sim')}>
              <div style={{ color: '#888', fontSize: 10 }}>prompt spread</div>
              <input style={NUM} type="number" step={0.01} min={0.2} max={0.95} value={targetSim}
                     onChange={(e) => setTargetSim(+e.target.value)} />
            </label>
            <label title={tip('batch')}>
              <div style={{ color: '#888', fontSize: 10 }}>images / simplex</div>
              <input style={NUM} type="number" min={20} max={2000} value={batch}
                     onChange={(e) => setBatch(+e.target.value)} />
            </label>
            <label>
              <div style={{ color: '#888', fontSize: 10 }}>total images</div>
              <input style={NUM} type="number" min={1} max={100000} value={total}
                     onChange={(e) => setTotal(+e.target.value)} />
            </label>
            <button onClick={start}
                    disabled={belowFloor || status?.status === 'running'}
                    style={{ padding: '4px 12px', background: belowFloor ? '#333' : '#2a5',
                             color: '#fff', border: 'none', borderRadius: 3,
                             cursor: belowFloor ? 'not-allowed' : 'pointer' }}>
              {status?.status === 'running' ? 'running…' : 'Start'}
            </button>
            {status?.status === 'running' && runId && (
              <button onClick={() => discoverCancel(runId)}
                      style={{ padding: '4px 10px', background: '#833', color: '#fff',
                               border: 'none', borderRadius: 3, cursor: 'pointer' }}>
                Cancel
              </button>
            )}
          </div>

          <div style={{ marginTop: 6, fontSize: 11, color: belowFloor ? '#f66' : '#777' }}>
            {belowFloor
              ? `commitment must exceed the k=${k} floor of ${floor.toFixed(3)} — a ${k}-prompt `
                + `mixture cannot be more balanced than 1/k`
              : <>Dirichlet α {alpha ? `≈ ${alpha}` : '(solved server-side)'} · uniform sampling
                  would give {(k === 3 ? 0.611 : k === 4 ? 0.521 : k === 5 ? 0.456 : k === 8 ? 0.339
                  : 1 / k).toFixed(3)} at this k, which is why this never samples uniformly</>}
          </div>

          {err && <div style={{ marginTop: 6, color: '#f66', fontSize: 11 }}>{err}</div>}

          {status && (
            <div style={{ marginTop: 10 }}>
              <div style={{ display: 'flex', gap: 16, fontSize: 12 }}>
                <span>{status.generated}/{status.total}</span>
                <span style={{ color: '#8cf' }}>
                  diversity {status.diversity != null ? status.diversity.toFixed(3) : '—'}
                </span>
                <span style={{ color: '#888' }}>{status.simplices.length} simplices</span>
                <span style={{ color: status.status === 'error' ? '#f66' : '#6a6' }}>
                  {status.status}
                </span>
              </div>
              {status.error && (
                <div style={{ color: '#f66', fontSize: 11, marginTop: 4 }}>{status.error}</div>
              )}
              <div style={{ marginTop: 6, maxHeight: 150, overflowY: 'auto' }}>
                {status.simplices.map((s) => (
                  <div key={s.index} style={{ fontSize: 11, color: '#aaa', padding: '2px 0',
                                              borderTop: '1px solid #222' }}>
                    <span style={{ color: '#fff' }}>#{s.index}</span>{' '}
                    <span style={{ color: '#777' }}>sim {s.mean_sim.toFixed(3)} · α {s.alpha.toFixed(2)}
                      {' '}· {s.done}/{s.n_points}
                      {s.diversity != null && <> · div {s.diversity.toFixed(3)}</>}
                    </span>
                    <div style={{ color: '#889' }}>{s.prompts.join(' · ')}</div>
                  </div>
                ))}
              </div>
              {status.generated > 0 && runId && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 2, marginTop: 8,
                              maxHeight: 220, overflowY: 'auto' }}>
                  {Array.from({ length: Math.min(status.generated, 240) }, (_, i) => (
                    <img key={i} src={`/api/discover/${runId}/${i}.jpg`} width={54} height={54}
                         loading="lazy" style={{ objectFit: 'cover', borderRadius: 2 }} />
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
