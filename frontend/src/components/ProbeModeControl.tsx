/**
 * "Probe mode" control shared by the Cascade panel and the Mixing desk (desktop only): the mode
 * selector, and the staged readout's step t and flag threshold θ in a collapsible "advanced"
 * section with the one-line explanation. Defaults come from ../probeDefaults (one constant).
 */
import { useState } from 'react';
import {
  PROBE_MODES, STAGED_EXPLAINER, clampStagedT, clampStagedTheta, type ProbeMode,
} from '../probeDefaults';

const SEL: React.CSSProperties = {
  background: '#0a0a1a', color: '#8a9', border: '1px solid #444', borderRadius: 3,
  padding: '2px 4px', fontSize: 12,
};
const NUM: React.CSSProperties = {
  width: 52, background: '#0a0a1a', color: '#fff', border: '1px solid #444',
  borderRadius: 3, padding: '2px 4px', fontSize: 12,
};

export default function ProbeModeControl({ mode, t, theta, steps, onMode, onT, onTheta }: {
  mode: ProbeMode; t: number; theta: number; steps: number;
  onMode: (m: ProbeMode) => void; onT: (t: number) => void; onTheta: (th: number) => void;
}) {
  const [adv, setAdv] = useState(false);
  const staged = mode === 'staged';
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
      <label title={STAGED_EXPLAINER} style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
        probe mode
        <select value={mode} style={SEL} onChange={(e) => onMode(e.target.value as ProbeMode)}>
          {PROBE_MODES.map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}
        </select>
      </label>
      <span role="button" tabIndex={0} onClick={() => setAdv(!adv)}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setAdv(!adv); }}
            title="the staged readout's step t and flag threshold θ"
            style={{ color: '#8a9', cursor: 'pointer', userSelect: 'none', fontSize: 11 }}>
        {adv ? '▾' : '▸'} advanced
      </span>
      {adv && (
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8, flexWrap: 'wrap',
                       color: staged ? undefined : '#667' }}>
          <label title={`readout step of the full ${steps}-step schedule (1 … ${steps - 1}); `
                        + 'a later step reads closer to the final image and costs more'}>
            t <input style={NUM} type="number" min={1} max={Math.max(1, steps - 1)} value={t}
                     disabled={!staged}
                     onChange={(e) => onT(clampStagedT(Number(e.target.value), steps))} /> / {steps}
          </label>
          <label title="a segment is finished when its two x̂0 readouts are at least θ apart (1 − cos); lower θ = more finished, higher recall, more cost">
            θ <input style={NUM} type="number" min={0.01} max={1} step={0.01} value={theta}
                     disabled={!staged}
                     onChange={(e) => onTheta(clampStagedTheta(Number(e.target.value)))} />
          </label>
          <span style={{ color: '#667', fontSize: 11 }}>{STAGED_EXPLAINER}</span>
        </span>
      )}
    </span>
  );
}
