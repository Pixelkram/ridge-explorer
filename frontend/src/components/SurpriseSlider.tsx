// The surprise slider: -1 = calm interiors, +1 = boundary hybrids.
// Measured basis (search_problem E31/E33/E62): novelty is monotone in boundary
// degree at a coherence price; this control spans that axis with one scalar.
import { useState } from 'react';
import { surpriseSample } from '../api/client';

export default function SurpriseSlider(props: {
  jobId: string | null;
  onSample: (cells: [number, number][]) => void;
}) {
  const [surprise, setSurprise] = useState(1.0);
  const [busy, setBusy] = useState(false);
  const label = surprise > 0.33 ? 'surprise' : surprise < -0.33 ? 'calm' : 'mixed';
  return (
    <div>
      <label style={{ fontSize: 10, color: '#888' }}>
        Surprise {surprise >= 0 ? '+' : ''}{surprise.toFixed(2)} ({label})
      </label>
      <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
        <input type="range" min={-1} max={1} step={0.05} value={surprise}
               onChange={e => setSurprise(+e.target.value)}
               style={{ display: 'block', width: 140 }} />
        <button
          disabled={!props.jobId || busy}
          style={{ fontSize: 10, padding: '2px 8px', background: '#0f3460',
                   color: '#fff', border: '1px solid #333', borderRadius: 4,
                   cursor: props.jobId && !busy ? 'pointer' : 'default' }}
          onClick={async () => {
            if (!props.jobId) return;
            setBusy(true);
            try {
              const r = await surpriseSample(props.jobId, surprise, 12);
              props.onSample(r.cells);
            } finally { setBusy(false); }
          }}>
          {busy ? '…' : 'draw 12'}
        </button>
      </div>
    </div>
  );
}
