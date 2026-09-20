// Surviving a phone lock.
//
// Mobile browsers evict backgrounded tabs to reclaim memory (iOS Safari within a minute
// or two), and the restored page is a COLD LOAD: React state is gone, so a running hike
// looked like it had vanished and the form came back blank. Nothing was actually lost —
// the backend keeps hike state in memory and mirrors it to hike.json, and
// GET /hike/{id}/status serves it either way — the browser just forgot which run it was
// watching.
//
// So persist the few identifiers needed to pick the thread back up, and nothing else.
// Deliberately NOT cached: statuses, images, cell arrays. They are large, they go stale,
// and the server is the only honest source for them.
const KEY = 'ridge.resume.v1';

export interface Resume {
  hikeId?: string | null;
  jobId?: string | null;
  promptA?: string;
  promptB?: string;
  promptC?: string;
  ts?: number;
}

// A hike id older than this is not worth resuming: the server may have restarted and
// dropped it, and silently polling a dead id was its own bug (fixed separately).
const MAX_AGE_MS = 12 * 60 * 60 * 1000;

export function loadResume(): Resume {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return {};
    const r = JSON.parse(raw) as Resume;
    if (r.ts && Date.now() - r.ts > MAX_AGE_MS) return { promptA: r.promptA, promptB: r.promptB, promptC: r.promptC };
    return r;
  } catch {
    return {};                     // private mode, quota, or corrupt value
  }
}

export function saveResume(patch: Resume): void {
  try {
    const cur = loadResume();
    localStorage.setItem(KEY, JSON.stringify({ ...cur, ...patch, ts: Date.now() }));
  } catch {
    /* storage unavailable: resuming is a convenience, never a requirement */
  }
}

/** Run `fn` when the tab comes back to the foreground, and once on mount. */
export function onResumeVisible(fn: () => void): () => void {
  const h = () => { if (document.visibilityState === 'visible') fn(); };
  document.addEventListener('visibilitychange', h);
  window.addEventListener('pageshow', h);   // iOS back-forward cache restores fire this
  return () => {
    document.removeEventListener('visibilitychange', h);
    window.removeEventListener('pageshow', h);
  };
}
