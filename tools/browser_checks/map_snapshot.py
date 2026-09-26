"""Start a cascade and snapshot the map at three moments, rendering the status payload
with the same projection + colour logic as CascadeMap.tsx."""
import json
import math
import time
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, Rectangle

BASE = "http://localhost:8001"
OUT = "/tmp/claude-1000/-home-student-ai-jspace/090e2078-48fc-4f5b-bdcf-cb67d0789297/scratchpad"


def call(path, body=None):
    if body is not None:
        req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
    else:
        req = urllib.request.Request(BASE + path)
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def div_color(d):
    if d is None:
        return (0x56 / 255, 0x5e / 255, 0x93 / 255)
    t = min(1.0, max(0.0, d / 0.7))
    lerp = lambda a, b: (a + (b - a) * t) / 255
    return (lerp(75, 233), lerp(90, 69), lerp(160, 96))


def verts(k):
    return [(math.cos(-math.pi / 2 + 2 * math.pi * i / k),
             math.sin(-math.pi / 2 + 2 * math.pi * i / k)) for i in range(k)]


def project(w, vs):
    x = sum(wi * v[0] for wi, v in zip(w, vs))
    y = sum(wi * v[1] for wi, v in zip(w, vs))
    return x, -y          # flip y for screen-like orientation


def snapshot(s, fname, note):
    k = len(s["prompts"])
    vs = verts(k)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor("#0d0d20")
    fig.patch.set_facecolor("#0d0d20")
    poly = MplPoly([(v[0], -v[1]) for v in vs], closed=True,
                   facecolor="#151538", edgecolor="#34346a", lw=1.5)
    ax.add_patch(poly)
    for c in s["chords"]:
        x1, y1 = project(c["a"], vs)
        x2, y2 = project(c["b"], vs)
        ax.plot([x1, x2], [y1, y2], color="#4a4f8f", lw=2, zorder=2)
    pts = s["points"]
    divs = s["point_divs"]
    for i, w in enumerate(pts):
        x, y = project(w, vs)
        d = divs[i] if i < len(divs) else None
        ax.scatter([x], [y], s=26 if d is not None else 16, color=div_color(d),
                   alpha=0.9 if d is not None else 0.55, zorder=3, linewidths=0)
    for p in s["patches"]:
        c = next((x for x in s["crossings"] if x["cid"] == p["cid"]), None)
        if c:
            x, y = project(c["weights"], vs)
            r = Rectangle((x - 0.055, y - 0.055), 0.11, 0.11, angle=45,
                          rotation_point="center", fill=False, lw=2.2,
                          edgecolor="#4ecca3" if p["significant"] else "#8a93b8",
                          linestyle="--" if p["exploration"] else "-", zorder=5)
            ax.add_patch(r)
    for c in s["crossings"]:
        x, y = project(c["weights"], vs)
        scored = c["b"] is not None
        rad = (4.5 + 6 * min(1, max(0, c["b"])) if scored else 4.5) * 6
        col = "#7d84c8" if not scored else ("#4ecca3" if c["significant"] else "#8a93b8")
        ax.scatter([x], [y], s=rad * 2.2, color=col, zorder=6,
                   edgecolors="#0d0d20", linewidths=1.2)
    for i, v in enumerate(vs):
        ax.scatter([v[0]], [-v[1]], s=42, color="#dfe3ff", zorder=7)
        ax.annotate(s["prompts"][i][:26], (v[0] * 1.12, -v[1] * 1.12),
                    color="#aeb6dd", fontsize=9, ha="center")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    divs_n = sum(1 for d in divs if d is not None)
    ax.set_title(f"{note} — phase {s['phase']} · {s['generated']} imgs · "
                 f"{divs_n} coloured · {len(s['crossings'])} crossings "
                 f"({sum(c['significant'] for c in s['crossings'])} certified)",
                 color="#dfe3f5", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=110, facecolor="#0d0d20")
    plt.close(fig)
    print(f"saved {fname}: {note}", flush=True)


while True:
    try:
        call("/health")
        break
    except Exception:
        time.sleep(3)
r = call("/api/cascade/start", {"k": 3, "n_chords": 10, "n_patches": 3, "seed": 7})
rid = r["run_id"]
print("run:", rid, flush=True)

taken = set()
t0 = time.time()
while True:
    s = call(f"/api/cascade/{rid}/status")
    el = time.time() - t0
    ph = s["phase"]
    if ph == "chords" and s["generated"] > 30 and "early" not in taken:
        snapshot(s, "map_t1_early.png", f"t+{el:.0f}s mid-isolation")
        taken.add("early")
    if ph in ("bisect", "score") and "mid" not in taken:
        snapshot(s, "map_t2_mid.png", f"t+{el:.0f}s {ph}")
        taken.add("mid")
    if s["status"] != "running":
        snapshot(s, "map_t3_final.png", f"t+{el:.0f}s complete")
        break
    time.sleep(4)
print("DONE", flush=True)
