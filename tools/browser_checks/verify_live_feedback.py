"""Verify the new pin/refine feedback signals appear in live status polls."""
import json
import time
import urllib.request

BASE = "http://localhost:8001"


def call(path, body=None):
    if body is not None:
        req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
    else:
        req = urllib.request.Request(BASE + path)
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


r = call("/api/cascade/start", {
    "k": 3, "prompts": ["a lighthouse in a storm", "a bowl of ramen",
                        "a paper crane"],
    "n_chords": 10, "n_patches": 2, "seed": 5})
rid = r["run_id"]
print("run", rid)

seen_open = seen_pinned = seen_partial = False
partial_snaps = []
last_phase = None
while True:
    s = call(f"/api/cascade/{rid}/status")
    if s["phase"] != last_phase:
        print(f"phase -> {s['phase']}  (imgs {s['generated']})")
        last_phase = s["phase"]
    bws = [c["bracket_w"] for c in s["crossings"] if c["bracket_w"] is not None]
    if bws and not seen_open and max(bws) > 0.014:
        seen_open = True
        print(f"  OPEN brackets visible: n={len(bws)} max_w={max(bws):.4f} "
              f"(reticles would show)")
    if bws and not seen_pinned and s["phase"] in ("score", "patches", "done") \
            and max(bws) <= 0.014:
        seen_pinned = True
        print(f"  all brackets PINNED: max_w={max(bws):.4f} (reticles collapsed)")
    for p in s.get("patches", []):
        cells = [t for row in p["grid"] for t in row]
        done = sum(1 for t in cells if t >= 0)
        if 0 < done < len(cells):
            seen_partial = True
            partial_snaps.append((p["region"], done, len(cells)))
    if s["status"] != "running":
        print(f"finished: {s['status']}, imgs {s['generated']}, "
              f"{len(s['crossings'])} crossings")
        break
    time.sleep(3)

if partial_snaps:
    regs = sorted({r0 for r0, _, _ in partial_snaps})
    print(f"  LIVE patch fills observed: {len(partial_snaps)} partial snapshots "
          f"across regions {regs}, e.g. {partial_snaps[:4]}")
print(f"VERDICT: open_brackets={seen_open} pinned={seen_pinned} "
      f"live_patch_fill={seen_partial}")
