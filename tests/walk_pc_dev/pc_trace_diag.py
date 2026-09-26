"""Dev-set diagnostic: continuation walks from every crossing, both directions; dump corrector traces."""
import json, time, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
base = 'http://127.0.0.1:8001'
def call(p, b=None):
    r = urllib.request.Request(base + p, data=json.dumps(b).encode() if b is not None else None,
                               headers={"Content-Type": "application/json"}, method="POST" if b is not None else "GET")
    return json.loads(urllib.request.urlopen(r, timeout=120).read())
sets = json.load(open("/home/student/ai/search_problem/outputs/h08_vector_uses/1_highdim/sets.json"))
prompts = next(s for s in sets if s["name"] == sys.argv[1])["prompts"]
rid = call("/api/cascade/start", {"k": 4, "prompts": prompts, "n_chords": 10, "n_patches": 2, "seed": int(sys.argv[2]), "steps": 8})["run_id"]
while call(f"/api/cascade/{rid}/status")["status"] == "running": time.sleep(5)
st = call(f"/api/cascade/{rid}/status"); print("run", rid, len(st["crossings"]), "crossings", flush=True)
def go(xd):
    x, d = xd
    w = call(f"/api/cascade/{rid}/walk", {"cid": x["cid"], "direction": d, "n_steps": 6, "sig_mode": "relative", "mode": sys.argv[3]})
    while True:
        s = call(f"/api/cascade/{rid}/walk/{w['walk_id']}")
        if s["status"] != "running": break
        time.sleep(2)
    print(x["cid"], d, s["status"], len(s["steps"]), s["images"], (s["notes"] or ["budget"])[-1], s.get("error") or "", flush=True)
    return {"cid": x["cid"], "dir": d, "sig": x["significant"], "walk": s}
with ThreadPoolExecutor(4) as ex:
    rows = list(ex.map(go, [(x, d) for x in st["crossings"] for d in (1, -1)]))
json.dump({"run": rid, "rows": rows}, open(sys.argv[4], "w"), indent=1)
print("saved", flush=True)
