#!/usr/bin/env bash
# Detached: restart pool (GPUs 0-3,5,6), start a k=4 cascade, run the A/B with the given arm. Log: logs/walk_ab_<arm>.log
set -u; cd "$(dirname "$0")/.."; ARM=${1:-relative}; STEPS=${2:-6}
CUDA_VISIBLE_DEVICES=0,1,2,3,5,6 bash tools/probe_server.sh 6 || exit 1
RID=$(python3 - <<'PY'
import json, urllib.request, time
base='http://127.0.0.1:8001'
def call(p,b=None):
    r=urllib.request.Request(base+p, data=json.dumps(b).encode() if b is not None else None, headers={"Content-Type":"application/json"}, method="POST" if b is not None else "GET"); return json.loads(urllib.request.urlopen(r, timeout=120).read())
prompts=["a photograph of an airplane","a photograph of a giraffe","a surrealist oil painting of a roll of toilet paper","a photograph of sushi on a plate"]
rid=call("/api/cascade/start", {"k":4,"prompts":prompts,"n_chords":10,"n_patches":2,"seed":42,"steps":8})["run_id"]; t0=time.time()
while call(f"/api/cascade/{rid}/status")["status"]=="running" and time.time()-t0<1500: time.sleep(5)
print(rid)
PY
)
echo "run $RID" ; echo "$RID" > tests/last_cascade_run.txt
python3 tests/walk_ab.py --run "$RID" --steps "$STEPS" --arm "$ARM"
echo "AB DONE"
