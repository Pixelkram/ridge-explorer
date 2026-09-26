"""Drive the confirmatory collection of PREREG_walk_continuation.md: C1-C3 always, then C4, C5 while the pooled
primary population (units from significant origins) has fewer than 30 units; D1 last (descriptive)."""
import json, subprocess, sys
RUNS = [("C1", "k4_T0", 7), ("C2", "k4_T2", 42), ("C3", "k4_T3", 42), ("C4", "k4_T1", 42), ("C5", "k4_T4", 42)]
def collect(label, pset, seed):
    subprocess.run([sys.executable, "tests/walk_pc_collect.py", "--set", pset, "--seed", str(seed), "--workers", "3",
                    "--out", f"tests/walk_pc_raw_{label}.json"], check=True)
    d = json.load(open(f"tests/walk_pc_raw_{label}.json"))
    return sum(1 for u in d["units"] if u["significant"])
n = 0
for i, (label, pset, seed) in enumerate(RUNS):
    if i >= 3 and n >= 30:
        print(f"stopping rule: {n} primary units after {i} runs", flush=True)
        break
    got = collect(label, pset, seed)
    n += got
    print(f"== {label} ({pset}, seed {seed}): {got} primary units; pooled {n}", flush=True)
collect("D1", "k4_T0", 42)
print("== ALL DONE", flush=True)
