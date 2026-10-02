"""Offline checks of the Metropolis sampler's LIVE half (round log + the five controls).

Companion to tests/metro_test.py, which checks the arithmetic and the one-shot run; this file
checks what the panel drives while a run is going round. No GPU: the synthetic two-basin field
and the `cascade.evaluate` stub are imported from metro_test rather than copied, so the two
suites cannot drift apart about what the field is.

The trick throughout parts (i)-(iv) is that the stub IS the panel: `cascade.evaluate` is called
once per round, so a control requested from inside it lands at the next round -- which is both
the cheapest way to drive the loop deterministically and the exact ordering the real panel
gets (the loop reads every control at the START of a round, never inside the batch in flight).

  (i)   ROUND LOG -- one row per COMPLETED round and no row for a round cancelled mid-batch;
        the eight documented fields plus the sigma/beta the round ran under; the accounting
        identities against `summary()` (proposals, acceptances, cost, wall time); and
        `last_events` / `recent_events`: every proposal recorded, out-of-simplex ones with
        S_prop None, the ring capped at metro.EVENTS_RING rounds.
  (ii)  PARAMS -- a change requested during round r is in force at r+1 and not before,
        measured two ways: the sigma each round's dispatch actually saw, and the proposal
        STEP LENGTH in that round's events. The run records it (`params_changed`, a note
        naming the round, `param_change_rounds` from the log alone).
  (iii) ADD A CHAIN -- joins at the next round with its initial energy measured in THAT
        round's batch (checked as an exact probe count, so a second batch cannot hide in it),
        id positional, `joined_round` set, not counted as a sample, proposing from the round
        after.
  (iv)  STOP A CHAIN -- frozen from the next round: no proposals, no events, samples kept,
        and the rest of the chains unaffected.
  (v)   PAUSE/RESUME -- the loop really blocks (two readings 0.4 s apart with nothing
        dispatched and no image generated), stays `status == "running"` while paused, resumes
        where it stopped, and a cancel issued WHILE paused still gets the thread out.
  (vi)  ROUTER -- every new route through TestClient: 404 unknown run, 409 on a run whose
        loop has stopped, 422 for out-of-bounds params, 400 for an empty params body and for
        a recipe that is not one, the add-chain cost refusal against MAX_IMAGE_EQ, 404 for an
        unknown chain, and the live fields on /status.
  (vii) THE SEED PHASE, which is the first minute or two of a real run and used to publish
        nothing: `seed_chords` as the chords are drawn, `seed_probes` as the probes ARRIVE
        (the `on_arrival` hook, with the divergence to the chord neighbour filled in once both
        of a pair exist), `seed_crossings` as pairs clear COS_T -- checked against the ordered
        pass that has the last word on them, which must find the same SET; and the chains
        published at step 0, every one at its seed with S null, BEFORE the batch that measures
        them. Plus the three fields on /status, including the 4,000-point stride.

Run: python tests/metro_live_test.py
"""
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backend.services import metro                      # noqa: E402
from backend.services import cascade as cs              # noqa: E402
from metro_test import _two_basin, _stub_evaluate       # noqa: E402

K = 4
fails = []


def check(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(label)


def _mk_run(chains=10, chain_steps=8, m=2, sigma=0.03, beta=1.0, n_seed_chords=14, seed=5):
    return metro.MetroRun(
        run_id="m", prompts=[f"p{i}" for i in range(K)], seed=seed, steps=8, height=64,
        width=64, guidance_scale=3.5, beta=beta, sigma=sigma, m=m, chains=chains,
        chain_steps=chain_steps, seed_mode="iur", n_seed_chords=n_seed_chords)


def _drive(run, field, panel=None):
    """Run the loop with `cascade.evaluate` stubbed, letting `panel(round, run)` act after
    each round's dispatch -- i.e. exactly where the real panel acts, between rounds.

    Returns (dispatches, kernels): one (label, n_weights, steps) per dispatch as metro_test's
    stub records it, and one (label, sigma, beta) per dispatch -- the kernel the loop was
    actually holding when it asked for that batch, which is what "applied next round" means.
    """
    dispatches, kernels = [], []
    stub = _stub_evaluate(run, field, dispatches)

    def driven(app, r, pool, weights, seed, label, ctl=None, on_arrival=None, steps=None):
        kernels.append((label, float(run.sigma), float(run.beta)))
        out = stub(app, r, pool, weights, seed, label, ctl, on_arrival, steps)
        if panel is not None and _mcmc_round(label) is not None:
            panel(_mcmc_round(label), run)
        return out

    orig = cs.evaluate
    cs.evaluate = driven
    try:
        metro.run_metro(None, run, None)
    finally:
        cs.evaluate = orig
    return dispatches, kernels


def _mcmc_round(label):
    """The round number a dispatch label names, or None for the seed/init/full phases."""
    return int(label[5:]) if label.startswith("metro") and label[5:].isdigit() else None


def _batch_of(dispatches):
    """round -> probes in that round's single batch."""
    return {_mcmc_round(lab): n for lab, n, _st in dispatches if _mcmc_round(lab) is not None}


def _events_at(run, r):
    """One round's events out of the ring (the ring must still hold it)."""
    for b in run.recent_events:
        if b["round"] == r:
            return b["events"]
    return None


# ------------------------------------------------------------------- (i) the per-round log
def part1():
    print("\n(i) the round log and the per-chain events")
    field, _p = _two_basin()
    m, chains, L = 2, 10, 8
    run = _mk_run(chains=chains, chain_steps=L, m=m)
    _drive(run, field)
    s = run.summary()
    log = run.rounds

    check(len(log) == L and [r["round"] for r in log] == list(range(1, L + 1)),
          "one row per round, numbered from 1", f"{len(log)} rows for {L} rounds")
    want = {"round", "n_proposed", "n_accepted", "acc_rate", "mean_S_states",
            "mean_S_accepted", "cost_so_far", "t_wall", "sigma", "beta"}
    check(all(set(r) == want for r in log),
          "every row carries the eight documented fields plus the kernel it ran under",
          f"{sorted(log[0])}")
    check(sum(r["n_proposed"] for r in log) == s["n_propose"]
          and sum(r["n_accepted"] for r in log) == s["n_samples"],
          "the rows add up to the run: proposals and acceptances",
          f"{sum(r['n_proposed'] for r in log)} proposals = {s['n_propose']}, "
          f"{sum(r['n_accepted'] for r in log)} accepted = {s['n_samples']} samples")
    check(all(r["acc_rate"] is not None
              and abs(r["acc_rate"] - r["n_accepted"] / r["n_proposed"]) < 1e-12
              for r in log),
          "acc_rate is this round's n_accepted / n_proposed, with out-of-simplex proposals "
          "counted in the denominator exactly as the summary's acceptance counts them")
    per_round = {}
    for x in run.samples:
        per_round[x["step"]] = per_round.get(x["step"], 0) + 1
    check(all(r["n_accepted"] == per_round.get(r["round"], 0) for r in log),
          "a row's n_accepted is exactly the samples stamped with that round")
    check(all(a["cost_so_far"] <= b["cost_so_far"] for a, b in zip(log, log[1:]))
          and abs(log[-1]["cost_so_far"] - run.cost_image_eq) < 1e-9,
          "cost_so_far only grows, and the last row is the run's whole bill",
          f"{log[-1]['cost_so_far']:.1f} = {run.cost_image_eq:.1f} image-eq")
    check(all(a["t_wall"] <= b["t_wall"] for a, b in zip(log, log[1:]))
          and log[0]["t_wall"] >= 0.0,
          "t_wall is monotone seconds since the run started",
          f"{log[0]['t_wall']:.3f} -> {log[-1]['t_wall']:.3f} s")
    live_now = [c["s"] for c in run.chains_state if c["s"] is not None]
    check(abs(log[-1]["mean_S_states"] - float(np.mean(live_now))) < 1e-12,
          "mean_S_states of the last row is the mean energy of the chain SET as it ended",
          f"{log[-1]['mean_S_states']:.4f} over {len(live_now)} chains")
    acc_last = [x["s"] for x in run.samples if x["step"] == L]
    check((log[-1]["mean_S_accepted"] is None and not acc_last)
          or abs(log[-1]["mean_S_accepted"] - float(np.mean(acc_last))) < 1e-12,
          "mean_S_accepted is over just that round's accepted proposals",
          f"{log[-1]['mean_S_accepted']} over {len(acc_last)} accepted")
    check(all(r["sigma"] == 0.03 and r["beta"] == 1.0 for r in log)
          and metro.param_change_rounds(log) == [] and run.params_changed is False,
          "an undriven run has one kernel in every row and no change recorded")

    # --- the events
    ev = run.last_events
    check(len(ev) == log[-1]["n_proposed"]
          and sum(1 for e in ev if e["accepted"]) == log[-1]["n_accepted"],
          "last_events holds the last round's proposals, with the accepted ones marked",
          f"{len(ev)} events, {sum(1 for e in ev if e['accepted'])} accepted")
    want_e = {"chain", "w_from", "w_prop", "S_from", "S_prop", "accepted", "image_idx_prop"}
    check(all(set(e) == want_e for e in ev), "each event has the documented seven fields",
          f"{sorted(ev[0])}")
    check(all(len(e["w_from"]) == K and len(e["w_prop"]) == K
              and metro.in_simplex(e["w_from"]) and e["S_from"] is not None for e in ev),
          "an event starts from a legal recipe with a known energy (a chain's state always "
          "is) and both vectors are k-long")
    allev = [e for b in run.recent_events for e in b["events"]]
    inside = [e for e in allev if metro.in_simplex(e["w_prop"])]
    check(all(e["S_prop"] is not None and e["image_idx_prop"] is not None for e in inside),
          "an in-simplex proposal carries the energy measured at it and the image rendered "
          "there", f"{len(inside)} inside over the ring")

    # out-of-simplex proposals are rare at the sigma of record, so they get their own run at
    # a sigma that makes them common
    wide = _mk_run(chains=10, chain_steps=4, m=1, sigma=0.15)
    wdisp, _k = _drive(wide, field)
    wev = [e for b in wide.recent_events for e in b["events"]]
    out = [e for e in wev if not metro.in_simplex(e["w_prop"])]
    check(out and all(e["S_prop"] is None and not e["accepted"]
                      and e["image_idx_prop"] is None for e in out),
          "at sigma 0.15 proposals DO leave the simplex, and each one is in the events as a "
          "rejection with nothing rendered -- the map draws it, so the acceptance rate on "
          "screen is explainable from the picture",
          f"{len(out)}/{len(wev)} outside at sigma 0.15")
    wbatch = _batch_of(wdisp)
    check(all(wbatch[r["round"]]
              == (1 + wide.m) * sum(1 for e in _events_at(wide, r["round"])
                                    if metro.in_simplex(e["w_prop"]))
              for r in wide.rounds),
          "and it costs nothing: a round's batch is (1 + m) probes per IN-SIMPLEX proposal "
          "only", f"batches {[wbatch[r['round']] for r in wide.rounds]} over "
          f"{[r['n_proposed'] for r in wide.rounds]} proposals")
    check(sum(1 for e in wev if not metro.in_simplex(e["w_prop"]))
          == sum(c["n_outside"] for c in wide.chains_state),
          "every out-of-simplex event is one the chain counters also counted",
          f"{len(out)} events = {sum(c['n_outside'] for c in wide.chains_state)} counted")
    heads = {c["chain"]: (tuple(c["weights"]), c["s"]) for c in run.chains_state}
    acc_ev = [e for e in ev if e["accepted"]]
    check(acc_ev and all(heads[e["chain"]] == (tuple(e["w_prop"]), e["S_prop"])
                         for e in acc_ev),
          "after an accepted event the chain's head IS the proposal -- which is what the map "
          "glides the head to", f"{len(acc_ev)} accepted events in the last round")
    check(len(run.recent_events) == metro.EVENTS_RING
          and [b["round"] for b in run.recent_events] == list(range(L - 4, L + 1))
          and run.recent_events[-1]["events"] is run.last_events,
          f"the ring keeps the last {metro.EVENTS_RING} rounds, newest last, ending at "
          f"last_events", f"{[b['round'] for b in run.recent_events]}")

    # a round cancelled mid-batch never completed, so it is not logged
    cut = _mk_run(chains=6, chain_steps=20, m=1)
    cdisp = []
    stub = _stub_evaluate(cut, field, cdisp)

    def cancelling(*a, **kw):
        res = stub(*a, **kw)
        if len(cdisp) >= 4:
            cut.status = "cancelled"          # the UI's Cancel, from inside a batch
        return res
    orig = cs.evaluate
    cs.evaluate = cancelling
    try:
        metro.run_metro(None, cut, None)
    finally:
        cs.evaluate = orig
    mcmc = [lab for lab, _n, _st in cdisp if _mcmc_round(lab) is not None]
    check(cut.status == "cancelled" and len(mcmc) > 1
          and len(cut.rounds) == len(mcmc) - 1
          and cut.rounds[-1]["round"] == cut.round_done == len(mcmc) - 1,
          "a round cancelled mid-batch is NOT logged: the log only ever holds rounds that "
          "finished", f"{len(mcmc)} MCMC dispatches, {len(cut.rounds)} rows, round_done "
          f"{cut.round_done}")


# ------------------------------------------------------------- (ii) a kernel change mid-run
def part2():
    print("\n(ii) sigma / beta changed mid-run apply at the START of the next round")
    field, _p = _two_basin()
    m, chains, L, at = 1, 12, 6, 2
    run = _mk_run(chains=chains, chain_steps=L, m=m)

    def panel(r, rn):
        if r == at:
            metro.request_params(rn, sigma=0.12, beta=1.5)

    _dispatches, kernels = _drive(run, field, panel)
    disp = {_mcmc_round(lab): (sg, bt) for lab, sg, bt in kernels
            if _mcmc_round(lab) is not None}
    check(all(disp[r] == (0.03, 1.0) for r in range(1, at + 1))
          and all(disp[r] == (0.12, 1.5) for r in range(at + 1, L + 1)),
          f"the change requested during round {at} is in force from round {at + 1}, and the "
          f"round it was asked for in finished under the old kernel",
          f"round {at}: {disp[at]}, round {at + 1}: {disp[at + 1]}")
    check([r["sigma"] for r in run.rounds] == [0.03] * at + [0.12] * (L - at)
          and [r["beta"] for r in run.rounds] == [1.0] * at + [1.5] * (L - at),
          "and the log says which kernel every round ran under",
          f"sigmas {[r['sigma'] for r in run.rounds]}")
    check(metro.param_change_rounds(run.rounds) == [at + 1],
          "param_change_rounds reads the change off the log alone -- one source of truth for "
          "the mark the panel puts on its sparkline",
          f"{metro.param_change_rounds(run.rounds)}")
    check(run.params_changed is True
          and any(f"round {at + 1}:" in n and "stationarity" in n for n in run.notes)
          and any("not a draw from one law" in n for n in run.notes),
          "the run records the change and what it costs, naming the round",
          next((n[:72] for n in run.notes if "stationarity" in n), "MISSING"))

    # the proposals themselves moved further: evidence the new sigma IS the kernel in use,
    # not merely the number on record
    step = {}
    for b in run.recent_events:
        if b["events"]:
            step[b["round"]] = float(np.mean(
                [np.linalg.norm(np.asarray(e["w_prop"]) - np.asarray(e["w_from"]))
                 for e in b["events"]]))
    before = [v for r, v in step.items() if r <= at]
    after = [v for r, v in step.items() if r > at]
    check(before and after and min(after) > max(before) * 1.8,
          "the proposal step length in the events jumps with sigma (0.03 -> 0.12), so the "
          "chain really is proposing from the new kernel",
          f"mean |dw| {np.mean(before):.4f} -> {np.mean(after):.4f}")
    check(run.sigma == 0.12 and run.beta == 1.5,
          "the run ends holding the kernel it was last given")

    # a request that changes nothing is not recorded as a change
    quiet = _mk_run(chains=6, chain_steps=4, m=1)
    _drive(quiet, field,
           lambda r, rn: metro.request_params(rn, sigma=0.03) if r == 1 else None)
    check(quiet.params_changed is False and metro.param_change_rounds(quiet.rounds) == []
          and not any("stationarity" in n for n in quiet.notes),
          "re-sending the sigma the run already has is not a kernel change, so a slider "
          "nudged back where it was leaves no caveat on the run")


# -------------------------------------------------------------- (iii) a chain added mid-run
def part3():
    print("\n(iii) a chain added mid-run joins at the next round")
    field, _p = _two_basin()
    m, chains, L, at = 2, 8, 6, 2
    run = _mk_run(chains=chains, chain_steps=L, m=m)
    w_new = [0.4, 0.3, 0.2, 0.1]
    asked = {}

    def panel(r, rn):
        if r == at:
            asked["cid"] = metro.queue_chain(rn, w_new)
            asked["n_before"] = len(rn.chains_state)

    dispatches, _kernels = _drive(run, field, panel)
    batch = _batch_of(dispatches)
    new = [c for c in run.chains_state if c["joined_round"]]
    check(len(new) == 1, "exactly one chain joined", f"{len(new)}")
    if len(new) != 1:
        return
    nc = new[0]

    check(asked["cid"] == chains and nc["chain"] == chains
          and run.chains_state[chains] is nc,
          "the id queue_chain predicted is the id the chain gets, and it is positional "
          "(chains_state is indexed by it, as the seeded chains are)",
          f"predicted {asked['cid']}, got {nc['chain']}")
    check(nc["joined_round"] == at + 1,
          f"it joins at round {at + 1}, the round after the one it was asked for in",
          f"joined_round {nc['joined_round']}")
    check(nc["seed_s"] is not None and nc["seed_image"] >= 0
          and np.allclose(nc["seed_weights"], w_new, atol=1e-9)
          and nc["weights"] == nc["seed_weights"] or nc["moved"],
          "its initial energy is measured at the recipe that was asked for -- a hand-placed "
          "recipe has no rendered image, so it has to be measured like any other state",
          f"S {nc['seed_s']:.4f} at {[round(v, 3) for v in nc['seed_weights']]}")

    # the exact probe arithmetic: the joining chain rode the SAME batch as that round's
    # proposals, so one more batch cannot be hiding in the round it joined at
    ev_join = _events_at(run, at + 1)
    inside = sum(1 for e in ev_join if metro.in_simplex(e["w_prop"]))
    check(batch[at + 1] == (1 + m) * (inside + 1),
          f"round {at + 1} is still ONE batch: (1 + m) probes per in-simplex proposal PLUS "
          f"(1 + m) for the joining chain's initial energy",
          f"{batch[at + 1]} probes = ({1 + m}) x ({inside} proposals + 1 joining)")
    check(not any(x["chain"] == chains and x["step"] == at + 1 for x in run.samples),
          "the joining state is a START state, not an accepted sample")
    check(all(e["chain"] != chains for b in run.recent_events if b["round"] <= at + 1
              for e in b["events"]),
          f"it proposes nothing up to and including round {at + 1}")
    check(any(e["chain"] == chains for b in run.recent_events if b["round"] > at + 1
              for e in b["events"]) and nc["n_propose"] == L - (at + 1),
          f"and proposes in every round from {at + 2} on",
          f"{nc['n_propose']} proposals over rounds {at + 2}..{L}")
    check(run.summary()["n_chains"] == chains + 1
          and len(run.chains_state) == asked["n_before"] + 1,
          "the run counts it", f"{run.summary()['n_chains']} chains")
    check(any("added by hand" in n and f"round {at + 1}" in n for n in run.notes),
          "and says on the record that a hand-placed chain is a choice in the sample",
          next((n[:70] for n in run.notes if "added by hand" in n), "MISSING"))


# -------------------------------------------------------------------- (iv) a chain stopped
def part4():
    print("\n(iv) a chain stopped mid-run freezes where it is")
    field, _p = _two_basin()
    m, chains, L, at = 1, 10, 8, 3
    run = _mk_run(chains=chains, chain_steps=L, m=m)
    frozen = {}

    def panel(r, rn):
        if r == at:
            # the busiest chain, so "its samples are kept" has something to keep
            tgt = max(rn.chains_state, key=lambda c: c["n_accept"])
            frozen.update(
                chain=int(tgt["chain"]), n_propose=int(tgt["n_propose"]),
                weights=list(tgt["weights"]), s=tgt["s"],
                samples=sum(1 for x in rn.samples if x["chain"] == tgt["chain"]),
                ok=metro.stop_chain(rn, tgt["chain"]))

    _drive(run, field, panel)
    cid = frozen["chain"]
    c = run.chains_state[cid]

    check(frozen["ok"] is True and c["stopped"] is True,
          "stop_chain found the chain and marked it stopped", f"chain {cid}")
    check(c["n_propose"] == frozen["n_propose"],
          f"it made no further proposal after round {at}",
          f"{frozen['n_propose']} proposals then, {c['n_propose']} at the end")
    check(c["weights"] == frozen["weights"] and c["s"] == frozen["s"],
          "its head stays exactly where it was frozen")
    check(sum(1 for x in run.samples if x["chain"] == cid) == frozen["samples"],
          "every state it had already accepted is kept -- stop is a freeze, not a delete",
          f"{frozen['samples']} samples kept")
    check(all(e["chain"] != cid for b in run.recent_events if b["round"] > at
              for e in b["events"]),
          "and it produces no events, so the map's animation leaves its head alone")
    check(run.summary()["n_chains_stopped"] == 1
          and run.summary()["n_chains_moving"]
          == sum(1 for x in run.chains_state if x["s"] is not None and not x["stopped"]),
          "the summary counts stopped and still-moving chains",
          f"{run.summary()['n_chains_stopped']} stopped, "
          f"{run.summary()['n_chains_moving']} moving")
    others = [x for x in run.chains_state if x["chain"] != cid and x["s"] is not None]
    check(others and all(x["n_propose"] == L for x in others),
          "the other chains kept proposing every round",
          f"{len(others)} chains x {L} rounds")
    check(not metro.stop_chain(run, 999),
          "stopping a chain that does not exist is False, not an exception")


# --------------------------------------------------------------------- (v) pause / resume
def part5():
    print("\n(v) pause blocks dispatch between rounds; resume continues; cancel still exits")
    field, _p = _two_basin()
    run = _mk_run(chains=6, chain_steps=60, m=1)
    dispatches = []
    stub = _stub_evaluate(run, field, dispatches)

    def slow(*a, **kw):
        time.sleep(0.01)                      # rounds short, but not instantaneous
        return stub(*a, **kw)

    orig = cs.evaluate
    cs.evaluate = slow
    th = threading.Thread(target=metro.run_metro, args=(None, run, None), daemon=True)
    try:
        th.start()
        end = time.monotonic() + 10
        while run.round_done < 3 and time.monotonic() < end:
            time.sleep(0.01)
        check(run.round_done >= 3, "the loop gets going", f"round {run.round_done}")

        run.paused = True
        time.sleep(0.3)                       # let the round already in flight finish
        a = (len(run.rounds), run.round_done, run.generated)
        time.sleep(0.4)
        b = (len(run.rounds), run.round_done, run.generated)
        check(a == b,
              "while paused nothing is dispatched: no round completes and no image is "
              "generated, so a paused run holds no GPU worker",
              f"(rounds, round_done, images) {a} then {b}")
        check(run.status == "running" and run.paused is True,
              "a paused run is still 'running' -- pause is not a status, so every "
              "`status == 'running'` check (cancel included) still means what it meant",
              f"status {run.status!r}, paused {run.paused}")
        check(th.is_alive(), "and the worker thread is still there, waiting")

        run.paused = False
        end = time.monotonic() + 10
        while run.round_done <= b[1] + 1 and time.monotonic() < end:
            time.sleep(0.01)
        check(run.round_done > b[1], "resume continues from the round it stopped at",
              f"{b[1]} -> {run.round_done}")
        check([r["round"] for r in run.rounds] == list(range(1, len(run.rounds) + 1)),
              "and the log has no gap across the pause",
              f"{len(run.rounds)} rows, last {run.rounds[-1]['round']}")
        gaps = [y["t_wall"] - x["t_wall"] for x, y in zip(run.rounds, run.rounds[1:])]
        check(max(gaps) > 0.3,
              "the pause shows up in the log as one long round-to-round gap rather than as a "
              "missing round", f"longest gap {max(gaps):.2f} s")

        # cancel WHILE paused must still get the thread out of _wait_while_paused
        run.paused = True
        time.sleep(0.2)
        run.status = "cancelled"
        th.join(timeout=5)
        check(not th.is_alive() and run.status == "cancelled",
              "a cancel issued while paused ends the run rather than deadlocking it",
              f"thread alive {th.is_alive()}")
    finally:
        run.status = "cancelled"
        run.paused = False
        cs.evaluate = orig
        th.join(timeout=5)


# ------------------------------------------------------------------------- (vi) the router
def part6():
    print("\n(vi) the live routes through TestClient (stubbed state, run_metro no-op)")
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
    except Exception as exc:                         # noqa: BLE001
        check(False, "TestClient available", str(exc))
        return
    from backend.routers import metro as router_mod

    class _NoCache:
        """A truthy stand-in ThumbnailStore keeps as-is, so nothing touches the disk."""
        def has(self, h):
            return False

        def save(self, h, data):
            pass

        def get_path(self, h):
            return None

    orig = metro.run_metro
    metro.run_metro = lambda app, run, pool: None     # the worker thread renders nothing
    try:
        app = FastAPI()
        app.include_router(router_mod.router)
        app.state.metro_runs = {}
        app.state.cascades = {}
        app.state.cache = _NoCache()
        app.state.gpu_pool = None
        c = TestClient(app, raise_server_exceptions=False)

        r = c.post("/api/metro/start", json={"prompts": ["a", "b", "c", "d"],
                                             "chains": 6, "chain_steps": 10})
        rid = r.json()["run_id"]
        run = app.state.metro_runs[rid]
        run.chains_state = [
            {"chain": i, "weights": [0.25] * 4, "s": 0.3 + 0.01 * i, "image": i,
             "seed_weights": [0.25] * 4, "seed_s": 0.3, "seed_image": i,
             "n_propose": 4, "n_accept": i % 3, "n_outside": 0, "moved": i % 3 > 0,
             "stopped": False, "joined_round": 0} for i in range(6)]
        run.rounds = [{"round": 1, "n_proposed": 6, "n_accepted": 3, "acc_rate": 0.5,
                       "mean_S_states": 0.31, "mean_S_accepted": 0.4, "cost_so_far": 9.0,
                       "t_wall": 1.5, "sigma": 0.03, "beta": 1.0}]
        run.last_events = [
            {"chain": 0, "w_from": [0.25] * 4, "w_prop": [0.3, 0.3, 0.2, 0.2],
             "S_from": 0.3, "S_prop": 0.42, "accepted": True, "image_idx_prop": 11},
            {"chain": 1, "w_from": [0.25] * 4, "w_prop": [0.6, 0.6, -0.1, -0.1],
             "S_from": 0.31, "S_prop": None, "accepted": False, "image_idx_prop": None}]
        run.recent_events = [{"round": 1, "events": run.last_events}]
        run.round_done = 1

        # --- /status serves the live half
        j = c.get(f"/api/metro/{rid}/status").json()
        check(len(j["rounds"]) == 1 and j["rounds"][0]["sigma"] == 0.03
              and j["rounds"][0]["acc_rate"] == 0.5
              and len(j["last_events"]) == 2 and j["last_events"][1]["S_prop"] is None
              and j["last_events"][0]["image_idx_prop"] == 11
              and len(j["recent_events"]) == 1 and j["recent_events"][0]["round"] == 1
              and len(j["recent_events"][0]["events"]) == 2,
              "/status carries the round log, last_events and the ring",
              f"{len(j['rounds'])} rows, {len(j['last_events'])} events")
        check(j["paused"] is False and j["params"] == {"sigma": 0.03, "beta": 1.0}
              and j["params_changed"] is False,
              "... plus the kernel in force and the paused flag", f"{j['params']}")
        check([x["acc_rate"] for x in j["chains_stats"]] == [0.0, 0.25, 0.5] * 2
              and all(x["stopped"] is False for x in j["chains_stats"])
              and all(x["joined_round"] == 0 for x in j["chains_stats"]),
              "... and the per-chain summary carries acc_rate, stopped and joined_round",
              f"{[x['acc_rate'] for x in j['chains_stats']]}")

        # --- params
        r = c.post(f"/api/metro/{rid}/params", json={"sigma": 0.08})
        check(r.status_code == 200 and r.json()["applies_at_round"] == 2
              and run.pending_params == {"sigma": 0.08, "beta": None},
              "/params queues the change for the next round",
              f"HTTP {r.status_code}: {r.text[:70]}")
        r = c.post(f"/api/metro/{rid}/params", json={"beta": 1.2, "sigma": 0.05})
        check(r.status_code == 200 and run.pending_params == {"sigma": 0.05, "beta": 1.2},
              "a second change before the round lands replaces the first, so a dragged "
              "slider costs one kernel change and not ten", f"{run.pending_params}")
        for bad, why in (({"sigma": 0.3}, "sigma past 0.2"), ({"sigma": 0.0}, "sigma 0"),
                         ({"beta": 2.0}, "beta 2.0"), ({"beta": 0.4}, "beta 0.4"),
                         ({"sigma": "big"}, "a sigma that is not a number")):
            r = c.post(f"/api/metro/{rid}/params", json=bad)
            check(r.status_code == 422, f"/params with {why} -> 422 (the start form's bounds)",
                  f"got HTTP {r.status_code}")
        r = c.post(f"/api/metro/{rid}/params", json={})
        check(r.status_code == 400 and "neither" in r.text,
              "/params with an empty body -> 400 rather than a silent no-op",
              f"HTTP {r.status_code}")

        # --- chains
        r = c.post(f"/api/metro/{rid}/chains", json={"w": [0.4, 0.3, 0.2, 0.1]})
        check(r.status_code == 200 and r.json()["chain"] == 6
              and r.json()["joins_at_round"] == 2 and len(run.pending_chains) == 1
              and np.allclose(run.pending_chains[0], [0.4, 0.3, 0.2, 0.1]),
              "/chains queues a chain at the recipe, with the id it will get",
              f"HTTP {r.status_code}: {r.text[:80]}")
        r = c.post(f"/api/metro/{rid}/chains", json={"w": [0.4, 0.3, 0.2, 0.1004]})
        check(r.status_code == 200
              and abs(float(np.sum(run.pending_chains[-1])) - 1.0) < 1e-12,
              "a recipe off by a slider's round-off is renormalised, not refused",
              f"sum {float(np.sum(run.pending_chains[-1])):.12f}")
        for bad, why, code in (
                ({"w": [0.5, 0.5]}, "fewer than 3 weights", 422),
                ({"w": [0.3, 0.3, 0.4]}, "the wrong k for this run", 400),
                ({"w": [0.6, 0.6, -0.1, -0.1]}, "a negative weight", 400),
                ({"w": [0.5, 0.5, 0.5, 0.5]}, "weights that do not sum to 1", 400)):
            r = c.post(f"/api/metro/{rid}/chains", json=bad)
            check(r.status_code == code, f"/chains with {why} -> {code}",
                  f"got HTTP {r.status_code}: {r.text[:70]}")
        n_queued = len(run.pending_chains)

        # the ceiling: 39 chains x 200 rounds at m=4 is legal and 40 is not, so a run started
        # at 39 must refuse the chain the start form would have refused
        r = c.post("/api/metro/start", json={"prompts": ["a", "b", "c"], "chains": 39,
                                             "chain_steps": 200, "m": 4})
        check(r.status_code == 200, "a run right under the ceiling starts",
              f"HTTP {r.status_code}: {r.text[:80]}")
        rid_full = r.json()["run_id"]
        full = app.state.metro_runs[rid_full]
        full.chains_state = [
            {"chain": i, "weights": [1 / 3] * 3, "s": 0.3, "image": -1,
             "seed_weights": [1 / 3] * 3, "seed_s": 0.3, "seed_image": -1, "n_propose": 0,
             "n_accept": 0, "n_outside": 0, "moved": False, "stopped": False,
             "joined_round": 0} for i in range(39)]
        r = c.post(f"/api/metro/{rid_full}/chains", json={"w": [0.5, 0.3, 0.2]})
        proj = metro.projected_cost(3, 4, 40, 200, 20, "iur", 4)[0]
        check(r.status_code == 400 and "ceiling" in r.text and f"{proj:.0f}" in r.text
              and not full.pending_chains,
              "one chain too many is refused with the projection, against the same ceiling "
              "the start form uses -- adding chains one click at a time must not get past a "
              "limit typing the number would have hit",
              f"HTTP {r.status_code}: {r.text[:120]}")

        # --- stop a chain
        r = c.post(f"/api/metro/{rid}/chains/2/stop")
        check(r.status_code == 200 and run.chains_state[2]["stopped"] is True
              and run.chains_state[1]["stopped"] is False
              and c.get(f"/api/metro/{rid}/status").json()["chains_stats"][2]["stopped"],
              "/chains/{id}/stop freezes exactly that chain, and /status says so",
              f"HTTP {r.status_code}")
        r = c.post(f"/api/metro/{rid}/chains/99/stop")
        check(r.status_code == 404 and "no chain 99" in r.text,
              "stopping a chain the run does not have -> 404",
              f"HTTP {r.status_code}: {r.text[:80]}")

        # --- pause / resume
        r = c.post(f"/api/metro/{rid}/pause")
        check(r.status_code == 200 and r.json()["paused"] is True and run.paused is True
              and run.status == "running"
              and c.get(f"/api/metro/{rid}/status").json()["paused"] is True,
              "/pause sets the flag and /status shows it, with the run still 'running'",
              f"HTTP {r.status_code}")
        r = c.post(f"/api/metro/{rid}/resume")
        check(r.status_code == 200 and r.json()["paused"] is False and run.paused is False,
              "/resume clears it", f"HTTP {r.status_code}")

        # --- the two refusals every live route shares
        for path in ("/api/metro/nope/params", "/api/metro/nope/chains",
                     "/api/metro/nope/chains/0/stop", "/api/metro/nope/pause",
                     "/api/metro/nope/resume"):
            body = ({"sigma": 0.05} if path.endswith("params")
                    else {"w": [0.4, 0.3, 0.3]} if path.endswith("chains") else None)
            r = c.post(path, json=body) if body is not None else c.post(path)
            check(r.status_code == 404 and "no metro run nope" in r.text,
                  f"{path[16:]} on an unknown run -> 404", f"got HTTP {r.status_code}")
        run.status = "complete"
        for path, body in ((f"/api/metro/{rid}/params", {"sigma": 0.05}),
                           (f"/api/metro/{rid}/chains", {"w": [0.4, 0.3, 0.2, 0.1]}),
                           (f"/api/metro/{rid}/chains/0/stop", None),
                           (f"/api/metro/{rid}/pause", None),
                           (f"/api/metro/{rid}/resume", None)):
            r = c.post(path, json=body) if body is not None else c.post(path)
            check(r.status_code == 409 and "round loop has stopped" in r.text,
                  f"{path.split(rid, 1)[1].lstrip('/')} on a finished run -> 409",
                  f"got HTTP {r.status_code}")
        check(len(run.pending_chains) == n_queued and run.paused is False,
              "and a refused control changed nothing on the run")

        # --- the export carries the history
        e = c.get(f"/api/metro/{rid}/samples.json").json()
        check(len(e["rounds"]) == 1 and e["rounds"][0]["beta"] == 1.0
              and e["params_changed"] is False,
              "samples.json exports the per-round log, so a stimulus set carries the kernel "
              "every state in it was drawn under", f"{len(e['rounds'])} rows")
    finally:
        metro.run_metro = orig


# ------------------------------------------------------- (vii) the seed phase, published live
def _drive_seed(run, field):
    """Run the loop with snapshots taken INSIDE the two phases that precede round 1.

    Returns a dict of readings nothing else can give: what `seed_chords` held when the seed
    batch was dispatched, what `seed_probes`/`seed_crossings` held after each arrival, the live
    crossing list the moment the survey ended (before the ordered pass replaces it), and the
    chain set as it stood when the initial-energy batch was dispatched.
    """
    snap = {"arrivals": [], "chords_at_dispatch": None, "live_crossings": None,
            "chains_at_init": None, "phase_at_init": None, "rounds_at_init": None}
    dispatches = []
    stub = _stub_evaluate(run, field, dispatches)

    def driven(app, r, pool, weights, seed, label, ctl=None, on_arrival=None, steps=None):
        if label.startswith("metroseed"):
            snap["chords_at_dispatch"] = [[list(a), list(b)] for a, b in run.seed_chords]
            inner = on_arrival

            def hook(li, gi, _inner=inner):
                _inner(li, gi)
                snap["arrivals"].append(
                    (len(run.seed_probes), len(run.seed_crossings), run.phase_done,
                     run.phase))
            on_arrival = hook
        if label.startswith("metroinit"):
            snap["chains_at_init"] = [dict(c) for c in run.chains_state]
            snap["phase_at_init"] = run.phase
            snap["rounds_at_init"] = len(run.rounds)
        out = stub(app, r, pool, weights, seed, label, ctl, on_arrival, steps)
        if label.startswith("metroseed"):
            snap["live_crossings"] = [dict(x) for x in run.seed_crossings]
        return out

    orig = cs.evaluate
    cs.evaluate = driven
    try:
        metro.run_metro(None, run, None)
    finally:
        cs.evaluate = orig
    return snap, dispatches


def part7():
    print("\n(vii) the seed phase publishes itself: chords, probes, crossings, chains at step 0")
    field, _p = _two_basin()
    chains, L = 10, 4
    run = _mk_run(chains=chains, chain_steps=L, m=2, n_seed_chords=14)
    snap, dispatches = _drive_seed(run, field)
    n_seed = next(n for lab, n, _st in dispatches if lab == "metroseed")
    key = (lambda w: tuple(round(float(v), 9) for v in w))

    # --- the chords
    check(len(run.seed_chords) == run.n_chords
          and len(snap["chords_at_dispatch"]) == run.n_chords,
          "every chord is published as it is DRAWN, so they are all on the map before the "
          "first probe of them was even dispatched",
          f"{len(snap['chords_at_dispatch'])} chords at dispatch = {run.n_chords} drawn")
    check(all(len(c) == 2 and len(c[0]) == K and len(c[1]) == K for c in run.seed_chords),
          "a chord is [w_start, w_end], both k-long")
    ends = {key(c[0]) for c in run.seed_chords} | {key(c[1]) for c in run.seed_chords}
    at = {key(p["w"]) for p in run.seed_probes}
    check(ends <= at,
          "and its two ends ARE probes of it (the first and the last), so the line drawn is "
          "the line measured rather than the unclipped chord it was taken from",
          f"{len(ends)} endpoints, all among {len(at)} probe positions")
    check(all(float(np.linalg.norm(np.asarray(c[1]) - np.asarray(c[0])))
              >= cs._min_chord_len(cs.STRIDE) - 2 * cs.STRIDE for c in run.seed_chords),
          "every published chord is a chord that was long enough to carry a bracket")

    # --- the probes, as they arrive
    check(len(run.seed_probes) == n_seed,
          "one seed probe is published per probe RENDERED -- the whole batch, nothing dropped",
          f"{len(run.seed_probes)} probes = {n_seed} in the metroseed batch")
    check(all(set(p) == {"w", "div", "image_idx"} for p in run.seed_probes),
          "each carries the documented three fields", f"{sorted(run.seed_probes[0])}")
    check(all(len(p["w"]) == K and p["image_idx"] in run.embeddings
              for p in run.seed_probes),
          "a probe names its own recipe and the image that was rendered at it")
    firsts = [a[0] for a in snap["arrivals"]]
    check(firsts == list(range(1, n_seed + 1))
          and snap["arrivals"][0][2] <= 1 and snap["arrivals"][0][3] == "seed chords",
          "they go up ONE AT A TIME as they arrive, from the first one on -- the panel does "
          "not wait for the batch",
          f"{len(firsts)} arrivals, {firsts[:3]}... with phase "
          f"{snap['arrivals'][0][3]!r}")
    check(all(p["div"] is not None for p in run.seed_probes),
          "every probe ends up with a divergence: a chord carries at least three probes "
          "(cascade._min_chord_len), so every probe has a chord neighbour to be read against")
    # the reading itself is the Cascade's: the cosine distance to a NEIGHBOUR's embedding.
    # Arrival order is dispatch order here, so a probe's chord neighbours are among its list
    # neighbours, and its div must be one of those two readings exactly -- not a placeholder,
    # and not a distance to something that is not next to it.
    bad = []
    for i, p in enumerate(run.seed_probes):
        nb = [x["image_idx"] for x in (run.seed_probes[max(0, i - 1):i]
                                       + run.seed_probes[i + 1:i + 2])]
        ds = [float(cs._cosd(run.embeddings[p["image_idx"]], run.embeddings[g])) for g in nb]
        if not any(abs(p["div"] - d) < 1e-12 for d in ds):
            bad.append(i)
    check(not bad,
          "and that divergence IS the cosine distance to one of its chord neighbours (the "
          "larger of the two where it has two, as the Cascade's `_set_div` keeps it)",
          f"{len(run.seed_probes)} probes checked against their neighbours")

    # --- the crossings, live against the ordered pass that has the last word on them
    live, final = snap["live_crossings"], run.seed_crossings
    check(live and len(live) == len(final)
          and {key(x["weights"]) for x in live} == {key(x["weights"]) for x in final},
          "the crossings detected live are exactly the set the ordered pass finds -- the live "
          "list is for watching, the ordered one is what seeds the chains, and they never "
          "disagree about WHICH crossings there are",
          f"{len(live)} live = {len(final)} ordered")
    check(all(x["divergence"] > cs.COS_T and metro.in_simplex(x["weights"]) for x in live),
          "every live crossing is a legal recipe past the threshold",
          f"min div {min(x['divergence'] for x in live):.3f} > {cs.COS_T}")
    mid = [a for a in snap["arrivals"] if a[0] < n_seed]
    check(any(a[1] > 0 for a in mid),
          "and the first of them lands while the survey is still probing, not at the end of "
          "it", f"{max(a[1] for a in mid)} crossings up before the batch finished")

    # --- the chains, published at step 0 before the batch that measures them
    c0 = snap["chains_at_init"]
    check(c0 is not None and len(c0) == chains and snap["phase_at_init"] == "initial energies"
          and snap["rounds_at_init"] == 0,
          f"all {chains} chains exist BEFORE the initial-energy batch is dispatched, in a "
          f"phase that names itself", f"{len(c0 or [])} chains, phase "
          f"{snap['phase_at_init']!r}, {snap['rounds_at_init']} rounds logged")
    check(all(c["s"] is None and c["seed_s"] is None and c["image"] == -1
              and c["seed_image"] == -1 for c in c0),
          "with no energy and no image yet: null, not 0 -- the map draws such a head hollow "
          "rather than inventing a reading for it")
    check(all(c["n_propose"] == 0 and c["n_accept"] == 0 and c["acc_rate"] is None
              for c in _as_stats(c0)),
          "and at step 0 with acc_rate null, which is what the chain table shows")
    check(all(c["weights"] == c["seed_weights"] and len(c["weights"]) == K for c in c0)
          and [c["chain"] for c in c0] == list(range(chains)),
          "every head sits at its own seed, and the ids are positional")
    seeds_at = {key(x["weights"]) for x in final}
    check(all(key(c["seed_weights"]) in seeds_at for c in c0),
          "and the seeds they sit on are crossings the survey found",
          f"{len(seeds_at)} distinct crossings for {chains} chains")
    check(len(run.chains_state) == chains
          and [c["chain"] for c in run.chains_state] == [c["chain"] for c in c0]
          and all(a["weights"] == b["seed_weights"] for a, b in zip(c0, run.chains_state))
          and all(c["s"] is not None and c["seed_s"] is not None
                  for c in run.chains_state),
          "when the batch lands the rows are FILLED IN, not appended a second time: the same "
          "chains, with the same ids, each still recording as its seed the position it was "
          "published at -- and now with an energy there",
          f"{len(run.chains_state)} chains at the end")
    check(not run.samples or min(x["step"] for x in run.samples) >= 1,
          "a start state is not a sample: nothing is stamped with round 0")


def _as_stats(rows):
    """The router's per-chain view of raw chain rows (routers/metro._chain_stats' arithmetic)."""
    out = []
    for c in rows:
        p = int(c.get("n_propose", 0))
        out.append({**c, "acc_rate": (float(c.get("n_accept", 0)) / p) if p else None})
    return out


# -------------------------------------------- (viii) the seed phase on /status, with its cap
def part8():
    print("\n(viii) /status serves the seed phase, probes strided to the served ceiling")
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
    except Exception as exc:                         # noqa: BLE001
        check(False, "TestClient available", str(exc))
        return
    from backend.routers import metro as router_mod

    class _NoCache:
        def has(self, h):
            return False

        def save(self, h, data):
            pass

        def get_path(self, h):
            return None

    orig = metro.run_metro
    metro.run_metro = lambda app, run, pool: None
    try:
        app = FastAPI()
        app.include_router(router_mod.router)
        app.state.metro_runs = {}
        app.state.cascades = {}
        app.state.cache = _NoCache()
        app.state.gpu_pool = None
        c = TestClient(app, raise_server_exceptions=False)
        r = c.post("/api/metro/start", json={"prompts": ["a", "b", "c", "d"], "chains": 4,
                                            "chain_steps": 5})
        rid = r.json()["run_id"]
        run = app.state.metro_runs[rid]

        # a fresh run: the three fields are there and empty, which is what an old client that
        # never heard of them also sees
        j = c.get(f"/api/metro/{rid}/status").json()
        check(j["seed_chords"] == [] and j["seed_probes"] == [] and j["seed_crossings"] == [],
              "a run that has not started seeding serves the three fields empty")

        run.seed_chords = [[[0.9, 0.1, 0.0, 0.0], [0.0, 0.1, 0.9, 0.0]],
                           [[0.25] * 4, [0.4, 0.2, 0.2, 0.2]]]
        run.seed_probes = [{"w": [0.25] * 4, "div": 0.41, "image_idx": 7},
                           {"w": [0.3, 0.3, 0.2, 0.2], "div": None, "image_idx": 8}]
        run.seed_crossings = [{"weights": [0.25] * 4, "divergence": 0.41},
                              {"weights": [0.5, 0.2, 0.2, 0.1], "divergence": None}]
        j = c.get(f"/api/metro/{rid}/status").json()
        check(j["seed_chords"] == [[[0.9, 0.1, 0.0, 0.0], [0.0, 0.1, 0.9, 0.0]],
                                  [[0.25] * 4, [0.4, 0.2, 0.2, 0.2]]],
              "the chords come through as [w_start, w_end] pairs",
              f"{len(j['seed_chords'])} chords")
        check(len(j["seed_probes"]) == 2 and j["seed_probes"][0] == {
            "w": [0.25] * 4, "div": 0.41, "image_idx": 7}
            and j["seed_probes"][1]["div"] is None,
              "a probe carries w, div and image_idx, with div null while it is unpaired",
              f"{j['seed_probes'][1]}")
        check(j["seed_crossings"] == [{"w": [0.25] * 4, "div": 0.41},
                                     {"w": [0.5, 0.2, 0.2, 0.1], "div": None}],
              "a crossing comes through as {w, div} (div null where the source run had none)")
        check(j["seeds"] == [[0.25] * 4, [0.5, 0.2, 0.2, 0.1]]
              and j["seed_divs"] == [0.41, None],
              "and `seeds`/`seed_divs` still say exactly what they said before, so a client "
              "that reads those is untouched", f"{j['seeds']}")

        # the cap: every probe is kept on the run, only the view is strided
        n = 4 * router_mod.SEED_PROBES_MAX + 1
        run.seed_probes = [{"w": [0.25] * 4, "div": i / n, "image_idx": i} for i in range(n)]
        j = c.get(f"/api/metro/{rid}/status").json()
        got = j["seed_probes"]
        check(len(got) <= router_mod.SEED_PROBES_MAX and len(got) > 0
              and got[0]["image_idx"] == 0
              and [p["image_idx"] for p in got] == list(range(0, n, 5)),
              f"{n} probes are strided down to at most {router_mod.SEED_PROBES_MAX} served, "
              f"evenly and from the first one, with the run keeping all of them",
              f"{len(got)} served of {len(run.seed_probes)} held")
        run.seed_probes = [{"w": [0.25] * 4, "div": None, "image_idx": i}
                           for i in range(router_mod.SEED_PROBES_MAX)]
        j = c.get(f"/api/metro/{rid}/status").json()
        check(len(j["seed_probes"]) == router_mod.SEED_PROBES_MAX,
              "exactly at the ceiling nothing is strided away",
              f"{len(j['seed_probes'])} served")
    finally:
        metro.run_metro = orig


if __name__ == "__main__":
    part1()
    part2()
    part3()
    part4()
    part5()
    part6()
    part7()
    part8()
    print(f"\n{'ALL CHECKS PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)
