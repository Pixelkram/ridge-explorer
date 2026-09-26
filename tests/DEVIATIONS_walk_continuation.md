# Deviations from PREREG_walk_continuation.md

1. **D1 is a fresh cascade, not the stored design run.** The PREREG describes D1 as "the design run's crossings
   (f91e00f7)"; that run lived only in server memory and was lost at the restarts before the freeze, so
   `walk_pc_run_all.py` started a fresh cascade with the same prompt set and seed (run 7f79fc8d). The cascade is
   deterministic: all 18 crossings reproduce f91e00f7's significance flags and B values exactly (max |ΔB| < 5e-5,
   checked against `walk_diag_f91e00f7_relative.json`). D1 is descriptive only. (Found by the verifier.)
2. **Analysis-script fix after verification.** `arcs()` stopped at the first zero-length segment instead of only at
   the 0.30 cap. No effect on any number: the shortest segment in the data is 0.014 (verifier); the re-run
   reproduces every registered value.
