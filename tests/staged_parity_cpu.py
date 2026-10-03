"""CPU bit-identity check of the staged-probe loop (backend/services/staged.py) against the real
FLUX.2-klein-base-4B pipeline. SLOW (minutes; loads the 4B transformer, the Qwen3-4B text encoder,
the VAE and DINOv2 on the CPU) -- NOT part of the fast suite. No GPU: run with
CUDA_VISIBLE_DEVICES="".

Loads the pipeline exactly as a gpu_pool worker does (bf16, cached empty negative embedding,
torch.hub DINOv2 + the worker's transform) and, at a tiny resolution with few steps, asserts:

  A  manual full run (staged.denoise 0..S) == pipe() -- the Cascade's Discover image path
     (gpu_pool._discover_point) -- final packed latents (captured by a step-end callback),
     decoded pixels and DINOv2 embedding;
  B  stop at t, latent -> bytes -> latent, resume t..S == the manual full run (torch.equal);
  C  worker_readout's latent == the full run's latent after step t, and its x̂0 embedding ==
     the full run's x̂0 at step t decoded through the same path;
  D  worker_resume from the readout's bytes == the full run's final image (DINOv2 equal) and a
     from-scratch resume (no latent) == the same;
  E  worker_trace_point: x̂0 at step t == the readout, t = S == the final image, the 4-step
     arm == _discover_point at those steps;
  F  worker_onepass (the generalised Fast-Scan readout) == gpu_pool._ScanContext.evaluate_points
     on a 3-prompt (alpha, beta) task, bit for bit;
  G  worker_parity reports bit-identity, identical pixels and cos 1 (incl. against pipe());
  H  gpu_pool._process_staged -- the worker loop's dispatch of all four task types, results
     pickled as the mp queue does -- reproduces C-G, and a failing point is reported, not raised.

Writes its output to tests/staged_parity_cpu.log as well.

Run: CUDA_VISIBLE_DEVICES="" python tests/staged_parity_cpu.py [--size 128] [--steps 4] [--t 2] [--append]
     (the recorded log holds two runs: 128 px / S = 4 / t = 2, and the tool's own 512 px / S = 8 / t = 4)
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import config                                  # noqa: E402
from backend.services import staged as st                   # noqa: E402
from backend.services import gpu_pool as gp                 # noqa: E402

LOG = ROOT / "tests" / "staged_parity_cpu.log"
_lines = []
fails = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def check(ok, label, detail=""):
    log(f"  [{'PASS' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--t", type=int, default=2)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--append", action="store_true", help="append to the log instead of replacing it")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    T0 = time.time()
    S, t, H = args.steps, args.t, args.size
    dev = torch.device("cpu")
    log(f"staged_parity_cpu  {time.strftime('%Y-%m-%d %H:%M:%S')}  torch {torch.__version__}  "
        f"threads {torch.get_num_threads()}  model {config.MODEL_ID}")
    log(f"settings: {H}x{H}, S = {S}, t = {t}, CFG 4.0, seed 42, bf16 on CPU")

    from diffusers import Flux2KleinPipeline
    import diffusers
    log(f"diffusers {diffusers.__version__}")
    t_load = time.time()
    pipe = Flux2KleinPipeline.from_pretrained(config.MODEL_ID, torch_dtype=torch.bfloat16).to(dev)
    pipe.set_progress_bar_config(disable=True)
    with torch.no_grad():
        pipe._cached_neg_embeds, _ = pipe.encode_prompt(prompt="")
    try:
        dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg').to(dev)
    except Exception as exc:                              # noqa: BLE001  offline fallback, same weights
        log(f"torch.hub github load failed ({exc}); loading the cached hub repo locally")
        hub = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
        dino = torch.hub.load(str(hub), 'dinov2_vitb14_reg', source="local").to(dev)
    dino.eval()
    from torchvision import transforms
    dino_transform = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    log(f"models loaded in {time.time() - t_load:.0f}s")

    prompts = ["a close-up photograph of cracked ice", "a photograph of icebergs at sea",
               "a watercolor painting of a jellyfish"]
    base = st.encode_basis(pipe, prompts)
    w = [0.2, 0.5, 0.3]
    seed, g = 42, 4.0
    common = (pipe, dino, dino_transform, dev, base)

    # ---------------- A: manual full run vs pipe()
    log("\nA  manual full run vs pipe() (the Discover image path)")
    t1 = time.time()
    emb = st.mix_discover(base, w)
    rec = st.StagedRecipe(pipe, emb, seed, H, H, S, g, dev)
    lat_after = {}
    x0_at = {}

    def hook(step, lat_in, v, sigma, lat_out):
        lat_after[step] = lat_out.clone()
        x0_at[step] = (lat_in.clone(), v.clone(), sigma.clone())
    manual = st.denoise(rec, rec.latents0, 0, S, hook)
    img_m = st.decode_pil(pipe, manual, rec.latent_ids)
    e_m = st.dino_embed(dino, dino_transform, img_m, dev)
    log(f"   manual run + decode: {time.time() - t1:.0f}s")
    t1 = time.time()
    cap = {}

    def cb(_p, i, _t, kw):
        cap[i] = kw["latents"].clone()
        return kw
    gen = torch.Generator(device=dev).manual_seed(seed)
    with torch.no_grad():
        img_p = pipe(prompt_embeds=emb, height=H, width=H, num_inference_steps=S,
                     negative_prompt_embeds=pipe._cached_neg_embeds, guidance_scale=g, generator=gen,
                     callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"]).images[0]
    e_p = st.dino_embed(dino, dino_transform, img_p, dev)
    img_d, e_d = gp._discover_point(pipe, dino, dino_transform, dev, base, w, seed, H, H, S, g)
    e_d = e_d.cpu().numpy().flatten()
    log(f"   pipe() x2: {time.time() - t1:.0f}s")
    d_lat = float((cap[S - 1].float() - manual.float()).abs().max())
    check(torch.equal(cap[S - 1], manual), "final packed latents: manual == pipe()",
          f"max |diff| {d_lat:.3g}, dtype {manual.dtype}, shape {tuple(manual.shape)}")
    per_step = [float((cap[i].float() - lat_after[i + 1].float()).abs().max()) for i in range(S)]
    check(max(per_step) == 0.0, "every intermediate latent: manual == pipe()", f"{per_step}")
    check(np.array_equal(np.asarray(img_m), np.asarray(img_p))
          and np.array_equal(np.asarray(img_m), np.asarray(img_d)),
          "decoded PIL pixels: manual == pipe() == _discover_point")
    check(np.array_equal(e_m, e_p) and np.array_equal(e_m, e_d),
          "DINOv2 embedding: manual == pipe() == _discover_point",
          f"cos {float(np.dot(e_m, e_p)):.9f}")

    # ---------------- B: stop at t + bytes roundtrip + resume
    log("\nB  stop at t -> bytes -> resume == manual full run")
    lat_t = st.denoise(rec, rec.latents0, 0, t)
    check(torch.equal(lat_t, lat_after[t]), f"the latent after step {t} == the full run's")
    b, shape, dt = st.latent_to_bytes(lat_t)
    back = st.bytes_to_latent(b, shape, dt, dev)
    check(torch.equal(back, lat_t) and back.dtype == lat_t.dtype,
          "latent -> bytes -> latent is bit-exact", f"{len(b)} bytes, {dt}, {shape}")
    resumed = st.denoise(rec, back, t, S)
    d_res = float((resumed.float() - manual.float()).abs().max())
    check(torch.equal(resumed, manual), "stop-at-t + resume == manual full run (exact)",
          f"max |diff| {d_res:.3g}")

    # ---------------- C: worker_readout
    log("\nC  worker_readout")
    rd = st.worker_readout(*common, w, seed, H, H, S, t, g)
    lat_rd = st.bytes_to_latent(rd["latent"], rd["shape"], rd["dtype"], dev)
    check(torch.equal(lat_rd, lat_after[t]), f"readout latent == the full run's after step {t}")
    li, v, sg = x0_at[t]
    _img, e_x0 = st.x0hat_embedding(rec, dino, dino_transform, li, v, sg)
    check(np.array_equal(rd["emb"], e_x0), f"readout x̂0 embedding == the full run's x̂0 at step {t}",
          f"cos(x̂0_t, final) {float(np.dot(rd['emb'], e_m)):.4f}; timing {rd['timing']}")
    check(len(rd["thumb"]) > 100 and rd["thumb"][:2] == b"\xff\xd8", "readout carries a JPEG preview")

    # ---------------- D: worker_resume
    log("\nD  worker_resume")
    rs = st.worker_resume(*common, w, seed, H, H, S, t, g, rd["latent"], rd["shape"], rd["dtype"])
    check(np.array_equal(rs["emb"], e_m), "resume from the readout's bytes == full image (DINOv2 equal)",
          f"timing {rs['timing']}")
    rs0 = st.worker_resume(*common, w, seed, H, H, S, t, g, None, None, None)
    check(np.array_equal(rs0["emb"], e_m) and rs0["timing"]["from_scratch"]
          and rs0["timing"]["steps"] == S,
          "from-scratch resume (evicted latent) == full image, flagged from_scratch")

    # ---------------- E: worker_trace_point
    log("\nE  worker_trace_point")
    fs_steps = max(1, S // 2)

    def fourstep(ww):
        img, e = gp._discover_point(pipe, dino, dino_transform, dev, base, ww, seed, H, H, fs_steps, g)
        return img, e.cpu().numpy().flatten()
    tr = st.worker_trace_point(*common, w, seed, H, H, S, g, fourstep)
    _i4, e4 = fourstep(w)
    check(tr["xhat"].shape == (S, 768) and np.array_equal(tr["xhat"][t - 1], rd["emb"])
          and np.array_equal(tr["xhat"][S - 1], e_m),
          f"trace: x̂0 at step {t} == readout, row S == the final image",
          f"cos to final per step {[round(float(np.dot(r, e_m)), 4) for r in tr['xhat']]}")
    check(np.array_equal(tr["fourstep"], e4), f"trace: {fs_steps}-step arm == _discover_point({fs_steps} steps)")
    cum = tr["denoise_cum_s"]
    check(all(b2 >= a2 for a2, b2 in zip(cum, cum[1:])) and len(tr["decode_dino_s"]) == S,
          "trace timings: cumulative denoising time non-decreasing, one decode per step",
          f"cum {[round(x, 1) for x in cum]} s, decode {[round(x, 1) for x in tr['decode_dino_s']]} s, "
          f"{fs_steps}-step {tr['fourstep_s']:.1f} s")

    # ---------------- F: Fast-Scan generalisation
    log("\nF  worker_onepass vs gpu_pool._ScanContext.evaluate_points")
    ab = [(0.5, 0.3), (0.1, 0.2), (0.0, 0.9)]
    task = gp.FastScanTask(job_id="x", row_indices=[0], alphas=np.array([a for a, _ in ab]),
                           betas=np.array([b2 for _, b2 in ab]), prompt_a=prompts[0], prompt_b=prompts[1],
                           prompt_c=prompts[2], grid_size=3, seed=seed, height=H, width=H, use_slerp=False)
    ctx = gp._ScanContext(pipe, dev, task)
    ref = ctx.evaluate_points([(float(a), float(b2)) for a, b2 in ab], batch_size=8)
    mine = st.worker_onepass(pipe, dev, prompts, [[1 - a - b2, a, b2] for a, b2 in ab], seed, H, H)
    check(ref.shape == mine.shape and np.array_equal(ref, mine.astype(ref.dtype)),
          "one-pass readout == the Fast Scan's own computation, bit for bit",
          f"shape {mine.shape}; consecutive 1-cos {[round(1 - float(np.dot(mine[i], mine[i + 1])), 4) for i in range(len(mine) - 1)]}")

    # ---------------- G: worker_parity
    log("\nG  worker_parity")
    pr = st.worker_parity(*common, [0.6, 0.1, 0.3], seed, H, H, S, t, g, pipe_check=True)
    check(pr["latent_bit_identical"] and pr["max_abs_latent_diff"] == 0.0 and pr["pixels_identical"]
          and pr["cos_resumed_vs_direct"] >= 0.9999999 and pr["max_abs_latent_diff_pipe"] == 0.0
          and pr["pixels_identical_pipe"] and pr["cos_resumed_vs_pipe"] >= 0.9999999,
          "parity: resumed == direct == pipe() (latents, pixels, cos)",
          f"{ {k: v for k, v in pr.items() if k != 'timing'} }")

    # ---------------- H: the worker's dispatch, end to end
    log("\nH  gpu_pool._process_staged (the worker loop's dispatch; results pickled as the mp queue does)")
    import pickle

    class Q:
        def __init__(self):
            self.items = []

        def put(self, x):
            self.items.append(pickle.loads(pickle.dumps(x)))
    w2 = [0.6, 0.1, 0.3]
    kw = dict(job_id="h", basis=prompts, seed=seed, height=H, width=H, steps=S, guidance_scale=g)
    q = Q()
    gp._process_staged(0, dev, pipe, dino, dino_transform,
                       gp.StagedReadoutTask(points=[(0, w), (1, w2)], t=t, **kw), q)
    rds = {r.index: r for r in q.items}
    check(sorted(rds) == [0, 1] and all(r.kind == "readout" and not r.error for r in q.items)
          and np.array_equal(rds[0].data["emb"], rd["emb"]) and rds[0].data["latent"] == rd["latent"],
          "StagedReadoutTask == worker_readout (embedding and latent bytes)")
    q = Q()
    d0 = rds[0].data
    gp._process_staged(0, dev, pipe, dino, dino_transform,
                       gp.StagedResumeTask(points=[(0, (w, d0["latent"], d0["shape"], d0["dtype"])),
                                                   (1, (w, None, None, None))], t=t, **kw), q)
    rs_ = {r.index: r for r in q.items}
    check(np.array_equal(rs_[0].data["emb"], e_m) and np.array_equal(rs_[1].data["emb"], e_m)
          and rs_[1].data["timing"]["from_scratch"] and not rs_[0].data["timing"]["from_scratch"],
          "StagedResumeTask from the task's latent bytes, and from scratch, == the full image")
    q = Q()
    chord3 = [w, w2, [0.1, 0.1, 0.8]]
    gp._process_staged(0, dev, pipe, dino, dino_transform,
                       gp.StagedTraceTask(points=[(0, w)], fourstep_steps=fs_steps, onepass_points=chord3,
                                          **kw), q)
    op = [r for r in q.items if r.kind == "onepass"]
    tq = [r for r in q.items if r.kind == "trace"]
    lat3 = st.worker_onepass(pipe, dev, prompts, chord3, seed, H, H)
    want_div = [float(1.0 - np.dot(lat3[i], lat3[i + 1])) for i in range(2)]
    check(len(op) == 1 and op[0].index == -1 and op[0].data["divs"] == want_div and op[0].data["n"] == 3
          and len(tq) == 1 and np.array_equal(tq[0].data["xhat"], tr["xhat"])
          and np.array_equal(tq[0].data["fourstep"], tr["fourstep"]),
          "StagedTraceTask: one chord-level 1-pass record (consecutive 1-cos of the Fast-Scan "
          "readout) and the per-point trace == worker_trace_point",
          f"onepass_div {[round(x, 4) for x in want_div]}, {op[0].data['onepass_s']:.1f} s")
    q = Q()
    gp._process_staged(0, dev, pipe, dino, dino_transform,
                       gp.StagedParityTask(points=[(0, w2)], t=t, pipe_check=False, **kw), q)
    check(len(q.items) == 1 and q.items[0].data["latent_bit_identical"]
          and q.items[0].data["cos_resumed_vs_direct"] >= 0.9999999,
          "StagedParityTask: bit-identical through the dispatch")
    q = Q()
    gp._process_staged(0, dev, pipe, dino, dino_transform,
                       gp.StagedReadoutTask(points=[(7, [0.25, 0.25, 0.25, 0.25])], t=t, **kw), q)
    check(len(q.items) == 1 and q.items[0].index == 7 and "IndexError" in q.items[0].error,
          "a point that fails is reported in its result's error; the task does not raise",
          q.items[0].error if q.items else "")

    log(f"\ntotal wall-clock {time.time() - T0:.0f}s ({(time.time() - T0) / 60:.1f} min)")
    log(f"\n{'ALL CHECKS PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    with open(LOG, "a" if args.append else "w") as fh:
        fh.write(("\n" + "=" * 100 + "\n" if args.append else "") + "\n".join(_lines) + "\n")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
