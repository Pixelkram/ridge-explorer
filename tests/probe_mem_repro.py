"""Standalone reproduction: load the pipe as gpu_pool.worker_main does, then run jvp_probe.ridge_normal on one
point while logging GPU memory at each stage. Usage: python tests/probe_mem_repro.py --gpu 3"""
import sys, argparse, time, torch
sys.path.insert(0, '.')
from backend import config
from backend.services import jvp_probe
ap = argparse.ArgumentParser(); ap.add_argument('--gpu', type=int, default=3); a = ap.parse_args()
dev = torch.device(f'cuda:{a.gpu}'); torch.cuda.set_device(dev)
mem = lambda tag: print(f"[{tag:<28}] allocated {torch.cuda.memory_allocated(dev)/2**30:6.2f} GiB  reserved {torch.cuda.memory_reserved(dev)/2**30:6.2f} GiB", flush=True)
from diffusers import Flux2KleinPipeline
pipe = Flux2KleinPipeline.from_pretrained(config.MODEL_ID, torch_dtype=torch.bfloat16).to(dev); mem("pipe loaded")
dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg').to(dev); dino.eval(); mem("dino loaded")
for n, m in (("transformer", pipe.transformer), ("text_encoder", pipe.text_encoder), ("vae", pipe.vae)):
    print(f"  {n}: {sum(p.numel()*p.element_size() for p in m.parameters())/2**30:.2f} GiB, dtype {next(m.parameters()).dtype}, device {next(m.parameters()).device}")
with torch.no_grad():
    neg, _ = pipe.encode_prompt(prompt=""); pipe._cached_neg_embeds = neg
    enc = [pipe.encode_prompt(prompt=p, device=dev) for p in ("a photograph of an airplane", "a photograph of a giraffe", "a surrealist oil painting of a roll of toilet paper")]
mem("after encoding")
embs = [e for e, _ in enc]; tids = enc[0][1]
# instrument the context: check offload really happened
orig = jvp_probe._forward_ad_ready
from contextlib import contextmanager
@contextmanager
def instrumented(p):
    with orig(p):
        mem("inside ctx (offloaded?)"); print("  text_encoder device now:", next(p.text_encoder.parameters()).device, flush=True)
        yield
    mem("after ctx restore")
jvp_probe._forward_ad_ready = instrumented
t0 = time.time()
try:
    r = jvp_probe.ridge_normal(pipe, dino, embs, tids, [0.35, 0.43, 0.22], seed=42, steps=8, guidance=4.0, height=512, width=512, use_slerp=False)
    print("OK", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in ('rank1_share', 'participation_ratio', 'sigma1_raw', 'wall_s')}, "normal_theta", [round(x, 3) for x in r['normal_theta']])
except Exception as e:
    print("FAILED:", type(e).__name__, str(e)[:200])
print(f"peak allocated {torch.cuda.max_memory_allocated(dev)/2**30:.2f} GiB, {time.time()-t0:.0f}s")
