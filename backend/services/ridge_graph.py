"""Ridge graph + itinerary: the ridge network as walkable arcs, not just a mask.

The tool already renders sensitivity as a field (heatmap) and as a thresholded set
(overlay contours). This adds the third view: the ridge as a GRAPH of arcs and junctions
that can be walked in order, with the flanking basins named at every station — two of
them, except where three or more meet, which each arc lists in `meetings`.

Method, ported from the validated offline pipeline:

  1. persistence-simplify S with an h-minima transform, so shallow noise minima do not
     each spawn their own basin;
  2. watershed with `watershed_line=True` — the line follows the crests of S;
  3. label the cells semantically by clustering the stored DINOv2 embeddings;
  4. CONTRACT: keep only the watershed line separating faces whose majority semantic
     labels differ. This is what turns a dense watershed skeleton into the small set of
     arcs that actually divide distinct content;
  5. split the retained line at junctions (degree >= 3) into simple arcs and order each
     one by walking it end to end.

Two implementation notes that matter:

* Clustering is done with scipy linkage on the condensed distance vector, NOT with
  `ridge_detector.compute_clusters`, which materialises a full N x N cosine matrix —
  0.8 GB at 100x100 and 10.9 GB at 192x192, both of which occur in the existing cache.
  The condensed vector is N(N-1)/2, so this halves that peak rather than removing it;
  at 192x192 it is still ~5 GB and is the largest allocation in the service.
* Everything is rebuilt from files in `results_dir` (`sensitivity.npy`, `embeddings.npy`)
  rather than from `app.state.jobs`, which is an in-memory dict that `POST /cancel`
  clears. That also makes the graph re-derivable for old jobs with no GPU.

Measured behaviour of this construction (50 triplets x 3 seeds, offline): arcs sit at the
~96th percentile of S, and 94.6% of arc pixels lie within 1 px of the boundary of the
semantic partition, against 17.4% for a phase-randomised control. The flanking pair is
constant along a typical arc — arcs terminate at junctions or the domain edge before the
neighbours change — so an "itinerary" is usually one pair, occasionally two.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
NB4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# ------------------------------------------------------------------ clustering

def semantic_labels(embeddings: np.ndarray, valid: np.ndarray, k: int) -> np.ndarray:
    """Agglomerative (cosine, average linkage) basin labels over the valid cells.

    Uses the condensed distance vector, so memory is N(N-1)/2 floats rather than the
    N x N matrix `ridge_detector.compute_clusters` builds.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    X = embeddings[valid].astype(np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(n, 1e-12)
    out = np.zeros(valid.shape, dtype=np.int32)
    if len(X) < 3:
        return out
    Z = linkage(pdist(X, metric="cosine"), method="average")
    out[valid] = fcluster(Z, k, criterion="maxclust")
    return out


# -------------------------------------------------------------------- watershed

def watershed_line(S: np.ndarray, valid: np.ndarray, h_frac: float) -> np.ndarray:
    """Persistence-simplified watershed of S; the line follows crests."""
    from scipy.ndimage import label as cc_label
    from skimage.morphology import h_minima
    from skimage.segmentation import watershed

    finite = S[valid & np.isfinite(S)]
    if finite.size == 0:
        return np.zeros_like(S, dtype=np.int32)
    span = float(np.nanmax(finite) - np.nanmin(finite))
    filled = np.where(valid & np.isfinite(S), S, np.nanmax(finite) + 1.0)
    markers, _ = cc_label(h_minima(filled, max(h_frac * span, 1e-9)) & valid)
    return watershed(filled, markers, mask=valid, watershed_line=True)


def contract(ws: np.ndarray, lab: np.ndarray, valid: np.ndarray,
             meetings: dict | None = None):
    """Keep only the watershed line separating different semantic basins.

    `flank` names two basins per pixel, which is a truncation where three faces meet.
    Pass a dict as `meetings` to also receive {pixel: all basins seen, sorted} for those
    pixels, so a caller can report the station as a 3-way meeting instead of taking the
    pair at face value. (Out-parameter rather than a fourth return value: `hiker.walk`
    and the tests unpack this as a 3-tuple.)
    """
    raw = (ws == 0) & valid
    face_sem = {}
    for f in np.unique(ws):
        if f == 0:
            continue
        m = (ws == f) & valid
        vals, cnt = np.unique(lab[m], return_counts=True)
        if len(vals):
            face_sem[int(f)] = int(vals[np.argmax(cnt)])

    H, W = ws.shape
    retained = np.zeros_like(raw)
    flank = {}
    ys, xs = np.nonzero(raw)
    for y, x in zip(ys, xs):
        sem = {}
        for dy, dx in NB8:
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W and ws[yy, xx] > 0:
                s = face_sem.get(int(ws[yy, xx]))
                if s is not None:
                    sem[s] = sem.get(s, 0) + 1
        if len(sem) >= 2:
            # neighbour-pixel counts tie routinely at a triple point (often all three at
            # 1), and a plain -count sort then falls through to dict insertion order,
            # i.e. the up-and-left-biased NB8 scan. Break the tie on basin id so the
            # emitted pair is at least reproducible, and record the meeting itself.
            top = sorted(sem.items(), key=lambda kv: (-kv[1], kv[0]))[:2]
            retained[y, x] = True
            flank[(int(y), int(x))] = (int(top[0][0]), int(top[1][0]))
            if meetings is not None and len(sem) >= 3:
                meetings[(int(y), int(x))] = tuple(sorted(int(s) for s in sem))
    return retained, flank, raw


# ring-ordered 8-neighbourhood, clockwise from north -- order matters for _crossing
RING8 = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def _ring_components(p, pts):
    """The mask neighbours of p, grouped into the branches they belong to.

    Neighbours that are adjacent to each other around the ring are one branch, so a
    corner yields one group and a T yields three.
    """
    v = [(p[0] + dy, p[1] + dx) if (p[0] + dy, p[1] + dx) in pts else None
         for dy, dx in RING8]
    if all(q is not None for q in v):
        return [set(q for q in v if q is not None)]
    groups, cur = [], []
    start = next(i for i in range(8) if v[i] is None)
    for k in range(1, 9):
        q = v[(start + k) % 8]
        if q is None:
            if cur:
                groups.append(set(cur))
                cur = []
        else:
            cur.append(q)
    if cur:
        groups.append(set(cur))
    return groups


def _crossing(p, pts) -> int:
    """Number of separate mask branches touching p (Rutovitz crossing number / 2).

    A raw 8-neighbour COUNT is not a branch test: on an 8-connected line an L-bend has
    two neighbours and a staircase step has three, but in a staircase those three are
    mutually adjacent and form a single branch. Counting degree>=3 therefore declares
    ordinary corners to be junctions -- measured at 82-94% false positives on real
    sensitivity fields, which shredded arcs and truncated them to a third of their
    length. Counting ring transitions instead only fires where branches genuinely part.
    """
    v = [1 if (p[0] + dy, p[1] + dx) in pts else 0 for dy, dx in RING8]
    return sum(abs(v[(i + 1) % 8] - v[i]) for i in range(8)) // 2


def split_arcs(mask: np.ndarray):
    """Split a thin mask into simple arcs by removing junctions, then walk each."""
    ys, xs = np.nonzero(mask)
    pts = set(zip(ys.tolist(), xs.tolist()))
    junctions = {p for p in pts if _crossing(p, pts) >= 3}
    body = pts - junctions
    # orthogonal neighbours first: with both a diagonal and an orthogonal step
    # available, taking the diagonal skips the pixel between them and drops it
    # from the arc.
    adj = {p: sorted([(p[0] + dy, p[1] + dx) for dy, dx in NB8
                      if (p[0] + dy, p[1] + dx) in body],
                     key=lambda q: (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2)
           for p in body}
    # Deleting the junction pixel does not separate its arms: on an 8-connected mask
    # the arms of a T remain diagonally adjacent to each other, so all three collapse
    # into one component and the junction yields a single arc. Cut only the links
    # between arms that are genuinely distinct branches of that junction.
    for j in junctions:
        groups = [g & body for g in _ring_components(j, pts)]
        for gi in range(len(groups)):
            for gj in range(gi + 1, len(groups)):
                for a in groups[gi]:
                    for b in groups[gj]:
                        if b in adj.get(a, ()):
                            adj[a].remove(b)
                        if a in adj.get(b, ()):
                            adj[b].remove(a)
    # Drop diagonal shortcuts: if a diagonal pair also shares a body neighbour, the
    # diagonal skips around a corner and is not a separate route. Leaving it in gives
    # every L-bend and staircase step degree 3, which would make the promotion below
    # re-split exactly the corners the crossing number was introduced to protect.
    for a in sorted(adj):
        for b in list(adj[a]):
            if (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 != 2:
                continue                       # orthogonal, always a real step
            if any(c in adj.get(b, ()) for c in adj[a] if c != b):
                adj[a].remove(b)
                if a in adj.get(b, ()):
                    adj[b].remove(a)

    # The crossing number cannot see a branch point that is two pixels wide: at an
    # L-shaped 2-wide corner two of the three arms are ring-adjacent, so they merge into
    # one ring run and crossing == 2 while the degree is 3. Measured on cached grids,
    # that silently dropped 1.7% of retained ridge pixels, including one contiguous
    # 9-pixel branch. Promote whatever is still over-connected after the cuts until
    # every remaining component is a simple path or a simple cycle.
    while True:
        extra = {p for p in body if len(adj[p]) >= 3}
        if not extra:
            break
        junctions |= extra
        body -= extra
        for p in sorted(extra):
            for q in adj[p]:
                if p in adj.get(q, ()):
                    adj[q].remove(p)
            del adj[p]

    seen, arcs = set(), []
    for p in sorted(body):
        if p in seen:
            continue
        comp, stack = [], [p]
        seen.add(p)
        while stack:
            q = stack.pop()
            comp.append(q)
            for r in adj[q]:
                if r not in seen:
                    seen.add(r)
                    stack.append(r)
        # Emit arcs until the component is fully covered. One walk per component is
        # only complete if the component is a simple path; a cycle-with-tail leaves
        # the far side of the cycle unwalked, and the leftovers used to be dropped
        # silently -- they appeared in neither `arcs` nor `junctions`.
        left = set(comp)
        while left:
            ends = sorted(q for q in left
                          if len([r for r in adj[q] if r in left]) <= 1)
            start = ends[0] if ends else sorted(left)[0]
            walk, cur, prev = [start], start, None
            left.discard(start)
            while True:
                nxt = [r for r in adj[cur] if r != prev and r in left]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                walk.append(cur)
                left.discard(cur)
            arcs.append(walk)
    return arcs, sorted(junctions)


# ------------------------------------------------------------------ public API

def build_ridge_graph(results_dir: Path, k: int = 8, h_frac: float = 0.10,
                      min_arc: int = 3) -> dict:
    """Rebuild the ridge graph from stored arrays. No GPU, no job state."""
    results_dir = Path(results_dir)
    S = np.load(results_dir / "sensitivity.npy")
    emb_path = results_dir / "embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(
            "embeddings.npy missing for this job. Jobs generated before embeddings "
            "were persisted cannot be re-analysed without regenerating.")
    E = np.load(emb_path).astype(np.float32)
    if E.ndim != 3 or S.shape != E.shape[:2]:
        raise ValueError(f"shape mismatch: sensitivity {S.shape}, embeddings {E.shape}")

    valid = np.isfinite(S) & (np.linalg.norm(E, axis=-1) > 0)
    lab = semantic_labels(E, valid, k)
    ws = watershed_line(S, valid, h_frac)
    meetings: dict = {}
    retained, flank, raw = contract(ws, lab, valid, meetings)
    arcs, junctions = split_arcs(retained)
    # every retained pixel must be in exactly one arc or be a junction; silent loss
    # here understates the ridge and is invisible in the API response
    covered = sum(len(a) for a in arcs) + len(junctions)
    if covered != int(retained.sum()):
        raise AssertionError(
            f"split_arcs lost pixels: {covered} covered of {int(retained.sum())} retained")

    pct = (lambda v: float((S[valid] < v).mean() * 100))
    edges = []
    for i, arc in enumerate(arcs):
        if len(arc) < min_arc:
            continue
        pairs = [flank[p] for p in arc if p in flank]
        if not pairs:
            continue
        seq, uniq = [], []
        for a, b in pairs:
            key = tuple(sorted((a, b)))
            if not seq or seq[-1] != key:
                seq.append(key)
            if key not in uniq:
                uniq.append(key)
        svals = [float(S[y, x]) for y, x in arc if np.isfinite(S[y, x])]
        # Stations where three or more basins meet: `flank` had to name two of them, so
        # the pair there — and any one-station leg it inserts into the itinerary — is a
        # choice, not a reading of the field. Most such pixels are junctions and never
        # reach an arc, but a triple point on the domain edge has one arm cut off, so
        # _crossing sees 2 and it stays. Report them rather than hide the truncation.
        meets = [[int(y), int(x), *meetings[(y, x)]]
                 for y, x in arc if (y, x) in meetings]
        edges.append(dict(
            id=i,
            stations=[[int(y), int(x)] for y, x in arc],
            length=len(arc),
            flank_pairs=[list(p) for p in uniq],
            itinerary=[list(p) for p in seq],
            meetings=meets,
            mean_sensitivity=float(np.mean(svals)) if svals else None,
            s_percentile=pct(float(np.median(svals))) if svals else None,
        ))
    edges.sort(key=lambda e: -e["length"])
    # n_arcs == 0 has four distinct causes and they need different user actions;
    # without this the UI shows the same empty graph for all of them.
    note = None
    if not edges:
        if not valid.any():
            note = "no valid cells: sensitivity is all-NaN or embeddings are all-zero"
        elif len(np.unique(lab[valid])) < 2:
            note = (f"only {len(np.unique(lab[valid]))} semantic basin(s) at k={k}: "
                    "the prompts are too similar to separate, or k is too low")
        elif not retained.any():
            note = (f"watershed lines exist but none separate two basins at "
                    f"h_frac={h_frac}: try a smaller h_frac")
        else:
            note = (f"{len(arcs)} arc(s) found but all shorter than min_arc="
                    f"{min_arc}: lower min_arc or use a finer grid")
    return dict(
        params=dict(k=k, h_frac=h_frac, min_arc=min_arc),
        grid_size=int(S.shape[0]),
        n_basins=int(len(np.unique(lab[valid]))),
        n_arcs=len(edges),
        note=note,
        junctions=[[int(y), int(x)] for y, x in junctions],
        edges=edges,
    )


def render_itinerary(results_dir: Path, graph: dict, edge_id: int,
                     thumb: int = 96, per_row: int = 12) -> bytes:
    """Filmstrip along one arc, sliced out of the stored images.png montage."""
    import io

    from PIL import Image, ImageDraw

    from backend import config

    edge = next((e for e in graph["edges"] if e["id"] == edge_id), None)
    if edge is None:
        raise KeyError(f"no arc with id {edge_id}")
    # a 192x192 montage of 256px thumbnails is 2.4e9 px, well past PIL's 1.79e8
    # DecompressionBombError limit; this montage is ours, so raise the ceiling
    # rather than let a legitimate render fail as a suspected attack.
    Image.MAX_IMAGE_PIXELS = None
    montage = Image.open(Path(results_dir) / "images.png").convert("RGB")
    gs = graph["grid_size"]
    ts = config.THUMBNAIL_SIZE

    tiles = []
    for n, (alpha_idx, beta_idx) in enumerate(edge["stations"]):
        # assemble_image_grid lays out row = gs-1-beta_idx, col = alpha_idx
        row, col = gs - 1 - beta_idx, alpha_idx
        box = (col * ts, row * ts, (col + 1) * ts, (row + 1) * ts)
        cell = montage.crop(box).resize((thumb, thumb), Image.LANCZOS)
        t = Image.new("RGB", (thumb, thumb + 14), "black")
        t.paste(cell, (0, 0))
        ImageDraw.Draw(t).text((3, thumb + 2), str(n), fill="white")
        tiles.append(t)

    rows = [tiles[i:i + per_row] for i in range(0, len(tiles), per_row)]
    out = Image.new("RGB", (per_row * thumb, len(rows) * (thumb + 14)), "black")
    for r, row_tiles in enumerate(rows):
        for c, t in enumerate(row_tiles):
            out.paste(t, (c * thumb, r * (thumb + 14)))
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def graph_path(results_dir: Path) -> Path:
    return Path(results_dir) / "ridge_graph.json"


def save_graph(results_dir: Path, graph: dict) -> Path:
    p = graph_path(results_dir)
    p.write_text(json.dumps(graph))
    return p
