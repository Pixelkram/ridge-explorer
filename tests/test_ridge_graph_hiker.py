"""Regression tests for the ridge-graph and hiker additions.

Runs without a GPU: the generation path is exercised only through the FastAPI layer with
a stubbed pool, everything else uses cached grids or synthetic fields.

    python3 -m pytest tests/test_ridge_graph_hiker.py -x -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402
from backend.services import ridge_graph as rg  # noqa: E402
from backend.services import hiker as hk  # noqa: E402

CACHED = Path("/home/student/ai/search_problem/outputs/22_definitive_10step/triplet_09")
HAS_CACHE = (CACHED / "dino_sensitivity_avg.npy").exists()


# --------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def real_grid():
    if not HAS_CACHE:
        pytest.skip("cached reference grid unavailable")
    S = np.load(CACHED / "dino_sensitivity_avg.npy")
    z = np.load(CACHED / "embeddings_seed42.npz")
    E = z["dino"].astype(np.float32)
    valid = z["valid"]
    return np.where(valid, S, np.nan), E, valid


@pytest.fixture(scope="module")
def job_dir(tmp_path_factory, real_grid):
    S, E, _ = real_grid
    d = tmp_path_factory.mktemp("job")
    np.save(d / "sensitivity.npy", S)
    np.save(d / "embeddings.npy", E)
    return d


# --------------------------------------------------------- ridge_graph

def test_graph_has_arcs_and_flanking_pairs(job_dir):
    g = rg.build_ridge_graph(job_dir, k=8, h_frac=0.10)
    assert g["n_arcs"] > 0
    assert g["n_basins"] >= 2
    for e in g["edges"]:
        assert e["length"] == len(e["stations"])
        assert e["flank_pairs"], "every retained arc must separate two basins"
        for a, b in e["flank_pairs"]:
            assert a != b, "a flanking pair must name two DIFFERENT basins"
        assert 0.0 <= e["s_percentile"] <= 100.0


def test_arcs_are_contiguous_walks(job_dir):
    """Stations must form a connected path in 8-neighbour order — the filmstrip is
    meaningless if the walk jumps."""
    g = rg.build_ridge_graph(job_dir, k=8, h_frac=0.10)
    for e in g["edges"]:
        pts = e["stations"]
        for (y0, x0), (y1, x1) in zip(pts, pts[1:]):
            assert max(abs(y1 - y0), abs(x1 - x0)) == 1, \
                f"arc {e['id']} jumps from {(y0, x0)} to {(y1, x1)}"


def test_arcs_sit_high_on_the_sensitivity_field(job_dir, real_grid):
    S, _, _ = real_grid
    g = rg.build_ridge_graph(job_dir, k=8, h_frac=0.10)
    vals = [S[y, x] for e in g["edges"] for y, x in e["stations"] if np.isfinite(S[y, x])]
    finite = S[np.isfinite(S)]
    pct = float((finite < np.median(vals)).mean() * 100)
    assert pct > 70, f"arcs should lie on crests; median sits at the {pct:.0f}th percentile"


def test_missing_embeddings_raises_clearly(tmp_path, real_grid):
    S, _, _ = real_grid
    np.save(tmp_path / "sensitivity.npy", S)
    with pytest.raises(FileNotFoundError) as exc:
        rg.build_ridge_graph(tmp_path)
    assert "embeddings.npy" in str(exc.value)


def test_shape_mismatch_raises(tmp_path, real_grid):
    S, E, _ = real_grid
    np.save(tmp_path / "sensitivity.npy", S)
    np.save(tmp_path / "embeddings.npy", E[:10, :10])
    with pytest.raises(ValueError):
        rg.build_ridge_graph(tmp_path)


def test_single_basin_grid_yields_no_arcs(tmp_path):
    """A uniform field has no boundary to contract onto; it must return cleanly."""
    gs = 12
    S = np.full((gs, gs), 0.5)
    E = np.tile(np.eye(1, 768, 0).astype(np.float32), (gs, gs, 1))
    np.save(tmp_path / "sensitivity.npy", S)
    np.save(tmp_path / "embeddings.npy", E)
    g = rg.build_ridge_graph(tmp_path, k=8)
    assert g["n_arcs"] == 0


def test_tiny_grid_does_not_crash(tmp_path):
    gs = 4
    rs = np.random.default_rng(0)
    S = rs.random((gs, gs))
    E = rs.random((gs, gs, 768)).astype(np.float32)
    E /= np.linalg.norm(E, axis=-1, keepdims=True)
    np.save(tmp_path / "sensitivity.npy", S)
    np.save(tmp_path / "embeddings.npy", E)
    g = rg.build_ridge_graph(tmp_path, k=8)
    assert isinstance(g["n_arcs"], int)


def test_k_greater_than_cell_count_is_safe(tmp_path):
    gs = 5
    rs = np.random.default_rng(1)
    S = rs.random((gs, gs))
    E = rs.random((gs, gs, 768)).astype(np.float32)
    E /= np.linalg.norm(E, axis=-1, keepdims=True)
    np.save(tmp_path / "sensitivity.npy", S)
    np.save(tmp_path / "embeddings.npy", E)
    g = rg.build_ridge_graph(tmp_path, k=500)
    assert isinstance(g["n_arcs"], int)


def test_itinerary_crop_matches_montage_layout(job_dir):
    """The montage is laid out row = gs-1-beta, col = alpha. Paint each cell a unique
    colour and check the filmstrip returns the cells the arc actually visits — an
    off-by-one here silently shows the wrong images."""
    from PIL import Image

    g = rg.build_ridge_graph(job_dir, k=8, h_frac=0.10)
    gs, ts = g["grid_size"], config.THUMBNAIL_SIZE
    canvas = np.zeros((gs * ts, gs * ts, 3), np.uint8)
    for i in range(gs):
        for j in range(gs):
            canvas[(gs - 1 - j) * ts:(gs - j) * ts, i * ts:(i + 1) * ts] = (i, j, 7)
    Image.fromarray(canvas).save(job_dir / "images.png")

    edge = g["edges"][0]
    data = rg.render_itinerary(job_dir, g, edge["id"], thumb=ts, per_row=len(edge["stations"]))
    strip = np.array(Image.open(__import__("io").BytesIO(data)).convert("RGB"))
    for n, (i, j) in enumerate(edge["stations"]):
        px = strip[ts // 2, n * ts + ts // 2]
        assert abs(int(px[0]) - i) <= 2 and abs(int(px[1]) - j) <= 2, \
            f"station {n} shows cell {(px[0], px[1])}, expected {(i, j)}"


def test_unknown_edge_id_raises(job_dir):
    g = rg.build_ridge_graph(job_dir, k=8, h_frac=0.10)
    with pytest.raises(KeyError):
        rg.render_itinerary(job_dir, g, 99999)


# --------------------------------------------------------------- hiker

def test_bary_and_simplex_points():
    gs = 10
    assert np.allclose(hk.bary(0, 0, gs), [1, 0, 0])
    assert np.allclose(hk.bary(gs - 1, 0, gs), [0, 1, 0])
    assert np.allclose(hk.bary(0, gs - 1, gs), [0, 0, 1])
    pts = hk.simplex_points(gs)
    assert len(pts) == gs * (gs + 1) // 2
    for i, j in pts:
        w = hk.bary(i, j, gs)
        assert w.min() >= -1e-9 and abs(w.sum() - 1) < 1e-9


def test_walk_finds_arcs_on_real_grid(real_grid):
    _, E, valid = real_grid
    w = hk.walk(E.astype(np.float64), valid, 50, k_basins=6, h_frac=0.10)
    assert w is not None and len(w["arcs"]) > 0
    for a in w["arcs"]:
        assert len(a) >= hk.MIN_ARC


def test_walk_returns_none_on_uniform_field():
    gs = 12
    E = np.tile(np.eye(1, 768, 0), (gs, gs, 1))
    assert hk.walk(E, np.ones((gs, gs), bool), gs) is None


def test_novelty_and_reach_semantics(real_grid):
    _, E, valid = real_grid
    X = E[valid].astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    assert np.isnan(hk.novelty(X[:20], []))
    assert hk.novelty(X[:20], [X]) == 0.0, "everything is already in the archive"
    assert hk.reach(X[0], []) == 0.0
    assert hk.reach(X[0], [X]) == pytest.approx(1.0, abs=1e-4)


def _text_enc():
    import open_clip
    import torch
    m, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu")
    m.eval()
    tok = open_clip.get_tokenizer("ViT-B-32")

    def enc(texts):
        with torch.no_grad():
            f = m.encode_text(tok(list(texts)))
            return (f / f.norm(dim=-1, keepdim=True)).numpy()
    return enc


@pytest.fixture(scope="module")
def text_enc():
    return _text_enc()


def test_chain_extends_and_respects_guards(text_enc):
    pool = config.HIKE_PROMPT_POOL
    gs = 20
    ch = hk.Chain(pool[:3], [[1., 0, 0], [0, 1., 0], [0, 0, 1.]], pool[:3])
    for hop in range(5):
        nc = hk.extend(ch, [(gs - 1, 0), (0, gs - 1), (0, 0), (9, 8)][hop % 4], gs, pool, text_enc)
        assert nc is not None, f"extend failed at hop {hop}"
        assert len(nc.coefs) == 3
        assert all(len(r) == len(nc.basis) for r in nc.coefs)
        assert abs(sum(nc.coefs[0]) - 1.0) < 1e-6, "exit coefficients must stay affine"
        L = text_enc(nc.labels)
        M = L @ L.T
        assert M[np.triu_indices(3, 1)].max() <= hk.DIV_MAX + 1e-6
        assert not nc.labels[0].startswith("exit(exit("), "labels must not nest"
        ch = nc


def test_distinct_exits_give_distinct_successors(text_enc):
    """If nearby exits collapsed to the same prompt pair the population could not
    diversify — this guards the beam against silent collapse."""
    pool = config.HIKE_PROMPT_POOL
    gs = 18
    ch = hk.Chain(pool[:3], [[1., 0, 0], [0, 1., 0], [0, 0, 1.]], pool[:3])
    keys = set()
    for e in [(gs - 1, 0), (0, gs - 1), (0, 0), (9, 8), (3, 14)]:
        nc = hk.extend(ch, e, gs, pool, text_enc)
        if nc:
            keys.add("|".join(nc.labels))
    assert len(keys) >= 4, f"only {len(keys)} distinct successors from 5 exits"


def test_exhausted_pool_returns_none(text_enc):
    pool = config.HIKE_PROMPT_POOL[:4]
    gs = 12
    ch = hk.Chain(pool[:3], [[1., 0, 0], [0, 1., 0], [0, 0, 1.]], pool[:3])
    assert hk.extend(ch, (gs - 1, 0), gs, pool, text_enc) is None


def test_prompt_pool_supports_a_real_hike():
    """Each hop consumes two prompts; a silent shortfall stopped hikes after one hop."""
    assert len(config.HIKE_PROMPT_POOL) >= 2 * 10 + 3


# ------------------------------------------------------------ endpoints

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    import backend.main as m
    return TestClient(m.app)


def test_graph_endpoints_roundtrip(client, real_grid):
    S, E, _ = real_grid
    jid = "pytest_graph_job"
    d = config.RESULTS_DIR / jid
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "sensitivity.npy", S)
    np.save(d / "embeddings.npy", E)
    r = client.post(f"/api/grid/{jid}/graph", json={"k": 8, "h_frac": 0.10})
    assert r.status_code == 200 and r.json()["n_arcs"] > 0
    g = client.get(f"/api/grid/{jid}/ridge_graph.json").json()
    assert g["n_arcs"] == r.json()["n_arcs"]


def test_graph_endpoint_reports_missing_inputs(client):
    r = client.post("/api/grid/definitely_not_a_job/graph", json={})
    assert r.status_code == 200
    assert "error" in r.json()["params"]


def test_hike_status_unknown_id(client):
    r = client.get("/api/grid/hike/nope/status")
    assert r.status_code == 200 and r.json()["status"] == "unknown"


# ------------------------------------------------- arc splitting (regression)
#
# split_arcs used a raw 8-neighbour COUNT to find junctions. On an 8-connected
# watershed line an L-bend has two neighbours and a staircase step has three, so
# ordinary corners were declared junctions: 82-94% false positives on real fields,
# which shredded arcs and cut their length to about a third. These pin the fix.

def _mask(pts, n=6):
    a = np.zeros((n, n), bool)
    for y, x in pts:
        a[y, x] = True
    return a


@pytest.mark.parametrize("name,pts,n_junc,n_arcs,longest", [
    ("straight",       {(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)}, 0, 1, 5),
    ("L-bend",         {(1, 1), (2, 1), (3, 1), (3, 2), (3, 3)}, 0, 1, 5),
    ("staircase",      {(0, 0), (1, 0), (1, 1), (2, 1),
                        (2, 2), (3, 2), (3, 3)},                 0, 1, 7),
    ("T",              {(0, 2), (1, 2), (2, 2), (3, 2),
                        (2, 0), (2, 1)},                         1, 3, 2),
    ("X",              {(0, 2), (1, 2), (2, 2), (3, 2), (4, 2),
                        (2, 0), (2, 1), (2, 3), (2, 4)},         1, 4, 2),
])
def test_corners_are_not_junctions_but_real_branches_are(name, pts, n_junc, n_arcs, longest):
    arcs, junctions = rg.split_arcs(_mask(pts))
    assert len(junctions) == n_junc, f"{name}: junctions {junctions}"
    assert len(arcs) == n_arcs, f"{name}: arcs {arcs}"
    assert max(len(a) for a in arcs) == longest, f"{name}: arcs {arcs}"
    # every pixel is accounted for: in an arc or declared a junction, never orphaned
    assert sum(len(a) for a in arcs) + len(junctions) == len(pts), f"{name}: lost pixels"


def test_arms_of_a_junction_do_not_merge_diagonally():
    """Deleting the junction pixel leaves a T's arms diagonally adjacent, so they used
    to reconnect into one component and the branch vanished."""
    arcs, junctions = rg.split_arcs(_mask({(0, 2), (1, 2), (2, 2), (3, 2), (2, 0), (2, 1)}))
    assert len(junctions) == 1
    bodies = [set(a) for a in arcs]
    assert {(0, 2), (1, 2)} in bodies
    assert {(2, 0), (2, 1)} in bodies
    assert {(3, 2)} in bodies


def test_ring_components_group_adjacent_neighbours():
    pts = {(1, 1), (0, 1), (0, 2)}          # two neighbours, adjacent to each other
    assert len(rg._ring_components((1, 1), pts)) == 1
    pts = {(1, 1), (0, 1), (2, 1)}          # two neighbours, opposite sides
    assert len(rg._ring_components((1, 1), pts)) == 2


def test_arcs_are_longer_after_the_junction_fix(real_grid):
    """On a real field the corrected splitter must not fragment more than the old one."""
    S, E, ok = real_grid
    valid = ok & np.isfinite(S) & (np.linalg.norm(E, axis=-1) > 0)
    lab = rg.semantic_labels(E, valid, 8)
    ws = rg.watershed_line(S, valid, 0.10)
    retained, _flank, _raw = rg.contract(ws, lab, valid)

    arcs, junctions = rg.split_arcs(retained)
    pts = set(zip(*(a.tolist() for a in np.nonzero(retained))))
    old_junc = {p for p in pts
                if sum((p[0] + dy, p[1] + dx) in pts for dy, dx in rg.NB8) >= 3}
    assert len(junctions) <= len(old_junc)
    assert max((len(a) for a in arcs), default=0) >= 3
    for a in arcs:                                  # still contiguous 8-walks
        for (y0, x0), (y1, x1) in zip(a, a[1:]):
            assert max(abs(y0 - y1), abs(x0 - x1)) == 1


def test_empty_graph_explains_itself(tmp_path):
    """n_arcs == 0 has four causes; the response must say which one."""
    gs = 12
    S = np.full((gs, gs), np.nan)
    for i in range(gs):
        for j in range(gs - i):
            S[i, j] = 0.01
    E = np.zeros((gs, gs, 16), np.float32)
    E[np.isfinite(S)] = np.eye(16, dtype=np.float32)[0]     # one basin only
    np.save(tmp_path / "sensitivity.npy", S)
    np.save(tmp_path / "embeddings.npy", E)
    g = rg.build_ridge_graph(tmp_path, k=8, h_frac=0.10, min_arc=3)
    assert g["n_arcs"] == 0
    assert g["note"] and "basin" in g["note"]


# ------------------------------------------------- arc coverage (regression 2)
#
# The first junction fix was incomplete. `_crossing` cannot see a branch point two
# pixels wide -- at an L-shaped 2-wide corner two arms are ring-adjacent, so they
# merge into one ring run and crossing == 2 while degree == 3. split_arcs then also
# emitted only ONE walk per component and dropped the rest. Measured on cached grids
# that lost 1.7% of retained ridge pixels, including a contiguous 9-pixel branch.

def _cover(mask):
    arcs, junctions = rg.split_arcs(mask)
    return sum(len(a) for a in arcs) + len(junctions), int(mask.sum()), arcs, junctions


def test_thick_branch_point_loses_no_pixels():
    rows = ["..#....", "..#..#.", "..###..", "..##...", ".#..#..", "....#..", "....#.."]
    m = np.array([[c == "#" for c in r] for r in rows])
    covered, total, _arcs, _j = _cover(m)
    assert covered == total, f"lost {total - covered} of {total} pixels"


def test_cycle_with_tail_loses_no_pixels():
    """One walk per component leaves the far side of a cycle unwalked."""
    m = np.zeros((7, 7), bool)
    for y, x in {(1, 1), (1, 2), (1, 3), (2, 1), (2, 3),
                 (3, 1), (3, 2), (3, 3), (4, 2), (5, 2)}:
        m[y, x] = True
    covered, total, _a, _j = _cover(m)
    assert covered == total, f"lost {total - covered} of {total} pixels"


def test_solid_block_is_all_junction_not_silently_dropped():
    m = np.zeros((6, 6), bool)
    m[2:4, 2:4] = True
    covered, total, _a, _j = _cover(m)
    assert covered == total


@pytest.mark.parametrize("seed", range(12))
def test_random_masks_lose_no_pixels(seed):
    from scipy import ndimage
    rs = np.random.default_rng(seed)
    f = ndimage.gaussian_filter(rs.normal(size=(40, 40)), 2.0)
    mk = ndimage.binary_closing(f > np.percentile(f, 88))
    sk = mk & ~ndimage.binary_erosion(mk)
    if sk.sum() < 5:
        pytest.skip("degenerate mask")
    covered, total, _a, _j = _cover(sk)
    assert covered == total, f"seed {seed}: lost {total - covered} of {total}"


def test_every_arc_pixel_appears_exactly_once(real_grid):
    S, E, ok = real_grid
    valid = ok & np.isfinite(S) & (np.linalg.norm(E, axis=-1) > 0)
    lab = rg.semantic_labels(E, valid, 8)
    ws = rg.watershed_line(S, valid, 0.10)
    retained, _f, _r = rg.contract(ws, lab, valid)
    arcs, junctions = rg.split_arcs(retained)
    flat = [p for a in arcs for p in a]
    assert len(flat) == len(set(flat)), "a pixel appears in two arcs"
    assert set(flat).isdisjoint(set(junctions))
    assert len(flat) + len(junctions) == int(retained.sum())


# ------------------------------------------------------- hike simplex map

def test_hike_map_renders_and_places_stations_correctly(tmp_path):
    """The map must use the same layout as the montage and render_itinerary
    (display row = gs-1-beta, display col = alpha), or a station appears in a
    different place here than in the main grid view."""
    from PIL import Image
    gs, cell = 8, 30
    S = np.full((gs, gs), np.nan)
    valid = np.zeros((gs, gs), bool)
    thumbs = {}
    for i in range(gs):
        for j in range(gs - i):
            S[i, j] = 0.1 + 0.02 * (i + j)
            valid[i, j] = True
            # paint each cell a unique colour so we can find it again
            thumbs[(i, j)] = _png_bytes((i * 8 % 256, j * 8 % 256, 7))
    best = [(0, 0), (1, 0), (2, 0), (3, 0)]
    arcs = [best, [(0, 3), (0, 4)]]
    out = tmp_path / "map.jpg"
    hk.render_hike_map(thumbs, S, valid, arcs, best, gs, out,
                       exit_ij=best[-1], junctions=[(2, 2)], cell=cell)
    assert out.exists() and out.stat().st_size > 0
    im = np.array(Image.open(out).convert("RGB"))
    assert im.shape == (gs * cell, gs * cell, 3)
    # a cell far from any drawn arc must still carry its own colour signature
    i, j = 1, 5
    px = im[(gs - 1 - j) * cell + cell // 2, i * cell + cell // 2]
    assert abs(int(px[0]) - i * 8) < 90 and abs(int(px[1]) - j * 8) < 90, \
        f"cell ({i},{j}) not at the expected display position, got {px}"


def _png_bytes(rgb):
    import io as _io
    from PIL import Image
    b = _io.BytesIO()
    Image.new("RGB", (16, 16), rgb).save(b, format="PNG")
    return b.getvalue()


def test_hike_map_survives_missing_thumbnails_and_all_nan(tmp_path):
    """A hop that lost cells must still produce a map rather than abort the hike."""
    gs = 6
    S = np.full((gs, gs), np.nan)
    valid = np.zeros((gs, gs), bool)
    out = tmp_path / "m.jpg"
    hk.render_hike_map({}, S, valid, [], [], gs, out, exit_ij=None)
    assert out.exists()


# --------------------------------------------------- ridge refinement (hikes)

def test_doubled_lattice_reproduces_coarse_weights_exactly():
    """The whole refinement rests on bary(i,j,gs) == bary(2i,2j,2gs-1). If that drifts,
    refinement silently relocates every image already generated."""
    for gs in (5, 8, 14, 20):
        gs2 = 2 * gs - 1
        for i in range(gs):
            for j in range(gs - i):
                assert np.allclose(hk.bary(i, j, gs), hk.bary(2 * i, 2 * j, gs2)), (gs, i, j)


def test_upscale_lattice_preserves_cells_and_leaves_gaps_invalid():
    gs, D = 6, 8
    rs = np.random.default_rng(0)
    dino = rs.normal(size=(gs, gs, D))
    valid = np.zeros((gs, gs), bool)
    thumbs = {}
    for i in range(gs):
        for j in range(gs - i):
            valid[i, j] = True
            thumbs[(i, j)] = b"x"
    d2, v2, t2, gs2, s2 = hk.upscale_lattice(dino, valid, thumbs, gs)
    assert gs2 == 2 * gs - 1
    # a carried cell now stands for a 2x2 block of the finer lattice
    assert all(v == 2 for v in s2.values()), "spans did not double with the lattice"
    assert v2.sum() == valid.sum(), "a generated cell was lost"
    for i in range(gs):
        for j in range(gs - i):
            assert np.allclose(d2[2 * i, 2 * j], dino[i, j])
            assert t2[(2 * i, 2 * j)] == thumbs[(i, j)]
    # the interleaved cells must be marked ungenerated, not zero-and-valid
    assert not v2[1, 0] and not v2[0, 1]


def test_refine_targets_stay_in_simplex_and_avoid_existing():
    gs2 = 15
    have = {(0, 0), (2, 0), (4, 0)}
    arcs = [[(0, 0), (1, 0), (2, 0)]]          # coarse indices -> doubled at 0,2,4
    tgt = hk.refine_targets(arcs, gs2, have, radius=1, cap=1000)
    assert tgt, "no refinement targets proposed"
    for (i, j) in tgt:
        assert (i, j) not in have, "proposed a cell that already exists"
        assert i >= 0 and j >= 0 and i + j <= gs2 - 1, f"({i},{j}) outside the simplex"
    assert len(set(tgt)) == len(tgt), "duplicate targets"


def test_refine_targets_respects_cap_and_is_deterministic():
    gs2 = 31
    arcs = [[(i, 0) for i in range(10)], [(0, j) for j in range(10)]]
    a = hk.refine_targets(arcs, gs2, set(), radius=2, cap=25)
    b = hk.refine_targets(arcs, gs2, set(), radius=2, cap=25)
    assert len(a) == 25 and a == b


def test_refined_map_paints_coarse_cells_over_their_whole_block(tmp_path):
    """After a refine, a coarse cell covers a 2x2 block. Painting every thumbnail at one
    size left the unrefined ground looking like holes -- the bug this pins."""
    from PIL import Image
    gs, cell = 9, 20
    S = np.full((gs, gs), np.nan)
    valid = np.zeros((gs, gs), bool)
    thumbs, spans = {}, {}
    for i in range(0, gs, 2):                       # coarse cells only, span 2
        for j in range(0, gs - i, 2):
            S[i, j] = 0.5
            valid[i, j] = True
            thumbs[(i, j)] = _png_bytes((200, 40, 40))
            spans[(i, j)] = 2
    out = tmp_path / "m.jpg"
    hk.render_hike_map(thumbs, S, valid, [], [], gs, out, cell=cell, spans=spans)
    im = np.array(Image.open(out).convert("RGB"))

    # the cell diagonally between four coarse anchors must be painted, not background
    i, j = 1, 1
    px = im[(gs - 1 - j) * cell + cell // 2, i * cell + cell // 2]
    assert int(px.max()) > 60, f"gap at ({i},{j}) left unpainted: {px}"

    # and a finer cell must win over the coarse block it subdivides
    thumbs[(1, 1)] = _png_bytes((30, 30, 240)); spans[(1, 1)] = 1; S[1, 1] = 0.5
    valid[1, 1] = True
    hk.render_hike_map(thumbs, S, valid, [], [], gs, out, cell=cell, spans=spans)
    im2 = np.array(Image.open(out).convert("RGB"))
    px2 = im2[(gs - 1 - 1) * cell + cell // 2, 1 * cell + cell // 2]
    assert int(px2[2]) > int(px2[0]), f"fine cell did not paint over the coarse block: {px2}"


def test_hiker_sensitivity_matches_ridge_detector():
    """hiker.sensitivity duplicates ridge_detector._sensitivity_2d. They disagreed for
    months (flat mean vs per-axis); pin them together so the hiker and the main grid
    cannot drift apart again."""
    from backend.services import ridge_detector as rd
    rs = np.random.default_rng(3)
    gs = 12
    E = rs.normal(size=(gs, gs, 24))
    E /= np.linalg.norm(E, axis=-1, keepdims=True)
    valid = np.zeros((gs, gs), bool)
    for i in range(gs):
        for j in range(gs - i):
            valid[i, j] = True
    E[~valid] = 0.0                       # ridge_detector detects gaps by zero norm
    a = hk.sensitivity(E, valid)
    b = rd.compute_sensitivity(E)
    m = valid & np.isfinite(a)
    assert np.allclose(a[m], b[m], atol=1e-9), \
        f"max |diff| = {np.nanmax(np.abs(a[m] - b[m])):.2e}"


def test_per_axis_changes_three_neighbour_edges_not_the_hypotenuse():
    """The fix corrects the alpha=0 / beta=0 edges (3 neighbours, lopsided axes) and
    provably cannot change the hypotenuse (1 neighbour per axis, already balanced)."""
    gs = 10
    rs = np.random.default_rng(4)
    E = rs.normal(size=(gs, gs, 16))
    E /= np.linalg.norm(E, axis=-1, keepdims=True)
    valid = np.zeros((gs, gs), bool)
    for i in range(gs):
        for j in range(gs - i):
            valid[i, j] = True

    def flat(i, j):
        d = [1.0 - float(E[i, j] @ E[i + di, j + dj])
             for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
             if 0 <= i + di < gs and 0 <= j + dj < gs and valid[i + di, j + dj]]
        return float(np.mean(d)) if d else np.nan

    S = hk.sensitivity(E, valid)
    hyp = [(i, gs - 1 - i) for i in range(1, gs - 1)]
    assert all(abs(S[i, j] - flat(i, j)) < 1e-9 for i, j in hyp), \
        "hypotenuse must be unchanged: one neighbour per axis is already balanced"
    edge = [(0, j) for j in range(1, gs - 2)]
    assert any(abs(S[i, j] - flat(i, j)) > 1e-6 for i, j in edge), \
        "the alpha=0 edge should have been reweighted"


# ------------------------------------------------------- simplex padding (margin)

def test_margin_zero_is_the_old_lattice_exactly():
    for gs in (8, 14, 20):
        assert hk.simplex_points(gs, 0) == [(i, j) for i in range(gs) for j in range(gs - i)]
        for i, j in hk.simplex_points(gs, 0):
            assert np.allclose(hk.bary(i, j, gs), hk.bary(i, j, gs, 0))
            assert hk.on_hull(i, j, gs, 0)


@pytest.mark.parametrize("gs,m", [(8, 1), (10, 2), (14, 3), (20, 2)])
def test_padding_surrounds_all_three_edges(gs, m):
    """All three hull edges must gain a ring; a cell on any of them needs a full
    4-neighbourhood once padded, which is the entire point."""
    pts = set(hk.simplex_points(gs, m))
    inner = [p for p in pts if hk.on_hull(*p, gs, m)]
    assert len(inner) == gs * (gs + 1) // 2, "the true simplex changed size"
    # every true-simplex cell now has all four neighbours present in the padded lattice
    for i, j in inner:
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            assert (i + di, j + dj) in pts, f"({i},{j}) still missing a neighbour"


@pytest.mark.parametrize("gs,m", [(8, 1), (10, 2), (14, 2)])
def test_padded_weights_are_affine_and_ring_is_outside_the_hull(gs, m):
    for i, j in hk.simplex_points(gs, m):
        w = hk.bary(i, j, gs, m)
        assert abs(w.sum() - 1.0) < 1e-9, "weights must stay affine"
        assert hk.on_hull(i, j, gs, m) == bool(w.min() >= -1e-9), \
            "on_hull must agree with the sign of the smallest weight"
    # the three corners of the TRUE simplex still map to the pure prompts
    assert np.allclose(hk.bary(m, m, gs, m), [1, 0, 0])
    assert np.allclose(hk.bary(gs - 1 + m, m, gs, m), [0, 1, 0])
    assert np.allclose(hk.bary(m, gs - 1 + m, gs, m), [0, 0, 1])


def test_padding_removes_the_missing_neighbour_problem():
    """The measured motivation: unpadded hull cells average 2.6 of 4 neighbours."""
    gs, m = 20, 2
    def mean_nbrs(margin):
        pts = set(hk.simplex_points(gs, margin))
        hull = [p for p in pts if hk.on_hull(*p, gs, margin)
                and min(hk.bary(*p, gs, margin)) < 1e-9]
        return np.mean([sum((p[0] + di, p[1] + dj) in pts
                            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)))
                        for p in hull])
    assert mean_nbrs(0) < 3.0, "precondition: unpadded hull is under-sampled"
    assert mean_nbrs(m) == 4.0, "padded hull cells must have all four neighbours"


def test_doubling_preserves_padded_coordinates():
    """Refinement doubles the lattice AND the margin; weights must be identical."""
    gs, m = 10, 2
    for i, j in hk.simplex_points(gs, m):
        assert np.allclose(hk.bary(i, j, gs, m), hk.bary(2 * i, 2 * j, 2 * gs - 1, 2 * m))


# ------------------------------------------------ approach path (entry continuity)

def _tri_valid(gs, m=0):
    v = np.zeros((gs + 3 * m, gs + 3 * m), bool)
    for i, j in hk.simplex_points(gs, m):
        v[i, j] = True
    return v


def test_approach_path_is_contiguous_and_half_open():
    v = _tri_valid(14)
    for target in [(4, 3), (9, 0), (0, 9), (1, 1), (6, 6)]:
        p = hk.approach_path((0, 0), target, v)
        if not p:
            continue
        assert p[0] == (0, 0), "must start at the entry cell"
        assert target not in p, "target belongs to the arc, not the approach"
        full = p + [target]
        for a, b in zip(full, full[1:]):
            assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1, f"jump {a}->{b}"
        for c in p:
            assert v[c], "approach must stay on generated cells"


def test_approach_path_degenerate_cases():
    v = _tri_valid(12)
    assert hk.approach_path((3, 3), (3, 3), v) == [], "no path to itself"
    assert hk.approach_path((3, 4), (3, 3), v) == [(3, 4)], "adjacent target: one cell"
    dead = np.zeros_like(v); dead[0, 0] = True          # entry with no valid neighbour
    assert hk.approach_path((0, 0), (5, 5), dead) == [], "unreachable target yields nothing"
    assert hk.approach_path((0, 0), (5, 5), np.zeros_like(v)) == [], "invalid entry"


def test_approach_path_respects_the_padding_margin():
    gs, m = 12, 2
    v = _tri_valid(gs, m)
    p = hk.approach_path((m, m), (m + 4, m + 3), v)     # vertex A of a padded lattice
    assert p and p[0] == (m, m)
    for c in p:
        assert v[c]


def test_prepending_leaves_the_exit_untouched():
    """The approach must be display-only: it changes the FRONT of the walk, so the exit
    (a[-1]) -- the only thing that moves the search -- is unaffected."""
    v = _tri_valid(14)
    arc = [(6, 3), (6, 2), (6, 1), (7, 0)]
    p = hk.approach_path((0, 0), arc[0], v)
    walk = p + arc
    assert walk[-1] == arc[-1], "the exit changed"
    assert walk[len(p):] == arc, "the ridge portion changed"
