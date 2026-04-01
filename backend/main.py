import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend import config
from backend.services.gpu_pool import GPUPool, CellResult, LatentResult, LatentBatchResult
from backend.services.ridge_detector import compute_sensitivity, compute_clusters, classify_ridges
from backend.services.visualization import render_heatmap, assemble_image_grid, render_overlay, render_clusters
from backend.cache.thumbnail_cache import ThumbnailCache
from backend.routers import health, grid

import numpy as np


async def result_collector(app: FastAPI):
    """Background task that drains GPU results and updates job state."""
    pool: GPUPool = app.state.gpu_pool
    cache: ThumbnailCache = app.state.cache
    jobs: dict = app.state.jobs

    while True:
        results = pool.collect_results()
        for r in results:
            job = jobs.get(r.job_id)
            if not job:
                continue

            if isinstance(r, LatentBatchResult):
                # Fast scan batch result — entire row of latent vectors
                if job["latents"] is None:
                    gs = job["grid_size"]
                    dim = r.latent_vectors.shape[1]
                    job["latents"] = np.zeros((gs, gs, dim))
                for idx, col in enumerate(r.cols):
                    job["latents"][r.row, col] = r.latent_vectors[idx]
                    job["cells"][r.row][col]["status"] = "scanned"
                job["cells_generated"] += len(r.cols)

                if job["cells_generated"] >= job["total_cells"] and job["phase"] == "scanning":
                    job["phase"] = "analyzing"
                    asyncio.create_task(asyncio.to_thread(
                        _analyze_fast_scan, job, r.job_id
                    ))

            elif isinstance(r, LatentResult):
                # Fast scan single result (legacy path)
                if job["latents"] is None:
                    gs = job["grid_size"]
                    dim = r.latent_vector.shape[0]
                    job["latents"] = np.zeros((gs, gs, dim))
                job["latents"][r.row, r.col] = r.latent_vector
                job["cells"][r.row][r.col]["status"] = "scanned"
                job["cells_generated"] += 1

                if job["cells_generated"] >= job["total_cells"] and job["phase"] == "scanning":
                    job["phase"] = "analyzing"
                    asyncio.create_task(asyncio.to_thread(
                        _analyze_fast_scan, job, r.job_id
                    ))

            elif isinstance(r, CellResult):
                is_3d = job.get("dimensions", 2) == 3

                # Find the cell entry
                if is_3d:
                    cell = None
                    try:
                        cell = job["cells"][r.row][r.col][r.depth]
                    except (IndexError, TypeError):
                        pass
                else:
                    cell = None
                    try:
                        cell = job["cells"][r.row][r.col]
                    except (IndexError, TypeError):
                        pass

                if cell is None:
                    continue

                # Save thumbnail
                key = (r.row, r.col, r.depth) if is_3d else (r.row, r.col)
                cache.save(r.thumbnail_hash, r.thumbnail_bytes)
                job["thumbnails"][key] = r.thumbnail_bytes
                job["thumbnail_hashes"][key] = r.thumbnail_hash

                # Store DINOv2 embedding
                if is_3d:
                    job["embeddings"][r.row, r.col, r.depth] = r.dino_embedding
                else:
                    job["embeddings"][r.row, r.col] = r.dino_embedding

                # Update cell status
                status = "hq" if r.is_hq else "generated"
                cell["status"] = status
                url_key = "hq_url" if r.is_hq else "thumbnail_url"
                cell[url_key] = cache.url(r.thumbnail_hash)
                job["cells_generated"] += 1

                # When all cells are generated, compute ridges
                if job["cells_generated"] >= job["total_cells"] and job["phase"] == "generating":
                    if job.get("type") == "fast_scan" and not job.get("_run_analysis"):
                        # Fast scan initial generate-selected: sensitivity already
                        # computed from Jacobian, skip DINOv2 analysis
                        job["phase"] = "complete"
                        job["status"] = "complete"
                        job["render_version"] = job.get("render_version", 0) + 1
                        n_imgs = sum(1 for row in job["cells"] for c in row
                                     if c is not None and c.get("thumbnail_url"))
                        print(f"Job {r.job_id}: fast scan image generation complete "
                              f"({n_imgs} images)", flush=True)
                    else:
                        job["phase"] = "analyzing"
                        asyncio.create_task(asyncio.to_thread(
                            _analyze_and_render, job, r.job_id
                        ))

        # Use shorter poll interval when results are flowing
        interval = 0.05 if results else config.RESULT_POLL_INTERVAL
        await asyncio.sleep(interval)


def _analyze_and_render(job: dict, job_id: str):
    """Compute ridge map + clusters + render all visualizations."""
    from backend.services.ridge_detector import extract_ridge_mesh
    import json

    results_dir = config.RESULTS_DIR / job_id
    results_dir.mkdir(parents=True, exist_ok=True)

    gs = job["grid_size"]
    is_3d = job.get("dimensions", 2) == 3
    embs = job["embeddings"]

    # Ridge sensitivity
    sensitivity = compute_sensitivity(embs)

    # For fast_scan jobs: cells without DINOv2 embeddings get sensitivity=0 from
    # compute_sensitivity. Preserve their Jacobian sensitivity from the scan phase.
    old_sensitivity = job.get("sensitivity")
    if job.get("type") == "fast_scan" and old_sensitivity is not None:
        # Where DINOv2 produced 0 (no embedding), keep old Jacobian value
        mask = sensitivity == 0
        if old_sensitivity.shape == sensitivity.shape:
            sensitivity[mask] = old_sensitivity[mask]
        else:
            # Grid was refined — scale up old sensitivity to new grid
            from scipy.ndimage import zoom as ndizoom
            scale = tuple(n / o for n, o in zip(sensitivity.shape, old_sensitivity.shape))
            old_upsampled = ndizoom(old_sensitivity, scale, order=1)
            sensitivity[mask] = old_upsampled[mask]

    job["sensitivity"] = sensitivity
    np.save(results_dir / "sensitivity.npy", sensitivity)

    if is_3d:
        gs_z = embs.shape[2]
        # Update 3D cell sensitivities
        for i in range(gs):
            for j in range(gs):
                for k in range(gs_z):
                    if job["cells"][i][j][k] is not None:
                        job["cells"][i][j][k]["sensitivity"] = float(sensitivity[i, j, k])

        # DBSCAN clustering
        clusters = compute_clusters(embs)
        job["clusters"] = clusters
        for i in range(gs):
            for j in range(gs):
                for k in range(gs_z):
                    if job["cells"][i][j][k] is not None:
                        job["cells"][i][j][k]["cluster"] = int(clusters[i, j, k])

        # Marching cubes ridge mesh
        mesh = extract_ridge_mesh(sensitivity, tau=1.5)
        if mesh:
            mesh_path = results_dir / "ridge_mesh.json"
            with open(mesh_path, 'w') as f:
                json.dump(mesh, f)
            job["ridge_mesh_path"] = str(mesh_path)

    else:
        gs_b = job.get("grid_size_b", gs)
        # Update 2D cell sensitivities
        for i in range(gs):
            for j in range(gs_b):
                if job["cells"][i][j] is not None:
                    job["cells"][i][j]["sensitivity"] = float(sensitivity[i, j])

        # DBSCAN clustering
        clusters = compute_clusters(embs)
        job["clusters"] = clusters
        for i in range(gs):
            for j in range(gs_b):
                if job["cells"][i][j] is not None:
                    job["cells"][i][j]["cluster"] = int(clusters[i, j])

    # 2D-only rendering (skip for 3D — visualization is done client-side with Plotly)
    if not is_3d:
        heatmap_path = results_dir / "heatmap.png"
        render_heatmap(sensitivity, job["alphas"], job["betas"], heatmap_path,
                       prompt_a=job["prompt_a"], prompt_b=job["prompt_b"], prompt_c=job["prompt_c"])
        job["heatmap_path"] = str(heatmap_path)

        image_grid_path = results_dir / "images.png"
        assemble_image_grid(job["thumbnails"], gs, image_grid_path)
        job["image_grid_path"] = str(image_grid_path)

        overlay_path = results_dir / "overlay.png"
        render_overlay(sensitivity, job["thumbnails"], gs, job["alphas"], job["betas"],
                       config.RIDGE_THRESHOLD_TAU, overlay_path)
        job["overlay_path"] = str(overlay_path)

        cluster_path = results_dir / "clusters.png"
        render_clusters(clusters, job["thumbnails"], gs, job["alphas"], job["betas"], cluster_path)
        job["cluster_path"] = str(cluster_path)

    job["phase"] = "complete"
    job["status"] = "complete"
    job["render_version"] = job.get("render_version", 0) + 1
    n_clusters = len(set(clusters.ravel()) - {-1}) if clusters is not None else 0
    print(f"Job {job_id}: analysis complete "
          f"({'3D' if is_3d else '2D'}, "
          f"m/m={sensitivity.max()/np.median(sensitivity):.1f}, "
          f"{n_clusters} clusters)", flush=True)


def _analyze_fast_scan(job: dict, job_id: str):
    """Compute Jacobian spectral norm from 1-step latents."""
    from backend.services.ridge_detector import compute_jacobian_sensitivity

    gs = job["grid_size"]
    latents = job["latents"]

    spectral, anisotropy = compute_jacobian_sensitivity(latents)
    job["sensitivity"] = spectral
    job["anisotropy"] = anisotropy

    # Update cell sensitivities
    for i in range(gs):
        for j in range(gs):
            if job["cells"][i][j] is not None:
                job["cells"][i][j]["sensitivity"] = float(spectral[i, j])
                job["cells"][i][j]["status"] = "scanned"

    # Render heatmap
    results_dir = config.RESULTS_DIR / job_id
    results_dir.mkdir(parents=True, exist_ok=True)

    heatmap_path = results_dir / "heatmap.png"
    render_heatmap(spectral, job["alphas"], job["betas"], heatmap_path,
                   prompt_a=job["prompt_a"], prompt_b=job["prompt_b"],
                   prompt_c=job.get("prompt_c", ""))
    job["heatmap_path"] = str(heatmap_path)

    # Save sensitivity
    np.save(results_dir / "spectral_norm.npy", spectral)
    np.save(results_dir / "anisotropy.npy", anisotropy)

    job["phase"] = "scan_complete"
    job["status"] = "scan_complete"
    job["render_version"] = job.get("render_version", 0) + 1

    print(f"Fast scan {job_id}: Jacobian spectral norm computed "
          f"(max={spectral.max():.4f}, median={np.median(spectral):.4f}, "
          f"grid={gs}x{gs})", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Ridge Explorer (FLUX Klein)...", flush=True)

    app.state.jobs = {}
    app.state.cache = ThumbnailCache()

    pool = GPUPool(n_gpus=config.N_GPUS)
    pool.start()
    pool.wait_ready()
    app.state.gpu_pool = pool

    collector = asyncio.create_task(result_collector(app))
    print(f"Ridge Explorer ready: {pool.ready_count}/{config.N_GPUS} GPUs", flush=True)
    yield

    collector.cancel()
    pool.shutdown()
    print("Ridge Explorer shut down.", flush=True)


app = FastAPI(title="Ridge Explorer", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

config.THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/cache/thumbnails", StaticFiles(directory=str(config.THUMBNAILS_DIR)), name="thumbnails")
app.mount("/cache/results", StaticFiles(directory=str(config.RESULTS_DIR)), name="results")

app.include_router(health.router)
app.include_router(grid.router)
