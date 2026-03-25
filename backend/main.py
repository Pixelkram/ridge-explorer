import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend import config
from backend.services.gpu_pool import GPUPool, CellResult
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

            if isinstance(r, CellResult):
                # Skip cells that aren't part of this job (sparse refine grids)
                cell = job["cells"][r.row][r.col] if (
                    r.row < len(job["cells"]) and
                    r.col < len(job["cells"][r.row]) and
                    job["cells"][r.row][r.col] is not None
                ) else None
                if cell is None:
                    continue

                # Save thumbnail
                cache.save(r.thumbnail_hash, r.thumbnail_bytes)
                job["thumbnails"][(r.row, r.col)] = r.thumbnail_bytes
                job["thumbnail_hashes"][(r.row, r.col)] = r.thumbnail_hash

                # Store DINOv2 embedding
                job["embeddings"][r.row, r.col] = r.dino_embedding

                # Update cell status
                status = "hq" if r.is_hq else "generated"
                cell["status"] = status
                url_key = "hq_url" if r.is_hq else "thumbnail_url"
                cell[url_key] = cache.url(r.thumbnail_hash)
                job["cells_generated"] += 1

                # When all cells are generated, compute ridges
                if job["cells_generated"] >= job["total_cells"] and job["phase"] == "generating":
                    job["phase"] = "analyzing"
                    asyncio.create_task(asyncio.to_thread(
                        _analyze_and_render, job, r.job_id
                    ))

        await asyncio.sleep(config.RESULT_POLL_INTERVAL)


def _analyze_and_render(job: dict, job_id: str):
    """Compute ridge map + clusters + render all visualizations."""
    results_dir = config.RESULTS_DIR / job_id
    results_dir.mkdir(parents=True, exist_ok=True)

    gs = job["grid_size"]
    gs_b = job.get("grid_size_b", gs)
    embs = job["embeddings"]  # (gs, gs_b, 768)

    # Ridge sensitivity (DINOv2 neighbor distance)
    sensitivity = compute_sensitivity(embs)
    job["sensitivity"] = sensitivity
    np.save(results_dir / "sensitivity.npy", sensitivity)

    # Update cell sensitivities (skip None cells in sparse grids)
    for i in range(gs):
        for j in range(gs_b):
            if job["cells"][i][j] is not None:
                job["cells"][i][j]["sensitivity"] = float(sensitivity[i, j])

    # DBSCAN clustering
    clusters = compute_clusters(embs)
    job["clusters"] = clusters
    np.save(results_dir / "clusters.npy", clusters)
    for i in range(gs):
        for j in range(gs_b):
            if job["cells"][i][j] is not None:
                job["cells"][i][j]["cluster"] = int(clusters[i, j])

    # Render heatmap
    heatmap_path = results_dir / "heatmap.png"
    render_heatmap(sensitivity, job["alphas"], job["betas"], heatmap_path,
                   prompt_a=job["prompt_a"], prompt_b=job["prompt_b"], prompt_c=job["prompt_c"])
    job["heatmap_path"] = str(heatmap_path)

    # Render image grid
    image_grid_path = results_dir / "images.png"
    assemble_image_grid(job["thumbnails"], gs, image_grid_path)
    job["image_grid_path"] = str(image_grid_path)

    # Render overlay
    overlay_path = results_dir / "overlay.png"
    render_overlay(sensitivity, job["thumbnails"], gs, job["alphas"], job["betas"],
                   config.RIDGE_THRESHOLD_TAU, overlay_path)
    job["overlay_path"] = str(overlay_path)

    # Render cluster map
    cluster_path = results_dir / "clusters.png"
    render_clusters(clusters, job["thumbnails"], gs, job["alphas"], job["betas"], cluster_path)
    job["cluster_path"] = str(cluster_path)

    job["phase"] = "complete"
    job["status"] = "complete"
    job["render_version"] = job.get("render_version", 0) + 1
    print(f"Job {job_id}: analysis complete "
          f"(m/m={sensitivity.max()/np.median(sensitivity):.1f}, "
          f"{len(set(clusters.ravel()) - {-1})} clusters)", flush=True)


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
