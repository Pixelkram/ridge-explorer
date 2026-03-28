# High-Dimensional Latent Space Exploration: Research Summary

## Context
We explore 2D planes in text embedding space by interpolating between 3 prompts
and generating images on a grid. This document surveys methods for extending to
3D, 4D, and higher-dimensional exploration.

---

## 1. Efficient Sampling in High-Dimensional Spaces

### Sobol Sequences (RECOMMENDED for 3-10 dims)
- **What**: Quasi-random low-discrepancy sequences that fill space more uniformly than random sampling
- **Convergence**: O(1/N) vs O(1/sqrt(N)) for Monte Carlo -- roughly 2x better convergence rate
- **Dimension performance**: Best overall for dimensions > 6. Halton is slightly better for d <= 6 but degrades substantially in higher dimensions
- **Sample counts**: Must be powers of 2 (balance properties break otherwise). For 3D: 512-4096 samples; for 5D: 2048-8192; for 10D: 8192-65536
- **Key constraint**: Extensible up to d=21,201 dimensions
- **Python**: `scipy.stats.qmc.Sobol(d=N).random(n=2**m)` -- ships with SciPy
- **Reference**: Niederreiter (1992); Joe & Kuo direction numbers

### Latin Hypercube Sampling (LHS)
- **What**: Ensures each marginal dimension is stratified with exactly one sample per bin
- **Strength**: Good for small sample counts; can outperform QMC for n < ~100 in low dims
- **Weakness**: Standard LHS doesn't control multi-dimensional uniformity. Strength-2 LHS improves this but is slower to generate
- **Python**: `scipy.stats.qmc.LatinHypercube(d=N, strength=2).random(n=50)`
- **Scaling**: O(n*d) generation, no power-of-2 requirement

### Halton Sequences
- **What**: QMC based on Van der Corput sequences in different prime bases per dimension
- **Performance**: Good for d <= 6, degrades for higher dimensions due to correlation between dimensions
- **Python**: `scipy.stats.qmc.Halton(d=N).random(n=samples)`
- **Verdict**: Use Sobol instead for d > 6

### Hilbert Space-Filling Curves
- **What**: Map 1D line to N-D space preserving locality
- **Use case**: Not ideal for initial sampling, but excellent for ordering/indexing samples for cache locality and spatial queries
- **Python**: `numpy-hilbert-curve` (supports arbitrary dims), `hilbertcurve` (multi-core support)
- **Practical use**: Order generated samples along Hilbert curve for efficient neighbor queries

### Practical Sample Counts for Our Setup
With 6 GPUs at ~0.3s/image (so ~20 images/sec total):
| Dims | Sobol Samples | Time     | Grid Equivalent |
|------|--------------|----------|-----------------|
| 3D   | 1024         | ~51s     | 10^3 grid       |
| 3D   | 4096         | ~3.4min  | 16^3 grid       |
| 4D   | 4096         | ~3.4min  | 8^4 grid        |
| 5D   | 8192         | ~6.8min  | 6^5 grid        |
| 6D   | 16384        | ~13.6min | 5^6 grid        |
| 8D   | 65536        | ~54min   | 4^8 grid        |
| 10D  | 65536        | ~54min   | 3^10 grid       |

---

## 2. Active Learning / Adaptive Sampling for Phase Boundaries

### GP-Based Level Set Estimation (HIGHEST PRIORITY)
- **Core idea**: Fit a Gaussian Process to sampled scalar values (e.g., Hessian ridge magnitude), use uncertainty to decide where to sample next
- **Straddle acquisition function**: `a(x) = 1.96*sigma(x) - |mu(x) - threshold|` -- prioritizes points near the level set boundary with high uncertainty
- **Efficiency**: In 2D phase diagrams, achieves < 5% error with only 8% of grid samples. 10-50x sample reduction typical
- **Key paper**: "Active Learning for Discovering Complex Phase Diagrams with Gaussian Processes" (arXiv:2409.07042, 2024) -- works in 2D and 3D phase spaces
- **Python**: GPyTorch + BoTorch (Meta's Bayesian optimization), or scikit-learn GaussianProcessRegressor

### PhaseXplorer (2025, ACS Nano)
- **What**: Closed-loop active learning platform that autonomously maps phase diagrams
- **Method**: GP regression as surrogate model + acquisition function balancing exploration/exploitation
- **Result**: Maps 4D phase diagrams 100x faster than traditional methods
- **Key insight**: Total iteration time ~7 min for 10 optimally selected samples in multi-dim space
- **Directly applicable**: Their approach of GP + acquisition function is exactly what we need for ridge boundary detection

### BE-CBO (ICML 2024)
- **What**: Boundary Exploration for Constrained Bayesian Optimization
- **Method**: Uses ensemble of neural networks to learn constraints and explores feasible/infeasible boundary
- **Why relevant**: Ridge boundaries are essentially constraint boundaries in embedding space
- **Code**: github.com/yunshengtian/BE-CBO

### `adaptive` Python Library (RECOMMENDED)
- **What**: Production-ready adaptive sampling library with N-dimensional support
- **LearnerND**: Delaunay triangulation-based adaptive sampling. Uses loss per simplex (deviation from linearity + volume). Samples densely where gradient is high
- **Key feature**: Automatically focuses samples on boundaries/ridges without explicit boundary detection
- **Parallel**: Built-in support for parallel function evaluation
- **Install**: `pip install adaptive` or `conda install adaptive`
- **Bounds**: Supports list of (min, max) per dimension or ConvexHull
- **Verdict**: BEST starting point -- drop-in N-dimensional adaptive sampler

### Algorithm for Our Ridge Problem
1. Initial Sobol sampling (256-512 points)
2. Compute ridge magnitude at each point
3. Fit GP to ridge magnitude values
4. Use straddle acquisition to find next batch of points near ridge threshold
5. Generate images, compute ridge values, update GP
6. Repeat until convergence (uncertainty near ridge < epsilon)

Expected: 500-2000 total samples to map ridge structure in 4-6D vs. 10,000+ for grid

---

## 3. Dimensionality Reduction that Preserves Boundaries

### Supervised UMAP (RECOMMENDED)
- **What**: UMAP with label information to guide embedding
- **Boundary preservation**: Cleanly separates known classes while preserving inter-class relationships. If we label points as "near ridge" vs "far from ridge", supervised UMAP will preserve the ridge boundary
- **Semi-supervised**: Can use partial labels (only label confident ridge/non-ridge points)
- **Python**: `umap.UMAP(target_metric='categorical').fit(X, y=labels)`
- **Parametric UMAP**: Neural network version enables fast embedding of new points without refitting

### UMAP (Unsupervised)
- **Strengths**: Preserves both local and global structure better than t-SNE. Fast (minutes for 100k points)
- **Weakness for boundaries**: Can introduce topological distortions -- may create spurious clusters or exaggerate separation. Two forms of map discontinuity documented
- **Parameters that matter**: `n_neighbors` (15-50), `min_dist` (0.0-0.1 for tighter clusters), `metric` (cosine for embeddings)
- **Python**: `umap-learn` package

### t-SNE
- **Strengths**: Excellent local structure preservation
- **Weakness**: Destroys global structure, computationally expensive for large N, non-deterministic
- **Verdict**: Use UMAP instead for our application

### Diffusion Maps
- **What**: Spectral decomposition of diffusion operator on similarity graph
- **Strength**: Highlights continuous transitions (good for ridges that are continuous surfaces)
- **Weakness**: Slower than UMAP, fewer tuning options
- **Python**: `pydiffmap` package

### Topology-Preserving Dimensionality Reduction (arXiv:2201.13012)
- **What**: Minimizes interleaving distance between persistent homology of Vietoris-Rips filtrations in original and embedded space
- **Strength**: Provably preserves topological features (clusters, holes) at identified scales
- **Weakness**: Computationally expensive, demonstrated mainly for linear projections
- **Status**: Research-stage, no production library yet

### Practical Recommendation
1. Run high-dim exploration (4-8D) with adaptive sampling
2. Compute DINOv2 embeddings + ridge magnitudes for all generated images
3. Use supervised UMAP with ridge magnitude as continuous label to project to 2D/3D
4. Overlay ridge contours on the UMAP projection
5. Use parametric UMAP for real-time embedding of new points

---

## 4. 3D Isosurface Extraction

### scikit-image Marching Cubes (RECOMMENDED for offline)
- **What**: Extract 2D mesh from 3D scalar field at given iso-value
- **API**: `skimage.measure.marching_cubes(volume, level=threshold, spacing=(dx,dy,dz))`
- **Input**: 3D NumPy array (M x N x P), float32
- **Output**: vertices, faces, normals, values
- **Performance**: Fast (< 1s for 100^3 grid), pure C implementation
- **Use case**: Extract ridge surface from 3D Hessian field

### PyVista (RECOMMENDED for interactive visualization)
- **What**: Pythonic wrapper around VTK for 3D visualization
- **Marching cubes**: `pyvista.wrap(volume).contour(isosurfaces=[threshold])`
- **Volume rendering**: `plotter.add_volume(grid, opacity='sigmoid')`
- **Integration**: Direct NumPy interface, works with scikit-image output
- **Features**: Interactive rotation, slicing planes, multiple isosurfaces, opacity control
- **Install**: `pip install pyvista`
- **Export**: Can export to HTML (via trame), STL, VTK formats

### Marching Tetrahedra
- **What**: Alternative to marching cubes, avoids ambiguity cases
- **When**: Use if marching cubes produces artifacts at ambiguous cube configurations
- **Python**: Available in VTK, less commonly needed

### Workflow for 3D Ridge Surfaces
```python
import numpy as np
from skimage.measure import marching_cubes
import pyvista as pv

# ridge_field is a 3D numpy array of ridge magnitudes
verts, faces, normals, values = marching_cubes(ridge_field, level=threshold)

# Visualize with PyVista
mesh = pv.PolyData(verts, np.column_stack([np.full(len(faces), 3), faces]))
plotter = pv.Plotter()
plotter.add_mesh(mesh, color='red', opacity=0.7)
plotter.show()
```

---

## 5. Interactive 3D/4D Visualization

### Plotly 3D (RECOMMENDED for web integration)
- **Isosurface**: `go.Isosurface(x=X, y=Y, z=Z, value=V, isomin=lo, isomax=hi, surface_count=5)`
- **Volume**: `go.Volume()` with opacity scale for depth effect
- **3D Scatter**: `go.Scatter3d()` for point clouds colored by ridge magnitude
- **Export**: `fig.write_html("output.html")` for standalone interactive HTML
- **Integration**: Works with Dash for full web apps
- **Limitation**: Slower for > 100k points, no true volume rendering

### PyVista + Trame (for heavy-duty visualization)
- **What**: VTK-powered 3D visualization with web frontend via trame
- **Strength**: Handles millions of points, true volume rendering, advanced lighting
- **Web**: `pyvista.Plotter(notebook=True)` or trame for standalone web app
- **4D**: Animate 4th dimension as time/slider parameter

### Three.js (for custom web UIs)
- **What**: JavaScript 3D library for browser-based rendering
- **Integration**: Use with Python backend serving data via API
- **Strength**: Full GPU acceleration, handles complex scenes
- **Existing work**: "Browsing the Latent Space" (BLS) tool uses web-based 3D for generative model exploration (ACM C&C 2023)
- **Effort**: Requires JavaScript expertise

### DINO Explorer
- **What**: Tool specifically for exploring DINOv2 embeddings
- **Method**: Extracts DINO embeddings, UMAP reduction, interactive visualization via FiftyOne/Voxel51
- **GitHub**: github.com/nityanandmathur/diex

### Plotly Dash + VTK
- **What**: Dash provides `dash_vtk` components for embedding VTK renderers in web apps
- **Strength**: Combines Plotly interactivity with VTK rendering power

### 4D Visualization Strategies
Since we can't directly visualize 4D:
1. **3D slices**: Fix one dimension, show 3D isosurface. Slider to vary the fixed dimension
2. **3D scatter with color**: Use color/size as 4th dimension
3. **Animation**: Animate through 4th dimension as time
4. **Parallel coordinates**: Show all dimensions simultaneously as parallel axes
5. **Multiple linked views**: 2D projections of all dimension pairs, linked brushing

---

## 6. Topological Methods for High-Dimensional Boundaries

### Persistent Homology (RECOMMENDED as feature extractor)
- **What**: Tracks birth/death of topological features (components, loops, voids) across scales
- **For ridges**: Ridge boundaries create 1-cycles (loops) in persistence diagrams. Long-lived features = robust boundaries
- **Scaling**: O(n^3) worst case, but practical implementations much faster with sparse matrices
- **Python libraries**:
  - `ripser` (fastest for Vietoris-Rips, pip install ripser)
  - `gudhi` (comprehensive TDA library, pip install gudhi)
  - `giotto-tda` (scikit-learn compatible pipeline, pip install giotto-tda)
- **Practical use**: Compute persistence on ridge magnitude field to identify significant ridge structures vs noise

### Morse-Smale Complexes
- **What**: Decomposes scalar field into cells based on gradient flow between critical points
- **For ridges**: Ridge lines ARE the 1-skeleton of the Morse-Smale complex (separatrices between descending manifolds)
- **High-dim**: Can be approximated on point clouds via k-nearest neighbor graph
- **Key challenge**: Robust computation in high dimensions is still active research
- **Python**: TTK (Topology ToolKit) has C++ implementation with Python bindings
- **Reference**: Gyulassy et al. (2008) "A Practical Approach to Morse-Smale Complex Computation"

### Contour Trees / Merge Trees
- **What**: Track connected components of level sets as threshold varies
- **For ridges**: Contour tree captures how ridge regions merge/split at different thresholds
- **Strength**: Works in arbitrary dimensions, captures nesting structure
- **Python**: TTK (topology-tool-kit.github.io) -- C++/Python, integrates with VTK/ParaView
- **Algorithm**: Carr et al. "Computing Contour Trees in All Dimensions" -- O(n log n)

### Reeb Graphs
- **What**: Quotient space of level sets -- captures topological skeleton of scalar field
- **Relationship**: Contour tree is Reeb graph for simply-connected domains
- **For ridges**: Provides a 1D skeleton that summarizes ridge connectivity

### TTK (Topology ToolKit) -- Central Hub
- **Implements**: Morse-Smale complexes, merge trees, contour trees, Reeb graphs, persistence diagrams, topological simplification
- **Bindings**: C++, VTK/Python, ParaView plugins, command-line tools
- **License**: BSD
- **Install**: Via conda or build from source with VTK
- **Website**: topology-tool-kit.github.io

### giotto-tda (for ML integration)
- **What**: Scikit-learn compatible TDA pipeline
- **Features**: Persistent homology computation, persistence diagram vectorization, feature extraction for ML
- **Workflow**: Raw data -> persistence diagram -> features -> classifier/regressor
- **Use case**: Extract topological features from ridge fields as input to downstream models
- **Install**: `pip install giotto-tda`

### Practical Topological Workflow
1. Sample points in high-dim space, compute ridge magnitude
2. Build filtered simplicial complex (Vietoris-Rips or alpha complex)
3. Compute persistent homology with ripser/gudhi
4. Identify significant H0 (components) and H1 (loops) features
5. Use persistence-based simplification to remove noise
6. Extract Morse-Smale complex for ridge skeleton

---

## 7. Random Projections and Johnson-Lindenstrauss

### Core Theory
- **JL Lemma**: N points in high-dim can be projected to k = O(log(N)/eps^2) dimensions while preserving all pairwise distances within (1 +/- eps)
- **For our case**: 10,000 samples need only k ~= 664 dims (eps=0.5) or ~11,841 dims (eps=0.1) to preserve distances. Since DINOv2 embeddings are 768-dim, we're already in a reasonable range

### Application to Ridge Detection
- **Key insight**: Random projections preserve pairwise distances, and therefore preserve gradient magnitudes (which are differences of nearby points). This means ridges in the full space will appear as ridges in random projections
- **Strategy**: Take multiple random 2D/3D projections of the high-dim space, detect ridges in each, combine evidence
- **Limitation**: A single projection may miss ridges that are oriented perpendicular to the projection plane. Multiple projections needed for coverage
- **Coverage**: For d dimensions, O(d) random projections suffice to detect ridges with high probability

### Multi-Projection Ridge Detection Algorithm
1. Generate K random projection matrices (K = 10-50 for d=768)
2. Project all points to 2D/3D via each matrix
3. Compute ridge magnitude in each projection
4. Aggregate: point is on ridge if it's on ridge in multiple projections
5. Use consensus to identify robust ridge structure

### Python Implementation
```python
from sklearn.random_projection import GaussianRandomProjection
# or SparseRandomProjection for memory efficiency

transformer = GaussianRandomProjection(n_components=3)
X_projected = transformer.fit_transform(X_high_dim)
# Now do ridge detection in 3D
```

### Feasibility Assessment
- **Pros**: Very fast (linear in n*d), no GP fitting, embarrassingly parallel
- **Cons**: Need multiple projections, may miss fine structure, theoretical but not proven for ridge detection specifically
- **Verdict**: Good for initial screening, then refine with adaptive sampling

---

## 8. Multi-Resolution / Octree Approaches

### Octree Adaptive Refinement (RECOMMENDED for 3D)
- **What**: Recursively subdivide 3D space into 8 octants, only refine cells where ridges detected
- **Efficiency**: Reduces active cells by orders of magnitude (1.32-1.67x speedup documented)
- **Algorithm**:
  1. Start with coarse 4^3 = 64 cells
  2. Sample center + corners of each cell (9 points)
  3. Compute ridge magnitude variance within cell
  4. If variance > threshold, subdivide into 8 children
  5. Repeat to desired depth (typically 4-6 levels = 16^3 to 64^3 effective resolution)
- **Sample savings**: Ridge occupies ~5-10% of volume -> need ~5-10% of full grid samples

### kd-Tree Adaptive Refinement (for 4D+)
- **What**: Binary space partition, splits along one axis at a time
- **Advantage over octree**: Scales to arbitrary dimensions (octree is 3D only)
- **Refinement criterion**: Split cells with high ridge magnitude variance
- **Python**: `scipy.spatial.KDTree` for queries, custom splitting logic for refinement
- **Reference**: Efficient local refinement near boundaries using kd-tree (Algorithms, 2022)

### Quadtree (2D) -> Octree (3D) -> 2^d-tree (dD) Generalization
- **Scaling**: Each cell splits into 2^d children. For d=4: 16 children, d=5: 32 children
- **Problem**: Exponential branching makes this impractical beyond d=4-5
- **Solution**: Use kd-tree (binary splits) instead of 2^d-tree for d > 4

### Python Implementations
- **3D Octree**: `open3d.geometry.Octree`, or roll your own with NumPy
- **kd-tree**: `scipy.spatial.KDTree` (pure Python) or `scipy.spatial.cKDTree` (C, faster)
- **Adaptive meshing**: `adaptive` library's LearnerND uses Delaunay triangulation (similar adaptive subdivision)

### Hierarchical Sampling Strategy for Our Problem
```
3D: Octree
  Level 0: 4^3 = 64 cells, 64 samples
  Level 1: ~20 refined cells * 8 = 160 new cells, +160 samples
  Level 2: ~50 refined * 8 = 400 new cells, +400 samples
  Level 3: ~100 refined * 8 = 800 new cells, +800 samples
  Total: ~1424 samples vs 64^3 = 262,144 for uniform grid (184x reduction)

4D: kd-tree
  Level 0: 4^4 = 256 cells, 256 samples
  Adaptive refinement near ridges...
  Total: ~3000-5000 samples vs 16^4 = 65,536 uniform (13-22x reduction)
```

---

## Combined Strategy: Recommended Pipeline

### Phase 1: Initial Survey (fast)
1. Generate Sobol samples in N-dimensional prompt-weight space
   - 3D: 512 points (~25s), 4D: 1024 points (~51s), 6D: 4096 points (~3.4min)
2. Generate images at each point
3. Compute DINOv2 embeddings + pairwise differences
4. Compute approximate ridge magnitude (gradient of embedding distance)

### Phase 2: Adaptive Refinement (efficient)
5. Fit GP to ridge magnitude field (GPyTorch/BoTorch)
6. Use straddle acquisition function to identify candidate points near ridge threshold
7. OR use `adaptive` library's LearnerND for automatic gradient-based refinement
8. Generate images at selected points, update model
9. Repeat 3-5 iterations until convergence

### Phase 3: Ridge Extraction
10. For 3D: Extract isosurface with marching cubes (scikit-image)
11. For 4D+: Extract level set via Morse-Smale complex (TTK) or persistence analysis (ripser/gudhi)
12. Simplify topology using persistence-based noise removal

### Phase 4: Visualization
13. 3D: PyVista interactive isosurface + Plotly web export
14. 4D+: Supervised UMAP projection to 2D/3D preserving ridge structure
15. Multiple linked views: 2D slices, 3D isosurfaces, parallel coordinates
16. Interactive exploration via Plotly Dash or trame

### Total Sample Budget Estimates
| Dims | Initial Sobol | Adaptive Rounds | Total Samples | Wall Time (6 GPU) |
|------|--------------|-----------------|---------------|-------------------|
| 3D   | 512          | 3 x 128 = 384   | ~900          | ~45s              |
| 4D   | 1024         | 4 x 256 = 1024  | ~2000         | ~100s             |
| 5D   | 2048         | 5 x 512 = 2560  | ~4600         | ~4min             |
| 6D   | 4096         | 5 x 512 = 2560  | ~6600         | ~5.5min           |
| 8D   | 8192         | 6 x 1024 = 6144 | ~14000        | ~12min            |

---

## Key Libraries Summary

| Library | Purpose | Install |
|---------|---------|---------|
| scipy.stats.qmc | Sobol/Halton/LHS sampling | `pip install scipy` |
| adaptive | N-dim adaptive sampling | `pip install adaptive` |
| GPyTorch + BoTorch | GP fitting + Bayesian optimization | `pip install botorch` |
| scikit-image | Marching cubes (3D isosurface) | `pip install scikit-image` |
| PyVista | Interactive 3D visualization | `pip install pyvista` |
| Plotly | Web-based 3D charts | `pip install plotly` |
| umap-learn | Dimensionality reduction | `pip install umap-learn` |
| ripser | Persistent homology (fast) | `pip install ripser` |
| gudhi | Comprehensive TDA | `pip install gudhi` |
| giotto-tda | TDA + ML pipeline | `pip install giotto-tda` |
| TTK | Morse-Smale, contour trees | conda or build from source |
| sklearn | Random projections, kd-tree | `pip install scikit-learn` |
| numpy-hilbert-curve | Space-filling curves | `pip install numpy-hilbert-curve` |

---

## Key References

1. Sobol sampling: Niederreiter (1992), Joe & Kuo (2010) direction numbers
2. GP phase boundaries: arXiv:2409.07042 (2024) - Active learning for complex phase diagrams
3. PhaseXplorer: ACS Nano (2025) - Closed-loop active learning for 4D phase diagrams
4. BE-CBO: ICML 2024 - Boundary exploration for Bayesian optimization
5. Straddle heuristic: Bryan et al. (2005) - Active learning for level set estimation
6. Topology-preserving DR: arXiv:2201.13012 - Interleaving optimization
7. Parametric UMAP: Neural Computation 33(11) 2021
8. BLS tool: ACM C&C 2023 - Browsing the Latent Space
9. TTK: arXiv:1805.09110 - The Topology ToolKit
10. Morse-Smale: Gyulassy et al. (2008) - Practical MS complex computation
11. Marching cubes: Lorensen & Cline (1987), Lewiner et al. (2003)
12. JL lemma: Johnson & Lindenstrauss (1984)
13. Adaptive library: python-adaptive/adaptive on GitHub
