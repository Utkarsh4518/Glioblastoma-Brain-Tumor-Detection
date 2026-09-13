"""
Isosurface mesh extraction from a predicted label volume.

Decoupled from app/inference.py and the training/model code: everything here
takes an already-computed integer label volume (e.g. the output of
inf.predict()) and turns it into renderable triangle meshes, one per class.
No model, dataset, or checkpoint logic lives here.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from skimage.measure import marching_cubes


@dataclass
class MeshData:
    """A single triangle mesh: vertices/faces/normals, all numpy arrays."""
    vertices: np.ndarray  # (N, 3) float32, voxel-index coordinates (i.e. mm, since 1mm isotropic)
    faces: np.ndarray     # (M, 3) int32, indices into `vertices`
    normals: np.ndarray   # (N, 3) float32, unit vertex normals

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])


def _adjacency_matrix(faces: np.ndarray, n_vertices: int) -> sparse.csr_matrix:
    """Symmetric 0/1 vertex-adjacency matrix built from triangle edges."""
    i = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    j = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    row = np.concatenate([i, j])
    col = np.concatenate([j, i])
    data = np.ones(row.shape[0], dtype=np.float32)
    A = sparse.coo_matrix((data, (row, col)), shape=(n_vertices, n_vertices)).tocsr()
    A.data[:] = 1.0  # collapse any duplicate edges back to a plain 0/1 adjacency
    return A


def laplacian_smooth(
    vertices: np.ndarray, faces: np.ndarray, iterations: int = 5, lam: float = 0.5
) -> np.ndarray:
    """
    Light Laplacian smoothing: each iteration moves every vertex a fraction
    `lam` of the way towards the average position of its neighbours. Raw
    marching-cubes output from voxel data is blocky; a handful of iterations
    rounds it off without a full remeshing library.
    """
    if iterations <= 0 or vertices.shape[0] == 0:
        return vertices
    A = _adjacency_matrix(faces, vertices.shape[0])
    degree = np.asarray(A.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0  # isolated vertex (shouldn't happen on a closed mesh); avoid div-by-0
    inv_degree = (1.0 / degree)[:, None]

    v = vertices.astype(np.float64, copy=True)
    for _ in range(iterations):
        neighbour_mean = A.dot(v) * inv_degree
        v = (1 - lam) * v + lam * neighbour_mean
    return v.astype(np.float32)


def recompute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals from face-normal accumulation (standard
    cross-product / scatter-add scheme), needed after smoothing moves vertices."""
    v = vertices
    f = faces
    tri = v[f]  # (M, 3, 3)
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # (M, 3), magnitude = 2*area

    vertex_normals = np.zeros_like(v, dtype=np.float64)
    for k in range(3):
        np.add.at(vertex_normals, f[:, k], face_normals)

    lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return (vertex_normals / lengths).astype(np.float32)


def extract_mesh(
    mask: np.ndarray,
    level: float = 0.5,
    smoothing_iterations: int = 5,
    smoothing_lambda: float = 0.5,
    step_size: int = 1,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> MeshData | None:
    """
    Extract a smoothed isosurface mesh from a binary (or boolean-like) mask.

    `step_size` is passed straight to skimage's marching_cubes: it strides
    the voxel grid before extraction, so step_size=2 roughly quarters the
    vertex/face count at the cost of some fine detail. Use it to keep large
    regions (e.g. edema) at a browser-friendly size; leave at 1 for small
    regions where detail matters more than size.

    Returns None if the mask has too few voxels to form a surface.
    """
    binary = np.asarray(mask, dtype=bool)
    if binary.sum() < 8:  # marching_cubes needs a non-degenerate volume around the isosurface
        return None
    try:
        verts, faces, normals, _values = marching_cubes(
            binary.astype(np.float32), level=level, spacing=spacing, step_size=step_size
        )
    except (ValueError, RuntimeError):
        return None  # region too small/thin to yield a surface at this step_size

    if smoothing_iterations > 0:
        verts = laplacian_smooth(verts, faces, smoothing_iterations, smoothing_lambda)
        normals = recompute_vertex_normals(verts, faces)

    return MeshData(
        vertices=verts.astype(np.float32),
        faces=faces.astype(np.int32),
        normals=normals.astype(np.float32),
    )


DEFAULT_STEP_SIZE = 2
"""
Verified against all 3 bundled examples at 128^3: step_size=1 can produce a
single region with 150k+ faces and ~13 MB of (JSON-serialized) mesh data --
sluggish in a browser and slow to re-send on every Streamlit rerun.
step_size=2 cuts faces/vertices by roughly 8-10x (worst combined case
~23 MB -> ~2.5 MB) while leaving several thousand vertices per region, which
is still plenty of detail after Laplacian smoothing. Kept as a module
constant (not buried as an unused default argument) so the reasoning above
travels with the value.
"""


def extract_all_regions(
    pred: np.ndarray,
    class_ids: tuple[int, ...] = (1, 2, 3),
    step_sizes: dict[int, int] | None = None,
    smoothing_iterations: int = 5,
    smoothing_lambda: float = 0.5,
) -> dict[int, MeshData | None]:
    """
    Extract one mesh per foreground class from a (D, H, W) integer label
    volume (e.g. the output of app.inference.predict). `step_sizes` optionally
    overrides the marching-cubes step size per class (default DEFAULT_STEP_SIZE
    for all -- see its docstring for why).
    """
    step_sizes = step_sizes or {}
    meshes: dict[int, MeshData | None] = {}
    for cls in class_ids:
        mask = pred == cls
        meshes[cls] = extract_mesh(
            mask,
            step_size=step_sizes.get(cls, DEFAULT_STEP_SIZE),
            smoothing_iterations=smoothing_iterations,
            smoothing_lambda=smoothing_lambda,
        )
    return meshes


def edges_from_faces(faces: np.ndarray) -> np.ndarray:
    """Unique undirected edges (E, 2) of vertex indices from a triangle mesh."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)  # so (a,b) and (b,a) collapse to the same row
    return np.unique(e, axis=0)


def wireframe_lines(md: MeshData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    x/y/z coordinate arrays (with NaN separators between segments) for drawing
    every edge of `md` as a single Plotly Scatter3d line trace -- the standard
    technique for many disconnected line segments in one trace. Purely a
    decorative/stylistic overlay: it does not alter the accurate coloured mesh
    it's drawn on top of.
    """
    edges = edges_from_faces(md.faces)
    v = md.vertices
    n = edges.shape[0]
    x = np.full(n * 3, np.nan, dtype=np.float32)
    y = np.full(n * 3, np.nan, dtype=np.float32)
    z = np.full(n * 3, np.nan, dtype=np.float32)
    x[0::3], x[1::3] = v[edges[:, 0], 0], v[edges[:, 1], 0]
    y[0::3], y[1::3] = v[edges[:, 0], 1], v[edges[:, 1], 1]
    z[0::3], z[1::3] = v[edges[:, 0], 2], v[edges[:, 1], 2]
    return x, y, z
