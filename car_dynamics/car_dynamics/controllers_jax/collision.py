"""
Universal obstacle representation converter for MPPI collision avoidance.

Accepts any common obstacle format (occupancy grid, bounding boxes, point cloud,
mesh, circles) and converts to a JAX-friendly Signed Distance Field (SDF).

SDF is the canonical internal format:
    positive = free space (distance to nearest obstacle in meters)
    negative = inside obstacle

Usage:
    # From bounding boxes (e.g., RoboCasa fixtures)
    obs_map = ObstacleMap.from_boxes(boxes, bounds=(floor_min, floor_max))

    # From existing occupancy grid (e.g., home-robot voxel map)
    obs_map = ObstacleMap.from_occupancy_grid(grid, origin, resolution)

    # From 3D mesh projected to 2D (e.g., MuJoCo scene)
    obs_map = ObstacleMap.from_mesh_2d(vertices, faces, bounds)

    # From point cloud (e.g., lidar scan)
    obs_map = ObstacleMap.from_point_cloud(points_xy, bounds)

    # From circles (e.g., simple obstacle list)
    obs_map = ObstacleMap.from_circles(centers, radii, bounds)
"""

import numpy as np
import jax
import jax.numpy as jnp
from scipy.ndimage import distance_transform_edt, binary_dilation


class ObstacleMap:
    """
    Immutable 2D obstacle representation backed by a signed distance field.
    Construct via from_* classmethods. Use query_sdf_batch for MPPI cost evaluation.
    """

    def __init__(self, sdf: jnp.ndarray, origin: jnp.ndarray, resolution: float):
        """
        Args:
            sdf: (W, H) signed distance in meters. Positive = free, negative = inside obstacle.
            origin: (2,) world XY coordinates of grid cell [0, 0].
            resolution: meters per grid cell.
        """
        self.sdf = sdf
        self.origin = origin
        self.resolution = resolution
        self.shape = sdf.shape

    # ------------------------------------------------------------------ #
    #  Constructors                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_occupancy_grid(cls, grid, origin, resolution, inflate_radius=0.0):
        """
        Create from a binary occupancy grid.

        Args:
            grid: (W, H) array — 1 = obstacle, 0 = free.
            origin: (2,) world coordinates of grid[0, 0].
            resolution: meters per cell.
            inflate_radius: inflate obstacles by this radius (meters) before SDF.
        """
        grid = np.asarray(grid, dtype=np.float32)
        origin = np.asarray(origin, dtype=np.float32)

        if inflate_radius > 0:
            r_cells = int(np.ceil(inflate_radius / resolution))
            struct = np.ones((2 * r_cells + 1, 2 * r_cells + 1))
            grid = binary_dilation(grid, structure=struct).astype(np.float32)

        sdf = _occupancy_to_sdf(grid, resolution)
        return cls(jnp.array(sdf), jnp.array(origin[:2]), float(resolution))

    @classmethod
    def from_boxes(cls, boxes, bounds, resolution=0.02, inflate_radius=0.15):
        """
        Create from axis-aligned bounding boxes.

        Args:
            boxes: list of 2-tuples (min_xy, max_xy), each as array-like of shape (2,).
                   Or list of 3-tuples (center_xy, half_extents_xy, yaw) — yaw is ignored
                   (projected to AABB).
            bounds: (min_xy, max_xy) of the full scene floor.
            resolution: meters per cell.
            inflate_radius: inflate obstacles by this radius (meters).
        """
        min_xy = np.asarray(bounds[0], dtype=np.float32)[:2]
        max_xy = np.asarray(bounds[1], dtype=np.float32)[:2]
        nx = int(np.ceil((max_xy[0] - min_xy[0]) / resolution))
        ny = int(np.ceil((max_xy[1] - min_xy[1]) / resolution))
        grid = np.zeros((nx, ny), dtype=np.float32)

        for box in boxes:
            if len(box) == 2:
                bmin = np.asarray(box[0], dtype=np.float32)[:2]
                bmax = np.asarray(box[1], dtype=np.float32)[:2]
            elif len(box) == 3:
                center = np.asarray(box[0], dtype=np.float32)[:2]
                half_ext = np.asarray(box[1], dtype=np.float32)[:2]
                bmin = center - half_ext
                bmax = center + half_ext
            else:
                raise ValueError(f"Box must be (min, max) or (center, half_ext, yaw), got len={len(box)}")

            i0 = max(0, int(np.floor((bmin[0] - min_xy[0]) / resolution)))
            i1 = min(nx, int(np.ceil((bmax[0] - min_xy[0]) / resolution)))
            j0 = max(0, int(np.floor((bmin[1] - min_xy[1]) / resolution)))
            j1 = min(ny, int(np.ceil((bmax[1] - min_xy[1]) / resolution)))
            grid[i0:i1, j0:j1] = 1.0

        return cls.from_occupancy_grid(grid, min_xy, resolution, inflate_radius)

    @classmethod
    def from_point_cloud(cls, points_xy, bounds, resolution=0.02, inflate_radius=0.15):
        """
        Create from 2D obstacle points.

        Args:
            points_xy: (N, 2) obstacle point positions.
            bounds: (min_xy, max_xy) of the scene.
            resolution: meters per cell.
            inflate_radius: inflate obstacles by this radius (meters).
        """
        points_xy = np.asarray(points_xy, dtype=np.float32)
        min_xy = np.asarray(bounds[0], dtype=np.float32)[:2]
        max_xy = np.asarray(bounds[1], dtype=np.float32)[:2]
        nx = int(np.ceil((max_xy[0] - min_xy[0]) / resolution))
        ny = int(np.ceil((max_xy[1] - min_xy[1]) / resolution))
        grid = np.zeros((nx, ny), dtype=np.float32)

        indices = ((points_xy[:, :2] - min_xy) / resolution).astype(int)
        valid = (
            (indices[:, 0] >= 0) & (indices[:, 0] < nx)
            & (indices[:, 1] >= 0) & (indices[:, 1] < ny)
        )
        grid[indices[valid, 0], indices[valid, 1]] = 1.0
        return cls.from_occupancy_grid(grid, min_xy, resolution, inflate_radius)

    @classmethod
    def from_mesh_2d(cls, vertices, faces, bounds, resolution=0.02,
                     height_range=(0.0, 1.5), inflate_radius=0.15):
        """
        Create from a 3D mesh by projecting to 2D. Only triangles within
        height_range are considered obstacles.

        Args:
            vertices: (V, 3) mesh vertex positions.
            faces: (F, 3) triangle face indices.
            bounds: (min_xy, max_xy) of the scene floor.
            resolution: meters per cell.
            height_range: (z_min, z_max) — only geometry in this range is rasterized.
            inflate_radius: inflate obstacles by this radius (meters).
        """
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)
        min_xy = np.asarray(bounds[0], dtype=np.float32)[:2]
        max_xy = np.asarray(bounds[1], dtype=np.float32)[:2]
        nx = int(np.ceil((max_xy[0] - min_xy[0]) / resolution))
        ny = int(np.ceil((max_xy[1] - min_xy[1]) / resolution))
        grid = np.zeros((nx, ny), dtype=np.float32)

        z_lo, z_hi = height_range
        for face in faces:
            tri_verts = vertices[face]
            if tri_verts[:, 2].max() < z_lo or tri_verts[:, 2].min() > z_hi:
                continue
            tmin = tri_verts[:, :2].min(axis=0)
            tmax = tri_verts[:, :2].max(axis=0)
            i0 = max(0, int(np.floor((tmin[0] - min_xy[0]) / resolution)))
            i1 = min(nx, int(np.ceil((tmax[0] - min_xy[0]) / resolution)))
            j0 = max(0, int(np.floor((tmin[1] - min_xy[1]) / resolution)))
            j1 = min(ny, int(np.ceil((tmax[1] - min_xy[1]) / resolution)))
            grid[i0:i1, j0:j1] = 1.0

        return cls.from_occupancy_grid(grid, min_xy, resolution, inflate_radius)

    @classmethod
    def from_circles(cls, centers, radii, bounds, resolution=0.02):
        """
        Create from circular obstacles with analytic SDF (no rasterization).

        Args:
            centers: (N, 2) obstacle center positions.
            radii: (N,) obstacle radii.
            bounds: (min_xy, max_xy) of the scene.
            resolution: meters per cell.
        """
        centers = np.asarray(centers, dtype=np.float32)
        radii = np.asarray(radii, dtype=np.float32)
        min_xy = np.asarray(bounds[0], dtype=np.float32)[:2]
        max_xy = np.asarray(bounds[1], dtype=np.float32)[:2]
        nx = int(np.ceil((max_xy[0] - min_xy[0]) / resolution))
        ny = int(np.ceil((max_xy[1] - min_xy[1]) / resolution))

        xs = np.linspace(min_xy[0] + resolution / 2, max_xy[0] - resolution / 2, nx)
        ys = np.linspace(min_xy[1] + resolution / 2, max_xy[1] - resolution / 2, ny)
        xx, yy = np.meshgrid(xs, ys, indexing='ij')
        coords = np.stack([xx, yy], axis=-1)  # (nx, ny, 2)

        sdf = np.full((nx, ny), np.inf, dtype=np.float32)
        for c, r in zip(centers, radii):
            dist = np.linalg.norm(coords - c.reshape(1, 1, 2), axis=-1) - r
            sdf = np.minimum(sdf, dist)

        return cls(jnp.array(sdf), jnp.array(min_xy), float(resolution))

    # ------------------------------------------------------------------ #
    #  SDF query (JAX-compatible, used inside MPPI cost)                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    @jax.jit
    def query_sdf_batch(positions_xy, sdf_grid, origin, resolution):
        """
        Batch SDF query for MPPI rollout positions.

        Args:
            positions_xy: (N, 2) world-frame positions to query.
            sdf_grid: (W, H) the SDF grid array.
            origin: (2,) world coords of grid[0, 0].
            resolution: meters per cell (scalar).

        Returns:
            (N,) signed distances in meters. Positive = free, negative = inside obstacle.
        """
        idx = ((positions_xy - origin) / resolution).astype(jnp.int32)
        ix = jnp.clip(idx[:, 0], 0, sdf_grid.shape[0] - 1)
        iy = jnp.clip(idx[:, 1], 0, sdf_grid.shape[1] - 1)
        return sdf_grid[ix, iy]

    # ------------------------------------------------------------------ #
    #  Visualization helper                                               #
    # ------------------------------------------------------------------ #

    def to_numpy(self):
        """Return (sdf, origin, resolution) as numpy arrays for plotting."""
        return np.asarray(self.sdf), np.asarray(self.origin), self.resolution


# ------------------------------------------------------------------ #
#  Internal helpers                                                   #
# ------------------------------------------------------------------ #

def _occupancy_to_sdf(grid, resolution):
    """Convert binary occupancy grid to signed distance field in meters."""
    free_dist = distance_transform_edt(1.0 - grid) * resolution
    obs_dist = distance_transform_edt(grid) * resolution
    return (free_dist - obs_dist).astype(np.float32)
