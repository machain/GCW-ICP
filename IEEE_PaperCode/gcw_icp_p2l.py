"""Point-to-plane geometric-confidence weighted ICP (GCW-ICP).

This implementation follows the GCW-ICP formulation in the manuscript and is
informed by the Fast-Robust-ICP code structure. It is intentionally written in
NumPy/SciPy so that it is easy to run without compiling Eigen/nanoflann code.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class GCWICPOptions:
    """Hyperparameters used by GCW-ICP point-to-plane registration."""

    k_nn: int = 10
    max_iter: int = 60
    min_iterations: int = 30
    tol: float = 1e-6
    rotation_tol: Optional[float] = None
    translation_tol: Optional[float] = None
    eta: float = 1.4826
    eps_nu: float = 1e-8
    eps_sigma: float = 1e-8
    min_total_weight: float = 1e-4
    enable_weight_fallback: bool = True
    damping: float = 1e-8
    normal_k: int = 20
    rank_tol: float = 1e-8
    enable_rank_check: bool = True
    max_correspondence_distance: Optional[float] = None
    distance_trim_std: Optional[float] = None
    distance_quantile: Optional[float] = None
    initial_scale_factor: float = 3.0
    final_scale_factor: float = 0.0
    normalize_input: bool = True
    multiscale: bool = True
    # Resolution scheduling. ``fallback`` first attempts inexpensive
    # single-scale local refinement and invokes the voxel hierarchy only when
    # its robust final correspondence distance indicates that the local basin
    # is unsatisfactory. ``always`` is retained for controlled ablations.
    multiscale_mode: str = "fallback"
    fallback_max_normalized_median_distance: float = 0.01
    multiscale_max_voxel_ratio: float = 0.05
    multiscale_min_voxel_ratio: float = 0.002
    multiscale_levels: int = 9
    preprocess_target_points: int = 45000


def _exp_se3(xi: np.ndarray) -> np.ndarray:
    """Convert a 6D twist to a 4x4 SE(3) matrix using the small-angle map."""
    w = xi[:3]
    v = xi[3:]
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        skew = np.array(
            [
                [0.0, -w[2], w[1]],
                [w[2], 0.0, -w[0]],
                [-w[1], w[0], 0.0],
            ]
        )
        R = np.eye(3) + skew
    else:
        K = np.array(
            [
                [0.0, -w[2], w[1]],
                [w[2], 0.0, -w[0]],
                [-w[1], w[0], 0.0],
            ]
        )
        R = np.eye(3) + np.sin(theta) / theta * K + (1.0 - np.cos(theta)) / (theta * theta) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = v
    return T


def _rotation_matrix_to_angle_axis(R: np.ndarray) -> float:
    """Return rotation angle in radians for a proper rotation matrix."""
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cos_angle))


def estimate_normals_scipy(points: np.ndarray, k: int = 20) -> np.ndarray:
    """Estimate oriented normals from a point cloud using local PCA."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")
    if len(points) < 3:
        return np.zeros_like(points)
    tree = cKDTree(points)
    kk = min(k, len(points))
    _, idx = tree.query(points, k=kk)
    if kk == 1:
        return np.tile(np.array([0.0, 0.0, 1.0]), (len(points), 1))
    neighbors = points[idx]
    centroid = neighbors.mean(axis=1, keepdims=True)
    centered = neighbors - centroid
    cov = np.einsum("nki,nkj->nij", centered, centered)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normals = eigvecs[..., 0]
    center = points.mean(axis=0)
    orientation = np.sign(np.sum(normals * (center[None, :] - points), axis=1))
    orientation[orientation == 0] = 1.0
    normals *= orientation[:, None]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return normals


def _estimate_normals_open3d(points: np.ndarray, k: int = 20) -> Optional[np.ndarray]:
    """Estimate normals with Open3D when it is installed."""
    try:
        import open3d as o3d
    except Exception:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k),
        fast_normal_computation=False,
    )
    pcd.orient_normals_towards_camera_location(pcd.get_center())
    return np.asarray(pcd.normals, dtype=np.float64)


def read_point_cloud(
    path: str,
    *,
    estimate_missing_normals: bool = True,
    normal_k: int = 20,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Read a local point cloud, with a dependency-free PLY fallback.

    Open3D remains the preferred reader for its broad format support. The
    fallback supports standard ASCII and binary-little-endian PLY vertex tables
    containing ``x``, ``y`` and ``z`` (and optionally ``nx``, ``ny``, ``nz``),
    allowing the registration core and its voxel hierarchy to run in minimal
    NumPy/SciPy environments.
    """
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points, dtype=np.float64)
        if len(points) == 0:
            raise ValueError(f"no points were loaded from {path}")

        normals: Optional[np.ndarray] = None
        if pcd.has_normals():
            normals = np.asarray(pcd.normals, dtype=np.float64)
        elif estimate_missing_normals:
            normals = _estimate_normals_open3d(points, k=normal_k)
    except Exception:
        points, normals = _read_ply_fallback(path)
        if normals is None and estimate_missing_normals:
            normals = estimate_normals_scipy(points, k=normal_k)

    return points, normals


def _read_ply_fallback(path: str) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Read a standard vertex-only PLY file without requiring Open3D."""
    type_map = {
        "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
        "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
        "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
        "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
    }
    file_path = Path(path)
    with file_path.open("rb") as handle:
        header_lines: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"invalid PLY header in {path}")
            decoded = line.decode("ascii").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break
        if not header_lines or header_lines[0] != "ply":
            raise ValueError(f"{path} is not a PLY file")

        format_name = next(
            (line.split()[1] for line in header_lines if line.startswith("format ")),
            None,
        )
        vertex_count = 0
        vertex_properties: list[tuple[str, str]] = []
        in_vertex = False
        for line in header_lines:
            tokens = line.split()
            if len(tokens) >= 3 and tokens[0] == "element":
                in_vertex = tokens[1] == "vertex"
                if in_vertex:
                    vertex_count = int(tokens[2])
            elif in_vertex and len(tokens) == 3 and tokens[0] == "property":
                if tokens[1] == "list":
                    raise ValueError("PLY fallback does not support list-valued vertex properties")
                vertex_properties.append((tokens[2], tokens[1]))

        if vertex_count <= 0 or not vertex_properties:
            raise ValueError(f"PLY file has no readable vertices: {path}")
        names = [name for name, _ in vertex_properties]
        if not {"x", "y", "z"}.issubset(names):
            raise ValueError(f"PLY vertices in {path} do not contain x, y, z")

        if format_name in ("binary_little_endian", "binary_big_endian"):
            endian = "<" if format_name == "binary_little_endian" else ">"
            dtype = np.dtype([(name, endian + type_map[ptype]) for name, ptype in vertex_properties])
            raw = np.fromfile(handle, dtype=dtype, count=vertex_count)
            if len(raw) != vertex_count:
                raise ValueError(f"PLY vertex data are truncated: {path}")
            points = np.column_stack([raw[axis] for axis in ("x", "y", "z")]).astype(np.float64)
            normal_axes = ("nx", "ny", "nz")
            normals = (
                np.column_stack([raw[axis] for axis in normal_axes]).astype(np.float64)
                if set(normal_axes).issubset(names)
                else None
            )
        elif format_name == "ascii":
            data = np.loadtxt(handle, max_rows=vertex_count)
            if data.ndim == 1:
                data = data[None, :]
            columns = {name: index for index, (name, _) in enumerate(vertex_properties)}
            points = np.column_stack([data[:, columns[axis]] for axis in ("x", "y", "z")])
            normal_axes = ("nx", "ny", "nz")
            normals = (
                np.column_stack([data[:, columns[axis]] for axis in normal_axes])
                if set(normal_axes).issubset(names)
                else None
            )
        else:
            raise ValueError(f"unsupported PLY format: {format_name}")
    return points, normals


def _voxel_downsample_numpy(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Centroid voxel downsampling with deterministic lexicographic cells."""
    points = np.asarray(points, dtype=np.float64)
    if voxel_size <= 0.0 or len(points) == 0:
        return points.copy()
    origin = np.min(points, axis=0)
    keys = np.floor((points - origin) / voxel_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.empty((len(counts), 3), dtype=np.float64)
    for axis in range(3):
        sums[:, axis] = np.bincount(inverse, weights=points[:, axis])
    return sums / counts[:, None]


def write_point_cloud(
    path: str,
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
) -> None:
    """Write points and optional normals to a local PLY/PCD file."""
    try:
        import open3d as o3d
    except Exception as exc:
        raise RuntimeError(
            "Open3D is required for write_point_cloud. Install it or save numpy arrays manually."
        ) from exc

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
    o3d.io.write_point_cloud(path, pcd)


class GCWICPPointToPlane:
    """Point-to-plane GCW-ICP with adaptive dual-confidence weighting."""

    def __init__(self, options: Optional[GCWICPOptions] = None) -> None:
        self.options = options or GCWICPOptions()

    @staticmethod
    def _prepare_normals(
        points: np.ndarray,
        normals: Optional[np.ndarray],
        use_open3d: bool,
        k: int,
    ) -> np.ndarray:
        if normals is not None:
            normals = np.asarray(normals, dtype=np.float64)
            if normals.shape != points.shape:
                raise ValueError("normals must have the same shape as points")
            normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            return normals
        if use_open3d:
            normals = _estimate_normals_open3d(points, k=k)
            if normals is not None:
                return normals
        return estimate_normals_scipy(points, k=k)

    @staticmethod
    def _compute_local_median_neighbors(
        tree: cKDTree,
        source: np.ndarray,
        k_nn: int,
    ) -> np.ndarray:
        k = min(k_nn + 1, len(source))
        _, idx = tree.query(source, k=k)
        if idx.ndim == 1:
            idx = idx[:, None]
        if k == k_nn + 1:
            # The first hit is usually the query point itself.
            return idx[:, 1:]
        return idx

    @staticmethod
    def _build_multiscale_voxel_sizes(extent: float, options: GCWICPOptions) -> list[float]:
        """Return descending voxel sizes for coarse-to-fine point-to-plane ICP."""
        if options.multiscale_levels <= 1:
            return []

        max_voxel = max(options.multiscale_max_voxel_ratio * extent, 1e-9)
        min_voxel = max(options.multiscale_min_voxel_ratio * extent, 1e-9)
        if max_voxel <= min_voxel:
            return [max_voxel]

        sizes = np.geomspace(
            max_voxel,
            min_voxel,
            num=max(2, int(options.multiscale_levels)),
        )
        unique = np.unique(np.round(sizes, decimals=3))[::-1]
        return [float(value) for value in unique]

    @staticmethod
    def _find_voxel_size_for_target_count(
        pcd: "open3d.geometry.PointCloud",
        target_count: int,
    ) -> float:
        """Find a voxel size that reduces a cloud to roughly target_count points."""
        points = np.asarray(pcd.points, dtype=np.float64)
        extent = float(np.linalg.norm(np.ptp(points, axis=0)))
        if extent <= 0.0:
            return 1e-6

        low = extent * 1e-6
        high = extent
        best = high
        for _ in range(24):
            mid = (low + high) * 0.5
            count = len(pcd.voxel_down_sample(mid).points)
            if count >= target_count:
                low = mid
            else:
                best = mid
                high = mid
        return best

    @staticmethod
    def _estimate_point_to_plane_bootstrap_open3d(
        source: np.ndarray,
        target: np.ndarray,
        voxel_sizes: list[float],
        normal_k: int,
    ) -> Optional[np.ndarray]:
        """Run a coarse-to-fine unweighted point-to-plane ICP bootstrap.

        This is an internal initialization stage. It uses the same point-to-plane
        objective as the final GCW-ICP refinement and no feature descriptors.
        """
        try:
            import open3d as o3d
        except Exception:
            return None

        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source)
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target)

        source_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamKNN(knn=normal_k)
        )
        source_pcd.orient_normals_towards_camera_location(source_pcd.get_center())
        target_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamKNN(knn=normal_k)
        )
        target_pcd.orient_normals_towards_camera_location(target_pcd.get_center())

        transform = np.eye(4)
        for voxel_size in voxel_sizes:
            source_level = source_pcd.voxel_down_sample(voxel_size)
            target_level = target_pcd.voxel_down_sample(voxel_size)
            if len(source_level.points) < 8 or len(target_level.points) < 8:
                continue
            result = o3d.pipelines.registration.registration_icp(
                source_level,
                target_level,
                1e9,
                transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
            )
            transform = np.asarray(result.transformation, dtype=np.float64)
        return transform

    def _fit_multiscale_gcw(
        self,
        source: np.ndarray,
        target: np.ndarray,
        source_normals: Optional[np.ndarray],
        target_normals: Optional[np.ndarray],
        *,
        use_open3d: bool,
        verbose: bool,
    ) -> dict:
        """Run the same GCW solver from coarse to fine resolution."""
        opt = self.options
        reference_center = np.mean(target, axis=0)
        scale = max(
            float(np.linalg.norm(np.ptp(source, axis=0))),
            float(np.linalg.norm(np.ptp(target, axis=0))),
            1e-12,
        )
        source_normalized = (source - reference_center) / scale
        target_normalized = (target - reference_center) / scale
        extent = max(
            float(np.linalg.norm(np.ptp(source_normalized, axis=0))),
            float(np.linalg.norm(np.ptp(target_normalized, axis=0))),
            1e-9,
        )
        voxel_sizes = self._build_multiscale_voxel_sizes(extent, opt)

        levels: list[tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = []
        for voxel_size in voxel_sizes:
            source_level = _voxel_downsample_numpy(source_normalized, voxel_size)
            target_level = _voxel_downsample_numpy(target_normalized, voxel_size)
            if len(source_level) < 8 or len(target_level) < 8:
                continue
            levels.append(
                (
                    source_level,
                    target_level,
                    None,
                )
            )
        levels.append((source_normalized, target_normalized, target_normals))

        transform = np.eye(4)
        all_history: list[dict] = []
        total_iterations = 0
        level_count = len(levels)
        for level_index, (source_level, target_level, level_normals) in enumerate(levels):
            is_final = level_index == level_count - 1
            level_iterations = opt.max_iter if is_final else min(opt.max_iter, 8)
            level_min_iterations = (
                opt.min_iterations if is_final else min(opt.min_iterations, level_iterations)
            )
            level_options = replace(
                opt,
                max_iter=level_iterations,
                min_iterations=level_min_iterations,
                normalize_input=False,
            )
            level_solver = GCWICPPointToPlane(level_options)
            level_result = level_solver._fit_single_scale(
                source_level,
                target_level,
                source_normals=None,
                target_normals=level_normals,
                init_transform=transform,
                use_open3d=True,
                verbose=(verbose and is_final),
            )
            transform = level_result["transform"]
            for entry in level_result["history"]:
                item = dict(entry)
                item["level"] = level_index
                item["iteration"] = total_iterations + item["iteration"]
                all_history.append(item)
            total_iterations += len(level_result["history"])

        final_result = dict(level_result)
        T_normalized = np.asarray(transform, dtype=np.float64)
        final_transform = np.eye(4)
        final_transform[:3, :3] = T_normalized[:3, :3]
        final_transform[:3, 3] = (
            reference_center
            - reference_center @ T_normalized[:3, :3].T
            + scale * T_normalized[:3, 3]
        )
        final_result["transform"] = final_transform
        final_result["history"] = all_history
        final_result["iterations"] = total_iterations
        final_result["multiscale_levels"] = level_count
        return final_result

    @staticmethod
    def _robust_registration_quality(
        source: np.ndarray,
        target: np.ndarray,
        transform: np.ndarray,
    ) -> dict[str, float]:
        """Return a ground-truth-free final-alignment diagnostic.

        The median nearest-neighbor distance is intentionally used instead of
        the mean/RMS distance. It remains informative when fewer than half of
        the source points are injected outliers, whereas a mean residual is
        dominated by those points. Distances are normalized by the larger
        cloud diagonal so the selector is insensitive to physical units.
        """
        extent = max(
            float(np.linalg.norm(np.ptp(source, axis=0))),
            float(np.linalg.norm(np.ptp(target, axis=0))),
            1e-12,
        )
        transformed = source @ transform[:3, :3].T + transform[:3, 3]
        distances, _ = cKDTree(target).query(transformed, k=1)
        median_distance = float(np.median(distances))
        return {
            "normalized_median_distance": median_distance / extent,
            "normalized_mean_distance": float(np.mean(distances)) / extent,
        }

    @staticmethod
    def _attach_schedule_metadata(
        result: dict,
        schedule: str,
        quality: Optional[dict[str, float]] = None,
    ) -> dict:
        """Attach scheduling diagnostics without changing solver outputs."""
        annotated = dict(result)
        annotated["resolution_schedule"] = schedule
        annotated.setdefault("multiscale_levels", 1)
        if quality is not None:
            annotated.update(quality)
        return annotated

    def fit(
        self,
        source: np.ndarray,
        target: np.ndarray,
        source_normals: Optional[np.ndarray] = None,
        target_normals: Optional[np.ndarray] = None,
        init_transform: Optional[np.ndarray] = None,
        *,
        use_open3d: bool = True,
        verbose: bool = False,
    ) -> dict:
        """Register source to target and return the estimated transformation."""
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if source.ndim != 2 or target.ndim != 2 or source.shape[1] != 3 or target.shape[1] != 3:
            raise ValueError("source and target must be Nx3 point clouds")

        opt = self.options
        mode = opt.multiscale_mode.lower()
        if mode not in {"fallback", "always"}:
            raise ValueError("multiscale_mode must be 'fallback' or 'always'")

        # A supplied transform explicitly requests local refinement. The
        # implementation does not attempt to infer the reliability of that
        # external estimate, so it never silently replaces it with a pyramid.
        if not opt.multiscale or init_transform is not None:
            result = self._fit_single_scale(
                source,
                target,
                source_normals=source_normals,
                target_normals=target_normals,
                init_transform=init_transform,
                use_open3d=use_open3d,
                verbose=verbose,
            )
            schedule = (
                "single_scale_external_initialization"
                if init_transform is not None
                else "single_scale_multiscale_disabled"
            )
            quality = self._robust_registration_quality(source, target, result["transform"])
            return self._attach_schedule_metadata(result, schedule, quality)

        # Small point sets do not benefit from a voxel pyramid and can become
        # unstable when the coarsest level contains only a handful of points.
        if min(len(source), len(target)) < 2000:
            result = self._fit_single_scale(
                source,
                target,
                source_normals=source_normals,
                target_normals=target_normals,
                init_transform=init_transform,
                use_open3d=use_open3d,
                verbose=verbose,
            )
            quality = self._robust_registration_quality(source, target, result["transform"])
            return self._attach_schedule_metadata(
                result, "single_scale_insufficient_points", quality
            )

        if mode == "always":
            result = self._fit_multiscale_gcw(
                source,
                target,
                source_normals,
                target_normals,
                use_open3d=use_open3d,
                verbose=verbose,
            )
            quality = self._robust_registration_quality(source, target, result["transform"])
            return self._attach_schedule_metadata(result, "multiscale_forced", quality)

        # Default: prefer a single full-resolution solve. This avoids the cost
        # and possible correspondence drift of an unconditional pyramid when
        # the single-scale candidate is already satisfactory.
        single_result = self._fit_single_scale(
            source,
            target,
            source_normals=source_normals,
            target_normals=target_normals,
            init_transform=None,
            use_open3d=use_open3d,
            verbose=verbose,
        )
        single_quality = self._robust_registration_quality(
            source, target, single_result["transform"]
        )
        if (
            single_quality["normalized_median_distance"]
            <= opt.fallback_max_normalized_median_distance
        ):
            return self._attach_schedule_metadata(
                single_result, "single_scale_auto_accepted", single_quality
            )

        # The local candidate is outside the prescribed robust-distance basin.
        # Re-run from the identity through the hierarchy, then retain whichever
        # candidate has the smaller robust nearest-neighbor median distance.
        multiscale_result = self._fit_multiscale_gcw(
            source,
            target,
            source_normals,
            target_normals,
            use_open3d=use_open3d,
            verbose=verbose,
        )
        multiscale_quality = self._robust_registration_quality(
            source, target, multiscale_result["transform"]
        )
        if (
            multiscale_quality["normalized_median_distance"]
            < single_quality["normalized_median_distance"]
        ):
            return self._attach_schedule_metadata(
                multiscale_result, "multiscale_fallback_selected", multiscale_quality
            )
        return self._attach_schedule_metadata(
            single_result, "single_scale_fallback_retained", single_quality
        )

    def _fit_single_scale(
        self,
        source: np.ndarray,
        target: np.ndarray,
        source_normals: Optional[np.ndarray] = None,
        target_normals: Optional[np.ndarray] = None,
        init_transform: Optional[np.ndarray] = None,
        *,
        use_open3d: bool = True,
        verbose: bool = False,
    ) -> dict:
        opt = self.options
        rotation_tol = (
            opt.tol if opt.rotation_tol is None else opt.rotation_tol
        )
        translation_tol = (
            opt.tol if opt.translation_tol is None else opt.translation_tol
        )
        if opt.normalize_input:
            source_center = np.mean(source, axis=0)
            target_center = np.mean(target, axis=0)
            source_extent = float(np.linalg.norm(np.ptp(source, axis=0)))
            target_extent = float(np.linalg.norm(np.ptp(target, axis=0)))
            scale = max(source_extent, target_extent, 1e-12)
            source = (source - source_center) / scale
            target = (target - target_center) / scale
        else:
            source_center = np.zeros(3)
            target_center = np.zeros(3)
            scale = 1.0

        target_normals = self._prepare_normals(
            target,
            target_normals,
            use_open3d=use_open3d,
            k=opt.normal_k,
        )

        tree_target = cKDTree(target)
        tree_source = cKDTree(source)
        source_neighbors = self._compute_local_median_neighbors(
            tree_source,
            source,
            opt.k_nn,
        )

        if init_transform is None:
            T = np.eye(4)
        else:
            T_orig = np.asarray(init_transform, dtype=np.float64)
            if T_orig.shape != (4, 4):
                raise ValueError("init_transform must be a 4x4 matrix")
            T = np.eye(4)
            T[:3, :3] = T_orig[:3, :3]
            T[:3, 3] = (
                T_orig[:3, 3] + source_center @ T_orig[:3, :3].T - target_center
            ) / scale

        history: list[dict] = []
        previous_delta_norm = np.inf
        converged = False

        for iteration in range(opt.max_iter):
            transformed_source = source @ T[:3, :3].T + T[:3, 3]
            _, target_idx = tree_target.query(transformed_source, k=1)
            q = target[target_idx]
            nq = target_normals[target_idx]
            correspondence_distance = np.linalg.norm(transformed_source - q, axis=1)

            # e_i in the manuscript. For the point-to-plane variant, e_i is the
            # signed point-to-plane distance; |e_i| feeds the robust weights.
            signed_residual = np.sum(nq * (transformed_source - q), axis=1)
            residual_abs = np.abs(signed_residual)

            # Paper Eq. for nu_k: eta * MAD(|e_i - median(e_i)|).
            residual_median = float(np.median(residual_abs))
            mad = float(np.median(np.abs(residual_abs - residual_median)))
            # The annealing floor is an implementation-level safeguard for
            # 40-50 degree initial misalignment; set both factors to 0.0 to use
            # the exact manuscript MAD formula without this floor.
            progress = iteration / max(1, opt.max_iter - 1)
            scale_annealing = (
                opt.initial_scale_factor * (1.0 - progress)
                + opt.final_scale_factor * progress
            )
            nu = max(
                opt.eta * mad,
                opt.eta * residual_median * scale_annealing,
                opt.eps_nu,
            )

            # bar_e_i: fixed source-neighborhood median of |e_i|.
            local_median = np.median(residual_abs[source_neighbors], axis=1)
            local_deviation = residual_abs - local_median
            local_mad = float(np.median(np.abs(local_deviation)))
            sigma = max(
                opt.eta * local_mad,
                opt.eta * residual_median * scale_annealing,
                opt.eps_sigma,
            )

            # w_i^(1,k) and w_i^(2,k).
            w_magnitude = np.exp(-0.5 * (residual_abs / max(nu, 1e-12)) ** 2)
            w_local = np.exp(-0.5 * (local_deviation / max(sigma, 1e-12)) ** 2)
            weights = w_magnitude * w_local

            # Degeneracy protection: total-weight fallback.
            weight_fallback_triggered = bool(
                np.sum(weights) < opt.min_total_weight * max(1, len(source))
            )
            if opt.enable_weight_fallback and weight_fallback_triggered:
                weights = np.ones_like(weights)

            # Trim clearly distant correspondences when a fixed distance or
            # robust distance criterion is configured. This is important for
            # large initial misalignment, where all-nearest-neighbor matching
            # can bias the update.
            if opt.distance_quantile is not None:
                distance_threshold = float(
                    np.quantile(correspondence_distance, opt.distance_quantile)
                )
            elif opt.distance_trim_std is not None:
                distance_median = float(np.median(correspondence_distance))
                distance_mad = float(
                    np.median(np.abs(correspondence_distance - distance_median))
                )
                distance_threshold = distance_median + opt.distance_trim_std * distance_mad
            elif opt.max_correspondence_distance is not None:
                distance_threshold = float(opt.max_correspondence_distance)
            else:
                distance_threshold = np.inf
            keep = correspondence_distance <= distance_threshold
            if int(np.sum(keep)) < max(6, int(0.1 * len(source))):
                keep = np.ones(len(source), dtype=bool)
            weights = np.where(keep, weights, 0.0)

            # Weighted point-to-plane normal equations, replacing the paper's
            # weighted point-to-point SVD with the point-to-plane linearization.
            J = np.empty((len(source), 6), dtype=np.float64)
            J[:, :3] = np.cross(transformed_source, nq)
            J[:, 3:] = nq
            WJ = weights[:, None] * J
            H = J.T @ WJ
            g = J.T @ (weights * signed_residual)

            # Freeze rotation when the rotational block is not well constrained.
            rot_block = H[:3, :3]
            singular_values = np.linalg.svd(rot_block, compute_uv=False)
            rank_scale = max(float(singular_values.max()), 1e-12)
            rank = int(np.sum(singular_values > opt.rank_tol * rank_scale))

            rank_check_triggered = bool(rank < 2)
            if opt.enable_rank_check and rank_check_triggered:
                # Retain the current rotation and solve only for translation,
                # as described by the degeneracy protection in the manuscript.
                H_translation = H[3:, 3:] + opt.damping * np.eye(3)
                g_translation = g[3:]
                delta = np.zeros(6, dtype=np.float64)
                delta[3:] = -np.linalg.solve(H_translation, g_translation)
            else:
                H_reg = H + opt.damping * np.eye(6)
                delta = -np.linalg.solve(H_reg, g)

            delta_norm = float(np.linalg.norm(delta))
            rotation_change = float(np.linalg.norm(delta[:3]))
            translation_change = float(np.linalg.norm(delta[3:]))
            T_delta = _exp_se3(delta)
            T = T_delta @ T

            history.append(
                {
                    "iteration": iteration,
                    "nu": nu,
                    "sigma": sigma,
                    "residual_rms": float(np.sqrt(np.mean(signed_residual * signed_residual))),
                    "mean_weight": float(np.mean(weights)),
                    "delta_norm": delta_norm,
                    "rank": rank,
                    "rank_check_triggered": rank_check_triggered,
                    "weight_fallback_triggered": weight_fallback_triggered,
                }
            )

            if verbose:
                print(
                    f"iter={iteration:02d} nu={nu:.3e} sigma={sigma:.3e} "
                    f"rms={history[-1]['residual_rms']:.3e} delta={delta_norm:.3e} "
                    f"rank={rank}"
                )

            if (
                iteration > 0
                and iteration + 1 >= opt.min_iterations
                and rotation_change < rotation_tol
                and translation_change < translation_tol
            ):
                converged = True
                break
            if (
                iteration + 1 >= opt.min_iterations
                and abs(delta_norm - previous_delta_norm) < 1e-16
            ):
                break
            previous_delta_norm = delta_norm

        transformed_source = source @ T[:3, :3].T + T[:3, 3]
        _, target_idx = tree_target.query(transformed_source, k=1)
        q = target[target_idx]
        nq = target_normals[target_idx]
        final_residual = np.sum(nq * (transformed_source - q), axis=1)
        final_correspondence_distance = np.linalg.norm(transformed_source - q, axis=1)
        point_to_point_rms = float(
            np.sqrt(np.mean(final_correspondence_distance * final_correspondence_distance))
        )

        T_out = T.copy()
        if opt.normalize_input:
            T_out[:3, 3] = (
                target_center
                - source_center @ T[:3, :3].T
                + scale * T[:3, 3]
            )

        return {
            "transform": T_out,
            "history": history,
            "converged": converged,
            "iterations": len(history),
            "residual_rms": float(np.sqrt(np.mean(final_residual * final_residual))),
            "mean_abs_residual": float(np.mean(np.abs(final_residual))),
            "point_to_point_rms": point_to_point_rms,
        }


def _demo() -> None:
    """Small synthetic demo with known ground truth."""
    rng = np.random.default_rng(7)
    x = rng.uniform(-0.8, 1.1, 2500)
    y = rng.uniform(-1.0, 0.7, 2500)
    z = 0.28 * np.sin(2.2 * x + 0.5) + 0.18 * np.cos(1.7 * y - 0.3) + 0.12 * x * y
    fx = 0.28 * 2.2 * np.cos(2.2 * x + 0.5) + 0.12 * y
    fy = -0.18 * 1.7 * np.sin(1.7 * y - 0.3) + 0.12 * x
    target_normals = np.column_stack([-fx, -fy, np.ones_like(x)])
    target_normals /= np.linalg.norm(target_normals, axis=1, keepdims=True)
    target = np.column_stack([x, y, z])

    T_gt = _exp_se3(np.array([0.22, -0.15, 0.25, 0.06, -0.04, 0.05]))
    R_gt = T_gt[:3, :3]
    t_gt = T_gt[:3, 3]
    source = target @ R_gt.T + t_gt
    source += 0.002 * rng.normal(size=source.shape)

    # The class stores rotation as the transpose of the row-vector active rotation.
    T_true = np.eye(4)
    T_true[:3, :3] = R_gt.T
    T_true[:3, 3] = -R_gt.T @ t_gt

    reg = GCWICPPointToPlane(
        GCWICPOptions(max_iter=40, tol=1e-8, damping=1e-8)
    )
    result = reg.fit(
        source,
        target,
        target_normals=target_normals,
        use_open3d=False,
        verbose=True,
    )
    T_est = result["transform"]
    angle_err = _rotation_matrix_to_angle_axis(T_est[:3, :3] @ T_true[:3, :3].T)
    trans_err = np.linalg.norm(T_est[:3, 3] - T_true[:3, 3])
    print(f"rotation error (deg): {np.rad2deg(angle_err):.6f}")
    print(f"translation error: {trans_err:.6f}")


if __name__ == "__main__":
    _demo()
