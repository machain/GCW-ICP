"""Run GCW-ICP point-to-plane registration on two local point clouds.

Example:
    python run_gcw_icp_local.py --target target.ply --source source.ply

The script registers source to target, saves the registered source cloud, and
opens an Open3D window showing target, source-before, and source-after.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gcw_icp_p2l import (
    GCWICPPointToPlane,
    GCWICPOptions,
    read_point_cloud,
    write_point_cloud,
)
from benchmark_gcw_icp import compute_paper_metrics


DEFAULT_TARGET = r"G:\SCI_MWJ\overlap_data\target_overlap_20pct.ply"
DEFAULT_SOURCE = r"G:\SCI_MWJ\overlap_data\source_overlap_20pct.ply"
DEFAULT_GROUND_TRUTH = r"G:\SCI_MWJ\overlap_data\T_gt_overlap_20pct.npy"
DEFAULT_OUTPUT = r"G:\SCI_MWJ\pointCloud\registered_source_20pct.ply"


def _make_pcd(
    points: np.ndarray,
    normals: np.ndarray | None,
    color: tuple[float, float, float],
) -> "open3d.geometry.PointCloud":
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
    pcd.paint_uniform_color(color)
    return pcd


def _maybe_downsample(
    pcd: "open3d.geometry.PointCloud",
    voxel_size: float | None,
) -> "open3d.geometry.PointCloud":
    if voxel_size is not None and voxel_size > 0.0:
        return pcd.voxel_down_sample(voxel_size)
    return pcd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register source to target with point-to-plane GCW-ICP."
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help="target point cloud path; edit DEFAULT_TARGET in this script to use a fixed path",
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="source point cloud path; edit DEFAULT_SOURCE in this script to use a fixed path",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="output path for registered source cloud",
    )
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--min-iterations", type=int, default=30)
    parser.add_argument("--k-nn", type=int, default=10)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--damping", type=float, default=1e-8)
    parser.add_argument(
        "--multiscale-mode",
        choices=("fallback", "always"),
        default="fallback",
        help="fallback: run the voxel hierarchy only after an unsatisfactory single-scale result; always: force the hierarchy",
    )
    parser.add_argument(
        "--fallback-distance",
        type=float,
        default=0.01,
        help="normalized median nearest-neighbor distance that triggers the multiscale fallback",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="optional voxel size for visualization only",
    )
    parser.add_argument(
        "--init-transform",
        default=None,
        help="optional 4x4 numpy .npy file used as initial transformation",
    )
    parser.add_argument(
        "--ground-truth",
        default=DEFAULT_GROUND_TRUTH,
        help="optional 4x4 numpy .npy ground-truth transform for paper-format metrics",
    )
    return parser.parse_args()


def main() -> None:
    import open3d as o3d

    args = parse_args()
    target_path = Path(args.target).expanduser()
    source_path = Path(args.source).expanduser()

    if args.ground_truth is None:
        inferred_gt = target_path.with_name(
            "T_gt_" + target_path.stem.replace("target_", "", 1) + ".npy"
        )
        if inferred_gt.exists():
            args.ground_truth = str(inferred_gt)
            print(f"Using inferred ground truth: {inferred_gt}")

    print(f"Loading target: {target_path}")
    target, target_normals = read_point_cloud(
        str(target_path),
        estimate_missing_normals=True,
        normal_k=20,
    )

    print(f"Loading source: {source_path}")
    source, source_normals = read_point_cloud(
        str(source_path),
        estimate_missing_normals=True,
        normal_k=20,
    )

    reg = GCWICPPointToPlane(
        GCWICPOptions(
            k_nn=args.k_nn,
            max_iter=args.max_iter,
            min_iterations=args.min_iterations,
            tol=args.tol,
            damping=args.damping,
            multiscale_mode=args.multiscale_mode,
            fallback_max_normalized_median_distance=args.fallback_distance,
        )
    )

    init_transform = None
    if args.init_transform is not None:
        init_transform = np.load(args.init_transform, allow_pickle=False)
        print(f"Using initial transform from: {args.init_transform}")
    else:
        print(
            "Starting from identity; GCW-ICP first runs single-scale refinement "
            "and may invoke the coarse-to-fine fallback."
        )

    print("Running GCW-ICP point-to-plane registration...")
    result = reg.fit(
        source,
        target,
        source_normals=source_normals,
        target_normals=target_normals,
        init_transform=init_transform,
        use_open3d=True,
        verbose=True,
    )

    T = result["transform"]
    registered_source = source @ T[:3, :3].T + T[:3, 3]

    output_path = Path(args.output).expanduser()
    write_point_cloud(
        str(output_path),
        registered_source,
        normals=source_normals,
    )
    print(f"Saved registered source to: {output_path}")
    print(f"iterations: {result['iterations']}")
    print(f"converged: {result['converged']}")
    print(f"residual_rms: {result['residual_rms']:.6e}")
    print(f"mean_abs_residual: {result['mean_abs_residual']:.6e}")
    print(f"resolution_schedule: {result['resolution_schedule']}")
    print(
        "normalized_median_distance: "
        f"{result['normalized_median_distance']:.6e}"
    )

    if args.ground_truth is not None:
        ground_truth = np.load(args.ground_truth, allow_pickle=False)
        metrics = compute_paper_metrics(source, target, T, ground_truth)
        print("Paper-format metrics")
        print(f"rotation_error_deg: {metrics['rotation_error_deg']:.6e}")
        print(f"translation_error: {metrics['translation_error']:.6e}")
        print(f"normalized_translation_error: {metrics['normalized_translation_error']:.6e}")
        print(f"normalized_rmse: {metrics['normalized_rmse']:.6e}")

    target_pcd = _make_pcd(target, target_normals, (0.35, 0.35, 0.35))
    source_before_pcd = _make_pcd(source, source_normals, (0.85, 0.15, 0.15))
    source_after_pcd = _make_pcd(registered_source, source_normals, (0.15, 0.75, 0.15))

    cloud_extent = float(np.linalg.norm(np.ptp(target, axis=0)))
    frame_size = max(0.02 * cloud_extent, 1e-6)
    initial_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
    initial_frame.translate(source.mean(axis=0))

    estimated_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
    active_rotation = T[:3, :3].T
    estimated_frame.rotate(active_rotation, center=(0.0, 0.0, 0.0))
    estimated_frame.translate(registered_source.mean(axis=0))

    target_pcd = _maybe_downsample(target_pcd, args.voxel_size)
    source_before_pcd = _maybe_downsample(source_before_pcd, args.voxel_size)
    source_after_pcd = _maybe_downsample(source_after_pcd, args.voxel_size)

    print("Opening before-registration window. Close it to continue.")
    o3d.visualization.draw_geometries(
        [
            target_pcd,
            source_before_pcd,
            initial_frame,
        ],
        window_name=(
            "Before registration: gray=target, red=source"
        ),
    )

    print("Opening after-registration window. Close it to exit.")
    o3d.visualization.draw_geometries(
        [
            target_pcd,
            source_after_pcd,
            estimated_frame,
        ],
        window_name=(
            "After registration: gray=target, green=registered source"
        ),
    )


if __name__ == "__main__":
    main()
