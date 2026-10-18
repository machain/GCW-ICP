"""Benchmark GCW-ICP point-to-plane registration against paper metrics.

This script links the local registration implementation to the error metrics
used in the manuscript:
  - rotation error in degrees
  - normalized translation error
  - normalized RMSE

It does not fabricate numbers; it reads local point clouds, runs GCW-ICP, and
computes these quantities from the estimated and ground-truth poses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from gcw_icp_p2l import (
    GCWICPPointToPlane,
    GCWICPOptions,
    _rotation_matrix_to_angle_axis,
    read_point_cloud,
)


DEFAULT_TARGET = r"G:\ICP_Paper_my\source_down_R_T.ply"
DEFAULT_SOURCE = r"G:\ICP_Paper_my\source_down.ply"
DEFAULT_GROUND_TRUTH = r"G:\ICP_Paper_my\ground_truth_transform.npy"


def normalize_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("transform must be a 4x4 matrix")
    return T


def compute_paper_metrics(
    source: np.ndarray,
    target: np.ndarray,
    T_est: np.ndarray,
    T_gt: np.ndarray,
) -> dict[str, float]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    T_est = normalize_transform(T_est)
    T_gt = normalize_transform(T_gt)

    R_est = T_est[:3, :3]
    t_est = T_est[:3, 3]
    R_gt = T_gt[:3, :3]
    t_gt = T_gt[:3, 3]

    # GCWICPPointToPlane stores rotation in row-vector convention:
    # transformed_points = points @ R_est.T + t_est. Therefore the active
    # rotation matrix is R_est.T.
    active_rotation_est = R_est.T
    active_rotation_gt = R_gt.T
    rotation_error = _rotation_matrix_to_angle_axis(
        active_rotation_est @ active_rotation_gt.T
    )

    source_gt = source @ R_gt.T + t_gt
    source_est = source @ R_est.T + t_est
    diagonal = float(np.linalg.norm(np.ptp(target, axis=0)))
    if diagonal <= 0.0:
        diagonal = 1.0

    translation_error = float(np.linalg.norm(t_est - t_gt))
    normalized_translation = translation_error / diagonal

    rmse_abs = float(np.sqrt(np.mean(np.sum((source_gt - source_est) ** 2, axis=1))))
    normalized_rmse = rmse_abs / diagonal

    # Same coarse success criterion as the manuscript: successful when rotation
    # is below 1 degree and normalized RMSE is below 1e-2.
    successful = bool(np.rad2deg(rotation_error) < 1.0 and normalized_rmse < 1e-2)

    return {
        "rotation_error_deg": float(np.rad2deg(rotation_error)),
        "translation_error": translation_error,
        "normalized_translation_error": normalized_translation,
        "rmse": rmse_abs,
        "normalized_rmse": normalized_rmse,
        "successful": successful,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GCW-ICP and compute manuscript-format registration metrics."
    )
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--ground-truth", default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--output-json", default="gcw_icp_benchmark.json")
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--k-nn", type=int, default=10)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--damping", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    target, target_normals = read_point_cloud(
        args.target,
        estimate_missing_normals=True,
        normal_k=20,
    )
    source, source_normals = read_point_cloud(
        args.source,
        estimate_missing_normals=True,
        normal_k=20,
    )
    T_gt = normalize_transform(np.load(args.ground_truth, allow_pickle=False))

    reg = GCWICPPointToPlane(
        GCWICPOptions(
            k_nn=args.k_nn,
            max_iter=args.max_iter,
            tol=args.tol,
            damping=args.damping,
        )
    )

    print("Running GCW-ICP point-to-plane registration...")
    result = reg.fit(
        source,
        target,
        source_normals=source_normals,
        target_normals=target_normals,
        use_open3d=False,
        verbose=True,
    )

    metrics = compute_paper_metrics(source, target, result["transform"], T_gt)
    metrics["iterations"] = result["iterations"]
    metrics["converged"] = result["converged"]
    metrics["point_to_plane_residual_rms"] = result["residual_rms"]

    print("\nBenchmark metrics")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    output_path = Path(args.output_json).expanduser()
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved metrics to: {output_path}")


if __name__ == "__main__":
    main()
