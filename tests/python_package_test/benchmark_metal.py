"""Opt-in benchmark harness for the Apple Metal tree learner."""

from __future__ import annotations

import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python-package"))

import lightgbm as lgb


def _physical_cpu_count() -> int:
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True)
        return max(1, int(out.strip()))
    except Exception:
        return max(1, (os.cpu_count() or 1))


def _make_binary_data(num_rows: int, num_features: int, seed: int) -> lgb.Dataset:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(num_rows, num_features)).astype(np.float32)
    # Keep missing-value handling in the benchmarked path.
    x[: max(1, num_rows // 128), 0] = np.nan
    weights = rng.normal(size=num_features).astype(np.float32)
    margin = x @ weights + 0.1 * rng.normal(size=num_rows).astype(np.float32)
    y = (margin > np.nanmedian(margin)).astype(np.float32)
    return lgb.Dataset(x, label=y, free_raw_data=False)


def _train_once(dataset: lgb.Dataset, params: dict, num_boost_round: int) -> float:
    start = time.perf_counter()
    lgb.train(params, dataset, num_boost_round=num_boost_round)
    return time.perf_counter() - start


def _benchmark_case(case: dict, cpu_threads: int) -> dict:
    dataset = _make_binary_data(case["rows"], case["features"], case["seed"])
    base_params = {
        "objective": "binary",
        "metric": "None",
        "num_leaves": 255,
        "min_data_in_leaf": 20,
        "learning_rate": 0.1,
        "max_bin": case["max_bin"],
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "verbose": -1,
        "seed": case["seed"],
        "deterministic": True,
    }
    cpu_params = dict(base_params, device_type="cpu", num_threads=cpu_threads)
    metal_params = dict(base_params, device_type="metal", num_threads=1)

    _train_once(dataset, cpu_params, case["rounds"])
    _train_once(dataset, metal_params, case["rounds"])

    cpu_runs = [_train_once(dataset, cpu_params, case["rounds"]) for _ in range(5)]
    metal_runs = [_train_once(dataset, metal_params, case["rounds"]) for _ in range(5)]

    cpu_median = statistics.median(cpu_runs)
    metal_median = statistics.median(metal_runs)
    return {
        "name": case["name"],
        "cpu_median": cpu_median,
        "metal_median": metal_median,
        "speedup": cpu_median / metal_median,
        "cpu_runs": cpu_runs,
        "metal_runs": metal_runs,
    }


def main() -> None:
    cpu_threads = _physical_cpu_count()
    cases = [
        {
            "name": "binary 50k x 200 x 10, max_bin=255",
            "rows": 50_000,
            "features": 200,
            "rounds": 10,
            "max_bin": 255,
            "seed": 101,
        },
        {
            "name": "binary 200k x 500 x 5, max_bin=255",
            "rows": 200_000,
            "features": 500,
            "rounds": 5,
            "max_bin": 255,
            "seed": 102,
        },
        {
            "name": "binary 200k x 500 x 5, max_bin=63",
            "rows": 200_000,
            "features": 500,
            "rounds": 5,
            "max_bin": 63,
            "seed": 103,
        },
    ]

    print(f"CPU baseline threads: {cpu_threads}")
    print("Warm-up: 1 run per device, measured runs: 5, reported statistic: median")
    print()

    results = [_benchmark_case(case, cpu_threads) for case in cases]
    geo_speedup = math.exp(
        statistics.mean(math.log(result["speedup"]) for result in results)
    )

    for result in results:
        print(result["name"])
        print(f"  cpu median   : {result['cpu_median']:.6f}s")
        print(f"  metal median : {result['metal_median']:.6f}s")
        print(f"  speedup      : {result['speedup']:.3f}x")
        print()

    print(f"geometric mean speedup: {geo_speedup:.3f}x")
    if geo_speedup >= 1.15 and all(r["speedup"] >= 1.0 for r in results):
        print("acceptance: PASS")
    else:
        print("acceptance: FAIL")


if __name__ == "__main__":
    main()
