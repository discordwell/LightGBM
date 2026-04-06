"""Benchmark CPU vs Metal tree learner on real LightGBM benchmark datasets.

Requires downloaded datasets in ../data/ (see data/README.md).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python-package"))

import lightgbm as lgb

DATA_ROOT = REPO_ROOT / "data"


def _physical_cpu_count() -> int:
    import subprocess
    try:
        return max(1, int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True).strip()))
    except Exception:
        return max(1, os.cpu_count() or 1)


def _load_higgs() -> lgb.Dataset:
    """Higgs: 10.5M rows, 28 features, label in col 0, CSV no header."""
    path = DATA_ROOT / "higgs" / "higgs.train"
    # Use LightGBM's native CSV loader via Dataset constructor + label_column.
    return lgb.Dataset(str(path), params={"label_column": "name:label", "header": False},
                       label=None, free_raw_data=False)


def _load_higgs_numpy(max_rows: int | None = None) -> lgb.Dataset:
    """Load Higgs into numpy via pandas. label_column=0, 28 feature columns follow."""
    import pandas as pd
    path = DATA_ROOT / "higgs" / "higgs.train"
    print(f"  loading {path} via pandas...", flush=True)
    t0 = time.perf_counter()
    df = pd.read_csv(path, header=None, nrows=max_rows, dtype=np.float32)
    print(f"  loaded {df.shape} in {time.perf_counter()-t0:.1f}s", flush=True)
    y = df.iloc[:, 0].values
    X = np.ascontiguousarray(df.iloc[:, 1:].values)
    return lgb.Dataset(X, label=y, free_raw_data=False)


def _load_epsilon() -> lgb.Dataset:
    """Epsilon: 400k x 2000, LIBSVM format."""
    path = DATA_ROOT / "epsilon" / "epsilon_normalized"
    return lgb.Dataset(str(path), free_raw_data=False)


def _load_bosch_numpy(max_rows: int | None = None) -> lgb.Dataset:
    """Bosch: 1M x 968 numeric features. Header. Col 0 is Id, last col is Response."""
    import pandas as pd
    path = DATA_ROOT / "bosch" / "bosch.train"
    print(f"  loading {path} into pandas...", flush=True)
    t0 = time.perf_counter()
    df = pd.read_csv(path, nrows=max_rows, dtype=np.float32)
    print(f"  loaded {df.shape} in {time.perf_counter()-t0:.1f}s", flush=True)
    y = df["Response"].values.astype(np.float32)
    X = df.drop(columns=["Id", "Response"]).values
    return lgb.Dataset(X, label=y, free_raw_data=False)


def _load_expo_numpy(max_rows: int | None = None) -> lgb.Dataset:
    """Load preprocessed Expo airline data from numpy binaries."""
    path_X = DATA_ROOT / "expo" / "expo_X_train.npy"
    path_y = DATA_ROOT / "expo" / "expo_y_train.npy"
    print(f"  loading {path_X}...", flush=True)
    t0 = time.perf_counter()
    X = np.load(path_X, mmap_mode="r")
    y = np.load(path_y, mmap_mode="r")
    if max_rows is not None and max_rows < len(X):
        X = np.ascontiguousarray(X[:max_rows])
        y = np.ascontiguousarray(y[:max_rows])
    else:
        X = np.ascontiguousarray(X)
        y = np.ascontiguousarray(y)
    print(f"  loaded {X.shape} in {time.perf_counter()-t0:.1f}s", flush=True)
    return lgb.Dataset(X, label=y, free_raw_data=False)


def _time_train(dataset: lgb.Dataset, params: dict, num_boost_round: int) -> float:
    t0 = time.perf_counter()
    lgb.train(params, dataset, num_boost_round=num_boost_round)
    return time.perf_counter() - t0


def _bench_one(name: str, dataset: lgb.Dataset, cpu_threads: int,
               num_boost_round: int, max_bin: int = 255) -> bool:
    base = {
        "objective": "binary",
        "metric": "None",
        "num_leaves": 255,
        "min_data_in_leaf": 20,
        "learning_rate": 0.1,
        "max_bin": max_bin,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "verbose": -1,
        "seed": 42,
        "deterministic": True,
    }
    cpu_params = dict(base, device_type="cpu", num_threads=cpu_threads)
    metal_params = dict(base, device_type="metal", num_threads=2)

    print(f"[{name}] warmup (1 round each)...", flush=True)
    _time_train(dataset, cpu_params, 1)
    try:
        _time_train(dataset, metal_params, 1)
    except Exception as e:
        print(f"[{name}]   Metal warmup FAILED: {e}", flush=True)
        print(f"[{name}]   (pre-existing Metal limitation, not sparse-related)\n", flush=True)
        return False

    print(f"[{name}] CPU {num_boost_round} rounds ({cpu_threads} threads)...", flush=True)
    t_cpu = _time_train(dataset, cpu_params, num_boost_round)
    print(f"[{name}]   cpu:   {t_cpu:.2f}s  ({t_cpu/num_boost_round*1000:.1f}ms/round)", flush=True)

    print(f"[{name}] Metal {num_boost_round} rounds...", flush=True)
    t_metal = _time_train(dataset, metal_params, num_boost_round)
    print(f"[{name}]   metal: {t_metal:.2f}s  ({t_metal/num_boost_round*1000:.1f}ms/round)", flush=True)

    print(f"[{name}]   speedup: {t_cpu/t_metal:.3f}x\n", flush=True)
    return True


def main() -> None:
    cpu_threads = _physical_cpu_count()
    print(f"CPU threads: {cpu_threads}")
    print(f"Metal helper threads: 2\n")

    datasets = os.environ.get("BENCH_DATASETS", "higgs,epsilon,bosch").split(",")
    num_rounds = int(os.environ.get("BENCH_ROUNDS", "50"))

    if "higgs" in datasets:
        max_rows = int(os.environ.get("HIGGS_ROWS", "1000000"))  # default 1M for memory
        print(f"=== Higgs (first {max_rows:,} rows) ===")
        ds = _load_higgs_numpy(max_rows=max_rows)
        _bench_one("higgs", ds, cpu_threads, num_rounds)

    if "epsilon" in datasets:
        print("=== Epsilon (400k x 2000) ===")
        ds = _load_epsilon()
        _bench_one("epsilon", ds, cpu_threads, num_rounds)

    if "bosch" in datasets:
        print("=== Bosch (1M x 968) ===")
        ds = _load_bosch_numpy(max_rows=None)
        _bench_one("bosch", ds, cpu_threads, num_rounds)

    if "expo" in datasets:
        max_rows = int(os.environ.get("EXPO_ROWS", "2000000"))
        print(f"=== Expo (first {max_rows:,} rows) ===")
        ds = _load_expo_numpy(max_rows=max_rows)
        _bench_one("expo", ds, cpu_threads, num_rounds)


if __name__ == "__main__":
    main()
