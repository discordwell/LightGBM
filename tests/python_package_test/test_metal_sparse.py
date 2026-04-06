"""Tests for Metal GPU backend: sparse features, narrow datasets, NaN handling.

Run with: python -m pytest tests/python_package_test/test_metal_sparse.py -v
Requires: Metal-enabled LightGBM build (USE_METAL=1).
"""

import numpy as np
import os
import pytest
import sys
from pathlib import Path

# Ensure we load the LOCAL build, not an installed package.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python-package"))
# Remove any installed lightgbm from path
sys.modules.pop("lightgbm", None)
for p in list(sys.path):
    if "site-packages" in p:
        sys.path.remove(p)
sys.path.insert(0, str(REPO_ROOT / "python-package"))

import lightgbm as lgb

# Verify we're using the local build
assert "LightGBM/python-package" in str(lgb.__file__), \
    f"Expected local build, got {lgb.__file__}"


def _train_and_compare(X, y, num_boost_round=10, num_leaves=31,
                       min_data_in_leaf=20, max_pred_diff=0.01):
    """Train on Metal and CPU, assert predictions match."""
    params = {
        "objective": "binary",
        "verbose": -1,
        "num_leaves": num_leaves,
        "min_data_in_leaf": min_data_in_leaf,
        "seed": 42,
    }
    ds_metal = lgb.Dataset(X, label=y, free_raw_data=False)
    bm = lgb.train(dict(params, device_type="metal", num_threads=2),
                    ds_metal, num_boost_round=num_boost_round)

    ds_cpu = lgb.Dataset(X, label=y, free_raw_data=False)
    bc = lgb.train(dict(params, device_type="cpu"),
                   ds_cpu, num_boost_round=num_boost_round)

    pm = bm.predict(X[:min(200, len(X))])
    pc = bc.predict(X[:min(200, len(X))])
    diff = np.max(np.abs(pm - pc))
    assert bm.num_trees() == bc.num_trees(), \
        f"Tree count mismatch: metal={bm.num_trees()}, cpu={bc.num_trees()}"
    assert diff <= max_pred_diff, \
        f"Prediction diff {diff:.6f} exceeds threshold {max_pred_diff}"
    return diff


class TestMetalDenseOnly:
    """Baseline: pure dense features on GPU."""

    def test_dense_200_features(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(10000, 200)).astype(np.float32)
        y = (X.sum(axis=1) > 0).astype(np.float32)
        _train_and_compare(X, y, max_pred_diff=0.001)


class TestMetalNarrowFallback:
    """Narrow datasets auto-fallback to CPU (< 32 dense groups)."""

    def test_28_features(self):
        """Higgs-like: 28 features → CPU fallback."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50000, 28)).astype(np.float32)
        y = (X.sum(axis=1) > 0).astype(np.float32)
        _train_and_compare(X, y, max_pred_diff=0.001)

    def test_10_features(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(5000, 10)).astype(np.float32)
        y = (X.sum(axis=1) > 0).astype(np.float32)
        _train_and_compare(X, y, max_pred_diff=0.001)


class TestMetalAllSparse:
    """All features are sparse → 0 dense groups → CPU fallback."""

    def test_all_sparse_30_features(self):
        rng = np.random.default_rng(42)
        X = np.zeros((5000, 30), dtype=np.float32)
        for j in range(30):
            idx = rng.random(5000) < 0.05
            X[idx, j] = rng.normal(size=idx.sum()).astype(np.float32)
        y = (X.sum(axis=1) > 0).astype(np.float32)
        diff = _train_and_compare(X, y, num_leaves=7, min_data_in_leaf=50,
                                  max_pred_diff=0.001)
        assert diff == 0.0

    def test_all_sparse_200_features(self):
        rng = np.random.default_rng(42)
        X = np.zeros((10000, 200), dtype=np.float32)
        for j in range(200):
            idx = rng.random(10000) < 0.03
            X[idx, j] = rng.normal(size=idx.sum()).astype(np.float32)
        y = (X.sum(axis=1) > 0).astype(np.float32)
        _train_and_compare(X, y, max_pred_diff=0.001)


class TestMetalMixedSparse:
    """Mixed dense + sparse features → CPU fallback (sparse present)."""

    def test_mixed_180_dense_20_sparse(self):
        rng = np.random.default_rng(42)
        X1 = rng.normal(size=(10000, 180)).astype(np.float32)
        X2 = np.zeros((10000, 20), dtype=np.float32)
        for j in range(20):
            idx = rng.random(10000) < 0.03
            X2[idx, j] = rng.normal(size=idx.sum()).astype(np.float32)
        X = np.hstack([X1, X2])
        y = (X.sum(axis=1) > 0).astype(np.float32)
        _train_and_compare(X, y, max_pred_diff=0.001)

    def test_mixed_180_dense_1_sparse(self):
        rng = np.random.default_rng(42)
        X1 = rng.normal(size=(10000, 180)).astype(np.float32)
        X2 = np.zeros((10000, 1), dtype=np.float32)
        idx = rng.random(10000) < 0.03
        X2[idx, 0] = rng.normal(size=idx.sum()).astype(np.float32)
        X = np.hstack([X1, X2])
        y = (X.sum(axis=1) > 0).astype(np.float32)
        _train_and_compare(X, y, max_pred_diff=0.001)


class TestMetalNaN:
    """NaN handling in Metal GPU histogram and partition."""

    def test_nan_50pct(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(10000, 100)).astype(np.float32)
        mask = rng.random(X.shape) < 0.5
        X[mask] = np.nan
        y = (rng.random(10000) < 0.3).astype(np.float32)
        _train_and_compare(X, y, max_pred_diff=0.001)

    def test_nan_80pct(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(10000, 200)).astype(np.float32)
        mask = rng.random(X.shape) < 0.8
        X[mask] = np.nan
        y = (rng.random(10000) < 0.3).astype(np.float32)
        _train_and_compare(X, y, max_pred_diff=0.001)

    def test_nan_column_wise(self):
        """Some columns fully NaN, some fully present."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(5000, 50)).astype(np.float32)
        # Make 10 columns entirely NaN
        X[:, 40:] = np.nan
        y = (X[:, :40].sum(axis=1) > 0).astype(np.float32)
        _train_and_compare(X, y, max_pred_diff=0.001)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
