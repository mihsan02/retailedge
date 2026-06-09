"""F2: OOS validation must apply the live ATR regime gate."""
import numpy as np
import pandas as pd
from research.atr_percentile_service import compute_atr_percentile_baseline
from research.worker import compute_bucket_features, run_oos_validation


def _fixture():
    rng = np.random.default_rng(3)
    n = 160
    rets = 0.001 + rng.normal(0, 0.002, n)
    close = 60000 * np.cumprod(1 + rets)
    # engineer a strong momentum signal at n-2 (survives dropna)
    close[-2] = close[-3] * 1.05
    high = close * 1.0005
    low = close * 0.9995
    # volatile high-vol tail -> atr_pct spikes -> percentile ~1.0 -> excluded
    high[-10:] = close[-10:] * 1.02
    low[-10:] = close[-10:] * 0.98
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n, freq="15min"),
        "open": close, "high": high, "low": low, "close": close,
        "volume": np.full(n, 5.0),
    })


def test_oos_is_regime_gated():
    feat = compute_bucket_features(_fixture())
    res = run_oos_validation(feat, atr_lookback=20)
    assert res.get("regime_gated") is True


def test_gate_excludes_high_vol_signal():
    feat = compute_bucket_features(_fixture())
    res = run_oos_validation(feat, atr_lookback=20)
    thr = res["threshold_long"]
    assert thr is not None
    split = int(len(feat) * 0.8)
    oos = feat.iloc[split:]
    ungated = int((oos["momentum_5"] > thr).sum())
    # the volatile tail must be flagged high-vol by the canonical gate
    last = feat.index[-1]
    base = compute_atr_percentile_baseline(feat.iloc[: last + 1], min_lookback=20)
    assert base["is_high_vol"] is True
    # gate can only remove, and here it removes at least the engineered tail signal
    assert res["trade_count"] < ungated
