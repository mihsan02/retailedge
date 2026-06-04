"""
research/worker.py — Sprint 3 S3-2
Research Worker: synthetic OHLCV + bucket baseline (threshold scan) + 80/20 OOS validation.

IMPORTANT CONSTRAINTS:
- Synthetic data only. Real exchange data is wired in Sprint 4+.
- Simple 80/20 split. CPCV/PBO/DSR deferred to Sprint 4.
- All candidates stored with stage=DRY_RUN_PLACEHOLDER.
  They CANNOT be promoted to live without full CPCV/DSR evidence (Stage A gate).
- cost_floor_pct = 0.0046 (binance_spot_usdt from venue_costs.json).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COST_FLOOR_PCT = 0.0046  # from venue_costs.json binance_spot_usdt
ATR_PERIOD = 14
MOMENTUM_PERIOD = 5
TRAIN_RATIO = 0.8
MIN_OOS_TRADES = 10  # Sprint 3 minimum; Stage A gate requires 50

# Threshold scan grid
MOMENTUM_THRESHOLDS = [-0.005, -0.002, 0.0, 0.002, 0.005, 0.010, 0.015]
ATR_PCT_THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Strategy memory table schema (v1.9 blueprint Section 7)
STRATEGY_MEMORY_INSERT = """
INSERT INTO strategy_memory (
    candidate_id,
    strategy_id,
    model_type,
    model_id,
    stage,
    oos_trade_count,
    oos_win_rate,
    oos_avg_pnl_pct,
    cost_floor_pct,
    cost_floor_cleared,
    best_momentum_threshold,
    best_atr_pct_threshold,
    train_ratio,
    n_candles_total,
    n_candles_oos,
    created_ts,
    evidence_json
) VALUES (
    :candidate_id,
    :strategy_id,
    :model_type,
    :model_id,
    :stage,
    :oos_trade_count,
    :oos_win_rate,
    :oos_avg_pnl_pct,
    :cost_floor_pct,
    :cost_floor_cleared,
    :best_momentum_threshold,
    :best_atr_pct_threshold,
    :train_ratio,
    :n_candles_total,
    :n_candles_oos,
    :created_ts,
    :evidence_json
)
"""

# ---------------------------------------------------------------------------
# 1. Synthetic OHLCV generation
# ---------------------------------------------------------------------------


def generate_synthetic_ohlcv(n_candles: int = 6000, seed: int = 42) -> pd.DataFrame:
    """
    Geometric Brownian Motion with regime switches.
    Produces OHLCV + volume. ATR and returns computed from OHLCV.

    WARNING: DGP is synthetic. OOS metrics on this data have no predictive
    value for real markets. This function exists purely for infrastructure
    validation in Sprint 3.
    """
    rng = np.random.default_rng(seed)

    # GBM parameters
    mu = 0.00005       # drift per candle
    sigma = 0.002      # vol per candle
    regime_switch_prob = 0.002  # probability of regime change per candle

    prices = np.zeros(n_candles)
    prices[0] = 30000.0  # BTC-like start price
    current_sigma = sigma

    for i in range(1, n_candles):
        if rng.random() < regime_switch_prob:
            # Switch between low/high vol regime
            current_sigma = sigma * rng.choice([0.5, 1.0, 2.0])
        shock = rng.normal(mu, current_sigma)
        prices[i] = prices[i - 1] * math.exp(shock)

    # Synthesize OHLC from close prices
    noise = rng.uniform(0.0005, 0.003, size=n_candles)
    high = prices * (1 + noise)
    low = prices * (1 - noise)
    open_ = np.roll(prices, 1)
    open_[0] = prices[0]
    volume = rng.uniform(100, 2000, size=n_candles)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": prices,
            "volume": volume,
        }
    )

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Feature computation
# ---------------------------------------------------------------------------


def _compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """True Range ATR."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def compute_bucket_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two features used by bucket baseline:
      - momentum: n-candle return (continuous, not bucketed — threshold scan does the bucketing)
      - atr_pct: ATR / close (normalized volatility)

    Returns a copy with columns added. Drops NaN rows from rolling windows.
    """
    df = df.copy()

    # Momentum: n-candle log return
    df["momentum"] = np.log(df["close"] / df["close"].shift(MOMENTUM_PERIOD))

    # ATR normalized by close
    atr = _compute_atr(df, period=ATR_PERIOD)
    df["atr_pct"] = atr / df["close"]

    # Forward return: 1-candle (entry at open+1, exit at close+1 approximation)
    df["fwd_return"] = df["close"].shift(-1) / df["close"] - 1

    df = df.dropna().reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 3. OOS validation — threshold scan
# ---------------------------------------------------------------------------


def _simulate_trades(
    df: pd.DataFrame,
    momentum_thresh: float,
    atr_pct_thresh: float,
    cost_floor_pct: float,
) -> dict[str, Any]:
    """
    Simple long-only bucket strategy:
      Enter if momentum > momentum_thresh AND atr_pct < atr_pct_thresh.
      Exit next candle close (1-candle hold, no compounding).
      Net PnL = fwd_return - cost_floor_pct (one-way cost applied once per trade).

    Returns dict with trade_count, win_rate, avg_pnl_pct.
    """
    signals = (df["momentum"] > momentum_thresh) & (df["atr_pct"] < atr_pct_thresh)
    trade_df = df[signals].copy()

    if len(trade_df) == 0:
        return {"trade_count": 0, "win_rate": 0.0, "avg_pnl_pct": float("nan")}

    net_returns = trade_df["fwd_return"] - cost_floor_pct
    wins = (net_returns > 0).sum()

    return {
        "trade_count": int(len(trade_df)),
        "win_rate": float(wins / len(trade_df)),
        "avg_pnl_pct": float(net_returns.mean()),
    }


def run_oos_validation(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    cost_floor_pct: float = COST_FLOOR_PCT,
) -> dict[str, Any]:
    """
    1. Split df into train/OOS folds.
    2. Grid search (momentum_thresh, atr_pct_thresh) on TRAIN fold only.
       Best = highest avg_pnl_pct with >= MIN_OOS_TRADES.
    3. Apply best threshold to OOS fold. Report OOS metrics only.

    Returns candidate dict ready for save_candidate().
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    oos_df = df.iloc[split_idx:].reset_index(drop=True)

    logger.info(
        "OOS split: %d train candles / %d OOS candles",
        len(train_df),
        len(oos_df),
    )

    # --- Grid search on train fold ---
    best_score = float("-inf")
    best_params: dict[str, float] = {}

    for m_thresh in MOMENTUM_THRESHOLDS:
        for a_thresh in ATR_PCT_THRESHOLDS:
            result = _simulate_trades(train_df, m_thresh, a_thresh, cost_floor_pct)
            if result["trade_count"] < MIN_OOS_TRADES:
                continue
            if math.isnan(result["avg_pnl_pct"]):
                continue
            # Objective: avg_pnl_pct (not Sharpe — Sprint 4 adds DSR/PBO)
            if result["avg_pnl_pct"] > best_score:
                best_score = result["avg_pnl_pct"]
                best_params = {
                    "momentum_threshold": m_thresh,
                    "atr_pct_threshold": a_thresh,
                }

    if not best_params:
        logger.warning("No valid threshold found on train fold. Returning null candidate.")
        return {
            "stage": "DRY_RUN_PLACEHOLDER",
            "status": "NO_VALID_THRESHOLD",
            "oos_trade_count": 0,
            "oos_win_rate": 0.0,
            "oos_avg_pnl_pct": float("nan"),
            "cost_floor_cleared": False,
            "best_momentum_threshold": None,
            "best_atr_pct_threshold": None,
        }

    # --- Apply best params to OOS fold ---
    oos_result = _simulate_trades(
        oos_df,
        best_params["momentum_threshold"],
        best_params["atr_pct_threshold"],
        cost_floor_pct,
    )

    cost_floor_cleared = (
        not math.isnan(oos_result["avg_pnl_pct"])
        and oos_result["avg_pnl_pct"] > 0
        and oos_result["trade_count"] >= MIN_OOS_TRADES
    )

    return {
        "stage": "DRY_RUN_PLACEHOLDER",
        "status": "OK",
        "oos_trade_count": oos_result["trade_count"],
        "oos_win_rate": oos_result["win_rate"],
        "oos_avg_pnl_pct": oos_result["avg_pnl_pct"],
        "cost_floor_pct": cost_floor_pct,
        "cost_floor_cleared": cost_floor_cleared,
        "best_momentum_threshold": best_params["momentum_threshold"],
        "best_atr_pct_threshold": best_params["atr_pct_threshold"],
        "n_candles_total": len(df),
        "n_candles_oos": len(oos_df),
        "train_ratio": train_ratio,
    }


# ---------------------------------------------------------------------------
# 4. Ledger persistence
# ---------------------------------------------------------------------------


def _ensure_strategy_memory_table(conn: sqlite3.Connection) -> None:
    """
    Create strategy_memory table if it doesn't exist.
    Mirrors the schema from ledger/schema.sql but is idempotent.
    This is a safety net only — the canonical schema lives in schema.sql.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_memory (
            candidate_id              TEXT PRIMARY KEY,
            strategy_id               TEXT NOT NULL,
            model_type                TEXT NOT NULL,
            model_id                  TEXT NOT NULL,
            stage                     TEXT NOT NULL,
            oos_trade_count           INTEGER,
            oos_win_rate              REAL,
            oos_avg_pnl_pct           REAL,
            cost_floor_pct            REAL,
            cost_floor_cleared        INTEGER,
            best_momentum_threshold   REAL,
            best_atr_pct_threshold    REAL,
            train_ratio               REAL,
            n_candles_total           INTEGER,
            n_candles_oos             INTEGER,
            created_ts                TEXT NOT NULL,
            evidence_json             TEXT
        )
        """
    )
    conn.commit()


def save_candidate(
    ledger_path: str,
    candidate: dict[str, Any],
    strategy_id: str = "trend_pullback_v1",
    model_type: str = "bucket_baseline",
) -> str:
    """
    Write candidate to strategy_memory. Returns candidate_id.

    candidate dict is the output of run_oos_validation().
    stage is always DRY_RUN_PLACEHOLDER for Sprint 3 output.
    """
    if candidate.get("stage") != "DRY_RUN_PLACEHOLDER":
        raise ValueError(
            "save_candidate only accepts DRY_RUN_PLACEHOLDER stage. "
            "Promotion requires full CPCV/DSR evidence."
        )

    candidate_id = f"bucket_baseline_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    model_id = candidate_id
    created_ts = datetime.now(timezone.utc).isoformat()

    # Fingerprint evidence for audit
    evidence_hash = hashlib.sha256(
        json.dumps(candidate, sort_keys=True, default=str).encode()
    ).hexdigest()

    row = {
        "candidate_id": candidate_id,
        "strategy_id": strategy_id,
        "model_type": model_type,
        "model_id": model_id,
        "stage": candidate["stage"],
        "oos_trade_count": candidate.get("oos_trade_count"),
        "oos_win_rate": candidate.get("oos_win_rate"),
        "oos_avg_pnl_pct": candidate.get("oos_avg_pnl_pct"),
        "cost_floor_pct": candidate.get("cost_floor_pct", COST_FLOOR_PCT),
        "cost_floor_cleared": int(bool(candidate.get("cost_floor_cleared", False))),
        "best_momentum_threshold": candidate.get("best_momentum_threshold"),
        "best_atr_pct_threshold": candidate.get("best_atr_pct_threshold"),
        "train_ratio": candidate.get("train_ratio", TRAIN_RATIO),
        "n_candles_total": candidate.get("n_candles_total"),
        "n_candles_oos": candidate.get("n_candles_oos"),
        "created_ts": created_ts,
        "evidence_json": json.dumps(
            {**candidate, "evidence_hash": evidence_hash}, default=str
        ),
    }

    conn = sqlite3.connect(ledger_path)
    try:
        _ensure_strategy_memory_table(conn)
        conn.execute(STRATEGY_MEMORY_INSERT, row)
        conn.commit()
        logger.info("Candidate saved: %s", candidate_id)
    finally:
        conn.close()

    return candidate_id


# ---------------------------------------------------------------------------
# 5. Orchestrator
# ---------------------------------------------------------------------------


def run_research_cycle(
    ledger_path: str | None = None,
    n_candles: int = 6000,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Full research cycle:
      generate → features → OOS validation → save candidate.

    Returns dict with candidate_id and OOS metrics.
    """
    if ledger_path is None:
        ledger_path = os.environ.get("LEDGER_DB_PATH", "./ledger/retailedge.db")

    logger.info("Starting research cycle. ledger=%s n_candles=%d", ledger_path, n_candles)

    df_raw = generate_synthetic_ohlcv(n_candles=n_candles, seed=seed)
    df_feat = compute_bucket_features(df_raw)
    candidate = run_oos_validation(df_feat)

    if candidate["status"] != "OK":
        logger.warning("Research cycle produced no valid candidate: %s", candidate["status"])
        return {"candidate_id": None, "result": candidate}

    candidate_id = save_candidate(ledger_path, candidate)

    return {
        "candidate_id": candidate_id,
        "stage": candidate["stage"],
        "oos_trade_count": candidate["oos_trade_count"],
        "oos_win_rate": candidate["oos_win_rate"],
        "oos_avg_pnl_pct": candidate["oos_avg_pnl_pct"],
        "cost_floor_cleared": candidate["cost_floor_cleared"],
        "best_momentum_threshold": candidate["best_momentum_threshold"],
        "best_atr_pct_threshold": candidate["best_atr_pct_threshold"],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run_research_cycle()
    print(json.dumps(result, indent=2, default=str))