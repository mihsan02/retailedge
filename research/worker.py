"""
RetailEdge — Research Worker
Sprint 3 Revisited, S3-R1: Real Freqtrade disk data loader.

CHANGE LOG:
- generate_synthetic_ohlcv() REPLACED by load_freqtrade_ohlcv()
- All other functions (compute_bucket_features, run_oos_validation,
  save_candidate, run_research_cycle) are UNCHANGED.

DATA FORMAT (Freqtrade default JSON):
  File: freqtrade/user_data/data/binance/{BASE}_{QUOTE}-{timeframe}.json
  Content: [[timestamp_ms, open, high, low, close, volume], ...]

SUPPORTED FORMATS (checked in order): json, feather, parquet
  feather/parquet require pyarrow. If absent and file is not json,
  raises ImportError with install instruction.

MINIMUM CANDLE REQUIREMENT: 5000 (from CLAUDE.md Hard Constraint #6)
  If data is absent or below threshold, raises DataInsufficientError.
  This is NOT a soft warning. It blocks execution.
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from research.atr_percentile_service import add_atr_pct
from research.atr_percentile_service import (
    add_atr_pct,
    compute_atr_percentile_baseline,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class DataInsufficientError(Exception):
    """Raised when on-disk data is absent or below the 5000-candle minimum."""


class DataFormatError(Exception):
    """Raised when the data file exists but cannot be parsed."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_CANDLES = 5000
REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
DEFAULT_DATA_DIR = "freqtrade/user_data/data/binance"
DEFAULT_TIMEFRAME = "15m"
DEFAULT_PAIRS = ["BTC/USDT", "ETH/USDT"]
FORMAT_EXTENSIONS = ["json", "feather", "parquet"]
# Trailing window for ATR percentile gate. Must equal RetailEdgeStrategy.ATR_MIN_LOOKBACK.
# INVARIANT: changing this without changing the strategy breaks train-serve parity.
ATR_PERCENTILE_LOOKBACK = 4999


# ---------------------------------------------------------------------------
# DOWNLOAD INSTRUCTIONS (printed when data is missing)
# ---------------------------------------------------------------------------

_DOWNLOAD_INSTRUCTIONS = """
=============================================================
MISSING DATA — Manual action required
=============================================================
No Freqtrade OHLCV data found at: {path}

Run the following command inside the Freqtrade container:

  docker exec retailedge_freqtrade \\
    freqtrade download-data \\
    --exchange binance \\
    --pairs BTC/USDT ETH/USDT \\
    --timeframe 15m \\
    --days 90

90 days of 15m candles = 8640 candles per pair (well above the 5000 minimum).

After download, re-run the research worker:
  python -m research.worker
=============================================================
"""


# ---------------------------------------------------------------------------
# Loader — replaces generate_synthetic_ohlcv()
# ---------------------------------------------------------------------------

def _find_data_file(data_dir: str, pair: str, timeframe: str) -> tuple[Optional[str], Optional[str]]:
    """
    Locate the data file for a pair/timeframe across supported formats.

    Returns (file_path, format_ext) or (None, None) if not found.
    """
    safe_pair = pair.replace("/", "_")
    for ext in FORMAT_EXTENSIONS:
        candidate = os.path.join(data_dir, f"{safe_pair}-{timeframe}.{ext}")
        if os.path.exists(candidate):
            return candidate, ext
    return None, None


def _load_json(path: str) -> pd.DataFrame:
    """Load Freqtrade JSON format: [[timestamp_ms, o, h, l, c, v], ...]"""
    with open(path, "r") as f:
        raw = json.load(f)

    if not isinstance(raw, list) or len(raw) == 0:
        raise DataFormatError(f"JSON file is empty or not a list: {path}")

    first = raw[0]
    if not isinstance(first, (list, tuple)) or len(first) < 6:
        raise DataFormatError(
            f"Expected rows of [timestamp, o, h, l, c, v], got: {first!r} in {path}"
        )

    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df[REQUIRED_COLUMNS].copy()


def _load_feather(path: str) -> pd.DataFrame:
    try:
        import pyarrow  # noqa: F401 — presence check
    except ImportError:
        raise ImportError(
            f"File {path} is feather format but pyarrow is not installed. "
            "Run: pip install pyarrow --break-system-packages"
        )
    df = pd.read_feather(path)
    return _normalize_columns(df, path)


def _load_parquet(path: str) -> pd.DataFrame:
    try:
        import pyarrow  # noqa: F401 — presence check
    except ImportError:
        raise ImportError(
            f"File {path} is parquet format but pyarrow is not installed. "
            "Run: pip install pyarrow --break-system-packages"
        )
    df = pd.read_parquet(path)
    return _normalize_columns(df, path)


def _normalize_columns(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """Normalize feather/parquet DataFrames to standard column set."""
    col_map = {}
    for col in df.columns:
        if col.lower() in ("date", "timestamp", "time", "open_time"):
            col_map[col] = "date"
        elif col.lower() == "open":
            col_map[col] = "open"
        elif col.lower() == "high":
            col_map[col] = "high"
        elif col.lower() == "low":
            col_map[col] = "low"
        elif col.lower() == "close":
            col_map[col] = "close"
        elif col.lower() == "volume":
            col_map[col] = "volume"

    df = df.rename(columns=col_map)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataFormatError(f"Missing columns {missing} in {path}")

    # Ensure date column is datetime with UTC
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], utc=True)

    return df[REQUIRED_COLUMNS].copy()


def load_freqtrade_ohlcv(
    pairs: list[str] = None,
    timeframe: str = DEFAULT_TIMEFRAME,
    data_dir: str = DEFAULT_DATA_DIR,
    min_candles: int = MIN_CANDLES,
) -> dict[str, pd.DataFrame]:
    """
    Load OHLCV data from Freqtrade disk storage for one or more pairs.

    Replaces generate_synthetic_ohlcv(). Returns a dict keyed by pair.

    Args:
        pairs: List of pairs e.g. ["BTC/USDT", "ETH/USDT"]
        timeframe: Freqtrade timeframe string e.g. "15m"
        data_dir: Path to exchange data directory
        min_candles: Minimum candle count required (default 5000 per CLAUDE.md)

    Returns:
        dict[pair_str, DataFrame] with columns [date, open, high, low, close, volume]

    Raises:
        DataInsufficientError: If any pair has no data or fewer than min_candles.
        DataFormatError: If data file exists but cannot be parsed.
    """
    if pairs is None:
        pairs = DEFAULT_PAIRS

    results = {}
    missing_pairs = []

    for pair in pairs:
        path, ext = _find_data_file(data_dir, pair, timeframe)

        if path is None:
            missing_pairs.append(pair)
            continue

        logger.info("Loading %s %s from %s (format: %s)", pair, timeframe, path, ext)

        loaders = {
            "json": _load_json,
            "feather": _load_feather,
            "parquet": _load_parquet,
        }
        df = loaders[ext](path)

        if len(df) < min_candles:
            raise DataInsufficientError(
                f"{pair} {timeframe}: only {len(df)} candles available, "
                f"need >= {min_candles}. "
                f"Run: freqtrade download-data --pairs {pair} --timeframe {timeframe} --days 90"
            )

        df = df.sort_values("date").reset_index(drop=True)
        results[pair] = df
        logger.info("Loaded %s %s: %d candles (%s to %s)",
                    pair, timeframe, len(df),
                    df["date"].iloc[0].isoformat(),
                    df["date"].iloc[-1].isoformat())

    if missing_pairs:
        search_path = os.path.join(data_dir, "*.json / *.feather / *.parquet")
        print(_DOWNLOAD_INSTRUCTIONS.format(path=search_path))
        raise DataInsufficientError(
            f"No data files found for pairs: {missing_pairs} "
            f"in {data_dir} (timeframe: {timeframe}). "
            "See download instructions above."
        )

    return results


# ---------------------------------------------------------------------------
# All functions below are UNCHANGED from Sprint 3 original.
# Do not modify without updating test_research_worker.py.
# ---------------------------------------------------------------------------

def compute_bucket_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add bucket features to OHLCV DataFrame.
    Inputs:  [date, open, high, low, close, volume]
    Outputs: adds [momentum_5, atr_pct, fwd_return]
    """
    df = df.copy()

    # Momentum: 5-candle log return
    df["momentum_5"] = np.log(df["close"] / df["close"].shift(5))

    # ATR percentile input: delegate to canonical Wilder ewm helper.
    # Single source of truth shared with live strategy. Do not open-code ATR here.
    df = add_atr_pct(df)

    # Forward return (1 candle ahead) — OOS target
    df["fwd_return"] = df["close"].shift(-1) / df["close"] - 1

    return df.dropna().reset_index(drop=True)


def run_oos_validation(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    cost_floor_pct: float = 0.0046,
    atr_lookback: int = ATR_PERCENTILE_LOOKBACK,
) -> dict:
    """
    Simple 80/20 OOS validation.
    Scans momentum_5 thresholds on train, applies best threshold to OOS.

    Returns OOS metrics dict.
    """
    if "momentum_5" not in df.columns or "fwd_return" not in df.columns:
        raise ValueError("DataFrame must have momentum_5 and fwd_return columns")

    n = len(df)
    split = int(n * train_ratio)
    train = df.iloc[:split]
    oos = df.iloc[split:].copy()

    # Threshold scan on train
    thresholds = np.percentile(train["momentum_5"].dropna(), [60, 70, 80, 90])
    best_threshold = None
    best_train_win_rate = -1.0

    for t in thresholds:
        signals = train[train["momentum_5"] > t]
        if len(signals) < 10:
            continue
        win_rate = (signals["fwd_return"] > 0).mean()
        if win_rate > best_train_win_rate:
            best_train_win_rate = win_rate
            best_threshold = t

    if best_threshold is None:
        return {
            "status": "INSUFFICIENT_TRAIN_SIGNALS",
            "threshold_long": None,
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_pnl_pct": 0.0,
            "cost_floor_cleared": False,
        }

    # Momentum signals on OOS (pre-gate)
    oos_signals = oos[oos["momentum_5"] > best_threshold].copy()

    if len(oos_signals) == 0:
        return {
            "status": "NO_OOS_SIGNALS",
            "threshold_long": float(best_threshold),
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_pnl_pct": 0.0,
            "cost_floor_cleared": False,
            "regime_gated": True,
        }

    # F2 parity gate: keep only rows live would actually enter.
    # Live blocks entry when trailing ATR percentile >= HIGH_VOL_THRESHOLD (0.80).
    # Each row evaluated causally (history up to that row only) — no lookahead.
    keep = []
    for i in oos_signals.index:
        res = compute_atr_percentile_baseline(df.iloc[: i + 1], min_lookback=atr_lookback)
        if res["status"] == "OK" and not res["is_high_vol"]:
            keep.append(i)

    gated = oos_signals.loc[keep]
    trade_count = len(gated)

    if trade_count == 0:
        return {
            "status": "NO_OOS_SIGNALS_AFTER_GATE",
            "threshold_long": float(best_threshold),
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_pnl_pct": 0.0,
            "cost_floor_cleared": False,
            "regime_gated": True,
        }

    win_rate = float((gated["fwd_return"] > 0).mean())
    avg_pnl_pct = float(gated["fwd_return"].mean())
    cost_floor_cleared = avg_pnl_pct > cost_floor_pct

    return {
        "status": "OK",
        "threshold_long": float(best_threshold),
        "trade_count": trade_count,
        "win_rate": round(win_rate, 4),
        "avg_pnl_pct": round(avg_pnl_pct, 6),
        "cost_floor_cleared": cost_floor_cleared,
        "regime_gated": True,
        "min_sample_ok": trade_count >= 30,
    }

    return {
        "status": "OK",
        "threshold_long": float(best_threshold),
        "trade_count": trade_count,
        "win_rate": round(win_rate, 4),
        "avg_pnl_pct": round(avg_pnl_pct, 6),
        "cost_floor_cleared": cost_floor_cleared,
    }


def save_candidate(ledger_path: str, candidate: dict) -> str:
    """
    Persist a research candidate to the strategy_memory table.

    Enforces: stage must not be DRY_RUN_PLACEHOLDER.
    Returns candidate_id.
    """
    import sqlite3

    stage = candidate.get("stage", "")
    if stage == "DRY_RUN_PLACEHOLDER":
        raise ValueError(
            "save_candidate: stage='DRY_RUN_PLACEHOLDER' is rejected. "
            "Use a real data stage such as 'RESEARCH_OOS' or 'REAL_DATA_OOS'."
        )

    candidate_id = candidate.get("candidate_id") or str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    row = {
        "candidate_id": candidate_id,
        "strategy_id": candidate.get("strategy_id", "bucket_baseline"),
        "model_type": candidate.get("model_type", "bucket_baseline"),
        "stage": stage,
        "oos_metrics": json.dumps(candidate.get("oos_metrics", {})),
        "pair": candidate.get("pair", ""),
        "data_source": candidate.get("data_source", ""),
        "created_ts": now,
        "notes": candidate.get("notes", ""),
    }

    with sqlite3.connect(ledger_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO strategy_memory
              (candidate_id, strategy_id, model_type, stage,
               oos_metrics, pair, data_source, created_ts, notes)
            VALUES
              (:candidate_id, :strategy_id, :model_type, :stage,
               :oos_metrics, :pair, :data_source, :created_ts, :notes)
        """, row)
        conn.commit()

    logger.info("Saved candidate %s (stage=%s, pair=%s)", candidate_id, stage, row["pair"])
    return candidate_id


def run_research_cycle(
    pairs: list[str] = None,
    timeframe: str = DEFAULT_TIMEFRAME,
    data_dir: str = DEFAULT_DATA_DIR,
    ledger_path: str = "./ledger/retailedge.db",
    cost_floor_pct: float = 0.0046,
) -> list[str]:
    """
    Orchestrate one full research cycle for all target pairs.

    For each pair:
      1. Load real OHLCV from Freqtrade disk
      2. Compute bucket features
      3. Run OOS validation
      4. Save candidate to ledger

    Returns list of saved candidate_ids.
    """
    if pairs is None:
        pairs = DEFAULT_PAIRS

    # Load all pairs — raises DataInsufficientError if any pair missing
    pair_data = load_freqtrade_ohlcv(
        pairs=pairs,
        timeframe=timeframe,
        data_dir=data_dir,
    )

    candidate_ids = []

    for pair, df_raw in pair_data.items():
        logger.info("Research cycle: %s (%d candles)", pair, len(df_raw))

        df_features = compute_bucket_features(df_raw)
        oos_result = run_oos_validation(df_features, cost_floor_pct=cost_floor_pct)

        candidate = {
            "strategy_id": "bucket_baseline",
            "model_type": "bucket_baseline",
            "stage": "REAL_DATA_OOS",
            "pair": pair,
            "data_source": f"freqtrade_disk:{data_dir}/{pair.replace('/', '_')}-{timeframe}",
            "oos_metrics": oos_result,
            "notes": f"Real Binance Spot {timeframe} data. OOS status: {oos_result.get('status')}",
        }

        cid = save_candidate(ledger_path, candidate)
        candidate_ids.append(cid)
        logger.info(
            "Candidate saved: %s | pair=%s | avg_pnl=%.4f%% | cost_floor_cleared=%s",
            cid, pair,
            oos_result.get("avg_pnl_pct", 0) * 100,
            oos_result.get("cost_floor_cleared"),
        )

    return candidate_ids


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ids = run_research_cycle()
    print(f"Research cycle complete. Candidates: {ids}")