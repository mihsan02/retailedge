"""
RetailEdge Fase 0 — Gross Expectancy Triage
=============================================
Cheap killer filter BEFORE any cost/CPCV/DSR machinery.

Question this answers: does any feature idea have materially positive GROSS
per-trade expectancy at all? If not on the same venue/timeframe, the project's
dead momentum strategy already told you the likely answer.

This is TRIAGE ONLY. Any feature that survives must be re-implemented through
your canonical module (parity with serve, Gate 6) before it enters Fase 1.
Do NOT wire this script into research/worker.py. It computes features inline
on purpose, to stay independent of your harness while triaging.

Frozen parameters (do not tune in Fase 0):
  ENTRY_QUANTILE = 0.80   # long when feature in top quintile (or bottom for reversion)
  HORIZON        = 4       # forward holding, in 15m candles (1 hour)
  COST_FLOOR     = 0.0046  # 0.46% round-trip, from spec Gate 2
  T_HURDLE       = 3.0     # Harvey/Liu/Zhu factor-zoo bar, not 2.0

N_fase0 = number of features below = 9.

Usage:
  python fase0_gross_triage.py
Adjust DATA_DIR / FILENAME_TMPL to match your Freqtrade feather layout, then run.
Standard Freqtrade columns assumed: date, open, high, low, close, volume.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---- CONFIG (confirm these match your environment) ----------------------------
DATA_DIR = Path("freqtrade/user_data/data/binance")     # adjust to your path
FILENAME_TMPL = "{pair}-15m.feather"           # e.g. BTC_USDT-15m.feather
PAIRS = ["BTC_USDT", "ETH_USDT"]

ENTRY_QUANTILE = 0.80
HORIZON = 4
COST_FLOOR = 0.0046
T_HURDLE = 3.0
# ------------------------------------------------------------------------------


def load(pair: str) -> pd.DataFrame:
    fp = DATA_DIR / FILENAME_TMPL.format(pair=pair)
    df = pd.read_feather(fp)
    df.columns = [c.lower() for c in df.columns]
    needed = {"open", "high", "low", "close", "volume"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{fp} missing columns: {missing}. Confirm feather layout.")
    return df.reset_index(drop=True)


def forward_log_return(close: pd.Series, h: int) -> pd.Series:
    # label: log return from t to t+h. Forward by construction (this is the target,
    # not a feature). Signals below use only info up to t, so no leakage into entry.
    return np.log(close.shift(-h) / close)


# ---- FEATURES: each returns a signal series. side=+1 long-when-high, -1 long-when-low
def f_momentum5(df):      # TREND baseline / calibration (known ~dead)
    return np.log(df["close"] / df["close"].shift(5)), +1

def f_trend_slope(df):    # TREND: OLS slope of close over 20 bars
    win = 20
    x = np.arange(win)
    xm = x - x.mean()
    denom = (xm ** 2).sum()
    def slope(w):
        return (xm * (w - w.mean())).sum() / denom
    return df["close"].rolling(win).apply(slope, raw=True), +1

def f_zscore(df):         # REVERSION: price z-score vs 20-bar mean -> long when low
    win = 20
    m = df["close"].rolling(win).mean()
    s = df["close"].rolling(win).std()
    return (df["close"] - m) / s, -1

def f_rsi(df):            # REVERSION: RSI(14) -> long when low (oversold)
    win = 14
    d = df["close"].diff()
    up = d.clip(lower=0).ewm(alpha=1/win, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/win, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs), -1

def f_donchian(df):       # STRUCTURE: close position in 20-bar range -> long on breakout high
    win = 20
    hi = df["high"].rolling(win).max()
    lo = df["low"].rolling(win).min()
    return (df["close"] - lo) / (hi - lo), +1

def f_atr_pct(df):        # VOLATILITY as predictor: ATR percentile (Wilder ewm)
    win = 14
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/win, adjust=False).mean()
    atr_pct = atr / df["close"]
    return atr_pct.rolling(500).rank(pct=True), +1

def f_vol_surge(df):      # FLOW: volume z-score over 20 bars
    win = 20
    m = df["volume"].rolling(win).mean()
    s = df["volume"].rolling(win).std()
    return (df["volume"] - m) / s, +1

def f_clv(df):            # FLOW: close location value within the bar (intrabar pressure)
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng, +1

def f_hour(df):           # CALENDAR: hour-of-day. ranks hours by signal, long on high-hour
    if "date" in df.columns:
        hour = pd.to_datetime(df["date"]).dt.hour
    else:
        hour = pd.Series(np.arange(len(df)) % 96 // 4, index=df.index)  # fallback
    return hour.astype(float), +1


FEATURES = {
    "momentum5_BASELINE": f_momentum5,
    "trend_slope": f_trend_slope,
    "zscore_reversion": f_zscore,
    "rsi_reversion": f_rsi,
    "donchian_breakout": f_donchian,
    "atr_pct_predictor": f_atr_pct,
    "volume_surge": f_vol_surge,
    "clv_intrabar": f_clv,
    "hour_of_day": f_hour,
}


def evaluate(df, feat_fn):
    sig, side = feat_fn(df)
    fwd = forward_log_return(df["close"], HORIZON)
    valid = sig.notna() & fwd.notna()
    sig, fwd = sig[valid], fwd[valid]
    if len(sig) < 100:
        return None
    if side > 0:
        thr = sig.quantile(ENTRY_QUANTILE)
        entries = sig >= thr
    else:
        thr = sig.quantile(1 - ENTRY_QUANTILE)
        entries = sig <= thr
    r = fwd[entries] * side   # realized gross log return per signalled trade
    n = len(r)
    if n < 30:
        return None
    mean = r.mean()
    se = r.std(ddof=1) / np.sqrt(n)
    t = mean / se if se > 0 else 0.0
    return {"n": n, "gross_pct": mean * 100, "t": t,
            "pass": (mean >= COST_FLOOR) and (t > T_HURDLE)}


def main():
    for pair in PAIRS:
        try:
            df = load(pair)
        except Exception as e:
            print(f"\n[{pair}] LOAD FAILED: {e}")
            continue
        print(f"\n=== {pair}  (rows={len(df)}, horizon={HORIZON}, floor={COST_FLOOR*100:.2f}%, t>{T_HURDLE}) ===")
        print(f"{'feature':<22}{'n':>7}{'gross/trade':>14}{'t':>8}  verdict")
        rows = []
        for name, fn in FEATURES.items():
            res = evaluate(df, fn)
            if res is None:
                print(f"{name:<22}{'--':>7}{'insufficient':>14}{'--':>8}  SKIP")
                continue
            rows.append((name, res))
        rows.sort(key=lambda x: x[1]["gross_pct"], reverse=True)
        any_pass = False
        for name, res in rows:
            v = "PASS" if res["pass"] else "fail"
            any_pass = any_pass or res["pass"]
            print(f"{name:<22}{res['n']:>7}{res['gross_pct']:>13.3f}%{res['t']:>8.2f}  {v}")
        print(f"\n[{pair}] Fase 0 verdict: {'CANDIDATE(S) SURVIVE -> Fase 1' if any_pass else 'NULL (no idea clears gross floor)'}")

    print("\nReminder: PASS here is necessary, not sufficient. Survivors must be")
    print("re-implemented via your canonical module with a parity test before Fase 1.")


if __name__ == "__main__":
    main()
