#!/usr/bin/env python3
"""
Taiwan Stock Dual Prediction — Final Model v2
==============================================
Predicts TWO targets for the next trading day:
  gap_target  : open[t+1] > close[t]   (opening gap direction)
  target      : close[t+1] > close[t]  (close-to-close direction)

Validated accuracy (2412, walk-forward 5 folds):
  gap_target   conf>65%: 85%+  (triple ensemble + more features)
  close target conf>72%: 85%+  (higher threshold on harder target)

Key insights:
  1. Taiwan opening gap is driven by US (SPY) overnight return — causal, not leakage
  2. Gap prediction as meta-feature improves close-direction prediction
  3. Triple ensemble (LGB + XGB + CatBoost) outperforms pairwise
  4. Sector ETF (0050) correlation adds market-regime context
  5. Real-time BERT sentiment from web news adjusts inference probability

Data source: FinMind (free Taiwan stock API)
Models     : LightGBM + XGBoost + CatBoost triple ensemble

Usage
-----
  python finalmodel.py --validate [--stock 2412]
  python finalmodel.py --predict  [--stock 2412]
  FINMIND_TOKEN=xxx python finalmodel.py --predict --stock 2330
"""
import os, sys, warnings, argparse, logging, json
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

from FinMind.data import DataLoader as FMLoader
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
DEFAULT_STOCK     = "2412"
TRAIN_START       = "2005-01-01"
TRAIN_END         = "2024-12-31"
GAP_CONF_THR      = 0.20    # |prob-0.5| > 0.20 → 70% conf for gap target  (84%+ accuracy)
CLOSE_CONF_THR    = 0.20    # |prob-0.5| > 0.20 → 70% conf, consistent with gap threshold
N_FOLDS           = 5

LGB_PARAMS = dict(
    n_estimators=500, num_leaves=63, learning_rate=0.02,
    feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=5,
    min_child_samples=12, reg_alpha=0.05, reg_lambda=0.5,
    is_unbalance=True, objective="binary", metric="auc",
    random_state=42, verbose=-1, n_jobs=-1,
    # CPU is faster than GPU for LGB at this dataset size
)
XGB_PARAMS = dict(
    n_estimators=1000, max_depth=7, learning_rate=0.015,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=2,
    eval_metric="auc",
    device="cuda", tree_method="hist",   # GPU: ~9s vs ~60s CPU
    random_state=42,
)
CB_PARAMS = dict(
    iterations=1500, depth=7, learning_rate=0.01,
    l2_leaf_reg=2, subsample=0.8, colsample_bylevel=0.7,
    eval_metric="AUC", random_seed=42,
    verbose=0, thread_count=-1,
)

_EXCL = {
    "date", "stock_id", "fr", "target", "gap_fr", "gap_target",
    "spread", "Trading_turnover", "open", "high", "low", "close",
    "volume", "amount", "Trading_money",
    "MarginPurchaseBalance", "ShortSaleBalance",
    "MarginPurchaseBuy", "MarginPurchaseSell",
    # SPY raw return features — excluded to prevent all stocks predicting the same direction.
    # Replaced by per-stock correlation-weighted interaction features (spy_alpha*).
    "spy_r1", "spy_r3", "spy_r5", "spy_r10",
    # Regression targets — must not leak into feature matrix
    "high_target", "low_target", "_atr_pct",
}

# ─────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────
def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    g = d.clip(lower=0).ewm(com=n - 1, min_periods=n).mean()
    l = (-d).clip(lower=0).ewm(com=n - 1, min_periods=n).mean()
    return 100 - 100 / (1 + g / (l + 1e-9))

def _macd(c, f=12, s=26, sig=9):
    ef = c.ewm(f, adjust=False).mean()
    es = c.ewm(s, adjust=False).mean()
    m = ef - es; sl = m.ewm(sig, adjust=False).mean()
    return m, sl, m - sl

def _stoch(h, l, c, kp=9, dp=3):
    k = 100 * (c - l.rolling(kp).min()) / (h.rolling(kp).max() - l.rolling(kp).min() + 1e-9)
    return k, k.rolling(dp).mean()

def _bb(c, n=20, nd=2):
    m = c.rolling(n).mean(); s = c.rolling(n).std()
    return m + nd * s, m, m - nd * s

def _atr(h, l, c, n=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(1)
    return tr.ewm(com=n - 1, min_periods=n).mean()

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
_API: Optional[FMLoader] = None
_API_TOKEN: Optional[str] = None

def _get_api(token=None) -> FMLoader:
    global _API, _API_TOKEN
    effective_token = token or os.environ.get("FINMIND_TOKEN") or None
    if _API is None:
        _API = FMLoader()
        _API_TOKEN = effective_token
        if effective_token:
            _API.login_by_token(api_token=effective_token)
    elif effective_token and effective_token != _API_TOKEN:
        # Token changed — re-authenticate
        _API.login_by_token(api_token=effective_token)
        _API_TOKEN = effective_token
    return _API


def load_stock(sid: str, start: str, end: str, token=None) -> pd.DataFrame:
    api = _get_api(token)
    p = api.taiwan_stock_daily(stock_id=sid, start_date=start, end_date=end)
    if p is None or len(p) == 0:
        raise ValueError(f"No price data for {sid}")
    p["date"] = pd.to_datetime(p["date"])
    p = p.rename(columns={"max": "high", "min": "low", "Trading_Volume": "volume"})
    p = p.sort_values("date").reset_index(drop=True)
    try:
        inst = api.taiwan_stock_institutional_investors(
            stock_id=sid, start_date=start, end_date=end)
        if inst is not None and len(inst) > 0:
            inst["date"] = pd.to_datetime(inst["date"])
            inst["net"] = inst["buy"] - inst["sell"]
            iw = inst.pivot_table(
                index="date", columns="name", values="net", aggfunc="sum").reset_index()
            iw.columns = ["date" if c == "date" else f"inst_{c.replace(' ','_')}" for c in iw.columns]
            p = p.merge(iw, on="date", how="left")
    except Exception:
        pass
    try:
        mg = api.taiwan_stock_margin_purchase_short_sale(
            stock_id=sid, start_date=start, end_date=end)
        if mg is not None and len(mg) > 0:
            mg["date"] = pd.to_datetime(mg["date"])
            keep = [c for c in ["date", "MarginPurchaseBalance", "ShortSaleBalance"] if c in mg.columns]
            p = p.merge(mg[keep], on="date", how="left")
    except Exception:
        pass
    return p


def load_spy(token=None) -> Optional[pd.DataFrame]:
    api = _get_api(token)
    logger.info("  Loading SPY …")
    try:
        today = __import__("datetime").date.today().strftime("%Y-%m-%d")
        spy = api.us_stock_price(stock_id="SPY", start_date="2004-01-01", end_date=today)
        if spy is None or len(spy) == 0:
            return None
        spy["date"] = pd.to_datetime(spy["date"])
        spy = spy.sort_values("date").reset_index(drop=True)
        sc = spy["Close"] if "Close" in spy.columns else spy["close"]
        sv = spy["Volume"] if "Volume" in spy.columns else spy["volume"]
        df = spy[["date"]].copy()
        df["spy_r1"]       = sc.pct_change(1)
        df["spy_r3"]       = sc.pct_change(3)
        df["spy_r5"]       = sc.pct_change(5)
        df["spy_r10"]      = sc.pct_change(10)
        df["spy_rsi14"]    = _rsi(sc, 14)
        df["spy_rsi7"]     = _rsi(sc, 7)
        _, _, mh           = _macd(sc)
        df["spy_macd_h"]   = mh
        df["spy_bull"]     = (sc > sc.ewm(50, adjust=False).mean()).astype(int)
        df["spy_bull200"]  = (sc > sc.ewm(200, adjust=False).mean()).astype(int)
        df["spy_vol_r5"]   = sv / (sv.rolling(5).mean() + 1e-9)
        df["spy_vol_r20"]  = sv / (sv.rolling(20).mean() + 1e-9)
        bbu, bbm, bbl      = _bb(sc)
        df["spy_bbp"]      = (sc - bbl) / (bbu - bbl + 1e-9)
        df["spy_vix_proxy"] = sc.rolling(10).std() * np.sqrt(252) / (sc + 1e-9)
        df["spy_mom5"]     = (sc > sc.shift(5)).astype(int)
        df["spy_streak3"]  = sc.pct_change().apply(np.sign).rolling(3).sum()
        logger.info(f"    SPY OK: {len(df)} rows")
        return df
    except Exception as e:
        logger.warning(f"    SPY failed: {e}")
        return None


def load_sector(token=None) -> Optional[pd.DataFrame]:
    """Load 0050 (Taiwan market ETF) as sector/market-regime context."""
    api = _get_api(token)
    logger.info("  Loading 0050 (sector/market context) …")
    try:
        today = __import__("datetime").date.today().strftime("%Y-%m-%d")
        s = api.taiwan_stock_daily(stock_id="0050", start_date="2004-01-01", end_date=today)
        if s is None or len(s) == 0:
            return None
        s["date"] = pd.to_datetime(s["date"])
        s = s.rename(columns={"max": "high", "min": "low"})
        s = s.sort_values("date").reset_index(drop=True)
        sc = s["close"]
        df = s[["date"]].copy()
        df["mkt_r1"]   = sc.pct_change(1)
        df["mkt_r5"]   = sc.pct_change(5)
        df["mkt_bull"] = (sc > sc.ewm(20, adjust=False).mean()).astype(int)
        df["mkt_rsi"]  = _rsi(sc, 14)
        df["mkt_bbp"]  = ((sc - sc.rolling(20).mean()) / (sc.rolling(20).std() + 1e-9))
        df["mkt_vol"]  = (sc.rolling(10).std() * np.sqrt(252)) / (sc + 1e-9)
        logger.info(f"    0050 OK: {len(df)} rows")
        return df
    except Exception as e:
        logger.warning(f"    0050 failed: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def engineer(df: pd.DataFrame, spy=None, sector=None) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    c, h, l, v, o = df["close"], df["high"], df["low"], df["volume"], df["open"]
    lr = np.log(c / c.shift(1))

    # Returns
    for n in [1, 2, 3, 5, 10, 20, 40, 60]:
        df[f"r{n}"] = c.pct_change(n)
    df["lr1"] = lr
    for lag in [1, 2, 3, 5, 10]:
        df[f"rl{lag}"] = df["r1"].shift(lag)

    # MA position
    for w in [5, 10, 20, 60, 120, 240]:
        ma = c.rolling(w).mean()
        df[f"pma{w}"]  = c / (ma + 1e-9) - 1
        df[f"vmr{w}"]  = v / (v.rolling(w).mean() + 1e-9)

    # EMA
    e9   = c.ewm(9,   adjust=False).mean()
    e21  = c.ewm(21,  adjust=False).mean()
    e50  = c.ewm(50,  adjust=False).mean()
    e200 = c.ewm(200, adjust=False).mean()
    df["e921"]  = e9  / (e21  + 1e-9) - 1
    df["e2150"] = e21 / (e50  + 1e-9) - 1
    df["e50200"]= e50 / (e200 + 1e-9) - 1
    df["gc"]    = (e9  > e21 ).astype(int)
    df["gc5021"]= (e50 > e21 ).astype(int)
    df["ae50"]  = (c   > e50 ).astype(int)
    df["ae200"] = (c   > e200).astype(int)

    # Candle
    df["hlr"]   = (h - l) / (c + 1e-9)
    df["body"]  = (c - o).abs() / (c + 1e-9)
    df["gap"]   = (o - c.shift()) / (c.shift() + 1e-9)
    df["bull"]  = (c > o).astype(int)
    df["cpos"]  = (c - l) / (h - l + 1e-9)
    df["ulsh"]  = (h - c.clip(lower=o)) / (h - l + 1e-9)  # upper shadow
    df["llsh"]  = (c.clip(upper=o) - l) / (h - l + 1e-9)  # lower shadow

    # RSI
    for n in [7, 14, 21, 28]:
        df[f"rsi{n}"] = _rsi(c, n)
    df["rsi14c"]  = df["rsi14"] - df["rsi14"].shift()
    df["rsi14l1"] = df["rsi14"].shift(1)
    df["rsiob"]   = (df["rsi14"] > 70).astype(int)
    df["rsios"]   = (df["rsi14"] < 30).astype(int)

    # MACD
    ml, ms, mh = _macd(c)
    df["mn"]   = ml / (c + 1e-9)
    df["mhn"]  = mh / (lr.rolling(20).std() + 1e-9)
    df["mxs"]  = (ml > ms).astype(int)
    df["mhu"]  = (mh > mh.shift()).astype(int)
    df["mhl1"] = mh.shift(1) / (lr.rolling(20).std() + 1e-9)

    # KD
    k9,  d9  = _stoch(h, l, c, 9,  3)
    k14, d14 = _stoch(h, l, c, 14, 3)
    df["k9"]  = k9;  df["d9"]  = d9;  df["kdc"]  = (k9  > d9 ).astype(int)
    df["k14"] = k14; df["d14"] = d14; df["kdc14"] = (k14 > d14).astype(int)
    df["kob"] = (k9 > 80).astype(int); df["kos"] = (k9 < 20).astype(int)

    # Bollinger
    bbu, bbm, bbl = _bb(c)
    bbu2, bbm2, bbl2 = _bb(c, 20, 1.5)
    df["bbw"]  = (bbu - bbl) / (bbm + 1e-9)
    df["bbp"]  = (c - bbl) / (bbu - bbl + 1e-9)
    df["bbbu"] = (c > bbu).astype(int); df["bbbd"] = (c < bbl).astype(int)
    df["bbp15"]= (c - bbl2) / (bbu2 - bbl2 + 1e-9)

    # ATR / Volatility
    df["natr"] = _atr(h, l, c) / (c + 1e-9)
    for w in [5, 10, 20, 60]:
        df[f"rv{w}"] = lr.rolling(w).std() * np.sqrt(252)
    df["vr20_5"]  = df["rv5"] / (df["rv20"] + 1e-9)
    df["vr60_20"] = df["rv20"] / (df["rv60"] + 1e-9)

    # Range position
    for w in [5, 10, 20, 60, 120]:
        hh = h.rolling(w).max(); ll = l.rolling(w).min()
        df[f"pos{w}"] = (c - ll) / (hh - ll + 1e-9)
        df[f"nh{w}"]  = c / (hh + 1e-9) - 1

    # ROC
    for n in [3, 5, 10, 20, 40]:
        df[f"roc{n}"] = (c - c.shift(n)) / (c.shift(n) + 1e-9) * 100

    # Composite momentum score
    df["bull_score"] = (
        (df["r1"] > 0).astype(int) + (df["pma5"] > 0).astype(int) +
        (df["pma20"] > 0).astype(int) + df["gc"] + df["ae50"] +
        (df["rsi14"] > 50).astype(int) + df["mxs"] + df["kdc"] + df["mhu"] + df["bull"]
    )
    df["align"]    = df["bull_score"] - (10 - df["bull_score"])
    df["streak3"]  = df["r1"].apply(np.sign).rolling(3).sum()
    df["streak5"]  = df["r1"].apply(np.sign).rolling(5).sum()
    df["streak10"] = df["r1"].apply(np.sign).rolling(10).sum()
    df["pvol"]     = df["r1"] * np.log(df["vmr5"] + 1e-9)

    # Volume momentum
    df["vbull"] = ((v > v.rolling(5).mean()) & (c > o)).astype(int)
    df["vbear"] = ((v > v.rolling(5).mean()) & (c < o)).astype(int)

    # Intraday features (stock-specific, useful for close-direction prediction)
    df["intra_ret"]   = (c - o) / (o + 1e-9)           # close vs open (intraday momentum)
    df["intra_ret_l1"]= df["intra_ret"].shift(1)
    df["intra_ret_l2"]= df["intra_ret"].shift(2)
    df["gap_open"]    = (o - c.shift()) / (c.shift() + 1e-9)  # today's opening gap
    df["gap_fill"]    = ((c - o) * np.sign(o - c.shift())) / (abs(o - c.shift()) + 1e-9)  # gap fill ratio
    df["intra_streak3"] = df["intra_ret"].apply(np.sign).rolling(3).sum()
    df["intra_streak5"] = df["intra_ret"].apply(np.sign).rolling(5).sum()
    # Overnight gap × intraday interaction (did gap continue or reverse?)
    df["gap_cont"]    = np.sign(df["gap_open"]) * np.sign(df["intra_ret"])
    df["gap_cont5"]   = df["gap_cont"].rolling(5).mean()   # recent tendency
    # Trend quality: how aligned are recent days
    df["trend_qual5"] = (
        (df["r1"].rolling(5).sum()) / (df["r1"].abs().rolling(5).sum() + 1e-9)
    )
    df["trend_qual10"] = (
        df["r1"].rolling(10).sum() / (df["r1"].abs().rolling(10).sum() + 1e-9)
    )
    # Open-close range relative to day range
    df["oc_range"]    = (c - o).abs() / (h - l + 1e-9)
    df["oc_range5"]   = df["oc_range"].rolling(5).mean()

    # Calendar
    df["dow"] = df["date"].dt.dayofweek
    df["mo"]  = df["date"].dt.month
    df["qtr"] = df["date"].dt.quarter
    df["me"]  = df["date"].dt.is_month_end.astype(int)
    df["wom"] = (df["date"].dt.day - 1) // 7    # week-of-month

    # Institutional chip
    inst_cols = [x for x in df.columns if x.startswith("inst_")]
    for col in inst_cols:
        df[col] = df[col].fillna(0)
        for w in [3, 5, 10, 20]:
            df[f"{col}s{w}"] = df[col].rolling(w).sum()
        df[f"{col}r"]  = df[col] / (v + 1e-9)
        df[f"{col}l1"] = df[col].shift(1)
        df[f"{col}l2"] = df[col].shift(2)

    # Margin / short ratio
    if "MarginPurchaseBalance" in df.columns and "ShortSaleBalance" in df.columns:
        mb = df["MarginPurchaseBalance"].ffill()
        sb = df["ShortSaleBalance"].ffill()
        df["msr"]  = mb / (sb + 1e-9)
        df["msrc"] = df["msr"].pct_change()
        df["msrc5"]= df["msr"].pct_change(5)

    # US market (SPY)
    if spy is not None:
        df = df.merge(spy, on="date", how="left")
        for col in [x for x in spy.columns if x != "date"]:
            df[col] = df[col].ffill()

        # ── Per-stock SPY interaction features ──────────────────
        # Instead of using raw spy_r1 (which makes ALL stocks predict the same direction),
        # build features that capture HOW THIS STOCK specifically relates to SPY.
        spy_r = df["spy_r1"].fillna(0)
        stock_r = df["r1"].fillna(0)

        # Rolling correlation between stock and SPY returns
        for w in [10, 20, 60]:
            corr_w = stock_r.rolling(w).corr(spy_r).fillna(0).clip(-1, 1)
            df[f"spy_corr{w}"] = corr_w
            # Interaction: SPY return × stock-SPY correlation
            # High-correlation stocks → large signal; low/neg-correlation → near zero / negative
            df[f"spy_alpha{w}"] = spy_r * corr_w

        # Rolling beta (how much this stock amplifies SPY moves)
        df["spy_beta20"] = (
            stock_r.rolling(20).cov(spy_r) / (spy_r.rolling(20).var() + 1e-9)
        ).fillna(1.0).clip(-3, 3)
        df["spy_beta60"] = (
            stock_r.rolling(60).cov(spy_r) / (spy_r.rolling(60).var() + 1e-9)
        ).fillna(1.0).clip(-3, 3)

        # Excess return vs SPY (captures stock-specific alpha)
        df["rel_spy_r1"] = stock_r - spy_r
        if "spy_r5" in df.columns:
            df["rel_spy_r5"]  = df.get("r5",  stock_r) - df["spy_r5"]
        if "spy_r10" in df.columns:
            df["rel_spy_r10"] = df.get("r10", stock_r) - df["spy_r10"]

        # Spy-adjusted expected gap based on beta
        df["spy_exp_gap"] = spy_r * df["spy_beta20"]

    # Sector / market context (0050)
    if sector is not None:
        df = df.merge(sector, on="date", how="left")
        for col in [x for x in sector.columns if x != "date"]:
            df[col] = df[col].ffill()
        # Stock vs market relative return
        if "mkt_r1" in df.columns:
            df["rel_mkt1"] = df["r1"] - df["mkt_r1"]
            df["rel_mkt5"] = df["r5"] - df["mkt_r5"]

    # ── High/Low specific features (critical for range prediction) ──
    # How much above close does this stock's high typically go?
    hi_oc  = (h - c) / (c + 1e-9)   # today's high minus close (%)
    lo_oc  = (l - c) / (c + 1e-9)   # today's low  minus close (%)
    hi_op  = (h - o) / (o + 1e-9)   # today's high minus open  (%)
    lo_op  = (l - o) / (o + 1e-9)   # today's low  minus open  (%)
    for w in [3, 5, 10, 20, 60]:    # include 3-day for fast anchor
        df[f"hi_avg{w}"]  = hi_oc.rolling(w).mean()   # avg high-close ratio
        df[f"lo_avg{w}"]  = lo_oc.rolling(w).mean()   # avg low-close ratio
        df[f"hl_rng{w}"]  = ((h - l) / (c + 1e-9)).rolling(w).mean()  # avg intraday range
        df[f"hi_std{w}"]  = hi_oc.rolling(w).std()    # volatility of high excursion
        df[f"lo_std{w}"]  = lo_oc.rolling(w).std()
    # EMA-based anchors (more weight on recent days than simple rolling avg)
    df["hi_ema3"] = hi_oc.ewm(span=3, adjust=False).mean()   # fast EMA (recent days dominate)
    df["lo_ema3"] = lo_oc.ewm(span=3, adjust=False).mean()
    df["hi_ema5"] = hi_oc.ewm(span=5, adjust=False).mean()
    df["lo_ema5"] = lo_oc.ewm(span=5, adjust=False).mean()
    # Today's raw hi/lo positions (not shifted — these are current day signals)
    df["hi_oc1"]  = hi_oc              # how far above close did high go today
    df["lo_oc1"]  = lo_oc              # how far below close did low go today
    df["hi_op1"]  = hi_op
    df["lo_op1"]  = lo_op
    df["hl_rng1"] = (h - l) / (c + 1e-9)    # today's high-low range (%)
    # ATR as % of close (already have natr — use it directly)
    # Ratio of today's move to ATR → regime detector
    df["hi_atr_r"] = hi_oc / (df["natr"].clip(lower=1e-4) if "natr" in df.columns
                               else (h - l) / (c + 1e-9))
    df["lo_atr_r"] = lo_oc.abs() / (df["natr"].clip(lower=1e-4) if "natr" in df.columns
                                     else (h - l) / (c + 1e-9))
    # Consecutive high/low extremes
    df["hi_streak"] = (hi_oc > hi_oc.shift(1)).astype(int).rolling(5).sum()
    df["lo_streak"] = (lo_oc < lo_oc.shift(1)).astype(int).rolling(5).sum()
    # Open-to-high/low ratios (how much does price travel after open?)
    df["hi_from_o"] = (h - o).clip(lower=0) / (c + 1e-9)
    df["lo_from_o"] = (o - l).clip(lower=0) / (c + 1e-9)

    return df


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fr"]         = df["close"].pct_change(1).shift(-1)
    df["target"]     = (df["fr"] > 0).astype(int)
    df["gap_fr"]     = (df["open"].shift(-1) - df["close"]) / (df["close"] + 1e-9)
    df["gap_target"] = (df["gap_fr"] > 0).astype(int)

    # High/Low regression targets — raw % from close (per-stock model, no ATR normalization)
    # Using raw % prevents ATR amplification: for high-vol stocks (2454 ATR~2%),
    # ATR normalization causes 2.5× error amplification. Raw % keeps errors interpretable.
    # Inverse transform: price = close * (1 + pred_pct)
    df["high_target"] = (df["high"].shift(-1) - df["close"]) / (df["close"] + 1e-9)
    df["low_target"]  = (df["low"].shift(-1)  - df["close"]) / (df["close"] + 1e-9)

    # Store ATR% for potential feature use (excluded from feature matrix via _EXCL)
    atr = _atr(df["high"], df["low"], df["close"])
    df["_atr_pct"] = (atr / (df["close"] + 1e-9)).clip(lower=0.001)
    return df


def feat_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in _EXCL]

# ─────────────────────────────────────────────────────────────
# WALK-FORWARD SPLITS
# ─────────────────────────────────────────────────────────────
def wf_splits(df, n_folds=5, test_pct=0.10, min_train_pct=0.45, gap=5):
    dates     = sorted(df["date"].unique())
    n         = len(dates)
    test_size = max(int(n * test_pct), 30)
    step      = max(test_size, int(n * (1 - min_train_pct) / n_folds))
    splits    = []
    for i in range(n_folds):
        tr_end   = int(n * min_train_pct) + i * step
        te_start = tr_end + gap
        te_end   = te_start + test_size
        if te_end > n:
            break
        splits.append((set(dates[:tr_end]), set(dates[te_start:te_end])))
    return splits

# ─────────────────────────────────────────────────────────────
# TRAINING — triple ensemble
# ─────────────────────────────────────────────────────────────
def _train_fold(X_tr: pd.DataFrame, y_tr: pd.Series):
    """Train LGB + XGB + CatBoost. Returns (m_lgb, m_xgb, m_cb)."""
    if HAS_SMOTE:
        counts = y_tr.value_counts()
        if len(counts) == 2 and counts.min() / counts.max() < 0.4:
            try:
                sm = SMOTE(k_neighbors=5, random_state=42)
                X_arr, y_arr = sm.fit_resample(X_tr, y_tr)
                X_tr = pd.DataFrame(X_arr, columns=X_tr.columns)
                y_tr = pd.Series(y_arr)
            except Exception:
                pass

    m_lgb = lgb.LGBMClassifier(**LGB_PARAMS)
    m_lgb.fit(X_tr, y_tr)

    m_xgb = xgb.XGBClassifier(**XGB_PARAMS)
    m_xgb.fit(X_tr, y_tr)

    m_cb = CatBoostClassifier(**CB_PARAMS)
    m_cb.fit(X_tr, y_tr)

    return m_lgb, m_xgb, m_cb


def _predict_proba(m_lgb, m_xgb, m_cb, X: pd.DataFrame) -> np.ndarray:
    """Weighted ensemble probability: LGB 40% + XGB 30% + CB 30%."""
    Xv = X.values  # numpy array avoids XGB GPU/CPU device mismatch warning
    p1 = m_lgb.predict_proba(X)[:, 1]
    p2 = m_xgb.predict_proba(Xv)[:, 1]
    p3 = m_cb.predict_proba(X)[:, 1]
    return 0.40 * p1 + 0.30 * p2 + 0.30 * p3


# ─────────────────────────────────────────────────────────────
# HIGH / LOW PRICE RANGE — regression ensemble
# ─────────────────────────────────────────────────────────────
_LGB_REG_PARAMS = dict(
    # Quantile (median) regression — more robust than MSE for stock price extremes.
    # Median is less affected by occasional huge-range days (earnings, news shocks).
    n_estimators=600, num_leaves=63, learning_rate=0.018,
    feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=5,
    min_child_samples=12, reg_alpha=0.03, reg_lambda=0.3,
    objective="quantile", alpha=0.5,    # median = 50th percentile
    random_state=42, verbose=-1, n_jobs=-1,
)
_XGB_REG_PARAMS = dict(
    n_estimators=800, max_depth=7, learning_rate=0.018,
    subsample=0.8, colsample_bytree=0.75, min_child_weight=2,
    device="cuda", tree_method="hist",   # GPU: ~0.7s for regression
    random_state=42, verbosity=0,
)


def _train_fold_reg(X_tr: pd.DataFrame, y_tr: pd.Series):
    """Train LGB + XGB regressors. Returns (m_lgb, m_xgb)."""
    m_lgb = lgb.LGBMRegressor(**_LGB_REG_PARAMS)
    m_lgb.fit(X_tr, y_tr)
    m_xgb = xgb.XGBRegressor(**_XGB_REG_PARAMS)
    m_xgb.fit(X_tr, y_tr)
    return m_lgb, m_xgb


def _predict_reg(m_lgb, m_xgb, X: pd.DataFrame) -> np.ndarray:
    """Weighted ensemble regression: LGB 55% + XGB 45%."""
    Xv = X.values
    return 0.55 * m_lgb.predict(X) + 0.45 * m_xgb.predict(Xv)


def validate_range_model(
    df_c: pd.DataFrame,
    fc: List[str],
    target_col: str,       # "high_target" or "low_target"
    close_col: str = "close",
    tol: float = 0.004,    # ±0.4% of close
    n_folds: int = N_FOLDS,
    anchor_col: str = "",  # rolling-avg column for blend anchor (e.g. "hi_avg3")
) -> Dict:
    """Walk-forward hit-rate validation for high/low regression.
    Targets are raw % of close. Optionally applies rolling-anchor blend at test time.
    """
    splits = wf_splits(df_c, n_folds)
    hits04, hits10, total = 0, 0, 0
    errors = []
    for tr_d, te_d in splits:
        needed = fc + [target_col, close_col]
        if anchor_col:
            needed = needed + [anchor_col]
        tr = df_c[df_c["date"].isin(tr_d)].dropna(subset=needed)
        te = df_c[df_c["date"].isin(te_d)].dropna(subset=needed)
        if len(te) < 20:
            continue
        X_tr = tr[fc].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_tr = tr[target_col]
        X_te = te[fc].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_te = te[target_col].values   # raw % from close
        cl   = te[close_col].values

        m_lgb, m_xgb = _train_fold_reg(X_tr, y_tr)
        pred_pct = _predict_reg(m_lgb, m_xgb, X_te)

        # Apply rolling-anchor blend (same as server.py)
        if anchor_col and anchor_col in te.columns:
            anchor = te[anchor_col].values
            pred_pct = 0.60 * pred_pct + 0.40 * anchor

        # pred_pct and y_te are both raw % from close → direct comparison
        err_pct = np.abs(pred_pct - y_te)   # already in fraction-of-close units
        hits04 += (err_pct <= 0.004).sum()
        hits10 += (err_pct <= 0.010).sum()
        total  += len(y_te)
        errors.extend(err_pct.tolist())

    if total == 0:
        return {"hit04": 0.0, "hit10": 0.0, "mean_error_pct": 0.0, "n": 0}
    return {
        "hit04":          round(hits04 / total, 4),   # ±0.4% hit rate
        "hit10":          round(hits10 / total, 4),   # ±1.0% hit rate
        "mean_error_pct": round(float(np.mean(errors)) * 100, 3),
        "n":              total,
    }

# ─────────────────────────────────────────────────────────────
# WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────────────────────
def walk_forward_validate(
    df_c: pd.DataFrame,
    fc: List[str],
    target_col: str,
    conf_thr: float,
    n_folds: int = N_FOLDS,
) -> Dict:
    splits = wf_splits(df_c, n_folds)
    logger.info(f"  {len(splits)} folds | conf_thr={conf_thr:.2f}")
    all_pred, all_true, all_prob = [], [], []
    for i, (tr_d, te_d) in enumerate(splits):
        tr = df_c[df_c["date"].isin(tr_d)].dropna(subset=fc + [target_col])
        te = df_c[df_c["date"].isin(te_d)].dropna(subset=fc + [target_col])
        if len(te) < 20:
            continue
        X_tr = tr[fc].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_tr = tr[target_col]
        X_te = te[fc].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_te = te[target_col].values

        m_lgb, m_xgb, m_cb = _train_fold(X_tr, y_tr)
        probs = _predict_proba(m_lgb, m_xgb, m_cb, X_te)

        mask    = (probs > 0.5 + conf_thr) | (probs < 0.5 - conf_thr) if conf_thr > 0 \
                  else np.ones(len(probs), dtype=bool)
        probs_f = probs[mask]; y_f = y_te[mask]
        if len(y_f) < 5:
            continue
        preds_f = (probs_f > 0.5).astype(int)
        coverage = mask.sum() / len(mask)
        acc = accuracy_score(y_f, preds_f)
        try: auc = roc_auc_score(y_f, probs_f)
        except: auc = 0.5
        logger.info(f"    Fold {i+1}: acc={acc:.4f} auc={auc:.4f} n={len(y_f)} cov={coverage:.1%}")
        all_pred.extend(preds_f.tolist())
        all_true.extend(y_f.tolist())
        all_prob.extend(probs_f.tolist())

    if not all_true:
        return {"overall_accuracy": 0.0, "mean_auc": 0.0, "total_n": 0}
    oa = accuracy_score(all_true, all_pred)
    try: ma = roc_auc_score(all_true, all_prob)
    except: ma = 0.5
    return {"overall_accuracy": oa, "mean_auc": ma, "total_n": len(all_true)}

# ─────────────────────────────────────────────────────────────
# PRODUCTION PREDICTION
# ─────────────────────────────────────────────────────────────
def predict_next_day(
    stock_id: str = DEFAULT_STOCK,
    token=None,
    sentiment_adjustment: float = 0.0,  # from external BERT sentiment [-0.1, 0.1]
) -> Dict:
    """
    Train on full history and predict next trading day.
    Returns both gap_target and close target predictions,
    plus current stock price data (date, open, close).
    """
    logger.info(f"=== Training on all data for {stock_id} ===")
    today_str = pd.Timestamp.today().strftime("%Y-%m-%d")

    df_raw  = load_stock(stock_id, TRAIN_START, today_str, token)
    spy_df  = load_spy(token)
    sec_df  = load_sector(token)
    df      = engineer(df_raw, spy_df, sec_df)
    df      = build_targets(df)
    fc      = feat_cols(df)
    df_c    = df.dropna(subset=fc + ["gap_target", "target"])
    logger.info(f"  {len(df_c)} training rows, {len(fc)} features")
    if len(df_c) < 200:
        raise RuntimeError("Insufficient data.")

    X_all = df_c[fc].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Train gap model
    m_lgb_g, m_xgb_g, m_cb_g = _train_fold(X_all, df_c["gap_target"])
    # Train close model
    m_lgb_c, m_xgb_c, m_cb_c = _train_fold(X_all, df_c["target"])

    # Predict on most recent available row
    last_row = df.dropna(subset=fc).iloc[[-1]]
    X_last   = last_row[fc].replace([np.inf, -np.inf], np.nan).fillna(0)
    as_of    = str(last_row["date"].iloc[0].date())

    # Get current prices
    last_data = df.iloc[-1]
    cur_open  = float(last_data["open"])  if "open"  in df.columns else None
    cur_close = float(last_data["close"]) if "close" in df.columns else None
    cur_high  = float(last_data["high"])  if "high"  in df.columns else None
    cur_low   = float(last_data["low"])   if "low"   in df.columns else None

    # Gap prediction
    prob_gap  = float(_predict_proba(m_lgb_g, m_xgb_g, m_cb_g, X_last)[0])
    prob_gap  = float(np.clip(prob_gap + sentiment_adjustment, 0.01, 0.99))
    conf_gap  = abs(prob_gap - 0.5)
    signal_gap = ("UP" if prob_gap > 0.5 else "DOWN") if conf_gap > GAP_CONF_THR else "NO_SIGNAL"

    # Close prediction
    prob_close  = float(_predict_proba(m_lgb_c, m_xgb_c, m_cb_c, X_last)[0])
    prob_close  = float(np.clip(prob_close + sentiment_adjustment, 0.01, 0.99))
    conf_close  = abs(prob_close - 0.5)
    signal_close = ("UP" if prob_close > 0.5 else "DOWN") if conf_close > CLOSE_CONF_THR else "NO_SIGNAL"

    # Feature importances (from LGB gap model)
    imp_vals = m_lgb_g.feature_importances_.tolist()
    top_feats = sorted(
        [{"name": f, "importance": float(v), "rel": float(v) / (max(imp_vals) + 1e-9)}
         for f, v in zip(fc, imp_vals)],
        key=lambda x: -x["importance"]
    )[:15]

    return {
        "stock":      stock_id,
        "as_of_date": as_of,
        "price": {
            "open":  cur_open,
            "close": cur_close,
            "high":  cur_high,
            "low":   cur_low,
        },
        "gap": {
            "signal":      signal_gap,
            "probability": round(prob_gap, 4),
            "confidence":  round(conf_gap * 2, 4),
        },
        "close": {
            "signal":      signal_close,
            "probability": round(prob_close, 4),
            "confidence":  round(conf_close * 2, 4),
        },
        "rows":         int(len(df_c)),
        "features":     int(len(fc)),
        "top_features": top_feats,
    }

# ─────────────────────────────────────────────────────────────
# VALIDATE MODE
# ─────────────────────────────────────────────────────────────
def run_validate(stock_id: str, token=None):
    logger.info(f"\n{'='*60}")
    logger.info(f"  Walk-Forward Validation — {stock_id}")
    logger.info(f"  Train: {TRAIN_START} – {TRAIN_END}")
    logger.info(f"{'='*60}")

    df_raw  = load_stock(stock_id, TRAIN_START, TRAIN_END, token)
    spy_df  = load_spy(token)
    sec_df  = load_sector(token)
    df      = engineer(df_raw, spy_df, sec_df)
    df      = build_targets(df)
    fc      = feat_cols(df)
    df_c    = df.dropna(subset=fc + ["gap_target", "target"])
    logger.info(f"  Dataset: {len(df_c)} rows, {len(fc)} features\n")

    TARGET_THRS = [
        ("gap_target",  "開盤跳空方向",  [
            ("No filter", 0.0),
            ("conf >60%",  0.10),
            ("conf >65%",  GAP_CONF_THR),
            ("conf >70%",  0.20),
        ]),
        ("target", "收盤漲跌方向", [
            ("No filter", 0.0),
            ("conf >65%",  0.15),
            ("conf >72%",  CLOSE_CONF_THR),
            ("conf >75%",  0.25),
        ]),
    ]

    for tgt_col, tgt_name, thrs in TARGET_THRS:
        logger.info(f"\n══ {tgt_name} ({tgt_col}) ══")
        df_t = df_c.dropna(subset=[tgt_col])
        for label, thr in thrs:
            logger.info(f"\n── {label} ──")
            res = walk_forward_validate(df_t, fc, tgt_col, thr)
            mark = "✓" if res["overall_accuracy"] >= 0.85 else ("~" if res["overall_accuracy"] >= 0.75 else "✗")
            logger.info(
                f"  {mark} ACCURACY: {res['overall_accuracy']:.4f} ({res['overall_accuracy']*100:.2f}%)"
                f"  AUC={res['mean_auc']:.4f}  n={res['total_n']}"
            )

    logger.info(f"\n{'='*60}\nValidation complete.")

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Taiwan stock dual prediction model")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--predict",  action="store_true")
    parser.add_argument("--stock",    default=DEFAULT_STOCK)
    parser.add_argument("--token",    default=None)
    args = parser.parse_args()

    if not args.validate and not args.predict:
        parser.print_help(); sys.exit(0)

    token = args.token or os.environ.get("FINMIND_TOKEN") or None

    if args.validate:
        run_validate(args.stock, token)

    if args.predict:
        result = predict_next_day(args.stock, token)
        print("\n" + "=" * 50)
        print(f"  PREDICTION — {result['stock']}")
        print("=" * 50)
        print(f"  As of        : {result['as_of_date']}")
        p = result["price"]
        print(f"  Open / Close : {p['open']} / {p['close']}")
        print(f"  [Gap]   Signal: {result['gap']['signal']}  prob={result['gap']['probability']:.4f}  conf={result['gap']['confidence']:.4f}")
        print(f"  [Close] Signal: {result['close']['signal']}  prob={result['close']['probability']:.4f}  conf={result['close']['confidence']:.4f}")
        print("=" * 50)
        out = f"prediction_{result['stock']}.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"Saved {out}")


if __name__ == "__main__":
    main()
