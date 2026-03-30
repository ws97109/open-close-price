#!/usr/bin/env python3
"""
Taiwan Stock Next-Day Up/Down Prediction System v2
Data Source: FinMind
Target: >75% accuracy
Method: LightGBM + XGBoost ensemble, custom TA indicators (no library dependency issues)
"""
import os
import sys
import warnings
import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

from FinMind.data import DataLoader
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ═══════════════════════════════════════════════════════════
# 1. CUSTOM TECHNICAL INDICATORS (pure pandas/numpy)
# ═══════════════════════════════════════════════════════════
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=n - 1, min_periods=n).mean()
    avg_loss = loss.ewm(com=n - 1, min_periods=n).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period=9, d_period=3) -> Tuple[pd.Series, pd.Series]:
    low_min = low.rolling(k_period).min()
    high_max = high.rolling(k_period).max()
    k = 100 * (close - low_min) / (high_max - low_min + 1e-9)
    d = k.rolling(d_period).mean()
    return k, d


def bollinger_bands(close: pd.Series, n=20, ndev=2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(n).mean()
    std = close.rolling(n).std()
    upper = mid + ndev * std
    lower = mid - ndev * std
    return upper, mid, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n=14) -> pd.Series:
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(com=n - 1, min_periods=n).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n=14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up_move = high.diff()
    dn_move = (-low.diff())
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)
    atr_val = atr(high, low, close, n)
    plus_di = 100 * plus_dm.ewm(com=n - 1, min_periods=n).mean() / (atr_val + 1e-9)
    minus_di = 100 * minus_dm.ewm(com=n - 1, min_periods=n).mean() / (atr_val + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx_val = dx.ewm(com=n - 1, min_periods=n).mean()
    return adx_val, plus_di, minus_di


def cci(high: pd.Series, low: pd.Series, close: pd.Series, n=20) -> pd.Series:
    tp = (high + low + close) / 3
    ma = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * mad + 1e-9)


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, n=14) -> pd.Series:
    high_max = high.rolling(n).max()
    low_min = low.rolling(n).min()
    return -100 * (high_max - close) / (high_max - low_min + 1e-9)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# ═══════════════════════════════════════════════════════════
# 2. DATA LOADING
# ═══════════════════════════════════════════════════════════
def load_data(
    stock_id: str,
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-31",
    api_token: Optional[str] = None,
) -> pd.DataFrame:
    api = DataLoader()
    if api_token:
        api.login_by_token(api_token=api_token)

    logger.info(f"[{stock_id}] Fetching price …")
    price = api.taiwan_stock_daily(
        stock_id=stock_id, start_date=start_date, end_date=end_date
    )
    if price is None or len(price) == 0:
        raise ValueError(f"No price data for {stock_id}")

    price["date"] = pd.to_datetime(price["date"])
    price = price.rename(
        columns={"max": "high", "min": "low",
                 "Trading_Volume": "volume", "Trading_money": "amount"}
    ).sort_values("date").reset_index(drop=True)

    # Institutional investors
    logger.info(f"[{stock_id}] Fetching institutional …")
    try:
        inst = api.taiwan_stock_institutional_investors(
            stock_id=stock_id, start_date=start_date, end_date=end_date
        )
        if inst is not None and len(inst) > 0:
            inst["date"] = pd.to_datetime(inst["date"])
            inst["net"] = inst["buy"] - inst["sell"]
            inst_wide = inst.pivot_table(
                index="date", columns="name", values="net", aggfunc="sum"
            ).reset_index()
            inst_wide.columns = [
                "date" if c == "date" else f"inst_{c.replace(' ', '_')}"
                for c in inst_wide.columns
            ]
            price = price.merge(inst_wide, on="date", how="left")
    except Exception as e:
        logger.warning(f"[{stock_id}] Institutional error: {e}")

    # Margin
    logger.info(f"[{stock_id}] Fetching margin …")
    try:
        margin = api.taiwan_stock_margin_purchase_short_sale(
            stock_id=stock_id, start_date=start_date, end_date=end_date
        )
        if margin is not None and len(margin) > 0:
            margin["date"] = pd.to_datetime(margin["date"])
            keep = [c for c in ["date", "MarginPurchaseBalance", "ShortSaleBalance",
                                 "MarginPurchaseBuy", "MarginPurchaseSell"] if c in margin.columns]
            price = price.merge(margin[keep], on="date", how="left")
    except Exception as e:
        logger.warning(f"[{stock_id}] Margin error: {e}")

    return price


# ═══════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    o = df["open"]
    log_ret = np.log(c / c.shift(1))

    # ── Returns ──────────────────────────────────────────
    for n in [1, 2, 3, 5, 10, 20]:
        df[f"ret_{n}d"] = c.pct_change(n)
    df["log_ret"] = log_ret

    # ── Lag features ─────────────────────────────────────
    for lag in [1, 2, 3, 5]:
        df[f"ret_lag{lag}"] = df["ret_1d"].shift(lag)
        df[f"vol_lag{lag}"] = v.shift(lag)
        df[f"close_lag{lag}"] = c.shift(lag)

    # ── Moving averages & price position ─────────────────
    for w in [5, 10, 20, 60]:
        ma = c.rolling(w).mean()
        df[f"ma{w}"] = ma
        df[f"price_vs_ma{w}"] = c / (ma + 1e-9) - 1
        df[f"vol_ma{w}"] = v.rolling(w).mean()
    df["vol_ratio5"] = v / (df["vol_ma5"] + 1e-9)
    df["vol_ratio20"] = v / (df["vol_ma20"] + 1e-9)
    df["vol_spike"] = (df["vol_ratio5"] > 2.0).astype(int)

    # ── EMA ───────────────────────────────────────────────
    for w in [9, 21, 50]:
        df[f"ema{w}"] = c.ewm(span=w, adjust=False).mean()
    df["ema9_21"] = df["ema9"] / (df["ema21"] + 1e-9) - 1
    df["ema21_50"] = df["ema21"] / (df["ema50"] + 1e-9) - 1
    df["golden_cross"] = (df["ema9"] > df["ema21"]).astype(int)
    df["above_ema50"] = (c > df["ema50"]).astype(int)

    # ── Candle structure ──────────────────────────────────
    df["hl_spread"] = (h - l) / (c + 1e-9)
    df["body"] = (c - o).abs() / (c + 1e-9)
    df["upper_shadow"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / (c + 1e-9)
    df["lower_shadow"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / (c + 1e-9)
    df["gap_open"] = (o - c.shift(1)) / (c.shift(1) + 1e-9)
    df["bull_candle"] = (c > o).astype(int)
    df["close_vs_hl"] = (c - l) / (h - l + 1e-9)  # where in range did close land

    # ── RSI ───────────────────────────────────────────────
    df["rsi_7"] = rsi(c, 7)
    df["rsi_14"] = rsi(c, 14)
    df["rsi_21"] = rsi(c, 21)
    df["rsi_14_lag1"] = df["rsi_14"].shift(1)
    df["rsi_chg"] = df["rsi_14"] - df["rsi_14_lag1"]
    df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
    df["rsi_oversold"] = (df["rsi_14"] < 30).astype(int)

    # ── MACD ──────────────────────────────────────────────
    macd_line, sig_line, hist = macd(c)
    df["macd"] = macd_line
    df["macd_sig"] = sig_line
    df["macd_hist"] = hist
    df["macd_hist_lag1"] = hist.shift(1)
    df["macd_cross"] = (macd_line > sig_line).astype(int)
    df["macd_hist_up"] = (hist > hist.shift(1)).astype(int)
    df["macd_norm"] = macd_line / (c + 1e-9)

    # ── KD (Stochastic) — very popular in Taiwan market ──
    k9, d9 = stochastic(h, l, c, k_period=9, d_period=3)
    df["stoch_k9"] = k9
    df["stoch_d9"] = d9
    df["kd_cross9"] = (k9 > d9).astype(int)

    k14, d14 = stochastic(h, l, c, k_period=14, d_period=3)
    df["stoch_k14"] = k14
    df["stoch_d14"] = d14
    df["kd_cross14"] = (k14 > d14).astype(int)

    # ── Williams %R ───────────────────────────────────────
    df["willr_14"] = williams_r(h, l, c, 14)

    # ── CCI ───────────────────────────────────────────────
    df["cci_14"] = cci(h, l, c, 14)
    df["cci_20"] = cci(h, l, c, 20)

    # ── ADX ───────────────────────────────────────────────
    adx_val, plus_di, minus_di = adx(h, l, c, 14)
    df["adx_14"] = adx_val
    df["adx_plus_di"] = plus_di
    df["adx_minus_di"] = minus_di
    df["adx_di_diff"] = plus_di - minus_di
    df["adx_strong"] = (adx_val > 25).astype(int)

    # ── Bollinger Bands ───────────────────────────────────
    bb_up, bb_mid, bb_dn = bollinger_bands(c, 20, 2)
    df["bb_width"] = (bb_up - bb_dn) / (bb_mid + 1e-9)
    df["bb_pct"] = (c - bb_dn) / (bb_up - bb_dn + 1e-9)
    df["bb_break_up"] = (c > bb_up).astype(int)
    df["bb_break_dn"] = (c < bb_dn).astype(int)

    # ── ATR / NATR ────────────────────────────────────────
    atr14 = atr(h, l, c, 14)
    df["atr_14"] = atr14
    df["natr_14"] = atr14 / (c + 1e-9)

    # ── Realized volatility ───────────────────────────────
    for w in [5, 10, 20]:
        df[f"rv{w}"] = log_ret.rolling(w).std() * np.sqrt(252)
        df[f"ret_skew{w}"] = log_ret.rolling(w).skew()
        df[f"ret_max{w}"] = log_ret.rolling(w).max()
        df[f"ret_min{w}"] = log_ret.rolling(w).min()

    # ── OBV ───────────────────────────────────────────────
    obv_val = obv(c, v)
    obv_ma20 = obv_val.rolling(20).mean()
    df["obv_trend"] = (obv_val > obv_ma20).astype(int)
    df["obv_chg"] = obv_val.pct_change(5)

    # ── ROC ───────────────────────────────────────────────
    for n in [5, 10, 20]:
        df[f"roc_{n}"] = (c - c.shift(n)) / (c.shift(n) + 1e-9) * 100

    # ── Range position ────────────────────────────────────
    for w in [5, 10, 20]:
        hi = h.rolling(w).max()
        lo = l.rolling(w).min()
        df[f"near_high{w}"] = c / (hi + 1e-9) - 1
        df[f"near_low{w}"] = c / (lo + 1e-9) - 1
        df[f"range_pos{w}"] = (c - lo) / (hi - lo + 1e-9)

    # ── Calendar ──────────────────────────────────────────
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)

    return df


def build_chip_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    inst_cols = [c for c in df.columns if c.startswith("inst_")]

    for col in inst_cols:
        df[col] = df[col].fillna(0)
        for w in [3, 5, 10, 20]:
            df[f"{col}_sum{w}"] = df[col].rolling(w).sum()
        df[f"{col}_ratio"] = df[col] / (df["volume"] + 1e-9)
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_sum5_lag1"] = df[f"{col}_sum5"].shift(1)

    if "MarginPurchaseBalance" in df.columns:
        mb = df["MarginPurchaseBalance"].ffill()
        df["margin_chg"] = mb.pct_change()
        df["margin_ma5"] = mb.rolling(5).mean()
        df["margin_vs_ma5"] = mb / (df["margin_ma5"] + 1e-9) - 1

    if "ShortSaleBalance" in df.columns:
        sb = df["ShortSaleBalance"].ffill()
        df["short_chg"] = sb.pct_change()

    if "MarginPurchaseBalance" in df.columns and "ShortSaleBalance" in df.columns:
        mb = df["MarginPurchaseBalance"].ffill()
        sb = df["ShortSaleBalance"].ffill()
        df["ms_ratio"] = mb / (sb + 1e-9)
        df["ms_ratio_lag1"] = df["ms_ratio"].shift(1)
        df["ms_ratio_chg"] = df["ms_ratio"].pct_change()

    return df


# ═══════════════════════════════════════════════════════════
# 4. TARGET
# ═══════════════════════════════════════════════════════════
def build_target(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    df = df.copy()
    df["future_ret"] = df["close"].pct_change(1).shift(-1)
    df["target"] = (df["future_ret"] > threshold).astype(int)
    return df


# ═══════════════════════════════════════════════════════════
# 5. FEATURE COLUMNS
# ═══════════════════════════════════════════════════════════
_EXCLUDE = {
    "date", "stock_id", "future_ret", "target", "spread", "Trading_turnover",
    "open", "high", "low", "close", "volume", "amount",
    "ma5", "ma10", "ma20", "ma60",
    "vol_ma5", "vol_ma10", "vol_ma20", "vol_ma60",
    "ema9", "ema21", "ema50",
    "MarginPurchaseBalance", "ShortSaleBalance",
    "MarginPurchaseBuy", "MarginPurchaseSell",
    "margin_ma5",
}

def get_feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in _EXCLUDE]


# ═══════════════════════════════════════════════════════════
# 6. WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════
def walk_forward_validate(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_builder,
    n_splits: int = 5,
    gap: int = 5,
) -> Dict:
    df_c = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
    X = df_c[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df_c["target"]

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    all_preds, all_true, all_probs = [], [], []
    fold_scores = []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        scaler = StandardScaler()
        X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
        X_te_s = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns)

        model = model_builder()
        model.fit(X_tr_s, y_tr)

        preds = model.predict(X_te_s)
        probs = model.predict_proba(X_te_s)[:, 1]

        acc = accuracy_score(y_te, preds)
        auc = roc_auc_score(y_te, probs)

        all_preds.extend(preds.tolist())
        all_true.extend(y_te.tolist())
        all_probs.extend(probs.tolist())
        fold_scores.append({"fold": fold + 1, "accuracy": acc, "auc": auc})
        logger.info(f"    Fold {fold+1}: acc={acc:.4f}  auc={auc:.4f}  n={len(te_idx)}")

    overall_acc = accuracy_score(all_true, all_preds)
    mean_auc = roc_auc_score(all_true, all_probs)
    return {
        "fold_scores": fold_scores,
        "overall_accuracy": overall_acc,
        "mean_auc": mean_auc,
    }


# ═══════════════════════════════════════════════════════════
# 7. STACKING ENSEMBLE
# ═══════════════════════════════════════════════════════════
def stacking_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_splits: int = 5,
    gap: int = 5,
) -> Dict:
    df_c = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
    X = df_c[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df_c["target"]

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    te_indices = []

    lgb_p = dict(n_estimators=600, num_leaves=63, learning_rate=0.03,
                 feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
                 min_child_samples=15, reg_alpha=0.1, reg_lambda=1.0,
                 is_unbalance=True, objective="binary", metric="auc",
                 random_state=42, verbose=-1, n_jobs=-1)
    xgb_p = dict(n_estimators=600, max_depth=5, learning_rate=0.03,
                 subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
                 reg_alpha=0.1, reg_lambda=1.0,
                 eval_metric="auc", use_label_encoder=False,
                 random_state=42, n_jobs=-1)

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr = y.iloc[tr_idx]

        scaler = StandardScaler()
        X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
        X_te_s = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns)

        m1 = lgb.LGBMClassifier(**lgb_p)
        m1.fit(X_tr_s, y_tr)
        oof_lgb[te_idx] = m1.predict_proba(X_te_s)[:, 1]

        m2 = xgb.XGBClassifier(**xgb_p)
        m2.fit(X_tr_s, y_tr)
        oof_xgb[te_idx] = m2.predict_proba(X_te_s)[:, 1]

        te_indices.extend(te_idx.tolist())

    # Collect aligned
    te_idx_arr = np.array(te_indices)
    true_vals = y.iloc[te_idx_arr].values
    lgb_probs = oof_lgb[te_idx_arr]
    xgb_probs = oof_xgb[te_idx_arr]

    # Average ensemble
    avg_probs = (lgb_probs + xgb_probs) / 2
    avg_preds = (avg_probs > 0.5).astype(int)
    acc = accuracy_score(true_vals, avg_preds)
    auc = roc_auc_score(true_vals, avg_probs)

    logger.info(f"    Ensemble (avg) acc={acc:.4f}  auc={auc:.4f}")
    return {"overall_accuracy": acc, "mean_auc": auc}


# ═══════════════════════════════════════════════════════════
# 8. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════
def run_stock(
    stock_id: str,
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-31",
    api_token: Optional[str] = None,
) -> Dict:
    df = load_data(stock_id, start_date, end_date, api_token)
    df = build_features(df)
    df = build_chip_features(df)
    df = build_target(df, threshold=0.0)

    feat_cols = get_feature_cols(df)
    df_c = df.dropna(subset=feat_cols + ["target"])
    logger.info(f"[{stock_id}] {len(df_c)} rows, {len(feat_cols)} features")

    if len(df_c) < 500:
        raise ValueError(f"Not enough data: {len(df_c)} rows")

    # LightGBM
    logger.info(f"[{stock_id}] LightGBM walk-forward …")
    lgbm_res = walk_forward_validate(
        df_c, feat_cols,
        lambda: lgb.LGBMClassifier(
            n_estimators=600, num_leaves=63, learning_rate=0.03,
            feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
            min_child_samples=15, reg_alpha=0.1, reg_lambda=1.0,
            is_unbalance=True, objective="binary", metric="auc",
            random_state=42, verbose=-1, n_jobs=-1,
        ),
    )

    # Stacking
    logger.info(f"[{stock_id}] Stacking ensemble walk-forward …")
    stack_res = stacking_cv(df_c, feat_cols)

    return {
        "lgbm_acc": lgbm_res["overall_accuracy"],
        "lgbm_auc": lgbm_res["mean_auc"],
        "stack_acc": stack_res["overall_accuracy"],
        "stack_auc": stack_res["mean_auc"],
        "best_acc": max(lgbm_res["overall_accuracy"], stack_res["overall_accuracy"]),
        "df": df_c,
        "feat_cols": feat_cols,
    }


if __name__ == "__main__":
    API_TOKEN = os.environ.get("FINMIND_TOKEN", "") or None
    TARGET_ACC = 0.75
    STOCK_IDS = ["2330", "2317", "2454", "2412", "1301"]
    START, END = "2015-01-01", "2024-12-31"

    logger.info("=== Taiwan Stock Prediction System v2 ===")
    results = {}

    for sid in STOCK_IDS:
        logger.info(f"\n{'='*60}\nStock: {sid}\n{'='*60}")
        try:
            r = run_stock(sid, START, END, API_TOKEN)
            results[sid] = r
            logger.info(f"[{sid}] LGBM={r['lgbm_acc']:.4f}  Stack={r['stack_acc']:.4f}  Best={r['best_acc']:.4f}")
        except Exception as e:
            import traceback
            logger.error(f"[{sid}] {e}")
            traceback.print_exc()

    logger.info("\n=== Summary ===")
    for sid, r in results.items():
        mark = "✓" if r["best_acc"] >= TARGET_ACC else "✗"
        logger.info(f"  {mark} {sid}: best_acc={r['best_acc']:.4f}")

    if results:
        best = max(results, key=lambda k: results[k]["best_acc"])
        logger.info(f"\nBest: {best} = {results[best]['best_acc']:.4f}")
        if results[best]["best_acc"] >= TARGET_ACC:
            logger.info("TARGET REACHED → ready to build finalmodel.py")
        else:
            logger.info("Need improvement → will try pooled multi-stock + advanced methods")
