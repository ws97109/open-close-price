#!/usr/bin/env python3
"""
Taiwan Stock Next-Day Up/Down Prediction System
Data Source: FinMind
Target: >75% accuracy
Strategy: LightGBM + XGBoost ensemble with technical + chip features
"""
import os
import sys
import warnings
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict, Tuple

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Packages ──────────────────────────────────────────────
from FinMind.data import DataLoader
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import ta


# ═══════════════════════════════════════════════════════════
# 1. DATA LOADING
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

    logger.info(f"[{stock_id}] Fetching price data …")
    price = api.taiwan_stock_daily(
        stock_id=stock_id, start_date=start_date, end_date=end_date
    )
    if price is None or len(price) == 0:
        raise ValueError(f"No price data for {stock_id}")

    price["date"] = pd.to_datetime(price["date"])
    price = price.rename(
        columns={
            "max": "high",
            "min": "low",
            "Trading_Volume": "volume",
            "Trading_money": "amount",
        }
    ).sort_values("date").reset_index(drop=True)

    # ── Institutional investors ──────────────────────────
    logger.info(f"[{stock_id}] Fetching institutional data …")
    try:
        inst = api.taiwan_stock_institutional_investors(
            stock_id=stock_id, start_date=start_date, end_date=end_date
        )
        if inst is not None and len(inst) > 0:
            inst["date"] = pd.to_datetime(inst["date"])
            inst["net"] = inst["buy"] - inst["sell"]
            inst_wide = (
                inst.pivot_table(
                    index="date", columns="name", values="net", aggfunc="sum"
                )
                .reset_index()
            )
            inst_wide.columns = [
                "date" if c == "date" else f"inst_{c.replace(' ', '_')}"
                for c in inst_wide.columns
            ]
            price = price.merge(inst_wide, on="date", how="left")
    except Exception as e:
        logger.warning(f"[{stock_id}] Institutional data error: {e}")

    # ── Margin / Short sale ──────────────────────────────
    logger.info(f"[{stock_id}] Fetching margin data …")
    try:
        margin = api.taiwan_stock_margin_purchase_short_sale(
            stock_id=stock_id, start_date=start_date, end_date=end_date
        )
        if margin is not None and len(margin) > 0:
            margin["date"] = pd.to_datetime(margin["date"])
            keep = [
                c for c in ["date", "MarginPurchaseBalance", "ShortSaleBalance",
                             "MarginPurchaseBuy", "MarginPurchaseSell",
                             "ShortSaleBuy", "ShortSaleSell"]
                if c in margin.columns
            ]
            price = price.merge(margin[keep], on="date", how="left")
    except Exception as e:
        logger.warning(f"[{stock_id}] Margin data error: {e}")

    return price


# ═══════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════
def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    # Returns & lags
    for n in [1, 2, 3, 5, 10, 20]:
        df[f"ret_{n}d"] = c.pct_change(n)
    log_ret = np.log(c / c.shift(1))
    df["log_ret"] = log_ret
    for lag in [1, 2, 3, 5]:
        df[f"ret_lag{lag}"] = df["ret_1d"].shift(lag)
        df[f"vol_lag{lag}"] = v.shift(lag)

    # Moving averages & position
    for w in [5, 10, 20, 60]:
        ma = c.rolling(w).mean()
        df[f"ma{w}"] = ma
        df[f"price_vs_ma{w}"] = c / ma - 1
        df[f"vol_ma{w}"] = v.rolling(w).mean()
    df["vol_ratio5"] = v / df["vol_ma5"]
    df["vol_ratio20"] = v / df["vol_ma20"]

    # EMA
    for w in [9, 21, 50]:
        df[f"ema{w}"] = c.ewm(span=w, adjust=False).mean()
    df["ema9_21"] = df["ema9"] / df["ema21"] - 1
    df["ema21_50"] = df["ema21"] / df["ema50"] - 1
    df["golden_cross"] = (df["ema9"] > df["ema21"]).astype(int)

    # Candle structure
    o = df["open"]
    df["hl_spread"] = (h - l) / c
    df["body"] = (c - o).abs() / c
    df["upper_shadow"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / c
    df["lower_shadow"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / c
    df["gap_open"] = (o - c.shift(1)) / c.shift(1)
    df["bull_candle"] = (c > o).astype(int)
    df["close_vs_high"] = (c - h) / (h - l + 1e-9)

    # RSI
    df["rsi_7"] = ta.momentum.rsi(c, window=7)
    df["rsi_14"] = ta.momentum.rsi(c, window=14)
    df["rsi_21"] = ta.momentum.rsi(c, window=21)
    df["rsi_14_lag1"] = df["rsi_14"].shift(1)
    df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
    df["rsi_oversold"] = (df["rsi_14"] < 30).astype(int)

    # MACD
    macd = ta.trend.MACD(c)
    df["macd"] = macd.macd()
    df["macd_sig"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    df["macd_hist_lag1"] = df["macd_hist"].shift(1)
    df["macd_cross"] = (df["macd"] > df["macd_sig"]).astype(int)
    df["macd_hist_up"] = (df["macd_hist"] > df["macd_hist_lag1"]).astype(int)

    # KD Stochastic (very popular in Taiwan)
    stoch = ta.momentum.StochasticOscillator(h, l, c, window=9, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    df["kd_cross"] = (df["stoch_k"] > df["stoch_d"]).astype(int)

    stoch14 = ta.momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
    df["stoch_k14"] = stoch14.stoch()
    df["stoch_d14"] = stoch14.stoch_signal()

    # Williams %R
    df["willr_14"] = ta.momentum.WilliamsRIndicator(h, l, c, lbp=14).williams_r()

    # CCI
    df["cci_14"] = ta.trend.CCIIndicator(h, l, c, window=14).cci()
    df["cci_20"] = ta.trend.CCIIndicator(h, l, c, window=20).cci()

    # ROC / Momentum
    df["roc_5"] = ta.momentum.ROCIndicator(c, window=5).roc()
    df["roc_10"] = ta.momentum.ROCIndicator(c, window=10).roc()
    df["roc_20"] = ta.momentum.ROCIndicator(c, window=20).roc()

    # ADX (trend strength)
    adx = ta.trend.ADXIndicator(h, l, c, window=14)
    df["adx_14"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()
    df["adx_di_diff"] = df["adx_pos"] - df["adx_neg"]
    df["adx_strong"] = (df["adx_14"] > 25).astype(int)

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + 1e-9)
    df["bb_pct"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
    df["bb_break_up"] = (c > df["bb_upper"]).astype(int)
    df["bb_break_dn"] = (c < df["bb_lower"]).astype(int)

    # ATR / NATR
    df["atr_14"] = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    df["natr_14"] = df["atr_14"] / c

    # Realized volatility
    for w in [5, 10, 20]:
        df[f"rv{w}"] = log_ret.rolling(w).std() * np.sqrt(252)
        df[f"ret_skew{w}"] = log_ret.rolling(w).skew()
        df[f"ret_max{w}"] = log_ret.rolling(w).max()
        df[f"ret_min{w}"] = log_ret.rolling(w).min()

    # OBV
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    df["obv_ma20"] = df["obv"].rolling(20).mean()
    df["obv_trend"] = (df["obv"] > df["obv_ma20"]).astype(int)

    # Rolling high / low position
    for w in [5, 10, 20]:
        hi = h.rolling(w).max()
        lo = l.rolling(w).min()
        df[f"near_high{w}"] = c / hi - 1
        df[f"near_low{w}"] = c / lo - 1
        df[f"range_pos{w}"] = (c - lo) / (hi - lo + 1e-9)

    # Calendar
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    return df


def add_chip_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Institutional investor net positions
    inst_cols = [c for c in df.columns if c.startswith("inst_")]
    for col in inst_cols:
        df[col] = df[col].fillna(0)
        for w in [3, 5, 10, 20]:
            df[f"{col}_sum{w}"] = df[col].rolling(w).sum()
        df[f"{col}_ratio"] = df[col] / (df["volume"] + 1e-9)
        # Shift by 1: chip data is same-day but after market close
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_sum5_lag1"] = df[f"{col}_sum5"].shift(1)
        df[f"{col}_sum10_lag1"] = df[f"{col}_sum10"].shift(1)

    # Margin / short sale
    if "MarginPurchaseBalance" in df.columns:
        df["MarginPurchaseBalance"] = df["MarginPurchaseBalance"].ffill()
        mb = df["MarginPurchaseBalance"]
        df["margin_chg"] = mb.pct_change()
        df["margin_ma5"] = mb.rolling(5).mean()
        df["margin_vs_ma5"] = mb / df["margin_ma5"] - 1

    if "ShortSaleBalance" in df.columns:
        df["ShortSaleBalance"] = df["ShortSaleBalance"].ffill()
        sb = df["ShortSaleBalance"]
        df["short_chg"] = sb.pct_change()
        df["short_ma5"] = sb.rolling(5).mean()

    if "MarginPurchaseBalance" in df.columns and "ShortSaleBalance" in df.columns:
        df["ms_ratio"] = df["MarginPurchaseBalance"] / (df["ShortSaleBalance"] + 1e-9)
        df["ms_ratio_lag1"] = df["ms_ratio"].shift(1)
        df["ms_ratio_chg"] = df["ms_ratio"].pct_change()

    return df


# ═══════════════════════════════════════════════════════════
# 3. TARGET
# ═══════════════════════════════════════════════════════════
def build_target(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    df = df.copy()
    df["future_ret"] = df["close"].pct_change(1).shift(-1)
    df["target"] = (df["future_ret"] > threshold).astype(int)
    return df


# ═══════════════════════════════════════════════════════════
# 4. FEATURE SELECTION
# ═══════════════════════════════════════════════════════════
_EXCLUDE = {
    "date", "stock_id", "future_ret", "target", "spread", "Trading_turnover",
    "open", "high", "low", "close", "volume", "amount",
    # raw derived columns not meant as features
    "ma5", "ma10", "ma20", "ma60",
    "vol_ma5", "vol_ma10", "vol_ma20", "vol_ma60",
    "ema9", "ema21", "ema50",
    "bb_upper", "bb_lower", "bb_mid",
    "obv", "obv_ma20",
}

def get_feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in _EXCLUDE]


# ═══════════════════════════════════════════════════════════
# 5. WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════
def walk_forward_validate(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_builder,
    n_splits: int = 5,
    gap: int = 5,
    verbose: bool = True,
) -> Dict:
    df_c = (
        df.dropna(subset=feature_cols + ["target"])
        .reset_index(drop=True)
    )
    X = df_c[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
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
        fold_scores.append({"fold": fold + 1, "accuracy": acc, "auc": auc, "n": len(te_idx)})

        all_preds.extend(preds.tolist())
        all_true.extend(y_te.tolist())
        all_probs.extend(probs.tolist())

        if verbose:
            logger.info(f"  Fold {fold+1}: acc={acc:.4f}  auc={auc:.4f}  n={len(te_idx)}")

    overall_acc = accuracy_score(all_true, all_preds)
    mean_auc = roc_auc_score(all_true, all_probs)

    return {
        "fold_scores": fold_scores,
        "overall_accuracy": overall_acc,
        "mean_auc": mean_auc,
    }


# ═══════════════════════════════════════════════════════════
# 6. STACKING ENSEMBLE (LGB + XGB → LR meta)
# ═══════════════════════════════════════════════════════════
def stacking_ensemble_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_splits: int = 5,
    gap: int = 5,
) -> Dict:
    df_c = (
        df.dropna(subset=feature_cols + ["target"])
        .reset_index(drop=True)
    )
    X = df_c[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df_c["target"]

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    lgb_params = dict(
        n_estimators=600, num_leaves=63, learning_rate=0.03,
        feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
        min_child_samples=15, reg_alpha=0.1, reg_lambda=1.0,
        is_unbalance=True, objective="binary", metric="auc",
        random_state=42, verbose=-1, n_jobs=-1,
    )
    xgb_params = dict(
        n_estimators=600, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="auc", use_label_encoder=False,
        random_state=42, n_jobs=-1,
    )

    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    all_true_all, all_preds_stack = [], []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        scaler = StandardScaler()
        X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
        X_te_s = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns)

        m_lgb = lgb.LGBMClassifier(**lgb_params)
        m_lgb.fit(X_tr_s, y_tr)
        oof_lgb[te_idx] = m_lgb.predict_proba(X_te_s)[:, 1]

        m_xgb = xgb.XGBClassifier(**xgb_params)
        m_xgb.fit(X_tr_s, y_tr)
        oof_xgb[te_idx] = m_xgb.predict_proba(X_te_s)[:, 1]

        all_true_all.extend(y_te.tolist())

    # Meta-learner on OOF predictions (only on non-zero oof slots)
    oof_mask = np.array(all_true_all)  # aligned already
    # rebuild aligned OOF arrays
    all_true2, oof_lgb2, oof_xgb2 = [], [], []
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        all_true2.extend(y.iloc[te_idx].tolist())
        oof_lgb2.extend(oof_lgb[te_idx].tolist())
        oof_xgb2.extend(oof_xgb[te_idx].tolist())

    # Ensemble: soft vote (average probabilities)
    oof_avg = (np.array(oof_lgb2) + np.array(oof_xgb2)) / 2
    preds_avg = (oof_avg > 0.5).astype(int)
    acc_ensemble = accuracy_score(all_true2, preds_avg)
    auc_ensemble = roc_auc_score(all_true2, oof_avg)

    # Meta LR
    meta_X = np.column_stack([oof_lgb2, oof_xgb2])
    meta_y = np.array(all_true2)
    meta = LogisticRegression(C=1.0, random_state=42)
    meta.fit(meta_X, meta_y)
    meta_preds = meta.predict(meta_X)
    acc_meta = accuracy_score(meta_y, meta_preds)

    logger.info(f"  Ensemble (avg) accuracy: {acc_ensemble:.4f}  AUC: {auc_ensemble:.4f}")
    logger.info(f"  Meta-LR accuracy (in-sample): {acc_meta:.4f}")

    return {
        "overall_accuracy": acc_ensemble,
        "mean_auc": auc_ensemble,
        "oof_lgb": oof_lgb2,
        "oof_xgb": oof_xgb2,
        "all_true": all_true2,
    }


# ═══════════════════════════════════════════════════════════
# 7. FULL PIPELINE
# ═══════════════════════════════════════════════════════════
def run_pipeline(
    stock_ids: List[str],
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-31",
    api_token: Optional[str] = None,
    target_acc: float = 0.75,
):
    results = {}
    for stock_id in stock_ids:
        logger.info(f"\n{'='*60}\nStock: {stock_id}\n{'='*60}")
        try:
            df = load_data(stock_id, start_date, end_date, api_token)
            df = add_technical_features(df)
            df = add_chip_features(df)
            df = build_target(df, threshold=0.0)

            feat_cols = get_feature_cols(df)
            df_clean = df.dropna(subset=feat_cols + ["target"])
            logger.info(f"[{stock_id}] {len(df_clean)} clean rows, {len(feat_cols)} features")

            if len(df_clean) < 500:
                logger.warning(f"[{stock_id}] Not enough data, skip")
                continue

            # LightGBM single
            logger.info(f"[{stock_id}] Walk-forward LightGBM …")
            lgbm_res = walk_forward_validate(
                df_clean, feat_cols,
                lambda: lgb.LGBMClassifier(
                    n_estimators=600, num_leaves=63, learning_rate=0.03,
                    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
                    min_child_samples=15, reg_alpha=0.1, reg_lambda=1.0,
                    is_unbalance=True, objective="binary", metric="auc",
                    random_state=42, verbose=-1, n_jobs=-1,
                ),
            )
            lgbm_acc = lgbm_res["overall_accuracy"]
            logger.info(f"[{stock_id}] LightGBM overall acc: {lgbm_acc:.4f}")

            # Stacking ensemble
            logger.info(f"[{stock_id}] Walk-forward Stacking Ensemble …")
            stack_res = stacking_ensemble_cv(df_clean, feat_cols)
            stack_acc = stack_res["overall_accuracy"]
            logger.info(f"[{stock_id}] Stack overall acc: {stack_acc:.4f}")

            best_acc = max(lgbm_acc, stack_acc)
            results[stock_id] = {
                "lgbm_acc": lgbm_acc,
                "stack_acc": stack_acc,
                "best_acc": best_acc,
                "df": df_clean,
                "feat_cols": feat_cols,
            }
            logger.info(f"[{stock_id}] Best accuracy: {best_acc:.4f}")

        except Exception as e:
            import traceback
            logger.error(f"[{stock_id}] Error: {e}")
            traceback.print_exc()

    return results


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    API_TOKEN = os.environ.get("FINMIND_TOKEN", "")
    TARGET_ACC = 0.75

    # Start with Taiwan's most liquid stocks
    STOCK_IDS = ["2330", "2317", "2454", "2412", "1301"]
    START, END = "2015-01-01", "2024-12-31"

    logger.info("=== Taiwan Stock Prediction System v1 ===")
    results = run_pipeline(STOCK_IDS, START, END, API_TOKEN or None, TARGET_ACC)

    logger.info("\n=== Summary ===")
    for sid, r in results.items():
        logger.info(f"  {sid}: LGBM={r['lgbm_acc']:.4f}  Stack={r['stack_acc']:.4f}  Best={r['best_acc']:.4f}")

    best_stock = max(results, key=lambda k: results[k]["best_acc"]) if results else None
    if best_stock:
        logger.info(f"\nBest stock: {best_stock} with acc={results[best_stock]['best_acc']:.4f}")
        if results[best_stock]["best_acc"] >= TARGET_ACC:
            logger.info("✓ Target accuracy reached!")
        else:
            logger.info("✗ Need improvement → will try advanced methods")
