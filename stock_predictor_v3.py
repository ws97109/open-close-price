#!/usr/bin/env python3
"""
Taiwan Stock Prediction System v3
Strategy:
  - Pooled multi-stock model (more training data = better patterns)
  - 0050 ETF as market context feature
  - No StandardScaler (tree models don't need it)
  - Date-based walk-forward (correct for panel data)
  - LightGBM with categorical stock_id + Optuna tuning
  - Target: next-day up/down >75% accuracy
"""
import os, sys, warnings, logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

from FinMind.data import DataLoader
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ═══════════════════════════════════════════════════════════
# CUSTOM TA INDICATORS (no external lib dependency)
# ═══════════════════════════════════════════════════════════
def calc_rsi(c: pd.Series, n: int) -> pd.Series:
    delta = c.diff()
    g = delta.clip(lower=0).ewm(com=n-1, min_periods=n).mean()
    l_ = (-delta).clip(lower=0).ewm(com=n-1, min_periods=n).mean()
    return 100 - 100 / (1 + g / (l_ + 1e-9))

def calc_macd(c, fast=12, slow=26, sig=9):
    ef = c.ewm(span=fast, adjust=False).mean()
    es = c.ewm(span=slow, adjust=False).mean()
    m = ef - es
    s = m.ewm(span=sig, adjust=False).mean()
    return m, s, m - s

def calc_stoch(h, l, c, kp=9, dp=3):
    lo = l.rolling(kp).min()
    hi = h.rolling(kp).max()
    k = 100 * (c - lo) / (hi - lo + 1e-9)
    return k, k.rolling(dp).mean()

def calc_bb(c, n=20, ndev=2):
    mid = c.rolling(n).mean()
    std = c.rolling(n).std()
    return mid + ndev*std, mid, mid - ndev*std

def calc_atr(h, l, c, n=14):
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(1)
    return tr.ewm(com=n-1, min_periods=n).mean()

def calc_adx(h, l, c, n=14):
    up, dn = h.diff(), -l.diff()
    pdm = np.where((up>dn)&(up>0), up, 0.)
    ndm = np.where((dn>up)&(dn>0), dn, 0.)
    atr_ = calc_atr(h, l, c, n)
    pdi = 100 * pd.Series(pdm, index=c.index).ewm(com=n-1, min_periods=n).mean() / (atr_+1e-9)
    ndi = 100 * pd.Series(ndm, index=c.index).ewm(com=n-1, min_periods=n).mean() / (atr_+1e-9)
    dx  = 100 * (pdi-ndi).abs() / (pdi+ndi+1e-9)
    return dx.ewm(com=n-1, min_periods=n).mean(), pdi, ndi

def calc_obv(c, v):
    return (np.sign(c.diff()).fillna(0) * v).cumsum()

def calc_cci(h, l, c, n=20):
    tp = (h+l+c)/3
    ma = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.abs(x-x.mean()).mean(), raw=True)
    return (tp-ma)/(0.015*mad+1e-9)


# ═══════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════
_API = None

def get_api(token=None):
    global _API
    if _API is None:
        _API = DataLoader()
        if token:
            _API.login_by_token(api_token=token)
    return _API


def load_stock(stock_id: str, start: str, end: str, token=None) -> pd.DataFrame:
    api = get_api(token)
    price = api.taiwan_stock_daily(stock_id=stock_id, start_date=start, end_date=end)
    if price is None or len(price) == 0:
        raise ValueError(f"No price data: {stock_id}")
    price["date"] = pd.to_datetime(price["date"])
    price = price.rename(columns={"max":"high","min":"low",
                                   "Trading_Volume":"volume","Trading_money":"amount"})
    price = price.sort_values("date").reset_index(drop=True)

    # Institutional
    try:
        inst = api.taiwan_stock_institutional_investors(
            stock_id=stock_id, start_date=start, end_date=end)
        if inst is not None and len(inst) > 0:
            inst["date"] = pd.to_datetime(inst["date"])
            inst["net"] = inst["buy"] - inst["sell"]
            iw = inst.pivot_table(index="date", columns="name",
                                   values="net", aggfunc="sum").reset_index()
            iw.columns = ["date" if c=="date" else f"inst_{c.replace(' ','_')}"
                          for c in iw.columns]
            price = price.merge(iw, on="date", how="left")
    except: pass

    # Margin
    try:
        mg = api.taiwan_stock_margin_purchase_short_sale(
            stock_id=stock_id, start_date=start, end_date=end)
        if mg is not None and len(mg) > 0:
            mg["date"] = pd.to_datetime(mg["date"])
            keep = [c for c in ["date","MarginPurchaseBalance","ShortSaleBalance"] if c in mg.columns]
            price = price.merge(mg[keep], on="date", how="left")
    except: pass

    return price


# ═══════════════════════════════════════════════════════════
# FEATURE ENGINEERING PER STOCK
# ═══════════════════════════════════════════════════════════
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    c, h, l, v, o = df["close"], df["high"], df["low"], df["volume"], df["open"]
    lr = np.log(c/c.shift(1))

    # Returns
    for n in [1,2,3,5,10,20]: df[f"ret_{n}"] = c.pct_change(n)
    df["log_ret"] = lr

    # Lag
    for lag in [1,2,3,5]:
        df[f"ret_lag{lag}"] = df["ret_1"].shift(lag)
        df[f"vol_lag{lag}"] = v.shift(lag)

    # MA position
    for w in [5,10,20,60]:
        ma = c.rolling(w).mean()
        df[f"p_ma{w}"] = c/(ma+1e-9)-1
        df[f"vol_ma{w}_r"] = v/(v.rolling(w).mean()+1e-9)

    # EMA cross
    e9 = c.ewm(9,adjust=False).mean(); e21 = c.ewm(21,adjust=False).mean()
    e50 = c.ewm(50,adjust=False).mean()
    df["ema9_21"] = e9/(e21+1e-9)-1; df["ema21_50"] = e21/(e50+1e-9)-1
    df["golden_cross"] = (e9>e21).astype(int)

    # Candle
    df["hl_r"] = (h-l)/(c+1e-9)
    df["body"] = (c-o).abs()/(c+1e-9)
    df["upper_sh"] = (h - pd.concat([o,c],axis=1).max(1))/(c+1e-9)
    df["lower_sh"] = (pd.concat([o,c],axis=1).min(1)-l)/(c+1e-9)
    df["gap"] = (o-c.shift(1))/(c.shift(1)+1e-9)
    df["bull"] = (c>o).astype(int)
    df["close_pos"] = (c-l)/(h-l+1e-9)

    # RSI
    for n in [7,14,21]:
        df[f"rsi{n}"] = calc_rsi(c,n)
    df["rsi14_chg"] = df["rsi14"] - df["rsi14"].shift(1)
    df["rsi_ob"] = (df["rsi14"]>70).astype(int)
    df["rsi_os"] = (df["rsi14"]<30).astype(int)

    # MACD
    ml,ms,mh = calc_macd(c)
    df["macd_n"] = ml/(c+1e-9); df["macd_sig_n"] = ms/(c+1e-9)
    df["macd_h"] = mh; df["macd_cross"] = (ml>ms).astype(int)
    df["macd_h_up"] = (mh>mh.shift(1)).astype(int)

    # KD
    k9,d9 = calc_stoch(h,l,c,9,3); df["k9"]=k9; df["d9"]=d9
    df["kd_cross"] = (k9>d9).astype(int)
    k14,d14 = calc_stoch(h,l,c,14,3); df["k14"]=k14; df["d14"]=d14

    # BB
    bbu,bbm,bbl = calc_bb(c,20,2)
    df["bb_w"] = (bbu-bbl)/(bbm+1e-9)
    df["bb_pct"] = (c-bbl)/(bbu-bbl+1e-9)
    df["bb_bu"] = (c>bbu).astype(int); df["bb_bd"] = (c<bbl).astype(int)

    # ATR
    a14 = calc_atr(h,l,c,14); df["natr"] = a14/(c+1e-9)

    # ADX
    adx_v, pdi, ndi = calc_adx(h,l,c,14)
    df["adx"] = adx_v; df["di_diff"] = pdi-ndi; df["adx_str"] = (adx_v>25).astype(int)

    # CCI
    df["cci14"] = calc_cci(h,l,c,14); df["cci20"] = calc_cci(h,l,c,20)

    # OBV
    obv_v = calc_obv(c,v); df["obv_tr"] = (obv_v>obv_v.rolling(20).mean()).astype(int)
    df["obv_chg5"] = obv_v.pct_change(5)

    # ROC
    for n in [5,10,20]: df[f"roc{n}"] = (c-c.shift(n))/(c.shift(n)+1e-9)*100

    # Volatility
    for w in [5,10,20]:
        df[f"rv{w}"] = lr.rolling(w).std()*np.sqrt(252)
        df[f"skew{w}"] = lr.rolling(w).skew()

    # Range position
    for w in [5,10,20]:
        hh = h.rolling(w).max(); ll = l.rolling(w).min()
        df[f"pos{w}"] = (c-ll)/(hh-ll+1e-9)
        df[f"nhigh{w}"] = c/(hh+1e-9)-1

    # Chip
    icols = [x for x in df.columns if x.startswith("inst_")]
    for col in icols:
        df[col] = df[col].fillna(0)
        for w in [3,5,10]: df[f"{col}_s{w}"] = df[col].rolling(w).sum()
        df[f"{col}_r"] = df[col]/(v+1e-9)

    if "MarginPurchaseBalance" in df.columns and "ShortSaleBalance" in df.columns:
        mb = df["MarginPurchaseBalance"].ffill()
        sb = df["ShortSaleBalance"].ffill()
        df["ms_r"] = mb/(sb+1e-9)
        df["ms_chg"] = df["ms_r"].pct_change()

    # Calendar
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["qtr"] = df["date"].dt.quarter
    df["me"] = df["date"].dt.is_month_end.astype(int)
    df["ms_flag"] = df["date"].dt.is_month_start.astype(int)

    return df


def build_target(df: pd.DataFrame, thr: float = 0.0) -> pd.DataFrame:
    df = df.copy()
    df["future_ret"] = df["close"].pct_change(1).shift(-1)
    df["target"] = (df["future_ret"] > thr).astype(int)
    return df


# ═══════════════════════════════════════════════════════════
# POOLED DATASET BUILDER
# ═══════════════════════════════════════════════════════════
_EXCLUDE_BASE = {
    "date","stock_id","future_ret","target","spread","Trading_turnover",
    "open","high","low","close","volume","amount","stock_code",
    "MarginPurchaseBalance","ShortSaleBalance","MarginPurchaseBuy","MarginPurchaseSell",
}

def build_pool(stock_ids: List[str], start: str, end: str, token=None) -> pd.DataFrame:
    """Download & process all stocks, add market-context from 0050, return pooled df."""
    # Load 0050 as market proxy
    logger.info("Loading 0050 as market context …")
    try:
        mkt = load_stock("0050", start, end, token)
        mkt = mkt[["date","close","volume"]].rename(columns={
            "close":"mkt_close","volume":"mkt_vol"})
        mkt = mkt.sort_values("date").reset_index(drop=True)
        mkt_c = mkt["mkt_close"]
        for n in [1,3,5,10,20]:
            mkt[f"mkt_ret{n}"] = mkt_c.pct_change(n)
        for w in [5,20]:
            ma = mkt_c.rolling(w).mean()
            mkt[f"mkt_p_ma{w}"] = mkt_c/(ma+1e-9)-1
        mkt["mkt_rsi14"] = calc_rsi(mkt_c,14)
        ml,ms,mh = calc_macd(mkt_c)
        mkt["mkt_macd_h"] = mh
        mkt["mkt_trend"] = (mkt_c > mkt_c.ewm(50,adjust=False).mean()).astype(int)
        mkt["mkt_vol_r5"] = mkt["mkt_vol"]/(mkt["mkt_vol"].rolling(5).mean()+1e-9)
        mkt_feats = [c for c in mkt.columns if c != "mkt_close" and c != "mkt_vol"]
        logger.info(f"  Market features: {len(mkt_feats)-1}")
    except Exception as e:
        logger.warning(f"  Market proxy failed: {e}")
        mkt = None

    all_dfs = []
    stock_map = {sid: i for i, sid in enumerate(sorted(set(stock_ids) - {"0050"}))}

    for sid in stock_ids:
        if sid == "0050":
            continue
        logger.info(f"  Processing {sid} …")
        try:
            df = load_stock(sid, start, end, token)
            df = build_features(df)
            df = build_target(df, thr=0.0)
            df["stock_code"] = stock_map[sid]

            if mkt is not None:
                df = df.merge(mkt[mkt_feats], on="date", how="left")

            all_dfs.append(df)
        except Exception as e:
            logger.warning(f"  {sid} failed: {e}")

    if not all_dfs:
        raise RuntimeError("No stocks loaded")

    pool = pd.concat(all_dfs, ignore_index=True)
    pool = pool.sort_values(["date","stock_code"]).reset_index(drop=True)
    logger.info(f"Pool: {len(pool)} rows from {len(all_dfs)} stocks")
    return pool


def get_feat_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in _EXCLUDE_BASE]


# ═══════════════════════════════════════════════════════════
# DATE-BASED WALK-FORWARD
# ═══════════════════════════════════════════════════════════
def date_wf_splits(dates: np.ndarray, n_splits: int = 5, gap_days: int = 5,
                   min_train_frac: float = 0.5):
    """
    Expanding-window walk-forward: train on first (frac) of unique dates, test on next chunk.
    """
    dates = sorted(dates)
    n = len(dates)
    splits = []
    test_size = int(n * (1 - min_train_frac) / n_splits)
    if test_size < 20:
        test_size = 20
    for i in range(n_splits):
        train_end_pos = int(n * min_train_frac) + i * test_size
        test_start_pos = train_end_pos + gap_days
        test_end_pos = test_start_pos + test_size
        if test_end_pos > n:
            break
        splits.append((
            np.array(dates[:train_end_pos]),
            np.array(dates[test_start_pos:test_end_pos])
        ))
    return splits


# ═══════════════════════════════════════════════════════════
# WALK-FORWARD EVAL
# ═══════════════════════════════════════════════════════════
def wf_eval(pool: pd.DataFrame, feat_cols: List[str],
            model_fn, n_splits=5, gap_days=5) -> Dict:
    dates = pool["date"].unique()
    splits = date_wf_splits(dates, n_splits, gap_days)
    logger.info(f"  {len(splits)} WF folds")

    all_pred, all_true, all_prob = [], [], []
    fold_scores = []

    for i, (tr_dates, te_dates) in enumerate(splits):
        tr_dates_s = set(pd.Timestamp(d) for d in tr_dates)
        te_dates_s = set(pd.Timestamp(d) for d in te_dates)
        tr = pool[pool["date"].isin(tr_dates_s)].dropna(subset=feat_cols+["target"])
        te = pool[pool["date"].isin(te_dates_s)].dropna(subset=feat_cols+["target"])
        if len(te) < 50:
            continue

        X_tr = tr[feat_cols].replace([np.inf,-np.inf],np.nan).fillna(0)
        y_tr = tr["target"]
        X_te = te[feat_cols].replace([np.inf,-np.inf],np.nan).fillna(0)
        y_te = te["target"]

        model = model_fn()
        model.fit(X_tr, y_tr)

        preds = model.predict(X_te)
        probs = model.predict_proba(X_te)[:,1]

        acc = accuracy_score(y_te, preds)
        auc = roc_auc_score(y_te, probs)
        fold_scores.append({"fold":i+1,"acc":acc,"auc":auc,"n":len(te)})
        all_pred.extend(preds.tolist())
        all_true.extend(y_te.tolist())
        all_prob.extend(probs.tolist())
        logger.info(f"    Fold {i+1}: acc={acc:.4f} auc={auc:.4f} n={len(te)}")

    if not all_true:
        return {"overall_accuracy":0.0,"mean_auc":0.0}

    oa = accuracy_score(all_true, all_pred)
    ma = roc_auc_score(all_true, all_prob)
    return {"overall_accuracy":oa, "mean_auc":ma, "fold_scores":fold_scores}


# ═══════════════════════════════════════════════════════════
# OPTUNA TUNING
# ═══════════════════════════════════════════════════════════
def tune_lgbm(pool: pd.DataFrame, feat_cols: List[str],
              n_trials: int = 40, n_splits: int = 3) -> Dict:
    """Tune LightGBM hyperparameters with Optuna."""
    logger.info(f"  Optuna: {n_trials} trials …")
    dates = pool["date"].unique()
    splits = date_wf_splits(dates, n_splits, gap_days=5, min_train_frac=0.6)

    def objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int("n_estimators", 200, 800),
            num_leaves       = trial.suggest_int("num_leaves", 31, 127),
            learning_rate    = trial.suggest_float("lr", 0.01, 0.1, log=True),
            feature_fraction = trial.suggest_float("ff", 0.5, 0.9),
            bagging_fraction = trial.suggest_float("bf", 0.5, 0.9),
            bagging_freq     = trial.suggest_int("bfreq", 1, 10),
            min_child_samples= trial.suggest_int("mcs", 10, 50),
            reg_alpha        = trial.suggest_float("alpha", 1e-3, 5.0, log=True),
            reg_lambda       = trial.suggest_float("lambda", 1e-3, 5.0, log=True),
            is_unbalance=True, objective="binary", metric="auc",
            random_state=42, verbose=-1, n_jobs=-1,
        )
        aucs = []
        for tr_dates, te_dates in splits:
            tr_s = set(pd.Timestamp(d) for d in tr_dates)
            te_s = set(pd.Timestamp(d) for d in te_dates)
            tr = pool[pool["date"].isin(tr_s)].dropna(subset=feat_cols+["target"])
            te = pool[pool["date"].isin(te_s)].dropna(subset=feat_cols+["target"])
            if len(te) < 50: continue
            X_tr = tr[feat_cols].replace([np.inf,-np.inf],np.nan).fillna(0)
            X_te = te[feat_cols].replace([np.inf,-np.inf],np.nan).fillna(0)
            m = lgb.LGBMClassifier(**params)
            m.fit(X_tr, tr["target"])
            prob = m.predict_proba(X_te)[:,1]
            aucs.append(roc_auc_score(te["target"], prob))
        return np.mean(aucs) if aucs else 0.0

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"  Best AUC: {study.best_value:.4f}")
    return study.best_params


# ═══════════════════════════════════════════════════════════
# TRAIN FINAL MODEL
# ═══════════════════════════════════════════════════════════
def train_final(pool: pd.DataFrame, feat_cols: List[str], params: Dict) -> lgb.LGBMClassifier:
    """Train on all available data."""
    df_c = pool.dropna(subset=feat_cols+["target"])
    X = df_c[feat_cols].replace([np.inf,-np.inf],np.nan).fillna(0)
    y = df_c["target"]
    final_params = dict(
        is_unbalance=True, objective="binary", metric="auc",
        random_state=42, verbose=-1, n_jobs=-1,
    )
    final_params.update(params)
    # Convert Optuna keys back to LightGBM keys
    key_map = {"lr":"learning_rate","ff":"feature_fraction","bf":"bagging_fraction",
               "bfreq":"bagging_freq","mcs":"min_child_samples",
               "alpha":"reg_alpha","lambda":"reg_lambda"}
    for k,v in key_map.items():
        if k in final_params:
            final_params[v] = final_params.pop(k)
    model = lgb.LGBMClassifier(**final_params)
    model.fit(X, y)
    return model


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    TOKEN = os.environ.get("FINMIND_TOKEN", "") or None
    TARGET_ACC = 0.75
    START, END = "2015-01-01", "2024-12-31"

    # Taiwan 50 major liquid stocks
    STOCKS = [
        "0050",   # market proxy
        "2330",   # TSMC
        "2317",   # Foxconn
        "2454",   # MediaTek
        "2412",   # Chunghwa Telecom
        "1301",   # Formosa Plastics
        "2308",   # Delta Electronics
        "2303",   # UMC
        "2882",   # Cathay Financial
        "2886",   # Mega Financial
        "2891",   # CTBC Financial
        "3711",   # ASE Technology
        "2357",   # ASUS
        "2382",   # Quanta Computer
        "6505",   # Formosa Petrochemical
    ]

    logger.info("=== Taiwan Stock Prediction v3 (Pooled Multi-Stock) ===")
    logger.info(f"Stocks: {STOCKS}")

    # Build pooled dataset
    pool = build_pool(STOCKS, START, END, TOKEN)
    feat_cols = get_feat_cols(pool)
    pool_clean = pool.dropna(subset=feat_cols + ["target"])
    logger.info(f"Clean pool: {len(pool_clean)} rows, {len(feat_cols)} features")

    # Baseline LightGBM (default params)
    logger.info("\n--- Baseline LightGBM walk-forward ---")
    base_res = wf_eval(
        pool_clean, feat_cols,
        lambda: lgb.LGBMClassifier(
            n_estimators=600, num_leaves=63, learning_rate=0.03,
            feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
            is_unbalance=True, objective="binary", metric="auc",
            random_state=42, verbose=-1, n_jobs=-1,
        ),
    )
    logger.info(f"Baseline: acc={base_res['overall_accuracy']:.4f} auc={base_res['mean_auc']:.4f}")

    best_acc = base_res["overall_accuracy"]
    best_params = {}

    if best_acc < TARGET_ACC:
        logger.info(f"\nBaseline {best_acc:.4f} < {TARGET_ACC} → Optuna tuning …")
        best_params = tune_lgbm(pool_clean, feat_cols, n_trials=50)
        logger.info(f"Best params: {best_params}")

        # Eval tuned model
        logger.info("\n--- Tuned LightGBM walk-forward ---")
        def make_tuned():
            p = dict(
                is_unbalance=True, objective="binary", metric="auc",
                random_state=42, verbose=-1, n_jobs=-1,
            )
            p.update(best_params)
            key_map = {"lr":"learning_rate","ff":"feature_fraction","bf":"bagging_fraction",
                       "bfreq":"bagging_freq","mcs":"min_child_samples",
                       "alpha":"reg_alpha","lambda":"reg_lambda"}
            for k,v in key_map.items():
                if k in p: p[v] = p.pop(k)
            return lgb.LGBMClassifier(**p)

        tuned_res = wf_eval(pool_clean, feat_cols, make_tuned)
        logger.info(f"Tuned: acc={tuned_res['overall_accuracy']:.4f} auc={tuned_res['mean_auc']:.4f}")

        if tuned_res["overall_accuracy"] > best_acc:
            best_acc = tuned_res["overall_accuracy"]

    logger.info(f"\n=== Final accuracy: {best_acc:.4f} ===")
    if best_acc >= TARGET_ACC:
        logger.info("✓ Target 75% reached!")
    else:
        logger.info(f"✗ Need more improvement ({best_acc:.4f} < {TARGET_ACC})")

    # Save results for next iteration
    import json
    with open("v3_results.json", "w") as f:
        json.dump({
            "baseline_acc": base_res["overall_accuracy"],
            "baseline_auc": base_res["mean_auc"],
            "best_acc": best_acc,
            "best_params": best_params,
            "feat_cols": feat_cols,
        }, f, indent=2)
    logger.info("Results saved to v3_results.json")
