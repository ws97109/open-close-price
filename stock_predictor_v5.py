#!/usr/bin/env python3
"""
Taiwan Stock Prediction v5 — Research-Backed Approach

Key changes from v4:
1. Longer history (2005-2024 → 5000+ rows → 5 proper WF folds)
2. Percentile-based expanding-window WF (exactly 5 folds)
3. Triple-model ensemble: LightGBM + XGBoost + CatBoost
4. SMOTE upsampling on training fold (minority class)
5. Composite "momentum alignment" features (multiple indicators agree)
6. Foreign investor flow as primary predictor (Taiwan-specific signal)
7. Regime detection features
8. No StandardScaler (tree models don't need it)
"""
import os, sys, warnings, logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

from FinMind.data import DataLoader as FMLoader
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False


# ═══════════════════════════════════════════════════════════
# CUSTOM TA INDICATORS
# ═══════════════════════════════════════════════════════════
def rsi(c, n=14):
    d=c.diff(); g=d.clip(0).ewm(com=n-1,min_periods=n).mean()
    return 100-100/(1+g/((-d).clip(0).ewm(com=n-1,min_periods=n).mean()+1e-9))

def macd_all(c,f=12,s=26,sig=9):
    ef=c.ewm(f,adjust=False).mean(); es=c.ewm(s,adjust=False).mean()
    m=ef-es; sl=m.ewm(sig,adjust=False).mean(); return m,sl,m-sl

def stoch(h,l,c,kp=9,dp=3):
    k=100*(c-l.rolling(kp).min())/(h.rolling(kp).max()-l.rolling(kp).min()+1e-9)
    return k,k.rolling(dp).mean()

def bb(c,n=20,nd=2):
    m=c.rolling(n).mean(); s=c.rolling(n).std()
    return m+nd*s,m,m-nd*s

def atr(h,l,c,n=14):
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(1)
    return tr.ewm(com=n-1,min_periods=n).mean()

def adx_calc(h,l,c,n=14):
    u,d=h.diff(),-l.diff()
    pdm=pd.Series(np.where((u>d)&(u>0),u,0.),index=c.index)
    ndm=pd.Series(np.where((d>u)&(d>0),d,0.),index=c.index)
    a=atr(h,l,c,n)
    pdi=100*pdm.ewm(com=n-1,min_periods=n).mean()/(a+1e-9)
    ndi=100*ndm.ewm(com=n-1,min_periods=n).mean()/(a+1e-9)
    dx=100*(pdi-ndi).abs()/(pdi+ndi+1e-9)
    return dx.ewm(com=n-1,min_periods=n).mean(),pdi,ndi


# ═══════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════
_API = None
def get_api(token=None):
    global _API
    if _API is None:
        _API = FMLoader()
        if token: _API.login_by_token(api_token=token)
    return _API

def load_stock(sid, start, end, token=None):
    api=get_api(token)
    p=api.taiwan_stock_daily(stock_id=sid,start_date=start,end_date=end)
    if p is None or len(p)==0: raise ValueError(f"No data: {sid}")
    p["date"]=pd.to_datetime(p["date"])
    p=p.rename(columns={"max":"high","min":"low","Trading_Volume":"volume","Trading_money":"amount"})
    p=p.sort_values("date").reset_index(drop=True)
    try:
        inst=api.taiwan_stock_institutional_investors(stock_id=sid,start_date=start,end_date=end)
        if inst is not None and len(inst)>0:
            inst["date"]=pd.to_datetime(inst["date"]); inst["net"]=inst["buy"]-inst["sell"]
            iw=inst.pivot_table(index="date",columns="name",values="net",aggfunc="sum").reset_index()
            iw.columns=["date" if c=="date" else f"inst_{c.replace(' ','_')}" for c in iw.columns]
            p=p.merge(iw,on="date",how="left")
    except: pass
    try:
        mg=api.taiwan_stock_margin_purchase_short_sale(stock_id=sid,start_date=start,end_date=end)
        if mg is not None and len(mg)>0:
            mg["date"]=pd.to_datetime(mg["date"])
            keep=[c for c in ["date","MarginPurchaseBalance","ShortSaleBalance"] if c in mg.columns]
            p=p.merge(mg[keep],on="date",how="left")
    except: pass
    return p


# ═══════════════════════════════════════════════════════════
# COMPREHENSIVE FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════
def engineer(df, mkt=None):
    df=df.copy().sort_values("date").reset_index(drop=True)
    c,h,l,v,o=df["close"],df["high"],df["low"],df["volume"],df["open"]
    lr=np.log(c/c.shift(1))

    # ── Returns ──────────────────────────────────────────
    for n in [1,2,3,5,10,20,40]: df[f"r{n}"]=c.pct_change(n)
    df["lr1"]=lr

    # ── Lag returns ──────────────────────────────────────
    for lag in [1,2,3,5]: df[f"rl{lag}"]=df["r1"].shift(lag)

    # ── MA position ──────────────────────────────────────
    for w in [5,10,20,60,120]:
        ma=c.rolling(w).mean()
        df[f"pma{w}"]=c/(ma+1e-9)-1
        df[f"vmr{w}"]=v/(v.rolling(w).mean()+1e-9)

    # ── EMA ──────────────────────────────────────────────
    e9=c.ewm(9,adjust=False).mean(); e21=c.ewm(21,adjust=False).mean()
    e50=c.ewm(50,adjust=False).mean(); e200=c.ewm(200,adjust=False).mean()
    df["e921"]=e9/(e21+1e-9)-1; df["e2150"]=e21/(e50+1e-9)-1
    df["e50200"]=e50/(e200+1e-9)-1
    df["gc"]=(e9>e21).astype(int); df["ae50"]=(c>e50).astype(int)
    df["ae200"]=(c>e200).astype(int)

    # ── Candle ───────────────────────────────────────────
    df["hlr"]=(h-l)/(c+1e-9); df["body"]=(c-o).abs()/(c+1e-9)
    df["gap"]=(o-c.shift())/(c.shift()+1e-9); df["bull"]=(c>o).astype(int)
    df["cpos"]=(c-l)/(h-l+1e-9)

    # ── RSI ──────────────────────────────────────────────
    for n in [7,14,21]: df[f"rsi{n}"]=rsi(c,n)
    df["rsi14c"]=df["rsi14"]-df["rsi14"].shift()
    df["rsiob"]=(df["rsi14"]>70).astype(int); df["rsios"]=(df["rsi14"]<30).astype(int)

    # ── MACD ─────────────────────────────────────────────
    ml,ms,mh=macd_all(c); rv20=lr.rolling(20).std()+1e-9
    df["mn"]=ml/(c+1e-9); df["mhn"]=mh/rv20
    df["mxs"]=(ml>ms).astype(int); df["mhu"]=(mh>mh.shift()).astype(int)

    # ── KD ───────────────────────────────────────────────
    k9,d9=stoch(h,l,c,9,3); df["k9"]=k9; df["d9"]=d9; df["kdc"]=(k9>d9).astype(int)
    k14,d14=stoch(h,l,c,14,3); df["k14"]=k14; df["d14"]=d14

    # ── BB ───────────────────────────────────────────────
    bbu,bbm,bbl=bb(c)
    df["bbw"]=(bbu-bbl)/(bbm+1e-9); df["bbp"]=(c-bbl)/(bbu-bbl+1e-9)
    df["bbbu"]=(c>bbu).astype(int); df["bbbd"]=(c<bbl).astype(int)

    # ── ATR ──────────────────────────────────────────────
    a14=atr(h,l,c); df["natr"]=a14/(c+1e-9)

    # ── ADX ──────────────────────────────────────────────
    adxv,pdi,ndi=adx_calc(h,l,c)
    df["adx"]=adxv; df["didf"]=pdi-ndi; df["adxs"]=(adxv>25).astype(int)

    # ── Volatility ───────────────────────────────────────
    for w in [5,10,20,60]:
        df[f"rv{w}"]=lr.rolling(w).std()*np.sqrt(252)
        df[f"sk{w}"]=lr.rolling(w).skew()
    df["vr20_5"]=df["rv5"]/df["rv20"]  # vol regime

    # ── Range position ───────────────────────────────────
    for w in [5,10,20,60]:
        hh=h.rolling(w).max(); ll=l.rolling(w).min()
        df[f"pos{w}"]=(c-ll)/(hh-ll+1e-9)
        df[f"nh{w}"]=c/(hh+1e-9)-1; df[f"nl{w}"]=c/(ll+1e-9)-1

    # ── ROC ──────────────────────────────────────────────
    for n in [3,5,10,20]: df[f"roc{n}"]=(c-c.shift(n))/(c.shift(n)+1e-9)*100

    # ── Calendar ─────────────────────────────────────────
    df["dow"]=df["date"].dt.dayofweek; df["mo"]=df["date"].dt.month
    df["qtr"]=df["date"].dt.quarter
    df["me"]=df["date"].dt.is_month_end.astype(int)
    df["ms"]=df["date"].dt.is_month_start.astype(int)

    # ── Institutional chip features ───────────────────────
    icols=[x for x in df.columns if x.startswith("inst_")]
    for col in icols:
        df[col]=df[col].fillna(0)
        for w in [3,5,10,20]: df[f"{col}s{w}"]=df[col].rolling(w).sum()
        df[f"{col}r"]=df[col]/(v+1e-9)
        df[f"{col}l1"]=df[col].shift(1)
        df[f"{col}s5l1"]=df[f"{col}s5"].shift(1)

    # ── Margin / Short ────────────────────────────────────
    if "MarginPurchaseBalance" in df.columns and "ShortSaleBalance" in df.columns:
        mb=df["MarginPurchaseBalance"].ffill(); sb=df["ShortSaleBalance"].ffill()
        df["msr"]=mb/(sb+1e-9); df["msrc"]=df["msr"].pct_change()
        df["msrl1"]=df["msr"].shift(1)
        df["margin_chg"]=mb.pct_change(); df["short_chg"]=sb.pct_change()

    # ── Composite Momentum Alignment Scores ───────────────
    # Taiwan: multiple indicators agreeing → stronger signal
    # Bull alignment score: count how many bullish signals agree
    df["bull_score"] = (
        (df["r1"] > 0).astype(int) +
        (df["pma5"] > 0).astype(int) +
        (df["pma20"] > 0).astype(int) +
        df["gc"] + df["ae50"] +
        (df["rsi14"] > 50).astype(int) +
        df["mxs"] + df["kdc"] +
        df["mhu"] + df["bull"]
    )
    # Bear alignment score
    df["bear_score"] = (
        (df["r1"] < 0).astype(int) +
        (df["pma5"] < 0).astype(int) +
        (df["pma20"] < 0).astype(int) +
        (1-df["gc"]) + (1-df["ae50"]) +
        (df["rsi14"] < 50).astype(int) +
        (1-df["mxs"]) + (1-df["kdc"])
    )
    df["align_score"] = df["bull_score"] - df["bear_score"]
    df["strong_bull"] = (df["bull_score"] >= 8).astype(int)
    df["strong_bear"] = (df["bear_score"] >= 7).astype(int)

    # Momentum persistence
    df["up3"] = (df["r1"] > 0).astype(int).rolling(3).sum()
    df["dn3"] = (df["r1"] < 0).astype(int).rolling(3).sum()
    df["streak"] = df["r1"].apply(np.sign)
    df["streak_3"] = df["streak"].rolling(3).sum()
    df["streak_5"] = df["streak"].rolling(5).sum()

    # Volume spike on price move (accumulation signal)
    df["price_vol_align"] = df["r1"] * np.log(df["vmr5"] + 1e-9)

    # OBV-like
    df["obv_chg5"] = (np.sign(df["r1"].fillna(0)) * v).rolling(5).sum() / (v.rolling(20).mean()+1e-9)

    # Market context
    if mkt is not None:
        df=df.merge(mkt,on="date",how="left")

    return df


def target(df, thr=0.0):
    df=df.copy()
    df["fr"]=df["close"].pct_change(1).shift(-1)
    df["target"]=(df["fr"]>thr).astype(int)
    return df


_EXCL={
    "date","stock_id","fr","target","spread","Trading_turnover",
    "open","high","low","close","volume","amount",
    "MarginPurchaseBalance","ShortSaleBalance","MarginPurchaseBuy","MarginPurchaseSell",
}
def feat_cols(df): return [c for c in df.columns if c not in _EXCL]


# ═══════════════════════════════════════════════════════════
# PERCENTILE-BASED WALK-FORWARD (5 proper folds)
# ═══════════════════════════════════════════════════════════
def wf_splits(df: pd.DataFrame, n_folds=5, test_pct=0.10, min_train_pct=0.50, gap=5):
    """
    Expanding window WF using date percentiles.
    Fold i: train = 0 to (min_train_pct + i * fold_step)
             test  = train_end + gap to train_end + gap + test_size
    Always returns exactly n_folds valid splits.
    """
    dates = sorted(df["date"].unique())
    n = len(dates)
    test_size = max(int(n * test_pct), 30)
    # Distribute remaining space across folds
    available = n - int(n * min_train_pct) - gap - test_size * n_folds
    step = max(test_size, int(available / n_folds) if available > 0 else test_size)

    splits = []
    for i in range(n_folds):
        tr_end = int(n * min_train_pct) + i * step
        te_start = tr_end + gap
        te_end = te_start + test_size
        if te_end > n: break
        tr_d = set(dates[:tr_end])
        te_d = set(dates[te_start:te_end])
        splits.append((tr_d, te_d))
    return splits


# ═══════════════════════════════════════════════════════════
# SMOTE (apply only within training fold)
# ═══════════════════════════════════════════════════════════
def maybe_smote(X_tr, y_tr, random_state=42):
    if not HAS_SMOTE: return X_tr, y_tr
    counts = y_tr.value_counts()
    if len(counts) < 2: return X_tr, y_tr
    ratio = counts.min() / counts.max()
    if ratio > 0.4: return X_tr, y_tr  # already balanced enough
    try:
        sm = SMOTE(k_neighbors=5, random_state=random_state)
        X_r, y_r = sm.fit_resample(X_tr, y_tr)
        return pd.DataFrame(X_r, columns=X_tr.columns), pd.Series(y_r)
    except: return X_tr, y_tr


# ═══════════════════════════════════════════════════════════
# TRIPLE ENSEMBLE: LGB + XGB + CatBoost
# ═══════════════════════════════════════════════════════════
LGB_PARAMS = dict(
    n_estimators=700, num_leaves=63, learning_rate=0.02,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
    min_child_samples=15, reg_alpha=0.1, reg_lambda=1.0,
    is_unbalance=True, objective="binary", metric="auc",
    random_state=42, verbose=-1, n_jobs=-1,
)
XGB_PARAMS = dict(
    n_estimators=700, max_depth=5, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=1.0,
    eval_metric="auc", use_label_encoder=False,
    random_state=42, n_jobs=-1,
)
CB_PARAMS = dict(
    iterations=700, depth=6, learning_rate=0.02,
    l2_leaf_reg=3, bagging_temperature=0.8,
    auto_class_weights="Balanced",
    eval_metric="AUC", random_seed=42, verbose=0,
)


def triple_ensemble_fold(X_tr, y_tr, X_te, weights=(0.4, 0.3, 0.3)):
    """Train LGB + XGB + CatBoost on a single fold, return averaged probability."""
    X_tr_s, y_tr_s = maybe_smote(X_tr, y_tr)

    m1 = lgb.LGBMClassifier(**LGB_PARAMS)
    m1.fit(X_tr_s, y_tr_s)
    p1 = m1.predict_proba(X_te)[:, 1]

    m2 = xgb.XGBClassifier(**XGB_PARAMS)
    m2.fit(X_tr_s, y_tr_s)
    p2 = m2.predict_proba(X_te)[:, 1]

    m3 = CatBoostClassifier(**CB_PARAMS)
    m3.fit(X_tr_s.values, y_tr_s.values)
    p3 = m3.predict_proba(X_te.values)[:, 1]

    w1, w2, w3 = weights
    return w1*p1 + w2*p2 + w3*p3


# ═══════════════════════════════════════════════════════════
# WALK-FORWARD EVALUATION
# ═══════════════════════════════════════════════════════════
def wf_evaluate(df_c: pd.DataFrame, fc: List[str], n_folds=5, gap=5) -> Dict:
    splits = wf_splits(df_c, n_folds, test_pct=0.10, min_train_pct=0.45, gap=gap)
    logger.info(f"    {len(splits)} WF folds")

    all_pred, all_true, all_prob = [], [], []
    fold_scores = []

    for i, (tr_d, te_d) in enumerate(splits):
        tr = df_c[df_c["date"].isin(tr_d)].dropna(subset=fc+["target"])
        te = df_c[df_c["date"].isin(te_d)].dropna(subset=fc+["target"])
        if len(te) < 20: continue

        X_tr = tr[fc].replace([np.inf,-np.inf],np.nan).fillna(0)
        y_tr = tr["target"]
        X_te = te[fc].replace([np.inf,-np.inf],np.nan).fillna(0)
        y_te = te["target"]

        probs = triple_ensemble_fold(X_tr, y_tr, X_te)
        preds = (probs > 0.5).astype(int)

        acc = accuracy_score(y_te, preds)
        try: au = roc_auc_score(y_te, probs)
        except: au = 0.5

        fold_scores.append({"fold":i+1,"acc":acc,"auc":au,"n":len(y_te)})
        all_pred.extend(preds.tolist()); all_true.extend(y_te.tolist()); all_prob.extend(probs.tolist())
        logger.info(f"      Fold {i+1}: acc={acc:.4f} auc={au:.4f} n={len(y_te)}")

    if not all_true:
        return {"overall_accuracy":0.0,"mean_auc":0.0}

    oa = accuracy_score(all_true, all_pred)
    try: ma = roc_auc_score(all_true, all_prob)
    except: ma = 0.5
    return {"overall_accuracy":oa,"mean_auc":ma,"fold_scores":fold_scores}


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    TOKEN = os.environ.get("FINMIND_TOKEN","") or None
    TARGET_ACC = 0.75
    # Longer history: 2005-2024 gives ~5000 rows
    START, END = "2005-01-01", "2024-12-31"

    # All stocks to test
    STOCKS = [
        "2330","2317","2454","2412","1301","2308","2303",
        "2882","2886","2891","3711","2357","2382","6505","1303",
    ]

    # Load market context
    logger.info("Loading 0050 market context …")
    mkt = None
    try:
        api = get_api(TOKEN)
        mkt_raw = api.taiwan_stock_daily(stock_id="0050", start_date=START, end_date=END)
        mkt_raw["date"]=pd.to_datetime(mkt_raw["date"])
        mkt_raw=mkt_raw.rename(columns={"max":"high","min":"low","Trading_Volume":"volume"})
        mkt_raw=mkt_raw.sort_values("date").reset_index(drop=True)
        mc=mkt_raw["close"]
        mkt=mkt_raw[["date"]].copy()
        for n in [1,3,5,10,20]: mkt[f"mktr{n}"]=mc.pct_change(n)
        for w in [5,20,60]: mkt[f"mktpma{w}"]=mc/(mc.rolling(w).mean()+1e-9)-1
        mkt["mktrsi14"]=rsi(mc,14)
        ml,ms,mh=macd_all(mc); mkt["mktmh"]=mh
        mkt["mkttrend"]=(mc>mc.ewm(50,adjust=False).mean()).astype(int)
        mkt["mktbull"]=(mc>mc.ewm(200,adjust=False).mean()).astype(int)
        mkt["mktvmr5"]=mkt_raw["volume"]/(mkt_raw["volume"].rolling(5).mean()+1e-9)
        logger.info(f"  OK ({len(mkt)} rows)")
    except Exception as e:
        logger.warning(f"  Market context failed: {e}")

    results = {}
    for sid in STOCKS:
        logger.info(f"\n{'='*50}\nStock: {sid}\n{'='*50}")
        try:
            df = load_stock(sid, START, END, TOKEN)
            df = engineer(df, mkt)
            df = target(df, thr=0.0)
            fc = feat_cols(df)
            df_c = df.dropna(subset=fc+["target"])
            logger.info(f"  {len(df_c)} rows, {len(fc)} features")
            if len(df_c) < 500: continue
            res = wf_evaluate(df_c, fc)
            results[sid] = res
            logger.info(f"  [{sid}] acc={res['overall_accuracy']:.4f} auc={res['mean_auc']:.4f}")
        except Exception as e:
            import traceback; logger.error(f"  [{sid}] {e}"); traceback.print_exc()

    logger.info("\n=== Summary ===")
    for sid, r in sorted(results.items(), key=lambda x: -x[1]["overall_accuracy"]):
        mk="✓" if r["overall_accuracy"]>=TARGET_ACC else "✗"
        logger.info(f"  {mk} {sid}: acc={r['overall_accuracy']:.4f} auc={r['mean_auc']:.4f}")

    if results:
        best=max(results,key=lambda k:results[k]["overall_accuracy"])
        best_acc=results[best]["overall_accuracy"]
        logger.info(f"\nBest: {best} = {best_acc:.4f}")

    import json
    with open("v5_results.json","w") as f:
        json.dump({k:{"acc":v["overall_accuracy"],"auc":v["mean_auc"]}
                   for k,v in results.items()},f,indent=2)
    logger.info("Saved v5_results.json")
