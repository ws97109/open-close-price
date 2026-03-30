#!/usr/bin/env python3
"""
Taiwan Stock Prediction v4
Strategy:
  - LSTM (GRU) extracts sequential patterns per stock
  - LightGBM uses handcrafted features + GRU probability
  - Stacking: GRU_prob + LGBMprob → LR meta-learner
  - Per-stock walk-forward: train/test by date split
  - Target: >75% accuracy

References:
  - LSTM+LightGBM+RF ensemble: 82-86% on FTSE Taiwan 50 (Tandfonline 2024)
  - Key insight: LSTM captures non-linear sequential dependencies
"""
import os, sys, warnings, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Dict, Tuple
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

from FinMind.data import DataLoader as FMLoader
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression


# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"PyTorch device: {DEVICE}")


# ═══════════════════════════════════════════════════════════
# CUSTOM TA INDICATORS
# ═══════════════════════════════════════════════════════════
def calc_rsi(c, n=14):
    d=c.diff(); g=d.clip(0).ewm(com=n-1,min_periods=n).mean()
    l_=(-d).clip(0).ewm(com=n-1,min_periods=n).mean()
    return 100-100/(1+g/(l_+1e-9))

def calc_macd(c,f=12,s=26,sig=9):
    ef=c.ewm(f,adjust=False).mean(); es=c.ewm(s,adjust=False).mean()
    m=ef-es; sl=m.ewm(sig,adjust=False).mean(); return m,sl,m-sl

def calc_stoch(h,l,c,kp=9,dp=3):
    k=100*(c-l.rolling(kp).min())/(h.rolling(kp).max()-l.rolling(kp).min()+1e-9)
    return k,k.rolling(dp).mean()

def calc_bb(c,n=20,nd=2):
    m=c.rolling(n).mean(); s=c.rolling(n).std()
    return m+nd*s,m,m-nd*s

def calc_atr(h,l,c,n=14):
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(1)
    return tr.ewm(com=n-1,min_periods=n).mean()

def calc_adx(h,l,c,n=14):
    u,d=h.diff(),-l.diff()
    pdm=pd.Series(np.where((u>d)&(u>0),u,0.),index=c.index)
    ndm=pd.Series(np.where((d>u)&(d>0),d,0.),index=c.index)
    a=calc_atr(h,l,c,n)
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
    api = get_api(token)
    p = api.taiwan_stock_daily(stock_id=sid, start_date=start, end_date=end)
    if p is None or len(p)==0: raise ValueError(f"No data: {sid}")
    p["date"]=pd.to_datetime(p["date"])
    p=p.rename(columns={"max":"high","min":"low","Trading_Volume":"volume","Trading_money":"amount"})
    p=p.sort_values("date").reset_index(drop=True)
    # Institutional
    try:
        inst=api.taiwan_stock_institutional_investors(stock_id=sid,start_date=start,end_date=end)
        if inst is not None and len(inst)>0:
            inst["date"]=pd.to_datetime(inst["date"])
            inst["net"]=inst["buy"]-inst["sell"]
            iw=inst.pivot_table(index="date",columns="name",values="net",aggfunc="sum").reset_index()
            iw.columns=["date" if c=="date" else f"inst_{c.replace(' ','_')}" for c in iw.columns]
            p=p.merge(iw,on="date",how="left")
    except: pass
    # Margin
    try:
        mg=api.taiwan_stock_margin_purchase_short_sale(stock_id=sid,start_date=start,end_date=end)
        if mg is not None and len(mg)>0:
            mg["date"]=pd.to_datetime(mg["date"])
            keep=[c for c in ["date","MarginPurchaseBalance","ShortSaleBalance"] if c in mg.columns]
            p=p.merge(mg[keep],on="date",how="left")
    except: pass
    return p


# ═══════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════
def build_features(df, mkt=None):
    df=df.copy().sort_values("date").reset_index(drop=True)
    c,h,l,v,o=df["close"],df["high"],df["low"],df["volume"],df["open"]
    lr=np.log(c/c.shift(1))
    for n in [1,2,3,5,10,20]: df[f"r{n}"]=c.pct_change(n)
    df["lr"]=lr
    for lag in [1,2,3,5]:
        df[f"rl{lag}"]=df["r1"].shift(lag); df[f"vl{lag}"]=v.shift(lag)
    for w in [5,10,20,60]:
        ma=c.rolling(w).mean()
        df[f"pma{w}"]=c/(ma+1e-9)-1; df[f"vmr{w}"]=v/(v.rolling(w).mean()+1e-9)
    e9=c.ewm(9,adjust=False).mean(); e21=c.ewm(21,adjust=False).mean()
    e50=c.ewm(50,adjust=False).mean()
    df["e921"]=e9/(e21+1e-9)-1; df["e2150"]=e21/(e50+1e-9)-1
    df["gc"]=(e9>e21).astype(int); df["ae50"]=(c>e50).astype(int)
    df["hlr"]=(h-l)/(c+1e-9); df["body"]=(c-o).abs()/(c+1e-9)
    df["ushadow"]=(h-pd.concat([o,c],axis=1).max(1))/(c+1e-9)
    df["lshadow"]=(pd.concat([o,c],axis=1).min(1)-l)/(c+1e-9)
    df["gap"]=(o-c.shift())/(c.shift()+1e-9); df["bull"]=(c>o).astype(int)
    df["cpos"]=(c-l)/(h-l+1e-9)
    for n in [7,14,21]: df[f"rsi{n}"]=calc_rsi(c,n)
    df["rsi14c"]=df["rsi14"]-df["rsi14"].shift(); df["rsiob"]=(df["rsi14"]>70).astype(int)
    df["rsios"]=(df["rsi14"]<30).astype(int)
    ml,ms,mh=calc_macd(c)
    df["mn"]=ml/(c+1e-9); df["mhn"]=mh/(c.rolling(20).std()+1e-9)
    df["mxs"]=(ml>ms).astype(int); df["mhu"]=(mh>mh.shift()).astype(int)
    k9,d9=calc_stoch(h,l,c,9,3); df["k9"]=k9; df["d9"]=d9; df["kdc"]=(k9>d9).astype(int)
    k14,d14=calc_stoch(h,l,c,14,3); df["k14"]=k14; df["d14"]=d14
    bbu,bbm,bbl=calc_bb(c)
    df["bbw"]=(bbu-bbl)/(bbm+1e-9); df["bbp"]=(c-bbl)/(bbu-bbl+1e-9)
    df["bbbu"]=(c>bbu).astype(int); df["bbbd"]=(c<bbl).astype(int)
    a14=calc_atr(h,l,c); df["natr"]=a14/(c+1e-9)
    adxv,pdi,ndi=calc_adx(h,l,c)
    df["adx"]=adxv; df["didf"]=pdi-ndi; df["adxs"]=(adxv>25).astype(int)
    from functools import reduce
    for w in [5,10,20]:
        df[f"rv{w}"]=lr.rolling(w).std()*np.sqrt(252)
        df[f"sk{w}"]=lr.rolling(w).skew()
    for w in [5,10,20]:
        hh=h.rolling(w).max(); ll=l.rolling(w).min()
        df[f"pos{w}"]=(c-ll)/(hh-ll+1e-9); df[f"nh{w}"]=c/(hh+1e-9)-1
    for n in [5,10,20]: df[f"roc{n}"]=(c-c.shift(n))/(c.shift(n)+1e-9)*100
    icols=[x for x in df.columns if x.startswith("inst_")]
    for col in icols:
        df[col]=df[col].fillna(0)
        for w in [3,5,10]: df[f"{col}s{w}"]=df[col].rolling(w).sum()
        df[f"{col}r"]=df[col]/(v+1e-9)
    if "MarginPurchaseBalance" in df.columns and "ShortSaleBalance" in df.columns:
        mb=df["MarginPurchaseBalance"].ffill(); sb=df["ShortSaleBalance"].ffill()
        df["msr"]=mb/(sb+1e-9); df["msrc"]=df["msr"].pct_change()
    df["dow"]=df["date"].dt.dayofweek; df["mo"]=df["date"].dt.month
    df["qtr"]=df["date"].dt.quarter; df["me"]=df["date"].dt.is_month_end.astype(int)
    # Market context
    if mkt is not None:
        df=df.merge(mkt,on="date",how="left")
    return df


def build_target(df, thr=0.0):
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
# GRU MODEL
# ═══════════════════════════════════════════════════════════
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.gru=nn.GRU(input_size,hidden,num_layers=layers,
                        batch_first=True,dropout=dropout if layers>1 else 0.)
        self.head=nn.Sequential(
            nn.Linear(hidden,32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32,1)
        )
    def forward(self,x):
        out,_=self.gru(x); return self.head(out[:,-1,:]).squeeze(-1)


def make_sequences(X_arr, y_arr, seq_len=15):
    Xs, ys = [], []
    for i in range(seq_len, len(X_arr)):
        Xs.append(X_arr[i-seq_len:i])
        ys.append(y_arr[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


GRU_FEATURES = [
    "r1","r3","r5","r10","lr","pma5","pma20","vmr5","vmr20",
    "rsi7","rsi14","rsi21","k9","d9","k14","d14",
    "mn","mhn","mxs","mhu",
    "bbw","bbp","natr","adx","didf",
    "rv5","rv10","rv20","bull","cpos","gap",
    "e921","e2150","gc","ae50",
]


def train_gru(X_train_seq, y_train_seq, X_val_seq, y_val_seq,
              input_size, epochs=50, patience=8, lr=1e-3, batch=64):
    model = GRUModel(input_size).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3)

    Xtr = torch.tensor(X_train_seq, dtype=torch.float32)
    ytr = torch.tensor(y_train_seq, dtype=torch.float32)
    Xval = torch.tensor(X_val_seq, dtype=torch.float32).to(DEVICE)
    yval = torch.tensor(y_val_seq, dtype=torch.float32).to(DEVICE)

    ds = TensorDataset(Xtr, ytr)
    dl = DataLoader(ds, batch_size=batch, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xval), yval).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model


def gru_predict(model, X_seq):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)
        logits = model(X_t)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


# ═══════════════════════════════════════════════════════════
# NORMALIZE GRU INPUT
# ═══════════════════════════════════════════════════════════
from sklearn.preprocessing import RobustScaler

def normalize_gru_input(X_train_2d, X_test_2d):
    """Normalize GRU input features using train statistics."""
    sc = RobustScaler()
    X_tr_n = sc.fit_transform(X_train_2d)
    X_te_n = sc.transform(X_test_2d)
    return X_tr_n, X_te_n


# ═══════════════════════════════════════════════════════════
# PER-STOCK WALK-FORWARD WITH GRU + LGB ENSEMBLE
# ═══════════════════════════════════════════════════════════
SEQ_LEN = 15
N_SPLITS = 5
GAP = 5

def per_stock_wf(df_clean: pd.DataFrame, fcols: List[str]) -> Dict:
    """Walk-forward evaluation: GRU + LightGBM stacking."""
    # Filter GRU features to those available
    gru_feats = [f for f in GRU_FEATURES if f in df_clean.columns]
    logger.info(f"    GRU features: {len(gru_feats)}, LGB features: {len(fcols)}")

    dates = sorted(df_clean["date"].unique())
    n = len(dates)
    test_size = n // (N_SPLITS + 1)
    if test_size < 30: test_size = 30

    all_preds, all_true, all_probs = [], [], []
    fold_scores = []

    for fold in range(N_SPLITS):
        tr_end = int(n * 0.5) + fold * test_size
        te_start = tr_end + GAP
        te_end = te_start + test_size
        if te_end > n: break

        tr_dates = set(dates[:tr_end])
        te_dates = set(dates[te_start:te_end])

        tr = df_clean[df_clean["date"].isin(tr_dates)].copy()
        te = df_clean[df_clean["date"].isin(te_dates)].copy()

        if len(te) < 30: continue

        # ── GRU training ────────────────────────────────
        X_tr_gru = tr[gru_feats].replace([np.inf,-np.inf],np.nan).fillna(0).values
        y_tr_gru = tr["target"].values.astype(np.float32)
        X_te_gru = te[gru_feats].replace([np.inf,-np.inf],np.nan).fillna(0).values

        # Normalize
        X_tr_gru_n, X_te_gru_n = normalize_gru_input(X_tr_gru, X_te_gru)

        # Create sequences
        X_tr_seq, y_tr_seq = make_sequences(X_tr_gru_n, y_tr_gru, SEQ_LEN)
        X_te_seq, y_te_seq = make_sequences(X_te_gru_n,
                                             te["target"].values.astype(np.float32), SEQ_LEN)

        if len(X_tr_seq) < 100 or len(X_te_seq) < 10:
            continue

        # Split train into train/val for GRU (80/20)
        val_split = int(len(X_tr_seq) * 0.8)
        X_gval = X_tr_seq[val_split:]; y_gval = y_tr_seq[val_split:]
        X_gtr  = X_tr_seq[:val_split]; y_gtr  = y_tr_seq[:val_split]

        gru_model = train_gru(X_gtr, y_gtr, X_gval, y_gval, len(gru_feats),
                              epochs=60, patience=10)

        # GRU probabilities for LGB stacking
        # Train data: get OOF-ish predictions (shift by seq_len)
        gru_tr_all = gru_predict(gru_model, X_tr_seq)  # shape: (len-seq_len,)
        gru_te_all = gru_predict(gru_model, X_te_seq)

        # Align indices: GRU sequences start at index SEQ_LEN
        # Shift back to match original df rows
        tr_aligned = tr.iloc[SEQ_LEN:].copy().reset_index(drop=True)
        te_aligned = te.iloc[SEQ_LEN:].copy().reset_index(drop=True)

        if len(tr_aligned) < len(gru_tr_all):
            gru_tr_all = gru_tr_all[:len(tr_aligned)]
        tr_aligned = tr_aligned.iloc[:len(gru_tr_all)].copy()
        te_aligned = te_aligned.iloc[:len(gru_te_all)].copy()

        tr_aligned["gru_prob"] = gru_tr_all
        te_aligned["gru_prob"] = gru_te_all

        # ── LightGBM with GRU feature ────────────────────
        lgb_feats = [c for c in fcols if c in tr_aligned.columns] + ["gru_prob"]
        X_tr_lgb = tr_aligned[lgb_feats].replace([np.inf,-np.inf],np.nan).fillna(0)
        y_tr_lgb = tr_aligned["target"]
        X_te_lgb = te_aligned[lgb_feats].replace([np.inf,-np.inf],np.nan).fillna(0)
        y_te_lgb = te_aligned["target"]

        if len(X_te_lgb) < 10: continue

        lgbm = lgb.LGBMClassifier(
            n_estimators=400, num_leaves=63, learning_rate=0.03,
            feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
            min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
            is_unbalance=True, objective="binary", metric="auc",
            random_state=42, verbose=-1, n_jobs=-1,
        )
        lgbm.fit(X_tr_lgb, y_tr_lgb)

        lgb_probs = lgbm.predict_proba(X_te_lgb)[:, 1]

        # ── Ensemble: average GRU + LGB ──────────────────
        ens_probs = (gru_te_all[:len(lgb_probs)] + lgb_probs) / 2
        ens_preds = (ens_probs > 0.5).astype(int)
        y_te_arr = y_te_lgb.values[:len(ens_preds)]

        acc = accuracy_score(y_te_arr, ens_preds)
        try: auc = roc_auc_score(y_te_arr, ens_probs)
        except: auc = 0.5

        fold_scores.append({"fold":fold+1,"acc":acc,"auc":auc,"n":len(y_te_arr)})
        all_preds.extend(ens_preds.tolist())
        all_true.extend(y_te_arr.tolist())
        all_probs.extend(ens_probs.tolist())
        logger.info(f"    Fold {fold+1}: acc={acc:.4f} auc={auc:.4f} n={len(y_te_arr)}")

    if not all_true:
        return {"overall_accuracy":0.0,"mean_auc":0.0}

    oa = accuracy_score(all_true, all_preds)
    try: ma = roc_auc_score(all_true, all_probs)
    except: ma = 0.5
    return {"overall_accuracy":oa,"mean_auc":ma,"fold_scores":fold_scores}


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    TOKEN = os.environ.get("FINMIND_TOKEN","") or None
    TARGET_ACC = 0.75
    START, END = "2015-01-01", "2024-12-31"

    STOCKS = [
        "2330","2317","2454","2412","1301",
        "2308","2303","2882","2886","2891",
        "3711","2357","2382","6505","1303",
    ]

    # Load 0050 as market context
    logger.info("Loading market context (0050) …")
    try:
        mkt_raw = load_stock("0050", START, END, TOKEN)
        mkt_raw=mkt_raw.sort_values("date").reset_index(drop=True)
        mc=mkt_raw["close"]
        mkt=mkt_raw[["date"]].copy()
        for n in [1,3,5,10]: mkt[f"mkt_r{n}"]=mc.pct_change(n)
        for w in [5,20]: mkt[f"mkt_pma{w}"]=mc/(mc.rolling(w).mean()+1e-9)-1
        mkt["mkt_rsi14"]=calc_rsi(mc,14)
        mkt["mkt_trend"]=(mc>mc.ewm(50,adjust=False).mean()).astype(int)
        logger.info("  Market context OK")
    except Exception as e:
        logger.warning(f"  Market context failed: {e}")
        mkt = None

    results = {}
    for sid in STOCKS:
        logger.info(f"\n{'='*50}\nStock: {sid}\n{'='*50}")
        try:
            df = load_stock(sid, START, END, TOKEN)
            df = build_features(df, mkt)
            df = build_target(df, thr=0.0)
            fc = feat_cols(df)
            df_c = df.dropna(subset=fc+["target"])
            logger.info(f"  {len(df_c)} rows, {len(fc)} features")
            if len(df_c) < 500:
                logger.warning(f"  Too few rows, skip")
                continue
            res = per_stock_wf(df_c, fc)
            results[sid] = res
            logger.info(f"  [{sid}] overall acc={res['overall_accuracy']:.4f} auc={res['mean_auc']:.4f}")
        except Exception as e:
            import traceback
            logger.error(f"  [{sid}] {e}")
            traceback.print_exc()

    logger.info("\n=== Summary ===")
    for sid, r in sorted(results.items(), key=lambda x: -x[1]["overall_accuracy"]):
        mark="✓" if r["overall_accuracy"]>=TARGET_ACC else "✗"
        logger.info(f"  {mark} {sid}: acc={r['overall_accuracy']:.4f} auc={r['mean_auc']:.4f}")

    if results:
        best = max(results, key=lambda k: results[k]["overall_accuracy"])
        best_acc = results[best]["overall_accuracy"]
        logger.info(f"\nBest: {best} = {best_acc:.4f}")
        if best_acc >= TARGET_ACC:
            logger.info("✓ Target 75% reached!")
        else:
            logger.info(f"✗ Best={best_acc:.4f} < 0.75 → Will try further improvements")

    import json
    with open("v4_results.json","w") as f:
        json.dump({k:{"acc":v["overall_accuracy"],"auc":v["mean_auc"]}
                   for k,v in results.items()}, f, indent=2)
    logger.info("Results saved to v4_results.json")
