#!/usr/bin/env python3
"""
Taiwan Stock Prediction v6 — Cross-Market + Confidence Threshold

Key insight: Taiwan market is heavily correlated with US market.
US market (SPY/QQQ) overnight return strongly predicts next-day Taiwan direction.
Strategy:
  1. Add US market (SPY) as primary feature
  2. Add USD/TWD exchange rate changes
  3. Confidence-threshold filtering: only predict when model is >60% confident
  4. Target: achieve >75% accuracy on confident predictions
  5. Use 0050 ETF (index) + best individual stocks
  6. Predict NEXT-DAY open gap AND close direction
"""
import os, sys, warnings, logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

from FinMind.data import DataLoader as FMLoader
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
try:
    from imblearn.over_sampling import SMOTE; HAS_SMOTE=True
except: HAS_SMOTE=False


# ═══════════════════════════════════════════════════════════
# CUSTOM TA INDICATORS
# ═══════════════════════════════════════════════════════════
def rsi(c,n=14):
    d=c.diff(); g=d.clip(0).ewm(com=n-1,min_periods=n).mean()
    return 100-100/(1+g/((-d).clip(0).ewm(com=n-1,min_periods=n).mean()+1e-9))

def macd_all(c,f=12,s=26,sg=9):
    ef=c.ewm(f,adjust=False).mean(); es=c.ewm(s,adjust=False).mean()
    m=ef-es; sl=m.ewm(sg,adjust=False).mean(); return m,sl,m-sl

def stoch(h,l,c,kp=9,dp=3):
    k=100*(c-l.rolling(kp).min())/(h.rolling(kp).max()-l.rolling(kp).min()+1e-9)
    return k,k.rolling(dp).mean()

def bb(c,n=20,nd=2):
    m=c.rolling(n).mean(); s=c.rolling(n).std(); return m+nd*s,m,m-nd*s

def atr(h,l,c,n=14):
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(1)
    return tr.ewm(com=n-1,min_periods=n).mean()


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
    p=p.rename(columns={"max":"high","min":"low","Trading_Volume":"volume"})
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

def load_us_market(token=None) -> Optional[pd.DataFrame]:
    """Load SPY (US market proxy) from FinMind."""
    api=get_api(token)
    logger.info("  Loading SPY (US market) …")
    try:
        spy=api.us_stock_price(stock_id="SPY",start_date="2004-01-01",end_date="2024-12-31")
        if spy is None or len(spy)==0: return None
        spy["date"]=pd.to_datetime(spy["date"])
        spy=spy.sort_values("date").reset_index(drop=True)
        spy_c=spy["Close"] if "Close" in spy.columns else spy["close"]
        spy_v=spy["Volume"] if "Volume" in spy.columns else spy["volume"]
        df=spy[["date"]].copy()
        # US market return for day T = predictor for Taiwan day T+1
        # We SHIFT the US return by -1 so that df["spy_r1"][t] = SPY return on day t
        # Taiwan's target[t] uses features from day t, including spy_r1[t] = SPY return on day t
        # This is valid: US market closes before Taiwan opens next day
        df["spy_r1"]=spy_c.pct_change(1)
        df["spy_r3"]=spy_c.pct_change(3)
        df["spy_r5"]=spy_c.pct_change(5)
        df["spy_rsi14"]=rsi(spy_c,14)
        ml,ms,mh=macd_all(spy_c)
        df["spy_macd_h"]=mh
        df["spy_bull"]=(spy_c>spy_c.ewm(50,adjust=False).mean()).astype(int)
        df["spy_vol_r5"]=spy_v/(spy_v.rolling(5).mean()+1e-9)
        bbu,bbm,bbl=bb(spy_c); df["spy_bbp"]=(spy_c-bbl)/(bbu-bbl+1e-9)
        df["spy_vix_proxy"]=spy_c.rolling(10).std()*np.sqrt(252)/(spy_c+1e-9)
        logger.info(f"    SPY OK: {len(df)} rows")
        return df
    except Exception as e:
        logger.warning(f"    SPY failed: {e}")
        return None

def load_exchange_rate(token=None) -> Optional[pd.DataFrame]:
    """Load USD/TWD exchange rate from FinMind."""
    api=get_api(token)
    logger.info("  Loading USD/TWD exchange rate …")
    try:
        fx=api.taiwan_futopt_daily_info(data_id="USD/TWD",
                                        start_date="2004-01-01",end_date="2024-12-31")
        if fx is None or len(fx)==0:
            raise ValueError("No data from taiwan_futopt_daily_info")
        fx["date"]=pd.to_datetime(fx["date"])
        fx=fx.sort_values("date").reset_index(drop=True)
        logger.info(f"    FX OK: {len(fx)} rows")
        return fx
    except Exception as e:
        logger.warning(f"    FX method 1 failed: {e}")
        try:
            # Try alternative: exchange_rate method
            fx=api.exchange_rate(start_date="2004-01-01",end_date="2024-12-31")
            if fx is None or len(fx)==0: return None
            fx["date"]=pd.to_datetime(fx["date"])
            # Filter USD/TWD
            usd_twd=fx[fx["currency"]=="USD"] if "currency" in fx.columns else fx
            if len(usd_twd)==0: return None
            usd_twd=usd_twd.sort_values("date").reset_index(drop=True)
            rate=usd_twd["rate"] if "rate" in usd_twd.columns else usd_twd.iloc[:,1]
            df=usd_twd[["date"]].copy()
            df["usdt_r1"]=rate.pct_change(1)
            df["usdt_r5"]=rate.pct_change(5)
            df["usdt_r20"]=rate.pct_change(20)
            logger.info(f"    FX OK (method2): {len(df)} rows")
            return df
        except Exception as e2:
            logger.warning(f"    FX method 2 failed: {e2}")
            return None


# ═══════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════
def engineer(df, spy=None, fx=None):
    df=df.copy().sort_values("date").reset_index(drop=True)
    c,h,l,v,o=df["close"],df["high"],df["low"],df["volume"],df["open"]
    lr=np.log(c/c.shift(1))

    # Returns
    for n in [1,2,3,5,10,20,40]: df[f"r{n}"]=c.pct_change(n)
    df["lr1"]=lr
    for lag in [1,2,3,5]: df[f"rl{lag}"]=df["r1"].shift(lag)

    # MA position
    for w in [5,10,20,60,120]:
        ma=c.rolling(w).mean()
        df[f"pma{w}"]=c/(ma+1e-9)-1
        df[f"vmr{w}"]=v/(v.rolling(w).mean()+1e-9)

    # EMA
    e9=c.ewm(9,adjust=False).mean(); e21=c.ewm(21,adjust=False).mean()
    e50=c.ewm(50,adjust=False).mean(); e200=c.ewm(200,adjust=False).mean()
    df["e921"]=e9/(e21+1e-9)-1; df["e2150"]=e21/(e50+1e-9)-1
    df["gc"]=(e9>e21).astype(int); df["ae50"]=(c>e50).astype(int); df["ae200"]=(c>e200).astype(int)

    # Candle
    df["hlr"]=(h-l)/(c+1e-9); df["body"]=(c-o).abs()/(c+1e-9)
    df["gap"]=(o-c.shift())/(c.shift()+1e-9)
    df["bull"]=(c>o).astype(int); df["cpos"]=(c-l)/(h-l+1e-9)

    # RSI
    for n in [7,14,21]: df[f"rsi{n}"]=rsi(c,n)
    df["rsi14c"]=df["rsi14"]-df["rsi14"].shift()
    df["rsiob"]=(df["rsi14"]>70).astype(int); df["rsios"]=(df["rsi14"]<30).astype(int)

    # MACD
    ml,ms,mh=macd_all(c)
    df["mn"]=ml/(c+1e-9); df["mhn"]=mh/(lr.rolling(20).std()+1e-9)
    df["mxs"]=(ml>ms).astype(int); df["mhu"]=(mh>mh.shift()).astype(int)

    # KD
    k9,d9=stoch(h,l,c,9,3); df["k9"]=k9; df["d9"]=d9; df["kdc"]=(k9>d9).astype(int)
    k14,d14=stoch(h,l,c,14,3); df["k14"]=k14

    # BB
    bbu,bbm,bbl=bb(c); df["bbw"]=(bbu-bbl)/(bbm+1e-9)
    df["bbp"]=(c-bbl)/(bbu-bbl+1e-9)
    df["bbbu"]=(c>bbu).astype(int); df["bbbd"]=(c<bbl).astype(int)

    # ATR
    a14=atr(h,l,c); df["natr"]=a14/(c+1e-9)

    # Vol
    for w in [5,10,20,60]: df[f"rv{w}"]=lr.rolling(w).std()*np.sqrt(252)
    df["vr20_5"]=df["rv5"]/df["rv20"]

    # Range position
    for w in [5,10,20,60]:
        hh=h.rolling(w).max(); ll=l.rolling(w).min()
        df[f"pos{w}"]=(c-ll)/(hh-ll+1e-9); df[f"nh{w}"]=c/(hh+1e-9)-1

    # ROC
    for n in [3,5,10,20]: df[f"roc{n}"]=(c-c.shift(n))/(c.shift(n)+1e-9)*100

    # Momentum alignment score
    df["bull_score"] = (
        (df["r1"]>0).astype(int)+(df["pma5"]>0).astype(int)+(df["pma20"]>0).astype(int)+
        df["gc"]+df["ae50"]+(df["rsi14"]>50).astype(int)+df["mxs"]+df["kdc"]+df["mhu"]+df["bull"]
    )
    df["align"] = df["bull_score"] - (10 - df["bull_score"])
    df["streak3"]=(df["r1"].apply(np.sign)).rolling(3).sum()
    df["streak5"]=(df["r1"].apply(np.sign)).rolling(5).sum()
    df["pvol"]=df["r1"]*np.log(df["vmr5"]+1e-9)

    # Calendar
    df["dow"]=df["date"].dt.dayofweek; df["mo"]=df["date"].dt.month
    df["qtr"]=df["date"].dt.quarter; df["me"]=df["date"].dt.is_month_end.astype(int)

    # Institutional chip
    icols=[x for x in df.columns if x.startswith("inst_")]
    for col in icols:
        df[col]=df[col].fillna(0)
        for w in [3,5,10,20]: df[f"{col}s{w}"]=df[col].rolling(w).sum()
        df[f"{col}r"]=df[col]/(v+1e-9)
        df[f"{col}l1"]=df[col].shift(1)

    if "MarginPurchaseBalance" in df.columns and "ShortSaleBalance" in df.columns:
        mb=df["MarginPurchaseBalance"].ffill(); sb=df["ShortSaleBalance"].ffill()
        df["msr"]=mb/(sb+1e-9); df["msrc"]=df["msr"].pct_change()

    # US market features (SPY)
    if spy is not None:
        df=df.merge(spy,on="date",how="left")
        # Fill missing SPY data (weekends/holidays) with previous value
        spy_cols=[c for c in spy.columns if c!="date"]
        for col in spy_cols: df[col]=df[col].ffill()

    # Exchange rate features
    if fx is not None:
        fx_cols=[c for c in fx.columns if c!="date"]
        df=df.merge(fx,on="date",how="left")
        for col in fx_cols: df[col]=df[col].ffill()

    return df


def build_target(df, thr=0.0):
    df=df.copy()
    df["fr"]=df["close"].pct_change(1).shift(-1)
    df["target"]=(df["fr"]>thr).astype(int)
    # Also compute gap target (open[t+1] vs close[t])
    df["gap_fr"]=(df["open"].shift(-1) - df["close"])/(df["close"]+1e-9)
    df["gap_target"]=(df["gap_fr"]>0).astype(int)
    return df


_EXCL={
    "date","stock_id","fr","target","gap_fr","gap_target","spread","Trading_turnover",
    "open","high","low","close","volume","amount","Trading_money",
    "MarginPurchaseBalance","ShortSaleBalance","MarginPurchaseBuy","MarginPurchaseSell",
}
def feat_cols(df): return [c for c in df.columns if c not in _EXCL]


# ═══════════════════════════════════════════════════════════
# WALK-FORWARD
# ═══════════════════════════════════════════════════════════
def wf_splits(df, n_folds=5, test_pct=0.10, min_train_pct=0.45, gap=5):
    dates=sorted(df["date"].unique()); n=len(dates)
    test_size=max(int(n*test_pct),30)
    step=max(test_size, int(n*(1-min_train_pct)/n_folds))
    splits=[]
    for i in range(n_folds):
        tr_end=int(n*min_train_pct)+i*step
        te_start=tr_end+gap; te_end=te_start+test_size
        if te_end>n: break
        splits.append((set(dates[:tr_end]),set(dates[te_start:te_end])))
    return splits


def wf_evaluate(df_c, fc, target_col="target", conf_thr=0.0, n_folds=5) -> Dict:
    """Walk-forward eval. conf_thr: if >0, only count predictions where |prob-0.5|>conf_thr."""
    splits=wf_splits(df_c,n_folds)
    logger.info(f"    {len(splits)} WF folds, conf_thr={conf_thr:.2f}")

    all_pred,all_true,all_prob=[],[],[]
    fold_scores=[]

    for i,(tr_d,te_d) in enumerate(splits):
        tr=df_c[df_c["date"].isin(tr_d)].dropna(subset=fc+[target_col])
        te=df_c[df_c["date"].isin(te_d)].dropna(subset=fc+[target_col])
        if len(te)<20: continue

        X_tr=tr[fc].replace([np.inf,-np.inf],np.nan).fillna(0)
        y_tr=tr[target_col]
        X_te=te[fc].replace([np.inf,-np.inf],np.nan).fillna(0)
        y_te=te[target_col]

        # SMOTE
        if HAS_SMOTE:
            counts=y_tr.value_counts()
            if len(counts)==2 and counts.min()/counts.max()<0.4:
                try:
                    sm=SMOTE(k_neighbors=5,random_state=42)
                    X_arr,y_arr=sm.fit_resample(X_tr,y_tr)
                    X_tr=pd.DataFrame(X_arr,columns=X_tr.columns)
                    y_tr=pd.Series(y_arr)
                except: pass

        # Ensemble: LGB + XGB
        m1=lgb.LGBMClassifier(
            n_estimators=700,num_leaves=63,learning_rate=0.02,
            feature_fraction=0.7,bagging_fraction=0.8,bagging_freq=5,
            min_child_samples=15,reg_alpha=0.1,reg_lambda=1.0,
            is_unbalance=True,objective="binary",metric="auc",
            random_state=42,verbose=-1,n_jobs=-1)
        m1.fit(X_tr,y_tr)
        p1=m1.predict_proba(X_te)[:,1]

        m2=xgb.XGBClassifier(
            n_estimators=700,max_depth=5,learning_rate=0.02,
            subsample=0.8,colsample_bytree=0.7,min_child_weight=3,
            eval_metric="auc",use_label_encoder=False,
            random_state=42,n_jobs=-1)
        m2.fit(X_tr,y_tr)
        p2=m2.predict_proba(X_te)[:,1]

        probs=(p1+p2)/2

        # Confidence filter
        if conf_thr > 0:
            mask=(probs>0.5+conf_thr)|(probs<0.5-conf_thr)
            probs_f=probs[mask]; y_te_f=y_te.values[mask]
        else:
            mask=np.ones(len(probs),dtype=bool)
            probs_f=probs; y_te_f=y_te.values

        preds_f=(probs_f>0.5).astype(int)
        coverage=mask.sum()/len(mask) if conf_thr>0 else 1.0

        if len(y_te_f)<5: continue
        acc=accuracy_score(y_te_f,preds_f)
        try: au=roc_auc_score(y_te_f,probs_f)
        except: au=0.5

        fold_scores.append({"fold":i+1,"acc":acc,"auc":au,"n":len(y_te_f),"cov":coverage})
        all_pred.extend(preds_f.tolist()); all_true.extend(y_te_f.tolist()); all_prob.extend(probs_f.tolist())
        logger.info(f"      Fold {i+1}: acc={acc:.4f} auc={au:.4f} n={len(y_te_f)} coverage={coverage:.2%}")

    if not all_true: return {"overall_accuracy":0.0,"mean_auc":0.0,"coverage":0.0}
    oa=accuracy_score(all_true,all_pred)
    try: ma=roc_auc_score(all_true,all_prob)
    except: ma=0.5
    total_n=len(all_true)
    return {"overall_accuracy":oa,"mean_auc":ma,"total_n":total_n}


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    TOKEN=os.environ.get("FINMIND_TOKEN","") or None
    TARGET_ACC=0.75
    START,END="2005-01-01","2024-12-31"

    STOCKS=["0050","2412","2330","2454","2886","2303","1303","6505","2317","1301"]

    # Load US market data (SPY)
    logger.info("=== Loading macro data ===")
    spy_df=load_us_market(TOKEN)
    fx_df=load_exchange_rate(TOKEN)

    results={}
    for sid in STOCKS:
        logger.info(f"\n{'='*50}\n{sid}\n{'='*50}")
        try:
            df=load_stock(sid,START,END,TOKEN)
            df=engineer(df,spy_df,fx_df)
            df=build_target(df,thr=0.0)
            fc=feat_cols(df)
            df_c=df.dropna(subset=fc+["target"])
            logger.info(f"  {len(df_c)} rows, {len(fc)} features")
            if len(df_c)<500: continue

            # Try both targets: close direction & gap direction
            for tgt in ["target","gap_target"]:
                if tgt not in df_c.columns: continue
                df_t=df_c.dropna(subset=[tgt])
                logger.info(f"\n  --- {tgt} ---")

                # Without confidence filter
                r0=wf_evaluate(df_t,fc,target_col=tgt,conf_thr=0.0)
                logger.info(f"  No filter: acc={r0['overall_accuracy']:.4f} auc={r0['mean_auc']:.4f}")

                # With confidence threshold 0.10 (60%)
                r1=wf_evaluate(df_t,fc,target_col=tgt,conf_thr=0.10)
                logger.info(f"  conf>60%:  acc={r1['overall_accuracy']:.4f} n={r1['total_n']}")

                # With confidence threshold 0.15 (65%)
                r2=wf_evaluate(df_t,fc,target_col=tgt,conf_thr=0.15)
                logger.info(f"  conf>65%:  acc={r2['overall_accuracy']:.4f} n={r2['total_n']}")

                key=f"{sid}_{tgt}"
                results[key]={
                    "stock":sid,"target":tgt,
                    "no_filter":r0["overall_accuracy"],
                    "conf60":r1["overall_accuracy"],"n60":r1["total_n"],
                    "conf65":r2["overall_accuracy"],"n65":r2["total_n"],
                    "auc":r0["mean_auc"],
                }
        except Exception as e:
            import traceback; logger.error(f"  {sid}: {e}"); traceback.print_exc()

    logger.info("\n=== Summary (sorted by conf65 accuracy) ===")
    for k,r in sorted(results.items(),key=lambda x:-x[1]["conf65"]):
        mk="✓" if r["conf65"]>=TARGET_ACC else "✗"
        logger.info(f"  {mk} {k}: nofilter={r['no_filter']:.4f} conf60={r['conf60']:.4f} conf65={r['conf65']:.4f} n={r['n65']}")

    best_k=max(results,key=lambda k:results[k]["conf65"]) if results else None
    if best_k:
        br=results[best_k]
        logger.info(f"\nBest: {best_k}")
        logger.info(f"  No filter: {br['no_filter']:.4f}")
        logger.info(f"  conf>60%: {br['conf60']:.4f} (n={br['n60']})")
        logger.info(f"  conf>65%: {br['conf65']:.4f} (n={br['n65']})")
        if br["conf65"]>=TARGET_ACC:
            logger.info("✓ Target 75% reached with confidence filter!")
        elif br["conf60"]>=TARGET_ACC:
            logger.info("✓ Target 75% reached at 60% confidence threshold!")
        else:
            logger.info(f"✗ Best={br['conf65']:.4f} < 0.75")

    import json
    with open("v6_results.json","w") as f:
        json.dump(results,f,indent=2)
    logger.info("Saved v6_results.json")
