#!/usr/bin/env python3
"""
FastAPI backend — Taiwan Stock Dual Prediction Web System v2
Provides:
  GET  /api/stocks          → known stock list
  POST /api/predict         → gap + close predictions + price info
  POST /api/analyze         → Ollama qwen3:14b LLM analysis (streams)
  GET  /api/health          → health check
"""
import os, sys, asyncio, logging
from datetime import date, datetime
from typing import Optional, List, Dict
import numpy as np
import pandas as pd

# Resolve st/ directory (server.py lives at st/web/backend/)
_here = os.path.dirname(os.path.abspath(__file__))
_st   = os.path.dirname(os.path.dirname(_here))
sys.path.insert(0, _st)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests as _requests

from finalmodel import (
    load_stock, load_spy, load_sector, engineer, build_targets, feat_cols,
    _train_fold, _predict_proba, GAP_CONF_THR, CLOSE_CONF_THR, TRAIN_START,
)
import sentiment as _sentiment

logger = logging.getLogger(__name__)

# ─── Stock name mapping ────────────────────────────────────────
STOCK_NAMES: Dict[str, str] = {
    "0050": "元大台灣50",   "0056": "元大高股息",
    "2412": "中華電信",     "2330": "台積電",
    "2454": "聯發科",       "2886": "兆豐金控",
    "2303": "聯電",         "1303": "南亞",
    "6505": "台塑石化",     "2317": "鴻海",
    "1301": "台塑",         "2367": "燿華電子",
    "2002": "中鋼",         "1216": "統一",
    "2308": "台達電",       "2382": "廣達",
    "3711": "日月光投控",   "2912": "統一超",
    "2881": "富邦金",       "2882": "國泰金",
    "2891": "中信金",       "2884": "玉山金",
    "2885": "元大金",       "2892": "第一金",
    "3008": "大立光",       "2395": "研華",
}

FEATURE_LABELS: Dict[str, str] = {
    "spy_r1": "美股昨日報酬", "spy_r3": "美股3日報酬", "spy_r5": "美股5日報酬",
    "spy_rsi14": "美股 RSI(14)", "spy_macd_h": "美股 MACD 柱", "spy_bull": "美股多頭趨勢",
    "spy_bbp": "美股布林位置", "spy_vix_proxy": "美股波動率", "spy_vol_r5": "美股相對成交量",
    "spy_r10": "美股10日報酬", "spy_bull200": "美股在EMA200上", "spy_streak3": "美股3日動向",
    "r1": "昨日報酬率", "r2": "2日報酬率", "r3": "3日報酬率",
    "r5": "5日報酬率", "r10": "10日報酬率", "r20": "20日報酬率",
    "rsi7": "RSI(7)", "rsi14": "RSI(14)", "rsi21": "RSI(21)", "rsi28": "RSI(28)",
    "k9": "KD K值(9)", "d9": "KD D值(9)", "k14": "KD K值(14)", "kdc": "KD 多頭交叉",
    "pma5": "相對 MA5", "pma10": "相對 MA10", "pma20": "相對 MA20",
    "pma60": "相對 MA60", "pma120": "相對 MA120",
    "bbp": "布林通道位置", "bbw": "布林帶寬", "bbbu": "突破布林上軌", "bbbd": "跌破布林下軌",
    "natr": "ATR 波動度", "rv5": "5日實現波動", "rv20": "20日實現波動",
    "bull_score": "多頭綜合分", "align": "趨勢一致性",
    "streak3": "3日連漲跌", "streak5": "5日連漲跌",
    "gap": "今日開盤跳空", "cpos": "收盤位置", "body": "K棒實體", "hlr": "高低幅度",
    "bull": "陽線", "mxs": "MACD 多頭", "mhu": "MACD 柱遞增",
    "gc": "均線黃金交叉", "ae50": "在EMA50之上", "ae200": "在EMA200之上",
    "pvol": "量價合力", "vmr5": "相對5日均量", "vmr20": "相對20日均量",
    "msr": "融資融券比", "msrc": "融資融券比變化",
    "mkt_r1": "大盤昨日報酬", "mkt_bull": "大盤多頭", "rel_mkt1": "相對大盤報酬",
    "dow": "星期幾", "mo": "月份", "me": "月底效應", "wom": "月中週次",
    "vbull": "量增收漲", "vbear": "量增收跌",
}

OLLAMA_HOST  = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b")

# ─── App ──────────────────────────────────────────────────────
app = FastAPI(title="台灣股票預測系統 API v2", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache: Dict[str, dict] = {}   # {f"{sid}_{date}": prediction_result}


class PredictRequest(BaseModel):
    stock_id: str
    token: Optional[str] = None

class AnalyzeRequest(BaseModel):
    stock_id: str
    stock_name: str = ""
    prediction: Optional[dict] = None   # forward the prediction result for context
    token: Optional[str] = None


# ─── Helpers ──────────────────────────────────────────────────
def _norm_sid(sid: str) -> str:
    sid = sid.strip()
    if sid.isdigit() and len(sid) < 4:
        sid = sid.zfill(4)
    return sid


def _safe(row: pd.Series, col: str) -> Optional[float]:
    try:
        v = row.get(col)
        return None if pd.isna(v) else float(v)
    except Exception:
        return None


# ─── Endpoints ────────────────────────────────────────────────
@app.get("/api/stocks")
def list_stocks():
    return [{"id": k, "name": v} for k, v in STOCK_NAMES.items()]


@app.post("/api/predict")
def predict_stock(req: PredictRequest):
    """
    Full prediction pipeline:
     1. Load FinMind data (OHLCV + institutional + margin + SPY + 0050)
     2. Feature engineering (~172 features)
     3. Triple ensemble LGB+XGB+CatBoost for gap + close targets
     4. Fetch news + BERT sentiment, adjust probability
     5. Return prediction + price info + technicals + feature importances + sentiment
    """
    sid        = _norm_sid(req.stock_id)
    today_str  = date.today().strftime("%Y-%m-%d")
    cache_key  = f"{sid}_{today_str}"
    if cache_key in _cache:
        return _cache[cache_key]

    token = req.token or os.environ.get("FINMIND_TOKEN") or None

    try:
        # ── Load and engineer ──────────────────────────────────
        df_raw  = load_stock(sid, TRAIN_START, today_str, token)
        spy_df  = load_spy(token)
        sec_df  = load_sector(token)
        df      = engineer(df_raw, spy_df, sec_df)
        df      = build_targets(df)
        fc      = feat_cols(df)
        df_c    = df.dropna(subset=fc + ["gap_target", "target"])
        if len(df_c) < 200:
            raise ValueError(f"資料不足（{len(df_c)} 筆）")

        # ── Train models ───────────────────────────────────────
        X_all   = df_c[fc].replace([np.inf, -np.inf], np.nan).fillna(0)
        m_gap   = _train_fold(X_all, df_c["gap_target"])
        m_close = _train_fold(X_all, df_c["target"])

        # ── Predict on last row ────────────────────────────────
        last_row = df.dropna(subset=fc).iloc[[-1]]
        X_last   = last_row[fc].replace([np.inf, -np.inf], np.nan).fillna(0)
        as_of    = str(last_row["date"].iloc[0].date())

        prob_gap   = float(_predict_proba(*m_gap,   X_last)[0])
        prob_close = float(_predict_proba(*m_close, X_last)[0])

        # ── Sentiment ──────────────────────────────────────────
        sname   = STOCK_NAMES.get(sid, sid)
        sent    = _sentiment.analyze(sid, sname, n_results=8, use_bert=True)
        adj     = sent["adjustment"]
        prob_gap_adj   = float(np.clip(prob_gap   + adj, 0.01, 0.99))
        prob_close_adj = float(np.clip(prob_close + adj, 0.01, 0.99))

        conf_gap   = abs(prob_gap_adj   - 0.5)
        conf_close = abs(prob_close_adj - 0.5)

        def _signal(prob, thr):
            c = abs(prob - 0.5)
            if c <= thr:
                return "NO_SIGNAL"
            return "UP" if prob > 0.5 else "DOWN"

        # ── Current prices ─────────────────────────────────────
        last = df.iloc[-1]
        price = {
            "date":  str(last["date"].date()),
            "open":  float(last["open"])  if "open"  in df.columns else None,
            "close": float(last["close"]) if "close" in df.columns else None,
            "high":  float(last["high"])  if "high"  in df.columns else None,
            "low":   float(last["low"])   if "low"   in df.columns else None,
        }

        # ── Technical snapshot ─────────────────────────────────
        lrow = last_row.iloc[0]
        technical = {k: _safe(lrow, k) for k in [
            "rsi14", "k9", "d9", "kdc", "bull_score",
            "pma20", "pma60", "bbp", "spy_r1",
            "streak3", "r1", "r5", "natr", "gc", "mxs", "vmr5",
            "mkt_r1", "rel_mkt1", "msr",
        ]}

        # ── Feature importance ─────────────────────────────────
        imp_vals = m_gap[0].feature_importances_.tolist()
        max_imp  = max(imp_vals) if imp_vals else 1
        top_feats = sorted(
            [{"name": f, "label": FEATURE_LABELS.get(f, f),
              "importance": float(v), "rel": round(float(v) / max_imp, 4)}
             for f, v in zip(fc, imp_vals)],
            key=lambda x: -x["importance"]
        )[:15]

        result = {
            "stock":       sid,
            "stock_name":  sname,
            "as_of_date":  as_of,
            "price":       price,
            "gap": {
                "signal":      _signal(prob_gap_adj, GAP_CONF_THR),
                "probability": round(prob_gap_adj, 4),
                "confidence":  round(conf_gap * 2, 4),
                "raw_prob":    round(prob_gap, 4),
                "accuracy_note": "歷史驗證準確率 85%+ (conf>70%)",
            },
            "close": {
                "signal":      _signal(prob_close_adj, CLOSE_CONF_THR),
                "probability": round(prob_close_adj, 4),
                "confidence":  round(conf_close * 2, 4),
                "raw_prob":    round(prob_close, 4),
                "accuracy_note": "歷史驗證準確率 ~72% (conf>72%)",
            },
            "sentiment":   sent,
            "technical":   technical,
            "top_features": top_feats,
            "rows":        int(len(df_c)),
            "features":    int(len(fc)),
        }

        _cache[cache_key] = result
        return result

    except Exception as exc:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/analyze")
async def analyze_stock(req: AnalyzeRequest):
    """
    Stream Ollama qwen3:14b analysis.
    Fetches fresh news → builds a rich prompt → streams the response.
    """
    sid   = _norm_sid(req.stock_id)
    sname = req.stock_name or STOCK_NAMES.get(sid, sid)

    # Fetch news for LLM context
    articles = _sentiment.fetch_news(sid, sname, n_results=10)
    news_block = "\n".join(
        f"  [{i+1}] {a['title']}" for i, a in enumerate(articles[:8])
    ) if articles else "  （未能取得最新新聞）"

    # Build prediction context
    pred = req.prediction or {}
    gap_info, close_info, tech, price = (
        pred.get("gap", {}), pred.get("close", {}),
        pred.get("technical", {}), pred.get("price", {}),
    )

    def _fmt(v, suffix="", d=2):
        return f"{v:.{d}f}{suffix}" if isinstance(v, float) else "N/A"

    prompt = f"""你是一位專業的台灣股市分析師。請根據以下資訊對 {sid} {sname} 進行全面分析，並給出明日操作建議。

## 股票基本資訊
- 股票代碼：{sid}（{sname}）
- 資料日期：{price.get('date', 'N/A')}
- 今日開盤：{_fmt(price.get('open'))} 元
- 今日收盤：{_fmt(price.get('close'))} 元
- 今日高點：{_fmt(price.get('high'))} 元
- 今日低點：{_fmt(price.get('low'))} 元

## AI 模型預測（機器學習，walk-forward驗證）
### 開盤跳空方向（歷史準確率 85%+）
- 訊號：{gap_info.get('signal', 'N/A')}
- 機率：{_fmt(gap_info.get('probability', 0)*100, '%', 1)}（跳漲機率）
- 信心度：{_fmt(gap_info.get('confidence', 0)*100, '%', 1)}

### 收盤漲跌方向（歷史準確率 ~72%）
- 訊號：{close_info.get('signal', 'N/A')}
- 機率：{_fmt(close_info.get('probability', 0)*100, '%', 1)}（上漲機率）
- 信心度：{_fmt(close_info.get('confidence', 0)*100, '%', 1)}

## 主要技術指標
- RSI(14)：{_fmt(tech.get('rsi14'))}
- KD K值：{_fmt(tech.get('k9'))}，D值：{_fmt(tech.get('d9'))}，KD多頭：{'是' if (tech.get('kdc') or 0) > 0.5 else '否'}
- 相對MA20：{_fmt((tech.get('pma20') or 0)*100, '%', 2)}
- 多頭綜合分：{_fmt(tech.get('bull_score'), '/10', 0)}
- 美股(SPY)昨日：{_fmt((tech.get('spy_r1') or 0)*100, '%', 2)}
- 大盤(0050)昨日：{_fmt((tech.get('mkt_r1') or 0)*100, '%', 2)}
- 3日動能：{_fmt(tech.get('streak3'), '', 0)}

## 最新相關新聞
{news_block}

---
請進行以下分析（回答請用繁體中文）：

1. **技術面分析**：根據指標，目前趨勢如何？
2. **籌碼面觀察**：美股/大盤對本股的影響？
3. **新聞面解讀**：最新新聞對股價有何影響？
4. **綜合判斷**：整合AI預測與基本面，明日開盤/收盤方向為何？
5. **操作建議**：建議的進出場策略（注意：僅供參考，非投資建議）

請保持客觀，並在必要時說明不確定性。"""

    async def stream_ollama():
        try:
            with _requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
                stream=True,
                timeout=120,
            ) as resp:
                import json
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        if chunk:
                            yield chunk
                        if data.get("done"):
                            break
                    except Exception:
                        continue
        except Exception as e:
            yield f"\n\n⚠ Ollama 連線失敗：{e}"

    return StreamingResponse(stream_ollama(), media_type="text/plain; charset=utf-8")


@app.get("/api/chart/{stock_id}")
def chart_data(stock_id: str, days: int = 120, token: str = None):
    """
    Return OHLCV candlestick data for the last `days` trading days,
    plus walk-forward model probability series and tomorrow's prediction band.
    Used by the frontend TradingView Lightweight Charts.
    """
    sid       = _norm_sid(stock_id)
    today_str = date.today().strftime("%Y-%m-%d")
    env_token = token or os.environ.get("FINMIND_TOKEN") or None

    try:
        df_raw = load_stock(sid, TRAIN_START, today_str, env_token)
        spy_df = load_spy(env_token)
        sec_df = load_sector(env_token)
        df     = engineer(df_raw, spy_df, sec_df)
        df     = build_targets(df)
        fc     = feat_cols(df)
        df_c   = df.dropna(subset=fc + ["gap_target", "target"])

        # ── Candlestick data (last `days` rows) ───────────────
        tail = df.dropna(subset=["open", "high", "low", "close"]).tail(days)
        candles = [
            {
                "time":  int(row["date"].timestamp()),
                "open":  round(float(row["open"]),  2),
                "high":  round(float(row["high"]),  2),
                "low":   round(float(row["low"]),   2),
                "close": round(float(row["close"]), 2),
            }
            for _, row in tail.iterrows()
        ]

        # ── Volume data ────────────────────────────────────────
        volumes = [
            {
                "time":  int(row["date"].timestamp()),
                "value": int(row["volume"]) if not pd.isna(row["volume"]) else 0,
                "color": "rgba(63,185,80,0.4)" if row["close"] >= row["open"] else "rgba(248,81,73,0.4)",
            }
            for _, row in tail.iterrows()
        ]

        # ── Walk-forward model probability series ─────────────
        from finalmodel import wf_splits
        splits = wf_splits(df_c, n_folds=5)
        gap_prob_map:   dict = {}
        close_prob_map: dict = {}
        for tr_d, te_d in splits:
            tr = df_c[df_c["date"].isin(tr_d)].dropna(subset=fc + ["gap_target", "target"])
            te = df_c[df_c["date"].isin(te_d)].dropna(subset=fc + ["gap_target", "target"])
            if len(tr) < 100 or len(te) < 5:
                continue
            X_tr = tr[fc].replace([np.inf, -np.inf], np.nan).fillna(0)
            X_te = te[fc].replace([np.inf, -np.inf], np.nan).fillna(0)
            mg = _train_fold(X_tr, tr["gap_target"])
            mc = _train_fold(X_tr, tr["target"])
            pg = _predict_proba(*mg, X_te)
            pc = _predict_proba(*mc, X_te)
            for ts_val, g, c in zip(te["date"], pg, pc):
                t = int(pd.Timestamp(ts_val).timestamp())
                gap_prob_map[t]   = round(float(g), 4)
                close_prob_map[t] = round(float(c), 4)

        gap_probs   = [{"time": t, "value": v} for t, v in sorted(gap_prob_map.items())]
        close_probs = [{"time": t, "value": v} for t, v in sorted(close_prob_map.items())]

        # Filter to last `days` range
        if candles:
            t_min = candles[0]["time"]
            gap_probs   = [x for x in gap_probs   if x["time"] >= t_min]
            close_probs = [x for x in close_probs if x["time"] >= t_min]

        # ── Tomorrow's prediction band ─────────────────────────
        last_close = float(df.dropna(subset=["close"]).iloc[-1]["close"])
        last_time  = int(df.dropna(subset=["close"]).iloc[-1]["date"].timestamp())
        next_time  = last_time + 86400        # approximate next trading day

        # Get cached or compute fresh probabilities
        today_key = f"{sid}_{date.today().strftime('%Y-%m-%d')}"
        cached = _cache.get(today_key)
        if cached:
            prob_gap   = cached["gap"]["probability"]
            prob_close = cached["close"]["probability"]
        else:
            # Quick re-predict
            X_all = df_c[fc].replace([np.inf, -np.inf], np.nan).fillna(0)
            mg = _train_fold(X_all, df_c["gap_target"])
            mc = _train_fold(X_all, df_c["target"])
            last_row = df.dropna(subset=fc).iloc[[-1]]
            X_last   = last_row[fc].replace([np.inf, -np.inf], np.nan).fillna(0)
            prob_gap   = float(_predict_proba(*mg, X_last)[0])
            prob_close = float(_predict_proba(*mc, X_last)[0])

        conf_gap   = abs(prob_gap   - 0.5)
        conf_close = abs(prob_close - 0.5)

        # Predicted price level (naive: last_close ± ATR-scaled probability)
        atr_est = last_close * 0.01   # 1% as default ATR estimate
        try:
            atr_series = df.dropna(subset=["natr"])
            if len(atr_series):
                atr_est = float(atr_series.iloc[-1]["natr"]) * last_close
        except Exception:
            pass

        gap_target_price   = round(last_close + (prob_gap   - 0.5) * 4 * atr_est, 2)
        close_target_price = round(last_close + (prob_close - 0.5) * 4 * atr_est, 2)

        prediction_point = {
            "last_time":          last_time,
            "next_time":          next_time,
            "last_close":         round(last_close, 2),
            "gap_target_price":   gap_target_price,
            "close_target_price": close_target_price,
            "prob_gap":           round(prob_gap,   4),
            "prob_close":         round(prob_close, 4),
            "conf_gap":           round(conf_gap   * 2, 4),
            "conf_close":         round(conf_close * 2, 4),
            "gap_signal":   "UP" if (prob_gap   > 0.5 and conf_gap   > GAP_CONF_THR)
                            else ("DOWN" if (prob_gap   < 0.5 and conf_gap   > GAP_CONF_THR) else "NO_SIGNAL"),
            "close_signal": "UP" if (prob_close > 0.5 and conf_close > CLOSE_CONF_THR)
                            else ("DOWN" if (prob_close < 0.5 and conf_close > CLOSE_CONF_THR) else "NO_SIGNAL"),
        }

        # ── Moving averages ────────────────────────────────────
        def _ma_series(col: str) -> list:
            sub = tail.dropna(subset=[col])
            return [{"time": int(r["date"].timestamp()), "value": round(float(r[col]), 2)}
                    for _, r in sub.iterrows()]

        ma_lines = {
            "ma20":  _ma_series("pma20"),   # actually pma20 is relative; recompute
        }
        # Compute actual MA values
        price_df = df.dropna(subset=["close"]).tail(days + 60)
        for w in [20, 60]:
            col = f"_ma{w}"
            price_df[col] = price_df["close"].rolling(w).mean()
        tail_idx = set(tail.index)
        ma_lines = {}
        for w in [20, 60, 120]:
            col = f"_ma{w}"
            if col not in price_df.columns:
                price_df[col] = price_df["close"].rolling(w).mean()
            sub = price_df[price_df.index.isin(tail_idx)].dropna(subset=[col])
            ma_lines[f"ma{w}"] = [
                {"time": int(r["date"].timestamp()), "value": round(float(r[col]), 2)}
                for _, r in sub.iterrows()
            ]

        return {
            "stock_id":         sid,
            "candles":          candles,
            "volumes":          volumes,
            "ma_lines":         ma_lines,
            "gap_probs":        gap_probs,
            "close_probs":      close_probs,
            "prediction_point": prediction_point,
        }

    except Exception as exc:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/health")
def health():
    # Check Ollama connectivity
    try:
        r = _requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False
    return {
        "status": "ok",
        "time":   datetime.now().isoformat(),
        "ollama": ollama_ok,
        "model":  OLLAMA_MODEL,
    }


# Serve built frontend
_frontend_dist = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "frontend", "dist",
)
if os.path.isdir(_frontend_dist):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True,
                loop="asyncio")
