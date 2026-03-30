# 台灣股票開盤預測系統

AI-powered Taiwan stock **opening gap & close direction** prediction system with real-time news sentiment and LLM analysis.

## Validated Accuracy (2412 中華電信, walk-forward 5 folds)

| Target | No filter | Filtered |
|--------|-----------|---------|
| Opening Gap (open[t+1] > close[t]) | 78.4% | **84.8%** (conf >70%) |
| Close Direction (close[t+1] > close[t]) | 59.3% | **77.4%** (conf >85%) |

## Features

- **Triple ensemble model**: LightGBM + XGBoost + CatBoost
- **172 features**: price/volume TA, institutional chip, SPY (US market), 0050 sector context
- **BERT sentiment**: multilingual BERT scores latest news from DuckDuckGo
- **Ollama qwen3:14b**: streams AI analysis combining technicals + news
- **Interactive chart**: TradingView Lightweight Charts with K-lines, MA20/60/120, prediction lines, model probability panel
- **Web UI**: TypeScript + Vite frontend, FastAPI backend

## Data Source

[FinMind](https://finmindtrade.com/) — free Taiwan stock data API
(`taiwan_stock_daily`, `taiwan_stock_institutional_investors`, `taiwan_stock_margin_purchase_short_sale`, `us_stock_price`)

## Quick Start

### Requirements
```
Python 3.10+   — pip install lightgbm xgboost catboost scikit-learn imbalanced-learn
                              FinMind fastapi uvicorn transformers ddgs pandas numpy
Node.js 18+    — nvm install --lts
Ollama         — ollama pull qwen3:14b
```

### Run

**Terminal 1 — Backend (port 8000):**
```bash
cd web/backend
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 --loop asyncio
```

**Terminal 2 — Frontend (port 5173):**
```bash
cd web/frontend
npm install
npm run dev
```

Open **http://localhost:5173**

### Command Line
```bash
# Predict next day (gap + close)
python3 finalmodel.py --predict --stock 2412

# Walk-forward validation
python3 finalmodel.py --validate --stock 2412

# News sentiment only
python3 sentiment.py 2412 中華電信
```

## Project Structure

```
├── finalmodel.py       # Core prediction model (CLI + importable)
├── sentiment.py        # DuckDuckGo news fetch + BERT sentiment
├── run.txt             # Detailed run instructions
└── web/
    ├── backend/
    │   └── server.py   # FastAPI: /api/predict, /api/chart, /api/analyze
    └── frontend/
        └── src/
            ├── main.ts     # UI logic
            ├── chart.ts    # TradingView Lightweight Charts
            ├── api.ts      # API client
            └── types.ts    # TypeScript types
```

## Key Insights

1. **Gap prediction**: Taiwan's opening gap correlates with US market (SPY) overnight return. However, using raw SPY return as a direct feature causes all stocks to predict the same direction. Instead, per-stock SPY interaction features are used: rolling stock-SPY correlation (`spy_corr20`), scaled SPY signal (`spy_alpha20 = spy_r1 × corr20`), and excess return vs SPY (`rel_spy_r1 = stock_r1 − spy_r1`). This allows high-beta stocks (TSMC) to follow SPY strongly while low-beta stocks (telecom) rely on their own signals.

2. **Close direction**: At conf>85% threshold, 77.4% accuracy. Lower confidence predictions show NO_SIGNAL — model is conservative to maintain quality.

## Disclaimer

For research purposes only. Not financial advice.
