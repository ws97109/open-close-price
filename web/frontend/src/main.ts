import './style.css'
import { predict, analyzeStream } from './api'
import { fetchChartData, renderChart } from './chart'
import type { PredictResponse, Signal, TechnicalData, SentimentArticle, PriceRange } from './types'

const QUICK_STOCKS = [
  { id: '2412', name: '中華電信' }, { id: '2330', name: '台積電' },
  { id: '0050', name: '台灣50' },   { id: '2454', name: '聯發科' },
  { id: '2317', name: '鴻海' },     { id: '2882', name: '國泰金' },
  { id: '2886', name: '兆豐金' },   { id: '2367', name: '燿華' },
]

const SIG: Record<Signal, { label: string; emoji: string; cls: string; desc: string }> = {
  UP:        { label: '看漲',   emoji: '↑', cls: 'sig-up',   desc: '模型預測該方向，信心達門檻' },
  DOWN:      { label: '看跌',   emoji: '↓', cls: 'sig-down', desc: '模型預測該方向，信心達門檻' },
  NO_SIGNAL: { label: '觀望',   emoji: '—', cls: 'sig-none', desc: '信心不足，不建議操作' },
}

function pct(v: number | null, d = 2): string {
  if (v == null) return 'N/A'
  return `${v >= 0 ? '+' : ''}${(v * 100).toFixed(d)}%`
}
function fmt(v: number | null, d = 1): string {
  if (v == null) return 'N/A'
  return v.toFixed(d)
}
function dir(v: number | null): 'up' | 'down' | 'neutral' {
  if (v == null) return 'neutral'
  return v > 0 ? 'up' : v < 0 ? 'down' : 'neutral'
}

function techRow(label: string, value: string, d: 'up' | 'down' | 'neutral' = 'neutral'): string {
  const cls = d === 'up' ? 'val-up' : d === 'down' ? 'val-down' : ''
  return `<div class="tech-row"><span class="tl">${label}</span><span class="tv ${cls}">${value}</span></div>`
}

function rangeCard(hl: PriceRange, curClose: number | null): string {
  const signPh = hl.high_pct >= 0 ? '+' : ''
  const signPl = hl.low_pct  >= 0 ? '+' : ''
  const rangePct = hl.high_pct - hl.low_pct
  return `
  <div class="pred-card range-card">
    <div class="pred-title">明日高低價預測</div>
    <div class="pred-acc">${hl.accuracy_note}</div>
    <div class="range-grid">
      <div class="range-cell">
        <div class="range-lbl">預測最高</div>
        <div class="range-val val-up">${hl.pred_high.toFixed(2)}</div>
        <div class="range-sub val-up">${signPh}${hl.high_pct.toFixed(2)}%</div>
      </div>
      <div class="range-sep">↕</div>
      <div class="range-cell">
        <div class="range-lbl">預測最低</div>
        <div class="range-val val-down">${hl.pred_low.toFixed(2)}</div>
        <div class="range-sub val-down">${signPl}${hl.low_pct.toFixed(2)}%</div>
      </div>
    </div>
    <div class="range-width">預期振幅 ${rangePct.toFixed(2)}%${curClose ? ` ≈ ${(curClose * rangePct / 100).toFixed(2)} 元` : ''}</div>
    <div class="pred-desc">基於 ATR、布林帶、波動率回歸預測</div>
  </div>`
}

function signalCard(title: string, accNote: string, side: PredictResponse['gap']): string {
  const sig = SIG[side.signal]
  const prob = Math.round(side.probability * 100)
  const conf = Math.round(side.confidence * 100)
  return `
  <div class="pred-card ${sig.cls}">
    <div class="pred-title">${title}</div>
    <div class="pred-acc">${accNote}</div>
    <div class="pred-sig">
      <span class="pred-arrow">${sig.emoji}</span>
      <span class="pred-label">${sig.label}</span>
    </div>
    <div class="pred-bars">
      <div class="pbar-row">
        <span class="pbar-lbl">漲機率</span>
        <div class="pbar-track">
          <div class="pbar-fill" style="width:${prob}%;background:${side.probability>0.5?'var(--green)':'var(--red)'}"></div>
        </div>
        <span class="pbar-val ${side.probability>0.5?'val-up':'val-down'}">${prob}%</span>
      </div>
      <div class="pbar-row">
        <span class="pbar-lbl">信心度</span>
        <div class="pbar-track">
          <div class="pbar-fill" style="width:${Math.min(conf,100)}%;background:var(--accent)"></div>
        </div>
        <span class="pbar-val">${conf}%</span>
      </div>
    </div>
    <div class="pred-desc">${sig.desc}</div>
  </div>`
}

function renderResult(d: PredictResponse): string {
  const t: TechnicalData = d.technical
  const p = d.price

  // Feature importance bars
  const featsHtml = d.top_features.map(f => `
    <div class="feat-row">
      <span class="feat-lbl">${f.label}</span>
      <div class="feat-track"><div class="feat-fill" style="width:${Math.round(f.rel*100)}%"></div></div>
      <span class="feat-num">${f.importance.toFixed(0)}</span>
    </div>`).join('')

  // Sentiment badge
  const sentScore = d.sentiment.score
  const sentCls = sentScore > 0.2 ? 'sent-pos' : sentScore < -0.2 ? 'sent-neg' : 'sent-neu'
  const sentLabel = d.sentiment.label
  const sentAdj = d.sentiment.adjustment

  // News list
  const newsHtml = d.sentiment.articles.slice(0, 6).map((a: SentimentArticle) => {
    const cls = a.label === '正面' ? 'val-up' : a.label === '負面' ? 'val-down' : ''
    const link = a.url ? `<a href="${a.url}" target="_blank" rel="noopener">${a.title}</a>` : a.title
    return `<div class="news-row"><span class="news-tag ${cls}">${a.label}</span>${link}</div>`
  }).join('')

  const priceChange = (p.close && p.open) ? p.close - p.open : null
  const priceChangePct = (priceChange && p.open) ? priceChange / p.open : null

  return `
  <div class="result-card">

    <!-- Header: price info -->
    <div class="rc-header">
      <div class="rc-title-row">
        <h2 class="rc-title">${d.stock} <span class="rc-name">${d.stock_name}</span></h2>
        <div class="rc-badge">
          <span class="rc-date">📅 ${p.date}</span>
        </div>
      </div>
      <div class="price-grid">
        <div class="price-cell">
          <div class="price-lbl">開盤</div>
          <div class="price-val">${p.open != null ? p.open.toFixed(2) : 'N/A'}</div>
        </div>
        <div class="price-cell highlight">
          <div class="price-lbl">收盤</div>
          <div class="price-val ${priceChange != null ? (priceChange >= 0 ? 'val-up' : 'val-down') : ''}">
            ${p.close != null ? p.close.toFixed(2) : 'N/A'}
            ${priceChangePct != null ? `<span class="price-chg">${pct(priceChangePct, 2)}</span>` : ''}
          </div>
        </div>
        <div class="price-cell">
          <div class="price-lbl">最高</div>
          <div class="price-val val-up">${p.high != null ? p.high.toFixed(2) : 'N/A'}</div>
        </div>
        <div class="price-cell">
          <div class="price-lbl">最低</div>
          <div class="price-val val-down">${p.low != null ? p.low.toFixed(2) : 'N/A'}</div>
        </div>
      </div>
      <p class="rc-meta">${d.rows.toLocaleString()} 筆歷史資料 · ${d.features} 個特徵 · 資料截至 ${d.as_of_date}</p>
    </div>

    <!-- Price chart -->
    <div class="section-title chart-section-title">
      📊 價格走勢圖
      <span class="title-sub">K線 · MA20/60/120 · 預測線 · 機率圖</span>
      <span class="chart-loading-badge" id="chart-badge">載入中…</span>
    </div>
    <div id="chart-container" class="chart-container">
      <div class="chart-spinner"><div class="spinner"></div></div>
    </div>

    <!-- Prediction cards -->
    <div class="section-title">明日預測</div>
    <div class="pred-grid">
      ${signalCard('開盤跳空方向', d.gap.accuracy_note, d.gap)}
      ${signalCard('收盤漲跌方向', d.close.accuracy_note, d.close)}
    </div>
    <div class="pred-grid-single">
      ${d.high_low ? rangeCard(d.high_low, d.price.close) : ''}
    </div>

    <!-- Sentiment -->
    <div class="section-title">
      新聞情緒分析 <span class="title-sub">（BERT + DuckDuckGo）</span>
      <span class="sent-badge ${sentCls}">${sentLabel} ${sentScore >= 0 ? '+' : ''}${sentScore.toFixed(2)}</span>
      ${Math.abs(sentAdj) > 0 ? `<span class="sent-adj">機率調整 ${sentAdj >= 0 ? '+' : ''}${(sentAdj * 100).toFixed(1)}%</span>` : ''}
    </div>
    <div class="news-list">${newsHtml || '<div class="news-empty">無法取得新聞資料</div>'}</div>

    <!-- Technical indicators -->
    <div class="section-title">技術指標</div>
    <div class="tech-grid">
      ${techRow('RSI(14)',      fmt(t.rsi14),                                      t.rsi14!=null?(t.rsi14>60?'up':t.rsi14<40?'down':'neutral'):'neutral')}
      ${techRow('K 值 (9)',    fmt(t.k9),                                          dir(t.k9 != null ? t.k9-50 : null))}
      ${techRow('D 值 (9)',    fmt(t.d9),                                          dir(t.d9 != null ? t.d9-50 : null))}
      ${techRow('KD 多頭交叉', t.kdc != null ? (t.kdc > 0.5 ? '✓ 是' : '✗ 否') : 'N/A', t.kdc!=null?(t.kdc>0.5?'up':'down'):'neutral')}
      ${techRow('均線黃金交叉', t.gc  != null ? (t.gc  > 0.5 ? '✓ 是' : '✗ 否') : 'N/A', t.gc!=null?(t.gc>0.5?'up':'down'):'neutral')}
      ${techRow('MACD 多頭',   t.mxs != null ? (t.mxs > 0.5 ? '✓ 是' : '✗ 否') : 'N/A', t.mxs!=null?(t.mxs>0.5?'up':'down'):'neutral')}
      ${techRow('相對 MA20',   pct(t.pma20),   dir(t.pma20))}
      ${techRow('相對 MA60',   pct(t.pma60),   dir(t.pma60))}
      ${techRow('布林位置',     fmt(t.bbp, 3),  t.bbp!=null?(t.bbp>0.7?'down':t.bbp<0.3?'up':'neutral'):'neutral')}
      ${techRow('昨日報酬',    pct(t.r1),      dir(t.r1))}
      ${techRow('5 日報酬',    pct(t.r5),      dir(t.r5))}
      ${techRow('多頭綜合分',  fmt(t.bull_score, 0) + ' / 10', t.bull_score!=null?(t.bull_score>=6?'up':t.bull_score<=4?'down':'neutral'):'neutral')}
      ${techRow('美股 SPY',    pct(t.spy_r1,2), dir(t.spy_r1))}
      ${techRow('大盤 0050',   pct(t.mkt_r1,2), dir(t.mkt_r1))}
      ${techRow('相對大盤',    pct(t.rel_mkt1,2),dir(t.rel_mkt1))}
      ${techRow('ATR 波動度',  t.natr != null ? (t.natr*100).toFixed(2)+'%' : 'N/A', 'neutral')}
    </div>

    <!-- Feature importance -->
    <div class="section-title">主要影響因素 <span class="title-sub">（LightGBM 特徵重要度）</span></div>
    <div class="feats">${featsHtml}</div>

    <!-- Ollama AI analysis -->
    <div class="section-title ai-title">
      🤖 AI 深度分析
      <span class="title-sub">（Ollama ${d.stock_name} · qwen3:14b）</span>
      <button id="analyze-btn" class="btn-analyze">開始分析</button>
    </div>
    <div id="ai-output" class="ai-output">
      <div class="ai-placeholder">點擊「開始分析」，AI 將結合技術面、籌碼面與最新新聞給出綜合判斷。</div>
    </div>

    <!-- Disclaimer -->
    <div class="disclaimer">
      ⚠ 開盤跳空 84%（conf&gt;70%）· 收盤方向 ~67%（conf&gt;70%）· 高低價 ±1% 命中率 94%+
      · 信心度門檻均為 70%，低於此值顯示「觀望」 · 以上資訊僅供研究參考，不構成投資建議
    </div>
  </div>`
}

// ─── Loading/Error ─────────────────────────────────────────────
function showLoading(sid: string): void {
  const sec = document.getElementById('result-section')!
  sec.innerHTML = `
    <div class="loading-card">
      <div class="spinner"></div>
      <p>正在分析 <strong>${sid}</strong>…</p>
      <p class="loading-sub">下載 20 年資料、訓練 GPU 加速模型、計算 BERT 情緒，首次約需 60–90 秒，同一股票當日第二次即時返回</p>
    </div>`
  sec.classList.remove('hidden')
}
function showError(msg: string): void {
  const sec = document.getElementById('result-section')!
  sec.innerHTML = `<div class="error-card"><div class="err-icon">⚠</div><p>${msg}</p></div>`
}

// ─── Ollama analysis ───────────────────────────────────────────
function attachAnalyzeBtn(result: PredictResponse): void {
  const btn = document.getElementById('analyze-btn') as HTMLButtonElement | null
  const out = document.getElementById('ai-output')
  if (!btn || !out) return
  btn.addEventListener('click', async () => {
    btn.disabled = true
    btn.textContent = '分析中…'
    out.innerHTML = '<div class="ai-streaming"></div>'
    const stream = out.querySelector('.ai-streaming')!

    let full = ''
    await analyzeStream(
      result.stock,
      result.stock_name,
      result,
      (chunk) => {
        full += chunk
        // Basic markdown-ish: bold, newlines
        stream.innerHTML = full
          .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
          .replace(/\n/g, '<br>')
      },
      () => {
        btn.textContent = '重新分析'
        btn.disabled = false
      },
    )
  })
}

// ─── Main predict flow ─────────────────────────────────────────
async function runPredict(): Promise<void> {
  const input = document.getElementById('stock-input') as HTMLInputElement
  const btn   = document.getElementById('predict-btn') as HTMLButtonElement
  const sec   = document.getElementById('result-section')!
  const sid   = input.value.trim()

  if (!sid) {
    input.classList.add('shake')
    input.focus()
    setTimeout(() => input.classList.remove('shake'), 400)
    return
  }
  btn.disabled = true
  showLoading(sid)

  try {
    const result = await predict(sid)
    sec.innerHTML = renderResult(result)
    sec.classList.remove('hidden')
    sec.scrollIntoView({ behavior: 'smooth', block: 'start' })
    attachAnalyzeBtn(result)
    loadChart(result.stock)
  } catch (err) {
    showError(err instanceof Error ? err.message : '未知錯誤')
  } finally {
    btn.disabled = false
  }
}

// ─── Chart loader ──────────────────────────────────────────────
async function loadChart(stockId: string): Promise<void> {
  const container = document.getElementById('chart-container')
  const badge     = document.getElementById('chart-badge')
  if (!container) return

  try {
    const data = await fetchChartData(stockId)
    renderChart(container, data)
    if (badge) {
      badge.textContent = `${data.candles.length} 根K線`
      badge.classList.add('chart-badge-ok')
    }
  } catch (err) {
    container.innerHTML = `<div class="chart-error">⚠ 圖表載入失敗：${err instanceof Error ? err.message : '未知錯誤'}</div>`
    if (badge) badge.textContent = '載入失敗'
  }
}

// ─── Init ──────────────────────────────────────────────────────
function init(): void {
  const container = document.getElementById('quick-stocks')!
  QUICK_STOCKS.forEach(({ id, name }) => {
    const chip = document.createElement('button')
    chip.className = 'chip'
    chip.textContent = `${id} ${name}`
    chip.addEventListener('click', () => {
      (document.getElementById('stock-input') as HTMLInputElement).value = id
    })
    container.appendChild(chip)
  })

  document.getElementById('predict-btn')!.addEventListener('click', runPredict)
  const input = document.getElementById('stock-input') as HTMLInputElement
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') runPredict() })
}

init()
