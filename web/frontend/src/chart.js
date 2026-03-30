import { createChart, CandlestickSeries, HistogramSeries, LineSeries, ColorType, CrosshairMode, LineStyle, } from 'lightweight-charts';
// ─── Chart state ────────────────────────────────────────────────
let _chart = null;
let _candles = null;
// ─── Fetch chart data ────────────────────────────────────────────
export async function fetchChartData(stockId) {
    const res = await fetch(`/api/chart/${stockId}?days=120`);
    if (!res.ok)
        throw new Error('圖表資料載入失敗');
    return res.json();
}
// ─── Render chart ────────────────────────────────────────────────
export function renderChart(container, data) {
    // Destroy old chart if exists
    if (_chart) {
        _chart.remove();
        _chart = null;
    }
    container.innerHTML = '';
    container.style.position = 'relative';
    // ── Main price chart ─────────────────────────────────────────
    const mainEl = document.createElement('div');
    mainEl.style.cssText = 'width:100%;height:340px;';
    container.appendChild(mainEl);
    _chart = createChart(mainEl, {
        layout: {
            background: { type: ColorType.Solid, color: '#161b22' },
            textColor: '#8b949e',
            fontSize: 11,
        },
        grid: {
            vertLines: { color: '#21262d' },
            horzLines: { color: '#21262d' },
        },
        crosshair: { mode: CrosshairMode.Normal },
        rightPriceScale: { borderColor: '#30363d', scaleMargins: { top: 0.08, bottom: 0.28 } },
        timeScale: { borderColor: '#30363d', timeVisible: true, secondsVisible: false },
        width: mainEl.clientWidth,
        height: 340,
    });
    // Responsive resize
    const resizeObs = new ResizeObserver(() => {
        if (_chart)
            _chart.applyOptions({ width: mainEl.clientWidth });
    });
    resizeObs.observe(mainEl);
    // ── Candlestick series ───────────────────────────────────────
    _candles = _chart.addSeries(CandlestickSeries, {
        upColor: '#3fb950',
        downColor: '#f85149',
        borderUpColor: '#3fb950',
        borderDownColor: '#f85149',
        wickUpColor: '#3fb950',
        wickDownColor: '#f85149',
    });
    _candles.setData(data.candles);
    // ── Volume (pane 0, lower portion) ──────────────────────────
    const volSeries = _chart.addSeries(HistogramSeries, {
        color: 'rgba(88,166,255,0.3)',
        priceFormat: { type: 'volume' },
        priceScaleId: 'vol',
    });
    volSeries.priceScale().applyOptions({
        scaleMargins: { top: 0.80, bottom: 0.00 },
    });
    volSeries.setData(data.volumes);
    // ── Moving averages ──────────────────────────────────────────
    const MA_COLORS = {
        ma20: '#58a6ff',
        ma60: '#d29922',
        ma120: '#bc8cff',
    };
    const MA_LABELS = {
        ma20: 'MA20', ma60: 'MA60', ma120: 'MA120',
    };
    const maSeries = {};
    for (const [key, pts] of Object.entries(data.ma_lines)) {
        if (!pts.length)
            continue;
        const s = _chart.addSeries(LineSeries, {
            color: MA_COLORS[key] ?? '#888',
            lineWidth: 1,
            title: MA_LABELS[key] ?? key,
            crosshairMarkerVisible: false,
            lastValueVisible: true,
            priceLineVisible: false,
        });
        s.setData(pts);
        maSeries[key] = s;
    }
    // ── Prediction point & lines ─────────────────────────────────
    const pp = data.prediction_point;
    const sigGap = pp.gap_signal;
    const sigClose = pp.close_signal;
    // Always draw prediction lines — grey when NO_SIGNAL, green/red when confident
    const gapColor = sigGap === 'UP' ? '#3fb950' : sigGap === 'DOWN' ? '#f85149' : '#4a5568';
    const closeColor = sigClose === 'UP' ? '#a8e6cf' : sigClose === 'DOWN' ? '#ff8b8b' : '#6b7280';
    const gapLabel = sigGap !== 'NO_SIGNAL'
        ? `開盤預測 ${sigGap === 'UP' ? '↑' : '↓'} ${(pp.prob_gap * 100).toFixed(0)}%`
        : `開盤 ${(pp.prob_gap * 100).toFixed(0)}%（觀望）`;
    const closeLabel = sigClose !== 'NO_SIGNAL'
        ? `收盤預測 ${sigClose === 'UP' ? '↑' : '↓'} ${(pp.prob_close * 100).toFixed(0)}%`
        : `收盤 ${(pp.prob_close * 100).toFixed(0)}%（觀望）`;
    const gapLine = _chart.addSeries(LineSeries, {
        color: gapColor,
        lineWidth: sigGap !== 'NO_SIGNAL' ? 2 : 1,
        lineStyle: LineStyle.Dashed,
        title: gapLabel,
        lastValueVisible: true,
        priceLineVisible: false,
        crosshairMarkerRadius: 5,
    });
    gapLine.setData([
        { time: pp.last_time, value: pp.last_close },
        { time: pp.next_time, value: pp.gap_target_price },
    ]);
    const closeLine = _chart.addSeries(LineSeries, {
        color: closeColor,
        lineWidth: sigClose !== 'NO_SIGNAL' ? 2 : 1,
        lineStyle: LineStyle.Dotted,
        title: closeLabel,
        lastValueVisible: true,
        priceLineVisible: false,
        crosshairMarkerRadius: 5,
    });
    closeLine.setData([
        { time: pp.last_time, value: pp.last_close },
        { time: pp.next_time, value: pp.close_target_price },
    ]);
    // ── Probability sub-chart ────────────────────────────────────
    const probEl = document.createElement('div');
    probEl.style.cssText = 'width:100%;height:130px;margin-top:1px;';
    container.appendChild(probEl);
    const probChart = createChart(probEl, {
        layout: {
            background: { type: ColorType.Solid, color: '#161b22' },
            textColor: '#8b949e',
            fontSize: 10,
        },
        grid: {
            vertLines: { color: '#21262d' },
            horzLines: { color: '#21262d' },
        },
        rightPriceScale: {
            borderColor: '#30363d',
            scaleMargins: { top: 0.05, bottom: 0.05 },
        },
        timeScale: { borderColor: '#30363d', timeVisible: true, visible: true },
        width: probEl.clientWidth,
        height: 130,
        crosshair: { mode: CrosshairMode.Normal },
    });
    const resizeObs2 = new ResizeObserver(() => {
        probChart.applyOptions({ width: probEl.clientWidth });
    });
    resizeObs2.observe(probEl);
    // 50% baseline
    if (data.gap_probs.length > 0) {
        const baselineData = [
            { time: data.gap_probs[0].time, value: 0.5 },
            { time: data.gap_probs[data.gap_probs.length - 1].time, value: 0.5 },
        ];
        const baseLine = probChart.addSeries(LineSeries, {
            color: '#30363d', lineWidth: 1, lineStyle: LineStyle.Dashed,
            lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false,
        });
        baseLine.setData(baselineData);
    }
    // Gap prob line
    if (data.gap_probs.length > 0) {
        const gapProbSeries = probChart.addSeries(LineSeries, {
            color: '#3fb950', lineWidth: 2,
            title: '開盤跳空機率',
            lastValueVisible: true, priceLineVisible: false,
            crosshairMarkerRadius: 3,
        });
        // Color each point
        const colored = data.gap_probs.map(p => ({
            time: p.time,
            value: p.value,
        }));
        gapProbSeries.setData(colored);
    }
    // Close prob line
    if (data.close_probs.length > 0) {
        const closeProbSeries = probChart.addSeries(LineSeries, {
            color: '#58a6ff', lineWidth: 2,
            title: '收盤漲跌機率',
            lastValueVisible: true, priceLineVisible: false,
            crosshairMarkerRadius: 3,
        });
        closeProbSeries.setData(data.close_probs);
    }
    // Sync time scales
    _chart.timeScale().subscribeVisibleLogicalRangeChange(range => {
        if (range)
            probChart.timeScale().setVisibleLogicalRange(range);
    });
    probChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
        if (range && _chart)
            _chart.timeScale().setVisibleLogicalRange(range);
    });
    _chart.timeScale().fitContent();
    probChart.timeScale().fitContent();
    // ── Legend overlay ───────────────────────────────────────────
    const legend = document.createElement('div');
    legend.className = 'chart-legend';
    legend.innerHTML = `
    <span class="leg-item" style="color:#58a6ff">■ MA20</span>
    <span class="leg-item" style="color:#d29922">■ MA60</span>
    <span class="leg-item" style="color:#bc8cff">■ MA120</span>
    <span class="leg-item" style="color:${gapColor}">--- 開盤${sigGap === 'UP' ? '↑' : sigGap === 'DOWN' ? '↓' : '？'}</span>
    <span class="leg-item" style="color:${closeColor}">··· 收盤${sigClose === 'UP' ? '↑' : sigClose === 'DOWN' ? '↓' : '？'}</span>
  `;
    mainEl.appendChild(legend);
    // ── Prob chart label ─────────────────────────────────────────
    const probLabel = document.createElement('div');
    probLabel.className = 'prob-chart-label';
    probLabel.innerHTML = `
    <span style="color:#3fb950">■ 開盤跳空機率</span>
    <span style="color:#58a6ff">■ 收盤漲跌機率</span>
    <span style="color:#30363d">--- 50% 基線</span>
  `;
    probEl.appendChild(probLabel);
}
