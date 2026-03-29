import type { PredictResponse, StockInfo } from './types'

const BASE = '/api'

export async function getStocks(): Promise<StockInfo[]> {
  const res = await fetch(`${BASE}/stocks`)
  if (!res.ok) throw new Error('無法載入股票列表')
  return res.json()
}

export async function predict(
  stockId: string,
  token?: string,
): Promise<PredictResponse> {
  const res = await fetch(`${BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ stock_id: stockId, token }),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: '伺服器錯誤' }))
    throw new Error(body.detail ?? '預測失敗')
  }
  return res.json()
}

export async function analyzeStream(
  stockId: string,
  stockName: string,
  prediction: PredictResponse,
  onChunk: (text: string) => void,
  onDone: () => void,
): Promise<void> {
  const res = await fetch(`${BASE}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ stock_id: stockId, stock_name: stockName, prediction }),
  })
  if (!res.ok || !res.body) {
    onChunk('⚠ 無法連線至 AI 分析服務')
    onDone()
    return
  }
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    onChunk(decoder.decode(value, { stream: true }))
  }
  onDone()
}
