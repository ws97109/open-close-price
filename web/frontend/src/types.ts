export interface StockInfo { id: string; name: string }

export type Signal = 'UP' | 'DOWN' | 'NO_SIGNAL'

export interface PredictionSide {
  signal: Signal
  probability: number
  confidence: number
  raw_prob: number
  accuracy_note: string
}

export interface SentimentArticle {
  title: string
  score: number
  label: string
  url: string
}

export interface Sentiment {
  score: number
  adjustment: number
  label: string
  articles: SentimentArticle[]
  n_articles: number
}

export interface TechnicalData {
  rsi14: number | null
  k9: number | null
  d9: number | null
  kdc: number | null
  bull_score: number | null
  pma20: number | null
  pma60: number | null
  bbp: number | null
  spy_r1: number | null
  streak3: number | null
  r1: number | null
  r5: number | null
  natr: number | null
  gc: number | null
  mxs: number | null
  vmr5: number | null
  mkt_r1: number | null
  rel_mkt1: number | null
  msr: number | null
}

export interface FeatureImportance {
  name: string
  label: string
  importance: number
  rel: number
}

export interface PriceData {
  date: string
  open: number | null
  close: number | null
  high: number | null
  low: number | null
}

export interface PredictResponse {
  stock: string
  stock_name: string
  as_of_date: string
  price: PriceData
  gap: PredictionSide
  close: PredictionSide
  sentiment: Sentiment
  technical: TechnicalData
  top_features: FeatureImportance[]
  rows: number
  features: number
}
