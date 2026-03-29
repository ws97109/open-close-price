#!/usr/bin/env python3
"""
sentiment.py — Real-time news fetch + BERT sentiment for Taiwan stocks
=======================================================================
1. fetch_news(stock_id, stock_name) → list of news headlines (DuckDuckGo)
2. compute_sentiment(headlines) → float score in [-1, 1]
3. sentiment_to_adjustment(score) → probability adjustment [-0.08, +0.08]

Sentiment model: multilingual sentiment classifier (XLM-RoBERTa / distilbert)
Loads lazily on first use; cached across calls.
"""
import re
import logging
from typing import List, Optional, Tuple, Dict

logger = logging.getLogger(__name__)

# ─── News fetching ────────────────────────────────────────────
def fetch_news(
    stock_id: str,
    stock_name: str = "",
    n_results: int = 10,
) -> List[Dict]:
    """
    Fetch recent news about a stock via DuckDuckGo text search.
    Returns list of {"title": ..., "body": ..., "url": ...}
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        queries = [
            f"{stock_id} {stock_name} 股票 新聞",
            f"台股 {stock_name} 今日",
        ]
        seen, articles = set(), []
        with DDGS() as ddgs:
            for q in queries:
                for r in ddgs.text(q, max_results=n_results, region="tw-tzh"):
                    title = r.get("title", "")
                    if title and title not in seen:
                        seen.add(title)
                        articles.append({
                            "title": title,
                            "body":  r.get("body", ""),
                            "url":   r.get("href", ""),
                        })
                    if len(articles) >= n_results:
                        break
                if len(articles) >= n_results:
                    break
        logger.info(f"  News: fetched {len(articles)} articles for {stock_id}")
        return articles
    except Exception as e:
        logger.warning(f"  News fetch failed: {e}")
        return []


# ─── Sentiment model (lazy singleton) ─────────────────────────
_sentiment_pipeline = None

def _get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is not None:
        return _sentiment_pipeline
    try:
        from transformers import pipeline
        # Multilingual sentiment model — supports Chinese / English
        logger.info("  Loading BERT sentiment model …")
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            top_k=None,
            truncation=True,
            max_length=512,
        )
        logger.info("  BERT sentiment model loaded.")
    except Exception as e:
        logger.warning(f"  BERT model load failed ({e}), falling back to keyword method")
        _sentiment_pipeline = "keyword"
    return _sentiment_pipeline


# ─── Keyword fallback ─────────────────────────────────────────
_POS_KW = re.compile(
    r"上漲|漲停|突破|創新高|利多|買超|增持|強勢|獲利|成長|"
    r"beat|surge|rally|gain|bullish|upgrade|buy|profit|outperform",
    re.IGNORECASE,
)
_NEG_KW = re.compile(
    r"下跌|跌停|破底|創新低|利空|賣超|減持|弱勢|虧損|衰退|"
    r"miss|slump|decline|loss|bearish|downgrade|sell|debt|underperform",
    re.IGNORECASE,
)

def _keyword_score(text: str) -> float:
    pos = len(_POS_KW.findall(text))
    neg = len(_NEG_KW.findall(text))
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total  # [-1, 1]


# ─── Public API ───────────────────────────────────────────────
def compute_sentiment(
    articles: List[Dict],
    use_bert: bool = True,
) -> Tuple[float, List[Dict]]:
    """
    Compute aggregate sentiment from news articles.

    Returns
    -------
    (score, detail_list)
      score       : float in [-1, 1] (positive = bullish, negative = bearish)
      detail_list : per-article {title, score, label}
    """
    if not articles:
        return 0.0, []

    texts = [f"{a['title']} {a['body'][:200]}" for a in articles]
    detail = []

    if use_bert:
        pipe = _get_sentiment_pipeline()
        if pipe != "keyword":
            try:
                results = pipe(texts)
                scores = []
                for art, res in zip(articles, results):
                    # res is list of {label, score}
                    pos_score = next((r["score"] for r in res if r["label"].lower() == "positive"), 0.0)
                    neg_score = next((r["score"] for r in res if r["label"].lower() == "negative"), 0.0)
                    s = pos_score - neg_score   # [-1, 1]
                    scores.append(s)
                    detail.append({
                        "title": art["title"][:80],
                        "score": round(s, 3),
                        "label": "正面" if s > 0.2 else ("負面" if s < -0.2 else "中性"),
                        "url":   art.get("url", ""),
                    })
                agg = float(sum(scores) / len(scores)) if scores else 0.0
                return agg, detail
            except Exception as e:
                logger.warning(f"  BERT inference failed ({e}), falling back to keyword")

    # Keyword fallback
    scores = []
    for art in articles:
        text = art["title"] + " " + art.get("body", "")[:300]
        s = _keyword_score(text)
        scores.append(s)
        detail.append({
            "title": art["title"][:80],
            "score": round(s, 3),
            "label": "正面" if s > 0 else ("負面" if s < 0 else "中性"),
            "url":   art.get("url", ""),
        })
    agg = float(sum(scores) / len(scores)) if scores else 0.0
    return agg, detail


def sentiment_to_adjustment(score: float) -> float:
    """
    Convert sentiment score [-1, 1] to probability adjustment [-0.08, +0.08].
    Only applies when |score| > 0.3 (avoid noise).
    """
    if abs(score) < 0.3:
        return 0.0
    # Scale: score ±1.0 → ±0.08 adjustment
    return float(max(-0.08, min(0.08, score * 0.08)))


def analyze(
    stock_id: str,
    stock_name: str = "",
    n_results: int = 8,
    use_bert: bool = True,
) -> Dict:
    """Convenience: fetch news + compute sentiment in one call."""
    articles = fetch_news(stock_id, stock_name, n_results)
    score, detail = compute_sentiment(articles, use_bert)
    adj = sentiment_to_adjustment(score)
    return {
        "score":      round(score, 3),
        "adjustment": round(adj, 4),
        "label":      "正面" if score > 0.2 else ("負面" if score < -0.2 else "中性"),
        "articles":   detail,
        "n_articles": len(detail),
    }


if __name__ == "__main__":
    import json, sys
    sid  = sys.argv[1] if len(sys.argv) > 1 else "2412"
    name = sys.argv[2] if len(sys.argv) > 2 else "中華電信"
    result = analyze(sid, name)
    print(f"\nSentiment for {sid} {name}:")
    print(f"  Score: {result['score']:+.3f}  ({result['label']})")
    print(f"  Prob adjustment: {result['adjustment']:+.4f}")
    print(f"  Articles ({result['n_articles']}):")
    for a in result["articles"][:5]:
        print(f"    [{a['label']}] {a['title']}")
