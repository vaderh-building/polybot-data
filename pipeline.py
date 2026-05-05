"""
Market Topic Scanner — Signal Pipeline
Runs 3x daily via GitHub Actions cron.
"""
import json
import os
import sys
from datetime import datetime, timezone

import feedparser
import requests
from anthropic import Anthropic

ENGLISH_FEEDS = [
    ("CNBC Top News", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("Reuters Business", "https://feeds.reuters.com/reuters/businessNews"),
    ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Cointelegraph", "https://cointelegraph.com/rss"),
    ("Politico", "https://www.politico.com/rss/politicopicks.xml"),
    ("The Hill", "https://thehill.com/news/feed/"),
]

CHINESE_FEEDS = [
    ("财新 Caixin", "https://www.caixin.com/rss/all.xml"),
    ("第一财经 Yicai", "https://www.yicai.com/feed/"),
    ("36氪 36Kr", "https://36kr.com/feed"),
]

HACKER_NEWS_API = "https://hacker-news.firebaseio.com/v0"
COINGECKO_TRENDING = "https://api.coingecko.com/api/v3/search/trending"
OUTPUT_FILE = "market-signals.json"
MAX_ITEMS_PER_FEED = 8


def fetch_rss(name, url, max_items=MAX_ITEMS_PER_FEED):
    try:
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:max_items]:
            items.append({
                "source": name,
                "title": (entry.get("title") or "").strip()[:300],
                "summary": (entry.get("summary") or "").strip()[:500],
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
            })
        print(f"  OK {name}: {len(items)} items", flush=True)
        return items
    except Exception as e:
        print(f"  FAIL {name}: {e}", flush=True)
        return []


def fetch_hacker_news(top_n=15):
    try:
        ids = requests.get(f"{HACKER_NEWS_API}/topstories.json", timeout=10).json()[:top_n]
        items = []
        for sid in ids:
            try:
                story = requests.get(f"{HACKER_NEWS_API}/item/{sid}.json", timeout=5).json()
                if story and story.get("type") == "story" and story.get("title"):
                    items.append({
                        "source": "Hacker News",
                        "title": story["title"][:300],
                        "summary": f"score: {story.get('score', 0)}, comments: {story.get('descendants', 0)}",
                        "link": story.get("url") or f"https://news.ycombinator.com/item?id={sid}",
                        "score": story.get("score", 0),
                    })
            except Exception:
                continue
        items.sort(key=lambda x: x.get("score", 0), reverse=True)
        print(f"  OK Hacker News: {len(items)} items", flush=True)
        return items[:8]
    except Exception as e:
        print(f"  FAIL Hacker News: {e}", flush=True)
        return []


def fetch_crypto_trending():
    try:
        data = requests.get(COINGECKO_TRENDING, timeout=10).json()
        coins = data.get("coins", [])[:7]
        items = []
        for c in coins:
            coin = c.get("item", {})
            items.append({
                "source": "CoinGecko Trending",
                "title": f"{coin.get('name', '?')} ({coin.get('symbol', '?')})",
                "summary": f"market cap rank: {coin.get('market_cap_rank', 'n/a')}",
                "link": f"https://www.coingecko.com/en/coins/{coin.get('slug', '')}",
            })
        print(f"  OK CoinGecko: {len(items)} items", flush=True)
        return items
    except Exception as e:
        print(f"  FAIL CoinGecko: {e}", flush=True)
        return []


ANALYSIS_PROMPT = """You are analyzing today's news and market signals for a prediction market analyst, crypto/macro trader, or VC investor. You have data from English news, Chinese news (Caixin, Yicai, 36Kr — flag when these surface stories before English media), Hacker News, and trending crypto tokens.

Today's date: 2026-05-05. All forward-looking deadlines must be in 2026 or 2027 — never use 2025 dates.

Synthesize three lists. Be specific and tied to actual signals.

SECTION 1: PREDICTION MARKETS TO WATCH
3-5 EXISTING prediction markets that deserve attention now. Hard rules:
- Must reference well-known entities (Bitcoin, Ethereum, Trump, Powell, Fed, OpenAI, Anthropic, NVDA, Tesla, etc.) — NOT obscure regulatory topics
- Must have plausible Polymarket/Kalshi/Manifold betting volume
- Phrase as a binary question with a number threshold or dated event

For each: market_title, venue (Polymarket | Kalshi | Manifold), why_now (one tight sentence), signal_source.

SECTION 2: MARKET QUESTIONS TO CREATE
3-5 NEW prediction market questions that don't exist yet. Hard rules:
- Must involve recognizable entities (companies, public figures, asset classes, indices)
- Must be interesting to a crypto/finance Twitter audience
- Avoid niche regulatory minutiae unless directly market-moving
- Resolution source must be a public, easily-checkable feed
- Deadlines: 30-180 days out, all in 2026 or 2027

For each: question, deadline (YYYY-MM-DD, 2026 or 2027 only), resolution_source, why_underpriced, trigger_signal.

SECTION 3: INVESTMENT THESES
3-5 thesis prompts. Hard rules:
- Must be expressible in liquid, recognizable instruments (BTC, ETH, SPY, NVDA, MU, DXY, gold, USDC yield, BTC perps, ETH/BTC ratio, sector ETFs)
- Avoid obscure single names unless they're a clean expression of a major theme
- Time horizon: days to months, not years

For each: thesis (one sentence, positional), horizon, supporting_signals (cite actual sources), counter_view (strongest objection), expression (specific instrument).

GUARDRAILS
- If Chinese sources surface something English media hasn't, FLAG IT
- Never invent specific numbers, prices, or quotes
- All dates must be 2026 or 2027
- If signals are weak, produce fewer items rather than padding

Return strictly valid JSON — no markdown fences, no extra text:

{
  "prediction_markets_to_watch": [
    {"market_title": "...", "venue": "Polymarket | Kalshi | Manifold", "why_now": "...", "signal_source": "..."}
  ],
  "market_questions_to_create": [
    {"question": "...", "deadline": "2026-MM-DD", "resolution_source": "...", "why_underpriced": "...", "trigger_signal": "..."}
  ],
  "investment_theses": [
    {"thesis": "...", "horizon": "...", "supporting_signals": ["...", "..."], "counter_view": "...", "expression": "..."}
  ]
}
"""


def build_llm_input(english, chinese, hn, crypto):
    parts = []
    parts.append("# ENGLISH NEWS")
    for item in english[:30]:
        parts.append(f"- [{item['source']}] {item['title']}")
        if item.get("summary"):
            parts.append(f"  {item['summary'][:200]}")
    parts.append("\n# 中文新闻 / CHINESE NEWS")
    for item in chinese[:20]:
        parts.append(f"- [{item['source']}] {item['title']}")
        if item.get("summary"):
            parts.append(f"  {item['summary'][:200]}")
    parts.append("\n# HACKER NEWS TOP DISCUSSIONS")
    for item in hn[:8]:
        parts.append(f"- {item['title']} ({item.get('summary', '')})")
    parts.append("\n# CRYPTO TRENDING")
    for item in crypto:
        parts.append(f"- {item['title']}")
    return "\n".join(parts)


def call_claude(signal_text):
    client = Anthropic()
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4000,
        system=ANALYSIS_PROMPT,
        messages=[{"role": "user", "content": signal_text}],
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw.strip())


def main():
    print("Fetching English RSS...", flush=True)
    english = []
    for name, url in ENGLISH_FEEDS:
        english.extend(fetch_rss(name, url))

    print("\nFetching Chinese RSS...", flush=True)
    chinese = []
    for name, url in CHINESE_FEEDS:
        chinese.extend(fetch_rss(name, url))

    print("\nFetching Hacker News...", flush=True)
    hn = fetch_hacker_news()

    print("\nFetching Crypto Trending...", flush=True)
    crypto = fetch_crypto_trending()

    print("\nCalling Claude for analysis...", flush=True)
    signal_text = build_llm_input(english, chinese, hn, crypto)
    try:
        analysis = call_claude(signal_text)
        print("  OK Analysis complete", flush=True)
    except Exception as e:
        print(f"  FAIL Analysis: {e}", flush=True)
        analysis = {
            "prediction_markets_to_watch": [],
            "market_questions_to_create": [],
            "investment_theses": [],
            "error": str(e),
        }

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "live_signals": {
            "english_news": english[:25],
            "chinese_news": chinese[:15],
            "hacker_news": hn,
            "crypto_trending": crypto,
        },
        "analysis": analysis,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nOK wrote {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
