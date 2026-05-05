"""
Market Topic Scanner — Signal Pipeline
Runs 3x daily via GitHub Actions cron.
Fetches multilingual news + crypto signals, asks Claude to synthesize 
into 3 sections: prediction markets to watch, market questions to create,
investment theses.
"""
import json
import os
import sys
from datetime import datetime, timezone
from urllib.parse import quote_plus

import feedparser
import requests
from anthropic import Anthropic

# ============================================================
# CONFIG
# ============================================================

# English business / news / crypto sources
ENGLISH_FEEDS = [
    ("CNBC Top News", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("Reuters Business", "https://feeds.reuters.com/reuters/businessNews"),
    ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Cointelegraph", "https://cointelegraph.com/rss"),
    ("Politico", "https://www.politico.com/rss/politicopicks.xml"),
    ("The Hill", "https://thehill.com/news/feed/"),
]

# Chinese sources (this is your edge — most candidates can't read these)
CHINESE_FEEDS = [
    ("财新 Caixin", "https://www.caixin.com/rss/all.xml"),
    ("第一财经 Yicai", "https://www.yicai.com/feed/"),
    ("36氪 36Kr", "https://36kr.com/feed"),
]

HACKER_NEWS_API = "https://hacker-news.firebaseio.com/v0"
COINGECKO_TRENDING = "https://api.coingecko.com/api/v3/search/trending"

OUTPUT_FILE = "market-signals.json"
MAX_ITEMS_PER_FEED = 8


# ============================================================
# DATA FETCHING
# ============================================================

def fetch_rss(name: str, url: str, max_items: int = MAX_ITEMS_PER_FEED) -> list:
    """Fetch and parse an RSS feed. Returns list of dicts."""
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
        print(f"  ✓ {name}: {len(items)} items", flush=True)
        return items
    except Exception as e:
        print(f"  ✗ {name}: {e}", flush=True)
        return []


def fetch_hacker_news(top_n: int = 15) -> list:
    """Fetch top HN stories."""
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
        print(f"  ✓ Hacker News: {len(items)} items", flush=True)
        return items[:8]
    except Exception as e:
        print(f"  ✗ Hacker News: {e}", flush=True)
        return []


def fetch_crypto_trending() -> list:
    """Fetch CoinGecko trending tokens."""
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
        print(f"  ✓ CoinGecko: {len(items)} items", flush=True)
        return items
    except Exception as e:
        print(f"  ✗ CoinGecko: {e}", flush=True)
        return []


# ============================================================
# LLM ANALYSIS
# ============================================================

ANALYSIS_PROMPT = """ANALYSIS_PROMPT = """You are analyzing today's news and market signals 
for a prediction market analyst, crypto/macro trader, or VC investor. 
You have data from English news, Chinese news (Caixin, Yicai, 36Kr — 
flag when these surface stories before English media), Hacker News, 
and trending crypto tokens.

Today's date: 2026-05-05. All forward-looking deadlines must be in 
2026 or 2027 — never use 2025 dates.

Synthesize three lists. Be specific and tied to actual signals.

═══════════════════════════════
SECTION 1: PREDICTION MARKETS TO WATCH
═══════════════════════════════
3-5 EXISTING prediction markets that deserve attention now. Hard rules:
- Must reference well-known entities (Bitcoin, Ethereum, Trump, Powell, 
  Fed, OpenAI, Anthropic, NVDA, Tesla, etc.) — NOT obscure regulatory 
  topics
- Must have plausible Polymarket/Kalshi/Manifold betting volume
- Phrase as a binary question with a number threshold or dated event

For each:
- market_title (specific, with threshold / deadline)
- venue (Polymarket / Kalshi / Manifold)
- why_now (one tight sentence — why today's signal makes this 
  market more interesting)
- signal_source

═══════════════════════════════
SECTION 2: MARKET QUESTIONS TO CREATE
═══════════════════════════════
3-5 NEW prediction market questions that don't exist yet. Hard rules:
- Must involve recognizable entities (companies, public figures, 
  asset classes, indices)
- Must be interesting to a crypto/finance Twitter audience
- Avoid niche regulatory minutiae (e.g., specific local manufacturing 
  rules in non-English-speaking regions) unless directly market-moving
- Resolution source must be a public, easily-checkable feed (price 
  feed, official announcement, count threshold)
- Deadlines: 30-180 days out, all in 2026 or 2027

For each:
- question (precise binary)
- deadline (YYYY-MM-DD format, must be 2026 or 2027)
- resolution_source
- why_underpriced (why no one's listed this yet)
- trigger_signal

═══════════════════════════════
SECTION 3: INVESTMENT THESES
═══════════════════════════════
3-5 thesis prompts. Hard rules:
- Must be expressible in liquid, recognizable instruments (BTC, ETH, 
  SPY, NVDA, MU, DXY, gold, USDC yield, BTC perps, ETH/BTC ratio, 
  major sector ETFs, etc.)
- Avoid obscure single names (e.g., specific small-cap industrial 
  Chinese stocks) unless they're a clean expression of a major theme
- Time horizon: days to months, not years

For each:
- thesis (one sentence, positional)
- horizon (e.g., "2-6 weeks")
- supporting_signals (cite actual sources from the input)
- counter_view (the strongest objection)
- expression (specific instrument: ticker, perp, options structure)

═══════════════════════════════
GUARDRAILS
═══════════════════════════════
- If Chinese sources surface something English media hasn't, FLAG IT — 
  this is the highest-value signal type
- Never invent specific numbers, prices, or quotes
- All dates must be 2026 or 2027
- If signals are weak, produce fewer items rather than padding

Return strictly valid JSON — no markdown fences, no extra text:

{
  "prediction_markets_to_watch": [
    {
      "market_title": "...",
      "venue": "Polymarket | Kalshi | Manifold",
      "why_now": "...",
      "signal_source": "..."
    }
  ],
  "market_questions_to_create": [
    {
      "question": "...",
      "deadline": "2026-MM-DD or 2027-MM-DD",
      "resolution_source": "...",
      "why_underpriced": "...",
      "trigger_signal": "..."
    }
  ],
  "investment_theses": [
    {
      "thesis": "...",
      "horizon": "...",
      "supporting_signals": ["...", "..."],
      "counter_view": "...",
      "expression": "..."
    }
  ]
}
"""

═══════════════════════════════
GUARDRAILS
═══════════════════════════════
- If Chinese sources surface something English media hasn't covered, 
  flag it explicitly — that's the highest-value signal type
- Don't make up specific numbers or quotes
- If signals are weak / day is quiet, say so honestly and produce fewer items

Return strictly valid JSON in this schema (no markdown fences, no extra text):

{
  "prediction_markets_to_watch": [
    {
      "market_title": "...",
      "venue": "Polymarket | Kalshi | Manifold",
      "why_now": "...",
      "signal_source": "..."
    }
  ],
  "market_questions_to_create": [
    {
      "question": "...",
      "deadline": "YYYY-MM-DD or relative (e.g., 'within 60 days')",
      "resolution_source": "...",
      "why_underpriced": "...",
      "trigger_signal": "..."
    }
  ],
  "investment_theses": [
    {
      "thesis": "...",
      "horizon": "...",
      "supporting_signals": ["...", "..."],
      "counter_view": "...",
      "expression": "..."
    }
  ]
}
"""


def build_llm_input(english: list, chinese: list, hn: list, crypto: list) -> str:
    """Format raw signals as text input for the LLM."""
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


def call_claude(signal_text: str) -> dict:
    """Send signals to Claude, parse structured response."""
    client = Anthropic()  # picks up ANTHROPIC_API_KEY from env
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4000,
        system=ANALYSIS_PROMPT,
        messages=[{"role": "user", "content": signal_text}],
    )
    raw = msg.content[0].text.strip()
    # Strip any markdown fences if model added them despite instruction
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw.strip())


# ============================================================
# MAIN
# ============================================================

def main():
    print("Fetching English RSS feeds...", flush=True)
    english = []
    for name, url in ENGLISH_FEEDS:
        english.extend(fetch_rss(name, url))

    print("\nFetching Chinese RSS feeds...", flush=True)
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
        print("  ✓ Analysis complete", flush=True)
    except Exception as e:
        print(f"  ✗ Analysis failed: {e}", flush=True)
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
    print(f"\n✓ Wrote {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
