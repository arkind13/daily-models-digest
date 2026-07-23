# python daily_digest.py

#!/usr/bin/env python3
"""
daily_digest.py — Daily automation digest.

  1. Fetch OpenRouter models released in the last 24 hours.
  2. Send one Telegram summary message of those models.
  3. For each model, search YouTube (via Data API v3) for videos about it,
     filter them, and send one Telegram message per model.
  4. Check tracked YouTube channels for uploads in the last 24 hours and
     send one Telegram message per channel.

Requirements:
    pip install requests python-dotenv google-genai
"""

import html
import json
import os
import re
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Gemini is kept for potential future use; video search now uses YouTube API directly.
GEMINI_MODEL = "gemini-2.0-flash"

YOUTUBE_CHANNELS = [
    "UCmTM_hPCeckqN3cPWtYZZcg",  # The Deshbhakt
    "UC5l7RouTQ60oUjLjt1Nh-UQ",  # AI Revolution
    "UClXAalunTPaX1YV185DWUeg",  # Vaibhav Sisinty
    "UCYwLV1gDwzGbg7jXQ52bVnQ",  # Universe of AI
    "UC-CSyyi47VX1lD9zyeABW3w",  # Dhruv Rathee
]

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models?sort=newest"
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

DAY_IN_SECONDS = 86400
TELEGRAM_MAX_LENGTH = 4096
HTTP_TIMEOUT = 30
SEPARATOR = "━━━━━━━━━━━━━━━━━━━━━━"

# google-genai SDK — optional import so the script never crashes if missing.
# Note: video search no longer relies on Gemini (uses YouTube API directly).
try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:  # FIX #5: catch ImportError only, not SystemExit/KeyboardInterrupt
    genai = None
    genai_types = None
    GENAI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def safe_err(exc, limit=300):
    """Truncated error string — never log full responses/URLs (may hold keys)."""
    return str(exc)[:limit]


def raise_for_status_safe(resp, label):
    """Like resp.raise_for_status() but never embeds the URL (which may
    contain the Telegram token / YouTube API key) in the exception text."""
    if not resp.ok:
        raise RuntimeError(f"{label} returned HTTP {resp.status_code}")


def utf16_len(text):
    """Telegram counts message length in UTF-16 code units (emoji = 2)."""
    return len(text.encode("utf-16-le")) // 2


def utf16_slice(text, utf16_offset, utf16_length):
    """
    Slice a string by UTF-16 code units rather than Unicode code points.

    This is necessary because Telegram measures message length in UTF-16
    code units, and slicing by Python's code-point indexing can produce
    slices whose UTF-16 length differs from the expected count (e.g. emoji
    occupy 2 UTF-16 code units but only 1 Python code point).
    """
    encoded = text.encode("utf-16-le")
    byte_start = utf16_offset * 2
    byte_end = (utf16_offset + utf16_length) * 2
    return encoded[byte_start:byte_end].decode("utf-16-le")


def split_message(text, limit=TELEGRAM_MAX_LENGTH):
    """Split text into chunks under the Telegram limit, on line boundaries.

    Uses UTF-16-aware splitting to handle emoji and other non-BMP characters
    that take 2 UTF-16 code units each.
    """
    if utf16_len(text) <= limit:
        return [text]
    chunks, current_lines, current_len = [], [], 0
    for line in text.split("\n"):
        # Pathological single long line — split with UTF-16 awareness
        while utf16_len(line) > limit:
            if current_lines:
                chunks.append("\n".join(current_lines))
                current_lines, current_len = [], 0
            # Binary search: find the largest code-point prefix whose
            # UTF-16 length fits within `limit`.
            lo, hi = 0, len(line)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if utf16_len(line[:mid]) <= limit:
                    lo = mid
                else:
                    hi = mid - 1
            # lo is now the max number of code points we can take.
            # (lo is guaranteed >= 1 since even a single emoji is only 2 UTF-16 units.)
            chunk = line[:lo]
            chunks.append(chunk)
            line = line[lo:]
        added = utf16_len(line) + (1 if current_lines else 0)
        if current_lines and current_len + added > limit:
            chunks.append("\n".join(current_lines))
            current_lines, current_len = [line], utf16_len(line)
        else:
            current_lines.append(line)
            current_len += added
    if current_lines:
        chunks.append("\n".join(current_lines))
    return chunks


def format_count(value):
    """1234567 -> '1.2M', 12345 -> '12.3K'."""
    try:
        n = int(float(str(value).replace(",", "").strip()))
    except (TypeError, ValueError):
        return str(value) if value is not None else "0"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def format_price(raw):
    """OpenRouter pricing is USD per token (string). Show USD per 1M tokens."""
    try:
        per_million = float(raw) * 1_000_000
    except (TypeError, ValueError):
        return "N/A"
    if per_million == 0:
        return "0.00"
    if per_million < 0.01:
        return f"{per_million:.4f}"
    return f"{per_million:.2f}"


def parse_datetime(value):
    """Tolerant parser for ISO-8601 / common date strings / unix timestamps.
    Returns timezone-aware datetime or None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%B %d, %Y", "%b %d, %Y", "%d %B %Y"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def parse_json_array(text):
    """Extract a JSON array from an LLM response (tolerates code fences)."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end == -1 or end < start:
        # Tolerate a single object instead of an array.
        o_start, o_end = cleaned.find("{"), cleaned.rfind("}")
        if o_start != -1 and o_end > o_start:
            try:
                return [json.loads(cleaned[o_start:o_end + 1])]
            except json.JSONDecodeError:
                return []
        return []
    try:
        result = json.loads(cleaned[start:end + 1])
        return result if isinstance(result, list) else []
    except json.JSONDecodeError as exc:
        print(f"[gemini] Could not parse JSON response: {safe_err(exc)}")
        return []

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------


def send_telegram(text):
    """Send a Telegram message (HTML parse_mode), splitting long messages."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[telegram] TELEGRAM_TOKEN / TELEGRAM_CHAT_ID not set — message not sent.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for chunk in split_message(text):
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            resp = requests.post(url, json=payload, timeout=HTTP_TIMEOUT)
            print(f"[telegram] sendMessage status: {resp.status_code}")
            if resp.status_code == 429:
                print("[telegram] Rate limited (429) — skipping this message.")
                return
            if resp.status_code == 400:
                # Probably an HTML entity problem — retry as plain text.
                print("[telegram] HTTP 400 — retrying without parse_mode.")
                payload.pop("parse_mode", None)
                resp = requests.post(url, json=payload, timeout=HTTP_TIMEOUT)
                print(f"[telegram] plain-text resend status: {resp.status_code}")
            raise_for_status_safe(resp, "Telegram")
        except Exception as exc:
            print(f"[telegram] Failed to send message: {safe_err(exc)}")

# ---------------------------------------------------------------------------
# Step 1: OpenRouter models (last 24 hours)
# ---------------------------------------------------------------------------


def fetch_new_models(since_ts):
    """Return list of models created since `since_ts`, or None on failure."""
    try:
        resp = requests.get(OPENROUTER_MODELS_URL, timeout=HTTP_TIMEOUT)
        if resp.status_code == 429:
            print("[openrouter] Rate limited (429) — skipping model fetch.")
            send_telegram("⚠️ Failed to fetch OpenRouter models\nRate limited (429)")
            return None
        raise_for_status_safe(resp, "OpenRouter")
        models = resp.json().get("data", [])
        new_models = [m for m in models if int(m.get("created") or 0) >= since_ts]
        print(f"[openrouter] Fetched {len(models)} models total, "
              f"{len(new_models)} new in the last 24h.")
        return new_models
    except Exception as exc:
        print(f"[openrouter] Failed to fetch models: {safe_err(exc)}")
        send_telegram("⚠️ Failed to fetch OpenRouter models\n"
                      f"{html.escape(safe_err(exc))}")
        return None

# ---------------------------------------------------------------------------
# Step 2: Model summary message
# ---------------------------------------------------------------------------


def format_model_summary(models):
    lines = ["📋 <b>NEW MODELS ON OPENROUTER</b>", SEPARATOR]
    for m in models:
        name = html.escape(str(m.get("name") or m.get("id") or "Unknown model"))
        pricing = m.get("pricing") or {}
        prompt_price = format_price(pricing.get("prompt"))
        completion_price = format_price(pricing.get("completion"))
        released = datetime.fromtimestamp(
            int(m.get("created") or 0), tz=timezone.utc).strftime("%Y-%m-%d")
        context = m.get("context_length")
        context_str = f"{int(context):,}" if isinstance(context, (int, float)) else "N/A"
        lines.append(f"• <b>Model:</b> {name}")
        lines.append(f"  <b>Pricing:</b> ${prompt_price} / ${completion_price} per 1M tokens")
        lines.append(f"  <b>Released:</b> {released}")
        lines.append(f"  <b>Context:</b> {context_str}")
        lines.append(SEPARATOR)
    lines.append(f"<b>Total:</b> {len(models)} new model(s)")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Step 3: Per-model YouTube video search via YouTube Data API v3
# ---------------------------------------------------------------------------


def search_youtube_videos(query, since_ts, max_results=2):
    """
    Search YouTube using the Data API v3 for videos matching `query`
    published after `since_ts`. Returns a list of video dicts with
    real metadata, or None on API failure, or [] if no results.

    This replaces the old Gemini-based search which hallucinated URLs
    and statistics. The YouTube API returns real, verified data.
    """
    if not YOUTUBE_API_KEY:
        print("[youtube] YOUTUBE_API_KEY not set — video search unavailable.")
        return None

    # RFC 3339 format (ISO 8601) required by YouTube API
    published_after = datetime.fromtimestamp(since_ts, tz=timezone.utc).isoformat()

    try:
        # Step 1: Search for videos matching the query
        search_params = {
            "part": "snippet",
            "q": query,
            "order": "relevance",
            "maxResults": min(max_results * 2, 5),  # fetch extra, filter down
            "type": "video",
            "publishedAfter": published_after,
            "key": YOUTUBE_API_KEY,
        }
        search_resp = requests.get(
            YOUTUBE_SEARCH_URL, params=search_params, timeout=HTTP_TIMEOUT
        )
        if search_resp.status_code == 429:
            print(f"[youtube] Rate limited (429) searching for '{query}' — skipping.")
            return None
        raise_for_status_safe(search_resp, "YouTube search")
        search_items = search_resp.json().get("items", [])

        if not search_items:
            print(f"[youtube] No search results for '{query}'.")
            return []

        # Step 2: Collect video IDs for statistics lookup
        video_ids = []
        for item in search_items:
            vid = (item.get("id") or {}).get("videoId")
            if vid:
                video_ids.append(vid)

        # Step 3: Fetch statistics (views, likes) for the found videos
        stats_params = {
            "part": "statistics,snippet",
            "id": ",".join(video_ids),
            "key": YOUTUBE_API_KEY,
        }
        stats_resp = requests.get(
            YOUTUBE_VIDEOS_URL, params=stats_params, timeout=HTTP_TIMEOUT
        )
        raise_for_status_safe(stats_resp, "YouTube videos")
        stats_items = stats_resp.json().get("items", [])

        # Step 4: Build result list with real verified data
        videos = []
        for item in stats_items:
            snippet = item.get("snippet") or {}
            statistics = item.get("statistics") or {}
            published_dt = parse_datetime(snippet.get("publishedAt"))
            if published_dt is None or published_dt.timestamp() < since_ts:
                continue
            videos.append({
                "title": snippet.get("title") or "Untitled",
                "url": f"https://youtube.com/watch?v={item['id']}",
                "viewCount": int(statistics.get("viewCount") or 0),
                "likeCount": int(statistics.get("likeCount") or 0),
                "publishedAt": snippet.get("publishedAt", ""),
                "_published_dt": published_dt,
            })

        # Sort by engagement (views + likes) descending
        videos.sort(key=lambda v: v["viewCount"] + v["likeCount"], reverse=True)
        result = videos[:max_results]
        print(f"[youtube] {len(result)} real video(s) found for '{query}'.")
        return result

    except Exception as exc:
        err = safe_err(exc)
        if "429" in err:
            print(f"[youtube] Rate limited searching for '{query}'.")
        else:
            print(f"[youtube] Search failed for '{query}': {err}")
        return None


def format_videos_message(model_name, videos):
    lines = [f"🎥 <b>VIDEOS FOR {html.escape(model_name)}</b>", SEPARATOR]
    for i, v in enumerate(videos):
        if i:
            lines.append("")
        lines.append(html.escape(str(v.get("title") or "Untitled")))
        lines.append(f"🔗 {html.escape(str(v.get('url') or ''))}")
        lines.append(f"👍 {format_count(v.get('likeCount'))} | "
                     f"👁 {format_count(v.get('viewCount'))} | "
                     f"📅 {v['_published_dt'].strftime('%Y-%m-%d')}")
        lines.append(SEPARATOR)
    return "\n".join(lines)


def process_model(model):
    """Steps 3a–3c for a single model. Never raises."""
    model_id = str(model.get("id") or "")
    model_name = str(model.get("name") or model_id or "Unknown model")
    vendor = model_id.split("/")[0] if "/" in model_id else "unknown vendor"
    created_ts = int(model.get("created") or 0)

    print(f"[videos] Searching YouTube (via Data API v3) for: {model_name}")
    videos = search_youtube_videos(
        query=model_name,
        since_ts=created_ts,  # only videos published after the model release
        max_results=2,
    )

    if videos is None:  # YouTube API failure
        send_telegram(f"🎥 <b>VIDEOS FOR {html.escape(model_name)}</b>\n"
                      f"{SEPARATOR}\n"
                      "⚠️ Video search unavailable")
        return

    if not videos:  # No results found
        send_telegram(f'❌ No YouTube videos found yet for "{html.escape(model_name)}"')
        print(f"[videos] No videos found for: {model_name}")
        return

    send_telegram(format_videos_message(model_name, videos))
    print(f"[videos] Sent video message for: {model_name}")

# ---------------------------------------------------------------------------
# Step 4: Tracked YouTube channels (last 24 hours)
# ---------------------------------------------------------------------------


def fetch_recent_channel_videos(channel_id, since_ts):
    """Return up to 2 videos from a channel published since `since_ts`,
    enriched with statistics, sorted by view count. [] if none."""
    resp = requests.get(YOUTUBE_SEARCH_URL, params={
        "part": "snippet",
        "channelId": channel_id,
        "order": "date",
        "maxResults": 5,
        "type": "video",
        "key": YOUTUBE_API_KEY,
    }, timeout=HTTP_TIMEOUT)
    if resp.status_code == 429:
        print(f"[youtube] Rate limited (429) for channel {channel_id} — skipping.")
        return []
    raise_for_status_safe(resp, "YouTube search")

    recent = []
    for item in resp.json().get("items", []):
        snippet = item.get("snippet") or {}
        published_dt = parse_datetime(snippet.get("publishedAt"))
        video_id = (item.get("id") or {}).get("videoId")
        if published_dt and video_id and published_dt.timestamp() >= since_ts:
            recent.append({
                "video_id": video_id,
                "title": snippet.get("title") or "Untitled",
                "channel_title": snippet.get("channelTitle") or "Unknown channel",
            })
    if not recent:
        return []

    # Statistics for the matching videos.
    stats = {}
    try:
        stats_resp = requests.get(YOUTUBE_VIDEOS_URL, params={
            "part": "statistics",
            "id": ",".join(v["video_id"] for v in recent),
            "key": YOUTUBE_API_KEY,
        }, timeout=HTTP_TIMEOUT)
        if stats_resp.status_code == 429:
            print("[youtube] Rate limited (429) on statistics — using zeros.")
        else:
            raise_for_status_safe(stats_resp, "YouTube videos")
            stats = {it["id"]: it.get("statistics") or {}
                     for it in stats_resp.json().get("items", [])}
    except Exception as exc:
        print(f"[youtube] Statistics fetch failed: {safe_err(exc)}")

    for v in recent:
        s = stats.get(v["video_id"], {})
        v["views"] = int(s.get("viewCount") or 0)
        v["likes"] = int(s.get("likeCount") or 0)
    recent.sort(key=lambda v: v["views"], reverse=True)
    return recent[:2]


def format_channel_message(channel_title, videos):
    lines = [f"📺 <b>NEW FROM {html.escape(channel_title)}</b> (last 24h)", SEPARATOR]
    for v in videos:
        lines.append(f"• {html.escape(v['title'])}")
        lines.append(f"  🔗 https://youtube.com/watch?v={v['video_id']}")
        lines.append(f"  👁 {format_count(v['views'])} views | 👍 {format_count(v['likes'])}")
        lines.append(SEPARATOR)
    return "\n".join(lines)


def check_tracked_channels(since_ts):
    if not YOUTUBE_API_KEY:
        print("[youtube] YOUTUBE_API_KEY not set — skipping channel check.")
        send_telegram("⚠️ YouTube search unavailable")
        return

    any_uploads = False
    successes = 0
    errors = []  # FIX #4: collect errors, send one consolidated message
    for channel_id in YOUTUBE_CHANNELS:
        try:
            videos = fetch_recent_channel_videos(channel_id, since_ts)
        except Exception as exc:
            print(f"[youtube] Failed for channel {channel_id}: {safe_err(exc)}")
            errors.append(channel_id)
            continue
        successes += 1
        if not videos:
            print(f"[youtube] No recent uploads for channel {channel_id}.")
            continue
        any_uploads = True
        channel_title = videos[0]["channel_title"]
        send_telegram(format_channel_message(channel_title, videos))
        print(f"[youtube] Sent channel update: {channel_title}")
        time.sleep(1)  # be gentle with the Telegram API

    # FIX #4: send a single error message instead of one per failed channel
    if errors:
        send_telegram(f"⚠️ YouTube search unavailable for {len(errors)} channel(s)")

    if not any_uploads and successes > 0:
        send_telegram("📺 No new uploads from tracked channels in the last 24 hours.")
        print("[youtube] No uploads from tracked channels in the last 24 hours.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def check_config():
    missing = [name for name, val in (
        ("TELEGRAM_TOKEN", TELEGRAM_TOKEN),
        ("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID),
        ("GEMINI_API_KEY", GEMINI_API_KEY),
        ("YOUTUBE_API_KEY", YOUTUBE_API_KEY),
    ) if not val]
    if missing:
        print(f"[config] Missing environment variables: {', '.join(missing)}")
    if not GENAI_AVAILABLE:
        print("[config] google-genai SDK not importable — Gemini functionality unavailable "
              "(video search uses YouTube API directly, so this is fine).")


def main():
    print("=== Daily digest started ===")
    check_config()
    since_ts = time.time() - DAY_IN_SECONDS

    # ---- Steps 1–3: OpenRouter models ----
    models = fetch_new_models(since_ts)
    if models is None:
        print("[main] OpenRouter fetch failed — skipping to channel check.")
    elif len(models) == 0:
        send_telegram("No new models released on OpenRouter in the last 24 hours.")
        print("[main] No new models in the last 24 hours.")
    else:
        print(f"[main] Fetched {len(models)} new model(s).")
        send_telegram(format_model_summary(models))
        print("[main] Sent model summary.")
        for model in models:
            try:
                process_model(model)
            except Exception as exc:
                print(f"[main] Unexpected error processing model: {safe_err(exc)}")
            time.sleep(1)  # be gentle with YouTube + Telegram rate limits

    # ---- Step 4: tracked YouTube channels ----
    try:
        check_tracked_channels(since_ts)
    except Exception as exc:
        print(f"[main] Channel check failed: {safe_err(exc)}")

    print("=== Daily digest finished ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # last-resort guard: the script must never crash
        print(f"[fatal] Unhandled error: {safe_err(exc)}")
        try:
            send_telegram(f"⚠️ daily_digest failed: {html.escape(safe_err(exc))}")
        except Exception:
            pass