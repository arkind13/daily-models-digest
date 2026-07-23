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
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # optional / future use
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

GEMINI_MODEL = "gemini-2.0-flash"

YOUTUBE_CHANNELS = [
    "UCmTM_hPCeckqN3cPWtYZZcg",  # The Deshbhakt
    "UC5l7RouTQ60oUjLjt1Nh-UQ",  # AI Revolution
    "UClXAalunTPaX1YV185DWUeg",  # Vaibhav Sisinty
    "UCYwLV1gDwzGbg7jXQ52bVnQ",  # Universe of AI
    "UC-CSyyi47VX1lD9zyeABW3w",  # Dhruv Rathee
]

# IMPORTANT: default API is text-only; "all" includes audio/image/embeddings
OPENROUTER_MODELS_URL = (
    "https://openrouter.ai/api/v1/models?sort=newest&output_modalities=all"
)
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

DAY_IN_SECONDS = 86400
TELEGRAM_MAX_LENGTH = 4096
HTTP_TIMEOUT = 30
SEPARATOR = "━━━━━━━━━━━━━━━━━━━━━━"
DESCRIPTION_MAX_CHARS = 160
SYDNEY_TZ = ZoneInfo("Australia/Sydney")

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    genai_types = None
    GENAI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def to_sydney(dt):
    """Convert a datetime to Australia/Sydney. Accepts aware or naive (assumed UTC)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(SYDNEY_TZ)


def format_sydney(dt, with_time=True):
    """
    Format a datetime in Sydney local time.
    Example: 2026-07-23 14:30 AEST   or   2026-07-23 14:30 AEDT
    """
    local = to_sydney(dt)
    if local is None:
        return "N/A"
    # %Z → AEST or AEDT depending on DST
    if with_time:
        return local.strftime("%Y-%m-%d %H:%M %Z")
    return local.strftime("%Y-%m-%d %Z")

def safe_err(exc, limit=300):
    """Truncated error string — never log full responses/URLs (may hold keys)."""
    return str(exc)[:limit]


def raise_for_status_safe(resp, label):
    """Like resp.raise_for_status() but never embeds the URL in the exception."""
    if not resp.ok:
        raise RuntimeError(f"{label} returned HTTP {resp.status_code}")


def utf16_len(text):
    """Telegram counts message length in UTF-16 code units (emoji = 2)."""
    return len(text.encode("utf-16-le")) // 2


def utf16_slice(text, utf16_offset, utf16_length):
    """Slice a string by UTF-16 code units rather than Unicode code points."""
    encoded = text.encode("utf-16-le")
    byte_start = utf16_offset * 2
    byte_end = (utf16_offset + utf16_length) * 2
    return encoded[byte_start:byte_end].decode("utf-16-le")


def split_message(text, limit=TELEGRAM_MAX_LENGTH):
    """Split text into chunks under the Telegram limit, on line boundaries."""
    if utf16_len(text) <= limit:
        return [text]
    chunks, current_lines, current_len = [], [], 0
    for line in text.split("\n"):
        while utf16_len(line) > limit:
            if current_lines:
                chunks.append("\n".join(current_lines))
                current_lines, current_len = [], 0
            lo, hi = 0, len(line)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if utf16_len(line[:mid]) <= limit:
                    lo = mid
                else:
                    hi = mid - 1
            chunks.append(line[:lo])
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


def _to_float(raw):
    """Parse an OpenRouter price string/number; return None if missing/invalid."""
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def format_price_per_million(raw):
    """USD-per-token string → human USD per 1M tokens."""
    val = _to_float(raw)
    if val is None:
        return None
    per_million = val * 1_000_000
    if per_million == 0:
        return "0.00"
    if per_million < 0.01:
        return f"{per_million:.4f}"
    if per_million < 1:
        return f"{per_million:.3f}"
    return f"{per_million:.2f}"


def format_price_per_unit(raw, unit_label):
    """USD-per-unit (image, request, etc.)."""
    val = _to_float(raw)
    if val is None:
        return None
    if val == 0:
        return f"0.00/{unit_label}"
    if val < 0.0001:
        return f"{val:.6f}/{unit_label}"
    if val < 0.01:
        return f"{val:.4f}/{unit_label}"
    return f"{val:.2f}/{unit_label}"


def format_price(raw):
    """Backward-compatible wrapper: USD per 1M tokens, or 'N/A'."""
    formatted = format_price_per_million(raw)
    return formatted if formatted is not None else "N/A"


def get_modalities(model):
    """Return (input_modalities, output_modalities) lists."""
    arch = model.get("architecture") or {}
    in_mods = arch.get("input_modalities") or []
    out_mods = arch.get("output_modalities") or []
    # Fallbacks for older/partial payloads
    if not in_mods and not out_mods:
        modality = arch.get("modality") or ""
        if "->" in str(modality):
            left, right = str(modality).split("->", 1)
            in_mods = [p.strip() for p in left.split("+") if p.strip()]
            out_mods = [p.strip() for p in right.split("+") if p.strip()]
        elif modality:
            out_mods = [str(modality)]
    if not out_mods:
        out_mods = ["text"]
    if not in_mods:
        in_mods = ["text"]
    return list(in_mods), list(out_mods)


def format_modalities(model):
    """e.g. 'text → audio' or 'text+image → text'."""
    in_mods, out_mods = get_modalities(model)
    left = "+".join(in_mods) if in_mods else "?"
    right = "+".join(out_mods) if out_mods else "?"
    return f"{left} → {right}"


def format_pricing_line(model):
    """
    Build a single human-readable pricing line from OpenRouter pricing.

    OpenRouter prices are USD per token / per image / per request / etc.
    Token prices are shown per 1M tokens; other units keep their native scale.
    """
    pricing = model.get("pricing") or {}
    if not isinstance(pricing, dict) or not pricing:
        return "N/A"

    # Accept both API snake_case and occasional camelCase from docs/SDKs
    def p(*keys):
        for k in keys:
            if k in pricing and pricing[k] is not None:
                return pricing[k]
        return None

    parts = []

    prompt = p("prompt")
    completion = p("completion")
    prompt_s = format_price_per_million(prompt)
    completion_s = format_price_per_million(completion)

    # Token I/O — show if either side exists (embeddings often have prompt only)
    if prompt_s is not None or completion_s is not None:
        if prompt_s is not None and completion_s is not None:
            if prompt_s == "0.00" and completion_s == "0.00":
                parts.append("Free")
            else:
                parts.append(f"${prompt_s} / ${completion_s} per 1M tokens (in/out)")
        elif prompt_s is not None:
            label = "Free input" if abs(_to_float(prompt) or 0) == 0 else (
                f"${prompt_s} per 1M input tokens"
            )
            parts.append(label)
        else:
            label = "Free output" if abs(_to_float(completion) or 0) == 0 else (
                f"${completion_s} per 1M output tokens"
            )
            parts.append(label)

    # Audio (STT input / TTS output) — USD per audio token, shown per 1M
    audio_in = p("audio", "audio_input")
    audio_out = p("audio_output", "audioOutput")
    audio_in_s = format_price_per_million(audio_in)
    audio_out_s = format_price_per_million(audio_out)
    if audio_in_s is not None and (_to_float(audio_in) or 0) > 0:
        parts.append(f"${audio_in_s} per 1M audio input tokens")
    if audio_out_s is not None and (_to_float(audio_out) or 0) > 0:
        parts.append(f"${audio_out_s} per 1M audio output tokens")

    # Images
    image_in = p("image")
    image_out = p("image_output", "imageOutput")
    image_tok = p("image_token", "imageToken")
    img_in_u = format_price_per_unit(image_in, "input image")
    img_out_u = format_price_per_unit(image_out, "output image")
    if img_in_u and (_to_float(image_in) or 0) > 0:
        parts.append(f"${img_in_u}")
    if img_out_u and (_to_float(image_out) or 0) > 0:
        parts.append(f"${img_out_u}")
    img_tok_s = format_price_per_million(image_tok)
    if img_tok_s is not None and (_to_float(image_tok) or 0) > 0:
        parts.append(f"${img_tok_s} per 1M image tokens")

    # Fixed per-request fee
    request = p("request")
    req_u = format_price_per_unit(request, "request")
    if req_u and (_to_float(request) or 0) > 0:
        parts.append(f"${req_u}")

    # Reasoning / cache / search — only if non-zero (keeps message short)
    extras = [
        (p("internal_reasoning", "internalReasoning"), "reasoning tokens", True),
        (p("input_cache_read", "inputCacheRead"), "cache-read tokens", True),
        (p("input_cache_write", "inputCacheWrite"), "cache-write tokens", True),
        (p("input_audio_cache", "inputAudioCache"), "audio cache tokens", True),
        (p("web_search", "webSearch"), "web search", False),
    ]
    for raw, label, per_million in extras:
        val = _to_float(raw)
        if val is None or val <= 0:
            continue
        if per_million:
            s = format_price_per_million(raw)
            if s:
                parts.append(f"${s} per 1M {label}")
        else:
            u = format_price_per_unit(raw, label)
            if u:
                parts.append(f"${u}")

    if not parts:
        # All zeros / empty → treat as free
        if any(_to_float(v) == 0 for v in pricing.values() if not isinstance(v, (list, dict))):
            return "Free"
        return "N/A"

    # De-dupe while preserving order ("Free" alone is enough)
    if parts == ["Free"] or (len(parts) == 1 and parts[0] == "Free"):
        return "Free"
    # If we already said Free for tokens but have paid extras, drop bare Free
    if parts[0] == "Free" and len(parts) > 1:
        parts = parts[1:]

    return " · ".join(parts)


def truncate_text(text, limit=DESCRIPTION_MAX_CHARS):
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def parse_datetime(value):
    """Tolerant parser for ISO-8601 / common date strings / unix timestamps."""
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
        # Newest first (API usually is, but don't rely on it after filtering)
        new_models.sort(key=lambda m: int(m.get("created") or 0), reverse=True)
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
        model_id = str(m.get("id") or "")
        name = html.escape(str(m.get("name") or model_id or "Unknown model"))
        pricing_line = html.escape(format_pricing_line(m))
        modality_line = html.escape(format_modalities(m))
        released = datetime.fromtimestamp(
            int(m.get("created") or 0), tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M UTC")
        context = m.get("context_length")
        if isinstance(context, (int, float)) and context > 0:
            context_str = f"{int(context):,}"
        else:
            context_str = "N/A"

        lines.append(f"• <b>Model:</b> {name}")
        if model_id:
            lines.append(f"  <b>ID:</b> <code>{html.escape(model_id)}</code>")
        lines.append(f"  <b>Modality:</b> {modality_line}")
        lines.append(f"  <b>Pricing:</b> {pricing_line}")
        lines.append(f"  <b>Released:</b> {released}")
        lines.append(f"  <b>Context:</b> {context_str}")

        desc = truncate_text(m.get("description") or "")
        if desc:
            lines.append(f"  <b>About:</b> {html.escape(desc)}")

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
    """
    if not YOUTUBE_API_KEY:
        print("[youtube] YOUTUBE_API_KEY not set — video search unavailable.")
        return None

    published_after = datetime.fromtimestamp(since_ts, tz=timezone.utc).isoformat()

    try:
        search_params = {
            "part": "snippet",
            "q": query,
            "order": "relevance",
            "maxResults": min(max_results * 2, 5),
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

        video_ids = []
        for item in search_items:
            vid = (item.get("id") or {}).get("videoId")
            if vid:
                video_ids.append(vid)

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
        published = format_sydney(v.get("_published_dt"))
        lines.append(
            f"👍 {format_count(v.get('likeCount'))} | "
            f"👁 {format_count(v.get('viewCount'))} | "
            f"📅 {published}"
        )
        lines.append(SEPARATOR)
    return "\n".join(lines)


def youtube_query_for_model(model):
    """
    Build a YouTube search query. Prefer the human name; strip noisy free/suffix tags.
    Fall back to the bare model slug without vendor if name is empty.
    """
    model_id = str(model.get("id") or "")
    name = str(model.get("name") or "").strip()
    if name:
        # "InclusionAI: Ling 3.0 Flash (free)" → "InclusionAI Ling 3.0 Flash"
        cleaned = re.sub(r"\s*\(free\)\s*", " ", name, flags=re.I)
        cleaned = cleaned.replace(":", " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned
    if "/" in model_id:
        return model_id.split("/", 1)[1].replace(":", " ").replace("-", " ")
    return model_id or "AI model"


def process_model(model):
    """Steps 3a–3c for a single model. Never raises."""
    model_id = str(model.get("id") or "")
    model_name = str(model.get("name") or model_id or "Unknown model")
    created_ts = int(model.get("created") or 0)

    query = youtube_query_for_model(model)
    print(f"[videos] Searching YouTube (via Data API v3) for: {model_name} (q={query!r})")
    videos = search_youtube_videos(
        query=query,
        since_ts=created_ts,  # only videos published after the model release
        max_results=2,
    )

    if videos is None:
        send_telegram(f"🎥 <b>VIDEOS FOR {html.escape(model_name)}</b>\n"
                      f"{SEPARATOR}\n"
                      "⚠️ Video search unavailable")
        return

    if not videos:
        send_telegram(f'❌ No YouTube videos found yet for "{html.escape(model_name)}"')
        print(f"[videos] No videos found for: {model_name}")
        return

    send_telegram(format_videos_message(model_name, videos))
    print(f"[videos] Sent video message for: {model_name}")

# ---------------------------------------------------------------------------
# Step 4: Tracked YouTube channels (last 24 hours)
# ---------------------------------------------------------------------------


def fetch_recent_channel_videos(channel_id, since_ts):
    """Return up to 2 videos from a channel published since `since_ts`."""
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
                "_published_dt": published_dt,
            })
    if not recent:
        return []

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
        published = format_sydney(v.get("_published_dt"))
        lines.append(f"• {html.escape(v['title'])}")
        lines.append(f"  🔗 https://youtube.com/watch?v={v['video_id']}")
        lines.append(
            f"  👁 {format_count(v['views'])} views | "
            f"👍 {format_count(v['likes'])} | "
            f"📅 {published}"
        )
        lines.append(SEPARATOR)
    return "\n".join(lines)


def check_tracked_channels(since_ts):
    if not YOUTUBE_API_KEY:
        print("[youtube] YOUTUBE_API_KEY not set — skipping channel check.")
        send_telegram("⚠️ YouTube search unavailable")
        return

    any_uploads = False
    successes = 0
    errors = []
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
        time.sleep(1)

    if errors:
        send_telegram(f"⚠️ YouTube search unavailable for {len(errors)} channel(s)")

    if not any_uploads and successes > 0:
        send_telegram("📺 No new uploads from tracked channels in the last 24 hours.")
        print("[youtube] No uploads from tracked channels in the last 24 hours.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def check_config():
    # Gemini is optional now (video path uses YouTube API).
    missing = [name for name, val in (
        ("TELEGRAM_TOKEN", TELEGRAM_TOKEN),
        ("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID),
        ("YOUTUBE_API_KEY", YOUTUBE_API_KEY),
    ) if not val]
    if missing:
        print(f"[config] Missing environment variables: {', '.join(missing)}")
    if not GEMINI_API_KEY:
        print("[config] GEMINI_API_KEY not set — optional, fine for current flow.")
    if not GENAI_AVAILABLE:
        print("[config] google-genai SDK not importable — Gemini functionality unavailable "
              "(video search uses YouTube API directly, so this is fine).")


def main():
    print("=== Daily digest started ===")
    check_config()
    since_ts = time.time() - DAY_IN_SECONDS

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
            time.sleep(1)

    try:
        check_tracked_channels(since_ts)
    except Exception as exc:
        print(f"[main] Channel check failed: {safe_err(exc)}")

    print("=== Daily digest finished ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[fatal] Unhandled error: {safe_err(exc)}")
        try:
            send_telegram(f"⚠️ daily_digest failed: {html.escape(safe_err(exc))}")
        except Exception:
            pass
