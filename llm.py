"""
OptiRAG-lite Provider-Agnostic LLM Interface & Zero-Cost Budget Guard.

Exposes a unified interface:
    generate(tier, prompt, live=False) -> dict

Providers supported:
- 'groq': Groq Cloud SDK (Llama 3.1 8B, Llama 3.3 70B)
- 'gemini': Google Gen AI SDK (Gemini 3.5 Flash-Lite, Gemini 2.5 Flash-Lite)

Hard Zero-Cost Rules enforced:
1. Persistent usage tracking in .cache/usage.json.
2. Safety ceilings: 80% of RPM & TPM, 90% of RPD.
3. Fail-closed: Refuses live calls if tier limits contain TODO/None.
4. Sleep-based pacing to stay <= safe RPM.
5. Max 3 retries on 429 / quota errors, then raises BudgetExceeded.
6. SHA-256 disk caching in .cache/llm/ (cache hit makes zero API calls).
7. DRY_RUN is default (simulated calls only unless live=True).
8. Logs Groq x-ratelimit-* headers to .cache/ratelimit_headers.jsonl.
"""

from __future__ import annotations

import os
import json
import time
import hashlib
import contextlib
from datetime import datetime, timezone
import zoneinfo
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tiktoken
from dotenv import load_dotenv

import config

# Load environment variables
load_dotenv(config.BASE_DIR / ".env")

# Ensure required cache folders exist
config.LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

# BPE Tokenizer for accurate local token estimation
_TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")


class BudgetExceeded(Exception):
    """Raised when an API call would exceed safety limits or 429 quota is exhausted."""
    pass


class ConfigIncompleteError(Exception):
    """Raised when tier limits contain None, TODO, or placeholder values."""
    pass


class UsageCorruptedError(Exception):
    """Raised when a usage log file is corrupt or unreadable, refusing to reset counters (fail-closed)."""
    pass


# In-memory tracking for rolling RPM/TPM rate pacing and dry-run telemetry
_CALL_TIMESTAMPS: Dict[str, List[float]] = {}
_TOKEN_TIMESTAMPS: Dict[str, List[Tuple[float, int]]] = {}
_LAST_CALL_TIME: Dict[str, float] = {}
_RUN_STATS = {
    "real_calls": 0,
    "cached_calls": 0,
    "dry_run_calls": 0,
    "real_prompt_tokens": 0,
    "real_completion_tokens": 0,
}


def count_tokens(text: str) -> int:
    """Return local BPE token count approximation using cl100k_base."""
    return len(_TIKTOKEN_ENC.encode(text))


def get_cache_key(provider: str, model: str, prompt: str) -> str:
    """Compute deterministic SHA-256 cache key for prompt and model."""
    raw = f"{provider}:{model}:{prompt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_cache(provider: str, model: str, prompt: str) -> Optional[Dict[str, Any]]:
    """Look up cached response on disk."""
    key = get_cache_key(provider, model, prompt)
    cache_path = config.LLM_CACHE_DIR / f"{key}.json"
    if cache_path.is_file():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                "text": data["text"],
                "in_tokens": data["in_tokens"],
                "out_tokens": data["out_tokens"],
                "latency_s": data.get("latency_s", 0.0),
                "cached": True,
                "provider": provider,
                "model": model,
            }
        except Exception:
            return None
    return None


def write_cache(provider: str, model: str, prompt: str, text: str, in_tokens: int, out_tokens: int, latency_s: float) -> None:
    """Persist completed response to disk cache."""
    key = get_cache_key(provider, model, prompt)
    cache_path = config.LLM_CACHE_DIR / f"{key}.json"
    data = {
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "text": text,
        "in_tokens": in_tokens,
        "out_tokens": out_tokens,
        "latency_s": latency_s,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    temp_path = config.LLM_CACHE_DIR / f"{key}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    temp_path.replace(cache_path)


def get_calendar_day(provider: str) -> str:
    """
    Get current calendar day in the provider's quota timezone.
    - Gemini: Midnight US Pacific time (America/Los_Angeles).
    - Groq: Midnight UTC (standard cloud reset).
    """
    if provider.lower() == "gemini":
        tz = zoneinfo.ZoneInfo("America/Los_Angeles")
    else:
        tz = timezone.utc
    return datetime.now(tz).strftime("%Y-%m-%d")


def get_usage_file_path(tier_key: str) -> Path:
    """Return dedicated path for a tier's usage file under .cache/."""
    sanitized = tier_key.replace(":", "_").replace("/", "_")
    return config.CACHE_DIR / f"usage_{sanitized}.json"


@contextlib.contextmanager
def file_lock(lock_path: Path, timeout: float = 15.0, retry_delay: float = 0.05):
    """
    Cross-process file lock using atomic OS-level file creation (O_CREAT | O_EXCL).
    Includes automatic stale lock recovery after 30s.
    """
    start = time.time()
    fd = None
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            try:
                os.write(fd, f"{os.getpid()}:{time.time()}".encode("utf-8"))
            except Exception:
                pass
            break
        except (FileExistsError, PermissionError):
            # On Windows, opening an already-locked file raises PermissionError or FileExistsError.
            # Check for stale lock file
            try:
                if lock_path.is_file():
                    age = time.time() - lock_path.stat().st_mtime
                    if age > 30.0:
                        try:
                            lock_path.unlink()
                        except Exception:
                            pass
            except Exception:
                pass
            if time.time() - start > timeout:
                raise TimeoutError(f"Could not acquire cross-process lock on {lock_path} within {timeout}s")
            time.sleep(retry_delay)

    try:
        yield
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
            try:
                if lock_path.is_file():
                    lock_path.unlink()
            except Exception:
                pass


def _load_json_with_retry(path: Path, max_retries: int = 5, retry_delay: float = 0.05) -> Dict[str, Any]:
    """
    Load JSON from file with brief retries.
    If the file is corrupt or unreadable, raises UsageCorruptedError (fail-closed).
    """
    last_err = None
    for _ in range(max_retries):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError, PermissionError) as e:
            last_err = e
            time.sleep(retry_delay)

    raise UsageCorruptedError(
        f"Usage file '{path}' is corrupt or unreadable after {max_retries} attempts: {last_err}. "
        f"Refusing to silently reset counters to zero (fail-closed)."
    )


def load_tier_usage(tier_key: str) -> Dict[str, Dict[str, int]]:
    """
    Load persistent usage log for a specific tier_key.
    Returns {day: {'requests': int, 'tokens': int}}.
    """
    usage_file = get_usage_file_path(tier_key)
    if not usage_file.is_file():
        # Check if legacy usage.json has this tier_key
        if config.USAGE_LOG_PATH.is_file():
            legacy_all = _load_json_with_retry(config.USAGE_LOG_PATH)
            if tier_key in legacy_all:
                return legacy_all[tier_key]
        return {}

    tier_data = _load_json_with_retry(usage_file)
    if tier_key in tier_data:
        return tier_data[tier_key]
    return tier_data


def save_tier_usage(tier_key: str, days_dict: Dict[str, Dict[str, int]]) -> None:
    """Save tier usage atomically via temp file + os.replace."""
    usage_file = get_usage_file_path(tier_key)
    temp_file = usage_file.with_suffix(".tmp")
    data = {tier_key: days_dict}
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(temp_file, usage_file)


def load_usage_log(tier_key: Optional[str] = None) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Load persistent usage log. If tier_key is specified, returns {tier_key: days_dict}.
    Otherwise aggregates all tier usage files and legacy usage.json into a combined dictionary.
    """
    if tier_key is not None:
        return {tier_key: load_tier_usage(tier_key)}

    combined: Dict[str, Dict[str, Dict[str, int]]] = {}

    # 1. Load legacy usage.json if present
    if config.USAGE_LOG_PATH.is_file():
        legacy = _load_json_with_retry(config.USAGE_LOG_PATH)
        for k, v in legacy.items():
            combined[k] = v

    # 2. Load dedicated usage_*.json files
    if config.CACHE_DIR.is_dir():
        for p in config.CACHE_DIR.glob("usage_*.json"):
            if p.name == "usage.json":
                continue
            data = _load_json_with_retry(p)
            for k, v in data.items():
                combined[k] = v

    return combined


def save_usage_log(usage: Dict[str, Any]) -> None:
    """
    Save usage log atomically per tier, using cross-process locks.
    """
    for tier_key, days_dict in usage.items():
        usage_file = get_usage_file_path(tier_key)
        lock_file = usage_file.with_suffix(".lock")
        with file_lock(lock_file):
            save_tier_usage(tier_key, days_dict)


def record_usage(provider: str, model: str, in_tokens: int, out_tokens: int, day_override: Optional[str] = None) -> None:
    """Record an executed API call in persistent usage log with file lock and atomic write."""
    day = day_override or get_calendar_day(provider)
    tier_key = f"{provider}:{model}"
    usage_file = get_usage_file_path(tier_key)
    lock_file = usage_file.with_suffix(".lock")

    with file_lock(lock_file):
        days_dict = load_tier_usage(tier_key)
        if day not in days_dict:
            days_dict[day] = {"requests": 0, "tokens": 0}

        days_dict[day]["requests"] += 1
        days_dict[day]["tokens"] += (in_tokens + out_tokens)
        save_tier_usage(tier_key, days_dict)

    _RUN_STATS["real_calls"] += 1
    _RUN_STATS["real_prompt_tokens"] += in_tokens
    _RUN_STATS["real_completion_tokens"] += out_tokens


def check_tier_configured(tier: Dict[str, Any]) -> None:
    """
    FAIL-CLOSED check: Verifies all required limits exist and are not None/TODO placeholders.
    """
    required_keys = ["provider", "model", "rpm", "tpm", "rpd"]
    missing_or_todo = []
    for k in required_keys:
        val = tier.get(k)
        if val is None:
            missing_or_todo.append(f"{k}=None")
        elif isinstance(val, str) and ("TODO" in val.upper() or val.strip() == ""):
            missing_or_todo.append(f"{k}='{val}'")

    if missing_or_todo:
        raise ConfigIncompleteError(
            f"Tier '{tier.get('model', 'unknown')}' is not fully configured. "
            f"Missing or TODO fields: {', '.join(missing_or_todo)}. "
            f"Real API calls refused (fail-closed)."
        )


def check_budget_and_pace(tier: Dict[str, Any], prompt: str, day_override: Optional[str] = None) -> None:
    """
    Enforces rate-limit pacing and checks persistent usage against safety caps.
    Raises BudgetExceeded if safety limits (80% RPM/TPM, 90% RPD) would be exceeded.
    """
    check_tier_configured(tier)

    provider = tier["provider"]
    model = tier["model"]
    tier_key = f"{provider}:{model}"
    day = day_override or get_calendar_day(provider)

    # 1. Pacing & Rolling RPM Check (Safety factor: 0.8)
    safe_rpm = tier["rpm"] * config.SAFETY_FACTOR
    min_interval = 60.0 / safe_rpm
    now = time.time()

    last_call = _LAST_CALL_TIME.get(tier_key, 0.0)
    elapsed = now - last_call
    if elapsed < min_interval:
        sleep_needed = min_interval - elapsed
        time.sleep(sleep_needed)

    # Rolling 60-second window
    now = time.time()
    timestamps = _CALL_TIMESTAMPS.setdefault(tier_key, [])
    # Prune timestamps older than 60s
    timestamps[:] = [ts for ts in timestamps if now - ts < 60.0]

    if len(timestamps) >= safe_rpm:
        oldest = timestamps[0]
        wait_s = max(0.1, 60.0 - (now - oldest) + 0.1)
        if wait_s > 60.0:
            raise BudgetExceeded(
                f"RPM safety limit reached for {tier_key} ({len(timestamps)} calls in last 60s >= safe RPM {safe_rpm:.1f})."
            )
        time.sleep(wait_s)

    # 2. Rolling 60-Second TPM Check (Safety factor: 0.8)
    est_call_tokens = count_tokens(prompt) + 50
    safe_tpm = int(tier["tpm"] * config.SAFETY_FACTOR)
    token_history = _TOKEN_TIMESTAMPS.setdefault(tier_key, [])
    now = time.time()
    # Prune token entries older than 60s
    token_history[:] = [(ts, tok) for ts, tok in token_history if now - ts < 60.0]
    current_window_tokens = sum(tok for _, tok in token_history)

    if current_window_tokens + est_call_tokens > safe_tpm:
        needed_free = (current_window_tokens + est_call_tokens) - safe_tpm
        freed = 0
        wait_s = 0.1
        for ts, tok in token_history:
            freed += tok
            wait_s = max(wait_s, 60.0 - (now - ts) + 0.1)
            if freed >= needed_free:
                break
        if wait_s > 60.0:
            raise BudgetExceeded(
                f"TPM safety limit reached for {tier_key}: "
                f"current window {current_window_tokens} + est {est_call_tokens} > safe TPM {safe_tpm}."
            )
        time.sleep(wait_s)

    # 3. Check Daily Safety Limits
    tier_dict = load_tier_usage(tier_key)
    today_stats = tier_dict.get(day, {"requests": 0, "tokens": 0})
    requests_today = today_stats["requests"]
    tokens_today = today_stats["tokens"]

    # RPD safety limit: 90%
    safe_rpd = int(tier["rpd"] * config.RPD_SAFETY_FACTOR)
    if requests_today + 1 > safe_rpd:
        raise BudgetExceeded(
            f"Daily requests safety limit reached for {tier_key}: "
            f"{requests_today + 1} requests would exceed safe ceiling {safe_rpd} "
            f"(90% of {tier['rpd']} RPD). Reset at midnight."
        )

    # TPD safety limit: 80% (if defined for tier)
    if tier.get("tpd") is not None:
        safe_tpd = int(tier["tpd"] * config.SAFETY_FACTOR)
        est_tokens = count_tokens(prompt) + 50
        if tokens_today + est_tokens > safe_tpd:
            raise BudgetExceeded(
                f"Daily tokens safety limit reached for {tier_key}: "
                f"{tokens_today + est_tokens} tokens would exceed safe ceiling {safe_tpd} "
                f"(80% of {tier['tpd']} TPD). Reset at midnight."
            )


def log_groq_ratelimit_headers(model: str, headers: Any) -> None:
    """Log x-ratelimit-* headers from Groq response. Never logs API keys."""
    ratelimit_dict = {}
    for k, v in headers.items():
        if k.lower().startswith("x-ratelimit-"):
            ratelimit_dict[k.lower()] = v

    if ratelimit_dict:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "headers": ratelimit_dict,
        }
        with open(config.RATELIMIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# Optional mock client hook for unit testing without network
_MOCK_DISPATCHER = None


def set_mock_dispatcher(dispatcher_fn) -> None:
    """Hook a mock dispatcher function for testing."""
    global _MOCK_DISPATCHER
    _MOCK_DISPATCHER = dispatcher_fn


def _call_groq(model: str, prompt: str) -> Tuple[str, int, int, float]:
    """Execute real Groq API call with temperature=0 and max_tokens=50."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in environment or .env file.")

    from groq import Groq
    client = Groq(api_key=api_key)

    call_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 75 if "gpt-oss" in model.lower() else 50,
    }
    if "gpt-oss" in model.lower():
        call_kwargs["reasoning_effort"] = "low"

    t0 = time.time()
    raw_response = client.chat.completions.with_raw_response.create(**call_kwargs)
    latency = time.time() - t0

    # Log rate limit headers
    log_groq_ratelimit_headers(model, raw_response.headers)

    completion = raw_response.parse()
    text = completion.choices[0].message.content or ""
    in_tok = completion.usage.prompt_tokens
    out_tok = completion.usage.completion_tokens

    return text.strip(), in_tok, out_tok, latency


def _call_gemini(model: str, prompt: str) -> Tuple[str, int, int, float]:
    """Execute real Gemini API call with official Google Gen AI SDK."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in environment or .env file.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    # Disable thinking mode / minimal thinking budget to save tokens and latency
    # Note: API rejects thinking_budget=0 with 400 INVALID_ARGUMENT; thinking_level=MINIMAL enforces minimal thinking.
    gen_config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=50,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
    )

    t0 = time.time()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=gen_config,
    )
    latency = time.time() - t0

    # Safely handle response.text which can be None or raise on safety blocks/empty output
    try:
        raw_text = response.text
    except Exception:
        raw_text = None

    finish_reason = ""
    if response.candidates:
        c = response.candidates[0]
        finish_reason = str(getattr(c, "finish_reason", ""))

    if raw_text is None or not raw_text.strip():
        text = f"[EMPTY_RESPONSE: finish_reason={finish_reason}]" if finish_reason else ""
    else:
        text = raw_text.strip()

    in_tok = response.usage_metadata.prompt_token_count if response.usage_metadata else count_tokens(prompt)
    out_tok = response.usage_metadata.candidates_token_count if response.usage_metadata else count_tokens(text)

    return text.strip(), in_tok, out_tok, latency


def _execute_with_retry(tier: Dict[str, Any], prompt: str) -> Tuple[str, int, int, float]:
    """
    Executes real API call with at most 3 retries on 429/quota errors.
    Uses exponential backoff (2s, 4s, 8s). Never retries indefinitely.
    """
    provider = tier["provider"]
    model = tier["model"]
    tier_key = f"{provider}:{model}"

    max_retries = 3
    base_backoff = 2.0

    for attempt in range(max_retries + 1):
        try:
            if _MOCK_DISPATCHER is not None:
                return _MOCK_DISPATCHER(provider, model, prompt)

            if provider.lower() == "groq":
                return _call_groq(model, prompt)
            elif provider.lower() == "gemini":
                return _call_gemini(model, prompt)
            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            err_str = str(e).lower()
            is_429 = "429" in err_str or "rate limit" in err_str or "quota" in err_str or "resourceexhausted" in err_str

            if is_429 and attempt < max_retries:
                sleep_s = base_backoff * (2 ** attempt)
                time.sleep(sleep_s)
                continue
            elif is_429:
                raise BudgetExceeded(
                    f"Rate limit / 429 quota error encountered after {max_retries} retries for {tier_key}: {e}"
                ) from e
            else:
                raise e


def generate(tier: Dict[str, Any], prompt: str, live: bool = False) -> Dict[str, Any]:
    """
    Unified provider-agnostic LLM interface.

    Args:
        tier: Dictionary specifying tier constants (provider, model, rpm, tpm, rpd, optional tpd)
        prompt: Raw prompt text string
        live: Explicit flag to permit real network API calls. Defaults to False.

    Returns:
        dict with keys:
            text (str): Model answer
            in_tokens (int): Input prompt token count
            out_tokens (int): Output generated token count
            latency_s (float): Round-trip duration in seconds
            cached (bool): True if result was served from disk cache
            provider (str): Provider name
            model (str): Model name
    """
    provider = tier["provider"]
    model = tier["model"]
    tier_key = f"{provider}:{model}"

    # 1. Check disk cache first (zero cost, zero budget hit)
    cached_resp = read_cache(provider, model, prompt)
    if cached_resp is not None:
        _RUN_STATS["cached_calls"] += 1
        return cached_resp

    # 2. Dry Run Mode: Return simulated response without network calls
    if not live:
        _RUN_STATS["dry_run_calls"] += 1
        est_in = count_tokens(prompt)
        return {
            "text": f"[DRY_RUN SIMULATED ANSWER for {model}]",
            "in_tokens": est_in,
            "out_tokens": 20,
            "latency_s": 0.0,
            "cached": False,
            "provider": provider,
            "model": model,
        }

    # 3. Fail-Closed check and budget / pacing verification
    check_budget_and_pace(tier, prompt)

    # 4. Execute Real Call with Retry
    text, in_tok, out_tok, latency = _execute_with_retry(tier, prompt)

    # Update in-memory pacing trackers
    now = time.time()
    _LAST_CALL_TIME[tier_key] = now
    _CALL_TIMESTAMPS.setdefault(tier_key, []).append(now)
    _TOKEN_TIMESTAMPS.setdefault(tier_key, []).append((now, in_tok + out_tok))

    # 5. Record Usage to Persistent File
    record_usage(provider, model, in_tok, out_tok)

    # 6. Save to Disk Cache
    write_cache(provider, model, prompt, text, in_tok, out_tok, latency)

    return {
        "text": text,
        "in_tokens": in_tok,
        "out_tokens": out_tok,
        "latency_s": latency,
        "cached": False,
        "provider": provider,
        "model": model,
    }


def get_usage_summary() -> Dict[str, Any]:
    """Return dictionary of current run statistics and remaining daily budget."""
    usage = load_usage_log()
    tier_summaries = {}

    for tier_obj in [config.SMALL_TIER, config.LARGE_TIER, config.LARGE_TIER_GROQ]:
        p = tier_obj["provider"]
        m = tier_obj["model"]
        k = f"{p}:{m}"
        day = get_calendar_day(p)
        today = usage.get(k, {}).get(day, {"requests": 0, "tokens": 0})

        rpd_lim = tier_obj.get("rpd")
        safe_rpd = int(rpd_lim * config.RPD_SAFETY_FACTOR) if rpd_lim is not None else None
        remaining_req = max(0, safe_rpd - today["requests"]) if safe_rpd is not None else "TODO"

        tpd_lim = tier_obj.get("tpd")
        safe_tpd = int(tpd_lim * config.SAFETY_FACTOR) if tpd_lim is not None else None
        remaining_tok = max(0, safe_tpd - today["tokens"]) if safe_tpd is not None else "N/A"

        tier_summaries[k] = {
            "day": day,
            "used_requests": today["requests"],
            "safe_rpd_limit": safe_rpd,
            "remaining_requests": remaining_req,
            "used_tokens": today["tokens"],
            "safe_tpd_limit": safe_tpd,
            "remaining_tokens": remaining_tok,
        }

    return {
        "run_stats": dict(_RUN_STATS),
        "daily_tiers": tier_summaries,
    }


def print_usage_summary() -> None:
    """Print formatted summary of calls, tokens, and daily budget status."""
    summary = get_usage_summary()
    stats = summary["run_stats"]
    print("\n" + "=" * 70)
    print("OPTIRAG USAGE & BUDGET SUMMARY")
    print("=" * 70)
    print(f"Run Real Calls Made   : {stats['real_calls']}")
    print(f"Run Cached Hits        : {stats['cached_calls']}")
    print(f"Run Dry-Run Calls      : {stats['dry_run_calls']}")
    print(f"Run Real Tokens (In/Out): {stats['real_prompt_tokens']} / {stats['real_completion_tokens']}")
    print("-" * 70)
    print("Daily Budget Status (Safety Ceilings Enforced):")
    for tier_key, info in summary["daily_tiers"].items():
        print(f"[{tier_key}] (Day: {info['day']})")
        print(f"  Requests: {info['used_requests']} used / {info['safe_rpd_limit']} safe cap (Remaining: {info['remaining_requests']})")
        print(f"  Tokens  : {info['used_tokens']} used / {info['safe_tpd_limit']} safe cap (Remaining: {info['remaining_tokens']})")
    print("=" * 70 + "\n")
