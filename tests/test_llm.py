"""
Unit tests for OptiRAG-lite LLM client & budget guard (Mock client only, zero network calls).
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

import config
import llm


@pytest.fixture(autouse=True)
def isolate_cache(tmp_path, monkeypatch):
    """Isolate LLM cache dir and usage log file for each test."""
    temp_llm_cache = tmp_path / "llm"
    temp_llm_cache.mkdir(parents=True, exist_ok=True)
    temp_usage_path = tmp_path / "usage.json"
    temp_ratelimit_log = tmp_path / "ratelimit_headers.jsonl"

    monkeypatch.setattr(config, "LLM_CACHE_DIR", temp_llm_cache)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "USAGE_LOG_PATH", temp_usage_path)
    monkeypatch.setattr(config, "RATELIMIT_LOG_PATH", temp_ratelimit_log)

    # Reset in-memory trackers
    monkeypatch.setattr(llm, "_CALL_TIMESTAMPS", {})
    monkeypatch.setattr(llm, "_LAST_CALL_TIME", {})
    monkeypatch.setattr(llm, "_RUN_STATS", {
        "real_calls": 0,
        "cached_calls": 0,
        "dry_run_calls": 0,
        "real_prompt_tokens": 0,
        "real_completion_tokens": 0,
    })

    # Speed up tests by bypassing real sleep delays
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)

    # Reset mock dispatcher
    llm.set_mock_dispatcher(None)


def test_dry_run_makes_no_calls():
    """Verify dry_run (live=False) makes zero network calls and returns simulated response."""
    mock_fn = MagicMock()
    llm.set_mock_dispatcher(mock_fn)

    tier = {
        "provider": "groq",
        "model": "allam-2-7b",
        "rpm": 30,
        "tpm": 6000,
        "rpd": 7000,
        "tpd": 500000,
    }

    res = llm.generate(tier, "What is the capital of Maine?", live=False)

    assert mock_fn.call_count == 0
    assert "[DRY_RUN SIMULATED ANSWER" in res["text"]
    assert res["cached"] is False
    assert res["latency_s"] == 0.0
    assert res["provider"] == "groq"
    assert res["model"] == "allam-2-7b"


def test_cache_hit_makes_no_network_call():
    """Verify first live call hits mock, second call returns disk cache with zero API calls."""
    mock_fn = MagicMock(return_value=("Augusta", 15, 2, 0.12))
    llm.set_mock_dispatcher(mock_fn)

    tier = {
        "provider": "groq",
        "model": "allam-2-7b",
        "rpm": 30,
        "tpm": 6000,
        "rpd": 7000,
        "tpd": 500000,
    }
    prompt = "What is the capital of Maine?"

    # First call: cache miss, invokes mock
    res1 = llm.generate(tier, prompt, live=True)
    assert res1["text"] == "Augusta"
    assert res1["cached"] is False
    assert mock_fn.call_count == 1

    # Second call: cache hit, zero mock calls
    res2 = llm.generate(tier, prompt, live=True)
    assert res2["text"] == "Augusta"
    assert res2["cached"] is True
    assert mock_fn.call_count == 1  # Still 1!


def test_fail_closed_on_todo_or_missing_limits():
    """Verify client refuses real calls if any tier limit is TODO, None, or empty."""
    mock_fn = MagicMock()
    llm.set_mock_dispatcher(mock_fn)

    # Case A: rpd is None
    tier_none = {
        "provider": "gemini",
        "model": "gemini-3.5-flash-lite",
        "rpm": 15,
        "tpm": 250000,
        "rpd": None,
    }
    with pytest.raises(llm.ConfigIncompleteError) as exc_none:
        llm.generate(tier_none, "Test prompt", live=True)
    assert "rpd=None" in str(exc_none.value)
    assert mock_fn.call_count == 0

    # Case B: model is a TODO placeholder
    tier_todo = {
        "provider": "gemini",
        "model": "TODO_AI_STUDIO_MODEL_ID",
        "rpm": 15,
        "tpm": 250000,
        "rpd": 1500,
    }
    with pytest.raises(llm.ConfigIncompleteError) as exc_todo:
        llm.generate(tier_todo, "Test prompt", live=True)
    assert "TODO" in str(exc_todo.value)
    assert mock_fn.call_count == 0


def test_budget_guard_blocks_when_over_safety_limit(monkeypatch):
    """Verify BudgetExceeded is raised when usage exceeds safety thresholds (80% / 90%)."""
    mock_fn = MagicMock(return_value=("Answer", 10, 2, 0.05))
    llm.set_mock_dispatcher(mock_fn)

    # Tier with small RPD limit: 10 RPD. With RPD_SAFETY_FACTOR = 0.9, safe ceiling is 9 calls.
    tier = {
        "provider": "groq",
        "model": "allam-2-7b",
        "rpm": 30,
        "tpm": 6000,
        "rpd": 10,
        "tpd": 100000,
    }

    # Simulate existing usage in usage.json at safe cap (9 requests)
    today = llm.get_calendar_day("groq")
    tier_key = "groq:allam-2-7b"
    llm.save_usage_log({
        tier_key: {
            today: {"requests": 9, "tokens": 500}
        }
    })

    # The 10th request would exceed safe ceiling 9 -> Must raise BudgetExceeded
    with pytest.raises(llm.BudgetExceeded) as exc:
        llm.generate(tier, "Next prompt", live=True)
    assert "Daily requests safety limit reached" in str(exc.value)
    assert mock_fn.call_count == 0


def test_daily_counter_rolls_over_on_new_day(monkeypatch):
    """Verify that full usage on day 1 does not block execution on day 2."""
    mock_fn = MagicMock(return_value=("Rollover Answer", 10, 2, 0.05))
    llm.set_mock_dispatcher(mock_fn)

    tier = {
        "provider": "groq",
        "model": "allam-2-7b",
        "rpm": 30,
        "tpm": 6000,
        "rpd": 10,
        "tpd": 100000,
    }

    tier_key = "groq:allam-2-7b"
    # Yesterday had 9 requests (fully exhausted safe quota)
    yesterday = "2026-10-27"
    today = "2026-10-28"

    llm.save_usage_log({
        tier_key: {
            yesterday: {"requests": 9, "tokens": 5000}
        }
    })

    # When date is today ("2026-10-28"), requests start at 0 and call succeeds
    monkeypatch.setattr(llm, "get_calendar_day", lambda provider: today)

    res = llm.generate(tier, "New day prompt", live=True)
    assert res["text"] == "Rollover Answer"
    assert mock_fn.call_count == 1

    # Verify usage was recorded under today, leaving yesterday intact
    usage = llm.load_usage_log()
    assert usage[tier_key][yesterday]["requests"] == 9
    assert usage[tier_key][today]["requests"] == 1


def test_429_retry_stops_after_3_tries():
    """Verify that 429 quota errors retry at most 3 times and then raise BudgetExceeded."""
    mock_fn = MagicMock(side_effect=Exception("Error 429: Rate limit exceeded or quota exhausted"))
    llm.set_mock_dispatcher(mock_fn)

    tier = {
        "provider": "groq",
        "model": "allam-2-7b",
        "rpm": 30,
        "tpm": 6000,
        "rpd": 7000,
        "tpd": 500000,
    }

    with pytest.raises(llm.BudgetExceeded) as exc:
        llm.generate(tier, "Retry test prompt", live=True)

    # Initial attempt + 3 retries = exactly 4 attempts
    assert mock_fn.call_count == 4
    assert "Rate limit / 429 quota error encountered after 3 retries" in str(exc.value)


def test_groq_rate_limit_headers_logged_without_secrets():
    """Verify Groq rate limit headers are logged to ratelimit_headers.jsonl and no keys are leaked."""
    headers = {
        "x-ratelimit-limit-requests": "14400",
        "x-ratelimit-remaining-requests": "14399",
        "authorization": "Bearer gsk_secret_12345",
        "api-key": "secret_key_abcdef",
        "content-type": "application/json",
    }
    llm.log_groq_ratelimit_headers("allam-2-7b", headers)

    assert config.RATELIMIT_LOG_PATH.is_file()
    with open(config.RATELIMIT_LOG_PATH, "r", encoding="utf-8") as f:
        line = f.readline()
        record = json.loads(line)

    assert record["model"] == "allam-2-7b"
    assert "x-ratelimit-limit-requests" in record["headers"]
    assert "x-ratelimit-remaining-requests" in record["headers"]
    # Ensure sensitive headers are NOT logged
    assert "authorization" not in record["headers"]
    assert "api-key" not in record["headers"]
    assert "gsk_secret" not in line


def test_rolling_tpm_pacing_guard(monkeypatch):
    """Verify rolling 60-second TPM pacing triggers sleep when token throughput exceeds 80% safe cap."""
    sleep_calls = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: sleep_calls.append(s))

    # Tier with 1000 TPM limit (safe ceiling is 800 TPM)
    tier = {
        "provider": "groq",
        "model": "allam-2-7b",
        "rpm": 60,
        "tpm": 1000,
        "rpd": 1000,
    }

    mock_fn = MagicMock(return_value=("Answer", 450, 10, 0.05))
    llm.set_mock_dispatcher(mock_fn)

    # First call: 450 tokens. Fits comfortably under 800 TPM.
    llm.generate(tier, "Prompt 1", live=True)
    assert mock_fn.call_count == 1
    assert len(sleep_calls) == 0

    # Second call: another 450 tokens (450 + 450 = 900 > 800 safe TPM ceiling).
    # Guard must detect that window would exceed 800 TPM and sleep until window clears!
    llm.generate(tier, "Prompt 2", live=True)
    assert mock_fn.call_count == 2
    assert len(sleep_calls) >= 1
    # Sleep duration must be positive and wait for window expiration
    assert any(s > 0.0 for s in sleep_calls)


def test_concurrent_writers_updating_counter():
    """Verify that concurrent writers updating the usage counter concurrently do not drop updates."""
    from concurrent.futures import ThreadPoolExecutor

    tier_key = "groq:allam-2-7b"
    today = llm.get_calendar_day("groq")
    calls_per_worker = 25

    def worker(worker_id: int):
        for _ in range(calls_per_worker):
            llm.record_usage("groq", "allam-2-7b", in_tokens=100, out_tokens=10)

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(worker, 1)
        f2 = executor.submit(worker, 2)
        f1.result()
        f2.result()

    tier_data = llm.load_tier_usage(tier_key)
    expected_requests = calls_per_worker * 2
    expected_tokens = expected_requests * 110

    assert tier_data[today]["requests"] == expected_requests
    assert tier_data[today]["tokens"] == expected_tokens


def test_concurrent_writers_different_tiers_isolated():
    """Verify parallel Small (Groq) and Large (Gemini) writers update independent dedicated usage files."""
    from concurrent.futures import ThreadPoolExecutor

    def small_worker():
        for _ in range(20):
            llm.record_usage("groq", "allam-2-7b", in_tokens=50, out_tokens=5)

    def large_worker():
        for _ in range(20):
            llm.record_usage("gemini", "gemini-3.5-flash-lite", in_tokens=200, out_tokens=20)

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(small_worker)
        f2 = executor.submit(large_worker)
        f1.result()
        f2.result()

    small_day = llm.get_calendar_day("groq")
    large_day = llm.get_calendar_day("gemini")

    small_data = llm.load_tier_usage("groq:allam-2-7b")
    large_data = llm.load_tier_usage("gemini:gemini-3.5-flash-lite")

    assert small_data[small_day]["requests"] == 20
    assert small_data[small_day]["tokens"] == 20 * 55

    assert large_data[large_day]["requests"] == 20
    assert large_data[large_day]["tokens"] == 20 * 220


def test_corrupt_usage_file_fails_closed():
    """Verify that a corrupted usage file raises UsageCorruptedError and refuses to reset to 0."""
    tier_key = "groq:allam-2-7b"
    usage_file = llm.get_usage_file_path(tier_key)

    # Write corrupt truncated JSON
    with open(usage_file, "w", encoding="utf-8") as f:
        f.write('{"groq:allam-2-7b": { "2026-10-28": { "requests": 50, "tokens": ')

    with pytest.raises(llm.UsageCorruptedError) as exc:
        llm.load_tier_usage(tier_key)

    assert "is corrupt or unreadable" in str(exc.value)
    assert "fail-closed" in str(exc.value)
