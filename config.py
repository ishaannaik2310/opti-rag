"""
OptiRAG-lite Configuration & Zero-Cost Budget Constraints.
Single source of truth for runtime constants, model tiers, and feature layouts.

FEATURE DEFINITIONS (applied in features.py):
- query-only:
    384-d MiniLM query embedding + query token count (385 features total).
- telemetry-only:
    8 retrieval-derived features only (NO query-derived features):
    1. top-1 score
    2. top-2 score
    3. top1-top2 gap
    4. top-5 mean score
    5. top-5 std score
    6. softmax entropy of scores
    7. mean pairwise cosine similarity among top-5 chunks
    8. context token count
    (8 features total).
- both:
    Concatenation of query-only and telemetry-only features (393 features total).
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache"
RESULTS_DIR = BASE_DIR / "results"
DOCS_DIR = BASE_DIR / "docs"
USAGE_LOG_PATH = CACHE_DIR / "usage.json"
LLM_CACHE_DIR = CACHE_DIR / "llm"
RATELIMIT_LOG_PATH = CACHE_DIR / "ratelimit_headers.jsonl"

# Fixed Reproducibility Seed
SEED = 42

# Dataset & Retrieval Parameters
TOTAL_SAMPLES = 300
TRAIN_SAMPLES = 200
TEST_SAMPLES = 100
TOP_K = 5
BYPASS_TOKENS = 400
COMPRESSION_RATE = 0.5

# Embedding Model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# LLMLingua-2 Model
LINGUA_MODEL_NAME = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

# =====================================================================
# HARD ZERO-COST RULES & RATE LIMIT SAFETY CONSTRAINTS
# =====================================================================
SAFETY_FACTOR = 0.8       # Never exceed 80% of published RPM and TPM
RPD_SAFETY_FACTOR = 0.9   # Never exceed 90% of published RPD
DRY_RUN = True            # Default True: simulated dry-runs only. Live API calls require --live

# SMALL TIER (Groq Allam 2 7B - active 7B model on Groq following Llama 3.1 8B deprecation)
SMALL_TIER = {
    "provider": "groq",
    "model": "allam-2-7b",
    "rpm": 30,
    "tpm": 6000,
    "rpd": 7000,
    "tpd": 500000,
}

# LARGE TIER: Google Gemini 3.5 Flash-Lite (AI Studio Free Tier)
# Limits: 15 RPM, 250,000 TPM.
# RPD = 500 (provisional, unverified; real value must be read from AI Studio)
# With RPD_SAFETY_FACTOR = 0.9, safe ceiling is 450 calls/day.
LARGE_TIER = {
    "provider": "gemini",
    "model": "gemini-3.5-flash-lite",
    "rpm": 15,
    "tpm": 250000,
    "rpd": 500,  # provisional, unverified; real value must be read from AI Studio
}

# BACKUP LARGE TIER CANDIDATE: Google Gemini 2.5 Flash-Lite
BACKUP_LARGE_TIER = {
    "provider": "gemini",
    "model": "gemini-2.5-flash-lite",
    "rpm": 10,
    "tpm": 250000,
    "rpd": None,  # TODO: Fill in exact RPD from Google AI Studio rate-limits page
}

# ALTERNATIVE LARGE TIER: Groq Llama 3.3 70B Versatile
LARGE_TIER_GROQ = {
    "provider": "groq",
    "model": "llama-3.3-70b-versatile",
    "rpm": 30,
    "tpm": 12000,
    "rpd": 1000,
    "tpd": 100000,  # 100k tokens/day ceiling
}

# Default Active Candidate for LARGE Tier
ACTIVE_LARGE_TIER = LARGE_TIER

# =====================================================================
# SIMULATED PRICING BENCHMARK (Per 1,000,000 Tokens)
# Clearly labeled: SIMULATED AT LIST PRICES, NOT BILLED ($0 actual spend)
# =====================================================================
PRICES_PER_MILLION = {
    "small": {
        "tier_name": "8B-class (Llama-3.1-8B / ALLaM-2-7B on Groq)",
        "prompt": 0.05,        # $0.05 per 1M prompt tokens
        "completion": 0.08,    # $0.08 per 1M completion tokens
        "source_url": "https://groq.com/pricing/",
        "retrieved_date": "2026-10-29",
        "billing_note": "SIMULATED AT LIST PRICES, NOT BILLED ($0 project spend)",
    },
    "large": {
        "tier_name": "Google Gemini 3.5 Flash-Lite",
        "prompt": 0.30,        # $0.30 per 1M prompt tokens
        "completion": 2.50,    # $2.50 per 1M completion tokens
        "source_url": "https://ai.google.dev/pricing",
        "retrieved_date": "2026-10-29",
        "billing_note": "SIMULATED AT LIST PRICES, NOT BILLED ($0 project spend)",
    },
}
