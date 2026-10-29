"""
OptiRAG-lite Demo Site Data Exporter (export_demo_data.py).

Reads exclusively from precomputed, read-only cached results on disk.
Makes ZERO LLM API calls and does not touch any secrets or .env files.
Exports:
1. demo_site/data/queries.json (100 test queries with full telemetry, routing, compression, and cached answers)
2. demo_site/data/summary.json (summary_v4.csv rows, extra_stats_v3.json, and simulated pricing table)
3. demo_site/data/ptm_parity.json (Python ptm.find_protected_spans ground truth on 100 test contexts + 25 edge cases)
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import config
import features
import metrics
import ptm


EDGE_CASE_STRINGS: List[str] = [
    "",
    "March 4, 2024",
    "03/04/2024",
    "2024-03-04",
    "4 March 2024",
    "Q3 2023",
    "1999",
    "in the 1990s",
    "$1.5M",
    "EUR 20",
    "20 USD",
    "Rs. 500",
    "€50",
    "£100",
    "₹1,000",
    "15%",
    "15 percent",
    "1,250,000",
    "3.14159",
    "3rd",
    "They don't want anything.",
    "He cannot and will not leave, nor will she.",
    "There was nobody and nothing found nowhere.",
    "This is another notable announcement about a normal notice for each person in the 5th percentile.",
    "Neither Alice nor Bob was present except on Friday, 12th August 2022, paying $50.75 without delay."
]


def compute_cost(in_tokens: int, out_tokens: int, tier: str) -> Tuple[float, float]:
    """Return (cost_per_query, cost_per_1k) using config list prices."""
    cfg = config.PRICES_PER_MILLION[tier]
    p_cost = (in_tokens / 1e6) * cfg["prompt"]
    c_cost = (out_tokens / 1e6) * cfg["completion"]
    total = float(p_cost + c_cost)
    return round(total, 7), round(total * 1000.0, 4)


def export_data():
    out_dir = PROJECT_ROOT / "demo_site" / "data"
    assets_dir = PROJECT_ROOT / "demo_site" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading source result files...")
    # 1. Labels Test
    labels_test_path = PROJECT_ROOT / "results" / "labels_test.jsonl"
    with open(labels_test_path, "r", encoding="utf-8") as f:
        labels_test = [json.loads(line) for line in f if line.strip()]

    # 2. Compressed Test
    comp_test_path = PROJECT_ROOT / "results" / "compressed_test.jsonl"
    with open(comp_test_path, "r", encoding="utf-8") as f:
        comp_test_dict = {json.loads(line)["question_id"]: json.loads(line) for line in f if line.strip()}

    # 3. Small Compressed Test
    small_comp_path = PROJECT_ROOT / "results" / "labels_small_test_compressed.jsonl"
    with open(small_comp_path, "r", encoding="utf-8") as f:
        small_comp_dict = {json.loads(line)["question_id"]: json.loads(line) for line in f if line.strip()}

    # 4. Large Compressed Test
    large_comp_path = PROJECT_ROOT / "results" / "labels_large_test_compressed.jsonl"
    with open(large_comp_path, "r", encoding="utf-8") as f:
        large_comp_dict = {json.loads(line)["question_id"]: json.loads(line) for line in f if line.strip()}

    # 5. Router Test Probs
    probs_path = PROJECT_ROOT / "results" / "router_test_probs.csv"
    router_probs_dict = {}
    with open(probs_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            router_probs_dict[row["question_id"]] = {
                "prob_heuristic": float(row["prob_heuristic"]),
                "prob_query_only_logistic": float(row["prob_query_only_logistic"]),
                "prob_query_only_mlp": float(row["prob_query_only_mlp"]),
                "prob_telemetry_only_logistic": float(row["prob_telemetry_only_logistic"]),
                "prob_telemetry_only_mlp": float(row["prob_telemetry_only_mlp"]),
                "prob_both_logistic": float(row["prob_both_logistic"]),
                "prob_both_mlp": float(row["prob_both_mlp"]),
                "hard_gold": int(float(row["hard_gold"])),
            }

    print(f"Loaded {len(labels_test)} test queries.")

    print("[2/5] Building queries.json...")
    queries_data: List[Dict[str, Any]] = []
    parity_contexts: List[Dict[str, Any]] = []

    for idx, rec in enumerate(labels_test):
        qid = rec["question_id"]
        q_text = rec["question"]
        gold = rec["gold_answer"]
        gold_all = bool(rec.get("gold_all_in_top5", False))

        comp_info = comp_test_dict.get(qid, {})
        comp_chunks = comp_info.get("retrieved_chunks", [])
        raw_chunks = rec.get("retrieved_chunks", [])

        # Reconstruct chunks with text, title, score
        structured_chunks = []
        for i, c in enumerate(raw_chunks):
            # c is [title, score]
            title = c[0] if isinstance(c, (list, tuple)) else str(c)
            score = float(c[1]) if isinstance(c, (list, tuple)) and len(c) > 1 else 0.0
            
            # Find matching text from comp_chunks [text, title]
            chunk_text = ""
            for cc in comp_chunks:
                if isinstance(cc, (list, tuple)) and len(cc) >= 2 and cc[1] == title:
                    chunk_text = cc[0]
                    break
            if not chunk_text and i < len(comp_chunks):
                chunk_text = comp_chunks[i][0]

            structured_chunks.append({
                "rank": i + 1,
                "title": title,
                "score": round(score, 4),
                "text": chunk_text
            })

        # Format original context string
        context_blocks = [
            f"[{c['rank']}] Title: {c['title']}\n{c['text']}"
            for c in structured_chunks
        ]
        original_context = "\n\n".join(context_blocks)

        # Telemetry Features (8 scalar features)
        # 1. top-1 score, 2. top-2 score, 3. gap, 4. mean, 5. std, 6. softmax entropy, 7. chunk sim mean, 8. context tokens
        telemetry_raw = features.extract_telemetry_features(raw_chunks)
        telemetry_feats = {
            "top1_score": round(float(telemetry_raw[0]), 4),
            "top2_score": round(float(telemetry_raw[1]), 4),
            "score_gap": round(float(telemetry_raw[2]), 4),
            "mean_score": round(float(telemetry_raw[3]), 4),
            "score_std": round(float(telemetry_raw[4]), 4),
            "softmax_entropy": round(float(telemetry_raw[5]), 4),
            "chunk_sim_mean": round(float(telemetry_raw[6]), 4),
            "context_tokens": int(telemetry_raw[7]),
        }

        # Router Probabilities
        r_probs = router_probs_dict.get(qid, {})
        router_data = {
            "query_only": {
                "logistic": round(r_probs.get("prob_query_only_logistic", 0.5), 4),
                "mlp": round(r_probs.get("prob_query_only_mlp", 0.5), 4),
                "variant_name": "Query-Only (385-d MiniLM + query length)",
            },
            "telemetry_only": {
                "logistic": round(r_probs.get("prob_telemetry_only_logistic", 0.5), 4),
                "mlp": round(r_probs.get("prob_telemetry_only_mlp", 0.5), 4),
                "variant_name": "Telemetry-Only (8-d retrieval confidence signals)",
            },
            "both": {
                "logistic": round(r_probs.get("prob_both_logistic", 0.5), 4),
                "mlp": round(r_probs.get("prob_both_mlp", 0.5), 4),
                "variant_name": "Both (393-d concatenated features)",
            },
            "heuristic": {
                "prob": round(r_probs.get("prob_heuristic", 0.7), 4),
                "predict_large": bool(r_probs.get("prob_heuristic", 0.7) >= 0.70),
                "variant_name": "Heuristic Rule (Top-1 score < 0.70 or keyword match)",
            },
            "hard_gold": r_probs.get("hard_gold", rec.get("hard", 0)),
        }

        # PTM Spans on original context
        spans = ptm.find_protected_spans(original_context)
        ptm_spans_list = [
            {"start": s[0], "end": s[1], "kind": s[2], "text": original_context[s[0]:s[1]]}
            for s in spans
        ]

        # For parity ground truth
        parity_contexts.append({
            "id": qid,
            "text": original_context,
            "spans": [{"start": s[0], "end": s[1], "kind": s[2]} for s in spans]
        })

        # Compression Data
        compressed_text = comp_info.get("compressed_context", original_context)
        orig_tokens = comp_info.get("original_tokens", rec.get("prompt_tokens", 700))
        comp_tokens = comp_info.get("compressed_tokens", orig_tokens)
        bypassed = bool(comp_info.get("bypassed", False))
        comp_seconds = round(float(comp_info.get("seconds", 0.0)), 3)
        pct_saved = round((1.0 - (comp_tokens / max(orig_tokens, 1))) * 100.0, 1)

        # Cached Answers (4 combinations)
        # 1. small_uncompressed
        s_uncomp_ans = rec.get("small_answer", "")
        s_uncomp_in = rec.get("small_in_tokens", orig_tokens)
        s_uncomp_out = rec.get("small_out_tokens", 10)
        s_uncomp_lat = round(float(rec.get("small_latency_s", 0.15)), 3)
        s_uncomp_corr = bool(rec.get("small_correct", metrics.is_correct(s_uncomp_ans, gold)))
        s_uncomp_cpq, s_uncomp_cp1k = compute_cost(s_uncomp_in, s_uncomp_out, "small")

        # 2. small_compressed
        s_comp_rec = small_comp_dict.get(qid, {})
        s_comp_ans = s_comp_rec.get("answer", s_uncomp_ans)
        s_comp_in = s_comp_rec.get("in_tokens", comp_tokens)
        s_comp_out = s_comp_rec.get("out_tokens", s_uncomp_out)
        s_comp_lat = round(float(s_comp_rec.get("latency_s", 0.14)), 3)
        s_comp_corr = bool(metrics.is_correct(s_comp_ans, gold))
        s_comp_cpq, s_comp_cp1k = compute_cost(s_comp_in, s_comp_out, "small")

        # 3. large_uncompressed
        l_uncomp_ans = rec.get("large_answer", "")
        l_uncomp_in = rec.get("large_in_tokens", orig_tokens)
        l_uncomp_out = rec.get("large_out_tokens", 10)
        l_uncomp_lat = round(float(rec.get("large_latency_s", 0.90)), 3)
        l_uncomp_corr = bool(rec.get("large_correct", metrics.is_correct(l_uncomp_ans, gold)))
        l_uncomp_cpq, l_uncomp_cp1k = compute_cost(l_uncomp_in, l_uncomp_out, "large")

        # 4. large_compressed
        l_comp_rec = large_comp_dict.get(qid, {})
        l_comp_ans = l_comp_rec.get("answer", l_uncomp_ans)
        l_comp_in = l_comp_rec.get("in_tokens", comp_tokens)
        l_comp_out = l_comp_rec.get("out_tokens", l_uncomp_out)
        l_comp_lat = round(float(l_comp_rec.get("latency_s", 0.85)), 3)
        l_comp_corr = bool(metrics.is_correct(l_comp_ans, gold))
        l_comp_cpq, l_comp_cp1k = compute_cost(l_comp_in, l_comp_out, "large")

        answers = {
            "small_uncompressed": {
                "tier": "small",
                "tier_name": "Small Tier",
                "model_name": "allam-2-7b",
                "provider": "groq",
                "compressed": False,
                "answer": s_uncomp_ans,
                "is_correct": s_uncomp_corr,
                "in_tokens": s_uncomp_in,
                "out_tokens": s_uncomp_out,
                "llm_latency_s": s_uncomp_lat,
                "compression_s": 0.0,
                "e2e_latency_s": s_uncomp_lat,
                "cost_per_query": s_uncomp_cpq,
                "cost_per_1k": s_uncomp_cp1k,
            },
            "small_compressed": {
                "tier": "small",
                "tier_name": "Small Tier",
                "model_name": "allam-2-7b",
                "provider": "groq",
                "compressed": True,
                "answer": s_comp_ans,
                "is_correct": s_comp_corr,
                "in_tokens": s_comp_in,
                "out_tokens": s_comp_out,
                "llm_latency_s": s_comp_lat,
                "compression_s": comp_seconds,
                "e2e_latency_s": round(s_comp_lat + comp_seconds, 3),
                "cost_per_query": s_comp_cpq,
                "cost_per_1k": s_comp_cp1k,
            },
            "large_uncompressed": {
                "tier": "large",
                "tier_name": "Large Tier",
                "model_name": "gemini-3.5-flash-lite",
                "provider": "gemini",
                "compressed": False,
                "answer": l_uncomp_ans,
                "is_correct": l_uncomp_corr,
                "in_tokens": l_uncomp_in,
                "out_tokens": l_uncomp_out,
                "llm_latency_s": l_uncomp_lat,
                "compression_s": 0.0,
                "e2e_latency_s": l_uncomp_lat,
                "cost_per_query": l_uncomp_cpq,
                "cost_per_1k": l_uncomp_cp1k,
            },
            "large_compressed": {
                "tier": "large",
                "tier_name": "Large Tier",
                "model_name": "gemini-3.5-flash-lite",
                "provider": "gemini",
                "compressed": True,
                "answer": l_comp_ans,
                "is_correct": l_comp_corr,
                "in_tokens": l_comp_in,
                "out_tokens": l_comp_out,
                "llm_latency_s": l_comp_lat,
                "compression_s": comp_seconds,
                "e2e_latency_s": round(l_comp_lat + comp_seconds, 3),
                "cost_per_query": l_comp_cpq,
                "cost_per_1k": l_comp_cp1k,
            },
        }

        query_item = {
            "id": qid,
            "idx": idx,
            "question": q_text,
            "gold_answer": gold,
            "gold_all_in_top5": gold_all,
            "chunks": structured_chunks,
            "telemetry_features": telemetry_feats,
            "router_probs": router_data,
            "original_context": original_context,
            "original_tokens": orig_tokens,
            "compressed_context": compressed_text,
            "compressed_tokens": comp_tokens,
            "tokens_saved_pct": pct_saved,
            "bypassed": bypassed,
            "compression_seconds": comp_seconds,
            "ptm_protected_spans": ptm_spans_list,
            "answers": answers,
        }
        queries_data.append(query_item)

    queries_out_path = out_dir / "queries.json"
    with open(queries_out_path, "w", encoding="utf-8") as f:
        json.dump(queries_data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(queries_data)} queries to {queries_out_path} ({queries_out_path.stat().st_size / 1024:.1f} KB)")

    print("[3/5] Building summary.json...")
    # Load summary_v4.csv
    summary_v4_path = PROJECT_ROOT / "results" / "summary_v4.csv"
    summary_rows = []
    with open(summary_v4_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            summary_rows.append(r)

    # Load extra_stats_v3.json
    extra_stats_path = PROJECT_ROOT / "results" / "extra_stats_v3.json"
    with open(extra_stats_path, "r", encoding="utf-8") as f:
        extra_stats = json.load(f)

    # Price table with mandatory labels
    price_table = {
        "disclaimer_label": "simulated list price, not billed",
        "small_proxy_label": "small-model price is a proxy",
        "small_tier": {
            "name": "ALLaM-2-7B on Groq Free Tier",
            "billed_cost": "$0.00 (Groq Free Tier, 0 billed spend)",
            "proxy_pricing_label": "small-model price is a proxy (8B-class list proxy)",
            "prompt_per_million": 0.05,
            "completion_per_million": 0.08,
            "source": "https://groq.com/pricing/",
        },
        "large_tier": {
            "name": "Gemini 3.5 Flash-Lite on Google AI Studio",
            "billed_cost": "$0.00 (AI Studio Free Tier, 0 billed spend)",
            "list_pricing_label": "simulated list price, not billed",
            "prompt_per_million": 0.30,
            "completion_per_million": 2.50,
            "source": "https://ai.google.dev/pricing",
        },
    }

    summary_data = {
        "benchmark_table": summary_rows,
        "extra_stats": extra_stats,
        "prices": price_table,
    }

    summary_out_path = out_dir / "summary.json"
    with open(summary_out_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"Wrote summary data to {summary_out_path} ({summary_out_path.stat().st_size / 1024:.1f} KB)")

    print("[4/5] Building ptm_parity.json...")
    edge_cases_data = []
    for s in EDGE_CASE_STRINGS:
        spans = ptm.find_protected_spans(s)
        edge_cases_data.append({
            "text": s,
            "spans": [{"start": sp[0], "end": sp[1], "kind": sp[2]} for sp in spans]
        })

    parity_data = {
        "test_contexts": parity_contexts,
        "edge_cases": edge_cases_data,
    }

    parity_out_path = out_dir / "ptm_parity.json"
    with open(parity_out_path, "w", encoding="utf-8") as f:
        json.dump(parity_data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(parity_contexts)} contexts + {len(edge_cases_data)} edge cases to {parity_out_path} ({parity_out_path.stat().st_size / 1024:.1f} KB)")

    print("[5/5] Copying figures to demo_site/assets/...")
    fig_sources = [
        ("results/retrieval_accuracy_v3.png", "demo_site/assets/retrieval_accuracy_v3.png"),
        ("results/ptm_survival_v4.png", "demo_site/assets/ptm_survival_v4.png"),
        ("results/pareto_v4.png", "demo_site/assets/pareto_v4.png"),
    ]
    for src_rel, dst_rel in fig_sources:
        src_path = PROJECT_ROOT / src_rel
        dst_path = PROJECT_ROOT / dst_rel
        if src_path.is_file():
            shutil.copy2(src_path, dst_path)
            print(f"Copied {src_rel} -> {dst_rel} ({dst_path.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"WARNING: Missing figure {src_rel}")

    # Report total size of demo_site/
    demo_dir = PROJECT_ROOT / "demo_site"
    total_bytes = sum(f.stat().st_size for f in demo_dir.rglob("*") if f.is_file())
    total_mb = total_bytes / (1024 * 1024)
    print("=" * 60)
    print(f"DEMO_SITE TOTAL SIZE: {total_mb:.2f} MB ({total_bytes:,} bytes)")
    print(f"Size constraint check (<15 MB): {'PASSED' if total_mb < 15.0 else 'FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    export_data()
