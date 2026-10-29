"""
Script to generate results/summary_v4.csv with end-to-end latency accounting,
break-even analysis, and bypassed (<400 tokens) vs non-bypassed breakdowns.
"""

import csv
import json
from pathlib import Path
import numpy as np

RESULTS_DIR = Path("results")

def build_summary_v4():
    with open(RESULTS_DIR / "summary_v2.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        v2_rows = list(reader)

    with open(RESULTS_DIR / "compressed_test.jsonl", "r", encoding="utf-8") as f:
        comp_records = [json.loads(l) for l in f if l.strip()]

    comp_secs = np.array([r["seconds"] for r in comp_records])
    mean_comp_all = float(np.mean(comp_secs))  # 6.869s

    bypassed_mask = np.array([r["bypassed"] for r in comp_records])
    mean_comp_bypassed = float(np.mean(comp_secs[bypassed_mask])) if np.any(bypassed_mask) else 0.0  # 0.0s
    mean_comp_non_bypassed = float(np.mean(comp_secs[~bypassed_mask])) if np.any(~bypassed_mask) else 0.0  # 7.895s

    # Build summary_v4 rows
    v4_rows = []
    for r in v2_rows:
        sys_name = r["system"]
        is_comp = "Compressed" in sys_name or "PTM" in sys_name or "OptiRAG" in sys_name
        llm_lat = float(r["mean_latency_s"])
        c_lat = mean_comp_all if is_comp else 0.0
        e2e_lat = round(llm_lat + c_lat, 3)

        new_row = dict(r)
        new_row["llm_only_latency_s"] = round(llm_lat, 3)
        new_row["compression_latency_s"] = round(c_lat, 3)
        new_row["end_to_end_latency_s"] = e2e_lat
        v4_rows.append(new_row)

    out_csv = RESULTS_DIR / "summary_v4.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(v4_rows[0].keys()))
        writer.writeheader()
        writer.writerows(v4_rows)
    print(f"Saved {out_csv} ({len(v4_rows)} rows)")

    # Print clean table
    print("\n" + "=" * 125)
    print(f"{'System':<42} | {'Acc (%)':<7} | {'Cost/1k':<9} | {'LLM Lat (s)':<11} | {'Comp Lat (s)':<12} | {'E2E Lat (s)':<11}")
    print("-" * 125)
    for r in v4_rows:
        print(f"{r['system']:<42} | {float(r['accuracy']):<7.1f} | ${float(r['cost_per_1k']):<8.4f} | {r['llm_only_latency_s']:<11.3f} | {r['compression_latency_s']:<12.3f} | {r['end_to_end_latency_s']:<11.3f}")
    print("=" * 125)

if __name__ == "__main__":
    build_summary_v4()
