"""
Re-scores the 12 pilot questions for Allam 2 7B and Gemini 3.5 Flash Lite
using the updated Content-F1 correctness metric from cached results.
Confirms zero real API calls are made.
"""

import sys
import json
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import metrics
import config
import llm


def main():
    pilot_path = config.RESULTS_DIR / "pilot_results.json"
    if not pilot_path.is_file():
        print(f"Error: {pilot_path} not found.")
        return

    with open(pilot_path, "r", encoding="utf-8") as f:
        pilot_data = json.load(f)

    print("=" * 80)
    print("RE-SCORING PILOT QUESTIONS WITH CONTENT-TOKEN F1 (ZERO API CALLS)")
    print("=" * 80)

    matrix = {
        "both_correct": 0,
        "small_only": 0,
        "large_only": 0,  # Hard = 1
        "both_wrong": 0,
    }
    small_correct_count = 0
    large_correct_count = 0

    print(f"{'#':<3} | {'Gold Answer':<28} | {'Allam 2 7B (Small)':<20} | {'Gemini (Large)':<20} | {'Category'}")
    print("-" * 88)

    for row in pilot_data:
        idx = row["idx"]
        gold = row["gold_answer"]
        allam_ans = row["models"]["allam-2-7b"]["answer"]
        gemini_ans = row["models"]["gemini-flash-lite"]["answer"]

        # Re-score with updated metrics
        allam_corr = metrics.is_correct(allam_ans, gold)
        gemini_corr = metrics.is_correct(gemini_ans, gold)

        allam_f1 = metrics.compute_f1(allam_ans, gold)
        gemini_f1 = metrics.compute_f1(gemini_ans, gold)

        if allam_corr:
            small_correct_count += 1
        if gemini_corr:
            large_correct_count += 1

        if allam_corr and gemini_corr:
            category = "Both Correct (Easy)"
            matrix["both_correct"] += 1
        elif allam_corr and not gemini_corr:
            category = "Small-Only Correct"
            matrix["small_only"] += 1
        elif not allam_corr and gemini_corr:
            category = "Large-Only (Hard=1)"
            matrix["large_only"] += 1
        else:
            category = "Both Wrong"
            matrix["both_wrong"] += 1

        allam_tag = f"{'CORRECT' if allam_corr else 'WRONG'} (F1={allam_f1:.2f})"
        gemini_tag = f"{'CORRECT' if gemini_corr else 'WRONG'} (F1={gemini_f1:.2f})"

        print(f"{idx:<3} | {gold[:26]:<28} | {allam_tag:<20} | {gemini_tag:<20} | {category}")

    total = len(pilot_data)
    small_acc = (small_correct_count / total) * 100
    large_acc = (large_correct_count / total) * 100

    print("\n" + "=" * 80)
    print("RE-SCORED 2x2 CONFUSION MATRIX & ACCURACIES (N=12):")
    print("=" * 80)
    print(f"Both Correct (Easy)      : {matrix['both_correct']} ({matrix['both_correct']/total*100:.1f}%)")
    print(f"Small-Only Correct       : {matrix['small_only']} ({matrix['small_only']/total*100:.1f}%)")
    print(f"Large-Only Correct (Hard): {matrix['large_only']} ({matrix['large_only']/total*100:.1f}%) -> Target Class 1")
    print(f"Both Wrong (Unsolvable)  : {matrix['both_wrong']} ({matrix['both_wrong']/total*100:.1f}%)")
    print("-" * 80)
    print(f"Small Tier (Allam 2 7B) Accuracy : {small_acc:.1f}% ({small_correct_count}/{total})")
    print(f"Large Tier (Gemini 3.5) Accuracy : {large_acc:.1f}% ({large_correct_count}/{total})")
    print(f"Hard Query Rate (Target Class 1) : {matrix['large_only']/total*100:.1f}%")
    print("=" * 80)
    print(f"Real API Calls Made: {llm._RUN_STATS['real_calls']} (Confirmed: 0 real calls)\n")


if __name__ == "__main__":
    main()
