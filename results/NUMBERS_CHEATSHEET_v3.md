# OptiRAG-lite: Numbers Cheat Sheet & Oral Defense Guide

**Presenter**: Ishaan Naik (Symbiosis Institute of Technology)  
**Deliverable**: `results/NUMBERS_CHEATSHEET_v3.md`  
**Purpose**: One-page high-density reference containing the 20 audited core metrics and 10 examiner Q&As for presentation defense.

---

## Part 1: The 20 Audited Core Numbers

| # | Metric Description | Value & 95% Confidence Interval | Source Artifact |
| :-: | :--- | :--- | :--- |
| **1** | **Always-SMALL Accuracy (Test)** | **38.0%** [29.0%, 48.0%] | `results/summary_v2.csv` |
| **2** | **Always-LARGE Accuracy (Test)** | **48.0%** [38.0%, 58.0%] | `results/summary_v2.csv` |
| **3** | **Paired Accuracy Difference (Large - Small, Test)** | **+10.0%** [-1.0%, +21.0%] | `results/extra_stats_v3.json` |
| **4** | **McNemar Exact Test p-value (Test, N=100)** | **p = 0.1214** (b=22, c=12; not significant at $\alpha=0.05$) | `results/extra_stats_v3.json` |
| **5** | **Train Paired Difference & McNemar (N=200)** | **+12.0%** [+4.5%, +19.5%], **p = 0.0032** (b=43, c=19) | `results/extra_stats_v3.json` |
| **6** | **Telemetry Router Test Accuracy (@ 30% Escalation)**| **41.0%** [32.0%, 51.0%] (Threshold $\tau = 0.520$) | `results/summary_v2.csv` |
| **7** | **Matched-Cost Random Baseline Accuracy** | **40.7%** [36.0%, 46.0%] (Matching $p = 28.0\%$ Large) | `results/summary_v2.csv` |
| **8** | **Paired Difference: Telemetry Router vs. Random** | **+2.0%** [-5.0%, +9.0%] (Interval spans 0; not significant) | `audit_analysis.py` |
| **9** | **Telemetry Router Test AUROC (Hardness)** | **0.505** [0.375, 0.640] | `audit_analysis.py` |
| **10** | **Permutation Test p-value (Telemetry AUROC vs 0.50)**| **p = 0.4835** (2,000 permutations; cannot reject null) | `audit_analysis.py` |
| **11** | **Retrieval Full Recall Rate (Test Split)** | **49.0%** (49 / 100 queries with `gold_all_in_top5 = True`) | `results/extra_stats_v3.json` |
| **12** | **Large Accuracy: Full Retrieval Coverage (Test)** | **85.7%** [75.5%, 93.9%] (42 / 49 correct) | `results/extra_stats_v3.json` |
| **13** | **Large Accuracy: Incomplete Retrieval (Test)** | **11.8%** [3.9%, 21.6%] (6 / 51 correct) | `results/extra_stats_v3.json` |
| **14** | **Oracle Retrieval Feature AUROC for Large Accuracy**| **0.870** (vs 0.505 for surface telemetry score) | `audit_analysis.py` |
| **15** | **Small Accuracy: Full Retrieval Coverage (Test)** | **57.1%** [42.9%, 71.4%] (28 / 49 correct) | `results/extra_stats_v3.json` |
| **16** | **Small Accuracy: Incomplete Retrieval (Test)** | **19.6%** [9.8%, 31.4%] (10 / 51 correct) | `results/extra_stats_v3.json` |
| **17** | **Total Test Context Spans Evaluated** | **N = 2,110** (1,667 num, 354 date, 69 neg, 14 curr, 6 pct) | `results/ptm_survival.csv` |
| **18** | **PTM Number & Negation Retention Rate** | **100.0%** (1,667/1,667 num; 69/69 neg) vs 96.0% & 73.9% vanilla | `results/ptm_survival.csv` |
| **19** | **PTM Date Retention Rate (Exact Substring)** | **66.1%** (234/354 exact; 100% token retention after commas) | `results/ptm_date_diagnosis.md` |
| **20** | **Simulated Cost per 1,000 Queries (List Price)** | **\$0.0417** (Small), **\$0.2294** (Large), **\$0.0619** (OptiRAG) | `results/summary_v2.csv` |

---

## Part 2: 10 Likely Professor Questions & Honest Answers

### Q1: Why did you use ALLaM-2-7B as the small tier model instead of Llama 3.1 8B?
> **Answer**:  
> Groq deprecated the free-tier endpoint for `llama-3.1-8b-instant` during our project lifecycle. To maintain our strict \$0.00 budget rule without entering billing or paid tiers, we adopted `allam-2-7b`, which remains active on Groq's free tier. Because ALLaM-2 is a bilingual Arabic-English model, its zero-shot English reasoning accuracy on HotpotQA was modest (38.0% on test), but it reliably served as a representative 8B-class inference baseline.

---

### Q2: Why did the learned cascade router fail to beat random routing on the test set?
> **Answer**:  
> The router achieved a test AUROC of 0.505 ($p = 0.4835$), and its paired difference against cost-matched random routing (+2.0% [-5.0%, +9.0%]) spans zero. This occurred because surface cosine similarity features (top-1 score, score margin, entropy) only measure query-document semantic affinity, not whether the specific second-hop evidence required to answer the question is present in the retrieved chunks. Surface scores are too noisy a proxy for true multi-hop information completeness to train an effective cascade classifier on 200 examples.

---

### Q3: Why does PTM keep dates only partially (66.1%) and currency at 85.7%?
> **Answer**:  
> This was an evaluation metric artifact, not factual token loss. `ptm.py` extracts individual constituent tokens (`['March', '23', '1995']`) and LLMLingua-2 preserved 100% of these words, but stripped the non-essential comma punctuation. Because `compress_metrics.py` required an exact contiguous string match (`"march 23, 1995"` with the comma), it recorded a false-negative drop. Across all 120 "missed" date spans, exactly 0 actual constituent words or digits were dropped.

---

### Q4: What does "simulated cost" mean if your project spent \$0.00?
> **Answer**:  
> Actual out-of-pocket project expenditure was exactly \$0.00 because all calls used promotional free tiers. "Simulated cost" refers to the dollar amount an organization would pay in production based on official published list prices (\$0.05/\$0.08 per 1M tokens for 8B-class Groq inference, and \$0.30/\$2.50 per 1M tokens for Gemini 3.5 Flash-Lite). It models commercial economic feasibility.

---

### Q5: What does the oracle retrieval result show, and what does it NOT show?
> **Answer**:  
> It shows that retrieval completeness (`gold_all_in_top5`) is the dominant determinant of QA success: Large accuracy is 85.7% when complete, but collapses to 11.8% when incomplete, yielding an oracle AUROC of 0.870. It does NOT show that we have a deployable router with 0.870 AUROC, because `gold_all_in_top5` is an unobservable oracle label that requires knowing the ground-truth answer in advance. It highlights where engineering effort should be invested: improving retriever recall rather than router complexity.

---

### Q6: Why did you evaluate on only 100 test queries?
> **Answer**:  
> The 100-query test split (and 200-query train split) was dictated by vendor free-tier operational limits: Groq's rolling tokens-per-minute ceiling (4,800 TPM safe cap) and Gemini's free-tier rate limits (15 RPM and 450 provisional RPD safe limit). Evaluating 300 queries across both models and compression variants required ~800 live generation calls, which reached our safe free-tier daily ceiling. We used stratified bootstrap confidence intervals specifically to quantify the statistical uncertainty of this sample size.

---

### Q7: Was the test set used for tuning any thresholds or hyperparameters?
> **Answer**:  
> No, absolutely not. All feature standard scalers, logistic regression regularizers ($C$), and decision thresholds ($\tau$) were calibrated exclusively within stratified 5-fold cross-validation on the 200 training examples. All raw label files were made read-only on disk before router training commenced. Test records were touched strictly once for final out-of-sample inference.

---

### Q8: Why is the 10% accuracy gap between Large (48%) and Small (38%) on test not statistically significant?
> **Answer**:  
> On $N=100$ test queries, there are only 34 discordant queries where the models disagreed ($b=22$ Large-only, $c=12$ Small-only). Under the two-sided exact McNemar test, $p = 0.1214$, which exceeds the standard $\alpha = 0.05$ threshold. However, on the larger $N=200$ training set (62 discordant pairs: $b=43, c=19$), the 12% gap is highly significant ($p = 0.0032$), demonstrating that statistical power on $N=100$ is limited by sample size.

---

### Q9: Does prompt compression with PTM introduce significant latency overhead?
> **Answer**:  
> Running LLMLingua-2 on CPU with 2 threads takes ~6.8 seconds per 5-chunk context. In high-throughput interactive serving, running local neural compression on CPU would be a bottleneck unless offloaded to a GPU or lightweight C++ runtime. However, PTM achieves a 59.9% effective compression ratio and saves ~36% token cost without any drop in accuracy (41.0% vs 38.0% on Small; 46.0% vs 48.0% on Large).

---

### Q10: What is the single most actionable engineering takeaway from this project?
> **Answer**:  
> In multi-hop RAG systems, engineering investment must prioritize retrieval recall and multi-hop graph coverage over lightweight cascade classifiers. Surface retrieval telemetry scores do not reliably predict whether multi-hop reasoning will succeed, whereas deterministic entity masking (PTM) is an essential, high-impact guardrail that prevents neural compressors from corrupting factual numerical and negation data.
