# OptiRAG-lite: Numbers Cheat Sheet & Oral Defense Guide (v4 Final Audited)

**Presenter**: Ishaan Naik (Symbiosis Institute of Technology)  
**Deliverable**: `results/NUMBERS_CHEATSHEET_v4.md`  
**Purpose**: High-density reference containing the 20 audited core metrics, 13 examiner Q&As, and an explicit changelog of all claims audited.

---

## Part 1: The 20 Audited Core Numbers

| # | Metric Description | Value & 95% Confidence Interval | Source Artifact |
| :-: | :--- | :--- | :--- |
| **1** | **Always-SMALL Accuracy (Test)** | **38.0%** [29.0%, 48.0%] | `results/summary_v4.csv` |
| **2** | **Always-LARGE Accuracy (Test)** | **48.0%** [38.0%, 58.0%] | `results/summary_v4.csv` |
| **3** | **Paired Accuracy Difference (Large - Small, Test)** | **+10.0%** [-1.0%, +21.0%] | `results/extra_stats_v3.json` |
| **4** | **McNemar Exact Test p-value (Test, N=100)** | **p = 0.1214** (b=22, c=12; not significant at $\alpha=0.05$) | `results/extra_stats_v3.json` |
| **5** | **Train Paired Difference & McNemar (N=200)** | **+12.0%** [+4.5%, +19.5%], **p = 0.0032** (b=43, c=19; significant) | `results/extra_stats_v3.json` |
| **6** | **Telemetry Router Test Accuracy (@ 30% Escalation)**| **41.0%** [32.0%, 51.0%] (Threshold $\tau = 0.520$) | `results/summary_v4.csv` |
| **7** | **Matched-Cost Random Baseline Accuracy** | **40.7%** [36.0%, 46.0%] (Matching $p = 28.0\%$ Large) | `results/summary_v4.csv` |
| **8** | **Paired Difference: Telemetry Router vs. Random** | **+2.0%** [-5.0%, +9.0%] (Interval spans 0; not significant) | `audit_analysis.py` |
| **9** | **Telemetry Router Test AUROC (Hardness)** | **0.505** [0.375, 0.640] | `audit_analysis.py` |
| **10** | **Permutation Test p-value (Telemetry AUROC vs 0.50)**| **p = 0.4835** (2,000 permutations; cannot reject null) | `audit_analysis.py` |
| **11** | **Retrieval Full Recall Rate (Test Split)** | **49.0%** (49 / 100 queries with `gold_all_in_top5 = True`) | `results/extra_stats_v3.json` |
| **12** | **Large Accuracy: Full Retrieval Coverage (Test)** | **85.7%** [75.5%, 93.9%] (42 / 49 correct) | `results/extra_stats_v3.json` |
| **13** | **Large Accuracy: Incomplete Retrieval (Test)** | **11.8%** [3.9%, 21.6%] (6 / 51 correct) | `results/extra_stats_v3.json` |
| **14** | **Oracle Retrieval AUROC for Large Correctness** | **0.870** (Oracle, uses gold labels, not deployable) | `audit_analysis.py` |
| **15** | **Oracle Retrieval AUROC for Hardness (hard=1)** | **0.681** (Oracle, uses gold labels, not deployable) | `audit_analysis.py` |
| **16** | **Small Accuracy: Full vs. Incomplete Retrieval** | Full: **57.1%** [42.9%, 71.4%] vs. Incomplete: **19.6%** [9.8%, 31.4%] | `results/extra_stats_v3.json` |
| **17** | **Total Test Context Spans Evaluated** | **N = 2,110** (1,667 num, 354 date, 69 neg, 14 curr, 6 pct) | `results/ptm_survival.csv` |
| **18** | **PTM Number & Negation Retention Rate** | **100.0%** (1,667/1,667 num; 69/69 neg) vs 96.0% & 73.9% vanilla | `results/ptm_survival.csv` |
| **19** | **PTM Date Retention (Exact / Norm / Word)** | Exact: **66.1%** (234/354); Norm: **99.7%** (353/354); Word: **100.0%** (900/900) | `scratch/test_token_survival.py` |
| **20** | **Simulated Cost per 1,000 Queries (List Price)** | Small: **\$0.0417** (proxy price); Large: **\$0.2294**; OptiRAG: **\$0.0619** | `results/summary_v4.csv` |

---

## Part 2: 13 Likely Professor Questions & Honest Answers

### Q1: Why did you use ALLaM-2-7B as the small tier model instead of Llama 3.1 8B?
> **Answer**:  
> During initial setup, the `llama-3.1-8b-instant` endpoint returned a 404 error and was not available to my Groq free-tier API key. To adhere strictly to our zero-budget rule without adding paid services or billing, we selected `allam-2-7b`, which was available on the key. Because ALLaM-2 is an Arabic-English bilingual model, its English zero-shot multi-hop reasoning on HotpotQA is relatively weak (38.0% test accuracy), which is an acknowledged limitation of this setup.

---

### Q2: Why did the learned cascade router fail to beat random routing on the test set?
> **Answer**:  
> The router achieved a test AUROC of 0.505 ($p = 0.4835$), and its paired difference against cost-matched random routing (+2.0% [-5.0%, +9.0%]) spans zero. Our leading hypothesis is that surface cosine similarity telemetry (top-1 score, margin, entropy) measures general query-chunk semantic match, but cannot detect whether the specific second-hop reasoning bridge is present in the retrieved chunks. We did not test this hypothesis directly with intermediate probe classifiers, but it is strongly supported by the oracle retrieval analysis.

---

### Q3: Why does PTM keep dates only partially (66.1%) under the original metric, but 100% at the word level?
> **Answer**:  
> In `ptm.py`, multi-word spans are split into constituent word tokens (`['March', '23', '1995']`). LLMLingua-2 successfully preserved 100% of these words and digits (900/900 across all 354 date spans), but pruned the punctuation comma `,`. Because the original metric in `compress_metrics.py` checked for the literal contiguous substring `"march 23, 1995"`, it recorded a false negative for 120 spans. When punctuation is normalized, date survival rises to 99.7% (353/354), and constituent word/digit survival is exactly 100.0%.

---

### Q4: What does "simulated cost" mean if your project spent \$0.00?
> **Answer**:  
> Actual out-of-pocket project expenditure was strictly \$0.00 because all calls were executed within free-tier quotas. "Simulated cost" models what an organization would pay in production using published commercial rates: an 8B-class proxy price of \$0.05 / \$0.08 per 1M tokens for Groq Small inference, and \$0.30 / \$2.50 per 1M tokens for Gemini 3.5 Flash-Lite.

---

### Q5: What does the oracle retrieval result show, and what does it NOT show?
> **Answer**:  
> An oracle feature indicating whether all gold supporting documents were retrieved (`gold_all_in_top5`) achieves an AUROC of 0.870 for predicting Large model correctness and 0.681 for predicting query hardness ($hard = 1$). Both are oracle metrics computed using ground-truth gold labels and are strictly not deployable at inference time. They demonstrate that retrieval recall is the primary bottleneck on downstream multi-hop reasoning.

---

### Q6: Why did you evaluate on only 100 test queries?
> **Answer**:  
> The 100-query test set (and 200-query train set) was constrained by free-tier operational limits: Groq's rolling 4,800 tokens-per-minute rate limit and Gemini's 15 RPM / 450 provisional daily request ceiling. Generating labels, pilot runs, and compressed evaluation across two models required ~800 live generation calls, reaching our safe daily free-tier limit. We report bootstrap confidence intervals on all metrics to quantify this sample variance explicitly.

---

### Q7: Was the test set used for tuning any thresholds or hyperparameters?
> **Answer**:  
> No. All feature standard scalers, logistic regression regularizers ($C$), and decision thresholds ($\tau$) were calibrated exclusively within stratified 5-fold cross-validation on the 200 training examples. Raw label files were marked read-only on disk before router training began, ensuring zero test leakage.

---

### Q8: Why is the 10% accuracy gap between Large (48%) and Small (38%) on test not statistically significant?
> **Answer**:  
> On $N=100$ test queries, there were only 34 discordant queries where the models disagreed ($b=22$ Large-only, $c=12$ Small-only), yielding a two-sided McNemar exact test p-value of $p = 0.1214$. On the larger $N=200$ training set (62 discordant queries: $b=43, c=19$), the 12% gap is statistically significant ($p = 0.0032$), indicating that the lack of significance on test is driven by statistical power under $N=100$.

---

### Q9: Does prompt compression with PTM save latency or degrade accuracy?
> **Answer**:  
> Running LLMLingua-2 on CPU with 2 threads takes an average of 6.87 seconds per query (7.90s on non-bypassed queries), whereas compression saves only +0.026s of Groq LLM latency and -0.003s of Gemini LLM latency. In 87 out of 100 queries, compression time exceeds the LLM time it could save; end-to-end latency increases to ~7.0s–7.8s. For accuracy, compression did not cause statistically significant degradation: Small accuracy shifted from 38.0% to 41.0% (paired difference +3.0% [-6.0%, +12.0%]), and Large accuracy shifted from 48.0% to 46.0% (paired difference -2.0% [-8.0%, +4.0%]).

---

### Q10: What is the most defensible takeaway supported by your data?
> **Answer**:  
> First, PTM guarantees that protected numerical, currency, and negation tokens survive compression by construction, preventing the 26.1% negation drops seen in vanilla LLMLingua-2 without significantly altering QA accuracy. Second, retrieval completeness is the overwhelming empirical driver of multi-hop QA performance (85.7% accuracy when complete vs. 11.8% when incomplete). Surface retrieval telemetry alone is insufficient to beat a cost-matched random router.

---

### Q11: Does compression save latency?
> **Answer**:  
> No. On this CPU setup, compression takes an average of 6.87 seconds per query, while reducing LLM generation time by at most ~0.026 seconds. Compression is strictly a token-reduction (~40% tokens saved) and financial cost-reduction (~36% cost saved) optimization, not a latency optimization.

---

### Q12: Is the 400-token bypass rule justified by your data?
> **Answer**:  
> Yes. For the 13 test queries with context below 400 tokens, bypassing compression saved 0.00s of CPU time without sacrificing any token savings, while preserving 100% of original context fidelity (accuracy was identical at 46.2% on Small and 61.5% on Large). Bypassing short contexts avoids spending ~7-8 seconds of CPU compute on queries where potential token savings are negligible.

---

### Q13: What would you do with another week on this project?
> **Answer**:  
> I would focus on two directions: first, replace surface cosine telemetry with graph-connectivity features (e.g. checking whether retrieved chunks share entity bridges required for multi-hop reasoning), which addresses the root cause of why the surface router failed. Second, move LLMLingua-2 compression from CPU to GPU or evaluate an ONNX-runtime export to reduce compression latency from 6.9s down to <200ms, making end-to-end latency competitive.

---

## Part 3: Explicit Log of Claims Audited & Corrected

1. **Model Selection**: Corrected claim from "Llama 3.1 8B was deprecated" to "endpoint returned 404 / was not available to key". Added explicit limitation that ALLaM-2-7B is a weak English baseline.
2. **Router Efficacy**: Removed claims that learned routers "significantly outperform" baselines; disclosed that test AUROC is 0.505 ($p = 0.4835$) and paired difference against random routing (+2.0% [-5.0%, +9.0%]) spans zero.
3. **Router Failure Mechanism**: Clarified that surface telemetry missing second-hop bridge documents is a hypothesis supported by oracle retrieval analysis, not directly tested via probe models.
4. **Oracle Scope**: Disclosed that the 0.870 AUROC is an unobservable oracle metric that uses gold labels, not a deployable classifier.
5. **PTM Survival**: Disclosed that regex token preservation works by construction. Clarified date survival: 66.1% exact substring vs. 99.7% punctuation-normalized vs. 100.0% constituent word/digit survival (900/900 words).
6. **Latency Accounting**: Replaced claim that compression reduces latency with full end-to-end accounting (6.87s CPU compression overhead; break-even fails on 87% of queries).
7. **Pricing Transparency**: Explicitly labeled Groq Small tier prices (\$0.05 / \$0.08) as an 8B-class proxy price benchmarked to published rates.
