# OptiRAG-lite Pipeline Execution Status Log

> **Protocol Note**: This log is strictly append-only. Every pipeline action, quota usage checkpoint, file creation, and milestone update is recorded chronologically. Zero secrets or keys are ever logged.

---

### [2026-10-29 04:00 IST] - Pipeline Initialization & Snapshot Backup
- **Action**: Safety Protocol A executed before any pipeline operations.
- **Backup Snapshot**: `backups/20261029_0400/` created and verified.
  - Copied: `results/`, `tests/`, `config.py`, `requirements.txt`, all root `*.py` files, and `.cache/llm/` (606 total files).
  - Verified: `.env` is NOT copied (file does not exist in backup).
- **Module 5 Status (Step 1)**:
  - `compress.py`, `compress_metrics.py`, and `tests/test_compress.py` complete (5/5 tests passing).
  - Compression pilot on 20 train queries evaluated at rate 0.5 across `none`, `ptm`, and `ptm_ent`.
  - Saved to `results/compress_pilot.json`.
  - PTM preserved 100% of currency, negation, and number tokens (overall entity survival 95.4% vs 91.4% baseline `none`), with ~40% token savings and ~7.4s latency on CPU (`torch.set_num_threads(2)`).
- **Next Action**: Monitor detached labeling workers until all 4 label files reach target row counts (200 train, 100 test per tier).

---

### [2026-10-29 04:12 IST] - Step 2: Label Merging & Hardness Analysis Complete
- **Labeling Status**: All 4 raw label files reached full targets (Small Train=200, Small Test=100, Large Train=200, Large Test=100).
- **Protection**: All 4 files marked read-only on disk per Safety Protocol A.3.
- **Merge Operation**: Executed merge_labels.py --split train and merge_labels.py --split test.
- **Train Split (N=200)**:
  - Both Correct (Easy)     : 75 (37.5%)
  - Small-Only Correct      : 19 (9.5%)
  - Large-Only Correct (Hard): 43 (21.5%) -> Class 1 (43 >= 15, eligible for both Logistic Regression and PyTorch MLP)
  - Both Wrong (Unsolvable) : 63 (31.5%)
  - Small Accuracy: 47.0%, Large Accuracy: 59.0%
  - Empty Responses: 0 (Small=0, Large=0)
  - Retrieval Stats: Any Gold In Top-5 = 97.5%, All Gold In Top-5 = 49.5%
  - Accuracy with Full Retrieval Coverage (99/200): Small=71.7%, Large=87.9%
  - Accuracy with Partial/No Coverage (101/200): Small=22.8%, Large=30.7%
- **Test Split (N=100)**:
  - Both Correct (Easy)     : 26 (26.0%)
  - Small-Only Correct      : 12 (12.0%)
  - Large-Only Correct (Hard): 22 (22.0%) -> Class 1
  - Both Wrong (Unsolvable) : 40 (40.0%)
  - Small Accuracy: 38.0%, Large Accuracy: 48.0%
  - Empty Responses: 0 (Small=0, Large=0)
  - Retrieval Stats: Any Gold In Top-5 = 99.0%, All Gold In Top-5 = 49.0%
  - Accuracy with Full Retrieval Coverage (49/100): Small=57.1%, Large=85.7%
  - Accuracy with Partial/No Coverage (51/100): Small=19.6%, Large=11.8%
- **Backup**: Created snapshot ackups/20261029_0412/ (676 files).
- **Next Step**: Step 3 (Test context compression & compressed evaluation calls).


### [2026-10-29 04:21 IST] - Step 4: Feature Extraction & Router Training Complete
- **Feature Representations**:
  - query-only: 385 features (384-d MiniLM embedding + 1-d token count).
  - 	elemetry-only: 8 features (top-1, top-2, gap, mean, std, softmax entropy, mean pairwise chunk cosine similarity, context token count).
  - oth: 393 features.
- **Strict Protocol Compliance**:
  - Scalers fit strictly on training folds / training set (zero test leakage).
  - Hyperparameter C for Logistic Regression tuned exclusively via Stratified 5-Fold CV on 
esults/labels_train.jsonl (43 hard=1, 157 hard=0).
- **Router Evaluation Summary**:
  - Heuristic Baseline: Train AUROC = 0.487, Test AUROC = 0.518, Test PR-AUC = 0.226.
  - Query-Only (Logistic, C=0.5): CV AUROC = 0.504 +/- 0.116, Test AUROC = 0.505, Test PR-AUC = 0.225.
  - Query-Only (PyTorch MLP): Test AUROC = 0.465, Test PR-AUC = 0.201.
  - Telemetry-Only (Logistic, C=0.1): CV AUROC = 0.571 +/- 0.033, Test AUROC = 0.505, Test PR-AUC = 0.302.
  - Telemetry-Only (PyTorch MLP): Test AUROC = 0.546, Test PR-AUC = 0.260.
  - Both (Logistic, C=0.5): CV AUROC = 0.525 +/- 0.126, Test AUROC = 0.522, Test PR-AUC = 0.235.
  - Both (PyTorch MLP): Test AUROC = 0.499, Test PR-AUC = 0.244.
- **Key Empirical Finding**:
  - 	elemetry-only features achieved the highest CV AUROC (0.571 +/- 0.033) and highest CV PR-AUC (0.317 +/- 0.065), and the PyTorch MLP on telemetry features achieved the highest Test AUROC (0.546). Retrieval confidence signals provide clearer hardness indicators than raw query semantics.
- **Outputs Saved**:
  - 
esults/router_test_probs.csv (contains predicted probabilities across all models and variants for all 100 test queries).
- **Unit Tests**: 13/13 passing in 	ests/test_features.py and 	ests/test_router.py.
- **Backup**: Created snapshot ackups/20261029_0421/ (680 files).


### [2026-10-29 05:11 IST] - Step 5: Full Benchmark Evaluation & Pareto Sweep Complete
- **Systems Evaluated on 100 Test Queries**:
  - Always-SMALL (llam-2-7b on Groq): Acc = 38.0% [29.0, 48.0], F1 = 0.346, Cost/1k = .0417, Latency = 0.17s.
  - Always-LARGE (gemini-3.5-flash-lite on AI Studio): Acc = 48.0% [38.0, 58.0], F1 = 0.452, Cost/1k = .2294, Latency = 0.92s.
  - Heuristic Router (threshold 0.70 / keyword): Acc = 46.0% [37.0, 56.0], F1 = 0.436, Cost/1k = .1713, Latency = 0.68s.
  - Random Routing (matched at 70% Large): Acc = 45.0% [36.0, 54.0], F1 = 0.423, Cost/1k = .1739, Latency = 0.70s.
  - Router: Query-Only (LogReg): Acc = 43.0% [34.0, 53.0], F1 = 0.392, Cost/1k = .0789, Latency = 0.33s.
  - Router: Telemetry-Only (LogReg): Acc = 44.0% [35.0, 54.0], F1 = 0.404, Cost/1k = .1164, Latency = 0.45s.
  - Router: Both (LogReg): Acc = 42.0% [33.0, 52.0], F1 = 0.382, Cost/1k = .0790, Latency = 0.33s.
  - Router: Telemetry-Only (MLP): Acc = 43.0% [34.0, 53.0], F1 = 0.405, Cost/1k = .1888, Latency = 0.75s.
  - Always-SMALL (PTM-Compressed): Acc = 41.0% [32.0, 51.0], F1 = 0.389, Cost/1k = .0271, Latency = 0.15s.
  - OptiRAG Pipeline (Telemetry Router + PTM): Acc = 42.0% [33.0, 52.0], F1 = 0.405, Cost/1k = .0741, Latency = 0.43s.

#### Honest 'What We Found' Analysis:
1. **Pareto Frontier Position**:
   - The Telemetry-Only Logistic Router achieves a strong trade-off: 44.0% accuracy at .1164 per 1k queries, cutting inference cost by 49.3% and latency by 51.1% compared to Always-Large (48.0%, .2294), while beating Always-Small by +6.0% accuracy.
   - The OptiRAG pipeline (Telemetry Router + PTM Compression) further reduces cost to .0741 per 1k queries (a 67.7% reduction vs. Always-Large) with 42.0% accuracy.
2. **Telemetry vs. Query-Only Features**:
   - In 5-fold cross-validation on train, Telemetry-Only significantly outperformed Query-Only (CV AUROC: 0.571 +/- 0.033 vs. 0.504 +/- 0.116; CV PR-AUC: 0.317 +/- 0.065 vs. 0.251 +/- 0.052).
   - On the test set, Telemetry-Only achieved the highest PR-AUC (0.302 vs 0.225) and higher accuracy (44.0% vs 43.0%), confirming that low-dimensional retrieval confidence metrics generalise better than 385-d dense query embeddings on small samples.
3. **Statistical Uncertainty & Inconclusive Comparisons**:
   - With N=100 test queries, bootstrap 95% confidence intervals on accuracy span approximately +/- 9 percentage points (e.g. Always-Large [38.0%, 58.0%], Telemetry Router [35.0%, 54.0%]). The 4% gap between Telemetry Router and Always-Large is within statistical margin of error.
   - The Heuristic router achieved 46.0% accuracy, but at a much higher cost (.1713 vs .1164). At that expenditure level, Random Routing achieved 45.0%, indicating that the heuristic is only marginally better than random assignment.
4. **Surprising Finding on PTM Compression**:
   - For Always-Small, PTM context compression actually *increased* accuracy from 38.0% to 41.0%, while decreasing token cost from .0417 to .0271 per 1k queries. Pruning peripheral distractor text while preserving core entities prevented the small model from being distracted by irrelevant background text.

- **Generated Artifacts**:
  - 
esults/summary.csv
  - 
esults/ablation.csv
  - 
esults/pareto.png
  - 
esults/entity_survival.png
- **Backup**: Created snapshot ackups/20261029_0511/ (865 files).



---

### [2026-10-29 09:00 IST] - Step 6: Post-Hoc Mathematical & Statistical Audit Complete
- **Backup**: Created pre-audit snapshot `backups/20261029_0846/` (865 files, `.env` excluded).
- **Audit Mandate**: Zero LLM API calls, zero modification to existing results files, strict creation of `_v2` artifacts only.
- **Key Audit Findings & Actions**:
  1. **Threshold Leakage Audit**: Confirmed original default threshold tau = 0.50 was not test-tuned, but lacked principled calibration. Established train-only thresholds via Stratified 5-Fold CV on train set only (targeting 30% escalation quantile: Query-Only tau=0.244 -> 26% test; Telemetry-Only tau=0.520 -> 28% test; Both tau=0.314 -> 25% test; Max Youden J: Telemetry tau=0.490 -> 41% test). Saved to `results/summary_v2.csv`.
  2. **Matched-Cost Random Routing Baseline**: Simulated 1,000 independent random assignments matching each router's escalation fraction. No learned router or heuristic router statistically significantly beats cost-matched random routing on test (Paired diffs: Heuristic +1.0% [-3%, +5%]; Query-Only -1.0% [-8%, +6%]; Telemetry-Only +2.0% [-5%, +9%]; Both +2.0% [-5%, +9%]; all 95% CIs span zero).
  3. **Router AUROC & Permutation Test**: Test AUROC for Telemetry Router was 0.505 [0.375, 0.640]. Permutation test (2,000 resamples) yielded p = 0.4835 (null hypothesis AUROC = 0.50 cannot be rejected).
  4. **Pricing Disclosures**: Disclosed that `allam-2-7b` on Groq has $0 billed inference cost; $0.05 / $0.08 rate is an 8B-class proxy price. Sensitivity analysis confirmed that even at 2x Small price ($0.10 / $0.16), Telemetry Router preserves a 44.9% cost reduction vs Always-Large.
  5. **PTM Disclosures & Span Counts**: Explicitly disclosed that regex masking preserves tokens by construction. Verified total spans (N=2,110 across 100 test contexts): Numbers (n=1,667, 100% PTM vs 96.0% vanilla), Dates (n=354, 66.1% PTM vs 57.9% vanilla), Negations (n=69, 100% PTM vs 73.9% vanilla), Currency (n=14, 85.7% PTM vs 35.7% vanilla), Percent (n=6, 100% PTM vs 16.7% vanilla). Gold answers at risk (n=61): 93.4% PTM vs 91.8% vanilla.
  6. **Retrieval Completeness Discovery (The Dominant Driver)**: When all gold facts are in top-5 (49% of test), Large accuracy is 85.7% and Small is 57.1%. When incomplete (51% of test), Large drops to 11.8% and Small to 19.6%. An oracle retrieval completeness feature achieves an AUROC of 0.870 for Large accuracy (vs 0.505 for surface telemetry). Proves that surface cosine similarity is too noisy a proxy for multi-hop evidence completeness to train an effective cascade classifier on N=200 queries.
  7. **Generated Visualizations & Documentation**:
     - `results/pareto_v2.png`: Pareto frontier with 95% CI error bars, random routing confidence band, and indicators that all routers fall within the random envelope.
     - `results/retrieval_accuracy.png`: Grouped bar chart illustrating the dramatic accuracy collapse when retrieval is incomplete.
     - `README_v2.md`: Comprehensive rewrite replacing all overstated claims with empirical statistical rigor.
     - `results/PRESENTATION_NOTES_v2.md`: Presentation guide and professor Q&A defense notes emphasizing the value of negative results and engineering lessons.
  8. **Data Leakage Verification**: Confirmed train/test isolation enforced in `data.py` (lines 45-62), `train_and_eval_router.py` (lines 149-160), and `audit_analysis.py` (lines 140-176). All raw label files confirmed read-only. Zero test leakage detected.


---

### [2026-10-29 09:20 IST] - Step 7: Phase 3 Deliverables & Final Offline Verification Complete
- **Backup Snapshot**: `backups/20261029_0920/` created (878 files, `.env` excluded).
- **Deliverables Completed**:
  1. **Figures**:
     - `results/retrieval_accuracy_v3.png`: Plain title, 2-line clean x-axis labels, 95% bootstrap error bars with labels comfortably above caps, legend in upper-left without overlap.
     - `results/pareto_v3.png`: Title "Accuracy vs. simulated cost (N=100 test queries)", legend below plot (outside), hollow markers for PTM-compressed points, matched-cost random routing 95% band, note on paired CIs spanning zero.
     - `results/ptm_survival_v3.png`: Grouped bar chart with counts n printed under each kind, note explaining small n for percent/currency and punctuation stripping for dates, legend at upper center without bar overlap.
  2. **Extra Stats**:
     - `results/extra_stats_v3.json`: Paired bootstrap 95% CI on test (Large - Small: +10.0% [-1.0%, +21.0%], McNemar exact p = 0.1214) and train (+12.0% [+4.5%, +19.5%], McNemar p = 0.0032), plus subgroup bootstrap CIs.
  3. **Read-Only PTM Investigation**:
     - `results/ptm_date_diagnosis.md`: Detailed root cause analysis showing that 100% of constituent date words and numbers survived (0 actual word drops across 120 spans), with 66.1% rate caused by metric checking for literal punctuation commas pruned by LLMLingua-2; includes 5 concrete examples and proposed normalized metric fix.
  4. **Cheat Sheet**:
     - `results/NUMBERS_CHEATSHEET_v3.md`: 20 audited core metrics with file provenance and 10 examiner Q&As for presentation defense.
  5. **Offline Demo**:
     - `demo.py`: Fully functional offline demonstration script executing in <1s with 0 API calls. Tested twice with verified usage counters (requests = 389 unchanged).
- **Test Suite Status**: All unit tests passing.


---

### [2026-10-29 09:35 IST] - Step 8: Phase 4 Empirical Audit & Token-Level Survival Verification Complete
- **Backup Snapshot**: `backups/20261029_0930/` created (878 files, `.env` excluded).
- **Deliverables Completed**:
  1. **Token-Level Date/Currency Survival Audit**:
     - Verified constituent-word survival on 100 test contexts: 100% of date words/digits survived (900/900 across 354 spans); 0 words dropped in the 120 missed date spans (sole cause: dropped punctuation comma `,`). Normalized date survival = 99.7% (100% when normalizing non-breaking space `\xa0`).
     - Currency survival: 12/14 exact spans (85.7%); 2 missed spans (`$100\xa0million`, `$45\xa0million`) dropped zero words (only `\xa0` normalized to regular space). Constituent words/digits = 100% preserved.
     - Confirmed variants `none` and `ptm_ent` word-level text on test was not retained on disk, faithfully reported as `NOT AVAILABLE`.
  2. **End-to-End Latency & Break-Even Accounting**:
     - Mean CPU compression latency across 100 test queries: 6.869s (87 non-bypassed: 7.895s; 13 bypassed: 0.000s).
     - In 87/100 queries (87% overall, 100% of non-bypassed), compression latency exceeded LLM latency savings.
     - Saved to `results/summary_v4.csv`.
  3. **Figures**:
     - `results/ptm_survival_v4.png`: Y-axis capped strictly at 100 with headroom for labels, counts n under each category, explicit span vs word survival bars.
     - `results/pareto_v4.png`: Two-panel layout showing Panel A (Accuracy vs Cost) and Panel B (End-to-End Latency vs Cost).
  4. **Cheat Sheet**:
     - `results/NUMBERS_CHEATSHEET_v4.md`: Audited metrics with file provenance, updated examiner Q&As (404 key limitation, hypothesis, oracle labels, paired CIs, CPU latency trade-off, proxy pricing labels, and 3 new questions), plus full claims changelog.
- **Verification**:
  - Full pytest test suite passing (45/45).
  - API usage counters verified unchanged at 389 requests (0 calls made).


---

### [2026-10-29 10:50 IST] - Step 9: Zero-Dependency Offline Web Demo (demo_site/) Complete
- **Backup Snapshot**: `backups/20261029_1032/` created (890 files, `.env` excluded).
- **Deliverables Completed**:
  1. `export_demo_data.py`:
     - Reads strictly from disk cache (zero API calls, $0 spend).
     - Generated `demo_site/data/queries.json` (100 test queries, full telemetry, 4 cached answers, PTM spans, router probabilities).
     - Generated `demo_site/data/summary.json` (audited benchmark table, extra stats, simulated pricing table with required proxy/list disclosures).
     - Generated `demo_site/data/ptm_parity.json` (ground truth PTM spans for all 100 test contexts + 25 edge-case strings).
     - Copied figures to `demo_site/assets/` (`retrieval_accuracy_v3.png`, `ptm_survival_v4.png`, `pareto_v4.png`).
  2. `demo_site/` Interactive Web Application:
     - Zero build step, plain HTML/CSS/vanilla JS, zero npm packages, zero external CDN scripts (works 100% offline).
     - Total size: 2.99 MB (well below 15 MB limit).
     - Projector-ready high contrast UI (min 16px body text).
     - Mandatory top banner: "Replay of cached results for 100 HotpotQA test questions. No live model calls. Costs are simulated at list prices (small model: proxy price)."
     - Four full tabs: Pipeline Demo (with threshold slider & compression toggle), Try the Mask (live client-side PTM sandbox with 3 examples), Results (summary table, figures, 3 core empirical findings), and About (architecture, models, limitations, academic attribution).
  3. Parity & Browser Testing:
     - JavaScript PTM engine (`demo_site/ptm.js`) achieved 100% byte-for-byte span parity with Python on 100/100 test contexts and 25/25 edge-case strings (verified in Node and via headless Chrome on `demo_site/test.html`).
     - Automated CDP browser test verified threshold slider (0.00 to 1.00), gold reveal, compression toggle, and all 4 tabs with zero layout defects.
     - Cross-check test verified 100% integrity across 5 randomly sampled questions against raw result files.
  4. Security & Deployment:
     - `demo_site/vercel.json` configured for cleanUrls and asset caching.
     - `demo_site/DEPLOY.md` written with exact deployment instructions (free Hobby plan, stop if credit card requested, local `python -m http.server 8000` backup).
     - Security audit verified: zero .env, zero .cache, zero API keys, zero absolute local paths, and zero external fetch requests.
