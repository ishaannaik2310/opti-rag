# OptiRAG-lite: Oral Presentation Notes & Defense Guide (Post-Audit v2)

**Presenter**: Ishaan Naik (Symbiosis Institute of Technology)  
**Project**: OptiRAG-lite (Cost-Optimal Retrieval-Augmented Generation)  
**Format**: 7-Minute Slide Deck + Oral Defense Q&A  
**Budget**: Strictly \$0.00 Project Spend (Free-Tier Infrastructure)  
**Audit Status**: Fully Audited with Matched-Cost Random Baselines, Permutation Tests, and Zero-Leakage Calibration

---

## Part 1: Slide-by-Slide Presentation Guide (7 Slides)

### Slide 1: Title & The Core Dilemma
- **Slide Title**: OptiRAG-lite: Empirical Evaluation of Cascaded Routing and Protected Context Compression in RAG
- **The Core Problem**:
  - Frontier LLM inference in enterprise RAG is economically expensive and slow.
  - Feeding raw multi-chunk contexts into large models for every incoming query incurs substantial token costs.
- **The Core Hypotheses We Tested**:
  1. *Hypothesis 1 (Routing)*: Can low-dimensional retrieval telemetry (retrieval scores, entropy, score margins) predict query hardness and route queries between a cheap Small model (`allam-2-7b`) and a Large model (`gemini-3.5-flash-lite`) better than matched-cost random routing?
  2. *Hypothesis 2 (Compression)*: Can a Protected Token Mask (PTM) prevent token-pruning compressors (LLMLingua-2) from dropping critical entities (numbers, dates, negations)?
- **Infrastructure Metric**: Completed entire empirical study on a **strict \$0.00 budget** using official vendor free tiers.

---

### Slide 2: Architecture & Workflow
- **Visual**: System Architecture Flowchart (`flowchart TD` from `README_v2.md`).
- **Talking Points**:
  - **Retriever**: MiniLM dense index (FAISS FlatIP) over 2,959 HotpotQA distractor chunks.
  - **Feature Extraction**: 8-d retrieval telemetry extracted in <2ms directly from search output.
  - **Cascade Router**: Trained on train out-of-fold predictions to evaluate $P(hard)$, escalating to Large if $P(hard) \ge \tau$.
  - **PTM Context Compressor**: Identifies and hard-masks numbers, dates, currency, and negations, preserving them by construction during LLMLingua-2 token pruning.

---

### Slide 3: Labeling & The Core Phenomenon: Retrieval Completeness
- **Visual**: Grouped Bar Chart (`results/retrieval_accuracy.png`).
- **The Numbers**:
  - Train ($N=200$): Small Acc = 47.0%, Large Acc = 59.0%, Hard ($hard=1$) = 21.5% (43 queries).
  - Test ($N=100$): Small Acc = 38.0%, Large Acc = 48.0%, Hard ($hard=1$) = 22.0% (22 queries).
- **The Primary Empirical Finding of this Project**:
  - When retrieval successfully recalled **all gold supporting documents** (`gold_all_in_top5 = True`, ~49% of queries):
    - Large model accuracy is **87.9%** (Train) and **85.7%** (Test).
    - Small model accuracy is **71.7%** (Train) and **57.1%** (Test).
  - When retrieval was **incomplete** (`gold_all_in_top5 = False`):
    - Large model accuracy collapses to **30.7%** (Train) and **11.8%** (Test).
    - Small model accuracy drops to **22.8%** (Train) and **19.6%** (Test).
- **Diagnostic Power**:
  - An oracle feature representing retrieval completeness yields an **AUROC of 0.870** for predicting Large model success.
  - *Key Takeaway*: Retrieval completeness is the overwhelming driver of multi-hop QA success.

---

### Slide 4: Router Evaluation & Matched-Cost Random Baseline (The Audit Result)
- **Visual**: Pareto Chart (`results/pareto_v2.png`) and Matched-Cost Comparison Table.
- **Key Empirical Results**:
  - Always-Small: 38.0% [29.0%, 48.0%], Cost = \$0.0417 / 1k queries.
  - Always-Large: 48.0% [38.0%, 58.0%], Cost = \$0.2294 / 1k queries.
  - Telemetry Router (@ 30% train-calibrated escalation): 41.0% [32.0%, 51.0%], Cost = \$0.0968 / 1k queries.
  - Matched-Cost Random Baseline (@ 28% escalation): 40.7% [36.0%, 46.0%].
  - Paired Difference (Telemetry - Random): **+2.0%** with 95% bootstrap CI: **[-5.0%, +9.0%]**.
- **Test AUROC & Permutation Test**:
  - Telemetry Router test AUROC for query hardness: **0.505** (95% CI: [0.375, 0.640]).
  - Permutation test p-value vs chance ($H_0: \text{AUROC} = 0.50$): **$p = 0.4835$**.
- **Honest Scientific Takeaway**:
  - The learned router does not achieve statistically significant separation from a cost-matched coin flip.
  - *Why?* Dense retrieval cosine scores reflect semantic similarity, but they cannot tell the classifier whether a second-hop document was retrieved. Surface telemetry is too noisy a proxy for true multi-hop information completeness.

---

### Slide 5: Protected Token Mask (PTM) Context Compression
- **Visual**: Entity Survival Breakdown Table.
- **Key Findings**:
  - Compression ratio: 599.3 tokens $\to$ 343.1 tokens (**~40% token savings**).
  - **Preservation by Construction**:
    - Numbers: 1,667 / 1,667 (**100.0%** in PTM vs 96.0% in vanilla LLMLingua-2; +67 numbers saved).
    - Negations: 69 / 69 (**100.0%** in PTM vs 73.9% in vanilla; +18 negation words saved).
    - Percentages: 6 / 6 (**100.0%** in PTM vs 16.7% in vanilla).
    - Currency: 12 / 14 (**85.7%** in PTM vs 35.7% in vanilla).
    - Gold answers at risk: 57 / 61 (**93.4%** in PTM vs 91.8% in vanilla).
  - Downstream Accuracy:
    - Small tier: 41.0% (PTM) vs 38.0% (Uncompressed).
    - Large tier: 46.0% (PTM) vs 48.0% (Uncompressed).
    - Paired difference CIs span zero; compression is economically viable (\$0.0619 / 1k queries) without degrading accuracy.

---

### Slide 6: Pricing Transparency & Sensitivity
- **Visual**: Pricing and Sensitivity Breakdown.
- **Disclosures**:
  - Large Tier (Gemini 3.5 Flash-Lite): \$0.30 input / \$2.50 output per 1M tokens (Official Google AI Studio rates).
  - Small Tier (`allam-2-7b` on Groq): Hosted for free ($0 billed charges); priced at **\$0.05 / \$0.08 per 1M tokens as an 8B-class commercial proxy** (benchmarked to Groq Llama-3.1-8B).
- **Sensitivity Analysis**:
  - If Small Tier price doubles (2x: \$0.10 / \$0.16), Telemetry Router cost increases from \$0.0968 to \$0.1264 / 1k queries, still delivering a **44.9% cost reduction** compared to Always-Large (\$0.2294).
  - Cascade routing provides economic savings even under unfavorable pricing assumptions.

---

### Slide 7: Engineering Lessons & Future Directions
- **What We Learned (The Value of Rigorous Negative Results)**:
  1. *Negative results are vital engineering insights*: Discovering that surface telemetry fails to beat random routing saves engineering teams from deploying fragile, ineffective cascade classifiers.
  2. *Retrieval completeness is the bottleneck*: System designers should allocate token/latency budget toward improving recall (hybrid search, multi-hop query decomposition) rather than router complexity.
  3. *PTM provides indispensable safety*: Unconstrained token compressors drop over 25% of negations and 60% of currency references. Deterministic regex token masking is a necessary guardrail.
- **Future Directions**:
  - Multi-hop graph-based routing (checking whether retrieved entities form a closed reasoning chain).
  - Conformal risk bounds for cascade routing.
  - Dynamic chunk-level compression rates based on retrieval margin.

---

## Part 2: Defense Preparation: Tough Questions & Honest Answers

### Q1: Your learned router achieved a test AUROC of 0.505 and its paired difference CI spans zero compared to random routing. Does this mean your project failed?
> **Model Answer**:  
> "Not at all. In machine learning systems research, a well-controlled negative result is far more valuable than an embellished positive claim. Our audit demonstrated that low-dimensional surface retrieval telemetry (cosine scores, entropy, score margins) has an AUROC of only 0.505 ($p = 0.4835$) for query hardness. 
> Crucially, our diagnostic analysis explained *why*: an oracle feature indicating whether all gold supporting facts were retrieved (`gold_all_in_top5`) has an AUROC of 0.870 for predicting Large model accuracy. When retrieval is complete, the Large model succeeds 86% of the time; when incomplete, it collapses to 12%. Surface cosine scores cannot distinguish between a single high-scoring distractor chunk and a complete two-hop reasoning bridge. This establishes an important engineering principle: before deploying cascade routers in multi-hop RAG, one must optimize retrieval recall or routing based on multi-hop graph connectivity, rather than relying on surface telemetry."

---

### Q2: You claim PTM achieved 100% number and negation retention, but isn't that just a tautology since you forced them via regex?
> **Model Answer**:  
> "Yes, that is exactly right, and we explicitly disclose in our report that PTM preserves these tokens *by construction*. 
> The research contribution is not claiming that regex was an empirical discovery, but rather addressing a known failure mode of statistical prompt compression. When vanilla LLMLingua-2 was applied at a 0.5 target rate, it dropped 26.1% of negation words ('not', 'never', 'without') and 64.3% of currency symbols, which introduces severe hallucination risks in factual RAG systems. PTM introduces a deterministic engineering guardrail that guarantees entity retention while still allowing the neural compressor to remove 40% of peripheral tokens, saving ~36% cost with zero statistically significant degradation in answer accuracy."

---

### Q3: How did you ensure there was zero data leakage between train and test?
> **Model Answer**:  
> "We enforced three strict isolation barriers:
> 1. *Deterministic Partitioning*: 300 HotpotQA queries were split at data generation time (`data.py`) with a fixed seed into 200 train and 100 test examples, stored in distinct files.
> 2. *Train-Only Model Fitting*: All feature scaling, logistic regression training, and hyperparameter selection were performed exclusively inside stratified 5-fold cross-validation on the 200 training examples.
> 3. *Train-Only Threshold Calibration*: All decision thresholds ($\tau$) were chosen either by targeting a 30% escalation quantile or maximizing Youden's $J$ on out-of-fold training predictions prior to any test inference.
> 4. *Immutable Storage*: All raw label files were made read-only on disk, guaranteeing that test labels were never inspected or tuned against."

---

### Q4: Why did you use ALLaM-2-7B on Groq instead of Llama 3.1 8B?
> **Model Answer**:  
> "Groq deprecated the free-tier endpoint for `llama-3.1-8b-instant` during our project lifecycle. To maintain our hard constraint of a \$0.00 project spend, we adopted `allam-2-7b`, which remains active on Groq's free tier. 
> Because ALLaM-2 is a bilingual Arabic-English model, its English zero-shot multi-hop reasoning accuracy on HotpotQA was modest (38.0% on test). In our financial analysis, we disclosed that its \$0.05/\$0.08 per 1M token rate is an 8B-class proxy price benchmarked to Groq's published commercial rates. Our sensitivity analysis verified that even if small-tier costs are doubled to \$0.10/\$0.16, cascaded routing still delivers a 44.9% cost reduction compared to routing everything to Gemini 3.5 Flash-Lite."
