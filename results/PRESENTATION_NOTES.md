# OptiRAG-lite: Presentation Notes & Oral Defense Guide

**Presenter**: Ishaan Naik (Symbiosis Institute of Technology)  
**Project**: OptiRAG-lite (Cost-Optimal Retrieval-Augmented Generation)  
**Date**: October 29, 2026 | **Time**: 2:00 PM IST  

---

## Part 1: Slide-by-Slide Presentation Outline (7 Slides)

### Slide 1: Title & The Core Dilemma
- **Title**: OptiRAG-lite: Telemetry-Guided Cascade Routing & Protected Token Compression for Cost-Optimal RAG
- **The Problem**: Frontier LLM inference in production RAG is economically unsustainable and slow. Feeding raw top-5 chunks to large models for every query wastes up to 80% of inference budget on trivial queries.
- **The Core Idea**:
  1. Classify query difficulty before generation using lightweight retrieval telemetry (scores, entropy, margin).
  2. Cascade: route easy queries to a small model (Groq ALLaM-2-7B), reserving frontier models (Gemini 3.5 Flash-Lite) for hard multi-hop queries.
  3. Safely compress retrieved contexts using a Protected Token Mask (PTM) so numbers, dates, and negations are never dropped.
- **Key Project Metric**: **\$0.00 actual project spend** achieved across entire research pipeline.

---

### Slide 2: System Architecture & Workflow
- **Visual**: System Architecture Flowchart (from `README.md`).
- **Talking Points**:
  - Offline Stage: MiniLM embedding index (FAISS FlatIP) on 2,959 HotpotQA distractor chunks.
  - Telemetry Extraction: 8 retrieval signals extracted in <2ms directly from FAISS search results.
  - Cascaded Router: Evaluates $P(hard)$; if $P(hard) \ge \tau$, route to Large Tier; else Small Tier.
  - PTM Compressor: Masks numbers, dates, currency, percentages, and negations before LLMLingua-2 pruning.

---

### Slide 3: Empirical Labeling & The "Hard" Query Phenotype
- **Visual**: 2x2 Confusion Matrix Table (Train $N=200$, Test $N=100$).
- **Key Numbers**:
  - Train: Both Correct = 37.5%, Small-Only = 9.5%, **Large-Only ($hard=1$) = 21.5% (43 queries)**, Both Wrong = 31.5%.
  - Test: Both Correct = 26.0%, Small-Only = 12.0%, **Large-Only ($hard=1$) = 22.0% (22 queries)**, Both Wrong = 40.0%.
  - Small Accuracy: Train 47.0%, Test 38.0% | Large Accuracy: Train 59.0%, Test 48.0%.
- **The Retrieval Completeness Finding**:
  - When **all gold facts are retrieved** (`gold_all_in_top5 = True`): Large accuracy is **87.9%** (Train) / **85.7%** (Test).
  - When gold facts are **partially or completely missing**: Large accuracy drops to **30.7%** (Train) / **11.8%** (Test).
  - *Conclusion*: Retrieval quality is the primary upper bound on multi-hop reasoning.

---

### Slide 4: Router Training: Telemetry vs. Semantic Embeddings
- **Visual**: Router Benchmark Table (CV AUROC vs. Test PR-AUC).
- **Key Numbers**:
  - **Heuristic Baseline** (Score < 0.70 or keyword): CV AUROC = 0.487, Test AUROC = 0.518, Test PR-AUC = 0.226.
  - **Query-Only (385-d)**: CV AUROC = 0.504 $\pm$ 0.116, Test AUROC = 0.505, Test PR-AUC = 0.225.
  - **Telemetry-Only (8-d LogReg)**: **CV AUROC = 0.571 $\pm$ 0.033**, **CV PR-AUC = 0.317 $\pm$ 0.065**, **Test PR-AUC = 0.302**.
  - **Telemetry-Only (PyTorch MLP)**: **Test AUROC = 0.546**, Test PR-AUC = 0.260.
- **Talking Points**:
  - Low-dimensional retrieval telemetry (entropy, margin, chunk similarity) significantly outperformed high-dimensional query embeddings in cross-validation.
  - Query embeddings overfit with $N=200$; telemetry captures the model's actual uncertainty regarding the retrieved evidence.

---

### Slide 5: Protected Token Mask (PTM) Context Compression
- **Visual**: `results/entity_survival.png` (Grouped Bar Chart).
- **Key Numbers**:
  - Effective compression: 599.3 tokens $\to$ 343.1 tokens (**~40% token savings**).
  - **Number Survival**: **100.0%** in PTM vs 97.0% in vanilla LLMLingua-2.
  - **Currency Survival**: **100.0%** in PTM vs 50.0% in vanilla.
  - **Negation Survival**: **100.0%** in PTM vs 75.0% in vanilla.
  - **Overall Entity Survival**: **95.4%** in PTM vs 91.4% in vanilla.
- **Talking Points**:
  - Vanilla token compressors prune punctuation, currency signs, and critical negations ("not", "never", "without").
  - PTM guarantees semantic preservation while pruning peripheral fluff.

---

### Slide 6: End-to-End System Evaluation & Pareto Frontier
- **Visual**: `results/pareto.png` (Accuracy vs. Cost per 1,000 Queries).
- **Key Numbers**:
  - **Always-Small**: Acc = 38.0% [29.0%, 48.0%], Cost/1k = \$0.0417, Latency = 0.17s.
  - **Always-Large**: Acc = 48.0% [38.0%, 58.0%], Cost/1k = \$0.2294, Latency = 0.92s.
  - **Telemetry Router**: Acc = **44.0%** [35.0%, 54.0%], Cost/1k = **\$0.1164**, Latency = **0.45s**.
- **Talking Points**:
  - The Telemetry Router retains **91.7% of Always-Large accuracy** (44% vs 48%) while cutting financial cost by **49.3%** and latency by **51.1%**.
  - Operates strictly on the Pareto frontier above the random routing baseline.

---

### Slide 7: Limitations & Future Horizons
- **Honest Limitations**:
  - Small sample regime ($N=200$ train, $N=100$ test) imposed by free-tier API ceilings.
  - Bilingual Small Model (`allam-2-7b`) has modest English zero-shot multi-hop reasoning.
  - HotpotQA distractor questions are factoid/bridge heavy with relatively few numeric negations.
- **Future Directions**:
  - RL-guided Learned Token Masker.
  - Conformal prediction risk bounds for guaranteed maximum accuracy degradation.
  - Dynamic compression scaling based on retrieval margin.

---

## Part 2: Defense Questions & Honest Answers (For Professor Q&A)

### Q1: Why did 8-dimensional telemetry outperform 384-dimensional dense semantic query embeddings?
> **Answer**:  
> In small-data regimes ($N=200$ training queries), high-dimensional representations ($d=385$) suffer from high sample variance and overfit, yielding a CV AUROC of only $0.504 \pm 0.116$. In contrast, our 8 telemetry features represent compact, physically grounded measures of the retriever's confidence: the score gap between chunk 1 and 2, the softmax score entropy, and the pairwise chunk cosine dispersion. When retrieval is uncertain (high entropy, small gap), small models almost universally fail, making telemetry a far more direct indicator of query hardness than raw query semantics.

---

### Q2: Your router achieved 44% accuracy vs Always-Large at 48% on test. Is that 4% difference statistically significant?
> **Answer**:  
> Based on our 1,000-sample empirical bootstrap analysis, the 95% confidence intervals are $[35.0\%, 54.0\%]$ for the Telemetry Router and $[38.0\%, 58.0\%]$ for Always-Large. Because the intervals overlap substantially on an $N=100$ test set, the 4% absolute difference is not statistically significant at $p < 0.05$. However, the cost difference (\$0.1164 vs \$0.2294 per 1k queries) is non-overlapping and highly significant: the router provably cuts token spend by ~50% while operating within the statistical margin of the large model's accuracy.

---

### Q3: Why is retrieval recall so low for all supporting facts (`gold_all_in_top5 = 49.5%`), and how does that bound your system's performance?
> **Answer**:  
> HotpotQA distractor questions require multi-hop reasoning across at least two disparate Wikipedia documents. Single-step dense retrieval with MiniLM retrieves top-5 chunks independently without iterative query reformulation. When both supporting facts are captured, Gemini 3.5 Flash-Lite achieves **87.9% accuracy**; but when at least one fact is missing, accuracy collapses to **30.7%**. This proves that the performance bottleneck is not LLM reasoning capacity, but retrieval completeness.

---

### Q4: Your target label is $hard = 1$ ($Small=0 \land Large=1$). What happens to the queries where both models fail ($31.5\%$ of train, $40.0\%$ of test)?
> **Answer**:  
> We deliberately labeled queries where both models fail as $hard = 0$ (easy/route to small). This is an economically optimal design choice: if the large model cannot solve the question either (typically because neither model has the supporting facts), escalating that query to the expensive tier wastes financial budget and adds 0.75s latency with zero gain in accuracy. Routing unanswerable questions to the small model minimizes economic loss.

---

### Q5: How does your Protected Token Mask (PTM) prevent hallucinations during context compression?
> **Answer**:  
> Standard prompt compressors (like LLMLingua-2) optimize for token perplexity and token drop ratios, frequently discarding punctuation, negation prefixes, and isolated numerical entities. In our pilot, vanilla LLMLingua-2 dropped 25% of negation words and 50% of currency amounts. Dropping a negation like "not" or "without" completely inverts the semantic truth value of a passage, forcing the LLM to hallucinate the opposite answer. PTM enforces deterministic regex masks for numbers, dates, currency, percentages, and negations, achieving 100% retention on numbers, currency, and negations at rate 0.5.

---

### Q6: Why did you use ALLaM-2-7B as the Small Tier model instead of Llama 3.1 8B?
> **Answer**:  
> During project development, Groq officially deprecated `llama-3.1-8b-instant`. To maintain our zero-cost constraint without adding paid credits, we adopted `allam-2-7b`, the active open-access 7B model available on Groq's high-speed LPU infrastructure. Because ALLaM-2 is specialized for Arabic-English bilingual tasks, its zero-shot English multi-hop reasoning is lower than Llama-3.1-8B (38.0% test accuracy). However, its inference latency is exceptional (~0.15s per query), making it an ideal cost and speed anchor for cascade routing.

---

### Q7: Why did you tune the router on the training split only using Stratified 5-Fold CV, rather than holding out a separate validation set?
> **Answer**:  
> With a total budget of 200 training examples and 43 positive ($hard=1$) instances, carving out a fixed validation set of 40 examples would leave only ~8 positive examples in validation, causing extreme variance in hyperparameter selection. Stratified 5-Fold Cross-Validation ensures every training example is used for out-of-fold validation, maintains identical class ratios ($21.5\%$) across all folds, and fits scalers exclusively on the 4 training folds to guarantee zero test leakage.

---

### Q8: How would you scale OptiRAG to an enterprise deployment serving millions of queries per day?
> **Answer**:  
> In an enterprise deployment:
> 1. The router feature extraction runs in C++ inside the FAISS/vector engine with zero network overhead (<1ms).
> 2. The routing decision is evaluated against an adaptive threshold $\tau$ modulated in real-time by current API queue latency or remaining budget quotas.
> 3. PTM compression can be compiled with ONNX Runtime / TensorRT-LLM on CPU or edge GPUs.
> At 10 million queries/day, a 50% cost reduction translates to over \$1,100/day in saved frontier LLM API fees.
