# OptiRAG-lite: Project Context (READ THIS FIRST)

You are helping a university student (Ishaan Naik, Symbiosis Institute of Technology) build a **deep learning course project in ONE DAY with a $0 budget**. This file is the single source of truth. Follow it. Do not add scope, paid services, or extra frameworks.

---

## 1. What we are building

**OptiRAG-lite** is a layer between a vector database and an LLM in a RAG (Retrieval-Augmented Generation) pipeline. It cuts cost and latency by doing two things:

1. **Retrieval-aware routing.** A *trained* classifier decides whether a question is "easy" (answer with a cheap/small LLM) or "hard" (answer with an expensive/large LLM). Unlike most routers, it looks not only at the question but also at *retrieval telemetry*: the similarity scores of the retrieved chunks. Example: "Who is the CEO?" is easy if one chunk matches at 0.95, but hard if five conflicting chunks match at 0.55.
2. **Safe context compression.** Retrieved text is compressed with LLMLingua-2 to save tokens, but a **Protected Token Mask (PTM)** (regex) force-keeps numbers, dates, currency, percentages, and negations (not, never, no, except, without) so compression cannot turn "Revenue is not growing" into "Revenue growing". Compression is skipped entirely if the context is under 400 tokens (**latency break-even bypass**).

## 2. Research question (what the report/presentation argues)

> Does adding retrieval telemetry (similarity-score features) to a learned router improve the cost/accuracy trade-off versus a query-only router, and does protected-token masking preserve facts under compression?

The **central scientific result is the ablation**: query-only vs. telemetry-only vs. query+telemetry router. The **hero chart** is a cost-vs-accuracy Pareto curve.

## 3. Hard constraints

- **$0.** Use only free tools. Groq free tier for LLM calls, local models for embeddings/compression, FAISS for vector search (no paid vector DB, no OpenAI).
- **One day.** Prefer simple, working code over elegant code. If something is blocking for more than ~15 minutes, simplify it and tell the user.
- **Never hardcode secrets.** API key goes in `.env` as `GROQ_API_KEY`. Provide `.env.example`. Add `.env` to `.gitignore`.
- **Cache every LLM call to disk** (key = hash of model + prompt). Never pay (in time or rate limit) for the same prompt twice. The labeling job must be **resumable**.
- **Do not guess library APIs.** For LLMLingua-2, Groq, and FAISS, read the installed package docs or run a tiny test before writing wrapper code.
- Python 3.10+. Pin versions in `requirements.txt`.
- Do **not** build: a web UI, a FastAPI server, a learned token masker, conformal prediction, Docker/Qdrant. (These are "future work" only.)

## 4. Models and data

- **Data:** HotpotQA (distractor setting) via the Hugging Face `datasets` library. Use **300 questions**: 200 train / 100 test, fixed seed (42). Pool all their paragraphs into one corpus and index them.
- **Embeddings/retrieval:** `sentence-transformers/all-MiniLM-L6-v2` + FAISS (cosine via normalized inner product). Retrieve **top-5** chunks with scores.
- **LLMs (via Groq, models set in `config.py`, easy to change):**
  - SMALL tier: an 8B-class Llama model.
  - LARGE tier: a 70B-class model.
  - Groq's available free models change; check the current list and put the chosen names in `config.py`. Record the model names and date in results.
- **Compressor:** LLMLingua-2, model `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`, run on CPU. Use the `llmlingua` package (`PromptCompressor(..., use_llmlingua2=True)` and its force-keep tokens option). Verify the exact argument names from the installed version.

## 5. Repo structure

```
optirag/
  docs/PROJECT_CONTEXT.md     <- this file
  config.py                   <- model names, price table, thresholds, seeds
  data.py                     <- load/split HotpotQA, build corpus
  retrieve.py                 <- MiniLM + FAISS, returns chunks + scores
  llm.py                      <- cached Groq client, token logging, retry/backoff
  run_labels.py               <- resumable job: answer all queries with both tiers
  ptm.py                      <- Protected Token Mask (regex)
  compress.py                 <- LLMLingua-2 + mask + 400-token bypass
  features.py                 <- query embedding + retrieval telemetry features
  router.py                   <- train query-only / telemetry-only / both
  eval.py                     <- baselines, metrics, Pareto plot, ablation table
  tests/                      <- test_ptm.py, test_features.py, test_router.py
  results/                    <- JSON/CSV outputs, figures
  requirements.txt, .env.example, .gitignore, README.md
```

## 6. Module specs

**data.py**: Load HotpotQA distractor, take 300 questions with a seeded shuffle, split 200/100. Each item: id, question, gold answer, its paragraphs. Build the pooled corpus (chunk = one paragraph, keep title).

**retrieve.py**: Embed corpus once (cache embeddings to disk). `retrieve(query, k=5)` returns list of (chunk_text, score) sorted by score.

**llm.py**: `generate(model, prompt) -> {text, in_tokens, out_tokens, latency_s}`. Disk cache. Retry with exponential backoff on rate limit errors. Prompt format: "Answer using only the context. Give a short answer (a few words)." + context + question. Temperature 0.

**run_labels.py**: For every question, retrieve top-5, build the prompt from *uncompressed* context, answer with both tiers, save to `results/labels.jsonl`. Skip items already done. Compute token F1 and Exact Match vs gold. **Label `hard = 1` if small F1 < 0.5 and large F1 >= 0.5; otherwise 0.** Also save `small_f1`, `large_f1` for cost-sensitive analysis. Print class balance when finished.

**ptm.py**: `find_protected_spans(text)` returns character spans for: numbers (incl. decimals, commas), dates (several formats, years, month names), currency ($, €, ₹, USD...), percentages, and negation words (not, never, no, none, nor, except, without, n't). `protected_tokens(text)` returns the strings to force-keep. Tests must include `"Revenue is not growing"` (keeps "not"), `"$1.5M in Q3 2023, up 15%"`, and a date like `03/04/2024`.

**compress.py**: `compress(context, query, use_mask=True)`:
- If token count < 400: return context unchanged (bypass) and flag `bypassed=True`.
- Else run LLMLingua-2 at rate ~0.5 with the PTM tokens passed as force-keep tokens (when `use_mask=True`).
- Return compressed text, original/compressed token counts, and wall-clock time.
- Implement **Entity Survival Rate** = fraction of protected tokens in the original that still appear in the compressed text. Report it with and without the mask.
- Note: HotpotQA contexts from 5 paragraphs may often be near/above 400 tokens; report how many queries were bypassed.

**features.py**: Per query, build:
- `q_emb`: 384-d MiniLM query embedding.
- Telemetry: top-1 score, top-2 score, top1-top2 gap, mean of top-5, std of top-5, entropy of softmax(scores), mean pairwise cosine similarity between the 5 chunk embeddings (conflict/redundancy proxy), context token count, query token count.

**router.py**: Train three variants, each as (a) logistic regression and (b) a small PyTorch MLP (1-2 hidden layers, dropout, early stopping on a validation fold from train):
- query-only (`q_emb` + query length)
- telemetry-only
- both
Report AUROC and accuracy on the test split. Output predicted P(hard) for the test set. Decision rule: route to LARGE if P(hard) >= threshold; sweep the threshold from 0 to 1 for the Pareto curve. Standardize features, fit scalers on train only. With only 200 train examples, use cross-validation for model selection and say that results are small-sample.

**eval.py**: On the 100 test queries, compare:
1. always-SMALL, 2. always-LARGE, 3. heuristic router (max score < 0.70 or query contains compare/analyze/summarize), 4. learned routers (three variants).
For each: mean F1, EM, **simulated cost per 1,000 queries** (token counts x per-million-token prices from `config.py`; label it "simulated, not billed"), mean latency. Use compressed context for the OptiRAG configurations and uncompressed for baselines (compression applies when context >= 400 tokens). Bootstrap 95% CIs (1000 resamples) for F1. Produce:
- `results/pareto.png`: x = simulated cost per 1k queries, y = F1, with the learned-router threshold sweep as a curve and baselines as labeled points.
- `results/ablation.csv` / printed table: query-only vs telemetry-only vs both (AUROC, F1 at matched cost).
- `results/ptm_survival.csv`: Entity Survival Rate with vs without mask.

## 7. Working rules for the agent

1. **Show a plan first** and wait for the user's approval before writing code.
2. Build **one module at a time**, in this order: config/data/retrieve -> llm -> run_labels (start it ASAP, it is the slow step) -> ptm (while labels run) -> compress -> features -> router -> eval -> README.
3. After each module, **run its tests / a tiny smoke test and show the output** before moving on.
4. Keep functions small and typed where cheap. No clever abstractions.
5. If a library call fails, show the real error and fix it; do not silently change the design.
6. Be honest in results: if the router doesn't beat the heuristic, report that. Do not tune on the test set. Report class balance and small-sample caveats.
7. At the end, write `README.md` with setup, how to reproduce, results table, the Pareto figure, limitations, and "future work" (learned token masker, conformal guarantees, adaptive compression ratio, larger scale).

## 8. Suggested timeline (about 12 hours)

| Block | Goal |
|---|---|
| 0-2h | Repo, data, retrieval, cached Groq wrapper, smoke tests |
| 2-4h | Start `run_labels.py` in background; write `ptm.py` + tests |
| 4-6h | Compression with mask and bypass; Entity Survival Rate |
| 6-8h | Features + router training + ablation |
| 8-10h | Test-set evaluation, Pareto chart, CIs |
| 10-12h | README, slides/report, buffer |

## 9. Definition of done

- `run_labels.py` finished; class balance reported.
- PTM unit tests pass; Entity Survival Rate shown with/without mask.
- Three router variants trained, AUROC reported, ablation table produced.
- Pareto chart saved, baselines included, CIs reported.
- README explains how to reproduce everything with `.env` and `requirements.txt`.
