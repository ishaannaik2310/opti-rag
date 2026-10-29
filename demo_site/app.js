/**
 * OptiRAG-lite Interactive Demonstration Client Application (app.js).
 * Runs 100% offline with zero CDN dependencies and zero external network calls.
 * All numbers and evaluations originate from pre-computed, audited data files.
 */

(function () {
  "use strict";

  // Application State
  let queriesData = [];
  let summaryData = null;
  let currentQuery = null;
  let goldRevealed = false;

  // DOM Elements - Tabs
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  // DOM Elements - Pipeline Demo
  const querySearch = document.getElementById("querySearch");
  const querySelect = document.getElementById("querySelect");
  const btnRandom = document.getElementById("btnRandom");
  const qidBadge = document.getElementById("qidBadge");
  const questionText = document.getElementById("questionText");
  const btnRevealGold = document.getElementById("btnRevealGold");
  const goldAnswerContainer = document.getElementById("goldAnswerContainer");
  const goldAnswerText = document.getElementById("goldAnswerText");
  const retrievalCoverageBadge = document.getElementById("retrievalCoverageBadge");
  const chunksList = document.getElementById("chunksList");
  const telemetryStrip = document.getElementById("telemetryStrip");

  const routerVariantSelect = document.getElementById("routerVariantSelect");
  const thresholdControlGroup = document.getElementById("thresholdControlGroup");
  const thresholdSlider = document.getElementById("thresholdSlider");
  const thresholdValue = document.getElementById("thresholdValue");
  const variantDesc = document.getElementById("variantDesc");
  const routerDecisionBanner = document.getElementById("routerDecisionBanner");
  const probHardValue = document.getElementById("probHardValue");
  const decisionText = document.getElementById("decisionText");

  const compressToggle = document.getElementById("compressToggle");
  const compBypassBadge = document.getElementById("compBypassBadge");
  const origTokCount = document.getElementById("origTokCount");
  const compTokCount = document.getElementById("compTokCount");
  const originalContextView = document.getElementById("originalContextView");
  const compressedContextView = document.getElementById("compressedContextView");

  const activeModelLabel = document.getElementById("activeModelLabel");
  const activeAnswerText = document.getElementById("activeAnswerText");
  const evalBadgeContainer = document.getElementById("evalBadgeContainer");
  const valLlmLats = document.getElementById("valLlmLats");
  const valE2eLats = document.getElementById("valE2eLats");
  const valCost1k = document.getElementById("valCost1k");
  const valCostQuery = document.getElementById("valCostQuery");

  // DOM Elements - Try the Mask
  const maskInput = document.getElementById("maskInput");
  const maskHighlightPreview = document.getElementById("maskHighlightPreview");
  const exBtnFinance = document.getElementById("exBtnFinance");
  const exBtnBio = document.getElementById("exBtnBio");
  const exBtnPlain = document.getElementById("exBtnPlain");
  const cntCurrency = document.getElementById("cntCurrency");
  const cntDate = document.getElementById("cntDate");
  const cntPercent = document.getElementById("cntPercent");
  const cntNegation = document.getElementById("cntNegation");
  const cntNumber = document.getElementById("cntNumber");
  const pctProtectedWords = document.getElementById("pctProtectedWords");

  // DOM Elements - Results Table
  const benchmarkTableBody = document.querySelector("#benchmarkTable tbody");

  // Variant Descriptions Map
  const VARIANT_DESCRIPTIONS = {
    telemetry_only_logistic: "8 low-dimensional retrieval confidence metrics. Trained with Stratified 5-Fold CV on train only (tau=0.520 escalates ~30% queries).",
    telemetry_only_mlp: "PyTorch 2-layer MLP on 8 retrieval confidence features. Achieved highest test AUROC (0.546) among learned variants.",
    query_only_logistic: "385 features (384-d MiniLM query embedding + query length). High-dimensional space generalises weakly on N=200 train queries.",
    both_logistic: "393 concatenated features (Query semantic embedding + retrieval telemetry).",
    heuristic: "Deterministic threshold rule: Escalate to Large if top-1 score < 0.70 or keyword triggers. Escalates 70% of test queries."
  };

  // Example Sentences for Mask Tester
  const MASK_EXAMPLES = {
    finance: "In Q3 2023, the firm reported revenue of $14.2 million, but net profits did not exceed 15 percent, neither meeting expectations nor paying dividends.",
    bio: "Jayne Mansfield (born April 19, 1933 – June 29, 1967) was an American actress who rose to stardom in the 1950s after appearing in 24 films.",
    plain: "The general consensus among the committee members was that regular attendance would greatly benefit overall team collaboration."
  };

  /**
   * Escape HTML to prevent XSS during string injection
   */
  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  /**
   * Highlight PTM spans within arbitrary text
   */
  function renderHighlightedText(text, spans) {
    if (!text) return "";
    if (!spans || spans.length === 0) return escapeHtml(text);

    // Sort spans ascending by start
    const sorted = [...spans].sort((a, b) => a.start - b.start);
    let html = "";
    let lastIdx = 0;

    for (const span of sorted) {
      if (span.start > lastIdx) {
        html += escapeHtml(text.slice(lastIdx, span.start));
      }
      const spanText = text.slice(span.start, span.end);
      const kind = span.kind || "number";
      html += `<span class="ptm-span ptm-${kind}" title="${kind.toUpperCase()}">${escapeHtml(spanText)}</span>`;
      lastIdx = span.end;
    }

    if (lastIdx < text.length) {
      html += escapeHtml(text.slice(lastIdx));
    }

    return html;
  }

  /**
   * Tab Navigation handler
   */
  function initTabs() {
    tabButtons.forEach(btn => {
      btn.addEventListener("click", () => {
        const tabId = btn.getAttribute("data-tab");
        tabButtons.forEach(b => b.classList.remove("active"));
        tabContents.forEach(c => c.classList.remove("active"));

        btn.classList.add("active");
        const activeContent = document.getElementById("tab-" + tabId);
        if (activeContent) activeContent.classList.add("active");
      });
    });
  }

  /**
   * Load data files
   */
  async function loadData() {
    try {
      const [queriesRes, summaryRes] = await Promise.all([
        fetch("./data/queries.json"),
        fetch("./data/summary.json")
      ]);

      if (!queriesRes.ok) throw new Error("Failed to load queries.json");
      if (!summaryRes.ok) throw new Error("Failed to load summary.json");

      queriesData = await queriesRes.json();
      summaryData = await summaryRes.json();

      populateQuerySelect(queriesData);
      populateBenchmarkTable(summaryData.benchmark_table);

      // Select first query by default
      if (queriesData.length > 0) {
        selectQuery(0);
      }
    } catch (err) {
      console.error("Initialization error:", err);
      questionText.textContent = "Error loading demonstration data: " + err.message;
    }
  }

  /**
   * Populate query dropdown
   */
  function populateQuerySelect(list) {
    querySelect.innerHTML = "";
    list.forEach(q => {
      const opt = document.createElement("option");
      opt.value = q.idx;
      const snippet = q.question.length > 70 ? q.question.slice(0, 70) + "..." : q.question;
      opt.textContent = `#${q.idx + 1} (${q.id.slice(0, 8)}) - ${snippet}`;
      querySelect.appendChild(opt);
    });
  }

  /**
   * Filter query dropdown based on text input
   */
  function filterQueries(searchStr) {
    const term = searchStr.trim().toLowerCase();
    const filtered = queriesData.filter(q => {
      return q.question.toLowerCase().includes(term) || q.id.toLowerCase().includes(term);
    });
    populateQuerySelect(filtered);
    if (filtered.length > 0) {
      selectQuery(filtered[0].idx);
    }
  }

  /**
   * Select a specific query by its dataset index
   */
  function selectQuery(idx) {
    currentQuery = queriesData.find(q => q.idx === Number(idx)) || queriesData[0];
    querySelect.value = currentQuery.idx;
    goldRevealed = false;
    updatePipelineUI();
  }

  /**
   * Update the entire Pipeline Demo tab for the current query
   */
  function updatePipelineUI() {
    if (!currentQuery) return;

    // 1. Question & Reveal Area
    qidBadge.textContent = "ID: " + currentQuery.id;
    questionText.textContent = currentQuery.question;
    goldAnswerText.textContent = currentQuery.gold_answer;

    if (goldRevealed) {
      goldAnswerContainer.style.display = "inline-flex";
      btnRevealGold.textContent = "Hide Gold Answer";
      retrievalCoverageBadge.style.display = "inline-block";
      if (currentQuery.gold_all_in_top5) {
        retrievalCoverageBadge.className = "badge badge-success";
        retrievalCoverageBadge.textContent = "All Gold Paragraphs Retrieved (Complete)";
      } else {
        retrievalCoverageBadge.className = "badge badge-danger";
        retrievalCoverageBadge.textContent = "Incomplete Retrieval (Partial / Missing Gold)";
      }
      evalBadgeContainer.style.display = "block";
    } else {
      goldAnswerContainer.style.display = "none";
      btnRevealGold.textContent = "Reveal Gold Answer";
      retrievalCoverageBadge.style.display = "none";
      evalBadgeContainer.style.display = "none";
    }

    // 2. Chunks List
    chunksList.innerHTML = "";
    currentQuery.chunks.forEach(c => {
      const card = document.createElement("div");
      card.className = "chunk-card";
      const scorePct = Math.min(100, Math.max(0, (c.score / 1.0) * 100));
      card.innerHTML = `
        <div class="chunk-header">
          <span class="chunk-title">#${c.rank} &bull; ${escapeHtml(c.title)}</span>
          <div class="score-bar-wrap">
            <div class="score-bar-bg">
              <div class="score-bar-fill" style="width: ${scorePct}%;"></div>
            </div>
            <span class="score-val">${c.score.toFixed(4)}</span>
          </div>
        </div>
        <div class="chunk-snippet">${escapeHtml(c.text)}</div>
      `;
      chunksList.appendChild(card);
    });

    // Telemetry Features Strip
    const tf = currentQuery.telemetry_features;
    telemetryStrip.innerHTML = `
      <div class="metric-box"><div class="label">Top-1 Score</div><div class="val">${tf.top1_score.toFixed(3)}</div></div>
      <div class="metric-box"><div class="label">Top-2 Score</div><div class="val">${tf.top2_score.toFixed(3)}</div></div>
      <div class="metric-box"><div class="label">Score Gap</div><div class="val">${tf.score_gap.toFixed(3)}</div></div>
      <div class="metric-box"><div class="label">Mean Score</div><div class="val">${tf.mean_score.toFixed(3)}</div></div>
      <div class="metric-box"><div class="label">Score Std</div><div class="val">${tf.score_std.toFixed(3)}</div></div>
      <div class="metric-box"><div class="label">Entropy</div><div class="val">${tf.softmax_entropy.toFixed(3)}</div></div>
      <div class="metric-box"><div class="label">Chunk Sim</div><div class="val">${tf.chunk_sim_mean.toFixed(3)}</div></div>
      <div class="metric-box"><div class="label">Context Toks</div><div class="val">${tf.context_tokens}</div></div>
    `;

    // 3. Router Decision Evaluation
    updateRouterDecision();

    // 4. Compression Display
    updateCompressionView();
  }

  /**
   * Recompute Router Decision based on variant and threshold
   */
  function updateRouterDecision() {
    if (!currentQuery) return;

    const variant = routerVariantSelect.value;
    variantDesc.textContent = VARIANT_DESCRIPTIONS[variant] || "";

    let p_hard = 0.5;
    let willEscalate = false;
    const threshold = parseFloat(thresholdSlider.value);

    if (variant === "heuristic") {
      thresholdControlGroup.style.opacity = "0.4";
      thresholdSlider.disabled = true;
      p_hard = currentQuery.router_probs.heuristic.prob;
      willEscalate = currentQuery.router_probs.heuristic.predict_large;
    } else {
      thresholdControlGroup.style.opacity = "1";
      thresholdSlider.disabled = false;

      if (variant === "telemetry_only_logistic") {
        p_hard = currentQuery.router_probs.telemetry_only.logistic;
      } else if (variant === "telemetry_only_mlp") {
        p_hard = currentQuery.router_probs.telemetry_only.mlp;
      } else if (variant === "query_only_logistic") {
        p_hard = currentQuery.router_probs.query_only.logistic;
      } else if (variant === "both_logistic") {
        p_hard = currentQuery.router_probs.both.logistic;
      }

      willEscalate = (p_hard >= threshold);
    }

    probHardValue.textContent = p_hard.toFixed(4);

    if (willEscalate) {
      routerDecisionBanner.className = "router-decision-box decision-escalate";
      decisionText.textContent = "ESCALATE -> LARGE TIER";
    } else {
      routerDecisionBanner.className = "router-decision-box decision-stay-small";
      decisionText.textContent = "ROUTE -> SMALL TIER";
    }

    // Now update Step 5 with the resulting model answer
    updateStep5Answer(willEscalate);
  }

  /**
   * Update Compression view
   */
  function updateCompressionView() {
    if (!currentQuery) return;

    origTokCount.textContent = `Tokens: ${currentQuery.original_tokens}`;
    
    // Highlight original context using PTM spans
    originalContextView.innerHTML = renderHighlightedText(
      currentQuery.original_context,
      currentQuery.ptm_protected_spans
    );

    if (currentQuery.bypassed) {
      compBypassBadge.innerHTML = '<span class="badge badge-info">Bypassed (&lt;400 tokens)</span>';
      compTokCount.textContent = `Tokens: ${currentQuery.original_tokens} (Unmodified)`;
      compressedContextView.textContent = currentQuery.original_context;
    } else {
      compBypassBadge.innerHTML = `<span class="badge badge-success">Saved ${currentQuery.tokens_saved_pct}% tokens (in ${currentQuery.compression_seconds}s)</span>`;
      compTokCount.textContent = `Tokens: ${currentQuery.compressed_tokens} (-${currentQuery.tokens_saved_pct}%)`;
      // Run PTM on compressed context as well for highlighting visual feedback
      const compSpans = findProtectedSpansJS(currentQuery.compressed_context);
      compressedContextView.innerHTML = renderHighlightedText(currentQuery.compressed_context, compSpans);
    }
  }

  /**
   * Update Final Answer and Economics (Step 5)
   */
  function updateStep5Answer(isLarge) {
    if (!currentQuery) return;

    const isCompressed = compressToggle.checked;
    let answerKey = "";

    if (isLarge) {
      answerKey = isCompressed ? "large_compressed" : "large_uncompressed";
    } else {
      answerKey = isCompressed ? "small_compressed" : "small_uncompressed";
    }

    const ansData = currentQuery.answers[answerKey];
    if (!ansData) return;

    // Model label
    if (isLarge) {
      activeModelLabel.innerHTML = `<span style="color: #f87171;">ESCALATED TO LARGE TIER</span> &bull; Gemini 3.5 Flash-Lite (Google AI Studio Free Tier)`;
    } else {
      activeModelLabel.innerHTML = `<span style="color: #4ade80;">SERVED BY SMALL TIER</span> &bull; ALLaM-2-7B (Groq Free Tier, 8B list proxy)`;
    }

    activeAnswerText.textContent = `"${ansData.answer}"`;

    // Latency & Cost Metrics
    valLlmLats.textContent = `${ansData.llm_latency_s.toFixed(3)} s`;
    valE2eLats.textContent = `${ansData.e2e_latency_s.toFixed(3)} s`;
    valCost1k.textContent = `$${ansData.cost_per_1k.toFixed(4)}`;
    valCostQuery.textContent = `$${ansData.cost_per_query.toFixed(6)}`;

    // Evaluation badge
    if (goldRevealed) {
      evalBadgeContainer.style.display = "block";
      if (ansData.is_correct) {
        evalBadgeContainer.innerHTML = '<span class="badge badge-success">CORRECT PREDICTION (+)</span>';
      } else {
        evalBadgeContainer.innerHTML = '<span class="badge badge-danger">INCORRECT PREDICTION (-)</span>';
      }
    } else {
      evalBadgeContainer.style.display = "none";
    }
  }

  /**
   * Populate Benchmark summary table in Results tab
   */
  function populateBenchmarkTable(rows) {
    benchmarkTableBody.innerHTML = "";
    if (!rows) return;

    rows.forEach(r => {
      const tr = document.createElement("tr");
      const isOptiRAG = r.system.includes("OptiRAG");
      if (isOptiRAG) {
        tr.style.background = "rgba(56, 189, 248, 0.12)";
        tr.style.fontWeight = "bold";
      }

      tr.innerHTML = `
        <td>${escapeHtml(r.system)}</td>
        <td>${escapeHtml(r.threshold_method)}</td>
        <td><code>${escapeHtml(r.decision_threshold)}</code></td>
        <td>${(parseFloat(r.fraction_to_large) * 100).toFixed(0)}%</td>
        <td><strong>${r.accuracy}%</strong> [${r.accuracy_ci_low}, ${r.accuracy_ci_high}]</td>
        <td>${parseFloat(r.mean_f1).toFixed(3)}</td>
        <td>$${parseFloat(r.cost_per_1k).toFixed(4)}</td>
        <td>${parseFloat(r.llm_only_latency_s).toFixed(2)} s</td>
        <td>${parseFloat(r.end_to_end_latency_s).toFixed(2)} s</td>
      `;
      benchmarkTableBody.appendChild(tr);
    });
  }

  /**
   * PTM Live Sandbox Handler
   */
  function updateMaskSandbox() {
    const text = maskInput.value || "";
    const spans = findProtectedSpansJS(text);

    // Counts per kind
    const counts = { currency: 0, date: 0, percent: 0, negation: 0, number: 0 };
    spans.forEach(s => {
      if (counts[s.kind] !== undefined) counts[s.kind]++;
    });

    cntCurrency.textContent = counts.currency;
    cntDate.textContent = counts.date;
    cntPercent.textContent = counts.percent;
    cntNegation.textContent = counts.negation;
    cntNumber.textContent = counts.number;

    // Word statistics
    const words = text.trim() ? text.trim().split(/\s+/) : [];
    const totalWords = words.length;

    let protectedWordCount = 0;
    if (totalWords > 0) {
      spans.forEach(s => {
        const spanStr = text.slice(s.start, s.end).trim();
        if (spanStr) {
          protectedWordCount += spanStr.split(/\s+/).length;
        }
      });
    }

    const pct = totalWords > 0 ? Math.min(100, (protectedWordCount / totalWords) * 100).toFixed(1) : "0.0";
    pctProtectedWords.textContent = `${pct}% (${protectedWordCount}/${totalWords} words)`;

    // Render highlighted preview
    maskHighlightPreview.innerHTML = renderHighlightedText(text, spans);
  }

  /**
   * Initialize Event Listeners
   */
  function initEvents() {
    initTabs();

    // Query selection & search
    querySelect.addEventListener("change", (e) => selectQuery(e.target.value));
    querySearch.addEventListener("input", (e) => filterQueries(e.target.value));

    btnRandom.addEventListener("click", () => {
      if (queriesData.length === 0) return;
      const randIdx = Math.floor(Math.random() * queriesData.length);
      selectQuery(randIdx);
    });

    // Reveal Gold Answer
    btnRevealGold.addEventListener("click", () => {
      goldRevealed = !goldRevealed;
      updatePipelineUI();
    });

    // Router variant & slider
    routerVariantSelect.addEventListener("change", updateRouterDecision);
    thresholdSlider.addEventListener("input", (e) => {
      thresholdValue.textContent = parseFloat(e.target.value).toFixed(2);
      updateRouterDecision();
    });

    // Compression toggle
    compressToggle.addEventListener("change", () => {
      const isLarge = decisionText.textContent.includes("LARGE");
      updateStep5Answer(isLarge);
    });

    // Mask Sandbox
    maskInput.addEventListener("input", updateMaskSandbox);

    exBtnFinance.addEventListener("click", () => {
      maskInput.value = MASK_EXAMPLES.finance;
      updateMaskSandbox();
    });

    exBtnBio.addEventListener("click", () => {
      maskInput.value = MASK_EXAMPLES.bio;
      updateMaskSandbox();
    });

    exBtnPlain.addEventListener("click", () => {
      maskInput.value = MASK_EXAMPLES.plain;
      updateMaskSandbox();
    });
  }

  // Startup
  window.addEventListener("DOMContentLoaded", () => {
    initEvents();
    loadData();

    // Prefill mask sandbox with finance example
    maskInput.value = MASK_EXAMPLES.finance;
    updateMaskSandbox();
  });

})();
