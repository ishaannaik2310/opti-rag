/**
 * OptiRAG-lite Interactive Demonstration Client Application (app.js).
 * Minimalist, cache-first, zero-cost web application for academic demonstration.
 * Features:
 *   - 6-Stage Live Demo Stepper (Query -> Retrieve -> Route -> Compress -> Answer -> Summary)
 *   - 3 Curated Presets (Easy, Hard, Mask Saves a Fact)
 *   - Offline Replay by default + Graceful Custom Question matching
 *   - Presenter Mode (Larger typography, keyboard stepping with Right/Left/R)
 *   - Simplified Pipeline Replay & Hero Benchmark views
 * Runs 100% offline with zero CDN dependencies and zero external API calls.
 */

(function () {
  "use strict";

  // =========================================================================
  // Application State
  // =========================================================================
  let queriesData = [];
  let summaryData = null;
  let currentQuery = null;
  let goldRevealed = false;

  // Live Demo State
  let currentLiveStep = 1; // 1 to 6
  let liveGoldRevealed = false;
  let presenterMode = false;
  let isCustomMode = false;
  let presetIndices = { easy: 7, hard: 57, mask: 28 };
  let runAllInterval = null;

  // =========================================================================
  // DOM Elements - Theme & Tabs
  // =========================================================================
  const btnThemeToggle = document.getElementById("btnThemeToggle");
  const themeIcon = document.getElementById("themeIcon");
  const themeLabel = document.getElementById("themeLabel");
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  // =========================================================================
  // DOM Elements - Live Demo Tab
  // =========================================================================
  const livePresetEasy = document.getElementById("livePresetEasy");
  const livePresetHard = document.getElementById("livePresetHard");
  const livePresetMask = document.getElementById("livePresetMask");
  const livePresetRandom = document.getElementById("livePresetRandom");
  const btnPresenterMode = document.getElementById("btnPresenterMode");
  const presenterIcon = document.getElementById("presenterIcon");
  const presenterText = document.getElementById("presenterText");

  const btnModeReplay = document.getElementById("btnModeReplay");
  const btnModeCustom = document.getElementById("btnModeCustom");
  const liveReplayWrap = document.getElementById("liveReplayWrap");
  const liveCustomWrap = document.getElementById("liveCustomWrap");
  const liveQuerySelect = document.getElementById("liveQuerySelect");
  const liveCustomInput = document.getElementById("liveCustomInput");
  const btnLiveCustomRun = document.getElementById("btnLiveCustomRun");
  const liveCustomNote = document.getElementById("liveCustomNote");

  const stepPills = document.querySelectorAll(".step-pill");
  const btnLivePrev = document.getElementById("btnLivePrev");
  const btnLiveNext = document.getElementById("btnLiveNext");
  const btnLiveRunAll = document.getElementById("btnLiveRunAll");
  const btnLiveReset = document.getElementById("btnLiveReset");

  const liveStepCards = [
    document.getElementById("liveStep1"),
    document.getElementById("liveStep2"),
    document.getElementById("liveStep3"),
    document.getElementById("liveStep4"),
    document.getElementById("liveStep5"),
    document.getElementById("liveStep6")
  ];

  const liveQidBadge = document.getElementById("liveQidBadge");
  const liveQuestionText = document.getElementById("liveQuestionText");
  const liveRetrievalConfidenceBadge = document.getElementById("liveRetrievalConfidenceBadge");
  const livePassagesList = document.getElementById("livePassagesList");
  const liveRouteDecisionBadge = document.getElementById("liveRouteDecisionBadge");
  const liveRouteReasonText = document.getElementById("liveRouteReasonText");
  const liveCompressionBadge = document.getElementById("liveCompressionBadge");
  const liveOrigContextView = document.getElementById("liveOrigContextView");
  const liveCompContextView = document.getElementById("liveCompContextView");
  const liveModelBadge = document.getElementById("liveModelBadge");
  const liveAnswerText = document.getElementById("liveAnswerText");
  const btnLiveRevealGold = document.getElementById("btnLiveRevealGold");
  const liveGoldContainer = document.getElementById("liveGoldContainer");
  const liveGoldAnswerText = document.getElementById("liveGoldAnswerText");
  const liveEvalStatusBadge = document.getElementById("liveEvalStatusBadge");
  const liveSummaryStrip = document.getElementById("liveSummaryStrip");

  // =========================================================================
  // DOM Elements - Pipeline Replay Tab (Existing)
  // =========================================================================
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

  // =========================================================================
  // DOM Elements - Try the Mask Tab
  // =========================================================================
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
   * Escape HTML to prevent XSS
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
   * Render original context with PTM spans highlighted AND dropped words faded with strikethrough.
   * This provides clear visual evidence of what LLMLingua-2 pruned while facts were saved.
   */
  function renderContextWithPrunedFade(origText, compText, spans) {
    if (!origText) return "";
    const compWords = new Set((compText.match(/[a-zA-Z0-9]+/g) || []).map(w => w.toLowerCase()));
    const sorted = [...(spans || [])].sort((a, b) => a.start - b.start);

    let html = "";
    let lastIdx = 0;

    for (const span of sorted) {
      if (span.start > lastIdx) {
        html += renderFadedSegment(origText.slice(lastIdx, span.start), compWords);
      }
      const spanText = origText.slice(span.start, span.end);
      const kind = span.kind || "number";
      html += `<span class="ptm-span ptm-${kind}" title="${kind.toUpperCase()}">${escapeHtml(spanText)}</span>`;
      lastIdx = span.end;
    }

    if (lastIdx < origText.length) {
      html += renderFadedSegment(origText.slice(lastIdx), compWords);
    }

    return html;
  }

  function renderFadedSegment(text, compWords) {
    const tokens = text.split(/(\s+)/);
    let res = "";
    for (const tok of tokens) {
      if (/^\s+$/.test(tok)) {
        res += tok;
      } else {
        const clean = tok.replace(/[^a-zA-Z0-9]/g, "").toLowerCase();
        if (!clean || compWords.has(clean)) {
          res += escapeHtml(tok);
        } else {
          res += `<span class="pruned">${escapeHtml(tok)}</span>`;
        }
      }
    }
    return res;
  }

  /**
   * Theme Switcher (Light / Dark mode)
   */
  function initTheme() {
    let saved = "dark";
    try {
      saved = localStorage.getItem("optirag_theme") || "dark";
    } catch (e) {}
    setTheme(saved);

    if (btnThemeToggle) {
      btnThemeToggle.addEventListener("click", () => {
        const current = document.documentElement.getAttribute("data-theme") || "dark";
        const next = current === "dark" ? "light" : "dark";
        setTheme(next);
      });
    }
  }

  function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("optirag_theme", theme);
    } catch (e) {}

    if (themeIcon && themeLabel) {
      if (theme === "dark") {
        themeIcon.textContent = "🌞";
        themeLabel.textContent = "Light Mode";
      } else {
        themeIcon.textContent = "🌙";
        themeLabel.textContent = "Dark Mode";
      }
    }
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
   * Find preset questions automatically from data
   */
  function findPresets(list) {
    // 1. Easy: routed to Small (p_hard < 0.45), Small correct, all gold retrieved
    const easy = list.find(q => {
      const p = q.router_probs.telemetry_only.logistic;
      return p < 0.45 && q.answers.small_compressed.is_correct && q.gold_all_in_top5;
    });
    if (easy) presetIndices.easy = easy.idx;

    // 2. Hard: routed to Large (p_hard >= 0.52), Small wrong, Large correct
    const hard = list.find(q => {
      const p = q.router_probs.telemetry_only.logistic;
      return p >= 0.52 && !q.answers.small_compressed.is_correct && q.answers.large_compressed.is_correct;
    });
    if (hard) presetIndices.hard = hard.idx;

    // 3. Mask saves a fact: Vanilla Small wrong, but PTM Small correct, has date/currency/negation
    const mask = list.find(q => {
      const kinds = q.ptm_protected_spans.map(s => s.kind);
      const hasFact = kinds.includes("date") || kinds.includes("negation") || kinds.includes("currency");
      return !q.answers.small_uncompressed.is_correct && q.answers.small_compressed.is_correct && hasFact;
    });
    if (mask) presetIndices.mask = mask.idx;
  }

  /**
   * Find closest cached query to an arbitrary user string using Jaccard word similarity
   */
  function findClosestCachedQuery(queryStr) {
    const terms = (queryStr.toLowerCase().match(/[a-zA-Z0-9]+/g) || []);
    if (terms.length === 0) return queriesData[0];
    const querySet = new Set(terms);

    let bestScore = -1;
    let bestQuery = queriesData[0];

    for (const q of queriesData) {
      const qTerms = (q.question.toLowerCase().match(/[a-zA-Z0-9]+/g) || []);
      let matchCount = 0;
      for (const t of qTerms) {
        if (querySet.has(t)) matchCount++;
      }
      const unionSize = new Set([...terms, ...qTerms]).size;
      const score = unionSize > 0 ? matchCount / unionSize : 0;
      if (score > bestScore) {
        bestScore = score;
        bestQuery = q;
      }
    }
    return bestQuery;
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

      findPresets(queriesData);
      populateQuerySelect(queriesData);
      populateLiveQuerySelect(queriesData);
      populateBenchmarkTable(summaryData.benchmark_table);

      // Select Easy preset or first query by default
      selectQuery(presetIndices.easy !== undefined ? presetIndices.easy : 0);
    } catch (err) {
      console.error("Initialization error:", err);
      if (questionText) questionText.textContent = "Error loading demonstration data: " + err.message;
    }
  }

  /**
   * Populate query dropdown in Pipeline Replay
   */
  function populateQuerySelect(list) {
    if (!querySelect) return;
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
   * Populate query dropdown in Live Demo
   */
  function populateLiveQuerySelect(list) {
    if (!liveQuerySelect) return;
    liveQuerySelect.innerHTML = "";
    list.forEach(q => {
      const opt = document.createElement("option");
      opt.value = q.idx;
      const snippet = q.question.length > 70 ? q.question.slice(0, 70) + "..." : q.question;
      opt.textContent = `#${q.idx + 1} - ${snippet}`;
      liveQuerySelect.appendChild(opt);
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
    if (querySelect) querySelect.value = currentQuery.idx;
    if (liveQuerySelect) liveQuerySelect.value = currentQuery.idx;
    goldRevealed = false;
    liveGoldRevealed = false;

    // Reset live demo stepper to stage 1 whenever question changes
    setLiveStep(1);

    updatePipelineUI();
    updateLiveUI();
  }

  // =========================================================================
  // LIVE DEMO STEPPER CONTROLS
  // =========================================================================

  /**
   * Update Stepper visible stage cards (Stages 1 through currentStep)
   */
  function setLiveStep(step) {
    currentLiveStep = Math.max(1, Math.min(6, step));

    // Update step pills
    stepPills.forEach((pill, idx) => {
      const pStep = idx + 1;
      pill.classList.remove("active", "completed");
      if (pStep === currentLiveStep) {
        pill.classList.add("active");
      } else if (pStep < currentLiveStep) {
        pill.classList.add("completed");
      }
    });

    // Update fluid progress bar
    const liveProgressFill = document.getElementById("liveProgressFill");
    if (liveProgressFill) {
      liveProgressFill.style.width = ((currentLiveStep / 6) * 100).toFixed(1) + "%";
    }

    // Update fluid flow nodes along the timeline spine
    for (let i = 1; i <= 6; i++) {
      const node = document.getElementById(`flowNode${i}`);
      if (node) {
        if (i <= currentLiveStep) {
          node.classList.add("active");
        } else {
          node.classList.remove("active");
        }
      }
    }

    // Reveal or hide cards
    liveStepCards.forEach((card, idx) => {
      const cardStep = idx + 1;
      if (cardStep <= currentLiveStep) {
        card.classList.remove("hidden");
        if (cardStep === currentLiveStep) {
          card.classList.add("revealing");
        }
      } else {
        card.classList.add("hidden");
        card.classList.remove("revealing");
      }
    });

    // Update Action Buttons
    if (btnLivePrev) btnLivePrev.disabled = (currentLiveStep === 1);
    if (btnLiveNext) {
      if (currentLiveStep === 6) {
        btnLiveNext.textContent = "Done ✓";
        btnLiveNext.disabled = true;
      } else {
        btnLiveNext.textContent = "Next Stage →";
        btnLiveNext.disabled = false;
      }
    }
  }

  function advanceLiveStep() {
    if (currentLiveStep < 6) {
      setLiveStep(currentLiveStep + 1);
    }
  }

  function previousLiveStep() {
    if (currentLiveStep > 1) {
      setLiveStep(currentLiveStep - 1);
    }
  }

  function resetLiveSteps() {
    if (runAllInterval) clearInterval(runAllInterval);
    liveGoldRevealed = false;
    setLiveStep(1);
    updateLiveUI();
  }

  function runAllLiveSteps() {
    if (runAllInterval) clearInterval(runAllInterval);
    setLiveStep(1);
    let s = 1;
    runAllInterval = setInterval(() => {
      s++;
      if (s <= 6) {
        setLiveStep(s);
      } else {
        clearInterval(runAllInterval);
      }
    }, 450);
  }

  /**
   * Render all 6 Live Demo Stages for currentQuery
   */
  function updateLiveUI() {
    if (!currentQuery) return;

    // -------------------------------------------------------------
    // Stage 1: Query
    // -------------------------------------------------------------
    liveQidBadge.textContent = "HotpotQA ID: " + currentQuery.id.slice(0, 8);
    liveQuestionText.textContent = currentQuery.question;

    // -------------------------------------------------------------
    // Stage 2: Dense Retrieval & Confidence
    // -------------------------------------------------------------
    const tf = currentQuery.telemetry_features;
    let confBadgeText = "Retrieval Confidence: High";
    let confBadgeClass = "badge-notion success";

    if (tf.top1_score >= 0.55 && tf.score_gap >= 0.08) {
      confBadgeText = "Retrieval Confidence: High";
      confBadgeClass = "badge-notion success";
    } else if (tf.top1_score >= 0.44 && tf.score_gap >= 0.03) {
      confBadgeText = "Retrieval Confidence: Medium";
      confBadgeClass = "badge-notion info";
    } else {
      confBadgeText = "Retrieval Confidence: Low (Ambiguous)";
      confBadgeClass = "badge-notion warning";
    }

    liveRetrievalConfidenceBadge.className = confBadgeClass;
    liveRetrievalConfidenceBadge.textContent = confBadgeText;

    // Render 5 clean passage snippets
    livePassagesList.innerHTML = "";
    currentQuery.chunks.forEach((c) => {
      const item = document.createElement("div");
      item.className = "live-passage-item";
      const snippet = c.text.length > 130 ? c.text.slice(0, 130) + "..." : c.text;
      item.innerHTML = `
        <div class="live-passage-item-top">
          <span class="live-passage-title">#${c.rank} &bull; ${escapeHtml(c.title)}</span>
          <span style="font-size: 11px; font-family: monospace; color: var(--text-tertiary);">${c.score.toFixed(3)}</span>
        </div>
        <div class="live-passage-snippet">${escapeHtml(snippet)}</div>
      `;
      livePassagesList.appendChild(item);
    });

    // -------------------------------------------------------------
    // Stage 3: Cascade Router Decision
    // -------------------------------------------------------------
    // Using calibrated telemetry logistic model (threshold = 0.520)
    const p_hard = currentQuery.router_probs.telemetry_only.logistic;
    const willEscalate = (p_hard >= 0.520);

    if (willEscalate) {
      liveRouteDecisionBadge.className = "badge-notion danger";
      liveRouteDecisionBadge.textContent = "ESCALATE TO LARGE MODEL (Gemini 3.5 Flash-Lite)";
      liveRouteReasonText.innerHTML = `
        <strong>Routing Reason:</strong> Retrieval signals indicate higher ambiguity or multi-hop difficulty 
        (P(hard) = ${p_hard.toFixed(3)} &ge; &tau; 0.520). The query is escalated to the frontier model for reasoning synthesis.
      `;
    } else {
      liveRouteDecisionBadge.className = "badge-notion success";
      liveRouteDecisionBadge.textContent = "ROUTE TO SMALL MODEL (ALLaM-2-7B)";
      liveRouteReasonText.innerHTML = `
        <strong>Routing Reason:</strong> Retrieval confidence is sufficient and question difficulty is manageable 
        (P(hard) = ${p_hard.toFixed(3)} &lt; &tau; 0.520). The query is handled by the 7B model at 90% lower inference cost.
      `;
    }

    // Update fluid routing gate visual paths
    const liveRouteGateVal = document.getElementById("liveRouteGateVal");
    const gateArmSmall = document.getElementById("gateArmSmall");
    const gateArmLarge = document.getElementById("gateArmLarge");
    if (liveRouteGateVal) liveRouteGateVal.textContent = `P(hard) = ${p_hard.toFixed(3)}`;
    if (gateArmSmall && gateArmLarge) {
      if (willEscalate) {
        gateArmLarge.classList.add("selected");
        gateArmLarge.classList.remove("dimmed");
        gateArmSmall.classList.add("dimmed");
        gateArmSmall.classList.remove("selected");
      } else {
        gateArmSmall.classList.add("selected");
        gateArmSmall.classList.remove("dimmed");
        gateArmLarge.classList.add("dimmed");
        gateArmLarge.classList.remove("selected");
      }
    }

    // -------------------------------------------------------------
    // Stage 4: PTM Context Compression
    // -------------------------------------------------------------
    const liveMeterRetained = document.getElementById("liveMeterRetained");
    const liveMeterPruned = document.getElementById("liveMeterPruned");
    const liveMeterRetainedText = document.getElementById("liveMeterRetainedText");
    const liveMeterPrunedText = document.getElementById("liveMeterPrunedText");

    if (currentQuery.bypassed) {
      liveCompressionBadge.className = "badge-notion info";
      liveCompressionBadge.textContent = "Bypassed (< 400 tokens) · Unmodified";
      liveOrigContextView.innerHTML = renderHighlightedText(currentQuery.original_context, currentQuery.ptm_protected_spans);
      liveCompContextView.textContent = currentQuery.original_context;
      if (liveMeterRetained && liveMeterPruned) {
        liveMeterRetained.style.width = "100%";
        liveMeterPruned.style.width = "0%";
      }
      if (liveMeterRetainedText) liveMeterRetainedText.textContent = `Retained Context (${currentQuery.original_tokens} tokens - Bypassed)`;
      if (liveMeterPrunedText) liveMeterPrunedText.textContent = "0% Pruned (Short Context)";
    } else {
      liveCompressionBadge.className = "badge-notion success";
      liveCompressionBadge.textContent = `${currentQuery.tokens_saved_pct}% Shorter · ${currentQuery.original_tokens} → ${currentQuery.compressed_tokens} tokens`;
      
      if (liveMeterRetained && liveMeterPruned) {
        const retainedPct = Math.max(10, 100 - currentQuery.tokens_saved_pct);
        liveMeterRetained.style.width = retainedPct + "%";
        liveMeterPruned.style.width = currentQuery.tokens_saved_pct + "%";
      }
      if (liveMeterRetainedText) liveMeterRetainedText.textContent = `Retained Context (${currentQuery.compressed_tokens} tokens)`;
      if (liveMeterPrunedText) liveMeterPrunedText.textContent = `Pruned Peripheral Tokens (-${currentQuery.tokens_saved_pct}%)`;

      // Before: Highlight protected tokens in color, fade pruned tokens with subtle strikethrough
      liveOrigContextView.innerHTML = renderContextWithPrunedFade(
        currentQuery.original_context,
        currentQuery.compressed_context,
        currentQuery.ptm_protected_spans
      );

      // After: Highlight preserved entities in compressed text
      const compSpans = findProtectedSpansJS(currentQuery.compressed_context);
      liveCompContextView.innerHTML = renderHighlightedText(currentQuery.compressed_context, compSpans);
    }

    // -------------------------------------------------------------
    // Stage 5: Final LLM Answer & Verification
    // -------------------------------------------------------------
    const activeKey = willEscalate ? "large_compressed" : "small_compressed";
    const ansData = currentQuery.answers[activeKey];

    if (willEscalate) {
      liveModelBadge.className = "badge-notion danger";
      liveModelBadge.textContent = "Served by Large Tier (Gemini 3.5 Flash-Lite)";
    } else {
      liveModelBadge.className = "badge-notion success";
      liveModelBadge.textContent = "Served by Small Tier (ALLaM-2-7B on Groq)";
    }

    liveAnswerText.textContent = `"${ansData.answer}"`;
    liveGoldAnswerText.textContent = currentQuery.gold_answer;

    if (liveGoldRevealed) {
      liveGoldContainer.style.display = "inline-flex";
      btnLiveRevealGold.textContent = "Hide Ground Truth";
      if (ansData.is_correct) {
        liveEvalStatusBadge.className = "badge-notion success";
        liveEvalStatusBadge.textContent = "CORRECT ✓";
      } else {
        liveEvalStatusBadge.className = "badge-notion danger";
        liveEvalStatusBadge.textContent = "INCORRECT ✗";
      }
    } else {
      liveGoldContainer.style.display = "none";
      btnLiveRevealGold.textContent = "Reveal Ground Truth";
    }

    // -------------------------------------------------------------
    // Stage 6: Summary Strip (Without OptiRAG vs. With OptiRAG)
    // -------------------------------------------------------------
    const baselineAns = currentQuery.answers.large_uncompressed;
    const optiAns = ansData;

    const tokSavedPct = Math.max(0, Math.round(((baselineAns.in_tokens - optiAns.in_tokens) / baselineAns.in_tokens) * 100));
    const costSavedPct = Math.max(0, Math.round(((baselineAns.cost_per_1k - optiAns.cost_per_1k) / baselineAns.cost_per_1k) * 100));

    liveSummaryStrip.innerHTML = `
      <div class="summary-metric-card">
        <div class="summary-metric-label">Tokens Used</div>
        <div class="summary-metric-value">${baselineAns.in_tokens} → ${optiAns.in_tokens} (${tokSavedPct}% shorter)</div>
        <div class="summary-metric-sub">Without OptiRAG: ${baselineAns.in_tokens} tokens</div>
      </div>
      <div class="summary-metric-card">
        <div class="summary-metric-label">Simulated Cost / 1k</div>
        <div class="summary-metric-value">$${baselineAns.cost_per_1k.toFixed(3)} → $${optiAns.cost_per_1k.toFixed(3)} (-${costSavedPct}%)</div>
        <div class="summary-metric-sub">Without OptiRAG: $${baselineAns.cost_per_1k.toFixed(3)}</div>
      </div>
      <div class="summary-metric-card">
        <div class="summary-metric-label">Model Tier Selected</div>
        <div class="summary-metric-value">${willEscalate ? "Large Tier (Gemini)" : "Small Tier (ALLaM-2)"}</div>
        <div class="summary-metric-sub">Without OptiRAG: Always Large ($0.23/1k)</div>
      </div>
    `;
  }

  // =========================================================================
  // PIPELINE REPLAY TAB (EXISTING LOGIC)
  // =========================================================================

  function updatePipelineUI() {
    if (!currentQuery) return;

    // 1. Question & Reveal Area
    if (qidBadge) qidBadge.textContent = "ID: " + currentQuery.id;
    if (questionText) questionText.textContent = currentQuery.question;
    if (goldAnswerText) goldAnswerText.textContent = currentQuery.gold_answer;

    if (goldRevealed) {
      if (goldAnswerContainer) goldAnswerContainer.style.display = "inline-flex";
      if (btnRevealGold) btnRevealGold.textContent = "Hide Gold Answer";
      if (retrievalCoverageBadge) {
        retrievalCoverageBadge.style.display = "inline-flex";
        if (currentQuery.gold_all_in_top5) {
          retrievalCoverageBadge.className = "badge-notion success";
          retrievalCoverageBadge.textContent = "All Gold Paragraphs Retrieved (Complete)";
        } else {
          retrievalCoverageBadge.className = "badge-notion danger";
          retrievalCoverageBadge.textContent = "Incomplete Retrieval (Partial / Missing Gold)";
        }
      }
      if (evalBadgeContainer) evalBadgeContainer.style.display = "block";
    } else {
      if (goldAnswerContainer) goldAnswerContainer.style.display = "none";
      if (btnRevealGold) btnRevealGold.textContent = "Reveal Gold Answer";
      if (retrievalCoverageBadge) retrievalCoverageBadge.style.display = "none";
      if (evalBadgeContainer) evalBadgeContainer.style.display = "none";
    }

    // 2. Chunks List
    if (chunksList) {
      chunksList.innerHTML = "";
      currentQuery.chunks.forEach(c => {
        const card = document.createElement("div");
        card.className = "chunk-row";
        const scorePct = Math.min(100, Math.max(0, (c.score / 1.0) * 100));
        card.innerHTML = `
          <div class="chunk-top">
            <span class="chunk-heading">#${c.rank} &bull; ${escapeHtml(c.title)}</span>
            <div class="score-badge">
              <div class="score-bar-bg">
                <div class="score-bar-fill" style="width: ${scorePct}%;"></div>
              </div>
              <span class="score-text">${c.score.toFixed(4)}</span>
            </div>
          </div>
          <div class="chunk-text">${escapeHtml(c.text)}</div>
        `;
        chunksList.appendChild(card);
      });
    }

    // Telemetry Features Strip
    if (telemetryStrip) {
      const tf = currentQuery.telemetry_features;
      telemetryStrip.innerHTML = `
        <div class="telemetry-card"><div class="telemetry-label">Top-1 Score</div><div class="telemetry-value">${tf.top1_score.toFixed(3)}</div></div>
        <div class="telemetry-card"><div class="telemetry-label">Top-2 Score</div><div class="telemetry-value">${tf.top2_score.toFixed(3)}</div></div>
        <div class="telemetry-card"><div class="telemetry-label">Score Gap</div><div class="telemetry-value">${tf.score_gap.toFixed(3)}</div></div>
        <div class="telemetry-card"><div class="telemetry-label">Mean Score</div><div class="telemetry-value">${tf.mean_score.toFixed(3)}</div></div>
        <div class="telemetry-card"><div class="telemetry-label">Score Std</div><div class="telemetry-value">${tf.score_std.toFixed(3)}</div></div>
        <div class="telemetry-card"><div class="telemetry-label">Entropy</div><div class="telemetry-value">${tf.softmax_entropy.toFixed(3)}</div></div>
        <div class="telemetry-card"><div class="telemetry-label">Chunk Sim</div><div class="telemetry-value">${tf.chunk_sim_mean.toFixed(3)}</div></div>
        <div class="telemetry-card"><div class="telemetry-label">Context Toks</div><div class="telemetry-value">${tf.context_tokens}</div></div>
      `;
    }

    // 3. Router Decision Evaluation
    updateRouterDecision();

    // 4. Compression Display
    updateCompressionView();
  }

  function updateRouterDecision() {
    if (!currentQuery || !routerVariantSelect) return;

    const variant = routerVariantSelect.value;
    if (variantDesc) variantDesc.textContent = VARIANT_DESCRIPTIONS[variant] || "";

    let p_hard = 0.5;
    let willEscalate = false;
    const threshold = parseFloat(thresholdSlider ? thresholdSlider.value : 0.52);

    if (variant === "heuristic") {
      if (thresholdControlGroup) thresholdControlGroup.style.opacity = "0.4";
      if (thresholdSlider) thresholdSlider.disabled = true;
      p_hard = currentQuery.router_probs.heuristic.prob;
      willEscalate = currentQuery.router_probs.heuristic.predict_large;
    } else {
      if (thresholdControlGroup) thresholdControlGroup.style.opacity = "1";
      if (thresholdSlider) thresholdSlider.disabled = false;

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

    if (probHardValue) probHardValue.textContent = p_hard.toFixed(4);

    if (routerDecisionBanner && decisionText) {
      if (willEscalate) {
        routerDecisionBanner.className = "router-decision-card decision-badge-escalate";
        decisionText.textContent = "ESCALATE -> LARGE TIER";
      } else {
        routerDecisionBanner.className = "router-decision-card decision-badge-small";
        decisionText.textContent = "ROUTE -> SMALL TIER";
      }
    }

    updateStep5Answer(willEscalate);
  }

  function updateCompressionView() {
    if (!currentQuery) return;

    if (origTokCount) origTokCount.textContent = `Tokens: ${currentQuery.original_tokens}`;
    
    if (originalContextView) {
      originalContextView.innerHTML = renderHighlightedText(
        currentQuery.original_context,
        currentQuery.ptm_protected_spans
      );
    }

    if (currentQuery.bypassed) {
      if (compBypassBadge) compBypassBadge.innerHTML = '<span class="badge-notion info">Bypassed (&lt;400 tokens)</span>';
      if (compTokCount) compTokCount.textContent = `Tokens: ${currentQuery.original_tokens} (Unmodified)`;
      if (compressedContextView) compressedContextView.textContent = currentQuery.original_context;
    } else {
      if (compBypassBadge) compBypassBadge.innerHTML = `<span class="badge-notion success">Saved ${currentQuery.tokens_saved_pct}% tokens (in ${currentQuery.compression_seconds}s)</span>`;
      if (compTokCount) compTokCount.textContent = `Tokens: ${currentQuery.compressed_tokens} (-${currentQuery.tokens_saved_pct}%)`;
      if (compressedContextView) {
        const compSpans = findProtectedSpansJS(currentQuery.compressed_context);
        compressedContextView.innerHTML = renderHighlightedText(currentQuery.compressed_context, compSpans);
      }
    }
  }

  function updateStep5Answer(isLarge) {
    if (!currentQuery || !activeModelLabel) return;

    const isCompressed = compressToggle ? compressToggle.checked : true;
    let answerKey = "";

    if (isLarge) {
      answerKey = isCompressed ? "large_compressed" : "large_uncompressed";
    } else {
      answerKey = isCompressed ? "small_compressed" : "small_uncompressed";
    }

    const ansData = currentQuery.answers[answerKey];
    if (!ansData) return;

    if (isLarge) {
      activeModelLabel.innerHTML = `<span style="color: var(--ptm-negation-text); font-weight: 600;">ESCALATED TO LARGE TIER</span> &bull; Gemini 3.5 Flash-Lite (Google AI Studio Free Tier)`;
    } else {
      activeModelLabel.innerHTML = `<span style="color: var(--ptm-currency-text); font-weight: 600;">SERVED BY SMALL TIER</span> &bull; ALLaM-2-7B (Groq Free Tier, 8B list proxy)`;
    }

    if (activeAnswerText) activeAnswerText.textContent = `"${ansData.answer}"`;

    if (valLlmLats) valLlmLats.textContent = `${ansData.llm_latency_s.toFixed(3)} s`;
    if (valE2eLats) valE2eLats.textContent = `${ansData.e2e_latency_s.toFixed(3)} s`;
    if (valCost1k) valCost1k.textContent = `$${ansData.cost_per_1k.toFixed(4)}`;
    if (valCostQuery) valCostQuery.textContent = `$${ansData.cost_per_query.toFixed(6)}`;

    if (goldRevealed && evalBadgeContainer) {
      evalBadgeContainer.style.display = "block";
      if (ansData.is_correct) {
        evalBadgeContainer.innerHTML = '<span class="badge-notion success">CORRECT PREDICTION (+)</span>';
      } else {
        evalBadgeContainer.innerHTML = '<span class="badge-notion danger">INCORRECT PREDICTION (-)</span>';
      }
    } else if (evalBadgeContainer) {
      evalBadgeContainer.style.display = "none";
    }
  }

  function populateBenchmarkTable(rows) {
    if (!benchmarkTableBody || !rows) return;
    benchmarkTableBody.innerHTML = "";

    rows.forEach(r => {
      const tr = document.createElement("tr");
      const isOptiRAG = r.system.includes("OptiRAG");
      if (isOptiRAG) {
        tr.style.background = "var(--callout-blue-bg)";
        tr.style.fontWeight = "600";
      }

      tr.innerHTML = `
        <td>${escapeHtml(r.system)}</td>
        <td>${escapeHtml(r.threshold_method)}</td>
        <td><code style="font-family: 'SFMono-Regular', Menlo, monospace; font-size: 13px;">${escapeHtml(r.decision_threshold)}</code></td>
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

  function updateMaskSandbox() {
    if (!maskInput || !maskHighlightPreview) return;
    const text = maskInput.value || "";
    const spans = findProtectedSpansJS(text);

    const counts = { currency: 0, date: 0, percent: 0, negation: 0, number: 0 };
    spans.forEach(s => {
      if (counts[s.kind] !== undefined) counts[s.kind]++;
    });

    if (cntCurrency) cntCurrency.textContent = counts.currency;
    if (cntDate) cntDate.textContent = counts.date;
    if (cntPercent) cntPercent.textContent = counts.percent;
    if (cntNegation) cntNegation.textContent = counts.negation;
    if (cntNumber) cntNumber.textContent = counts.number;

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
    if (pctProtectedWords) pctProtectedWords.textContent = `${pct}% (${protectedWordCount}/${totalWords} words)`;

    maskHighlightPreview.innerHTML = renderHighlightedText(text, spans);
  }

  // =========================================================================
  // Event Listeners Initialization
  // =========================================================================
  function initEvents() {
    initTheme();
    initTabs();

    // Live Demo Presets
    if (livePresetEasy) {
      livePresetEasy.addEventListener("click", () => selectQuery(presetIndices.easy));
    }
    if (livePresetHard) {
      livePresetHard.addEventListener("click", () => selectQuery(presetIndices.hard));
    }
    if (livePresetMask) {
      livePresetMask.addEventListener("click", () => selectQuery(presetIndices.mask));
    }
    if (livePresetRandom) {
      livePresetRandom.addEventListener("click", () => {
        if (queriesData.length === 0) return;
        const randIdx = Math.floor(Math.random() * queriesData.length);
        selectQuery(randIdx);
      });
    }

    // Live Demo Stepper Buttons
    if (btnLiveNext) btnLiveNext.addEventListener("click", advanceLiveStep);
    if (btnLivePrev) btnLivePrev.addEventListener("click", previousLiveStep);
    if (btnLiveReset) btnLiveReset.addEventListener("click", resetLiveSteps);
    if (btnLiveRunAll) btnLiveRunAll.addEventListener("click", runAllLiveSteps);

    // Step pill direct clicking
    stepPills.forEach((pill, idx) => {
      pill.addEventListener("click", () => setLiveStep(idx + 1));
    });

    // Gold Reveal in Live Demo
    if (btnLiveRevealGold) {
      btnLiveRevealGold.addEventListener("click", () => {
        liveGoldRevealed = !liveGoldRevealed;
        updateLiveUI();
      });
    }

    // Live Query Select
    if (liveQuerySelect) {
      liveQuerySelect.addEventListener("change", (e) => selectQuery(e.target.value));
    }

    // Mode Switcher (Replay vs Custom)
    if (btnModeReplay && btnModeCustom) {
      btnModeReplay.addEventListener("click", () => {
        isCustomMode = false;
        btnModeReplay.classList.add("active");
        btnModeCustom.classList.remove("active");
        if (liveReplayWrap) liveReplayWrap.style.display = "block";
        if (liveCustomWrap) liveCustomWrap.style.display = "none";
      });

      btnModeCustom.addEventListener("click", () => {
        isCustomMode = true;
        btnModeCustom.classList.add("active");
        btnModeReplay.classList.remove("active");
        if (liveReplayWrap) liveReplayWrap.style.display = "none";
        if (liveCustomWrap) liveCustomWrap.style.display = "block";
      });
    }

    // Custom Question Match & Run
    if (btnLiveCustomRun) {
      btnLiveCustomRun.addEventListener("click", () => {
        const text = (liveCustomInput ? liveCustomInput.value.trim() : "");
        if (!text) return;
        const matched = findClosestCachedQuery(text);
        if (liveCustomNote) {
          liveCustomNote.style.display = "block";
          liveCustomNote.textContent = `Matched to closest benchmark query #${matched.idx + 1}: "${matched.question.slice(0, 70)}..." (Offline Replay)`;
        }
        selectQuery(matched.idx);
      });
    }

    // Presenter Mode Toggle
    if (btnPresenterMode) {
      btnPresenterMode.addEventListener("click", () => {
        presenterMode = !presenterMode;
        document.body.classList.toggle("presenter-mode", presenterMode);
        if (presenterText) {
          presenterText.textContent = presenterMode ? "Exit Presenter Mode" : "Presenter Mode";
        }
        if (btnPresenterMode) {
          btnPresenterMode.classList.toggle("primary", presenterMode);
        }
      });
    }

    // Keyboard Stepping (Right Arrow = Next, Left Arrow = Prev, R = Reset)
    window.addEventListener("keydown", (e) => {
      // Don't intercept if user is typing in an input or select
      const activeTag = document.activeElement ? document.activeElement.tagName.toLowerCase() : "";
      if (activeTag === "input" || activeTag === "textarea" || activeTag === "select") {
        return;
      }

      // Check if Live Demo tab is active
      const liveTab = document.getElementById("tab-live");
      if (!liveTab || !liveTab.classList.contains("active")) {
        return;
      }

      if (e.key === "ArrowRight") {
        e.preventDefault();
        advanceLiveStep();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        previousLiveStep();
      } else if (e.key === "r" || e.key === "R") {
        e.preventDefault();
        resetLiveSteps();
      }
    });

    // Pipeline Replay Events
    if (querySelect) querySelect.addEventListener("change", (e) => selectQuery(e.target.value));
    if (querySearch) querySearch.addEventListener("input", (e) => filterQueries(e.target.value));

    if (btnRandom) {
      btnRandom.addEventListener("click", () => {
        if (queriesData.length === 0) return;
        const randIdx = Math.floor(Math.random() * queriesData.length);
        selectQuery(randIdx);
      });
    }

    if (btnRevealGold) {
      btnRevealGold.addEventListener("click", () => {
        goldRevealed = !goldRevealed;
        updatePipelineUI();
      });
    }

    if (routerVariantSelect) routerVariantSelect.addEventListener("change", updateRouterDecision);
    if (thresholdSlider) {
      thresholdSlider.addEventListener("input", (e) => {
        if (thresholdValue) thresholdValue.textContent = parseFloat(e.target.value).toFixed(2);
        updateRouterDecision();
      });
    }

    if (compressToggle) {
      compressToggle.addEventListener("change", () => {
        const isLarge = decisionText ? decisionText.textContent.includes("LARGE") : false;
        updateStep5Answer(isLarge);
      });
    }

    // Mask Sandbox Events
    if (maskInput) maskInput.addEventListener("input", updateMaskSandbox);

    if (exBtnFinance) {
      exBtnFinance.addEventListener("click", () => {
        if (maskInput) maskInput.value = MASK_EXAMPLES.finance;
        updateMaskSandbox();
      });
    }

    if (exBtnBio) {
      exBtnBio.addEventListener("click", () => {
        if (maskInput) maskInput.value = MASK_EXAMPLES.bio;
        updateMaskSandbox();
      });
    }

    if (exBtnPlain) {
      exBtnPlain.addEventListener("click", () => {
        if (maskInput) maskInput.value = MASK_EXAMPLES.plain;
        updateMaskSandbox();
      });
    }
  }

  // Startup initialization
  window.addEventListener("DOMContentLoaded", () => {
    initEvents();
    loadData();

    if (maskInput) {
      maskInput.value = MASK_EXAMPLES.finance;
      updateMaskSandbox();
    }
  });

})();
