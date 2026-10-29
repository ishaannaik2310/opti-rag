/**
 * OptiRAG-lite Protected Token Mask (PTM) - JavaScript Engine.
 * Provides 100% exact parity with Python ptm.py across all entity kinds:
 * (currency, date, percent, negation, number).
 *
 * Runs 100% client-side with zero external dependencies.
 */

const PTM_PRIORITY = {
  currency: 5,
  date: 4,
  percent: 3,
  negation: 2,
  number: 1,
};

// Python-compatible Unicode word boundaries
const WB_START = "(?<![\\p{L}\\p{M}\\p{N}\\p{Pc}])";
const WB_END = "(?![\\p{L}\\p{M}\\p{N}\\p{Pc}])";

const PTM_NEGATION_WORDS = `${WB_START}(?:not|no|never|none|nor|neither|nobody|nothing|nowhere|except|without|cannot)${WB_END}`;
const PTM_NEGATION_CONTRACTIONS = `${WB_START}[\\p{L}\\p{M}\\p{N}\\p{Pc}]+(?:n['’]t)${WB_END}`;

const PTM_MONTHS = "(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)";
const PTM_CURRENCY_SYMBOLS = "[$€£¥₹]|(?:Rs\\.?)";
const PTM_CURRENCY_CODES = "(?:USD|EUR|GBP|JPY|INR|CAD|AUD|dollars?|cents?|euros?|pounds?|rupees?|yen)";

const PTM_RE_CURRENCY = new RegExp(
  `(?:(${PTM_CURRENCY_SYMBOLS})\\s*\\d+(?:,\\d{3})*(?:\\.\\d+)?(?:\\s*(?:billion|million|trillion|k|m|b|thousand|crore|lakh))?${WB_END})|(?:(${PTM_CURRENCY_SYMBOLS})\\s*\\d+(?:\\.\\d+)?[MBKmbk]${WB_END})|(?:${WB_START}\\d+(?:,\\d{3})*(?:\\.\\d+)?\\s*${PTM_CURRENCY_CODES}${WB_END})`,
  "giu"
);

const PTM_RE_PERCENT = new RegExp(
  `${WB_START}\\d+(?:,\\d{3})*(?:\\.\\d+)?\\s*(?:%|percent|percentage)${WB_END}|${WB_START}\\d+(?:\\.\\d+)?%`,
  "giu"
);

const PTM_RE_DATE = new RegExp(
  `${WB_START}Q[1-4](?:\\s*(?:19|20)\\d{2})?${WB_END}|${WB_START}\\d{4}-\\d{2}-\\d{2}${WB_END}|${WB_START}\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4}${WB_END}|${WB_START}${PTM_MONTHS}\\s+\\d{1,2}(?:st|nd|rd|th)?,?\\s+(?:19|20)\\d{2}${WB_END}|${WB_START}\\d{1,2}(?:st|nd|rd|th)?\\s+${PTM_MONTHS},?\\s+(?:19|20)\\d{2}${WB_END}|${WB_START}${PTM_MONTHS}\\s+\\d{1,2}(?:st|nd|rd|th)?${WB_END}|${WB_START}${PTM_MONTHS}\\s+(?:19|20)\\d{2}${WB_END}|${WB_START}(?:19|20)\\d{2}s${WB_END}`,
  "giu"
);

const PTM_RE_NEGATION = new RegExp(
  `${PTM_NEGATION_WORDS}|${PTM_NEGATION_CONTRACTIONS}`,
  "giu"
);

const PTM_RE_NUMBER = new RegExp(
  `${WB_START}\\d+(?:st|nd|rd|th)${WB_END}|${WB_START}\\d{1,3}(?:,\\d{3})+(?:\\.\\d+)?${WB_END}|${WB_START}\\d+\\.\\d+${WB_END}|${WB_START}\\d+${WB_END}`,
  "giu"
);

/**
 * Finds all character-level protected spans (start, end, kind).
 * Merges overlapping and contiguous spans using priority resolution.
 * @param {string} text - Input text
 * @returns {Array<{start: number, end: number, kind: string}>}
 */
function findProtectedSpansJS(text) {
  if (!text) return [];

  const raw_spans = [];
  const patterns = [
    { regex: PTM_RE_CURRENCY, kind: "currency" },
    { regex: PTM_RE_PERCENT, kind: "percent" },
    { regex: PTM_RE_DATE, kind: "date" },
    { regex: PTM_RE_NEGATION, kind: "negation" },
    { regex: PTM_RE_NUMBER, kind: "number" },
  ];

  for (const { regex, kind } of patterns) {
    regex.lastIndex = 0;
    let match;
    while ((match = regex.exec(text)) !== null) {
      raw_spans.push({
        start: match.index,
        end: match.index + match[0].length,
        kind: kind,
      });
    }
  }

  if (raw_spans.length === 0) return [];

  // Sort spans by start asc, length desc
  raw_spans.sort((a, b) => {
    if (a.start !== b.start) return a.start - b.start;
    return (b.end - b.start) - (a.end - a.start);
  });

  const merged = [];
  for (const cur of raw_spans) {
    if (merged.length === 0) {
      merged.push({ start: cur.start, end: cur.end, kind: cur.kind });
      continue;
    }

    const prev = merged[merged.length - 1];
    if (cur.start <= prev.end) {
      const new_start = Math.min(prev.start, cur.start);
      const new_end = Math.max(prev.end, cur.end);
      const prev_pri = PTM_PRIORITY[prev.kind] || 0;
      const cur_pri = PTM_PRIORITY[cur.kind] || 0;
      const resolved_kind = prev_pri >= cur_pri ? prev.kind : cur.kind;
      prev.start = new_start;
      prev.end = new_end;
      prev.kind = resolved_kind;
    } else {
      merged.push({ start: cur.start, end: cur.end, kind: cur.kind });
    }
  }

  return merged;
}

// Export for Node environments if present
if (typeof module !== "undefined" && module.exports) {
  module.exports = { findProtectedSpansJS };
}
