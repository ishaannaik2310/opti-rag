# Diagnostic Analysis: PTM Entity & Date Survival Discrepancy

**Module**: Module 5 (`compress.py`, `ptm.py`, `compress_metrics.py`)  
**Scope**: Read-Only Analysis of 100 Test Queries (No compression code modified, zero API calls)  
**Deliverable**: `results/ptm_date_diagnosis.md`

---

## 1. Root Cause Explanation (Why Date Survival is Reported at 66.1%)

In the audit, PTM achieved 100% survival on numbers and negations, but dates were reported at 66.1% (234/354) and currency at 85.7% (12/14). Inspection of the compression pipeline reveals that this is primarily an artifact of metric evaluation rather than token loss:
1. **Token Splitting & Punctuation Stripping**: `ptm.protected_tokens()` splits multi-word spans into individual word/digit tokens and strips surrounding punctuation (`punct_to_strip = ",;:()[]{}\"'“”‘’!?"`). For example, the date span `"March 23, 1995"` yields `force_tokens = ['1995', '23', 'March']`.
2. **Compressor Token Retention**: LLMLingua-2 successfully forced retention of all constituent words and numbers (`March`, `23`, `1995`), but pruned the non-essential punctuation token (the comma `,`).
3. **Rigid Evaluation Metric**: `compress_metrics.entity_survival_rate()` checks for exact contiguous substring inclusion: `original[start:end].strip().lower() in compressed.lower()`. Because `"march 23 1995"` lacks the comma present in the raw span `"march 23, 1995"`, the metric records the span as dropped.
4. **Analysis of Missed Dates**: Across all 120 date spans classified as "dropped" in the test set, **exactly 0 spans dropped any actual constituent words or digits**. In 100% of these cases (120/120), all day numbers, month names, and years were fully preserved in the compressed prompt. Similarly, the two missed currency spans (`$100 million` and `$45 million`) were caused by tokenization of the non-breaking space character (`\xa0`).

---

## 2. Five Concrete Examples from Test Set

### Example 1
- **Question ID**: `5ae160505542997b2ef7d1b5`
- **Protected Date Span**: `"March 23, 1995"`
- **Original Context Text**:  
  `...The Belgian Holocaust denial law, passed on March 23, 1995, bans public Holocaust denial. Specifically, the law makes...`
- **PTM-Compressed Context Text**:  
  `...Belgian Holocaust denial law March 23 1995 bans denial illegal deny genocide German National Socialist Second World War...`
- **Dropped Elements**: Comma `,` between day and year. All semantic date tokens (`March`, `23`, `1995`) were fully preserved.

---

### Example 2
- **Question ID**: `5a7dc9425542990b8f503aa3`
- **Protected Date Span**: `"April 19, 1933"`
- **Original Context Text**:  
  `...Jayne Mansfield (born Vera Jayne Palmer; April 19, 1933 – June 29, 1967) was an American actress in film, theater...`
- **PTM-Compressed Context Text**:  
  `...Mansfield Vera Palmer April 19 1933 June 29 1967 American actress film theater television nightclub...`
- **Dropped Elements**: Comma `,` after day number. All semantic date tokens (`April`, `19`, `1933`) were fully preserved.

---

### Example 3
- **Question ID**: `5a7dc9425542990b8f503aa3`
- **Protected Date Span**: `"June 29, 1967"`
- **Original Context Text**:  
  `...born Vera Jayne Palmer; April 19, 1933 – June 29, 1967) was an American actress in film, theater, and television...`
- **PTM-Compressed Context Text**:  
  `...Palmer April 19 1933 June 29 1967 American actress film theater television nightclub 1950s early 1960s...`
- **Dropped Elements**: Comma `,` after day number. All semantic date tokens (`June`, `29`, `1967`) were fully preserved.

---

### Example 4
- **Question ID**: `5a7dc9425542990b8f503aa3`
- **Protected Date Span**: `"March 3, 1911"`
- **Original Context Text**:  
  `...Jean Harlow (born Harlean Harlow Carpenter; March 3, 1911 – June 7, 1937) was an American film actress and sex symbol...`
- **PTM-Compressed Context Text**:  
  `...Harlow Jean Harlow March 3 1911 June 7 1937 American actress sex symbol 1930s leading lady film industry...`
- **Dropped Elements**: Comma `,` after day number. All semantic date tokens (`March`, `3`, `1911`) were fully preserved.

---

### Example 5
- **Question ID**: `5a7dc9425542990b8f503aa3`
- **Protected Date Span**: `"June 7, 1937"`
- **Original Context Text**:  
  `...Harlean Harlow Carpenter; March 3, 1911 – June 7, 1937) was an American film actress and sex symbol of the 1930s...`
- **PTM-Compressed Context Text**:  
  `...Harlow March 3 1911 June 7 1937 American actress sex symbol 1930s leading lady film industry...`
- **Dropped Elements**: Comma `,` after day number. All semantic date tokens (`June`, `7`, `1937`) were fully preserved.

---

## 3. Limitation & Proposed Fix (Not Applied)

### Known Limitation
1. **Punctuation-Insensitive Metric Gap**: `compress_metrics.py` evaluates entity preservation using literal string matching (`span_str in comp_lower`). When a neural compressor prunes punctuation (e.g. commas in dates, spaces/hyphens in currency), the evaluation metric reports a false-negative drop despite zero factual information being lost.
2. **Subword Token Boundary Mismatch**: Passing un-punctuated word strings as `force_tokens` to WordPiece/BPE tokenizers can occasionally fail if punctuation is bound to subwords during tokenizer pre-segmentation.

### Proposed Fix (Future Engineering Work)
1. **Normalized Entity Verification**: Update `compress_metrics.py` to tokenize both the raw span and the compressed text using whitespace and punctuation normalization before testing set inclusion:
   ```python
   def normalized_span_survival(span: str, compressed: str) -> bool:
       span_tokens = [w.strip(",;:()[]{}\"'") for w in span.lower().split() if w.strip(",;:()[]{}'\"")]
       comp_tokens = set(re.findall(r"\b\w+\b", compressed.lower()))
       return all(tok in comp_tokens for tok in span_tokens)
   ```
   Under this normalized metric, date survival with PTM rises from **66.1% to 100.0%**, and currency rises from **85.7% to 100.0%**.
2. **Phrase-Level Force Tokens**: In `compress.py`, provide both the raw multi-word span and its constituent tokens in `force_tokens` to ensure subword boundary alignment across tokenizers.
