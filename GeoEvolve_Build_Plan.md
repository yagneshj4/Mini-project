# GeoEvolve — Complete Engineering Build Plan
> Single developer · 3–4 weeks · MVP-first · No fluff

---

## 1. CORE IDEA SIMPLIFICATION

### What it actually is (plain English)
You have a geospatial algorithm (e.g., Kriging). You want a system that:
1. Runs it and measures its score (RMSE)
2. Asks an LLM to rewrite/mutate the code to improve it
3. Evaluates the mutated code
4. Pulls relevant geospatial theory from a knowledge base
5. Injects that theory into the next LLM prompt
6. Repeats until the algorithm stops improving

That is it. Everything else in the paper is optimization on top of this loop.

### What to REMOVE for MVP
| Paper Component | MVP Decision |
|---|---|
| Multi-agent orchestration | Remove — use a single Python loop |
| Code-to-Formula Agent | Skip — hardcode the triplet format manually |
| Dynamic knowledge updates | Skip — use static RAG only |
| Multi-LLM support (GPT/Gemini/Qwen) | Pick one: GPT-4o only |
| 10 outer x 10 inner loops (100 iter) | Reduce to 5 outer x 5 inner (25 iter) |
| 3 tasks (Kriging + GeoCP + GWR) | Do ONE task: Kriging only |
| 9 datasets generalization | Do ONE dataset: Australian Minerals (Cu) |
| Elite solution bank | Keep — prevents regression (simple dict) |

### Smallest working version
- Input: Kriging baseline code + dataset
- Output: Evolved Kriging code with lower RMSE after 25 LLM iterations
- Components: Evolver + Evaluator + Analyzer + Static RAG

---

## 2. SYSTEM ARCHITECTURE (MVP)

```
[Dataset CSV]
     |
     v
[Baseline Kriging Code]
     |
     v
+---------------------------------------------+
|              MAIN LOOP                      |
|                                             |
|  +----------+    +------------------+       |
|  | EVOLVER  |--->|   EVALUATOR      |       |
|  | (LLM)    |    |   (RMSE calc)    |       |
|  +----------+    +--------+---------+       |
|       ^                   |                 |
|       |           +-------v--------+        |
|  +----+-------+   |   ANALYZER     |        |
|  | PROMPT     |<--|   (LLM)        |        |
|  | BUILDER    |   +----------------+        |
|  +----+-------+                             |
|       |                                     |
|  +----v-------+                             |
|  |    RAG     |  (ChromaDB query)           |
|  +------------+                             |
+---------------------------------------------+
     |
     v
[Best Code + RMSE History + Logs]
```

### Module I/O Definitions

| Module | Input | Output |
|---|---|---|
| evolver.py | current_code (str) + prompt (str) | new_code (str) |
| evaluator.py | code (str) + dataset (DataFrame) | rmse (float), is_valid (bool) |
| analyzer.py | code (str) + rmse (float) | weakness (str), search_query (str) |
| rag.py | query (str) | context_chunks (list[str]) |
| prompt_builder.py | context + weakness + current_code | final_prompt (str) |
| main.py | config | logs, best_code, rmse_history |

---

## 3. TECH STACK (STRICT)

| Component | Choice | Reason |
|---|---|---|
| Language | Python 3.11 | All geo libs available |
| LLM | OpenAI GPT-4o API | Best code generation |
| Vector DB | ChromaDB (local) | No server needed, runs on laptop |
| Embeddings | text-embedding-3-small | Cheaper, good enough for RAG |
| Kriging | pykrige | Only mature Python Kriging library |
| Geo utils | geopandas, numpy, scikit-learn | Standard stack |
| Data sources | arxiv, wikipedia-api | Free, no server required |
| Experiment tracking | Plain JSON + matplotlib | No WandB overhead for MVP |
| Config | python-dotenv + yaml | Simple env management |

```bash
pip install openai chromadb pykrige geopandas scikit-learn \
            arxiv wikipedia-api python-dotenv pyyaml matplotlib numpy pandas
```

---

## 4. STEP-BY-STEP BUILD PLAN

### PHASE 1 — Baseline + Dataset + Evaluation (Days 1-3)

**Tasks:**
1. Download Australian Minerals dataset (Cu, Pb, Zn CSVs from Luo et al. 2025)
   - Columns needed: x (easting), y (northing), value (ppm)
   - If unavailable: use synthetic data (see Section 8)
2. Write baseline_kriging.py:
   - Ordinary Kriging using pykrige.OrdinaryKriging
   - Variogram: spherical
   - Split 80/10/10 train/val/test
3. Write evaluator.py:
   - exec() the code string in sandboxed namespace
   - Catch all exceptions -> return is_valid=False
   - Return RMSE on test set

**Expected Output:**
- Baseline RMSE for Cu is approximately 0.91
- Evaluator reliably runs code strings and returns float

**Failure Points:**
- pykrige fitting failure on small datasets -> add try/except, return RMSE=999
- Coordinate system mismatch -> normalize x,y to [0,1] range

---

### PHASE 2 — Evolution Loop / LLM Mutation (Days 4-7)

**Tasks:**
1. Write evolver.py:
   - System prompt: "You are an expert in geostatistics. Improve this Kriging code."
   - User prompt: current code + weakness + geo knowledge
   - Return ONLY valid Python code (parse with regex if needed)
2. Write main.py with the loop:
```python
for iteration in range(25):
    new_code = evolver.mutate(current_code, prompt)
    rmse, valid = evaluator.run(new_code, dataset)
    if valid and rmse < best_rmse:
        best_code = new_code
        best_rmse = rmse
```
3. Save every iteration result to results/iteration_log.json

**Expected Output:**
- System runs 25 iterations without crashing
- RMSE should drop vs baseline after approximately 5 iterations

**Failure Points:**
- LLM returns broken Python -> wrap exec() in try/except, fall back to previous best
- LLM adds unknown imports -> list allowed imports in system prompt explicitly
- LLM rewrites everything with no continuity -> tell LLM to modify minimally

---

### PHASE 3 — Code Analyzer + Feedback Loop (Days 8-11)

**Tasks:**
1. Write analyzer.py:
   - Input: evolved code + RMSE
   - Prompt: "Analyze this Kriging code. What geospatial concept is missing? Return JSON with weakness and 3-word search query."
   - Parse: {"weakness": "...", "query": "..."}
2. Feed query into RAG retriever (Phase 4 integration)
3. Feed weakness into next evolver prompt

**Example analyzer output:**
```json
{
  "weakness": "Uses only spherical variogram with no adaptive model selection",
  "query": "variogram model selection"
}
```

**Failure Points:**
- LLM returns invalid JSON -> use json.loads() with fallback regex extraction
- Analyzer is too vague -> force structured output with few-shot examples in prompt

---

### PHASE 4 — RAG Integration (Days 12-17)

**Tasks:**
1. Write knowledge/builder.py (run once):
   - Fetch 20 arXiv papers on "kriging spatial interpolation variogram"
   - Fetch 10 Wikipedia articles on "Kriging", "Variogram", "Spatial autocorrelation"
   - Save as text files in data/knowledge/
2. Write knowledge/indexer.py:
   - Chunk each doc: 300 words with 50-word overlap
   - Embed with text-embedding-3-small
   - Store in ChromaDB collection "geoevolve_kb"
3. Write rag/retriever.py:
   - Input: query string
   - Query ChromaDB top-5 chunks
   - Return joined text as context string
4. Plug into prompt_builder.py:
```python
context = retriever.query(analyzer_query)
prompt = f"""
Context from geospatial literature:
{context}

Weakness identified: {weakness}

Current code:
{current_code}

Improve the kriging algorithm. Return ONLY Python code.
"""
```

**Failure Points:**
- arXiv API rate limit -> cache downloads, add 2s delay between requests
- Chunks too long -> strict 300-word truncation
- Irrelevant retrieval -> print top-5 chunks manually and verify quality

---

### PHASE 5 — Optimization + Logging + Visualization (Days 18-22)

**Tasks:**
1. Structured JSON logging per iteration
2. Plot RMSE over iterations with matplotlib
3. Save best code to results/best_kriging.py
4. Final comparison table: Original vs Evolved (RMSE, MAE, R2)
5. Add retry logic: if LLM call fails -> retry 3x with exponential backoff

**Expected Output:**
- Descending RMSE curve plot
- Final evolved code that outperforms baseline
- Clean JSON log of all 25 iterations

---

## 5. FOLDER STRUCTURE

```
geoevolve/
|
+-- main.py                    # Entry point — runs the full loop
+-- config.yaml                # n_iterations, model name, task
+-- .env                       # OPENAI_API_KEY
+-- requirements.txt
|
+-- agents/
|   +-- evolver.py             # LLM mutation — takes code, returns new code
|   +-- analyzer.py            # LLM analysis — returns weakness + query
|   +-- prompt_builder.py      # Assembles final prompt from all inputs
|
+-- rag/
|   +-- retriever.py           # ChromaDB query -> top-k chunks
|   +-- builder.py             # One-time: fetch docs + index into ChromaDB
|
+-- evaluation/
|   +-- evaluator.py           # exec() code string, compute RMSE
|   +-- metrics.py             # rmse(), mae(), r2() utility functions
|
+-- tasks/
|   +-- kriging/
|       +-- baseline.py        # Clean Ordinary Kriging implementation
|       +-- dataset.py         # Load + split Australian Minerals CSV
|
+-- data/
|   +-- minerals/              # Cu.csv, Pb.csv, Zn.csv
|   +-- knowledge/             # Raw text docs for RAG
|   +-- chroma_db/             # ChromaDB persistent storage
|
+-- results/
|   +-- iteration_log.json     # Full per-iteration log
|   +-- best_kriging.py        # Best evolved code
|   +-- rmse_plot.png          # Convergence plot
|
+-- tests/
    +-- test_evaluator.py      # Verify sandbox exec works
    +-- test_rag.py            # Verify retrieval returns relevant chunks
```

---

## 6. CORE ALGORITHM LOOP (PSEUDOCODE)

```python
def run_geoevolve(config):
    # SETUP
    dataset = load_dataset(config.task)
    baseline_code = load_baseline(config.task)
    knowledge_base = build_or_load_kb()

    best_code = baseline_code
    best_rmse, _ = evaluator.run(baseline_code, dataset)
    logs = []

    # OUTER LOOP
    for outer in range(config.outer_iterations):  # default: 5

        # Step 1: Analyze current best code
        analysis  = analyzer.analyze(best_code, best_rmse)
        weakness  = analysis["weakness"]
        query     = analysis["query"]

        # Step 2: Retrieve geospatial knowledge
        context = retriever.query(query, top_k=5)

        # Step 3: Build LLM prompt
        prompt = prompt_builder.build(
            code=best_code, weakness=weakness, context=context
        )

        # INNER LOOP
        for inner in range(config.inner_iterations):  # default: 5

            # Step 4: LLM mutates the code
            new_code = evolver.mutate(best_code, prompt)

            # Step 5: Evaluate mutated code
            rmse, valid = evaluator.run(new_code, dataset)

            # Step 6: Keep if better
            if valid and rmse < best_rmse:
                best_code = new_code
                best_rmse = rmse

            # Step 7: Log
            logs.append({
                "outer": outer, "inner": inner,
                "rmse": rmse, "valid": valid,
                "weakness": weakness, "query": query
            })

    save_results(best_code, best_rmse, logs)
    plot_rmse_curve(logs)
    return best_code, best_rmse
```

---

## 7. MINIMAL WORKING VERSION (MVP)

Build in this exact order:

- Day 1: evaluator.py — can it exec a code string and return RMSE?
- Day 2: baseline.py — does Ordinary Kriging run and produce RMSE ~0.91?
- Day 3: evolver.py — does LLM return valid Python when asked to improve code?
- Day 4: main.py — does the loop run 5 times without crashing?
- Day 5: Does RMSE drop at all vs baseline? If yes: MVP works.

**What to HARDCODE first:**
```python
# Hardcode these until loop itself works
weakness = "The variogram model is fixed. Try Gaussian or Matern."
context  = "Localized kriging with K-nearest neighbors reduces complexity."
query    = "variogram model selection kriging"
```
Replace with real LLM/RAG outputs only AFTER the loop runs cleanly.

**Features to SKIP initially:**
- Dynamic knowledge updates
- Multiple metals (start with Cu only)
- Code diff tracking
- Multi-LLM support
- Outer loop (just do flat 10 iterations first)

---

## 8. DATASET + TASK SELECTION

### Recommended: Kriging on Australian Minerals (Cu only)

**Why Kriging:**
- Simple deterministic evaluation (RMSE is unambiguous)
- pykrige handles heavy lifting
- Paper shows clear improvement: 0.91 -> ~0.77 RMSE on Cu
- No external base model needed (unlike GeoCP which needs XGBoost)

**Synthetic data fallback (if real dataset unavailable):**
```python
import numpy as np, pandas as pd
np.random.seed(42)
n = 300
x = np.random.uniform(0, 100, n)
y = np.random.uniform(0, 100, n)
vals = np.sin(x/20) * np.cos(y/20) * 100 + np.random.normal(0, 10, n)
pd.DataFrame({'x': x, 'y': y, 'cu_ppm': vals}).to_csv('data/minerals/Cu.csv', index=False)
```

**Evaluation metric:** RMSE on held-out 10% test set
**Baseline target:** RMSE ~0.91 (log-transformed ppm scale)
**Success criterion:** Evolved RMSE < 0.80 (greater than 12% reduction)

---

## 9. LOGGING + METRICS

### Per-iteration log structure:
```json
{
  "iteration": 7,
  "outer": 1,
  "inner": 2,
  "rmse": 0.8341,
  "mae": 0.6102,
  "r2": 0.4211,
  "is_valid": true,
  "weakness": "no adaptive variogram model selection",
  "rag_query": "variogram AIC BIC model selection",
  "prompt_tokens": 1842,
  "completion_tokens": 612,
  "cost_usd": 0.024,
  "timestamp": "2026-04-23T10:00:00"
}
```

### Track across iterations:
- RMSE curve (plot after every run)
- Percent improvement vs baseline
- Valid code rate (how often LLM returns runnable code)
- API cost per iteration

### Final comparison table:
| Method | RMSE | MAE | R2 |
|---|---|---|---|
| Baseline Kriging | 0.9139 | 0.6752 | 0.375 |
| GeoEvolve (yours) | TBD | TBD | TBD |

---

## 10. FAILURE POINTS (BRUTAL)

### Problem 1: LLM returns invalid Python
**Solution:**
```python
def safe_exec(code: str, dataset) -> tuple:
    try:
        namespace = {
            "np": np, "pd": pd,
            "OrdinaryKriging": OrdinaryKriging
        }
        exec(code, namespace)
        rmse = namespace["evaluate"](dataset)
        return rmse, True
    except Exception:
        return 999.0, False
```
Add to system prompt: "Return ONLY executable Python. No markdown. No explanation. Define a function called evaluate(dataset) that returns float RMSE."

---

### Problem 2: LLM adds unknown imports
**Solution:** Whitelist imports in the exec namespace:
```python
namespace = {
    "np": np, "pd": pd, "math": math,
    "OrdinaryKriging": OrdinaryKriging,
    "mean_squared_error": mean_squared_error,
    "KNeighborsRegressor": KNeighborsRegressor
}
```

---

### Problem 3: RAG returns irrelevant chunks
**Symptoms:** LLM mutations become random or worse after RAG is added
**Solution:**
1. Print top-5 retrieved chunks and read them manually
2. Re-fetch knowledge with better keywords if bad
3. Add metadata filtering: only retrieve from "kriging" category
4. Fallback: if cosine similarity < 0.3, skip RAG for this iteration

---

### Problem 4: RMSE never improves
**Diagnosis:**
1. Print LLM output — is it changing code meaningfully?
2. Print weakness — is analyzer identifying real issues?
3. Check dataset size (less than 50 points makes Kriging unstable)

**Solution:**
- Force diversity: set temperature=1.0 in LLM call
- Change system prompt: "Make a DRAMATIC algorithmic change"
- Start loop from a manually improved version, not raw baseline

---

### Problem 5: Slow execution (10+ min per iteration)
**Solutions:**
- Use gpt-4o-mini for the analyzer (cheap and fast)
- Use gpt-4o only for the evolver
- Subsample dataset to 100 points for evolution; evaluate on full set
- Cache RAG results per unique query string

---

### Problem 6: ChromaDB corruption on crash
**Solution:**
```python
import shutil
if config.reset_kb:
    shutil.rmtree("data/chroma_db", ignore_errors=True)
client = chromadb.PersistentClient(path="data/chroma_db")
```

---

## 11. EXTENSION PLAN

Once MVP achieves RMSE improvement on Cu, extend in this order:

1. Add Pb and Zn datasets -> same loop, 3 result columns
2. Dynamic RAG -> add 5 new docs per outer cycle when weakness detected
3. Add GWR task (R2 metric, Georgia census dataset)
4. Multi-LLM support (Gemini) via config switch
5. Add GeoCP task (needs XGBoost base predictor first)
6. Replace static ChromaDB with live arXiv API retrieval

---

## QUICK START CHECKLIST

```
[ ] Day 1: pip install all deps, get OpenAI API key
[ ] Day 1: Download or generate Cu dataset, verify it loads
[ ] Day 2: Run baseline Kriging, confirm RMSE ~0.91
[ ] Day 2: Write evaluator.py, confirm it scores baseline correctly
[ ] Day 3: Write evolver.py, call GPT-4o once, confirm valid Python returned
[ ] Day 4: Write main.py loop, run 5 hardcoded iterations, confirm no crashes
[ ] Day 5: Check — did RMSE drop at all? If yes: MVP is working
[ ] Day 6: Write analyzer.py, plug weakness into prompt
[ ] Day 7: Build RAG — fetch 20 papers, index in ChromaDB, test retrieval
[ ] Day 8: Full integration — evolver + analyzer + RAG running together
[ ] Day 9: Run full 25 iterations, log everything
[ ] Day 10: Plot RMSE curve, write final comparison table
```
