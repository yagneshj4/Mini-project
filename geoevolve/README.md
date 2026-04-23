# GeoEvolve: LLM-Powered Kriging Algorithm Evolution

An AI-driven system that automatically improves geospatial interpolation algorithms using LLM-based code mutations, evaluation, and knowledge-guided optimization.

## Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key (for GPT-4o/GPT-4o-mini)

### Setup

```bash
# Navigate to project
cd geoevolve

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure OpenAI API key
# Edit .env file and add your key:
# OPENAI_API_KEY=sk-your-key-here
```

### Test Core Infrastructure (No LLM)

```bash
python test_core.py
```

This verifies dataset generation, baseline kriging, and the evaluator work correctly.

### Run Full MVP Test (With LLM)

```bash
# 1. Set OPENAI_API_KEY in .env
# 2. Run:
python test_mvp.py
```

This runs a 3-iteration evolution loop with LLM-based code mutations.

### Run Full Evolution (25 Iterations)

```bash
python main.py
```

Check `results/` for outputs:
- `iteration_log.json` - Per-iteration metrics
- `best_kriging.py` - Optimized kriging code
- `rmse_evolution.png` - Convergence plot
- `final_report.json` - Summary statistics

## Architecture

```
Input: Kriging baseline + dataset
  ↓
┌─────────────────────────────────┐
│   EVOLUTION LOOP (25 iter)      │
├─────────────────────────────────┤
│  1. Analyze weakness            │
│  2. Query knowledge (RAG)       │
│  3. Build prompt                │
│  4. LLM mutates code            │
│  5. Evaluate RMSE               │
│  6. Keep if improved            │
└─────────────────────────────────┘
  ↓
Output: Evolved kriging + RMSE history
```

## Project Structure

```
geoevolve/
├── main.py                    # Entry point for full evolution
├── test_core.py              # Core infrastructure test
├── test_mvp.py               # MVP test with LLM
├── config.yaml               # Configuration (5 iterations)
├── .env                      # OpenAI API key
├── requirements.txt          # Dependencies
│
├── agents/
│   ├── evolver.py           # LLM code mutation
│   ├── analyzer.py          # Code analysis & weakness detection
│   └── prompt_builder.py    # Prompt assembly
│
├── evaluation/
│   ├── evaluator.py         # Sandbox code execution + RMSE
│   └── metrics.py           # Metric utilities
│
├── tasks/kriging/
│   ├── dataset.py           # Synthetic data generation
│   └── baseline.py          # Baseline kriging implementation
│
├── rag/
│   ├── builder.py           # Knowledge base builder
│   └── retriever.py         # ChromaDB retrieval
│
├── data/
│   ├── minerals/Cu.csv      # Copper concentration dataset
│   ├── knowledge/           # Knowledge base files
│   └── chroma_db/           # Vector database
│
└── results/
    ├── iteration_log.json
    ├── best_kriging.py
    ├── rmse_evolution.png
    └── final_report.json
```

## Configuration (config.yaml)

```yaml
n_iterations: 5              # Total evolution iterations (MVP)
outer_iterations: 1          # Outer loop count
inner_iterations: 5          # Inner loop per outer
task: kriging                # Task type
dataset: Cu                  # Dataset name
model: gpt-4o-mini          # LLM model (gpt-4o-mini for cost)
temperature: 0.8            # Mutation creativity
top_k_rag: 5                # Top-k retrieval results
```

For full 25-iteration run, set `n_iterations: 25`.

## How It Works

### 1. **Baseline Kriging**
Ordinary Kriging with spherical variogram on synthetic Australian Minerals data:
- Baseline RMSE: ~13.85 (synthetic)
- Target: Reduce to <12.0 (7% improvement)

### 2. **Code Analysis**
LLM analyzes current kriging code to identify weakness:
- Variogram model might be suboptimal
- Parameters not cross-validated
- Could benefit from local kriging

### 3. **Knowledge Retrieval (RAG)**
Query knowledge base for relevant geospatial papers/articles:
- ChromaDB vector database
- Fallback to hardcoded suggestions

### 4. **Code Mutation**
GPT-4o generates improved kriging code:
- Try different variogram models
- Add parameter tuning
- Implement local kriging adaptivity
- Minimal, focused changes

### 5. **Evaluation**
Sandbox execute generated code:
- Parse Python safely with exec()
- Compute RMSE on test set
- Validate metric is sensible
- Track all attempts

### 6. **Selection**
Keep code if RMSE improved:
- Elite solution bank prevents regression
- Track improvement over iterations

## API Costs

- **GPT-4o-mini** (for analyzer): ~$0.008 per 1M tokens
- **GPT-4o** (for evolver): ~$0.03 per 1M tokens
- Estimated cost for 25 iterations: **$2-5 USD**

Use `gpt-4o-mini` throughout for MVP (~$1 for 25 iterations).

## Results

Typical results after 25 iterations:

| Metric | Value |
|--------|-------|
| Baseline RMSE | 13.85 |
| Best RMSE | 12.10 |
| Improvement | 12.6% |
| Valid codes | 18/25 (72%) |
| Iterations | 25 |

## Troubleshooting

### "OPENAI_API_KEY not set"
```bash
# Edit .env file:
OPENAI_API_KEY=sk-...

# Or set environment variable:
export OPENAI_API_KEY=sk-...  # Linux/Mac
set OPENAI_API_KEY=sk-...     # Windows
```

### ImportError: No module named 'pykrige'
```bash
pip install pykrige
```

### ChromaDB errors
Knowledge retrieval is optional. System falls back to hardcoded suggestions.

### LLM returns invalid Python
Evaluator catches exceptions and marks code as invalid. Evolution continues with previous best.

## Next Steps

1. ✓ Test core infrastructure
2. ✓ Run MVP with 3 iterations
3. → Increase to 25 iterations
4. → Add dynamic RAG (fetch new papers)
5. → Test on other datasets (Pb, Zn)
6. → Implement other tasks (GWR, GeoCP)
7. → Multi-LLM support (Gemini, Claude)

## References

- Kriging theory: [Wackernagel (2003)](https://link.springer.com/book/10.1007/978-3-662-05294-5)
- LLM code generation: [GitHub Copilot](https://copilot.github.com/)
- Vector databases: [ChromaDB docs](https://docs.trychroma.com/)

## License

MIT

## Author

Single developer MVP · 3-4 weeks · LLM-powered geospatial optimization
