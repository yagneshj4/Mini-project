#!/usr/bin/env python3
"""
Quick Start Guide - Print this to get started immediately
"""

QUICK_START = """
╔════════════════════════════════════════════════════════════════════════════════╗
║                       GeoEvolve Quick Start Guide                             ║
╚════════════════════════════════════════════════════════════════════════════════╝

✨ WHAT IS GEOEVOLVE?
═══════════════════════════════════════════════════════════════════════════════
An AI system that uses GPT-4o to automatically improve kriging algorithms by:
1. Analyzing the current algorithm for weaknesses
2. Retrieving relevant geospatial research (arXiv papers, Wikipedia)
3. Asking LLM to generate improved code
4. Testing and measuring improvement
5. Repeating until RMSE is optimized

Expected Results:
• Baseline RMSE: ~13.85 (synthetic copper concentration data)
• Target RMSE: <12.0 (7% improvement)
• Actual improvement: 5-15% typical
• Cost: ~$1-2 for 25 iterations


🚀 GET STARTED (3 STEPS)
═══════════════════════════════════════════════════════════════════════════════

STEP 1: Test Without LLM (2 minutes)
──────────────────────────────────────────────────────────────────────────────
   $ cd geoevolve
   $ python test_core.py

   ✓ Tests dataset generation, kriging, and evaluator
   ✓ Confirms all dependencies work
   ✓ Shows baseline RMSE: 13.85
   ✓ Cost: $0


STEP 2: Get OpenAI API Key (1 minute)
──────────────────────────────────────────────────────────────────────────────
   1. Go to: https://platform.openai.com/api-keys
   2. Click: "Create new secret key"
   3. Copy the key (starts with "sk-")
   
   Then edit .env file:
   ┌──────────────────────────────────────┐
   │ OPENAI_API_KEY=sk-your-key-here     │
   └──────────────────────────────────────┘


STEP 3: Run Evolution Tests (5-10 minutes)
──────────────────────────────────────────────────────────────────────────────
   Option A: Quick MVP Test (3 iterations)
   $ python test_mvp.py
   
   ✓ Runs 3 LLM-based mutations
   ✓ Shows each iteration's RMSE
   ✓ Cost: ~$0.15
   ✓ Time: 3-5 minutes


   Option B: Full Evolution (25 iterations) 
   $ python main.py
   
   ✓ Runs complete 25-iteration loop
   ✓ Saves best code to results/best_kriging.py
   ✓ Generates convergence plot
   ✓ Cost: ~$1.00
   ✓ Time: 20-30 minutes


📊 EXPECTED OUTPUT
═══════════════════════════════════════════════════════════════════════════════
After running main.py, check results/ folder:

   results/
   ├── iteration_log.json         ← All 25 iterations with metrics
   ├── best_kriging.py            ← Best evolved code
   ├── rmse_evolution.png         ← Convergence plot
   └── final_report.json          ← Summary statistics

   Sample final_report.json:
   ┌────────────────────────────────────────┐
   │ {                                      │
   │   "total_iterations": 25,              │
   │   "best_rmse": 12.10,                  │
   │   "improvement_percent": 12.6,         │
   │   "valid_codes": 18                    │
   │ }                                      │
   └────────────────────────────────────────┘


⚙️ CONFIGURATION
═══════════════════════════════════════════════════════════════════════════════
Edit config.yaml to customize:

   n_iterations: 25        ← Number of evolution iterations
   model: gpt-4o-mini      ← (gpt-4o-mini is cheaper, gpt-4o is better)
   temperature: 0.8        ← Higher = more creative mutations


🐛 TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════════════════

ERROR: "OPENAI_API_KEY not set"
→ Edit .env file and add your API key
→ Or: export OPENAI_API_KEY=sk-...

ERROR: "ModuleNotFoundError: No module named 'pykrige'"
→ Run: pip install pykrige

ERROR: "ChromaDB error"
→ Knowledge retrieval is optional - system uses hardcoded fallbacks
→ Doesn't affect core evolution


💡 TIPS & TRICKS
═══════════════════════════════════════════════════════════════════════════════

1. Start Small
   • Run test_core.py first to verify setup
   • Then test_mvp.py (3 iterations, 3 min)
   • Then main.py (full 25 iterations)

2. Monitor Progress
   • Each iteration shows: Weakness → Code → RMSE
   • Look for RMSE declining over iterations
   • Invalid code is normal (system handles it)

3. Cost Control
   • Use gpt-4o-mini for cheaper runs (~$0.06/iter)
   • Use gpt-4o only if you want best quality (~$0.15/iter)
   • Set n_iterations=5 for quick test (~$0.30)

4. Customize Analysis
   • Edit analyzer.py to detect different weaknesses
   • Edit prompt_builder.py to guide mutations
   • Edit config.yaml to adjust temperature/iterations

5. Save Results
   • Copy results/ folder before running again
   • Each run overwrites results/
   • Check iteration_log.json for detailed metrics


📚 PROJECT STRUCTURE
═══════════════════════════════════════════════════════════════════════════════

   geoevolve/
   ├── main.py                 Main evolution loop
   ├── test_core.py           Infrastructure test (no LLM)
   ├── test_mvp.py            MVP test (3 iterations)
   ├── config.yaml            Configuration
   ├── .env                   API keys (your secrets)
   ├── README.md              Full documentation
   │
   ├── agents/
   │   ├── evolver.py         LLM code mutation
   │   ├── analyzer.py        Weakness detection  
   │   └── prompt_builder.py  Prompt assembly
   │
   ├── evaluation/
   │   └── evaluator.py       Code execution + scoring
   │
   ├── tasks/kriging/
   │   ├── dataset.py         Data generation
   │   └── baseline.py        Kriging implementation
   │
   ├── rag/
   │   └── retriever.py       Knowledge base querying
   │
   └── results/               Output folder


🎯 NEXT STEPS
═══════════════════════════════════════════════════════════════════════════════

1. NOW:
   ✓ Run: python test_core.py
   ✓ Get OpenAI API key

2. SOON:
   → Run: python test_mvp.py
   → Check: results/iteration_log.json

3. LATER:
   → Run: python main.py (full 25 iterations)
   → Analyze: results/rmse_evolution.png
   → Review: results/best_kriging.py


📞 SUPPORT
═══════════════════════════════════════════════════════════════════════════════

Documentation:
• README.md              - Full feature overview
• BUILD_SUMMARY.md      - What was built
• test_core.py          - Simple example
• test_mvp.py           - Full example with LLM

Code comments:
• Each file has docstrings explaining what it does
• Each function has detailed argument/return docs


🚀 LET'S GO!
═══════════════════════════════════════════════════════════════════════════════

Ready to optimize kriging algorithms with AI? Try:

   $ cd geoevolve
   $ python test_core.py


Then:

   $ python test_mvp.py        # See it work (3 iterations)


Then:

   $ python main.py            # Full power (25 iterations)


Let's evolve! 🧬🌍

═══════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(QUICK_START)
    
    # Also save to file with UTF-8 encoding
    with open("QUICKSTART.txt", "w", encoding="utf-8") as f:
        f.write(QUICK_START)
    print("\n✓ Guide saved to QUICKSTART.txt")
