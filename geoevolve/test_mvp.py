#!/usr/bin/env python3
"""
Quick test of GeoEvolve MVP without RAG.
Tests: Dataset → Baseline → Evolution Loop (hardcoded analysis first)
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add geoevolve to path
sys.path.insert(0, os.getcwd())

from tasks.kriging.dataset import generate_synthetic_dataset, load_and_split_dataset
from tasks.kriging.baseline import baseline_kriging
from evaluation.evaluator import safe_exec
from agents.evolver import CodeEvolver
from agents.analyzer import CodeAnalyzer
from agents.prompt_builder import PromptBuilder


def test_mvp():
    """Run minimal MVP test."""
    
    print("=" * 60)
    print("GeoEvolve MVP Test")
    print("=" * 60)
    
    # 1. Generate dataset
    print("\n1. Generating dataset...")
    dataset_path = "data/minerals/Cu.csv"
    if not os.path.exists(dataset_path):
        df = generate_synthetic_dataset(n_samples=300)
        os.makedirs("data/minerals", exist_ok=True)
        df.to_csv(dataset_path, index=False)
    splits = load_and_split_dataset(dataset_path)
    print(f"   ✓ Dataset loaded: {len(splits['train'][0])} train, {len(splits['test'][0])} test")
    
    # 2. Get baseline
    print("\n2. Computing baseline...")
    result = baseline_kriging(splits)
    baseline_code = result['code']
    baseline_rmse = result['rmse']
    print(f"   ✓ Baseline RMSE: {baseline_rmse:.6f}")
    
    # 3. Initialize agents
    print("\n3. Initializing agents...")
    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("   ✗ ERROR: GOOGLE_API_KEY or GEMINI_API_KEY not set. Skipping LLM tests.")
        print("   Set GOOGLE_API_KEY=your_key in .env file or environment")
        return
    
    evolver = CodeEvolver(model="gemini-flash-latest", temperature=0.8)
    analyzer = CodeAnalyzer(model="gemini-flash-latest")
    builder = PromptBuilder()
    print("   ✓ Agents initialized")
    
    # 4. Test analyzer
    print("\n4. Testing analyzer...")
    try:
        analysis = analyzer.analyze(baseline_code, baseline_rmse)
        weakness = analysis.get('weakness', 'Unknown')
        query = analysis.get('query', 'kriging')
        print(f"   ✓ Weakness identified: {weakness[:60]}...")
        print(f"   ✓ Query generated: {query}")
    except Exception as e:
        print(f"   ✗ Analyzer failed: {e}")
        return
    
    # 5. Test prompt builder
    print("\n5. Building prompt...")
    prompt = builder.build(
        code=baseline_code,
        weakness=weakness,
        suggestion="Try different variogram model"
    )
    print(f"   ✓ Prompt built ({len(prompt)} chars)")
    
    # 6. Test evolver (single mutation)
    print("\n6. Testing LLM code mutation...")
    try:
        new_code = evolver.mutate(baseline_code, prompt)
        print(f"   ✓ Code generated ({len(new_code)} chars)")
        print(f"   Sample: {new_code[:200]}...")
    except Exception as e:
        print(f"   ✗ Evolver failed: {e}")
        return
    
    # 7. Test evaluator
    print("\n7. Testing evaluator on generated code...")
    rmse, valid = safe_exec(new_code, splits)
    print(f"   ✓ Evaluation: RMSE={rmse:.6f}, Valid={valid}")
    
    if valid:
        improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100
        status = "✓ IMPROVED" if rmse < baseline_rmse else "✗ WORSE"
        print(f"   {status}: {improvement:+.2f}% vs baseline")
    else:
        print(f"   ✗ Generated code is invalid")
    
    # 8. Quick evolution loop (3 iterations)
    print("\n8. Running 3-iteration evolution loop...")
    best_code = baseline_code
    best_rmse = baseline_rmse
    
    for i in range(3):
        print(f"\n   Iteration {i+1}:")
        
        # Analyze
        analysis = analyzer.analyze(best_code, best_rmse)
        weakness = analysis.get('weakness', 'Weakness unknown')
        
        # Build prompt
        prompt = builder.build(best_code, weakness)
        
        # Mutate
        new_code = evolver.mutate(best_code, prompt)
        
        # Evaluate
        rmse, valid = safe_exec(new_code, splits)
        
        # Check improvement
        if valid and rmse < best_rmse:
            improvement = best_rmse - rmse
            print(f"     ✓ Improved: {best_rmse:.6f} → {rmse:.6f} ({improvement:+.4f})")
            best_code = new_code
            best_rmse = rmse
        else:
            status = "Invalid" if not valid else "Worse"
            print(f"     ✗ {status}: RMSE={rmse:.6f}")
    
    print("\n" + "=" * 60)
    print(f"Final Results:")
    print(f"  Baseline RMSE: {baseline_rmse:.6f}")
    print(f"  Best RMSE:     {best_rmse:.6f}")
    total_improvement = ((baseline_rmse - best_rmse) / baseline_rmse) * 100
    print(f"  Improvement:   {total_improvement:+.2f}%")
    print("=" * 60)
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "test_results.json", 'w') as f:
        json.dump({
            'baseline_rmse': baseline_rmse,
            'best_rmse': best_rmse,
            'improvement_percent': total_improvement,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    with open(results_dir / "best_code_test.py", 'w') as f:
        f.write(best_code)
    
    print("\nResults saved to results/")


if __name__ == "__main__":
    test_mvp()
