#!/usr/bin/env python3
"""
Test core GeoEvolve infrastructure (no LLM required).
Tests: Dataset generation, baseline kriging, evaluator.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

from tasks.kriging.dataset import generate_synthetic_dataset, load_and_split_dataset
from tasks.kriging.baseline import baseline_kriging
from evaluation.evaluator import safe_exec


def test_core():
    """Test core functionality without LLM."""
    
    print("=" * 70)
    print("GeoEvolve Core Infrastructure Test (No LLM Required)")
    print("=" * 70)
    
    # 1. Dataset generation
    print("\n[1/4] Generating synthetic dataset...")
    try:
        df = generate_synthetic_dataset(n_samples=300, seed=42)
        print(f"      ✓ Generated {len(df)} samples with columns: {list(df.columns)}")
        print(f"      ✓ Shape: {df.shape}, Memory: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    except Exception as e:
        print(f"      ✗ Failed: {e}")
        return False
    
    # 2. Dataset loading and splitting
    print("\n[2/4] Loading and splitting dataset...")
    try:
        dataset_path = "data/minerals/Cu.csv"
        os.makedirs("data/minerals", exist_ok=True)
        df.to_csv(dataset_path, index=False)
        
        splits = load_and_split_dataset(dataset_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        
        train_size = len(splits['train'][0])
        val_size = len(splits['val'][0])
        test_size = len(splits['test'][0])
        
        print(f"      ✓ Train set: {train_size} samples")
        print(f"      ✓ Val set:   {val_size} samples")
        print(f"      ✓ Test set:  {test_size} samples")
        print(f"      ✓ Total:     {train_size + val_size + test_size} samples")
    except Exception as e:
        print(f"      ✗ Failed: {e}")
        return False
    
    # 3. Baseline kriging
    print("\n[3/4] Computing baseline Kriging...")
    try:
        result = baseline_kriging(splits)
        baseline_rmse = result['rmse']
        baseline_mae = result['mae']
        baseline_r2 = result['r2']
        
        print(f"      ✓ Baseline computed successfully")
        print(f"      ✓ RMSE: {baseline_rmse:.6f}")
        print(f"      ✓ MAE:  {baseline_mae:.6f}")
        print(f"      ✓ R²:   {baseline_r2:.6f}")
    except Exception as e:
        print(f"      ✗ Failed: {e}")
        return False
    
    # 4. Code evaluator
    print("\n[4/4] Testing evaluator with baseline code...")
    try:
        code = result['code']
        rmse_eval, valid = safe_exec(code, splits)
        
        if valid:
            print(f"      ✓ Code executed successfully")
            print(f"      ✓ RMSE from evaluator: {rmse_eval:.6f}")
            print(f"      ✓ Match with baseline: {abs(rmse_eval - baseline_rmse) < 0.0001}")
        else:
            print(f"      ✗ Code execution failed")
            return False
    except Exception as e:
        print(f"      ✗ Failed: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✓ All core tests passed!")
    print("=" * 70)
    
    print("\nNext steps:")
    print("  1. Set OPENAI_API_KEY in .env file")
    print("  2. Run: python test_mvp.py")
    print("\nProject structure:")
    print("  geoevolve/")
    print("  ├── main.py                 # Full evolution loop")
    print("  ├── test_mvp.py             # MVP test with LLM")
    print("  ├── config.yaml             # Configuration")
    print("  ├── agents/                 # Evolver, Analyzer")
    print("  ├── evaluation/             # Evaluator, metrics")
    print("  ├── tasks/kriging/          # Baseline, dataset")
    print("  ├── rag/                    # Knowledge retrieval")
    print("  ├── results/                # Output directory")
    print("  └── data/")
    print("      ├── minerals/           # Datasets")
    print("      ├── knowledge/          # Knowledge base")
    print("      └── chroma_db/          # Vector DB")
    
    return True


if __name__ == "__main__":
    success = test_core()
    sys.exit(0 if success else 1)
