#!/usr/bin/env python3
"""
GeoEvolve Main Loop
Orchestrates the entire kriging code evolution process.
"""

import os
import json
import sys
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Import project modules
from tasks.kriging.dataset import load_and_split_dataset, generate_synthetic_dataset
from tasks.kriging.baseline import baseline_kriging
from evaluation.evaluator import safe_exec
from agents.evolver import CodeEvolver
from agents.analyzer import CodeAnalyzer
from agents.prompt_builder import PromptBuilder
from rag.retriever import RAGRetriever


class GeoEvolve:
    """
    Main orchestrator for kriging code evolution.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize GeoEvolve with configuration.
        
        Args:
            config_path: Path to YAML config file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.iteration_log = []
        self.best_code = None
        self.best_rmse = float('inf')
        
        print(f"GeoEvolve initialized with config:")
        print(f"  Iterations: {self.config['n_iterations']}")
        print(f"  Task: {self.config['task']}")
        print(f"  Dataset: {self.config['dataset']}")
    
    def setup_dataset(self) -> Dict:
        """
        Load or generate dataset and split into train/val/test.
        
        Returns:
            dict: Dataset splits
        """
        dataset_path = f"data/minerals/{self.config['dataset']}.csv"
        
        if not os.path.exists(dataset_path):
            print(f"Dataset not found. Generating synthetic dataset...")
            df = generate_synthetic_dataset(n_samples=300)
            os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
            df.to_csv(dataset_path, index=False)
        
        print(f"Loading dataset from {dataset_path}")
        splits = load_and_split_dataset(dataset_path)
        
        print(f"Dataset sizes:")
        print(f"  Train: {len(splits['train'][0])}")
        print(f"  Val: {len(splits['val'][0])}")
        print(f"  Test: {len(splits['test'][0])}")
        
        return splits
    
    def get_baseline(self, splits: Dict) -> Tuple[str, float]:
        """
        Get baseline kriging code and RMSE.
        
        Args:
            splits: Dataset splits
            
        Returns:
            tuple: (baseline_code, baseline_rmse)
        """
        print("\n=== Computing Baseline ===")
        result = baseline_kriging(splits)
        
        baseline_code = result['code']
        baseline_rmse = result['rmse']
        
        print(f"Baseline RMSE: {baseline_rmse:.6f}")
        
        return baseline_code, baseline_rmse
    
    def run_evolution_loop(self, splits: Dict):
        """
        Main evolution loop: mutate code, evaluate, analyze, improve.
        
        Args:
            splits: Dataset splits
        """
        # Setup
        baseline_code, baseline_rmse = self.get_baseline(splits)
        self.best_code = baseline_code
        self.best_rmse = baseline_rmse
        
        # Initialize agents
        evolver = CodeEvolver(
            model=self.config.get('model', 'gpt-4o-mini'),
            temperature=self.config.get('temperature', 0.8)
        )
        analyzer = CodeAnalyzer()
        builder = PromptBuilder()
        retriever = RAGRetriever()
        
        # Try to load knowledge base
        retriever.load_and_index_files()
        
        print(f"\n=== Starting Evolution Loop ({self.config['n_iterations']} iterations) ===\n")
        
        for iteration in range(self.config['n_iterations']):
            print(f"\n--- Iteration {iteration + 1}/{self.config['n_iterations']} ---")
            
            try:
                # Step 1: Analyze current best code
                print("Analyzing code...")
                analysis = analyzer.analyze(self.best_code, self.best_rmse)
                weakness = analysis.get('weakness', 'Unknown weakness')
                query = analysis.get('query', 'kriging improvement')
                suggestion = analysis.get('suggestion', '')
                
                print(f"Weakness: {weakness[:80]}...")
                print(f"Query: {query}")
                
                # Step 2: Retrieve knowledge
                print("Retrieving knowledge...")
                rag_context = retriever.query(query, top_k=self.config.get('top_k_rag', 5))
                
                # Step 3: Build prompt
                prompt = builder.build(
                    code=self.best_code,
                    weakness=weakness,
                    rag_context=rag_context,
                    suggestion=suggestion
                )
                
                # Step 4: Mutate code
                print("Generating improved code...")
                new_code = evolver.mutate(self.best_code, prompt)
                
                # Step 5: Evaluate
                print("Evaluating...")
                rmse, valid = safe_exec(new_code, splits)
                
                # Step 6: Log results
                log_entry = {
                    'iteration': iteration + 1,
                    'rmse': rmse,
                    'valid': valid,
                    'improvement': self.best_rmse - rmse,
                    'weakness': weakness,
                    'query': query,
                    'timestamp': datetime.now().isoformat()
                }
                self.iteration_log.append(log_entry)
                
                # Step 7: Update best if improved
                if valid and rmse < self.best_rmse:
                    print(f"✓ Improved! RMSE: {self.best_rmse:.6f} → {rmse:.6f}")
                    self.best_code = new_code
                    self.best_rmse = rmse
                else:
                    improvement_text = "Improvement" if valid else "Invalid"
                    print(f"✗ {improvement_text}. Current best RMSE: {self.best_rmse:.6f}")
                
                # Print log
                print(f"Log: RMSE={rmse:.6f}, Valid={valid}")
                
            except Exception as e:
                print(f"Error in iteration: {e}")
                import traceback
                traceback.print_exc()
                
                log_entry = {
                    'iteration': iteration + 1,
                    'rmse': 999.0,
                    'valid': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                self.iteration_log.append(log_entry)
            
            # Save checkpoint
            if (iteration + 1) % 5 == 0:
                self.save_checkpoint()
    
    def save_checkpoint(self):
        """Save current progress to results."""
        # Save iteration log
        log_path = self.results_dir / "iteration_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.iteration_log, f, indent=2)
        
        # Save best code
        if self.best_code:
            code_path = self.results_dir / "best_kriging.py"
            with open(code_path, 'w') as f:
                f.write(self.best_code)
        
        print(f"Checkpoint saved to {self.results_dir}")
    
    def generate_report(self):
        """Generate final report and visualizations."""
        print("\n=== Final Report ===")
        
        if not self.iteration_log:
            print("No iterations completed")
            return
        
        # Extract metrics
        iterations = [log['iteration'] for log in self.iteration_log]
        rmses = [log['rmse'] for log in self.iteration_log]
        valid_count = sum(1 for log in self.iteration_log if log['valid'])
        
        # Statistics
        valid_rmses = [log['rmse'] for log in self.iteration_log if log['valid']]
        
        print(f"\nResults:")
        print(f"  Iterations: {len(self.iteration_log)}")
        print(f"  Valid codes: {valid_count}")
        print(f"  Best RMSE: {self.best_rmse:.6f}")
        print(f"  Worst RMSE: {max(rmses):.6f}")
        if valid_rmses:
            print(f"  Mean valid RMSE: {np.mean(valid_rmses):.6f}")
        
        # Try to plot
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # RMSE over iterations
            ax1.plot(iterations, rmses, 'b-o', label='RMSE')
            ax1.axhline(self.best_rmse, color='g', linestyle='--', label='Best')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('RMSE')
            ax1.set_title('RMSE Evolution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Valid vs invalid
            valid_iter = [log['iteration'] for log in self.iteration_log if log['valid']]
            valid_rmse = [log['rmse'] for log in self.iteration_log if log['valid']]
            invalid_rmse = [log['rmse'] for log in self.iteration_log if not log['valid']]
            invalid_iter = [log['iteration'] for log in self.iteration_log if not log['valid']]
            
            if valid_iter:
                ax2.scatter(valid_iter, valid_rmse, color='green', label='Valid', s=100)
            if invalid_iter:
                ax2.scatter(invalid_iter, invalid_rmse, color='red', label='Invalid', s=100)
            
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('RMSE')
            ax2.set_title('Code Validity')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plot_path = self.results_dir / "rmse_evolution.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            print(f"\nPlot saved to {plot_path}")
            plt.close()
        
        except ImportError:
            print("Matplotlib not available for plotting")
        
        # Save final report
        report = {
            'total_iterations': len(self.iteration_log),
            'valid_codes': valid_count,
            'baseline_rmse': self.iteration_log[0]['rmse'] if self.iteration_log else None,
            'best_rmse': self.best_rmse,
            'improvement_percent': ((self.iteration_log[0]['rmse'] - self.best_rmse) / self.iteration_log[0]['rmse'] * 100) if self.iteration_log else 0,
            'logs': self.iteration_log
        }
        
        report_path = self.results_dir / "final_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {report_path}")


def main():
    """Main entry point."""
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY not set in environment")
        print("Please set your OpenAI API key in .env file")
        sys.exit(1)
    
    # Initialize and run
    geoevolve = GeoEvolve()
    
    # Load dataset
    splits = geoevolve.setup_dataset()
    
    # Run evolution
    geoevolve.run_evolution_loop(splits)
    
    # Save final results
    geoevolve.save_checkpoint()
    geoevolve.generate_report()
    
    print("\n=== GeoEvolve Complete ===")


if __name__ == "__main__":
    main()
