import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from .dataset import load_and_split_dataset

def baseline_kriging(dataset_splits):
    """
    Run standard Ordinary Kriging on training data.
    Evaluate on test set.
    
    Args:
        dataset_splits: Output from load_and_split_dataset
        
    Returns:
        dict: {
            'code': Python code string,
            'rmse': RMSE on test set,
            'mae': MAE on test set,
            'r2': R2 on test set,
            'predictions': Predictions on test set
        }
    """
    x_train, y_train, z_train = dataset_splits['train']
    x_test, y_test, z_test = dataset_splits['test']
    
    # Perform Ordinary Kriging
    ok = OrdinaryKriging(
        x_train, y_train, z_train,
        variogram_model='spherical',
        verbose=False,
        enable_plotting=False
    )
    
    # Predict on test set
    z_pred, ss = ok.execute('points', x_test, y_test)
    
    # Compute metrics
    rmse = np.sqrt(mean_squared_error(z_test, z_pred))
    mae = mean_absolute_error(z_test, z_pred)
    r2 = r2_score(z_test, z_pred)
    
    # Generate the code as a string
    code = """
import numpy as np
from pykrige.ok import OrdinaryKriging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate(dataset):
    '''
    Baseline Ordinary Kriging with spherical variogram.
    '''
    x_train, y_train, z_train = dataset['train']
    x_test, y_test, z_test = dataset['test']
    
    # Ordinary Kriging
    ok = OrdinaryKriging(
        x_train, y_train, z_train,
        variogram_model='spherical',
        verbose=False,
        enable_plotting=False
    )
    
    z_pred, ss = ok.execute('points', x_test, y_test)
    
    rmse = np.sqrt(mean_squared_error(z_test, z_pred))
    return rmse
"""
    
    return {
        'code': code,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': z_pred
    }

if __name__ == "__main__":
    # Test the baseline
    dataset_path = "data/minerals/Cu.csv"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Run dataset.py to generate it first.")
    else:
        splits = load_and_split_dataset(dataset_path)
        result = baseline_kriging(splits)
        
        print(f"Baseline Kriging Results:")
        print(f"RMSE: {result['rmse']:.4f}")
        print(f"MAE: {result['mae']:.4f}")
        print(f"R2: {result['r2']:.4f}")
        print(f"\nGenerated Code:")
        print(result['code'])
