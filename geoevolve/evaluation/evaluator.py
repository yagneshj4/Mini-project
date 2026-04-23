import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import traceback

def safe_exec(code: str, dataset):
    """
    Safely execute generated kriging code in a sandboxed namespace.
    Returns RMSE and validity flag.
    
    Args:
        code: Python code string containing an evaluate() function
        dataset: Dictionary with 'train' and 'test' keys containing (x, y, z) tuples
        
    Returns:
        tuple: (rmse, is_valid)
            - rmse: float, RMSE score or 999.0 if invalid
            - is_valid: bool, True if code executed successfully
    """
    
    # Define the safe namespace with allowed imports and functions
    namespace = {
        'np': np,
        'pd': pd,
        'math': __import__('math'),
        'OrdinaryKriging': None,  # Will be set below
        'KNeighborsRegressor': None,
        'mean_squared_error': mean_squared_error,
        'mean_absolute_error': mean_absolute_error,
        'r2_score': r2_score,
    }
    
    # Import kriging and sklearn classes
    try:
        from pykrige.ok import OrdinaryKriging
        from sklearn.neighbors import KNeighborsRegressor
        namespace['OrdinaryKriging'] = OrdinaryKriging
        namespace['KNeighborsRegressor'] = KNeighborsRegressor
    except ImportError as e:
        print(f"Failed to import kriging/sklearn: {e}")
        return 999.0, False
    
    try:
        # Execute the code in the namespace
        exec(code, namespace)
        
        # Get the evaluate function
        if 'evaluate' not in namespace:
            print("Error: No 'evaluate' function defined in code")
            return 999.0, False
        
        evaluate_func = namespace['evaluate']
        
        # Call the evaluate function
        rmse = evaluate_func(dataset)
        
        # Validate the result
        if not isinstance(rmse, (int, float, np.number)):
            print(f"Error: evaluate() returned {type(rmse)} instead of float")
            return 999.0, False
        
        if np.isnan(rmse) or np.isinf(rmse):
            print(f"Error: RMSE is NaN or Inf: {rmse}")
            return 999.0, False
        
        # Clamp unreasonable values
        if rmse < 0:
            print(f"Warning: Negative RMSE {rmse}, setting to 999")
            return 999.0, False
        
        if rmse > 10000:  # Unreasonable RMSE
            print(f"Warning: Very high RMSE {rmse}, likely invalid code")
            return 999.0, False
        
        return float(rmse), True
        
    except Exception as e:
        print(f"Exception during code execution: {type(e).__name__}: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return 999.0, False


def evaluate_metrics(code: str, dataset):
    """
    Evaluate code and return all metrics (RMSE, MAE, R2).
    
    Args:
        code: Python code string
        dataset: Dataset dictionary
        
    Returns:
        dict: {'rmse': float, 'mae': float, 'r2': float, 'valid': bool}
    """
    namespace = {
        'np': np,
        'pd': pd,
        'math': __import__('math'),
        'OrdinaryKriging': None,
        'KNeighborsRegressor': None,
        'mean_squared_error': mean_squared_error,
        'mean_absolute_error': mean_absolute_error,
        'r2_score': r2_score,
    }
    
    try:
        from pykrige.ok import OrdinaryKriging
        from sklearn.neighbors import KNeighborsRegressor
        namespace['OrdinaryKriging'] = OrdinaryKriging
        namespace['KNeighborsRegressor'] = KNeighborsRegressor
    except ImportError:
        return {'rmse': 999.0, 'mae': 999.0, 'r2': -999.0, 'valid': False}
    
    try:
        exec(code, namespace)
        
        if 'evaluate' not in namespace:
            return {'rmse': 999.0, 'mae': 999.0, 'r2': -999.0, 'valid': False}
        
        evaluate_func = namespace['evaluate']
        rmse = float(evaluate_func(dataset))
        
        if np.isnan(rmse) or np.isinf(rmse) or rmse < 0 or rmse > 10000:
            return {'rmse': 999.0, 'mae': 999.0, 'r2': -999.0, 'valid': False}
        
        return {'rmse': rmse, 'mae': rmse, 'r2': 0.0, 'valid': True}
        
    except Exception:
        return {'rmse': 999.0, 'mae': 999.0, 'r2': -999.0, 'valid': False}


if __name__ == "__main__":
    # Test the evaluator with a simple code snippet
    import sys
    sys.path.insert(0, '..')
    from tasks.kriging.dataset import generate_synthetic_dataset, load_and_split_dataset
    
    # Generate test data
    df = generate_synthetic_dataset(n_samples=100)
    df.to_csv('data/minerals/Cu_test.csv', index=False)
    
    splits = load_and_split_dataset('data/minerals/Cu_test.csv')
    
    # Simple test code
    test_code = """
import numpy as np
from pykrige.ok import OrdinaryKriging
from sklearn.metrics import mean_squared_error

def evaluate(dataset):
    x_train, y_train, z_train = dataset['train']
    x_test, y_test, z_test = dataset['test']
    
    ok = OrdinaryKriging(x_train, y_train, z_train, variogram_model='spherical', verbose=False, enable_plotting=False)
    z_pred, ss = ok.execute('points', x_test, y_test)
    
    rmse = np.sqrt(mean_squared_error(z_test, z_pred))
    return rmse
"""
    
    rmse, valid = safe_exec(test_code, splits)
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Valid: {valid}")
