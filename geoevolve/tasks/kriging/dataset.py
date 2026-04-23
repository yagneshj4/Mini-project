import numpy as np
import pandas as pd

def load_and_split_dataset(filepath, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Load dataset and split into train/val/test.
    
    Args:
        filepath: Path to CSV file with columns [x, y, cu_ppm]
        train_ratio, val_ratio, test_ratio: Split proportions
        seed: Random seed
        
    Returns:
        dict: {
            'train': (x_train, y_train, z_train),
            'val': (x_val, y_val, z_val),
            'test': (x_test, y_test, z_test),
            'full_df': full dataframe
        }
    """
    df = pd.read_csv(filepath)
    
    # Shuffle with seed
    np.random.seed(seed)
    df = df.sample(frac=1).reset_index(drop=True)
    
    n = len(df)
    train_idx = int(n * train_ratio)
    val_idx = train_idx + int(n * val_ratio)
    
    train_df = df.iloc[:train_idx]
    val_df = df.iloc[train_idx:val_idx]
    test_df = df.iloc[val_idx:]
    
    splits = {
        'train': (train_df['x'].values, train_df['y'].values, train_df['cu_ppm'].values),
        'val': (val_df['x'].values, val_df['y'].values, val_df['cu_ppm'].values),
        'test': (test_df['x'].values, test_df['y'].values, test_df['cu_ppm'].values),
        'full_df': df
    }
    
    return splits

def generate_synthetic_dataset(n_samples=300, seed=42):
    """
    Generate synthetic geospatial copper concentration data.
    Mimics Australian Minerals dataset structure.
    
    Returns:
        DataFrame with columns: x (easting), y (northing), cu_ppm (copper in ppm)
    """
    np.random.seed(seed)
    
    # Generate coordinates
    x = np.random.uniform(0, 100, n_samples)
    y = np.random.uniform(0, 100, n_samples)
    
    # Generate values based on spatial pattern
    # Create smooth spatial correlation using sine/cosine patterns
    vals = (
        np.sin(x / 20) * np.cos(y / 20) * 50 +
        0.3 * x + 0.2 * y +
        np.random.normal(0, 10, n_samples)
    )
    
    # Ensure positive values (copper concentration)
    vals = np.abs(vals) + 10
    
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'cu_ppm': vals
    })
    
    return df

if __name__ == "__main__":
    # Generate and save the dataset
    df = generate_synthetic_dataset(n_samples=300)
    output_path = "data/minerals/Cu.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nBasic stats:")
    print(df.describe())
