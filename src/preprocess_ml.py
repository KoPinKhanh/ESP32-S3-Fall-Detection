import os
import glob
import numpy as np
import pandas as pd

def process_kfall_ml(data_dir, window_size=200, step_size=100):
    """
    Reads KFall dataset CSV files, applies sliding window, and extracts 
    statistical features for Classical Machine Learning models.
    """
    csv_files = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return None, None
        
    print(f"Found {len(csv_files)} CSV files. Extracting ML features...")
    
    X_features = []
    y = []
    
    # 6 axes for MPU6050
    features = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
    label_col = 'FallCheck'
    
    # Process files
    # Using a simple counter instead of tqdm for simplicity
    count = 0
    total = len(csv_files)
    
    for file_path in csv_files:
        count += 1
        if count % 500 == 0:
            print(f"Processed {count}/{total} files...")
            
        try:
            df = pd.read_csv(file_path)
            if not all(col in df.columns for col in features + [label_col]):
                continue
                
            data = df[features].values
            labels = df[label_col].values
            num_samples = len(data)
            
            # Sliding window
            for start in range(0, num_samples - window_size + 1, step_size):
                end = start + window_size
                window_data = data[start:end]
                window_labels = labels[start:end]
                
                # Label is 1 if any fall occurs in the window
                window_label = 1 if np.any(window_labels == 1) else 0
                
                # --- FEATURE EXTRACTION ---
                # For each of the 6 axes, compute 5 statistical features:
                # mean, std, min, max, variance
                # Shape of window_data is (200, 6). We compute along axis 0.
                mean_vals = np.mean(window_data, axis=0)
                std_vals = np.std(window_data, axis=0)
                min_vals = np.min(window_data, axis=0)
                max_vals = np.max(window_data, axis=0)
                var_vals = np.var(window_data, axis=0)
                
                # Concatenate all features into a single 1D vector (length = 30)
                feature_vector = np.concatenate([
                    mean_vals, std_vals, min_vals, max_vals, var_vals
                ])
                
                X_features.append(feature_vector)
                y.append(window_label)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
            
    X_features = np.array(X_features)
    y = np.array(y)
    
    print(f"Extraction complete! Extracted {len(X_features)} windows.")
    print(f"Shape of X_features: {X_features.shape}")
    print(f"Shape of y: {y.shape}")
    
    return X_features, y

if __name__ == "__main__":
    DATASET_DIR = "../dataset/KFall"
    OUTPUT_DIR = "../dataset/processed"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    X_features, y = process_kfall_ml(DATASET_DIR, window_size=200, step_size=100)
    
    if X_features is not None:
        np.save(os.path.join(OUTPUT_DIR, "X_features.npy"), X_features)
        # We can overwrite or save a separate y.npy, but they match 1:1
        np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
        print("Done! ML features saved to", OUTPUT_DIR)
