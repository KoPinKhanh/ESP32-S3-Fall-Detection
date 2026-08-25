import os
import glob
import numpy as np
import pandas as pd
def tqdm(x): return x

def process_kfall_data(data_dir, window_size=200, step_size=100):
    """
    Reads KFall dataset CSV files, extracts 6-axis features,
    and applies sliding window.
    """
    # Look for all csv files in the dataset folder recursively
    csv_files = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return None, None
        
    print(f"Found {len(csv_files)} CSV files. Processing...")
    
    X = []
    y = []
    
    # KFall features we want to use (Assuming MPU6050: 6-axis)
    # We drop Euler angles as they are not always available on standard 6-axis IMUs
    features = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
    label_col = 'FallCheck'
    
    # To avoid memory explosion, process files iteratively
    for file_path in tqdm(csv_files):
        try:
            df = pd.read_csv(file_path)
            # Ensure required columns exist
            if not all(col in df.columns for col in features + [label_col]):
                continue
                
            # Convert to numpy array for faster sliding window
            data = df[features].values
            labels = df[label_col].values
            
            num_samples = len(data)
            
            # Sliding window
            for start in range(0, num_samples - window_size + 1, step_size):
                end = start + window_size
                window_data = data[start:end]
                window_labels = labels[start:end]
                
                # Rule for labeling the window:
                # If any point in the window is a fall (1), label the window as fall (1)
                window_label = 1 if np.any(window_labels == 1) else 0
                
                X.append(window_data)
                y.append(window_label)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
            
    X = np.array(X)
    y = np.array(y)
    print(f"Extracted {len(X)} windows. Shape of X: {X.shape}, Shape of y: {y.shape}")
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"Class distribution: {dist}")
    
    return X, y

if __name__ == "__main__":
    # Path to dataset (Update if necessary)
    DATASET_DIR = "../dataset/KFall"
    OUTPUT_DIR = "../dataset/processed"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set window size (200 samples = 2 seconds at 100Hz)
    WINDOW_SIZE = 200
    STEP_SIZE = 100 # 50% overlap
    
    print("Starting Data Preprocessing...")
    X, y = process_kfall_data(DATASET_DIR, window_size=WINDOW_SIZE, step_size=STEP_SIZE)
    
    if X is not None:
        print("Saving processed data...")
        np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
        np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
        print("Done! Data saved to", OUTPUT_DIR)
