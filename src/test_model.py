import os
import random
import glob
import numpy as np
import pandas as pd
import joblib

def extract_features_from_window(window_data):
    """Tính 30 đặc trưng toán học cho 1 cửa sổ 200 mẫu"""
    mean_vals = np.mean(window_data, axis=0)
    std_vals = np.std(window_data, axis=0)
    min_vals = np.min(window_data, axis=0)
    max_vals = np.max(window_data, axis=0)
    var_vals = np.var(window_data, axis=0)
    
    return np.concatenate([mean_vals, std_vals, min_vals, max_vals, var_vals])

def test_random_csv(model_path, data_dir, window_size=200, step_size=100):
    # Load model
    print(f"Loading model từ: {model_path}...")
    model = joblib.load(model_path)
    
    # Chọn ngẫu nhiên 1 file CSV
    csv_files = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)
    random_file = random.choice(csv_files)
    print(f"Đang Test trên file ngẫu nhiên: {os.path.basename(random_file)}")
    
    features = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
    df = pd.read_csv(random_file)
    data = df[features].values
    
    try:
        labels = df['FallCheck'].values
    except KeyError:
        labels = None
    
    num_samples = len(data)
    print(f"File có {num_samples} mẫu (tương đương {num_samples/100:.2f} giây). Bắt đầu mô phỏng AI quét liên tục...\n")
    
    # Mô phỏng quét qua thời gian (Sliding Window)
    for start in range(0, num_samples - window_size + 1, step_size):
        end = start + window_size
        window_data = data[start:end]
        
        # 1. Trích xuất 30 đặc trưng
        feature_vector = extract_features_from_window(window_data)
        feature_vector = feature_vector.reshape(1, -1) # reshape thành mảng 2D cho scikit-learn
        
        # 2. AI Dự đoán
        prediction = model.predict(feature_vector)[0]
        
        # Lấy nhãn thực tế để so sánh (nếu có)
        actual_label = "Không xác định"
        if labels is not None:
            actual_label = "TÉ NGÃ!" if np.any(labels[start:end] == 1) else "Bình thường"
            
        pred_text = "TÉ NGÃ!" if prediction == 1 else "Bình thường"
        
        # Chỉ in ra nếu có té ngã (hoặc cả 2)
        print(f"Thời gian [{start/100:.1f}s - {end/100:.1f}s]: AI dự đoán -> {pred_text} | Thực tế: {actual_label}")

if __name__ == "__main__":
    MODEL_PATH = "../models/random_forest_model.pkl"
    DATASET_DIR = "../dataset/KFall/Sample_Training" 
    
    if not os.path.exists(MODEL_PATH):
        print("Không tìm thấy model. Hãy chạy train_random_forest.py trước!")
    else:
        test_random_csv(MODEL_PATH, DATASET_DIR)
