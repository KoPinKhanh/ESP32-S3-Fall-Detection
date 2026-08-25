import os
import time
import numpy as np
import joblib

def benchmark_model(model_path, features_path):
    print("=" * 50)
    print("BENCHMARK TỐC ĐỘ XỬ LÝ - MÔ HÌNH HIỆN TẠI")
    print("=" * 50)
    
    model = joblib.load(model_path)
    X = np.load(features_path)
    
    # Lấy 1 mẫu đơn để benchmark inference đơn lẻ
    single_sample = X[0:1]
    
    # --- Đo tốc độ inference MỘT mẫu ---
    N = 1000  # chạy 1,000 lần để có số liệu ổn định
    start = time.perf_counter()
    for _ in range(N):
        _ = model.predict(single_sample)
    end = time.perf_counter()
    
    total_time_single = end - start
    avg_ms_per_sample = (total_time_single / N) * 1000  # đổi sang mili-giây
    
    print(f"\n[1] Inference TỪng CỬA SỔ ĐƠN LẺ (200 mẫu x 6 trục):")
    print(f"    -> Thời gian trung bình: {avg_ms_per_sample:.4f} ms / lần")
    print(f"    -> Tốc độ xử lý:        {1000/avg_ms_per_sample:.0f} cửa sổ/giây (window/s)")

    # --- Đo tốc độ inference TOÀN BỘ dataset ---
    start = time.perf_counter()
    _ = model.predict(X)
    end = time.perf_counter()
    total_batch_ms = (end - start) * 1000
    
    print(f"\n[2] Inference TOÀN BỘ {len(X)} cửa sổ (batch mode):")
    print(f"    -> Tổng thời gian:  {total_batch_ms:.2f} ms  ({total_batch_ms/1000:.2f} giây)")
    print(f"    -> Tốc độ batch:    {len(X)/(total_batch_ms/1000):.0f} cửa sổ/giây")
    
    print(f"\n[Thông tin model]")
    print(f"    Số cây (n_estimators):   {model.n_estimators}")
    print(f"    Độ sâu tối đa (max_depth): {model.max_depth}")
    print(f"    Số đặc trưng đầu vào:    {model.n_features_in_}")

if __name__ == "__main__":
    benchmark_model(
        "../models/random_forest_model.pkl",
        "../dataset/processed/X_features.npy"
    )
