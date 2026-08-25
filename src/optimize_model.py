import os
import time
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================================================
# Script so sánh nhiều cấu hình Random Forest và tìm ra
# mô hình cân bằng tốt nhất giữa Tốc độ và Độ chính xác
# =========================================================

def benchmark_single(model, sample):
    """Đo tốc độ inference trung bình cho 1 mẫu đơn (1,000 lần)"""
    N = 1000
    start = time.perf_counter()
    for _ in range(N):
        _ = model.predict(sample)
    end = time.perf_counter()
    return ((end - start) / N) * 1000  # ms

def train_and_eval(X_train, y_train, X_test, y_test, sample, config):
    model = RandomForestClassifier(**config, class_weight='balanced',
                                   random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    avg_ms = benchmark_single(model, sample)
    return model, acc, avg_ms

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    PROCESSED_DIR = "../dataset/processed"
    MODELS_DIR    = "../models"

    print("Đang nạp dữ liệu...")
    X = np.load(os.path.join(PROCESSED_DIR, "X_features.npy"))
    y = np.load(os.path.join(PROCESSED_DIR, "y.npy"))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                         random_state=42, stratify=y)
    sample = X_test[0:1]

    # ---- Thông số gốc (baseline) ----
    BASELINE_MS  = 31.79  # từ kết quả benchmark trước
    BASELINE_ACC = 91.56  # từ lần train trước

    print("\n" + "=" * 62)
    print(f"  BASELINE (50 cây, max_depth=10): {BASELINE_ACC:.2f}% acc | {BASELINE_MS:.2f} ms")
    print("=" * 62)

    # ---- Các cấu hình cần thử ----
    configs = [
        {"label": "Opt-A  (20 cây, depth=8)",  "n_estimators": 20,  "max_depth":  8},
        {"label": "Opt-B  (10 cây, depth=8)",  "n_estimators": 10,  "max_depth":  8},
        {"label": "Opt-C  (10 cây, depth=6)",  "n_estimators": 10,  "max_depth":  6},
        {"label": "Opt-D  ( 5 cây, depth=6)",  "n_estimators":  5,  "max_depth":  6},
        {"label": "Opt-E  ( 5 cây, depth=5)",  "n_estimators":  5,  "max_depth":  5},
    ]

    best_model  = None
    best_label  = ""
    best_acc    = 0
    best_ms     = 9999

    print("\n  Huấn luyện và đo tốc độ từng cấu hình...\n")
    for cfg in configs:
        label = cfg.pop("label")
        model, acc, avg_ms = train_and_eval(
            X_train, y_train, X_test, y_test, sample, cfg
        )
        speedup = BASELINE_MS / avg_ms
        print(f"  {label}: Accuracy={acc:.2f}%  |  Inference={avg_ms:.3f} ms  "
              f"|  Tăng tốc x{speedup:.1f} so với baseline")
        cfg["label"] = label   # restore

        # Chọn model có tốc độ nhanh nhất nhưng accuracy >= 88%
        if acc >= 88.0 and avg_ms < best_ms:
            best_model = model
            best_label = label
            best_ms    = avg_ms
            best_acc   = acc

    print("\n" + "=" * 62)
    print(f"  ✅ Mô hình tối ưu được chọn: {best_label}")
    print(f"     Accuracy  : {best_acc:.2f}%")
    print(f"     Inference : {best_ms:.3f} ms  (nhanh hơn x{BASELINE_MS/best_ms:.1f})")
    print("=" * 62)

    # Lưu mô hình tối ưu
    opt_path = os.path.join(MODELS_DIR, "random_forest_optimized.pkl")
    joblib.dump(best_model, opt_path)
    print(f"\n  Mô hình tối ưu đã lưu tại: {opt_path}")

    # In classification report của mô hình tốt nhất
    y_pred_best = best_model.predict(X_test)
    print("\n  Classification Report (Mô hình tối ưu):")
    print(classification_report(y_test, y_pred_best,
                                 target_names=["Bình thường", "Té ngã"]))
