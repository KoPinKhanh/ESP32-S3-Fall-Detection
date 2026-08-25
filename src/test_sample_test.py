import os
import glob
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from datetime import datetime

def extract_features(window_data):
    """Tính 30 đặc trưng cho 1 cửa sổ (phải khớp với lúc training)"""
    mean_vals = np.mean(window_data, axis=0)
    std_vals  = np.std(window_data,  axis=0)
    min_vals  = np.min(window_data,  axis=0)
    max_vals  = np.max(window_data,  axis=0)
    var_vals  = np.var(window_data,  axis=0)
    return np.concatenate([mean_vals, std_vals, min_vals, max_vals, var_vals])

def run_test_on_sample_test(model_path, sample_test_dir,
                            window_size=200, step_size=100,
                            output_dir="../results"):
    # ── 0. Chuẩn bị ──────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    model = joblib.load(model_path)
    FEATURES = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']

    csv_files = sorted(glob.glob(
        os.path.join(sample_test_dir, '**', '*.csv'), recursive=True))
    print(f"Tìm thấy {len(csv_files)} file CSV trong Sample_Test.\n")

    # ── 1. Thu thập kết quả theo từng file ───────────────────────
    rows = []            # chi tiết từng cửa sổ
    file_summaries = []  # tóm tắt theo từng file

    all_y_true, all_y_pred = [], []

    for csv_path in csv_files:
        subject = os.path.basename(os.path.dirname(csv_path))   # SA21, SA23, ...
        filename = os.path.splitext(os.path.basename(csv_path))[0]

        try:
            df = pd.read_csv(csv_path)
            if not all(c in df.columns for c in FEATURES + ['FallCheck']):
                continue
            data   = df[FEATURES].values
            labels = df['FallCheck'].values
        except Exception as e:
            print(f"Lỗi đọc {csv_path}: {e}")
            continue

        n = len(data)
        file_preds, file_truths = [], []

        for start in range(0, n - window_size + 1, step_size):
            end = start + window_size
            win_data   = data[start:end]
            win_labels = labels[start:end]
            true_label = 1 if np.any(win_labels == 1) else 0

            feat  = extract_features(win_data).reshape(1, -1)
            pred  = int(model.predict(feat)[0])

            t_start = f"{start/100:.1f}s"
            t_end   = f"{end/100:.1f}s"

            rows.append({
                "Subject":   subject,
                "File":      filename,
                "Window":    f"{t_start}–{t_end}",
                "True":      true_label,
                "Predicted": pred,
                "Correct":   "✅" if pred == true_label else "❌"
            })

            file_preds.append(pred)
            file_truths.append(true_label)
            all_y_true.append(true_label)
            all_y_pred.append(pred)

        # Tóm tắt cho file này
        if file_truths:
            fa = accuracy_score(file_truths, file_preds)
            has_fall = any(t == 1 for t in file_truths)
            detected = any(p == 1 for p in file_preds)
            file_summaries.append({
                "Subject":      subject,
                "File":         filename,
                "Windows":      len(file_truths),
                "Has_Fall":     "Có" if has_fall  else "Không",
                "Fall_Detected":"Có" if detected  else "Không",
                "Accuracy(%)":  round(fa * 100, 2),
                "Status":       "✅ Đúng" if (has_fall == detected) else "❌ Sai"
            })

    # ── 2. Chỉ số tổng thể ───────────────────────────────────────
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    overall_acc = accuracy_score(all_y_true, all_y_pred) * 100
    cm = confusion_matrix(all_y_true, all_y_pred)
    report = classification_report(all_y_true, all_y_pred,
                                   target_names=["Bình thường", "Té ngã"])

    # ── 3. In kết quả ra terminal ────────────────────────────────
    print("=" * 60)
    print("  KẾT QUẢ TEST TRÊN TẬP Sample_Test (Dữ liệu THỰC TẾ)")
    print("=" * 60)
    print(f"  Tổng số file CSV  : {len(file_summaries)}")
    print(f"  Tổng cửa sổ test  : {len(all_y_true)}")
    print(f"  Accuracy tổng thể : {overall_acc:.2f}%\n")
    print("  Ma trận nhầm lẫn:")
    print(f"                       Dự đoán Bình thường | Dự đoán Té ngã")
    print(f"  Thực tế Bình thường |       {cm[0][0]:<12}|    {cm[0][1]}")
    print(f"  Thực tế Té ngã      |       {cm[1][0]:<12}|    {cm[1][1]}\n")
    print("  Classification Report:")
    print(report)
    print("=" * 60)

    # ── 4. Xuất file kết quả ─────────────────────────────────────
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path  = os.path.join(output_dir, f"result_detail_{timestamp}.csv")
    summary_path = os.path.join(output_dir, f"result_summary_{timestamp}.csv")
    report_path  = os.path.join(output_dir, f"result_report_{timestamp}.txt")

    pd.DataFrame(rows).to_csv(detail_path,  index=False, encoding='utf-8-sig')
    pd.DataFrame(file_summaries).to_csv(summary_path, index=False, encoding='utf-8-sig')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("KẾT QUẢ TEST TRÊN TẬP Sample_Test (Dữ liệu THỰC TẾ)\n")
        f.write(f"Thời gian chạy  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model sử dụng   : {os.path.basename(model_path)}\n")
        f.write(f"Thư mục test    : {sample_test_dir}\n")
        f.write(f"Tổng số file    : {len(file_summaries)}\n")
        f.write(f"Tổng cửa sổ     : {len(all_y_true)}\n")
        f.write(f"Accuracy tổng   : {overall_acc:.2f}%\n\n")
        f.write("Ma trận nhầm lẫn:\n")
        f.write(f"  True Negative (BT→BT)  : {cm[0][0]}\n")
        f.write(f"  False Positive (BT→TN) : {cm[0][1]}\n")
        f.write(f"  False Negative (TN→BT) : {cm[1][0]}\n")
        f.write(f"  True Positive (TN→TN)  : {cm[1][1]}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(f"  📄 Chi tiết từng cửa sổ : {detail_path}")
    print(f"  📄 Tóm tắt từng file    : {summary_path}")
    print(f"  📄 Báo cáo tổng hợp     : {report_path}")

if __name__ == "__main__":
    # Dùng mô hình tối ưu (Opt-B: 10 cây, depth=8)
    MODEL_PATH       = "../models/random_forest_optimized.pkl"
    SAMPLE_TEST_DIR  = "../dataset/KFall/Sample_Test"

    run_test_on_sample_test(MODEL_PATH, SAMPLE_TEST_DIR)
