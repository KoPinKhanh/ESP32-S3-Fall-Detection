import os
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_full_dataset(model_path, features_path, labels_path):
    print(f"1. Đang nạp mô hình từ: {model_path}")
    model = joblib.load(model_path)
    
    print(f"2. Đang nạp toàn bộ dữ liệu từ: {features_path}")
    X = np.load(features_path)
    y = np.load(labels_path)
    
    print(f"-> Tổng số lượng mẫu dữ liệu: {len(X)} cửa sổ thời gian.")
    
    print("3. AI đang tiến hành dự đoán trên toàn bộ tập dữ liệu...")
    y_pred = model.predict(X)
    
    print("\n" + "="*50)
    print("KẾT QUẢ ĐÁNH GIÁ TRÊN TOÀN BỘ DATASET")
    print("="*50)
    
    acc = accuracy_score(y, y_pred)
    print(f"Độ chính xác tổng thể (Accuracy): {acc * 100:.2f}%\n")
    
    print("Ma trận nhầm lẫn (Confusion Matrix):")
    print("                     Dự đoán: Bình thường | Dự đoán: Té Ngã")
    cm = confusion_matrix(y, y_pred)
    print(f"Thực tế: Bình thường |        {cm[0][0]:<11} |      {cm[0][1]}")
    print(f"Thực tế: Té Ngã      |        {cm[1][0]:<11} |      {cm[1][1]}\n")
    
    print("Báo cáo phân loại chi tiết (Classification Report):")
    print(classification_report(y, y_pred, target_names=["Bình thường (0)", "Té ngã (1)"]))

if __name__ == "__main__":
    MODEL_PATH = "../models/random_forest_model.pkl"
    FEATURES_PATH = "../dataset/processed/X_features.npy"
    LABELS_PATH = "../dataset/processed/y.npy"
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        print("Lỗi: Không tìm thấy model hoặc dữ liệu. Vui lòng kiểm tra lại!")
    else:
        evaluate_full_dataset(MODEL_PATH, FEATURES_PATH, LABELS_PATH)
