import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

if __name__ == "__main__":
    PROCESSED_DIR = "../dataset/processed"
    MODELS_DIR = "../models"
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("Loading ML features data...")
    X = np.load(os.path.join(PROCESSED_DIR, "X_features.npy"))
    y = np.load(os.path.join(PROCESSED_DIR, "y.npy"))
    
    print(f"Data shape - X: {X.shape}, y: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    
    # Initialize Random Forest (using class_weight for imbalanced data)
    print("\nTraining Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=50, 
        max_depth=10, 
        class_weight='balanced', 
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating model on test set...")
    y_pred = rf_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc * 100:.2f}%\n")
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the scikit-learn model
    model_path = os.path.join(MODELS_DIR, "random_forest_model.pkl")
    joblib.dump(rf_model, model_path)
    print(f"\nModel saved to {model_path}")
