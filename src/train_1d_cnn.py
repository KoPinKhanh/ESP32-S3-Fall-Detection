import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

def build_model(input_shape):
    """
    Builds a lightweight 1D-CNN suitable for microcontrollers.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

def convert_to_tflite(model, X_train, save_path):
    """
    Converts a Keras model to TensorFlow Lite and applies INT8 quantization.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Representative dataset for INT8 quantization
    def representative_dataset_gen():
        for i in range(100):
            # Get a batch of data
            data = X_train[i:i+1].astype(np.float32)
            yield [data]
            
    converter.representative_dataset = representative_dataset_gen
    
    # Restrict ops to INT8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_quant_model = converter.convert()
    
    with open(save_path, 'wb') as f:
        f.write(tflite_quant_model)
    
    print(f"Quantized TFLite model saved to {save_path} (Size: {len(tflite_quant_model)} bytes)")

if __name__ == '__main__':
    PROCESSED_DIR = "../dataset/processed"
    MODELS_DIR = "../models"
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("Loading preprocessed data...")
    X = np.load(os.path.join(PROCESSED_DIR, "X.npy"))
    y = np.load(os.path.join(PROCESSED_DIR, "y.npy"))
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train shapes - X: {X_train.shape}, y: {y_train.shape}")
    
    # Compute class weights for imbalanced data
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight = {0: weights[0], 1: weights[1]}
    print(f"Class weights applied: {class_weight}")
    
    # Build and train model
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.summary()
    
    print("\nTraining the model...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=64,
        validation_split=0.2,
        class_weight=class_weight,
        verbose=1
    )
    
    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {acc*100:.2f}%")
    
    # Save the Keras model
    keras_path = os.path.join(MODELS_DIR, "fall_detect_model.keras")
    model.save(keras_path)
    print(f"Keras model saved to {keras_path}")
    
    # Convert to TFLite (Quantized)
    tflite_path = os.path.join(MODELS_DIR, "fall_detect_quantized.tflite")
    convert_to_tflite(model, X_train, tflite_path)
