#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Include the generated model (from xxd)
#include "fall_detect_model.h" 

// Globals
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

// Adjust this depending on your model size
constexpr int kTensorArenaSize = 30 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// For sliding window buffer (200 samples x 6 features)
const int WINDOW_SIZE = 200;
const int NUM_FEATURES = 6;
float sensor_buffer[WINDOW_SIZE][NUM_FEATURES];

void setup() {
  Serial.begin(115200);
  
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure
  model = tflite::GetModel(fall_detect_quantized_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in all the TFLM ops
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  // Initialize Sensor (MPU6050/BNO085) here...
}

void loop() {
  // 1. Read Sensor Data (AccX, AccY, AccZ, GyrX, GyrY, GyrZ)
  // Shift buffer and add new sample...
  
  // 2. Populate input tensor (Note: We used INT8 quantization)
  // You may need to quantize your float values to INT8 using input->params.scale and zero_point
  for (int i = 0; i < WINDOW_SIZE; i++) {
    for (int j = 0; j < NUM_FEATURES; j++) {
      float val = sensor_buffer[i][j];
      // int8_t quantized_val = val / input->params.scale + input->params.zero_point;
      // input->data.int8[i * NUM_FEATURES + j] = quantized_val;
    }
  }

  // 3. Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
  }

  // 4. Read the output
  int8_t y_quantized = output->data.int8[0];
  // Dequantize back to float probability
  // float y_pred = (y_quantized - output->params.zero_point) * output->params.scale;
  
  // if (y_pred > 0.5) {
  //    Serial.println("FALL DETECTED!!!");
  // }
  
  delay(10);
}
