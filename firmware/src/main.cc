// Gesture classifier on a Cortex-M4F: TensorFlow Lite Micro with CMSIS-NN.
//
// Runs the embedded int8 model over a fixed set of int8 windows and prints,
// for each one, the raw int8 output vector and the DWT cycle count of the
// Invoke() call. firmware/tools/compare.py reads that log and checks every
// output against the desktop TFLite interpreter's answer for the same window.
//
// Output protocol, one record per line, all integers in decimal:
//
//   SEMG BEGIN
//   SEMG MODEL bytes=<n> windows=<n> input=<n> output=<n>
//   SEMG ARENA size=<n> used=<n>
//   SEMG W <index> cycles=<n> top=<class> out=<v0>,<v1>,...
//   SEMG END status=ok
//
// Under Renode the cycle count is an emulator estimate (one cycle per executed
// instruction), not a hardware measurement.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "board.h"
#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern "C" const uint8_t g_model[];
extern "C" const uint8_t g_model_end[];
extern "C" const int8_t g_windows[];
extern "C" const int8_t g_windows_end[];

namespace {

alignas(16) uint8_t g_arena[ARENA_BYTES];

char g_line[512];

void Emit(const char* text) { board_uart_write(text); }

void DebugLog(const char* text) {
  board_uart_write("SEMG LOG ");
  board_uart_write(text);
}

[[noreturn]] void Fail(const char* reason) {
  std::snprintf(g_line, sizeof(g_line), "SEMG END status=fail reason=%s\n", reason);
  Emit(g_line);
  for (;;) {
    __asm volatile("wfi");
  }
}

}  // namespace

int main() {
  board_uart_init();
  board_cycle_counter_init();
  RegisterDebugLogCallback(DebugLog);

  Emit("SEMG BEGIN\n");

  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Fail("schema-version");
  }

  // Exactly the four operators the exporter allows (src/quantise/tflite_export.py).
  tflite::MicroMutableOpResolver<4> resolver;
  if (resolver.AddConv2D() != kTfLiteOk || resolver.AddMaxPool2D() != kTfLiteOk ||
      resolver.AddFullyConnected() != kTfLiteOk || resolver.AddReshape() != kTfLiteOk) {
    Fail("resolver");
  }

  tflite::MicroInterpreter interpreter(model, resolver, g_arena, sizeof(g_arena));
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    Fail("allocate-tensors");
  }

  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8) {
    Fail("not-int8");
  }

  const size_t input_bytes = input->bytes;
  const size_t output_bytes = output->bytes;
  const size_t windows_bytes = static_cast<size_t>(g_windows_end - g_windows);
  if (input_bytes == 0 || windows_bytes % input_bytes != 0) {
    Fail("window-size");
  }
  const size_t n_windows = windows_bytes / input_bytes;

  std::snprintf(g_line, sizeof(g_line), "SEMG MODEL bytes=%u windows=%u input=%u output=%u\n",
                static_cast<unsigned>(g_model_end - g_model), static_cast<unsigned>(n_windows),
                static_cast<unsigned>(input_bytes), static_cast<unsigned>(output_bytes));
  Emit(g_line);
  std::snprintf(g_line, sizeof(g_line), "SEMG ARENA size=%u used=%u\n",
                static_cast<unsigned>(sizeof(g_arena)),
                static_cast<unsigned>(interpreter.arena_used_bytes()));
  Emit(g_line);

  for (size_t w = 0; w < n_windows; ++w) {
    std::memcpy(input->data.int8, g_windows + w * input_bytes, input_bytes);

    const uint32_t start = board_cycles();
    const TfLiteStatus status = interpreter.Invoke();
    const uint32_t cycles = board_cycles() - start;
    if (status != kTfLiteOk) {
      Fail("invoke");
    }

    const int8_t* out = output->data.int8;
    size_t top = 0;
    for (size_t k = 1; k < output_bytes; ++k) {
      if (out[k] > out[top]) {
        top = k;
      }
    }

    int n = std::snprintf(g_line, sizeof(g_line), "SEMG W %u cycles=%lu top=%u out=",
                          static_cast<unsigned>(w), static_cast<unsigned long>(cycles),
                          static_cast<unsigned>(top));
    for (size_t k = 0; k < output_bytes && n < static_cast<int>(sizeof(g_line)) - 8; ++k) {
      n += std::snprintf(g_line + n, sizeof(g_line) - n, k == 0 ? "%d" : ",%d",
                         static_cast<int>(out[k]));
    }
    std::snprintf(g_line + n, sizeof(g_line) - n, "\n");
    Emit(g_line);
  }

  Emit("SEMG END status=ok\n");
  for (;;) {
    __asm volatile("wfi");
  }
}
