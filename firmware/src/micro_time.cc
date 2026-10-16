// TFLite Micro's timer hooks, on the DWT cycle counter (board.c).
//
// Replaces tensorflow/lite/micro/cortex_m_generic/micro_time.cc, which reads
// the same counter through a CMSIS device header the exported tree lacks.

#include "tensorflow/lite/micro/micro_time.h"

#include "board.h"

namespace tflite {

// Zero means "ticks are not wall-clock time", which under emulation is true.
uint32_t ticks_per_second() { return 0; }

uint32_t GetCurrentTimeTicks() { return board_cycles(); }

}  // namespace tflite
