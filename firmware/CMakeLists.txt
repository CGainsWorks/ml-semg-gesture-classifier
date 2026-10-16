# Gesture classifier firmware: the int8 TFLite model, TensorFlow Lite Micro
# with the CMSIS-NN kernels and a bare-metal harness for a Cortex-M4F with the
# STM32F411 memory map (512 KiB flash, 128 KiB RAM).
#
# The memory map is enforced by the linker script, so if the model, the arena
# and the stack stop fitting in an STM32F411 the link fails here rather than on
# a board.
#
#   cmake -S firmware -B build/firmware -G Ninja \
#         -DCMAKE_TOOLCHAIN_FILE=firmware/cmake/arm-none-eabi.cmake \
#         -DTFLM_TREE=/opt/tflm
#   cmake --build build/firmware

cmake_minimum_required(VERSION 3.22)
project(semg_firmware C CXX ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(TFLM_TREE "$ENV{TFLM_TREE}" CACHE PATH "TFLite Micro source tree from create_tflm_tree.py")
if(NOT EXISTS "${TFLM_TREE}/tensorflow/lite/micro/micro_interpreter.h")
  message(FATAL_ERROR "TFLM_TREE does not point at a TFLite Micro tree: '${TFLM_TREE}'")
endif()

set(MODEL_TFLITE "${CMAKE_CURRENT_SOURCE_DIR}/model/gesture_cnn_int8.tflite"
    CACHE FILEPATH "int8 TFLite model to embed")
set(WINDOWS_BIN "${CMAKE_CURRENT_SOURCE_DIR}/vectors/windows_int8.bin"
    CACHE FILEPATH "Fixed int8 input windows, back to back")
# Sized from what the interpreter reports it uses, plus headroom. The firmware
# prints the figure it actually used on every run.
set(ARENA_BYTES 40960 CACHE STRING "TFLite Micro tensor arena, bytes")

# ---------------------------------------------------------------------------
# TensorFlow Lite Micro, compiled from the exported tree.
# ---------------------------------------------------------------------------
file(GLOB_RECURSE TFLM_SOURCES CONFIGURE_DEPENDS
  "${TFLM_TREE}/tensorflow/*.cc"
  "${TFLM_TREE}/tensorflow/*.c"
  "${TFLM_TREE}/third_party/cmsis_nn/*.c")
# The generic Cortex-M timer reads the DWT through a CMSIS device header that
# the exported tree does not carry. src/micro_time.cc provides the same two
# functions on the same counter.
list(FILTER TFLM_SOURCES EXCLUDE REGEX "cortex_m_generic/micro_time[.]cc$")

add_library(tflm STATIC ${TFLM_SOURCES})
target_include_directories(tflm SYSTEM PUBLIC
  "${TFLM_TREE}"
  "${TFLM_TREE}/third_party/flatbuffers/include"
  "${TFLM_TREE}/third_party/gemmlowp"
  "${TFLM_TREE}/third_party/ruy"
  "${TFLM_TREE}/third_party/cmsis_nn"
  "${TFLM_TREE}/third_party/cmsis_nn/Include"
  "${TFLM_TREE}/third_party/cmsis/CMSIS/Core/Include")
target_compile_definitions(tflm PUBLIC
  TF_LITE_STATIC_MEMORY
  TF_LITE_DISABLE_X86_NEON
  TF_LITE_MCU_DEBUG_LOG
  CMSIS_NN
  KERNELS_OPTIMIZED_FOR_SPEED
  CPU_M4=1)
target_compile_options(tflm PRIVATE
  -O2 -fno-unwind-tables -ffunction-sections -fdata-sections -fsigned-char
  -Wno-type-limits -Wno-unused-private-field -Wno-unused-parameter
  $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti -fno-exceptions -fno-threadsafe-statics>)

# ---------------------------------------------------------------------------
# The firmware.
# ---------------------------------------------------------------------------
add_executable(semg_firmware.elf
  src/main.cc
  src/startup.c
  src/board.c
  src/micro_time.cc
  src/blobs.S)
target_link_libraries(semg_firmware.elf PRIVATE tflm)
target_compile_definitions(semg_firmware.elf PRIVATE
  ARENA_BYTES=${ARENA_BYTES}
  MODEL_TFLITE_PATH="${MODEL_TFLITE}"
  WINDOWS_BIN_PATH="${WINDOWS_BIN}")
target_compile_options(semg_firmware.elf PRIVATE
  -O2 -ffunction-sections -fdata-sections -Wall -Wextra
  $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti -fno-exceptions -fno-threadsafe-statics>)
target_link_options(semg_firmware.elf PRIVATE
  -T${CMAKE_CURRENT_SOURCE_DIR}/src/stm32f411.ld
  -Wl,--gc-sections
  -Wl,--print-memory-usage
  -Wl,-Map=${CMAKE_CURRENT_BINARY_DIR}/semg_firmware.map)
set_property(SOURCE src/blobs.S APPEND PROPERTY OBJECT_DEPENDS
  "${MODEL_TFLITE}" "${WINDOWS_BIN}")
set_property(TARGET semg_firmware.elf APPEND PROPERTY LINK_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/stm32f411.ld")

add_custom_command(TARGET semg_firmware.elf POST_BUILD
  COMMAND ${CMAKE_SIZE} -A -d $<TARGET_FILE:semg_firmware.elf>
          > ${CMAKE_CURRENT_BINARY_DIR}/semg_firmware.size.txt
  COMMAND ${CMAKE_SIZE} -B -d $<TARGET_FILE:semg_firmware.elf>
  VERBATIM)
