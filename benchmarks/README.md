# Benchmarks

## What is in here

| File | Measured? | What it is |
| --- | --- | --- |
| `latency_host.json` | **Yes** | Single-window inference latency for all four exported artefacts, on the machine named inside the file. Single-threaded, batch size one. |
| `latency_targets.json` | **No** | Estimated inference latency on an STM32F411, an STM32H743 and a Jetson Nano, from arithmetic, plus the Renode figure for the STM32F411. RAM needed is taken from the emulated firmware build. Every row carries `"measured": false`. |
| `renode_cortex_m4.json` | **Emulated** | The int8 model in TensorFlow Lite Micro with CMSIS-NN, cross-built with the Arm GNU Toolchain and run under Renode on an emulated STM32F411. Cycle counts are emulator estimates; flash, RAM and arena figures are exact for the build; every output is compared with the desktop interpreter. Regenerated and checked by the `firmware` CI job. |
| `control_loop_db5.json` | **Yes** | End-to-end accuracy and latency of the streaming controller on the real recordings: causal filtering, int8 arithmetic and the voting rule from the config, whatever it is set to. |

## The honest position on target hardware

**No STM32 and no Jetson board was used for this work.** There are no
hardware latency numbers in this repository.

There are two kinds of estimate instead and both are labelled as such
wherever they appear.

**Arithmetic.** `src/embedded/estimate.py` computes an inference time from the
model's multiply-accumulate count and a documented per-board throughput
assumption. The assumptions are written down in that module, along with where
they come from and how wrong they could be. Treat them as an order of
magnitude.

**Emulation.** `firmware/` is a bare-metal Cortex-M4F program that embeds the
int8 model and runs it in TensorFlow Lite Micro with the CMSIS-NN kernels. It
is cross-compiled with the Arm GNU Toolchain and CMake, linked against the
STM32F411 memory map (512 KiB flash, 128 KiB RAM) and run under Renode on a
platform description with the same sizes (`firmware/renode/stm32f411.repl`).
For each of a fixed set of int8 windows it prints the raw int8 output and the
DWT cycle count of the `Invoke()` call. CI compares every output with what the
desktop TFLite interpreter computes for the same window, using TFLite's
reference kernels and fails on any difference.

What that does and does not establish:

* **The outputs are real.** The firmware computes the same int8 vector as the
  desktop interpreter, byte for byte. That is a functional check of the whole
  embedded path: the converter's operator set, TFLite Micro, CMSIS-NN and the
  compiler.
* **The memory figures are real.** Flash and RAM are what the linker placed
  and the arena is what the TFLite Micro allocator reported. The link itself
  fails if the image ever stops fitting an STM32F411.
* **The cycle count is an estimate.** Renode retires one instruction per
  cycle at the configured rate and does not model pipeline stalls,
  multi-cycle instructions, or flash wait states (an STM32F411 at 100 MHz runs
  its flash with three wait states, softened by the ART accelerator). A real
  board will take longer than the emulator says, by an amount only a board
  can tell you.
* **The weights are synthetic.** The firmware's model has the shipped
  architecture (same operators, tensor shapes, size and MAC count, which a
  test asserts) but was trained on the synthetic generator, so that the
  committed binary carries nothing derived from the research-use dataset.
  Memory and cycles depend on the architecture, not the weights; the
  bit-exactness check does not depend on whether the answer is right.

The measured host numbers are real, and they bound the problem usefully. The
int8 model is a couple of hundred kilobytes and takes a fraction of a
millisecond on a desktop CPU; the exact figures are in `latency_host.json` and
in the README's deployment table. See
[`docs/LATENCY_BUDGET.md`](../docs/LATENCY_BUDGET.md).

## Producing a real on-target measurement

The benchmark is deliberately portable: it takes an artefact path and writes
JSON. Nothing about it is specific to this machine.

### On a Linux-class board (Jetson Nano, Raspberry Pi, any ARM SBC)

```bash
git clone <this repository>
cd ml-semg-gesture-classifier
pip install numpy onnxruntime
# copy artifacts/deploy/db5/ across from the training machine, then:
python -m src.embedded.benchmark \
    --artefact artifacts/deploy/db5/gesture_cnn_int8.onnx \
    --threads 1 \
    --label jetson-nano
```

That writes `benchmarks/latency_jetson-nano.json` in the same format as
`latency_host.json`. Commit it, and the estimate for that board can be
deleted.

TensorFlow is not needed for the ONNX artefacts, so the install is small
enough for a board with limited storage.

### On a microcontroller (STM32)

There is no Python on a Cortex-M, so this is a porting job rather than a
command but most of the port already exists. `firmware/` builds and runs
under emulation today:

```bash
make firmware          # or, by hand:
docker build -f docker/firmware.Dockerfile -t semg-firmware docker
docker run --rm -v "$PWD:/work" -w /work semg-firmware bash firmware/scripts/build_and_run.sh
```

To move it to a real STM32F411 (a Nucleo-F411RE, say):

1. Add clock set-up (HSE and the PLL to 100 MHz, three flash wait states with
   the ART accelerator on) and enable the USART2 and GPIOA clocks in the RCC.
   The emulator does not model clock gating, so `firmware/src/board.c` skips
   all of it.
2. Route PA2 to USART2 TX (alternate function 7) and recompute `BRR` for the
   APB1 clock you chose.
3. Flash `build/firmware/semg_firmware.elf` with a debug probe and read the
   UART. The protocol is in `firmware/src/main.cc` and
   `firmware/tools/renode_report.py` parses a log from a board as happily as
   one from Renode. Label the result as measured.
4. To run the DB5-trained model instead of the synthetic-weight one, point
   CMake at it: `-DMODEL_TFLITE=artifacts/deploy/db5/gesture_cnn_int8.tflite`
   and regenerate the vectors from that model with
   `python -m src.embedded.firmware_vectors --model <path>`.
5. The input is the scaled, int8-quantised window. The per-channel scaler is
   `artifacts/models/db5/channel_scaler.npz` and the input scale and zero point
   are in `results/db5/export_report.json`; apply both on the device before
   the model, or fold the scaler into the input quantisation.

Write the result out in the same JSON shape and drop it in here, alongside
the emulated one rather than in place of it.

## Reading the numbers

Latency is reported as a distribution, not a mean. **Read the p99.** A control
loop is not judged on its average case; it is judged on whether it ever
stalls, and a mean hides exactly that.

Two separate constraints are reported, and both have to hold:

* `budget.meets_budget`: total wearer-perceived delay is inside 300 ms.
* `budget.keeps_up_with_stride`: one inference finishes inside one stride, so
  decisions do not queue.
