# Benchmarks

## What is in here

| File | Measured? | What it is |
| --- | --- | --- |
| `latency_host.json` | **Yes** | Single-window inference latency for all four exported artefacts, on the machine named inside the file. Single-threaded, batch size one. |
| `latency_targets.json` | **No** | Estimated inference latency on an STM32F411, an STM32H743 and a Jetson Nano. Every row carries `"measured": false`. |
| `control_loop_db5.json` | **Yes** | End-to-end accuracy and latency of the streaming controller on the real recordings: causal filtering, int8 arithmetic and the voting rule from the config, whatever it is set to. |

## The honest position on target hardware

**No STM32 and no Jetson was available for this work.** The brief asks for
inference latency on target hardware and this repository does not have it.

Rather than omit the section or quote a number that reads like a measurement,
`src/embedded/estimate.py` computes an estimate from the model's
multiply-accumulate count and a documented per-board throughput assumption,
and every consumer of it labels the result as an estimate. The assumptions are
written down in that module, along with where they come from and how wrong
they could be. Treat them as an order of magnitude.

The measured host numbers are real, and they bound the problem usefully. The
int8 model is a couple of hundred kilobytes and takes tens of microseconds on
a desktop CPU; the exact figures are in `latency_host.json` and in the
README's deployment table. Scaling that to a 480 MHz Cortex-M7 by clock ratio
and instruction width is not rigorous, but it is enough to say with confidence
that inference is not what threatens the 300 ms budget. Filling the window is.
See [`docs/LATENCY_BUDGET.md`](../docs/LATENCY_BUDGET.md).

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
command. What the repository gives you:

* `artifacts/deploy/db5/gesture_cnn_int8.tflite`, fully integer-quantised with
  int8 input and output tensors. It will build for TFLite Micro without a
  float fallback; the export refuses to emit anything else, and a test asserts
  it.
* `artifacts/models/db5/channel_scaler.npz`, the per-channel mean and scale
  the model expects its input to have been normalised by. Sixteen floats each.
  Apply them on the device before the model, or fold them into the input
  quantisation parameters.
* `artifacts/deploy/db5/export_report.json`, which records the int8 input
  quantisation scale and zero point so the fixed-point conversion on the
  device matches what the model was calibrated for.

Rough shape of the port:

1. `xxd -i gesture_cnn_int8.tflite > model_data.cc` to get the weights as a C
   array.
2. Build TFLite Micro with CMSIS-NN enabled. Without it the integer
   convolutions fall back to reference kernels and run roughly an order of
   magnitude slower.
3. Register the operators the model uses. `export_report.json` records them
   under `artefacts.tflite_int8.operators`, and there are only four:
   `CONV_2D`, `MAX_POOL_2D`, `FULLY_CONNECTED` and `RESHAPE`. That short list
   is not an accident. Exporting with a dynamic batch dimension adds `SHAPE`,
   `STRIDED_SLICE` and `PACK` so the reshape can compute its target at run
   time, which is three more operators to register and defeats static memory
   planning; the export pins the signature to a batch of one, and a test
   asserts the shape operators stay out.
4. Size the arena. `export_report.json` records a floor from the sum of the
   intermediate tensor sizes; the real figure is smaller because the allocator
   reuses buffers, and the only way to know it is to build and read what the
   allocator reports.
5. Time it with DWT cycle counting rather than a millisecond tick. The
   inference is short enough that a 1 kHz tick would quantise the measurement
   into uselessness.

Write the result out in the same JSON shape and drop it in here.

## Reading the numbers

Latency is reported as a distribution, not a mean. **Read the p99.** A control
loop is not judged on its average case; it is judged on whether it ever
stalls, and a mean hides exactly that.

Two separate constraints are reported, and both have to hold:

* `budget.meets_budget`: total wearer-perceived delay is inside 300 ms.
* `budget.keeps_up_with_stride`: one inference finishes inside one stride, so
  decisions do not queue.
