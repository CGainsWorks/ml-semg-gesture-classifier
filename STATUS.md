# Status

Last updated 16 October 2026.

## Complete

* **Pipeline on Ninapro DB5.** Filtering, windowing, time-domain features,
  LDA, SVM and random forest baselines and twin PyTorch and TensorFlow CNNs
  from one architecture spec, evaluated under three protocols (within-subject,
  leave-one-subject-out and leave-one-subject-out with calibration). Every
  result is committed under `results/db5/` and every README table is
  regenerated from it and checked in CI.
* **Quantised deployment.** int8 TFLite (fully integer, four operators) and
  int8 ONNX, scored through the deployed runtimes. Host latency measured on
  the development machine.
* **Firmware under emulation.** `firmware/` builds the int8 model into a
  bare-metal Cortex-M4F program with TensorFlow Lite Micro and CMSIS-NN, using
  the Arm GNU Toolchain 14.2.Rel1 and CMake, linked against the STM32F411
  memory map. CI runs it under Renode 1.16.1 on every push and compares all 36
  int8 outputs with the desktop TFLite interpreter (reference kernels): they
  are bit-exact. The emulator's cycle count, the flash and RAM use and the
  arena size are in `benchmarks/renode_cortex_m4.json` and CI checks that the
  committed copy is exactly what the run produces.
* **The STM32F411 memory question is settled.** The build uses 52.9 KiB of
  the part's 128 KiB of RAM (29.7 KiB of it the tensor arena) and 287 KiB of
  its 512 KiB of flash. The earlier 217 KiB "arena upper bound" was the sum of
  every tensor including the weights; it is now reported under that name and
  no longer used as a RAM figure.
* **The STM32F411 latency question is answered, as an estimate.** The
  emulator counts 15.8 million instructions per inference, 157.7 ms at
  100 MHz, which misses the 50 ms stride and the 300 ms budget. On a
  single-issue Cortex-M4 that is close to a floor. The recommended part is the
  STM32H743, on arithmetic only.
* **Architecture prose generated from data.** The paragraphs under the
  architecture table (1-D against 2-D, augmentation, wrapped electrode axis,
  electrode rotation) are generated from
  `experiments/results/db5/architecture_comparison.json`, comparative wording
  included and CI fails if they drift. They now say what the data says: the
  2-D design wins within subject and the 1-D design wins on the single
  cross-subject fold.
* **Link to the consumer.** The README links
  [emb-neurotech-prosthetic-arm](https://github.com/CGainsWorks/emb-neurotech-prosthetic-arm),
  which consumes these decisions as `arm_msgs/GestureCommand`.
* **Tests.** 201 tests, run on Python 3.10, 3.11 and 3.12 in CI and inside
  the container. Ruff on `src`, `tests`, `experiments` and `firmware`.

## In progress

Nothing.

## Not started

* **A `GestureCommand` publisher.** The streaming controller produces the
  class, the confidence and the inference time; there is no validity flag and
  nothing publishes the message.
* **An eight-channel model.** The arm repository's board has eight channels
  and this model takes sixteen. That needs a retrain on a channel subset.
* **Cortex-M7 emulation.** The STM32H743 figure is arithmetic. Renode ships an
  STM32H743 platform, so the same firmware could be built for a Cortex-M7 and
  counted the same way.
* **Ninapro DB3 or DB7.** Amputee subjects. Nothing here has been run on
  them.
* **Ten-fold architecture ablation.** The architecture comparison is one fold
  per protocol and every difference in it is inside the fold-to-fold spread.

## Limitations

* **No hardware.** No STM32, Jetson, acquisition board or arm was used. The
  20 ms acquisition and actuation allowance in the latency budget is an
  assumption.
* **Emulator timing is an estimate.** Renode retires one instruction per
  cycle and does not model pipeline stalls, multi-cycle instructions or flash
  wait states. A real STM32F411 will be slower than the emulated figure, by an
  amount only a board can measure. The outputs and the memory figures from the
  emulated build are exact.
* **The firmware's weights are synthetic.** The firmware model has the shipped
  architecture but was trained on the synthetic generator, so the committed
  binary carries nothing derived from the research-use dataset. Memory and
  cycles depend on the architecture; bit-exactness does not depend on the
  answer being right. `benchmarks/README.md` says how to build it with the
  DB5-trained model instead.
* **Cross-subject accuracy is not good enough to ship.** About 0.32 balanced
  accuracy on 18 classes without calibration, 0.47 with it.
* **Ten intact-limbed subjects, one session each.** No amputees, no donning
  variation, no fatigue.
