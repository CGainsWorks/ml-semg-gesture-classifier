# How the code is put together

## The shape of it

```
src/
  config.py              one dataclass tree, loaded from config/*.yaml
  data/
    ninapro.py           read DB5 .mat files out of the distributed zips
    download.py          fetch and checksum the archives
    synthetic.py         surrogate generator for CI and tests
    windows.py           sliding windows, as an index rather than an array
    splits.py            the three evaluation protocols
    dataset.py           assemble and cache
    gestures.py          restimulus code to human-readable name
    build_report.py      regenerate the dataset card's numbers
  preprocessing/
    filters.py           Butterworth bandpass, zero-phase and causal
    normalise.py         per-channel scaler with an explicit fit
  features/
    time_domain.py       Hudgins-style features, vectorised over all windows
  models/
    architecture.py      the shape, described once for both frameworks
    augment.py           training-time perturbation, shared by both
    torch_cnn.py         PyTorch implementation
    tf_cnn.py            TensorFlow implementation
    classical.py         features plus LDA, SVM or random forest
  quantise/
    tflite_export.py     TFLite float32 and full-integer int8
    onnx_export.py       ONNX float32 and static int8
    export.py            produce every artefact and measure what int8 cost
  embedded/
    runtime.py           one interface over all four artefacts
    benchmark.py         single-window latency, as a distribution
    estimate.py          on-target estimates, labelled as estimates
    control_loop.py      the streaming controller, causal filter and all
  eval/
    metrics.py           scoring, including the two activation error rates
    plots.py             figures
    make_figures.py      the signal figure the README opens with
  embedded/
    estimate_report.py   the on-target table, every row labelled an estimate
  experiment.py          the fold loop, shared by every model and protocol
  train.py               CLI
  calibrate.py           fit the model to one wearer from their demonstrations
  evaluate.py            results to figures, tables and a report
  check_readme.py        the README's numbers still match the JSON

experiments/             ablations whose numbers the docs cite
config/                  default.yaml is the shipped run, ci.yaml is the smoke run
results/                 committed evidence: one JSON per scored run
benchmarks/              committed evidence: latency, measured and estimated
analysis/db5/            committed evidence: figures and the evaluation report
artifacts/               regenerated, never committed: weights, caches, binaries
```

## Decisions worth explaining

### Windows are an index, not an array

`WindowIndex` holds the filtered continuous signal plus the start offsets, and
`materialise` builds the tensor with `sliding_window_view`, which is a view
rather than a copy. Ten subjects of exercise B are about 75,000 windows at the
default geometry; materialising every subject up front would cost roughly half
a gigabyte to hold data that is three-quarters duplicated.

`concatenate` joins per-subject indices into one shared signal buffer with a
window of zeros between subjects, so no window can span two people. There is a
test for that, because the failure would be silent and would corrupt every
subject-independent number.

### The architecture is described once

`ArchitectureSpec` is a frozen dataclass that both `torch_cnn` and `tf_cnn`
read. It also derives the parameter count and the MAC count arithmetically,
and tests assert both frameworks against the derived figure. The point of a
TensorFlow-versus-PyTorch comparison is that the two are the same model, and
the only way to keep that true across edits is to stop writing the numbers
down twice.

### The input layout is the same everywhere

Everything produces `(batch, electrodes, time)`. Keras wants channels last for
`Conv2D`, so the reshape lives *inside* the Keras graph rather than in the
caller. The exported TFLite model and the exported ONNX model therefore take
identical input, which means the embedded runtime wrapper does not need to
know which one it loaded.

### Augmentation is NumPy, called from both training loops

`src/models/augment.py` is framework-agnostic and both loops call it, Keras
through a `PyDataset`. Using each framework's own preprocessing layers would
have been tidier and would have quietly meant the two models were trained on
different data.

### The classical baseline scales itself

`ClassicalPipeline` fits its own per-channel sigmas because three of its
features need an amplitude threshold, and a threshold fixed in signal units
cannot transfer between people whose electrode gain differs by a factor of
two. It is handed unscaled windows for that reason, while the networks get the
`ChannelScaler`-transformed ones. Both fit their statistics on the training
fold only.

### Zero-phase offline, causal on the device

Offline filtering uses `sosfiltfilt`, which runs the filter both ways and so
uses future samples. That is legitimate for an offline result and impossible
in a controller. `src/embedded/control_loop.py` re-runs the whole pipeline
through a one-directional filter with state carried across blocks and reports
what changes, rather than leaving the reader to wonder. The state-carrying is
tested by streaming the same signal at two different block sizes and asserting
the decisions match.

### Evidence is committed, intermediates are not

`artifacts/` holds trained weights, the window cache and the exported
binaries. All of it is regenerated by `make demo` and none of it is tracked.

`results/` holds the scored runs, `benchmarks/` holds the latency
measurements, `experiments/results/` holds the ablations and
`data/dataset_stats.json` holds the dataset counts. All of that is tracked,
because `python -m src.check_readme` verifies the README against those files
and CI runs it on a fresh checkout, where `artifacts/` does not exist.

### Every artefact directory is named for its data source

Results from the synthetic generator land under `.../synthetic/` and results
from Ninapro land under `.../db5/`. Reports carry a `synthetic` flag and the
smoke script prints a banner. It is not possible to confuse a wiring-check
number for a result by looking at a file path.

## Adding things

**Another dataset.** Implement a loader returning a `Recording`, add a branch
to `load_raw_recording`, and add a dataset card. Everything downstream is
agnostic. DB2 (12 channels at 2 kHz) needs `bandpass_high_hz` raised and the
mains notch turned on.

**Another model.** Add it to `MODELS` in `src/experiment.py` and dispatch it in
`run_protocol`. The fold loop, the metrics and the reporting are shared;
`tests/test_pipeline.py::test_every_model_name_is_runnable` will fail if a name
is added without a dispatch.

**Another protocol.** Add a builder to `src/data/splits.py`, list it in
`PROTOCOLS`, and dispatch it in `make_folds`.

**Another deployment target.** Add a `Target` to `src/embedded/estimate.py`
for the estimate, or better, run the benchmark on the board and drop the JSON
in `benchmarks/`. A measurement replaces an estimate.
