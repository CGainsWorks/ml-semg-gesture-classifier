# ml-semg-gesture-classifier

> **Status: planned. This repository is a brief, not an implementation.**
> Nothing here is built yet. The sections below define what goes in it.

Real-time gesture classification from surface electromyography, driving a six-axis prosthetic arm. Built on the public Ninapro dataset rather than personal subject recordings, covering the full path from raw signal to a quantised model running on embedded hardware within a stated latency budget.

---

## What this covers

CV entries this repository is the evidence for:

- sEMG gesture classifier (NeuroTech Biowares)

---

## Requirements

### Hardware

- Training PC with a GPU
- Target embedded board for inference (STM32, Jetson Nano or similar)
- Optional: sEMG acquisition board and electrodes for live capture

### Software

- TensorFlow and PyTorch
- Signal processing: filtering, windowing, feature extraction
- Imitation learning and learning-by-demonstration training loop
- Quantised embedded inference runtime (TFLite Micro or ONNX Runtime)
- Ninapro or another public sEMG dataset

### Report and documentation deliverables

- Dataset card
- Preprocessing description
- Confusion matrix
- Per-class precision and recall
- Inference latency measurements on target hardware
- Subject-independent generalisation results

---

## Inputs

- Multi-channel raw sEMG at a stated sample rate
- Gesture labels
- Demonstration recordings
- Subject metadata

## Outputs

- Trained model weights
- Quantised deployment artefact
- Gesture predictions with confidence
- Evaluation report
- Latency benchmark

---

## Intended structure

```
src/
  data/                   Ninapro loaders, windowing, splits
  preprocessing/          filtering, rectification, feature extraction
  models/                 TF and PyTorch implementations
  train.py
  evaluate.py
  quantise/               TFLite / ONNX export
  embedded/               target inference runtime
data/                     DATASET_CARD.md
benchmarks/               latency on target hardware
analysis/                 confusion matrices, subject-independent eval
docs/                     MODEL_CARD.md
```

Plus the standard baseline: `README.md`, `LICENSE`, `.gitignore`, `.gitattributes`
(Git LFS rules for weights, bags and media), `.github/workflows/ci.yml`, `docs/`,
`scripts/setup.sh`, `scripts/run_demo.sh` and `tests/`.

---

## Build checklist

- [ ] Public sEMG dataset loader with a dataset card
- [ ] Filtering and windowing pipeline
- [ ] Classical feature extraction baseline
- [ ] TensorFlow model with reported accuracy
- [ ] PyTorch equivalent for comparison
- [ ] Subject-independent evaluation split
- [ ] Confusion matrix and per-class analysis
- [ ] Quantisation and export
- [ ] Embedded inference with measured latency
- [ ] Latency budget against the control loop requirement
- [ ] Model card

---

## Definition of done

This repository is finished when:

- [ ] The whole thing runs from one command on a clean machine.
- [ ] The README opens with a diagram or a GIF, not a wall of text.
- [ ] Every number quoted in the README is reproducible from a script in the repo.
- [ ] Limitations are stated honestly in their own section.
- [ ] A licence is declared.
- [ ] Large binaries live in Releases or LFS, not in the Git history.
- [ ] The repository URL is added to the matching CV bullet point.

---

## Starter prompt

Paste this into your coding agent of choice to begin.

<details>
<summary>Expand</summary>

```
You are helping me build a portfolio project from scratch. Read this brief and then
propose a concrete implementation plan before writing any code.

PROJECT: ml-semg-gesture-classifier
Real-time gesture classification from surface electromyography, driving a six-axis prosthetic arm. Built on the public Ninapro dataset rather than personal subject recordings, covering the full path from raw signal to a quantised model running on embedded hardware within a stated latency budget.

TARGET STACK
- TensorFlow and PyTorch
- Signal processing: filtering, windowing, feature extraction
- Imitation learning and learning-by-demonstration training loop
- Quantised embedded inference runtime (TFLite Micro or ONNX Runtime)
- Ninapro or another public sEMG dataset

HARDWARE CONTEXT
- Training PC with a GPU
- Target embedded board for inference (STM32, Jetson Nano or similar)
- Optional: sEMG acquisition board and electrodes for live capture

THE SYSTEM MUST TAKE THESE INPUTS
- Multi-channel raw sEMG at a stated sample rate
- Gesture labels
- Demonstration recordings
- Subject metadata

AND PRODUCE THESE OUTPUTS
- Trained model weights
- Quantised deployment artefact
- Gesture predictions with confidence
- Evaluation report
- Latency benchmark

INTENDED REPOSITORY LAYOUT
src/
  data/                   Ninapro loaders, windowing, splits
  preprocessing/          filtering, rectification, feature extraction
  models/                 TF and PyTorch implementations
  train.py
  evaluate.py
  quantise/               TFLite / ONNX export
  embedded/               target inference runtime
data/                     DATASET_CARD.md
benchmarks/               latency on target hardware
analysis/                 confusion matrices, subject-independent eval
docs/                     MODEL_CARD.md

CONSTRAINTS
- Everything must run from a single command: `docker compose up` or `make demo`.
- No placeholder or stub code presented as working. If something is not implemented, say so.
- Every quantitative claim in the README must be reproducible from a script in this repo.
- Use only public or synthetic data. Record the licence of every dataset used.


START BY
1. Confirming or challenging the proposed repository layout.
2. Listing the specific libraries and versions you intend to pin.
3. Identifying the single riskiest technical unknown and how to de-risk it first.
Then work through the milestones in the README build checklist, one at a time.

```

</details>

---

## Notes

- Do not backdate commits to match CV dates. Publishing older work today is normal and expected.
- Record dataset licences before training on anything. Several common research datasets are
  non-commercial only.
- Flip this repository to public once there is something in it worth showing.
