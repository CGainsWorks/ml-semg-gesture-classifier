# Dataset card: Ninapro DB5

## In one line

Sixteen channels of surface electromyography from two Myo armbands, sampled at
200 Hz, recorded from ten intact-limbed adults performing 52 hand and wrist
movements, six repetitions each.

## Provenance

| | |
| --- | --- |
| Name | Ninapro DB5 (Non-Invasive Adaptive Prosthetics database 5) |
| Publisher | Ninapro project, HES-SO Valais / University of Padova |
| Home page | https://ninapro.hevs.ch/instructions/DB5.html |
| Reference | Pizzolato et al., "Comparison of six electromyography acquisition setups on hand movement classification tasks", *PLOS ONE* 12(10), 2017 |
| Licence | Research use, with attribution. See "Licence" below. |
| Downloaded | 21 August 2026 |
| Size | 10 archives, roughly 200 MB total |
| Retrieval | `python -m src.data.download`, or `make data` |
| Integrity | sha256 of every archive in [`DB5_CHECKSUMS.txt`](DB5_CHECKSUMS.txt) |

This repository does **not** redistribute the data. It downloads it from the
publisher, checksums it, and leaves the archives sealed on disk. Nothing under
`data/raw/` is committed.

## Why this database and not another

Ninapro has ten databases. DB5 was chosen over the more commonly used DB1 and
DB2 for reasons that are specific to a prosthetic control project:

* **The sensor is the one a prosthesis would use.** DB5 uses two Thalmic Myo
  armbands: consumer-grade, dry-electrode, wireless, and cheap. DB2 uses a
  Delsys Trigno system that costs more than the prosthesis. A result on DB5
  transfers to a device somebody could actually wear.
* **DB1 is not raw sEMG.** DB1 records the output of Otto Bock amplifiers,
  which is an already-rectified, already-smoothed envelope at 100 Hz. A
  filtering and windowing pipeline has nothing to do on it.
* **It is small enough to work with honestly.** All ten subjects come to
  200 MB, so the leave-one-subject-out evaluation runs over every subject
  rather than a convenient subset. DB2 is 470 MB *per subject*.

The trade-off is real and should be stated: at 200 Hz the usable sEMG band is
capped at 100 Hz, and sEMG energy extends well past that. DB5 accuracies in
the literature run below DB2 accuracies on the same movements for exactly this
reason. The numbers in this repository are therefore not comparable to
DB2-based results, and they are not intended to be.

## What is in a file

One `.mat` file per subject per exercise, MATLAB v5 format, inside one zip per
subject.

| Field | Shape | Meaning |
| --- | --- | --- |
| `emg` | (n, 16) | Raw signal, both armbands. Signed 8-bit counts stored as float. |
| `acc` | (n, 3) | Accelerometer. Not used here. |
| `glove` | (n, 22) | Cyberglove joint angles. Not used here. |
| `stimulus` | (n, 1) | The movement the subject was asked to make. |
| `restimulus` | (n, 1) | The movement refined to the actual contraction. **This is the label used.** |
| `repetition` | (n, 1) | Requested repetition index. |
| `rerepetition` | (n, 1) | Refined repetition index. **This is the one used.** |
| `subject`, `age`, `gender`, `height`, `weight`, `circumference`, `laterality` | scalar | Subject metadata. |
| `frequency` | scalar | 200 Hz. |
| `sensor` | scalar | `"Double Myo"`. |

`restimulus` and `rerepetition` rather than `stimulus` and `repetition`: the
unrefined fields mark the whole period the subject was being *asked* to move,
including the time before they started and after they stopped. Training on
those labels teaches the model that rest is sometimes called a gesture.

### Two quirks that will bite you

1. **The `exercise` field inside the file is wrong.** `S1_E1_A1.mat` reports
   `exercise == 3`. The mapping is shifted across all ten subjects. The
   filename is authoritative. `src/data/ninapro.py` ignores the field and
   records the discrepancy in the recording metadata so it stays visible.

2. **`rerepetition` is 0 throughout every rest period.** Rest therefore
   belongs to no repetition, and a repetition-wise train/test split would put
   all of it on one side. `assign_rest_repetitions` gives each contiguous rest
   segment the repetition index of the movement it leads into. Rest before the
   first movement of a recording is attributed to nothing and those windows
   are dropped.

## Exercises

| File | Exercise | Movements | Used here |
| --- | --- | --- | --- |
| `S*_E1_A1.mat` | A | 12 isolated finger movements | No |
| `S*_E2_A1.mat` | B | 17 basic hand configurations and wrist movements | **Yes, by default** |
| `S*_E3_A1.mat` | C | 23 grasps and functional movements | Available via config |

Exercise B is the default because it contains the six wrist degrees of freedom
plus fist, point and finger spread. Between them those are what a
wrist-and-grip prosthesis is asked to do, and a confusion between any two of
them is a visible error rather than a near miss.

With rest, the default label space is **18 classes**.

## Subjects

Ten adults, all intact-limbed. Read out of the recordings themselves by
`python -m src.data.build_report`, not copied from the paper:

| Subject | Age | Sex | Height (cm) | Weight (kg) | Forearm circumference (cm) |
| --- | --- | --- | --- | --- | --- |
| 1 | 23 | m | 187 | 67 | 24.0 |
| 2 | 28 | m | 187 | 75 | 26.0 |
| 3 | 28 | m | 170 | 63 | 25.0 |
| 4 | 22 | f | 156 | 52 | 22.0 |
| 5 | 28 | f | 160 | 61 | 25.0 |
| 6 | 24 | m | 170 | 65 | 23.5 |
| 7 | 32 | m | 172 | 78 | 28.5 |
| 8 | 31 | m | 170 | 74 | 26.5 |
| 9 | 34 | m | 176 | 68 | 27.0 |
| 10 | 30 | m | 173 | 83 | 27.5 |

Ages 22 to 34, eight male and two female, all right-handed, forearm
circumference 22 to 28.5 cm, 2.81 hours of recording in total.

Three things follow from that, and they bound every claim in this repository:

* **No amputees.** Ninapro DB3 and DB7 include amputee subjects; DB5 does not.
  Residual-limb sEMG differs from intact-limb sEMG in amplitude, in which
  muscles survive, and in how reliably the subject can produce a requested
  movement. A model that generalises across ten intact subjects has not been
  shown to generalise to an amputee, and that is the population a prosthesis
  is for.
* **Eight of ten subjects are male, and the forearm circumference range is
  narrow.** Two subjects cannot support any claim about sex differences, and
  22 to 28.5 cm does not cover the population a device would be fitted to.
* **Ten subjects is a small sample.** The leave-one-subject-out standard
  deviation reported in the README is computed over ten folds. Treat it as an
  indication of spread, not as a confidence interval.

## Preprocessing applied here

Set in `config/default.yaml`; the code is in `src/preprocessing/filters.py`
and `src/data/windows.py`.

1. **Scale.** Divide by 128 to bring the signed 8-bit Myo counts into roughly
   [-1, 1].
2. **Bandpass.** Fourth-order Butterworth, 10 to 99 Hz, zero-phase
   (`sosfiltfilt`). The upper edge sits just under the 100 Hz Nyquist
   frequency.
3. **No mains notch.** The Myo filters interference in hardware, and at 200 Hz
   a 50 Hz notch would gouge a hole in the middle of the useful band. The
   notch is implemented and configurable for higher-rate sources.
4. **No rectification.** The convolutional models read the oscillating signal.
   The classical feature set rectifies internally where a feature needs it.
5. **Window.** 200 ms with a 50 ms stride, by default. Windows straddling a
   label change or a repetition change are dropped.
6. **Balance rest.** Rest arrives at roughly seventeen times the rate of any
   single movement. It is subsampled per subject and per repetition to
   approximately the mean movement-class count.

The zero-phase filter uses future samples and a real controller cannot. That
is why `src/embedded/control_loop.py` re-runs the whole thing through a causal
filter and reports the difference rather than assuming it away.

## Resulting dataset

Default configuration, all ten subjects, exercise B:

| | |
| --- | --- |
| Windows after balancing | 74,764 |
| Shape per window | 16 channels x 40 samples |
| Classes | 18 (rest plus 17 movements) |
| Windows per class | 3,395 to 4,833 |
| Repetitions | 6 per subject per movement |

Regenerate with `python -m src.data.build_report`.

## Licence

Ninapro is released for research use with attribution. The project page states
the terms and asks that the relevant paper be cited in any publication using
the data.

**This is not a commercial licence.** If you intend to use DB5 in a product,
read the terms on the publisher's site and contact them. Nothing in this
repository grants any right to the data; the MIT licence on this repository
covers the code only.

Cite:

> Pizzolato S, Tagliapietra L, Cognolato M, Reggiani M, Müller H, Atzori M.
> Comparison of six electromyography acquisition setups on hand movement
> classification tasks. *PLOS ONE* 12(10): e0186132, 2017.

and the database paper:

> Atzori M, Gijsberts A, Castellini C, Caputo B, Hager A-G M, Elsig S,
> Giatsidis G, Bassetto F, Müller H. Electromyography data for non-invasive
> naturally-controlled robotic hand prostheses. *Scientific Data* 1: 140053,
> 2014.

## Ethics

The recordings were collected by the Ninapro project under their own ethical
approval, from consenting adult volunteers, and are published in de-identified
form. No personal recordings were made for this repository, and none of the
subject metadata is identifying.

## The synthetic alternative

`src/data/synthetic.py` generates surrogate sEMG with the same field layout.
It exists so the pipeline, the tests and CI can run without the download.

**No accuracy figure from the synthetic source is a result.** Every script
that touches it stamps `SYNTHETIC` on its output and every artefact directory
it writes to is named `synthetic`. It is a wiring check.

It is not arbitrary noise: each gesture is a pattern of activation across six
notional muscles, observed through a per-subject electrode mixing matrix, with
band-limited noise as the carrier. The per-subject mixing is what makes
cross-subject transfer hard in the synthetic data for the same reason it is
hard in the real data, and there is a test asserting that the gap survives.
