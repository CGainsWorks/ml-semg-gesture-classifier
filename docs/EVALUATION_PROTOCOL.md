# How this repository is evaluated, and why it matters

## The short version

**Within-subject accuracy on sEMG is easy and close to useless.** It measures a
model that has already met its user. A prosthesis is fitted to somebody the
model has never seen.

So this repository reports three numbers, leads with the hard one, and states
the gap between them explicitly.

## The three protocols

### 1. Within-subject

Train and test on the same person, split by repetition. Repetitions 1, 3, 4
and 6 train; 2 and 5 test. This is the Ninapro convention and it is what most
sEMG papers report.

What it tells you: whether the features and the model can separate these
gestures at all, for a user who has sat through a full recording session.

What it does not tell you: anything about a new wearer.

### 2. Subject-independent (leave one subject out)

Train on nine people, test on the tenth. Rotate over all ten. Report the mean
and the spread.

This is the number this repository leads with. It is what a device faces on
the day it is fitted, before any personalisation.

### 3. Subject-independent plus calibration

Leave one subject out, then fine-tune on a single repetition from the held-out
subject before testing on their remaining repetitions.

This is the realistic middle ground. Nobody ships a prosthesis with no fitting
appointment, and nobody makes a new amputee sit through a full Ninapro
protocol either. One repetition of each gesture is a few minutes of a fitting
appointment. The question this answers is how much of the generalisation gap
those few minutes buy back.

The calibration repetition is **removed from the test set**, not shared with
it. Scoring on data the model was just fine-tuned on would make the number
meaningless, and there is a test that asserts the two sets are disjoint
(`tests/test_data.py::TestSplits::test_calibration_windows_are_not_also_test_windows`).

## The rules that keep the numbers honest

These are the ways an sEMG result gets silently inflated. Each is closed
deliberately, and each has a test.

### Normalisation is fit on the training fold only

`ChannelScaler` is a separate object with an explicit `fit`, so the statistics
have to be passed in rather than being available ambiently. Fitting them over
all subjects would hand the model the held-out subject's channel gains, which
is precisely the information the subject-independent protocol is supposed to
withhold. The same applies to the classical baseline's per-channel sigmas,
which set its zero-crossing and Willison thresholds.

Asserted by `tests/test_models.py::TestNoLeakage`.

### The validation split is by repetition, never at random

Consecutive windows overlap by three quarters at the default geometry. A
random validation split would put a window in training and its near-duplicate
in validation, so the validation curve would track memorisation rather than
generalisation, and early stopping would fire at the wrong epoch.

`train_validation_split` splits on whole (subject, repetition) groups.
Asserted by
`tests/test_data.py::TestSplits::test_validation_split_holds_out_whole_repetitions`.

### Windows never straddle a label or repetition boundary

A window spanning the transition from rest into a movement has no correct
label. Ninapro's convention is to drop them and this follows it, checking
homogeneity in the label and the repetition together.

### Rest is attributed to a repetition before splitting

Ninapro leaves `rerepetition` at 0 throughout rest, so a naive repetition
split puts every rest window on one side. `assign_rest_repetitions` gives each
rest segment the repetition of the movement it leads into. Without this the
within-subject test fold would contain almost no rest and its accuracy would
be measured on an easier problem than the training fold.

### Rest is subsampled

Exercise B alternates each of 17 movements with a rest period of comparable
length, so rest arrives at roughly seventeen times the rate of any single
movement. A classifier that always answers rest would score about 50% accuracy
and be completely useless. Rest is capped at approximately the mean
movement-class count, per subject and per repetition.

Balanced accuracy and macro F1 are reported alongside plain accuracy for the
same reason, and they are the ones to read.

### No subject appears in both halves of a subject-independent fold

Obvious, and asserted anyway, because it is the one mistake that would make
every headline number wrong at once:
`tests/test_data.py::TestSplits::test_subject_independent_never_shares_a_subject`.

## Metrics, and why these ones

| Metric | Why it is here |
| --- | --- |
| Accuracy | Familiar. Read it last. |
| Balanced accuracy | The classes are close to balanced after subsampling but not exactly, and rest is the easiest class. |
| Macro F1 | Catches a model that gets its aggregate score from a few easy classes. |
| Cohen's kappa | Accuracy corrected for chance, which matters at 18 classes. |
| **False activation rate** | Rest windows classified as a movement. **The arm moves when the wearer did not ask it to.** |
| **Missed activation rate** | Movement windows classified as rest. The arm fails to respond. |

The last two are broken out because they are not symmetric in consequence. A
confusion between two similar grasps costs the wearer a retry. An unbidden
movement is what makes people stop wearing the device.

## What is pooled and what is averaged

Per-fold results are combined by **summing their confusion matrices** rather
than by averaging their accuracies, so each fold is weighted by its size.

The per-fold mean and standard deviation are reported separately, because for
the subject-independent protocol the spread between subjects is as informative
as the mean. A model that scores well on eight subjects and collapses on two
is not the same product as one that is mediocre on all ten, and the mean alone
cannot tell them apart. The worst and best fold are reported for the same
reason.

## What this evaluation still does not tell you

Stated plainly, because a protocol document that only lists its own strengths
is marketing:

* **No amputees.** DB5 is ten intact-limbed adults. Residual-limb sEMG is a
  different signal. Generalising across ten intact subjects is not evidence of
  generalising to an amputee.
* **No electrode repositioning between sessions.** Every repetition for a
  subject comes from one session with the armband in one position. Day-to-day
  donning variation is one of the largest practical failure modes in
  myoelectric control and this dataset cannot measure it.
* **No fatigue, no limb position variation, no sweat.** All three shift the
  signal and none is present here.
* **Ten subjects.** The reported standard deviation is over ten folds. It
  indicates spread; it is not a confidence interval.
* **Offline, not closed loop.** Classification accuracy on recorded windows is
  not the same thing as a user completing a task with the device. Users adapt
  to a controller's errors, and a controller with lower offline accuracy can
  be better to use. Measuring that needs users and hardware, and this
  repository has neither.
