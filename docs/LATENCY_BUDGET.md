# The latency budget

## The requirement

**A myoelectric controller has roughly 300 ms from the wearer's intention to
the arm moving.** Past that, users stop experiencing the hand as part of
themselves and start experiencing it as a machine they are operating.

That figure is not arbitrary and it is not this project's invention. The
myoelectric control literature converges on it from two directions:

* Englehart and Hudgins (2003) established that a controller delay under
  roughly 300 ms is imperceptible in continuous control, and used that to
  justify the 200 ms analysis window that has been standard ever since.
* Farrell and Weir (2007) measured it directly with users and found optimal
  controller delay between 100 and 125 ms, with acceptability degrading
  steadily beyond that and users rejecting the device past about 300 ms.

This repository takes **300 ms as a hard ceiling** and treats anything under
250 ms as comfortable. `config/default.yaml` sets `latency.budget_ms: 300.0`,
and every latency figure produced here is scored against it.

## What is in the budget

The mistake worth avoiding is treating "latency" as the model's inference
time. Inference is the smallest term. Here is the whole thing:

| Term | What it is | Cost |
| --- | --- | --- |
| Window fill | The classifier cannot decide until it has a full window of samples. | **200 ms**, and it dominates |
| Filter group delay | The causal bandpass cannot respond before the signal arrives. | ~3 ms |
| Feature extraction | Only for the classical baseline; the network reads the window directly. | < 1 ms |
| Inference | Forward pass of the quantised model. | See the measured table in the README |
| Majority vote | The controller waits for `vote_window` decisions to agree before moving. | `(vote_window - 1) x stride` |
| Acquisition and actuation | Reading the electrodes, driving the motors. | **Not implemented here.** Allowance: 20 ms |

The filter group delay is measured, not assumed:
`tests/test_preprocessing.py::TestCausalBandpass` compares the energy centroid
of a burst through the causal filter against the zero-phase one and asserts
the delay is under five samples. At 200 Hz that is 25 ms of headroom for a
term that is actually about 3 ms, and it is small enough to sit inside the
20 ms input/output allowance rather than getting its own line in the
arithmetic.

The 20 ms allowance is a **stated assumption, not a measurement**. This
repository has no acquisition board and no actuators. If your hardware is
slower than that, the budget shrinks by the difference.

## The finding

The first version of this project used a 200 ms window, a 50 ms stride and a
three-decision majority vote, and it **missed the budget**:

```
200 ms  window fill
100 ms  majority vote (2 further decisions at 50 ms each)
  0 ms  inference (under 0.1 ms, measured; rounds to nothing)
 20 ms  input/output allowance
-----
320 ms  against a 300 ms budget: 20 ms over
```

Inference was 0.01% of the total. The model was never the problem. The
signal-processing geometry was, and no amount of quantisation or kernel
optimisation would have fixed it.

`experiments/latency_accuracy_tradeoff.py` prices the geometry properly. It
runs in two stages, and the second stage exists because the first one is
misleading.

**Stage one** sweeps five window lengths, two strides and three vote lengths,
trains a model at each of the ten distinct geometries and scores all thirty
combinations on one held-out subject.

**Stage two** takes the two best geometries that fit inside the budget and
re-runs each over the full ten-fold leave-one-subject-out rotation.

What the sweep settled:

**The majority vote buys nothing, so the default is not to vote.** At four of
the five window lengths a single decision beat a three-decision vote, and a
five-decision vote was worse again at all five. Consecutive windows overlap by
three quarters, so their decisions are highly correlated and a vote over them
corrects almost nothing while smearing the answer across gesture transitions.
Each extra decision also costs a stride of latency. Dropping the vote is what
brought the controller inside its deadline, and it cost no accuracy at all.

That one deserves a caveat rather than a victory lap. The offline protocol
drops windows that straddle a label change, and transitions are exactly where
a vote would plausibly earn its keep. The honest statement is that voting buys
nothing *measurable under this protocol*. The mechanism is kept, it stays
configurable, and it is counted in the budget whenever it is switched on.

**A 50 ms stride beats a 25 ms one.** At four of the five window lengths, and
the fifth is a tie to four decimal places. Halving the stride doubles the
number of training windows, which sounds like it should help until you notice
those windows then overlap by seven eighths and are very largely the same data
again. It doubles the cost of every training run for nothing.

**Window length is where stage two earned its keep, by refusing to give an
answer.** On the single fold, accuracy rose with window length all the way to
300 ms, and 300 ms costs 320 ms end to end and misses the budget. Among the
geometries that fit, that fold ranked a 200 ms window first.

Over ten folds:

| Geometry | Balanced accuracy | Per-fold mean and sd | Worst / best fold | End to end |
| --- | --- | --- | --- | --- |
| 200 ms window, 50 ms stride | 0.3201 | 0.3197 +/- 0.0752 | see the results file | **220 ms** |
| 250 ms window, 50 ms stride | 0.3273 | 0.3270 +/- 0.0687 | see the results file | 270 ms |

The 250 ms window comes out 0.0072 ahead. The standard error of that
difference over ten folds is 0.0322, so the gap is **0.22 standard errors**,
which is nothing. An earlier run of the same experiment, before an unrelated
bug fix changed the training, put the 200 ms window ahead by 0.0035 instead:
the ordering flips with the random seed, which is what a tie looks like.

So the two window lengths are indistinguishable on accuracy, and one of them
costs fifty milliseconds more. The shipped configuration takes the cheaper
one, and the claim made for it is "tied on accuracy and faster", not "more
accurate". The single fold would have given a confident answer to a question
that does not have one, which is a better argument for the confirmation stage
than getting a different winner would have been.

The shipped geometry is therefore a **200 ms window, a 50 ms stride and no
vote**: 220 ms end to end against a 300 ms budget, with 80 ms of headroom that
the assumed 20 ms input and output allowance can eat into without the deadline
being missed.


## Measuring it on real hardware

`benchmarks/latency_host.json` is measured, on the machine named in the file,
single-threaded, batch size one, reported as a distribution because a control
loop is judged on its worst case rather than its average.

`benchmarks/latency_targets.json` is **not measured**. No STM32 and no Jetson
board was used for this work. Every row in that file carries
`"measured": false` and is arithmetic from the model's MAC count and a
documented throughput assumption per board. See `src/embedded/estimate.py` for
the assumptions and where they come from.

`benchmarks/renode_cortex_m4.json` is **emulated**. The int8 model runs in
TensorFlow Lite Micro with CMSIS-NN on an emulated Cortex-M4F under Renode
and its cycle count is an emulator estimate: one cycle per instruction, no
pipeline stalls, no flash wait states. Its outputs are checked bit for bit
against the desktop interpreter and its flash and RAM figures are exact for
the build. See `benchmarks/README.md` for what that does and does not
establish.

To replace an estimate with a measurement, run the same benchmark on the
board. `benchmarks/README.md` has the instructions. The output is the same
JSON in the same format, and dropping it into `benchmarks/` is all that is
needed.

## Two rates, not one

Meeting the budget is necessary and not sufficient. There is a second
constraint that is easy to miss:

* **Latency**: total delay must be under 300 ms. This is what the wearer
  feels.
* **Throughput**: the model must finish one inference within one stride, or
  decisions queue up and the delay grows without bound. At the shipped 50 ms
  stride that means inference must complete in under 50 ms.

`budget_check` in `src/embedded/benchmark.py` reports both, as `meets_budget`
and `keeps_up_with_stride`. On this hardware the second is met by nearly three
orders of magnitude. It is the constraint that actually bites on the small
microcontroller: the estimate for a 100 MHz Cortex-M4 is tens of milliseconds
per inference, which is comfortably inside the 300 ms latency budget and close
to the 50 ms stride. A board in that position has to run at a longer stride,
emit fewer decisions per second, or carry a smaller model. That is why the
estimate table reports both verdicts per board rather than one.
