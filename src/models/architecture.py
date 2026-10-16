"""The one architecture, described once.

Both framework implementations read their shape from here. If the TensorFlow
and PyTorch models are going to be compared, they have to be the same model,
and the only way to keep them the same as the code changes is to stop writing
the numbers down twice.

**Why two-dimensional convolutions over a one-dimensional signal.** The
obvious design for multi-channel sEMG is a 1-D convolution along time with the
electrodes as input channels, which lets the first layer mix all sixteen
electrodes at once. That was the first thing built here.

The electrodes are not an unordered feature vector, though. They are two rings
of eight around the forearm and adjacent electrodes see overlapping muscle
activity. Treating the window as a 16 x 40 image and convolving a 3 x 5 kernel
over it gives the model that adjacency as an inductive bias instead of asking
it to learn the layout from four thousand training windows. A ring is also
periodic, so wrapping the electrode axis was tried too and dropped as
complexity the single-fold result does not justify.

Being straight about the strength of that evidence
(`experiments/architecture_comparison.py`, one fold per protocol): the 2-D
design is ahead within subject and has the smaller overfitting gap there and
on the one leave-one-subject-out fold it is *behind* the 1-D design, with the
larger overfitting gap. The cross-subject difference is inside the
fold-to-fold spread of the ten-fold run, so it does not decide anything
either way. The README's architecture section states the figures; it is
generated from `experiments/results/db5/architecture_comparison.json` and CI
checks it, so the numbers live there and not here.

The model is small on purpose. The target is a microcontroller, and a 200 ms
window of 16-channel sEMG at 200 Hz does not carry enough information to feed
anything larger without memorising the training set.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import TrainConfig


@dataclass(frozen=True)
class ArchitectureSpec:
    """Shape of the network, derived from the config and the data.

    The input is always (n_channels, window_samples), the layout the rest of
    the repository uses. Both implementations reshape it to a single-channel
    image internally, so the exported TFLite and ONNX graphs take the same
    thing the training code does.
    """

    n_channels: int
    window_samples: int
    n_classes: int
    conv_channels: tuple[int, ...]
    kernel_electrodes: int
    kernel_time: int
    dense_units: int
    dropout: float

    @classmethod
    def from_config(
        cls, train: TrainConfig, n_channels: int, window_samples: int, n_classes: int
    ) -> ArchitectureSpec:
        if n_classes < 2:
            raise ValueError(f"need at least two classes, got {n_classes}")
        stages = len(train.conv_channels)
        if window_samples < 2**stages:
            raise ValueError(
                f"a {window_samples}-sample window cannot survive {stages} pooling stages"
            )
        if n_channels < 2**stages:
            raise ValueError(
                f"{n_channels} electrodes cannot survive {stages} pooling stages"
            )
        return cls(
            n_channels=n_channels,
            window_samples=window_samples,
            n_classes=n_classes,
            conv_channels=tuple(train.conv_channels),
            kernel_electrodes=train.kernel_electrodes,
            kernel_time=train.kernel_time,
            dense_units=train.dense_units,
            dropout=train.dropout,
        )

    @property
    def pooled_shape(self) -> tuple[int, int]:
        """(electrodes, time steps) left after one halving per block."""
        electrodes, time = self.n_channels, self.window_samples
        for _ in self.conv_channels:
            electrodes //= 2
            time //= 2
        if electrodes < 1 or time < 1:
            raise ValueError("pooling reduced an axis below one step")
        return electrodes, time

    @property
    def flattened_units(self) -> int:
        electrodes, time = self.pooled_shape
        return electrodes * time * self.conv_channels[-1]

    def parameter_count(self) -> int:
        """Trainable parameters, derived rather than read off a summary.

        Batch-norm running statistics are excluded: PyTorch holds them as
        buffers and Keras counts them as non-trainable weights, so counting
        them would make the two frameworks disagree about an architecture they
        actually share. A test asserts both against this number.
        """
        total = 0
        in_ch = 1
        kernel = self.kernel_electrodes * self.kernel_time
        for out_ch in self.conv_channels:
            total += in_ch * out_ch * kernel + out_ch  # convolution
            total += 2 * out_ch  # batch-norm scale and offset
            in_ch = out_ch
        total += self.flattened_units * self.dense_units + self.dense_units
        total += self.dense_units * self.n_classes + self.n_classes
        return total

    def multiply_accumulates(self) -> int:
        """MACs for one forward pass, for the embedded cost estimate.

        Convolutions and dense layers only. Batch norm, activations and
        pooling are elementwise and negligible beside these.
        """
        total = 0
        in_ch = 1
        electrodes, time = self.n_channels, self.window_samples
        kernel = self.kernel_electrodes * self.kernel_time
        for out_ch in self.conv_channels:
            total += electrodes * time * in_ch * out_ch * kernel
            in_ch = out_ch
            electrodes //= 2
            time //= 2
        total += self.flattened_units * self.dense_units
        total += self.dense_units * self.n_classes
        return total

    def describe(self) -> dict[str, object]:
        electrodes, time = self.pooled_shape
        return {
            "input_shape_channels_time": [self.n_channels, self.window_samples],
            "conv_channels": list(self.conv_channels),
            "kernel_electrodes_time": [self.kernel_electrodes, self.kernel_time],
            "pooled_shape_electrodes_time": [electrodes, time],
            "dense_units": self.dense_units,
            "dropout": self.dropout,
            "n_classes": self.n_classes,
            "parameters": self.parameter_count(),
            "macs_per_inference": self.multiply_accumulates(),
        }
