"""Why the shipped model looks the way it does.

Every architecture claim made in `src/models/architecture.py` and
`src/models/augment.py` is measured here rather than asserted. The variants:

``conv1d``
    Convolution along time with the sixteen electrodes as input channels. The
    obvious design, and the first one built. It treats the electrodes as an
    unordered feature vector.

``conv2d``
    The shipped design. The window is a 16 x 40 electrode-by-time image and a
    3 x 5 kernel moves over both axes, so adjacent electrodes are adjacent to
    the model.

``conv2d_circular``
    As above with the electrode axis wrapped, since the electrodes are a ring.
    Physically motivated, and it turns out not to matter.

``conv2d_noaug`` and ``conv2d_electrode_shift``
    The augmentation ablation. The second one randomly rotates each armband's
    eight electrodes during training to simulate the band being worn
    differently. It sounds right and it is badly wrong: which electrode sees
    which muscle is the signal, not nuisance variation.

One fold per protocol, not the full rotation. These numbers are noisier than
the headline results and are not quoted as such. What is being compared is one
architecture against another under identical conditions, and a single fold is
enough for that.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.config import Config, load_config, resolve
from src.data.dataset import build_dataset, class_codes, encode_labels
from src.data.splits import make_folds, train_validation_split
from src.models.architecture import ArchitectureSpec
from src.models.augment import Augmenter
from src.models.torch_cnn import ToImage, TorchGestureCNN, resolve_device
from src.preprocessing.normalise import ChannelScaler

PROTOCOLS = ("within_subject", "subject_independent")


class Conv1dGestureCNN(nn.Module):
    """The rejected design: time-only convolution, electrodes as input channels."""

    def __init__(self, spec: ArchitectureSpec):
        super().__init__()
        self.spec = spec
        blocks: list[nn.Module] = []
        in_ch = spec.n_channels
        for out_ch in spec.conv_channels:
            blocks += [
                nn.Conv1d(in_ch, out_ch, spec.kernel_time, padding=spec.kernel_time // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        pooled_time = spec.window_samples
        for _ in spec.conv_channels:
            pooled_time //= 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(spec.dropout),
            nn.Linear(pooled_time * spec.conv_channels[-1], spec.dense_units),
            nn.ReLU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.dense_units, spec.n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class _CircularElectrodePad(nn.Module):
    """Wrap the electrode axis, zero-pad time. The electrodes form a ring."""

    def __init__(self, electrodes: int, time: int):
        super().__init__()
        self.electrodes, self.time = electrodes, time

    def forward(self, x):
        x = F.pad(x, (0, 0, self.electrodes, self.electrodes), mode="circular")
        return F.pad(x, (self.time, self.time, 0, 0), mode="constant", value=0.0)


class Conv2dCircularCNN(nn.Module):
    """The shipped design with a wrapped electrode axis."""

    def __init__(self, spec: ArchitectureSpec):
        super().__init__()
        self.spec = spec
        pe, pt = spec.kernel_electrodes // 2, spec.kernel_time // 2
        kernel = (spec.kernel_electrodes, spec.kernel_time)
        layers: list[nn.Module] = [ToImage()]
        in_ch = 1
        for out_ch in spec.conv_channels:
            layers += [
                _CircularElectrodePad(pe, pt),
                nn.Conv2d(in_ch, out_ch, kernel, padding=0),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.flattened_units, spec.dense_units),
            nn.ReLU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.dense_units, spec.n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ElectrodeShiftAugmenter(Augmenter):
    """Adds a random rotation of each armband's eight electrodes.

    Kept here rather than in ``src/models/augment.py`` because it is a
    negative result, not a shipped feature.
    """

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        out = super().__call__(batch)
        half = out.shape[1] // 2
        first = self._rng.integers(0, half)
        second = self._rng.integers(0, half)
        return np.concatenate(
            [np.roll(out[:, :half], first, axis=1), np.roll(out[:, half:], second, axis=1)],
            axis=1,
        )


def _build(variant: str, spec: ArchitectureSpec) -> nn.Module:
    if variant.startswith("conv2d"):
        if variant == "conv2d_circular":
            return Conv2dCircularCNN(spec)
        return TorchGestureCNN(spec)
    if variant == "conv1d":
        return Conv1dGestureCNN(spec)
    raise ValueError(f"unknown variant {variant!r}")


def _augmenter(variant: str, config: Config) -> Augmenter:
    if variant == "conv2d_noaug":
        return Augmenter(noise_sd=0.0, scale_sd=0.0, seed=config.train.seed)
    if variant == "conv2d_electrode_shift":
        return ElectrodeShiftAugmenter(
            noise_sd=config.train.augment_noise_sd,
            scale_sd=config.train.augment_scale_sd,
            seed=config.train.seed,
        )
    return Augmenter(
        noise_sd=config.train.augment_noise_sd,
        scale_sd=config.train.augment_scale_sd,
        seed=config.train.seed,
    )


def _to_device(x: np.ndarray, device: torch.device, dtype) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x, dtype=dtype)).to(device)


def _accuracy(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
) -> float:
    """Accuracy over ``x``, in batches, on tensors already on the device.

    In batches because it is not optional. Pushing a whole leave-one-subject-out
    fold through this network in one call means tens of thousands of windows
    times thirty-two feature maps of 16 by 40 activations, which is several
    gigabytes of intermediate tensor and will exhaust an 8 GB card. Already on
    the device because this runs once per epoch, and re-uploading the
    validation set every time costs more than the forward pass does.
    """
    correct = 0
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb, yb = x[start : start + batch_size], y[start : start + batch_size]
            correct += int((model(xb).argmax(dim=1) == yb).sum().item())
    return correct / max(1, int(x.shape[0]))


def _train_and_score(
    variant: str, config: Config, fold, codes: np.ndarray, verbose: bool
) -> dict[str, object]:
    spec = ArchitectureSpec.from_config(
        config.train, fold.train.n_channels, fold.train.length, codes.size
    )
    device = resolve_device(config.train.device)
    torch.manual_seed(config.train.seed)

    raw_train = fold.train.materialise("ct")
    scaler = ChannelScaler.fit(raw_train)
    x = scaler.transform(raw_train)
    y = encode_labels(fold.train.labels, codes)
    x_test = scaler.transform(fold.test.materialise("ct"))
    y_test = encode_labels(fold.test.labels, codes)
    train_idx, val_idx = train_validation_split(
        fold.train, config.train.val_fraction, config.train.seed
    )

    model = _build(variant, spec).to(device)
    augmenter = _augmenter(variant, config)
    augmenter.reset()
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    # Sliced once. Fancy indexing copies, and doing it inside the epoch loop
    # would rebuild a thirty-megabyte array on every pass for no reason. The
    # validation and test sets go to the device once and stay there; only the
    # training batches make the trip, because they have to be augmented on the
    # host first.
    x_fit, y_fit = x[train_idx], y[train_idx]
    y_fit_device = _to_device(y_fit, device, np.int64)
    x_val_device = _to_device(x[val_idx], device, np.float32)
    y_val_device = _to_device(y[val_idx], device, np.int64)
    x_test_device = _to_device(x_test, device, np.float32)
    y_test_device = _to_device(y_test, device, np.int64)
    x_fit_device = _to_device(x_fit, device, np.float32)

    best, best_state, since = -1.0, None, 0
    rng = np.random.default_rng(config.train.seed)
    for _ in range(config.train.epochs):
        model.train()
        order = rng.permutation(x_fit.shape[0])
        for start in range(0, order.size, config.train.batch_size):
            idx = order[start : start + config.train.batch_size]
            xb = torch.from_numpy(augmenter(x_fit[idx])).to(device)
            yb = y_fit_device[torch.from_numpy(idx).to(device)]
            optimiser.zero_grad(set_to_none=True)
            criterion(model(xb), yb).backward()
            optimiser.step()
        model.eval()
        accuracy = _accuracy(
            model, x_val_device, y_val_device, config.train.batch_size
        )
        if accuracy > best:
            best, since = accuracy, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= config.train.early_stopping_patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    test = _accuracy(model, x_test_device, y_test_device, config.train.batch_size)
    train = _accuracy(model, x_fit_device, y_fit_device, config.train.batch_size)

    row = {
        "variant": variant,
        "train_accuracy": train,
        "val_accuracy": best,
        "test_accuracy": test,
        "overfit_gap": train - test,
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    if verbose:
        print(
            f"  {variant:24s} train {train:.3f}  val {best:.3f}  test {test:.3f}  "
            f"({row['parameters']:,} parameters)"
        )
    return row


VARIANTS = (
    "conv1d",
    "conv2d",
    "conv2d_circular",
    "conv2d_noaug",
    "conv2d_electrode_shift",
)


def run(config: Config, fold_index: int = 0, verbose: bool = True) -> dict[str, object]:
    index = build_dataset(config)
    codes = class_codes(index)
    out: dict[str, object] = {
        "synthetic": config.dataset.source == "synthetic",
        "fold_index": fold_index,
        "results": {},
    }
    for protocol in PROTOCOLS:
        fold = make_folds(index, config, protocol)[fold_index]
        if verbose:
            print(f"{protocol} ({fold.name}): {len(fold.train)} train, {len(fold.test)} test")
        out["results"][protocol] = [
            _train_and_score(variant, config, fold, codes, verbose) for variant in VARIANTS
        ]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    report = run(config, fold_index=args.fold)

    tag = "synthetic" if config.dataset.source == "synthetic" else "db5"
    out = (
        Path(args.out)
        if args.out
        else resolve("experiments") / "results" / tag / "architecture_comparison.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
