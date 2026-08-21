"""PyTorch implementation of the shared architecture."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.config import TrainConfig
from src.models.architecture import ArchitectureSpec
from src.models.augment import Augmenter, from_config


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but no CUDA device is visible")
    return torch.device(requested)


class ToImage(nn.Module):
    """(batch, electrodes, time) -> (batch, 1, electrodes, time).

    A layer rather than a reshape at the call site so the transformation is
    part of the exported graph. The deployed model then takes the same input
    layout the training code does, and there is no chance of a caller on an
    embedded target getting the reshape wrong.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1)


class TorchGestureCNN(nn.Module):
    """Three Conv2d-BN-ReLU-MaxPool blocks over the electrode-by-time image,
    then two dense layers.

    Takes (batch, n_channels, window_samples). Emits logits; softmax is
    applied by the caller so the exported graph keeps a single, unambiguous
    output.
    """

    def __init__(self, spec: ArchitectureSpec):
        super().__init__()
        self.spec = spec
        kernel = (spec.kernel_electrodes, spec.kernel_time)
        padding = (spec.kernel_electrodes // 2, spec.kernel_time // 2)

        blocks: list[nn.Module] = [ToImage()]
        in_ch = 1
        for out_ch in spec.conv_channels:
            blocks += [
                nn.Conv2d(in_ch, out_ch, kernel, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.flattened_units, spec.dense_units),
            nn.ReLU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.dense_units, spec.n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion, batch: int):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for start in range(0, x.shape[0], batch):
            xb, yb = x[start : start + batch], y[start : start + batch]
            logits = model(xb)
            loss_sum += criterion(logits, yb).item() * xb.shape[0]
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            total += int(yb.numel())
    return correct / max(1, total), loss_sum / max(1, total)


def train_model(
    spec: ArchitectureSpec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig,
    model: TorchGestureCNN | None = None,
    epochs: int | None = None,
    learning_rate: float | None = None,
    augmenter: Augmenter | None = None,
    verbose: bool = True,
) -> tuple[TorchGestureCNN, dict[str, object]]:
    """Train, with early stopping on validation accuracy.

    Passing an existing ``model`` continues from its weights, which is how the
    calibration condition fine-tunes a subject-independent model on a single
    repetition from the new wearer.
    """
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    epochs = config.epochs if epochs is None else epochs
    learning_rate = config.learning_rate if learning_rate is None else learning_rate
    augmenter = from_config(config) if augmenter is None else augmenter
    augmenter.reset()

    if model is None:
        model = TorchGestureCNN(spec)
    model = model.to(device)

    # Without augmentation the training set is moved to the device once rather
    # than copied per batch; the whole of DB5 is a few hundred megabytes and
    # fits comfortably. With augmentation the windows have to be perturbed on
    # the host anyway, so they stay there and each batch is transferred after
    # it has been augmented, which is one copy rather than a round trip.
    x_host = np.ascontiguousarray(x_train, dtype=np.float32)
    y_device = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.int64)).to(device)
    x_device = None if augmenter.enabled else torch.from_numpy(x_host).to(device)
    xv_device = torch.from_numpy(np.ascontiguousarray(x_val, dtype=np.float32)).to(device)
    yv_device = torch.from_numpy(np.ascontiguousarray(y_val, dtype=np.int64)).to(device)

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator(device=device).manual_seed(config.seed)

    best_accuracy = -1.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch, since_improvement = 0, 0
    history: list[dict[str, float]] = []
    n = x_host.shape[0]
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(n, device=device, generator=generator)
        running_loss, seen = 0.0, 0
        for start in range(0, n, config.batch_size):
            idx = order[start : start + config.batch_size]
            yb = y_device[idx]
            if augmenter.enabled:
                batch = augmenter(x_host[idx.cpu().numpy()])
                xb = torch.from_numpy(batch).to(device, non_blocking=True)
            else:
                xb = x_device[idx]
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            running_loss += loss.item() * xb.shape[0]
            seen += xb.shape[0]

        accuracy, val_loss = _evaluate(
            model, xv_device, yv_device, criterion, config.batch_size
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(1, seen),
                "val_loss": val_loss,
                "val_accuracy": accuracy,
            }
        )
        if verbose:
            print(
                f"  epoch {epoch:3d}/{epochs}  train_loss {history[-1]['train_loss']:.4f}"
                f"  val_loss {val_loss:.4f}  val_acc {accuracy:.4f}"
            )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch, since_improvement = epoch, 0
        else:
            since_improvement += 1
            if since_improvement >= config.early_stopping_patience:
                if verbose:
                    print(f"  early stop at epoch {epoch}; best was {best_epoch}")
                break

    model.load_state_dict(best_state)
    return model, {
        "framework": "pytorch",
        "device": str(device),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "train_seconds": time.perf_counter() - started,
        "augmented": augmenter.enabled,
        "history": history,
    }


def predict_proba(
    model: TorchGestureCNN, x: np.ndarray, batch_size: int = 1024, device: str = "auto"
) -> np.ndarray:
    """Class probabilities for (n_windows, n_channels, window_samples)."""
    torch_device = resolve_device(device)
    model = model.to(torch_device).eval()
    if x.shape[0] == 0:
        return np.empty((0, model.spec.n_classes), dtype=np.float32)
    out = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch = torch.from_numpy(
                np.ascontiguousarray(x[start : start + batch_size], dtype=np.float32)
            ).to(torch_device)
            out.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def get_weights(model: TorchGestureCNN) -> dict:
    """A detached copy of the weights, for caching a trained model."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def clone_with_weights(spec: ArchitectureSpec, weights: dict) -> TorchGestureCNN:
    """A fresh model carrying ``weights``.

    Used to hand the same trained base model to two protocols without letting
    the second one's fine-tuning mutate what the first one measured.
    """
    model = TorchGestureCNN(spec)
    model.load_state_dict(weights)
    return model.eval()


def save(model: TorchGestureCNN, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "spec": model.spec.__dict__}, path)


def load(path: str | Path) -> TorchGestureCNN:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    spec = ArchitectureSpec(**blob["spec"])
    model = TorchGestureCNN(spec)
    model.load_state_dict(blob["state_dict"])
    return model.eval()
