"""Figures.

Matplotlib with the Agg backend so this works headless, in CI and in Docker.
No seaborn: one fewer pin for four plots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIGSIZE_SQUARE = (9.0, 8.0)
FIGSIZE_WIDE = (11.0, 5.5)


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    path: str | Path,
    title: str = "Confusion matrix",
    normalise: bool = True,
) -> Path:
    """Row-normalised confusion matrix.

    Row-normalised, so each cell reads as "of the windows that really were
    this gesture, what fraction went here". The class counts are close but not
    identical, and an unnormalised matrix would let those differences read as
    accuracy differences.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square matrix, got {matrix.shape}")
    if len(labels) != matrix.shape[0]:
        raise ValueError(f"{len(labels)} labels for a {matrix.shape[0]}-class matrix")

    display = matrix.copy()
    if normalise:
        totals = display.sum(axis=1, keepdims=True)
        display = np.divide(display, totals, out=np.zeros_like(display), where=totals > 0)

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    image = ax.imshow(display, cmap="viridis", vmin=0.0, vmax=1.0 if normalise else None)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)

    # Annotate only the diagonal and the worst confusions. Annotating a whole
    # 18 by 18 grid is unreadable.
    n = display.shape[0]
    off_diagonal = display.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    threshold = np.percentile(off_diagonal, 99.0) if n > 3 else 0.0
    for i in range(n):
        for j in range(n):
            value = display[i, j]
            if i == j or (value >= threshold and value > 0.05):
                ax.text(
                    j,
                    i,
                    f"{value:.2f}" if normalise else f"{int(value)}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if value < 0.6 else "black",
                )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04,
                 label="fraction of true class" if normalise else "windows")
    return _save(fig, path)


def per_class_bars(
    names: list[str],
    precision: list[float],
    recall: list[float],
    path: str | Path,
    title: str = "Per-class precision and recall",
) -> Path:
    order = np.argsort(recall)
    names = [names[i] for i in order]
    precision = [precision[i] for i in order]
    recall = [recall[i] for i in order]

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10.0, max(4.0, 0.38 * len(names) + 2.0)))
    ax.barh(y - 0.2, recall, height=0.38, label="recall", color="#3b7dd8")
    ax.barh(y + 0.2, precision, height=0.38, label="precision", color="#e08a3c")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("score")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, path)


def protocol_comparison(
    rows: list[dict],
    path: str | Path,
    title: str = "Accuracy by evaluation protocol",
    n_classes: int = 18,
) -> Path:
    """Grouped bars: one group per model, one bar per protocol.

    The figure the whole repository exists to produce. The gap between the
    within-subject bar and the subject-independent bar is the point, and the
    chance line is there so the reader can see how much of the remaining
    distance to it the subject-independent bars have given back.
    """
    models = sorted({r["model"] for r in rows})
    protocols = [
        "within_subject",
        "subject_independent_calibrated",
        "subject_independent",
    ]
    present = [p for p in protocols if any(r["protocol"] == p for r in rows)]
    lookup = {(r["model"], r["protocol"]): r for r in rows}

    x = np.arange(len(models))
    width = 0.8 / max(1, len(present))
    colours = {
        "within_subject": "#3b7dd8",
        "subject_independent_calibrated": "#8c62c4",
        "subject_independent": "#d84c3b",
    }

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for k, protocol in enumerate(present):
        values, errors = [], []
        for model in models:
            row = lookup.get((model, protocol))
            values.append(row["balanced_accuracy"] if row else 0.0)
            errors.append(row.get("fold_sd", 0.0) if row else 0.0)
        offset = (k - (len(present) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            yerr=errors,
            capsize=3,
            label=protocol.replace("_", " "),
            color=colours.get(protocol, "#777777"),
        )
        for bar, value, error in zip(bars, values, errors):
            if value > 0:
                # Above the error bar, not on top of it: the caps sit at
                # value + error and would otherwise run through the digits.
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + error + 0.02,
                    f"{value:.3f}",
                    ha="center",
                    fontsize=7,
                )

    # Drawn before the legend so that it appears in it.
    ax.axhline(
        1.0 / n_classes, color="#555555", linestyle=":", linewidth=1.2, label="chance"
    )

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ") for m in models], fontsize=9)
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def latency_accuracy(rows: list[dict], budget_ms: float, path: str | Path) -> Path:
    """Accuracy against end-to-end latency, with the deadline marked."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    by_vote: dict[int, list[dict]] = {}
    for row in rows:
        by_vote.setdefault(int(row["vote_window"]), []).append(row)

    markers = {1: "o", 3: "s", 5: "^"}
    for vote, group in sorted(by_vote.items()):
        group = sorted(group, key=lambda r: r["geometry_latency_ms"])
        ax.plot(
            [r["geometry_latency_ms"] for r in group],
            [r["balanced_accuracy"] for r in group],
            marker=markers.get(vote, "o"),
            linestyle="none",
            markersize=7,
            alpha=0.85,
            label=f"{vote}-decision vote",
        )

    ax.axvline(budget_ms, color="#d84c3b", linestyle="--", linewidth=1.5)
    ax.text(
        budget_ms - 4,
        ax.get_ylim()[0] + 0.01,
        f"{budget_ms:.0f} ms budget",
        rotation=90,
        va="bottom",
        ha="right",
        fontsize=8,
        color="#d84c3b",
    )
    ax.set_xlabel("end-to-end controller latency (ms)")
    ax.set_ylabel("balanced accuracy, subject-independent")
    ax.set_title("What longer windows and longer votes buy, and what they cost")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def signal_example(
    raw: np.ndarray,
    filtered: np.ndarray,
    envelope_signal: np.ndarray,
    labels: np.ndarray,
    sample_rate_hz: int,
    path: str | Path,
    channel: int = 0,
) -> Path:
    """Raw, bandpassed and enveloped traces of one channel, with the labels.

    The README's opening figure. It shows what the preprocessing does and what
    a gesture looks like in the signal, which is more use than a paragraph
    saying the same thing.
    """
    t = np.arange(raw.shape[0]) / sample_rate_hz
    fig, axes = plt.subplots(4, 1, figsize=(12.0, 8.0), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 3, 1]})

    axes[0].plot(t, raw[:, channel], linewidth=0.5, color="#888888")
    axes[0].set_ylabel("raw")
    axes[0].set_title(
        f"Ninapro DB5, channel {channel}: raw Myo output, bandpassed, and linear envelope"
    )

    axes[1].plot(t, filtered[:, channel], linewidth=0.5, color="#3b7dd8")
    axes[1].set_ylabel("10-99 Hz")

    axes[2].plot(t, envelope_signal[:, channel], linewidth=0.9, color="#e08a3c")
    axes[2].set_ylabel("envelope")

    axes[3].step(t, labels, where="post", linewidth=1.0, color="#444444")
    axes[3].set_ylabel("gesture")
    axes[3].set_xlabel("time (s)")
    axes[3].set_yticks(sorted({int(v) for v in np.unique(labels)}))

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.align_ylabels(axes)
    return _save(fig, path)
