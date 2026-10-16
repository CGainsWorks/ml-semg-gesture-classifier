"""On-target latency estimates, and why they are only estimates.

No STM32 or Jetson board was used for this work. Rather than quietly omit the
target-hardware number, or worse invent one, this module computes it from the
model's operation count and a documented throughput figure for each candidate
board and labels every result as an estimate.

The arithmetic is deliberately simple, because a complicated model of a cache
hierarchy would only make a guess look like a measurement:

    latency = MACs / (clock_hz * macs_per_cycle) * overhead_factor

The inputs and where they come from:

``macs_per_cycle``
    For the Cortex-M targets, from the CMSIS-NN papers: the int8 convolution
    kernels use the SIMD ``SMLAD`` instruction, which retires two 16-bit
    multiply-accumulates per cycle, and in practice the kernels sustain
    somewhat less than that once address generation and the requantisation
    step are counted. For the Cortex-A57 in a Jetson Nano, from the NEON
    int8 dot-product throughput.

``overhead_factor``
    Everything the MAC count ignores: im2col buffer copies, requantisation,
    pooling, and the interpreter's own dispatch. Set from the general
    observation that TFLite Micro layers spend roughly a third of their time
    outside the multiply-accumulate inner loop.

Both of those are approximations, and neither was calibrated against these
particular boards. Treat the output as an order of magnitude.

The Cortex-M4 row has a second, independent estimate: the firmware under
``firmware/`` runs the same int8 model in TFLite Micro with CMSIS-NN on an
emulated Cortex-M4F and ``benchmarks/renode_cortex_m4.json`` records the
emulator's cycle count. That is also an estimate (the emulator retires one
instruction per cycle) but it is not arithmetic from a MAC count. The memory
figures from that build are not estimates: they are what the linker placed
and what the TFLite Micro allocator reported.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    name: str
    clock_mhz: float
    macs_per_cycle: float
    overhead_factor: float
    flash_kib: int
    ram_kib: int
    runtime: str
    note: str


# Representative parts, not a recommendation. The two Cortex-M entries bracket
# what a self-contained prosthetic controller would plausibly use; the Jetson
# is there because the original plan names it and because a Linux-class board
# changes the answer completely.
TARGETS: tuple[Target, ...] = (
    Target(
        name="STM32F411 (Cortex-M4F, 100 MHz)",
        clock_mhz=100.0,
        macs_per_cycle=1.4,
        overhead_factor=1.5,
        flash_kib=512,
        ram_kib=128,
        runtime="TFLite Micro + CMSIS-NN",
        note="No cache. Weights execute from flash, which costs wait states.",
    ),
    Target(
        name="STM32H743 (Cortex-M7, 480 MHz)",
        clock_mhz=480.0,
        macs_per_cycle=1.8,
        overhead_factor=1.4,
        flash_kib=2048,
        ram_kib=1024,
        runtime="TFLite Micro + CMSIS-NN",
        note="Dual-issue with an L1 cache; the whole model fits in tightly coupled RAM.",
    ),
    Target(
        name="Jetson Nano (Cortex-A57, 1.43 GHz, CPU only)",
        clock_mhz=1430.0,
        macs_per_cycle=6.0,
        overhead_factor=1.3,
        flash_kib=16 * 1024 * 1024,
        ram_kib=4 * 1024 * 1024,
        runtime="ONNX Runtime, one thread",
        note="CPU only. The 128-core GPU would be faster and is not modelled here.",
    ),
)


def estimate_latency_ms(macs: int, target: Target) -> float:
    """Estimated single-window inference time in milliseconds."""
    if macs <= 0:
        raise ValueError(f"MAC count must be positive, got {macs}")
    cycles = macs / target.macs_per_cycle * target.overhead_factor
    return cycles / (target.clock_mhz * 1e6) * 1e3


def fits_in_memory(
    model_bytes: int, ram_bytes: int, target: Target, ram_source: str
) -> dict[str, object]:
    """Whether the weights fit in flash and the working set fits in RAM.

    ``ram_bytes`` is what the deployment needs in RAM: for the Cortex-M
    targets, everything the linker placed in RAM for the emulated firmware
    build (statics, the tensor arena, heap and stack). The weights are not in
    it, because TFLite Micro reads them from flash.
    """
    return {
        "model_kib": model_bytes / 1024,
        "flash_kib": target.flash_kib,
        "fits_flash": model_bytes / 1024 <= target.flash_kib,
        "ram_needed_kib": ram_bytes / 1024,
        "ram_needed_source": ram_source,
        "ram_kib": target.ram_kib,
        "fits_ram": ram_bytes / 1024 <= target.ram_kib,
    }


def estimate_all(
    macs: int,
    model_bytes: int,
    ram_bytes: int,
    ram_source: str,
    window_ms: float,
    vote_ms: float,
    io_ms: float,
    budget_ms: float,
) -> list[dict[str, object]]:
    """One row per candidate target, with the budget verdict for each."""
    rows = []
    for target in TARGETS:
        inference_ms = estimate_latency_ms(macs, target)
        total = window_ms + vote_ms + inference_ms + io_ms
        rows.append(
            {
                "target": target.name,
                "runtime": target.runtime,
                "clock_mhz": target.clock_mhz,
                "estimated_inference_ms": inference_ms,
                "window_fill_ms": window_ms,
                "majority_vote_ms": vote_ms,
                "io_allowance_ms": io_ms,
                "estimated_total_ms": total,
                "budget_ms": budget_ms,
                "estimated_headroom_ms": budget_ms - total,
                "estimated_to_meet_budget": bool(total <= budget_ms),
                "memory": fits_in_memory(
                    model_bytes,
                    ram_bytes,
                    target,
                    ram_source
                    if target.runtime.startswith("TFLite Micro")
                    else "the TFLite Micro figure, for scale only: ONNX Runtime on "
                    "Linux needs more and the board has gigabytes",
                ),
                "note": target.note,
                "measured": False,
            }
        )
    return rows
