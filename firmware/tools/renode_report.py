"""Turn a Renode UART log into the emulator report and fail on any mismatch.

Standard library only, so it runs in the firmware image without the project's
Python dependencies.

Reads the firmware's UART log (protocol in firmware/src/main.cc), the section
sizes from ``arm-none-eabi-size -A`` and the desktop interpreter's outputs in
``firmware/vectors/expected.json``. Writes one JSON report and exits non-zero
if the firmware did not finish, did not run every window, or produced any
output that differs from the reference by more than the tolerance (zero by
default: the check is bit-exact).

Every timing in the report is labelled as an emulator estimate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CLOCK_MHZ = 100.0
FLASH_BYTES = 512 * 1024
RAM_BYTES = 128 * 1024

FLASH_SECTIONS = (
    ".isr_vector",
    ".text",
    ".rodata",
    ".ARM.extab",
    ".ARM.exidx",
    ".preinit_array",
    ".init_array",
    ".fini_array",
    ".data",
)
RAM_SECTIONS = (".data", ".bss", "._user_heap_stack")

LINE = re.compile(r"^SEMG (?P<kind>[A-Z]+)\s*(?P<rest>.*)$")
FIELDS = re.compile(r"(\w+)=(\S+)")


def parse_uart(text: str) -> dict[str, object]:
    """Pull the records out of the firmware's log. Ignores anything else."""
    info: dict[str, object] = {"windows": [], "log": []}
    for raw in text.splitlines():
        match = LINE.match(raw.strip())
        if not match:
            continue
        kind, rest = match["kind"], match["rest"]
        fields = dict(FIELDS.findall(rest))
        if kind == "MODEL":
            info["model"] = {k: int(v) for k, v in fields.items()}
        elif kind == "ARENA":
            info["arena"] = {k: int(v) for k, v in fields.items()}
        elif kind == "W":
            index = int(rest.split()[0])
            info["windows"].append(
                {
                    "index": index,
                    "cycles": int(fields["cycles"]),
                    "top": int(fields["top"]),
                    "out": [int(v) for v in fields["out"].split(",")],
                }
            )
        elif kind == "END":
            info["end"] = fields
        elif kind == "LOG":
            info["log"].append(rest)
    return info


def parse_size(text: str) -> dict[str, int]:
    """Section sizes from ``arm-none-eabi-size -A -d``."""
    sizes: dict[str, int] = {}
    for raw in text.splitlines():
        parts = raw.split()
        if len(parts) >= 3 and parts[0].startswith(".") and parts[1].isdigit():
            sizes[parts[0]] = int(parts[1])
    return sizes


def compare(
    firmware: list[dict[str, object]], expected: list[list[int]], tolerance: int
) -> dict[str, object]:
    worst = 0
    exact = 0
    top_agree = 0
    mismatched = []
    for row, want in zip(firmware, expected):
        got = row["out"]
        if len(got) != len(want):
            mismatched.append(row["index"])
            worst = max(worst, 255)
            continue
        diff = max(abs(a - b) for a, b in zip(got, want))
        worst = max(worst, diff)
        exact += diff == 0
        top_agree += row["top"] == max(range(len(want)), key=lambda k: (want[k], -k))
        if diff > tolerance:
            mismatched.append(row["index"])
    return {
        "compared": len(firmware),
        "bit_exact": exact,
        "max_abs_diff_lsb": worst,
        "tolerance_lsb": tolerance,
        "top_class_agrees": top_agree,
        "mismatched_windows": mismatched,
    }


def build_report(args: argparse.Namespace) -> tuple[dict[str, object], list[str]]:
    uart = parse_uart(Path(args.uart).read_text(encoding="utf-8", errors="replace"))
    sizes = parse_size(Path(args.size).read_text(encoding="utf-8"))
    expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))

    problems: list[str] = []
    end = uart.get("end")
    if end is None:
        problems.append("the firmware never printed its END line")
    elif end.get("status") != "ok":
        problems.append(f"the firmware reported failure: {end}")

    windows = sorted(uart["windows"], key=lambda w: w["index"])
    n_expected = len(expected["outputs"])
    if len(windows) != n_expected:
        problems.append(f"{len(windows)} windows ran, {n_expected} expected")

    model = uart.get("model", {})
    if model and model.get("bytes") != expected["model"]["bytes"]:
        problems.append(
            f"the firmware embeds a {model.get('bytes')}-byte model but expected.json "
            f"was made from a {expected['model']['bytes']}-byte one; regenerate the vectors"
        )

    outputs = compare(windows, expected["outputs"], args.tolerance)
    if outputs["mismatched_windows"]:
        problems.append(
            f"outputs differ from the reference beyond {args.tolerance} LSB on windows "
            f"{outputs['mismatched_windows']}"
        )

    cycles = [w["cycles"] for w in windows]
    flash_used = sum(sizes.get(s, 0) for s in FLASH_SECTIONS)
    ram_used = sum(sizes.get(s, 0) for s in RAM_SECTIONS)
    mean = sum(cycles) / len(cycles) if cycles else 0.0

    report = {
        "emulated": True,
        "disclaimer": (
            "Emulator estimates, not hardware measurements. Renode retires one "
            "instruction per cycle and does not model pipeline stalls, multi-cycle "
            "instructions or flash wait states, so a real STM32F411 will take longer. "
            "Memory figures are exact for this build."
        ),
        "target": {
            "name": "STM32F411 memory map (Cortex-M4F, 100 MHz, 512 KiB flash, 128 KiB RAM)",
            "platform": "firmware/renode/stm32f411.repl",
            "linker_script": "firmware/src/stm32f411.ld",
            "clock_mhz": CLOCK_MHZ,
        },
        "toolchain": {
            "emulator": args.renode_version,
            "compiler": args.gcc_version,
            "cmake": args.cmake_version,
            "tflite_micro_commit": args.tflm_commit,
            "kernels": "CMSIS-NN",
        },
        "model": {
            "path": expected["model"]["path"],
            "bytes": expected["model"]["bytes"],
            "sha256": expected["model"]["sha256"],
            "synthetic_weights": True,
        },
        "memory": {
            "flash_bytes": FLASH_BYTES,
            "flash_used_bytes": flash_used,
            "ram_bytes": RAM_BYTES,
            "ram_used_bytes": ram_used,
            "ram_used_includes": "statics, the tensor arena, a 4 KiB heap and an 8 KiB stack",
            "arena_reserved_bytes": uart.get("arena", {}).get("size"),
            "arena_used_bytes": uart.get("arena", {}).get("used"),
            "fits_flash": flash_used <= FLASH_BYTES,
            "fits_ram": ram_used <= RAM_BYTES,
        },
        "cycles": {
            "label": "emulator estimate",
            "windows": len(cycles),
            "mean": mean,
            "min": min(cycles) if cycles else None,
            "max": max(cycles) if cycles else None,
            "estimated_inference_ms_at_clock": mean / (CLOCK_MHZ * 1e3) if cycles else None,
            "per_window": cycles,
        },
        "outputs": {
            "reference": expected["reference"]["runtime"] + ", " + expected["reference"]["kernels"],
            **outputs,
        },
    }
    return report, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uart", required=True)
    parser.add_argument("--size", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tolerance", type=int, default=0, help="allowed |diff| in int8 LSB")
    parser.add_argument("--renode-version", default="unknown")
    parser.add_argument("--gcc-version", default="unknown")
    parser.add_argument("--cmake-version", default="unknown")
    parser.add_argument("--tflm-commit", default="unknown")
    args = parser.parse_args(argv)

    report, problems = build_report(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    memory, cycles, outputs = report["memory"], report["cycles"], report["outputs"]
    print(
        f"flash {memory['flash_used_bytes']} / {memory['flash_bytes']} bytes, "
        f"RAM {memory['ram_used_bytes']} / {memory['ram_bytes']} bytes, "
        f"arena used {memory['arena_used_bytes']} bytes"
    )
    if cycles["windows"]:
        print(
            f"EMULATOR ESTIMATE: {cycles['mean']:.0f} cycles per inference "
            f"({cycles['estimated_inference_ms_at_clock']:.2f} ms at {CLOCK_MHZ:.0f} MHz)"
        )
    print(
        f"outputs: {outputs['bit_exact']}/{outputs['compared']} bit-exact, "
        f"max |diff| {outputs['max_abs_diff_lsb']} LSB, "
        f"top class agrees on {outputs['top_class_agrees']}"
    )
    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
