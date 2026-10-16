"""The Cortex-M firmware's inputs, its reference outputs and its report.

The firmware itself is cross-compiled and run under Renode by the ``firmware``
CI job (``firmware/scripts/build_and_run.sh``), which needs the Arm GNU
Toolchain and the emulator. These tests hold the other end of that chain on
any machine with the project's Python dependencies:

* the committed reference outputs are still what the desktop TFLite
  interpreter computes for the committed model and windows, so the CI job is
  comparing the firmware against the interpreter and not against a stale file;
* the firmware's model has the shipped architecture, so the memory and cycle
  figures it produces describe the model the README is about;
* the committed emulator report says what the README says it says and is
  labelled as an emulator estimate.
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from src.config import load_config, resolve
from src.embedded import firmware_vectors as fv

REPORT = "benchmarks/renode_cortex_m4.json"


def _renode_report_module():
    path = resolve("firmware/tools/renode_report.py")
    spec = importlib.util.spec_from_file_location("renode_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def expected() -> dict:
    return json.loads(resolve(fv.EXPECTED_PATH).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(resolve(REPORT).read_text(encoding="utf-8"))


class TestReferenceOutputs:
    def test_the_digests_match_the_committed_files(self, expected):
        assert fv.sha256(resolve(fv.MODEL_PATH)) == expected["model"]["sha256"]
        assert fv.sha256(resolve(fv.WINDOWS_PATH)) == expected["windows"]["sha256"]

    def test_the_interpreter_still_computes_the_committed_outputs(self, expected):
        quant = fv.quantisation(resolve(fv.MODEL_PATH))
        windows = fv.load_windows(resolve(fv.WINDOWS_PATH), quant["input_shape"])
        outputs = fv.reference_outputs(resolve(fv.MODEL_PATH), windows, reference=True)
        assert outputs.tolist() == expected["outputs"]

    def test_every_class_is_represented(self, expected):
        assert sorted(set(expected["labels"])) == list(range(18))
        assert len(expected["outputs"]) == len(expected["labels"])


class TestFirmwareModelIsTheShippedArchitecture:
    def test_same_operators_and_shapes_as_the_db5_export(self):
        from src.quantise.tflite_export import operators

        db5 = json.loads(resolve("results/db5/export_report.json").read_text(encoding="utf-8"))
        shipped = db5["artefacts"]["tflite_int8"]
        assert operators(resolve(fv.MODEL_PATH)) == shipped["operators"]
        quant = fv.quantisation(resolve(fv.MODEL_PATH))
        assert quant["input_shape"] == shipped["input"]["shape"]
        assert quant["output_shape"] == shipped["output"]["shape"]
        assert resolve(fv.MODEL_PATH).stat().st_size == shipped["bytes"]

    def test_the_firmware_config_differs_from_the_default_only_in_data_and_schedule(self):
        default, firmware = load_config(), load_config(resolve("config/firmware.yaml"))
        for field in ("conv_channels", "kernel_electrodes", "kernel_time", "dense_units"):
            assert getattr(firmware.train, field) == getattr(default.train, field), field
        assert firmware.window.length_ms == default.window.length_ms
        assert firmware.dataset.source == "synthetic"


class TestRenodeReportParser:
    LOG = "\n".join(
        [
            "boot noise",
            "SEMG BEGIN",
            "SEMG MODEL bytes=100 windows=2 input=4 output=3",
            "SEMG ARENA size=4096 used=1234",
            "SEMG W 0 cycles=1000 top=2 out=-5,0,7",
            "SEMG W 1 cycles=1100 top=0 out=9,1,-128",
            "SEMG END status=ok",
        ]
    )

    def test_records_are_parsed(self):
        parsed = _renode_report_module().parse_uart(self.LOG)
        assert parsed["model"] == {"bytes": 100, "windows": 2, "input": 4, "output": 3}
        assert parsed["arena"] == {"size": 4096, "used": 1234}
        assert [w["cycles"] for w in parsed["windows"]] == [1000, 1100]
        assert parsed["windows"][1]["out"] == [9, 1, -128]
        assert parsed["end"] == {"status": "ok"}

    def test_a_one_lsb_difference_fails_a_bit_exact_check(self):
        module = _renode_report_module()
        windows = module.parse_uart(self.LOG)["windows"]
        result = module.compare(windows, [[-5, 0, 7], [9, 2, -128]], tolerance=0)
        assert result["mismatched_windows"] == [1]
        assert result["max_abs_diff_lsb"] == 1
        assert result["bit_exact"] == 1
        loose = module.compare(windows, [[-5, 0, 7], [9, 2, -128]], tolerance=1)
        assert loose["mismatched_windows"] == []

    def test_section_sizes_are_read(self):
        sizes = _renode_report_module().parse_size(
            "section   size   addr\n.text   1000   134217728\n.bss   200   536870912\n"
        )
        assert sizes == {".text": 1000, ".bss": 200}


class TestCommittedEmulatorReport:
    def test_it_is_labelled_an_estimate(self, report):
        assert report["emulated"] is True
        assert report["cycles"]["label"] == "emulator estimate"
        assert "not hardware measurements" in report["disclaimer"]

    def test_it_was_produced_from_the_committed_model(self, report, expected):
        assert report["model"]["sha256"] == expected["model"]["sha256"]
        assert report["outputs"]["compared"] == len(expected["outputs"])

    def test_every_output_matched_the_interpreter(self, report):
        outputs = report["outputs"]
        assert outputs["mismatched_windows"] == []
        assert outputs["max_abs_diff_lsb"] <= outputs["tolerance_lsb"]

    def test_the_build_fits_an_stm32f411(self, report):
        memory = report["memory"]
        assert memory["ram_bytes"] == 128 * 1024
        assert memory["flash_bytes"] == 512 * 1024
        assert memory["fits_ram"] and memory["fits_flash"]
        assert memory["arena_used_bytes"] <= memory["arena_reserved_bytes"]

    def test_the_cycle_summary_matches_the_per_window_counts(self, report):
        cycles = report["cycles"]
        per_window = cycles["per_window"]
        assert cycles["windows"] == len(per_window)
        assert cycles["mean"] == pytest.approx(np.mean(per_window))
        assert cycles["min"] == min(per_window) and cycles["max"] == max(per_window)
