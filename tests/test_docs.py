"""Documentation that has to stay true.

The README's numbers are checked by ``src.check_readme``. These are the
claims that a number checker cannot see: that the diagram in the README is
well-formed and its labels fit inside the boxes they belong to, that the
scripts the docs tell people to run exist, and that the shipped configuration
matches what the docs say it is.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ElementTree

import pytest

from src.config import load_config, resolve

SVG = "{http://www.w3.org/2000/svg}"
BACKSLASH = chr(92)
DIAGRAM = "docs/pipeline.svg"


def _text_width(text: str, size: float, bold: bool) -> float:
    """Rough advance width for DejaVu Sans, which is what renders these.

    Deliberately approximate. It exists to catch a label that has grown past
    its box, not to typeset anything.
    """
    return len(text) * size * (0.60 if bold else 0.55)


@pytest.fixture(scope="module")
def diagram():
    return ElementTree.parse(resolve(DIAGRAM)).getroot()


class TestPipelineDiagram:
    def test_it_is_well_formed_svg(self, diagram):
        assert diagram.tag == f"{SVG}svg"
        assert diagram.get("viewBox")

    def test_no_label_overflows_its_box(self, diagram):
        boxes = [
            (float(r.get("x")), float(r.get("y")), float(r.get("width")), float(r.get("height")))
            for r in diagram.iter(f"{SVG}rect")
            if r.get("x") is not None
        ]
        overflows = []
        for node in diagram.iter(f"{SVG}text"):
            x, y = float(node.get("x")), float(node.get("y"))
            size = float(node.get("font-size", 12))
            bold = node.get("font-weight") in {"600", "700", "bold"}
            end = x + _text_width(node.text or "", size, bold)
            for bx, by, bw, bh in boxes:
                # The full-canvas background rect is not a label container.
                if bw >= 500:
                    continue
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    if end > bx + bw - 4:
                        overflows.append((node.text, end, bx + bw))
                    break
        assert not overflows, f"labels running past their boxes: {overflows}"

    def test_nothing_runs_off_the_canvas(self, diagram):
        canvas = float(diagram.get("viewBox").split()[2])
        for node in diagram.iter(f"{SVG}text"):
            end = float(node.get("x")) + _text_width(
                node.text or "",
                float(node.get("font-size", 12)),
                node.get("font-weight") in {"600", "700", "bold"},
            )
            assert end <= canvas - 4, f"{node.text!r} ends at {end:.0f}, canvas is {canvas}"

    def test_the_latency_bar_is_drawn_to_scale(self, diagram):
        """The window-fill block has to be 200/300 of the track, or it lies."""
        config = load_config()
        rects = [r for r in diagram.iter(f"{SVG}rect") if r.get("x") is not None]
        track = next(r for r in rects if float(r.get("width")) == 980.0)
        window = next(r for r in rects if float(r.get("width")) == 653.0)
        expected = float(track.get("width")) * (
            config.window.length_ms / config.latency.budget_ms
        )
        assert float(window.get("width")) == pytest.approx(expected, abs=2.0)


class TestDocsMatchTheConfiguration:
    def test_the_diagram_quotes_the_shipped_geometry(self, diagram):
        config = load_config()
        labels = " ".join(node.text or "" for node in diagram.iter(f"{SVG}text"))
        assert f"{config.window.length_ms:.0f} ms" in labels
        assert f"{config.window.stride_ms:.0f} ms stride" in labels
        assert f"{config.latency.budget_ms:.0f} ms" in labels
        if config.evaluation.vote_window == 1:
            assert "no majority vote" in labels

    def test_the_model_card_quotes_the_derived_architecture(self):
        from src.models.architecture import ArchitectureSpec

        config = load_config()
        spec = ArchitectureSpec.from_config(config.train, 16, config.window_samples, 18)
        card = resolve("docs/MODEL_CARD.md").read_text(encoding="utf-8")
        assert f"{spec.parameter_count():,}" in card
        assert f"{spec.multiply_accumulates():,}" in card

    def test_every_script_the_docs_name_exists(self):
        named = set()
        for name in ("README.md", "docs/LATENCY_BUDGET.md", "docs/ARCHITECTURE.md"):
            text = resolve(name).read_text(encoding="utf-8")
            named |= set(re.findall(r"(scripts/[\w.\-]+\.sh)", text))
        assert named, "the docs name no scripts at all, which is suspicious"
        missing = [s for s in named if not resolve(s).exists()]
        assert not missing, f"the docs point at scripts that do not exist: {missing}"

    def test_the_dataset_card_matches_the_generated_statistics(self):
        """The card quotes counts. Those counts come from a script, so check them.

        Skipped when the statistics have not been generated, which is the case
        on a checkout that has never seen the data. When they are present, the
        card has to agree with them.
        """
        import json

        stats_path = resolve("data/dataset_stats.json")
        if not stats_path.exists():
            pytest.skip("data/dataset_stats.json not generated on this checkout")
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        card = resolve("data/DATASET_CARD.md").read_text(encoding="utf-8")

        assert f"{stats['windows']:,}" in card
        assert f"{stats['n_classes']}" in card
        assert f"{stats['window_ms']:.0f} ms" in card
        assert f"{stats['stride_ms']:.0f} ms stride" in card
        ages = stats["age_range"]
        assert f"Ages {ages[0]} to {ages[1]}" in card

    def test_no_committed_evidence_came_from_the_synthetic_generator(self):
        """The failure this exists to prevent.

        The synthetic pipeline writes the same file names as the real one. Run
        the smoke pipeline, forget to re-run the real one, and the repository
        now quotes wiring-check numbers as results. Every report carries a
        ``synthetic`` flag for exactly this reason, and the committed ones must
        all have it false.
        """
        import json

        offenders = []
        candidates = [
            *sorted(resolve("benchmarks").glob("*.json")),
            *sorted(resolve("results").glob("db5/*.json")),
            *sorted(resolve("experiments/results").glob("db5/*.json")),
        ]
        for path in candidates:
            # The synthetic outputs are gitignored by name; the point here is
            # the ones that are not.
            if "synthetic" in path.name or "synthetic" in path.parent.name:
                continue
            blob = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(blob, dict) and blob.get("synthetic") is True:
                offenders.append(str(path.relative_to(resolve("."))))
        assert not offenders, (
            "these committed files were produced by the synthetic generator and "
            f"would be read as results: {offenders}"
        )

    def test_every_committed_result_came_from_the_same_configuration(self):
        """All the result files have to be from one run, or the table is a mixture.

        Each scored run records the full config it was produced under. If two
        of them disagree, the results directory holds output from two different
        runs and the comparison across models in the README is not a
        comparison. This is not hypothetical: a training process that survived
        being cancelled once wrote half of this directory from an older
        revision of the code, and the timestamps were the only clue.
        """
        import json

        directory = resolve("results") / "db5"
        if not directory.exists():
            pytest.skip("no db5 results on this checkout")
        runs = {
            path.name: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("*__*.json"))
        }
        if len(runs) < 2:
            pytest.skip("need at least two results to compare")

        # The training seed, the window geometry and the preprocessing must be
        # identical. The model and protocol are of course meant to differ.
        def fingerprint(run):
            config = run["config"]
            return {
                "dataset": config["dataset"],
                "preprocessing": config["preprocessing"],
                "window": config["window"],
                "features": config["features"],
                "evaluation": config["evaluation"],
                "train": {
                    k: v for k, v in config["train"].items() if k != "device"
                },
            }

        reference_name, reference = next(iter(runs.items()))
        expected = fingerprint(reference)
        mismatched = [
            name for name, run in runs.items() if fingerprint(run) != expected
        ]
        assert not mismatched, (
            f"these results were produced under a different configuration from "
            f"{reference_name}: {mismatched}"
        )

    def test_every_committed_result_came_from_the_same_source_revision(self):
        """Same config is not enough; the code has to match too.

        Each result records a digest of ``src/``. Two results with the same
        config and different digests were produced by different code, which is
        exactly the failure this repository hit once: a cancelled training
        process kept running and wrote half a results directory from an older
        revision whose batch-norm configuration differed by a factor of ten.
        The config files were identical and said nothing about it.
        """
        import json

        directory = resolve("results") / "db5"
        if not directory.exists():
            pytest.skip("no db5 results on this checkout")
        fingerprints = {}
        for path in sorted(directory.glob("*__*.json")):
            blob = json.loads(path.read_text(encoding="utf-8"))
            fingerprints[path.name] = blob.get("source_fingerprint")
        if len(fingerprints) < 2:
            pytest.skip("need at least two results to compare")

        unstamped = [name for name, f in fingerprints.items() if f is None]
        if unstamped:
            pytest.skip(
                "some results predate the source fingerprint and cannot be "
                f"checked this way: {unstamped}. Their provenance was verified "
                "by modification time instead. Re-run the training to stamp them."
            )

        distinct = set(fingerprints.values())
        assert len(distinct) == 1, (
            "these results came from more than one revision of src/: "
            f"{fingerprints}"
        )

    def test_every_committed_result_saw_the_same_dataset(self):
        """Same windows, same classes, same subjects, across every result."""
        import json

        directory = resolve("results") / "db5"
        if not directory.exists():
            pytest.skip("no db5 results on this checkout")
        summaries = {
            path.name: json.loads(path.read_text(encoding="utf-8"))["dataset"]
            for path in sorted(directory.glob("*__*.json"))
        }
        if len(summaries) < 2:
            pytest.skip("need at least two results to compare")
        reference_name, reference = next(iter(summaries.items()))
        mismatched = [n for n, d in summaries.items() if d != reference]
        assert not mismatched, (
            f"these results were scored on a different dataset build from "
            f"{reference_name}: {mismatched}"
        )

    def test_committed_evidence_records_no_machine_specific_paths(self):
        """These files are read on other machines, including by CI on Linux.

        An absolute Windows path in a committed JSON is both machine-specific
        and unparseable by ``pathlib`` on a POSIX host, where it comes back as
        one long filename and a lookup by basename silently misses. That is not
        hypothetical: it blanked the latency columns in the README's
        quantisation table on the first CI run.
        """
        import json

        offenders = []
        for path in sorted(resolve("benchmarks").glob("*.json")):
            text = path.read_text(encoding="utf-8")
            blob = json.loads(text)
            if isinstance(blob, dict) and blob.get("synthetic") is True:
                continue
            if BACKSLASH in text or ":/" in text.replace("://", ""):
                offenders.append(path.name)
        assert not offenders, (
            "these committed files contain machine-specific paths: " f"{offenders}"
        )

    def test_no_em_or_en_dashes_in_the_prose(self):
        """House style. Easy to reintroduce by accident, so it is checked."""
        offenders = []
        for path in sorted(resolve(".").rglob("*.md")):
            if any(p in {".venv", "artifacts", "analysis"} for p in path.parts):
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if "—" in line or "–" in line:
                    offenders.append(f"{path.name}:{number}")
        assert not offenders, f"em or en dashes found at: {offenders}"
