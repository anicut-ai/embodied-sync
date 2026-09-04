"""Committed-fixture guard: ``data/fixtures/synth_mini_aligned`` (session 14).

Sibling of ``tests/test_fixture_synth_mini.py``. Where that fixture pins
the on-disk *run* format v0 (D-0005), this one pins the on-disk *aligned
episode* format v0 (D-0021). Regenerate with:

    embsync align data/fixtures/synth_mini \\
        --out data/fixtures/synth_mini_aligned \\
        --target-rate-hz 10.0

(from the repo root — pytest's rootdir is the same directory, so the
relative ``data/fixtures/synth_mini`` source path resolves both under
manual regeneration and under the regeneration test.)

If either the alignment engine (D-0020) or the episode format
(D-0021) changes, the byte-identity test fails and the diff must be
made deliberately. Change ``data/fixtures/README.md`` in the same
PR so the regeneration command matches the new expected bytes.
"""

from __future__ import annotations

from pathlib import Path

import json

from embodied_sync.align import align_run
from embodied_sync.cli.main import main
from embodied_sync.datasets.io import load_episode, load_run
from embodied_sync.reports import build_report

FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "fixtures"
    / "synth_mini_aligned"
)
SOURCE_RUN = Path("data") / "fixtures" / "synth_mini"


def test_fixture_loads_and_matches_engine() -> None:
    """Loaded episode matches a fresh ``align_run`` on the source run."""
    loaded = load_episode(FIXTURE_DIR)
    expected = align_run(load_run(SOURCE_RUN), target_rate_hz=10.0)
    assert loaded == expected


def test_fixture_is_byte_identical_to_regeneration(tmp_path: Path) -> None:
    """CLI-driven regeneration reproduces every legacy committed byte.

    Requires pytest's rootdir to be the repository root so the relative
    source-run path resolves. The rootdir is set implicitly by
    ``pyproject.toml`` at repo root (``testpaths = ["tests"]``).

    The fixture intentionally predates deterministic replay provenance and
    remains a compatibility pin for episodes without that additive manifest
    block. New CLI output must match after removing only ``provenance``;
    ``tests/test_provenance.py`` pins the new block itself.
    """
    regen_dir = tmp_path / "synth_mini_aligned"
    regen_args = [
        "align",
        str(SOURCE_RUN),
        "--out",
        str(regen_dir),
        "--target-rate-hz",
        "10.0",
    ]
    assert main(regen_args) == 0

    fixture_files = sorted(
        p.relative_to(FIXTURE_DIR) for p in FIXTURE_DIR.rglob("*.json*")
    )
    regen_files = sorted(
        p.relative_to(regen_dir) for p in regen_dir.rglob("*.json*")
    )
    assert fixture_files == regen_files
    for rel in fixture_files:
        fixture_bytes = (FIXTURE_DIR / rel).read_bytes()
        regenerated_bytes = (regen_dir / rel).read_bytes()
        if rel == Path("manifest.json"):
            regenerated_manifest = json.loads(regenerated_bytes)
            assert "provenance" in regenerated_manifest
            regenerated_manifest.pop("provenance")
            regenerated_bytes = (
                json.dumps(regenerated_manifest, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
        assert fixture_bytes == regenerated_bytes, (
            f"format drift in {rel}: committed fixture differs from regeneration "
            f"(see data/fixtures/README.md)"
        )


def test_fixture_manifest_carries_median_skew_ns_field() -> None:
    """Session-13 additive field survives on the committed fixture.

    Pins that ``AlignmentReport.median_skew_ns`` (D-0024-additive) is
    populated by ``align_run`` and echoed by ``save_episode`` — a
    regression here would mean either the engine stopped populating
    the field or ``save_episode`` stopped writing it.
    """
    loaded = load_episode(FIXTURE_DIR)
    assert loaded.report.median_skew_ns
    # Regular streams are grid-aligned at 10 Hz → skew 0.
    for name in (
        "actions",
        "audio",
        "cam_front",
        "cam_wrist",
        "robot_state",
        "tactile",
    ):
        assert loaded.report.median_skew_ns[name] == 0
    # Events stream is irregular; still populated.
    assert loaded.report.median_skew_ns["events"] is not None


def test_fixture_three_way_median_skew_ns_cross_check() -> None:
    """Three-way pin on the persisted median_skew_ns contract.

    All three surfaces target the same numbers via independent
    code paths on the committed fixture:

    - ``SyncQualityReport.streams[i].median_skew_ns`` — computed
      from the frames' metadata via ``statistics.median``
      (``embodied_sync/reports/sync_quality.py::build_report``).
    - ``AlignedRun.report.median_skew_ns[name]`` — populated at
      ``align_run`` time via
      ``embodied_sync/align/engine.py::_median_skew_ns_by_stream``,
      round-tripped through ``load_episode`` from the manifest.
    - ``manifest["median_skew_ns"][name]`` — written by
      ``embodied_sync/datasets/io.py::save_episode`` from the
      typed report, then read back from disk directly.

    A regression in any one path shows up as a mismatch here on the
    committed clean-run fixture; the sibling test
    ``tests/test_reports_sync_quality.py::TestBuildReport::
    test_median_skew_matches_alignment_report_median_skew`` runs the
    same invariant against a corrupted synth run.
    """
    loaded = load_episode(FIXTURE_DIR)
    report = build_report(loaded)
    manifest = json.loads(
        (FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    manifest_median = manifest["median_skew_ns"]
    for stats in report.streams:
        typed_value = loaded.report.median_skew_ns[stats.name]
        raw_value = manifest_median[stats.name]
        assert stats.median_skew_ns == typed_value == raw_value, (
            f"{stats.name}: report={stats.median_skew_ns}, "
            f"typed={typed_value}, manifest={raw_value} — the three "
            f"median_skew_ns code paths diverged."
        )
