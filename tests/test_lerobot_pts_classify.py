"""Classification arithmetic for the LeRobot PTS audit (C4, D-0043).

These tests build :class:`VideoEvidence` directly rather than encoding video,
which is the point of keeping the rules pure: the #2364 signature lives an
hour into a recording and the #3177 signature needs many episodes, and
neither is worth minutes of H.264 to reproduce. The evidence type is bounded
by episode count, not frame count, so these cases stay cheap.

No optional dependency is imported here — the module under test only reaches
for ``pyarrow``/``av`` inside :func:`audit_lerobot_dataset`.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodied_sync.inspect.lerobot_pts import (
    DEFAULT_TOLERANCE_S,
    VideoEvidence,
    _read_pts_timeline,
    classify_video,
    float32_ulp_ns,
    legacy_float32_residual_ns,
    precision_band_ns,
)

TOLERANCE_NS = 100_000  # 1e-4 s, LeRobot's default
FRAME_NS = round(1e9 / 30)


def _evidence(**overrides: object) -> VideoEvidence:
    """Evidence for a clean 24-frame video, before any override."""
    base: dict[str, object] = {
        "tolerance_ns": TOLERANCE_NS,
        "frame_duration_ns": FRAME_NS,
        "timeline_length": 24,
        "expected_frame_count": 24,
        "query_count": 24,
        "violation_count": 0,
        "legacy_violation_count": 0,
        "out_of_range_count": 0,
    }
    base.update(overrides)
    return VideoEvidence(**base)  # type: ignore[arg-type]


def _violations(
    count: int,
    *,
    frame_steps: int = 0,
    precision: int = 0,
    episode_means: tuple[int, ...] = (),
    ulp_ns: int = 0,
) -> VideoEvidence:
    return _evidence(
        violation_count=count,
        frame_step_violation_count=frame_steps,
        precision_band_violation_count=precision,
        violating_episode_means_ns=episode_means,
        max_float32_ulp_ns=ulp_ns,
    )


# ------------------------------------------------------------- precision


def test_default_tolerance_matches_lerobot() -> None:
    assert DEFAULT_TOLERANCE_S == 1e-4
    assert round(DEFAULT_TOLERANCE_S * 1e9) == TOLERANCE_NS


def test_float32_spacing_matches_the_reported_magnitudes() -> None:
    """Near 1025 s a float32 timeline steps in ~122 us, per #2364."""
    assert float32_ulp_ns(1025.0) == pytest.approx(122_070, rel=1e-3)
    # Near the start of a recording the same timeline is ~60 ns coarse, so
    # the same tolerance means something completely different there.
    assert float32_ulp_ns(1.0) == pytest.approx(119, rel=1e-2)
    assert float32_ulp_ns(0.0) == 0


def test_precision_band_tracks_tolerance_and_float32_spacing() -> None:
    """The band is never allowed to grow with the frame period."""
    # Early in a recording: twice the tolerance, not half a frame (16.7 ms).
    assert precision_band_ns(TOLERANCE_NS, float32_ulp_ns(1.0)) == 2 * TOLERANCE_NS
    # An hour in, float32 spacing dominates and the band widens honestly.
    band = precision_band_ns(TOLERANCE_NS, float32_ulp_ns(1025.0))
    assert 200_000 < band < 250_000


def test_ten_millisecond_error_is_not_precision_noise() -> None:
    """The regression this rule exists for: 100x tolerance, at 30 Hz.

    Half a frame would have swallowed everything up to 16.7 ms, filing real
    offsets under "precision" — exactly the bugs the audit is meant to find.
    """
    ten_ms = _violations(1, precision=0, ulp_ns=float32_ulp_ns(1.0))
    assert 10_000_000 > precision_band_ns(TOLERANCE_NS, ten_ms.max_float32_ulp_ns)
    assert classify_video(ten_ms) == "unclassified"


def test_one_nanosecond_over_the_tolerance_is_a_precision_boundary() -> None:
    assert classify_video(_violations(1, precision=1)) == "precision_boundary"


def test_hour_scale_boundary_violation_is_a_precision_boundary() -> None:
    """#2364/#2814 values: over tolerance, but inside float32's own spacing."""
    ulp = float32_ulp_ns(1025.0)
    assert classify_video(_violations(4, precision=4, ulp_ns=ulp)) == (
        "precision_boundary"
    )


# ------------------------------------------------------------ legacy path


def test_legacy_float32_residual_reproduces_the_2364_magnitude() -> None:
    """An exact residual well under tolerance, rejected by float32 code.

    At 1025 s the float32 grid is ~122 us, so a query and a PTS that differ
    by 93.75 us exactly can land a whole quantum apart once both are
    rounded to float32 — over the tolerance, for a dataset that is fine.
    """
    query_s = 1025.0
    selected_pts_ns = round(query_s * 1e9) + 93_750
    exact = selected_pts_ns - round(query_s * 1e9)
    legacy = legacy_float32_residual_ns(query_s, selected_pts_ns)

    assert exact == 93_750
    assert exact < TOLERANCE_NS  # current LeRobot accepts it
    assert abs(legacy) > TOLERANCE_NS  # older LeRobot does not
    assert abs(legacy) == pytest.approx(122_070, rel=1e-3)


def test_legacy_only_violation_is_named_rather_than_reported_clean() -> None:
    """Answering #2814 with "works for me" is the failure mode here."""
    evidence = _evidence(violation_count=0, legacy_violation_count=6)
    assert classify_video(evidence) == "legacy_precision_false_positive"


def test_exact_violation_outranks_a_legacy_one() -> None:
    evidence = replace(_violations(3, precision=3), legacy_violation_count=3)
    assert classify_video(evidence) == "precision_boundary"


# -------------------------------------------------------- episode offsets


def test_growing_episode_offsets_are_cumulative_drift() -> None:
    """#3177: each episode's declared offset drifts further than the last."""
    evidence = _evidence(
        boundary_offsets_ns=(0, 20_000, 900_000, 3_000_000),
        boundary_fault_count=2,
    )
    assert classify_video(evidence) == "cumulative_episode_offset"


def test_a_single_whole_frame_episode_offset_is_a_frame_step() -> None:
    """One episode pointing a frame into the video is not a trend."""
    evidence = _evidence(
        boundary_offsets_ns=(0, FRAME_NS, 0),
        boundary_fault_count=1,
    )
    assert classify_video(evidence) == "frame_step_offset"


def test_uniform_whole_frame_offset_is_a_frame_step_not_drift() -> None:
    evidence = _evidence(
        boundary_offsets_ns=(-FRAME_NS, -FRAME_NS, -FRAME_NS),
        boundary_fault_count=3,
    )
    assert classify_video(evidence) == "frame_step_offset"


def test_boundary_faults_outrank_residual_shape() -> None:
    """An offset that leaves no residual must still decide the verdict."""
    evidence = _evidence(
        boundary_offsets_ns=(0, FRAME_NS, 0),
        boundary_fault_count=1,
        violation_count=0,
    )
    assert classify_video(evidence) == "frame_step_offset"


def test_unrecognized_boundary_offset_is_unclassified() -> None:
    evidence = _evidence(
        boundary_offsets_ns=(0, round(0.4 * FRAME_NS), 0),
        boundary_fault_count=1,
    )
    assert classify_video(evidence) == "unclassified"


# ------------------------------------------------------------ frame count


def test_short_video_is_missing_or_extra_frames() -> None:
    assert classify_video(_evidence(timeline_length=12)) == "missing_or_extra_frames"


def test_extra_frames_are_detected_without_an_out_of_range_index() -> None:
    """25 encoded frames against 24 declared rows never goes out of range."""
    evidence = _evidence(timeline_length=25, out_of_range_count=0)
    assert classify_video(evidence) == "missing_or_extra_frames"


def test_frame_count_fault_outranks_every_other_signature() -> None:
    """Residuals computed against a clamped index are artifacts, not data."""
    evidence = _evidence(
        timeline_length=12,
        out_of_range_count=6,
        violation_count=6,
        precision_band_violation_count=6,
        boundary_offsets_ns=(0, 900_000),
        boundary_fault_count=1,
    )
    assert classify_video(evidence) == "missing_or_extra_frames"


# -------------------------------------------------------------- residuals


def test_growing_residuals_across_episodes_are_cumulative_drift() -> None:
    evidence = _violations(
        16, episode_means=(140_000, 400_000, 900_000), precision=16
    )
    assert classify_video(evidence) == "cumulative_episode_offset"


def test_whole_frame_residuals_are_a_frame_step() -> None:
    assert classify_video(_violations(4, frame_steps=4)) == "frame_step_offset"


def test_mixed_residual_shapes_fall_back_to_unclassified() -> None:
    """Half the violations look like frames, half do not: no verdict."""
    assert classify_video(_violations(4, frame_steps=2, precision=1)) == "unclassified"


def test_clean_evidence_is_within_tolerance() -> None:
    assert classify_video(_evidence()) == "within_tolerance"


# ------------------------------------------------- packet/frame cardinality


class _FakeStream:
    """Just enough of a PyAV video stream to exercise the timeline guard."""

    def __init__(self, declared_frames: int) -> None:
        self.time_base = Fraction(1, 30)
        self.average_rate = Fraction(30, 1)
        self.frames = declared_frames


class _FakeContainer:
    def __init__(self, declared_frames: int, packet_count: int) -> None:
        self.streams = SimpleNamespace(video=[_FakeStream(declared_frames)])
        self._packets = [SimpleNamespace(pts=index) for index in range(packet_count)]

    def demux(self, stream: object) -> list[object]:
        return list(self._packets)

    def __enter__(self) -> "_FakeContainer":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _stub_av(monkeypatch: pytest.MonkeyPatch, declared: int, packets: int) -> None:
    container = _FakeContainer(declared, packets)
    monkeypatch.setitem(
        sys.modules, "av", SimpleNamespace(open=lambda *a, **k: container)
    )


def test_packet_count_matching_the_declared_frames_is_verified(monkeypatch) -> None:
    _stub_av(monkeypatch, declared=10, packets=10)
    timeline, average_fps, verified = _read_pts_timeline(Path("stub.mp4"))

    assert len(timeline) == 10
    assert average_fps == 30.0
    assert verified


def test_packet_frame_mismatch_is_refused_rather_than_audited(monkeypatch) -> None:
    """A codec emitting several frames per packet would silently shift every
    index, so an unverifiable timeline must not be quietly used."""
    _stub_av(monkeypatch, declared=10, packets=7)

    with pytest.raises(ValueError, match="packets and frames"):
        _read_pts_timeline(Path("stub.mp4"))


def test_container_declaring_no_frame_count_is_reported_unverified(monkeypatch) -> None:
    _stub_av(monkeypatch, declared=0, packets=7)
    _, _, verified = _read_pts_timeline(Path("stub.mp4"))

    assert not verified
