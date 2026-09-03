"""End-to-end LeRobot PTS audit against real encoded video (C4, D-0043).

Everything here is generated locally: a LeRobot-v3-shaped dataset whose
metadata, Parquet rows and MP4 files are written in the test, then corrupted
in one specific way per case. The point is that each corruption has a *known*
cause, so the audit is checked for naming the right one rather than merely
noticing that something is wrong.

Marked ``optional_dep``: needs ``pyarrow`` and ``av`` (the ``lerobot`` extra).
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from embodied_sync.cli.main import main

FPS = 30
FRAME_NS = round(1e9 / FPS)


def _write_video(
    path: Path, frame_count: int, *, fps: int = FPS, pts_shift_frames: int = 0
) -> None:
    """Encode a real video, optionally with its whole PTS timeline shifted.

    ``pts_shift_frames`` reproduces #3177's shape: the encoded frames sit a
    whole frame later than the grid the Parquet timestamps assume, while the
    container's average rate stays exactly ``fps`` — so the decoder's index
    projection is unaffected and only the PTS comparison notices.
    """
    import av
    import numpy as np
    from fractions import Fraction

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.codec_context.time_base = Fraction(1, fps)
        stream.time_base = Fraction(1, fps)
        for index in range(frame_count):
            frame = av.VideoFrame.from_ndarray(
                np.full((48, 64, 3), (index * 17) % 255, dtype=np.uint8), format="rgb24"
            )
            frame.pts = index + pts_shift_frames
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _build_dataset(
    root: Path,
    *,
    episode_lengths: Sequence[int] = (8, 8, 8),
    cameras: Sequence[str] = ("observation.images.top",),
    from_timestamp_offsets_s: Sequence[float] | None = None,
    encoded_frames: int | None = None,
    pts_shift_frames: int = 0,
    codebase_version: str = "v3.0",
) -> Path:
    """A minimal but structurally real LeRobot v3 dataset.

    All episodes of a camera share one physical video, which is what makes
    ``from_timestamp`` load-bearing and what the audit has to group by.
    ``from_timestamp_offsets_s`` perturbs each episode's declared video
    offset; ``encoded_frames`` under- or over-fills the video itself.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    offsets = list(from_timestamp_offsets_s or [0.0] * len(episode_lengths))
    total_frames = sum(episode_lengths)

    timestamps: list[float] = []
    frame_indices: list[int] = []
    episode_indices: list[int] = []
    episode_rows: list[dict[str, Any]] = []
    frames_before = 0
    for episode_index, length in enumerate(episode_lengths):
        timestamps.extend(index / FPS for index in range(length))
        frame_indices.extend(range(length))
        episode_indices.extend([episode_index] * length)
        row: dict[str, Any] = {
            "episode_index": episode_index,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": frames_before,
            "dataset_to_index": frames_before + length,
            "length": length,
            "tasks": ["synthetic audit fixture"],
        }
        for camera in cameras:
            row[f"videos/{camera}/chunk_index"] = 0
            row[f"videos/{camera}/file_index"] = 0
            row[f"videos/{camera}/from_timestamp"] = (
                frames_before / FPS + offsets[episode_index]
            )
        episode_rows.append(row)
        frames_before += length

    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                # float32 is what LeRobot stores; the audit must cope with
                # its quantization rather than assume float64 inputs.
                "timestamp": pa.array(timestamps, type=pa.float32()),
                "frame_index": pa.array(frame_indices, type=pa.int64()),
                "episode_index": pa.array(episode_indices, type=pa.int64()),
                "index": pa.array(range(total_frames), type=pa.int64()),
            }
        ),
        data_dir / "file-000.parquet",
    )

    episodes_dir = root / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(episode_rows), episodes_dir / "file-000.parquet")
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array([0], type=pa.int64()),
                "__index_level_0__": pa.array(["synthetic audit fixture"], type=pa.string()),
            }
        ),
        root / "meta" / "tasks.parquet",
    )

    features: dict[str, Any] = {
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
    }
    for camera in cameras:
        features[camera] = {"dtype": "video", "shape": [48, 64, 3]}
    info = {
        "codebase_version": codebase_version,
        "robot_type": "synthetic",
        "total_episodes": len(episode_lengths),
        "total_frames": total_frames,
        "fps": FPS,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    for camera in cameras:
        _write_video(
            root / "videos" / camera / "chunk-000" / "file-000.mp4",
            total_frames if encoded_frames is None else encoded_frames,
            pts_shift_frames=pts_shift_frames,
        )
    return root


def _audit(root: Path) -> Any:
    from embodied_sync.inspect.lerobot_pts import audit_lerobot_dataset

    return audit_lerobot_dataset(root)


pytestmark = pytest.mark.optional_dep


@pytest.fixture(autouse=True)
def _require_optional_deps() -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("av")
    pytest.importorskip("numpy")


# ------------------------------------------------------------------ clean


def test_clean_dataset_passes_and_groups_the_shared_video(tmp_path) -> None:
    root = _build_dataset(
        tmp_path / "clean",
        cameras=("observation.images.top", "observation.images.wrist"),
    )
    audit = _audit(root)

    assert audit.passed
    assert audit.total_violations == 0
    assert audit.total_queries == 24 * 2  # 3 episodes x 8 frames x 2 cameras
    assert audit.total_legacy_violations == 0
    # One entry per camera, because both cameras' episodes share one file.
    assert [video.camera_key for video in audit.videos] == [
        "observation.images.top",
        "observation.images.wrist",
    ]
    for video in audit.videos:
        assert video.episode_indices == (0, 1, 2)
        assert video.timeline_length == 24
        assert video.likely_cause == "within_tolerance"
        assert video.first_violation is None
        assert video.evidence.expected_frame_count == 24
        assert video.timeline_verified
        assert all(check.within_tolerance for check in video.boundary_checks)


def test_clean_dataset_exits_zero_and_writes_json(tmp_path, capsys) -> None:
    root = _build_dataset(tmp_path / "clean")
    out = tmp_path / "evidence.json"

    assert main(["audit-lerobot-pts", str(root), "--json", str(out)]) == 0

    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["passed"] is True
    assert document["tolerance_ns"] == 100_000
    assert document["fps"] == 30
    assert "PASS" in capsys.readouterr().out


def test_audit_does_not_modify_the_dataset(tmp_path) -> None:
    """Read-only is a promise, so it is checked rather than asserted in prose."""
    root = _build_dataset(tmp_path / "clean")
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    _audit(root)
    after = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    assert before == after


def test_result_size_tracks_episodes_not_frames(tmp_path) -> None:
    """A finished audit keeps exemplars, never one object per frame.

    Retaining every query made a large dataset's audit cost gigabytes for
    output that quotes aggregates and a handful of samples.
    """
    from embodied_sync.inspect.lerobot_pts import PTSQuery

    root = _build_dataset(
        tmp_path / "big", episode_lengths=(200, 200, 200), cameras=("cam.a", "cam.b")
    )
    audit = _audit(root)
    assert audit.total_queries == 1200

    retained = [
        sample
        for video in audit.videos
        for sample in (
            video.first_violation,
            video.worst_violation,
            video.first_legacy_violation,
        )
        if isinstance(sample, PTSQuery)
    ]
    # At most first/worst/first-legacy per video, whatever the frame count.
    assert len(retained) <= 3 * len(audit.videos)
    # Per-episode trend, not per-frame rows.
    assert all(len(video.episode_trend) == 3 for video in audit.videos)


def test_max_episodes_limits_the_audit(tmp_path) -> None:
    from embodied_sync.inspect.lerobot_pts import audit_lerobot_dataset

    root = _build_dataset(tmp_path / "clean")
    audit = audit_lerobot_dataset(root, max_episodes=2)
    assert audit.audited_episodes == 2
    assert audit.videos[0].episode_indices == (0, 1)


# ------------------------------------------------------------- corrupted


def test_precision_boundary_fixture_names_its_cause(tmp_path, capsys) -> None:
    """Every episode offset by a hair over the tolerance, and no more."""
    root = _build_dataset(
        tmp_path / "boundary",
        from_timestamp_offsets_s=[110e-6, 110e-6, 110e-6],
    )
    out = tmp_path / "evidence.json"
    assert main(["audit-lerobot-pts", str(root), "--json", str(out)]) == 1

    document = json.loads(out.read_text(encoding="utf-8"))
    video = document["videos"][0]
    assert video["likely_cause"] == "precision_boundary"
    assert video["violation_count"] > 0
    first = video["first_violation"]
    assert first["episode_index"] == 0
    assert 100_000 < first["abs_residual_ns"] < 200_000
    assert first["video_path"].endswith("file-000.mp4")
    assert "likely_cause: precision_boundary" in capsys.readouterr().out


def test_cumulative_drift_fixture_names_the_first_affected_episode(tmp_path) -> None:
    """#3177's shape: episode 0 fine, later episodes progressively wrong."""
    root = _build_dataset(
        tmp_path / "drift",
        episode_lengths=(8, 8, 8, 8),
        from_timestamp_offsets_s=[0.0, 20e-6, 900e-6, 3_000e-6],
    )
    audit = _audit(root)
    video = audit.videos[0]

    assert video.likely_cause == "cumulative_episode_offset"
    assert video.first_violation is not None
    assert video.first_violation.episode_index == 2
    assert video.worst_violation is not None
    assert video.worst_violation.episode_index == 3
    trend = {t.episode_index: t.violation_count for t in video.episode_trend}
    assert trend[0] == 0 and trend[1] == 0
    assert trend[2] > 0 and trend[3] > 0


def test_one_frame_offset_fixture_reports_a_frame_step(tmp_path) -> None:
    """The encoded timeline sits one whole frame off the Parquet grid.

    """
    root = _build_dataset(tmp_path / "frame_step", pts_shift_frames=1)
    audit = _audit(root)
    video = audit.videos[0]

    assert video.likely_cause == "frame_step_offset"
    assert video.worst_violation is not None
    # ~33.3 ms: one frame at 30 Hz, not precision noise.
    assert abs(video.worst_violation.abs_residual_ns - FRAME_NS) < 1_000_000
    assert video.evidence.out_of_range_count == 0


def test_short_video_reports_missing_or_extra_frames_without_raising(tmp_path) -> None:
    """A timeline shorter than the row count must not index out of bounds."""
    root = _build_dataset(tmp_path / "short", encoded_frames=12)
    audit = _audit(root)
    video = audit.videos[0]

    assert video.timeline_length == 12
    assert video.evidence.expected_frame_count == 24
    assert video.likely_cause == "missing_or_extra_frames"
    assert video.evidence.out_of_range_count > 0


def test_middle_episode_offset_by_one_frame_is_caught(tmp_path) -> None:
    """The regression: a whole-frame offset that leaves no residual at all.

    Shifting one episode's ``from_timestamp`` by a frame moves the query and
    the projected index together, so every residual stays zero and a
    residual-only audit reports a clean dataset. Only comparing the declared
    offset against the boundary frame's actual PTS sees it.
    """
    root = _build_dataset(
        tmp_path / "middle",
        episode_lengths=(8, 8, 8),
        from_timestamp_offsets_s=[0.0, 1 / FPS, 0.0],
    )
    audit = _audit(root)
    video = audit.videos[0]

    assert not audit.passed
    assert video.likely_cause == "frame_step_offset"
    # The residuals really are clean; the boundary check is what fires.
    assert video.violation_count == 0
    faults = video.boundary_faults
    assert [check.episode_index for check in faults] == [1]
    assert faults[0].expected_frame_index == 8
    assert abs(faults[0].offset_ns - FRAME_NS) < 1_000_000


def test_extra_encoded_frames_are_caught(tmp_path) -> None:
    """25 encoded frames against 24 declared rows never goes out of range."""
    root = _build_dataset(tmp_path / "extra", encoded_frames=25)
    audit = _audit(root)
    video = audit.videos[0]

    assert not audit.passed
    assert video.timeline_length == 25
    assert video.evidence.expected_frame_count == 24
    assert video.evidence.out_of_range_count == 0
    assert video.likely_cause == "missing_or_extra_frames"


def test_legacy_only_failure_is_reported_rather_than_passed(tmp_path) -> None:
    """A dataset current LeRobot accepts and the reporter's version rejects.

    #2814's own numbers: the distance lands on exactly ``tensor([0.0001])``,
    which the current inclusive comparison accepts and the older strict one
    rejects. Exiting 0 here would answer that issue with "works for me".
    (The float32 half of the legacy path, which needs hour-scale timestamps,
    is covered in the pure tests rather than by encoding an hour of video.)
    """
    root = _build_dataset(
        tmp_path / "legacy",
        episode_lengths=(1, 1),
        from_timestamp_offsets_s=[1e-4, 1e-4],
    )
    out = tmp_path / "evidence.json"
    assert main(["audit-lerobot-pts", str(root), "--json", str(out)]) == 1

    audit = _audit(root)
    video = audit.videos[0]
    assert video.violation_count == 0
    assert video.evidence.legacy_violation_count > 0
    assert video.likely_cause == "legacy_precision_false_positive"

    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["passed"] is False
    assert document["total_violations"] == 0
    assert document["total_legacy_violations"] == 2
    legacy = document["videos"][0]["first_legacy_violation"]
    # The distance is the tolerance: accepted inclusively, rejected strictly.
    assert abs(legacy["signed_residual_ns"]) == document["tolerance_ns"]
    assert legacy["within_tolerance"] is True
    assert legacy["legacy_within_tolerance"] is False


def test_violation_json_carries_everything_an_issue_report_needs(tmp_path) -> None:
    root = _build_dataset(
        tmp_path / "drift",
        episode_lengths=(8, 8, 8),
        from_timestamp_offsets_s=[0.0, 900e-6, 3_000e-6],
    )
    out = tmp_path / "evidence.json"
    assert main(["audit-lerobot-pts", str(root), "--json", str(out)]) == 1

    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["passed"] is False
    assert document["codebase_version"] == "v3.0"
    video = document["videos"][0]
    assert set(video["residual_ns"]) == {"p50", "p95", "p99", "max"}
    assert video["episode_trend"][0]["episode_index"] == 0
    worst = video["worst_violation"]
    for field in (
        "query_ns",
        "selected_index",
        "selected_pts_ns",
        "nearest_index",
        "nearest_pts_ns",
        "signed_residual_ns",
        "abs_residual_ns",
        "camera_key",
        "video_path",
        "episode_index",
        "frame_index",
        "parquet_timestamp_ns",
        "from_timestamp_ns",
    ):
        assert field in worst


def test_custom_tolerance_is_a_diagnostic_not_a_fix(tmp_path) -> None:
    """Widening the tolerance changes the verdict, and nothing else."""
    root = _build_dataset(
        tmp_path / "boundary", from_timestamp_offsets_s=[110e-6] * 3
    )
    assert main(["audit-lerobot-pts", str(root)]) == 1
    assert main(["audit-lerobot-pts", str(root), "--tolerance-s", "0.001"]) == 0


# ------------------------------------------------------------ exit code 2


def test_missing_video_exits_two_with_partial_evidence(tmp_path, capsys) -> None:
    root = _build_dataset(
        tmp_path / "gone",
        cameras=("observation.images.top", "observation.images.wrist"),
    )
    (root / "videos" / "observation.images.wrist" / "chunk-000" / "file-000.mp4").unlink()
    out = tmp_path / "evidence.json"

    assert main(["audit-lerobot-pts", str(root), "--json", str(out)]) == 2

    captured = capsys.readouterr()
    assert "video file not found" in captured.err
    document = json.loads(out.read_text(encoding="utf-8"))
    assert "error" in document
    # The camera that did resolve keeps its evidence.
    assert [video["camera_key"] for video in document["videos"]] == [
        "observation.images.top"
    ]


def test_malformed_metadata_exits_two_without_a_traceback(tmp_path, capsys) -> None:
    root = _build_dataset(tmp_path / "bad", codebase_version="v2.1")
    assert main(["audit-lerobot-pts", str(root)]) == 2
    assert "unsupported LeRobot codebase_version" in capsys.readouterr().err

    truncated = tmp_path / "truncated"
    truncated.mkdir()
    (truncated / "meta").mkdir()
    (truncated / "meta" / "info.json").write_text("{not json", encoding="utf-8")
    assert main(["audit-lerobot-pts", str(truncated)]) == 2
    assert "malformed" in capsys.readouterr().err


def test_absent_pyav_exits_two_with_the_install_hint(tmp_path, capsys, monkeypatch) -> None:
    """An unavailable optional dependency is a setup problem, not a violation."""
    import importlib.abc

    root = _build_dataset(tmp_path / "clean")

    class _BlockAv(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):  # type: ignore[no-untyped-def]
            if fullname.split(".")[0] == "av":
                raise ModuleNotFoundError("No module named 'av'", name="av")
            return None

    for module in [name for name in sys.modules if name.split(".")[0] == "av"]:
        monkeypatch.delitem(sys.modules, module)
    monkeypatch.setattr(sys, "meta_path", [_BlockAv(), *sys.meta_path])

    assert main(["audit-lerobot-pts", str(root)]) == 2
    error = capsys.readouterr().err
    assert "missing optional dependency 'av'" in error
    assert "embodied-sync[lerobot]" in error


def test_directory_without_a_dataset_exits_two(tmp_path, capsys) -> None:
    assert main(["audit-lerobot-pts", str(tmp_path)]) == 2
    assert "not a LeRobot dataset" in capsys.readouterr().err


def test_dataset_without_video_features_exits_two(tmp_path, capsys) -> None:
    root = _build_dataset(tmp_path / "novideo")
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    del info["features"]["observation.images.top"]
    info_path.write_text(json.dumps(info), encoding="utf-8")

    assert main(["audit-lerobot-pts", str(root)]) == 2
    assert "no video features" in capsys.readouterr().err
