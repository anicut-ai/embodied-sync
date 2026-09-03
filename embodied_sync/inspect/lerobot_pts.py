"""Audit a LeRobot v3 dataset's Parquet timestamps against encoded video PTS.

Why this exists
---------------
LeRobot resolves a video frame by adding a per-episode video offset to a
Parquet-stored, episode-relative timestamp, projecting that query onto a
frame index, and refusing the result when the selected frame's presentation
timestamp sits further from the query than ``tolerance_s`` (default 1e-4).
Three reported failures share that check and nothing else:

- `#2814 <https://github.com/huggingface/lerobot/issues/2814>`_ fails at a
  distance of ``tensor([0.0001])`` — the tolerance itself, which only fails
  under a *strict* comparison, in float32;
- `#2364 <https://github.com/huggingface/lerobot/issues/2364>`_ fails near
  1025 s, where a float32 timeline's own spacing is about 122 us and any
  comparison at 100 us precision is measuring rounding;
- `#3177 <https://github.com/huggingface/lerobot/issues/3177>`_ fails
  because accumulated episode ``from_timestamp`` values drift from the
  concatenated video's actual PTS by a whole frame.

Those are three different defects with one error message. This module
reproduces the check with full precision and says, per camera and physical
video, which of them a local dataset actually has.

Four measurements, not one
--------------------------
A residual alone cannot separate them, so the audit takes four:

1. the **exact** residual — the current float64, inclusive-comparison path;
2. the **legacy** residual — the same query in float32 with a strict
   comparison, because a dataset that passes today can still be the one the
   reporter's older LeRobot rejects, and reporting only (1) would answer
   #2814 with "works for me";
3. the **episode boundary** offset — each declared ``from_timestamp``
   against the PTS of the frame it should land on, derived from the frame
   counts of preceding episodes in the same physical video. A whole-frame
   error in ``from_timestamp`` moves the query *and* the projected index
   together, so it leaves no residual at all; only this comparison sees it,
   and it is what makes the #3177 signature detectable rather than assumed;
4. the **frame count** — declared rows against the encoded timeline, which
   catches extra frames as well as missing ones.

What it does not claim (D-0043)
-------------------------------
This audit validates *internal consistency* between Parquet timestamps,
episode video offsets, the decoder's index projection, and encoded PTS. It
says nothing about acquisition-time sensor synchronization: LeRobot v3
discards independent sensor clocks, so no ``likely_cause`` here may be read
as a sensor-sync verdict. It is also strictly read-only — it never rewrites
timestamps, alters PTS, widens the tolerance, or drops samples.

The ordinary importer stays unchanged for the same reason: folding decoded
PTS into ``import-lerobot`` would manufacture acquisition clocks the source
does not have, and would make every import pay for video decoding.

Precision discipline
--------------------
Container timing stays rational (``Fraction`` over the stream time base)
until one final conversion to integer nanoseconds; parquet-derived times
convert through :meth:`Fraction.from_float`, which is exact for the stored
float64 value. Every comparison is between integers. Nothing here compares
formatted decimal strings, and no pixel data is decoded or retained — the
PTS timeline comes from demuxed packet timestamps, with the packet/frame
cardinality verified rather than assumed.
"""

from __future__ import annotations

import json
import math
import struct
from array import array
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import ceil
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "DEFAULT_TOLERANCE_S",
    "BoundaryCheck",
    "DatasetAudit",
    "EpisodeTrend",
    "LeRobotPTSAuditError",
    "LikelyCause",
    "PTSQuery",
    "VideoAudit",
    "VideoEvidence",
    "audit_lerobot_dataset",
    "classify_video",
    "float32_ulp_ns",
    "legacy_float32_residual_ns",
    "precision_band_ns",
    "render_audit_text",
]

#: LeRobot's own default video timestamp tolerance, in seconds.
DEFAULT_TOLERANCE_S = 1e-4

_NS = 1_000_000_000

LikelyCause = Literal[
    "within_tolerance",
    "precision_boundary",
    "legacy_precision_false_positive",
    "cumulative_episode_offset",
    "frame_step_offset",
    "missing_or_extra_frames",
    "unclassified",
]

#: A residual counts as "one frame step" when it lands within this fraction
#: of a frame duration of an integer multiple of it. Encoders round PTS onto
#: the container time base, so an exact multiple is not reachable in general.
_FRAME_STEP_FRACTION = 0.1

#: How much the last episode's offset must exceed the first one's before a
#: growing trend is called cumulative drift rather than scatter.
_CUMULATIVE_GROWTH_RATIO = 1.5

#: Fewer faulty episodes than this is a single wrong offset, not a trend.
_CUMULATIVE_MIN_EPISODES = 2


class LeRobotPTSAuditError(RuntimeError):
    """An audit that could not finish, carrying what it did finish.

    ``partial`` holds the video audits completed before the failure so a CI
    run or an issue report still has evidence when the command exits 2.
    """

    def __init__(self, message: str, *, partial: DatasetAudit | None = None) -> None:
        super().__init__(message)
        self.partial = partial


# --------------------------------------------------------------- precision


def _float32(value: float) -> float:
    """``value`` as it survives a round trip through IEEE-754 binary32."""
    return float(struct.unpack("f", struct.pack("f", value))[0])


def float32_ulp_ns(seconds: float) -> int:
    """Spacing of float32 values at this magnitude, in nanoseconds.

    This is the resolution floor of any timeline stored as float32: near
    1025 s it is about 122 us, so a 100 us tolerance there is finer than the
    numbers being compared and *must not* be read as a timing fault.
    """
    if seconds == 0.0 or not math.isfinite(seconds):
        return 0
    _, exponent = math.frexp(abs(seconds))
    # binary32 carries a 24-bit significand, so ulp = 2**(exponent - 24).
    return round(2.0 ** (exponent - 24) * _NS)


def legacy_float32_residual_ns(query_s: float, selected_pts_ns: int) -> int:
    """The residual an older, float32 LeRobot would have computed.

    Newer LeRobot compares in float64 and accepts the boundary; older code
    did neither. Reproducing that path is the only way to explain a dataset
    that this audit finds clean and the reporter's install rejects.
    """
    query32 = _float32(query_s)
    pts32 = _float32(selected_pts_ns / 1e9)
    return round(Fraction.from_float(_float32(pts32 - query32)) * _NS)


def precision_band_ns(tolerance_ns: int, ulp_ns: int) -> int:
    """Widest residual still explainable by timestamp precision alone.

    Tied to the tolerance under test and to the float32 spacing at the
    timestamps involved — deliberately *not* to the frame period. Half a
    frame at 30 Hz is 16.7 ms, which would file a 100x-over-tolerance error
    under "precision noise" and bury exactly the bugs this tool exists to
    surface.
    """
    return max(2 * tolerance_ns, ulp_ns + tolerance_ns)


def _is_frame_step(offset_ns: int, frame_duration_ns: int, slack_ns: int) -> bool:
    """Is ``offset`` an integer number of whole frames, within slack?"""
    if frame_duration_ns <= 0:
        return False
    steps = round(abs(offset_ns) / frame_duration_ns)
    return steps >= 1 and abs(abs(offset_ns) - steps * frame_duration_ns) <= slack_ns


def _grows(values: Sequence[int], tolerance_ns: int) -> bool:
    """Does this ordered series climb, rather than scatter?

    Slack of one tolerance in the monotonicity test: episode-to-episode
    variation below the very threshold under test is not evidence either way.
    """
    if len(values) < 2:
        return False
    magnitudes = [abs(value) for value in values]
    non_decreasing = all(
        later >= earlier - tolerance_ns
        for earlier, later in zip(magnitudes, magnitudes[1:])
    )
    grows = magnitudes[-1] > max(
        magnitudes[0] * _CUMULATIVE_GROWTH_RATIO, magnitudes[0] + tolerance_ns
    )
    return non_decreasing and grows


def _percentile(values: Sequence[int], fraction: float) -> int:
    """Nearest-rank percentile, matching the convention used elsewhere."""
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(0, ceil(fraction * len(ordered)) - 1)
    return ordered[min(rank, len(ordered) - 1)]


# ------------------------------------------------------------------ model


@dataclass(frozen=True, slots=True)
class PTSQuery:
    """One frame lookup as LeRobot would perform it, resolved exactly.

    Retained only for the exemplars a report quotes — first and worst
    violation, first legacy-only violation — never for every frame.
    """

    episode_index: int
    camera_key: str
    video_path: str
    frame_index: int
    parquet_timestamp_ns: int
    from_timestamp_ns: int
    query_ns: int
    #: ``round(query_s * average_fps)`` — the decoder's own projection,
    #: before any clamping. Out-of-range values are reported, not hidden.
    selected_index: int
    selected_pts_ns: int
    #: Index of the truly nearest PTS. Differing from ``selected_index``
    #: separates a bad index projection from a bad timeline.
    nearest_index: int
    nearest_pts_ns: int
    #: ``selected_pts − query``. The decoder compares against the frame it
    #: selected, so this — not the nearest-PTS distance — decides the check.
    signed_residual_ns: int
    abs_residual_ns: int
    within_tolerance: bool
    #: Same comparison in float32 with a strict ``<``: what an older LeRobot
    #: would have measured, and why a clean dataset can still be rejected.
    legacy_residual_ns: int
    legacy_within_tolerance: bool
    #: float32 spacing at this query's magnitude — the resolution floor
    #: below which the comparison is measuring rounding.
    float32_ulp_ns: int
    #: The projected index fell outside the physical video's timeline.
    index_out_of_range: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_index": self.episode_index,
            "camera_key": self.camera_key,
            "video_path": self.video_path,
            "frame_index": self.frame_index,
            "parquet_timestamp_ns": self.parquet_timestamp_ns,
            "from_timestamp_ns": self.from_timestamp_ns,
            "query_ns": self.query_ns,
            "selected_index": self.selected_index,
            "selected_pts_ns": self.selected_pts_ns,
            "nearest_index": self.nearest_index,
            "nearest_pts_ns": self.nearest_pts_ns,
            "signed_residual_ns": self.signed_residual_ns,
            "abs_residual_ns": self.abs_residual_ns,
            "within_tolerance": self.within_tolerance,
            "legacy_residual_ns": self.legacy_residual_ns,
            "legacy_within_tolerance": self.legacy_within_tolerance,
            "float32_ulp_ns": self.float32_ulp_ns,
            "index_out_of_range": self.index_out_of_range,
        }


@dataclass(frozen=True, slots=True)
class BoundaryCheck:
    """One episode's declared video offset against the PTS it should hit.

    The frame a ``from_timestamp`` is supposed to land on is fixed by the
    lengths of the episodes stored before it in the same physical video, so
    this comparison needs no pixels — only arithmetic the metadata already
    determines.
    """

    episode_index: int
    expected_frame_index: int
    declared_from_timestamp_ns: int
    #: ``None`` when the expected frame is past the end of the timeline,
    #: which the frame-count check reports separately.
    actual_boundary_pts_ns: int | None
    #: ``declared − actual``. Zero for a dataset whose episode offsets agree
    #: with the video it points into.
    offset_ns: int | None
    within_tolerance: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_index": self.episode_index,
            "expected_frame_index": self.expected_frame_index,
            "declared_from_timestamp_ns": self.declared_from_timestamp_ns,
            "actual_boundary_pts_ns": self.actual_boundary_pts_ns,
            "offset_ns": self.offset_ns,
            "within_tolerance": self.within_tolerance,
        }


@dataclass(frozen=True, slots=True)
class EpisodeTrend:
    """Per-episode residual summary — the shape drift classification reads."""

    episode_index: int
    query_count: int
    violation_count: int
    legacy_violation_count: int
    mean_abs_residual_ns: int
    max_abs_residual_ns: int

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_index": self.episode_index,
            "query_count": self.query_count,
            "violation_count": self.violation_count,
            "legacy_violation_count": self.legacy_violation_count,
            "mean_abs_residual_ns": self.mean_abs_residual_ns,
            "max_abs_residual_ns": self.max_abs_residual_ns,
        }


@dataclass(frozen=True, slots=True)
class VideoEvidence:
    """Everything :func:`classify_video` reads, and nothing per-frame.

    Bounded by episode count rather than frame count, which is what lets the
    audit summarize a million-frame dataset — and what lets a test reproduce
    an hour-long or hundred-episode signature without encoding one.
    """

    tolerance_ns: int
    frame_duration_ns: int
    timeline_length: int
    expected_frame_count: int
    query_count: int
    violation_count: int
    legacy_violation_count: int
    out_of_range_count: int
    #: Signed boundary offset per audited episode, in episode order.
    boundary_offsets_ns: tuple[int, ...] = ()
    boundary_fault_count: int = 0
    #: Mean absolute residual for each episode that has a violation, in
    #: episode order.
    violating_episode_means_ns: tuple[int, ...] = ()
    frame_step_violation_count: int = 0
    precision_band_violation_count: int = 0
    max_float32_ulp_ns: int = 0

    @property
    def frame_count_mismatch(self) -> bool:
        return self.timeline_length != self.expected_frame_count

    def to_dict(self) -> dict[str, object]:
        return {
            "timeline_length": self.timeline_length,
            "expected_frame_count": self.expected_frame_count,
            "frame_count_mismatch": self.frame_count_mismatch,
            "out_of_range_count": self.out_of_range_count,
            "boundary_fault_count": self.boundary_fault_count,
            "frame_step_violation_count": self.frame_step_violation_count,
            "precision_band_violation_count": self.precision_band_violation_count,
            "max_float32_ulp_ns": self.max_float32_ulp_ns,
            "precision_band_ns": precision_band_ns(
                self.tolerance_ns, self.max_float32_ulp_ns
            ),
        }


def classify_video(evidence: VideoEvidence) -> LikelyCause:
    """Name the failure mode behind one physical video's findings.

    Pure arithmetic over a bounded summary, so every signature — including
    the hour-scale and hundred-episode ones — is reproducible in a test.

    The rules, in the order they are applied:

    1. the encoded frame count disagrees with the declared rows, or an index
       projection left the timeline — ``missing_or_extra_frames``. It
       outranks everything because every residual computed against a
       clamped index is an artifact of the clamp;
    2. declared episode offsets disagree with the PTS they should land on.
       These are graded on their own, because a whole-frame error here
       leaves *no* residual to grade: it moves query and index together;
    3. otherwise the per-query residuals are graded the same way — growing
       across episodes, clustered at whole frames, or inside the precision
       band;
    4. no exact violation but a legacy float32 one —
       ``legacy_precision_false_positive``: correct today, rejected by an
       older LeRobot, which is the state #2814 reports from;
    5. nothing at all — ``within_tolerance``.

    A residual that fits no rule is ``unclassified`` rather than guessed.
    """
    if evidence.frame_count_mismatch or evidence.out_of_range_count:
        return "missing_or_extra_frames"

    slack_ns = max(
        evidence.tolerance_ns,
        int(evidence.frame_duration_ns * _FRAME_STEP_FRACTION),
    )
    band_ns = precision_band_ns(evidence.tolerance_ns, evidence.max_float32_ulp_ns)

    if evidence.boundary_fault_count:
        offsets = evidence.boundary_offsets_ns
        faults = [offset for offset in offsets if abs(offset) > evidence.tolerance_ns]
        if len(faults) >= _CUMULATIVE_MIN_EPISODES and _grows(
            offsets, evidence.tolerance_ns
        ):
            return "cumulative_episode_offset"
        if all(
            _is_frame_step(offset, evidence.frame_duration_ns, slack_ns)
            for offset in faults
        ):
            return "frame_step_offset"
        if all(abs(offset) <= band_ns for offset in faults):
            return "precision_boundary"
        return "unclassified"

    if evidence.violation_count:
        means = evidence.violating_episode_means_ns
        if len(means) >= _CUMULATIVE_MIN_EPISODES and _grows(
            means, evidence.tolerance_ns
        ):
            return "cumulative_episode_offset"
        if evidence.frame_step_violation_count == evidence.violation_count:
            return "frame_step_offset"
        if evidence.precision_band_violation_count == evidence.violation_count:
            return "precision_boundary"
        return "unclassified"

    if evidence.legacy_violation_count:
        return "legacy_precision_false_positive"
    return "within_tolerance"


@dataclass(frozen=True, slots=True)
class VideoAudit:
    """Every finding about one physical video.

    One entry per (camera, physical video file): a camera's shards can fail
    for different reasons, and a single verdict over all of them would name
    at most one.
    """

    camera_key: str
    video_path: str
    average_fps: float
    #: False when the container did not declare a frame count to check the
    #: demuxed packet count against; the timeline is then unverified.
    timeline_verified: bool
    episode_indices: tuple[int, ...]
    evidence: VideoEvidence
    likely_cause: LikelyCause
    episode_trend: tuple[EpisodeTrend, ...]
    boundary_checks: tuple[BoundaryCheck, ...]
    p50_abs_residual_ns: int
    p95_abs_residual_ns: int
    p99_abs_residual_ns: int
    max_abs_residual_ns: int
    first_violation: PTSQuery | None
    worst_violation: PTSQuery | None
    #: First query the current code accepts and an older LeRobot would not.
    first_legacy_violation: PTSQuery | None

    @property
    def timeline_length(self) -> int:
        return self.evidence.timeline_length

    @property
    def query_count(self) -> int:
        return self.evidence.query_count

    @property
    def violation_count(self) -> int:
        return self.evidence.violation_count

    @property
    def passed(self) -> bool:
        return self.likely_cause == "within_tolerance"

    @property
    def boundary_faults(self) -> tuple[BoundaryCheck, ...]:
        return tuple(check for check in self.boundary_checks if not check.within_tolerance)

    def to_dict(self) -> dict[str, object]:
        return {
            "camera_key": self.camera_key,
            "video_path": self.video_path,
            "average_fps": self.average_fps,
            "timeline_length": self.timeline_length,
            "timeline_verified": self.timeline_verified,
            "episode_indices": list(self.episode_indices),
            "query_count": self.query_count,
            "violation_count": self.violation_count,
            "legacy_violation_count": self.evidence.legacy_violation_count,
            "likely_cause": self.likely_cause,
            "passed": self.passed,
            "evidence": self.evidence.to_dict(),
            "residual_ns": {
                "p50": self.p50_abs_residual_ns,
                "p95": self.p95_abs_residual_ns,
                "p99": self.p99_abs_residual_ns,
                "max": self.max_abs_residual_ns,
            },
            "episode_trend": [trend.to_dict() for trend in self.episode_trend],
            "boundary_checks": [check.to_dict() for check in self.boundary_checks],
            "first_violation": (
                self.first_violation.to_dict() if self.first_violation else None
            ),
            "worst_violation": (
                self.worst_violation.to_dict() if self.worst_violation else None
            ),
            "first_legacy_violation": (
                self.first_legacy_violation.to_dict()
                if self.first_legacy_violation
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class DatasetAudit:
    """The audited dataset: what was checked, and what failed."""

    dataset_path: str
    codebase_version: str
    fps: float
    tolerance_s: float
    tolerance_ns: int
    audited_episodes: int
    videos: tuple[VideoAudit, ...]

    @property
    def total_queries(self) -> int:
        return sum(video.query_count for video in self.videos)

    @property
    def total_violations(self) -> int:
        return sum(video.violation_count for video in self.videos)

    @property
    def total_legacy_violations(self) -> int:
        return sum(video.evidence.legacy_violation_count for video in self.videos)

    @property
    def passed(self) -> bool:
        """No finding of any kind, including boundary and legacy findings.

        Deliberately stricter than "no query exceeded the tolerance": a
        whole-frame episode offset produces no residual, and a legacy-only
        failure is the exact state a reporter is asking about. Exiting 0 on
        either would hand back a clean bill of health for a broken dataset.
        """
        return all(video.passed for video in self.videos)

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "lerobot_pts_audit",
            "dataset_path": self.dataset_path,
            "codebase_version": self.codebase_version,
            "fps": self.fps,
            "tolerance_s": self.tolerance_s,
            "tolerance_ns": self.tolerance_ns,
            "audited_episodes": self.audited_episodes,
            "total_queries": self.total_queries,
            "total_violations": self.total_violations,
            "total_legacy_violations": self.total_legacy_violations,
            "passed": self.passed,
            "videos": [video.to_dict() for video in self.videos],
        }


# --------------------------------------------------------------- container


def _read_pts_timeline(video_path: Path) -> tuple[tuple[int, ...], float, bool]:
    """``(pts_ns, average_fps, verified)`` for one physical video.

    Demux only: packet timestamps carry presentation times without decoding
    a single pixel, and sorting them recovers presentation order even when
    the container stores frames in a different decode order.

    A packet is not universally a frame, though — a codec may emit zero or
    several frames for one packet, and TorchCodec indexes *frames*. So the
    packet count is checked against the count the container declares, and a
    container where they disagree is refused rather than silently audited
    against the wrong timeline. ``verified`` is False when the container
    declares no count to check against.
    """
    import av

    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError(f"no video stream in {video_path}")
        stream = container.streams.video[0]
        time_base = stream.time_base
        if time_base is None:
            raise ValueError(f"video stream has no time base: {video_path}")
        average_rate = stream.average_rate
        declared_frames = int(stream.frames or 0)
        rational_pts = sorted(
            Fraction(packet.pts) * time_base
            for packet in container.demux(stream)
            if packet.pts is not None
        )

    if declared_frames > 0 and declared_frames != len(rational_pts):
        raise ValueError(
            f"{video_path}: {len(rational_pts)} demuxed packet timestamps but the "
            f"container declares {declared_frames} frames. This audit reads packet "
            "timestamps and cannot resolve a container whose packets and frames "
            "are not one to one; re-mux the video or audit it with a decoder-based "
            "tool instead of trusting a timeline that would be silently wrong"
        )
    return (
        tuple(round(pts * _NS) for pts in rational_pts),
        float(average_rate) if average_rate else 0.0,
        declared_frames > 0,
    )


def _resolve_query(
    query_ns: int,
    query_s: float,
    timeline_ns: Sequence[int],
    average_fps: float,
) -> tuple[int, int, int, int, bool]:
    """``(selected_index, selected_pts, nearest_index, nearest_pts, out_of_range)``.

    The projection is TorchCodec's, in float64 rather than float32 — the
    audit must not reproduce the precision bug it is measuring. An index
    outside the timeline is reported as-is and clamped only for the lookup,
    so a short or long timeline surfaces as a fact instead of an exception.
    """
    count = len(timeline_ns)
    projected = round(query_s * average_fps)
    out_of_range = not 0 <= projected < count
    selected_index = min(max(projected, 0), count - 1)

    position = bisect_left(timeline_ns, query_ns)
    neighbours = [i for i in (position - 1, position) if 0 <= i < count]
    nearest_index = min(neighbours, key=lambda i: abs(timeline_ns[i] - query_ns))
    return (
        projected,
        timeline_ns[selected_index],
        nearest_index,
        timeline_ns[nearest_index],
        out_of_range,
    )


def _episode_timestamps(
    root: Path,
    data_path_tpl: str,
    episode: dict[str, Any],
    file_base: dict[tuple[int, int], int],
    table_cache: dict[tuple[int, int], Any],
) -> tuple[list[float], list[int]]:
    """This episode's Parquet ``timestamp`` and ``frame_index`` columns."""
    import pyarrow.parquet as pq

    chunk = int(episode["data/chunk_index"])
    file_index = int(episode["data/file_index"])
    key = (chunk, file_index)
    if key not in table_cache:
        data_file = root / data_path_tpl.format(chunk_index=chunk, file_index=file_index)
        if not data_file.is_file():
            raise FileNotFoundError(f"episode data file not found: {data_file}")
        table_cache[key] = pq.read_table(data_file, columns=["timestamp", "frame_index"])
    table = table_cache[key]
    start = int(episode["dataset_from_index"]) - file_base[key]
    stop = int(episode["dataset_to_index"]) - file_base[key]
    if start < 0 or stop > table.num_rows or stop < start:
        raise ValueError(
            f"episode {int(episode['episode_index'])}: row range [{start}, {stop}) "
            f"outside data file with {table.num_rows} rows"
        )
    rows = table.slice(start, stop - start)
    return (
        [float(value) for value in rows.column("timestamp").to_pylist()],
        [int(value) for value in rows.column("frame_index").to_pylist()],
    )


class _VideoAccumulator:
    """Running statistics for one physical video.

    Holds one int64 per query rather than one object per query: exact
    percentiles still need the distribution, but a dataset with millions of
    frames must not pay a Python object for each of them. Everything else —
    per-episode trend, violation counts, exemplars — is incremental.
    """

    __slots__ = (
        "abs_residuals",
        "episode_order",
        "episode_stats",
        "first_legacy_violation",
        "first_violation",
        "frame_step_violations",
        "legacy_violations",
        "max_ulp_ns",
        "out_of_range",
        "precision_band_violations",
        "violations",
        "worst_violation",
    )

    def __init__(self) -> None:
        self.abs_residuals: array[int] = array("q")
        self.episode_order: list[int] = []
        # episode -> [queries, violations, legacy violations, sum, max]
        self.episode_stats: dict[int, list[int]] = {}
        self.violations = 0
        self.legacy_violations = 0
        self.out_of_range = 0
        self.frame_step_violations = 0
        self.precision_band_violations = 0
        self.max_ulp_ns = 0
        self.first_violation: PTSQuery | None = None
        self.worst_violation: PTSQuery | None = None
        self.first_legacy_violation: PTSQuery | None = None

    def add(
        self,
        query: PTSQuery,
        *,
        frame_duration_ns: int,
        tolerance_ns: int,
    ) -> None:
        residual = query.abs_residual_ns
        self.abs_residuals.append(residual)
        stats = self.episode_stats.get(query.episode_index)
        if stats is None:
            stats = [0, 0, 0, 0, 0]
            self.episode_stats[query.episode_index] = stats
            self.episode_order.append(query.episode_index)
        stats[0] += 1
        stats[3] += residual
        stats[4] = max(stats[4], residual)
        self.max_ulp_ns = max(self.max_ulp_ns, query.float32_ulp_ns)
        if query.index_out_of_range:
            self.out_of_range += 1
        if not query.within_tolerance:
            stats[1] += 1
            self.violations += 1
            slack = max(tolerance_ns, int(frame_duration_ns * _FRAME_STEP_FRACTION))
            if _is_frame_step(residual, frame_duration_ns, slack):
                self.frame_step_violations += 1
            if residual <= precision_band_ns(tolerance_ns, query.float32_ulp_ns):
                self.precision_band_violations += 1
            if self.first_violation is None:
                self.first_violation = query
            if self.worst_violation is None or residual > self.worst_violation.abs_residual_ns:
                self.worst_violation = query
        if not query.legacy_within_tolerance:
            stats[2] += 1
            self.legacy_violations += 1
            if self.first_legacy_violation is None:
                self.first_legacy_violation = query

    def episode_trend(self) -> tuple[EpisodeTrend, ...]:
        return tuple(
            EpisodeTrend(
                episode_index=episode,
                query_count=self.episode_stats[episode][0],
                violation_count=self.episode_stats[episode][1],
                legacy_violation_count=self.episode_stats[episode][2],
                mean_abs_residual_ns=round(
                    self.episode_stats[episode][3] / self.episode_stats[episode][0]
                ),
                max_abs_residual_ns=self.episode_stats[episode][4],
            )
            for episode in sorted(self.episode_order)
        )

    def violating_episode_means(self) -> tuple[int, ...]:
        return tuple(
            trend.mean_abs_residual_ns
            for trend in self.episode_trend()
            if trend.violation_count
        )


# ------------------------------------------------------------------- audit


def _feature_dtype(spec: object) -> str:
    """``dtype`` of one info.json feature, tolerating malformed entries.

    A feature written as a bare string (``"camera": "video"``) is malformed
    metadata, not a crash: it must reach the caller as a refusal to audit.
    """
    if not isinstance(spec, dict):
        return ""
    return str(spec.get("dtype", ""))


def audit_lerobot_dataset(
    path: str | Path,
    *,
    tolerance_s: float = DEFAULT_TOLERANCE_S,
    max_episodes: int | None = None,
) -> DatasetAudit:
    """Check every video query in a local LeRobot v3 dataset, read-only.

    Requires ``pyarrow`` and ``av`` (both in the ``lerobot`` extra), imported
    lazily so ``import embodied_sync`` stays a base-install operation.

    Raises :class:`FileNotFoundError` or :class:`ValueError` for a dataset
    that cannot be audited at all, and :class:`LeRobotPTSAuditError` — with
    the completed video audits attached — when a later video or data file
    turns out to be missing or malformed.
    """
    from embodied_sync.adapters.lerobot import (
        _data_file_row_bases,
        _read_episode_metadata,
    )

    if tolerance_s < 0:
        raise ValueError(f"tolerance_s must be >= 0, got {tolerance_s!r}")

    root = Path(path)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"not a LeRobot dataset (no meta/info.json): {root}")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed {info_path}: {exc}") from exc
    if not isinstance(info, dict):
        raise ValueError(f"malformed {info_path}: expected a JSON object")

    version = str(info.get("codebase_version", ""))
    if not version.startswith("v3"):
        raise ValueError(
            f"unsupported LeRobot codebase_version {version!r} in {info_path}; "
            "this audit supports v3.x"
        )
    if "fps" not in info:
        raise ValueError(f"malformed {info_path}: missing 'fps'")
    try:
        fps = float(info["fps"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"malformed {info_path}: 'fps' is not a number") from exc
    if fps <= 0:
        raise ValueError(f"invalid fps {fps!r} in {info_path}")
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"malformed {info_path}: missing 'features'")
    malformed = [name for name, spec in features.items() if not isinstance(spec, dict)]
    if malformed:
        raise ValueError(
            f"malformed {info_path}: feature(s) {malformed!r} are not JSON objects; "
            "each feature must be a mapping carrying at least a 'dtype'"
        )
    video_keys = [
        name for name, spec in features.items() if _feature_dtype(spec) == "video"
    ]
    if not video_keys:
        raise ValueError(f"no video features to audit in {info_path}")
    video_path_tpl = str(info.get("video_path", ""))
    if not video_path_tpl:
        raise ValueError(f"malformed {info_path}: missing 'video_path'")
    data_path_tpl = str(info.get("data_path", ""))
    if not data_path_tpl:
        raise ValueError(f"malformed {info_path}: missing 'data_path'")

    meta_dir = root / "meta"
    all_episode_rows = _read_episode_metadata(meta_dir)
    file_base = _data_file_row_bases(meta_dir)
    episode_rows = (
        all_episode_rows if max_episodes is None else all_episode_rows[:max_episodes]
    )
    if not episode_rows:
        raise ValueError(f"no episodes to audit in {root}")

    tolerance_ns = round(tolerance_s * _NS)
    frame_duration_ns = round(_NS / fps)
    audited = DatasetAudit(
        dataset_path=str(root),
        codebase_version=version,
        fps=fps,
        tolerance_s=tolerance_s,
        tolerance_ns=tolerance_ns,
        audited_episodes=len(episode_rows),
        videos=(),
    )

    table_cache: dict[tuple[int, int], Any] = {}
    videos: list[VideoAudit] = []
    for camera_key in video_keys:
        chunk_field = f"videos/{camera_key}/chunk_index"
        file_field = f"videos/{camera_key}/file_index"

        # Every episode of the file, selected or not: an episode's position
        # inside its physical video is fixed by the ones stored before it,
        # so --max-episodes must not be able to move a boundary.
        placement: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for episode in all_episode_rows:
            if chunk_field not in episode or file_field not in episode:
                raise LeRobotPTSAuditError(
                    f"episode {int(episode['episode_index'])} metadata has no "
                    f"{chunk_field!r}: camera {camera_key!r} is declared in "
                    "info.json but not resolvable to a physical video",
                    partial=_with_videos(audited, videos),
                )
            placement.setdefault(
                (int(episode[chunk_field]), int(episode[file_field])), []
            ).append(episode)

        selected = {int(episode["episode_index"]) for episode in episode_rows}
        for (chunk, file_index), file_episodes in sorted(placement.items()):
            ordered = sorted(file_episodes, key=lambda row: int(row["episode_index"]))
            audit_here = [
                episode
                for episode in ordered
                if int(episode["episode_index"]) in selected
            ]
            if not audit_here:
                continue

            relative = video_path_tpl.format(
                video_key=camera_key, chunk_index=chunk, file_index=file_index
            )
            video_path = root / relative
            try:
                if not video_path.is_file():
                    raise FileNotFoundError(f"video file not found: {video_path}")
                timeline_ns, average_fps, verified = _read_pts_timeline(video_path)
            except (FileNotFoundError, OSError, ValueError) as exc:
                raise LeRobotPTSAuditError(
                    f"camera {camera_key!r}: {exc}",
                    partial=_with_videos(audited, videos),
                ) from exc
            if not timeline_ns:
                raise LeRobotPTSAuditError(
                    f"camera {camera_key!r}: {video_path} carries no frame "
                    "timestamps to compare against",
                    partial=_with_videos(audited, videos),
                )
            if average_fps <= 0:
                # Without a container rate there is no index projection to
                # reproduce; the dataset's own fps is what LeRobot wrote.
                average_fps = fps

            expected_frame_count = sum(int(row["length"]) for row in ordered)
            accumulator = _VideoAccumulator()
            boundary_checks: list[BoundaryCheck] = []
            frames_before = 0
            for episode in ordered:
                episode_index = int(episode["episode_index"])
                length = int(episode["length"])
                if episode_index not in selected:
                    frames_before += length
                    continue
                try:
                    timestamps, frame_indices = _episode_timestamps(
                        root, data_path_tpl, episode, file_base, table_cache
                    )
                    from_timestamp = float(episode[f"videos/{camera_key}/from_timestamp"])
                except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
                    raise LeRobotPTSAuditError(
                        f"camera {camera_key!r}, episode {episode_index}: {exc}",
                        partial=_with_videos(audited, videos),
                    ) from exc
                from_ns = round(Fraction.from_float(from_timestamp) * _NS)

                # The boundary this episode's offset should land on is fixed
                # by the frames stored before it. Comparing the two is the
                # only way to see a whole-frame from_timestamp error, which
                # moves the query and the projected index together and so
                # leaves no residual behind.
                boundary_pts = (
                    timeline_ns[frames_before]
                    if 0 <= frames_before < len(timeline_ns)
                    else None
                )
                boundary_offset = (
                    from_ns - boundary_pts if boundary_pts is not None else None
                )
                boundary_checks.append(
                    BoundaryCheck(
                        episode_index=episode_index,
                        expected_frame_index=frames_before,
                        declared_from_timestamp_ns=from_ns,
                        actual_boundary_pts_ns=boundary_pts,
                        offset_ns=boundary_offset,
                        within_tolerance=(
                            boundary_offset is not None
                            and abs(boundary_offset) <= tolerance_ns
                        ),
                    )
                )

                for timestamp, frame_index in zip(timestamps, frame_indices):
                    query_s = from_timestamp + timestamp
                    query_ns = round(Fraction.from_float(query_s) * _NS)
                    (
                        selected_index,
                        selected_pts_ns,
                        nearest_index,
                        nearest_pts_ns,
                        out_of_range,
                    ) = _resolve_query(query_ns, query_s, timeline_ns, average_fps)
                    signed = selected_pts_ns - query_ns
                    legacy = legacy_float32_residual_ns(query_s, selected_pts_ns)
                    accumulator.add(
                        PTSQuery(
                            episode_index=episode_index,
                            camera_key=camera_key,
                            video_path=str(video_path),
                            frame_index=frame_index,
                            parquet_timestamp_ns=round(
                                Fraction.from_float(timestamp) * _NS
                            ),
                            from_timestamp_ns=from_ns,
                            query_ns=query_ns,
                            selected_index=selected_index,
                            selected_pts_ns=selected_pts_ns,
                            nearest_index=nearest_index,
                            nearest_pts_ns=nearest_pts_ns,
                            signed_residual_ns=signed,
                            abs_residual_ns=abs(signed),
                            within_tolerance=abs(signed) <= tolerance_ns,
                            legacy_residual_ns=legacy,
                            # Older LeRobot compared strictly, which is why
                            # #2814 fails at exactly the tolerance.
                            legacy_within_tolerance=abs(legacy) < tolerance_ns,
                            float32_ulp_ns=float32_ulp_ns(query_s),
                            index_out_of_range=out_of_range,
                        ),
                        frame_duration_ns=frame_duration_ns,
                        tolerance_ns=tolerance_ns,
                    )
                frames_before += length

            evidence = VideoEvidence(
                tolerance_ns=tolerance_ns,
                frame_duration_ns=frame_duration_ns,
                timeline_length=len(timeline_ns),
                expected_frame_count=expected_frame_count,
                query_count=len(accumulator.abs_residuals),
                violation_count=accumulator.violations,
                legacy_violation_count=accumulator.legacy_violations,
                out_of_range_count=accumulator.out_of_range,
                boundary_offsets_ns=tuple(
                    check.offset_ns
                    for check in boundary_checks
                    if check.offset_ns is not None
                ),
                boundary_fault_count=sum(
                    1 for check in boundary_checks if not check.within_tolerance
                ),
                violating_episode_means_ns=accumulator.violating_episode_means(),
                frame_step_violation_count=accumulator.frame_step_violations,
                precision_band_violation_count=accumulator.precision_band_violations,
                max_float32_ulp_ns=accumulator.max_ulp_ns,
            )
            residuals = accumulator.abs_residuals
            videos.append(
                VideoAudit(
                    camera_key=camera_key,
                    video_path=str(video_path),
                    average_fps=average_fps,
                    timeline_verified=verified,
                    episode_indices=tuple(
                        int(episode["episode_index"]) for episode in audit_here
                    ),
                    evidence=evidence,
                    likely_cause=classify_video(evidence),
                    episode_trend=accumulator.episode_trend(),
                    boundary_checks=tuple(boundary_checks),
                    p50_abs_residual_ns=_percentile(residuals, 0.50),
                    p95_abs_residual_ns=_percentile(residuals, 0.95),
                    p99_abs_residual_ns=_percentile(residuals, 0.99),
                    max_abs_residual_ns=max(residuals) if residuals else 0,
                    first_violation=accumulator.first_violation,
                    worst_violation=accumulator.worst_violation,
                    first_legacy_violation=accumulator.first_legacy_violation,
                )
            )

    result = _with_videos(audited, videos)
    if result.total_queries == 0:
        raise ValueError(f"no video queries to audit in {root}")
    return result


def _with_videos(audit: DatasetAudit, videos: Sequence[VideoAudit]) -> DatasetAudit:
    return DatasetAudit(
        dataset_path=audit.dataset_path,
        codebase_version=audit.codebase_version,
        fps=audit.fps,
        tolerance_s=audit.tolerance_s,
        tolerance_ns=audit.tolerance_ns,
        audited_episodes=audit.audited_episodes,
        videos=tuple(videos),
    )


# ------------------------------------------------------------------ render


def _us(nanoseconds: int) -> str:
    """Microseconds at nanosecond resolution: 100.000 us vs 100.001 us."""
    return f"{nanoseconds / 1000:.3f} us"


def _query_lines(label: str, query: PTSQuery) -> list[str]:
    return [
        f"    {label}: episode {query.episode_index}, frame {query.frame_index}",
        f"      parquet timestamp   {query.parquet_timestamp_ns} ns",
        f"      from_timestamp      {query.from_timestamp_ns} ns",
        f"      query               {query.query_ns} ns",
        f"      selected index      {query.selected_index}"
        + (" (outside timeline)" if query.index_out_of_range else ""),
        f"      selected PTS        {query.selected_pts_ns} ns",
        f"      nearest PTS         {query.nearest_pts_ns} ns "
        f"(index {query.nearest_index})",
        f"      residual            {query.signed_residual_ns:+d} ns "
        f"= {_us(query.signed_residual_ns)}",
        f"      legacy float32      {query.legacy_residual_ns:+d} ns "
        f"= {_us(query.legacy_residual_ns)}"
        + ("" if query.legacy_within_tolerance else "  (rejected by older LeRobot)"),
        f"      float32 spacing     {query.float32_ulp_ns} ns "
        f"= {_us(query.float32_ulp_ns)}",
    ]


def render_audit_text(audit: DatasetAudit) -> str:
    """A console report whose numbers are precise enough to act on."""
    lines = [
        f"dataset   {audit.dataset_path}",
        f"version   {audit.codebase_version}   fps {audit.fps}   "
        f"episodes audited {audit.audited_episodes}",
        f"tolerance {audit.tolerance_s} s = {audit.tolerance_ns} ns "
        f"({_us(audit.tolerance_ns)})",
        f"queries   {audit.total_queries}   violations {audit.total_violations}   "
        f"legacy-only violations {audit.total_legacy_violations}",
        "",
    ]
    for video in audit.videos:
        lines.append(f"  camera {video.camera_key}  -> {video.video_path}")
        lines.append(
            f"    episodes {list(video.episode_indices)}   "
            f"timeline {video.timeline_length} frames "
            f"(expected {video.evidence.expected_frame_count})   "
            f"average_fps {video.average_fps:g}"
        )
        if not video.timeline_verified:
            lines.append(
                "    note: the container declares no frame count, so the "
                "packet-derived timeline could not be verified"
            )
        lines.append(
            f"    likely_cause: {video.likely_cause}   "
            f"violations {video.violation_count}/{video.query_count}"
        )
        lines.append(
            "    |residual| ns  "
            f"p50 {video.p50_abs_residual_ns}  "
            f"p95 {video.p95_abs_residual_ns}  "
            f"p99 {video.p99_abs_residual_ns}  "
            f"max {video.max_abs_residual_ns} ({_us(video.max_abs_residual_ns)})"
        )
        if video.evidence.frame_count_mismatch:
            difference = video.timeline_length - video.evidence.expected_frame_count
            lines.append(
                f"    frame count: video holds {difference:+d} frames against the "
                f"{video.evidence.expected_frame_count} rows the metadata declares"
            )
        for check in video.boundary_faults:
            lines.append(
                f"    episode {check.episode_index} offset: declared "
                f"{check.declared_from_timestamp_ns} ns, frame "
                f"{check.expected_frame_index} sits at "
                f"{check.actual_boundary_pts_ns} ns, off by "
                f"{check.offset_ns:+d} ns"
                + (f" = {_us(check.offset_ns)}" if check.offset_ns is not None else "")
            )
        if video.first_violation is not None:
            lines.extend(_query_lines("first violation", video.first_violation))
        if video.worst_violation is not None and (
            video.worst_violation is not video.first_violation
        ):
            lines.extend(_query_lines("worst violation", video.worst_violation))
        if video.first_violation is None and video.first_legacy_violation is not None:
            lines.extend(
                _query_lines("first legacy-only violation", video.first_legacy_violation)
            )
        if video.violation_count:
            lines.append("    per-episode |residual| ns (mean / max):")
            for trend in video.episode_trend:
                lines.append(
                    f"      episode {trend.episode_index}: "
                    f"{trend.mean_abs_residual_ns} / {trend.max_abs_residual_ns}  "
                    f"({trend.violation_count}/{trend.query_count} over tolerance)"
                )
        lines.append("")

    scope = (
        " This audit compares Parquet timestamps, episode video offsets and "
        "encoded PTS only; it cannot speak to acquisition-time sensor sync, "
        "which LeRobot v3 does not retain."
    )
    if audit.passed:
        lines.append(
            "PASS: every audited query is within tolerance, every episode "
            "offset lands on its boundary frame, and every video holds the "
            "frames its metadata declares."
        )
    elif audit.total_violations:
        findings = f"{audit.total_violations} of {audit.total_queries} queries exceed the tolerance"
        if audit.total_legacy_violations:
            findings += (
                f"; {audit.total_legacy_violations} would also be rejected by an "
                "older float32 LeRobot"
            )
        lines.append(f"FAIL: {findings}." + scope)
    elif audit.total_legacy_violations:
        # The dataset current LeRobot accepts and an older one refuses. Saying
        # "0 queries exceed the tolerance" and stopping there is how a report
        # like #2814 gets answered with "works for me".
        lines.append(
            f"FAIL: every query is within tolerance under the current float64, "
            f"inclusive comparison, but {audit.total_legacy_violations} of "
            f"{audit.total_queries} would be rejected by an older LeRobot that "
            "compared in float32 with a strict '<'. Upgrading LeRobot, not "
            "editing the dataset, is what resolves this." + scope
        )
    else:
        lines.append(
            "FAIL: every query is within tolerance, but the episode offsets or "
            "frame counts above do not agree with the encoded video." + scope
        )
    return "\n".join(lines)
