"""LeRobot adapter: v3.0 native reader + Milestone 5 synthetic contract path.

Two entry points:

- :func:`load_lerobot_dataset` (D-0033) reads a real LeRobot **v3.0**
  dataset directory (``meta/info.json``, ``meta/tasks.parquet``,
  ``meta/episodes/*.parquet``, chunked data parquet files, video/image
  features) into the canonical run model. ``pyarrow`` is imported lazily
  inside the function so the base import surface stays light.
- :func:`load_lerobot_run` reads the deterministic JSON contract fixture
  used by CI (no optional dependencies at all).

Timestamp semantics for the native reader
-----------------------------------------
LeRobot stores per-episode-relative ``timestamp`` as float32 seconds.
The reader converts each stored value to integer nanoseconds with
``round(float(ts) * 1e9)`` — float32 quantization (e.g. ``0.1`` →
``100_000_001 ns``) is preserved, not regridded: it *is* the dataset's
timing. Episodes are composed onto one global monotonic timeline by
offsetting episode ``e`` by ``round(1e9 * frames_before_e / fps)``
(the frame-grid boundary, matching the concatenated-video
``from_timestamp`` bookkeeping). LeRobot has no delivery-time concept,
so ``receive_time_ns = acquisition_time_ns`` — the importer must not
invent transport latency.

Video features become payload-ref streams: each sample's
``payload_ref`` names the concatenated episode video file plus the
seek timestamp inside it (``from_timestamp`` + in-episode timestamp)
and the in-episode frame index. Parquet-embedded ``image`` features
reference their data file, row, and column instead.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from embodied_sync.core.sample import Modality, Sample
from embodied_sync.datasets.io import _record_to_sample
from embodied_sync.exporters.lerobot import _DATASET_NAME, _FORMAT

__all__ = ["load_lerobot_dataset", "load_lerobot_run"]

#: Parquet columns that describe rows rather than sensor data; never streams.
_BOOKKEEPING = frozenset(
    {"timestamp", "episode_index", "frame_index", "index", "task_index"}
)
_NUMERIC_DTYPES = frozenset({"float32", "float64"})

#: All samples share the dataset's composed global timeline.
LEROBOT_CLOCK_DOMAIN = "lerobot"


def _feature_modality(name: str, dtype: str) -> Modality:
    if dtype in ("video", "image") or "image" in name:
        return Modality.CAMERA
    if "state" in name or "effort" in name or "velocity" in name:
        return Modality.ROBOT_STATE
    if name == "action" or name.startswith("action."):
        return Modality.ACTION
    return Modality.OTHER


def _read_tasks(meta_dir: Path) -> dict[int, str]:
    """Map ``task_index`` to task string from ``meta/tasks.parquet``.

    The task string column is the parquet file's pandas index artifact
    (``__index_level_0__``) in real exports; fall back to the first
    non-``task_index`` string column.
    """
    import pyarrow.parquet as pq

    path = meta_dir / "tasks.parquet"
    if not path.is_file():
        return {}
    table = pq.read_table(path)
    names = [n for n in table.schema.names if n != "task_index"]
    if not names or "task_index" not in table.schema.names:
        return {}
    indices = table.column("task_index").to_pylist()
    strings = table.column(names[0]).to_pylist()
    return {int(i): str(s) for i, s in zip(indices, strings)}


def _read_episode_metadata(meta_dir: Path) -> list[dict[str, Any]]:
    """All episode rows from ``meta/episodes/**/*.parquet``, sorted by index."""
    import pyarrow.parquet as pq

    episodes_dir = meta_dir / "episodes"
    files = sorted(episodes_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no episode metadata parquet files under {episodes_dir}"
        )
    rows: list[dict[str, Any]] = []
    for f in files:
        rows.extend(pq.read_table(f).to_pylist())
    rows.sort(key=lambda r: int(r["episode_index"]))
    return rows


def _data_file_row_bases(meta_dir: Path) -> dict[tuple[int, int], int]:
    """Global row offset of each data file, keyed by ``(chunk, file)``.

    ``dataset_from_index`` is a global row number and each file holds a
    contiguous range, so a file's base is the smallest from-index among the
    episodes stored in it. Computed over the *full* metadata so importing a
    subset of episodes cannot shift a base.
    """
    bases: dict[tuple[int, int], int] = {}
    for row in _read_episode_metadata(meta_dir):
        key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        base = int(row["dataset_from_index"])
        bases[key] = min(base, bases.get(key, base))
    return bases


def load_lerobot_dataset(
    path: str | Path,
    *,
    max_episodes: int | None = None,
) -> tuple[dict[str, list[Sample]], dict[str, Any]]:
    """Load a real LeRobot v3.0 dataset directory as a canonical run.

    Returns ``(run, dataset_info)``. ``run`` maps stream name to samples
    on one global monotonic timeline; ``dataset_info`` carries the
    provenance a run manifest should record: source path, codebase
    version, fps, robot type, tasks, and the per-episode boundary table
    (``episode_index``, ``start_time_ns``, ``length``, ``tasks``).

    ``max_episodes`` imports only the first N episodes (dataset order) —
    the boundary table still reflects exactly what was imported.

    Requires ``pyarrow`` (installed with the ``lerobot`` extra). Raises
    :class:`FileNotFoundError` for a missing dataset and
    :class:`ValueError` for unsupported layouts (e.g. v2.x ``jsonl``
    metadata).
    """
    import pyarrow.parquet as pq

    root = Path(path)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"not a LeRobot dataset (no meta/info.json): {root}"
        )
    info = json.loads(info_path.read_text(encoding="utf-8"))
    version = str(info.get("codebase_version", ""))
    if not version.startswith("v3"):
        raise ValueError(
            f"unsupported LeRobot codebase_version {version!r} in {info_path}; "
            "this reader supports v3.x (v2.x jsonl-metadata layouts are not "
            "implemented yet)"
        )
    fps = float(info["fps"])
    if fps <= 0:
        raise ValueError(f"invalid fps {fps!r} in {info_path}")
    data_path_tpl = str(info["data_path"])
    video_path_tpl = str(info.get("video_path", ""))
    features: dict[str, Any] = dict(info["features"])

    numeric_keys = [
        name
        for name, spec in features.items()
        if name not in _BOOKKEEPING and str(spec.get("dtype")) in _NUMERIC_DTYPES
    ]
    video_keys = [n for n, s in features.items() if str(s.get("dtype")) == "video"]
    image_keys = [n for n, s in features.items() if str(s.get("dtype")) == "image"]

    tasks = _read_tasks(root / "meta")
    episode_rows = _read_episode_metadata(root / "meta")
    if max_episodes is not None:
        episode_rows = episode_rows[:max_episodes]
    if not episode_rows:
        raise ValueError(f"LeRobot dataset has no episodes: {root}")

    file_base = _data_file_row_bases(root / "meta")

    stream_order = list(features)  # info.json order drives stream order
    run: dict[str, list[Sample]] = {
        name: []
        for name in stream_order
        if name in numeric_keys or name in video_keys or name in image_keys
    }
    seq: dict[str, int] = {name: 0 for name in run}
    boundaries: list[dict[str, Any]] = []

    needed_columns = ["timestamp", "frame_index", *numeric_keys, *image_keys]
    table_cache: dict[tuple[int, int], Any] = {}
    frames_before = 0
    for ep in episode_rows:
        episode_index = int(ep["episode_index"])
        length = int(ep["length"])
        chunk = int(ep["data/chunk_index"])
        file_idx = int(ep["data/file_index"])
        key = (chunk, file_idx)
        if key not in table_cache:
            data_file = root / data_path_tpl.format(
                chunk_index=chunk, file_index=file_idx
            )
            available = set(pq.ParquetFile(data_file).schema_arrow.names)
            table_cache[key] = pq.read_table(
                data_file, columns=[c for c in needed_columns if c in available]
            )
        table = table_cache[key]
        local_start = int(ep["dataset_from_index"]) - file_base[key]
        local_stop = int(ep["dataset_to_index"]) - file_base[key]
        if local_start < 0 or local_stop > table.num_rows:
            raise ValueError(
                f"episode {episode_index}: row range [{local_start}, {local_stop}) "
                f"outside data file with {table.num_rows} rows"
            )
        rows = table.slice(local_start, local_stop - local_start)

        offset_ns = round(1e9 * frames_before / fps)
        ts_s: list[float] = [float(v) for v in rows.column("timestamp").to_pylist()]
        acq_ns = [offset_ns + round(t * 1e9) for t in ts_s]
        frame_indices = [int(v) for v in rows.column("frame_index").to_pylist()]
        data_rel = data_path_tpl.format(chunk_index=chunk, file_index=file_idx)

        for name in numeric_keys:
            if name not in rows.schema.names:
                continue
            modality = _feature_modality(name, str(features[name].get("dtype")))
            values = rows.column(name).to_pylist()
            samples = run[name]
            for i, value in enumerate(values):
                payload = (
                    [float(v) for v in value]
                    if isinstance(value, (list, tuple))
                    else float(value)
                )
                samples.append(
                    Sample(
                        stream_name=name,
                        modality=modality,
                        sequence_id=seq[name],
                        acquisition_time_ns=acq_ns[i],
                        receive_time_ns=acq_ns[i],
                        source_clock_domain=LEROBOT_CLOCK_DOMAIN,
                        payload=payload,
                    )
                )
                seq[name] += 1

        for name in video_keys:
            video_chunk = int(ep[f"videos/{name}/chunk_index"])
            video_file = int(ep[f"videos/{name}/file_index"])
            from_ts = float(ep[f"videos/{name}/from_timestamp"])
            video_rel = video_path_tpl.format(
                video_key=name, chunk_index=video_chunk, file_index=video_file
            )
            samples = run[name]
            for i in range(len(acq_ns)):
                ref = (
                    f"{video_rel}#t={from_ts + ts_s[i]:.9f}"
                    f"&episode_frame={frame_indices[i]}"
                )
                samples.append(
                    Sample(
                        stream_name=name,
                        modality=Modality.CAMERA,
                        sequence_id=seq[name],
                        acquisition_time_ns=acq_ns[i],
                        receive_time_ns=acq_ns[i],
                        source_clock_domain=LEROBOT_CLOCK_DOMAIN,
                        payload_ref=ref,
                    )
                )
                seq[name] += 1

        for name in image_keys:
            if name not in rows.schema.names:
                continue
            samples = run[name]
            for i in range(len(acq_ns)):
                ref = f"{data_rel}#row={local_start + i}&column={name}"
                samples.append(
                    Sample(
                        stream_name=name,
                        modality=Modality.CAMERA,
                        sequence_id=seq[name],
                        acquisition_time_ns=acq_ns[i],
                        receive_time_ns=acq_ns[i],
                        source_clock_domain=LEROBOT_CLOCK_DOMAIN,
                        payload_ref=ref,
                    )
                )
                seq[name] += 1

        boundaries.append(
            {
                "episode_index": episode_index,
                "start_time_ns": offset_ns,
                "length": length,
                "tasks": [str(t) for t in ep.get("tasks") or []],
            }
        )
        frames_before += length

    for name, samples in run.items():
        for prev, cur in zip(samples, samples[1:]):
            if cur.acquisition_time_ns < prev.acquisition_time_ns:
                warnings.warn(
                    f"lerobot import: stream {name!r} is non-monotonic at "
                    f"sequence {cur.sequence_id} — source timestamps overrun "
                    "their episode's frame-grid boundary",
                    stacklevel=2,
                )
                break

    dataset_info: dict[str, Any] = {
        "source_path": str(root),
        "codebase_version": version,
        "fps": fps,
        "robot_type": info.get("robot_type"),
        "total_episodes": info.get("total_episodes"),
        "imported_episodes": len(episode_rows),
        "tasks": {str(i): s for i, s in sorted(tasks.items())},
        "episodes": boundaries,
    }
    return run, dataset_info


def load_lerobot_run(path: str | Path, *, episode_index: int = 0) -> dict[str, list[Sample]]:
    """Load a deterministic local LeRobot-style dataset directory.

    External Hugging Face downloads are deliberately out of scope; callers
    provide a local path. The optional ``lerobot`` package is not imported for
    this CI contract path.
    """
    dataset_path = Path(path) / _DATASET_NAME
    document = json.loads(dataset_path.read_text(encoding="utf-8"))
    if document.get("format") != _FORMAT:
        raise ValueError(
            f"unsupported LeRobot contract format {document.get('format')!r} "
            f"in {dataset_path}"
        )
    if document.get("type") != "run":
        raise ValueError(f"expected LeRobot run document, got {document.get('type')!r}")
    episodes = document.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"invalid LeRobot dataset in {dataset_path}: missing episodes")
    for episode in episodes:
        if episode.get("episode_index") == episode_index:
            streams = episode.get("streams")
            if not isinstance(streams, dict):
                raise ValueError(
                    f"invalid LeRobot episode {episode_index}: missing streams"
                )
            return {
                name: [_record_to_sample(record) for record in records]
                for name, records in streams.items()
            }
    raise ValueError(f"LeRobot episode_index {episode_index} not found in {dataset_path}")
