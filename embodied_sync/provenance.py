"""Deterministic alignment provenance and replay verification.

The run and episode formats already preserve the data needed to load an
alignment.  This module adds *identity*: canonical SHA-256 fingerprints for
the source records, selected sample mapping, and recorded episode content.
It also records the fully resolved per-stream alignment policy so replay does
not silently inherit different tolerance defaults from a newer release.

Two guarantees are deliberately distinct:

``selection``
    Replay selects the same source sample identity at every target time.
``content``
    Selection matches and all recorded inline payloads / locally resolvable
    payload references have the same bytes.  This does not promise identical
    decoded pixels across different codec builds.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path

from embodied_sync import __version__
from embodied_sync.align import MethodArg, align_run
from embodied_sync.core import AlignmentPolicy
from embodied_sync.core.episode import AlignedFrame, AlignedRun
from embodied_sync.core.sample import Sample
from embodied_sync.datasets.io import sample_to_record
from embodied_sync.time import (
    ClockDomain,
    ClockKind,
    LatencyEstimate,
    latency_estimate_to_dict,
    translate_ns,
)

PROVENANCE_FORMAT_VERSION = 0
DIGEST_ALGORITHM = "sha256"

__all__ = [
    "DIGEST_ALGORITHM",
    "PROVENANCE_FORMAT_VERSION",
    "ReplayVerification",
    "build_provenance",
    "content_digest",
    "fingerprint_source",
    "parse_recorded_seeds",
    "selection_digest",
    "software_identity",
    "verify_replay",
]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _stream_fingerprint(samples: list[Sample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(_canonical_bytes(sample_to_record(sample)))
        digest.update(b"\n")
    return digest.hexdigest()


def _manifest_stream_is_metadata_only(
    source_manifest: Mapping[str, object], stream: str
) -> bool:
    raw_streams = source_manifest.get("streams")
    if not isinstance(raw_streams, dict):
        return False
    raw_info = raw_streams.get(stream)
    return isinstance(raw_info, dict) and raw_info.get("persist") == "metadata"


def _resolve_payload_ref(source_path: Path, payload_ref: str) -> Path | None:
    if "://" in payload_ref or "#" in payload_ref:
        return None
    candidate = Path(payload_ref)
    if not candidate.is_absolute():
        base = source_path if source_path.is_dir() else source_path.parent
        candidate = base / candidate
    return candidate if candidate.is_file() else None


def fingerprint_source(
    run: dict[str, list[Sample]],
    *,
    source_path: str | Path,
    source_manifest: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return a canonical identity for a loaded source run.

    Stream hashes include every persisted :class:`Sample` field.  Referenced
    local files are hashed once per distinct reference.  Opaque references
    (URLs, container keys, missing files) remain selection-verifiable but
    prevent a content-level claim.
    """

    path = Path(source_path)
    manifest: Mapping[str, object] = source_manifest or {}
    streams: dict[str, object] = {}
    payload_refs: set[str] = set()
    metadata_only = False
    for name, samples in run.items():
        streams[name] = {
            "sample_count": len(samples),
            DIGEST_ALGORITHM: _stream_fingerprint(samples),
        }
        metadata_only = metadata_only or _manifest_stream_is_metadata_only(
            manifest, name
        )
        payload_refs.update(
            sample.payload_ref
            for sample in samples
            if sample.payload_ref is not None
        )

    external_payloads: dict[str, object] = {}
    unresolved_payload = False
    for ref in sorted(payload_refs):
        resolved = _resolve_payload_ref(path, ref)
        if resolved is None:
            unresolved_payload = True
            external_payloads[ref] = {"status": "unavailable"}
            continue
        digest, size = _hash_file(resolved)
        external_payloads[ref] = {
            "status": "fingerprinted",
            DIGEST_ALGORITHM: digest,
            "size_bytes": size,
        }

    identity: dict[str, object] = {
        "streams": streams,
        "manifest_sha256": _sha256(manifest) if manifest else None,
        "external_payloads": external_payloads,
    }
    level = "selection" if metadata_only or unresolved_payload else "content"
    return {
        "algorithm": DIGEST_ALGORITHM,
        "digest": _sha256(identity),
        "reproducibility_level": level,
        **identity,
    }


def _selection_record(frame: AlignedFrame) -> dict[str, object]:
    return {
        "target_time_ns": frame.target_time_ns,
        "samples": {
            name: (
                None
                if sample is None
                else {
                    "stream_name": sample.stream_name,
                    "sequence_id": sample.sequence_id,
                    "acquisition_time_ns": sample.acquisition_time_ns,
                }
            )
            for name, sample in frame.samples.items()
        },
    }


def _frame_record(frame: AlignedFrame) -> dict[str, object]:
    return {
        "target_time_ns": frame.target_time_ns,
        "samples": {
            name: sample_to_record(sample) if sample is not None else None
            for name, sample in frame.samples.items()
        },
        "metadata": {
            name: {
                "source_time_ns": item.source_time_ns,
                "skew_ns": item.skew_ns,
                "method": item.method,
                "missing": item.missing,
                "confidence": item.confidence,
            }
            for name, item in frame.metadata.items()
        },
    }


def _frames_digest(frames: list[AlignedFrame], *, content: bool) -> str:
    digest = hashlib.sha256()
    for frame in frames:
        record = _frame_record(frame) if content else _selection_record(frame)
        digest.update(_canonical_bytes(record))
        digest.update(b"\n")
    return digest.hexdigest()


def selection_digest(aligned: AlignedRun) -> str:
    """Hash target times and selected source identities, excluding payloads."""

    return _frames_digest(aligned.frames, content=False)


def content_digest(aligned: AlignedRun) -> str:
    """Hash complete persisted frame records, including payload and metadata."""

    return _frames_digest(aligned.frames, content=True)


def _method_entry(
    method: MethodArg, stream: str
) -> tuple[str, int | None, str]:
    if isinstance(method, str):
        return method, None, "derived"
    entry = method.get(stream, "nearest_neighbor")
    if isinstance(entry, AlignmentPolicy):
        source = "explicit" if entry.tolerance_ns is not None else "derived"
        return entry.method, entry.tolerance_ns, source
    return entry, None, "derived"


def _resolved_policy(
    run: dict[str, list[Sample]],
    target_rate_hz: float,
    method: MethodArg,
    clock_mappings: Mapping[str, LatencyEstimate],
) -> dict[str, object]:
    period_ns = round(1e9 / target_rate_hz)
    resolved: dict[str, object] = {}
    for name, samples in run.items():
        picked, override, source = _method_entry(method, name)
        mapping = clock_mappings.get(name)
        times = [
            (
                sample.acquisition_time_ns
                if mapping is None
                else translate_ns(sample.acquisition_time_ns, mapping)
            )
            for sample in samples
        ]
        if len(times) >= 2:
            diffs = sorted(
                times[index + 1] - times[index]
                for index in range(len(times) - 1)
            )
            interval_ns = diffs[len(diffs) // 2]
        else:
            interval_ns = 0
        default_tolerance = (
            interval_ns // 2 if interval_ns > 0 else period_ns // 2
        )
        resolved[name] = {
            "method": picked,
            "tolerance_ns": override if override is not None else default_tolerance,
            "tolerance_source": source,
        }
    return resolved


def _requested_policy(method: MethodArg) -> object:
    if isinstance(method, str):
        return method
    return {
        stream: (
            entry
            if isinstance(entry, str)
            else {
                "method": entry.method,
                "tolerance_ns": entry.tolerance_ns,
            }
        )
        for stream, entry in method.items()
    }


def software_identity() -> dict[str, object]:
    """Return installed-package and source-control identity when available."""

    try:
        version = importlib_metadata.version("embodied-sync")
    except importlib_metadata.PackageNotFoundError:
        version = __version__

    identity: dict[str, object] = {
        "package": "embodied-sync",
        "version": version,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
    package_root = Path(__file__).resolve().parent
    implementation = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        implementation.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        implementation.update(b"\0")
        implementation.update(path.read_bytes())
        implementation.update(b"\0")
    identity["implementation_sha256"] = implementation.hexdigest()

    repository = package_root.parent
    if not (repository / ".git").exists():
        return identity
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return identity
    identity["git_revision"] = revision
    return identity


def parse_recorded_seeds(values: list[str]) -> dict[str, int]:
    """Parse repeatable ``NAME=INT`` seed declarations from the CLI."""

    seeds: dict[str, int] = {}
    for value in values:
        name, separator, raw_seed = value.partition("=")
        if not separator or not name.strip():
            raise ValueError(
                f"recorded seed must be NAME=INT, got {value!r}"
            )
        try:
            seed = int(raw_seed)
        except ValueError as exc:
            raise ValueError(
                f"recorded seed {name!r} must be an integer, got {raw_seed!r}"
            ) from exc
        if name in seeds:
            raise ValueError(f"recorded seed {name!r} was provided more than once")
        seeds[name] = seed
    return seeds


def _known_source_seeds(
    source_manifest: Mapping[str, object], supplied: Mapping[str, int]
) -> dict[str, int]:
    seeds = dict(supplied)
    synthetic = source_manifest.get("synthetic")
    if isinstance(synthetic, dict) and isinstance(synthetic.get("seed"), int):
        seeds.setdefault("embodied_sync.synthetic", synthetic["seed"])
    corruption = source_manifest.get("corruption")
    if isinstance(corruption, dict) and isinstance(
        corruption.get("profile_seed"), int
    ):
        seeds.setdefault(
            "embodied_sync.corruption", corruption["profile_seed"]
        )
    return seeds


def build_provenance(
    run: dict[str, list[Sample]],
    aligned: AlignedRun,
    *,
    source_path: str | Path,
    source_manifest: Mapping[str, object] | None,
    target_rate_hz: float,
    method: MethodArg,
    adapter: str = "run",
    recorded_seeds: Mapping[str, int] | None = None,
    clock_mappings: Mapping[str, LatencyEstimate] | None = None,
) -> dict[str, object]:
    """Build the versioned manifest block for one alignment."""

    manifest: Mapping[str, object] = source_manifest or {}
    source = fingerprint_source(
        run, source_path=source_path, source_manifest=manifest
    )
    source["path"] = str(Path(source_path))
    source["adapter"] = adapter
    level = str(source["reproducibility_level"])
    mappings = dict(clock_mappings or {})
    return {
        "format_version": PROVENANCE_FORMAT_VERSION,
        "source": source,
        "alignment": {
            "target_rate_hz": target_rate_hz,
            "target_period_ns": round(1e9 / target_rate_hz),
            "target_grid": {
                "first_time_ns": (
                    aligned.frames[0].target_time_ns if aligned.frames else None
                ),
                "last_time_ns": (
                    aligned.frames[-1].target_time_ns if aligned.frames else None
                ),
                "frame_count": len(aligned.frames),
            },
            "requested_policy": _requested_policy(method),
            "resolved_policy": _resolved_policy(
                run, target_rate_hz, method, mappings
            ),
            "clock_mappings": {
                stream: latency_estimate_to_dict(mapping)
                for stream, mapping in mappings.items()
            },
        },
        "outputs": {
            "selection_sha256": selection_digest(aligned),
            "content_sha256": content_digest(aligned),
        },
        "stochastic": {
            "seeds": _known_source_seeds(manifest, recorded_seeds or {})
        },
        "software": software_identity(),
        "guarantee": {
            "level": level,
            "decoded_media": False,
            "description": (
                "same selected sample identities and recorded content"
                if level == "content"
                else "same selected sample identities; payload bytes are not fully available"
            ),
        },
    }


def _policy_from_provenance(provenance: Mapping[str, object]) -> dict[str, AlignmentPolicy]:
    alignment = provenance.get("alignment")
    if not isinstance(alignment, dict):
        raise ValueError("provenance has no alignment block")
    raw_policy = alignment.get("resolved_policy")
    if not isinstance(raw_policy, dict):
        raise ValueError("provenance has no resolved alignment policy")
    policy: dict[str, AlignmentPolicy] = {}
    for stream, raw_entry in raw_policy.items():
        if not isinstance(stream, str) or not isinstance(raw_entry, dict):
            raise ValueError("provenance resolved policy has an invalid entry")
        method = raw_entry.get("method")
        tolerance = raw_entry.get("tolerance_ns")
        if not isinstance(method, str) or not isinstance(tolerance, int):
            raise ValueError(
                f"provenance resolved policy for {stream!r} is incomplete"
            )
        policy[stream] = AlignmentPolicy(method=method, tolerance_ns=tolerance)
    return policy


def _clock_mappings_from_provenance(
    provenance: Mapping[str, object],
) -> dict[str, LatencyEstimate]:
    alignment = provenance.get("alignment")
    if not isinstance(alignment, dict):
        raise ValueError("provenance has no alignment block")
    raw_mappings = alignment.get("clock_mappings", {})
    if not isinstance(raw_mappings, dict):
        raise ValueError("provenance clock_mappings must be an object")
    mappings: dict[str, LatencyEstimate] = {}
    for stream, raw in raw_mappings.items():
        if not isinstance(stream, str) or not isinstance(raw, dict):
            raise ValueError("provenance clock mapping has an invalid entry")
        source = raw.get("source")
        target = raw.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError(f"clock mapping for {stream!r} has invalid domains")
        def integer(name: str) -> int:
            value = raw.get(name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"clock mapping for {stream!r} has invalid {name}"
                )
            return value

        mappings[stream] = LatencyEstimate(
            source=ClockDomain(source, ClockKind.UNKNOWN),
            target=ClockDomain(target, ClockKind.UNKNOWN),
            offset_ns=integer("offset_ns"),
            drift_ppb=integer("drift_ppb"),
            anchor_time_ns=integer("anchor_time_ns"),
            variance_ns=integer("variance_ns"),
            epoch=integer("epoch"),
        )
    return mappings


def _software_matches(
    expected: object, actual: Mapping[str, object]
) -> bool:
    if not isinstance(expected, dict):
        return False
    required = ("package", "version", "python", "implementation_sha256")
    return all(expected.get(name) == actual.get(name) for name in required)


def _first_selection_difference(
    expected: AlignedRun, actual: AlignedRun
) -> str | None:
    if len(expected.frames) != len(actual.frames):
        return (
            f"frame count differs: recorded={len(expected.frames)}, "
            f"replayed={len(actual.frames)}"
        )
    for index, (left, right) in enumerate(zip(expected.frames, actual.frames)):
        if left.target_time_ns != right.target_time_ns:
            return (
                f"frame {index} target differs: recorded={left.target_time_ns}, "
                f"replayed={right.target_time_ns}"
            )
        stream_names = list(dict.fromkeys([*left.samples, *right.samples]))
        for stream in stream_names:
            left_sample = left.samples.get(stream)
            right_sample = right.samples.get(stream)
            left_id = (
                None
                if left_sample is None
                else (left_sample.sequence_id, left_sample.acquisition_time_ns)
            )
            right_id = (
                None
                if right_sample is None
                else (right_sample.sequence_id, right_sample.acquisition_time_ns)
            )
            if left_id != right_id:
                return (
                    f"frame {index} stream {stream!r} selection differs: "
                    f"recorded={left_id}, replayed={right_id}"
                )
    return None


def _changed_source_parts(
    expected: Mapping[str, object], actual: Mapping[str, object]
) -> list[str]:
    changed: list[str] = []
    expected_streams = expected.get("streams")
    actual_streams = actual.get("streams")
    if isinstance(expected_streams, dict) and isinstance(actual_streams, dict):
        for name in sorted(set(expected_streams) | set(actual_streams)):
            if expected_streams.get(name) != actual_streams.get(name):
                changed.append(f"source stream {name!r} fingerprint changed")
    if expected.get("manifest_sha256") != actual.get("manifest_sha256"):
        changed.append("source manifest fingerprint changed")
    if expected.get("external_payloads") != actual.get("external_payloads"):
        changed.append("one or more external payload fingerprints changed")
    return changed


@dataclass(frozen=True, slots=True)
class ReplayVerification:
    """Structured result returned by :func:`verify_replay`."""

    verified: bool
    source_matches: bool
    software_matches: bool
    episode_integrity_matches: bool
    selection_matches: bool
    content_matches: bool | None
    messages: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "verified": self.verified,
            "source_matches": self.source_matches,
            "software_matches": self.software_matches,
            "episode_integrity_matches": self.episode_integrity_matches,
            "selection_matches": self.selection_matches,
            "content_matches": self.content_matches,
            "messages": list(self.messages),
        }


def verify_replay(
    run: dict[str, list[Sample]],
    recorded_episode: AlignedRun,
    provenance: Mapping[str, object],
    *,
    source_path: str | Path,
    source_manifest: Mapping[str, object] | None,
) -> ReplayVerification:
    """Replay an alignment from recorded provenance and verify its digests."""

    version = provenance.get("format_version")
    if version != PROVENANCE_FORMAT_VERSION:
        raise ValueError(
            f"unsupported provenance format_version {version!r}; "
            f"this build reads {PROVENANCE_FORMAT_VERSION}"
        )
    expected_source = provenance.get("source")
    outputs = provenance.get("outputs")
    alignment = provenance.get("alignment")
    if not isinstance(expected_source, dict) or not isinstance(outputs, dict):
        raise ValueError("provenance is missing source or output fingerprints")
    if not isinstance(alignment, dict):
        raise ValueError("provenance is missing alignment configuration")

    actual_source = fingerprint_source(
        run,
        source_path=source_path,
        source_manifest=source_manifest,
    )
    source_matches = expected_source.get("digest") == actual_source.get("digest")
    messages = [] if source_matches else _changed_source_parts(
        expected_source, actual_source
    )
    if not source_matches and not messages:
        messages.append("source fingerprint changed")

    expected_software = provenance.get("software")
    actual_software = software_identity()
    software_matches = _software_matches(expected_software, actual_software)
    if not software_matches:
        messages.append(
            f"software identity changed: recorded={expected_software!r}, "
            f"current={actual_software!r}"
        )

    target_rate = alignment.get("target_rate_hz")
    if not isinstance(target_rate, (int, float)) or isinstance(target_rate, bool):
        raise ValueError("provenance target_rate_hz is invalid")
    policy = _policy_from_provenance(provenance)
    clock_mappings = _clock_mappings_from_provenance(provenance)
    replayed = align_run(
        run,
        target_rate_hz=float(target_rate),
        method=policy,
        clock_map=clock_mappings,
    )

    expected_selection = outputs.get("selection_sha256")
    expected_content = outputs.get("content_sha256")
    selection_matches = selection_digest(replayed) == expected_selection
    if not selection_matches:
        messages.append(
            _first_selection_difference(recorded_episode, replayed)
            or "replayed selection fingerprint changed"
        )

    recorded_selection_matches = (
        selection_digest(recorded_episode) == expected_selection
    )
    recorded_content_matches = content_digest(recorded_episode) == expected_content
    episode_integrity_matches = recorded_selection_matches and recorded_content_matches
    if not episode_integrity_matches:
        messages.append("recorded episode content no longer matches its provenance")

    level = expected_source.get("reproducibility_level")
    content_matches: bool | None = None
    if level == "content":
        content_matches = content_digest(replayed) == expected_content
        if not content_matches:
            messages.append("replayed frame content fingerprint changed")

    verified = (
        source_matches
        and software_matches
        and episode_integrity_matches
        and selection_matches
        and content_matches is not False
    )
    if verified:
        messages.append(
            "content replay verified"
            if content_matches is True
            else "selection replay verified; payload content was not fully available"
        )
    return ReplayVerification(
        verified=verified,
        source_matches=source_matches,
        software_matches=software_matches,
        episode_integrity_matches=episode_integrity_matches,
        selection_matches=selection_matches,
        content_matches=content_matches,
        messages=tuple(messages),
    )
