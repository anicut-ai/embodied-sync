"""Deterministic replay provenance and CLI verification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from embodied_sync.align import align_run
from embodied_sync.cli.main import main
from embodied_sync.datasets.io import save_run
from embodied_sync.provenance import (
    build_provenance,
    fingerprint_source,
    verify_replay,
)
from embodied_sync.streams.synthetic import generate_synthetic_run
from embodied_sync.time import ClockDomain, ClockKind, LatencyEstimate


def _recorded_episode(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    episode_dir = tmp_path / "episode"
    save_run(
        generate_synthetic_run(duration_s=0.5, seed=7),
        run_dir,
        extra_manifest={
            "synthetic": {"seed": 7, "duration_s": 0.5, "start_time_ns": 0}
        },
    )
    assert (
        main(
            [
                "align",
                str(run_dir),
                "--out",
                str(episode_dir),
                "--target-rate-hz",
                "10",
                "--record-seed",
                "policy_sampler=123",
            ]
        )
        == 0
    )
    return run_dir, episode_dir


def _manifest(directory: Path) -> dict[str, object]:
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))


def test_align_records_versioned_provenance_and_resolved_policy(
    tmp_path: Path,
) -> None:
    run_dir, episode_dir = _recorded_episode(tmp_path)
    manifest = _manifest(episode_dir)
    provenance = manifest["provenance"]
    assert provenance["format_version"] == 0
    assert provenance["source"]["path"] == str(run_dir)
    assert provenance["source"]["digest"]
    assert provenance["source"]["reproducibility_level"] == "content"
    assert provenance["outputs"]["selection_sha256"]
    assert provenance["outputs"]["content_sha256"]
    assert provenance["software"]["version"]

    policies = provenance["alignment"]["resolved_policy"]
    assert set(policies) == set(manifest["streams"])
    assert all(entry["method"] == "nearest_neighbor" for entry in policies.values())
    assert all(isinstance(entry["tolerance_ns"], int) for entry in policies.values())
    assert all(entry["tolerance_source"] == "derived" for entry in policies.values())

    seeds = provenance["stochastic"]["seeds"]
    assert seeds["embodied_sync.synthetic"] == 7
    assert seeds["policy_sampler"] == 123


def test_replay_cli_verifies_unchanged_episode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, episode_dir = _recorded_episode(tmp_path)
    capsys.readouterr()
    assert main(["replay", str(episode_dir), "--verify"]) == 0
    output = capsys.readouterr().out
    assert "PASS" in output
    assert "content replay verified" in output


def test_replay_cli_accepts_relocated_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir, episode_dir = _recorded_episode(tmp_path)
    manifest_path = episode_dir / "manifest.json"
    manifest = _manifest(episode_dir)
    manifest["provenance"]["source"]["path"] = "/does/not/exist"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    capsys.readouterr()

    assert (
        main(
            [
                "replay",
                str(episode_dir),
                "--verify",
                "--source",
                str(run_dir),
            ]
        )
        == 0
    )
    assert "PASS" in capsys.readouterr().out


def test_replay_reports_changed_source_stream(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir, episode_dir = _recorded_episode(tmp_path)
    stream_path = run_dir / "streams" / "cam_front.jsonl"
    records = stream_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(records[0])
    first["receive_time_ns"] += 1
    records[0] = json.dumps(first, separators=(",", ":"))
    stream_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    capsys.readouterr()

    assert main(["replay", str(episode_dir), "--verify"]) == 1
    output = capsys.readouterr().out
    assert "FAIL" in output
    assert "source stream 'cam_front' fingerprint changed" in output


def test_replay_reports_modified_recorded_episode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, episode_dir = _recorded_episode(tmp_path)
    frames_path = episode_dir / "frames.jsonl"
    records = frames_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(records[0])
    first_stream = next(
        name for name, sample in first["samples"].items() if sample is not None
    )
    first["samples"][first_stream]["receive_time_ns"] += 1
    records[0] = json.dumps(first, separators=(",", ":"))
    frames_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    capsys.readouterr()

    assert main(["replay", str(episode_dir), "--verify"]) == 1
    output = capsys.readouterr().out
    assert "recorded episode content no longer matches" in output


def test_replay_detects_software_version_change(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, episode_dir = _recorded_episode(tmp_path)
    manifest_path = episode_dir / "manifest.json"
    manifest = _manifest(episode_dir)
    manifest["provenance"]["software"]["version"] = "different-version"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    capsys.readouterr()

    assert main(["replay", str(episode_dir), "--verify"]) == 1
    output = capsys.readouterr().out
    assert "software identity changed" in output


def test_metadata_only_session_claims_selection_not_content(tmp_path: Path) -> None:
    run = generate_synthetic_run(duration_s=0.1, seed=0)
    manifest = {
        "format_version": 0,
        "streams": {
            name: {"persist": "metadata", "sample_count": len(samples)}
            for name, samples in run.items()
        },
    }
    fingerprint = fingerprint_source(
        run,
        source_path=tmp_path,
        source_manifest=manifest,
    )
    assert fingerprint["reproducibility_level"] == "selection"


def test_local_payload_reference_is_hashed(tmp_path: Path) -> None:
    payload_path = tmp_path / "frame.bin"
    payload_path.write_bytes(b"frame-v1")
    run = generate_synthetic_run(duration_s=0.1, seed=0)
    sample = run["cam_front"][0]
    run["cam_front"][0] = type(sample)(
        stream_name=sample.stream_name,
        modality=sample.modality,
        sequence_id=sample.sequence_id,
        acquisition_time_ns=sample.acquisition_time_ns,
        receive_time_ns=sample.receive_time_ns,
        source_clock_domain=sample.source_clock_domain,
        payload=sample.payload,
        payload_ref="frame.bin",
        quality_flags=sample.quality_flags,
    )

    before = fingerprint_source(run, source_path=tmp_path)
    assert before["reproducibility_level"] == "content"
    assert before["external_payloads"]["frame.bin"]["status"] == "fingerprinted"
    payload_path.write_bytes(b"frame-v2")
    after = fingerprint_source(run, source_path=tmp_path)
    assert after["digest"] != before["digest"]


def test_replay_reconstructs_recorded_clock_mapping(tmp_path: Path) -> None:
    run = generate_synthetic_run(duration_s=0.5, seed=0)
    mapping = LatencyEstimate(
        source=ClockDomain("cam_front_hw", ClockKind.HARDWARE),
        target=ClockDomain("host_mono", ClockKind.MONOTONIC),
        offset_ns=5_000_000,
        drift_ppb=250,
        anchor_time_ns=0,
        variance_ns=100,
    )
    clock_map = {"cam_front": mapping}
    aligned = align_run(
        run,
        target_rate_hz=10.0,
        clock_map=clock_map,
    )
    provenance = build_provenance(
        run,
        aligned,
        source_path=tmp_path,
        source_manifest={},
        target_rate_hz=10.0,
        method="nearest_neighbor",
        clock_mappings=clock_map,
    )

    recorded_mapping = provenance["alignment"]["clock_mappings"]["cam_front"]
    assert recorded_mapping["offset_ns"] == 5_000_000
    result = verify_replay(
        run,
        aligned,
        provenance,
        source_path=tmp_path,
        source_manifest={},
    )
    assert result.verified
    assert result.selection_matches
    assert result.content_matches is True


def test_invalid_recorded_seed_fails_alignment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    save_run(generate_synthetic_run(duration_s=0.1, seed=0), run_dir)
    assert (
        main(
            [
                "align",
                str(run_dir),
                "--out",
                str(tmp_path / "episode"),
                "--target-rate-hz",
                "10",
                "--record-seed",
                "policy_sampler=not-an-int",
            ]
        )
        == 1
    )
    assert "must be an integer" in capsys.readouterr().err
