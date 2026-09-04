"""`embsync` command-line entry point.

Milestone 1 subcommands: `synth`, `corrupt`, `align`, `report`.
Milestone 5 native-format subcommands (D-0033): `import-lerobot`,
`export-lerobot`.
Increment-2 calibration subcommand (A5): `calibrate clap`.

Every subcommand here is a shell: parse arguments, call one library
function, print a summary, translate exceptions into an exit code. No
subcommand contains logic that a library caller could not reach — which
is why `calibrate clap`'s audio loading lives in
:mod:`embodied_sync.calibrate.audio_io` and its output shape in
:mod:`embodied_sync.calibrate.report`, not here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

from embodied_sync import __version__
from embodied_sync.align import (
    LINEAR_INTERPOLATION,
    NEAREST_NEIGHBOR,
    ZERO_ORDER_HOLD,
    Method,
    MethodArg,
    align_run,
)
from embodied_sync.core import AlignmentPolicy
from embodied_sync.core.sample import Sample
from embodied_sync.corrupt import CorruptionResult, apply_profile, load_profile
from embodied_sync.datasets.io import (
    CORRUPTION_GROUND_TRUTH_NAME,
    load_corruption_ground_truth,
    load_episode,
    load_run,
    save_corruption_ground_truth,
    save_episode,
    save_run,
)
from embodied_sync.adapters.lerobot import load_lerobot_dataset
from embodied_sync.adapters.mcap import load_mcap_run
from embodied_sync.adapters.qut import load_qut_dataset
from embodied_sync.exporters.lerobot import export_lerobot_dataset
from embodied_sync.exporters.mcap import save_mcap_run
from embodied_sync.exporters.umi import export_umi_zarr
from embodied_sync.ingest import (
    AmbiguousImportError,
    DatasetImportAgent,
    ImportPlan,
    InferenceResult,
    load_import_plan,
    load_inference_result,
    plan_source_rate_hz,
    save_json_document,
)
from embodied_sync.provenance import (
    build_provenance,
    parse_recorded_seeds,
    verify_replay,
)
from embodied_sync.reports import build_report, report_summary_dict, save_report_html
from embodied_sync.streams.synthetic import generate_synthetic_run

_RUN_ADAPTER = "run"
_MCAP_ADAPTER = "mcap"
_RUN_ADAPTERS = [_RUN_ADAPTER, _MCAP_ADAPTER]


def build_parser() -> argparse.ArgumentParser:
    # Imported here rather than at module scope so the CLI's import surface
    # stays as light as every other subcommand's (D-0001).
    from embodied_sync.inspect.lerobot_pts import DEFAULT_TOLERANCE_S

    parser = argparse.ArgumentParser(
        prog="embsync",
        description="Sync-quality validation for robot-learning data.",
    )
    parser.add_argument("--version", action="version", version=f"embodied-sync {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_synth = sub.add_parser("synth", help="Generate a deterministic synthetic run.")
    p_synth.add_argument("--out", required=True, help="Output run directory.")
    p_synth.add_argument(
        "--adapter",
        choices=_RUN_ADAPTERS,
        default=_RUN_ADAPTER,
        help="Output adapter format (default: run).",
    )
    p_synth.add_argument("--seed", type=int, default=0)
    p_synth.add_argument("--duration-s", type=float, default=10.0)
    p_synth.add_argument(
        "--start-time-ns",
        type=int,
        default=0,
        help=(
            "Integer-ns offset for the first acquisition timestamp of every "
            "regular stream (default 0). Threaded verbatim into the synth "
            "harness and echoed into the manifest under `synthetic`."
        ),
    )

    p_corrupt = sub.add_parser("corrupt", help="Apply a corruption profile to a run.")
    p_corrupt.add_argument("run_dir")
    p_corrupt.add_argument(
        "--adapter",
        choices=_RUN_ADAPTERS,
        default=_RUN_ADAPTER,
        help="Input/output adapter format (default: run).",
    )
    p_corrupt.add_argument("--profile", required=True, help="Corruption profile YAML.")
    p_corrupt.add_argument(
        "--out",
        default=None,
        help=(
            "Output run directory. Required unless --preview is set (preview "
            "mode does not touch disk)."
        ),
    )
    p_corrupt.add_argument(
        "--preview",
        action="store_true",
        help=(
            "Show what would change on stdout without writing anything. "
            "Prints per-stream sample counts and receive-time deltas so a "
            "new profile can be iterated on without an intermediate "
            "`rm -rf` cycle."
        ),
    )

    p_align = sub.add_parser("align", help="Align a run into policy-ready frames.")
    p_align.add_argument("run_dir")
    p_align.add_argument(
        "--adapter",
        choices=_RUN_ADAPTERS,
        default=_RUN_ADAPTER,
        help="Input adapter format (default: run).",
    )
    p_align.add_argument("--out", required=True)
    p_align.add_argument(
        "--target-rate-hz",
        type=float,
        default=None,
        help=(
            "Alignment target frame rate in Hz (e.g. 10.0 for a 10 Hz "
            "policy). Defaults to the run manifest's source_rate_hz when "
            "the run was imported from a rated source (e.g. import-lerobot)."
        ),
    )
    p_align.add_argument(
        "--check-ground-truth",
        action="store_true",
        help=(
            f"Load {CORRUPTION_GROUND_TRUTH_NAME} from run_dir and populate "
            "AlignmentReport.ground_truth_missing_count."
        ),
    )
    p_align.add_argument(
        "--method",
        choices=[NEAREST_NEIGHBOR, ZERO_ORDER_HOLD, LINEAR_INTERPOLATION],
        default=NEAREST_NEIGHBOR,
        help="Alignment policy (default: nearest_neighbor).",
    )
    p_align.add_argument(
        "--alignment-policy",
        default=None,
        help=(
            "JSON per-stream policy mapping accepted by align_run, e.g. "
            '{"cam_front":"zoh","robot_state":{"method":"linear_interp",'
            '"tolerance_ns":5000000}}. Overrides --method when supplied.'
        ),
    )
    p_align.add_argument(
        "--record-seed",
        action="append",
        default=[],
        metavar="NAME=INT",
        help=(
            "Record a namespaced stochastic seed in episode provenance; "
            "repeatable (for example policy_sampler=42). Embodied-Sync's "
            "synthetic/corruption seeds are copied automatically when present."
        ),
    )

    p_replay = sub.add_parser(
        "replay",
        help="Replay and verify an aligned episode from its provenance.",
    )
    p_replay.add_argument("episode_dir", help="Aligned episode to verify.")
    p_replay.add_argument(
        "--source",
        default=None,
        help="Source run path; defaults to the path recorded in provenance.",
    )
    p_replay.add_argument(
        "--adapter",
        choices=_RUN_ADAPTERS,
        default=None,
        help="Source adapter; defaults to the adapter recorded in provenance.",
    )
    p_replay.add_argument(
        "--verify",
        action="store_true",
        required=True,
        help="Verify source, software, selected samples, and available content.",
    )
    p_replay.add_argument(
        "--json",
        action="store_true",
        help="Print the structured verification result as JSON.",
    )

    p_report = sub.add_parser("report", help="Generate a sync-quality report.")
    p_report.add_argument(
        "episode_dir",
        help=(
            "Aligned-episode directory, or a run directory (the run is "
            "aligned in-memory at --target-rate-hz / the manifest's "
            "source_rate_hz first)."
        ),
    )
    p_report.add_argument(
        "--out",
        default=None,
        help="Report output path (HTML). Default: <episode_dir>/sync_report.html.",
    )
    p_report.add_argument(
        "--target-rate-hz",
        type=float,
        default=None,
        help=(
            "Only used when episode_dir is a run directory: alignment rate "
            "for the in-memory alignment (defaults to the run manifest's "
            "source_rate_hz)."
        ),
    )
    p_report.add_argument(
        "--json-summary",
        default=None,
        help="Optional path to also write the structured summary as JSON.",
    )
    p_report.add_argument(
        "--title",
        default="Sync-quality report",
        help="HTML page title.",
    )

    p_calibrate = sub.add_parser(
        "calibrate",
        help="Measure a clock mapping from physical calibration events.",
    )
    calibrate_sub = p_calibrate.add_subparsers(dest="calibrate_command")
    p_clap = calibrate_sub.add_parser(
        "clap",
        help="Fit an audio-clock -> event-clock mapping from clap transients.",
    )
    p_clap.add_argument(
        "--audio",
        default=None,
        help=(
            "Audio recording of the claps (.wav PCM, or .npy with "
            "--sample-rate-hz). Mutually exclusive with --audio-events."
        ),
    )
    p_clap.add_argument(
        "--audio-events",
        default=None,
        help=(
            "JSON file of already-detected audio onset times in integer ns, "
            "for callers running their own detector. Skips --audio entirely."
        ),
    )
    p_clap.add_argument(
        "--events",
        required=True,
        help=(
            "JSON file of visual/reference event times in integer ns: either "
            'a bare array or an object with an "events_ns" key.'
        ),
    )
    p_clap.add_argument("--out", required=True, help="Calibration JSON output path.")
    p_clap.add_argument(
        "--sample-rate-hz",
        type=float,
        default=None,
        help="Sample rate for a .npy --audio array (WAV carries its own).",
    )
    p_clap.add_argument(
        "--start-time-ns",
        type=int,
        default=0,
        help="Clock time of the first audio sample (default 0).",
    )
    p_clap.add_argument(
        "--max-offset-ms",
        type=float,
        default=1000.0,
        help=(
            "Half-width of the coarse offset search. Make it comfortably "
            "larger than the offset you expect (default 1000)."
        ),
    )
    p_clap.add_argument("--max-drift-ppm", type=float, default=500.0)
    p_clap.add_argument(
        "--match-tolerance-ms",
        type=float,
        default=None,
        help="Matching gate (default: 10%% of the median inter-event interval).",
    )
    p_clap.add_argument(
        "--threshold",
        type=float,
        default=6.0,
        help="Onset threshold in robust sigmas above the median (default 6).",
    )
    p_clap.add_argument("--frame-ms", type=float, default=10.0)
    p_clap.add_argument("--hop-ms", type=float, default=2.5)
    p_clap.add_argument("--min-separation-ms", type=float, default=50.0)
    p_clap.add_argument(
        "--no-refine",
        action="store_true",
        help=(
            "Skip the sub-millisecond waveform refinement pass. Refinement "
            "is on by default: it re-times each detected onset against the "
            "raw samples, removing the ~10 ms frame bias of the coarse "
            "detector without changing which onsets were found."
        ),
    )
    p_clap.add_argument(
        "--source-domain",
        default=None,
        help="Clock-domain name for the audio times (source of the mapping).",
    )
    p_clap.add_argument(
        "--target-domain",
        default=None,
        help="Clock-domain name for the event times (target of the mapping).",
    )
    p_clap.add_argument(
        "--epoch",
        type=int,
        default=0,
        help=(
            "Clock generation this calibration was taken in (see "
            "SyncSession.clock_epoch). Stamped onto the mapping so a session "
            "can refuse it after a device reconnect."
        ),
    )

    p_import_lr = sub.add_parser(
        "import-lerobot",
        help="Import a real LeRobot v3.0 dataset directory as a run.",
    )
    p_import_lr.add_argument("dataset_dir", help="LeRobot dataset root (contains meta/info.json).")
    p_import_lr.add_argument("--out", required=True, help="Output run directory.")
    p_import_lr.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Import only the first N episodes (default: all).",
    )

    p_import_qut = sub.add_parser(
        "import-qut",
        help="Import the QUT Hugging Face dataset-example directory as a run.",
    )
    p_import_qut.add_argument("dataset_dir", help="Dataset root (contains episodes/).")
    p_import_qut.add_argument("--out", required=True, help="Output run directory.")
    p_import_qut.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Import only the first N episodes (default: all).",
    )

    p_audit_pts = sub.add_parser(
        "audit-lerobot-pts",
        help="Check a LeRobot v3 dataset's Parquet timestamps against video PTS.",
    )
    p_audit_pts.add_argument("dataset_dir", help="LeRobot dataset root (contains meta/info.json).")
    p_audit_pts.add_argument(
        "--tolerance-s",
        type=float,
        default=DEFAULT_TOLERANCE_S,
        help=f"LeRobot's video timestamp tolerance (default {DEFAULT_TOLERANCE_S}).",
    )
    p_audit_pts.add_argument(
        "--json",
        default=None,
        help="Write the full-precision evidence document to this path.",
    )
    p_audit_pts.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Audit only the first N episodes (default: all).",
    )

    p_inspect_dataset = sub.add_parser(
        "inspect-dataset",
        help="Inspect an unfamiliar dataset without importing payload data.",
    )
    p_inspect_dataset.add_argument("dataset_path")
    p_inspect_dataset.add_argument(
        "--out",
        default=None,
        help="Optional dataset-profile JSON path (default: print JSON).",
    )

    p_verify = sub.add_parser(
        "verify",
        help="Ask an optional deep service for a second opinion on an AV offset.",
    )
    p_verify.add_argument("reference_uri", help="Video/reference URI understood by the service.")
    p_verify.add_argument("candidate_uri", help="Audio/candidate URI understood by the service.")
    p_verify.add_argument("--offset-ms", type=float, required=True, help="Classical offset proposal.")
    p_verify.add_argument(
        "--search-radius-ms", type=float, default=400.0, help="Deep search half-width (default 400)."
    )
    p_verify.add_argument(
        "--tolerance-ms",
        type=float,
        default=200.0,
        help="Disagreement threshold for human inspection (default 200).",
    )
    p_verify.add_argument(
        "--api-url",
        default=None,
        help="Verifier base URL; defaults to EMBODIED_SYNC_VERIFY_URL.",
    )
    p_verify.add_argument(
        "--token-env",
        default="EMBODIED_SYNC_VERIFY_TOKEN",
        help="Environment variable containing the bearer token.",
    )
    p_verify.add_argument(
        "--timeout-s", type=float, default=120.0, help="HTTP timeout (default 120)."
    )
    p_verify.add_argument(
        "--metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Opaque metadata forwarded to the service; repeatable.",
    )
    p_verify.add_argument("--out", default=None, help="Write result JSON instead of stdout.")

    p_infer_import = sub.add_parser(
        "infer-import",
        help="Generate and score deterministic import-plan candidates.",
    )
    p_infer_import.add_argument("dataset_path")
    p_infer_import.add_argument(
        "--out",
        default=None,
        help="Optional inference JSON path (default: print JSON).",
    )
    p_infer_import.add_argument(
        "--rate-hz",
        type=float,
        default=None,
        help="Known native rate; otherwise inferred from media/timestamp evidence.",
    )
    p_infer_import.add_argument("--min-confidence", type=float, default=0.75)
    p_infer_import.add_argument("--min-margin", type=float, default=0.12)

    p_import_auto = sub.add_parser(
        "import-auto",
        help="Infer, confidence-gate, and execute a deterministic import plan.",
    )
    p_import_auto.add_argument("dataset_path")
    p_import_auto.add_argument("--out", required=True, help="Output run directory.")
    p_import_auto.add_argument(
        "--plan",
        default=None,
        help="Reviewed import-plan or inference JSON; skips fresh inference.",
    )
    p_import_auto.add_argument("--rate-hz", type=float, default=None)
    p_import_auto.add_argument("--min-confidence", type=float, default=0.75)
    p_import_auto.add_argument("--min-margin", type=float, default=0.12)
    p_import_auto.add_argument(
        "--accept-ambiguous",
        action="store_true",
        help="Execute the highest-scoring candidate when the confidence gate blocks it.",
    )
    p_import_auto.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Import only the first N episodes when supported.",
    )

    p_export_lr = sub.add_parser(
        "export-lerobot",
        help="Export an aligned episode as a minimal LeRobot v3.0 dataset.",
    )
    p_export_lr.add_argument("episode_dir", help="Aligned-episode directory (from `embsync align`).")
    p_export_lr.add_argument("--out", required=True, help="Output dataset directory.")
    p_export_lr.add_argument(
        "--task",
        default=None,
        help=(
            "Task string for meta/tasks.parquet. Default: first task of the "
            "source run's LeRobot manifest when traceable, else 'unknown'."
        ),
    )

    p_export_umi = sub.add_parser(
        "export-umi",
        help="Export an aligned episode as a UMI/diffusion-policy Zarr replay buffer.",
    )
    p_export_umi.add_argument(
        "episode_dir",
        help="Aligned-episode directory (from `embsync align`).",
    )
    p_export_umi.add_argument("--out", required=True, help="Output Zarr directory.")
    p_export_umi.add_argument(
        "--target-rate-hz",
        type=float,
        default=None,
        help=(
            "Replay-buffer frame rate. Defaults to the aligned episode "
            "manifest's target_rate_hz."
        ),
    )

    return parser


def _cmd_synth(args: argparse.Namespace) -> int:
    """Generate a clean deterministic run and save it to ``args.out``.

    The generator inputs are recorded in the manifest under ``"synthetic"``
    so the run is reproducible from its manifest alone (D-0005).
    """
    run = generate_synthetic_run(
        duration_s=args.duration_s,
        seed=args.seed,
        start_time_ns=args.start_time_ns,
    )
    try:
        if args.adapter == _MCAP_ADAPTER:
            save_mcap_run(run, args.out)
        else:
            save_run(
                run,
                args.out,
                extra_manifest={
                    "synthetic": {
                        "seed": args.seed,
                        "duration_s": args.duration_s,
                        "start_time_ns": args.start_time_ns,
                    }
                },
            )
    except FileExistsError as exc:
        print(f"embsync synth: {exc}", file=sys.stderr)
        return 1
    total = sum(len(samples) for samples in run.values())
    print(f"wrote {total} samples across {len(run)} streams to {args.out}")
    return 0


def _cmd_corrupt(args: argparse.Namespace) -> int:
    """Apply a validated corruption profile to a saved run.

    ``--preview`` runs the corruption in-memory and prints a per-stream
    diff to stdout without touching disk — the safe way to iterate on a
    new profile.
    """
    if not args.preview and args.out is None:
        print(
            "embsync corrupt: --out is required unless --preview is set",
            file=sys.stderr,
        )
        return 2
    try:
        run = _load_run_for_adapter(args.run_dir, args.adapter)
        profile = load_profile(args.profile)
        result = apply_profile(run, profile)
        if args.preview:
            _print_corrupt_preview(run, result, args.profile, profile.seed)
            return 0
        if args.adapter == _MCAP_ADAPTER:
            save_mcap_run(result.run, args.out)
        else:
            save_run(
                result.run,
                args.out,
                extra_manifest={
                    "corruption": {
                        "profile_path": str(Path(args.profile)),
                        "profile_seed": profile.seed,
                    }
                },
            )
            save_corruption_ground_truth(
                result.dropped,
                args.out,
                extra_metadata={
                    "profile_path": str(Path(args.profile)),
                    "profile_seed": profile.seed,
                },
            )
    except (FileExistsError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"embsync corrupt: {exc}", file=sys.stderr)
        return 1

    total = sum(len(samples) for samples in result.run.values())
    dropped = sum(len(samples) for samples in result.dropped.values())
    print(
        f"wrote {total} samples across {len(result.run)} streams to {args.out} "
        f"({dropped} dropped samples recorded)"
    )
    return 0


def _print_corrupt_preview(
    original: dict[str, list[Sample]],
    result: CorruptionResult,
    profile_path: str,
    profile_seed: int,
) -> None:
    """Print a per-stream summary of the corruption to stdout.

    Format is diffable and stable so callers may snapshot it or grep
    fields out during profile iteration.
    """
    print(
        f"embsync corrupt --preview: {profile_path} (seed {profile_seed})"
    )
    print("stream               samples_before   samples_after    dropped   duplicated   recv_delta_max_ns")
    stream_names = sorted(set(original) | set(result.run))
    for name in stream_names:
        before = original.get(name, [])
        after = result.run.get(name, [])
        dropped_here = len(result.dropped.get(name, ()))
        added = max(0, len(after) - (len(before) - dropped_here))
        # Max receive-time delta over the samples that survived at the
        # same sequence position — a diagnostic for latency/jitter
        # corruptions that shift receive_time_ns without removing
        # samples.
        by_seq_before = {s.sequence_id: s for s in before}
        max_delta = 0
        for sample in after:
            src = by_seq_before.get(sample.sequence_id)
            if src is None:
                continue
            delta = abs(sample.receive_time_ns - src.receive_time_ns)
            if delta > max_delta:
                max_delta = delta
        print(
            f"{name:<20} {len(before):>14}   {len(after):>14}   "
            f"{dropped_here:>7}   {added:>10}   {max_delta:>18}"
        )
    total_before = sum(len(s) for s in original.values())
    total_after = sum(len(s) for s in result.run.values())
    total_dropped = sum(len(s) for s in result.dropped.values())
    print(
        f"totals: samples_before={total_before} samples_after={total_after} "
        f"dropped={total_dropped} (no files written)"
    )


def _cmd_align(args: argparse.Namespace) -> int:
    """Align a run into fixed-rate policy frames and save the episode.

    ``--check-ground-truth`` opts in to reading the corruption sidecar
    from ``run_dir`` so the report's ``ground_truth_missing_count`` is
    populated; if the sidecar is missing, alignment still proceeds and a
    warning is printed to stderr.
    """
    try:
        run = _load_run_for_adapter(args.run_dir, args.adapter)
        target_rate_hz = _resolve_target_rate(
            args.target_rate_hz, args.run_dir, command="align"
        )
        alignment_policy = _parse_alignment_policy(args.alignment_policy)
        method: MethodArg = alignment_policy if alignment_policy is not None else args.method
        recorded_seeds = parse_recorded_seeds(args.record_seed)
        ground_truth: dict[str, tuple[Sample, ...]] | None = None
        if args.check_ground_truth:
            try:
                ground_truth = load_corruption_ground_truth(args.run_dir)
            except FileNotFoundError:
                print(
                    f"embsync align: {args.run_dir!s} has no "
                    f"{CORRUPTION_GROUND_TRUTH_NAME}; continuing without ground-truth check.",
                    file=sys.stderr,
                )
        aligned = align_run(
            run,
            target_rate_hz=target_rate_hz,
            method=method,
            ground_truth=ground_truth,
        )
        source_manifest = _read_manifest(args.run_dir)
        provenance = build_provenance(
            run,
            aligned,
            source_path=args.run_dir,
            source_manifest=source_manifest,
            target_rate_hz=target_rate_hz,
            method=method,
            adapter=args.adapter,
            recorded_seeds=recorded_seeds,
        )
        save_episode(
            aligned,
            args.out,
            target_rate_hz=target_rate_hz,
            alignment_policy=alignment_policy,
            extra_manifest={
                "source_run": str(Path(args.run_dir)),
                "method": args.method,
                "provenance": provenance,
            },
        )
    except (FileExistsError, FileNotFoundError, TypeError, ValueError) as exc:
        print(f"embsync align: {exc}", file=sys.stderr)
        return 1

    missing_summary = ", ".join(
        f"{name}={count}" for name, count in aligned.report.missing_count.items()
    )
    print(
        f"wrote {len(aligned.frames)} aligned frames to {args.out} "
        f"(missing per stream: {missing_summary})"
    )
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    """Re-run an episode's recorded alignment and verify its provenance."""

    try:
        episode_manifest = _read_manifest(args.episode_dir)
        raw_provenance = episode_manifest.get("provenance")
        if not isinstance(raw_provenance, dict):
            raise ValueError(
                "episode has no provenance block; align it with this version first"
            )
        raw_source = raw_provenance.get("source")
        if not isinstance(raw_source, dict):
            raise ValueError("episode provenance has no source block")
        source_path = args.source or raw_source.get("path")
        if not isinstance(source_path, str) or not source_path:
            raise ValueError(
                "no source path was recorded; pass --source RUN_DIR"
            )
        recorded_adapter = raw_source.get("adapter", _RUN_ADAPTER)
        adapter = args.adapter or recorded_adapter
        if adapter not in _RUN_ADAPTERS:
            raise ValueError(
                f"unsupported recorded adapter {adapter!r}; pass --adapter"
            )
        run = _load_run_for_adapter(source_path, adapter)
        source_manifest = _read_manifest(source_path)
        episode = load_episode(args.episode_dir)
        result = verify_replay(
            run,
            episode,
            raw_provenance,
            source_path=source_path,
            source_manifest=source_manifest,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"embsync replay: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        status = "PASS" if result.verified else "FAIL"
        print(f"replay verification: {status}")
        for message in result.messages:
            print(f"- {message}")
    return 0 if result.verified else 1


def _parse_alignment_policy(raw: str | None) -> MethodArg | None:
    if raw is None:
        return None
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--alignment-policy must be JSON: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise ValueError("--alignment-policy must be a JSON object mapping stream names")

    policy: dict[str, Method | AlignmentPolicy] = {}
    for stream, entry in decoded.items():
        if not isinstance(stream, str):
            raise ValueError("--alignment-policy stream names must be strings")
        if isinstance(entry, str):
            policy[stream] = _alignment_method_from_value(entry)
            continue
        if not isinstance(entry, dict):
            raise ValueError(
                f"--alignment-policy entry for {stream!r} must be a method string or object"
            )
        method = _alignment_method_from_value(entry.get("method", NEAREST_NEIGHBOR))
        tolerance_ns = entry.get("tolerance_ns")
        policy[stream] = AlignmentPolicy(method=method, tolerance_ns=tolerance_ns)
    return policy


def _alignment_method_from_value(value: object) -> Method:
    if not isinstance(value, str):
        raise TypeError(f"alignment method must be a string, got {type(value).__name__}")
    if value not in (NEAREST_NEIGHBOR, ZERO_ORDER_HOLD, LINEAR_INTERPOLATION):
        raise ValueError(
            f"unknown alignment method {value!r}; "
            f"known methods: {[NEAREST_NEIGHBOR, ZERO_ORDER_HOLD, LINEAR_INTERPOLATION]}"
        )
    return cast(Method, value)


def _load_run_for_adapter(path: str | Path, adapter: str) -> dict[str, list[Sample]]:
    if adapter == _MCAP_ADAPTER:
        return load_mcap_run(path)
    return load_run(path)


def _read_manifest(directory: str | Path) -> dict[str, object]:
    manifest_path = Path(directory) / "manifest.json"
    if not manifest_path.is_file():
        return {}
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _resolve_target_rate(
    cli_value: float | None, run_dir: str | Path, *, command: str
) -> float:
    """CLI ``--target-rate-hz`` wins; else the run manifest's source rate.

    Imported runs (``import-lerobot``) record ``source_rate_hz`` so the
    natural alignment rate needs no flag. Synthetic/corrupted runs do
    not — there the flag stays required.
    """
    if cli_value is not None:
        return cli_value
    rate = _read_manifest(run_dir).get("source_rate_hz")
    if isinstance(rate, (int, float)) and not isinstance(rate, bool) and rate > 0:
        return float(rate)
    raise ValueError(
        f"--target-rate-hz is required: the run manifest of {run_dir!s} does "
        f"not record a source_rate_hz (embsync {command})"
    )


def _cmd_report(args: argparse.Namespace) -> int:
    """Emit a sync-quality HTML report from an aligned episode or a run.

    A run directory (e.g. written by ``import-lerobot``) is aligned
    in-memory first — at ``--target-rate-hz`` or the manifest's
    ``source_rate_hz`` — so a dataset can be audited in one command
    without persisting an episode directory.
    """
    try:
        manifest = _read_manifest(args.episode_dir)
        source_run: str | None = None
        target_rate_hz: float | None = None
        if manifest.get("type") == "aligned_episode":
            aligned = load_episode(args.episode_dir)
            raw_run = manifest.get("source_run")
            source_run = str(raw_run) if isinstance(raw_run, str) else None
            raw_rate = manifest.get("target_rate_hz")
            target_rate_hz = (
                float(raw_rate) if isinstance(raw_rate, (int, float)) else None
            )
        else:
            run = load_run(args.episode_dir)
            target_rate_hz = _resolve_target_rate(
                args.target_rate_hz, args.episode_dir, command="report"
            )
            source_run = str(Path(args.episode_dir))
            aligned = align_run(run, target_rate_hz=target_rate_hz)
        out = (
            Path(args.out)
            if args.out is not None
            else Path(args.episode_dir) / "sync_report.html"
        )
        save_report_html(
            aligned,
            out,
            title=args.title,
            source_run=source_run,
            target_rate_hz=target_rate_hz,
        )
        if args.json_summary:
            summary_path = Path(args.json_summary)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(report_summary_dict(build_report(aligned)), indent=2),
                encoding="utf-8",
            )
    except (FileNotFoundError, ValueError) as exc:
        print(f"embsync report: {exc}", file=sys.stderr)
        return 1

    print(f"wrote sync-quality report to {out}")
    return 0


def _load_event_times(path: str | Path, *, label: str) -> list[int]:
    """Read integer-ns event times from a JSON array or ``{"events_ns": [...]}``.

    Both shapes are accepted because both are what people actually have:
    a bare array falls out of ``json.dump(times)``, and the keyed object
    falls out of anything that also wanted to record metadata. Floats
    are refused rather than rounded — a float in a timestamp field means
    the producer was working in seconds or milliseconds, and guessing
    which would be worse than saying so (D-0002).
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"no such {label} file: {resolved}")
    decoded = json.loads(resolved.read_text(encoding="utf-8"))
    if isinstance(decoded, dict):
        decoded = decoded.get("events_ns")
    if not isinstance(decoded, list):
        raise ValueError(
            f"{resolved}: expected a JSON array of integer-ns {label} times, or "
            f'an object with an "events_ns" array'
        )
    times: list[int] = []
    for index, value in enumerate(decoded):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(
                f"{resolved}: {label} time at index {index} is "
                f"{type(value).__name__}, expected int nanoseconds (D-0002). "
                f"Convert seconds/milliseconds to integer ns before writing."
            )
        times.append(value)
    if not times:
        raise ValueError(f"{resolved}: contains no {label} times")
    return times


def _cmd_calibrate_clap(args: argparse.Namespace) -> int:
    """Fit an audio→event clock mapping from claps and write it as JSON.

    Thin by construction: detect (or read) onsets, call
    ``align_clap_events``, serialise with ``clap_report_dict``. The
    printed summary leads with the *evidence* — matched count, residual
    p95, confidence — rather than the offset alone, because an offset
    with no matched events is a number, not a measurement.
    """
    from embodied_sync.calibrate import (
        align_clap_events,
        clap_report_dict,
        detect_audio_onsets,
        load_waveform,
    )

    if (args.audio is None) == (args.audio_events is None):
        print(
            "embsync calibrate clap: pass exactly one of --audio or "
            "--audio-events",
            file=sys.stderr,
        )
        return 2
    try:
        refined = not args.no_refine
        inputs: dict[str, object] = {
            "events": str(Path(args.events)),
            "max_offset_ms": args.max_offset_ms,
            "max_drift_ppm": args.max_drift_ppm,
            "match_tolerance_ms": args.match_tolerance_ms,
        }
        if args.audio_events is not None:
            audio_onsets = _load_event_times(args.audio_events, label="audio onset")
            inputs["audio_events"] = str(Path(args.audio_events))
        else:
            waveform, sample_rate_hz = load_waveform(
                args.audio, sample_rate_hz=args.sample_rate_hz
            )
            audio_onsets = detect_audio_onsets(
                waveform,
                sample_rate_hz,
                start_time_ns=args.start_time_ns,
                frame_ms=args.frame_ms,
                hop_ms=args.hop_ms,
                threshold=args.threshold,
                min_separation_ms=args.min_separation_ms,
                refine=refined,
            )
            if not audio_onsets:
                raise ValueError(
                    f"{args.audio}: no onsets crossed the {args.threshold:g}-sigma "
                    f"threshold. Lower --threshold, or check that the recording "
                    f"actually contains the claps."
                )
            inputs.update(
                {
                    "audio": str(Path(args.audio)),
                    "sample_rate_hz": sample_rate_hz,
                    "start_time_ns": args.start_time_ns,
                    "frame_ms": args.frame_ms,
                    "hop_ms": args.hop_ms,
                    "threshold": args.threshold,
                    "min_separation_ms": args.min_separation_ms,
                    "refined": refined,
                }
            )
        visual_events = _load_event_times(args.events, label="event")
        alignment = align_clap_events(
            audio_onsets,
            visual_events,
            max_offset_ms=args.max_offset_ms,
            max_drift_ppm=args.max_drift_ppm,
            match_tolerance_ms=args.match_tolerance_ms,
            source_domain=args.source_domain,
            target_domain=args.target_domain,
        )
        if args.epoch:
            alignment = replace(
                alignment,
                fit=replace(
                    alignment.fit,
                    mapping=alignment.fit.mapping.with_epoch(args.epoch),
                ),
            )
        document = clap_report_dict(
            alignment,
            audio_onsets_ns=audio_onsets,
            visual_events_ns=visual_events,
            inputs=inputs,
        )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"embsync calibrate clap: {exc}", file=sys.stderr)
        return 1

    mapping = alignment.fit.mapping
    print(
        f"matched {len(alignment.matched)} of {len(audio_onsets)} audio onsets "
        f"against {len(visual_events)} events "
        f"(confidence {alignment.confidence:.3f})"
    )
    print(
        f"  offset {mapping.offset_ns / 1e6:+.3f} ms, drift "
        f"{mapping.drift_ppb / 1000.0:+.1f} ppm, residual scale "
        f"{mapping.variance_ns / 1e6:.3f} ms, residual p95 "
        f"{alignment.residual_p95_ns / 1e6:.3f} ms"
    )
    if alignment.fit.n_pairs < 3:
        print(
            "  note: fewer than 3 matched pairs — the drift is unverified. "
            "Clap at the start *and* the end of the recording to measure one."
        )
    print(f"wrote clap calibration to {out}")
    return 0


def _plan_label(plan: ImportPlan) -> str:
    clock = plan.parameters.get("clock")
    strategy = str(clock.get("strategy")) if isinstance(clock, dict) else None
    suffix = f"/{strategy}" if strategy else ""
    return f"{plan.executor}{suffix} confidence={plan.confidence:.3f}"


def _print_inference_summary(result: InferenceResult) -> None:
    print(result.decision)
    for index, candidate in enumerate(result.candidates, start=1):
        marker = "selected" if result.selected == candidate else "candidate"
        print(f"  {index}. [{marker}] {_plan_label(candidate)}")
        for warning in candidate.warnings:
            print(f"     warning: {warning}")


def _cmd_inspect_dataset(args: argparse.Namespace) -> int:
    from embodied_sync.ingest import inspect_dataset

    try:
        profile = inspect_dataset(args.dataset_path)
        if args.out is not None:
            save_json_document(profile, args.out)
            print(f"wrote dataset profile to {args.out}")
        else:
            print(json.dumps(profile.to_dict(), indent=2, sort_keys=True))
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"embsync inspect-dataset: {exc}", file=sys.stderr)
        return 1
    return 0


def _write_audit_json(document: dict[str, object], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cmd_audit_lerobot_pts(args: argparse.Namespace) -> int:
    """Audit LeRobot video timestamp resolution; never modify the dataset.

    Exit ``0`` only when the audit found nothing at all, ``1`` when it
    completed and found something, ``2`` when the dataset could not be
    audited. "Something" is wider than a residual over the tolerance: a
    whole-frame episode offset produces no residual, and a legacy float32
    rejection is the state a bug reporter is asking about, so both exit
    ``1`` rather than reporting a clean dataset.

    The evidence JSON is written for ``1``, and for ``2`` whenever there is
    anything to write — a CI run or an issue report that only gets an exit
    code has to be re-run by hand to learn anything.
    """
    from embodied_sync.inspect.lerobot_pts import (
        LeRobotPTSAuditError,
        audit_lerobot_dataset,
        render_audit_text,
    )

    try:
        audit = audit_lerobot_dataset(
            args.dataset_dir,
            tolerance_s=args.tolerance_s,
            max_episodes=args.max_episodes,
        )
    except ModuleNotFoundError as exc:
        print(
            f"embsync audit-lerobot-pts: missing optional dependency {exc.name!r} "
            f"(pip install 'embodied-sync[lerobot]')",
            file=sys.stderr,
        )
        return 2
    except LeRobotPTSAuditError as exc:
        print(f"embsync audit-lerobot-pts: {exc}", file=sys.stderr)
        if args.json is not None and exc.partial is not None:
            document = exc.partial.to_dict()
            document["error"] = str(exc)
            _write_audit_json(document, args.json)
            print(f"wrote partial audit evidence to {args.json}", file=sys.stderr)
        return 2
    except (
        AttributeError,
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        # Malformed metadata reaches here in whatever shape the file's own
        # damage produces; none of it may surface as a traceback.
        print(f"embsync audit-lerobot-pts: {exc}", file=sys.stderr)
        return 2

    if args.json is not None:
        _write_audit_json(audit.to_dict(), args.json)
    print(render_audit_text(audit))
    if args.json is not None:
        print(f"wrote audit evidence to {args.json}")
    return 0 if audit.passed else 1


def _cmd_verify(args: argparse.Namespace) -> int:
    """Call the library verifier client and emit its stable review document."""
    from embodied_sync.inspect import (
        HTTPAlignmentVerifier,
        VerificationRequest,
        VerificationServiceError,
        verification_document,
        verify_alignment,
    )
    from embodied_sync.time.clock_domain import ClockDomain, ClockKind, LatencyEstimate

    api_url = args.api_url or os.environ.get("EMBODIED_SYNC_VERIFY_URL")
    if not api_url:
        print(
            "embsync verify: --api-url or EMBODIED_SYNC_VERIFY_URL is required",
            file=sys.stderr,
        )
        return 2
    try:
        metadata: list[tuple[str, str]] = []
        for item in args.metadata:
            if "=" not in item:
                raise ValueError(f"--metadata must be KEY=VALUE, got {item!r}")
            key, value = item.split("=", 1)
            if not key:
                raise ValueError("--metadata key must be non-empty")
            metadata.append((key, value))
        offset_ns = round(args.offset_ms * 1_000_000)
        radius_ns = round(args.search_radius_ms * 1_000_000)
        tolerance_ns = round(args.tolerance_ms * 1_000_000)
        request = VerificationRequest(
            reference_uri=args.reference_uri,
            candidate_uri=args.candidate_uri,
            proposed_offset_ns=offset_ns,
            search_radius_ns=radius_ns,
            metadata=tuple(metadata),
        )
        mapping = LatencyEstimate(
            source=ClockDomain("reference", ClockKind.UNKNOWN),
            target=ClockDomain("candidate", ClockKind.UNKNOWN),
            offset_ns=offset_ns,
            variance_ns=0,
        )
        verifier = HTTPAlignmentVerifier(
            api_url,
            timeout_s=args.timeout_s,
            token=os.environ.get(args.token_env) if args.token_env else None,
        )
        review = verify_alignment(
            mapping, request, verifier, tolerance_ns=tolerance_ns
        )
        rendered = json.dumps(verification_document(request, review), indent=2) + "\n"
        if args.out:
            output = Path(args.out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered, encoding="utf-8")
            print(f"wrote verification review to {output}")
        else:
            print(rendered, end="")
    except (OSError, ValueError, VerificationServiceError) as exc:
        print(f"embsync verify: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_infer_import(args: argparse.Namespace) -> int:
    try:
        agent = DatasetImportAgent(
            min_confidence=args.min_confidence,
            min_margin=args.min_margin,
        )
        result = agent.analyze(args.dataset_path, rate_hz=args.rate_hz)
        if args.out is not None:
            save_json_document(result, args.out)
            print(f"wrote import inference to {args.out}")
            _print_inference_summary(result)
        else:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"embsync infer-import: {exc}", file=sys.stderr)
        return 1
    return 0


def _reviewed_plan(path: str | Path, *, accept_ambiguous: bool) -> ImportPlan:
    try:
        return load_import_plan(path)
    except ValueError:
        if not accept_ambiguous:
            raise
        inference = load_inference_result(path)
        if not inference.candidates:
            raise ValueError(f"{path!s} contains no import candidates")
        return inference.candidates[0]


def _cmd_import_auto(args: argparse.Namespace) -> int:
    try:
        agent = DatasetImportAgent(
            min_confidence=args.min_confidence,
            min_margin=args.min_margin,
        )
        reviewed_plan = (
            _reviewed_plan(args.plan, accept_ambiguous=args.accept_ambiguous)
            if args.plan is not None
            else None
        )
        run, dataset_info, inference = agent.import_dataset(
            args.dataset_path,
            plan=reviewed_plan,
            rate_hz=args.rate_hz,
            accept_ambiguous=args.accept_ambiguous,
            max_episodes=args.max_episodes,
        )
        selected = reviewed_plan
        if selected is None and inference is not None:
            selected = inference.selected
            if selected is None and args.accept_ambiguous and inference.candidates:
                selected = inference.candidates[0]
        if selected is None:
            raise ValueError("automatic import completed without a selected plan")
        extra_manifest: dict[str, object] = {"auto_import": dataset_info}
        source_rate_hz = plan_source_rate_hz(selected)
        if source_rate_hz is not None:
            extra_manifest["source_rate_hz"] = source_rate_hz
        save_run(run, args.out, extra_manifest=extra_manifest)
    except ModuleNotFoundError as exc:
        print(
            f"embsync import-auto: executor dependency {exc.name!r} is not installed",
            file=sys.stderr,
        )
        return 1
    except AmbiguousImportError as exc:
        print(
            f"embsync import-auto: {exc}; review `embsync infer-import` output, "
            "supply --plan, or use --accept-ambiguous",
            file=sys.stderr,
        )
        return 1
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"embsync import-auto: {exc}", file=sys.stderr)
        return 1

    total = sum(len(samples) for samples in run.values())
    accepted = " (ambiguity override)" if args.accept_ambiguous else ""
    print(
        f"imported {total} samples across {len(run)} streams to {args.out} "
        f"using {_plan_label(selected)}{accepted}"
    )
    return 0


def _cmd_import_lerobot(args: argparse.Namespace) -> int:
    """Import a real LeRobot v3.0 dataset directory into a run directory.

    The run manifest records the dataset provenance under ``"lerobot"``
    (fps, tasks, per-episode boundary table with global start offsets)
    plus a top-level ``source_rate_hz`` so `align`/`report` can default
    their target rate to the dataset's native fps.
    """
    try:
        run, dataset_info = load_lerobot_dataset(
            args.dataset_dir, max_episodes=args.max_episodes
        )
        save_run(
            run,
            args.out,
            extra_manifest={
                "lerobot": dataset_info,
                "source_rate_hz": dataset_info["fps"],
            },
        )
    except ModuleNotFoundError as exc:
        print(
            f"embsync import-lerobot: missing optional dependency {exc.name!r} "
            f"(pip install 'embodied-sync[lerobot]')",
            file=sys.stderr,
        )
        return 1
    except (FileExistsError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"embsync import-lerobot: {exc}", file=sys.stderr)
        return 1

    total = sum(len(samples) for samples in run.values())
    print(
        f"imported {dataset_info['imported_episodes']} episodes "
        f"({total} samples across {len(run)} streams) from {args.dataset_dir} "
        f"to {args.out}"
    )
    return 0


def _cmd_import_qut(args: argparse.Namespace) -> int:
    """Import the QUT Hugging Face dataset-example into a run directory."""
    try:
        run, dataset_info = load_qut_dataset(
            args.dataset_dir, max_episodes=args.max_episodes
        )
        save_run(
            run,
            args.out,
            extra_manifest={
                "qut": dataset_info,
                "source_rate_hz": dataset_info["source_rate_hz"],
            },
        )
    except (FileExistsError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"embsync import-qut: {exc}", file=sys.stderr)
        return 1

    total = sum(len(samples) for samples in run.values())
    print(
        f"imported {dataset_info['imported_episodes']} episodes "
        f"({total} samples across {len(run)} streams) from {args.dataset_dir} "
        f"to {args.out}"
    )
    return 0


def _lerobot_task_for_episode(episode_dir: str | Path) -> str | None:
    """First task string of the episode's source run, if traceable.

    Follows the episode manifest's ``source_run`` back to the imported
    run manifest written by ``import-lerobot``. Best-effort: any missing
    link returns ``None``.
    """
    source_run = _read_manifest(episode_dir).get("source_run")
    if not isinstance(source_run, str):
        return None
    lerobot = _read_manifest(source_run).get("lerobot")
    if not isinstance(lerobot, dict):
        return None
    for episode in lerobot.get("episodes", []):
        if isinstance(episode, dict):
            tasks = episode.get("tasks")
            if isinstance(tasks, list) and tasks and isinstance(tasks[0], str):
                return tasks[0]
    return None


def _cmd_export_lerobot(args: argparse.Namespace) -> int:
    """Export an aligned episode as a minimal LeRobot v3.0 dataset."""
    try:
        aligned = load_episode(args.episode_dir)
        manifest = _read_manifest(args.episode_dir)
        raw_rate = manifest.get("target_rate_hz")
        if not isinstance(raw_rate, (int, float)) or isinstance(raw_rate, bool):
            raise ValueError(
                f"episode manifest of {args.episode_dir!s} has no target_rate_hz"
            )
        task = args.task or _lerobot_task_for_episode(args.episode_dir) or "unknown"
        export_lerobot_dataset(
            aligned,
            args.out,
            target_rate_hz=float(raw_rate),
            task=task,
        )
    except ModuleNotFoundError as exc:
        print(
            f"embsync export-lerobot: missing optional dependency {exc.name!r} "
            f"(pip install 'embodied-sync[lerobot]')",
            file=sys.stderr,
        )
        return 1
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"embsync export-lerobot: {exc}", file=sys.stderr)
        return 1

    print(f"exported {len(aligned.frames)} frames to LeRobot dataset at {args.out}")
    return 0


def _cmd_export_umi(args: argparse.Namespace) -> int:
    """Export an aligned episode as a UMI-style Zarr replay buffer."""
    try:
        aligned = load_episode(args.episode_dir)
        manifest = _read_manifest(args.episode_dir)
        raw_rate = args.target_rate_hz
        if raw_rate is None:
            manifest_rate = manifest.get("target_rate_hz")
            if not isinstance(manifest_rate, (int, float)) or isinstance(manifest_rate, bool):
                raise ValueError(
                    f"episode manifest of {args.episode_dir!s} has no target_rate_hz"
                )
            raw_rate = float(manifest_rate)
        export_umi_zarr(aligned, args.out, target_rate_hz=float(raw_rate))
    except ModuleNotFoundError as exc:
        print(
            f"embsync export-umi: missing optional dependency {exc.name!r} "
            f"(pip install 'embodied-sync[umi]')",
            file=sys.stderr,
        )
        return 1
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"embsync export-umi: {exc}", file=sys.stderr)
        return 1

    print(f"exported {len(aligned.frames)} frames to UMI Zarr buffer at {args.out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "synth":
        return _cmd_synth(args)
    if args.command == "corrupt":
        return _cmd_corrupt(args)
    if args.command == "align":
        return _cmd_align(args)
    if args.command == "replay":
        return _cmd_replay(args)
    if args.command == "report":
        return _cmd_report(args)
    if args.command == "calibrate":
        if args.calibrate_command == "clap":
            return _cmd_calibrate_clap(args)
        # Bare `embsync calibrate`: show what calibrators exist rather than
        # failing with an argparse error that names none of them.
        build_parser().parse_args(["calibrate", "--help"])
        return 2
    if args.command == "inspect-dataset":
        return _cmd_inspect_dataset(args)
    if args.command == "audit-lerobot-pts":
        return _cmd_audit_lerobot_pts(args)
    if args.command == "verify":
        return _cmd_verify(args)
    if args.command == "infer-import":
        return _cmd_infer_import(args)
    if args.command == "import-auto":
        return _cmd_import_auto(args)
    if args.command == "import-lerobot":
        return _cmd_import_lerobot(args)
    if args.command == "import-qut":
        return _cmd_import_qut(args)
    if args.command == "export-lerobot":
        return _cmd_export_lerobot(args)
    if args.command == "export-umi":
        return _cmd_export_umi(args)
    # argparse's choices restrict `command` to the registered
    # subcommands (or None, handled above), so this branch is
    # defensive rather than reachable.
    print(f"embsync {args.command}: unknown subcommand", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
