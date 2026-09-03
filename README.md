# embodied-sync

**Know whether your sensors agree before your policy pays the price.**

`embodied-sync` is an open-source Python library for synchronizing, temporally
calibrating, and validating multi-sensor robot-learning datasets and live sensor
streams. It handles different sample rates, clock domains, clock skew and drift,
late packets, jitter, missing frames, and causal policy observations.

Cameras run at one rate, robot state at another, packets arrive late, and a
device reconnect can silently reset its clock. `embodied-sync` gives you one
place to align, replay, inspect, validate, and stress-test that multimodal timing
before a policy trains on or deploys with the wrong observations. It fits around
UMI, LeRobot, ROS 2/rosbag2 + MCAP, LSL/XDF, Rerun, SurgSync-style datasets, and
custom Python sensor SDKs; ROS is optional.

> **Project status:** alpha. The live and recorded workflows work today and
> have test coverage. Adapter support varies by format; [Current
> scope](#current-scope) spells out what each one can do and which tests need
> local data.

This project focuses on **temporal** calibration and alignment. It does not
calibrate camera intrinsics, sensor extrinsics, or hand-eye transforms; configure
PTP; emulate a network; or promise hard real-time scheduling.

Task-oriented guides:

- [Synchronize a multi-sensor robot-learning dataset in Python](https://anicut-ai.github.io/embodied-sync/robot-learning-dataset-synchronization.html)
- [Validate LeRobot v3 timestamps and temporal alignment](https://anicut-ai.github.io/embodied-sync/lerobot-timestamp-validation.html)
- [Test robot-policy pipelines against latency, jitter, and delay](https://anicut-ai.github.io/embodied-sync/latency-jitter-robot-policy-testing.html)
- [Handle multiple clocks, clock drift, and late packets](https://anicut-ai.github.io/embodied-sync/multi-clock-sensor-stream-synchronization.html)

## Start where your data is

You do not need to reorganize your workflow around the library.

| If you have... | Why use embodied-sync? | Start here |
| --- | --- | --- |
| A recording on disk | Check skew, missing observations, and alignment policy before you spend compute on training. | [`embsync align`](#align-a-recorded-run) or the [Python API](#use-the-python-api) |
| Sensors running now | Build causal policy observations, surface stale or unmapped streams immediately, and record the session for later review. | [`SyncSession`](#synchronize-a-live-robot) |
| A result you want to inspect visually | Open a portable report instead of parsing logs. You can send the same file to a collaborator. | [Browser GUI](#review-results-in-the-browser) |
| An alignment that is expensive to get wrong | Ask for a second, independent review before you accept or publish the run. | [Verifier API](#get-a-second-opinion-with-the-verifier-api) |

Use the CLI for shell and batch workflows, the browser report and inspector
for visual review, or the Python API inside experiments and services. They all
work with the same timestamp model and alignment metadata.

Want to see the full workflow first? The
[`sync_quality_demo.ipynb`](examples/notebooks/sync_quality_demo.ipynb) notebook
walks through corruption, recorded alignment, live alignment, reports, and a
real LeRobot import. For a version with plain text output, use
[`sync_quality_demo_plain.ipynb`](examples/notebooks/sync_quality_demo_plain.ipynb).

## Install

The package is on PyPI - the easiest way to start using is the latest official pip wheel:

```bash
pip install embodied-sync
```

In case you want to customize or install from source:

```bash
pip install -e .          # core types, alignment, calibration, and live sessions
pip install -e ".[dev]"   # add the development and test tools
pip install -e ".[full]"  # add format adapters and inspection tools
```

If you only need one ecosystem, install just that adapter:

```bash
pip install -e ".[mcap]"
pip install -e ".[lerobot]"
pip install -e ".[lsl]"
pip install -e ".[surg_sync]"
pip install -e ".[umi]"
pip install -e ".[rerun]"
```

The base install is designed to be deliberately small: `numpy` and `pyyaml` only.

## Align a recorded run

**Why you need it.** A dataset can look plausible frame by frame and still be
wrong for learning. A nearest camera frame may come from the future, a fast
state stream may be held too long, or dropped samples may disappear inside a
clean-looking tensor. Run the aligner before training. It records the choice,
skew, confidence, and missing status for every policy frame.

**How to use it.** Start with a deterministic example, inject known timing
problems, then generate an aligned episode and report:

```bash
embsync synth --out runs/clean --seed 0 --duration-s 10
embsync corrupt runs/clean \
  --profile configs/corrupt_kitchen_sink.yaml \
  --out runs/bad
embsync align runs/bad \
  --out episodes/bad_10hz \
  --target-rate-hz 10 \
  --check-ground-truth
embsync report episodes/bad_10hz \
  --out reports/bad.html \
  --json-summary reports/bad.json
```

Use `nearest_neighbor`, `zoh`, or `linear_interp` globally, or set a policy per
stream. The right choice depends on what the signal means. The decision table
in [`choosing_alignment_policy.md`](docs/user/choosing_alignment_policy.md)
will help you choose instead of relying on the default.

The [sync-quality notebook](examples/notebooks/sync_quality_demo.ipynb) runs
this same workflow in memory and plots where latency, jitter, drift, drops,
and stalls appear.

## Synchronize a live robot

**Why you need it.** A running policy cannot look ahead. It can only use
samples that have actually arrived, which means an offline-clean dataset can
still produce stale observations on the robot. `SyncSession` makes that
causal boundary visible. You can keep your existing SDK callbacks and control
loop.

**How to use it.** Attach callbacks for push-based sensors, push polled values
directly, and ask for a synchronized bundle at each policy tick:

```python
import embodied_sync as embsync

with embsync.init(
    run_dir="runs/experiment_001",
    streams={
        "camera": embsync.StreamConfig(rate_hz=30, tolerance_ms=20.0),
        "robot": embsync.StreamConfig(rate_hz=250, tolerance_ms=4.0),
    },
    primary="camera",
) as sync:
    camera_sdk.on_frame(
        sync.attach("camera", timestamp=lambda frame: frame.device_ts_ns)
    )

    while running:
        sync.push("robot", robot_sdk.read_state())
        bundle = sync.get()
        if bundle.ok:
            act(bundle["camera"], bundle["robot"])
```

Each bundle includes per-stream skew and quality metadata. The session raises
a typed `SyncViolation` for stale holds, missing streams, clock resets, and
unmapped clock domains. It does not quietly treat them as valid observations.

The session also records to the same run format used by the recorded-data
tools. You can stop the robot, run `embsync report runs/experiment_001`, and
inspect exactly what the policy saw. The online section of the
[`sync_quality_demo.ipynb`](examples/notebooks/sync_quality_demo.ipynb)
compares live observation staleness with recorded alignment of the same data.
For the complete session API, see
[`sync_session_api.md`](docs/design/sync_session_api.md).

## Recover a shared clock

**Why you need it.** Two precise timestamps are not comparable just because
they are both measured in nanoseconds. Device clocks can start at different
origins, drift at different rates, or reset after a reconnect. Do not
hard-code an offset in preprocessing and hope it still holds next week.
Measure the mapping instead.

**How to use it.** Record a physical event that both devices can observe, such
as a clap, and fit the mapping:

```bash
embsync calibrate clap \
  --audio recording.wav \
  --events visual_events.json \
  --source-domain microphone \
  --target-domain camera \
  --out calibration.json
```

The calibration package also supports matched event trains and visual
timestamps. Start with
[`timestamps_clock_domains.md`](docs/concepts/timestamps_clock_domains.md)
for the mental model and practical limits of each method.

## Bring your existing dataset

**Why you need it.** You probably did not come to the lab to rewrite a dataset
loader. The import tools preserve source timestamps and put each supported
format into the same sample model. Your alignment and report code can then
stay the same across datasets.

**How to use it.** If you do not recognize the layout of a local directory,
profile it first. The importer will suggest how to read it:

```bash
embsync inspect-dataset /data/recording --out profile.json
embsync infer-import /data/recording --out inference.json
embsync import-auto /data/recording \
  --plan inference.json \
  --out runs/recording
```

The tool scores the possible formats and clock mappings. It imports the data
only when one option wins by a clear margin. If the answer is ambiguous, it
stops and shows you the evidence. It only runs built-in import code; it never
writes code from a guess. See
[`automatic_dataset_import.md`](docs/user/automatic_dataset_import.md) for the
full workflow.

LeRobot v3.0 has a direct path:

```bash
embsync import-lerobot data/external/lerobot/pusht \
  --out runs/lerobot_pusht
embsync report runs/lerobot_pusht
embsync align runs/lerobot_pusht --out episodes/pusht
embsync export-lerobot episodes/pusht --out out/pusht_lerobot
```

You can also export numeric aligned episodes as UMI/diffusion-policy Zarr
replay buffers:

```bash
embsync export-umi episodes/pusht --out out/pusht_umi.zarr
```

The LeRobot section of the
[`sync_quality_demo_plain.ipynb`](examples/notebooks/sync_quality_demo_plain.ipynb)
shows the equivalent Python workflow and explains what happens to timestamp
precision, episode boundaries, and video references.

When LeRobot itself refuses to decode a frame — "no frame within tolerance" —
a read-only audit says which of several failure modes is behind it, with
nanosecond evidence you can attach to an issue:

```bash
embsync audit-lerobot-pts data/external/lerobot/pusht --json pts_evidence.json
```

It exits `0` only when the audit finds nothing — no residual over tolerance,
every episode offset landing on its boundary frame, every video holding the
frames its metadata declares — `1` on any finding, and `2` when the dataset
cannot be audited. See
[`audit_lerobot_pts.md`](docs/user/audit_lerobot_pts.md) for how to read the
result and what it deliberately does not claim.

## Review results in the browser

**Why you need it.** Timing bugs are easier to discuss when everyone can see
the same missing rates, skew, confidence, and alignment policy. The browser
view is one self-contained HTML file. Attach it to an experiment record, serve
it from a lab machine, or send it to a collaborator. You do not need a
dashboard server.

**How to use it.** Point `report` at either an aligned episode or an imported
run, then open the generated file in any browser:

```bash
embsync report runs/recording \
  --out reports/recording.html \
  --json-summary reports/recording.json \
  --title "grasping rig - camera replacement"
```

For event-train calibration, the inspector shows the selected match next to
the neighboring matches that the aligner rejected. You can then see whether
the events really match. The public API lives in `embodied_sync.inspect`. Its
provider interface accepts your own video, audio, force, or device-specific
media reader.

See [`interpreting_sync_reports.md`](docs/user/interpreting_sync_reports.md)
for a field guide to every report column.

## Use the Python API

**Why you need it.** CLI commands are convenient for one run, but experiments,
dataset gates, and CI checks often need typed results. The Python API returns
the same alignment metadata and report objects directly. You do not need to
parse files or subprocess output.

**How to use it.** Load a normalized run, align it, inspect the result, and
write the same HTML shown above:

```python
from embodied_sync.align import align_run
from embodied_sync.datasets.io import load_run
from embodied_sync.reports import build_report, save_report_html

run = load_run("runs/recording")
aligned = align_run(run, target_rate_hz=10.0, method="zoh")
report = build_report(aligned)

for stream in report.streams:
    print(stream.name, stream.missing_rate, stream.median_skew_ns)

save_report_html(aligned, "reports/recording.html")
```

The notebooks are the most complete executable API examples:
[`visual version`](examples/notebooks/sync_quality_demo.ipynb) and
[`plain version`](examples/notebooks/sync_quality_demo_plain.ipynb).

## Get a second opinion with the Verifier API

**Why you need it.** A classical alignment fit can be internally consistent
and still pair the wrong events. That risk matters most on long collections,
high-value demonstrations, and datasets that will be shared across a team.
The Verifier API is the premium review path. It checks the proposed offset
independently. If it disagrees by enough, it marks the result for inspection.
It never overwrites the classical fit or its evidence.

**How to use it.** Connect the client to your verifier endpoint, then send the
reference and candidate URIs with the proposed offset:

```bash
export EMBODIED_SYNC_VERIFY_URL=https://verifier.example.com
export EMBODIED_SYNC_VERIFY_TOKEN='your-token'

embsync verify \
  file:///data/video.mp4 \
  file:///data/audio.wav \
  --offset-ms 20 \
  --search-radius-ms 400 \
  --tolerance-ms 200 \
  --metadata scene=pick_001 \
  --out verification.json
```

The client sends URIs and alignment metadata, not the media bytes themselves.
Your robots and CI machines do not need the verification models installed.
One controlled service can review runs from every rig. The response includes
the verifier identity, proposed offset, confidence, and whether a person
should inspect the result. The public Python adapter adds this review to the
HTML inspector.

See the
[`Verifier API guide`](docs/user/optional_deep_verification.md) for the Python
client, inspector integration, authentication, and v1 wire contract.

## Test timing failures before they happen

**Why you need it.** A clean fixture proves the happy path; it does not tell
you how a pipeline behaves when a camera stalls or a clock drifts. Controlled
corruptions make those failures repeatable. Because the tool records exactly
what it changed or removed, you can check the report against known truth.

**How to use it.** Apply one of the profiles in [`configs/`](configs), or
compose your own from fixed latency, jitter, dropped frames, clock drift,
burst stalls, duplicates, non-monotonic delivery, and missing intervals:

```bash
embsync corrupt runs/clean \
  --profile configs/corrupt_camera_jitter.yaml \
  --out runs/camera_jitter
embsync align runs/camera_jitter \
  --out episodes/camera_jitter \
  --target-rate-hz 10 \
  --check-ground-truth
```

The first half of the
[`sync_quality_demo.ipynb`](examples/notebooks/sync_quality_demo.ipynb) plots
each failure shape and compares reported missing frames with known removals.

## External datasets

`embodied-sync` does not download external datasets or accept license and
access agreements on your behalf. Core tests run without them. To exercise an
adapter against real data, point the test suite at files you supplied locally:

```bash
export EMBODIED_SYNC_EXTERNAL_DATA_ROOT=/path/to/data/external
```

Use this layout (it is git-ignored):

```text
data/external/
  umi/
  lerobot/
  mcap/
  qut/
  xdf/
  surg_sync/
  rerun/
```

Install the matching extra, then run the relevant external-data tests. If the
dataset is missing, the test tells you why it skipped. The core suite still
runs:

```bash
pip install -e ".[mcap]"
pytest -q -m external_data tests/test_adapter_mcap.py
```

See [`manual_dataset_setup.md`](docs/user/manual_dataset_setup.md) and
[`TESTING_STRATEGY.md`](TESTING_STRATEGY.md) for details. Keep downloaded
datasets, archives, installers, and generated outputs out of git; only small,
redistributable fixtures belong in [`data/fixtures/`](data/fixtures).

## Current scope

Our philosophy is to support fewer formats well than claim a long adapter list.
Today you can use the core run and episode formats, corruption engine,
recorded and live aligners, `SyncSession`, clock calibration, reports, and
automatic dataset import.

Tests cover the native LeRobot v3.0, LabRecorder XDF, and SurgSync v1.0 readers
with data that users provide locally. CI also checks the format contracts for
MCAP, UMI, LSL/XDF, Rerun, and SurgSync. Install the matching extra when a
format needs one. Native UMI Zarr import is still planned. The project does
not promise hard real-time control, lock you into a vendor SDK, or distribute
third-party datasets.

## Repository map

| Path | What you will find there |
| --- | --- |
| [`embodied_sync/`](embodied_sync) | Core model, sessions, calibration, alignment, adapters, reports, and CLI |
| [`examples/`](examples) | Runnable examples and notebooks |
| [`docs/user/`](docs/user) | Task-oriented guides |
| [`docs/concepts/`](docs/concepts) | Timing, clock-domain, and alignment concepts |
| [`configs/`](configs) | Ready-to-run corruption and alignment configurations |
| [`tests/`](tests) | Deterministic unit, contract, and external-data tests |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Package layout and data-model decisions |
| [`TESTING_STRATEGY.md`](TESTING_STRATEGY.md) | Test tiers, skip policy, and determinism rules |

## License and contributing

All dependencies must use compatible licenses, and CI checks the full environment.
To run the same check locally, install `.[full,dev]` and run `python scripts/check_licenses.py`.

Contributions are welcome. Start with [`ARCHITECTURE.md`](ARCHITECTURE.md) for
the package boundaries and [`TESTING_STRATEGY.md`](TESTING_STRATEGY.md) for the
project's determinism, fixture, and external-data expectations.

The public docs landing page is available at [`docs/README.md`](docs/README.md),
with the same content published at the
[documentation site](https://anicut-ai.github.io/embodied-sync/).
