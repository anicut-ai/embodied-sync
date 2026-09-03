# Decision records

Design decisions carry a stable `D-NNNN` number. Code and docs cite that
number inline (`# ... (D-0033)`) so a reader can trace a constraint back to
the reasoning that produced it.

## Status of this file

This file was cited by the codebase before it existed: seven modules,
`pyproject.toml`, and `scripts/check_licenses.py` referred to
`DECISIONS.md D-NNNN` while no such file was committed. That gap is why
`D-0042` was very nearly assigned twice — there was nowhere to look up
which numbers were taken.

So this file starts as two things: an **allocation registry**, which is
complete and authoritative from today, and a **set of full records**, which
is not. Records D-0001 through D-0042 exist as inline rationale at the
citation sites listed below; they are transcribed here as each is next
revisited. Do not renumber them, and do not treat an unwritten record as an
unmade decision.

**Before using a new number, check the registry below and take the next
free one.**

## Allocation registry

| ID | Subject | Full record | First citation |
| -- | ------- | ----------- | -------------- |
| D-0001 | Layered optional dependencies; base install stays numpy + pyyaml | inline | `pyproject.toml` |
| D-0002 | Sample contract | inline | `embodied_sync/core/sample.py` |
| D-0003 | Sample contract | inline | `embodied_sync/core/sample.py` |
| D-0004 | Synthetic stream design | inline | `embodied_sync/streams/synthetic.py` |
| D-0005 | Run/episode IO design | inline | `embodied_sync/datasets/io.py` |
| D-0006 | Synthetic stream design | inline | `embodied_sync/streams/synthetic.py` |
| D-0007 | Synthetic stream design | inline | `embodied_sync/streams/synthetic.py` |
| D-0008 | No `python_version` pin for mypy | inline | `pyproject.toml` |
| D-0009 | Corruption profile model | inline | `embodied_sync/corrupt/profile.py` |
| D-0010 | Corruption application | inline | `embodied_sync/corrupt/apply.py` |
| D-0014 | Clock domains | inline | `embodied_sync/time/clock_domain.py` |
| D-0015 | Robot policy observations | inline | `docs/concepts/robot_policy_observations.md` |
| D-0017 | Kitchen-sink corruption coverage | inline | `tests/test_kitchen_sink.py` |
| D-0018 | Non-monotonic corruption | inline | `tests/test_corrupt_non_monotonic.py` |
| D-0019 | Missing-interval corruption | inline | `tests/test_corrupt_missing_interval.py` |
| D-0020 | Aligned episode model | inline | `embodied_sync/core/episode.py` |
| D-0021 | Run IO | inline | `embodied_sync/datasets/io.py` |
| D-0022 | Aligned episode model | inline | `embodied_sync/core/episode.py` |
| D-0023 | Sync-quality reporting | inline | `embodied_sync/reports/sync_quality.py` |
| D-0024 | Sync report contract | inline | `embodied_sync/core/sync_report.py` |
| D-0025 | Alignment engine | inline | `embodied_sync/align/engine.py` |
| D-0026 | Online alignment | inline | `embodied_sync/align/online.py` |
| D-0027 | Online alignment | inline | `embodied_sync/align/online.py` |
| D-0029 | Run manifest | inline | `embodied_sync/core/manifest.py` |
| D-0033 | Native LeRobot v3.0 reader/exporter | inline | `embodied_sync/adapters/lerobot.py` |
| D-0034 | LSL adapter | inline | `embodied_sync/adapters/lsl.py` |
| D-0035 | SurgSync adapter | inline | `embodied_sync/adapters/surg_sync.py` |
| D-0036 | UMI/Zarr exporter | inline | `embodied_sync/exporters/umi.py` |
| D-0037 | Live `SyncSession` | inline | `embodied_sync/session/session.py` |
| D-0038 | Calibration layer stays numpy-only | inline | `pyproject.toml` |
| D-0040 | Approximate-time synchronization | inline | `embodied_sync/session/approximate.py` |
| D-0041 | Latency estimates | inline | `embodied_sync/time/clock_domain.py` |
| D-0042 | Fail CI on a copyleft dependency | inline | `scripts/check_licenses.py` |
| D-0043 | LeRobot PTS audit model | below | `embodied_sync/inspect/lerobot_pts.py` |

D-0011 to D-0013, D-0016, D-0028, D-0030 to D-0032, and D-0039 are unused.
Leave them unused: a gap is cheaper than a number that means two things.

---

## D-0043 — LeRobot PTS audit: what it measures, and what it refuses to

**Status:** accepted. **Supersedes:** nothing. **Cited by:**
`embodied_sync/inspect/lerobot_pts.py`, `docs/user/audit_lerobot_pts.md`.

### Context

LeRobot rejects a video frame when the selected frame's presentation
timestamp is further from the query than `tolerance_s`. At least three
distinct defects produce that one error message: a boundary case at exactly
the tolerance (#2814), a float32 timeline whose own spacing exceeds the
tolerance being tested (#2364), and episode `from_timestamp` values that
drift from the concatenated video's PTS by a whole frame (#3177). Users
cannot tell them apart, and the residual alone does not separate them.

### Decision

`embsync audit-lerobot-pts` reproduces the check and reports a
`likely_cause` per camera and physical video, from a closed vocabulary:
`within_tolerance`, `precision_boundary`, `legacy_precision_false_positive`,
`cumulative_episode_offset`, `frame_step_offset`, `missing_or_extra_frames`,
`unclassified`.

Four independent measurements back that verdict, because one does not
suffice:

1. **exact residual** — float64, inclusive comparison, the current path;
2. **legacy residual** — float32 with a strict `<`, because a dataset that
   is clean today is still the one an older LeRobot rejects, and reporting
   only (1) answers #2814 with "works for me";
3. **episode boundary offset** — each declared `from_timestamp` against the
   PTS of the frame it should land on, derived from the frame counts of
   preceding episodes in the same physical video;
4. **frame count** — declared rows against the encoded timeline.

Measurement (3) is load-bearing rather than redundant: a whole-frame error
in `from_timestamp` shifts the query *and* the projected index together, so
it produces no residual at all. A residual-only audit reports such a dataset
as clean. This was found by testing rather than assumed, and it is the
reason the #3177 signature is detected rather than inferred.

`precision_boundary` is bounded by `max(2 * tolerance, float32_ulp +
tolerance)` — tied to the tolerance under test and the float32 spacing at
the timestamps involved. An earlier draft bounded it at half a frame
period, which at 30 Hz filed every error up to 16.7 ms under "precision
noise" and would have buried exactly the bugs this tool exists to surface.

The PTS timeline is read by demuxing packets, never by decoding frames. A
packet is not universally a frame, so the packet count is checked against
the count the container declares, and a container where they disagree is
refused rather than audited against a timeline that would be silently
wrong.

### Consequences

- Exit `0` requires *no* finding of any kind, not merely "no residual over
  tolerance": a whole-frame episode offset and a legacy-only rejection both
  exit `1`. Exiting `0` on either would hand back a clean bill of health for
  a dataset that fails in the field.
- The audit is strictly read-only. It never rewrites timestamps, alters PTS,
  widens the tolerance, or drops samples.
- No `likely_cause` may be read as an acquisition-time sensor-sync verdict.
  LeRobot v3 keeps one per-episode timeline and discards independent sensor
  clocks, so the question this library exists to answer cannot be answered
  from a v3 dataset. Saying so is part of the decision.
- `import-lerobot` is unchanged: folding decoded PTS into the ordinary
  importer would manufacture acquisition clocks the source does not have,
  and make every import pay for video decoding. Synthetic video timestamps
  therefore stay out of the normal sync-quality report.
- Per-frame `PTSQuery` objects are not retained. Residuals accumulate as
  int64 (exact percentiles still need the distribution); per-episode trend,
  counts, and exemplars accumulate incrementally, so memory tracks episode
  count rather than frame count.

### Rejected alternatives

- **Decode frames to build the timeline.** Correct in general, but it makes
  the audit pay for video decoding, which is the cost the importer avoids
  for the same reason. Verifying packet/frame cardinality and refusing
  ambiguous containers keeps the audit cheap and honest about its limits.
- **Report only the exact residual.** Simpler, and unable to explain the
  reports that motivated the tool.
- **Infer a cause when nothing fits.** `unclassified` is a usable answer;
  a wrong cause on a bug report costs a maintainer more than no cause.
