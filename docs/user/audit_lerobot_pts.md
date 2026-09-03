# Audit LeRobot video timestamps

`embsync audit-lerobot-pts` answers one question about a local LeRobot v3
dataset: would LeRobot's video timestamp check reject it, and why?

```bash
pip install 'embodied-sync[lerobot]'

embsync audit-lerobot-pts /data/my_dataset \
  --tolerance-s 0.0001 \
  --json pts_evidence.json
```

Exit codes are the contract:

| code | meaning |
| ---- | ------- |
| `0`  | the audit found nothing: no residual over tolerance, every episode offset on its boundary frame, every video holding the frames its metadata declares |
| `1`  | the audit completed and found something |
| `2`  | the dataset could not be audited (bad layout, missing video, missing optional dependency) |

"Something" is deliberately wider than "a residual over the tolerance". A
whole-frame episode offset produces no residual at all, and a dataset that
only an older LeRobot rejects is exactly the state a bug report describes;
both exit `1` rather than reporting a clean dataset.

The JSON evidence is written for `1`, and for `2` whenever anything was
resolved before the failure — a CI run that only gets an exit code has to be
repeated by hand to learn anything.

## What it checks

For every selected episode and declared video feature, the audit rebuilds the
lookup LeRobot performs:

1. resolve the physical video through the v3 episode metadata, reading each
   video's presentation-timestamp (PTS) timeline once even when several
   episodes share it;
2. rebuild the query as `videos/<key>/from_timestamp + parquet_timestamp`;
3. reproduce the decoder's index projection, `round(query_s * average_fps)`,
   in float64 — the audit must not acquire the float32 error it is measuring;
4. compare the selected frame's actual PTS against the query, and separately
   report the truly nearest PTS, which distinguishes a bad index projection
   from a bad timeline.

That residual alone cannot separate the failure modes, so three more
measurements sit beside it:

- the **legacy residual**: the same comparison in float32 with a strict `<`,
  which is what an older LeRobot did. A dataset can be clean under today's
  arithmetic and still be the one a reporter's install rejects.
- the **episode boundary offset**: each declared `from_timestamp` against the
  PTS of the frame it should land on, derived from the frame counts of the
  episodes stored before it in the same physical video.
- the **frame count**: declared rows against the encoded timeline, which
  catches extra frames as well as missing ones.

The boundary check is not redundant with the residual. A whole-frame error in
`from_timestamp` moves the query *and* the projected index together, so it
leaves no residual behind: the query lands exactly on a real frame, just the
wrong one. Only comparing the declared offset against the boundary frame's
actual PTS sees it, which is what makes the #3177 signature detectable.

Container timing stays rational until one final conversion to integer
nanoseconds, and every comparison is between integers, so `100.000 us` and
`100.001 us` are different answers rather than the same rounded string. The
timeline itself comes from demuxed packet timestamps — no pixels are decoded
— and because a packet is not universally a frame, the packet count is
checked against the count the container declares. A container where the two
disagree is refused rather than audited against a timeline that would be
silently wrong.

## Reading the result

Each camera and physical video gets a `likely_cause` from a closed vocabulary:

| cause | what it means |
| ----- | ------------- |
| `within_tolerance` | nothing to fix here |
| `precision_boundary` | over the tolerance, but within `max(2 × tolerance, float32 spacing + tolerance)` at the timestamps involved — the [#2364](https://github.com/huggingface/lerobot/issues/2364) shape, where the timeline's own float precision is coarser than the threshold being tested |
| `legacy_precision_false_positive` | correct under the current float64, inclusive comparison; rejected by an older LeRobot comparing in float32 with a strict `<`. The [#2814](https://github.com/huggingface/lerobot/issues/2814) shape, where the reported distance is *exactly* the tolerance. Upgrading LeRobot, not editing the dataset, is what resolves it |
| `cumulative_episode_offset` | episode offsets (or residuals) grow across later episodes — accumulated `from_timestamp` drift, the [#3177](https://github.com/huggingface/lerobot/issues/3177) shape |
| `frame_step_offset` | offsets cluster at an integer multiple of `1/fps` — a whole-frame selection error, not timing noise |
| `missing_or_extra_frames` | the encoded timeline and the declared row count disagree, in either direction, or a projected index falls outside the timeline |
| `unclassified` | the measurements fit none of the above, reported as such rather than guessed |

The console report and the JSON both name the camera, the physical video, the
episode and frame, the Parquet timestamp, the episode offset, the query, the
selected and nearest PTS, the signed and absolute residual in nanoseconds, the
legacy float32 residual and the float32 spacing at that timestamp, the
violation count, p50/p95/p99/max residuals, the per-episode residual trend,
every episode boundary check, and the first and worst violating samples (plus
the first legacy-only one, when the exact path is clean).

## Attaching evidence to an issue

`pts_evidence.json` is meant to be attached as-is to an issue such as
[#2814](https://github.com/huggingface/lerobot/issues/2814). It already
contains everything a maintainer would otherwise ask the reporter to add debug
prints for: the query, the selected and nearest PTS, the residual, the
tolerance in force, the episode/frame/camera/video identity, and the aggregate
trend that separates a boundary case from accumulated drift. Quote the
`likely_cause` and the first violating episode in the issue body, and attach
the file for the rest.

If you want to see how far off the dataset is, `--tolerance-s` can be widened
as a *diagnostic experiment* only. A wider tolerance does not fix anything: it
accepts a frame that may be the wrong frame.

## Limits (D-0043)

This audit validates consistency among Parquet timestamps, episode video
offsets, the decoder's index selection and encoded PTS. It does **not**
recover or validate independent acquisition clocks: LeRobot v3 stores one
per-episode timeline and discards the per-sensor clocks that a sync-quality
question is really about. No `likely_cause` above may be read as an
acquisition-time sensor-sync verdict.

What it *can* see, and by a route worth knowing about: shifting
`from_timestamp` by a whole frame moves the projected index by the same
frame, so the query lands exactly on a real frame and no residual appears.
Such a dataset decodes the wrong picture and passes LeRobot's own check. The
episode boundary comparison catches it anyway, because the frame each offset
should land on is fixed by the episodes stored before it — no image content
required.

The remaining blind spot is a video whose frames are internally consistent
but depict the wrong moment: if the encoded timeline, the offsets and the row
counts all agree with each other, timestamps have nothing left to disagree
about. Catching that needs image content, which is the alignment inspector's
job — `embodied_sync.inspect.collect_evidence` and `build_page` render the
chosen frame beside the ones that were rejected, for a person to judge. That
is a library API, not a CLI command; no timestamp audit substitutes for it.

The command is strictly read-only. It never rewrites timestamps, alters MP4
PTS, widens the tolerance on its own, or drops samples — and for the same
reason, `embsync import-lerobot` is unchanged: folding decoded PTS into the
ordinary importer would manufacture acquisition clocks the source does not
have, and make every import pay for video decoding.

## Library API

The CLI only parses arguments, renders, and maps outcomes to exit codes. All
of the work is reachable directly:

```python
from embodied_sync.inspect.lerobot_pts import audit_lerobot_dataset

audit = audit_lerobot_dataset("/data/my_dataset", tolerance_s=1e-4)
for video in audit.videos:
    if not video.passed:
        print(video.camera_key, video.video_path, video.likely_cause)
        print(video.first_violation or video.first_legacy_violation)
        for check in video.boundary_faults:
            print("episode", check.episode_index, "off by", check.offset_ns, "ns")
```

`classify_video` is a pure function over a `VideoEvidence` summary that is
bounded by episode count rather than frame count, so the long-timeline and
multi-episode signatures can be reproduced in a test without encoding hours of
video — and so a large dataset's memory use tracks its episodes, not its
frames.
