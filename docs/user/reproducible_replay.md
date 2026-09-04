# Reproducible replay provenance

An aligned episode is useful for comparison only when another person can
identify its inputs and reproduce the same observation pairing. Embodied-Sync
therefore writes a versioned `provenance` block into every episode produced by
`embsync align` and can verify it by replaying the alignment.

## Record and verify an episode

```bash
embsync align runs/experiment_001 \
  --out episodes/experiment_001_10hz \
  --target-rate-hz 10 \
  --record-seed policy_sampler=123

embsync replay episodes/experiment_001_10hz \
  --source runs/experiment_001 \
  --verify
```

The source path is recorded, so `--source` is optional on the machine that
created the episode. A third party will normally provide it because their copy
of the recording lives elsewhere. Use `--json` for a machine-readable result.

`--record-seed NAME=INT` is repeatable. Names should identify the owner, for
example `policy_sampler=123`, `evaluation.bootstrap=8`, or
`torch.action_sampler=42`. Embodied-Sync automatically copies its synthetic
generator and corruption-profile seeds when they are present in the source
manifest.

Recording a downstream seed does not apply that seed or make a downstream
runner deterministic. It preserves the value alongside the observations so
the runner and its result can be audited. An unrecorded seed cannot be
recovered retroactively.

## What the provenance block records

The episode manifest contains:

- a SHA-256 fingerprint for every canonical source stream and the source
  manifest;
- SHA-256 fingerprints for locally resolvable files named by `payload_ref`;
- the requested alignment policy and the resolved method and tolerance for
  every stream;
- the target rate, integer period, first and last target times, and frame
  count;
- a selection fingerprint over each target time and selected
  `(stream_name, sequence_id, acquisition_time_ns)` identity;
- a content fingerprint over the complete persisted aligned-frame records;
- the Embodied-Sync version, Python version, and Git revision when running
  from a checkout; and
- known Embodied-Sync seeds plus values supplied with `--record-seed`.

Resolved tolerances are important. Replay uses those recorded integer values
instead of asking the installed version to infer new defaults.

## Two guarantee levels

The provenance block reports one of two levels:

### `selection`

Replay selected the same source sample identity for every stream at every
target time. This is the strongest claim available for a metadata-only live
recording or when a payload reference cannot be opened and fingerprinted.

### `content`

Selection matches, the complete persisted aligned records match, and all
locally resolvable referenced payload files have matching bytes. This proves
the identity of the recorded inputs; it does not promise identical decoded
pixels or audio across different codec, driver, or hardware builds.

The manifest sets `guarantee.decoded_media` to `false` to make that boundary
machine-readable.

## Interpreting failures

Verification exits with status `0` only when every applicable check passes.
It reports status `1` and identifies the first useful mismatch otherwise, for
example:

```text
replay verification: FAIL
- source stream 'camera' fingerprint changed
- frame 17 stream 'camera' selection differs: recorded=(511, 1700000000), replayed=(512, 1733333333)
```

Checks cover:

- source stream, manifest, and referenced-payload identity;
- software identity;
- integrity of the recorded episode itself;
- replayed sample selection; and
- replayed content when the source supports a content-level guarantee.

An old episode without a provenance block remains loadable, but it cannot be
verified retroactively. Re-run `embsync align` from the original source to
create a provenance-enabled episode.
