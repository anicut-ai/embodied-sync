# embodied-sync documentation

`embodied-sync` measures and reports synchronization quality for recorded and
live robot-learning streams. It provides synchronization assurance; it does
not replace hardware triggering or prove physical simultaneity by timestamps
alone.

## Start here

- [Synchronize robot sensor streams in Python](user/quickstart.md)
- [Message-filters-style synchronization without ROS](user/sync_session_api.md)
- [Choose an alignment policy](user/choosing_alignment_policy.md)
- [Interpret a sync-quality report](user/interpreting_sync_reports.md)
- [Hardware-ground-truth capture protocol](../examples/notebooks/hardware_ground_truth_capture_protocol.ipynb)

## Concepts and operations

- [Timestamps and clock domains](concepts/timestamps_clock_domains.md)
- [Latency, jitter, and drift](concepts/latency_vs_jitter_vs_drift.md)
- [Online versus offline alignment](concepts/online_vs_offline_alignment.md)
- [Robot policy observations](concepts/robot_policy_observations.md)
- [Dataset-quality reports](concepts/dataset_quality_report.md)
- [Manual dataset setup](user/manual_dataset_setup.md)
- [Acceptance-report template](acceptance_report_template.md)
- [Worked fixture example](worked_example.md)
- [Optional deep verification](user/optional_deep_verification.md)
- [Audit LeRobot video timestamps](user/audit_lerobot_pts.md)

The [repository README](../README.md) has installation and CLI instructions.
The [sitemap](sitemap.xml) lists the crawlable site pages.
