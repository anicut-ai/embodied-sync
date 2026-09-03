"""Optional-dependency discipline check (NEXT_TASKS #3).

The base install of ``embodied_sync`` intentionally stays lightweight — no
``mcap``, ``lerobot``, ``pyarrow``, ``pylsl``, ``pyxdf``,
``zarr``/``numcodecs``, ``rerun``/``rerun_sdk`` at
package import time. The in-process check in
``tests/test_package.py::test_import_pulls_no_optional_dependencies``
passes trivially when those packages are not installed at all
(``sys.modules`` never contains them because ``import`` was never
attempted).

This test is the stricter version: a *subprocess* installs a
:class:`_ForbidImportsFinder` on ``sys.meta_path`` that raises
``ImportError`` for the forbidden names before ``embodied_sync`` is
imported. If a future PR adds a stray top-level ``import mcap`` (or
similar) to any module reachable from ``embodied_sync/__init__.py``, that
import will fire, hit the finder, and the subprocess will exit non-zero
— even on a developer machine where the optional dep happens to be
installed. Guards the "base install stays lightweight" rule before the
Milestone 4+ adapters land.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


FORBIDDEN = (
    "mcap",
    "lerobot",
    "pyarrow",
    "av",
    "pylsl",
    "pyxdf",
    "zarr",
    "numcodecs",
    "rerun",
    "rerun_sdk",
    # calibrate/ is numpy-only by design (D-0038): scipy for signal
    # processing and cv2/pyzbar for QR decoding are the two dependencies it
    # would be most natural to reach for, and both are forbidden here.
    "scipy",
    "cv2",
    "pyzbar",
)


def test_package_import_does_not_touch_optional_dependencies() -> None:
    """Subprocess: importing ``embodied_sync`` must not trigger a forbidden import."""
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import importlib.machinery
        import sys

        FORBIDDEN = {FORBIDDEN!r}

        class _ForbidImportsFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                root = fullname.split(".", 1)[0]
                if root in FORBIDDEN:
                    raise ImportError(
                        f"embodied_sync base install must not import "
                        f"optional dependency {{root!r}} at package import "
                        f"time (attempted: {{fullname!r}})"
                    )
                return None

        sys.meta_path.insert(0, _ForbidImportsFinder())

        # If any top-level module reachable from `embodied_sync` tries to
        # import one of the forbidden names, the finder above raises.
        import embodied_sync  # noqa: F401
        from embodied_sync.cli.main import build_parser  # noqa: F401
        from embodied_sync.align import align_run  # noqa: F401
        from embodied_sync.corrupt import apply_profile  # noqa: F401
        from embodied_sync.reports import build_report  # noqa: F401
        from embodied_sync.datasets.io import load_run  # noqa: F401
        # Adapters/exporters live behind pip extras (D-0001). The
        # subpackages themselves must be importable without any extra
        # installed — the concrete adapter/exporter modules do their
        # heavy imports inside function bodies (adapter authoring
        # guide + exporter contracts docs).
        import embodied_sync.adapters  # noqa: F401
        import embodied_sync.exporters  # noqa: F401
        import embodied_sync.adapters.lerobot  # noqa: F401
        import embodied_sync.adapters.lsl  # noqa: F401
        import embodied_sync.adapters.mcap  # noqa: F401
        import embodied_sync.adapters.surg_sync  # noqa: F401
        import embodied_sync.adapters.umi  # noqa: F401
        import embodied_sync.exporters.lerobot  # noqa: F401
        import embodied_sync.exporters.mcap  # noqa: F401
        import embodied_sync.exporters.rerun  # noqa: F401
        import embodied_sync.exporters.umi  # noqa: F401
        # The live session and the calibration layer (D-0037/D-0038) are
        # base-install surfaces: numpy + stdlib only. `calibrate` in
        # particular must not reach for scipy or OpenCV -- the QR decoder
        # is an extras-gated stub precisely so this stays true.
        import embodied_sync.session  # noqa: F401
        import embodied_sync.session.session  # noqa: F401
        import embodied_sync.session.recorder  # noqa: F401
        import embodied_sync.calibrate  # noqa: F401
        import embodied_sync.calibrate.clap  # noqa: F401
        import embodied_sync.calibrate.estimator  # noqa: F401
        import embodied_sync.calibrate.events  # noqa: F401
        import embodied_sync.calibrate.semantic  # noqa: F401
        import embodied_sync.calibrate.visual_timestamp  # noqa: F401
        # The LeRobot PTS audit needs pyarrow and av, both only inside
        # function bodies: the module (and the CLI parser that reads its
        # default tolerance) must import on a base install.
        import embodied_sync.inspect.lerobot_pts  # noqa: F401
        # The top-level lazy re-exports must resolve without pulling
        # anything forbidden either (PEP 562 __getattr__).
        embodied_sync.init  # noqa: B018
        embodied_sync.SyncSession  # noqa: B018
        embodied_sync.StreamConfig  # noqa: B018
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        "embodied_sync import surface touches a forbidden optional dep.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
