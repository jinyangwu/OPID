"""Runtime compatibility patches for the local development environment.

Python imports ``sitecustomize`` automatically during startup when the module is
available on ``sys.path``. Keeping the patch here lets us avoid mutating the
shared conda environment while still smoothing over known dependency issues.
"""

from __future__ import annotations

import os


def _patch_multiprocess_resource_tracker() -> None:
    try:
        from multiprocess import resource_tracker
    except Exception:
        return

    resource_tracker_cls = resource_tracker.ResourceTracker
    if getattr(resource_tracker_cls, "_skillrl_py312_patch", False):
        return

    def _recursion_count(lock) -> int:
        counter = getattr(lock, "_recursion_count", None)
        if callable(counter):
            return counter()
        return 0

    def _stop_locked(self, close=None, waitpid=None, waitstatus_to_exitcode=None):
        if _recursion_count(self._lock) > 1:
            return self._reentrant_call_error()
        if self._fd is None or self._pid is None:
            return

        close = close or os.close
        waitpid = waitpid or os.waitpid

        close(self._fd)
        self._fd = None

        waitpid(self._pid, 0)
        self._pid = None

    resource_tracker_cls._stop_locked = _stop_locked
    resource_tracker_cls._skillrl_py312_patch = True


_patch_multiprocess_resource_tracker()
