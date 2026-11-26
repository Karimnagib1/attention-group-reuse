"""Supervise a Ministral command and stop its complete process group on interrupt."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time


def _signal_group(pid, signum):
    try:
        os.killpg(pid, signum)
        return True
    except ProcessLookupError:
        return False


def run(command):
    if not command:
        raise ValueError("A command is required.")
    stop_signal = 0
    warning = False
    force_stop = False
    child = None

    def handle_signal(signum, frame):
        nonlocal stop_signal, warning, force_stop
        if signum == signal.SIGUSR1:
            warning = True
        elif stop_signal:
            force_stop = True
        else:
            stop_signal = signum

    previous = {sig: signal.signal(sig, handle_signal)
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1)}
    try:
        # A dedicated session isolates this run from the user's shell and other jobs.
        # The existing requeue helper and every Python worker inherit this group.
        child = subprocess.Popen(command, start_new_session=True)
        deadline = None
        while child.poll() is None:
            if warning and not stop_signal:
                warning = False
                # Only the helper receives USR1; it forwards the checkpoint warning.
                child.send_signal(signal.SIGUSR1)
            if stop_signal and deadline is None:
                print("Stopping Ministral and its workers; completed checkpoints are preserved.", flush=True)
                _signal_group(child.pid, signal.SIGTERM)
                deadline = time.monotonic() + 5
            if deadline is not None and (force_stop or time.monotonic() >= deadline):
                _signal_group(child.pid, signal.SIGKILL)
            try:
                child.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass
        if stop_signal:
            return 128 + stop_signal
        return child.returncode if child.returncode >= 0 else 128 - child.returncode
    finally:
        # The helper may exit before its descendants. Reap our child and clean up
        # any remaining workers even on errors, without touching another run.
        if child is not None:
            _signal_group(child.pid, signal.SIGTERM)
            deadline = time.monotonic() + 5
            while not force_stop and time.monotonic() < deadline and _signal_group(child.pid, 0):
                child.poll()
                time.sleep(0.05)
            _signal_group(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
