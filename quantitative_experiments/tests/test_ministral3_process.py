"""Exercise the real shell requeue helper with interrupt-ignoring descendants."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from quantitative_experiments.ministral3_process import run


ROOT = Path(__file__).resolve().parents[1]


def _alive(pid):
    path = Path(f"/proc/{pid}/stat")
    return path.exists() and path.read_text().rsplit(")", 1)[1].split()[0] != "Z"


@pytest.fixture
def launch(tmp_path):
    child_script = tmp_path / "child.py"
    child_script.write_text('''
import json, os, signal, subprocess, sys, time
from pathlib import Path
if os.environ.get("TEST_IGNORE_TERM"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
def checkpoint_stop(*args):
    raise SystemExit(75)
signal.signal(signal.SIGUSR1, checkpoint_stop)
worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
Path(os.environ["TEST_READY"]).write_text(json.dumps({
    "parent": os.getpid(), "worker": worker.pid,
    "ignores_sigint": signal.getsignal(signal.SIGINT) == signal.SIG_IGN,
}))
while True:
    time.sleep(0.1)
''')
    ready = tmp_path / "ready.json"
    requeue = tmp_path / "requeue.log"
    scontrol = tmp_path / "scontrol"
    scontrol.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$TEST_REQUEUE"\n')
    scontrol.chmod(0o755)
    processes = []

    def start(ignore_term=False):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), TEST_READY=str(ready),
                   TEST_REQUEUE=str(requeue), SCONTROL_BIN=str(scontrol), SLURM_JOB_ID="12345")
        env.pop("SLURM_ARRAY_JOB_ID", None)
        env.pop("SLURM_ARRAY_TASK_ID", None)
        if ignore_term:
            env["TEST_IGNORE_TERM"] = "1"
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "quantitative_experiments.ministral3_process",
             "bash", str(ROOT / "scripts/correctness_stratified/run_with_slurm_requeue.sh"),
             sys.executable, "-u", str(child_script)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        processes.append(proc)
        deadline = time.monotonic() + 10
        while not ready.exists():
            assert proc.poll() is None, proc.communicate()[0]
            assert time.monotonic() < deadline, "Child did not become ready"
            time.sleep(0.05)
        state = json.loads(ready.read_text())
        assert state["ignores_sigint"]  # Reproduce the original shell inheritance.
        return proc, state, requeue

    yield start
    for proc in processes:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.communicate(timeout=15)


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_interrupt_stops_background_parent_and_worker(launch, signum):
    proc, state, requeue = launch()
    proc.send_signal(signum)
    output, _ = proc.communicate(timeout=15)
    assert proc.returncode == 128 + signum, output
    assert not _alive(state["parent"])
    assert not _alive(state["worker"])
    assert not requeue.exists()  # Manual interruption must not resubmit a job.


def test_interrupt_force_stops_term_ignoring_workers(launch):
    proc, state, _ = launch(ignore_term=True)
    proc.send_signal(signal.SIGINT)
    output, _ = proc.communicate(timeout=15)
    assert proc.returncode == 130, output
    assert not _alive(state["parent"])
    assert not _alive(state["worker"])


def test_slurm_warning_still_checkpoints_and_requeues(launch):
    proc, state, requeue = launch()
    proc.send_signal(signal.SIGUSR1)
    output, _ = proc.communicate(timeout=15)
    assert proc.returncode == 0, output
    assert requeue.read_text().splitlines() == ["requeue", "12345"]
    assert not _alive(state["parent"])
    assert not _alive(state["worker"])


def test_child_exit_status_is_preserved():
    assert run([sys.executable, "-c", "raise SystemExit(7)"]) == 7
