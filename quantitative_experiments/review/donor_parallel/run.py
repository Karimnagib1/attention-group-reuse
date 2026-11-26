from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from donor_ablation.engine import locked
from donor_ablation.protocol import load_protocol, load_settings, prepare
from .engine import finalize, load_manifest, setup, tasks, unit_path, work

REVIEW = Path(__file__).resolve().parents[1]


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Must be positive.")
    return result


def settings(args):
    output = args.output_dir or REVIEW / "runs" / f"{args.config.stem}_parallel_{args.dtype}"
    overrides = [*args.set, f"model.dtype={args.dtype}", f"patching.batch_size={args.batch_size}",
                 f"patching.donor_batch_size={args.donor_batch_size}", "run.device=cuda",
                 "run.multi_gpu=false", "run.data_parallel=false", "run.device_map=null", "run.max_memory=null"]
    return load_settings(args.config, overrides, args.samples_per_language, output)


def gpu_tokens(count, visible=None):
    """Preserve Slurm's GPU indices/UUIDs when masking each fresh subprocess."""
    values = [str(i) for i in range(count)] if visible is None else [v.strip() for v in visible.split(",") if v.strip()]
    if len(values) != count or len(set(values)) != len(values) or any(v == "-1" for v in values):
        raise RuntimeError("CUDA_VISIBLE_DEVICES disagrees with available GPUs; check the allocation.")
    return values


class Children:
    def __init__(self):
        self.items = []
        self.handles = []

    def start(self, output, device, stage, rank=0, workers=1, limit=None):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        cmd = [sys.executable, "-m", "donor_parallel.run", "_child", "--run-dir", str(output),
               "--stage", stage, "--rank", str(rank), "--workers", str(workers)]
        if limit:
            cmd.extend(["--limit", str(limit)])
        log = None
        if stage == "work":
            directory = output / "worker_logs"
            directory.mkdir(exist_ok=True)
            path = directory / f"worker_{rank}.log"
            log = path.open("a", buffering=1)
            self.handles.append(log)
            log.write(f"\n=== {datetime.now().isoformat()} node={socket.gethostname()} GPU={device} ===\n")
            print(f"Worker {rank}: GPU {device}; log {path}", flush=True)
        process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT if log else None)
        self.items.append((process, stage, rank))
        return process

    def wait(self, progress=None):
        last_report = time.monotonic()
        while True:
            states = [(p.poll(), stage, rank) for p, stage, rank in self.items]
            for status, stage, rank in states:
                if status is not None and status != 0:
                    raise RuntimeError(f"{stage} process {rank} failed with exit code {status}. "
                                       "See the job error log or worker_logs; completed checkpoints are retained.")
            if all(status is not None for status, _, _ in states):
                if progress:
                    progress()
                return
            if progress and time.monotonic() - last_report >= 30:
                progress()
                last_report = time.monotonic()
            time.sleep(1)

    def close(self):
        for p, _, _ in self.items:
            if p.poll() is None:
                p.terminate()
        deadline = time.monotonic() + 10
        for p, _, _ in self.items:
            try:
                p.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for handle in self.handles:
            handle.close()


@contextmanager
def managed_children():
    children = Children()
    previous = {}
    def stop(signum, frame):
        raise SystemExit(128 + signum)
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, stop)
        yield children
    finally:
        # A second termination request must not interrupt child cleanup.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        children.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def controller(args):
    cfg = settings(args)
    output = Path(cfg["run"]["output_dir"])
    print(f"Node: {socket.gethostname()}; Slurm job: {os.environ.get('SLURM_JOB_ID', 'interactive')}\n"
          f"Output: {output}\nPrecision: {cfg['model']['dtype']}; batch per GPU: {args.batch_size}; "
          f"donor batch: {args.donor_batch_size}; workers: {args.workers}", flush=True)
    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        return
    devices = []
    if not args.prepare_only:
        import torch
        count = torch.cuda.device_count()
        if count < args.workers:
            raise RuntimeError(f"Requested {args.workers} workers but only {count} GPUs are visible.")
        devices = gpu_tokens(count, os.environ.get("CUDA_VISIBLE_DEVICES"))[:args.workers]
    with locked(output):
        path = output / "protocol.json"
        # Refuse an old serial run before writing anything into it.
        existing = output / "run_manifest.json"
        if existing.exists() and json.loads(existing.read_text()).get("runner") != "donor_parallel_v1":
            raise ValueError("This directory belongs to the serial runner; choose a new --output-dir.")
        if not path.exists() and any(p.name != ".lock" for p in output.iterdir()):
            raise ValueError("Output contains files without a saved protocol; choose a new directory.")
        protocol = load_protocol(path, cfg) if path.exists() else prepare(cfg, path)
        print(f"Recipients: {len(protocol['recipient_ids'])} questions x {len(protocol['languages'])} languages; "
              f"{len(protocol['variants'])} donor sweeps", flush=True)
        if args.prepare_only:
            return
        if existing.exists():
            load_manifest(output, protocol)
        with managed_children() as children:
            # Initial preparation is deliberately single-writer. Its model exits
            # before all patch workers start, freeing that GPU's memory.
            print("Stage 1: shared donor capture and clean predictions on the first GPU.", flush=True)
            children.start(output, devices[0], "setup")
            children.wait()
            manifest = load_manifest(output, protocol)
            paths = [unit_path(output, *task) for task in tasks(protocol, manifest)]
            initial = sum(p.exists() for p in paths)
            started = time.monotonic()
            (output / "_SUCCESS").unlink(missing_ok=True)
            def progress():
                done = sum(p.exists() for p in paths)
                newly = done - initial
                elapsed = time.monotonic() - started
                estimate = f"; remaining ~{elapsed / newly * (len(paths) - done) / 3600:.1f}h" if newly >= 2 * args.workers else ""
                print(f"Parallel progress: {done}/{len(paths)} units{estimate}", flush=True)
            print("Stage 2: independent model copies patch different groups concurrently.", flush=True)
            for rank, device in enumerate(devices):
                children.start(output, device, "work", rank, args.workers, args.benchmark_units_per_worker)
            children.wait(progress)
        if any(not p.exists() for p in paths):
            if not args.benchmark_units_per_worker:
                raise RuntimeError("Workers exited but checkpoints are missing.")
            print("Benchmark stopped after the requested units. Repeat without --benchmark-units-per-worker "
                  "to resume the full experiment.", flush=True)
            return
        finalize(protocol, output)
        if not args.skip_analysis:
            # The coordinator already holds the same output lock used by analyze().
            from donor_ablation.analysis import _analyze
            _analyze(output, None)
        print(f"Completed: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Parallel donor ablations: one complete model per GPU.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, default=REVIEW / "configs/qwen_mmluprox.yaml")
    run.add_argument("--samples-per-language", type=positive)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--workers", type=positive, default=4)
    run.add_argument("--batch-size", type=positive, default=4, help="Per GPU; overrides the serial YAML batch size")
    run.add_argument("--donor-batch-size", type=positive, default=4)
    run.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    run.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    run.add_argument("--skip-analysis", action="store_true")
    run.add_argument("--dry-run", action="store_true", help="Print effective config without GPU access or output writes")
    run.add_argument("--prepare-only", action="store_true", help="Freeze question/donor selections without loading a model")
    run.add_argument("--benchmark-units-per-worker", type=positive, help="Stop after this many NEW units per worker; same run can resume")
    child = commands.add_parser("_child", help=argparse.SUPPRESS)
    child.add_argument("--run-dir", type=Path, required=True)
    child.add_argument("--stage", choices=("setup", "work"), required=True)
    child.add_argument("--rank", type=int, default=0)
    child.add_argument("--workers", type=positive, default=1)
    child.add_argument("--limit", type=positive)
    args = parser.parse_args()
    if args.command == "run":
        controller(args)
    else:
        protocol = load_protocol(args.run_dir / "protocol.json")
        # Prevent accidental duplicate invocations of a stage/worker.
        with locked(args.run_dir / "worker_locks" / f"{args.stage}_{args.rank}"):
            if args.stage == "setup":
                setup(protocol, args.run_dir)
            else:
                work(protocol, args.run_dir, args.rank, args.workers, limit=args.limit)


if __name__ == "__main__":
    main()
