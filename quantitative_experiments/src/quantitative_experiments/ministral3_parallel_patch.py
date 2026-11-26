"""Ministral replicas partition patch units while sharing one scientific setup."""
from __future__ import annotations

import copy
import json
import os
import signal
import traceback
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from quantitative_experiments.multilingual_head_patch import run as sweeping
from quantitative_experiments.multilingual_head_patch.checkpoint import SweepCheckpoint, UNIT_KEYS


class ReplicaCheckpoint(SweepCheckpoint):
    """Publish setup once and give each layer/group exactly one writer."""

    def __init__(self, output_dir, manifest, *, resume):
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        if self.rank == 0:
            super().__init__(output_dir, manifest, resume=resume)
        dist.barrier()
        if self.rank != 0:
            super().__init__(output_dir, manifest, resume=True)

    def _load_setup(self, loader, *args, **kwargs):
        if self.rank == 0:
            result = loader(*args, **kwargs)
            if result is not None:
                dist.barrier()
            # A cache miss is followed by save_* on rank zero, which releases
            # the peers only after the complete setup file has been published.
            return result
        dist.barrier()
        result = loader(*args, **kwargs)
        if result is None:
            raise RuntimeError("Replica zero did not publish the shared patching setup.")
        return result

    def load_donors(self, *args, **kwargs):
        return self._load_setup(super().load_donors, *args, **kwargs)

    def save_donors(self, *args, **kwargs):
        super().save_donors(*args, **kwargs)
        dist.barrier()

    def load_baseline(self):
        return self._load_setup(super().load_baseline)

    def save_baseline(self, baseline):
        super().save_baseline(baseline)
        dist.barrier()

    def _owns(self, layer_idx, patch_idx):
        return (layer_idx * self.n_patch_units + patch_idx) % self.world_size == self.rank

    def load_unit(self, layer_idx, patch_idx):
        if not self._owns(layer_idx, patch_idx):
            # Tell the existing sweep to skip another worker's unit. These
            # placeholders are never written or used for the final summaries.
            return {key: np.zeros(self.n_eval_samples, dtype=np.float32) for key in UNIT_KEYS}
        return super().load_unit(layer_idx, patch_idx)

    def save_unit(self, layer_idx, patch_idx, arrays):
        if not self._owns(layer_idx, patch_idx):
            raise RuntimeError("A replica attempted to save another replica's patch unit.")
        super().save_unit(layer_idx, patch_idx, arrays)


def _patch_worker(rank, cfg, world_size, rendezvous, device_type="cuda"):
    from quantitative_experiments.ministral3 import MinistralRunner

    sweeping._install_graceful_stop_handler(cfg)
    if device_type == "cuda":
        torch.cuda.set_device(rank)
    try:
        # Workers exchange checkpoints on disk, never GPU tensors. CPU barriers
        # avoid NCCL teardown hanging when a peer fails during shared setup.
        dist.init_process_group(
            backend="gloo",
            init_method=rendezvous, rank=rank, world_size=world_size,
            timeout=timedelta(hours=24),
        )
        print(f"Ministral patch replica {rank + 1}/{world_size}: one complete model.", flush=True)
        with patch.object(sweeping, "HeadPatchingRunner", MinistralRunner), \
             patch.object(sweeping, "SweepCheckpoint", ReplicaCheckpoint):
            # Only checkpoints are used; incomplete per-worker summaries are discarded.
            sweeping.run_sweep(cfg)
    except sweeping.GracefulStopRequested:
        raise SystemExit(sweeping.GRACEFUL_STOP_EXIT_CODE)
    except Exception:
        # Surface the originating error before cleanup or peer connection errors.
        traceback.print_exc()
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _wait_for_workers(context):
    """Forward Slurm's warning and preserve exit 75 for the requeue wrapper."""
    signaled = False
    try:
        while True:
            if sweeping._GRACEFUL_STOP_REQUESTED and not signaled:
                for process in context.processes:
                    if process.is_alive():
                        try:
                            os.kill(process.pid, signal.SIGUSR1)
                        except ProcessLookupError:
                            pass
                signaled = True
            if context.join(timeout=1):
                break
    except (mp.ProcessRaisedException, mp.ProcessExitedException) as exc:
        if sweeping._GRACEFUL_STOP_REQUESTED or getattr(exc, "exit_code", None) == sweeping.GRACEFUL_STOP_EXIT_CODE:
            raise sweeping.GracefulStopRequested from exc
        raise
    finally:
        # Also clean up children if the parent is interrupted from the terminal.
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
    sweeping._raise_if_graceful_stop_requested()


def _assemble_results(cfg):
    """Reuse the normal sweep's aggregation without loading another model."""
    cfg = copy.deepcopy(cfg)
    cfg["run"]["checkpoint"]["resume"] = True
    output = Path(cfg["run"]["output_dir"])
    manifest = json.loads((output / ".sweep_checkpoint/manifest.json").read_text())
    checkpoint = SweepCheckpoint(output, manifest, resume=True)
    if checkpoint.completed_unit_count() != checkpoint.n_layers * checkpoint.n_patch_units:
        raise RuntimeError("The replicated sweep has incomplete patch units; resume before aggregating.")
    if not all((checkpoint.root / filename).is_file() for filename in ("donors.npz", "baseline.npz")):
        raise RuntimeError("The replicated sweep is missing its shared setup.")
    model = manifest["model"]
    runner = SimpleNamespace(
        n_layers=model["n_layers"], n_heads=model["n_heads"],
        n_kv_heads=model["n_kv_heads"], d_head=model["d_head"],
        option_labels=cfg["eval"]["option_labels"],
        uses_model_sharding=False, uses_data_parallel=False,
        base_model=SimpleNamespace(config=SimpleNamespace(_commit_hash=model["resolved_commit"])),
    )
    # The normal checkpoint validation still checks config, samples, and shapes.
    # Every unit is cached, so no model forward or attention hook is needed.
    with patch.object(sweeping, "HeadPatchingRunner", return_value=runner):
        return sweeping.run_sweep(cfg)


def run_replicated_patch(cfg):
    if not cfg.get("run", {}).get("checkpoint", {}).get("enabled", False):
        raise ValueError("Replicated patching requires run.checkpoint.enabled=true.")
    world_size = torch.cuda.device_count()
    if not torch.cuda.is_available() or world_size == 0:
        raise ValueError("Replicated patching requires visible CUDA GPUs.")
    sweeping._raise_if_graceful_stop_requested()
    print(f"Replicated patching: {world_size} GPUs, different layer/group units per replica, "
          f"batch size {cfg['patching'].get('batch_size', 4)} per GPU.", flush=True)
    with TemporaryDirectory(prefix="ministral-patch-replicas-") as temporary:
        context = mp.spawn(
            _patch_worker,
            args=(cfg, world_size, f"file://{Path(temporary) / 'rendezvous'}"),
            nprocs=world_size, join=False,
        )
        _wait_for_workers(context)
    return _assemble_results(cfg)
