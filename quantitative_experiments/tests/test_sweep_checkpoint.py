from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from quantitative_experiments.multilingual_head_patch.checkpoint import (
    CheckpointMismatchError,
    SweepCheckpoint,
    build_manifest,
)
from quantitative_experiments.multilingual_head_patch.run import save_outputs


ROOT = Path(__file__).resolve().parents[1]
SWEEP_SCRIPT = ROOT / "scripts" / "correctness_stratified" / "run_correctness_stratified_sweep.sh"
REQUEUE_HELPER = ROOT / "scripts" / "correctness_stratified" / "run_with_slurm_requeue.sh"


def _sample(sample_id: str, prompt: str) -> SimpleNamespace:
    return SimpleNamespace(
        sample_id=sample_id,
        parallel_id=sample_id.split("_", 1)[0],
        language="en",
        prompt=prompt,
        correct_idx=1,
        question="Question?",
        context="Context",
        options=("one", "two", "three", "four"),
    )


def _config(output_dir: Path) -> dict:
    return {
        "model": {"name_or_path": "synthetic/model", "revision": "abc"},
        "data": {
            "allowed_parallel_ids_path": str(output_dir.parent / "selected_ids.json")
        },
        "eval": {"option_labels": ["A", "B", "C", "D"]},
        "patching": {"patch_mode": "kv_group", "batch_size": 2},
        "run": {
            "output_dir": str(output_dir),
            "checkpoint": {"enabled": True, "resume": True},
        },
    }


def _manifest(output_dir: Path, samples: list[SimpleNamespace]) -> dict:
    return build_manifest(
        cfg=_config(output_dir),
        samples=samples,
        model_commit="resolved-commit",
        n_eval_samples=len(samples),
        n_layers=2,
        n_patch_units=2,
        n_heads=4,
        n_kv_heads=2,
        d_head=3,
        n_options=4,
    )


class SweepCheckpointTests(unittest.TestCase):
    def test_resume_restores_completed_units_and_cached_setup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "output"
            samples = [_sample(f"q{index}_en", f"prompt {index}") for index in range(3)]
            checkpoint = SweepCheckpoint(output_dir, _manifest(output_dir, samples), resume=True)

            baseline = np.arange(12, dtype=np.float32).reshape(3, 4)
            donors = {
                "en": torch.arange(24, dtype=torch.float32).reshape(2, 4, 3),
                "es": torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) + 100,
            }
            checkpoint.save_baseline(baseline)
            checkpoint.save_donors(donors, ["en", "es"])

            expected = {
                key: np.zeros((3, 2, 2), dtype=np.float32)
                for key in (
                    "pct_var_mean_incorrect",
                    "pct_var_max_incorrect",
                    "rank_drop_correct",
                )
            }
            for layer_idx in range(2):
                for patch_idx in range(2):
                    base = np.arange(3, dtype=np.float32) + 10 * layer_idx + patch_idx
                    expected["pct_var_mean_incorrect"][:, layer_idx, patch_idx] = base
                    expected["pct_var_max_incorrect"][:, layer_idx, patch_idx] = base + 100
                    expected["rank_drop_correct"][:, layer_idx, patch_idx] = base + 200

            checkpoint.save_unit(
                0,
                0,
                {key: values[:, 0, 0] for key, values in expected.items()},
            )
            checkpoint.save_unit(
                0,
                1,
                {key: values[:, 0, 1] for key, values in expected.items()},
            )

            # Simulate a process dying while writing the next unit. Only final
            # layer_*_unit_*.npz files count as completed.
            interrupted = checkpoint.units_dir / ".layer_0001_unit_0000.npz.123.tmp"
            interrupted.write_bytes(b"incomplete")
            self.assertEqual(checkpoint.completed_unit_count(), 2)

            resumed = SweepCheckpoint(output_dir, _manifest(output_dir, samples), resume=True)
            self.assertTrue(np.array_equal(resumed.load_baseline(), baseline))
            restored_donors = resumed.load_donors(
                ["en", "es"],
                n_layers=2,
                n_heads=4,
                d_head=3,
            )
            self.assertIsNotNone(restored_donors)
            for language in ("en", "es"):
                self.assertTrue(torch.equal(restored_donors[language], donors[language]))

            assembled = {key: np.zeros_like(values) for key, values in expected.items()}
            recomputed = []
            for layer_idx in range(2):
                for patch_idx in range(2):
                    unit = resumed.load_unit(layer_idx, patch_idx)
                    if unit is None:
                        recomputed.append((layer_idx, patch_idx))
                        unit = {
                            key: values[:, layer_idx, patch_idx]
                            for key, values in expected.items()
                        }
                        resumed.save_unit(layer_idx, patch_idx, unit)
                    for key, values in unit.items():
                        assembled[key][:, layer_idx, patch_idx] = values

            self.assertEqual(recomputed, [(1, 0), (1, 1)])
            for key in expected:
                self.assertTrue(np.array_equal(assembled[key], expected[key]))
            self.assertEqual(resumed.completed_unit_count(), 4)

    def test_changed_samples_or_config_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "output"
            selected_ids = output_dir.parent / "selected_ids.json"
            selected_ids.write_text('["q0"]\n', encoding="utf-8")
            samples = [_sample("q0_en", "original")]
            checkpoint = SweepCheckpoint(output_dir, _manifest(output_dir, samples), resume=True)
            checkpoint.save_unit(
                0,
                0,
                {
                    "pct_var_mean_incorrect": np.zeros(1, dtype=np.float32),
                    "pct_var_max_incorrect": np.zeros(1, dtype=np.float32),
                    "rank_drop_correct": np.zeros(1, dtype=np.float32),
                },
            )

            changed_samples = [_sample("q0_en", "changed prompt")]
            with self.assertRaisesRegex(CheckpointMismatchError, "sample_fingerprint"):
                SweepCheckpoint(output_dir, _manifest(output_dir, changed_samples), resume=True)

            selected_ids.write_text('["q1"]\n', encoding="utf-8")
            with self.assertRaisesRegex(CheckpointMismatchError, "input_fingerprints"):
                SweepCheckpoint(output_dir, _manifest(output_dir, samples), resume=True)
            selected_ids.write_text('["q0"]\n', encoding="utf-8")

            changed_config = _config(output_dir)
            changed_config["patching"]["batch_size"] = 8
            manifest = build_manifest(
                cfg=changed_config,
                samples=samples,
                model_commit="resolved-commit",
                n_eval_samples=1,
                n_layers=2,
                n_patch_units=2,
                n_heads=4,
                n_kv_heads=2,
                d_head=3,
                n_options=4,
            )
            with self.assertRaisesRegex(CheckpointMismatchError, "config_fingerprint"):
                SweepCheckpoint(output_dir, manifest, resume=True)

            with self.assertRaisesRegex(CheckpointMismatchError, "resume=false"):
                SweepCheckpoint(output_dir, _manifest(output_dir, samples), resume=False)


class FinalOutputTests(unittest.TestCase):
    def test_save_outputs_preserves_public_npz_schema_and_marks_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "output"
            cfg = {"run": {"output_dir": str(output_dir)}}
            metric = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
            baseline_logits = np.arange(8, dtype=np.float32).reshape(2, 4)
            results = {
                "samples": [{"sample_index": 0}, {"sample_index": 1}],
                "metadata": {"n_samples": 2},
                "language_means": {"en": {}},
                "spearman": {},
                "jaccard": {},
                "baseline": {
                    "option_logits": baseline_logits,
                    "base_margin_mean": np.ones(2, dtype=np.float32),
                    "base_margin_max": np.ones(2, dtype=np.float32) * 2,
                    "base_correct_rank": np.ones(2, dtype=np.float32) * 3,
                },
                "importance": {
                    "pct_var_mean_incorrect": metric,
                    "pct_var_max_incorrect": metric + 1,
                    "rank_drop_correct": metric + 2,
                },
                "flat_vectors": {
                    "pct_var_mean_incorrect": metric.reshape(2, -1),
                    "pct_var_max_incorrect": (metric + 1).reshape(2, -1),
                    "rank_drop_correct": (metric + 2).reshape(2, -1),
                },
                "_checkpoint_fingerprint": "fingerprint",
            }

            save_outputs(cfg, results)

            with np.load(output_dir / "importance_tensors.npz") as saved:
                self.assertEqual(
                    set(saved.files),
                    {
                        "baseline_option_logits",
                        "base_margin_mean",
                        "base_margin_max",
                        "base_correct_rank",
                        "pct_var_mean_incorrect",
                        "pct_var_max_incorrect",
                        "rank_drop_correct",
                        "flat_pct_var_mean_incorrect",
                        "flat_pct_var_max_incorrect",
                        "flat_rank_drop_correct",
                    },
                )
                self.assertTrue(np.array_equal(saved["pct_var_mean_incorrect"], metric))
            success = json.loads((output_dir / "_SUCCESS").read_text(encoding="utf-8"))
            self.assertEqual(success["status"], "complete")
            self.assertEqual(success["checkpoint_fingerprint"], "fingerprint")
            self.assertFalse(list(output_dir.glob(".*.tmp")))


class SlurmRequeueTests(unittest.TestCase):
    def test_wrapper_requests_warning_signal_and_requeue(self) -> None:
        text = SWEEP_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --time=48:00:00", text)
        self.assertIn("#SBATCH --signal=B:USR1@600", text)
        self.assertIn("#SBATCH --requeue", text)

    def test_warning_requeues_only_current_array_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready = root / "ready"
            scontrol_log = root / "scontrol.log"
            child = root / "child.sh"
            child.write_text(
                "#!/usr/bin/env bash\n"
                "trap 'exit 75' USR1\n"
                "touch \"$1\"\n"
                "while true; do sleep 0.05; done\n",
                encoding="utf-8",
            )
            child.chmod(0o755)
            fake_scontrol = root / "scontrol"
            fake_scontrol.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$*\" > \"$SCONTROL_LOG\"\n",
                encoding="utf-8",
            )
            fake_scontrol.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "SLURM_JOB_ID": "999",
                    "SLURM_ARRAY_JOB_ID": "123",
                    "SLURM_ARRAY_TASK_ID": "3",
                    "SCONTROL_BIN": str(fake_scontrol),
                    "SCONTROL_LOG": str(scontrol_log),
                }
            )
            process = subprocess.Popen(
                ["bash", str(REQUEUE_HELPER), str(child), str(ready)],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(ready.exists(), "stub child did not become ready")
            process.send_signal(signal.SIGUSR1)
            stdout, stderr = process.communicate(timeout=5)

            self.assertEqual(process.returncode, 0, msg=f"stdout={stdout}\nstderr={stderr}")
            self.assertEqual(scontrol_log.read_text(encoding="utf-8").strip(), "requeue 123_3")

    def test_real_child_failure_is_not_requeued(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            scontrol_log = Path(temporary) / "scontrol.log"
            env = os.environ.copy()
            env.update(
                {
                    "SLURM_JOB_ID": "999",
                    "SCONTROL_BIN": "/bin/false",
                    "SCONTROL_LOG": str(scontrol_log),
                }
            )
            result = subprocess.run(
                ["bash", str(REQUEUE_HELPER), "bash", "-c", "exit 42"],
                env=env,
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(result.returncode, 42)
            self.assertFalse(scontrol_log.exists())


if __name__ == "__main__":
    unittest.main()
