from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from donor_ablation.analysis import analyze
from donor_ablation.backend import Backend
from donor_ablation.engine import run as serial_run
from donor_ablation.protocol import build_protocol, load_protocol, seed_for
from donor_parallel.engine import finalize, load_manifest, setup, tasks, unit_path, work
from donor_parallel.run import Children, gpu_tokens, settings
from test_donor_ablation import FakeBackend, fixture
from quantitative_experiments.multilingual_head_patch.run import HeadPatchingRunner


def process_worker(directory, rank, workers):
    with redirect_stdout(io.StringIO()):
        path = Path(directory)
        work(load_protocol(path / "protocol.json"), path, rank, workers, FakeBackend())


def save_fixture(directory):
    cfg, samples = fixture(directory)
    protocol = build_protocol(cfg, samples, ["en", "es"])
    (Path(directory) / "protocol.json").write_text(json.dumps(protocol))
    return protocol


def real_backend(model):
    runner = object.__new__(HeadPatchingRunner)
    runner.model = runner.base_model = model
    runner.layers = model.model.layers
    runner.n_layers, runner.n_kv_heads = 2, 2
    runner.uses_data_parallel = False
    runner.option_token_ids = [1, 2, 3, 4]
    def encode(prompts):
        rows = []
        for p in prompts:
            rng = np.random.default_rng(seed_for(p))
            rows.append(rng.integers(5, 32, size=3 + seed_for(p) % 4).tolist())
        width = max(map(len, rows))
        return (torch.tensor([[0] * (width - len(row)) + row for row in rows]),
                torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in rows]))
    runner._encode_prompts = encode
    return Backend(runner)


class ParallelTests(unittest.TestCase):
    def test_two_process_results_equal_serial_and_resume_with_different_worker_count(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            output, serial = Path(tmp) / "parallel", Path(tmp) / "serial"
            output.mkdir()
            protocol = save_fixture(output)
            manifest = setup(protocol, output, FakeBackend())
            self.assertEqual(work(protocol, output, 0, 2, FakeBackend(), limit=1), 1)
            first = unit_path(output, *tasks(protocol, manifest)[0])
            timestamp = first.stat().st_mtime_ns
            with self.assertRaises(FileNotFoundError):
                finalize(protocol, output)
            self.assertFalse((output / "_SUCCESS").exists())
            ctx = multiprocessing.get_context("spawn")
            children = [ctx.Process(target=process_worker, args=(str(output), rank, 2)) for rank in range(2)]
            try:
                for p in children:
                    p.start()
                for p in children:
                    p.join(60)
                    self.assertEqual(p.exitcode, 0)
            finally:
                for p in children:
                    if p.is_alive():
                        p.terminate()
                        p.join()
            finalize(protocol, output)
            self.assertEqual(first.stat().st_mtime_ns, timestamp)
            serial_run(protocol, serial, FakeBackend())
            for variant in protocol["variants"]:
                name = Path("variants") / variant["name"] / "importance_tensors.npz"
                with np.load(output / name) as a, np.load(serial / name) as b:
                    self.assertEqual(set(a.files), set(b.files))
                    for key in a.files:
                        np.testing.assert_array_equal(a[key], b[key])
            analyze(output)
            self.assertTrue((output / "analysis/summary.json").exists())
            for rank in range(3):
                backend = FakeBackend()
                self.assertEqual(work(protocol, output, rank, 3, backend), 0)
                self.assertEqual(backend.calls, {"capture": 0, "logits": 0, "patched": 0})

    def test_wrong_model_or_missing_shared_cache_cannot_write_patch_results(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            output = Path(tmp)
            protocol = save_fixture(output)
            setup(protocol, output, FakeBackend())
            backend = FakeBackend()
            backend.identity = lambda: {"model": "wrong-model"}
            with self.assertRaisesRegex(ValueError, "Worker model differs"):
                work(protocol, output, 0, 1, backend)
            (output / "donor_cache/_READY.json").unlink()
            with self.assertRaisesRegex(ValueError, "Shared donor cache is missing"):
                work(protocol, output, 0, 1, FakeBackend())
            self.assertFalse((output / "checkpoints").exists())

    def test_bad_checkpoint_and_implementation_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            output = Path(tmp)
            protocol = save_fixture(output)
            manifest = setup(protocol, output, FakeBackend())
            with patch("donor_parallel.engine.code_digest", return_value="changed"):
                with self.assertRaisesRegex(ValueError, "implementation changed"):
                    load_manifest(output, protocol)
            p = unit_path(output, *tasks(protocol, manifest)[0])
            p.parent.mkdir(parents=True)
            np.savez(p, fingerprint="wrong")
            with self.assertRaisesRegex(ValueError, "different run"):
                work(protocol, output, 0, 1, FakeBackend())

    def test_real_qwen_and_llama_hooks_match_serial_results(self):
        from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM
        torch.set_num_threads(1)
        for config_cls, model_cls in ((Qwen3Config, Qwen3ForCausalLM), (LlamaConfig, LlamaForCausalLM)):
            with self.subTest(model=model_cls.__name__), tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
                output, serial = Path(tmp) / "parallel", Path(tmp) / "serial"
                output.mkdir()
                protocol = save_fixture(output)
                torch.manual_seed(17)
                model = model_cls(config_cls(vocab_size=32, hidden_size=32, intermediate_size=48,
                                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                             head_dim=8, attention_dropout=0.0)).eval()
                backend = real_backend(model)
                setup(protocol, output, backend)
                for rank in range(2):
                    work(protocol, output, rank, 2, backend)
                finalize(protocol, output)
                serial_run(protocol, serial, backend)
                for variant in protocol["variants"]:
                    name = Path("variants") / variant["name"] / "importance_tensors.npz"
                    with np.load(output / name) as a, np.load(serial / name) as b:
                        for key in a.files:
                            np.testing.assert_array_equal(a[key], b[key])
                self.assertTrue(all(not layer.self_attn.o_proj._forward_pre_hooks for layer in model.model.layers))


class LauncherTests(unittest.TestCase):
    def test_gpu_mask_preserves_slurm_mapping(self):
        self.assertEqual(gpu_tokens(4, "2,3,6,7"), ["2", "3", "6", "7"])
        self.assertEqual(gpu_tokens(2, "GPU-abc,GPU-def"), ["GPU-abc", "GPU-def"])
        self.assertEqual(gpu_tokens(2), ["0", "1"])
        for count, visible in ((2, "0"), (2, "0,0"), (1, "-1")):
            with self.assertRaises(RuntimeError):
                gpu_tokens(count, visible)

    def test_defaults_override_large_serial_batch_and_disable_sharding(self):
        from argparse import Namespace
        args = Namespace(config=Path(__file__).resolve().parents[1] / "configs/qwen_mmluprox.yaml",
                         output_dir=None, dtype="float32", batch_size=4, donor_batch_size=4,
                         set=[], samples_per_language=100)
        cfg = settings(args)
        self.assertEqual(cfg["patching"]["batch_size"], 4)
        self.assertEqual(cfg["model"]["dtype"], "float32")
        self.assertFalse(cfg["run"]["multi_gpu"])
        self.assertFalse(cfg["run"]["data_parallel"])
        self.assertIsNone(cfg["run"]["device_map"])
        self.assertTrue(cfg["run"]["output_dir"].endswith("qwen_mmluprox_parallel_float32"))

    def test_child_masks_one_gpu_without_changing_parent(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3,6,7"}):
            children = Children()
            with patch("donor_parallel.run.subprocess.Popen") as popen:
                children.start(Path(tmp), "6", "work", 2, 4)
                self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "6")
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2,3,6,7")
            for handle in children.handles:
                handle.close()

    def test_failed_worker_is_reported_and_cleanup_stops_other_children(self):
        children = Children()
        failed = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"])
        active = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        children.items = [(failed, "work", 0), (active, "work", 1)]
        try:
            failed.wait(timeout=10)
            with self.assertRaisesRegex(RuntimeError, "exit code 7"):
                children.wait()
        finally:
            children.close()
        self.assertIsNotNone(active.poll())


if __name__ == "__main__":
    unittest.main()
