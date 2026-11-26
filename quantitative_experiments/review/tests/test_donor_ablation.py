from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
from itertools import permutations
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from donor_ablation.analysis import analyze, bootstrap_weights, similarity_matrices, weighted_statistics
from donor_ablation.backend import Backend
from donor_ablation.engine import donor_vectors, prepare_donors, run
from donor_ablation.protocol import build_protocol, digest, load_protocol, load_settings, seed_for, validate_protocol
from quantitative_experiments.multilingual_head_patch.run import HeadPatchingRunner, Sample, _render_prompt_from_parts

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def fixture(output):
    cfg = load_settings(CONFIGS / "qwen_belebele.yaml", sample_count=3, output=Path(output))
    cfg["donors"].update(pool_questions=2, mean_repeats=2, random_repeats=2)
    cfg["analysis"].update(bootstrap_resamples=32, jaccard_k=2)
    cfg["patching"].update(batch_size=5, donor_batch_size=5)
    samples = []
    for q in range(8):
        for lang in ("en", "es"):
            question, options = f"Question {q} in {lang}?", ("one", "two", "three", "four")
            prompt = _render_prompt_from_parts(question=question, context="", options=options, cfg=cfg)
            samples.append(Sample(f"q{q}_{lang}", f"q{q}", lang, prompt, q % 4, question, "", options))
    return cfg, samples


class FakeBackend:
    n_layers, n_groups, width, n_options = 2, 2, 4, 4

    def __init__(self):
        self.calls = {"capture": 0, "logits": 0, "patched": 0}

    def identity(self):
        return {"model": "deterministic-test-backend"}

    def activation(self, prompts):
        return np.stack([np.random.default_rng(seed_for(p)).normal(size=(2, 4)) for p in prompts]).astype(np.float32)

    def capture(self, prompts):
        self.calls["capture"] += 1
        return self.activation(prompts)

    def logits(self, prompts):
        self.calls["logits"] += 1
        return np.stack([np.random.default_rng(seed_for(p, "logits")).normal(size=4) for p in prompts]).astype(np.float32)

    def patched_logits(self, prompts, layer, group, donors):
        self.calls["patched"] += 1
        base = self.logits(prompts)
        sl = slice(group * 2, (group + 1) * 2)
        delta = (donors[:, sl] - self.activation(prompts)[:, layer, sl]).sum(axis=1)
        return base + delta[:, None] * np.array([0.5, -0.2, 0.3, -0.7], dtype=np.float32) * (layer + 1)


class ProtocolTests(unittest.TestCase):
    def test_configs_default_to_100_balanced_questions(self):
        for path in CONFIGS.glob("*.yaml"):
            cfg = load_settings(path)
            self.assertEqual(cfg["selection"]["samples_per_language"], 100)
            self.assertEqual(len(cfg["data"]["languages"]), 5)
            self.assertFalse(cfg["run"]["data_parallel"])
            ids = json.loads(Path(cfg["data"]["allowed_parallel_ids_path"]).read_text())
            self.assertGreaterEqual(len(ids), 200)
        cfg = load_settings(CONFIGS / "llama_mmluprox.yaml", sample_count=17,
                            overrides=["donors.random_repeats=5"])
        self.assertEqual(cfg["selection"]["samples_per_language"], 17)
        self.assertEqual(cfg["donors"]["random_repeats"], 5)

    def test_selection_donor_constraints_and_reproducibility(self):
        cfg, samples = fixture("/tmp/test-donor-protocol")
        protocol = build_protocol(cfg, samples, ["en", "es"])
        self.assertEqual(protocol, build_protocol(cfg, samples, ["en", "es"]))
        reordered = build_protocol(cfg, list(reversed(samples)), ["en", "es"])
        self.assertEqual(protocol["recipient_ids"], reordered["recipient_ids"])
        self.assertEqual(len(protocol["recipient_rows"]), 6)
        self.assertEqual(len(protocol["expanded_samples"]), 24)
        self.assertEqual(protocol["variants_per_base"], [4] * 6)
        for variant in protocol["variants"]:
            if variant["kind"] == "disjoint_mean":
                self.assertFalse(set(variant["parallel_ids"]) & set(protocol["recipient_ids"]))
            if variant["kind"] == "random_in_language":
                for recipient, donor in zip(protocol["recipient_rows"], variant["donor_indices"]):
                    self.assertEqual(samples[recipient].language, samples[donor].language)
                    self.assertNotEqual(samples[recipient].parallel_id, samples[donor].parallel_id)
        protocol["recipient_ids"][0] = "tampered"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_protocol(protocol)

    def test_invalid_pools_and_limits_fail(self):
        cfg, samples = fixture("/tmp/test-donor-protocol")
        cfg["selection"]["samples_per_language"] = 9
        with self.assertRaisesRegex(ValueError, "only 8"):
            build_protocol(cfg, samples, ["en", "es"])
        cfg["selection"]["samples_per_language"] = 7
        with self.assertRaisesRegex(ValueError, "only 1"):
            build_protocol(cfg, samples, ["en", "es"])
        with self.assertRaisesRegex(ValueError, "data limits"):
            load_settings(CONFIGS / "qwen_belebele.yaml", ["data.max_samples=100"])
        with self.assertRaisesRegex(ValueError, "data_parallel|DataParallel"):
            load_settings(CONFIGS / "qwen_belebele.yaml", ["run.data_parallel=true"])

    def test_saved_config_cannot_silently_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, samples = fixture(tmp)
            protocol = build_protocol(cfg, samples, ["en", "es"])
            path = Path(tmp) / "protocol.json"
            path.write_text(json.dumps(protocol))
            self.assertEqual(load_protocol(path, cfg)["sha256"], protocol["sha256"])
            changed = copy.deepcopy(cfg)
            changed["donors"]["seed"] += 1
            with self.assertRaisesRegex(ValueError, "Configuration differs"):
                load_protocol(path, changed)


class EngineTests(unittest.TestCase):
    def test_all_donor_formulas_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            cfg, samples = fixture(tmp)
            protocol = build_protocol(cfg, samples, ["en", "es"])
            backend = FakeBackend()
            sums, banks = prepare_donors(protocol, backend, Path(tmp), "test")
            activations = backend.activation([s.prompt for s in samples])
            for variant in protocol["variants"]:
                for layer in range(2):
                    actual = donor_vectors(protocol, variant, sums, banks[layer], layer)
                    for row, recipient_index in enumerate(protocol["recipient_rows"]):
                        recipient = samples[recipient_index]
                        pool = [i for i, s in enumerate(samples) if s.language == recipient.language]
                        if variant["kind"] == "leave_one_out":
                            pool.remove(recipient_index)
                        elif variant["kind"] == "disjoint_mean":
                            pool = [i for i in pool if samples[i].parallel_id in variant["parallel_ids"]]
                        elif variant["kind"] == "random_in_language":
                            pool = [variant["donor_indices"][row]]
                        expected = activations[pool, layer].astype(np.float64).mean(axis=0)
                        np.testing.assert_allclose(actual[row], expected, rtol=1e-6, atol=1e-7)
            calls = backend.calls.copy()
            prepare_donors(protocol, backend, Path(tmp), "test")
            self.assertEqual(calls, backend.calls)

    def test_end_to_end_scores_resume_and_analysis(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            cfg, samples = fixture(tmp)
            protocol = build_protocol(cfg, samples, ["en", "es"])
            output, backend = Path(tmp), FakeBackend()
            run(protocol, output, backend)
            self.assertTrue((output / "_SUCCESS").exists())
            # Independently check per-order percentage drops are averaged AFTER normalization.
            prompts = [s["prompt"] for s in protocol["expanded_samples"]]
            targets = np.array([s["correct_idx"] for s in protocol["expanded_samples"]])
            base_ids = np.array(protocol["eval_to_base"])
            activations = backend.activation([s.prompt for s in samples])
            recipient_donors = np.stack([activations[[i for i, s in enumerate(samples) if s.language == samples[r].language], 0].mean(axis=0)
                                         for r in protocol["recipient_rows"]])
            baseline = backend.logits(prompts)
            patched = backend.patched_logits(prompts, 0, 0, recipient_donors[base_ids])
            def margin(logits):
                gold = logits[np.arange(len(logits)), targets]
                return gold - (logits.sum(axis=1) - gold) / 3
            expected = 100 * (margin(baseline) - margin(patched)) / (np.abs(margin(baseline)) + 1e-8)
            expected = np.array([expected[base_ids == i].mean() for i in range(6)])
            with np.load(output / "variants/language_mean/importance_tensors.npz") as result:
                np.testing.assert_allclose(result["pct_var_mean_incorrect"][:, 0, 0], expected, rtol=3e-5, atol=3e-4)
            calls = backend.calls.copy()
            run(protocol, output, backend)
            self.assertEqual(calls, backend.calls)
            # Removing one unit recomputes just that unit, using the existing donors/baseline.
            (output / "checkpoints/language_mean/layer_000_group_000.npz").unlink()
            run(protocol, output, backend)
            self.assertEqual(backend.calls["patched"] - calls["patched"], 5)  # ceil(24/5)
            self.assertEqual(backend.calls["capture"], calls["capture"])
            summary = analyze(output)
            self.assertEqual(len(summary["similarity"]), 12)  # six donors x two metrics
            reference = next(r for r in summary["signature_stability"] if r["variant"] == "language_mean")
            self.assertAlmostEqual(reference["spearman_vs_reference"], 1)
            self.assertTrue((output / "analysis/signature_stability_per_question.csv").exists())
            # A changed identity must not reuse results or destroy the success marker.
            with patch.object(backend, "identity", return_value={"model": "different"}):
                with self.assertRaisesRegex(ValueError, "Existing run differs"):
                    run(protocol, output, backend)
            self.assertTrue((output / "_SUCCESS").exists())


class AnalysisTests(unittest.TestCase):
    def test_null_matches_exhaustive_permutations_and_cluster_resampling(self):
        values = np.array([[4, 1, 2, 3], [1, 4, 3, 2], [2, 3, 4, 1]], dtype=float)
        for matrix in similarity_matrices(values, values, 2).values():
            observed = weighted_statistics(matrix, np.ones((1, 3)))[0]
            exact = np.mean([matrix[np.arange(3), p].mean() for p in permutations(range(3))])
            self.assertAlmostEqual(observed[0], 1)
            self.assertAlmostEqual(observed[1], exact)
            self.assertGreater(observed[2], 0)
            w = np.array([[2, 0, 1]], dtype=float)
            duplicated = matrix[np.ix_([0, 0, 2], [0, 0, 2])]
            actual = weighted_statistics(matrix, w)[0]
            np.testing.assert_allclose(actual, [np.diag(duplicated).mean(), duplicated.mean(), np.diag(duplicated).mean() - duplicated.mean()])
        np.testing.assert_array_equal(bootstrap_weights(3, 10, 42), bootstrap_weights(3, 10, 42))

    def test_ties_and_constant_signatures(self):
        a = np.array([[1, 1, 2], [3, 3, 3]], dtype=float)
        matrices = similarity_matrices(a, a, 1)
        self.assertAlmostEqual(matrices["spearman"][0, 0], 1)
        self.assertTrue(np.isnan(matrices["spearman"][1]).all())
        stats = weighted_statistics(matrices["spearman"], np.ones((1, 2)))[0]
        np.testing.assert_allclose(stats, [1, 1, 0], atol=1e-12)


class RealHookTests(unittest.TestCase):
    def test_llama_and_qwen_group_slice_and_hook_cleanup(self):
        from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM
        torch.set_num_threads(1)
        for config_class, model_class, head_dim in ((LlamaConfig, LlamaForCausalLM, 8), (Qwen3Config, Qwen3ForCausalLM, 16)):
            with self.subTest(model=model_class.__name__):
                torch.manual_seed(11)
                config = config_class(vocab_size=32, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, head_dim=head_dim)
                model = model_class(config).eval()
                runner = object.__new__(HeadPatchingRunner)
                runner.model = runner.base_model = model
                runner.layers = model.model.layers
                runner.n_layers, runner.n_kv_heads = 2, 2
                runner.uses_data_parallel = False
                runner.option_token_ids = [1, 2, 3, 4]
                # Left-padded variable-length inputs; use the actual original option-logit forward.
                runner._encode_prompts = lambda prompts: (
                    torch.tensor([[0, 0, 5, 6], [7, 8, 9, 10]][:len(prompts)]),
                    torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]][:len(prompts)]))
                backend = Backend(runner)
                self.assertEqual(backend.width, head_dim * 4)
                prompts = ["short", "long"]
                baseline = backend.logits(prompts)
                captured = backend.capture(prompts)
                for layer in range(2):
                    for group in range(2):
                        np.testing.assert_allclose(backend.patched_logits(prompts, layer, group, captured[:, layer]), baseline,
                                                   rtol=1e-6, atol=1e-7)
                block = runner.layers[0].self_attn.o_proj
                clean = []
                handle = block.register_forward_pre_hook(lambda module, args: clean.append(args[0].detach().clone()))
                backend.logits(prompts)
                handle.remove()
                after = []
                original_forward = runner._forward_option_logits
                def inspect_forward(prompts):
                    handle = block.register_forward_pre_hook(lambda module, args: after.append(args[0].detach().clone()))
                    try:
                        return original_forward(prompts)
                    finally:
                        handle.remove()
                with patch.object(runner, "_forward_option_logits", side_effect=inspect_forward):
                    backend.patched_logits(prompts, 0, 1, captured[:, 0] + 2)
                expected = clean[0].clone()
                expected[:, -1, backend.width // 2:] += 2
                torch.testing.assert_close(after[0], expected)
                with patch.object(runner, "_forward_option_logits", side_effect=RuntimeError("test failure")):
                    with self.assertRaisesRegex(RuntimeError, "test failure"):
                        backend.patched_logits(prompts, 0, 0, captured[:, 0])
                    with self.assertRaisesRegex(RuntimeError, "test failure"):
                        backend.capture(prompts)
                self.assertTrue(all(not layer.self_attn.o_proj._forward_pre_hooks for layer in runner.layers))


if __name__ == "__main__":
    unittest.main()
