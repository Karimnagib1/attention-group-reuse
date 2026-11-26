from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import random
import tempfile
from typing import Dict, List, Tuple, Optional
import unittest

import numpy as np
import torch

from review.latent_country.backend import Backend, replace_groups
from review.latent_country.protocol import (
    ROOT, build_protocol, digest, historical_hashes, historical_prompt,
    load_data, read_config, validate_protocol, write_json,
)
from review.latent_country.run import execute
from review.latent_country.statistics import random_sites, summarize_records


def small_config():
    cfg = read_config(ROOT / "review/configs/latent_country_qwen.json")
    cfg["protocol"].update(localization_pairs=2, evaluation_pairs=3, num_examples=2,
                           languages=["en", "es"], historical_exclusion={"num_pairs": 3, "num_examples": 2})
    cfg["model"] = {"name_or_path": "tiny-random-test-model"}
    cfg["selection"] = {"count": 2, "published_sites": [[0, 0], [1, 1]]}
    cfg["evaluation"].update(random_draws=3, bootstrap_resamples=30)
    cfg["inference"]["batch_size"] = 2
    return cfg


class ProtocolTests(unittest.TestCase):
    def test_historical_replay_matches_existing_generator(self):
        # Extract only the original pure function; never import or execute its main.
        tree = ast.parse((ROOT / "ICL_prompt_generator.py").read_text())
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "generate_icl_prompt")
        namespace = {"Dict": Dict, "List": List, "Tuple": Tuple, "Optional": Optional, "random": random,
                     "ALLOWED_SOURCE_TYPES": {"city", "university", "landmark"},
                     "ALLOWED_TARGET_TYPES": {"capital", "country calling code"}, "ALLOWED_DIRECTIONS": {"ltr", "rtl"}}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "original_generator", "exec"), namespace)
        data = load_data(ROOT / "data", ["en", "es", "de", "ru", "ar"])
        for lang, countries in data.items():
            for source in ("city", "university", "landmark"):
                for seed in (0, 73, 200):
                    old_target = "capital" if source == "city" else "country calling code"
                    old = namespace["generate_icl_prompt"](source, old_target, 20,
                                                            "rtl" if lang == "ar" else "ltr", seed, countries)
                    new = historical_prompt(countries, lang, source,
                                            "capital" if source == "city" else "country_code", seed, 20)
                    self.assertEqual(old, new)

    def test_reproducible_disjoint_aligned_prompts_and_target_language(self):
        cfg = small_config()
        protocol = build_protocol(cfg["protocol"], ROOT / "data")
        self.assertEqual(protocol, build_protocol(cfg["protocol"], ROOT / "data"))
        data = load_data(ROOT / "data", cfg["protocol"]["languages"])
        history = historical_hashes(data, cfg["protocol"])
        for split in ("localization", "evaluation"):
            for row in protocol[split]:
                self.assertNotEqual(row["donor_country"], row["recipient_country"])
                for key in ("recipient_prompt", "donor_prompt"):
                    self.assertNotIn(digest(row[key]), history)
                expected = " " + data[row["recipient_language"]][row["donor_country"]]["capital"]
                self.assertEqual(row["type_corrected_answer"], expected)
        # Every translated condition uses the same underlying countries for a seed/family.
        rows = protocol["evaluation"]
        for source in cfg["protocol"]["donor_sources"]:
            for index in range(3):
                aligned = [r for r in rows if r["donor_source"] == source and r["pair_index"] == index]
                self.assertEqual(len({(r["donor_country"], r["recipient_country"]) for r in aligned}), 1)
        corrupted = copy.deepcopy(protocol)
        corrupted["evaluation"][0]["recipient_prompt"] = protocol["localization"][0]["recipient_prompt"]
        corrupted["sha256"] = digest({k: v for k, v in corrupted.items() if k != "sha256"})
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_protocol(corrupted)
        corrupted = copy.deepcopy(protocol)
        corrupted["evaluation"][0]["recipient_prompt"] += " edited"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_protocol(corrupted)


class PatchTests(unittest.TestCase):
    def test_only_selected_group_at_final_position_changes(self):
        z = torch.arange(2 * 3 * 16).reshape(2, 3, 16).float()
        before = z.clone()
        donor = torch.full((2, 16), -11.0)
        patched = replace_groups(z, donor, [1, 3], 4)
        self.assertTrue(torch.equal(z, before))
        self.assertTrue(torch.equal(patched[:, :-1], before[:, :-1]))
        for group in (0, 2):
            self.assertTrue(torch.equal(patched[:, -1, group * 4:(group + 1) * 4], before[:, -1, group * 4:(group + 1) * 4]))
        for group in (1, 3):
            self.assertTrue(torch.all(patched[:, -1, group * 4:(group + 1) * 4] == -11))

    def test_actual_llama_and_qwen_hooks_restore_and_handle_head_dimensions(self):
        from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM
        class Tokenizer:
            pad_token_id = 0
            def __call__(self, prompts, **kwargs):
                if isinstance(prompts, str):
                    return {"input_ids": [ord(c) % 30 + 1 for c in prompts]}
                values = [[ord(c) % 30 + 1 for c in text] for text in prompts]
                width = max(map(len, values))
                return {"input_ids": torch.tensor([[0] * (width - len(row)) + row for row in values]),
                        "attention_mask": torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in values])}
        configs = [
            (LlamaForCausalLM, LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=48,
                                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)),
            # Deliberately make attention projection width differ from hidden_size.
            (Qwen3ForCausalLM, Qwen3Config(vocab_size=32, hidden_size=24, head_dim=8, intermediate_size=48,
                                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)),
        ]
        for model_class, config in configs:
            config._attn_implementation = "eager"
            torch.manual_seed(42)
            backend = Backend(model_class(config), Tokenizer(), 100)
            encoded = {"input_ids": torch.tensor([[0, 2, 3], [4, 5, 6]]),
                       "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]])}
            donors, hooks = {}, []
            for index, layer in enumerate(backend.layers):
                def capture(module, args, index=index):
                    donors[index] = args[0][:, -1, :].detach().clone()
                hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(capture))
            clean = backend.forward(encoded).clone()
            for hook in hooks:
                hook.remove()
            with backend.patch(donors, [[0, 0], [1, 1]]):
                self.assertTrue(torch.allclose(clean, backend.forward(encoded), atol=1e-6))
            with self.assertRaisesRegex(RuntimeError, "test exception"):
                with backend.patch(donors, [[0, 0]]):
                    raise RuntimeError("test exception")
            self.assertTrue(torch.equal(clean, backend.forward(encoded)))
            self.assertTrue(all(not layer.self_attn.o_proj._forward_pre_hooks for layer in backend.layers))
            alternative = {index: value + 2 for index, value in donors.items()}
            with backend.patch(alternative, [[0, 0]]):
                self.assertFalse(torch.allclose(clean, backend.forward(encoded)))
            # Exercise the actual capture/encode/score path, including left padding.
            prompts = ["abc", "defgh"]
            captured = backend.capture(prompts, [0, 1])
            encoded = backend.encode(prompts)
            baseline = backend.score(encoded, [1, 2], [3, 4])
            patched = backend.score(encoded, [1, 2], [3, 4], captured, [[0, 0], [1, 1]])
            self.assertEqual(baseline, patched)
            self.assertTrue(all(row["target_rank"] >= 1 for row in baseline))
            with self.assertRaisesRegex(ValueError, "truncation"):
                backend.encode(["x" * 101])


class StatisticsTests(unittest.TestCase):
    def test_random_controls_preserve_layer_counts(self):
        sites = [[1, 2], [1, 4], [3, 1]]
        first = random_sites(sites, 8, 30, 23)
        self.assertEqual(first, random_sites(sites, 8, 30, 23))
        for draw in first:
            self.assertEqual([site[0] for site in draw], [1, 1, 3])
            self.assertEqual(len({tuple(site) for site in draw}), 3)

    def test_paired_cluster_bootstrap_preserves_cross_condition_dependence(self):
        records = []
        for condition in ("en_to_en", "en_to_es"):
            for i in range(4):
                records.append({"condition": condition, "pair_index": i, "donor_language": "en",
                                "recipient_language": "en" if condition == "en_to_en" else "es",
                                "baseline": {"target_rank": 20}, "selected": {"target_rank": 1 if i < 2 else 2},
                                "random": [{"target_rank": 20}] * 3, "answer_token_collision": False})
        report = summarize_records(records, 200, 5)
        all_metrics = report["groups"]["all"]["metrics"]
        self.assertEqual(all_metrics["selected_top1_percent"]["estimate"], 50)
        self.assertEqual(all_metrics["selected_minus_baseline_top1_pp"]["estimate"], 50)
        self.assertEqual(all_metrics["selected_minus_random_top1_pp"]["estimate"], 50)
        self.assertEqual(all_metrics["selected_top10_percent"]["estimate"], 100)
        with self.assertRaisesRegex(ValueError, "complete"):
            summarize_records(records[:-1], 20, 5)

    def test_antiphase_conditions_have_constant_macro_average_in_every_resample(self):
        # Resampling conditions independently would incorrectly widen this CI.
        records = []
        for condition in range(2):
            for i in range(4):
                success = (i < 2) if condition == 0 else (i >= 2)
                records.append({"condition": str(condition), "pair_index": i, "donor_language": "en",
                                "recipient_language": "es", "baseline": {"target_rank": 20},
                                "selected": {"target_rank": 1 if success else 20},
                                "random": [{"target_rank": 20}], "answer_token_collision": False})
        summary = summarize_records(records, 100, 9)
        self.assertEqual(summary["groups"]["all"]["metrics"]["selected_top1_percent"]["ci95"], [50.0, 50.0])


class FakeBackend:
    n_layers, n_groups = 2, 2
    def __init__(self):
        self.calls = 0
    def identity(self):
        return {"kind": "deterministic-test-double"}
    def token_ids(self, rows):
        return [1] * len(rows), [2] * len(rows)
    def encode(self, prompts):
        return prompts
    def capture(self, prompts, layers):
        return {}
    def score(self, encoded, original, target, donors=None, sites=None):
        self.calls += 1
        effect = sum(1 + layer * 2 + group for layer, group in sites) if sites else 0
        return [{"original_rank": 1, "target_rank": 1 if effect >= 4 else 15,
                 "original_minus_target_logit": 10.0 - effect} for _ in encoded]


class WorkflowTests(unittest.TestCase):
    def test_localize_evaluate_resume_and_reject_changed_inputs(self):
        cfg = small_config()
        protocol = build_protocol(cfg["protocol"], ROOT / "data")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = FakeBackend()
            execute(cfg, protocol, root / "local", "localize", backend=backend)
            selection_path = root / "local/selection.json"
            selection = json.loads(selection_path.read_text())
            self.assertEqual(selection["sites"], [[1, 1], [1, 0]])
            execute(cfg, protocol, root / "eval", "evaluate", selection_path, backend)
            self.assertTrue((root / "eval/_SUCCESS").exists())
            self.assertTrue((root / "eval/summary.csv").exists())
            count = backend.calls
            execute(cfg, protocol, root / "eval", "evaluate", selection_path, backend)
            self.assertEqual(count, backend.calls)
            # Simulate interruption after one complete record disappeared.
            next((root / "eval/records").glob("*.json")).unlink()
            execute(cfg, protocol, root / "eval", "evaluate", selection_path, backend)
            self.assertEqual(backend.calls - count, 5)  # baseline + selected + three controls
            changed = copy.deepcopy(cfg)
            changed["evaluation"]["random_seed"] += 1
            with self.assertRaisesRegex(ValueError, "differs"):
                execute(changed, protocol, root / "eval", "evaluate", selection_path, backend)
            changed_protocol = copy.deepcopy(protocol)
            changed_protocol["settings"]["seed"] += 1
            changed_protocol["sha256"] = digest({k: v for k, v in changed_protocol.items() if k != "sha256"})
            changed_cfg = copy.deepcopy(cfg)
            changed_cfg["protocol"] = changed_protocol["settings"]
            with self.assertRaisesRegex(ValueError, "different protocol"):
                execute(changed_cfg, changed_protocol, root / "bad", "evaluate", selection_path, backend)
            execute(cfg, protocol, root / "fixed", "evaluate", backend=backend)
            record = next((root / "fixed/records").glob("*.json"))
            value = json.loads(record.read_text())
            value["prompt_sha256"] = "corrupt"
            write_json(record, value)
            with self.assertRaisesRegex(ValueError, "different run"):
                execute(cfg, protocol, root / "fixed", "evaluate", backend=backend)


if __name__ == "__main__":
    unittest.main()
