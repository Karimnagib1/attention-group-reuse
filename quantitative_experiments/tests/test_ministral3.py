"""Offline checks using a tiny real Ministral architecture; no model downloads."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from unittest.mock import patch

pytest.importorskip("transformers.models.ministral3")
from transformers import Mistral3Config, Mistral3ForConditionalGeneration
from transformers.models.ministral3 import Ministral3Config
from transformers.models.pixtral import PixtralVisionConfig

from quantitative_experiments import ministral3 as adapter


class TinyTokenizer:
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0
    padding_side = "left"

    def __call__(self, text, add_special_tokens=False):
        if text in (" A", " B", " C", " D"):
            return SimpleNamespace(input_ids=[3 + ord(text[-1]) - ord("A")])
        return SimpleNamespace(input_ids=[16 + ord(char) % 100 for char in text])

    def apply_chat_template(self, messages, *, tokenize, return_dict):
        assert tokenize and not return_dict
        return [1, 7] + self(messages[0]["content"]).input_ids + [8]

    def pad(self, encoded, **kwargs):
        rows = encoded["input_ids"]
        width = max(map(len, rows))
        return {
            "input_ids": torch.tensor([[0] * (width - len(row)) + row for row in rows]),
            "attention_mask": torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in rows]),
        }

    def decode(self, tokens, **kwargs):
        return {3: " A", 4: "B.", 5: "Answer", 6: " C", 7: "(A)"}.get(int(tokens[0]), "?")


@pytest.fixture
def cfg(tmp_path):
    config = adapter.load_config([
        str(Path(__file__).resolve().parents[1] / "configs/ministral3/mmluprox.yaml")
    ])
    config["model"].update(name_or_path="tiny-ministral", dtype="float32")
    config["data"].update(languages=["en", "es"], option_fields=[f"option_{i}" for i in range(4)])
    config["eval"].update(option_labels=list("ABCD"), max_input_length=64)
    config["run"].update(device="cpu", device_map=None, output_dir=str(tmp_path))
    config["patching"]["choice_order_perturbations"]["enabled"] = False
    return config


def make_tiny_model():
    torch.manual_seed(7)
    text = Ministral3Config(
        vocab_size=128, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "llama_4_scaling_beta": 0.1},
    )
    text.rope_parameters["original_max_position_embeddings"] = 128
    vision = PixtralVisionConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                                num_attention_heads=2, image_size=4, patch_size=2)
    config = Mistral3Config(text_config=text, vision_config=vision,
                           image_token_index=127, tie_word_embeddings=False)
    config._attn_implementation = "sdpa"
    return Mistral3ForConditionalGeneration(config).eval()


@pytest.fixture
def tiny_model(monkeypatch):
    model = make_tiny_model()
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda *a, **kw: model.config)
    monkeypatch.setattr("transformers.MistralCommonBackend.from_pretrained", lambda *a, **kw: TinyTokenizer())
    monkeypatch.setattr(Mistral3ForConditionalGeneration, "from_pretrained", lambda *a, **kw: model)
    return model


@pytest.fixture
def rows():
    return [dict(question_id=f"q{i}", language=lang, question=f"Question {i} in {lang}?",
                 answer_index=i, **{f"option_{j}": f"Choice {j}" for j in range(4)})
            for i in range(2) for lang in ("en", "es")]


@pytest.mark.parametrize("prompt_format", ["raw", "chat"])
def test_shared_encoding_and_explicit_head_width(cfg, tiny_model, prompt_format):
    cfg["eval"]["prompt_format"] = prompt_format
    scorer = adapter.MinistralScorer(cfg, "cpu")
    runner = adapter.MinistralRunner(cfg)
    assert runner.d_head == 8 != runner.d_model // runner.n_heads
    prompts = ["Short", "A longer prompt" * 10]
    for a, b in zip(scorer._encode_prompts(prompts), runner._encode_prompts(prompts)):
        torch.testing.assert_close(a, b)
    ids, mask = runner._encode_prompts(prompts)
    assert ids.shape == (2, 64)
    assert mask[0, 0] == 0 and torch.all(mask[:, -1] == 1)
    expected = tiny_model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -1, [3, 4, 5, 6]]
    torch.testing.assert_close(runner._forward_option_logits(prompts), expected)


def test_generation_and_baseline_records(cfg, tiny_model, rows, monkeypatch):
    scorer = adapter.MinistralScorer(cfg, "cpu")
    # Run real HF generation once, then supply controlled completions for parsing/accuracy.
    assert len(scorer.generate_choices(["Question?"], list("ABCD"), 2, 0.0, False, 1)) == 1

    def generate(input_ids, **kwargs):
        tokens = torch.tensor([[3], [5], [4], [4]])[:input_ids.shape[0]]
        return torch.cat([input_ids, tokens], dim=1)

    monkeypatch.setattr(tiny_model, "generate", generate)
    monkeypatch.setattr(adapter.evaluation, "load_mmluprox_dataset", lambda cfg: rows)
    original = adapter.evaluation.Scorer
    result = adapter.run(cfg, "evaluate")
    assert adapter.evaluation.Scorer is original
    assert result["metrics"]["accuracy"] == 0.75
    assert result["all_correct_ids"] == ["q1"]
    assert result["records"][1]["pred"] == "unknown"  # 'Answer' is not A.
    output = Path(cfg["run"]["output_dir"])
    assert json.loads((output / "all_correct_ids.json").read_text()) == ["q1"]
    assert len((output / "predictions.jsonl").read_text().splitlines()) == 4
    monkeypatch.setattr(adapter.sweeping, "load_eval_dataset", lambda cfg: rows)
    sweep = adapter.run(cfg, "patch")
    assert {sample["parallel_id"] for sample in sweep["samples"]} == {"q1"}
    # One donor per language must be a no-op when there are no permutations.
    np.testing.assert_allclose(sweep["importance"]["pct_var_mean_incorrect"], 0, atol=1e-4)


def test_group_sweep_matches_independent_intervention(cfg, tiny_model, rows, monkeypatch):
    cfg["eval"]["max_input_length"] = 128
    cfg["data"]["allowed_parallel_ids_path"] = None
    monkeypatch.setattr(adapter.sweeping, "load_eval_dataset", lambda cfg: rows)
    runner = adapter.MinistralRunner(cfg)
    samples, _ = adapter.sweeping._build_parallel_samples(cfg)
    prompts = [sample.prompt for sample in samples]
    captures = []
    projection = runner.layers[0].self_attn.o_proj
    handle = projection.register_forward_pre_hook(lambda module, inputs: captures.append(inputs[0].detach().clone()))
    with torch.no_grad():
        clean = runner._forward_option_logits(prompts)
    handle.remove()
    original_z = captures[0]
    # Independent flat slices: group 1 contains query heads 2 and 3 (16:32),
    # despite a hidden size of only 16. Average donor vectors by language.
    donor = original_z[:, -1, 16:32].clone()
    for indices in ([0, 2], [1, 3]):
        donor[indices] = donor[indices].mean(dim=0)

    def intervene(module, inputs):
        changed = inputs[0].clone()
        changed[:, -1, 16:32] = donor
        torch.testing.assert_close(changed[:, :-1], inputs[0][:, :-1])
        torch.testing.assert_close(changed[:, -1, :16], inputs[0][:, -1, :16])
        return (changed,)

    handle = projection.register_forward_pre_hook(intervene)
    patched = runner._forward_option_logits(prompts)
    handle.remove()
    assert not torch.allclose(clean, patched, atol=1e-7)
    gold = torch.tensor([sample.correct_idx for sample in samples])

    def margin(logits):
        correct = logits[torch.arange(len(gold)), gold]
        return correct - (logits.sum(dim=1) - correct) / 3

    before, after = margin(clean), margin(patched)
    expected = ((before - after) / (before.abs() + 1e-8) * 100).numpy()
    original = adapter.sweeping.HeadPatchingRunner
    result = adapter.run(cfg, "patch")
    assert adapter.sweeping.HeadPatchingRunner is original
    assert result["metadata"]["d_head"] == 8
    assert result["importance"]["pct_var_mean_incorrect"].shape == (4, 2, 2)
    np.testing.assert_allclose(result["importance"]["pct_var_mean_incorrect"][:, 0, 1], expected, rtol=2e-4, atol=2e-4)
    assert (Path(cfg["run"]["output_dir"]) / "_SUCCESS").is_file()
    # A resumed sweep should reuse the completed units without running the model.
    with patch.object(tiny_model, "forward", side_effect=AssertionError("Should resume cached units")):
        resumed = adapter.run(cfg, "patch")
    np.testing.assert_array_equal(resumed["importance"]["pct_var_mean_incorrect"], result["importance"]["pct_var_mean_incorrect"])


def test_choice_order_averaging(cfg, tiny_model, rows, monkeypatch):
    cfg["data"]["allowed_parallel_ids_path"] = None
    cfg["patching"]["choice_order_perturbations"]["enabled"] = True
    monkeypatch.setattr(adapter.sweeping, "load_eval_dataset", lambda cfg: rows)
    result = adapter.run(cfg, "patch")
    assert result["metadata"]["n_eval_samples_before_perturb_aggregation"] == 16
    assert result["importance"]["pct_var_mean_incorrect"].shape == (4, 2, 2)
    assert all(sample["n_choice_order_variants"] == 4 for sample in result["samples"])


def test_missing_or_mismatched_baseline_is_rejected(cfg):
    with pytest.raises(FileNotFoundError, match="Run evaluate first"):
        adapter.run(cfg, "patch")
    output = Path(cfg["run"]["output_dir"])
    (output / "all_correct_ids.json").write_text('["q1"]')
    old = copy.deepcopy(cfg)
    old["eval"]["prompt_format"] = "chat"
    (output / "evaluate_config.json").write_text(json.dumps(old))
    with pytest.raises(ValueError, match="Baseline eval config differs"):
        adapter.run(cfg, "patch")


def test_rejects_wrong_architecture_quantization_and_distributed(cfg, tiny_model, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="one process"):
        adapter.MinistralRunner(cfg)
    monkeypatch.setenv("WORLD_SIZE", "1")
    tiny_model.config.quantization_config = {"quant_method": "fp8"}
    with pytest.raises(ValueError, match="BF16"):
        adapter.MinistralRunner(cfg)
    del tiny_model.config.quantization_config
    tiny_model.config.text_config.model_type = "llama"
    with pytest.raises(ValueError, match="Ministral 3"):
        adapter.MinistralRunner(cfg)


class ReplicaTestScorer:
    """Deterministic answers for checking aggregation across real processes."""

    def __init__(self, cfg, device):
        pass

    def generate_choices(self, prompts, labels, **kwargs):
        return ["B" if "wrong" in prompt else "A" for prompt in prompts]


def _cpu_replica_test_worker(rank, cfg, world_size, rendezvous, result_path, rows):
    # Patch inside each spawned process; exercise the actual worker, sampler,
    # shared evaluation loop, reductions, and prediction gathering on CPU/Gloo.
    with patch.object(adapter, "MinistralScorer", ReplicaTestScorer), \
         patch.object(adapter.evaluation, "load_mmluprox_dataset", return_value=rows), \
         patch.object(torch.cuda, "is_available", return_value=False):
        adapter._evaluation_worker(rank, cfg, world_size, rendezvous, result_path, device_type="cpu")


@pytest.mark.parametrize("world_size,n_rows", [(2, 5), (3, 2)])
def test_replicas_merge_exactly_once_with_uneven_or_empty_shards(cfg, tmp_path, world_size, n_rows):
    rows = [dict(question_id=f"q{i // 2}", language="en" if i % 2 == 0 else "es",
                 question="wrong" if i == 2 else "correct", answer_index=0,
                 **{f"option_{j}": f"Option {j}" for j in range(4)}) for i in range(n_rows)]
    with patch.object(adapter, "MinistralScorer", ReplicaTestScorer), \
         patch.object(adapter.evaluation, "load_mmluprox_dataset", return_value=rows), \
         patch.object(torch.cuda, "is_available", return_value=False):
        single = adapter._evaluate(cfg, adapter.DistInfo(False, 1, 0, 0))
    cfg["run"]["data_parallel"] = True
    output = tmp_path / "gathered.json"
    adapter.mp.spawn(
        _cpu_replica_test_worker,
        args=(cfg, world_size, f"file://{tmp_path / 'rendezvous'}", str(output), rows),
        nprocs=world_size, join=True,
    )
    merged = json.loads(output.read_text())
    for key in ("metrics", "per_language", "all_correct_ids"):
        assert merged[key] == single[key]
    key = lambda row: (row["id"], row["language"])
    assert sorted(merged["records"], key=key) == sorted(single["records"], key=key)
    assert len({key(row) for row in merged["records"]}) == n_rows
    assert merged["all_correct_ids"] == ["q0"]
    # The parent publishes the usual files, compatible with sharded patching.
    with patch.object(adapter, "_evaluate_replicated", return_value=merged):
        adapter.run(cfg, "evaluate")
    assert json.loads((tmp_path / "all_correct_ids.json").read_text()) == ["q0"]
    assert len((tmp_path / "predictions.jsonl").read_text().splitlines()) == n_rows


def test_replica_load_is_pinned_to_its_own_device(cfg, tiny_model, monkeypatch):
    cfg["run"].update(data_parallel=True, device_map="auto", max_memory={0: "8GiB"})
    monkeypatch.setattr(adapter.dist, "is_initialized", lambda: True)
    with patch.object(Mistral3ForConditionalGeneration, "from_pretrained", return_value=tiny_model) as load:
        scorer = adapter.MinistralScorer(cfg, "cuda:1")
    assert load.call_args.kwargs["device_map"] == {"": "cuda:1"}
    assert "max_memory" not in load.call_args.kwargs
    assert not scorer.uses_model_sharding


def test_replication_errors_do_not_publish_results(cfg, tmp_path, monkeypatch):
    cfg["run"]["data_parallel"] = True
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(ValueError, match="visible CUDA GPUs"):
        adapter.run(cfg, "evaluate")
    from quantitative_experiments.ministral3_parallel_patch import run_replicated_patch
    with pytest.raises(ValueError, match="visible CUDA GPUs"):
        run_replicated_patch(cfg)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with patch.object(adapter.mp, "spawn", side_effect=RuntimeError("Worker failed")):
        with pytest.raises(RuntimeError, match="Worker failed"):
            adapter.run(cfg, "evaluate")
    assert not (tmp_path / "metrics.json").exists()
    assert not (tmp_path / "evaluate_config.json").exists()
