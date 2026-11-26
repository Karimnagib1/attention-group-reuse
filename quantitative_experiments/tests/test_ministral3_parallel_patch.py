"""Real CPU replicas verify patching equivalence without model downloads."""
import copy
import json
import signal
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

pytest.importorskip("transformers.models.ministral3")
from .test_ministral3 import TinyTokenizer, cfg, make_tiny_model, rows, tiny_model
from quantitative_experiments import ministral3 as adapter
from quantitative_experiments import ministral3_parallel_patch as parallel


def _cpu_patch_worker(rank, config, world_size, rendezvous, records, trace_dir, fail_baseline=False):
    model = make_tiny_model().to(dtype=getattr(torch, config["model"]["dtype"]))
    events = []
    model.register_forward_pre_hook(lambda *args: events.append(["forward"]))

    class TracedCheckpoint(parallel.ReplicaCheckpoint):
        def save_donors(self, *args, **kwargs):
            events.append(["donors"])
            return super().save_donors(*args, **kwargs)

        def save_baseline(self, *args, **kwargs):
            events.append(["baseline"])
            return super().save_baseline(*args, **kwargs)

        def save_unit(self, layer, unit, arrays):
            events.append(["unit", layer, unit])
            return super().save_unit(layer, unit, arrays)

    with patch.object(torch.cuda, "is_available", return_value=False), \
         patch("transformers.AutoConfig.from_pretrained", return_value=model.config), \
         patch("transformers.MistralCommonBackend.from_pretrained", return_value=TinyTokenizer()), \
         patch("transformers.Mistral3ForConditionalGeneration.from_pretrained", return_value=model), \
         patch.object(adapter.sweeping, "load_eval_dataset", return_value=records), \
         patch.object(parallel, "ReplicaCheckpoint", TracedCheckpoint), \
         (patch.object(adapter.MinistralRunner, "_forward_option_logits",
                       side_effect=TypeError("Injected baseline failure")) if fail_baseline else nullcontext()):
        parallel._patch_worker(rank, config, world_size, rendezvous, device_type="cpu")
    (Path(trace_dir) / f"rank_{rank}.json").write_text(json.dumps(events))


def _spawn_cpu_patch(config, world_size, records, directory):
    directory.mkdir()
    context = parallel.mp.spawn(
        _cpu_patch_worker,
        args=(config, world_size, f"file://{directory / 'rendezvous'}", records, str(directory)),
        nprocs=world_size, join=False,
    )
    parallel._wait_for_workers(context)
    return [json.loads((directory / f"rank_{rank}.json").read_text()) for rank in range(world_size)]


@pytest.mark.parametrize("permutations,dtype", [(False, "float32"), (True, "float32"), (True, "bfloat16")])
def test_parallel_patching_matches_serial_and_resumes(cfg, tiny_model, rows, tmp_path, monkeypatch, permutations, dtype):
    cfg["model"]["dtype"] = dtype
    tiny_model.to(dtype=getattr(torch, dtype))
    rows += [dict(question_id="q2", language=lang, question=f"Third question in {lang}?",
                  answer_index=2, **{f"option_{j}": f"Choice {j}" for j in range(4)})
             for lang in ("en", "es")]
    cfg["data"]["allowed_parallel_ids_path"] = None
    cfg["eval"]["max_input_length"] = 128
    cfg["patching"].update(batch_size=5, donor_batch_size=2)
    cfg["patching"]["choice_order_perturbations"]["enabled"] = permutations
    cfg["run"]["output_dir"] = str(tmp_path / "serial")
    monkeypatch.setattr(adapter.sweeping, "load_eval_dataset", lambda cfg: rows)
    serial = adapter.run(cfg, "patch")
    assert tiny_model.lm_head.weight.dtype == getattr(torch, dtype)
    assert serial["baseline"]["option_logits"].dtype == np.float32

    replicated = copy.deepcopy(cfg)
    replicated["run"].update(data_parallel=True, output_dir=str(tmp_path / "replicated"))
    # Four units over three replicas tests uneven ownership; six questions and
    # batches of five also exercise a short final forward batch.
    traces = _spawn_cpu_patch(replicated, 3, rows, tmp_path / "first")
    writes = []
    for rank, events in enumerate(traces):
        for event in events:
            if event[0] in ("donors", "baseline"):
                assert rank == 0
            if event[0] == "unit":
                _, layer, unit = event
                assert (layer * 2 + unit) % 3 == rank
                writes.append((layer, unit))
    assert sorted(writes) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert sum(events.count(["donors"]) for events in traces) == 1
    assert sum(events.count(["baseline"]) for events in traces) == 1
    assert not (tmp_path / "replicated/_SUCCESS").exists()

    # Assembly uses only checkpoint metadata, never a further model load.
    with patch.object(adapter.MinistralRunner, "__init__", side_effect=AssertionError("No model needed")):
        combined = parallel._assemble_results(replicated)
    for section in ("baseline", "importance", "flat_vectors"):
        for metric in serial[section]:
            np.testing.assert_allclose(combined[section][metric], serial[section][metric], atol=1e-5, rtol=1e-5)
    for key in ("metadata", "samples", "spearman", "jaccard", "language_means"):
        assert combined[key] == serial[key]

    # Simulate interruption with one unfinished unit; resume with fewer GPUs.
    units = tmp_path / "replicated/.sweep_checkpoint/units"
    (units / "layer_0000_unit_0001.npz").unlink()
    with pytest.raises(RuntimeError, match="incomplete patch units"):
        parallel._assemble_results(replicated)
    traces = _spawn_cpu_patch(replicated, 2, rows, tmp_path / "resumed")
    writes = [event for events in traces for event in events if event[0] == "unit"]
    assert writes == [["unit", 0, 1]]
    assert not any(event[0] in ("donors", "baseline") for events in traces for event in events)
    assert ["forward"] not in traces[0]
    resumed = parallel._assemble_results(replicated)
    for key in combined["importance"]:
        np.testing.assert_array_equal(resumed["importance"][key], combined["importance"][key])

    # The public entry point publishes only the assembled results.
    with patch.object(parallel, "run_replicated_patch", return_value=resumed):
        adapter.run(replicated, "patch")
    assert (tmp_path / "replicated/_SUCCESS").is_file()
    assert json.loads((tmp_path / "replicated/patch_config.json").read_text())["run"]["data_parallel"]


def _signal_test_worker(rank, ready_dir):
    def stop(*args):
        raise SystemExit(adapter.sweeping.GRACEFUL_STOP_EXIT_CODE)

    signal.signal(signal.SIGUSR1, stop)
    (Path(ready_dir) / str(rank)).touch()
    while True:
        time.sleep(0.05)


def test_setup_failure_exits_instead_of_hanging(cfg, rows, tmp_path, capfd):
    cfg["data"]["allowed_parallel_ids_path"] = None
    cfg["run"]["data_parallel"] = True
    context = parallel.mp.spawn(
        _cpu_patch_worker,
        args=(cfg, 2, f"file://{tmp_path / 'rendezvous'}", rows, str(tmp_path), True),
        nprocs=2, join=False,
    )
    with pytest.raises(parallel.mp.ProcessRaisedException):
        parallel._wait_for_workers(context)
    assert not any(process.is_alive() for process in context.processes)
    assert "Injected baseline failure" in capfd.readouterr().err
    assert (tmp_path / ".sweep_checkpoint/donors.npz").is_file()
    assert not (tmp_path / ".sweep_checkpoint/baseline.npz").exists()
    assert not (tmp_path / "_SUCCESS").exists()


def test_parent_forwards_slurm_stop_and_cleans_up(tmp_path, monkeypatch):
    context = parallel.mp.spawn(_signal_test_worker, args=(str(tmp_path),), nprocs=2, join=False)
    try:
        deadline = time.monotonic() + 30
        while not all((tmp_path / str(rank)).exists() for rank in range(2)):
            assert time.monotonic() < deadline, "Workers failed to start"
            time.sleep(0.05)
        monkeypatch.setattr(adapter.sweeping, "_GRACEFUL_STOP_REQUESTED", True)
        with pytest.raises(adapter.sweeping.GracefulStopRequested):
            parallel._wait_for_workers(context)
        assert not any(process.is_alive() for process in context.processes)
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


def test_replication_requires_checkpoints(cfg):
    cfg["run"]["checkpoint"]["enabled"] = False
    with pytest.raises(ValueError, match="checkpoint.enabled=true"):
        parallel.run_replicated_patch(cfg)
