# Parallel donor ablations

This separate runner uses **one complete model per GPU**, with an independent
Python process and patch hooks for each copy. Four GPUs patch four different
attention groups concurrently. It imports the existing review and original
helpers without editing them. The serial runner, its configs, and its saved
outputs remain usable.

## Submit

From the `thesis3` repository root:

```bash
mkdir -p quantitative_experiments/review/logs
sbatch quantitative_experiments/review/run_donor_parallel.sbatch \
  qwen_mmluprox --samples-per-language 100
```

The condition can also be `qwen_belebele`, `llama_mmluprox`, or `llama_belebele`.
The launcher activates `thesis` and requests four GPUs, 16 CPUs, 180 GB host RAM,
and 48 hours, using the same partition/QoS and `gpu-05` exclusion as the current
serial launcher. It logs the node, visible GPU IDs, precision, and effective
batch sizes. No existing job is cancelled by submitting this script.

Defaults are **FP32**, four workers, **batch size 4 per GPU**, and donor batch
size 4. These override the serial YAML's precision and batch sizes: a serial
batch of 64 must not be silently carried over to a full model on one GPU.
The default output is:

```text
quantitative_experiments/review/runs/qwen_mmluprox_parallel_float32/
```

All four workers share the same 100 recipient question IDs, four answer
orderings, donor choices, and eight donor variants. Work is split over
`(variant, layer, group)` units; question count is **not divided by four**.
Each unit still evaluates all 2,000 recipient prompts for the default setup.

## Flow and monitoring

1. The coordinator freezes the protocol and holds the run's writer lock.
2. A preparation process on the first GPU captures the shared donor activations
   and computes clean predictions. **Only one GPU is active in this stage.**
   It exits before the patch workers load, releasing its model memory.
3. Four fresh processes each see exactly one allocated GPU and load one model.
   They receive disjoint round-robin lists of patch units. There is no model
   sharding, cross-GPU activation transfer, `DataParallel`, or gradient sync.
4. The coordinator validates all unit checkpoints, exports the usual variant
   tensors, and runs the existing CPU statistical analysis.

Check progress with:

```bash
tail -f quantitative_experiments/review/logs/donor_parallel_JOB_ID.out
tail -f quantitative_experiments/review/runs/qwen_mmluprox_parallel_float32/worker_logs/worker_0.log
```

Workers 1, 2, and 3 have their own logs beside `worker_0.log`. Worker logs include
seconds per completed unit. During patching the coordinator prints combined
progress every 30 seconds and an approximate remaining time after enough units
complete. This estimate includes worker startup and should stabilize with time.
Setup errors appear in the job `.err`; patch-worker tracebacks are in their
individual logs. If one worker fails, the coordinator stops the other workers
and retains their completed checkpoints.

## Short throughput check

Run a small number of real patch units before committing to the full sweep:

```bash
sbatch quantitative_experiments/review/run_donor_parallel.sbatch \
  qwen_mmluprox --samples-per-language 100 \
  --benchmark-units-per-worker 4
```

This still performs shared donor capture and clean prediction on first use.
It then computes at most four new units per worker (16 total) and exits
successfully without final exports/analysis unless all units are already done.
Submit the same command **without** `--benchmark-units-per-worker` to resume and
finish. The benchmark's completed units are reused. Actual GPU speedup and peak
memory must be measured on the allocated node; CPU correctness tests do not
establish a runtime improvement.

## Batch sizes, precision, and resumption

Use the parallel runner's explicit batch options:

```bash
sbatch quantitative_experiments/review/run_donor_parallel.sbatch \
  qwen_mmluprox --samples-per-language 100 \
  --batch-size 2 --donor-batch-size 2 \
  --output-dir quantitative_experiments/review/runs/qwen_mmluprox_parallel_fp32_bs2
```

An 8B FP32 model needs roughly 32 GB for weights alone. Batch 4 is a conservative
starting point for a 40 GB A100, not a guarantee for every prompt/model. If a
worker runs out of memory, reduce both batch sizes in a **new output directory**.
There is no automatic precision change or CPU offload.

Optional BF16 uses less weight memory but changes numerical precision:

```bash
sbatch quantitative_experiments/review/run_donor_parallel.sbatch \
  qwen_mmluprox --samples-per-language 100 \
  --dtype bfloat16 --batch-size 8 --donor-batch-size 8
```

That defaults to `runs/qwen_mmluprox_parallel_bfloat16`; donors, clean predictions,
and patched logits are all recomputed in that precision. The output metadata
records the model dtype. Do not combine FP32 and BF16 scores as one experiment.

Repeating an unchanged command resumes atomic per-unit checkpoints. Completed
donor capture and clean predictions are reused; incomplete donor capture restarts.
An interrupted patch unit repeats. A different batch size, precision, scientific
configuration, or implementation requires a new output directory. Existing serial
checkpoints are not imported. Worker count and the benchmark limit are scheduling
options and can change when resuming a stopped parallel run. Only one coordinator
may write a given run directory at a time.

For fewer GPUs, change both allocation and worker count, for example:

```bash
sbatch --gres=gpu:2 quantitative_experiments/review/run_donor_parallel.sbatch \
  qwen_mmluprox --workers 2
```

The launcher retains Slurm's GPU selection for each worker, including UUID-based
visibility. It does not hard-code physical GPUs 0–3. `--requeue` allows scheduler
requeues; it does not automatically extend the 48-hour time limit. Resubmit the
same command after a time-limit interruption. TERM/INT stop the child processes.

## Local inspection and validation

No allocation or model loading is needed to print the effective configuration:

```bash
conda activate thesis
bash quantitative_experiments/review/run_donor_parallel.sh run \
  --config quantitative_experiments/review/configs/qwen_mmluprox.yaml --dry-run
```

`--prepare-only` saves the recipient/donor protocol without loading a model.
Other scientific settings can be passed with `--set`, as in the serial runner;
the explicit parallel precision/batch/device controls take precedence over
conflicting `--set` values. CPU analysis can be repeated with the existing
`run_donor_ablation.sh analyze --run-dir PARALLEL_OUTPUT` command.

CPU checks cover concurrent processes, exact agreement of all exported arrays
with the serial engine, analysis compatibility, partial-run resumption with a
different worker count, checkpoint/model mismatches, GPU masking, process cleanup,
and real miniature Qwen/Llama patch hooks:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest discover -s quantitative_experiments/review/tests -v
```
