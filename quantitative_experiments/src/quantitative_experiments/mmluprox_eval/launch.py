from __future__ import annotations

import argparse
import os
from typing import List

import torch
import torch.multiprocessing as mp

from .run import main as run_main


def _worker(rank: int, world_size: int, master_port: int, argv: List[str]) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)

    # Rebuild argv for the child process
    os.sys.argv = argv
    run_main()


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch MMLU-Prox eval on all GPUs")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--nproc", type=int, default=None, help="Number of GPUs to use")
    parser.add_argument("--master_port", type=int, default=29500)
    args, unknown = parser.parse_known_args()
    print(args)
    world_size = args.nproc or torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPUs available. Use quantitative_experiments.mmluprox_eval.run for CPU.")

    # Recreate argv for child processes
    argv = ["quantitative_experiments.mmluprox_eval.run"]
    for c in args.config:
        argv += ["--config", c]
    for s in args.set:
        argv += ["--set", s]
    argv += unknown

    mp.spawn(_worker, args=(world_size, args.master_port, argv), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
