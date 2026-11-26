from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    enabled: bool
    world_size: int
    rank: int
    local_rank: int


def init_distributed(backend: str = "nccl", master_port: Optional[int] = None) -> DistInfo:
    if dist.is_available() and dist.is_initialized():
        return DistInfo(True, dist.get_world_size(), dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0)))

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return DistInfo(False, 1, 0, 0)

    if master_port is not None:
        os.environ.setdefault("MASTER_PORT", str(master_port))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    return DistInfo(True, world_size, rank, local_rank)


# def barrier() -> None:
#     if dist.is_available() and dist.is_initialized():
#         dist.barrier()
def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def destroy_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def all_gather_object(obj):
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]
    output = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(output, obj)
    return output


def gather_object_to_rank0(obj):
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    output = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(obj, output, dst=0)
    return output
