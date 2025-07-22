import torch
import torch.distributed as dist
import time
import argparse
import os

from vtimeline import vinit, CUPTI, TracePoint


RANK = None
LOCAL_RANK = None
WORLD_SIZE = None
PROCESS_GROUP = None


def setup_distributed():
    global RANK, LOCAL_RANK, WORLD_SIZE, PROCESS_GROUP

    RANK = int(os.environ["RANK"])
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(LOCAL_RANK)

    dist.init_process_group(backend="nccl")

    PROCESS_GROUP = dist.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")


def create_dummy_computation(tensor_size, compute_steps):
    def compute_task():
        device = torch.cuda.current_device()
        temp_tensor = torch.randn(tensor_size, device=device)

        for _ in range(compute_steps):
            tp = TracePoint("compute", "compute", stream=torch.cuda.current_stream())
            tp.begin()
            temp_tensor = torch.mm(temp_tensor, temp_tensor.t())
            temp_tensor = torch.relu(temp_tensor)
            temp_tensor = temp_tensor / (torch.norm(temp_tensor) + 1e-8)
            tp.end()

        return temp_tensor

    return compute_task


def create_all_gather(tensor_size, comm_steps, async_op):
    def all_gather_async_task():
        device = torch.cuda.current_device()
        local_tensor = torch.randn(tensor_size, device=device) * (RANK + 1)
        gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(WORLD_SIZE)]

        handlers = []

        for _ in range(comm_steps):
            tp = TracePoint("all_gather_async", "all_gather", stream=PROCESS_GROUP)
            tp.begin()
            handlers.append(
                dist.all_gather(
                    gathered_tensors, local_tensor, group=PROCESS_GROUP, async_op=True
                )
            )
            tp.end()

        return handlers

    def all_gather_task():
        device = torch.cuda.current_device()
        local_tensor = torch.randn(tensor_size, device=device) * (RANK + 1)
        gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(WORLD_SIZE)]

        for _ in range(comm_steps):
            tp = TracePoint("all_gather_sync", "all_gather", stream=PROCESS_GROUP)
            tp.begin()
            dist.all_gather(gathered_tensors, local_tensor, async_op=False)
            tp.end()

    if async_op:
        return all_gather_async_task
    return all_gather_task


def overlap_allgather_compute(
    compute_tensor_size, comm_tensor_size, compute_steps, comm_steps
):
    compute_task = create_dummy_computation(compute_tensor_size, compute_steps)
    comm_async_task = create_all_gather(comm_tensor_size, comm_steps, True)
    comm_sync_task = create_all_gather(comm_tensor_size, comm_steps, False)

    dist.barrier()
    tp = TracePoint("overlap", "OVERLAP")
    tp.begin()
    start_time = time.time()

    handlers = comm_async_task()
    compute_task()

    for handler in handlers:
        handler.wait()

    overlap_time = time.time() - start_time
    torch.cuda.synchronize()
    tp.end()

    if RANK == 0:
        print(f"Overlap method completed in: {overlap_time:.4f} seconds")

    dist.barrier()
    tp = TracePoint("sequential", "SEQUENTIAL")
    tp.begin()
    start_time = time.time()

    comm_sync_task()
    compute_task()

    torch.cuda.synchronize()
    tp.end()
    sequential_time = time.time() - start_time
    dist.barrier()

    if RANK == 0:
        print(f"Sequential method completed in: {sequential_time:.4f} seconds")
        speedup = sequential_time / overlap_time
        print(f"Speedup: {speedup:.2f}x")

    return overlap_time, sequential_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overlap AllGather and Compute Example"
    )
    parser.add_argument(
        "--compute-tensor-size",
        type=int,
        default=4096,
        help="Size of the square tensor",
    )
    parser.add_argument(
        "--comm-tensor-size",
        type=int,
        default=8192,
        help="Size of the square tensor",
    )
    parser.add_argument(
        "--compute-steps",
        type=int,
        default=500,
        help="Number of computation steps",
    )
    parser.add_argument(
        "--comm-steps",
        type=int,
        default=50,
        help="Number of communication steps",
    )

    args = parser.parse_args()

    os.environ["CUPTI_HOME"] = os.path.dirname(__file__)
    vinit()

    setup_distributed()

    print(f"Initialized rank {RANK}/{WORLD_SIZE} on device {LOCAL_RANK}")

    compute_tensor_size = (args.compute_tensor_size, args.compute_tensor_size)
    comm_tensor_size = (args.comm_tensor_size, args.comm_tensor_size)

    try:
        for _ in range(0, 10):
            CUPTI.step()
            overlap_time, sequential_time = overlap_allgather_compute(
                compute_tensor_size,
                comm_tensor_size,
                args.compute_steps,
                args.comm_steps,
            )

        if RANK == 0:
            print(f"\n{'=' * 50}")
            print("Performance Summary:")
            print(f"Overlap time: {overlap_time:.4f}s")
            print(f"Sequential time: {sequential_time:.4f}s")
            print(f"Speedup: {sequential_time / overlap_time:.2f}x")
            print(f"{'=' * 50}")

    except Exception as e:
        print(f"Rank {RANK}: Error occurred: {e}")
        raise
    finally:
        dist.destroy_process_group()
