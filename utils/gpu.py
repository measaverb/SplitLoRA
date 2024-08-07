import argparse
import os

import torch
import torch.distributed as dist


def add_gpu_params(parser: argparse.ArgumentParser):
    parser.add_argument("--platform", default="k8s", type=str, help="platform cloud")
    parser.add_argument("--local_rank", default=0, type=int, help="local rank")
    parser.add_argument("--rank", default=0, type=int, help="rank")
    parser.add_argument("--device", default=0, type=int, help="device")
    parser.add_argument("--world_size", default=0, type=int, help="world size")
    parser.add_argument("--random_seed", default=10, type=int, help="random seed")


def distributed_opt(args, model, opt, grad_acc=1):
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=False,
        broadcast_buffers=False,
    )
    return model, opt


def distributed_gather(args, tensor):
    g_y = [torch.zeros_like(tensor) for _ in range(args.world_size)]
    torch.distributed.all_gather(g_y, tensor, async_op=False)
    return torch.stack(g_y)


def distributed_sync(args):
    if args.platform == "azure":
        args.hvd.allreduce(torch.tensor(0), name="barrier")
    else:
        args.dist.barrier()


def parse_gpu(args):
    torch.manual_seed(args.random_seed)

    if args.platform == "local":
        dist.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        args.rank = local_rank
        args.device = device
        args.world_size = torch.distributed.get_world_size()
        args.dist = dist
    elif args.platform == "k8s":
        master_uri = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        args.local_rank = local_rank
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        world_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        rank = world_rank
        torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend="nccl",
            init_method=master_uri,
            world_size=world_size,
            rank=world_rank,
        )
        device = torch.device("cuda", local_rank)
        args.rank = rank
        args.device = device
        args.world_size = world_size
        args.dist = dist
    print(
        "myrank:",
        args.rank,
        "local_rank:",
        args.local_rank,
        "device_count:",
        torch.cuda.device_count(),
        "world_size:",
        args.world_size,
    )


def cleanup(args):
    if args.platform == "k8s" or args.platform == "philly":
        args.dist.destroy_process_group()
