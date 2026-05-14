import os
import socket
from datetime import timedelta

import torch
import torch.nn as nn
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device(f"cuda:0")

    print(f"[Rank {rank}] Starting...")
    print(f"[Rank {rank}] Hostname: {socket.gethostname()}")

    dist.init_process_group(
        backend="nccl",
        init_method="tcp://10.50.0.57:29500",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )

    torch.cuda.set_device(device)

    # =========================
    # Rank 0
    # =========================
    if rank == 0:

        model_part1 = nn.Linear(4, 8).to(device)

        x = torch.randn(1, 4).to(device)

        print(f"\n[Rank 0] Input:")
        print(x)

        hidden = model_part1(x)

        print(f"\n[Rank 0] Hidden:")
        print(hidden)

        # shape
        shape_tensor = torch.tensor(hidden.shape,
                                    dtype=torch.long,
                                    device=device)

        dist.send(shape_tensor, dst=1)

        dist.send(hidden, dst=1)

        print("\n[Rank 0] Hidden sent")

    # =========================
    # Rank 1
    # =========================
    elif rank == 1:

        recv_shape = torch.zeros(2,
                                 dtype=torch.long,
                                 device=device)

        dist.recv(recv_shape, src=0)

        hidden_shape = tuple(recv_shape.tolist())

        hidden = torch.zeros(hidden_shape,
                             device=device)

        dist.recv(hidden, src=0)

        print(f"\n[Rank 1] Hidden received:")
        print(hidden)

        model_part2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(8, 2)
        ).to(device)

        output = model_part2(hidden)

        print(f"\n[Rank 1] Output:")
        print(output)

    dist.barrier()

    print(f"\n[Rank {rank}] SUCCESS")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()