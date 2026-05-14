import os
import socket
from datetime import timedelta

import torch
import torch.nn as nn
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"[Rank {rank}] Starting...")
    print(f"[Rank {rank}] Hostname: {socket.gethostname()}")

    dist.init_process_group(
        backend="gloo",
        init_method="tcp://10.50.0.57:29500",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )

    # =========================
    # Rank 0 : first stage
    # =========================
    if rank == 0:

        model_part1 = nn.Linear(4, 8)

        x = torch.randn(1, 4)

        print(f"\n[Rank 0] Input:")
        print(x)

        hidden = model_part1(x)

        print(f"\n[Rank 0] Hidden:")
        print(hidden)

        # ?? shape
        shape_tensor = torch.tensor(hidden.shape, dtype=torch.long)
        dist.send(shape_tensor, dst=1)

        # ?? activation
        dist.send(hidden, dst=1)

        print("\n[Rank 0] Hidden states sent to Rank 1")

    # =========================
    # Rank 1 : second stage
    # =========================
    elif rank == 1:

        # ?? shape
        recv_shape = torch.zeros(2, dtype=torch.long)
        dist.recv(recv_shape, src=0)

        hidden_shape = tuple(recv_shape.tolist())

        # ?? buffer
        hidden = torch.zeros(hidden_shape)

        # ?? activation
        dist.recv(hidden, src=0)

        print(f"\n[Rank 1] Hidden received:")
        print(hidden)

        model_part2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(8, 2)
        )

        output = model_part2(hidden)

        print(f"\n[Rank 1] Final Output:")
        print(output)

    dist.barrier()

    print(f"\n[Rank {rank}] SUCCESS")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()