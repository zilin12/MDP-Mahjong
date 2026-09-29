"""Render the learned contact policy's training data in parallel (simulation ground-truth labels)."""

import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np


def job(seed):
    import torch

    torch.set_num_threads(1)
    from mahjong_robot.sim.learned import generate_contact_data

    X, S, H, Y = generate_contact_data(seed, n_scenes=int(sys.argv[2]), per_scene=24)
    return X, S, H, Y


if __name__ == "__main__":
    out = Path(sys.argv[1])
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    with mp.get_context("spawn").Pool(procs) as pool:
        parts = pool.map(job, [1000 + i for i in range(procs)])
    cat = [np.concatenate([p[k] for p in parts]) for k in range(4)]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=cat[0], S=cat[1], H=cat[2], Y=cat[3])
    print("samples", len(cat[0]))
