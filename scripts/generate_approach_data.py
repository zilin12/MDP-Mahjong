"""Planner demonstrations for the learned-only approach policy (ApproachNet)."""

import multiprocessing as mp
import sys

import numpy as np


def job(seed):
    import torch

    torch.set_num_threads(1)
    from mahjong_robot.sim.learned import generate_approach_data

    return generate_approach_data(seed, n_paths=int(sys.argv[2]))


if __name__ == "__main__":
    procs = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    with mp.get_context("spawn").Pool(procs) as pool:
        parts = pool.map(job, [5000 + i for i in range(procs)])
    np.savez(sys.argv[1], AX=np.concatenate([p[0] for p in parts]), AY=np.concatenate([p[1] for p in parts]))
    print("steps", sum(len(p[0]) for p in parts))
