"""Imitation warm start: copy the training heuristic before PPO fine-tuning.

Tables of four training heuristics are played on training seeds, and each recorded seat's
decisions form one episode (observation, legal mask, the heuristic's action). The policy is
fitted with cross-entropy over the masked logits. Its value head is fitted to the same return
PPO uses (terminal win reward minus the deal-in penalty, plus the telescoped shaping term),
so fine-tuning starts from a critic that already fits the imitated policy. The held-out
heuristic is never used here.
"""

from __future__ import annotations

import multiprocessing as mp
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..agents.baselines import HeuristicPlayer
from ..agents.features import encode, potential
from ..agents.policy import PolicyNet, save_policy
from ..game.engine import Engine

IMITATION_SEED_BASE = 500_000_000  # inside the training range (< 1e9), away from PPO's seeds


def _record(args):
    seeds, shaping, dealin = args
    torch.set_num_threads(1)
    eps = []
    for seed in seeds:
        e = Engine(seed)
        players = [HeuristicPlayer() for _ in range(4)]
        for s, p in enumerate(players):
            p.reset(s, seed * 4 + s)
        keep = random.Random(seed).sample(range(4), 2)
        buf = {s: {"obs": [], "mask": [], "act": [], "pot": []} for s in keep}
        while not e.done:
            s = e.to_act
            v = e.view(s)
            a = players[s].act(v)
            if s in buf:
                o, m = encode(v)
                b = buf[s]
                b["obs"].append(o.astype(np.float16))
                b["mask"].append(m)
                b["act"].append(a)
                b["pot"].append(potential(v))
            e.step(a)
        r = e.result
        for s, b in buf.items():
            if not b["act"]:
                continue
            final = (1.0 if r.winner == s else 0.0) - (dealin if (r.win_type == "discard" and r.from_seat == s) else 0.0)
            # Return from each step with gamma = 1: telescoped shaping plus the terminal reward.
            ret = np.array([shaping * (0.0 - p) + final for p in b["pot"]], dtype=np.float32)
            eps.append({"obs": np.stack(b["obs"]), "mask": np.stack(b["mask"]), "act": np.array(b["act"], np.int64),
                        "ret": ret})
    return eps


def record(n_hands: int, shaping: float, dealin: float, processes: int = 20) -> list[dict]:
    seeds = list(range(IMITATION_SEED_BASE, IMITATION_SEED_BASE + n_hands))
    chunks = [seeds[i::processes] for i in range(processes)]
    with mp.get_context("fork").Pool(processes) as pool:
        parts = pool.map(_record, [(c, shaping, dealin) for c in chunks])
    return [e for p in parts for e in p]


def _pad(arrs, T, fill=0):
    out = np.full((len(arrs), T) + arrs[0].shape[1:], fill, dtype=arrs[0].dtype)
    for i, a in enumerate(arrs):
        out[i, : len(a)] = a
    return out


def fit(episodes: list[dict], out: Path, epochs: int = 6, hidden: int = 256, device: str | None = None,
        batch: int = 512, seed: int = 0) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(episodes))
    n_val = max(1, len(episodes) // 50)
    val, tr = [episodes[i] for i in idx[:n_val]], [episodes[i] for i in idx[n_val:]]
    net = PolicyNet(hidden=hidden).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    steps = epochs * ((len(tr) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)

    def tensors(eps):
        T = max(len(e["act"]) for e in eps)
        obs = torch.from_numpy(_pad([e["obs"] for e in eps], T).astype(np.float32)).to(device)
        mask = torch.from_numpy(_pad([e["mask"] for e in eps], T, fill=True)).to(device)
        act = torch.from_numpy(_pad([e["act"] for e in eps], T)).to(device)
        ret = torch.from_numpy(_pad([e["ret"] for e in eps], T)).to(device)
        valid = torch.from_numpy(_pad([np.ones(len(e["act"]), np.float32) for e in eps], T)).to(device)
        return obs, mask, act, ret, valid

    def losses(eps):
        obs, mask, act, ret, valid = tensors(eps)
        logits, values, _ = net(obs, mask, net.initial_state(len(eps)).to(device))
        ce = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), act.reshape(-1), reduction="none")
        n = valid.sum()
        pi = (ce * valid.reshape(-1)).sum() / n
        v = (((values - ret) ** 2) * valid).sum() / n
        agree = (((logits.argmax(-1) == act).float() * valid).sum() / n)
        return pi, v, agree

    hist = []
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(tr))
        for i in range(0, len(order), batch):
            pi, v, _ = losses([tr[j] for j in order[i : i + batch]])
            loss = pi + 0.5 * v
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            parts = [losses(val[i : i + batch]) for i in range(0, len(val), batch)]
        row = {"epoch": ep, "val_ce": float(np.mean([p[0].item() for p in parts])),
               "val_v_mse": float(np.mean([p[1].item() for p in parts])),
               "val_agreement": float(np.mean([p[2].item() for p in parts]))}
        hist.append(row)
        print(row, flush=True)
    net = net.cpu()
    save_policy(net, out, {"imitation": hist[-1], "episodes": len(episodes)})
    return {"history": hist, "episodes": len(episodes)}
