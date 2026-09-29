"""Recurrent PPO for the robot seat: fixed-opponent curriculum, then saved-policy league.

Stages (configs/gate1.yaml): random opponents -> heuristic opponents -> league of heuristic
opponents and frozen snapshots of the learner. Checkpoints are selected on validation seeds
against heuristic opponents; held-out test seeds and the held-out opponent are never used.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from ..agents.policy import PolicyNet, save_policy
from . import seeds as S
from .collect import collect


@dataclass
class Stage:
    name: str
    opponents: list[str]  # sampled uniformly per seat
    league_prob: float = 0.0  # probability a seat is a frozen snapshot instead
    promote_winrate: float | None = None
    min_iters: int = 0
    max_iters: int | None = None


@dataclass
class TrainConfig:
    run_dir: str = "runs/gate1"
    seed: int = 0
    workers: int = 20
    hands_per_iter: int = 4096
    iterations: int = 300
    lr: float = 3e-4
    epochs: int = 4
    minibatch_episodes: int = 256
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    ent_coef_final: float = 0.002
    max_grad_norm: float = 0.5
    gamma: float = 1.0
    lam: float = 0.95
    shaping: float = 0.5
    dealin_penalty: float = 0.0
    hidden: int = 256
    val_every: int = 5
    val_hands: int = 2000
    val_opponents: list[str] = field(default_factory=lambda: ["heuristic"])
    snapshot_every: int = 10
    league_size: int = 8
    torch_threads: int = 16
    stages: list[Stage] = field(default_factory=list)
    # Fine-tuning from an imitation checkpoint (configs/gate1_v2.yaml).
    init_checkpoint: str | None = None
    kl_ref_coef: float = 0.0  # weight of KL(reference || policy); the reference is the initial policy
    kl_ref_coef_final: float | None = None
    lr_final_frac: float = 0.1
    device: str = "cpu"  # where PPO updates run; rollouts always run on CPU workers
    select_from_stage: int = 1  # first stage whose checkpoints may be selected as best
    train_seed_iteration_offset: int = 0

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        raw = yaml.safe_load(Path(path).read_text())
        raw["stages"] = [Stage(**s) for s in raw.get("stages", [])]
        return cls(**raw)


class OpponentSampler:
    def __init__(self, stage: Stage, league: list[str]):
        self.stage = stage
        self.league = league

    def __call__(self, rng: random.Random) -> list[str]:
        out = []
        for _ in range(3):
            if self.league and rng.random() < self.stage.league_prob:
                out.append("policy:" + rng.choice(self.league))
            else:
                out.append(rng.choice(self.stage.opponents))
        return out


def _worker_init() -> None:
    torch.set_num_threads(1)


def _collect_task(args):
    state, hidden, seeds, sampler, shaping, dealin, greedy, torch_seed = args
    net = PolicyNet(hidden=hidden)
    net.load_state_dict(state)
    net.eval()
    return collect(net, seeds, sampler, shaping, dealin, greedy=greedy, torch_seed=torch_seed)


def parallel_collect(pool, net, seeds, sampler, cfg: TrainConfig, greedy=False, torch_seed=0):
    state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    n = cfg.workers
    chunks = [seeds[i::n] for i in range(n) if seeds[i::n]]
    tasks = [
        (state, cfg.hidden, c, sampler, cfg.shaping, cfg.dealin_penalty, greedy, torch_seed * 1000 + i)
        for i, c in enumerate(chunks)
    ]
    episodes, outcomes = [], []
    for eps, outs in pool.map(_collect_task, tasks):
        episodes += eps
        outcomes += outs
    return episodes, outcomes


def gae(ep: dict, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    r, v = ep["rew"], ep["val"]
    T = len(r)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nv = v[t + 1] if t + 1 < T else 0.0
        delta = r[t] + gamma * nv - v[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + v


def _pad(arrs, T, fill=0):
    out = np.full((len(arrs), T) + arrs[0].shape[1:], fill, dtype=arrs[0].dtype)
    for i, a in enumerate(arrs):
        out[i, : len(a)] = a
    return out


def ppo_update(net, opt, episodes, cfg: TrainConfig, ent_coef: float, ref=None, kl_ref_coef: float = 0.0) -> dict:
    for ep in episodes:
        ep["adv"], ep["ret"] = gae(ep, cfg.gamma, cfg.lam)
    all_adv = np.concatenate([ep["adv"] for ep in episodes])
    mu, sd = all_adv.mean(), all_adv.std() + 1e-8
    stats = {"pi_loss": [], "v_loss": [], "entropy": [], "kl": [], "clipfrac": [], "kl_ref": []}
    dev = next(net.parameters()).device
    idx = np.arange(len(episodes))
    for _ in range(cfg.epochs):
        np.random.shuffle(idx)
        for start in range(0, len(idx), cfg.minibatch_episodes):
            batch = [episodes[i] for i in idx[start : start + cfg.minibatch_episodes]]
            T = max(len(ep["act"]) for ep in batch)
            obs = torch.from_numpy(_pad([ep["obs"] for ep in batch], T)).to(dev)
            mask = torch.from_numpy(_pad([ep["mask"] for ep in batch], T, fill=True)).to(dev)
            act = torch.from_numpy(_pad([ep["act"] for ep in batch], T)).to(dev)
            old = torch.from_numpy(_pad([ep["logp"] for ep in batch], T)).to(dev)
            adv = torch.from_numpy(_pad([(ep["adv"] - mu) / sd for ep in batch], T)).to(dev)
            ret = torch.from_numpy(_pad([ep["ret"] for ep in batch], T)).to(dev)
            valid = torch.from_numpy(_pad([np.ones(len(ep["act"]), dtype=np.float32) for ep in batch], T)).to(dev)
            logits, values, _ = net(obs, mask, net.initial_state(len(batch)).to(dev))
            logp_all = torch.log_softmax(logits, -1)
            logp = logp_all.gather(-1, act[..., None])[..., 0]
            ratio = torch.exp(logp - old)
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv
            n = valid.sum()
            pi_loss = -(torch.min(s1, s2) * valid).sum() / n
            v_loss = (((values - ret) ** 2) * valid).sum() / n
            p = logp_all.exp()
            ent = -((p * logp_all.clamp(min=-30)).sum(-1) * valid).sum() / n
            loss = pi_loss + cfg.vf_coef * v_loss - ent_coef * ent
            kl_ref = torch.zeros((), device=dev)
            if ref is not None and kl_ref_coef > 0:
                with torch.no_grad():
                    ref_logits, _, _ = ref(obs, mask, ref.initial_state(len(batch)).to(dev))
                    ref_logp = torch.log_softmax(ref_logits, -1)
                ref_p = ref_logp.exp()
                kl_ref = ((ref_p * (ref_logp.clamp(min=-30) - logp_all.clamp(min=-30))).sum(-1) * valid).sum() / n
                loss = loss + kl_ref_coef * kl_ref
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                stats["pi_loss"].append(float(pi_loss))
                stats["v_loss"].append(float(v_loss))
                stats["entropy"].append(float(ent))
                stats["kl"].append(float((((old - logp) * valid).sum() / n)))
                stats["clipfrac"].append(float(((ratio - 1).abs() > cfg.clip).float().mul(valid).sum() / n))
                stats["kl_ref"].append(float(kl_ref))
    return {k: float(np.mean(v)) for k, v in stats.items()}


def summarize(outcomes: list[dict]) -> dict:
    n = max(1, len(outcomes))
    return {
        "hands": len(outcomes),
        "win_rate": sum(o["win"] for o in outcomes) / n,
        "dealin_rate": sum(o["dealin"] for o in outcomes) / n,
        "exhausted_rate": sum(o["exhausted"] for o in outcomes) / n,
        "self_draw_rate": sum(o["win_type"] == "self_draw" for o in outcomes) / n,
        "mean_steps": sum(o["steps"] for o in outcomes) / n,
    }


def _provenance() -> dict:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best effort
        rev = ""
    return {"git_rev": rev or None, "torch": torch.__version__, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}


def train(cfg: TrainConfig) -> Path:
    run = Path(cfg.run_dir)
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run / "league").mkdir(exist_ok=True)
    cfg_dict = asdict(cfg)
    (run / "config.yaml").write_text(yaml.safe_dump(cfg_dict, sort_keys=False))
    (run / "provenance.json").write_text(json.dumps(_provenance(), indent=2))
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_num_threads(cfg.torch_threads)
    net = PolicyNet(hidden=cfg.hidden)
    ref = None
    if cfg.init_checkpoint:
        from ..agents.policy import load_policy

        net.load_state_dict(load_policy(cfg.init_checkpoint).state_dict())
        ref = load_policy(cfg.init_checkpoint).to(cfg.device)
        for p_ in ref.parameters():
            p_.requires_grad_(False)
    net.to(cfg.device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    log = (run / "log.jsonl").open("a")
    league: list[str] = []
    stage_i, stage_start = 0, 0
    recent: list[float] = []
    best = {"val_win_rate": -1.0, "iteration": None}
    val_sampler = OpponentSampler(Stage("val", cfg.val_opponents), [])
    ctx = mp.get_context("fork")
    with ctx.Pool(cfg.workers, initializer=_worker_init) as pool:
        if cfg.init_checkpoint:  # score the starting policy on the same validation hands
            net.eval()
            _, vout = parallel_collect(pool, net, S.val_seeds(cfg.val_hands), val_sampler, cfg, greedy=True)
            vs = summarize(vout)
            best = {"val_win_rate": vs["win_rate"], "iteration": -1, "stage": "initial"}
            save_policy(net, run / "checkpoints" / "best.pt", {"iteration": -1, "val": vs})
            (run / "best.json").write_text(json.dumps(best, indent=2))
            log.write(json.dumps({"iteration": -1, "stage": "initial", "val_win_rate": vs["win_rate"],
                                  "val_dealin_rate": vs["dealin_rate"]}) + "\n")
            log.flush()
            print(f"initial policy: val_win_rate={vs['win_rate']:.4f} val_dealin_rate={vs['dealin_rate']:.4f}", flush=True)
        for it in range(cfg.iterations):
            t0 = time.time()
            stage = cfg.stages[stage_i]
            frac = it / max(1, cfg.iterations - 1)
            for g in opt.param_groups:
                g["lr"] = cfg.lr * (1.0 - (1.0 - cfg.lr_final_frac) * frac)
            ent_coef = cfg.ent_coef + (cfg.ent_coef_final - cfg.ent_coef) * frac
            kl_final = cfg.kl_ref_coef if cfg.kl_ref_coef_final is None else cfg.kl_ref_coef_final
            kl_ref_coef = cfg.kl_ref_coef + (kl_final - cfg.kl_ref_coef) * frac
            sampler = OpponentSampler(stage, league)
            net.eval()
            episodes, outcomes = parallel_collect(
                pool, net, S.train_seeds(it + cfg.train_seed_iteration_offset, cfg.hands_per_iter), sampler, cfg,
                torch_seed=cfg.seed * 100_003 + it
            )
            t1 = time.time()
            net.train()
            upd = ppo_update(net, opt, episodes, cfg, ent_coef, ref, kl_ref_coef)
            summ = summarize(outcomes)
            recent = (recent + [summ["win_rate"]])[-3:]
            row = {"iteration": it, "stage": stage.name, **summ, **upd,
                   "collect_s": round(t1 - t0, 1), "update_s": round(time.time() - t1, 1)}

            save_policy(net, run / "checkpoints" / "latest.pt", {"iteration": it, "stage": stage.name})
            if stage.league_prob > 0 or stage_i == len(cfg.stages) - 1:
                if it % cfg.snapshot_every == 0 or not league:
                    path = run / "league" / f"iter_{it:04d}.pt"
                    save_policy(net, path, {"iteration": it})
                    league = (league + [str(path)])[-cfg.league_size:]

            if (it + 1) % cfg.val_every == 0 or it == cfg.iterations - 1:
                net.eval()
                _, vout = parallel_collect(pool, net, S.val_seeds(cfg.val_hands), val_sampler, cfg, greedy=True)
                vs = summarize(vout)
                row["val_win_rate"] = vs["win_rate"]
                row["val_dealin_rate"] = vs["dealin_rate"]
                ck = run / "checkpoints" / f"iter_{it:04d}.pt"
                save_policy(net, ck, {"iteration": it, "stage": stage.name, "val": vs})
                if vs["win_rate"] > best["val_win_rate"] and stage_i >= cfg.select_from_stage:
                    best = {"val_win_rate": vs["win_rate"], "iteration": it, "stage": stage.name}
                    save_policy(net, run / "checkpoints" / "best.pt", {"iteration": it, "val": vs})
                    (run / "best.json").write_text(json.dumps(best, indent=2))

            since = it - stage_start + 1
            promote = stage_i < len(cfg.stages) - 1 and (
                (stage.promote_winrate is not None and since >= stage.min_iters and len(recent) == 3
                 and min(recent) >= stage.promote_winrate)
                or (stage.max_iters is not None and since >= stage.max_iters)
            )
            if promote:
                stage_i, stage_start, recent = stage_i + 1, it + 1, []
                row["promoted_to"] = cfg.stages[stage_i].name
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                  flush=True)
    return run
