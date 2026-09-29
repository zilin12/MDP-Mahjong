"""Legal-action-masked recurrent actor-critic and a Player wrapper around it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..game import actions as A
from ..game.engine import PlayerView
from .baselines import Player
from .features import OBS_DIM, encode

NEG = -1e9


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = A.N_ACTIONS, hidden: int = 256):
        super().__init__()
        self.hidden = hidden
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ReLU(), nn.Linear(512, hidden), nn.ReLU()
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.pi = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))
        self.v = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def initial_state(self, batch: int) -> torch.Tensor:
        return torch.zeros(1, batch, self.hidden)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor, h: torch.Tensor):
        """obs [B,T,D], mask [B,T,A] bool, h [1,B,H] -> masked logits [B,T,A], values [B,T], h."""
        z = self.encoder(obs)
        y, h = self.gru(z, h)
        feat = torch.cat([z, y], dim=-1)
        logits = self.pi(feat).masked_fill(~mask, NEG)
        return logits, self.v(feat).squeeze(-1), h

    @torch.no_grad()
    def step(self, obs: np.ndarray, mask: np.ndarray, h: torch.Tensor):
        """One decision for a batch: obs [B,D], mask [B,A] -> logits [B,A], values [B], h."""
        logits, v, h = self(torch.from_numpy(obs)[:, None], torch.from_numpy(mask)[:, None], h)
        return logits[:, 0], v[:, 0], h


def save_policy(net: PolicyNet, path: Path, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    torch.save({"state_dict": state, "hidden": net.hidden, "meta": meta or {}}, path)


def load_policy(path: str | Path) -> PolicyNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = PolicyNet(hidden=ckpt["hidden"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net


class PolicyPlayer(Player):
    """Plays one seat with a trained policy; greedy by default (evaluation mode)."""

    name = "policy"

    def __init__(self, net: PolicyNet | str | Path, greedy: bool = True):
        self.net = load_policy(net) if isinstance(net, (str, Path)) else net
        self.greedy = greedy

    def reset(self, seat: int, seed: int) -> None:
        super().reset(seat, seed)
        self.h = self.net.initial_state(1)
        self.gen = torch.Generator().manual_seed(seed)

    def act(self, view: PlayerView) -> int:
        obs, mask = encode(view)
        logits, _, self.h = self.net.step(obs[None], mask[None], self.h)
        if self.greedy:
            return int(logits[0].argmax())
        return int(torch.multinomial(torch.softmax(logits[0], -1), 1, generator=self.gen))
