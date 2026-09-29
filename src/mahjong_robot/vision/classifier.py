"""Tile-type classifier on rectified 40 x 52 face crops.

A small CNN trained on rendered crops with photometric and geometric augmentation. Face-up
crops have a 180-degree ambiguity (PCA gives the axis, not the direction), so training
rotates them by 180 degrees at random. ``TileClassifier.predict`` returns the type and the
softmax confidence; callers treat low confidence as uncertain identity (a fault, not a guess).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..game.tiles import N_TYPES

MEAN, STD = 150.0, 60.0


class TileNet(nn.Module):
    def __init__(self):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())

        self.features = nn.Sequential(block(3, 32), nn.MaxPool2d(2), block(32, 64), nn.MaxPool2d(2),
                                      block(64, 128), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, N_TYPES))

    def forward(self, x):
        return self.head(self.features(x))


def _to_tensor(X: np.ndarray) -> torch.Tensor:
    return (torch.from_numpy(np.ascontiguousarray(X)).float().permute(0, 3, 1, 2) - MEAN) / STD


def _augment(x: torch.Tensor, view: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    n = x.shape[0]
    dev = x.device
    flip = (torch.rand(n, generator=gen, device=dev) < 0.5) & (view == 0)
    x = torch.where(flip[:, None, None, None], torch.flip(x, dims=(2, 3)), x)
    gain = 1 + 0.25 * (torch.rand(n, 1, 1, 1, generator=gen, device=dev) - 0.5) * 2
    bias = 0.3 * (torch.rand(n, 1, 1, 1, generator=gen, device=dev) - 0.5) * 2
    tint = 1 + 0.08 * (torch.rand(n, 3, 1, 1, generator=gen, device=dev) - 0.5) * 2
    x = x * gain * tint + bias
    x = x + 0.08 * torch.randn(x.shape, generator=gen, device=dev)
    # Small random translations (up to 3 px).
    theta = torch.zeros(n, 2, 3, device=dev)
    ang = (torch.rand(n, generator=gen, device=dev) - 0.5) * 2 * 0.08
    theta[:, 0, 0] = torch.cos(ang)
    theta[:, 0, 1] = -torch.sin(ang)
    theta[:, 1, 0] = torch.sin(ang)
    theta[:, 1, 1] = torch.cos(ang)
    theta[:, :, 2] = (torch.rand(n, 2, generator=gen, device=dev) - 0.5) * 2 * 0.12
    scale = 1 + (torch.rand(n, 1, 1, generator=gen, device=dev) - 0.5) * 0.12
    theta[:, :, :2] *= scale
    grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
    return torch.nn.functional.grid_sample(x, grid, padding_mode="border", align_corners=False)


def train(train_npz: Path, val_npz: Path, out: Path, epochs: int = 30, device: str | None = None, seed: int = 0) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    tr, va = np.load(train_npz), np.load(val_npz)
    Xtr, ytr, vtr = _to_tensor(tr["X"]).to(device), torch.from_numpy(tr["y"]).long().to(device), torch.from_numpy(tr["view"]).to(device)
    Xva, yva = _to_tensor(va["X"]).to(device), torch.from_numpy(va["y"]).long().to(device)
    net = TileNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = epochs * ((len(Xtr) + 255) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    gen = torch.Generator(device=device).manual_seed(seed)
    best, hist = 0.0, []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr), device=device, generator=gen)
        for i in range(0, len(Xtr), 256):
            idx = perm[i : i + 256]
            loss = nn.functional.cross_entropy(net(_augment(Xtr[idx], vtr[idx], gen)), ytr[idx], label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            acc = float((net(Xva).argmax(1) == yva).float().mean())
        hist.append({"epoch": ep, "loss": float(loss), "val_acc": acc})
        if acc >= best:
            best = acc
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), out)
    return {"best_val_acc": best, "history": hist}


class TileClassifier:
    def __init__(self, path: str | Path, device: str = "cpu"):
        self.net = TileNet()
        self.net.load_state_dict(torch.load(path, map_location=device))
        self.net.eval().to(device)
        self.device = device

    @torch.no_grad()
    def predict(self, crops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if not crops:
            return np.zeros(0, int), np.zeros(0)
        x = _to_tensor(np.stack(crops)).to(self.device)
        p = torch.softmax(self.net(x), 1)
        conf, t = p.max(1)
        return t.cpu().numpy(), conf.cpu().numpy()


def recognition_test(model: Path, test_npz: Path, per_type: int = 10, seed: int = 0) -> dict:
    """Gate 2 recognition criterion on held-out views: >=90% overall, >=80% for every type.

    Draws ``per_type`` held-out views of each type for each view kind (overhead face-up and
    wrist face-on) and reports accuracy overall, per type, and per view kind.
    """
    z = np.load(test_npz)
    X, y, v = z["X"], z["y"], z["view"]
    rng = np.random.default_rng(seed)
    idx = []
    for view in (0, 1):
        for t in range(N_TYPES):
            pool = np.flatnonzero((y == t) & (v == view))
            if len(pool) < per_type:
                raise ValueError(f"only {len(pool)} held-out views of type {t} (view {view})")
            idx.extend(rng.choice(pool, per_type, replace=False))
    idx = np.array(idx)
    clf = TileClassifier(model)
    pred, conf = clf.predict(list(X[idx]))
    ok = pred == y[idx]
    per_t = {int(t): float(ok[y[idx] == t].mean()) for t in range(N_TYPES)}
    rep = {
        "views": int(len(idx)),
        "per_type_views": per_type * 2,
        "overall_acc": float(ok.mean()),
        "min_type_acc": min(per_t.values()),
        "per_type_acc": per_t,
        "per_view_acc": {"overhead_face_up": float(ok[v[idx] == 0].mean()), "wrist_face_on": float(ok[v[idx] == 1].mean())},
        "mean_confidence_correct": float(conf[ok].mean()) if ok.any() else 0.0,
        "mean_confidence_wrong": float(conf[~ok].mean()) if (~ok).any() else None,
    }
    rep["passed"] = rep["overall_acc"] >= 0.90 and rep["min_type_acc"] >= 0.80
    return rep


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
