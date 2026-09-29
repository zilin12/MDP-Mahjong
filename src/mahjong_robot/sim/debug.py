"""Frame recorder for inspecting operations (development aid, not part of the robot's inputs)."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


class Recorder:
    def __init__(self, world, lookat, distance=0.25, azimuth=90.0, elevation=-30.0, every=25, size=(360, 480)):
        self.w = world
        self.r = mujoco.Renderer(world.m, *size)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = lookat
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
        self.every, self.k, self.frames = every, 0, []
        world.arm.step_hook = self._hook

    def _hook(self):
        self.k += 1
        if self.k % self.every == 0:
            self.r.update_scene(self.w.d, self.cam)
            self.frames.append(self.r.render().copy())

    def save(self, path: str | Path, fps: int = 20, stride: int = 1) -> None:
        frames = [Image.fromarray(f) for f in self.frames[::stride]]
        if frames:
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)

    def contact_sheet(self, path: str | Path, n: int = 12) -> None:
        idx = np.linspace(0, len(self.frames) - 1, n).astype(int)
        imgs = [self.frames[i] for i in idx]
        h, w, _ = imgs[0].shape
        cols = 4
        rows = (len(imgs) + cols - 1) // cols
        sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        for j, im in enumerate(imgs):
            sheet[(j // cols) * h:(j // cols + 1) * h, (j % cols) * w:(j % cols + 1) * w] = im
        Image.fromarray(sheet).save(path)

    def close(self):
        self.w.arm.step_hook = None
        self.r.close()
