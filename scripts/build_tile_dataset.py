"""Render the tile classifier's train/val/test crops (held-out test uses its own seeds)."""

from pathlib import Path

from mahjong_robot.vision.dataset import build_parallel

if __name__ == "__main__":
    build_parallel(Path("data/tiles_train.npz"), 11, flat_scenes=1200, face_scenes=1800)
    build_parallel(Path("data/tiles_val.npz"), 12, flat_scenes=96, face_scenes=144)
    build_parallel(Path("data/tiles_test.npz"), 13, flat_scenes=192, face_scenes=288)
    print("done")
