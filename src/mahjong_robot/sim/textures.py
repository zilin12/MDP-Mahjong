"""Procedural cube textures for the 34 tile types.

Each texture is a 3x4 MuJoCo cube grid (layout ".U..LFRB.D..", 64 px cells). The engraved
face is F (+z), the back B (-z) is green, and the four edges are ivory with a dark rim so
adjacent upright tiles stay separable in RGB.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..game.tiles import N_TYPES, is_suited, rank_of, suit_of

CELL = 64
IVORY = (236, 230, 210)
GREEN = (40, 120, 70)
RIM = (90, 80, 60)
SUIT_COLORS = {0: (170, 25, 30), 1: (20, 120, 50), 2: (25, 60, 160)}
HONOR_GLYPHS = {27: "東", 28: "南", 29: "西", 30: "北", 31: "中", 32: "發", 33: ""}
HONOR_COLORS = {27: (20, 20, 20), 28: (20, 20, 20), 29: (20, 20, 20), 30: (20, 20, 20),
                31: (190, 20, 25), 32: (20, 130, 50), 33: (30, 70, 170)}
CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
LATIN_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size=size)


def face_image(t: int, w: int = 100, h: int = 130) -> Image.Image:
    """The engraved face at the tile's true aspect ratio (width x height)."""
    img = Image.new("RGB", (w, h), IVORY)
    d = ImageDraw.Draw(img)
    d.rectangle([2, 2, w - 3, h - 3], outline=RIM, width=4)
    if is_suited(t):
        s, r = suit_of(t), rank_of(t)
        col = SUIT_COLORS[s]
        d.text((w // 2, 30), str(r), fill=(15, 15, 15), font=_font(LATIN_FONT, 44), anchor="mm")
        if s == 0:
            d.text((w // 2, 92), "萬", fill=col, font=_font(CJK_FONT, 54), anchor="mm")
        else:
            cols = 3 if r > 4 else 2
            rows = (r + cols - 1) // cols
            for k in range(r):
                cx = w // 2 + (k % cols - (cols - 1) / 2) * 26
                cy = 90 + (k // cols - (rows - 1) / 2) * 20
                if s == 1:
                    d.rectangle([cx - 5, cy - 8, cx + 5, cy + 8], fill=col)
                else:
                    d.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], outline=col, width=4)
    elif t == 33:
        d.rectangle([18, 22, w - 19, h - 23], outline=HONOR_COLORS[t], width=8)
    else:
        d.text((w // 2, h // 2 + 2), HONOR_GLYPHS[t], fill=HONOR_COLORS[t], font=_font(CJK_FONT, 78), anchor="mm")
    return img


def _edge_cell(color=IVORY) -> Image.Image:
    img = Image.new("RGB", (CELL, CELL), color)
    ImageDraw.Draw(img).rectangle([0, 0, CELL - 1, CELL - 1], outline=RIM, width=3)
    return img


def cube_texture(t: int) -> Image.Image:
    grid = Image.new("RGB", (4 * CELL, 3 * CELL), (0, 0, 0))
    face = face_image(t).resize((CELL, CELL), Image.BICUBIC)
    cells = {"U": _edge_cell(), "D": _edge_cell(), "L": _edge_cell(), "R": _edge_cell(),
             "F": face, "B": _edge_cell(GREEN)}
    layout = ".U..LFRB.D.."
    for i, ch in enumerate(layout):
        if ch in cells:
            grid.paste(cells[ch], ((i % 4) * CELL, (i // 4) * CELL))
    return grid


def write_textures(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for t in range(N_TYPES):
        p = out_dir / f"tile_{t:02d}.png"
        if not p.exists():
            cube_texture(t).save(p)
        paths.append(p)
    return paths
