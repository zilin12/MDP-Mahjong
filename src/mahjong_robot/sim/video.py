"""Film a simulated hand: 1080p footage plus a readable game record.

``HandRecorder`` is an observer for ``gate3.play_hand``. It renders a table view, the
robot's two camera views and an information panel every ``every`` physics steps and writes
them to an MP4. Rendering only reads the simulation, so the recorded hand is the same hand
the gate would play. The panel's move log comes from the table's record (what a spectator
sees); the robot's own hand is drawn from its belief, i.e. what it perceived.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..game import actions as A
from ..game.tiles import LONG_NAMES, name, type_of
from .textures import face_image

W, H = 1920, 1080
BG, PANEL, INK, INK2, ACCENT, RULE = (15, 20, 18), (24, 31, 28), (236, 242, 238), (160, 174, 167), (76, 185, 143), (44, 54, 49)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SEAT = {0: "Robot", 1: "Right seat", 2: "Across seat", 3: "Left seat"}
ACTIVITY = {
    "pick": "Picking up a tile", "insert": "Inserting the tile into its hand", "tip_over": "Turning a tile face-up (tip-over)",
    "place_flat": "Placing a claimed tile in its meld", "push": "Closing a gap in the hand (compaction)",
}


def _font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


class HandRecorder:
    def __init__(self, world, table, robot, path: Path, every: int = 40, fps: int = 30, title: str = ""):
        self.w, self.table, self.robot = world, table, robot
        self.path, self.every, self.fps, self.title = Path(path), every, fps, title
        m = world.m
        self.r_main = mujoco.Renderer(m, 720, 1280)
        self.r_small = mujoco.Renderer(m, 360, 640)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [-0.05, 0.31, 0.0]  # front-left of the robot: the arm stays clear of the hand row
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.0, 45.0, -46.0
        self.opt = mujoco.MjvOption()
        self.faces = {t: face_image(t).resize((34, 44), Image.LANCZOS) for t in range(34)}
        self.f_title, self.f_h, self.f_b, self.f_s = _font(30, True), _font(21, True), _font(19), _font(16)
        self.k = 0
        self.frames = 0
        self.activity = "Dealing"
        self.decision = "—"
        self.log: list[str] = []
        self.decisions: list[dict] = []
        self.events_seen = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio_ffmpeg.write_frames(str(self.path), (W, H), fps=fps, codec="libx264", quality=7,
                                                  pix_fmt_out="yuv420p", macro_block_size=8)
        self.writer.send(None)
        self._wrap()
        world.arm.step_hook = self._hook

    # ------------------------------------------------------------ what the robot is doing

    def _wrap(self):
        rb = self.robot
        op, decide, park, view = rb._op, rb.decide, rb.ops.park, rb._wrist_view

        def _op(fn, *args):
            self.activity = ACTIVITY.get(fn.__name__, fn.__name__)
            return op(fn, *args)

        def _decide(phase):
            a = decide(phase)
            self.decision = A.describe(a)
            self.decisions.append({"sim_time_s": round(float(self.w.d.time), 1), "phase": phase, "action": A.describe(a)})
            self.log.append(f"Robot's policy chooses: {A.describe(a)}")
            return a

        def _park():
            self.activity = "Parking out of the overhead camera's view to look"
            return park()

        def _view(key, *args):
            self.activity = "Reading tile faces with the wrist camera"
            return view(key, *args)

        rb._op, rb.decide, rb.ops.park, rb._wrist_view = _op, _decide, _park, _view

    def _pull_events(self):
        ev = self.table.engine.events
        seat = self.table.seat
        for e in ev[self.events_seen:]:
            rel = (e[1] - seat) % 4 if len(e) > 1 and isinstance(e[1], int) else None
            who = SEAT.get(rel, "")
            k = e[0]
            if k == "discard":
                self.log.append(f"{who} discards {LONG_NAMES[type_of(e[2])]}")
            elif k in ("draw", "deal_draw", "replacement_draw") and rel == 0:
                self.log.append("Robot draws; the tile is stood up in the shared zone")
            elif k in ("pung", "chow", "kong_exposed"):
                frm = SEAT[(e[3] - seat) % 4]
                self.log.append(f"{who} claims {LONG_NAMES[type_of(e[2])]} from {frm}: {k.replace('_exposed', '')}")
            elif k == "kong_concealed":
                self.log.append(f"{who} declares a concealed kong of {LONG_NAMES[type_of(e[2][0])]}")
            elif k == "kong_added":
                self.log.append(f"{who} adds to a pung: kong of {LONG_NAMES[type_of(e[2])]}")
            elif k == "win":
                how = "self-draw" if e[2] == "self_draw" else f"on {SEAT[(e[4] - seat) % 4]}'s discard"
                self.log.append(f"{who} WINS by {how} ({LONG_NAMES[type_of(e[3])]})")
            elif k == "exhausted":
                self.log.append("Wall exhausted: drawn hand")
        self.events_seen = len(ev)

    # ------------------------------------------------------------ frames

    def _hook(self):
        self.k += 1
        if self.k % self.every == 0:
            self._frame()

    def _render(self, r, cam):
        r.update_scene(self.w.d, cam, self.opt) if not isinstance(cam, str) else r.update_scene(self.w.d, cam)
        return Image.fromarray(r.render())

    def _tiles(self, dr, x, y, types, gap=3):
        for t in types:
            dr._image.paste(self.faces[t], (x, y))
            x += 34 + gap
        return x

    def _frame(self, banner: str | None = None):
        self._pull_events()
        img = Image.new("RGB", (W, H), BG)
        img.paste(self._render(self.r_main, self.cam), (0, 64))
        img.paste(self._render(self.r_small, "overhead_d435i"), (1280, 64))
        img.paste(self._render(self.r_small, "wrist_d405"), (1280, 424))
        dr = ImageDraw.Draw(img)
        dr._image = img
        # header
        dr.text((24, 16), "Single-arm Mahjong", font=self.f_title, fill=INK)
        dr.text((24 + dr.textlength("Single-arm Mahjong", font=self.f_title) + 24, 24), self.title, font=self.f_b, fill=INK2)
        sp = self.every * self.w.m.opt.timestep * self.fps
        dr.text((W - 24, 24), f"sim time {self.w.d.time:6.1f} s   ·   playback ×{sp:.1f}", font=self.f_b, fill=INK2, anchor="ra")
        for (x, y, label) in ((16, 76, "TABLE VIEW (spectator camera)"), (1296, 76, "OVERHEAD D435i · what the robot sees"),
                              (1296, 436, "WRIST D405 · on the gripper")):
            tw = dr.textlength(label, font=self.f_s)
            dr.rectangle([x - 6, y - 4, x + tw + 6, y + 20], fill=(0, 0, 0))
            dr.text((x, y), label, font=self.f_s, fill=INK)
        # bottom panel
        dr.rectangle([0, 784, W, H], fill=PANEL)
        dr.line([0, 784, W, 784], fill=RULE, width=2)
        b = self.robot.b
        dr.text((24, 800), "ROBOT'S HAND (as perceived)", font=self.f_s, fill=ACCENT)
        hand = [h.tile_type for h in b.hand]
        x = self._tiles(dr, 24, 826, hand)
        if b.drawn is not None:
            dr.text((x + 10, 836), "+ drawn", font=self.f_s, fill=INK2)
            self._tiles(dr, x + 90, 826, [b.drawn])
        dr.text((24, 884), "MELDS", font=self.f_s, fill=ACCENT)
        x = 24
        for m in b.own_melds:
            x = self._tiles(dr, x, 908, [type_of(t) for t in m.tiles], gap=1) + 18
        if not b.own_melds:
            dr.text((24, 916), "none yet", font=self.f_s, fill=INK2)
        dr.text((24, 972), "NOW", font=self.f_s, fill=ACCENT)
        dr.text((90, 968), self.activity, font=self.f_h, fill=INK)
        dr.text((24, 1012), "LAST DECISION", font=self.f_s, fill=ACCENT)
        dr.text((168, 1008), self.decision, font=self.f_h, fill=INK)
        dr.text((24, 1048), f"wall: {self.table.engine.wall_remaining} tiles   ·   robot seat {self.table.seat} (dealer = seat 0)",
                 font=self.f_s, fill=INK2)
        # move log
        dr.line([1100, 800, 1100, 1064], fill=RULE, width=1)
        dr.text((1124, 800), "MOVE LOG", font=self.f_s, fill=ACCENT)
        for i, line in enumerate(self.log[-10:]):
            col = INK if line.startswith("Robot") else INK2
            if "WINS" in line or "exhausted" in line:
                col = ACCENT
            dr.text((1124, 826 + i * 24), line, font=self.f_s, fill=col)
        if banner:
            dr.rectangle([340, 360, 940, 480], fill=(0, 0, 0))
            dr.text((640, 420), banner, font=self.f_title, fill=ACCENT, anchor="mm")
        self.writer.send(np.asarray(img).tobytes())
        self.frames += 1

    # ------------------------------------------------------------ end

    def finish(self, result: dict):
        self.w.arm.step_hook = None
        self._pull_events()
        outcome = result.get("outcome")
        banner = {"robot_win": "The robot wins", "other_win": "Another seat wins", "exhausted": "Drawn hand"}.get(
            outcome, f"Stopped: {result.get('fault')}")
        for _ in range(self.fps * 4):
            self._frame(banner)
        self.writer.close()
        self.r_main.close()
        self.r_small.close()
        self._write_record(result)

    def _write_record(self, result: dict):
        eng = self.table.engine
        seat = self.table.seat
        lines = [f"# Game record: simulated hand {result['seed']}", "",
                 f"- Mode: {result['mode']} manipulation; robot seat {seat} (seat 0 deals)",
                 f"- Outcome: {result.get('outcome')} {('(' + result['win_type'] + ')') if result.get('win_type') else ''}",
                 f"- Completed: {result['completed']}; simulated time {result['sim_time_s']} s; "
                 f"tile operations {result.get('operations')}; recoveries {result.get('recoveries')}",
                 "", "Seats are named from the robot's point of view.", "", "| # | Seat | Event |", "| --- | --- | --- |"]
        n = 0
        for e in eng.events:
            k = e[0]
            if k == "deal":
                continue
            rel = (e[1] - seat) % 4 if len(e) > 1 and isinstance(e[1], int) else None
            who = SEAT.get(rel, "")
            if k in ("draw", "deal_draw", "replacement_draw"):
                txt = ("replacement " if k == "replacement_draw" else "") + "draw" + (f" ({name(type_of(e[2]))})" if rel == 0 else "")
            elif k == "discard":
                txt = f"discard **{name(type_of(e[2]))}**"
            elif k in ("pung", "chow", "kong_exposed"):
                txt = f"{k.replace('_exposed', '')} of {name(type_of(e[2]))} from {SEAT[(e[3] - seat) % 4]}"
            elif k == "kong_concealed":
                txt = f"concealed kong of {name(type_of(e[2][0]))}"
            elif k == "kong_added":
                txt = f"added kong of {name(type_of(e[2]))}"
            elif k == "win":
                txt = f"**WIN** by {'self-draw' if e[2] == 'self_draw' else 'discard from ' + SEAT[(e[4] - seat) % 4]} on {name(type_of(e[3]))}"
            else:
                txt = k
            n += 1
            lines.append(f"| {n} | {who} | {txt} |")
        final = sorted(type_of(t) for t in eng.hands[seat])
        lines += ["", f"Robot's final concealed hand: {' '.join(name(t) for t in final)}",
                  "Robot's melds: " + ("; ".join(" ".join(name(type_of(t)) for t in m.tiles) for m in eng.melds[seat]) or "none")]
        self.path.with_suffix(".md").write_text("\n".join(lines) + "\n")
        self.path.with_suffix(".json").write_text(json.dumps({"result": result, "robot_decisions": self.decisions,
                                                               "events": [list(map(str, e)) for e in eng.events]}, indent=1))
