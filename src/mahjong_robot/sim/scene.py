"""Compose the MuJoCo scene: YAM + linear_4310, cameras, table, and the 136 tiles.

The arm and gripper come unmodified from the vendored i2rt v1 models (assets/i2rt,
SOURCE.md). Composition follows the Agentic workspace's yam_model.py: merge the gripper
into the terminal ``gripper`` body, add position actuators and a box contact surrogate for
the fingers. It deliberately differs in one way. The pads are 15 mm wide, the measured width
of the upstream distal fingertip, not Agentic's 24 mm; wider pads would pinch neighbouring
tiles in a close-packed hand. The pads also follow the upstream visual meshes, which are closed
at joint7 = 0 and open at 0.0475, so the pad gap is 2 * joint7. Agentic's surrogate is mirrored
(open at 0); which convention matches the hardware must be checked on the real gripper.
"""

from __future__ import annotations

import copy
import os
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from ..game.tiles import N_INSTANCES, N_TYPES, type_of
from . import layout as L
from .textures import write_textures

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "assets"
YAM_XML = ASSETS / "i2rt" / "yam" / "yam.xml"
GRIPPER_XML = ASSETS / "i2rt" / "linear_4310" / "linear_4310.xml"
TEXTURE_DIR = ASSETS / "tiles"

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
ARM_KP = [80, 80, 80, 10, 10, 10]  # i2rt yam_v1.yml gains, as used in Agentic
ARM_DAMPING = [5, 5, 5, 1.5, 1.5, 1.5]
GRIPPER_KP, GRIPPER_KV, GRIPPER_FORCE = 2000, 10, 40
GRIPPER_RANGE = 0.0475
# Pad gap (m) = 2 * joint7: closed at 0, 95 mm at the upper limit.

# Wrist D405 extrinsic from i2rt's yam_station_linear_4310_d405 (flange -> optical frame).
WRIST_CAM_POS = np.array([-0.07, 0.0, -0.077])
WRIST_CAM_QUAT_CV = np.array([0.153, 0.69, 0.69, 0.153])
D405_FOVY, D435_FOVY = 58.0, 42.0
OVERHEAD_POS = np.array([0.0, 0.42, 0.85])
# Mounting yaw of the YAM base about z. It centres joint 6's +/-120 degree range on the work areas.
BASE_YAW = float(os.environ.get("MAHJONG_BASE_YAW", "0.0"))


def gripper_command_for_gap(gap: float) -> float:
    return float(np.clip(gap / 2.0, 0.0, GRIPPER_RANGE))


def _quat_cv_to_mj(q: np.ndarray) -> np.ndarray:
    """OpenCV optical frame (z forward, y down) -> MuJoCo camera frame (-z forward, y up)."""
    q = q / np.linalg.norm(q)
    flip = np.array([0.0, 1.0, 0.0, 0.0])  # 180 deg about x
    return L.quat_mul(q, flip)


def _f(v) -> str:
    return " ".join(f"{x:.6g}" for x in v)


def _compose_arm() -> tuple[ET.Element, list[ET.Element]]:
    arm = ET.parse(YAM_XML).getroot()
    grip = ET.parse(GRIPPER_XML).getroot()
    meshes = []
    for root, src in ((arm, YAM_XML), (grip, GRIPPER_XML)):
        for mesh in root.find("asset"):
            mesh = copy.deepcopy(mesh)
            mesh.set("file", str((src.parent / mesh.get("file")).resolve()))
            meshes.append(mesh)
    base = arm.find("./worldbody/body[@name='base']")
    mount = base.find(".//body[@name='gripper']")
    gbody = grip.find("./worldbody/body[@name='gripper']")
    mount.remove(mount.find("inertial"))
    mount.insert(0, copy.deepcopy(gbody.find("inertial")))
    frame = ET.SubElement(mount, "frame", {"pos": gbody.get("pos", "0 0 0"), "quat": gbody.get("quat", "1 0 0 0")})
    for child in gbody:
        if child.tag != "inertial":
            frame.append(copy.deepcopy(child))
    # Contact surrogate: visual meshes do not collide; boxes do.
    for g in base.iter("geom"):
        g.set("contype", "0")
        g.set("conaffinity", "0")
        g.set("group", "1")
    ET.SubElement(frame, "geom", {
        "name": "gripper_housing_contact", "type": "box", "size": "0.033 0.049 0.025",
        "pos": "0 0 -0.035", "rgba": "0.2 0.2 0.2 0.0", "group": "3", "contype": "1", "conaffinity": "1",
    })
    for name, sign in (("tip_left", -1), ("tip_right", 1)):
        body = frame.find(f".//body[@name='{name}']")
        # Surrogate boxes are placed in the gripper frame; convert into the tip body frame.
        body.set("gravcomp", "1")
        for suffix, z, half_z, fric in (("finger", -0.088, 0.034, "0.8 0.01 0.0001"),
                                        ("pad", -0.133, 0.011, "1.2 0.01 0.0001")):
            ET.SubElement(body, "geom", {
                "name": f"{name}_{suffix}_contact", "type": "box", "size": f"0.0075 0.006 {half_z}",
                "rgba": "0.9 0.5 0.1 0.35" if suffix == "pad" else "0.3 0.3 0.3 0.0", "group": "3",
                "friction": fric, "condim": "4", "contype": "1", "conaffinity": "1",
                "_gripper_frame_pos": _f([0.0, sign * 0.006, z]),
            })
    for body in base.iter("body"):
        body.set("gravcomp", "1")
    for j in base.iter("joint"):
        j.set("actuatorgravcomp", "true")
        name = j.get("name")
        if name in ARM_JOINTS:
            j.set("damping", str(ARM_DAMPING[ARM_JOINTS.index(name)]))
            j.set("armature", "0.01")
        else:
            j.set("damping", "0.5")
            j.set("armature", "0.001")
    mount_frame = frame
    cam_quat = _quat_cv_to_mj(WRIST_CAM_QUAT_CV)
    ET.SubElement(mount_frame, "camera", {"name": "wrist_d405", "pos": _f(WRIST_CAM_POS), "quat": _f(cam_quat),
                                          "fovy": str(D405_FOVY)})
    ET.SubElement(mount_frame, "site", {"name": "tcp", "pos": "0 0 -0.133", "size": "0.003", "rgba": "1 0 1 1",
                                        "group": "4"})
    ET.SubElement(mount_frame, "site", {"name": "wrist_d405_site", "pos": _f(WRIST_CAM_POS), "size": "0.004",
                                        "rgba": "0 0 1 1", "group": "4"})
    extras = [grip.find("equality"), grip.find("contact")]
    return base, meshes, extras


def build_xml(sleep: bool = True) -> str:
    textures = write_textures(TEXTURE_DIR)
    base, meshes, extras = _compose_arm()
    root = ET.Element("mujoco", {"model": "mahjong_yam"})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
    opt = ET.SubElement(root, "option", {"timestep": "0.002", "integrator": "implicitfast", "cone": "elliptic",
                                         "impratio": "10"})
    if sleep:
        ET.SubElement(opt, "flag", {"sleep": "enable"})
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", {"offwidth": "1280", "offheight": "720"})
    ET.SubElement(vis, "quality", {"shadowsize": "0"})
    ET.SubElement(vis, "headlight", {"ambient": "0.35 0.35 0.35", "diffuse": "0.45 0.45 0.45", "specular": "0 0 0"})
    ET.SubElement(vis, "map", {"znear": "0.005"})
    asset = ET.SubElement(root, "asset")
    for m in meshes:
        asset.append(m)
    for t in range(N_TYPES):
        ET.SubElement(asset, "texture", {"name": f"tile_tex_{t}", "type": "cube", "file": str(textures[t]),
                                         "gridsize": "3 4", "gridlayout": ".U..LFRB.D.."})
        ET.SubElement(asset, "material", {"name": f"tile_mat_{t}", "texture": f"tile_tex_{t}",
                                          "specular": "0.1", "shininess": "0.2"})
    ET.SubElement(asset, "texture", {"name": "felt", "type": "2d", "builtin": "flat", "rgb1": "0.12 0.36 0.22",
                                     "width": "16", "height": "16"})
    ET.SubElement(asset, "material", {"name": "felt", "texture": "felt", "rgba": "1 1 1 1"})

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", {"name": "key", "pos": "0 0.3 1.5", "dir": "0 0 -1", "diffuse": "0.55 0.55 0.55",
                                   "directional": "true"})
    ET.SubElement(world, "geom", {"name": "floor", "type": "plane", "size": "3 3 0.1", "pos": "0 0 -0.75",
                                  "rgba": "0.5 0.5 0.5 1"})
    cy = (L.TABLE_Y0 + L.TABLE_Y1) / 2
    ET.SubElement(world, "geom", {"name": "table", "type": "box", "material": "felt",
                                  "size": _f([L.TABLE_HALF_X, (L.TABLE_Y1 - L.TABLE_Y0) / 2, 0.02]),
                                  "pos": _f([0, cy, -0.02]), "friction": "0.5 0.005 0.0001"})
    ET.SubElement(world, "geom", {"name": "shared_zone_mark", "type": "box", "size": "0.03 0.03 0.0002",
                                  "pos": _f([*L.SHARED_XY, 0.0002]), "rgba": "0.85 0.8 0.3 1",
                                  "contype": "0", "conaffinity": "0"})
    ET.SubElement(world, "camera", {"name": "overhead_d435i", "pos": _f(OVERHEAD_POS), "xyaxes": "0 1 0 -1 0 0",
                                    "fovy": str(D435_FOVY)})
    base.set("quat", _f([np.cos(BASE_YAW / 2), 0.0, 0.0, np.sin(BASE_YAW / 2)]))
    world.append(base)
    # Movable camera with D405 intrinsics, used only to render classifier training/test views.
    dc = ET.SubElement(world, "body", {"name": "data_cam_mount", "mocap": "true", "pos": "0 -0.3 0.3"})
    ET.SubElement(dc, "camera", {"name": "data_cam", "fovy": str(D405_FOVY)})
    for i in range(N_INSTANCES):
        body = ET.SubElement(world, "body", {"name": f"tile_{i}", "pos": _f(L.storage_pos(i)), "gravcomp": "1"})
        ET.SubElement(body, "freejoint", {"name": f"tile_{i}_free"})
        ET.SubElement(body, "geom", {
            "name": f"tile_{i}_geom", "type": "box", "size": _f(L.TILE_HALF), "material": f"tile_mat_{type_of(i)}",
            "mass": str(L.TILE_MASS), "friction": "0.35 0.005 0.0001", "condim": "4",
            "contype": "0", "conaffinity": "0", "solref": "0.004 1",
        })
    for e in extras:
        if e is not None:
            root.append(copy.deepcopy(e))
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    ET.SubElement(contact, "exclude", {"body1": "base", "body2": "link1"})
    act = ET.SubElement(root, "actuator")
    for j, kp in zip(ARM_JOINTS, ARM_KP):
        ET.SubElement(act, "position", {"name": f"{j}_position", "joint": j, "kp": str(kp), "inheritrange": "1",
                                        "forcerange": "-10 10"})
    ET.SubElement(act, "position", {"name": "gripper_position", "joint": "joint7", "kp": str(GRIPPER_KP),
                                    "kv": str(GRIPPER_KV), "ctrlrange": f"0 {GRIPPER_RANGE}",
                                    "forcerange": f"-{GRIPPER_FORCE} {GRIPPER_FORCE}"})
    xml = ET.tostring(root, encoding="unicode")
    return xml


def _place_surrogates(model_xml: str) -> str:
    """Resolve the gripper-frame surrogate positions into each tip body's frame."""
    tmp = ET.fromstring(model_xml)
    probe = ET.fromstring(model_xml)
    for g in probe.iter("geom"):
        g.attrib.pop("_gripper_frame_pos", None)
    m = mujoco.MjModel.from_xml_string(ET.tostring(probe, encoding="unicode"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    gid = m.body("gripper").id
    Rg, pg = d.xmat[gid].reshape(3, 3), d.xpos[gid]
    for body in tmp.iter("body"):
        name = body.get("name")
        if name not in ("tip_left", "tip_right"):
            continue
        bid = m.body(name).id
        Rb, pb = d.xmat[bid].reshape(3, 3), d.xpos[bid]
        for g in body.findall("geom"):
            gp = g.attrib.pop("_gripper_frame_pos", None)
            if gp is None:
                continue
            world = pg + Rg @ np.array([float(x) for x in gp.split()])
            local = Rb.T @ (world - pb)
            qrel = np.zeros(4)
            mujoco.mju_mat2Quat(qrel, (Rb.T @ Rg).flatten())
            g.set("pos", _f(local))
            g.set("quat", _f(qrel))
    return ET.tostring(tmp, encoding="unicode")


@lru_cache(maxsize=2)
def scene_xml(sleep: bool = True) -> str:
    return _place_surrogates(build_xml(sleep))


def load_model(sleep: bool = True) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(scene_xml(sleep))
