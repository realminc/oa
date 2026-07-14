#!/usr/bin/env python3
"""walk_to_usda.py — emit a UsdSkel ASCII USD (.usda) of the walking skeleton.

Same skeleton + clean walk cycle as walk_to_fbx.py, written as a UsdSkel clip:
a SkelRoot > Skeleton (joints + bind/rest transforms) + SkelAnimation (per-frame
translations + rotations). This is the prototype for the "squashed USD" dataset
format — valid USD, opens in
usdview / any USD tool, and round-trips to the engine.

Reuses the bone layout + motion from walk_to_fbx.py so FBX and USD stay in sync.

Usage:  python Tools/Gen3dAnim/walk_to_usda.py --out <path.usda> [--frames 72 --fps 30]
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from walk_to_fbx import BONES, walk_frame  # shared skeleton + motion


def joint_paths():
    """Full UsdSkel joint paths, e.g. 'root/pelvis/spine_01'."""
    parent = {n: p for n, p, _ in BONES}
    paths = {}
    for name, _, _ in BONES:
        chain, cur = [], name
        while cur is not None:
            chain.append(cur)
            cur = parent[cur]
        paths[name] = "/".join(reversed(chain))
    return paths


def world_pos(name):
    """Bind-pose world position = sum of ancestor local translations (rest = identity rot)."""
    parent = {n: p for n, p, _ in BONES}
    rest = {n: t for n, _, t in BONES}
    x = y = z = 0.0
    cur = name
    while cur is not None:
        tx, ty, tz = rest[cur]
        x += tx; y += ty; z += tz
        cur = parent[cur]
    return (x, y, z)


def euler_deg_to_quat(rx, ry, rz):
    """XYZ-intrinsic Euler (degrees) -> quaternion (w, x, y, z)."""
    hx, hy, hz = (math.radians(a) * 0.5 for a in (rx, ry, rz))
    cx, sx = math.cos(hx), math.sin(hx)
    cy, sy = math.cos(hy), math.sin(hy)
    cz, sz = math.cos(hz), math.sin(hz)
    # q = qz * qy * qx
    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    return (w, x, y, z)


def mat4_translate(tx, ty, tz):
    """USD matrix4d (row-major; translation in 4th row) for a pure translation."""
    return "( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (%s, %s, %s, 1) )" % (
        repr(tx), repr(ty), repr(tz))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=72)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    frames, fps = args.frames, args.fps
    secs = [i / fps for i in range(frames)]
    names = [n for n, _, _ in BONES]
    paths = joint_paths()
    rest = {n: t for n, _, t in BONES}

    joints_tok = ", ".join('"%s"' % paths[n] for n in names)
    bind = ", ".join(mat4_translate(*world_pos(n)) for n in names)
    restm = ", ".join(mat4_translate(*rest[n]) for n in names)

    # Per-frame translations (local) + rotations (quat) for every joint.
    trans_ts, rot_ts = [], []
    for i, t in enumerate(secs):
        tr_row, ro_row = [], []
        for n in names:
            tr, ro = walk_frame(n, t)
            lt = tr if tr is not None else rest[n]
            tr_row.append("(%s, %s, %s)" % tuple(repr(float(v)) for v in lt))
            w, x, y, z = euler_deg_to_quat(*(ro if ro is not None else (0.0, 0.0, 0.0)))
            ro_row.append("(%s, %s, %s, %s)" % (repr(w), repr(x), repr(y), repr(z)))
        trans_ts.append("            %d: [%s]," % (i, ", ".join(tr_row)))
        rot_ts.append("            %d: [%s]," % (i, ", ".join(ro_row)))
    scales = ", ".join(["(1, 1, 1)"] * len(names))

    L = []
    a = L.append
    a("#usda 1.0")
    a("(")
    a('    defaultPrim = "Walk"')
    a('    upAxis = "Z"')
    a("    metersPerUnit = 0.01")
    a("    timeCodesPerSecond = %g" % fps)
    a("    startTimeCode = 0")
    a("    endTimeCode = %d" % (frames - 1))
    a("    doc = \"OA Gen3dAnim synthetic reference walk (squashed-USD prototype)\"")
    a(")")
    a("")
    a('def SkelRoot "Walk"')
    a("{")
    a('    def Skeleton "Skel" (')
    a('        prepend apiSchemas = ["SkelBindingAPI"]')
    a("    )")
    a("    {")
    a("        uniform token[] joints = [%s]" % joints_tok)
    a("        uniform matrix4d[] bindTransforms = [%s]" % bind)
    a("        uniform matrix4d[] restTransforms = [%s]" % restm)
    a("        rel skel:animationSource = </Walk/Anim>")
    a("    }")
    a("")
    a('    def SkelAnimation "Anim"')
    a("    {")
    a("        uniform token[] joints = [%s]" % joints_tok)
    a("        float3[] translations.timeSamples = {")
    a("\n".join(trans_ts))
    a("        }")
    a("        quatf[] rotations.timeSamples = {")
    a("\n".join(rot_ts))
    a("        }")
    a("        half3[] scales = [%s]" % scales)
    a("    }")
    a("}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    print("Wrote %s" % args.out)
    print("  joints=%d  frames=%d @ %g fps (%.2fs)" % (len(names), frames, fps, frames / fps))


if __name__ == "__main__":
    main()
