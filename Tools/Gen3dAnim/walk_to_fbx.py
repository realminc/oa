#!/usr/bin/env python3
"""walk_to_fbx.py — emit a VALID ASCII FBX (7500) of a simple walking skeleton.

Writes a Z-up, centimetre-unit skeleton (~18 bones) with a clean deterministic
walk cycle. The file is shaped like a real FBX SDK / Unreal export (full
FBXHeaderExtension with CreationTimeStamp + SceneInfo, PropertyTemplate
definitions, real Model/NodeAttribute layout, and a Takes section) so the
Autodesk FBX SDK importer — used by Maya and FBX Review — actually accepts it
and shows the joints.

This is a *synthetic reference* gait, not an ALM rollout. It validates the
pose->skeleton->FBX path used by the ALM motion representation.

Usage:  python Tools/Gen3dAnim/walk_to_fbx.py --out <path.fbx> [--frames 72 --fps 30]
"""

import argparse
import datetime
import math

FBX_TIME_PER_SEC = 46186158000  # FBX KTime units / second (TimeMode 6 base)
TAKE = "Take 001"

# (name, parent, local translation cm; Z-up, X-forward, Y-left)
BONES = [
    ("root",       None,         (0.0,   0.0,   0.0)),
    ("pelvis",     "root",       (0.0,   0.0,  96.0)),
    ("spine_01",   "pelvis",     (0.0,   0.0,  18.0)),
    ("spine_02",   "spine_01",   (0.0,   0.0,  18.0)),
    ("neck",       "spine_02",   (0.0,   0.0,  16.0)),
    ("head",       "neck",       (0.0,   0.0,  12.0)),
    ("clavicle_l", "spine_02",   (0.0,   7.0,  12.0)),
    ("upperarm_l", "clavicle_l", (0.0,  14.0,   0.0)),
    ("lowerarm_l", "upperarm_l", (0.0,  26.0,   0.0)),
    ("clavicle_r", "spine_02",   (0.0,  -7.0,  12.0)),
    ("upperarm_r", "clavicle_r", (0.0, -14.0,   0.0)),
    ("lowerarm_r", "upperarm_r", (0.0, -26.0,   0.0)),
    ("thigh_l",    "pelvis",     (0.0,   9.0,   0.0)),
    ("calf_l",     "thigh_l",    (0.0,   0.0, -44.0)),
    ("foot_l",     "calf_l",     (0.0,   0.0, -42.0)),
    ("thigh_r",    "pelvis",     (0.0,  -9.0,   0.0)),
    ("calf_r",     "thigh_r",    (0.0,   0.0, -44.0)),
    ("foot_r",     "calf_r",     (0.0,   0.0, -42.0)),
]

FORWARD_SPEED = 90.0   # cm/s
CADENCE_HZ    = 1.0    # walk cycles / sec


def walk_frame(name, t):
    """(translation|None, rotation_euler_deg|None) for a bone at time t (sec)."""
    ph = 2.0 * math.pi * CADENCE_HZ * t
    if name == "root":
        # -X: FBX FrontAxisSign=-1, so character faces -X; forward locomotion is -X.
        return ((-FORWARD_SPEED * t, 0.0, 2.0 * math.sin(2.0 * ph)), None)
    if name == "thigh_l":
        return (None, (0.0,  28.0 * math.sin(ph),           0.0))
    if name == "thigh_r":
        return (None, (0.0,  28.0 * math.sin(ph + math.pi), 0.0))
    if name == "calf_l":
        return (None, (0.0, -25.0 * (0.5 + 0.5 * math.sin(ph + 2.0)),           0.0))
    if name == "calf_r":
        return (None, (0.0, -25.0 * (0.5 + 0.5 * math.sin(ph + 2.0 + math.pi)), 0.0))
    if name == "upperarm_l":
        return (None, (0.0,  20.0 * math.sin(ph + math.pi), 0.0))
    if name == "upperarm_r":
        return (None, (0.0,  20.0 * math.sin(ph),           0.0))
    if name == "spine_01":
        return (None, (0.0, 0.0, 4.0 * math.sin(ph)))
    return (None, None)


class IdGen:
    def __init__(self, start=1000000000):
        self._n = start

    def next(self):
        self._n += 1
        return self._n


def ff(vals):
    return ",".join(repr(float(v)) for v in vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=72)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--static", action="store_true",
                    help="skeleton only — no animation curves / Takes (isolate skeleton validity)")
    args = ap.parse_args()

    frames, fps = args.frames, args.fps
    ids = IdGen()
    bone_id = {n: ids.next() for n, _, _ in BONES}
    attr_id = {n: ids.next() for n, _, _ in BONES}

    times = [int(round(i * FBX_TIME_PER_SEC / fps)) for i in range(frames)]
    secs = [i / fps for i in range(frames)]
    stop_time = times[-1] if times else 0

    # Build animation tracks (skipped entirely in --static mode).
    anim = []
    for name, _, _ in ([] if args.static else BONES):
        has_t = any(walk_frame(name, s)[0] is not None for s in secs)
        has_r = any(walk_frame(name, s)[1] is not None for s in secs)
        for kind, prop, sel in (("T", "Lcl Translation", 0), ("R", "Lcl Rotation", 1)):
            if (kind == "T" and not has_t) or (kind == "R" and not has_r):
                continue
            cols = [[], [], []]
            for s in secs:
                v = walk_frame(name, s)[sel] or (0.0, 0.0, 0.0)
                for k in range(3):
                    cols[k].append(v[k])
            anim.append({
                "bone": name, "kind": kind, "prop": prop, "node_id": ids.next(),
                "curves": [("d|X", ids.next(), cols[0]),
                           ("d|Y", ids.next(), cols[1]),
                           ("d|Z", ids.next(), cols[2])],
            })

    stack_id, layer_id, doc_id = ids.next(), ids.next(), ids.next()
    n_nodes = len(anim)
    n_curves = sum(len(a["curves"]) for a in anim)
    now = datetime.datetime.now()

    o = []
    w = o.append

    # ── FBXHeaderExtension (real-shaped: importer validates this) ──
    w("; FBX 7.5.0 project file")
    w("; ----------------------------------------------------\n")
    w("FBXHeaderExtension:  {")
    w("\tFBXHeaderVersion: 1003")
    w("\tFBXVersion: 7500")
    w("\tCreationTimeStamp:  {")
    w("\t\tVersion: 1000")
    w("\t\tYear: %d" % now.year)
    w("\t\tMonth: %d" % now.month)
    w("\t\tDay: %d" % now.day)
    w("\t\tHour: %d" % now.hour)
    w("\t\tMinute: %d" % now.minute)
    w("\t\tSecond: %d" % now.second)
    w("\t\tMillisecond: 0")
    w("\t}")
    w("\tCreator: \"FBX SDK/FBX Plugins version 2020.2\"")
    w("\tSceneInfo: \"SceneInfo::GlobalInfo\", \"UserData\" {")
    w("\t\tType: \"UserData\"")
    w("\t\tVersion: 100")
    w("\t\tMetaData:  {")
    w("\t\t\tVersion: 100")
    w("\t\t\tTitle: \"OA Gen3dAnim\"")
    w("\t\t\tSubject: \"\"")
    w("\t\t\tAuthor: \"\"")
    w("\t\t\tKeywords: \"\"")
    w("\t\t\tRevision: \"\"")
    w("\t\t\tComment: \"\"")
    w("\t\t}")
    w("\t}")
    w("}")

    # ── GlobalSettings (Z-up, cm, 30fps TimeMode 6) ──
    w("GlobalSettings:  {")
    w("\tVersion: 1000")
    w("\tProperties70:  {")
    for p in (
        ('UpAxis', 'int', 'Integer', 2), ('UpAxisSign', 'int', 'Integer', 1),
        ('FrontAxis', 'int', 'Integer', 1), ('FrontAxisSign', 'int', 'Integer', -1),
        ('CoordAxis', 'int', 'Integer', 0), ('CoordAxisSign', 'int', 'Integer', 1),
        ('OriginalUpAxis', 'int', 'Integer', 2), ('OriginalUpAxisSign', 'int', 'Integer', 1),
        ('UnitScaleFactor', 'double', 'Number', 1), ('OriginalUnitScaleFactor', 'double', 'Number', 1),
        ('TimeMode', 'enum', '', 6),
    ):
        w("\t\tP: \"%s\", \"%s\", \"%s\", \"\",%s" % p)
    w("\t\tP: \"TimeSpanStart\", \"KTime\", \"Time\", \"\",0")
    w("\t\tP: \"TimeSpanStop\", \"KTime\", \"Time\", \"\",%d" % stop_time)
    w("\t}")
    w("}")

    # ── Documents (ActiveAnimStackName makes the take play) ──
    w("Documents:  {")
    w("\tCount: 1")
    w("\tDocument: %d, \"\", \"Scene\" {" % doc_id)
    w("\t\tProperties70:  {")
    w("\t\t\tP: \"SourceObject\", \"object\", \"\", \"\"")
    w("\t\t\tP: \"ActiveAnimStackName\", \"KString\", \"\", \"\", \"%s\"" % ("" if args.static else TAKE))
    w("\t\t}")
    w("\t\tRootNode: 0")
    w("\t}")
    w("}")
    w("References:  {")
    w("}")

    # ── Definitions (with PropertyTemplates) ──
    total = 1 + len(BONES) * 2 + 2 + n_nodes + n_curves
    w("Definitions:  {")
    w("\tVersion: 100")
    w("\tCount: %d" % total)
    w("\tObjectType: \"GlobalSettings\" {\n\t\tCount: 1\n\t}")
    w("\tObjectType: \"AnimationStack\" {")
    w("\t\tCount: 1")
    w("\t\tPropertyTemplate: \"FbxAnimStack\" {")
    w("\t\t\tProperties70:  {")
    w("\t\t\t\tP: \"Description\", \"KString\", \"\", \"\", \"\"")
    w("\t\t\t\tP: \"LocalStart\", \"KTime\", \"Time\", \"\",0")
    w("\t\t\t\tP: \"LocalStop\", \"KTime\", \"Time\", \"\",0")
    w("\t\t\t\tP: \"ReferenceStart\", \"KTime\", \"Time\", \"\",0")
    w("\t\t\t\tP: \"ReferenceStop\", \"KTime\", \"Time\", \"\",0")
    w("\t\t\t}\n\t\t}\n\t}")
    w("\tObjectType: \"AnimationLayer\" {")
    w("\t\tCount: 1")
    w("\t\tPropertyTemplate: \"FbxAnimLayer\" {")
    w("\t\t\tProperties70:  {\n\t\t\t\tP: \"Weight\", \"Number\", \"\", \"A\",100\n\t\t\t}\n\t\t}\n\t}")
    w("\tObjectType: \"NodeAttribute\" {")
    w("\t\tCount: %d" % len(BONES))
    w("\t\tPropertyTemplate: \"FbxSkeleton\" {")
    w("\t\t\tProperties70:  {")
    w("\t\t\t\tP: \"Color\", \"ColorRGB\", \"Color\", \"\",0.8,0.8,0.8")
    w("\t\t\t\tP: \"Size\", \"double\", \"Number\", \"\",100")
    w("\t\t\t}\n\t\t}\n\t}")
    w("\tObjectType: \"Model\" {")
    w("\t\tCount: %d" % len(BONES))
    w("\t\tPropertyTemplate: \"FbxNode\" {")
    w("\t\t\tProperties70:  {")
    for p in (
        ('RotationActive', 'bool', '', 0), ('InheritType', 'enum', '', 0),
        ('ScalingMax', 'Vector3D', 'Vector', '0,0,0'),
        ('DefaultAttributeIndex', 'int', 'Integer', -1),
    ):
        w("\t\t\t\tP: \"%s\", \"%s\", \"%s\", \"\",%s" % p)
    w("\t\t\t\tP: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\",0,0,0")
    w("\t\t\t\tP: \"Lcl Rotation\", \"Lcl Rotation\", \"\", \"A\",0,0,0")
    w("\t\t\t\tP: \"Lcl Scaling\", \"Lcl Scaling\", \"\", \"A\",1,1,1")
    w("\t\t\t\tP: \"Visibility\", \"Visibility\", \"\", \"A\",1")
    w("\t\t\t}\n\t\t}\n\t}")
    w("\tObjectType: \"AnimationCurveNode\" {")
    w("\t\tCount: %d" % n_nodes)
    w("\t\tPropertyTemplate: \"FbxAnimCurveNode\" {")
    w("\t\t\tProperties70:  {\n\t\t\t\tP: \"d\", \"Compound\", \"\", \"\"\n\t\t\t}\n\t\t}\n\t}")
    w("\tObjectType: \"AnimationCurve\" {\n\t\tCount: %d\n\t}" % n_curves)
    w("}")

    # ── Objects ──
    w("Objects:  {")
    for name, parent, _ in BONES:
        if parent is None:
            w("\tNodeAttribute: %d, \"NodeAttribute::%s\", \"Root\" {" % (attr_id[name], name))
            w("\t\tTypeFlags: \"Null\", \"Skeleton\", \"Root\"")
            w("\t}")
        else:
            w("\tNodeAttribute: %d, \"NodeAttribute::%s\", \"LimbNode\" {" % (attr_id[name], name))
            w("\t\tProperties70:  {\n\t\t\tP: \"Size\", \"double\", \"Number\", \"\",6\n\t\t}")
            w("\t\tTypeFlags: \"Skeleton\"")
            w("\t}")
    for name, parent, rest in BONES:
        w("\tModel: %d, \"Model::%s\", \"LimbNode\" {" % (bone_id[name], name))
        w("\t\tVersion: 232")
        w("\t\tProperties70:  {")
        w("\t\t\tP: \"DefaultAttributeIndex\", \"int\", \"Integer\", \"\",0")
        w("\t\t\tP: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A+\",%s" % ff(rest))
        w("\t\t\tP: \"Lcl Rotation\", \"Lcl Rotation\", \"\", \"A+\",0,0,0")
        w("\t\t}")
        w("\t\tShading: Y")
        w("\t\tCulling: \"CullingOff\"")
        w("\t}")
    if anim:
        w("\tAnimationStack: %d, \"AnimStack::%s\", \"\" {" % (stack_id, TAKE))
        w("\t\tProperties70:  {")
        w("\t\t\tP: \"LocalStart\", \"KTime\", \"Time\", \"\",0")
        w("\t\t\tP: \"LocalStop\", \"KTime\", \"Time\", \"\",%d" % stop_time)
        w("\t\t\tP: \"ReferenceStart\", \"KTime\", \"Time\", \"\",0")
        w("\t\t\tP: \"ReferenceStop\", \"KTime\", \"Time\", \"\",%d" % stop_time)
        w("\t\t}")
        w("\t}")
        w("\tAnimationLayer: %d, \"AnimLayer::Base Layer\", \"\" {\n\t}" % layer_id)
    for a in anim:
        w("\tAnimationCurveNode: %d, \"AnimCurveNode::%s\", \"\" {" % (a["node_id"], a["kind"]))
        w("\t\tProperties70:  {")
        for chan, _cid, vals in a["curves"]:
            w("\t\t\tP: \"%s\", \"Number\", \"\", \"A\",%s" % (chan, repr(float(vals[0]))))
        w("\t\t}")
        w("\t}")
        for _chan, cid, vals in a["curves"]:
            w("\tAnimationCurve: %d, \"AnimCurve::\", \"\" {" % cid)
            w("\t\tDefault: %s" % repr(float(vals[0])))
            w("\t\tKeyVer: 4009")
            w("\t\tKeyTime: *%d {\n\t\t\ta: %s\n\t\t}" % (frames, ",".join(str(t) for t in times)))
            w("\t\tKeyValueFloat: *%d {\n\t\t\ta: %s\n\t\t}" % (frames, ff(vals)))
            w("\t\tKeyAttrFlags: *1 {\n\t\t\ta: 24840\n\t\t}")
            w("\t\tKeyAttrDataFloat: *4 {\n\t\t\ta: 0,0,0,0\n\t\t}")
            w("\t\tKeyAttrRefCount: *1 {\n\t\t\ta: %d\n\t\t}" % frames)
            w("\t}")
    w("}")

    # ── Connections ──
    w("Connections:  {")
    for name, parent, _ in BONES:
        dst = 0 if parent is None else bone_id[parent]
        w("\t;Model::%s, Model::%s" % (name, parent if parent else "RootNode"))
        w("\tC: \"OO\",%d,%d" % (bone_id[name], dst))
        w("\t;NodeAttribute::%s, Model::%s" % (name, name))
        w("\tC: \"OO\",%d,%d" % (attr_id[name], bone_id[name]))
    w("\t;AnimLayer::Base Layer, AnimStack::%s" % TAKE)
    w("\tC: \"OO\",%d,%d" % (layer_id, stack_id))
    for a in anim:
        w("\tC: \"OO\",%d,%d" % (a["node_id"], layer_id))
        w("\tC: \"OP\",%d,%d, \"%s\"" % (a["node_id"], bone_id[a["bone"]], a["prop"]))
        for chan, cid, _v in a["curves"]:
            w("\tC: \"OP\",%d,%d, \"%s\"" % (cid, a["node_id"], chan))
    w("}")

    # ── Takes (legacy; Maya importer still reads it) ──
    if anim:
        w("Takes:  {")
        w("\tCurrent: \"%s\"" % TAKE)
        w("\tTake: \"%s\" {" % TAKE)
        w("\t\tFileName: \"Take_001.tak\"")
        w("\t\tLocalTime: 0,%d" % stop_time)
        w("\t\tReferenceTime: 0,%d" % stop_time)
        w("\t}")
        w("}")

    with open(args.out, "w", encoding="ascii") as f:
        f.write("\n".join(o) + "\n")

    print("Wrote %s" % args.out)
    print("  bones=%d  curvenodes=%d  curves=%d  frames=%d @ %g fps (%.2fs)"
          % (len(BONES), n_nodes, n_curves, frames, fps, frames / fps))


if __name__ == "__main__":
    main()
