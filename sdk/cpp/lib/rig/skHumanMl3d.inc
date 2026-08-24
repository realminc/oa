// SkHumanMl3d.inc - the HumanML3D / SMPL 22-joint body skeleton definition.
//
// This is the skeleton the HumanML3D text-to-motion datasets (and AnimationGPT's
// CombatMotionProcessed, CMP) are authored on: 22 SMPL body joints, the
// `t2m_kinematic_chain`. The 263-dim HumanML3D feature vector and the per-frame
// `new_joints` array [T, 22, 3] both index these joints in this order.
//
// Reuses the same BuiltinJoint struct as SkMetaHuman.inc (which must be included
// first, inside the same anonymous namespace in Skeleton.cpp).
//
// Parent table == t2m_kinematic_chain expanded:
//   [0,2,5,8,11] [0,1,4,7,10] [0,3,6,9,12,15] [9,14,17,19,21] [9,13,16,18,20]
//
// Rest offsets (Tx,Ty,Tz) are left at 0: HumanML3D is a positions-first format,
// so the USD preview path derives per-joint bind/rest from each clip's frame-0
// world positions (see OaUsdClipFromWorldJoints). Exact rest offsets can later be
// seeded from HumanML3D's `t2m_raw_offsets` if a standalone rest pose is needed.
// RotDof=3 everywhere (the feature vector carries 6D rotations for the 21
// non-root joints); only the root carries a live translate channel.

// clang-format off
constexpr BuiltinJoint kHumanMl3dBody[] = {
	// Name            Parent          HikId  Tx Ty Tz   Rx Ry Rz   Mass  HasT   RotDof  HikSlot
	{ "pelvis",        "",                 0,  0, 0, 0,   0, 0, 0,   0.0f, true,  3, "" },
	{ "left_hip",      "pelvis",           0,  0, 0, 0,   0, 0, 0,   0.0f, false, 3, "" },
	{ "right_hip",     "pelvis",           0,  0, 0, 0,   0, 0, 0,   0.0f, false, 3, "" },
	{ "spine1",        "pelvis",           0,  0, 0, 0,   0, 0, 0,   0.0f, false, 3, "" },
	{ "left_knee",     "left_hip",         0,  0, 0, 0,   0, 0, 0,   0.0f, false, 3, "" },
	{ "right_knee",    "right_hip",        0,  0, 0, 0,   0, 0, 0,   0.0f, false, 3, "" },
	{ "spine2",        "spine1",           0,  0, 0, 0,   0, 0, 0,   0.0f, false, 3, "" },
	{ "left_ankle",    "left_knee",        0,  0, 0, 0,   0, 0, 0,   0.0f, false, 3, "" },
	{ "right_ankle",   "right_knee",       0,  0, 0, 0,   0, 0, 0,   0.0f, false, 3, "" },
	{ "spine3",        "spine2",           0,  0, 0, 0,   0, 0, 0,   0.0f, false, 3, "" },
	{ "left_foot",     "left_ankle",       0,  0, 0, 0,   0, 0, 0,   0.0f, false, 3, "" },
	{ "right_foot",    "right_ankle",      0,  0, 0, 0,   0, 0, 0,   0.0f, false, 3, "" },
	{ "neck",          "spine3",           0,  0, 0, 0,   0, 0, 0,   0.0f, false, 3, "" },
	{ "left_collar",   "spine3",           0,  0, 0, 0,   0, 0, 0,   0.0f, false, 3, "" },
	{ "right_collar",  "spine3",           0,  0, 0, 0,   0, 0, 0,   0.0f, false, 3, "" },
	{ "head",          "neck",             0,  0, 0, 0,   0, 0, 0,   0.0f, false, 3, "" },
	{ "left_shoulder", "left_collar",      0,  0, 0, 0,   0, 0, 0,   0.0f, false, 3, "" },
	{ "right_shoulder","right_collar",     0,  0, 0, 0,   0, 0, 0,   0.0f, false, 3, "" },
	{ "left_elbow",    "left_shoulder",    0,  0, 0, 0,   0, 0, 0,   0.0f, false, 3, "" },
	{ "right_elbow",   "right_shoulder",   0,  0, 0, 0,   0, 0, 0,   0.0f, false, 3, "" },
	{ "left_wrist",    "left_elbow",       0,  0, 0, 0,   0, 0, 0,   0.0f, false, 3, "" },
	{ "right_wrist",   "right_elbow",      0,  0, 0, 0,   0, 0, 0,   0.0f, false, 3, "" },
};
// clang-format on
