#include <Rig/Skeleton.h>
#include <Rig/SkeletonUsd.h>

#include <Core/Transform.h>
#include <Oa/Core/Vlm.h>

#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <sstream>

// ─── Built-in skeleton tables ────────────────────────────────────────────────

namespace {

#include "SkMetaHuman.inc"
#include "SkHumanMl3d.inc"

constexpr OaI32 kJointCount = static_cast<OaI32>(sizeof(kMetaHumanBody) / sizeof(kMetaHumanBody[0]));

// Index of a UE bone name in the builtin table (== joint index, since joints are
// added in table order). Renaming-independent, unlike OaSkeleton::IndexOf.
OaI32 TableIndexOf(OaStringView InName) {
	for (OaI32 k = 0; k < kJointCount; ++k) {
		if (InName == kMetaHumanBody[k].Name) {
			return k;
		}
	}
	return -1;
}

// Build a skeleton from the manny table. `InUseHikNames` swaps each joint's UE
// bone name for its HumanIK slot name (geometry/hierarchy unchanged).
OaSkeleton BuildBody(OaStringView InName, OaU32 InId, bool InUseHikNames) {
	OaSkeleton sk;
	sk.Name       = OaString(InName);
	sk.SkeletonId = InId;
	for (OaI32 i = 0; i < kJointCount; ++i) {
		const BuiltinJoint& b = kMetaHumanBody[i];
		OaSkelJoint j;
		// HumanIK-named build uses the slot name when one exists, else the UE name.
		j.Name             = (InUseHikNames && b.HikSlot[0] != '\0') ? b.HikSlot : b.Name;
		j.HumanIkId        = b.HumanIkId;
		j.Mass             = b.Mass;
		j.Rest.Translate   = { b.Tx, b.Ty, b.Tz };
		j.Rest.JointOrient = OaEulerXyzDegToQuat({ b.Rx, b.Ry, b.Rz });
		j.Rest.Rotate      = { 0.0f, 0.0f, 0.0f, 1.0f };
		j.HasTranslate     = b.HasTranslate;
		j.RotDof           = b.RotDof;
		// Resolve the parent against the table (UE names), not the possibly-renamed
		// joints already pushed — otherwise HumanIK-named builds can't find parents.
		j.ParentIndex      = (b.Parent[0] == '\0') ? -1 : TableIndexOf(b.Parent);
		sk.Joints.PushBack(std::move(j));
	}
	// Contacts reference the foot joints by their (possibly renamed) slot.
	sk.ContactJoints.PushBack(sk.IndexOf(InUseHikNames ? "LeftFoot" : "foot_l"));
	sk.ContactJoints.PushBack(sk.IndexOf(InUseHikNames ? "RightFoot" : "foot_r"));
	return sk;
}

// Build the HumanML3D / SMPL 22-joint body (SkeletonId 2) from kHumanMl3dBody.
// Same BuiltinJoint shape as the MetaHuman table; the rest offsets are 0 (see
// SkHumanMl3d.inc) — the USD preview supplies per-frame world positions.
OaSkeleton BuildHumanMl3d() {
	constexpr OaI32 n = static_cast<OaI32>(sizeof(kHumanMl3dBody) / sizeof(kHumanMl3dBody[0]));
	auto tableIndexOf = [](OaStringView InName) -> OaI32 {
		for (OaI32 k = 0; k < n; ++k) {
			if (InName == kHumanMl3dBody[k].Name) { return k; }
		}
		return -1;
	};
	OaSkeleton sk;
	sk.Name       = "humanml3d_body";
	sk.SkeletonId = 2u;
	for (OaI32 i = 0; i < n; ++i) {
		const BuiltinJoint& b = kHumanMl3dBody[i];
		OaSkelJoint j;
		j.Name             = b.Name;
		j.HumanIkId        = b.HumanIkId;
		j.Mass             = b.Mass;
		j.Rest.Translate   = { b.Tx, b.Ty, b.Tz };
		j.Rest.JointOrient = OaEulerXyzDegToQuat({ b.Rx, b.Ry, b.Rz });
		j.Rest.Rotate      = { 0.0f, 0.0f, 0.0f, 1.0f };
		j.HasTranslate     = b.HasTranslate;
		j.RotDof           = b.RotDof;
		j.ParentIndex      = (b.Parent[0] == '\0') ? -1 : tableIndexOf(b.Parent);
		sk.Joints.PushBack(std::move(j));
	}
	sk.ContactJoints.PushBack(sk.IndexOf("left_foot"));
	sk.ContactJoints.PushBack(sk.IndexOf("right_foot"));
	return sk;
}

} // namespace

const OaSkeleton& OaSkMetaHuman() {
	static const OaSkeleton kSkeleton = BuildBody("metahuman_body", 0u, /*hik=*/false);
	return kSkeleton;
}

const OaSkeleton& OaSkHumanIk() {
	static const OaSkeleton kSkeleton = BuildBody("humanik", 1u, /*hik=*/true);
	return kSkeleton;
}

const OaSkeleton& OaSkHumanMl3d() {
	static const OaSkeleton kSkeleton = BuildHumanMl3d();
	return kSkeleton;
}

// ─── Queries ─────────────────────────────────────────────────────────────────

OaI32 OaSkeleton::IndexOf(OaStringView InName) const noexcept {
	for (OaI32 i = 0; i < JointCount(); ++i) {
		if (Joints[static_cast<OaUsize>(i)].Name == InName) {
			return i;
		}
	}
	return -1;
}

namespace {

// Forward-kinematics walk over the rest OaJoints: returns world position and
// (optionally) world orientation of InJoint. Translate of joint k is expressed
// in its parent's frame, rotated by the parent's accumulated world rotation;
// orientation accumulates the rest JointOrients (animated Rotate is identity at
// rest). Mirrors the quaternion FK in OaPosePack::Pack.
void RestFk(const OaVec<OaSkelJoint>& InJoints, OaI32 InJoint, VlmVec3& OutPos, VlmQuat& OutRot) {
	const OaI32 n = static_cast<OaI32>(InJoints.Size());
	// Chain root → InJoint.
	OaVec<OaI32> chain;
	for (OaI32 cur = InJoint; cur >= 0 && cur < n; cur = InJoints[static_cast<OaUsize>(cur)].ParentIndex) {
		chain.PushBack(cur);
	}
	VlmVec3 pos = { 0.0f, 0.0f, 0.0f };
	VlmQuat rot = { 0.0f, 0.0f, 0.0f, 1.0f };
	for (OaI32 i = static_cast<OaI32>(chain.Size()) - 1; i >= 0; --i) {
		const OaSkelJoint& j = InJoints[static_cast<OaUsize>(chain[static_cast<OaUsize>(i)])];
		pos = Vlm::Add(pos, Vlm::RotateVector(rot, j.Rest.Translate));
		rot = VlmQuatMul(rot, j.Rest.OrientedRotation());
	}
	OutPos = pos;
	OutRot = rot;
}

} // namespace

VlmVec3 OaSkeleton::RestWorld(OaI32 InJoint) const noexcept {
	VlmVec3 pos = {}; VlmQuat rot = {};
	RestFk(Joints, InJoint, pos, rot);
	return pos;
}

VlmQuat OaSkeleton::RestWorldRotation(OaI32 InJoint) const noexcept {
	VlmVec3 pos = {}; VlmQuat rot = {};
	RestFk(Joints, InJoint, pos, rot);
	return rot;
}

bool OaSkeleton::IsValid() const noexcept {
	if (Joints.Size() == 0) {
		return false;
	}
	if (Joints[0].ParentIndex != -1) {
		return false; // root must be index 0
	}
	for (OaI32 i = 0; i < JointCount(); ++i) {
		const OaI32 p = Joints[static_cast<OaUsize>(i)].ParentIndex;
		if (i == 0) {
			continue;
		}
		if (p < 0 || p >= i) {
			return false; // every parent must precede its child
		}
	}
	for (OaI32 c : ContactJoints) {
		if (c < 0 || c >= JointCount()) {
			return false;
		}
	}
	return true;
}

// ─── `.skel` JSON IO ─────────────────────────────────────────────────────────
// Hand-rolled emitter + tolerant scoped parser. The schema is fixed and small,
// so we avoid a general JSON dependency (there is none in-tree) while keeping
// the file standard JSON that opens in any viewer. v2 stores the rest
// orientation quaternion alongside the local offset.

OaStatus OaSkeleton::WriteSkel(const OaPath& InPath) const {
	if (!IsValid()) {
		return OaStatus::InvalidArgument("OaSkeleton::WriteSkel: invalid skeleton");
	}

	std::ostringstream out;
	out << std::setprecision(9);
	out << "{\n";
	out << "  \"format\": \"oa.skel\",\n";
	out << "  \"version\": " << FormatVersion << ",\n";
	out << "  \"name\": \"" << Name << "\",\n";
	out << "  \"skeletonId\": " << SkeletonId << ",\n";

	out << "  \"contactJoints\": [";
	for (OaUsize i = 0; i < ContactJoints.Size(); ++i) {
		out << (i ? ", " : "") << ContactJoints[i];
	}
	out << "],\n";

	out << "  \"joints\": [\n";
	for (OaI32 i = 0; i < JointCount(); ++i) {
		const OaSkelJoint& j = Joints[static_cast<OaUsize>(i)];
		const VlmVec3& t = j.Rest.Translate;
		const VlmQuat& q = j.Rest.JointOrient;
		out << "    { \"name\": \"" << j.Name << "\""
			<< ", \"parent\": " << j.ParentIndex
			<< ", \"humanIkId\": " << j.HumanIkId
			<< ", \"rest\": [" << t.X << ", " << t.Y << ", " << t.Z << "]"
			<< ", \"orient\": [" << q.X << ", " << q.Y << ", " << q.Z << ", " << q.W << "]"
			<< ", \"mass\": " << j.Mass
			<< ", \"hasTranslate\": " << (j.HasTranslate ? 1 : 0)
			<< ", \"rotDof\": " << static_cast<int>(j.RotDof)
			<< " }" << (i + 1 < JointCount() ? "," : "") << "\n";
	}
	out << "  ]\n";
	out << "}\n";

	return OaFileIo::WriteText(InPath, OaString(out.str()));
}

namespace {

// Minimal cursor over the JSON text. Tolerant of whitespace; understands only
// what WriteSkel emits (objects, arrays, strings, numbers).
struct JsonCursor {
	const char* P;
	const char* End;

	void SkipWs() {
		while (P < End && (std::isspace(static_cast<unsigned char>(*P)) || *P == ',')) {
			++P;
		}
	}
	bool Eat(char c) {
		SkipWs();
		if (P < End && *P == c) { ++P; return true; }
		return false;
	}
	bool Peek(char c) {
		SkipWs();
		return P < End && *P == c;
	}
	bool Key(OaStringView InKey) {
		SkipWs();
		if (P >= End || *P != '"') { return false; }
		const char* save = P;
		OaString k;
		if (!Str(k)) { P = save; return false; }
		if (!Eat(':')) { P = save; return false; }
		if (k != InKey) { P = save; return false; }
		return true;
	}
	bool Str(OaString& Out) {
		SkipWs();
		if (P >= End || *P != '"') { return false; }
		++P;
		Out.Clear();
		while (P < End && *P != '"') {
			if (*P == '\\' && P + 1 < End) { ++P; }
			Out.PushBack(*P++);
		}
		if (P >= End) { return false; }
		++P; // closing quote
		return true;
	}
	bool Num(double& Out) {
		SkipWs();
		char* fin = nullptr;
		Out = std::strtod(P, &fin);
		if (fin == P) { return false; }
		P = fin;
		return true;
	}
};

} // namespace

OaResult<OaSkeleton> OaSkeleton::ReadSkel(const OaPath& InPath) {
	auto textResult = OaFileIo::ReadText(InPath);
	if (!textResult.IsOk()) {
		return textResult.GetStatus();
	}
	const OaString& text = *textResult;

	JsonCursor c{ text.Data(), text.Data() + text.Size() };
	OaSkeleton sk;

	if (!c.Eat('{')) {
		return OaStatus::InvalidArgument("OaSkeleton::ReadSkel: expected object");
	}

	while (!c.Peek('}')) {
		c.SkipWs();
		if (c.P >= c.End) {
			return OaStatus::InvalidArgument("OaSkeleton::ReadSkel: unexpected end");
		}

		double num = 0.0;
		OaString str;
		if (c.Key("name")) {
			if (!c.Str(sk.Name)) {
				return OaStatus::InvalidArgument("OaSkeleton::ReadSkel: bad name");
			}
		} else if (c.Key("skeletonId")) {
			if (!c.Num(num)) {
				return OaStatus::InvalidArgument("OaSkeleton::ReadSkel: bad skeletonId");
			}
			sk.SkeletonId = static_cast<OaU32>(num);
		} else if (c.Key("contactJoints")) {
			if (!c.Eat('[')) {
				return OaStatus::InvalidArgument("OaSkeleton::ReadSkel: bad contactJoints");
			}
			while (!c.Peek(']')) {
				if (!c.Num(num)) {
					return OaStatus::InvalidArgument("OaSkeleton::ReadSkel: bad contact index");
				}
				sk.ContactJoints.PushBack(static_cast<OaI32>(num));
			}
			c.Eat(']');
		} else if (c.Key("joints")) {
			if (!c.Eat('[')) {
				return OaStatus::InvalidArgument("OaSkeleton::ReadSkel: bad joints array");
			}
			while (!c.Peek(']')) {
				if (!c.Eat('{')) {
					return OaStatus::InvalidArgument("OaSkeleton::ReadSkel: bad joint object");
				}
				OaSkelJoint j;
				j.Rest.Rotate = { 0.0f, 0.0f, 0.0f, 1.0f };
				while (!c.Peek('}')) {
					if (c.Key("name")) {
						if (!c.Str(j.Name)) {
							return OaStatus::InvalidArgument("OaSkeleton::ReadSkel: bad joint name");
						}
					} else if (c.Key("parent")) {
						if (!c.Num(num)) { return OaStatus::InvalidArgument("OaSkeleton::ReadSkel: bad parent"); }
						j.ParentIndex = static_cast<OaI32>(num);
					} else if (c.Key("humanIkId")) {
						if (!c.Num(num)) { return OaStatus::InvalidArgument("OaSkeleton::ReadSkel: bad humanIkId"); }
						j.HumanIkId = static_cast<OaI32>(num);
					} else if (c.Key("mass")) {
						if (!c.Num(num)) { return OaStatus::InvalidArgument("OaSkeleton::ReadSkel: bad mass"); }
						j.Mass = static_cast<OaF32>(num);
					} else if (c.Key("hasTranslate")) {
						if (!c.Num(num)) { return OaStatus::InvalidArgument("OaSkeleton::ReadSkel: bad hasTranslate"); }
						j.HasTranslate = (num != 0.0);
					} else if (c.Key("rotDof")) {
						if (!c.Num(num)) { return OaStatus::InvalidArgument("OaSkeleton::ReadSkel: bad rotDof"); }
						j.RotDof = static_cast<OaU8>(num);
					} else if (c.Key("rest")) {
						if (!c.Eat('[')) { return OaStatus::InvalidArgument("OaSkeleton::ReadSkel: bad rest"); }
						double v[3] = { 0, 0, 0 };
						for (double& vi : v) {
							if (!c.Num(vi)) { return OaStatus::InvalidArgument("OaSkeleton::ReadSkel: bad rest value"); }
						}
						c.Eat(']');
						j.Rest.Translate = { static_cast<OaF32>(v[0]), static_cast<OaF32>(v[1]), static_cast<OaF32>(v[2]) };
					} else if (c.Key("orient")) {
						if (!c.Eat('[')) { return OaStatus::InvalidArgument("OaSkeleton::ReadSkel: bad orient"); }
						double v[4] = { 0, 0, 0, 1 };
						for (double& vi : v) {
							if (!c.Num(vi)) { return OaStatus::InvalidArgument("OaSkeleton::ReadSkel: bad orient value"); }
						}
						c.Eat(']');
						j.Rest.JointOrient = { static_cast<OaF32>(v[0]), static_cast<OaF32>(v[1]),
						                       static_cast<OaF32>(v[2]), static_cast<OaF32>(v[3]) };
					} else {
						return OaStatus::InvalidArgument("OaSkeleton::ReadSkel: unknown joint key");
					}
				}
				c.Eat('}');
				sk.Joints.PushBack(std::move(j));
			}
			c.Eat(']');
		} else if (c.Key("format") || c.Key("version")) {
			if (!c.Str(str) && !c.Num(num)) {
				return OaStatus::InvalidArgument("OaSkeleton::ReadSkel: bad header value");
			}
		} else {
			return OaStatus::InvalidArgument("OaSkeleton::ReadSkel: unknown top-level key");
		}
	}

	if (!sk.IsValid()) {
		return OaStatus::InvalidArgument("OaSkeleton::ReadSkel: invalid skeleton");
	}
	return sk;
}

// ─── Skeleton-agnostic USD bridge ────────────────────────────────────────────

namespace {

// Full UsdSkel path of a joint: "pelvis/left_hip/left_knee/..." (root → joint).
OaString JointPathFor(const OaSkeleton& InSkel, OaI32 InJoint) {
	OaVec<OaI32> chain;
	for (OaI32 cur = InJoint; cur >= 0 && cur < InSkel.JointCount();
	     cur = InSkel.Joints[static_cast<OaUsize>(cur)].ParentIndex) {
		chain.PushBack(cur);
	}
	OaString out;
	for (OaI32 i = static_cast<OaI32>(chain.Size()) - 1; i >= 0; --i) {
		if (!out.Empty()) { out += "/"; }
		out += InSkel.Joints[static_cast<OaUsize>(chain[static_cast<OaUsize>(i)])].Name;
	}
	return out;
}

VlmMat4 TranslateMat4(VlmVec3 InT) {
	VlmMat4 m = VlmMat4::Identity();
	m.M[3][0] = InT.X;
	m.M[3][1] = InT.Y;
	m.M[3][2] = InT.Z;
	return m;
}

} // namespace

OaUsdSkelClip OaUsdClipFromWorldJoints(const OaSkeleton& InSkel,
                                       OaSpan<const OaF32> InWorldXyz,
                                       OaI32 InFrames, OaF32 InFps,
                                       OaI32 InUpAxis, OaF32 InScale) {
	const OaI32 j = InSkel.JointCount();
	OaUsdSkelClip clip;
	clip.FrameCount = static_cast<OaU32>(InFrames < 0 ? 0 : InFrames);
	clip.Fps        = InFps;
	clip.UpAxis     = InUpAxis;

	for (OaI32 k = 0; k < j; ++k) {
		clip.JointPaths.PushBack(JointPathFor(InSkel, k));
	}

	auto world = [&](OaI32 InFrame, OaI32 InJoint) -> VlmVec3 {
		const OaI64 base = (static_cast<OaI64>(InFrame) * j + InJoint) * 3;
		return { InWorldXyz[static_cast<OaUsize>(base + 0)] * InScale,
		         InWorldXyz[static_cast<OaUsize>(base + 1)] * InScale,
		         InWorldXyz[static_cast<OaUsize>(base + 2)] * InScale };
	};

	// Bind = frame-0 world; rest = frame-0 local offset to parent.
	for (OaI32 k = 0; k < j; ++k) {
		const VlmVec3 w = world(0, k);
		clip.BindTransforms.PushBack(TranslateMat4(w));
		const OaI32 parent = InSkel.Joints[static_cast<OaUsize>(k)].ParentIndex;
		const VlmVec3 local = parent >= 0
			? VlmVec3{ w.X - world(0, parent).X, w.Y - world(0, parent).Y, w.Z - world(0, parent).Z }
			: w;
		clip.RestTransforms.PushBack(TranslateMat4(local));
	}

	clip.Translations.Resize(static_cast<OaUsize>(InFrames) * static_cast<OaUsize>(j));
	clip.Rotations.Resize(static_cast<OaUsize>(InFrames) * static_cast<OaUsize>(j));
	for (OaI32 f = 0; f < InFrames; ++f) {
		for (OaI32 k = 0; k < j; ++k) {
			const OaI32 parent = InSkel.Joints[static_cast<OaUsize>(k)].ParentIndex;
			const VlmVec3 w = world(f, k);
			const VlmVec3 local = parent >= 0
				? VlmVec3{ w.X - world(f, parent).X, w.Y - world(f, parent).Y, w.Z - world(f, parent).Z }
				: w;
			const OaUsize idx = static_cast<OaUsize>(f) * static_cast<OaUsize>(j) + static_cast<OaUsize>(k);
			clip.Translations[idx] = local;
			clip.Rotations[idx]    = Vlm::QuaternionIdentity();
		}
	}
	return clip;
}
