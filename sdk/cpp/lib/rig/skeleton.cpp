#include <rig/skeleton.h>
#include <core/streamText.h>
#include <rig/skeletonUsd.h>

#include <core/transform.h>
#include <oa/core/vlm.h>

#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <sstream>

// ─── Built-in skeleton tables ────────────────────────────────────────────────

namespace {

#include "skMetaHuman.inc"
#include "skHumanMl3d.inc"

constexpr oa::I32 kJointCount = static_cast<oa::I32>(sizeof(kMetaHumanBody) / sizeof(kMetaHumanBody[0]));

// index of a UE bone name in the builtin table (== joint index, since joints are
// added in table order). Renaming-independent, unlike oa::Skeleton::indexOf.
oa::I32 tableIndexOf(oa::StringView inName) {
	for (oa::I32 k = 0; k < kJointCount; ++k) {
		if (inName == kMetaHumanBody[k].name) {
			return k;
		}
	}
	return -1;
}

// Build a skeleton from the manny table. `inUseHikNames` swaps each joint's UE
// bone name for its HumanIK slot name (geometry/hierarchy unchanged).
oa::Skeleton buildBody(oa::StringView inName, oa::U32 inId, bool inUseHikNames) {
	oa::Skeleton sk;
	sk.name       = oa::String(inName);
	sk.skeletonId = inId;
	for (oa::I32 i = 0; i < kJointCount; ++i) {
		const BuiltinJoint& b = kMetaHumanBody[i];
		oa::SkelJoint j;
		// HumanIK-named build uses the slot name when one exists, else the UE name.
		j.name             = (inUseHikNames && b.hikSlot[0] != '\0') ? b.hikSlot : b.name;
		j.humanIkId        = b.humanIkId;
		j.mass             = b.mass;
		j.rest.translate   = { b.tx, b.ty, b.tz };
		j.rest.jointOrient = oa::eulerXyzDegToQuat({ b.rx, b.ry, b.rz });
		j.rest.rotate      = { 0.0f, 0.0f, 0.0f, 1.0f };
		j.hasTranslate     = b.hasTranslate;
		j.rotDof           = b.rotDof;
		// Resolve the parent against the table (UE names), not the possibly-renamed
		// joints already pushed — otherwise HumanIK-named builds can't find parents.
		j.parentIndex      = (b.parent[0] == '\0') ? -1 : tableIndexOf(b.parent);
		sk.joints.pushBack(std::move(j));
	}
	// Contacts reference the foot joints by their (possibly renamed) slot.
	sk.contactJoints.pushBack(sk.indexOf(inUseHikNames ? "LeftFoot" : "foot_l"));
	sk.contactJoints.pushBack(sk.indexOf(inUseHikNames ? "RightFoot" : "foot_r"));
	return sk;
}

// Build the HumanML3D / SMPL 22-joint body (skeletonId 2) from kHumanMl3dBody.
// Same BuiltinJoint shape as the MetaHuman table; the rest offsets are 0 (see
// SkHumanMl3d.inc) — the USD preview supplies per-frame world positions.
oa::Skeleton buildHumanMl3d() {
	constexpr oa::I32 n = static_cast<oa::I32>(sizeof(kHumanMl3dBody) / sizeof(kHumanMl3dBody[0]));
	auto tableIndexOf = [](oa::StringView inName) -> oa::I32 {
		for (oa::I32 k = 0; k < n; ++k) {
			if (inName == kHumanMl3dBody[k].name) { return k; }
		}
		return -1;
	};
	oa::Skeleton sk;
	sk.name       = "humanml3d_body";
	sk.skeletonId = 2u;
	for (oa::I32 i = 0; i < n; ++i) {
		const BuiltinJoint& b = kHumanMl3dBody[i];
		oa::SkelJoint j;
		j.name             = b.name;
		j.humanIkId        = b.humanIkId;
		j.mass             = b.mass;
		j.rest.translate   = { b.tx, b.ty, b.tz };
		j.rest.jointOrient = oa::eulerXyzDegToQuat({ b.rx, b.ry, b.rz });
		j.rest.rotate      = { 0.0f, 0.0f, 0.0f, 1.0f };
		j.hasTranslate     = b.hasTranslate;
		j.rotDof           = b.rotDof;
		j.parentIndex      = (b.parent[0] == '\0') ? -1 : tableIndexOf(b.parent);
		sk.joints.pushBack(std::move(j));
	}
	sk.contactJoints.pushBack(sk.indexOf("left_foot"));
	sk.contactJoints.pushBack(sk.indexOf("right_foot"));
	return sk;
}

} // namespace

const oa::Skeleton& oa::skMetaHuman() {
	static const oa::Skeleton kSkeleton = buildBody("metahuman_body", 0u, /*hik=*/false);
	return kSkeleton;
}

const oa::Skeleton& oa::skHumanIk() {
	static const oa::Skeleton kSkeleton = buildBody("humanik", 1u, /*hik=*/true);
	return kSkeleton;
}

const oa::Skeleton& oa::skHumanMl3d() {
	static const oa::Skeleton kSkeleton = buildHumanMl3d();
	return kSkeleton;
}

// ─── Queries ─────────────────────────────────────────────────────────────────

oa::I32 oa::Skeleton::indexOf(oa::StringView inName) const noexcept {
	for (oa::I32 i = 0; i < jointCount(); ++i) {
		if (joints[static_cast<oa::Usize>(i)].name == inName) {
			return i;
		}
	}
	return -1;
}

namespace {

// forward-kinematics walk over the rest OaJoints: returns world position and
// (optionally) world orientation of inJoint. translate of joint k is expressed
// in its parent's frame, rotated by the parent's accumulated world rotation;
// orientation accumulates the rest jointOrients (animated rotate is identity at
// rest). Mirrors the quaternion FK in oa::PosePack::Pack.
void restFk(
	const oa::Vector<oa::SkelJoint>& inJoints,
	oa::I32 inJoint,
	oa::vlm::Vec3& outPos,
	oa::vlm::Quat& outRot) {
	const oa::I32 n = static_cast<oa::I32>(inJoints.size());
	// Chain root → inJoint.
	oa::Vector<oa::I32> chain;
	for (oa::I32 cur = inJoint; cur >= 0 && cur < n; cur = inJoints[static_cast<oa::Usize>(cur)].parentIndex) {
		chain.pushBack(cur);
	}
	oa::vlm::Vec3 pos = { 0.0f, 0.0f, 0.0f };
	oa::vlm::Quat rot = { 0.0f, 0.0f, 0.0f, 1.0f };
	for (oa::I32 i = static_cast<oa::I32>(chain.size()) - 1; i >= 0; --i) {
		const oa::SkelJoint& j = inJoints[static_cast<oa::Usize>(chain[static_cast<oa::Usize>(i)])];
		pos = oa::vlm::add(pos, oa::vlm::rotateVector(rot, j.rest.translate));
		rot = rot * j.rest.orientedRotation();
	}
	outPos = pos;
	outRot = rot;
}

} // namespace

oa::vlm::Vec3 oa::Skeleton::restWorld(oa::I32 inJoint) const noexcept {
	oa::vlm::Vec3 pos = {}; oa::vlm::Quat rot = {};
	restFk(joints, inJoint, pos, rot);
	return pos;
}

oa::vlm::Quat oa::Skeleton::restWorldRotation(oa::I32 inJoint) const noexcept {
	oa::vlm::Vec3 pos = {}; oa::vlm::Quat rot = {};
	restFk(joints, inJoint, pos, rot);
	return rot;
}

bool oa::Skeleton::isValid() const noexcept {
	if (joints.size() == 0) {
		return false;
	}
	if (joints[0].parentIndex != -1) {
		return false; // root must be index 0
	}
	for (oa::I32 i = 0; i < jointCount(); ++i) {
		const oa::I32 p = joints[static_cast<oa::Usize>(i)].parentIndex;
		if (i == 0) {
			continue;
		}
		if (p < 0 || p >= i) {
			return false; // every parent must precede its child
		}
	}
	for (oa::I32 c : contactJoints) {
		if (c < 0 || c >= jointCount()) {
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

oa::Status oa::Skeleton::writeSkel(const oa::Path& inPath) const {
	if (!isValid()) {
		return oa::Status::invalidArgument("oa::Skeleton::writeSkel: invalid skeleton");
	}

	std::ostringstream out;
	out << std::setprecision(9);
	out << "{\n";
	out << "  \"format\": \"oa.skel\",\n";
	out << "  \"version\": " << formatVersion << ",\n";
	out << "  \"name\": \"";
	out.write(name.data(), static_cast<std::streamsize>(name.size()));
	out << "\",\n";
	out << "  \"skeletonId\": " << skeletonId << ",\n";

	out << "  \"contactJoints\": [";
	for (oa::Usize i = 0; i < contactJoints.size(); ++i) {
		out << (i ? ", " : "") << contactJoints[i];
	}
	out << "],\n";

	out << "  \"joints\": [\n";
	for (oa::I32 i = 0; i < jointCount(); ++i) {
		const oa::SkelJoint& j = joints[static_cast<oa::Usize>(i)];
		const oa::vlm::Vec3& t = j.rest.translate;
		const oa::vlm::Quat& q = j.rest.jointOrient;
		out << "    { \"name\": \"";
		out.write(j.name.data(), static_cast<std::streamsize>(j.name.size()));
		out << "\"" << ", \"parent\": " << j.parentIndex
			<< ", \"humanIkId\": " << j.humanIkId
			<< ", \"rest\": [" << t.x << ", " << t.y << ", " << t.z << "]"
			<< ", \"orient\": [" << q.x << ", " << q.y << ", " << q.z << ", " << q.w << "]"
			<< ", \"mass\": " << j.mass
			<< ", \"hasTranslate\": " << (j.hasTranslate ? 1 : 0)
			<< ", \"rotDof\": " << static_cast<int>(j.rotDof)
			<< " }" << (i + 1 < jointCount() ? "," : "") << "\n";
	}
	out << "  ]\n";
	out << "}\n";

	return oa::Filesystem::writeText(inPath, oa::sdk::fromStdString(out.str()));
}

namespace {

// Minimal cursor over the JSON text. Tolerant of whitespace; understands only
// what writeSkel emits (objects, arrays, strings, numbers).
struct JsonCursor {
	const char* p;
	const char* end;

	void skipWs() {
		while (p < end && (std::isspace(static_cast<unsigned char>(*p)) || *p == ',')) {
			++p;
		}
	}
	bool eat(char c) {
		skipWs();
		if (p < end && *p == c) { ++p; return true; }
		return false;
	}
	bool peek(char c) {
		skipWs();
		return p < end && *p == c;
	}
	bool key(oa::StringView inKey) {
		skipWs();
		if (p >= end || *p != '"') { return false; }
		const char* save = p;
		oa::String k;
		if (!str(k)) { p = save; return false; }
		if (!eat(':')) { p = save; return false; }
		if (k != inKey) { p = save; return false; }
		return true;
	}
	bool str(oa::String& out) {
		skipWs();
		if (p >= end || *p != '"') { return false; }
		++p;
		out.clear();
		while (p < end && *p != '"') {
			if (*p == '\\' && p + 1 < end) { ++p; }
			out.pushBack(*p++);
		}
		if (p >= end) { return false; }
		++p; // closing quote
		return true;
	}
	bool num(double& out) {
		skipWs();
		char* fin = nullptr;
		out = std::strtod(p, &fin);
		if (fin == p) { return false; }
		p = fin;
		return true;
	}
};

} // namespace

oa::Result<oa::Skeleton> oa::Skeleton::readSkel(const oa::Path& inPath) {
	auto textResult = oa::Filesystem::readText(inPath);
	if (!textResult.isOk()) {
		return textResult.getStatus();
	}
	const oa::String& text = *textResult;

	JsonCursor c{ text.data(), text.data() + text.size() };
	oa::Skeleton sk;

	if (!c.eat('{')) {
		return oa::Status::invalidArgument("oa::Skeleton::ReadSkel: expected object");
	}

	while (!c.peek('}')) {
		c.skipWs();
		if (c.p >= c.end) {
			return oa::Status::invalidArgument("oa::Skeleton::ReadSkel: unexpected end");
		}

		double num = 0.0;
		oa::String str;
		if (c.key("name")) {
			if (!c.str(sk.name)) {
				return oa::Status::invalidArgument("oa::Skeleton::ReadSkel: bad name");
			}
		} else if (c.key("skeletonId")) {
			if (!c.num(num)) {
				return oa::Status::invalidArgument("oa::Skeleton::ReadSkel: bad skeletonId");
			}
			sk.skeletonId = static_cast<oa::U32>(num);
		} else if (c.key("contactJoints")) {
			if (!c.eat('[')) {
				return oa::Status::invalidArgument("oa::Skeleton::ReadSkel: bad contactJoints");
			}
			while (!c.peek(']')) {
				if (!c.num(num)) {
					return oa::Status::invalidArgument("oa::Skeleton::ReadSkel: bad contact index");
				}
				sk.contactJoints.pushBack(static_cast<oa::I32>(num));
			}
			c.eat(']');
		} else if (c.key("joints")) {
			if (!c.eat('[')) {
				return oa::Status::invalidArgument("oa::Skeleton::ReadSkel: bad joints array");
			}
			while (!c.peek(']')) {
				if (!c.eat('{')) {
					return oa::Status::invalidArgument("oa::Skeleton::ReadSkel: bad joint object");
				}
				oa::SkelJoint j;
				j.rest.rotate = { 0.0f, 0.0f, 0.0f, 1.0f };
				while (!c.peek('}')) {
					if (c.key("name")) {
						if (!c.str(j.name)) {
							return oa::Status::invalidArgument("oa::Skeleton::ReadSkel: bad joint name");
						}
					} else if (c.key("parent")) {
						if (!c.num(num)) { return oa::Status::invalidArgument("oa::Skeleton::ReadSkel: bad parent"); }
						j.parentIndex = static_cast<oa::I32>(num);
					} else if (c.key("humanIkId")) {
						if (!c.num(num)) { return oa::Status::invalidArgument("oa::Skeleton::ReadSkel: bad humanIkId"); }
						j.humanIkId = static_cast<oa::I32>(num);
					} else if (c.key("mass")) {
						if (!c.num(num)) { return oa::Status::invalidArgument("oa::Skeleton::ReadSkel: bad mass"); }
						j.mass = static_cast<oa::F32>(num);
					} else if (c.key("hasTranslate")) {
						if (!c.num(num)) { return oa::Status::invalidArgument("oa::Skeleton::ReadSkel: bad hasTranslate"); }
						j.hasTranslate = (num != 0.0);
					} else if (c.key("rotDof")) {
						if (!c.num(num)) { return oa::Status::invalidArgument("oa::Skeleton::ReadSkel: bad rotDof"); }
						j.rotDof = static_cast<oa::U8>(num);
					} else if (c.key("rest")) {
						if (!c.eat('[')) { return oa::Status::invalidArgument("oa::Skeleton::ReadSkel: bad rest"); }
						double v[3] = { 0, 0, 0 };
						for (double& vi : v) {
							if (!c.num(vi)) { return oa::Status::invalidArgument("oa::Skeleton::ReadSkel: bad rest value"); }
						}
						c.eat(']');
						j.rest.translate = { static_cast<oa::F32>(v[0]), static_cast<oa::F32>(v[1]), static_cast<oa::F32>(v[2]) };
					} else if (c.key("orient")) {
						if (!c.eat('[')) { return oa::Status::invalidArgument("oa::Skeleton::ReadSkel: bad orient"); }
						double v[4] = { 0, 0, 0, 1 };
						for (double& vi : v) {
							if (!c.num(vi)) { return oa::Status::invalidArgument("oa::Skeleton::ReadSkel: bad orient value"); }
						}
						c.eat(']');
						j.rest.jointOrient = { static_cast<oa::F32>(v[0]), static_cast<oa::F32>(v[1]),
						                       static_cast<oa::F32>(v[2]), static_cast<oa::F32>(v[3]) };
					} else {
						return oa::Status::invalidArgument("oa::Skeleton::ReadSkel: unknown joint key");
					}
				}
				c.eat('}');
				sk.joints.pushBack(std::move(j));
			}
			c.eat(']');
		} else if (c.key("format") || c.key("version")) {
			if (!c.str(str) && !c.num(num)) {
				return oa::Status::invalidArgument("oa::Skeleton::ReadSkel: bad header value");
			}
		} else {
			return oa::Status::invalidArgument("oa::Skeleton::ReadSkel: unknown top-level key");
		}
	}

	if (!sk.isValid()) {
		return oa::Status::invalidArgument("oa::Skeleton::ReadSkel: invalid skeleton");
	}
	return sk;
}

// ─── Skeleton-agnostic USD bridge ────────────────────────────────────────────

namespace {

// Full UsdSkel path of a joint: "pelvis/left_hip/left_knee/..." (root → joint).
oa::String jointPathFor(const oa::Skeleton& inSkel, oa::I32 inJoint) {
	oa::Vector<oa::I32> chain;
	for (oa::I32 cur = inJoint; cur >= 0 && cur < inSkel.jointCount();
	     cur = inSkel.joints[static_cast<oa::Usize>(cur)].parentIndex) {
		chain.pushBack(cur);
	}
	oa::String out;
	for (oa::I32 i = static_cast<oa::I32>(chain.size()) - 1; i >= 0; --i) {
		if (!out.empty()) { out += "/"; }
		out += inSkel.joints[static_cast<oa::Usize>(chain[static_cast<oa::Usize>(i)])].name;
	}
	return out;
}

oa::vlm::Mat4 translateMat4(oa::vlm::Vec3 inT) {
	oa::vlm::Mat4 m = oa::vlm::Mat4::identity();
	m.m[3][0] = inT.x;
	m.m[3][1] = inT.y;
	m.m[3][2] = inT.z;
	return m;
}

} // namespace

oa::UsdSkelClip oa::usdClipFromWorldJoints(const oa::Skeleton& inSkel,
                                       oa::Span<const oa::F32> inWorldXyz,
                                       oa::I32 inFrames, oa::F32 inFps,
                                       oa::I32 inUpAxis, oa::F32 inScale) {
	const oa::I32 j = inSkel.jointCount();
	oa::UsdSkelClip clip;
	clip.frameCount = static_cast<oa::U32>(inFrames < 0 ? 0 : inFrames);
	clip.fps        = inFps;
	clip.upAxis     = inUpAxis;

	for (oa::I32 k = 0; k < j; ++k) {
		clip.jointPaths.pushBack(jointPathFor(inSkel, k));
	}

	auto world = [&](oa::I32 inFrame, oa::I32 inJoint) -> oa::vlm::Vec3 {
		const oa::I64 base = (static_cast<oa::I64>(inFrame) * j + inJoint) * 3;
		return { inWorldXyz[static_cast<oa::Usize>(base + 0)] * inScale,
		         inWorldXyz[static_cast<oa::Usize>(base + 1)] * inScale,
		         inWorldXyz[static_cast<oa::Usize>(base + 2)] * inScale };
	};

	// Bind = frame-0 world; rest = frame-0 local offset to parent.
	for (oa::I32 k = 0; k < j; ++k) {
		const oa::vlm::Vec3 w = world(0, k);
		clip.bindTransforms.pushBack(translateMat4(w));
		const oa::I32 parent = inSkel.joints[static_cast<oa::Usize>(k)].parentIndex;
		const oa::vlm::Vec3 local = parent >= 0
			? oa::vlm::Vec3{ w.x - world(0, parent).x, w.y - world(0, parent).y, w.z - world(0, parent).z }
			: w;
		clip.restTransforms.pushBack(translateMat4(local));
	}

	clip.translations.resize(static_cast<oa::Usize>(inFrames) * static_cast<oa::Usize>(j));
	clip.rotations.resize(static_cast<oa::Usize>(inFrames) * static_cast<oa::Usize>(j));
	for (oa::I32 f = 0; f < inFrames; ++f) {
		for (oa::I32 k = 0; k < j; ++k) {
			const oa::I32 parent = inSkel.joints[static_cast<oa::Usize>(k)].parentIndex;
			const oa::vlm::Vec3 w = world(f, k);
			const oa::vlm::Vec3 local = parent >= 0
				? oa::vlm::Vec3{ w.x - world(f, parent).x, w.y - world(f, parent).y, w.z - world(f, parent).z }
				: w;
			const oa::Usize idx = static_cast<oa::Usize>(f) * static_cast<oa::Usize>(j) + static_cast<oa::Usize>(k);
			clip.translations[idx] = local;
			clip.rotations[idx]    = oa::vlm::quaternionIdentity();
		}
	}
	return clip;
}
