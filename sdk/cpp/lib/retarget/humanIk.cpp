#include <retarget/humanIk.h>

#include <core/transform.h>

namespace {

#include "rtgPoseData.inc"

constexpr oa::I32 kCoreCount = static_cast<oa::I32>(sizeof(kHumanIkMap) / sizeof(kHumanIkMap[0]));

oa::RefPose buildPose(const RtgRefJoint* inTable) {
	oa::RefPose pose;
	for (oa::I32 i = 0; i < kCoreCount; ++i) {
		const RtgRefJoint& r = inTable[i];
		oa::RefJoint j;
		j.name        = r.name;
		j.localOrient = oa::eulerXyzDegToQuat({ r.rx, r.ry, r.rz });
		j.localTrans  = { r.tx, r.ty, r.tz };
		pose.joints.pushBack(std::move(j));
	}
	return pose;
}

oa::Vec<oa::HumanIkSlot> buildCharacterization() {
	oa::Vec<oa::HumanIkSlot> slots;
	for (oa::I32 i = 0; i < kCoreCount; ++i) {
		const RtgHikSlot& s = kHumanIkMap[i];
		oa::HumanIkSlot o;
		o.slot = s.slot;
		o.id   = s.id;
		o.node = s.node;
		slots.pushBack(std::move(o));
	}
	return slots;
}

} // namespace

oa::vlm::Quat oa::RefPose::orientOf(oa::StringView inName) const noexcept {
	for (const oa::RefJoint& j : joints) {
		if (j.name == inName) {
			return j.localOrient;
		}
	}
	return { 0.0f, 0.0f, 0.0f, 1.0f };
}

const oa::Vec<oa::HumanIkSlot>& oa::humanIkCharacterization() {
	static const oa::Vec<oa::HumanIkSlot> kMap = buildCharacterization();
	return kMap;
}

const oa::RefPose& oa::refPoseFor(oa::Mannequin inWho, oa::PoseKind inKind) {
	static const oa::RefPose kMannyT = buildPose(kRtgPoseMannyT);
	static const oa::RefPose kMannyA = buildPose(kRtgPoseMannyA);
	static const oa::RefPose kQuinnT = buildPose(kRtgPoseQuinnT);
	static const oa::RefPose kQuinnA = buildPose(kRtgPoseQuinnA);
	if (inWho == oa::Mannequin::Manny) {
		return inKind == oa::PoseKind::TPose ? kMannyT : kMannyA;
	}
	return inKind == oa::PoseKind::TPose ? kQuinnT : kQuinnA;
}
