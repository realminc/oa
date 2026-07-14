#include <Retarget/HumanIk.h>

#include <Core/Transform.h>

namespace {

#include "RtgPoseData.inc"

constexpr OaI32 kCoreCount = static_cast<OaI32>(sizeof(kHumanIkMap) / sizeof(kHumanIkMap[0]));

OaRefPose BuildPose(const RtgRefJoint* InTable) {
	OaRefPose pose;
	for (OaI32 i = 0; i < kCoreCount; ++i) {
		const RtgRefJoint& r = InTable[i];
		OaRefJoint j;
		j.Name        = r.Name;
		j.LocalOrient = OaEulerXyzDegToQuat({ r.Rx, r.Ry, r.Rz });
		j.LocalTrans  = { r.Tx, r.Ty, r.Tz };
		pose.Joints.PushBack(std::move(j));
	}
	return pose;
}

OaVec<OaHumanIkSlot> BuildCharacterization() {
	OaVec<OaHumanIkSlot> slots;
	for (OaI32 i = 0; i < kCoreCount; ++i) {
		const RtgHikSlot& s = kHumanIkMap[i];
		OaHumanIkSlot o;
		o.Slot = s.Slot;
		o.Id   = s.Id;
		o.Node = s.Node;
		slots.PushBack(std::move(o));
	}
	return slots;
}

} // namespace

VlmQuat OaRefPose::OrientOf(OaStringView InName) const noexcept {
	for (const OaRefJoint& j : Joints) {
		if (j.Name == InName) {
			return j.LocalOrient;
		}
	}
	return { 0.0f, 0.0f, 0.0f, 1.0f };
}

const OaVec<OaHumanIkSlot>& OaHumanIkCharacterization() {
	static const OaVec<OaHumanIkSlot> kMap = BuildCharacterization();
	return kMap;
}

const OaRefPose& OaRefPoseFor(OaMannequin InWho, OaPoseKind InKind) {
	static const OaRefPose kMannyT = BuildPose(kRtgPoseMannyT);
	static const OaRefPose kMannyA = BuildPose(kRtgPoseMannyA);
	static const OaRefPose kQuinnT = BuildPose(kRtgPoseQuinnT);
	static const OaRefPose kQuinnA = BuildPose(kRtgPoseQuinnA);
	if (InWho == OaMannequin::Manny) {
		return InKind == OaPoseKind::TPose ? kMannyT : kMannyA;
	}
	return InKind == OaPoseKind::TPose ? kQuinnT : kQuinnA;
}
