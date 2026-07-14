// OaFnLoss — Extension loss function implementations.

#include <Ml/FnLoss.h>
#include <Ml/Autograd/Matrix/AutogradMatrix.h>

#include <Anim/PoseClip.h>
#include <Anim/PosePack.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Core/Vlm.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Ml/FnMatrix.h>
#include <Oa/Ml/FnLoss.h>
#include <Oa/Runtime/Context.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

// A non-trainable constant matrix of the given shape holding `InData` — used to
// inject the per-channel weight row into the graph. Built from host bytes (proper
// host→device upload), so it records no producing op and carries no gradient of
// its own; it just scales the squared error.
OaMatrix MakeConstRow_(OaMatrixShape InShape, const OaF32* InData, OaUsize InCount) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(InData), InCount * sizeof(float)),
		InShape, OaScalarType::Float32);
}

} // namespace

OaMatrix OaFnLoss::Gen3dAnim(const OaMatrix& InPredFlat, const OaMatrix& InTargetFlat, const OaGen3dAnimLossConfig& InConfig) {
	OaFnLoss::SetLastName("pose_mse");
	const OaI32 poseDim = InConfig.PoseDim;
	const OaI32 seq     = InConfig.SeqLen;
	const OaI64 nTokens = InTargetFlat.Size(0);
	const OaI32 bs      = static_cast<OaI32>(nTokens / seq);

	// Per-channel-type path: weight each channel by its ChannelSpec type so the
	// scalar hinge angles, 6D rotation blocks, translate, and contact channels are
	// balanced (see OaFnGen3dAnim::MakeChannelWeights for building ChannelWeights).
	// Active iff a full-length weight vector is supplied; otherwise the legacy uniform path below runs unchanged.
	const bool typed = InConfig.ChannelWeights.Size() == static_cast<OaUsize>(poseDim);

	auto diff = InPredFlat - InTargetFlat;
	auto sq   = OaFnMatrix::Mul(diff, diff);

	auto pred3 = InPredFlat.Reshape(OaMatrixShape{bs, seq, poseDim});
	auto targ3 = InTargetFlat.Reshape(OaMatrixShape{bs, seq, poseDim});

	// Velocity / delta consistency along time (dim 1) — teaches dynamics, kills jitter.
	auto p_d    = OaFnMatrix::Slice(pred3, 1, 1, seq) - OaFnMatrix::Slice(pred3, 1, 0, seq - 1);
	auto t_d    = OaFnMatrix::Slice(targ3, 1, 1, seq) - OaFnMatrix::Slice(targ3, 1, 0, seq - 1);
	auto vd     = p_d - t_d;
	auto vel_sq = OaFnMatrix::Mul(vd, vd);

	OaMatrix loss;
	if (typed) {
		// Weighted means: Σ(w·err²) / (tokens · Σw). Σw normalizes the scale so the
		// loss magnitude stays comparable to the uniform path regardless of weights.
		float sumW = 0.0f;
		for (OaUsize i = 0; i < InConfig.ChannelWeights.Size(); ++i) {
			sumW += InConfig.ChannelWeights[i];
		}
		if (sumW <= 0.0f) { sumW = 1.0f; }

		const OaF32* w = InConfig.ChannelWeights.Data();
		auto wRow = MakeConstRow_(OaMatrixShape{1, poseDim}, w, static_cast<OaUsize>(poseDim));   // [1, D]
		auto wVel = MakeConstRow_(OaMatrixShape{1, 1, poseDim}, w, static_cast<OaUsize>(poseDim)); // [1,1,D]

		// Pose: contacts are folded into the weights here, so no separate term below.
		auto pose_loss = OaFnMatrix::Scale(
			OaFnMatrix::Sum(OaFnMatrix::Mul(sq, wRow)),
			InConfig.PoseWeight / (static_cast<float>(nTokens) * sumW));

		const OaI64 velTokens = static_cast<OaI64>(bs) * (seq - 1);
		auto vel_loss = OaFnMatrix::Scale(
			OaFnMatrix::Sum(OaFnMatrix::Mul(vel_sq, wVel)),
			InConfig.VelWeight / (static_cast<float>(velTokens) * sumW));

		loss = pose_loss + vel_loss;
	} else {
		// Legacy uniform layout (synthetic-gait / tutorial parity, bit-for-bit).
		auto pose_loss = OaFnMatrix::Scale(
			OaFnMatrix::Sum(sq),
			InConfig.PoseWeight / static_cast<float>(sq.NumElements()));

		auto vel_loss = OaFnMatrix::Scale(
			OaFnMatrix::Sum(vel_sq),
			InConfig.VelWeight / static_cast<float>(vel_sq.NumElements()));

		// Contact channels (trailing ContactDims) get extra emphasis for planting.
		auto c_pred       = OaFnMatrix::Slice(pred3, 2, poseDim - InConfig.ContactDims, poseDim);
		auto c_targ       = OaFnMatrix::Slice(targ3, 2, poseDim - InConfig.ContactDims, poseDim);
		auto c_err        = c_pred - c_targ;
		auto c_sq         = OaFnMatrix::Mul(c_err, c_err);
		auto contact_loss = OaFnMatrix::Scale(
			OaFnMatrix::Sum(c_sq),
			InConfig.ContactWeight / static_cast<float>(c_sq.NumElements()));

		loss = pose_loss + vel_loss + contact_loss;
	}

	// Root-motion consistency (leading RootDims = root trans3 + rot6) — punishes
	// drift. Off by default (RootWeight 0 ⇒ skipped entirely, exact baseline parity).
	if (InConfig.RootWeight > 0.0f && InConfig.RootDims > 0) {
		auto r_pred  = OaFnMatrix::Slice(pred3, 2, 0, InConfig.RootDims);
		auto r_targ  = OaFnMatrix::Slice(targ3, 2, 0, InConfig.RootDims);
		auto rp_d    = OaFnMatrix::Slice(r_pred, 1, 1, seq) - OaFnMatrix::Slice(r_pred, 1, 0, seq - 1);
		auto rt_d    = OaFnMatrix::Slice(r_targ, 1, 1, seq) - OaFnMatrix::Slice(r_targ, 1, 0, seq - 1);
		auto r_sq    = OaFnMatrix::Mul(rp_d - rt_d, rp_d - rt_d);
		auto r_loss  = OaFnMatrix::Scale(
			OaFnMatrix::Sum(r_sq),
			InConfig.RootWeight / static_cast<float>(r_sq.NumElements()));
		loss = loss + r_loss;
	}

	return loss;
}

namespace {

OaMatrix ConstF32_(OaSpan<const OaF32> InData, OaMatrixShape InShape) {
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(InData.data()), InData.size() * sizeof(OaF32)),
		InShape,
		OaScalarType::Float32);
}

OaMatrix ConstVec3_(OaI64 InRows, const VlmVec3& InV) {
	std::vector<OaF32> data(static_cast<size_t>(InRows) * 3);
	for (OaI64 i = 0; i < InRows; ++i) {
		data[static_cast<size_t>(i) * 3 + 0] = InV.X;
		data[static_cast<size_t>(i) * 3 + 1] = InV.Y;
		data[static_cast<size_t>(i) * 3 + 2] = InV.Z;
	}
	return ConstF32_(OaSpan<const OaF32>(data.data(), data.size()), OaMatrixShape{InRows, 3});
}

OaMatrix ConstMat3_(OaI64 InRows, const VlmQuat& InQ) {
	const VlmMat4 m = Vlm::QuaternionToMatrix(InQ);
	std::vector<OaF32> data(static_cast<size_t>(InRows) * 9);
	for (OaI64 i = 0; i < InRows; ++i) {
		OaF32* out = &data[static_cast<size_t>(i) * 9];
		out[0] = m.M[0][0]; out[1] = m.M[0][1]; out[2] = m.M[0][2];
		out[3] = m.M[1][0]; out[4] = m.M[1][1]; out[5] = m.M[1][2];
		out[6] = m.M[2][0]; out[7] = m.M[2][1]; out[8] = m.M[2][2];
	}
	return ConstF32_(OaSpan<const OaF32>(data.data(), data.size()), OaMatrixShape{InRows, 3, 3});
}

OaMatrix Normalize3_(const OaMatrix& InV) {
	auto sq = OaFnMatrix::Mul(InV, InV);
	auto len = OaFnMatrix::Sqrt(OaFnMatrix::AddScalar(OaFnMatrix::Sum(sq, 1), 1e-8f)).Reshape(OaMatrixShape{InV.Size(0), 1});
	return OaFnMatrix::Div(InV, len);
}

OaMatrix Cross3_(const OaMatrix& InA, const OaMatrix& InB) {
	auto ax = OaFnMatrix::Slice(InA, 1, 0, 1);
	auto ay = OaFnMatrix::Slice(InA, 1, 1, 2);
	auto az = OaFnMatrix::Slice(InA, 1, 2, 3);
	auto bx = OaFnMatrix::Slice(InB, 1, 0, 1);
	auto by = OaFnMatrix::Slice(InB, 1, 1, 2);
	auto bz = OaFnMatrix::Slice(InB, 1, 2, 3);
	OaMatrix parts[] = {
		OaFnMatrix::Sub(OaFnMatrix::Mul(ay, bz), OaFnMatrix::Mul(az, by)),
		OaFnMatrix::Sub(OaFnMatrix::Mul(az, bx), OaFnMatrix::Mul(ax, bz)),
		OaFnMatrix::Sub(OaFnMatrix::Mul(ax, by), OaFnMatrix::Mul(ay, bx)),
	};
	return OaFnMatrix::Concat(OaSpan<OaMatrix>(parts, 3), 1);
}

OaMatrix SixDToMat3_(const OaMatrix& InSix) {
	auto a1 = OaFnMatrix::Slice(InSix, 1, 0, 3);
	auto a2 = OaFnMatrix::Slice(InSix, 1, 3, 6);
	auto b1 = Normalize3_(a1);
	auto dot = OaFnMatrix::Sum(OaFnMatrix::Mul(b1, a2), 1).Reshape(OaMatrixShape{InSix.Size(0), 1});
	auto b2 = Normalize3_(OaFnMatrix::Sub(a2, OaFnMatrix::Mul(b1, dot)));
	auto b3 = Cross3_(b1, b2);
	OaMatrix cols[] = {
		b1.Reshape(OaMatrixShape{InSix.Size(0), 3, 1}),
		b2.Reshape(OaMatrixShape{InSix.Size(0), 3, 1}),
		b3.Reshape(OaMatrixShape{InSix.Size(0), 3, 1}),
	};
	return OaFnMatrix::Concat(OaSpan<OaMatrix>(cols, 3), 2);
}

OaMatrix HingeZToMat3_(const OaMatrix& InAngle, const VlmQuat& InRestOrient) {
	const OaI64 tokens = InAngle.Size(0);
	auto cs = OaFnMatrix::Cos(InAngle);
	auto sn = OaFnMatrix::Sin(InAngle);
	auto zero = OaFnMatrix::Zeros(OaMatrixShape{tokens, 1});
	auto one = OaFnMatrix::Ones(OaMatrixShape{tokens, 1});

	OaMatrix col0Parts[] = {cs, sn, zero};
	auto negSn = OaFnMatrix::Neg(sn);
	OaMatrix col1Parts[] = {negSn, cs, zero};
	OaMatrix col2Parts[] = {zero, zero, one};
	auto col0 = OaFnMatrix::Concat(OaSpan<OaMatrix>(col0Parts, 3), 1).Reshape(OaMatrixShape{tokens, 3, 1});
	auto col1 = OaFnMatrix::Concat(OaSpan<OaMatrix>(col1Parts, 3), 1).Reshape(OaMatrixShape{tokens, 3, 1});
	auto col2 = OaFnMatrix::Concat(OaSpan<OaMatrix>(col2Parts, 3), 1).Reshape(OaMatrixShape{tokens, 3, 1});
	OaMatrix cols[] = {col0, col1, col2};
	auto rz = OaFnMatrix::Concat(OaSpan<OaMatrix>(cols, 3), 2);
	return OaFnMatrix::Bmm(ConstMat3_(tokens, InRestOrient), rz);
}

OaMatrix SkPoseFkWorld_(const OaMatrix& InFlat, const OaSkeleton& InSkeleton) {
	const OaI64 tokens = InFlat.Size(0);
	const OaI32 joints = InSkeleton.JointCount();
	std::vector<OaMatrix> worldR(static_cast<size_t>(joints));
	std::vector<OaMatrix> worldP(static_cast<size_t>(joints));
	std::vector<OaMatrix> posParts;
	posParts.reserve(static_cast<size_t>(joints));

	for (OaI32 j = 0; j < joints; ++j) {
		const OaSkelJoint& joint = InSkeleton.Joints[static_cast<OaUsize>(j)];
		OaI64 c = InSkeleton.ChannelOffset(j);
		OaMatrix localT;
		if (joint.HasTranslate) {
			localT = OaFnMatrix::Slice(InFlat, 1, c, c + 3);
			c += 3;
		} else {
			localT = ConstVec3_(tokens, joint.Rest.Translate);
		}

		OaMatrix localR;
		if (joint.RotDof == 3) {
			localR = SixDToMat3_(OaFnMatrix::Slice(InFlat, 1, c, c + 6));
		} else if (joint.RotDof == 1) {
			localR = HingeZToMat3_(OaFnMatrix::Slice(InFlat, 1, c, c + 1), joint.Rest.JointOrient);
		} else {
			localR = ConstMat3_(tokens, joint.Rest.OrientedRotation());
		}

		if (joint.ParentIndex < 0) {
			worldR[static_cast<size_t>(j)] = localR;
			worldP[static_cast<size_t>(j)] = localT;
		} else {
			const OaMatrix& pr = worldR[static_cast<size_t>(joint.ParentIndex)];
			const OaMatrix& pp = worldP[static_cast<size_t>(joint.ParentIndex)];
			worldR[static_cast<size_t>(j)] = OaFnMatrix::Bmm(pr, localR);
			auto localCol = localT.Reshape(OaMatrixShape{tokens, 3, 1});
			auto rotated = OaFnMatrix::Bmm(pr, localCol).Reshape(OaMatrixShape{tokens, 3});
			worldP[static_cast<size_t>(j)] = OaFnMatrix::Add(pp, rotated);
		}
		posParts.push_back(worldP[static_cast<size_t>(j)].Reshape(OaMatrixShape{tokens, 1, 3}));
	}

	return OaFnMatrix::Concat(OaSpan<OaMatrix>(posParts.data(), posParts.size()), 1);
}

} // namespace

OaMatrix OaFnLoss::SkPoseFkLoss(
	const OaMatrix& InPredFlat,
	const OaMatrix& InTargetWorld,
	const OaSkeleton& InSkeleton,
	const OaSkPoseFkLossConfig& InConfig
) {
	OaFnLoss::SetLastName("skposefk_loss");
	OA_ASSERT(InPredFlat.Rank() == 2 && "SkPoseFkLoss expects pred [B*T, PoseDim]");
	OA_ASSERT(InTargetWorld.Rank() == 3 && "SkPoseFkLoss expects target world [B*T, J, 3]");
	OA_ASSERT(InConfig.PoseDim == InPredFlat.Size(1));
	OA_ASSERT(InConfig.PoseDim == InSkeleton.PoseDim());
	OA_ASSERT(InTargetWorld.Size(0) == InPredFlat.Size(0));
	OA_ASSERT(InTargetWorld.Size(1) == InSkeleton.JointCount());
	OA_ASSERT(InTargetWorld.Size(2) == 3);

	auto predWorld = SkPoseFkWorld_(InPredFlat, InSkeleton);
	auto err = predWorld - InTargetWorld;
	auto posSq = OaFnMatrix::Mul(err, err);
	auto loss = OaFnMatrix::Scale(OaFnMatrix::Sum(posSq),
		InConfig.PositionWeight / static_cast<OaF32>(posSq.NumElements()));

	if (InConfig.VelWeight > 0.0f && InConfig.SeqLen > 1) {
		const OaI64 tokens = InPredFlat.Size(0);
		const OaI64 batch = tokens / InConfig.SeqLen;
		auto p3 = predWorld.Reshape(OaMatrixShape{batch, InConfig.SeqLen, InSkeleton.JointCount(), 3});
		auto t3 = InTargetWorld.Reshape(OaMatrixShape{batch, InConfig.SeqLen, InSkeleton.JointCount(), 3});
		auto pd = OaFnMatrix::Slice(p3, 1, 1, InConfig.SeqLen) - OaFnMatrix::Slice(p3, 1, 0, InConfig.SeqLen - 1);
		auto td = OaFnMatrix::Slice(t3, 1, 1, InConfig.SeqLen) - OaFnMatrix::Slice(t3, 1, 0, InConfig.SeqLen - 1);
		auto vd = pd - td;
		auto vSq = OaFnMatrix::Mul(vd, vd);
		loss = loss + OaFnMatrix::Scale(OaFnMatrix::Sum(vSq),
			InConfig.VelWeight / static_cast<OaF32>(vSq.NumElements()));
	}

	return loss;
}

namespace {

std::vector<VlmVec3> Gen3dAnimWorldPositions_(const OaUsdSkelClip& InUsd,
                                             const OaSkeleton& InSkel) {
	using namespace Vlm;
	const OaI32 n = InSkel.JointCount();
	std::vector<VlmVec3> out(static_cast<size_t>(InUsd.FrameCount) * static_cast<size_t>(n));
	std::vector<VlmQuat> worldRot(static_cast<size_t>(InUsd.FrameCount) * static_cast<size_t>(n));

	for (OaU32 f = 0; f < InUsd.FrameCount; ++f) {
		for (OaI32 j = 0; j < n; ++j) {
			const size_t idx = static_cast<size_t>(f) * static_cast<size_t>(n) + static_cast<size_t>(j);
			const VlmVec3 lt = InUsd.Translations[idx];
			const VlmQuat lr = InUsd.Rotations[idx];
			const OaI32 parent = InSkel.Joints[static_cast<OaUsize>(j)].ParentIndex;
			if (parent < 0) {
				worldRot[idx] = lr;
				out[idx] = lt;
			} else {
				const size_t pidx = static_cast<size_t>(f) * static_cast<size_t>(n) + static_cast<size_t>(parent);
				worldRot[idx] = VlmQuatMul(worldRot[pidx], lr);
				out[idx] = Add(out[pidx], RotateVector(worldRot[pidx], lt));
			}
		}
	}
	return out;
}

OaResult<OaUsdSkelClip> Gen3dAnimRawToUsd_(OaSpan<const OaF32> InRaw,
                                           OaI32 InFrameCount,
                                           OaI32 InPoseDim,
                                           OaF32 InFps,
                                           const OaSkeleton& InSkeleton) {
	auto clipResult = OaPoseClip::Create(
		static_cast<OaU32>(InFrameCount),
		static_cast<OaU32>(InPoseDim),
		InFps,
		InSkeleton.SkeletonId,
		InRaw);
	if (!clipResult.IsOk()) {
		return clipResult.GetStatus();
	}
	return OaPosePack::Unpack(*clipResult, InSkeleton);
}

} // namespace

OaSkPoseFkMetrics OaFnLoss::SkPoseFk(
	OaSpan<const OaF32> InPredRaw,
	OaSpan<const OaF32> InTargetRaw,
	OaI32 InFrameCount,
	OaI32 InPoseDim,
	OaF32 InFps,
	const OaSkeleton& InSkeleton,
	OaF32 InContactThreshold
) {
	OaFnLoss::SetLastName("skposefk_mse");
	using namespace Vlm;

	OaSkPoseFkMetrics out;
	if (InFrameCount <= 0 || InPoseDim <= 0 || !InSkeleton.IsValid()) {
		return out;
	}
	const OaUsize expected = static_cast<OaUsize>(InFrameCount) * static_cast<OaUsize>(InPoseDim);
	if (InPredRaw.size() != expected || InTargetRaw.size() != expected ||
	    InPoseDim != InSkeleton.PoseDim()) {
		return out;
	}

	auto predUsd = Gen3dAnimRawToUsd_(InPredRaw, InFrameCount, InPoseDim, InFps, InSkeleton);
	auto targetUsd = Gen3dAnimRawToUsd_(InTargetRaw, InFrameCount, InPoseDim, InFps, InSkeleton);
	if (!predUsd.IsOk() || !targetUsd.IsOk()) {
		return out;
	}

	const OaI32 n = InSkeleton.JointCount();
	// Root is joint 0; pelvis is the only other translated joint. Both get full
	// translation + rotation tracking below.
	const OaI32 pelvisIdx = InSkeleton.IndexOf("pelvis");

	// ── Rotation geodesic error (degrees) on the unpacked per-joint LOCAL quats ──
	// USD stores rotations as quaternions; the model's 6D/hinge packing is only a
	// network encoding. So compare quats directly — no FK, no bone length. Only
	// joints with a live rotation channel (RotDof != 0) are scored; locked joints
	// are never predicted. Geodesic angle = 2·acos(|dot(q_pred, q_target)|).
	{
		double sumRot = 0.0;
		OaI64 rotCount = 0;
		double maxRot = 0.0;
		double rootRot = 0.0;
		double pelvisRot = 0.0;
		for (OaU32 f = 0; f < predUsd->FrameCount; ++f) {
			for (OaI32 j = 0; j < n; ++j) {
				if (InSkeleton.Joints[static_cast<OaUsize>(j)].RotDof == 0) { continue; }
				const size_t idx = static_cast<size_t>(f) * static_cast<size_t>(n) + static_cast<size_t>(j);
				const VlmQuat& a = predUsd->Rotations[idx];
				const VlmQuat& b = targetUsd->Rotations[idx];
				double d = static_cast<double>(a.W) * b.W + static_cast<double>(a.X) * b.X
				         + static_cast<double>(a.Y) * b.Y + static_cast<double>(a.Z) * b.Z;
				d = std::min(1.0, std::max(-1.0, std::fabs(d)));
				const double deg = 2.0 * std::acos(d) * (180.0 / 3.14159265358979323846);
				sumRot += deg;
				++rotCount;
				maxRot = std::max(maxRot, deg);
				if (j == 0)         { rootRot += deg; }
				if (j == pelvisIdx) { pelvisRot += deg; }
			}
		}
		const double frames = predUsd->FrameCount > 0 ? static_cast<double>(predUsd->FrameCount) : 1.0;
		out.MeanJointRotDeg = rotCount > 0 ? sumRot / static_cast<double>(rotCount) : 0.0;
		out.MaxJointRotDeg  = maxRot;
		out.RootRotDeg      = rootRot / frames;
		out.PelvisRotDeg    = pelvisIdx >= 0 ? pelvisRot / frames : 0.0;
		out.RotJointsScored = rotCount > 0 ? static_cast<OaI32>(rotCount / static_cast<OaI64>(predUsd->FrameCount)) : 0;
	}

	const std::vector<VlmVec3> predWorld = Gen3dAnimWorldPositions_(*predUsd, InSkeleton);
	const std::vector<VlmVec3> targetWorld = Gen3dAnimWorldPositions_(*targetUsd, InSkeleton);

	double sumJoint = 0.0;
	double sumRoot = 0.0;
	double maxRoot = 0.0;
	double sumPelvis = 0.0;
	double maxPelvis = 0.0;
	for (OaI32 f = 0; f < InFrameCount; ++f) {
		for (OaI32 j = 0; j < n; ++j) {
			const size_t idx = static_cast<size_t>(f) * static_cast<size_t>(n) + static_cast<size_t>(j);
			const double d = static_cast<double>(Length(Sub(predWorld[idx], targetWorld[idx])));
			sumJoint += d;
			if (j == 0) {
				sumRoot += d;
				maxRoot = std::max(maxRoot, d);
			}
			if (j == pelvisIdx) {
				sumPelvis += d;
				maxPelvis = std::max(maxPelvis, d);
			}
		}
	}

	const OaI32 contactOff = InSkeleton.ContactOffset();
	double footSkate = 0.0;
	OaI64 footSkateCount = 0;
	for (OaI32 f = 1; f < InFrameCount; ++f) {
		for (OaI32 ci = 0; ci < static_cast<OaI32>(InSkeleton.ContactJoints.Size()); ++ci) {
			const OaI32 contactCh = contactOff + ci;
			if (contactCh < 0 || contactCh >= InPoseDim) {
				continue;
			}
			const float targetContact = InTargetRaw[static_cast<OaUsize>(f) * InPoseDim + contactCh];
			if (targetContact < InContactThreshold) {
				continue;
			}

			const OaI32 joint = InSkeleton.ContactJoints[static_cast<OaUsize>(ci)];
			const size_t cur = static_cast<size_t>(f) * static_cast<size_t>(n) + static_cast<size_t>(joint);
			const size_t prev = static_cast<size_t>(f - 1) * static_cast<size_t>(n) + static_cast<size_t>(joint);
			const VlmVec3 delta = Sub(predWorld[cur], predWorld[prev]);
			footSkate += std::sqrt(static_cast<double>(delta.X * delta.X + delta.Z * delta.Z));
			++footSkateCount;
		}
	}

	out.MpjpeCm = sumJoint / static_cast<double>(InFrameCount * n);
	out.RootMeanCm = sumRoot / static_cast<double>(InFrameCount);
	out.RootMaxCm = maxRoot;
	out.PelvisMeanCm = pelvisIdx >= 0 ? sumPelvis / static_cast<double>(InFrameCount) : 0.0;
	out.PelvisMaxCm  = maxPelvis;
	out.FootSkateCmPerFrame = footSkateCount > 0 ? footSkate / static_cast<double>(footSkateCount) : 0.0;
	out.Ok = std::isfinite(out.MpjpeCm) && std::isfinite(out.RootMeanCm)
		&& std::isfinite(out.RootMaxCm) && std::isfinite(out.FootSkateCmPerFrame)
		&& std::isfinite(out.MeanJointRotDeg) && std::isfinite(out.MaxJointRotDeg)
		&& std::isfinite(out.RootRotDeg) && std::isfinite(out.PelvisRotDeg)
		&& std::isfinite(out.PelvisMeanCm) && std::isfinite(out.PelvisMaxCm);
	return out;
}

OaSkPoseFkMetrics OaFnLoss::Gen3dAnimPoseMetrics(
	OaSpan<const OaF32> InPredRaw,
	OaSpan<const OaF32> InTargetRaw,
	OaI32 InFrameCount,
	OaI32 InPoseDim,
	OaF32 InFps,
	const OaSkeleton& InSkeleton,
	OaF32 InContactThreshold
) {
	return OaFnLoss::SkPoseFk(
		InPredRaw,
		InTargetRaw,
		InFrameCount,
		InPoseDim,
		InFps,
		InSkeleton,
		InContactThreshold);
}

// ─── Fused SmoothL1Mean + VelSmoothL1 (OaAlm Phase F3) ─────────────────────

OaMatrix OaFnLoss::SmoothL1Mean(const OaMatrix& InA, const OaMatrix& InB) {
	OaFnLoss::SetLastName("smooth_l1_mean");
	auto& ctx = OaContext::GetDefault();
	const OaU32 count = static_cast<OaU32>(InA.NumElements());

	// Degenerate empty input: the shader divides the reduction by count, so
	// count == 0 would produce a NaN loss (and inf gradients) that silently
	// poisons training. Return a zero scalar instead.
	if (count == 0) { return OaFnMatrix::Zeros(OaMatrixShape{1}, InA.GetDtype()); }

	OaMatrix out = OaFnMatrix::Empty(OaMatrixShape{1}, InA.GetDtype());
	struct { OaU32 Count; } push{ .Count = count };
	OaBufferAccess access[] = {
		OaBufferAccess::Read,   // a
		OaBufferAccess::Read,   // b
		OaBufferAccess::Write   // out
	};
	ctx.Add("SmoothL1Mean", {&InA, &InB, &out}, access, &push, sizeof(push), 1);

	if (OaFnAutograd::IsEnabled() and InA.RequiresGrad()) {
		auto gradFn = OaMakeSharedPtr<OaGradSmoothL1Mean>();
		gradFn->Saved_ = OaVec<OaMatrix>{InA, InB};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InA, InB});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->Count_ = count;
		out.MutAutograd().GradFn = gradFn;
		out.SetRequiresGrad(true);
	}
	return out;
}

OaMatrix OaFnLoss::SmoothL1MeanBwd(const OaMatrix& InA, const OaMatrix& InB, const OaMatrix& InDOut) {
	OaFnLoss::SetLastName("smooth_l1_mean");
	auto& ctx = OaContext::GetDefault();
	const OaU32 count = static_cast<OaU32>(InA.NumElements());

	OaMatrix dA = OaFnMatrix::Empty(InA.GetShape(), InA.GetDtype());
	struct { OaU32 Count; } push{ .Count = count };
	OaBufferAccess access[] = {
		OaBufferAccess::Read,   // a
		OaBufferAccess::Read,   // b
		OaBufferAccess::Read,   // d_out
		OaBufferAccess::Write   // d_a
	};
	ctx.Add("SmoothL1MeanBwd", {&InA, &InB, &InDOut, &dA}, access, &push, sizeof(push), OaDivCeil(count, 256U));
	return dA;
}

OaMatrix OaFnLoss::VelSmoothL1(const OaMatrix& InPred, const OaMatrix& InTarget) {
	OaFnLoss::SetLastName("vel_smooth_l1");
	auto& ctx = OaContext::GetDefault();
	const OaI32 batch = static_cast<OaI32>(InPred.Size(0));
	const OaI32 seqLen = static_cast<OaI32>(InPred.Size(1));
	const OaI32 featDim = static_cast<OaI32>(InPred.Size(2));
	const OaU32 count = static_cast<OaU32>(batch) * static_cast<OaU32>(seqLen - 1) * static_cast<OaU32>(featDim);

	// A single-frame window (seqLen < 2) has no velocity term; count == 0 would
	// divide the reduction by zero → NaN loss. Return a zero scalar, matching the
	// FK-loss builder's own `SeqLen > 1` guard.
	if (count == 0) { return OaFnMatrix::Zeros(OaMatrixShape{1}, InPred.GetDtype()); }

	OaMatrix out = OaFnMatrix::Empty(OaMatrixShape{1}, InPred.GetDtype());
	struct {
		OaU32 Batch;
		OaU32 SeqLen;
		OaU32 FeatDim;
		OaU32 Count;
	} push{
		.Batch = static_cast<OaU32>(batch),
		.SeqLen = static_cast<OaU32>(seqLen),
		.FeatDim = static_cast<OaU32>(featDim),
		.Count = count};
	OaBufferAccess access[] = {
		OaBufferAccess::Read,   // pred
		OaBufferAccess::Read,   // target
		OaBufferAccess::Write   // out
	};
	ctx.Add("VelSmoothL1", {&InPred, &InTarget, &out}, access, &push, sizeof(push), 1);

	if (OaFnAutograd::IsEnabled() and InPred.RequiresGrad()) {
		auto gradFn = OaMakeSharedPtr<OaGradVelSmoothL1>();
		gradFn->Saved_ = OaVec<OaMatrix>{InPred, InTarget};
		gradFn->SetGraphInputs(OaVec<OaMatrix>{InPred, InTarget});
		gradFn->SequenceNr_ = OaFnAutograd::NextSeq();
		gradFn->Batch_ = batch;
		gradFn->SeqLen_ = seqLen;
		gradFn->FeatDim_ = featDim;
		gradFn->Count_ = count;
		out.MutAutograd().GradFn = gradFn;
		out.SetRequiresGrad(true);
	}
	return out;
}

OaMatrix OaFnLoss::VelSmoothL1Bwd(const OaMatrix& InPred, const OaMatrix& InTarget, const OaMatrix& InDOut) {
	OaFnLoss::SetLastName("vel_smooth_l1");
	auto& ctx = OaContext::GetDefault();
	const OaI32 batch = static_cast<OaI32>(InPred.Size(0));
	const OaI32 seqLen = static_cast<OaI32>(InPred.Size(1));
	const OaI32 featDim = static_cast<OaI32>(InPred.Size(2));
	const OaU32 count = static_cast<OaU32>(batch) * static_cast<OaU32>(seqLen - 1) * static_cast<OaU32>(featDim);
	const OaU32 total = static_cast<OaU32>(InPred.NumElements());

	OaMatrix dPred = OaFnMatrix::Empty(InPred.GetShape(), InPred.GetDtype());
	struct {
		OaU32 Batch;
		OaU32 SeqLen;
		OaU32 FeatDim;
		OaU32 Count;
	} push{
		.Batch = static_cast<OaU32>(batch),
		.SeqLen = static_cast<OaU32>(seqLen),
		.FeatDim = static_cast<OaU32>(featDim),
		.Count = count};
	OaBufferAccess access[] = {
		OaBufferAccess::Read,   // pred
		OaBufferAccess::Read,   // target
		OaBufferAccess::Read,   // d_out
		OaBufferAccess::Write   // d_pred
	};
	ctx.Add("VelSmoothL1Bwd", {&InPred, &InTarget, &InDOut, &dPred}, access, &push, sizeof(push), OaDivCeil(total, 256U));
	return dPred;
}

// ─── SkPoseFk helpers ───────────────────────────────────────────────────────

OaMatrix OaFnLoss::SkPoseFkTargetWorld(
	OaSpan<const OaF32> InRaw,
	OaI32 InFrameCount,
	OaI32 InPoseDim,
	OaF32 InFps,
	const OaSkeleton& InSkeleton
) {
	const OaI32 joints = InSkeleton.JointCount();
	const OaUsize expected = static_cast<OaUsize>(InFrameCount) * static_cast<OaUsize>(InPoseDim);
	if (InFrameCount <= 0 || InPoseDim <= 0 || !InSkeleton.IsValid()
		|| InRaw.size() != expected || InPoseDim != InSkeleton.PoseDim()) {
		return OaFnMatrix::Zeros(OaMatrixShape{std::max<OaI32>(InFrameCount, 0), std::max<OaI32>(joints, 0), 3});
	}

	auto usd = Gen3dAnimRawToUsd_(InRaw, InFrameCount, InPoseDim, InFps, InSkeleton);
	if (!usd.IsOk()) {
		return OaFnMatrix::Zeros(OaMatrixShape{InFrameCount, joints, 3});
	}
	const std::vector<VlmVec3> world = Gen3dAnimWorldPositions_(*usd, InSkeleton);
	std::vector<OaF32> packed(static_cast<size_t>(InFrameCount) * static_cast<size_t>(joints) * 3);
	for (OaI32 f = 0; f < InFrameCount; ++f) {
		for (OaI32 j = 0; j < joints; ++j) {
			const size_t src = static_cast<size_t>(f) * static_cast<size_t>(joints) + static_cast<size_t>(j);
			const size_t dst = src * 3;
			packed[dst + 0] = world[src].X;
			packed[dst + 1] = world[src].Y;
			packed[dst + 2] = world[src].Z;
		}
	}
	return OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(reinterpret_cast<const OaU8*>(packed.data()), packed.size() * sizeof(OaF32)),
		OaMatrixShape{InFrameCount, joints, 3},
		OaScalarType::Float32);
}
