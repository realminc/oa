// oa::FnLoss — extension loss function implementations.

#include <ml/fnLoss.h>
#include <ml/autograd/matrix/autogradMatrix.h>

#include <anim/poseClip.h>
#include <anim/posePack.h>
#include <oa/core/bufferAccess.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/op.h>
#include <oa/core/vlm.h>
#include <oa/ml/autograd.h>
#include <oa/ml/fnMatrix.h>
#include <oa/ml/fnLoss.h>
#include <oa/runtime/executionSession.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <utility>
#include <vector>

namespace {

// A non-trainable constant matrix of the given shape holding `inData` — used to
// inject the per-channel weight row into the graph. Built from host bytes (proper
// host→device upload), so it records no producing op and carries no gradient of
// its own; it just scales the squared error.
oa::Matrix makeConstRow_(oa::MatrixShape inShape, const oa::F32* inData, oa::Usize inCount) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(inData), inCount * sizeof(float)),
		inShape, oa::ScalarType::Float32);
}

constexpr oa::U64 kIdentityOffsetBasis = 14695981039346656037ULL;
constexpr oa::U64 kIdentityPrime = 1099511628211ULL;

void hashUnsigned_(oa::U64& inOutHash, oa::U64 inValue) {
	for (oa::U32 byte = 0; byte < 8U; ++byte) {
		inOutHash ^= (inValue >> (byte * 8U)) & 0xffU;
		inOutHash *= kIdentityPrime;
	}
}

void hashFloat_(oa::U64& inOutHash, oa::F32 inValue) {
	hashUnsigned_(inOutHash, std::bit_cast<oa::U32>(inValue));
}

void hashString_(oa::U64& inOutHash, const oa::String& inValue) {
	hashUnsigned_(inOutHash, static_cast<oa::U64>(inValue.size()));
	for (oa::Usize index = 0; index < inValue.size(); ++index) {
		inOutHash ^= static_cast<oa::U8>(inValue[index]);
		inOutHash *= kIdentityPrime;
	}
}

oa::U64 gen3dAnimConfigIdentity_(
	const oa::Gen3dAnimLossConfig& inConfig) {
	oa::U64 hash = kIdentityOffsetBasis;
	hashUnsigned_(hash, static_cast<oa::U64>(inConfig.poseDim));
	hashUnsigned_(hash, static_cast<oa::U64>(inConfig.seqLen));
	hashUnsigned_(hash, static_cast<oa::U64>(inConfig.contactDims));
	hashUnsigned_(hash, static_cast<oa::U64>(inConfig.rootDims));
	hashFloat_(hash, inConfig.poseWeight);
	hashFloat_(hash, inConfig.velWeight);
	hashFloat_(hash, inConfig.contactWeight);
	hashFloat_(hash, inConfig.rootWeight);
	hashUnsigned_(hash, static_cast<oa::U64>(inConfig.channelWeights.size()));
	for (oa::F32 weight : inConfig.channelWeights) {
		hashFloat_(hash, weight);
	}
	return hash;
}

oa::U64 skeletonIdentity_(const oa::Skeleton& inSkeleton) {
	oa::U64 hash = kIdentityOffsetBasis;
	hashString_(hash, inSkeleton.name);
	hashUnsigned_(hash, inSkeleton.skeletonId);
	hashUnsigned_(hash, static_cast<oa::U64>(inSkeleton.joints.size()));
	for (const oa::SkelJoint& joint : inSkeleton.joints) {
		hashString_(hash, joint.name);
		hashUnsigned_(hash, static_cast<oa::U64>(joint.parentIndex));
		hashUnsigned_(hash, static_cast<oa::U64>(joint.humanIkId));
		hashFloat_(hash, joint.mass);
		hashFloat_(hash, joint.rest.translate.x);
		hashFloat_(hash, joint.rest.translate.y);
		hashFloat_(hash, joint.rest.translate.z);
		hashFloat_(hash, joint.rest.rotate.x);
		hashFloat_(hash, joint.rest.rotate.y);
		hashFloat_(hash, joint.rest.rotate.z);
		hashFloat_(hash, joint.rest.rotate.w);
		hashFloat_(hash, joint.rest.scale.x);
		hashFloat_(hash, joint.rest.scale.y);
		hashFloat_(hash, joint.rest.scale.z);
		hashFloat_(hash, joint.rest.jointOrient.x);
		hashFloat_(hash, joint.rest.jointOrient.y);
		hashFloat_(hash, joint.rest.jointOrient.z);
		hashFloat_(hash, joint.rest.jointOrient.w);
		hashUnsigned_(hash, joint.hasTranslate ? 1U : 0U);
		hashUnsigned_(hash, joint.rotDof);
	}
	hashUnsigned_(hash, static_cast<oa::U64>(inSkeleton.contactJoints.size()));
	for (oa::I32 joint : inSkeleton.contactJoints) {
		hashUnsigned_(hash, static_cast<oa::U64>(joint));
	}
	return hash;
}

oa::Matrix commitExtensionLoss_(
	oa::Matrix inResult,
	oa::OpLoweringScope& inLowering,
	const oa::OpContract& inContract,
	std::initializer_list<const oa::Matrix*> inInputs,
	std::initializer_list<oa::OpAttribute> inAttributes = {}) {
	auto semantic = inLowering.commitWithId(
		inContract, inInputs, {&inResult}, inAttributes);
	if (not semantic.isOk()) return {};
	if (auto grad = inResult.getGradFn()) {
		if (not oa::FnAutograd::attachSemantic(
			grad, semantic.getValue()).isOk())
		{
			return {};
		}
	}
	return inResult;
}

} // namespace

oa::Matrix oa::FnLoss::gen3dAnim(const oa::Matrix& inPredFlat, const oa::Matrix& inTargetFlat, const oa::Gen3dAnimLossConfig& inConfig) {
	oa::FnLoss::setLastName("pose_mse");
	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	const oa::I32 poseDim = inConfig.poseDim;
	const oa::I32 seq     = inConfig.seqLen;
	const oa::I64 nTokens = inTargetFlat.size(0);
	const oa::I32 bs      = static_cast<oa::I32>(nTokens / seq);

	// Per-channel-type path: weight each channel by its ChannelSpec type so the
	// scalar hinge angles, 6D rotation blocks, translate, and contact channels are
	// balanced (see oa::FnGen3dAnim::MakeChannelWeights for building channelWeights).
	// active iff a full-length weight vector is supplied; otherwise the legacy uniform path below runs unchanged.
	const bool typed = inConfig.channelWeights.size() == static_cast<oa::Usize>(poseDim);

	auto diff = inPredFlat - inTargetFlat;
	auto sq   = oa::FnMatrix::mul(diff, diff);

	auto pred3 = inPredFlat.reshape(oa::MatrixShape{bs, seq, poseDim});
	auto targ3 = inTargetFlat.reshape(oa::MatrixShape{bs, seq, poseDim});

	// velocity / delta consistency along time (dim 1) — teaches dynamics, kills jitter.
	auto p_d    = oa::FnMatrix::slice(pred3, 1, 1, seq) - oa::FnMatrix::slice(pred3, 1, 0, seq - 1);
	auto t_d    = oa::FnMatrix::slice(targ3, 1, 1, seq) - oa::FnMatrix::slice(targ3, 1, 0, seq - 1);
	auto vd     = p_d - t_d;
	auto vel_sq = oa::FnMatrix::mul(vd, vd);

	oa::Matrix loss;
	if (typed) {
		// Weighted means: Σ(w·err²) / (tokens · Σw). Σw normalizes the scale so the
		// loss magnitude stays comparable to the uniform path regardless of weights.
		float sumW = 0.0f;
		for (oa::Usize i = 0; i < inConfig.channelWeights.size(); ++i) {
			sumW += inConfig.channelWeights[i];
		}
		if (sumW <= 0.0f) { sumW = 1.0f; }

		const oa::F32* w = inConfig.channelWeights.data();
		auto wRow = makeConstRow_(oa::MatrixShape{1, poseDim}, w, static_cast<oa::Usize>(poseDim));   // [1, D]
		auto wVel = makeConstRow_(oa::MatrixShape{1, 1, poseDim}, w, static_cast<oa::Usize>(poseDim)); // [1,1,D]

		// Pose: contacts are folded into the weights here, so no separate term below.
		auto pose_loss = oa::FnMatrix::scale(
			oa::FnMatrix::sum(oa::FnMatrix::mul(sq, wRow)),
			inConfig.poseWeight / (static_cast<float>(nTokens) * sumW));

		const oa::I64 velTokens = static_cast<oa::I64>(bs) * (seq - 1);
		auto vel_loss = oa::FnMatrix::scale(
			oa::FnMatrix::sum(oa::FnMatrix::mul(vel_sq, wVel)),
			inConfig.velWeight / (static_cast<float>(velTokens) * sumW));

		loss = pose_loss + vel_loss;
	} else {
		// Legacy uniform layout (synthetic-gait / tutorial parity, bit-for-bit).
		auto pose_loss = oa::FnMatrix::scale(
			oa::FnMatrix::sum(sq),
			inConfig.poseWeight / static_cast<float>(sq.numElements()));

		auto vel_loss = oa::FnMatrix::scale(
			oa::FnMatrix::sum(vel_sq),
			inConfig.velWeight / static_cast<float>(vel_sq.numElements()));

		// Contact channels (trailing contactDims) get extra emphasis for planting.
		auto c_pred       = oa::FnMatrix::slice(pred3, 2, poseDim - inConfig.contactDims, poseDim);
		auto c_targ       = oa::FnMatrix::slice(targ3, 2, poseDim - inConfig.contactDims, poseDim);
		auto c_err        = c_pred - c_targ;
		auto c_sq         = oa::FnMatrix::mul(c_err, c_err);
		auto contact_loss = oa::FnMatrix::scale(
			oa::FnMatrix::sum(c_sq),
			inConfig.contactWeight / static_cast<float>(c_sq.numElements()));

		loss = pose_loss + vel_loss + contact_loss;
	}

	// root-motion consistency (leading rootDims = root trans3 + rot6) — punishes
	// drift. Off by default (rootWeight 0 ⇒ skipped entirely, exact baseline parity).
	if (inConfig.rootWeight > 0.0f && inConfig.rootDims > 0) {
		auto r_pred  = oa::FnMatrix::slice(pred3, 2, 0, inConfig.rootDims);
		auto r_targ  = oa::FnMatrix::slice(targ3, 2, 0, inConfig.rootDims);
		auto rp_d    = oa::FnMatrix::slice(r_pred, 1, 1, seq) - oa::FnMatrix::slice(r_pred, 1, 0, seq - 1);
		auto rt_d    = oa::FnMatrix::slice(r_targ, 1, 1, seq) - oa::FnMatrix::slice(r_targ, 1, 0, seq - 1);
		auto r_sq    = oa::FnMatrix::mul(rp_d - rt_d, rp_d - rt_d);
		auto r_loss  = oa::FnMatrix::scale(
			oa::FnMatrix::sum(r_sq),
			inConfig.rootWeight / static_cast<float>(r_sq.numElements()));
		loss = loss + r_loss;
	}

	return commitExtensionLoss_(
		std::move(loss), lowering,
		oa::detail::opRegistry::FnLoss::gen3dAnim,
		{&inPredFlat, &inTargetFlat},
		{oa::OpAttribute::fromUnsignedInteger(
			"configIdentity", gen3dAnimConfigIdentity_(inConfig))});
}

namespace {

oa::Matrix constF32_(oa::Span<const oa::F32> inData, oa::MatrixShape inShape) {
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(inData.data()), inData.size() * sizeof(oa::F32)),
		inShape,
		oa::ScalarType::Float32);
}

oa::Matrix constVec3_(oa::I64 inRows, const oa::vlm::Vec3& inV) {
	std::vector<oa::F32> data(static_cast<size_t>(inRows) * 3);
	for (oa::I64 i = 0; i < inRows; ++i) {
		data[static_cast<size_t>(i) * 3 + 0] = inV.x;
		data[static_cast<size_t>(i) * 3 + 1] = inV.y;
		data[static_cast<size_t>(i) * 3 + 2] = inV.z;
	}
	return constF32_(oa::Span<const oa::F32>(data.data(), data.size()), oa::MatrixShape{inRows, 3});
}

oa::Matrix constMat3_(oa::I64 inRows, const oa::vlm::Quat& inQ) {
	const oa::vlm::Mat4 m = oa::vlm::quaternionToMatrix(inQ);
	std::vector<oa::F32> data(static_cast<size_t>(inRows) * 9);
	for (oa::I64 i = 0; i < inRows; ++i) {
		oa::F32* out = &data[static_cast<size_t>(i) * 9];
		out[0] = m.m[0][0]; out[1] = m.m[0][1]; out[2] = m.m[0][2];
		out[3] = m.m[1][0]; out[4] = m.m[1][1]; out[5] = m.m[1][2];
		out[6] = m.m[2][0]; out[7] = m.m[2][1]; out[8] = m.m[2][2];
	}
	return constF32_(oa::Span<const oa::F32>(data.data(), data.size()), oa::MatrixShape{inRows, 3, 3});
}

oa::Matrix normalize3_(const oa::Matrix& inV) {
	auto sq = oa::FnMatrix::mul(inV, inV);
	auto len = oa::FnMatrix::sqrt(oa::FnMatrix::addScalar(oa::FnMatrix::sum(sq, 1), 1e-8f)).reshape(oa::MatrixShape{inV.size(0), 1});
	return oa::FnMatrix::div(inV, len);
}

oa::Matrix cross3_(const oa::Matrix& inA, const oa::Matrix& inB) {
	auto ax = oa::FnMatrix::slice(inA, 1, 0, 1);
	auto ay = oa::FnMatrix::slice(inA, 1, 1, 2);
	auto az = oa::FnMatrix::slice(inA, 1, 2, 3);
	auto bx = oa::FnMatrix::slice(inB, 1, 0, 1);
	auto by = oa::FnMatrix::slice(inB, 1, 1, 2);
	auto bz = oa::FnMatrix::slice(inB, 1, 2, 3);
	oa::Matrix parts[] = {
		oa::FnMatrix::sub(oa::FnMatrix::mul(ay, bz), oa::FnMatrix::mul(az, by)),
		oa::FnMatrix::sub(oa::FnMatrix::mul(az, bx), oa::FnMatrix::mul(ax, bz)),
		oa::FnMatrix::sub(oa::FnMatrix::mul(ax, by), oa::FnMatrix::mul(ay, bx)),
	};
	return oa::FnMatrix::concat(oa::Span<oa::Matrix>(parts, 3), 1);
}

oa::Matrix sixDToMat3_(const oa::Matrix& inSix) {
	auto a1 = oa::FnMatrix::slice(inSix, 1, 0, 3);
	auto a2 = oa::FnMatrix::slice(inSix, 1, 3, 6);
	auto b1 = normalize3_(a1);
	auto dot = oa::FnMatrix::sum(oa::FnMatrix::mul(b1, a2), 1).reshape(oa::MatrixShape{inSix.size(0), 1});
	auto b2 = normalize3_(oa::FnMatrix::sub(a2, oa::FnMatrix::mul(b1, dot)));
	auto b3 = cross3_(b1, b2);
	oa::Matrix cols[] = {
		b1.reshape(oa::MatrixShape{inSix.size(0), 3, 1}),
		b2.reshape(oa::MatrixShape{inSix.size(0), 3, 1}),
		b3.reshape(oa::MatrixShape{inSix.size(0), 3, 1}),
	};
	return oa::FnMatrix::concat(oa::Span<oa::Matrix>(cols, 3), 2);
}

oa::Matrix hingeZToMat3_(const oa::Matrix& inAngle, const oa::vlm::Quat& inRestOrient) {
	const oa::I64 tokens = inAngle.size(0);
	auto cs = oa::FnMatrix::cos(inAngle);
	auto sn = oa::FnMatrix::sin(inAngle);
	auto zero = oa::FnMatrix::zeros(oa::MatrixShape{tokens, 1});
	auto one = oa::FnMatrix::ones(oa::MatrixShape{tokens, 1});

	oa::Matrix col0Parts[] = {cs, sn, zero};
	auto negSn = oa::FnMatrix::neg(sn);
	oa::Matrix col1Parts[] = {negSn, cs, zero};
	oa::Matrix col2Parts[] = {zero, zero, one};
	auto col0 = oa::FnMatrix::concat(oa::Span<oa::Matrix>(col0Parts, 3), 1).reshape(oa::MatrixShape{tokens, 3, 1});
	auto col1 = oa::FnMatrix::concat(oa::Span<oa::Matrix>(col1Parts, 3), 1).reshape(oa::MatrixShape{tokens, 3, 1});
	auto col2 = oa::FnMatrix::concat(oa::Span<oa::Matrix>(col2Parts, 3), 1).reshape(oa::MatrixShape{tokens, 3, 1});
	oa::Matrix cols[] = {col0, col1, col2};
	auto rz = oa::FnMatrix::concat(oa::Span<oa::Matrix>(cols, 3), 2);
	return oa::FnMatrix::bmm(constMat3_(tokens, inRestOrient), rz);
}

oa::Matrix skPoseFkWorld_(const oa::Matrix& inFlat, const oa::Skeleton& inSkeleton) {
	const oa::I64 tokens = inFlat.size(0);
	const oa::I32 joints = inSkeleton.jointCount();
	std::vector<oa::Matrix> worldR(static_cast<size_t>(joints));
	std::vector<oa::Matrix> worldP(static_cast<size_t>(joints));
	std::vector<oa::Matrix> posParts;
	posParts.reserve(static_cast<size_t>(joints));

	for (oa::I32 j = 0; j < joints; ++j) {
		const oa::SkelJoint& joint = inSkeleton.joints[static_cast<oa::Usize>(j)];
		oa::I64 c = inSkeleton.channelOffset(j);
		oa::Matrix localT;
		if (joint.hasTranslate) {
			localT = oa::FnMatrix::slice(inFlat, 1, c, c + 3);
			c += 3;
		} else {
			localT = constVec3_(tokens, joint.rest.translate);
		}

		oa::Matrix localR;
		if (joint.rotDof == 3) {
			localR = sixDToMat3_(oa::FnMatrix::slice(inFlat, 1, c, c + 6));
		} else if (joint.rotDof == 1) {
			localR = hingeZToMat3_(oa::FnMatrix::slice(inFlat, 1, c, c + 1), joint.rest.jointOrient);
		} else {
			localR = constMat3_(tokens, joint.rest.orientedRotation());
		}

		if (joint.parentIndex < 0) {
			worldR[static_cast<size_t>(j)] = localR;
			worldP[static_cast<size_t>(j)] = localT;
		} else {
			const oa::Matrix& pr = worldR[static_cast<size_t>(joint.parentIndex)];
			const oa::Matrix& pp = worldP[static_cast<size_t>(joint.parentIndex)];
			worldR[static_cast<size_t>(j)] = oa::FnMatrix::bmm(pr, localR);
			auto localCol = localT.reshape(oa::MatrixShape{tokens, 3, 1});
			auto rotated = oa::FnMatrix::bmm(pr, localCol).reshape(oa::MatrixShape{tokens, 3});
			worldP[static_cast<size_t>(j)] = oa::FnMatrix::add(pp, rotated);
		}
		posParts.push_back(worldP[static_cast<size_t>(j)].reshape(oa::MatrixShape{tokens, 1, 3}));
	}

	return oa::FnMatrix::concat(oa::Span<oa::Matrix>(posParts.data(), posParts.size()), 1);
}

} // namespace

oa::Matrix oa::FnLoss::skPoseFkLoss(
	const oa::Matrix& inPredFlat,
	const oa::Matrix& inTargetWorld,
	const oa::Skeleton& inSkeleton,
	const oa::SkPoseFkLossConfig& inConfig
) {
	oa::FnLoss::setLastName("skposefk_loss");
	OA_ASSERT(inPredFlat.rank() == 2 && "SkPoseFkLoss expects pred [B*T, poseDim]");
	OA_ASSERT(inTargetWorld.rank() == 3 && "SkPoseFkLoss expects target world [B*T, J, 3]");
	OA_ASSERT(inConfig.poseDim == inPredFlat.size(1));
	OA_ASSERT(inConfig.poseDim == inSkeleton.poseDim());
	OA_ASSERT(inTargetWorld.size(0) == inPredFlat.size(0));
	OA_ASSERT(inTargetWorld.size(1) == inSkeleton.jointCount());
	OA_ASSERT(inTargetWorld.size(2) == 3);

	auto& context = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(context);
	auto predWorld = skPoseFkWorld_(inPredFlat, inSkeleton);
	auto err = predWorld - inTargetWorld;
	auto posSq = oa::FnMatrix::mul(err, err);
	auto loss = oa::FnMatrix::scale(oa::FnMatrix::sum(posSq),
		inConfig.positionWeight / static_cast<oa::F32>(posSq.numElements()));

	if (inConfig.velWeight > 0.0f && inConfig.seqLen > 1) {
		const oa::I64 tokens = inPredFlat.size(0);
		const oa::I64 batch = tokens / inConfig.seqLen;
		auto p3 = predWorld.reshape(oa::MatrixShape{batch, inConfig.seqLen, inSkeleton.jointCount(), 3});
		auto t3 = inTargetWorld.reshape(oa::MatrixShape{batch, inConfig.seqLen, inSkeleton.jointCount(), 3});
		auto pd = oa::FnMatrix::slice(p3, 1, 1, inConfig.seqLen) - oa::FnMatrix::slice(p3, 1, 0, inConfig.seqLen - 1);
		auto td = oa::FnMatrix::slice(t3, 1, 1, inConfig.seqLen) - oa::FnMatrix::slice(t3, 1, 0, inConfig.seqLen - 1);
		auto vd = pd - td;
		auto vSq = oa::FnMatrix::mul(vd, vd);
		loss = loss + oa::FnMatrix::scale(oa::FnMatrix::sum(vSq),
			inConfig.velWeight / static_cast<oa::F32>(vSq.numElements()));
	}

	return commitExtensionLoss_(
		std::move(loss), lowering,
		oa::detail::opRegistry::FnLoss::skPoseFkLoss,
		{&inPredFlat, &inTargetWorld},
		{
			oa::OpAttribute::fromUnsignedInteger(
				"skeletonIdentity", skeletonIdentity_(inSkeleton)),
			oa::OpAttribute::fromSignedInteger(
				"poseDim", inConfig.poseDim),
			oa::OpAttribute::fromSignedInteger(
				"seqLen", inConfig.seqLen),
			oa::OpAttribute::fromFloat(
				"positionWeight", inConfig.positionWeight),
			oa::OpAttribute::fromFloat(
				"velWeight", inConfig.velWeight),
		});
}

namespace {

std::vector<oa::vlm::Vec3> gen3dAnimWorldPositions_(const oa::UsdSkelClip& inUsd,
                                             const oa::Skeleton& inSkel) {
	const oa::I32 n = inSkel.jointCount();
	std::vector<oa::vlm::Vec3> out(static_cast<size_t>(inUsd.frameCount) * static_cast<size_t>(n));
	std::vector<oa::vlm::Quat> worldRot(
		static_cast<size_t>(inUsd.frameCount) * static_cast<size_t>(n));

	for (oa::U32 f = 0; f < inUsd.frameCount; ++f) {
		for (oa::I32 j = 0; j < n; ++j) {
			const size_t idx = static_cast<size_t>(f) * static_cast<size_t>(n) + static_cast<size_t>(j);
			const oa::vlm::Vec3 lt = inUsd.translations[idx];
			const oa::vlm::Quat lr = inUsd.rotations[idx];
			const oa::I32 parent = inSkel.joints[static_cast<oa::Usize>(j)].parentIndex;
			if (parent < 0) {
				worldRot[idx] = lr;
				out[idx] = lt;
			} else {
				const size_t pidx = static_cast<size_t>(f) * static_cast<size_t>(n) + static_cast<size_t>(parent);
				worldRot[idx] = worldRot[pidx] * lr;
				out[idx] = oa::vlm::add(
					out[pidx], oa::vlm::rotateVector(worldRot[pidx], lt));
			}
		}
	}
	return out;
}

oa::Result<oa::UsdSkelClip> gen3dAnimRawToUsd_(oa::Span<const oa::F32> inRaw,
                                           oa::I32 inFrameCount,
                                           oa::I32 inPoseDim,
                                           oa::F32 inFps,
                                           const oa::Skeleton& inSkeleton) {
	auto clipResult = oa::PoseClip::create(
		static_cast<oa::U32>(inFrameCount),
		static_cast<oa::U32>(inPoseDim),
		inFps,
		inSkeleton.skeletonId,
		inRaw);
	if (!clipResult.isOk()) {
		return clipResult.getStatus();
	}
	return oa::PosePack::unpack(*clipResult, inSkeleton);
}

} // namespace

oa::SkPoseFkMetrics oa::FnLoss::skPoseFk(
	oa::Span<const oa::F32> inPredRaw,
	oa::Span<const oa::F32> inTargetRaw,
	oa::I32 inFrameCount,
	oa::I32 inPoseDim,
	oa::F32 inFps,
	const oa::Skeleton& inSkeleton,
	oa::F32 inContactThreshold
) {
	oa::FnLoss::setLastName("skposefk_mse");
	oa::SkPoseFkMetrics out;
	if (inFrameCount <= 0 || inPoseDim <= 0 || !inSkeleton.isValid()) {
		return out;
	}
	const oa::Usize expected = static_cast<oa::Usize>(inFrameCount) * static_cast<oa::Usize>(inPoseDim);
	if (inPredRaw.size() != expected || inTargetRaw.size() != expected ||
	    inPoseDim != inSkeleton.poseDim()) {
		return out;
	}

	auto predUsd = gen3dAnimRawToUsd_(inPredRaw, inFrameCount, inPoseDim, inFps, inSkeleton);
	auto targetUsd = gen3dAnimRawToUsd_(inTargetRaw, inFrameCount, inPoseDim, inFps, inSkeleton);
	if (!predUsd.isOk() || !targetUsd.isOk()) {
		return out;
	}

	const oa::I32 n = inSkeleton.jointCount();
	// root is joint 0; pelvis is the only other translated joint. Both get full
	// translation + rotation tracking below.
	const oa::I32 pelvisIdx = inSkeleton.indexOf("pelvis");

	// ── rotation geodesic error (degrees) on the unpacked per-joint LOCAL quats ──
	// USD stores rotations as quaternions; the model's 6D/hinge packing is only a
	// network encoding. so compare quats directly — no FK, no bone length. Only
	// joints with a live rotation channel (rotDof != 0) are scored; locked joints
	// are never predicted. Geodesic angle = 2·acos(|dot(q_pred, q_target)|).
	{
		double sumRot = 0.0;
		oa::I64 rotCount = 0;
		double maxRot = 0.0;
		double rootRot = 0.0;
		double pelvisRot = 0.0;
		for (oa::U32 f = 0; f < predUsd->frameCount; ++f) {
			for (oa::I32 j = 0; j < n; ++j) {
				if (inSkeleton.joints[static_cast<oa::Usize>(j)].rotDof == 0) { continue; }
				const size_t idx = static_cast<size_t>(f) * static_cast<size_t>(n) + static_cast<size_t>(j);
				const oa::vlm::Quat& a = predUsd->rotations[idx];
				const oa::vlm::Quat& b = targetUsd->rotations[idx];
				double d = static_cast<double>(a.w) * b.w + static_cast<double>(a.x) * b.x
				         + static_cast<double>(a.y) * b.y + static_cast<double>(a.z) * b.z;
				d = std::min(1.0, std::max(-1.0, std::fabs(d)));
				const double deg = 2.0 * std::acos(d) * (180.0 / 3.14159265358979323846);
				sumRot += deg;
				++rotCount;
				maxRot = std::max(maxRot, deg);
				if (j == 0)         { rootRot += deg; }
				if (j == pelvisIdx) { pelvisRot += deg; }
			}
		}
		const double frames = predUsd->frameCount > 0 ? static_cast<double>(predUsd->frameCount) : 1.0;
		out.meanJointRotDeg = rotCount > 0 ? sumRot / static_cast<double>(rotCount) : 0.0;
		out.maxJointRotDeg  = maxRot;
		out.rootRotDeg      = rootRot / frames;
		out.pelvisRotDeg    = pelvisIdx >= 0 ? pelvisRot / frames : 0.0;
		out.rotJointsScored = rotCount > 0 ? static_cast<oa::I32>(rotCount / static_cast<oa::I64>(predUsd->frameCount)) : 0;
	}

	const std::vector<oa::vlm::Vec3> predWorld = gen3dAnimWorldPositions_(*predUsd, inSkeleton);
	const std::vector<oa::vlm::Vec3> targetWorld = gen3dAnimWorldPositions_(*targetUsd, inSkeleton);

	double sumJoint = 0.0;
	double sumRoot = 0.0;
	double maxRoot = 0.0;
	double sumPelvis = 0.0;
	double maxPelvis = 0.0;
	for (oa::I32 f = 0; f < inFrameCount; ++f) {
		for (oa::I32 j = 0; j < n; ++j) {
			const size_t idx = static_cast<size_t>(f) * static_cast<size_t>(n) + static_cast<size_t>(j);
			const double d = static_cast<double>(oa::vlm::length(
				oa::vlm::sub(predWorld[idx], targetWorld[idx])));
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

	const oa::I32 contactOff = inSkeleton.contactOffset();
	double footSkate = 0.0;
	oa::I64 footSkateCount = 0;
	for (oa::I32 f = 1; f < inFrameCount; ++f) {
		for (oa::I32 ci = 0; ci < static_cast<oa::I32>(inSkeleton.contactJoints.size()); ++ci) {
			const oa::I32 contactCh = contactOff + ci;
			if (contactCh < 0 || contactCh >= inPoseDim) {
				continue;
			}
			const float targetContact = inTargetRaw[static_cast<oa::Usize>(f) * inPoseDim + contactCh];
			if (targetContact < inContactThreshold) {
				continue;
			}

			const oa::I32 joint = inSkeleton.contactJoints[static_cast<oa::Usize>(ci)];
			const size_t cur = static_cast<size_t>(f) * static_cast<size_t>(n) + static_cast<size_t>(joint);
			const size_t prev = static_cast<size_t>(f - 1) * static_cast<size_t>(n) + static_cast<size_t>(joint);
			const oa::vlm::Vec3 delta = oa::vlm::sub(
				predWorld[cur], predWorld[prev]);
			footSkate += std::sqrt(static_cast<double>(delta.x * delta.x + delta.z * delta.z));
			++footSkateCount;
		}
	}

	out.mpjpeCm = sumJoint / static_cast<double>(inFrameCount * n);
	out.rootMeanCm = sumRoot / static_cast<double>(inFrameCount);
	out.rootMaxCm = maxRoot;
	out.pelvisMeanCm = pelvisIdx >= 0 ? sumPelvis / static_cast<double>(inFrameCount) : 0.0;
	out.pelvisMaxCm  = maxPelvis;
	out.footSkateCmPerFrame = footSkateCount > 0 ? footSkate / static_cast<double>(footSkateCount) : 0.0;
	out.ok = std::isfinite(out.mpjpeCm) && std::isfinite(out.rootMeanCm)
		&& std::isfinite(out.rootMaxCm) && std::isfinite(out.footSkateCmPerFrame)
		&& std::isfinite(out.meanJointRotDeg) && std::isfinite(out.maxJointRotDeg)
		&& std::isfinite(out.rootRotDeg) && std::isfinite(out.pelvisRotDeg)
		&& std::isfinite(out.pelvisMeanCm) && std::isfinite(out.pelvisMaxCm);
	return out;
}

// ─── Fused SmoothL1Mean + velSmoothL1 (oa::Alm phase F3) ─────────────────────

oa::Matrix oa::FnLoss::smoothL1Mean(const oa::Matrix& inA, const oa::Matrix& inB) {
	oa::FnLoss::setLastName("smooth_l1_mean");
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	const oa::U32 count = static_cast<oa::U32>(inA.numElements());

	// Degenerate empty input: the shader divides the reduction by count, so
	// count == 0 would produce a NaN loss (and inf gradients) that silently
	// poisons training. Return a zero scalar instead.
	if (count == 0) {
		return commitExtensionLoss_(
			oa::FnMatrix::zeros(oa::MatrixShape{1}, inA.getDtype()),
			lowering, oa::detail::opRegistry::FnLoss::smoothL1Mean,
			{&inA, &inB});
	}

	oa::Matrix out = oa::FnMatrix::empty(oa::MatrixShape{1}, inA.getDtype());
	struct { oa::U32 count; } push{ .count = count };
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,   // a
		oa::BufferAccess::Read,   // b
		oa::BufferAccess::Write   // out
	};
	ctx.add( "SmoothL1Mean", {&inA, &inB, &out}, access, &push, sizeof(push), 1);

	if (oa::FnAutograd::isEnabled() and inA.requiresGrad()) {
		auto gradFn = oa::makeShared<oa::GradSmoothL1Mean>();
		gradFn->saveForBackward({inA, inB});
		gradFn->setGraphInputs(oa::Vector<oa::Matrix>{inA, inB});
		gradFn->sequenceNr_ = oa::FnAutograd::nextSeq();
		gradFn->count_ = count;
		out.mutAutograd().gradFn = gradFn;
		out.setRequiresGrad(true);
	}
	return commitExtensionLoss_(
		std::move(out), lowering,
		oa::detail::opRegistry::FnLoss::smoothL1Mean, {&inA, &inB});
}

oa::Matrix oa::FnLoss::smoothL1MeanBwd(const oa::Matrix& inA, const oa::Matrix& inB, const oa::Matrix& inDOut) {
	oa::FnLoss::setLastName("smooth_l1_mean");
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	const oa::U32 count = static_cast<oa::U32>(inA.numElements());

	oa::Matrix dA = oa::FnMatrix::empty(inA.getShape(), inA.getDtype());
	struct { oa::U32 count; } push{ .count = count };
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,   // a
		oa::BufferAccess::Read,   // b
		oa::BufferAccess::Read,   // d_out
		oa::BufferAccess::Write   // d_a
	};
	ctx.add( "SmoothL1MeanBwd", {&inA, &inB, &inDOut, &dA}, access, &push, sizeof(push), oa::divCeil(count, 256U));
	return commitExtensionLoss_(
		std::move(dA), lowering,
		oa::detail::opRegistry::FnLoss::smoothL1MeanBwd,
		{&inA, &inB, &inDOut});
}

oa::Matrix oa::FnLoss::velSmoothL1(const oa::Matrix& inPred, const oa::Matrix& inTarget) {
	oa::FnLoss::setLastName("vel_smooth_l1");
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	const oa::I32 batch = static_cast<oa::I32>(inPred.size(0));
	const oa::I32 seqLen = static_cast<oa::I32>(inPred.size(1));
	const oa::I32 featDim = static_cast<oa::I32>(inPred.size(2));
	const oa::U32 count = static_cast<oa::U32>(batch) * static_cast<oa::U32>(seqLen - 1) * static_cast<oa::U32>(featDim);

	// A single-frame window (seqLen < 2) has no velocity term; count == 0 would
	// divide the reduction by zero → NaN loss. Return a zero scalar, matching the
	// FK-loss builder's own `seqLen > 1` guard.
	if (count == 0) {
		return commitExtensionLoss_(
			oa::FnMatrix::zeros(oa::MatrixShape{1}, inPred.getDtype()),
			lowering, oa::detail::opRegistry::FnLoss::velSmoothL1,
			{&inPred, &inTarget});
	}

	oa::Matrix out = oa::FnMatrix::empty(oa::MatrixShape{1}, inPred.getDtype());
	struct {
		oa::U32 batch;
		oa::U32 seqLen;
		oa::U32 featDim;
		oa::U32 count;
	} push{
		.batch = static_cast<oa::U32>(batch),
		.seqLen = static_cast<oa::U32>(seqLen),
		.featDim = static_cast<oa::U32>(featDim),
		.count = count};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,   // pred
		oa::BufferAccess::Read,   // target
		oa::BufferAccess::Write   // out
	};
	ctx.add( "VelSmoothL1", {&inPred, &inTarget, &out}, access, &push, sizeof(push), 1);

	if (oa::FnAutograd::isEnabled() and inPred.requiresGrad()) {
		auto gradFn = oa::makeShared<oa::GradVelSmoothL1>();
		gradFn->saveForBackward({inPred, inTarget});
		gradFn->setGraphInputs(oa::Vector<oa::Matrix>{inPred, inTarget});
		gradFn->sequenceNr_ = oa::FnAutograd::nextSeq();
		gradFn->batch_ = batch;
		gradFn->seqLen_ = seqLen;
		gradFn->featDim_ = featDim;
		gradFn->count_ = count;
		out.mutAutograd().gradFn = gradFn;
		out.setRequiresGrad(true);
	}
	return commitExtensionLoss_(
		std::move(out), lowering,
		oa::detail::opRegistry::FnLoss::velSmoothL1,
		{&inPred, &inTarget});
}

oa::Matrix oa::FnLoss::velSmoothL1Bwd(const oa::Matrix& inPred, const oa::Matrix& inTarget, const oa::Matrix& inDOut) {
	oa::FnLoss::setLastName("vel_smooth_l1");
	auto& ctx = oa::ExecutionSession::getActive();
	oa::OpLoweringScope lowering(ctx);
	const oa::I32 batch = static_cast<oa::I32>(inPred.size(0));
	const oa::I32 seqLen = static_cast<oa::I32>(inPred.size(1));
	const oa::I32 featDim = static_cast<oa::I32>(inPred.size(2));
	const oa::U32 count = static_cast<oa::U32>(batch) * static_cast<oa::U32>(seqLen - 1) * static_cast<oa::U32>(featDim);
	const oa::U32 total = static_cast<oa::U32>(inPred.numElements());

	oa::Matrix dPred = oa::FnMatrix::empty(inPred.getShape(), inPred.getDtype());
	struct {
		oa::U32 batch;
		oa::U32 seqLen;
		oa::U32 featDim;
		oa::U32 count;
	} push{
		.batch = static_cast<oa::U32>(batch),
		.seqLen = static_cast<oa::U32>(seqLen),
		.featDim = static_cast<oa::U32>(featDim),
		.count = count};
	oa::BufferAccess access[] = {
		oa::BufferAccess::Read,   // pred
		oa::BufferAccess::Read,   // target
		oa::BufferAccess::Read,   // d_out
		oa::BufferAccess::Write   // d_pred
	};
	ctx.add( "VelSmoothL1Bwd", {&inPred, &inTarget, &inDOut, &dPred}, access, &push, sizeof(push), oa::divCeil(total, 256U));
	return commitExtensionLoss_(
		std::move(dPred), lowering,
		oa::detail::opRegistry::FnLoss::velSmoothL1Bwd,
		{&inPred, &inTarget, &inDOut});
}

// ─── SkPoseFk helpers ───────────────────────────────────────────────────────

oa::Matrix oa::FnLoss::skPoseFkTargetWorld(
	oa::Span<const oa::F32> inRaw,
	oa::I32 inFrameCount,
	oa::I32 inPoseDim,
	oa::F32 inFps,
	const oa::Skeleton& inSkeleton
) {
	const oa::I32 joints = inSkeleton.jointCount();
	const oa::Usize expected = static_cast<oa::Usize>(inFrameCount) * static_cast<oa::Usize>(inPoseDim);
	if (inFrameCount <= 0 || inPoseDim <= 0 || !inSkeleton.isValid()
		|| inRaw.size() != expected || inPoseDim != inSkeleton.poseDim()) {
		return oa::FnMatrix::zeros(oa::MatrixShape{std::max<oa::I32>(inFrameCount, 0), std::max<oa::I32>(joints, 0), 3});
	}

	auto usd = gen3dAnimRawToUsd_(inRaw, inFrameCount, inPoseDim, inFps, inSkeleton);
	if (!usd.isOk()) {
		return oa::FnMatrix::zeros(oa::MatrixShape{inFrameCount, joints, 3});
	}
	const std::vector<oa::vlm::Vec3> world = gen3dAnimWorldPositions_(*usd, inSkeleton);
	std::vector<oa::F32> packed(static_cast<size_t>(inFrameCount) * static_cast<size_t>(joints) * 3);
	for (oa::I32 f = 0; f < inFrameCount; ++f) {
		for (oa::I32 j = 0; j < joints; ++j) {
			const size_t src = static_cast<size_t>(f) * static_cast<size_t>(joints) + static_cast<size_t>(j);
			const size_t dst = src * 3;
			packed[dst + 0] = world[src].x;
			packed[dst + 1] = world[src].y;
			packed[dst + 2] = world[src].z;
		}
	}
	return oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(packed.data()), packed.size() * sizeof(oa::F32)),
		oa::MatrixShape{inFrameCount, joints, 3},
		oa::ScalarType::Float32);
}
