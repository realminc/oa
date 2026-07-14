// OaFnDataset — Unified dataset operations implementation

#include <Oa/Data/FnDataset.h>
#include <Oa/Core/Log.h>
#include <random>
#include <numeric>
#include <algorithm>

namespace OaFnDataset {

namespace {

bool IsValidSampleMatrix(const OaMatrix& InMatrix) {
	return !InMatrix.IsEmpty() && InMatrix.HasStorage() &&
		InMatrix.Rank() < OA_MAX_TENSOR_DIMS && InMatrix.ByteSize() > 0;
}

OaMatrixShape BatchedShape(const OaMatrixShape& InItemShape, OaI64 InBatchSize) {
	OaMatrixShape shape = InItemShape;
	for (OaI32 i = shape.Rank; i > 0; --i) {
		shape.Dims[static_cast<OaUsize>(i)] =
			shape.Dims[static_cast<OaUsize>(i) - 1];
	}
	shape.Dims[0] = InBatchSize;
	++shape.Rank;
	return shape;
}

bool ValidateLike(const OaMatrix& InMatrix, const OaMatrix& InReference) {
	return IsValidSampleMatrix(InMatrix) &&
		InMatrix.GetShape() == InReference.GetShape() &&
		InMatrix.GetDtype() == InReference.GetDtype() &&
		InMatrix.ByteSize() == InReference.ByteSize();
}

} // namespace

// ============================================================================
// Index shuffling and splitting
// ============================================================================

void Shuffle(OaVec<OaI64>& InOutIndices, OaU64 InSeed) {
	if (InSeed == 0) {
		std::random_device rd;
		InSeed = rd();
	}
	std::mt19937 rng(static_cast<OaU32>(InSeed));
	std::shuffle(InOutIndices.Begin(), InOutIndices.End(), rng);
}

SplitResult RandomSplit(OaI64 InTotalSize, OaF32 InTrainRatio, OaF32 InValRatio, OaU64 InSeed) {
	SplitResult result;
	if (InTotalSize <= 0) return result;
	InTrainRatio = std::clamp(InTrainRatio, 0.0f, 1.0f);
	InValRatio = std::clamp(InValRatio, 0.0f, 1.0f - InTrainRatio);

	OaVec<OaI64> indices(static_cast<OaUsize>(InTotalSize));
	std::iota(indices.Begin(), indices.End(), 0);
	Shuffle(indices, InSeed);

	OaI64 trainCount = static_cast<OaI64>(static_cast<OaF32>(InTotalSize) * InTrainRatio);
	OaI64 valCount   = static_cast<OaI64>(static_cast<OaF32>(InTotalSize) * InValRatio);
	OaI64 testCount  = InTotalSize - trainCount - valCount;
	if (testCount < 0) {
		trainCount += testCount; // absorb deficit into train
		testCount = 0;
	}
	if (valCount < 0) {
		trainCount += valCount;
		valCount = 0;
	}

	result.Train.Reserve(static_cast<OaUsize>(trainCount));
	result.Val.Reserve(static_cast<OaUsize>(valCount));
	result.Test.Reserve(static_cast<OaUsize>(testCount));

	for (OaI64 i = 0; i < trainCount; ++i)
		result.Train.PushBack(indices[static_cast<OaUsize>(i)]);
	for (OaI64 i = 0; i < valCount; ++i)
		result.Val.PushBack(indices[static_cast<OaUsize>(trainCount + i)]);
	for (OaI64 i = 0; i < testCount; ++i)
		result.Test.PushBack(indices[static_cast<OaUsize>(trainCount + valCount + i)]);

	return result;
}

// ============================================================================
// Batch collation
// ============================================================================

Batch Collate(OaSpan<const OaDataset::Sample> InSamples) {
	if (InSamples.Size() == 0) return {};

	const auto& first = InSamples[0];
	if (!IsValidSampleMatrix(first.X)) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"OaFnDataset::Collate: first X is empty, unstored, or has maximum rank");
		return {};
	}

	const bool hasLabel = first.HasLabel();
	if (hasLabel && !IsValidSampleMatrix(first.Y)) {
		OA_LOG_ERROR(OaLogComponent::ML,
			"OaFnDataset::Collate: first Y is invalid");
		return {};
	}

	for (OaUsize i = 1; i < InSamples.Size(); ++i) {
		const auto& sample = InSamples[i];
		if (!ValidateLike(sample.X, first.X)) {
			OA_LOG_ERROR(OaLogComponent::ML,
				"OaFnDataset::Collate: X shape, dtype, or storage mismatch at sample %zu", i);
			return {};
		}
		if (sample.HasLabel() != hasLabel ||
			(hasLabel && !ValidateLike(sample.Y, first.Y))) {
			OA_LOG_ERROR(OaLogComponent::ML,
				"OaFnDataset::Collate: Y presence, shape, dtype, or storage mismatch at sample %zu", i);
			return {};
		}
	}

	OaI64 actualBatch = static_cast<OaI64>(InSamples.Size());

	// X batch: prepend batch dimension
	OaMatrixShape xShape = BatchedShape(first.X.GetShape(), actualBatch);
	auto xBatch = OaFnMatrix::Empty(xShape, first.X.GetDtype());
	if (!xBatch.HasStorage()) return {};

	// Y batch: optional
	OaMatrix yBatch;
	if (hasLabel) {
		OaMatrixShape yShape = BatchedShape(first.Y.GetShape(), actualBatch);
		yBatch = OaFnMatrix::Empty(yShape, first.Y.GetDtype());
		if (!yBatch.HasStorage()) return {};
	}

	OaUsize xItemBytes = static_cast<OaUsize>(first.X.ByteSize());
	OaUsize yItemBytes = hasLabel ? static_cast<OaUsize>(first.Y.ByteSize()) : 0;
	for (OaUsize i = 0; i < InSamples.Size(); ++i) {
		const auto& sample = InSamples[i];
		OaMemcpy(xBatch.DataAs<OaU8>() + i * xItemBytes,
			sample.X.DataAs<const OaU8>(), xItemBytes);
		if (hasLabel) {
			OaMemcpy(yBatch.DataAs<OaU8>() + i * yItemBytes,
				sample.Y.DataAs<const OaU8>(), yItemBytes);
		}
	}

	return Batch{std::move(xBatch), std::move(yBatch)};
}

// ============================================================================
// Normalization & augmentation helpers
// ============================================================================

OaStatus Normalize(OaMatrix& InOutBatch, const OaNormalizationParams& InParams) {
	// Delegate to OaFnImage for image-shaped tensors; generic path falls back
	// to OaFnMatrix elementwise ops (TODO: channel-aware 1D/2D normalize).
	auto shape = InOutBatch.GetShape();
	if (shape.Rank >= 3) {
		InOutBatch = OaFnImage::Normalize(InOutBatch, InParams);
		if (InOutBatch.IsEmpty()) {
			return OaStatus::Error(OaStatusCode::Internal,
				"OaFnDataset::Normalize: image normalization failed");
		}
		return OaStatus::Ok();
	}
	return OaStatus::Unimplemented(
		"OaFnDataset::Normalize: generic rank < 3 normalization is not implemented");
}

OaStatus ScaleToRange(OaMatrix& InOutBatch, OaF32 InMin, OaF32 InMax) {
	(void)InOutBatch;
	if (InMin >= InMax) {
		return OaStatus::InvalidArgument(
			"OaFnDataset::ScaleToRange: minimum must be smaller than maximum");
	}
	return OaStatus::Unimplemented(
		"OaFnDataset::ScaleToRange: cast and scale kernel is not implemented");
}

OaResult<OaMatrix> Cast(const OaMatrix& InBatch, OaScalarType InDtype) {
	if (InBatch.IsEmpty() || !InBatch.HasStorage()) {
		return OaResult<OaMatrix>(
			OaStatus::InvalidArgument("OaFnDataset::Cast: input has no storage"));
	}
	if (InBatch.GetDtype() == InDtype) return OaResult<OaMatrix>(InBatch);
	if (InBatch.GetDtype() != OaScalarType::UInt8 ||
		InDtype != OaScalarType::Float32) {
		return OaResult<OaMatrix>(OaStatus::Unimplemented(
			"OaFnDataset::Cast: only UInt8 to Float32 is implemented"));
	}

	OaVec<OaU8> bytes(InBatch.NumElements());
	auto status = OaFnMatrix::CopyToHost(
		InBatch, bytes.Data(), static_cast<OaU64>(bytes.Size()));
	if (!status.IsOk()) return OaResult<OaMatrix>(std::move(status));
	auto out = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(bytes.Data(), bytes.Size()),
		InBatch.GetShape(), OaScalarType::Float32);
	if (out.IsEmpty()) {
		return OaResult<OaMatrix>(
			OaStatus::Error("OaFnDataset::Cast: allocation or upload failed"));
	}
	return OaResult<OaMatrix>(std::move(out));
}

} // namespace OaFnDataset
